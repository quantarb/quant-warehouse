from __future__ import annotations

from datetime import timedelta
from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import normalize_prices, symbol_provider_key
from quant_warehouse.ingest.openbb_fetch import fetch_dataframe
from quant_warehouse.ingest.providers import DEFAULT_PRICE_PROVIDERS, validate_price_provider
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.merge import merge_upsert

PRICES_LIBRARY = "prices"
GAP_OVERLAP_DAYS = 5
GAP_FILL_RETRY_LOOKBACK_DAYS = 30
EQUITY_PRICE_ADJUSTMENT = "splits_and_dividends"


def parse_symbol_provider_key(storage_symbol: str) -> tuple[str, str] | None:
    text = str(storage_symbol).strip()
    if "__" not in text:
        return None
    symbol, provider = text.rsplit("__", 1)
    symbol = symbol.strip().upper()
    provider = provider.strip().lower()
    if not symbol or not provider:
        return None
    return symbol, provider


def list_arctic_price_underlyings(
    backend: ArcticBackend,
    *,
    provider: str = "fmp",
    library: str = PRICES_LIBRARY,
) -> list[str]:
    """Return underlying symbols stored in Arctic for a price vendor."""

    target_provider = str(provider).strip().lower()
    symbols: list[str] = []
    seen: set[str] = set()
    for storage_symbol in backend.list_symbols(library):
        parsed = parse_symbol_provider_key(storage_symbol)
        if parsed is None:
            continue
        symbol, stored_provider = parsed
        if stored_provider != target_provider or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return sorted(symbols)


class PricesStore:
    """ArcticDB-backed historical OHLCV store with per-vendor symbols and gap-fill."""

    def __init__(
        self,
        config: WarehouseConfig | None = None,
        *,
        backend: StorageBackend | None = None,
        catalog: CatalogStore | None = None,
    ) -> None:
        self.config = config or WarehouseConfig.from_env()
        self.config.ensure_dirs()
        self.backend: ArcticBackend = backend or open_backend(self.config)
        self.storage_kind = "arctic"
        self.catalog = catalog or CatalogStore(self.config.catalog_path)

    def refresh(
        self,
        symbol: str,
        *,
        providers: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
    ) -> dict[str, dict[str, object]]:
        symbol = symbol.strip().upper()
        provider_list = [validate_price_provider(p) for p in (providers or DEFAULT_PRICE_PROVIDERS)]
        stats: dict[str, dict[str, object]] = {}

        for provider in provider_list:
            fetch_start = start_date
            if fetch_start is None and not full_refresh:
                fetch_start = self._gap_fill_start(symbol, provider)

            kwargs: dict[str, str] = {}
            if fetch_start:
                kwargs["start_date"] = fetch_start
            if end_date:
                kwargs["end_date"] = end_date
            kwargs["adjustment"] = EQUITY_PRICE_ADJUSTMENT

            raw = fetch_dataframe("prices", symbol=symbol, provider=provider, **kwargs)
            history_floor = self.catalog.equity_historical_start(symbol)
            frame = normalize_prices(raw, provider=provider, min_date=history_floor)
            if (
                frame.empty
                and fetch_start
                and not full_refresh
                and end_date
            ):
                state = self.catalog.get(symbol=symbol, section="prices", provider=provider)
                if state is not None and state.max_date:
                    wider_start = pd.Timestamp(state.max_date) - timedelta(days=GAP_FILL_RETRY_LOOKBACK_DAYS)
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["start_date"] = wider_start.strftime("%Y-%m-%d")
                    raw = fetch_dataframe("prices", symbol=symbol, provider=provider, **retry_kwargs)
                    frame = normalize_prices(raw, provider=provider, min_date=history_floor)
                    if not frame.empty:
                        fetch_start = retry_kwargs["start_date"]
            storage_symbol = symbol_provider_key(symbol, provider)

            existing = self.backend.read(PRICES_LIBRARY, storage_symbol)
            merged = merge_upsert(existing, frame)
            rows_written = 0
            if not merged.empty:
                self.backend.write(PRICES_LIBRARY, storage_symbol, merged)
                rows_written = len(merged)

            min_date = None
            max_date = None
            if not merged.empty:
                min_date = merged.index.min().strftime("%Y-%m-%d")
                max_date = merged.index.max().strftime("%Y-%m-%d")

            self.catalog.upsert(
                symbol=symbol,
                section="prices",
                provider=provider,
                min_date=min_date,
                max_date=max_date,
                row_count=rows_written,
                columns_present=[c for c in merged.columns],
            )
            stats[provider] = {
                "rows": rows_written,
                "fetched_rows": len(frame),
                "min_date": min_date,
                "max_date": max_date,
                "storage_symbol": storage_symbol,
                "fetch_start": fetch_start,
                "storage_backend": self.storage_kind,
            }

        return stats

    def ingest_frame(
        self,
        symbol: str,
        *,
        provider: str,
        frame: pd.DataFrame,
        merge: bool = True,
    ) -> dict[str, object]:
        symbol = symbol.strip().upper()
        provider = validate_price_provider(provider)
        history_floor = self.catalog.equity_historical_start(symbol)
        normalized = normalize_prices(frame, provider=provider, min_date=history_floor)
        storage_symbol = symbol_provider_key(symbol, provider)

        merged = normalized
        if merge:
            existing = self.backend.read(PRICES_LIBRARY, storage_symbol)
            merged = merge_upsert(existing, normalized)

        rows_written = 0
        if not merged.empty:
            self.backend.write(PRICES_LIBRARY, storage_symbol, merged)
            rows_written = len(merged)

        min_date = None
        max_date = None
        if not merged.empty:
            min_date = merged.index.min().strftime("%Y-%m-%d")
            max_date = merged.index.max().strftime("%Y-%m-%d")

        self.catalog.upsert(
            symbol=symbol,
            section="prices",
            provider=provider,
            min_date=min_date,
            max_date=max_date,
            row_count=rows_written,
            columns_present=[c for c in merged.columns],
        )
        return {
            "rows": rows_written,
            "fetched_rows": len(normalized),
            "min_date": min_date,
            "max_date": max_date,
            "storage_symbol": storage_symbol,
            "storage_backend": self.storage_kind,
        }

    def read(
        self,
        symbol: str,
        *,
        provider: str = "yfinance",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        provider = validate_price_provider(provider)
        storage_symbol = symbol_provider_key(symbol, provider)
        df = self.backend.read(PRICES_LIBRARY, storage_symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        return _slice_dates(df, start=start, end=end)

    def list_providers(self, symbol: str) -> list[str]:
        rows = self.catalog.list_symbol(symbol.strip().upper())
        return [row.provider for row in rows if row.section == "prices"]

    def _gap_fill_start(self, symbol: str, provider: str) -> str | None:
        state = self.catalog.get(symbol=symbol, section="prices", provider=provider)
        if state is None or not state.max_date:
            return None
        resume = pd.Timestamp(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
        return resume.strftime("%Y-%m-%d")


def _slice_dates(
    df: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    out = df.copy()
    index_tz = getattr(out.index, "tz", None)
    if start is not None:
        start_ts = pd.Timestamp(start)
        if index_tz is not None and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(index_tz)
        elif index_tz is None and start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert(None)
        out = out.loc[out.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        if index_tz is not None and end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(index_tz)
        elif index_tz is None and end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert(None)
        out = out.loc[out.index <= end_ts]
    return out
