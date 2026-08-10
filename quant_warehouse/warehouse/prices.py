from __future__ import annotations

from datetime import timedelta
from datetime import datetime
from typing import Literal, Sequence

import polars as pl

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import normalize_prices, symbol_provider_key
from quant_warehouse.ingest.openbb_fetch import fetch_dataframe
from quant_warehouse.ingest.providers import DEFAULT_PRICE_PROVIDERS, validate_price_provider
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.merge import merge_upsert
from quant_warehouse.warehouse.sections import ETF_PRICES_LIBRARY, FUND_PRICES_LIBRARY
from quant_warehouse.warehouse.storage import read_provider_frame, provider_library

PRICES_LIBRARY = "prices"
GAP_OVERLAP_DAYS = 5
GAP_FILL_RETRY_LOOKBACK_DAYS = 30
EQUITY_PRICE_ADJUSTMENT = "splits_and_dividends"
RAW_EQUITY_PRICE_ADJUSTMENT = "unadjusted"
PRICE_ADJUSTMENTS: tuple[str, ...] = (
    "splits_only",
    "splits_and_dividends",
    "unadjusted",
)
PriceAdjustment = Literal["splits_only", "splits_and_dividends", "unadjusted"]


def validate_price_adjustment(adjustment: str | None) -> PriceAdjustment:
    value = str(adjustment or EQUITY_PRICE_ADJUSTMENT).strip().lower()
    if value not in PRICE_ADJUSTMENTS:
        raise ValueError(
            f"Unknown equity price adjustment {adjustment!r}; "
            f"expected one of {PRICE_ADJUSTMENTS}"
        )
    return value  # type: ignore[return-value]


def price_library_for_adjustment(adjustment: str | None = EQUITY_PRICE_ADJUSTMENT) -> str:
    value = validate_price_adjustment(adjustment)
    if value == EQUITY_PRICE_ADJUSTMENT:
        return PRICES_LIBRARY
    return f"{PRICES_LIBRARY}_{value}"


def price_section_for_adjustment(adjustment: str | None = EQUITY_PRICE_ADJUSTMENT) -> str:
    return price_library_for_adjustment(adjustment)


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
    libraries = [provider_library(library, target_provider)]
    for library_name in libraries:
        try:
            library_symbols = backend.list_symbols(library_name)
        except Exception:
            continue
        for storage_symbol in library_symbols:
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
        adjustment: str = EQUITY_PRICE_ADJUSTMENT,
    ) -> dict[str, dict[str, object]]:
        symbol = symbol.strip().upper()
        adjustment = validate_price_adjustment(adjustment)
        base_library = price_library_for_adjustment(adjustment)
        section = price_section_for_adjustment(adjustment)
        provider_list = [
            validate_price_provider(p)
            for p in (providers or DEFAULT_PRICE_PROVIDERS)
        ]
        stats: dict[str, dict[str, object]] = {}

        for provider in provider_list:
            fetch_start = start_date
            if fetch_start is None and not full_refresh:
                fetch_start = self._gap_fill_start(symbol, provider, adjustment=adjustment)

            kwargs: dict[str, str] = {}
            if fetch_start:
                kwargs["start_date"] = fetch_start
            if end_date:
                kwargs["end_date"] = end_date

            raw = fetch_dataframe(
                "prices",
                symbol=symbol,
                provider=provider,
                adjustment=adjustment,
                **kwargs,
            )
            history_floor = self.catalog.equity_historical_start(symbol)
            frame = normalize_prices(raw, provider=provider, min_date=history_floor)
            if (
                frame.is_empty()
                and fetch_start
                and not full_refresh
                and end_date
            ):
                state = self.catalog.get(symbol=symbol, section=section, provider=provider)
                if state is not None and state.max_date:
                    wider_start = datetime.fromisoformat(state.max_date) - timedelta(
                        days=GAP_FILL_RETRY_LOOKBACK_DAYS
                    )
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["start_date"] = wider_start.strftime("%Y-%m-%d")
                    raw = fetch_dataframe(
                        "prices",
                        symbol=symbol,
                        provider=provider,
                        adjustment=adjustment,
                        **retry_kwargs,
                    )
                    frame = normalize_prices(raw, provider=provider, min_date=history_floor)
                    if not frame.is_empty():
                        fetch_start = retry_kwargs["start_date"]
            storage_symbol = symbol_provider_key(symbol, provider)

            library = provider_library(base_library, provider)
            existing = read_provider_frame(
                self.backend,
                base_library=base_library,
                provider=provider,
                symbol=storage_symbol,
            )
            merged = merge_upsert(existing, frame)
            rows_written = 0
            if not merged.is_empty():
                self.backend.write(library, storage_symbol, merged)
                rows_written = len(merged)

            min_date = None
            max_date = None
            if not merged.is_empty():
                min_date = merged["date"].min().strftime("%Y-%m-%d")
                max_date = merged["date"].max().strftime("%Y-%m-%d")

            self.catalog.upsert(
                symbol=symbol,
                section=section,
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
                "library": library,
                "adjustment": adjustment,
                "fetch_start": fetch_start,
                "storage_backend": self.storage_kind,
            }

        return stats

    def ingest_frame(
        self,
        symbol: str,
        *,
        provider: str,
        frame: pl.DataFrame,
        merge: bool = True,
        adjustment: str = EQUITY_PRICE_ADJUSTMENT,
    ) -> dict[str, object]:
        symbol = symbol.strip().upper()
        provider = validate_price_provider(provider)
        adjustment = validate_price_adjustment(adjustment)
        base_library = price_library_for_adjustment(adjustment)
        section = price_section_for_adjustment(adjustment)
        history_floor = self.catalog.equity_historical_start(symbol)
        normalized = normalize_prices(frame, provider=provider, min_date=history_floor)
        storage_symbol = symbol_provider_key(symbol, provider)

        merged = normalized
        if merge:
            library = provider_library(base_library, provider)
            existing = read_provider_frame(
                self.backend,
                base_library=base_library,
                provider=provider,
                symbol=storage_symbol,
            )
            merged = merge_upsert(existing, normalized)
        else:
            library = provider_library(base_library, provider)

        rows_written = 0
        if not merged.is_empty():
            self.backend.write(library, storage_symbol, merged)
            rows_written = len(merged)

        min_date = None
        max_date = None
        if not merged.is_empty():
            min_date = merged["date"].min().strftime("%Y-%m-%d")
            max_date = merged["date"].max().strftime("%Y-%m-%d")

        self.catalog.upsert(
            symbol=symbol,
            section=section,
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
            "library": library,
            "adjustment": adjustment,
            "storage_backend": self.storage_kind,
        }

    def read(
        self,
        symbol: str,
        *,
        provider: str = "yfinance",
        start: str | None = None,
        end: str | None = None,
        adjustment: str = EQUITY_PRICE_ADJUSTMENT,
        output_format: Literal["polars"] = "polars",
    ) -> pl.DataFrame:
        provider = validate_price_provider(provider)
        adjustment = validate_price_adjustment(adjustment)
        base_library = price_library_for_adjustment(adjustment)
        storage_symbol = symbol_provider_key(symbol, provider)
        df = read_provider_frame(
            self.backend,
            base_library=base_library,
            provider=provider,
            symbol=storage_symbol,
            output_format=output_format,
        )
        if adjustment == EQUITY_PRICE_ADJUSTMENT and (df is None or df.is_empty()):
            df = read_provider_frame(
                self.backend,
                base_library=ETF_PRICES_LIBRARY,
                provider=provider,
                symbol=storage_symbol,
                output_format=output_format,
            )
        if adjustment == EQUITY_PRICE_ADJUSTMENT and (df is None or df.is_empty()):
            df = read_provider_frame(
                self.backend,
                base_library=FUND_PRICES_LIBRARY,
                provider=provider,
                symbol=storage_symbol,
                output_format=output_format,
            )
        if df is None or df.is_empty():
            return pl.DataFrame()
        return _slice_dates(df, start=start, end=end)

    def list_providers(
        self,
        symbol: str,
        *,
        adjustment: str = EQUITY_PRICE_ADJUSTMENT,
    ) -> list[str]:
        section = price_section_for_adjustment(adjustment)
        rows = self.catalog.list_symbol(symbol.strip().upper())
        return [row.provider for row in rows if row.section == section]

    def _gap_fill_start(
        self,
        symbol: str,
        provider: str,
        *,
        adjustment: str = EQUITY_PRICE_ADJUSTMENT,
    ) -> str | None:
        section = price_section_for_adjustment(adjustment)
        state = self.catalog.get(symbol=symbol, section=section, provider=provider)
        if state is None or not state.max_date:
            return None
        resume = datetime.fromisoformat(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
        return resume.strftime("%Y-%m-%d")


def _slice_dates(
    df: pl.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> pl.DataFrame:
    if "date" not in df.columns:
        return df
    predicate = pl.lit(True)
    if start is not None:
        predicate = predicate & (pl.col("date") >= pl.lit(datetime.fromisoformat(start)))
    if end is not None:
        predicate = predicate & (pl.col("date") <= pl.lit(datetime.fromisoformat(end)))
    return df.filter(predicate)
