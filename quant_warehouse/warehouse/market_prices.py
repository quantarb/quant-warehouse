from __future__ import annotations

from datetime import timedelta
from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import normalize_prices, symbol_provider_key
from quant_warehouse.ingest.openbb_fetch import fetch_dataframe
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.merge import merge_upsert
from quant_warehouse.warehouse.prices import _slice_dates
from quant_warehouse.warehouse.sections import (
    MARKET_PRICE_SECTIONS,
    MIN_HISTORICAL_DATE,
)

GAP_OVERLAP_DAYS = 5
GAP_FILL_RETRY_LOOKBACK_DAYS = 30


class MarketPricesStore:
    """Arctic-backed OHLCV for crypto, FX, and index symbols."""

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
        section: str,
        provider: str = "fmp",
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
    ) -> dict[str, object]:
        section_name = str(section).strip()
        library = MARKET_PRICE_SECTIONS.get(section_name)
        if library is None:
            raise ValueError(f"Unknown market price section: {section}")

        symbol = symbol.strip().upper()
        provider_name = str(provider or "fmp").strip().lower()
        fetch_start = start_date
        if fetch_start is None and not full_refresh:
            fetch_start = self._gap_fill_start(symbol, section_name, provider_name)

        kwargs: dict[str, str] = {}
        if fetch_start:
            kwargs["start_date"] = fetch_start
        if end_date:
            kwargs["end_date"] = end_date

        raw = fetch_dataframe(section_name, symbol=symbol, provider=provider_name, **kwargs)
        frame = normalize_prices(raw, provider=provider_name, min_date=MIN_HISTORICAL_DATE)
        if frame.empty and fetch_start and not full_refresh and end_date:
            state = self.catalog.get(symbol=symbol, section=section_name, provider=provider_name)
            if state is not None and state.max_date:
                wider_start = pd.Timestamp(state.max_date) - timedelta(days=GAP_FILL_RETRY_LOOKBACK_DAYS)
                retry_kwargs = dict(kwargs)
                retry_kwargs["start_date"] = wider_start.strftime("%Y-%m-%d")
                raw = fetch_dataframe(section_name, symbol=symbol, provider=provider_name, **retry_kwargs)
                frame = normalize_prices(raw, provider=provider_name, min_date=MIN_HISTORICAL_DATE)
                if not frame.empty:
                    fetch_start = retry_kwargs["start_date"]

        storage_symbol = symbol_provider_key(symbol, provider_name)
        existing = self.backend.read(library, storage_symbol)
        merged = merge_upsert(existing, frame)
        if not merged.empty:
            self.backend.write(library, storage_symbol, merged)

        min_date = None
        max_date = None
        if not merged.empty:
            min_date = merged.index.min().strftime("%Y-%m-%d")
            max_date = merged.index.max().strftime("%Y-%m-%d")

        self.catalog.upsert(
            symbol=symbol,
            section=section_name,
            provider=provider_name,
            min_date=min_date,
            max_date=max_date,
            row_count=int(len(merged)),
            columns_present=[str(column) for column in merged.columns],
        )
        return {
            "symbol": symbol,
            "section": section_name,
            "provider": provider_name,
            "rows": int(len(merged)),
            "fetched_rows": int(len(frame)),
            "min_date": min_date,
            "max_date": max_date,
            "storage_symbol": storage_symbol,
            "fetch_start": fetch_start,
            "storage_backend": self.storage_kind,
        }

    def read(
        self,
        symbol: str,
        *,
        section: str,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        section_name = str(section).strip()
        library = MARKET_PRICE_SECTIONS.get(section_name)
        if library is None:
            raise ValueError(f"Unknown market price section: {section}")
        storage_symbol = symbol_provider_key(symbol.strip().upper(), provider.strip().lower())
        frame = self.backend.read(library, storage_symbol)
        if frame is None or frame.empty:
            return pd.DataFrame()
        return _slice_dates(frame, start=start, end=end)

    def _gap_fill_start(self, symbol: str, section: str, provider: str) -> str | None:
        state = self.catalog.get(symbol=symbol, section=section, provider=provider)
        if state is None or not state.max_date:
            return None
        resume = pd.Timestamp(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
        return resume.strftime("%Y-%m-%d")


def refresh_market_price_universe(
    store: MarketPricesStore,
    symbols: Sequence[str],
    *,
    section: str,
    provider: str = "fmp",
    start_date: str | None = None,
    end_date: str | None = None,
    full_refresh: bool = False,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for symbol in symbols:
        symbol_text = str(symbol).strip().upper()
        if not symbol_text:
            continue
        try:
            results.append(
                store.refresh(
                    symbol_text,
                    section=section,
                    provider=provider,
                    start_date=start_date,
                    end_date=end_date,
                    full_refresh=full_refresh,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "symbol": symbol_text,
                    "section": section,
                    "provider": provider,
                    "status": "error",
                    "error": str(exc),
                }
            )
    return results