from __future__ import annotations

from datetime import timedelta

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.equity_calendar_fetch import fetch_equity_calendar_range
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.merge import merge_panel_upsert
from quant_warehouse.warehouse.sections import (
    EQUITY_CALENDAR_BUNDLE_SYMBOL,
    EQUITY_CALENDAR_SECTIONS,
    MIN_HISTORICAL_DATE,
    fundamental_library,
)
from quant_warehouse.warehouse.storage import provider_library

GAP_OVERLAP_DAYS = 5


def _min_date_text(frame: pd.DataFrame) -> str | None:
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    return frame.index.min().strftime("%Y-%m-%d")


def _max_date_text(frame: pd.DataFrame) -> str | None:
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    return frame.index.max().strftime("%Y-%m-%d")


class EquityCalendarStore:
    """Market-wide equity event calendars (earnings, dividends, splits, IPOs)."""

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
        self.catalog = catalog or CatalogStore(self.config.catalog_path)

    def refresh_section(
        self,
        section: str,
        *,
        provider: str = "fmp",
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
    ) -> dict[str, object]:
        if section not in EQUITY_CALENDAR_SECTIONS:
            raise ValueError(f"Unknown equity calendar section: {section}")

        provider = str(provider or "fmp").strip().lower()
        fetch_start = str(start_date or MIN_HISTORICAL_DATE)[:10]
        if not full_refresh:
            state = self.catalog.get(
                symbol=EQUITY_CALENDAR_BUNDLE_SYMBOL,
                section=section,
                provider=provider,
            )
            if state is not None and state.max_date:
                fetch_start = (
                    pd.Timestamp(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
                ).strftime("%Y-%m-%d")

        library = provider_library(fundamental_library(section), provider)
        storage_symbol = symbol_provider_key(EQUITY_CALENDAR_BUNDLE_SYMBOL, provider)
        existing = self.backend.read(library, storage_symbol)
        if full_refresh:
            existing = pd.DataFrame()

        incoming = fetch_equity_calendar_range(
            section,
            provider=provider,
            start_date=fetch_start,
            end_date=end_date,
        )
        merged = merge_panel_upsert(existing, incoming)
        if not merged.empty:
            self.backend.write(library, storage_symbol, merged)

        self.catalog.upsert(
            symbol=EQUITY_CALENDAR_BUNDLE_SYMBOL,
            section=section,
            provider=provider,
            min_date=_min_date_text(merged),
            max_date=_max_date_text(merged),
            row_count=int(len(merged)),
            columns_present=[str(column) for column in merged.columns],
        )
        return {
            "section": section,
            "provider": provider,
            "rows": int(len(merged)),
            "fetched_rows": int(len(incoming)),
            "fetch_start": fetch_start,
            "min_date": _min_date_text(merged),
            "max_date": _max_date_text(merged),
        }

    def refresh_all(
        self,
        *,
        provider: str = "fmp",
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
    ) -> dict[str, dict[str, object]]:
        return {
            section: self.refresh_section(
                section,
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                full_refresh=full_refresh,
            )
            for section in EQUITY_CALENDAR_SECTIONS
        }

    def read(
        self,
        section: str,
        *,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        provider = str(provider or "fmp").strip().lower()
        library = provider_library(fundamental_library(section), provider)
        storage_symbol = symbol_provider_key(EQUITY_CALENDAR_BUNDLE_SYMBOL, provider)
        frame = self.backend.read(library, storage_symbol)
        if frame is None or frame.empty:
            return pd.DataFrame()
        out = frame.copy()
        if start is not None:
            out = out.loc[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out.loc[out.index <= pd.Timestamp(end)]
        return out
