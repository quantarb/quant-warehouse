from __future__ import annotations

from datetime import timedelta
from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.macro_fetch import (
    fetch_economic_indicator_series,
    fetch_economy_calendar_range,
    fetch_risk_premium_snapshot,
    fetch_treasury_rates_wide,
    fetch_yield_curve_history,
    treasury_series_code,
    yield_curve_series_code,
)
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.merge import merge_panel_upsert, merge_upsert
from quant_warehouse.warehouse.prices import _slice_dates
from quant_warehouse.warehouse.storage import read_provider_frame, provider_library
from quant_warehouse.warehouse.sections import (
    DEFAULT_ECONOMIC_SERIES,
    MACRO_CALENDAR_LIBRARY,
    MACRO_CALENDAR_SECTION,
    MACRO_CALENDAR_SYMBOL,
    MACRO_ECONOMIC_LIBRARY,
    MACRO_ECONOMIC_SECTION,
    MACRO_RISK_PREMIUM_LIBRARY,
    MACRO_RISK_PREMIUM_SECTION,
    MACRO_TREASURY_LIBRARY,
    MACRO_TREASURY_SECTION,
    MACRO_YIELD_CURVE_LIBRARY,
    MACRO_YIELD_CURVE_SECTION,
    MIN_HISTORICAL_DATE,
    RISK_PREMIUM_SYMBOL,
    TREASURY_BUNDLE_SYMBOL,
    YIELD_CURVE_BUNDLE_SYMBOL,
)

GAP_OVERLAP_DAYS = 5


class MacroStore:
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

    def refresh_economic_series(
        self,
        series_name: str,
        *,
        provider: str = "fmp",
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
    ) -> dict[str, object]:
        series_name = str(series_name).strip()
        provider = str(provider or "fmp").strip().lower()
        fetch_start = start_date
        if fetch_start is None and not full_refresh:
            fetch_start = self._gap_fill_start(series_name, MACRO_ECONOMIC_SECTION, provider)

        frame = fetch_economic_indicator_series(
            series_name,
            provider=provider,
            start_date=fetch_start,
            end_date=end_date,
        )
        storage_symbol = symbol_provider_key(series_name, provider)
        library = provider_library(MACRO_ECONOMIC_LIBRARY, provider)
        existing = read_provider_frame(
            self.backend,
            base_library=MACRO_ECONOMIC_LIBRARY,
            provider=provider,
            symbol=storage_symbol,
        )
        merged = merge_upsert(existing, frame)
        if not merged.empty:
            self.backend.write(library, storage_symbol, merged)
        self._upsert_catalog_state(
            symbol=series_name,
            section=MACRO_ECONOMIC_SECTION,
            provider=provider,
            frame=merged,
        )
        return {
            "series": series_name,
            "provider": provider,
            "rows": int(len(merged)),
            "fetched_rows": int(len(frame)),
            "min_date": _min_date_text(merged),
            "max_date": _max_date_text(merged),
            "fetch_start": fetch_start,
            "library": library,
        }

    def refresh_treasury_rates(
        self,
        *,
        provider: str = "fmp",
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
    ) -> dict[str, object]:
        provider = str(provider or "fmp").strip().lower()
        fetch_start = start_date
        if fetch_start is None and not full_refresh:
            fetch_start = self._gap_fill_start(TREASURY_BUNDLE_SYMBOL, MACRO_TREASURY_SECTION, provider)

        wide = fetch_treasury_rates_wide(
            provider=provider,
            start_date=fetch_start,
            end_date=end_date,
        )
        updated: dict[str, int] = {}
        for column in [col for col in wide.columns]:
            code = treasury_series_code(column)
            series_frame = wide[[column]].rename(columns={column: "value"})
            storage_symbol = symbol_provider_key(code, provider)
            library = provider_library(MACRO_TREASURY_LIBRARY, provider)
            existing = read_provider_frame(
                self.backend,
                base_library=MACRO_TREASURY_LIBRARY,
                provider=provider,
                symbol=storage_symbol,
            )
            merged = merge_upsert(existing, series_frame)
            if not merged.empty:
                self.backend.write(library, storage_symbol, merged)
            self._upsert_catalog_state(
                symbol=code,
                section=MACRO_TREASURY_SECTION,
                provider=provider,
                frame=merged,
            )
            updated[code] = int(len(merged))

        self._upsert_catalog_state(
            symbol=TREASURY_BUNDLE_SYMBOL,
            section=MACRO_TREASURY_SECTION,
            provider=provider,
            frame=wide.rename(columns={column: treasury_series_code(column) for column in wide.columns}),
        )
        return {
            "provider": provider,
            "series_count": len(updated),
            "rows_by_series": updated,
            "fetch_start": fetch_start,
            "max_date": _max_date_text(wide),
        }

    def refresh_yield_curve_history(
        self,
        *,
        provider: str = "fmp",
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
        step_days: int = 1,
    ) -> dict[str, object]:
        provider = str(provider or "fmp").strip().lower()
        fetch_start = str(start_date or MIN_HISTORICAL_DATE)[:10]
        if not full_refresh:
            state = self.catalog.get(
                symbol=YIELD_CURVE_BUNDLE_SYMBOL,
                section=MACRO_YIELD_CURVE_SECTION,
                provider=provider,
            )
            if state is not None and state.max_date:
                fetch_start = (
                    pd.Timestamp(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
                ).strftime("%Y-%m-%d")

        yield_curve_library = provider_library(MACRO_YIELD_CURVE_LIBRARY, provider)
        existing_bundle = read_provider_frame(
            self.backend,
            base_library=MACRO_YIELD_CURVE_LIBRARY,
            provider=provider,
            symbol=symbol_provider_key(YIELD_CURVE_BUNDLE_SYMBOL, provider),
        )
        existing_dates: set[pd.Timestamp] = set()
        if existing_bundle is not None and not existing_bundle.empty:
            existing_dates = {pd.Timestamp(value).normalize() for value in existing_bundle.index}

        incoming = fetch_yield_curve_history(
            provider=provider,
            start_date=fetch_start,
            end_date=end_date,
            existing_dates=existing_dates if not full_refresh else set(),
            step_days=step_days,
        )
        wide = merge_upsert(existing_bundle, incoming)
        updated: dict[str, int] = {}
        for column in [col for col in wide.columns]:
            code = yield_curve_series_code(column)
            series_frame = wide[[column]].rename(columns={column: "value"})
            storage_symbol = symbol_provider_key(code, provider)
            existing = read_provider_frame(
                self.backend,
                base_library=MACRO_YIELD_CURVE_LIBRARY,
                provider=provider,
                symbol=storage_symbol,
            )
            merged = merge_upsert(existing, series_frame)
            if not merged.empty:
                self.backend.write(yield_curve_library, storage_symbol, merged)
            self._upsert_catalog_state(
                symbol=code,
                section=MACRO_YIELD_CURVE_SECTION,
                provider=provider,
                frame=merged,
            )
            updated[code] = int(len(merged))

        bundle_symbol = symbol_provider_key(YIELD_CURVE_BUNDLE_SYMBOL, provider)
        if not wide.empty:
            self.backend.write(
                yield_curve_library,
                bundle_symbol,
                wide.rename(columns={column: yield_curve_series_code(column) for column in wide.columns}),
            )
        self._upsert_catalog_state(
            symbol=YIELD_CURVE_BUNDLE_SYMBOL,
            section=MACRO_YIELD_CURVE_SECTION,
            provider=provider,
            frame=wide.rename(columns={column: yield_curve_series_code(column) for column in wide.columns}),
        )
        return {
            "provider": provider,
            "series_count": len(updated),
            "rows_by_series": updated,
            "fetched_dates": int(len(incoming)),
            "fetch_start": fetch_start,
            "min_date": _min_date_text(wide),
            "max_date": _max_date_text(wide),
        }

    def refresh_calendar(
        self,
        *,
        provider: str = "fmp",
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
    ) -> dict[str, object]:
        provider = str(provider or "fmp").strip().lower()
        fetch_start = str(start_date or MIN_HISTORICAL_DATE)[:10]
        if not full_refresh:
            state = self.catalog.get(
                symbol=MACRO_CALENDAR_SYMBOL,
                section=MACRO_CALENDAR_SECTION,
                provider=provider,
            )
            if state is not None and state.max_date:
                fetch_start = (
                    pd.Timestamp(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
                ).strftime("%Y-%m-%d")

        storage_symbol = symbol_provider_key(MACRO_CALENDAR_SYMBOL, provider)
        calendar_library = provider_library(MACRO_CALENDAR_LIBRARY, provider)
        existing = read_provider_frame(
            self.backend,
            base_library=MACRO_CALENDAR_LIBRARY,
            provider=provider,
            symbol=storage_symbol,
        )
        if full_refresh:
            existing = pd.DataFrame()

        incoming = fetch_economy_calendar_range(
            provider=provider,
            start_date=fetch_start,
            end_date=end_date,
        )
        merged = merge_panel_upsert(existing, incoming)
        if not merged.empty:
            self.backend.write(calendar_library, storage_symbol, merged)
        self._upsert_catalog_state(
            symbol=MACRO_CALENDAR_SYMBOL,
            section=MACRO_CALENDAR_SECTION,
            provider=provider,
            frame=merged,
        )
        return {
            "provider": provider,
            "rows": int(len(merged)),
            "fetched_rows": int(len(incoming)),
            "fetch_start": fetch_start,
            "min_date": _min_date_text(merged),
            "max_date": _max_date_text(merged),
        }

    def refresh_risk_premium(self, *, provider: str = "fmp") -> dict[str, object]:
        provider = str(provider or "fmp").strip().lower()
        frame = fetch_risk_premium_snapshot(provider=provider)
        storage_symbol = symbol_provider_key(RISK_PREMIUM_SYMBOL, provider)
        library = provider_library(MACRO_RISK_PREMIUM_LIBRARY, provider)
        if not frame.empty:
            snapshot = frame.reset_index()
            as_of = pd.Timestamp.utcnow().normalize()
            snapshot.index = pd.DatetimeIndex([as_of] * len(snapshot))
            snapshot.index.name = "as_of"
            self.backend.write(library, storage_symbol, snapshot)
            frame = snapshot
        self.catalog.upsert(
            symbol=RISK_PREMIUM_SYMBOL,
            section=MACRO_RISK_PREMIUM_SECTION,
            provider=provider,
            min_date=None,
            max_date=None,
            row_count=int(len(frame)),
            columns_present=[str(column) for column in frame.columns],
        )
        return {
            "provider": provider,
            "rows": int(len(frame)),
            "countries": int(len(frame)),
        }

    def refresh(
        self,
        *,
        economic_series: Sequence[str] | None = None,
        include_treasury_rates: bool = True,
        include_yield_curve: bool = False,
        include_calendar: bool = False,
        include_risk_premium: bool = False,
        provider: str = "fmp",
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
    ) -> dict[str, object]:
        provider = str(provider or "fmp").strip().lower()
        stats: dict[str, object] = {"economic": {}, "treasury": {}}
        for series_name in list(economic_series or DEFAULT_ECONOMIC_SERIES):
            stats["economic"][series_name] = self.refresh_economic_series(
                series_name,
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                full_refresh=full_refresh,
            )
        if include_treasury_rates:
            stats["treasury"] = self.refresh_treasury_rates(
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                full_refresh=full_refresh,
            )
        if include_yield_curve:
            stats["yield_curve"] = self.refresh_yield_curve_history(
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                full_refresh=full_refresh,
            )
        if include_calendar:
            stats["calendar"] = self.refresh_calendar(
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                full_refresh=full_refresh,
            )
        if include_risk_premium:
            stats["risk_premium"] = self.refresh_risk_premium(provider=provider)
        return stats

    def read_series(
        self,
        series_code: str,
        *,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        series_code = str(series_code).strip()
        provider = str(provider or "fmp").strip().lower()
        lowered = series_code.lower()
        if lowered.startswith("macro__ust_"):
            library = MACRO_TREASURY_LIBRARY
        elif lowered.startswith("macro__yc_"):
            library = MACRO_YIELD_CURVE_LIBRARY
        else:
            library = MACRO_ECONOMIC_LIBRARY
        storage_symbol = symbol_provider_key(series_code, provider)
        frame = read_provider_frame(
            self.backend,
            base_library=library,
            provider=provider,
            symbol=storage_symbol,
        )
        if frame is None or frame.empty:
            return pd.Series(dtype=float)
        sliced = _slice_dates(frame, start=start, end=end)
        if sliced.empty:
            return pd.Series(dtype=float)
        column = "value" if "value" in sliced.columns else sliced.columns[0]
        series = pd.to_numeric(sliced[column], errors="coerce")
        series.index = pd.DatetimeIndex(sliced.index).normalize()
        return series.dropna()

    def read_panel(
        self,
        series_codes: Sequence[str],
        *,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for code in series_codes:
            series_name = str(code).strip()
            if not series_name:
                continue
            series = self.read_series(series_name, provider=provider, start=start, end=end)
            if not series.empty:
                columns[series_name] = series
        if not columns:
            return pd.DataFrame()
        panel = pd.DataFrame(columns).sort_index()
        panel.index = pd.DatetimeIndex(panel.index).normalize()
        return panel

    def read_calendar(
        self,
        *,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        provider = str(provider or "fmp").strip().lower()
        storage_symbol = symbol_provider_key(MACRO_CALENDAR_SYMBOL, provider)
        frame = read_provider_frame(
            self.backend,
            base_library=MACRO_CALENDAR_LIBRARY,
            provider=provider,
            symbol=storage_symbol,
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        return _slice_dates(frame, start=start, end=end)

    def read_risk_premium(self, *, provider: str = "fmp") -> pd.DataFrame:
        provider = str(provider or "fmp").strip().lower()
        storage_symbol = symbol_provider_key(RISK_PREMIUM_SYMBOL, provider)
        frame = read_provider_frame(
            self.backend,
            base_library=MACRO_RISK_PREMIUM_LIBRARY,
            provider=provider,
            symbol=storage_symbol,
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        latest = frame.loc[frame.index.max()]
        if isinstance(latest, pd.Series):
            return latest.to_frame().T
        return latest.copy()

    def list_treasury_series_codes(self, *, provider: str = "fmp") -> list[str]:
        provider = str(provider or "fmp").strip().lower()
        return sorted(
            {
                row.symbol.lower()
                for row in self.catalog.list_section(MACRO_TREASURY_SECTION, provider=provider)
                if row.symbol != TREASURY_BUNDLE_SYMBOL and int(row.row_count) > 0
            }
        )

    def _gap_fill_start(self, symbol: str, section: str, provider: str) -> str | None:
        state = self.catalog.get(symbol=symbol, section=section, provider=provider)
        if state is None or not state.max_date:
            return None
        resume = pd.Timestamp(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
        return resume.strftime("%Y-%m-%d")

    def _upsert_catalog_state(
        self,
        *,
        symbol: str,
        section: str,
        provider: str,
        frame: pd.DataFrame,
    ) -> None:
        min_date = _min_date_text(frame)
        max_date = _max_date_text(frame)
        row_count = int(len(frame))
        columns_present = [str(column) for column in frame.columns]
        self.catalog.upsert(
            symbol=symbol,
            section=section,
            provider=provider,
            min_date=min_date,
            max_date=max_date,
            row_count=row_count,
            columns_present=columns_present,
        )


def _min_date_text(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    return frame.index.min().strftime("%Y-%m-%d")


def _max_date_text(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    return frame.index.max().strftime("%Y-%m-%d")
