from __future__ import annotations

from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import (
    normalize_dated_snapshot_frame,
    normalize_panel_frame,
    normalize_snapshot_frame,
    normalize_vendor_frame,
    symbol_provider_key,
)
from quant_warehouse.ingest.openbb_fetch import fetch_dataframe, fetch_openbb, provider_period
from quant_warehouse.ingest.providers import (
    DEFAULT_FUNDAMENTAL_PROVIDERS,
    validate_fundamental_provider,
)
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.merge import merge_panel_upsert, merge_upsert
from quant_warehouse.warehouse.prices import _slice_dates
from quant_warehouse.warehouse.sections import (
    ALL_FUNDAMENTAL_SECTIONS,
    DATED_SNAPSHOT_SECTIONS,
    ETF_FUNDAMENTAL_SECTIONS,
    EQUITY_FUNDAMENTAL_SECTIONS,
    PANEL_FUNDAMENTAL_SECTIONS,
    PERIOD_FUNDAMENTAL_SECTIONS,
    SNAPSHOT_FUNDAMENTAL_SECTIONS,
    fundamental_library,
    fundamental_period_for_section,
    normalize_fundamental_period,
)


class FundamentalsStore:
    """ArcticDB-backed store with one library per OpenBB fundamental route."""

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
        sections: Sequence[str] | None = None,
        providers: Sequence[str] | None = None,
        period: str = "quarter",
        **fetch_kwargs: object,
    ) -> dict[str, int]:
        symbol = symbol.strip().upper()
        section_list = list(sections or EQUITY_FUNDAMENTAL_SECTIONS)
        provider_list = [validate_fundamental_provider(p) for p in (providers or DEFAULT_FUNDAMENTAL_PROVIDERS)]
        preferred_period = normalize_fundamental_period(period)
        history_floor = self.catalog.equity_historical_start(symbol)
        stats: dict[str, int] = {}

        for section in section_list:
            self._validate_section(section)
            for provider in provider_list:
                key = f"{section}:{provider}"
                kwargs = dict(fetch_kwargs)
                section_period = fundamental_period_for_section(section, preferred=preferred_period)
                if section_period is not None:
                    kwargs.setdefault("period", provider_period(provider, section_period))

                raw = fetch_dataframe(section, symbol=symbol, provider=provider, **kwargs)
                if section in DATED_SNAPSHOT_SECTIONS:
                    frame = normalize_dated_snapshot_frame(raw, section=section)
                elif section in SNAPSHOT_FUNDAMENTAL_SECTIONS:
                    frame = normalize_snapshot_frame(raw)
                elif section in PANEL_FUNDAMENTAL_SECTIONS:
                    frame = normalize_panel_frame(
                        raw,
                        provider=provider,
                        vendor_only_prefix=None,
                        min_date=history_floor,
                    )
                else:
                    frame = normalize_vendor_frame(
                        raw,
                        provider=provider,
                        vendor_only_prefix=None,
                        min_date=history_floor,
                    )

                library = fundamental_library(section)
                storage_symbol = symbol_provider_key(symbol, provider)
                existing = self.backend.read(library, storage_symbol)

                if section in PANEL_FUNDAMENTAL_SECTIONS or section in DATED_SNAPSHOT_SECTIONS:
                    merged = merge_panel_upsert(existing, frame)
                elif isinstance(frame.index, pd.DatetimeIndex):
                    merged = merge_upsert(existing, frame)
                else:
                    merged = frame

                if not merged.empty:
                    self.backend.write(library, storage_symbol, merged)

                min_date = None
                max_date = None
                if not merged.empty and isinstance(merged.index, pd.DatetimeIndex):
                    min_date = merged.index.min().strftime("%Y-%m-%d")
                    max_date = merged.index.max().strftime("%Y-%m-%d")

                self.catalog.upsert(
                    symbol=symbol,
                    section=section,
                    provider=provider,
                    min_date=min_date,
                    max_date=max_date,
                    row_count=len(merged),
                    columns_present=[c for c in merged.columns],
                )
                stats[key] = len(merged)

        return stats

    def refresh_section(
        self,
        symbol: str,
        section: str,
        *,
        provider: str,
        period: str = "annual",
        **fetch_kwargs: object,
    ) -> dict[str, object]:
        stats = self.refresh(
            symbol,
            sections=[section],
            providers=[provider],
            period=period,
            **fetch_kwargs,
        )
        key = f"{section}:{provider}"
        state = self.catalog.get(symbol=symbol, section=section, provider=provider)
        return {
            "section": section,
            "provider": provider,
            "rows": stats.get(key, 0),
            "min_date": state.min_date if state else None,
            "max_date": state.max_date if state else None,
            "storage_symbol": symbol_provider_key(symbol, provider),
            "library": fundamental_library(section),
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
        self._validate_section(section)
        provider = validate_fundamental_provider(provider)
        storage_symbol = symbol_provider_key(symbol, provider)
        library = fundamental_library(section)
        df = self.backend.read(library, storage_symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        if section in SNAPSHOT_FUNDAMENTAL_SECTIONS:
            out = df.copy()
        else:
            out = _slice_dates(df, start=start, end=end)
        out.attrs["section"] = section
        out.attrs["provider"] = provider
        return out

    def ingest_frame(
        self,
        symbol: str,
        *,
        section: str,
        provider: str,
        frame: pd.DataFrame,
        merge: bool = True,
    ) -> dict[str, object]:
        self._validate_section(section)
        provider = validate_fundamental_provider(provider)
        symbol = symbol.strip().upper()

        if section in DATED_SNAPSHOT_SECTIONS:
            normalized = normalize_dated_snapshot_frame(frame, section=section)
        elif isinstance(frame.index, pd.DatetimeIndex):
            normalized = frame.copy()
        elif section in SNAPSHOT_FUNDAMENTAL_SECTIONS:
            normalized = normalize_snapshot_frame(frame)
        elif section in PANEL_FUNDAMENTAL_SECTIONS:
            normalized = normalize_panel_frame(frame, provider=provider, vendor_only_prefix=None)
        else:
            normalized = normalize_vendor_frame(frame, provider=provider, vendor_only_prefix=None)

        library = fundamental_library(section)
        storage_symbol = symbol_provider_key(symbol, provider)
        merged = normalized
        if merge:
            existing = self.backend.read(library, storage_symbol)
            if section in PANEL_FUNDAMENTAL_SECTIONS or section in DATED_SNAPSHOT_SECTIONS:
                merged = merge_panel_upsert(existing, normalized)
            elif isinstance(normalized.index, pd.DatetimeIndex):
                merged = merge_upsert(existing, normalized)

        rows_written = 0
        if not merged.empty:
            self.backend.write(library, storage_symbol, merged)
            rows_written = len(merged)

        min_date = None
        max_date = None
        if not merged.empty and isinstance(merged.index, pd.DatetimeIndex):
            min_date = merged.index.min().strftime("%Y-%m-%d")
            max_date = merged.index.max().strftime("%Y-%m-%d")

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
            "section": section,
            "rows": rows_written,
            "min_date": min_date,
            "max_date": max_date,
            "storage_symbol": storage_symbol,
            "library": library,
            "storage_backend": self.storage_kind,
        }

    def refresh_transcripts(
        self,
        symbol: str,
        *,
        provider: str = "fmp",
        start_year: int = 2005,
        end_year: int | None = None,
        quarters: Sequence[int] = (1, 2, 3, 4),
    ) -> dict[str, object]:
        """Fetch quarterly earnings transcripts and merge into a dated panel."""
        symbol = symbol.strip().upper()
        provider = validate_fundamental_provider(provider)
        end_year = int(end_year or pd.Timestamp.utcnow().year)
        frames: list[pd.DataFrame] = []
        fetched_periods = 0
        for year in range(int(start_year), end_year + 1):
            for quarter in quarters:
                try:
                    result = fetch_openbb(
                        "transcript",
                        symbol=symbol,
                        provider=provider,
                        year=year,
                        quarter=int(quarter),
                    )
                except Exception:
                    continue
                if result.df is None or result.df.empty:
                    continue
                frames.append(result.df.copy())
                fetched_periods += 1

        if not frames:
            return {
                "symbol": symbol,
                "provider": provider,
                "rows": 0,
                "fetched_periods": 0,
            }

        combined = pd.concat(frames, ignore_index=True)
        stats = self.ingest_frame(
            symbol,
            section="transcript",
            provider=provider,
            frame=combined,
            merge=True,
        )
        stats["fetched_periods"] = fetched_periods
        return stats

    def list_sections(self, symbol: str) -> list[str]:
        rows = self.catalog.list_symbol(symbol.strip().upper())
        return sorted({row.section for row in rows if row.section in ALL_FUNDAMENTAL_SECTIONS})

    @staticmethod
    def _validate_section(section: str) -> None:
        if section not in ALL_FUNDAMENTAL_SECTIONS:
            allowed = ", ".join(sorted(ALL_FUNDAMENTAL_SECTIONS))
            raise ValueError(f"Unknown fundamental section '{section}'. Supported: {allowed}")