from __future__ import annotations

from datetime import timedelta
from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore, SymbolProfile
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import normalize_prices, symbol_provider_key
from quant_warehouse.ingest.openbb_fetch import fetch_dataframe, fetch_openbb
from quant_warehouse.ingest.providers import DEFAULT_PRICE_PROVIDERS, validate_price_provider
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.merge import merge_upsert
from quant_warehouse.warehouse.prices import _slice_dates
from quant_warehouse.warehouse.fundamentals import FundamentalsStore
from quant_warehouse.warehouse.sections import (
    ETF_FUNDAMENTAL_SECTIONS,
    ETF_PRICES_LIBRARY,
    ETF_PRICES_SECTION,
    ETF_PROFILE_SECTION,
)

GAP_OVERLAP_DAYS = 5


class EtfStore:
    """ArcticDB-backed ETF OHLCV and profile store using OpenBB etf.* routes."""

    def __init__(
        self,
        config: WarehouseConfig | None = None,
        *,
        backend: StorageBackend | None = None,
        catalog: CatalogStore | None = None,
        fundamentals: FundamentalsStore | None = None,
    ) -> None:
        self.config = config or WarehouseConfig.from_env()
        self.config.ensure_dirs()
        self.backend: ArcticBackend = backend or open_backend(self.config)
        self.storage_kind = "arctic"
        self.catalog = catalog or CatalogStore(self.config.catalog_path)
        self.fundamentals = fundamentals or FundamentalsStore(
            self.config,
            backend=self.backend,
            catalog=self.catalog,
        )

    def refresh_prices(
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

            raw = fetch_dataframe("etf_prices", symbol=symbol, provider=provider, **kwargs)
            frame = normalize_prices(raw, provider=provider)
            if frame.empty and fetch_start and not full_refresh and end_date:
                state = self.catalog.get(symbol=symbol, section=ETF_PRICES_SECTION, provider=provider)
                if state is not None and state.max_date:
                    from datetime import timedelta

                    wider_start = pd.Timestamp(state.max_date) - timedelta(days=30)
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["start_date"] = wider_start.strftime("%Y-%m-%d")
                    raw = fetch_dataframe("etf_prices", symbol=symbol, provider=provider, **retry_kwargs)
                    frame = normalize_prices(raw, provider=provider)
                    if not frame.empty:
                        fetch_start = retry_kwargs["start_date"]
            storage_symbol = symbol_provider_key(symbol, provider)

            existing = self.backend.read(ETF_PRICES_LIBRARY, storage_symbol)
            merged = merge_upsert(existing, frame)
            rows_written = 0
            if not merged.empty:
                self.backend.write(ETF_PRICES_LIBRARY, storage_symbol, merged)
                rows_written = len(merged)

            min_date = None
            max_date = None
            if not merged.empty:
                min_date = merged.index.min().strftime("%Y-%m-%d")
                max_date = merged.index.max().strftime("%Y-%m-%d")

            self.catalog.upsert(
                symbol=symbol,
                section=ETF_PRICES_SECTION,
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

    def ingest_prices_frame(
        self,
        symbol: str,
        *,
        provider: str,
        frame: pd.DataFrame,
        merge: bool = True,
    ) -> dict[str, object]:
        symbol = symbol.strip().upper()
        provider = validate_price_provider(provider)
        normalized = normalize_prices(frame, provider=provider)
        storage_symbol = symbol_provider_key(symbol, provider)

        merged = normalized
        if merge:
            existing = self.backend.read(ETF_PRICES_LIBRARY, storage_symbol)
            merged = merge_upsert(existing, normalized)

        rows_written = 0
        if not merged.empty:
            self.backend.write(ETF_PRICES_LIBRARY, storage_symbol, merged)
            rows_written = len(merged)

        min_date = None
        max_date = None
        if not merged.empty:
            min_date = merged.index.min().strftime("%Y-%m-%d")
            max_date = merged.index.max().strftime("%Y-%m-%d")

        self.catalog.upsert(
            symbol=symbol,
            section=ETF_PRICES_SECTION,
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

    def read_prices(
        self,
        symbol: str,
        *,
        provider: str = "yfinance",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        provider = validate_price_provider(provider)
        storage_symbol = symbol_provider_key(symbol, provider)
        df = self.backend.read(ETF_PRICES_LIBRARY, storage_symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        return _slice_dates(df, start=start, end=end)

    def refresh_profile(self, symbol: str, *, provider: str) -> dict[str, object]:
        symbol = symbol.strip().upper()
        provider = validate_price_provider(provider)
        result = fetch_openbb("etf_profile", symbol=symbol, provider=provider)
        record = dict(result.records[0]) if result.records else {}
        if not record and not result.df.empty:
            record = result.df.iloc[0].to_dict()
        self.catalog.upsert_etf_profile(
            symbol=symbol,
            provider=provider,
            source_provider=result.provider_used,
            payload=record,
        )
        return {
            "symbol": symbol,
            "provider_requested": result.provider_requested,
            "source_provider": result.provider_used,
            "fields_populated": len([key for key, value in record.items() if value is not None]),
        }

    def read_profile(self, symbol: str, *, provider: str) -> SymbolProfile | None:
        return self.catalog.get_etf_profile(symbol=symbol, provider=validate_price_provider(provider))

    def refresh_fundamentals(
        self,
        symbol: str,
        *,
        sections: Sequence[str] | None = None,
        providers: Sequence[str] | None = None,
        period: str = "annual",
        **fetch_kwargs: object,
    ) -> dict[str, int]:
        section_list = list(sections or ETF_FUNDAMENTAL_SECTIONS)
        for section in section_list:
            if section not in ETF_FUNDAMENTAL_SECTIONS:
                raise ValueError(f"Unknown ETF fundamental section: {section}")
        return self.fundamentals.refresh(
            symbol,
            sections=section_list,
            providers=providers,
            period=period,
            **fetch_kwargs,
        )

    def read_fundamentals(
        self,
        symbol: str,
        *,
        section: str,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if section not in ETF_FUNDAMENTAL_SECTIONS:
            raise ValueError(f"Unknown ETF fundamental section: {section}")
        return self.fundamentals.read(
            symbol,
            section=section,
            provider=provider,
            start=start,
            end=end,
        )

    def refresh_nport_disclosure_history(
        self,
        symbol: str,
        *,
        provider: str = "fmp",
        start_year: int = 2019,
        end_year: int | None = None,
        quarters: Sequence[int] = (1, 2, 3, 4),
    ) -> dict[str, object]:
        """Fetch quarterly ETF N-PORT filings and merge into a dated panel."""
        symbol = symbol.strip().upper()
        provider = validate_price_provider(provider)
        end_year = int(end_year or pd.Timestamp.utcnow().year)
        frames: list[pd.DataFrame] = []
        fetched_periods = 0
        for year in range(int(start_year), end_year + 1):
            for quarter in quarters:
                try:
                    result = fetch_openbb(
                        "etf_nport_disclosure",
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
        stats = self.fundamentals.ingest_frame(
            symbol,
            section="etf_nport_disclosure",
            provider=provider,
            frame=combined,
            merge=True,
        )
        stats["fetched_periods"] = fetched_periods
        return stats

    def _gap_fill_start(self, symbol: str, provider: str) -> str | None:
        state = self.catalog.get(symbol=symbol, section=ETF_PRICES_SECTION, provider=provider)
        if state is None or not state.max_date:
            return None
        resume = pd.Timestamp(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
        return resume.strftime("%Y-%m-%d")