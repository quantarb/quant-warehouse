from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Sequence

import polars as pl

from quant_warehouse.catalog.store import CatalogStore, SymbolProfile
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import normalize_prices, symbol_provider_key
from quant_warehouse.ingest.openbb_fetch import fetch_dataframe, fetch_openbb
from quant_warehouse.ingest.providers import DEFAULT_PRICE_PROVIDERS, validate_price_provider
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.merge import merge_upsert
from quant_warehouse.warehouse.prices import _slice_dates
from quant_warehouse.warehouse.fundamentals import FundamentalsStore
from quant_warehouse.warehouse.storage import read_provider_frame, provider_library
from quant_warehouse.warehouse.sections import (
    ETF_FUNDAMENTAL_SECTIONS,
    ETF_PRICES_LIBRARY,
    ETF_PRICES_SECTION,
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
            if frame.is_empty() and fetch_start and not full_refresh and end_date:
                state = self.catalog.get(symbol=symbol, section=ETF_PRICES_SECTION, provider=provider)
                if state is not None and state.max_date:
                    wider_start = datetime.fromisoformat(state.max_date) - timedelta(days=30)
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["start_date"] = wider_start.strftime("%Y-%m-%d")
                    raw = fetch_dataframe("etf_prices", symbol=symbol, provider=provider, **retry_kwargs)
                    frame = normalize_prices(raw, provider=provider)
                    if not frame.is_empty():
                        fetch_start = retry_kwargs["start_date"]
            storage_symbol = symbol_provider_key(symbol, provider)

            library = provider_library(ETF_PRICES_LIBRARY, provider)
            existing = read_provider_frame(
                self.backend,
                base_library=ETF_PRICES_LIBRARY,
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
                "library": library,
                "fetch_start": fetch_start,
                "storage_backend": self.storage_kind,
            }

        return stats

    def ingest_prices_frame(
        self,
        symbol: str,
        *,
        provider: str,
        frame: pl.DataFrame,
        merge: bool = True,
    ) -> dict[str, object]:
        symbol = symbol.strip().upper()
        provider = validate_price_provider(provider)
        normalized = normalize_prices(frame, provider=provider)
        storage_symbol = symbol_provider_key(symbol, provider)

        merged = normalized
        library = provider_library(ETF_PRICES_LIBRARY, provider)
        if merge:
            existing = read_provider_frame(
                self.backend,
                base_library=ETF_PRICES_LIBRARY,
                provider=provider,
                symbol=storage_symbol,
            )
            merged = merge_upsert(existing, normalized)

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
            "library": library,
            "storage_backend": self.storage_kind,
        }

    def read_prices(
        self,
        symbol: str,
        *,
        provider: str = "yfinance",
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        provider = validate_price_provider(provider)
        storage_symbol = symbol_provider_key(symbol, provider)
        df = read_provider_frame(
            self.backend,
            base_library=ETF_PRICES_LIBRARY,
            provider=provider,
            symbol=storage_symbol,
        )
        if df is None or df.is_empty():
            return pl.DataFrame()
        return _slice_dates(df, start=start, end=end)

    def refresh_profile(self, symbol: str, *, provider: str) -> dict[str, object]:
        symbol = symbol.strip().upper()
        provider = validate_price_provider(provider)
        result = fetch_openbb("etf_profile", symbol=symbol, provider=provider)
        record = dict(result.records[0]) if result.records else {}
        if not record and not result.df.is_empty():
            record = result.df.row(0, named=True)
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
        output_format: Literal["polars"] = "polars",
    ) -> pl.DataFrame:
        if section not in ETF_FUNDAMENTAL_SECTIONS:
            raise ValueError(f"Unknown ETF fundamental section: {section}")
        return self.fundamentals.read(
            symbol,
            section=section,
            provider=provider,
            start=start,
            end=end,
            output_format=output_format,
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
        end_year = int(end_year or datetime.utcnow().year)
        frames: list[pl.DataFrame] = []
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
                if result.df is None or result.df.is_empty():
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

        combined = pl.concat(frames, how="diagonal_relaxed")
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
        resume = datetime.fromisoformat(state.max_date) - timedelta(days=GAP_OVERLAP_DAYS)
        return resume.strftime("%Y-%m-%d")
