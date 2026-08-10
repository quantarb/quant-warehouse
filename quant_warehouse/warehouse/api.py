from __future__ import annotations

import threading
from typing import Literal, Sequence

import polars as pl

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.backend import StorageBackend, open_backend
from quant_warehouse.warehouse.equity_calendar import EquityCalendarStore
from quant_warehouse.warehouse.etf import EtfStore
from quant_warehouse.warehouse.fundamentals import FundamentalsStore
from quant_warehouse.warehouse.prices import EQUITY_PRICE_ADJUSTMENT, PricesStore
from quant_warehouse.warehouse.macro import MacroStore
from quant_warehouse.warehouse.market_prices import MarketPricesStore
from quant_warehouse.warehouse.news import CompanyNewsStore
from quant_warehouse.warehouse.profile import ProfileStore
from quant_warehouse.warehouse.sections import EQUITY_FUNDAMENTAL_SECTIONS, ETF_FUNDAMENTAL_SECTIONS

DEFAULT_SECTIONS = ("prices",)
DEFAULT_PROVIDERS = ("fmp",)


class Warehouse:
    def __init__(
        self,
        config: WarehouseConfig | None = None,
        *,
        backend: StorageBackend | None = None,
    ) -> None:
        self.config = config or WarehouseConfig.from_env()
        self.config.ensure_dirs()
        self.storage_lock = threading.RLock()
        shared_backend = backend or open_backend(self.config, storage_lock=self.storage_lock)
        shared_catalog = CatalogStore(self.config.catalog_path, storage_lock=self.storage_lock)
        self.prices = PricesStore(self.config, backend=shared_backend, catalog=shared_catalog)
        self.profiles = ProfileStore(self.config, catalog=shared_catalog)
        self.fundamentals = FundamentalsStore(
            self.config,
            backend=shared_backend,
            catalog=shared_catalog,
        )
        self.etf = EtfStore(
            self.config,
            backend=shared_backend,
            catalog=shared_catalog,
            fundamentals=self.fundamentals,
        )
        self.macro = MacroStore(
            self.config,
            backend=shared_backend,
            catalog=shared_catalog,
        )
        self.market_prices = MarketPricesStore(
            self.config,
            backend=shared_backend,
            catalog=shared_catalog,
        )
        self.equity_calendar = EquityCalendarStore(
            self.config,
            backend=shared_backend,
            catalog=shared_catalog,
        )
        self.news = CompanyNewsStore(
            self.config,
            backend=shared_backend,
            catalog=shared_catalog,
        )
        self.backend = shared_backend
        self.catalog = shared_catalog

    def refresh(
        self,
        symbol: str,
        *,
        sections: Sequence[str] | None = None,
        providers: Sequence[str] | None = None,
        period: str = "annual",
        **fetch_kwargs: object,
    ) -> dict[str, int]:
        symbol = symbol.strip().upper()
        section_list = list(sections or DEFAULT_SECTIONS)
        provider_list = list(providers or DEFAULT_PROVIDERS)
        price_adjustment = str(fetch_kwargs.pop("adjustment", EQUITY_PRICE_ADJUSTMENT))
        stats: dict[str, int] = {}

        price_sections = [s for s in section_list if s == "prices"]
        profile_sections = [s for s in section_list if s == "profile"]
        equity_fundamental_sections = [s for s in section_list if s in EQUITY_FUNDAMENTAL_SECTIONS]
        etf_fundamental_sections = [s for s in section_list if s in ETF_FUNDAMENTAL_SECTIONS]

        for section in price_sections:
            for provider in provider_list:
                price_stats = self.prices.refresh(
                    symbol,
                    providers=[provider],
                    adjustment=price_adjustment,
                )
                stats[f"{section}:{provider}"] = int(price_stats[provider]["rows"])

        for section in profile_sections:
            for provider in provider_list:
                self.profiles.refresh(symbol, provider=provider)
                state = self.catalog.get(symbol=symbol, section="profile", provider=provider)
                stats[f"{section}:{provider}"] = int(state.row_count) if state else 0

        if equity_fundamental_sections:
            stats.update(
                self.fundamentals.refresh(
                    symbol,
                    sections=equity_fundamental_sections,
                    providers=provider_list,
                    period=period,
                    **fetch_kwargs,
                )
            )

        if etf_fundamental_sections:
            stats.update(
                self.etf.refresh_fundamentals(
                    symbol,
                    sections=etf_fundamental_sections,
                    providers=provider_list,
                    period=period,
                    **fetch_kwargs,
                )
            )

        unknown = [
            s
            for s in section_list
            if s
            not in (
                {"prices", "profile"}
                | set(EQUITY_FUNDAMENTAL_SECTIONS)
                | set(ETF_FUNDAMENTAL_SECTIONS)
            )
        ]
        if unknown:
            raise ValueError(f"Unknown sections: {', '.join(unknown)}")

        return stats

    def refresh_prices(
        self,
        symbol: str,
        *,
        providers: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        full_refresh: bool = False,
        adjustment: str = EQUITY_PRICE_ADJUSTMENT,
    ) -> dict[str, dict[str, object]]:
        return self.prices.refresh(
            symbol,
            providers=providers,
            start_date=start_date,
            end_date=end_date,
            full_refresh=full_refresh,
            adjustment=adjustment,
        )

    def refresh_fundamentals(
        self,
        symbol: str,
        *,
        sections: Sequence[str] | None = None,
        providers: Sequence[str] | None = None,
        period: str = "annual",
        **fetch_kwargs: object,
    ) -> dict[str, int]:
        return self.fundamentals.refresh(
            symbol,
            sections=sections,
            providers=providers,
            period=period,
            **fetch_kwargs,
        )

    def refresh_profile(
        self,
        symbol: str,
        *,
        provider: str = "fmp",
    ) -> dict[str, object]:
        return self.profiles.refresh(symbol, provider=provider)

    def read_profile(self, symbol: str, *, provider: str = "fmp"):
        return self.profiles.read(symbol, provider=provider)

    def read_prices(
        self,
        symbol: str,
        *,
        provider: str = "yfinance",
        start: str | None = None,
        end: str | None = None,
        adjustment: str = EQUITY_PRICE_ADJUSTMENT,
        output_format: Literal["polars"] = "polars",
    ) -> pl.DataFrame:
        return self.prices.read(
            symbol,
            provider=provider,
            start=start,
            end=end,
            adjustment=adjustment,
            output_format=output_format,
        )

    def read_fundamentals(
        self,
        symbol: str,
        *,
        section: str = "income",
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        output_format: Literal["polars"] = "polars",
    ) -> pl.DataFrame:
        if section in ETF_FUNDAMENTAL_SECTIONS:
            return self.etf.read_fundamentals(
                symbol,
                section=section,
                provider=provider,
                start=start,
                end=end,
                output_format=output_format,
            )
        return self.fundamentals.read(
            symbol,
            section=section,
            provider=provider,
            start=start,
            end=end,
            period=period,
            output_format=output_format,
        )

    def read_features(
        self,
        symbol: str,
        *,
        recipe: str,
        start: str | None = None,
        end: str | None = None,
        output_format: Literal["polars"] = "polars",
    ) -> pl.DataFrame:
        storage_symbol = f"{symbol.strip().upper()}__{recipe}"
        df = self.backend.read("features", storage_symbol, output_format=output_format)
        if df is None or df.is_empty():
            return pl.DataFrame()
        return _slice_dates(df, start=start, end=end)

    def read_news(
        self,
        symbol: str,
        *,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
        output_format: Literal["polars"] = "polars",
    ) -> pl.DataFrame:
        return self.news.read(
            symbol, provider=provider, start=start, end=end, output_format=output_format
        )

    def refresh_macro(
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
        return self.macro.refresh(
            economic_series=economic_series,
            include_treasury_rates=include_treasury_rates,
            include_yield_curve=include_yield_curve,
            include_calendar=include_calendar,
            include_risk_premium=include_risk_premium,
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            full_refresh=full_refresh,
        )

    def read_macro_panel(
        self,
        series_codes: Sequence[str],
        *,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
        output_format: Literal["polars"] = "polars",
    ) -> pl.DataFrame:
        return self.macro.read_panel(
            series_codes,
            provider=provider,
            start=start,
            end=end,
            output_format=output_format,
        )

    def read_macro_calendar(
        self,
        *,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
        output_format: Literal["polars"] = "polars",
    ) -> pl.DataFrame:
        """Read stored macro economic-calendar releases."""
        return self.macro.read_calendar(
            provider=provider, start=start, end=end, output_format=output_format
        )

    def status(self, symbol: str) -> list:
        return self.catalog.list_symbol(symbol)


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
        predicate = predicate & (pl.col("date") >= pl.lit(start).str.to_datetime())
    if end is not None:
        predicate = predicate & (pl.col("date") <= pl.lit(end).str.to_datetime())
    return df.filter(predicate)
