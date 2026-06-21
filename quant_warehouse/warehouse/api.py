from __future__ import annotations

from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.backend import StorageBackend, open_backend
from quant_warehouse.warehouse.equity_calendar import EquityCalendarStore
from quant_warehouse.warehouse.etf import EtfStore
from quant_warehouse.warehouse.fundamentals import FundamentalsStore
from quant_warehouse.warehouse.prices import PricesStore
from quant_warehouse.warehouse.macro import MacroStore
from quant_warehouse.warehouse.market_prices import MarketPricesStore
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
        self.prices = PricesStore(self.config, backend=backend)
        self.profiles = ProfileStore(self.config, catalog=self.prices.catalog)
        self.fundamentals = FundamentalsStore(
            self.config,
            backend=self.prices.backend,
            catalog=self.prices.catalog,
        )
        self.etf = EtfStore(
            self.config,
            backend=self.prices.backend,
            catalog=self.prices.catalog,
            fundamentals=self.fundamentals,
        )
        self.macro = MacroStore(
            self.config,
            backend=self.prices.backend,
            catalog=self.prices.catalog,
        )
        self.market_prices = MarketPricesStore(
            self.config,
            backend=self.prices.backend,
            catalog=self.prices.catalog,
        )
        self.equity_calendar = EquityCalendarStore(
            self.config,
            backend=self.prices.backend,
            catalog=self.prices.catalog,
        )
        self.backend = self.prices.backend
        self.catalog = self.prices.catalog

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
        stats: dict[str, int] = {}

        price_sections = [s for s in section_list if s == "prices"]
        profile_sections = [s for s in section_list if s == "profile"]
        equity_fundamental_sections = [s for s in section_list if s in EQUITY_FUNDAMENTAL_SECTIONS]
        etf_fundamental_sections = [s for s in section_list if s in ETF_FUNDAMENTAL_SECTIONS]

        for section in price_sections:
            for provider in provider_list:
                price_stats = self.prices.refresh(symbol, providers=[provider])
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
    ) -> dict[str, dict[str, object]]:
        return self.prices.refresh(
            symbol,
            providers=providers,
            start_date=start_date,
            end_date=end_date,
            full_refresh=full_refresh,
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
    ) -> pd.DataFrame:
        return self.prices.read(symbol, provider=provider, start=start, end=end)

    def read_fundamentals(
        self,
        symbol: str,
        *,
        section: str = "income",
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if section in ETF_FUNDAMENTAL_SECTIONS:
            return self.etf.read_fundamentals(
                symbol,
                section=section,
                provider=provider,
                start=start,
                end=end,
            )
        return self.fundamentals.read(
            symbol,
            section=section,
            provider=provider,
            start=start,
            end=end,
        )

    def read_features(
        self,
        symbol: str,
        *,
        recipe: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        storage_symbol = f"{symbol.strip().upper()}__{recipe}"
        df = self.backend.read("features", storage_symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        return _slice_dates(df, start=start, end=end)

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
    ) -> pd.DataFrame:
        return self.macro.read_panel(series_codes, provider=provider, start=start, end=end)

    def status(self, symbol: str) -> list:
        return self.catalog.list_symbol(symbol)


def _slice_dates(
    df: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out