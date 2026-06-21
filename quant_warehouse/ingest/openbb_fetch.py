from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_warehouse.ingest.credentials import configure_openbb_credentials
from quant_warehouse.warehouse.sections import (
    EQUITY_FUNDAMENTAL_SECTIONS,
    ETF_FUNDAMENTAL_SECTIONS,
)

SECTION_ROUTES: dict[str, str] = {
    "prices": "equity.price.historical",
    "crypto_prices": "crypto.price.historical",
    "currency_prices": "currency.price.historical",
    "index_prices": "index.price.historical",
    "profile": "equity.profile",
    "etf_prices": "etf.historical",
    "etf_profile": "etf.info",
    "income": "equity.fundamental.income",
    "balance": "equity.fundamental.balance",
    "cash": "equity.fundamental.cash",
    "metrics": "equity.fundamental.metrics",
    "ratios": "equity.fundamental.ratios",
    "income_growth": "equity.fundamental.income_growth",
    "balance_growth": "equity.fundamental.balance_growth",
    "cash_growth": "equity.fundamental.cash_growth",
    "dividends": "equity.fundamental.dividends",
    "historical_eps": "equity.fundamental.historical_eps",
    "historical_splits": "equity.fundamental.historical_splits",
    "reported_financials": "equity.fundamental.reported_financials",
    "revenue_per_geography": "equity.fundamental.revenue_per_geography",
    "revenue_per_segment": "equity.fundamental.revenue_per_segment",
    "trailing_dividend_yield": "equity.fundamental.trailing_dividend_yield",
    "employee_count": "equity.fundamental.employee_count",
    "etf_holdings": "etf.holdings",
    "etf_sectors": "etf.sectors",
    "etf_countries": "etf.countries",
    "etf_equity_exposure": "etf.equity_exposure",
    "etf_nport_disclosure": "etf.nport_disclosure",
    "etf_price_performance": "etf.price_performance",
    "historical_market_cap": "equity.historical_market_cap",
    "esg_score": "equity.fundamental.esg_score",
    "management_compensation": "equity.fundamental.management_compensation",
    "management": "equity.fundamental.management",
    "filings": "equity.fundamental.filings",
    "transcript": "equity.fundamental.transcript",
    "compare_peers": "equity.compare.peers",
    "estimates_historical": "equity.estimates.historical",
    "estimates_consensus": "equity.estimates.consensus",
    "estimates_forward_eps": "equity.estimates.forward_eps",
    "estimates_forward_ebitda": "equity.estimates.forward_ebitda",
    "estimates_price_target": "equity.estimates.price_target",
    "ownership_insider_trading": "equity.ownership.insider_trading",
    "ownership_government_trades": "equity.ownership.government_trades",
    "ownership_institutional": "equity.ownership.institutional",
    "ownership_share_statistics": "equity.ownership.share_statistics",
    "equity_calendar_earnings": "equity.calendar.earnings",
    "equity_calendar_dividend": "equity.calendar.dividend",
    "equity_calendar_splits": "equity.calendar.splits",
    "equity_calendar_ipo": "equity.calendar.ipo",
}

EQUITY_FUNDAMENTAL_ROUTE_SECTIONS: tuple[str, ...] = EQUITY_FUNDAMENTAL_SECTIONS
ETF_FUNDAMENTAL_ROUTE_SECTIONS: tuple[str, ...] = ETF_FUNDAMENTAL_SECTIONS


@dataclass(frozen=True)
class OpenBBFetchResult:
    section: str
    symbol: str
    provider_requested: str
    provider_used: str
    df: pd.DataFrame
    records: tuple[dict[str, Any], ...]


def provider_period(provider: str, period: str) -> str:
    if provider == "sec" and period == "quarter":
        return "quarterly"
    return period


def _call_route(route: str, *, symbol: str | None, provider: str, **kwargs: Any):
    try:
        from openbb import obb
    except ImportError as exc:
        raise ImportError("Install OpenBB: pip install quant-warehouse[openbb]") from exc

    configure_openbb_credentials()

    parts = route.split(".")
    obj = obb
    for part in parts:
        obj = getattr(obj, part)
    call_kwargs = dict(kwargs)
    call_kwargs["provider"] = provider
    if symbol:
        call_kwargs["symbol"] = symbol
    return obj(**call_kwargs)


def fetch_openbb(
    section: str,
    *,
    symbol: str,
    provider: str,
    **kwargs: Any,
) -> OpenBBFetchResult:
    route = SECTION_ROUTES.get(section)
    if route is None:
        raise ValueError(f"Unknown section: {section}")

    result = _call_route(route, symbol=symbol, provider=provider, **kwargs)
    df = result.to_df()
    if df is None:
        df = pd.DataFrame()
    else:
        df = df.copy()

    records: list[dict[str, Any]] = []
    for item in list(getattr(result, "results", None) or []):
        if hasattr(item, "model_dump"):
            records.append(item.model_dump())
        elif isinstance(item, dict):
            records.append(dict(item))

    provider_used = str(getattr(result, "provider", None) or provider).strip().lower()
    return OpenBBFetchResult(
        section=section,
        symbol=symbol.strip().upper(),
        provider_requested=str(provider).strip().lower(),
        provider_used=provider_used,
        df=df,
        records=tuple(records),
    )


def fetch_dataframe(
    section: str,
    *,
    symbol: str,
    provider: str,
    **kwargs: Any,
) -> pd.DataFrame:
    return fetch_openbb(section, symbol=symbol, provider=provider, **kwargs).df


def fetch_route_dataframe(route: str, *, provider: str, **kwargs: Any) -> pd.DataFrame:
    """Fetch a route that does not require a symbol (calendars, search, etc.)."""
    result = _call_route(route, symbol=None, provider=provider, **kwargs)
    df = result.to_df()
    if df is None:
        return pd.DataFrame()
    return df.copy()