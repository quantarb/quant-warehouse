from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from time import sleep

import polars as pl

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
    "ratings_historical": "equity.fundamental.ratings_historical",
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
    "options_eod": "derivatives.options.chains",
    "company_news": "news.company",
}

EQUITY_FUNDAMENTAL_ROUTE_SECTIONS: tuple[str, ...] = EQUITY_FUNDAMENTAL_SECTIONS
ETF_FUNDAMENTAL_ROUTE_SECTIONS: tuple[str, ...] = ETF_FUNDAMENTAL_SECTIONS


@dataclass(frozen=True)
class OpenBBFetchResult:
    section: str
    symbol: str
    provider_requested: str
    provider_used: str
    df: pl.DataFrame
    records: tuple[dict[str, Any], ...]


def provider_period(provider: str, period: str) -> str:
    if provider == "sec" and period == "quarter":
        return "quarterly"
    return period


def _is_empty_fetch_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "[empty]" in message
        or "no results found" in message
        or "no data found" in message
    )


def _call_route(route: str, *, symbol: str | None, provider: str, **kwargs: Any):
    try:
        from openbb import obb
    except ImportError as exc:
        raise ImportError("OpenBB is a required quant-warehouse dependency; reinstall quant-warehouse.") from exc

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


def _call_route_with_retries(route: str, *, symbol: str, provider: str, **kwargs: Any):
    """Retry transient transport failures, retaining all other provider errors."""
    attempts = 5
    for attempt in range(attempts):
        try:
            return _call_route(route, symbol=symbol, provider=provider, **kwargs)
        except Exception as exc:
            transport_failure = isinstance(exc, (TimeoutError, ConnectionError)) or any(
                name in str(exc) for name in ("TimeoutError", "ClientConnectorError", "ServerDisconnectedError")
            )
            if not transport_failure or attempt == attempts - 1:
                raise
            sleep(1.0 * (2 ** attempt))


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

    call_kwargs = dict(kwargs)
    try:
        result = _call_route_with_retries(route, symbol=symbol, provider=provider, **call_kwargs)
    except Exception as exc:
        if _is_empty_fetch_error(exc):
            return OpenBBFetchResult(
                section=section,
                symbol=symbol.strip().upper(),
                provider_requested=str(provider).strip().lower(),
                provider_used=str(provider).strip().lower(),
                df=pl.DataFrame(),
                records=(),
            )
        raise
    records = _result_records(result)
    df = _records_frame(records)

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
) -> pl.DataFrame:
    return fetch_openbb(section, symbol=symbol, provider=provider, **kwargs).df


def fetch_route_dataframe(
    route: str,
    *,
    provider: str,
    **kwargs: Any,
) -> pl.DataFrame:
    """Fetch a route that does not require a symbol (calendars, search, etc.)."""
    result = _call_route(route, symbol=None, provider=provider, **kwargs)
    return _records_frame(_result_records(result))


def _result_records(result):
    records=[]
    for item in list(getattr(result, "results", None) or []):
        if hasattr(item, "model_dump"):
            records.append(item.model_dump())
        elif isinstance(item, dict):
            records.append(dict(item))
        else:
            raise TypeError("OpenBB results must contain structured row records")
    return records


def _records_frame(records):
    """Build Polars directly; SDK to_polars currently round-trips through pandas."""
    def value(item):
        if isinstance(item,datetime):
            return item.astimezone(timezone.utc).replace(tzinfo=None) if item.tzinfo else item
        if isinstance(item,date):return datetime.combine(item,datetime.min.time())
        if isinstance(item,dict):return {k:value(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [value(v) for v in item]
        return item
    return pl.from_dicts([value(row) for row in records],infer_schema_length=None,strict=False) if records else pl.DataFrame()
