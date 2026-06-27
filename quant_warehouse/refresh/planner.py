from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

import pandas as pd

from quant_warehouse.catalog.listing_date import equity_historical_floor
from quant_warehouse.catalog.store import CatalogStore, SectionState
from quant_warehouse.warehouse.sections import (
    DATE_WINDOW_SECTIONS,
    DEFAULT_ECONOMIC_SERIES,
    MACRO_TREASURY_SECTION,
    MIN_HISTORICAL_DATE,
    PANEL_FUNDAMENTAL_SECTIONS,
    PERIOD_FUNDAMENTAL_SECTIONS,
    SNAPSHOT_FUNDAMENTAL_SECTIONS,
    TREASURY_BUNDLE_SYMBOL,
)

EARLY_HISTORY_TOLERANCE_DAYS = 30
COVERAGE_THRESHOLD = 0.75
GAP_OVERLAP_DAYS = 5
PRICE_STALENESS_DAYS = 7
PERIODIC_TAIL_BUFFER_DAYS = 120

@dataclass(frozen=True)
class HistoricalFetchPlan:
    needs_refresh: bool
    mode: str
    reason: str
    fetch_ranges: tuple[tuple[date, date], ...]
    target_start: date
    target_end: date


PANEL_SECTION_DIMENSIONS: dict[str, str] = {
    "revenue_per_segment": "business_line",
    "revenue_per_geography": "region",
    "etf_nport_disclosure": "cusip",
    "ownership_insider_trading": "owner_name",
    "ownership_government_trades": "representative",
    "estimates_price_target": "analyst_name",
    "filings": "report_type",
    "transcript": "quarter",
    "equity_calendar_earnings": "symbol",
    "equity_calendar_dividend": "symbol",
    "equity_calendar_splits": "symbol",
    "equity_calendar_ipo": "symbol",
}


def expected_latest_price_date(*, now: datetime | None = None) -> date:
    """Latest date for which US equity prices are expected to be complete."""
    now_et = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("America/New_York")
    if now_et.weekday() < 5 and now_et.hour >= 17:
        return now_et.date()
    return (now_et.normalize() - pd.offsets.BDay(1)).date()


def catalog_price_max_date(
    catalog: CatalogStore,
    symbol: str,
    provider: str,
    *,
    section: str = "prices",
) -> date | None:
    state = catalog.get(symbol=symbol.strip().upper(), section=section, provider=provider)
    if state is None:
        return None
    return _parse_date(state.max_date)


def symbol_has_fresh_prices(
    catalog: CatalogStore,
    symbol: str,
    providers: Sequence[str],
    *,
    target_end_date: date,
    section: str = "prices",
) -> bool:
    for provider in providers:
        max_date = catalog_price_max_date(catalog, symbol, provider, section=section)
        if max_date is not None and max_date >= target_end_date:
            return True
    return False


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            return None
        if parsed.tzinfo is None:
            return parsed.to_pydatetime().replace(tzinfo=timezone.utc)
        return parsed.to_pydatetime()
    except (TypeError, ValueError):
        return None


def _section_staleness_days(section: str, *, staleness_days: int) -> int:
    if section in {"prices", "etf_prices"}:
        return PRICE_STALENESS_DAYS
    return max(1, int(staleness_days))


def _supports_date_window(section: str) -> bool:
    return section in DATE_WINDOW_SECTIONS


def _coverage_effective_start(
    catalog: CatalogStore,
    symbol: str,
    *,
    target_start: date,
    min_date: date | None,
    is_etf: bool,
) -> date:
    if min_date is None:
        return target_start
    if is_etf:
        return target_start
    if catalog.resolve_equity_ipo_date(symbol):
        return target_start
    if min_date > target_start:
        return min_date
    return target_start


def _coverage_ratio(
    *,
    min_date: date | None,
    max_date: date | None,
    target_start: date,
    target_end: date,
    section: str,
) -> float:
    if min_date is None or max_date is None:
        return 0.0
    effective_start = target_start
    if section in PERIOD_FUNDAMENTAL_SECTIONS and effective_start < min_date <= effective_start + timedelta(days=120):
        effective_start = min_date
    window_days = max(1, (target_end - effective_start).days + 1)
    covered_start = max(min_date, effective_start)
    if section in PERIOD_FUNDAMENTAL_SECTIONS:
        covered_end = min(max_date + timedelta(days=PERIODIC_TAIL_BUFFER_DAYS), target_end)
    else:
        covered_end = min(max_date, target_end)
    if covered_end < covered_start:
        return 0.0
    covered_days = (covered_end - covered_start).days + 1
    return min(1.0, max(0.0, covered_days / window_days))


def historical_fetch_plan(
    catalog: CatalogStore,
    symbol: str,
    section: str,
    provider: str,
    *,
    target_end_date: date | None = None,
    staleness_days: int = 90,
    skip_recent_hours: float = 24.0,
    is_etf: bool = False,
) -> HistoricalFetchPlan:
    """Plan tail/head/full/skip refreshes using catalog min/max dates."""
    symbol = symbol.strip().upper()
    section = str(section).strip()
    target_end = target_end_date or expected_latest_price_date()
    target_start = (
        pd.Timestamp(MIN_HISTORICAL_DATE).date()
        if is_etf
        else _equity_history_start_date(catalog, symbol)
    )
    threshold_days = _section_staleness_days(section, staleness_days=staleness_days)
    state = catalog.get(symbol=symbol, section=section, provider=provider)

    if state is None or int(state.row_count) <= 0:
        return HistoricalFetchPlan(True, "full", "missing", (), target_start, target_end)
    if _is_below_min_historical_date(state.min_date):
        return HistoricalFetchPlan(True, "full", "below_min_historical_date", (), target_start, target_end)
    if section in PANEL_FUNDAMENTAL_SECTIONS and not _panel_has_dimension_column(state, section=section):
        return HistoricalFetchPlan(True, "full", "upgrade_panel_schema", (), target_start, target_end)

    min_date = _parse_date(state.min_date)
    max_date = _parse_date(state.max_date)
    hours = _hours_since(_parse_timestamp(state.last_fetched_at))
    recent_attempt = hours is not None and hours < max(0.0, float(skip_recent_hours))
    has_date_window = _supports_date_window(section)

    if section in SNAPSHOT_FUNDAMENTAL_SECTIONS and section not in DATE_WINDOW_SECTIONS:
        if max_date is None:
            return HistoricalFetchPlan(True, "full", "missing_max_date", (), target_start, target_end)
        age_days = (datetime.now(timezone.utc).date() - max_date).days
        if age_days > threshold_days:
            return HistoricalFetchPlan(True, "full", "stale_max_date", (), target_start, target_end)
        if recent_attempt:
            return HistoricalFetchPlan(False, "skip", "recent_attempt", (), target_start, target_end)
        return HistoricalFetchPlan(False, "skip", "fresh", (), target_start, target_end)

    coverage_start = _coverage_effective_start(
        catalog,
        symbol,
        target_start=target_start,
        min_date=min_date,
        is_etf=is_etf,
    )
    coverage_ratio = _coverage_ratio(
        min_date=min_date,
        max_date=max_date,
        target_start=coverage_start,
        target_end=target_end,
        section=section,
    )
    recent_cutoff = (
        target_end - timedelta(days=max(PERIODIC_TAIL_BUFFER_DAYS, threshold_days))
        if section in PERIOD_FUNDAMENTAL_SECTIONS
        else target_end - timedelta(days=threshold_days)
    )
    is_recent_enough = bool(max_date and max_date >= recent_cutoff)
    head_needed = False
    if has_date_window and min_date is not None and not is_etf:
        ipo_date = catalog.resolve_equity_ipo_date(symbol)
        if ipo_date:
            expected_start = _equity_history_start_date(catalog, symbol)
            head_needed = _equity_history_incomplete(state, expected_start=expected_start)
    elif has_date_window and min_date is not None and is_etf:
        head_needed = (min_date - target_start).days > EARLY_HISTORY_TOLERANCE_DAYS

    if recent_attempt:
        if max_date is not None and max_date >= target_end:
            return HistoricalFetchPlan(False, "skip", "recent_attempt", (), target_start, target_end)
        if coverage_ratio >= COVERAGE_THRESHOLD and not head_needed:
            return HistoricalFetchPlan(False, "skip", "defer_recent_attempt", (), target_start, target_end)

    if has_date_window and max_date is not None and min_date is not None:
        if coverage_ratio >= COVERAGE_THRESHOLD:
            if max_date < target_end:
                tail_start = max(target_start, max_date - timedelta(days=GAP_OVERLAP_DAYS))
                return HistoricalFetchPlan(
                    True,
                    "tail",
                    "stale_max_date",
                    ((tail_start, target_end),),
                    target_start,
                    target_end,
                )
            if not is_recent_enough:
                verify_start = max(target_start, target_end - timedelta(days=threshold_days))
                return HistoricalFetchPlan(
                    True,
                    "tail",
                    "stale_max_date",
                    ((verify_start, target_end),),
                    target_start,
                    target_end,
                )
            if head_needed:
                head_end = min_date - timedelta(days=1)
                if head_end >= target_start:
                    return HistoricalFetchPlan(
                        True,
                        "head",
                        "incomplete_early_history",
                        ((target_start, head_end),),
                        target_start,
                        target_end,
                    )
            if recent_attempt:
                return HistoricalFetchPlan(False, "skip", "recent_attempt", (), target_start, target_end)
            return HistoricalFetchPlan(False, "skip", "fresh", (), target_start, target_end)
        if head_needed:
            head_end = min_date - timedelta(days=1)
            if head_end >= target_start:
                return HistoricalFetchPlan(
                    True,
                    "head",
                    "incomplete_early_history",
                    ((target_start, head_end),),
                    target_start,
                    target_end,
                )

    if max_date is None:
        return HistoricalFetchPlan(True, "full", "missing_max_date", (), target_start, target_end)
    if max_date < target_end:
        if has_date_window:
            tail_start = max(target_start, max_date - timedelta(days=GAP_OVERLAP_DAYS))
            return HistoricalFetchPlan(
                True,
                "tail",
                "stale_max_date",
                ((tail_start, target_end),),
                target_start,
                target_end,
            )
        return HistoricalFetchPlan(True, "full", "stale_max_date", (), target_start, target_end)
    if head_needed and has_date_window:
        head_end = min_date - timedelta(days=1)
        if head_end >= target_start:
            return HistoricalFetchPlan(
                True,
                "head",
                "incomplete_early_history",
                ((target_start, head_end),),
                target_start,
                target_end,
            )
    age_days = (datetime.now(timezone.utc).date() - max_date).days
    if age_days > threshold_days:
        return HistoricalFetchPlan(True, "full", "stale_max_date", (), target_start, target_end)
    if recent_attempt:
        return HistoricalFetchPlan(False, "skip", "recent_attempt", (), target_start, target_end)
    return HistoricalFetchPlan(False, "skip", "fresh", (), target_start, target_end)


def fetch_kwargs_from_plan(
    plan: HistoricalFetchPlan,
    *,
    default_start: str | None = None,
) -> dict[str, object]:
    """Translate a fetch plan into store refresh kwargs."""
    if not plan.needs_refresh:
        return {}
    kwargs: dict[str, object] = {"end_date": plan.target_end.isoformat()}
    if plan.mode == "full":
        kwargs["start_date"] = default_start or plan.target_start.isoformat()
        kwargs["full_refresh"] = plan.reason in {"missing", "below_min_historical_date", "upgrade_panel_schema"}
        return kwargs
    if plan.fetch_ranges:
        start, end = plan.fetch_ranges[0]
        kwargs["start_date"] = start.isoformat()
        kwargs["end_date"] = end.isoformat()
        kwargs["full_refresh"] = False
        return kwargs
    kwargs["start_date"] = default_start or plan.target_start.isoformat()
    kwargs["full_refresh"] = False
    return kwargs


def _hours_since(timestamp: datetime | None, *, now: datetime | None = None) -> float | None:
    if timestamp is None:
        return None
    now_utc = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (now_utc - timestamp).total_seconds() / 3600.0


def price_refresh_needs_update(
    catalog: CatalogStore,
    symbol: str,
    provider: str,
    *,
    target_end_date: date | None = None,
    skip_recent_hours: float = 24.0,
    section: str = "prices",
) -> tuple[bool, str]:
    target_end_date = target_end_date or expected_latest_price_date()
    state = catalog.get(symbol=symbol.strip().upper(), section=section, provider=provider)
    if state is None or int(state.row_count) <= 0:
        return True, "missing"
    if _is_below_min_historical_date(state.min_date):
        return True, "below_min_historical_date"
    max_date = _parse_date(state.max_date)
    if max_date is None:
        return True, "missing_max_date"
    if max_date < target_end_date:
        return True, "stale_max_date"
    hours = _hours_since(_parse_timestamp(state.last_fetched_at))
    if hours is not None and hours < max(0.0, float(skip_recent_hours)):
        return False, "recent_attempt"
    return False, "fresh"


def fundamental_refresh_needs_update(
    catalog: CatalogStore,
    symbol: str,
    section: str,
    provider: str,
    *,
    staleness_days: int = 90,
    skip_recent_hours: float = 24.0,
) -> tuple[bool, str]:
    state = catalog.get(symbol=symbol.strip().upper(), section=section, provider=provider)
    if state is None or int(state.row_count) <= 0:
        return True, "missing"
    max_date = _parse_date(state.max_date)
    if max_date is None:
        return True, "missing_max_date"
    age_days = (datetime.now(timezone.utc).date() - max_date).days
    if age_days > max(1, int(staleness_days)):
        return True, "stale_max_date"
    hours = _hours_since(_parse_timestamp(state.last_fetched_at))
    if hours is not None and hours < max(0.0, float(skip_recent_hours)):
        return False, "recent_attempt"
    return False, "fresh"


def _equity_history_start_date(catalog: CatalogStore, symbol: str) -> date:
    return equity_historical_floor(ipo_date=catalog.resolve_equity_ipo_date(symbol))


def _equity_history_incomplete(state: SectionState, *, expected_start: date) -> bool:
    min_date = _parse_date(state.min_date)
    if min_date is None:
        return False
    return (min_date - expected_start).days > EARLY_HISTORY_TOLERANCE_DAYS


def price_backfill_needs_update(
    catalog: CatalogStore,
    symbol: str,
    provider: str,
    *,
    target_end_date: date | None = None,
    skip_recent_hours: float = 24.0,
    section: str = "prices",
    is_etf: bool = False,
) -> tuple[bool, str]:
    """Skip prices already present unless stale or stored with a pre-floor min_date."""
    plan = historical_fetch_plan(
        catalog,
        symbol,
        section,
        provider,
        target_end_date=target_end_date,
        skip_recent_hours=skip_recent_hours,
        is_etf=is_etf,
    )
    return plan.needs_refresh, plan.reason


def profile_refresh_needs_update(
    catalog: CatalogStore,
    symbol: str,
    provider: str,
    *,
    refresh_days: int = 30,
    is_etf: bool = False,
) -> tuple[bool, str]:
    symbol = symbol.strip().upper()
    profile = catalog.get_etf_profile(symbol=symbol, provider=provider) if is_etf else catalog.get_profile(symbol=symbol, provider=provider)
    if profile is None:
        return True, "missing"
    fetched_at = _parse_timestamp(profile.fetched_at)
    hours = _hours_since(fetched_at)
    if hours is None:
        return True, "missing_fetched_at"
    if hours >= max(1, int(refresh_days)) * 24.0:
        return True, "stale_profile"
    return False, "fresh"


def macro_refresh_needs_update(
    catalog: CatalogStore,
    symbol: str,
    section: str,
    provider: str,
    *,
    target_end_date: date | None = None,
    history_start_date: date | None = None,
    skip_recent_hours: float = 24.0,
) -> tuple[bool, str]:
    target_end_date = target_end_date or expected_latest_price_date()
    state = catalog.get(symbol=symbol.strip().upper(), section=section, provider=provider)
    if state is None or int(state.row_count) <= 0:
        return True, "missing"
    if history_start_date is not None:
        min_date = _parse_date(state.min_date)
        if min_date is not None and min_date > history_start_date:
            return True, "incomplete_early_history"
    if _is_below_min_historical_date(state.min_date):
        return True, "below_min_historical_date"
    max_date = _parse_date(state.max_date)
    if max_date is None:
        return True, "missing_max_date"
    if max_date < target_end_date:
        return True, "stale_max_date"
    hours = _hours_since(_parse_timestamp(state.last_fetched_at))
    if hours is not None and hours < max(0.0, float(skip_recent_hours)):
        return False, "recent_attempt"
    return False, "fresh"


def _min_historical_floor() -> date:
    return pd.Timestamp(MIN_HISTORICAL_DATE).date()


def _is_below_min_historical_date(value: Any) -> bool:
    parsed = _parse_date(value)
    return parsed is not None and parsed < _min_historical_floor()


def _panel_has_dimension_column(state: SectionState, *, section: str) -> bool:
    dimension = PANEL_SECTION_DIMENSIONS.get(section)
    if dimension is None:
        return True
    return dimension in tuple(state.columns_present or ())


def backfill_fundamental_needs_update(
    catalog: CatalogStore,
    symbol: str,
    section: str,
    provider: str,
    *,
    staleness_days: int = 90,
    skip_recent_hours: float = 24.0,
    is_etf: bool = False,
) -> tuple[bool, str]:
    """Skip fundamentals already present unless stale or stored with a collapsed panel schema."""
    plan = historical_fetch_plan(
        catalog,
        symbol,
        section,
        provider,
        staleness_days=staleness_days,
        skip_recent_hours=skip_recent_hours,
        is_etf=is_etf,
    )
    return plan.needs_refresh, plan.reason


def nport_disclosure_needs_update(
    catalog: CatalogStore,
    symbol: str,
    provider: str,
    *,
    start_year: int = 2019,
    staleness_days: int = 90,
    skip_recent_hours: float = 24.0,
) -> tuple[bool, str]:
    """Skip ETF N-PORT panels that already span the requested history window."""
    state = catalog.get(
        symbol=symbol.strip().upper(),
        section="etf_nport_disclosure",
        provider=provider,
    )
    if state is None or int(state.row_count) <= 0:
        return True, "missing"
    if _is_below_min_historical_date(state.min_date):
        return True, "below_min_historical_date"
    if not _panel_has_dimension_column(state, section="etf_nport_disclosure"):
        return True, "upgrade_panel_schema"
    min_date = _parse_date(state.min_date)
    if min_date is not None and min_date.year > int(start_year) + 1:
        return True, "incomplete_history"
    return fundamental_refresh_needs_update(
        catalog,
        symbol,
        "etf_nport_disclosure",
        provider,
        staleness_days=staleness_days,
        skip_recent_hours=skip_recent_hours,
    )


def macro_backfill_needs_update(
    catalog: CatalogStore,
    *,
    provider: str = "fmp",
    economic_series: Sequence[str] | None = None,
    include_treasury_rates: bool = True,
    target_end_date: date | None = None,
    history_start_date: date | None = None,
    skip_recent_hours: float = 24.0,
) -> bool:
    """Return True when any macro series still needs a backfill refresh."""
    target_end_date = target_end_date or expected_latest_price_date()
    history_start = history_start_date or _min_historical_floor()
    provider_name = str(provider or "fmp").strip().lower()
    for series_name in list(economic_series or DEFAULT_ECONOMIC_SERIES):
        needs_refresh, _ = macro_refresh_needs_update(
            catalog,
            series_name,
            "macro_economic",
            provider_name,
            target_end_date=target_end_date,
            history_start_date=history_start,
            skip_recent_hours=skip_recent_hours,
        )
        if needs_refresh:
            return True
    if include_treasury_rates:
        needs_refresh, _ = macro_refresh_needs_update(
            catalog,
            TREASURY_BUNDLE_SYMBOL,
            MACRO_TREASURY_SECTION,
            provider_name,
            target_end_date=target_end_date,
            history_start_date=history_start,
            skip_recent_hours=skip_recent_hours,
        )
        if needs_refresh:
            return True
    return False


def section_state_summary(state: SectionState | None) -> dict[str, Any]:
    if state is None:
        return {}
    return {
        "min_date": state.min_date,
        "max_date": state.max_date,
        "row_count": int(state.row_count),
        "last_fetched_at": state.last_fetched_at,
    }
