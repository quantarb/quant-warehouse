from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore, SectionState
from quant_warehouse.refresh.planner import (
    backfill_fundamental_needs_update,
    expected_latest_price_date,
    fundamental_refresh_needs_update,
    historical_fetch_plan,
    macro_backfill_needs_update,
    macro_refresh_needs_update,
    nport_disclosure_needs_update,
    price_backfill_needs_update,
    price_refresh_needs_update,
    symbol_has_fresh_prices,
)
from quant_warehouse.warehouse.sections import DEFAULT_ECONOMIC_SERIES, MIN_HISTORICAL_DATE


class FakeCatalog:
    def __init__(
        self,
        states: dict[tuple[str, str, str], SectionState] | None = None,
        *,
        ipo_dates: dict[str, str] | None = None,
    ) -> None:
        self.states = dict(states or {})
        self.ipo_dates = dict(ipo_dates or {})

    def get(self, *, symbol: str, section: str, provider: str) -> SectionState | None:
        return self.states.get((symbol.upper(), section, provider))

    def resolve_equity_ipo_date(self, symbol: str) -> str | None:
        return self.ipo_dates.get(symbol.upper())


def test_expected_latest_price_date_after_close():
    now = datetime(2026, 6, 18, 22, 0, tzinfo=timezone.utc)  # 6pm ET on Thursday
    assert expected_latest_price_date(now=now).isoformat() == "2026-06-18"


def test_price_refresh_needs_update_when_max_date_stale():
    catalog = FakeCatalog(
        {
            ("AAPL", "prices", "fmp"): SectionState(
                symbol="AAPL",
                section="prices",
                provider="fmp",
                min_date="2020-01-02",
                max_date="2026-06-10",
                row_count=100,
                columns_present=("close",),
                last_fetched_at="2026-06-10T12:00:00+00:00",
            )
        }
    )
    needs, reason = price_refresh_needs_update(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=0.0,
    )
    assert needs is True
    assert reason == "stale_max_date"


def test_price_refresh_skips_when_fresh_and_recent():
    catalog = FakeCatalog(
        {
            ("AAPL", "prices", "fmp"): SectionState(
                symbol="AAPL",
                section="prices",
                provider="fmp",
                min_date="2020-01-02",
                max_date="2026-06-17",
                row_count=100,
                columns_present=("close",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    needs, reason = price_refresh_needs_update(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=24.0,
    )
    assert needs is False
    assert reason == "recent_attempt"


def test_symbol_has_fresh_prices_when_any_provider_current():
    catalog = FakeCatalog(
        {
            ("AAPL", "prices", "fmp"): SectionState(
                symbol="AAPL",
                section="prices",
                provider="fmp",
                min_date="2020-01-02",
                max_date="2026-06-15",
                row_count=100,
                columns_present=("close",),
                last_fetched_at="2026-06-10T12:00:00+00:00",
            ),
            ("AAPL", "prices", "yfinance"): SectionState(
                symbol="AAPL",
                section="prices",
                provider="yfinance",
                min_date="2020-01-02",
                max_date="2026-06-17",
                row_count=100,
                columns_present=("close",),
                last_fetched_at="2026-06-10T12:00:00+00:00",
            ),
        }
    )
    assert symbol_has_fresh_prices(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        ("fmp", "yfinance"),
        target_end_date=pd.Timestamp("2026-06-17").date(),
    )


def test_fundamental_refresh_needs_update_when_missing():
    catalog = FakeCatalog()
    needs, reason = fundamental_refresh_needs_update(
        catalog,  # type: ignore[arg-type]
        "MSFT",
        "metrics",
        "fmp",
    )
    assert needs is True
    assert reason == "missing"


def test_backfill_fundamental_skips_when_panel_schema_is_current():
    catalog = FakeCatalog(
        {
            ("AAPL", "revenue_per_segment", "fmp"): SectionState(
                symbol="AAPL",
                section="revenue_per_segment",
                provider="fmp",
                min_date="2020-01-01",
                max_date="2026-03-31",
                row_count=500,
                columns_present=("business_line", "revenue"),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    needs, reason = backfill_fundamental_needs_update(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "revenue_per_segment",
        "fmp",
        skip_recent_hours=24.0,
    )
    assert needs is False
    assert reason in {"recent_attempt", "defer_recent_attempt", "fresh"}


def test_backfill_fundamental_upgrades_collapsed_panel_schema():
    catalog = FakeCatalog(
        {
            ("AAPL", "revenue_per_segment", "fmp"): SectionState(
                symbol="AAPL",
                section="revenue_per_segment",
                provider="fmp",
                min_date="2020-01-01",
                max_date="2026-03-31",
                row_count=16,
                columns_present=("revenue",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    needs, reason = backfill_fundamental_needs_update(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "revenue_per_segment",
        "fmp",
    )
    assert needs is True
    assert reason == "upgrade_panel_schema"


def test_nport_disclosure_skips_when_history_is_complete():
    catalog = FakeCatalog(
        {
            ("SPY", "etf_nport_disclosure", "fmp"): SectionState(
                symbol="SPY",
                section="etf_nport_disclosure",
                provider="fmp",
                min_date="2019-03-31",
                max_date="2026-03-31",
                row_count=3866,
                columns_present=("cusip", "weight"),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    needs, reason = nport_disclosure_needs_update(
        catalog,  # type: ignore[arg-type]
        "SPY",
        "fmp",
        start_year=2019,
        skip_recent_hours=24.0,
    )
    assert needs is False
    assert reason == "recent_attempt"


def test_price_refresh_flags_below_min_historical_date():
    catalog = FakeCatalog(
        {
            ("AAPL", "prices", "fmp"): SectionState(
                symbol="AAPL",
                section="prices",
                provider="fmp",
                min_date="1899-12-31",
                max_date="2026-06-17",
                row_count=100,
                columns_present=("close",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    needs, reason = price_refresh_needs_update(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=0.0,
    )
    assert needs is True
    assert reason == "below_min_historical_date"


def test_macro_refresh_flags_incomplete_early_history():
    catalog = FakeCatalog(
        {
            ("GDP", "macro_economic", "fmp"): SectionState(
                symbol="GDP",
                section="macro_economic",
                provider="fmp",
                min_date="2005-01-01",
                max_date="2026-06-17",
                row_count=85,
                columns_present=("value",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    needs, reason = macro_refresh_needs_update(
        catalog,  # type: ignore[arg-type]
        "GDP",
        "macro_economic",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        history_start_date=pd.Timestamp(MIN_HISTORICAL_DATE).date(),
        skip_recent_hours=0.0,
    )
    assert needs is True
    assert reason == "incomplete_early_history"


def test_macro_backfill_skips_when_treasury_is_current():
    target_end = expected_latest_price_date()
    fresh_at = datetime.now(timezone.utc).isoformat()
    economic_states = {
        (str(series).upper(), "macro_economic", "fmp"): SectionState(
            symbol=str(series).upper(),
            section="macro_economic",
            provider="fmp",
            min_date=MIN_HISTORICAL_DATE,
            max_date=target_end.isoformat(),
            row_count=85,
            columns_present=("value",),
            last_fetched_at=fresh_at,
        )
        for series in DEFAULT_ECONOMIC_SERIES
    }
    catalog = FakeCatalog(
        {
            **economic_states,
            ("TREASURY_CURVE", "macro_treasury", "fmp"): SectionState(
                symbol="TREASURY_CURVE",
                section="macro_treasury",
                provider="fmp",
                min_date=MIN_HISTORICAL_DATE,
                max_date=target_end.isoformat(),
                row_count=5371,
                columns_present=("macro__ust_year10",),
                last_fetched_at=fresh_at,
            ),
        }
    )
    assert macro_backfill_needs_update(
        catalog,  # type: ignore[arg-type]
        provider="fmp",
        target_end_date=target_end,
        skip_recent_hours=24.0,
    ) is False


def test_price_backfill_flags_incomplete_history_when_ipo_known():
    catalog = FakeCatalog(
        {
            ("RIVN", "prices", "fmp"): SectionState(
                symbol="RIVN",
                section="prices",
                provider="fmp",
                min_date="2022-01-01",
                max_date="2026-06-17",
                row_count=100,
                columns_present=("close",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        },
        ipo_dates={"RIVN": "2021-11-10"},
    )
    needs, reason = price_backfill_needs_update(
        catalog,  # type: ignore[arg-type]
        "RIVN",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=0.0,
    )
    assert needs is True
    assert reason == "incomplete_early_history"


def test_historical_fetch_plan_tail_when_max_date_stale():
    catalog = FakeCatalog(
        {
            ("AAPL", "income", "fmp"): SectionState(
                symbol="AAPL",
                section="income",
                provider="fmp",
                min_date="2010-01-01",
                max_date="2026-03-31",
                row_count=64,
                columns_present=("total_revenue",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    plan = historical_fetch_plan(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "income",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=0.0,
    )
    assert plan.needs_refresh is True
    assert plan.mode == "tail"
    assert plan.reason == "stale_max_date"
    assert plan.fetch_ranges
    assert plan.fetch_ranges[0][1].isoformat() == "2026-06-17"


def test_historical_fetch_plan_head_when_early_history_missing():
    catalog = FakeCatalog(
        {
            ("RIVN", "income", "fmp"): SectionState(
                symbol="RIVN",
                section="income",
                provider="fmp",
                min_date="2022-01-01",
                max_date="2026-06-17",
                row_count=18,
                columns_present=("total_revenue",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        },
        ipo_dates={"RIVN": "2021-11-10"},
    )
    plan = historical_fetch_plan(
        catalog,  # type: ignore[arg-type]
        "RIVN",
        "income",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=0.0,
    )
    assert plan.needs_refresh is True
    assert plan.mode == "head"
    assert plan.reason == "incomplete_early_history"
    assert plan.fetch_ranges[0][0].isoformat() == "2021-11-10"
    assert plan.fetch_ranges[0][1].isoformat() == "2021-12-31"


def test_historical_fetch_plan_skips_when_fresh():
    catalog = FakeCatalog(
        {
            ("AAPL", "income", "fmp"): SectionState(
                symbol="AAPL",
                section="income",
                provider="fmp",
                min_date="2010-01-01",
                max_date="2026-06-17",
                row_count=64,
                columns_present=("total_revenue",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    plan = historical_fetch_plan(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "income",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=24.0,
    )
    assert plan.needs_refresh is False
    assert plan.mode == "skip"
    assert plan.reason == "recent_attempt"


def test_historical_fetch_plan_defers_recent_incomplete_retry():
    catalog = FakeCatalog(
        {
            ("AAPL", "income", "fmp"): SectionState(
                symbol="AAPL",
                section="income",
                provider="fmp",
                min_date="2010-01-01",
                max_date="2026-06-10",
                row_count=64,
                columns_present=("total_revenue",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    plan = historical_fetch_plan(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "income",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=24.0,
    )
    assert plan.needs_refresh is False
    assert plan.mode == "skip"
    assert plan.reason == "defer_recent_attempt"


def test_price_backfill_skips_when_fresh():
    catalog = FakeCatalog(
        {
            ("AAPL", "prices", "fmp"): SectionState(
                symbol="AAPL",
                section="prices",
                provider="fmp",
                min_date="1980-12-12",
                max_date="2026-06-17",
                row_count=100,
                columns_present=("close",),
                last_fetched_at=datetime.now(timezone.utc).isoformat(),
            )
        }
    )
    needs, reason = price_backfill_needs_update(
        catalog,  # type: ignore[arg-type]
        "AAPL",
        "fmp",
        target_end_date=pd.Timestamp("2026-06-17").date(),
        skip_recent_hours=24.0,
    )
    assert needs is False
    assert reason == "recent_attempt"