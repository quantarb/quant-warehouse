from __future__ import annotations

import pandas as pd

from quant_warehouse.ingest.macro_fetch import (
    _normalize_treasury_column_name,
    _yield_curve_wide_from_long,
    normalize_calendar_frame,
    normalize_risk_premium_frame,
    treasury_series_code,
    yield_curve_series_code,
)


def test_treasury_column_normalization_matches_fmp_codes():
    assert _normalize_treasury_column_name("month_1") == "month1"
    assert _normalize_treasury_column_name("year_10") == "year10"
    assert treasury_series_code("year_10") == "macro__ust_year10"
    assert yield_curve_series_code("year_10") == "macro__yc_year10"


def test_yield_curve_wide_from_long_pivots_maturities():
    raw = pd.DataFrame(
        {
            "date": ["2024-06-03", "2024-06-03", "2024-06-04", "2024-06-04"],
            "maturity": ["month_1", "year_10", "month_1", "year_10"],
            "rate": [0.05, 0.04, 0.051, 0.041],
        }
    )
    wide = _yield_curve_wide_from_long(raw)
    assert list(wide.columns) == ["month1", "year10"]
    assert len(wide) == 2
    assert wide.loc["2024-06-03", "year10"] == 0.04


def test_normalize_calendar_frame_indexes_events_by_date():
    raw = pd.DataFrame(
        {
            "date": ["2024-06-01", "2024-06-01"],
            "country": ["US", "US"],
            "event": ["CPI", "Payrolls"],
            "actual": [3.1, 200.0],
        }
    )
    out = normalize_calendar_frame(raw)
    assert out.index.name == "date"
    assert len(out) == 2
    assert "country" in out.columns


def test_normalize_risk_premium_frame_indexes_by_country():
    raw = pd.DataFrame(
        {
            "country": ["United States", "Canada"],
            "continent": ["North America", "North America"],
            "total_equity_risk_premium": [5.0, 4.5],
            "country_risk_premium": [0.0, 0.0],
        }
    )
    out = normalize_risk_premium_frame(raw)
    assert out.index.name == "country"
    assert len(out) == 2