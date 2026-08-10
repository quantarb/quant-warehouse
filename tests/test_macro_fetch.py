import polars as pl
from datetime import datetime

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
    raw = pl.DataFrame({"date": ["2024-06-03", "2024-06-03", "2024-06-04", "2024-06-04"], "maturity": ["month_1", "year_10", "month_1", "year_10"], "rate": [0.05, 0.04, 0.051, 0.041]})
    wide = _yield_curve_wide_from_long(raw)
    assert set(wide.columns) == {"date", "month1", "year10"}
    assert wide.filter(pl.col("date") == datetime(2024, 6, 3)).item(0, "year10") == 0.04


def test_normalize_calendar_frame_preserves_zero_values_and_aliases():
    raw = pl.DataFrame({"date": ["2024-06-01"], "country": ["US"], "event": ["Rate decision"], "importance": ["High"], "consensus": [0], "previous": [0], "actual": [0], "change_percent": [0]})
    out = normalize_calendar_frame(raw)
    row = out.row(0, named=True)
    assert row["impact"] == "High"
    assert row["estimate"] == 0
    assert row["actual"] == 0


def test_normalize_risk_premium_frame_preserves_country_rows():
    raw = pl.DataFrame({"country": ["United States", "Canada"], "total_equity_risk_premium": [5.0, 4.5], "country_risk_premium": [0.0, 0.0]})
    out = normalize_risk_premium_frame(raw)
    assert out.height == 2
    assert set(out["country"]) == {"United States", "Canada"}
