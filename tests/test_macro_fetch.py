from __future__ import annotations

import pandas as pd

from quant_warehouse.ingest.macro_fetch import (
    _fetch_fmp_economic_calendar_direct,
    _fetch_fmp_economic_indicator_series_direct,
    _normalize_treasury_column_name,
    _yield_curve_wide_from_long,
    normalize_calendar_frame,
    normalize_risk_premium_frame,
    treasury_series_code,
    yield_curve_series_code,
)


def test_direct_fmp_economic_indicator_fetch_normalizes_records(monkeypatch):
    calls = {}

    def fake_key(*, required=False):
        calls["required"] = required
        return "test-key"

    class FakeResponse:
        def raise_for_status(self):
            calls["raise_for_status"] = True

        def json(self):
            return [
                {"date": "2024-01-01", "value": "1.25"},
                {"date": "2024-02-01", "value": "1.50"},
            ]

    def fake_get(url, *, params, timeout):
        calls.update({"url": url, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("quant_warehouse.ingest.macro_fetch.resolve_fmp_api_key", fake_key)
    monkeypatch.setattr("requests.get", fake_get)

    frame = _fetch_fmp_economic_indicator_series_direct(
        "GDP",
        start_date="2024-01-01",
        end_date="2024-02-29",
    )

    assert calls == {
        "required": True,
        "url": "https://financialmodelingprep.com/stable/economic-indicators",
        "params": {
            "apikey": "test-key",
            "name": "GDP",
            "from": "2024-01-01",
            "to": "2024-02-29",
        },
        "timeout": (5, 30),
        "raise_for_status": True,
    }
    assert list(frame["value"]) == [1.25, 1.50]


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


def test_normalize_calendar_frame_preserves_zero_values_and_aliases():
    raw = pd.DataFrame(
        {
            "date": ["2024-06-01"],
            "country": ["US"],
            "event": ["Rate decision"],
            "importance": ["High"],
            "consensus": [0],
            "previous": [0],
            "actual": [0],
            "change_percent": [0],
        }
    )
    out = normalize_calendar_frame(raw)
    row = out.iloc[0]
    assert row["impact"] == "High"
    assert row["estimate"] == 0
    assert row["previous"] == 0
    assert row["actual"] == 0
    assert row["changePercentage"] == 0


def test_direct_fmp_calendar_fetch_preserves_multiple_events_and_zero_values(monkeypatch):
    calls = {}

    def fake_key(*, required=False):
        calls["required"] = required
        return "test-key"

    class FakeResponse:
        def raise_for_status(self):
            calls["raised"] = True

        def json(self):
            return [
                {"date": "2024-06-01 08:30:00", "country": "US", "event": "A", "previous": 0, "actual": 0},
                {"date": "2024-06-01 08:30:00", "country": "US", "event": "B", "previous": 1, "actual": 2},
            ]

    def fake_get(url, *, params, timeout):
        calls.update({"url": url, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("quant_warehouse.ingest.macro_fetch.resolve_fmp_api_key", fake_key)
    monkeypatch.setattr("requests.get", fake_get)
    out = _fetch_fmp_economic_calendar_direct(start_date="2024-06-01", end_date="2024-06-02")
    assert len(out) == 2
    assert list(out["actual"]) == [0, 2]
    assert calls["params"]["from"] == "2024-06-01"


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
