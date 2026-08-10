from __future__ import annotations

import datetime as dt

import polars as pl

from quant_warehouse.ingest.equity_calendar_fetch import normalize_equity_calendar_frame
from quant_warehouse.ingest.normalize import (
    coerce_object_dates,
    normalize_dated_snapshot_frame,
    normalize_etf_composition_frame,
)


def test_normalize_etf_holdings_uses_updated_as_of_index():
    raw = pl.DataFrame(
        {
            "symbol": ["NVDA", "AAPL"],
            "name": ["NVIDIA", "Apple"],
            "weight": [0.1, 0.08],
            "updated": ["2026-06-21 03:06:06", "2026-06-21 03:06:06"],
        }
    )
    out = normalize_etf_composition_frame(raw, section="etf_holdings")
    assert out.schema["as_of"] == pl.Datetime
    assert len(out) == 2


def test_normalize_etf_sectors_stamps_as_of_index():
    raw = pl.DataFrame({"symbol": ["SPY", "SPY"], "sector": ["Technology", "Healthcare"], "weight": [0.4, 0.2]})
    out = normalize_etf_composition_frame(raw, section="etf_sectors")
    assert out.schema["as_of"] == pl.Datetime
    assert len(out) == 2


def test_normalize_management_stamps_as_of_index():
    raw = pl.DataFrame({"title": ["CEO", "CFO"], "name": ["Tim", "Luca"], "pay": [1, 2]})
    out = normalize_dated_snapshot_frame(raw, section="management")
    assert out.schema["as_of"] == pl.Datetime
    assert len(out) == 2


def test_coerce_object_dates_converts_python_dates():
    raw = pl.DataFrame(
        {
            "report_date": ["2024-01-31"],
            "symbol": ["AAPL"],
            "last_updated": [dt.date(2024, 1, 30)],
        }
    )
    out = coerce_object_dates(raw)
    assert out.schema["last_updated"] == pl.Datetime


def test_normalize_equity_calendar_dividend_coerces_record_date():
    raw = pl.DataFrame(
        {
            "ex_dividend_date": ["2024-01-31"],
            "symbol": ["AAPL"],
            "record_date": [dt.date(2024, 2, 2)],
        }
    )
    out = normalize_equity_calendar_frame(raw, section="equity_calendar_dividend")
    assert len(out) == 1
    assert out.schema["record_date"] in {pl.Date, pl.Datetime}
