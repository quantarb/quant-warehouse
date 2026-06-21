from __future__ import annotations

import datetime as dt

import pandas as pd

from quant_warehouse.ingest.equity_calendar_fetch import normalize_equity_calendar_frame
from quant_warehouse.ingest.normalize import (
    coerce_object_dates,
    normalize_dated_snapshot_frame,
    normalize_etf_composition_frame,
)


def test_normalize_etf_holdings_uses_updated_as_of_index():
    raw = pd.DataFrame(
        {
            "symbol": ["NVDA", "AAPL"],
            "name": ["NVIDIA", "Apple"],
            "weight": [0.1, 0.08],
            "updated": ["2026-06-21 03:06:06", "2026-06-21 03:06:06"],
        }
    )
    out = normalize_etf_composition_frame(raw, section="etf_holdings")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "as_of"
    assert len(out) == 2


def test_normalize_etf_sectors_stamps_as_of_index():
    raw = pd.DataFrame({"symbol": ["SPY", "SPY"], "sector": ["Technology", "Healthcare"], "weight": [0.4, 0.2]})
    out = normalize_etf_composition_frame(raw, section="etf_sectors")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 2


def test_normalize_management_stamps_as_of_index():
    raw = pd.DataFrame({"title": ["CEO", "CFO"], "name": ["Tim", "Luca"], "pay": [1, 2]})
    out = normalize_dated_snapshot_frame(raw, section="management")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 2


def test_coerce_object_dates_converts_python_dates():
    raw = pd.DataFrame(
        {
            "report_date": ["2024-01-31"],
            "symbol": ["AAPL"],
            "last_updated": [dt.date(2024, 1, 30)],
        }
    )
    out = coerce_object_dates(raw)
    assert pd.api.types.is_datetime64_any_dtype(out["last_updated"])


def test_normalize_equity_calendar_dividend_coerces_record_date():
    raw = pd.DataFrame(
        {
            "ex_dividend_date": ["2024-01-31"],
            "symbol": ["AAPL"],
            "record_date": [dt.date(2024, 2, 2)],
        }
    )
    out = normalize_equity_calendar_frame(raw, section="equity_calendar_dividend")
    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["record_date"])