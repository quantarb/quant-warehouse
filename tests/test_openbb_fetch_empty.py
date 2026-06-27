from __future__ import annotations

import pandas as pd

from quant_warehouse.ingest.openbb_fetch import _is_empty_fetch_error, fetch_openbb


def test_is_empty_fetch_error_detects_openbb_empty_messages():
    assert _is_empty_fetch_error(Exception("[Empty] -> No results found."))
    assert _is_empty_fetch_error(Exception("No data found for the given symbols."))
    assert not _is_empty_fetch_error(Exception("connection reset"))


def test_fetch_openbb_returns_empty_frame_on_provider_empty(monkeypatch):
    def _raise(*args, **kwargs):
        raise Exception("[Empty] -> No results found. Try adjusting the query parameters.")

    monkeypatch.setattr("quant_warehouse.ingest.openbb_fetch._call_route", _raise)
    result = fetch_openbb("etf_holdings", symbol="AAAD", provider="fmp")
    assert isinstance(result.df, pd.DataFrame)
    assert result.df.empty