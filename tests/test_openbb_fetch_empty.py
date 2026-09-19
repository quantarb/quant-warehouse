from __future__ import annotations

import polars as pl

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
    assert isinstance(result.df, pl.DataFrame)
    assert result.df.is_empty()


def test_fetch_openbb_polars_path_does_not_convert_to_legacy_frame(monkeypatch):
    class Result:
        provider = "fmp"
        results = [{"symbol": "AAA", "value": 1.0}]

        def to_polars(self):
            raise AssertionError("Do not use SDK pandas-backed conversion")

        def to_df(self):
            raise AssertionError("Polars fetch must not call to_df")

    monkeypatch.setattr(
        "quant_warehouse.ingest.openbb_fetch._call_route",
        lambda *args, **kwargs: Result(),
    )
    result = fetch_openbb(
        "profile",
        symbol="AAA",
        provider="fmp",
        dataframe_type="polars",
    )
    assert isinstance(result.df, pl.DataFrame)
    assert result.df.height == 1


def test_record_conversion_handles_mixed_dates_without_pandas(monkeypatch):
    from datetime import date,datetime,timezone
    class Result:
        results=[{'published_date':date(2020,1,1),'value':1.},
                 {'published_date':datetime(2020,1,2,tzinfo=timezone.utc),'value':2.}]
        def to_polars(self):raise AssertionError('pandas-backed SDK conversion')
    monkeypatch.setattr('quant_warehouse.ingest.openbb_fetch._call_route',lambda *a,**k:Result())
    frame=fetch_openbb('estimates_price_target',symbol='A',provider='fmp').df
    assert frame['published_date'].to_list()==[datetime(2020,1,1),datetime(2020,1,2)]


def test_transient_transport_failure_retries_without_changing_request(monkeypatch):
    from quant_warehouse.ingest import openbb_fetch
    calls = []
    result = object()
    def call(route, **kwargs):
        calls.append((route, kwargs))
        if len(calls) < 3:
            raise RuntimeError('[Unexpected Error] -> TimeoutError ->')
        return result
    monkeypatch.setattr(openbb_fetch, '_call_route', call)
    monkeypatch.setattr(openbb_fetch, 'sleep', lambda seconds: None)
    assert openbb_fetch._call_route_with_retries('equity.fundamental.income', symbol='AAPL', provider='fmp', limit=1000) is result
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]


def test_transient_transport_failure_uses_five_bounded_attempts(monkeypatch):
    import pytest
    from quant_warehouse.ingest import openbb_fetch
    calls = []
    delays = []
    def call(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('[Unexpected Error] -> ClientConnectorError ->')
    monkeypatch.setattr(openbb_fetch, '_call_route', call)
    monkeypatch.setattr(openbb_fetch, 'sleep', delays.append)
    with pytest.raises(RuntimeError, match='ClientConnectorError'):
        openbb_fetch._call_route_with_retries('equity.fundamental.income', symbol='AAPL', provider='fmp')
    assert len(calls) == 5
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_provider_validation_errors_are_not_retried(monkeypatch):
    import pytest
    from quant_warehouse.ingest import openbb_fetch
    calls = []
    def call(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError('Invalid data model')
    monkeypatch.setattr(openbb_fetch, '_call_route', call)
    with pytest.raises(ValueError, match='Invalid data model'):
        openbb_fetch._call_route_with_retries('equity.fundamental.income', symbol='AAPL', provider='fmp')
    assert len(calls) == 1
