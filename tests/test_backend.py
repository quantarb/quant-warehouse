from pathlib import Path
from datetime import datetime

import polars as pl
import pytest

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.backend import ArcticBackend, ProviderRoutingBackend, open_backend
from quant_warehouse.warehouse.storage import provider_from_library, provider_library


def _config(tmp_path: Path) -> WarehouseConfig:
    return WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame({"date": [datetime(2024, 1, 1), datetime(2024, 1, 2)], "close": [100.0, 101.0], "volume": [1000, 1100]})


def test_arctic_backend_roundtrip(tmp_path: Path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = _sample_frame()
    backend.write("prices", "AAPL__yfinance", frame)
    out = backend.read("prices", "AAPL__yfinance")
    assert out is not None
    assert len(out) == 2
    assert out.filter(pl.col("date") == datetime(2024, 1, 2))["close"][0] == 101.0


def test_partially_dated_panel_preserves_undated_rows_and_supports_date_reads(tmp_path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = pl.DataFrame({'date': [None, datetime(2024, 1, 2), datetime(2024, 2, 1)],
                          'amount': [10., 20., 30.]})
    backend.write('events', 'ABC__fmp', frame)
    out = backend.read('events', 'ABC__fmp')
    assert out.height == 3 and out['date'].null_count() == 1
    selected = backend.read('events', 'ABC__fmp',
                            date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
                            columns=['amount'])
    assert selected.to_dicts() == [{'amount': 20.}]


@pytest.mark.parametrize('date_dtype', [pl.Null, pl.String, pl.Datetime('ns')])
def test_all_missing_filing_dates_are_stored_without_object_inference(tmp_path, monkeypatch, date_dtype):
    import arcticdb.version_store._normalization as normalization
    original = normalization.get_sample_from_non_empty_arr
    def sample(array, name):
        assert name != 'filing_date', 'Missing filing dates must not use object inference'
        return original(array, name)
    monkeypatch.setattr(normalization, 'get_sample_from_non_empty_arr', sample)
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = _sample_frame().with_columns(pl.Series('filing_date', [None, None], dtype=date_dtype))
    backend.write('statements', 'AAPL__fmp', frame)
    out = backend.read('statements', 'AAPL__fmp')
    assert out.height == frame.height
    assert out['filing_date'].to_list() == [None, None]
    assert out['close'].to_list() == frame['close'].to_list()
    assert out['date'].to_list() == frame['date'].to_list()
    # A later refresh can populate the same date field without losing rows.
    updated = frame.with_columns(pl.Series('filing_date', [None, datetime(2024, 1, 3)]))
    backend.write('statements', 'AAPL__fmp', updated)
    assert backend.read('statements', 'AAPL__fmp')['filing_date'].to_list() == [None, datetime(2024, 1, 3)]


def test_open_backend_uses_arctic(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QW_HOME", str(tmp_path / "home"))
    backend = open_backend(WarehouseConfig.from_env())
    assert backend.kind == "arctic"
    assert isinstance(backend, ProviderRoutingBackend)


def test_provider_library_names_are_provider_scoped_without_duplicate_provider():
    assert provider_library("prices", "yfinance") == "yfinance_equity_prices"
    assert provider_library("prices_unadjusted", "fmp") == "fmp_equity_prices_unadjusted"
    assert provider_library("etf_prices", "fmp") == "fmp_etf_prices"
    assert provider_library("fund_prices", "yfinance") == "yfinance_fund_prices"
    assert provider_library("fundamental_income", "fmp") == "fmp_equity_fundamental_income"
    assert provider_library("options_thetadata_eod", "thetadata") == "thetadata_derivatives_options_eod"


def test_provider_from_library_understands_route_family_names():
    assert provider_from_library("fmp_equity_prices") == "fmp"
    assert provider_from_library("thetadata_derivatives_options_eod") == "thetadata"
    assert provider_from_library("federal_reserve_macro_treasury") == "federal_reserve"
    assert provider_from_library("prices") is None


def test_provider_arctic_uri_defaults_to_separate_lmdb_roots(tmp_path: Path):
    config = _config(tmp_path)
    assert config.provider_arctic_uri("fmp") == f"lmdb://{tmp_path / 'arctic' / 'providers' / 'fmp'}"
    assert config.provider_arctic_uri("thetadata") == (
        f"lmdb://{tmp_path / 'arctic' / 'providers' / 'thetadata'}"
    )


def test_provider_routing_backend_separates_provider_roots(tmp_path: Path):
    config = _config(tmp_path)
    backend = ProviderRoutingBackend(config)
    frame = _sample_frame()

    backend.write("fmp_equity_prices", "AAPL__fmp", frame)
    backend.write("yfinance_equity_prices", "AAPL__yfinance", frame)

    fmp_backend = ArcticBackend(config.provider_arctic_uri("fmp"))
    yfinance_backend = ArcticBackend(config.provider_arctic_uri("yfinance"))
    default_backend = ArcticBackend(config.arctic_uri)

    assert fmp_backend.read("fmp_equity_prices", "AAPL__fmp") is not None
    assert yfinance_backend.read("yfinance_equity_prices", "AAPL__yfinance") is not None
    assert default_backend.read("fmp_equity_prices", "AAPL__fmp") is None
