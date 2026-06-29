from pathlib import Path

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.backend import ArcticBackend, ProviderRoutingBackend, open_backend
from quant_warehouse.warehouse.storage import provider_from_library, provider_library


def _config(tmp_path: Path) -> WarehouseConfig:
    return WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100.0, 101.0], "volume": [1000, 1100]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )


def test_arctic_backend_roundtrip(tmp_path: Path):
    backend = ArcticBackend(_config(tmp_path).arctic_uri)
    frame = _sample_frame()
    backend.write("prices", "AAPL__yfinance", frame)
    out = backend.read("prices", "AAPL__yfinance")
    assert out is not None
    assert len(out) == 2
    assert out.loc["2024-01-02", "close"] == 101.0


def test_open_backend_uses_arctic(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QW_HOME", str(tmp_path / "home"))
    backend = open_backend(WarehouseConfig.from_env())
    assert backend.kind == "arctic"
    assert isinstance(backend, ProviderRoutingBackend)


def test_provider_library_names_are_provider_scoped_without_duplicate_provider():
    assert provider_library("prices", "yfinance") == "yfinance_equity_prices"
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
