from pathlib import Path

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.backend import ArcticBackend, open_backend


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