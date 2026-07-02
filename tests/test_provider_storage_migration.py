from pathlib import Path

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.migrate.provider_storage import migrate_legacy_provider_storage
from quant_warehouse.platforms.data_providers.thetadata.options import OPTIONS_THETADATA_EOD_LIBRARY
from quant_warehouse.warehouse.backend import ArcticBackend
from quant_warehouse.warehouse.storage import provider_library


def _config(tmp_path: Path) -> WarehouseConfig:
    return WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    frame.index.name = "date"
    return frame


def test_migrate_provider_storage_copies_and_deletes_provider_keyed_symbol(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = ArcticBackend(config.arctic_uri)
    source.write("prices", "AAPL__fmp", _frame())

    rows = migrate_legacy_provider_storage(
        "fmp",
        libraries=("prices",),
        dry_run=False,
        delete_verified_legacy=True,
        config=config,
    )

    assert len(rows) == 1
    assert rows[0].status == "copied"
    assert rows[0].deleted_legacy is True

    target = ArcticBackend(config.provider_arctic_uri("fmp"))
    assert target.read(provider_library("prices", "fmp"), "AAPL__fmp") is not None
    assert source.read("prices", "AAPL__fmp") is None


def test_migrate_provider_storage_handles_thetadata_library_symbols(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = ArcticBackend(config.arctic_uri)
    source.write(OPTIONS_THETADATA_EOD_LIBRARY, "AAPL", _frame())

    rows = migrate_legacy_provider_storage(
        "thetadata",
        libraries=(OPTIONS_THETADATA_EOD_LIBRARY,),
        dry_run=False,
        delete_verified_legacy=True,
        config=config,
    )

    assert len(rows) == 1
    assert rows[0].target_symbol == "AAPL"
    assert rows[0].target_library == provider_library(OPTIONS_THETADATA_EOD_LIBRARY, "thetadata")
    assert rows[0].deleted_legacy is True

    target = ArcticBackend(config.provider_arctic_uri("thetadata"))
    assert target.read(provider_library(OPTIONS_THETADATA_EOD_LIBRARY, "thetadata"), "AAPL") is not None
    assert source.read(OPTIONS_THETADATA_EOD_LIBRARY, "AAPL") is None
