from __future__ import annotations

from datetime import date
import polars as pl

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.macro import MacroStore
from quant_warehouse.warehouse.sections import (
    MACRO_CALENDAR_LIBRARY,
    MACRO_ECONOMIC_LIBRARY,
    MACRO_RISK_PREMIUM_LIBRARY,
    MACRO_TREASURY_LIBRARY,
)
from quant_warehouse.warehouse.storage import provider_library


class FakeBackend:
    def __init__(self) -> None:
        self.frames: dict[tuple[str, str], pl.DataFrame] = {}

    def read(self, library: str, symbol: str, **kwargs) -> pl.DataFrame:
        del kwargs
        return self.frames.get((library, symbol), pl.DataFrame())

    def write(self, library: str, symbol: str, frame: pl.DataFrame) -> None:
        self.frames[(library, symbol)] = frame.clone()


def test_read_panel_joins_economic_and_treasury_series(tmp_path):
    home = tmp_path / "qw"
    config = WarehouseConfig(
        home=home,
        arctic_uri=f"lmdb://{home / 'arctic'}",
        catalog_path=home / "catalog.sqlite",
    )
    backend = FakeBackend()
    catalog = CatalogStore(config.catalog_path)
    store = MacroStore(config=config, backend=backend, catalog=catalog)

    economic = pl.DataFrame({"date": ["2024-01-01", "2024-02-01"], "value": [1.0, 2.0]}).with_columns(pl.col("date").str.to_datetime())
    treasury = pl.DataFrame({"date": ["2024-01-01", "2024-02-01"], "value": [4.0, 4.1]}).with_columns(pl.col("date").str.to_datetime())
    backend.write(provider_library(MACRO_ECONOMIC_LIBRARY, "fmp"), "GDP__fmp", economic)
    backend.write(provider_library(MACRO_TREASURY_LIBRARY, "fmp"), "MACRO__UST_YEAR10__fmp", treasury)
    store._upsert_catalog_state(symbol="GDP", section="macro_economic", provider="fmp", frame=economic)
    store._upsert_catalog_state(
        symbol="macro__ust_year10",
        section="macro_treasury",
        provider="fmp",
        frame=treasury,
    )

    panel = store.read_panel(["GDP", "macro__ust_year10"], provider="fmp")
    assert list(panel.columns) == ["date", "GDP", "macro__ust_year10"]
    assert panel.height == 2


def test_read_panel_collapses_duplicate_macro_dates(tmp_path):
    home = tmp_path / "qw"
    config = WarehouseConfig(
        home=home,
        arctic_uri=f"lmdb://{home / 'arctic'}",
        catalog_path=home / "catalog.sqlite",
    )
    backend = FakeBackend()
    catalog = CatalogStore(config.catalog_path)
    store = MacroStore(config=config, backend=backend, catalog=catalog)

    economic = pl.DataFrame({"date": ["2024-01-01 08:00", "2024-01-01 16:00", "2024-02-01 00:00"], "value": [1.0, 2.0, 3.0]}).with_columns(pl.col("date").str.to_datetime())
    backend.write(provider_library(MACRO_ECONOMIC_LIBRARY, "fmp"), "GDP__fmp", economic)

    panel = store.read_panel(["GDP"], provider="fmp")

    assert panel["date"].dt.date().to_list() == [date(2024, 1, 1), date(2024, 2, 1)]
    assert panel[0, "GDP"] == 2.0


def test_read_risk_premium_and_calendar(tmp_path):
    home = tmp_path / "qw"
    config = WarehouseConfig(
        home=home,
        arctic_uri=f"lmdb://{home / 'arctic'}",
        catalog_path=home / "catalog.sqlite",
    )
    backend = FakeBackend()
    catalog = CatalogStore(config.catalog_path)
    store = MacroStore(config=config, backend=backend, catalog=catalog)

    risk = pl.DataFrame({"as_of": ["2024-06-01"], "country": ["United States"],
                         "total_equity_risk_premium": [5.0], "country_risk_premium": [0.0]}).with_columns(pl.col("as_of").str.to_datetime())
    calendar = pl.DataFrame({"date": ["2024-06-01"], "country": ["US"],
                             "event": ["CPI"], "actual": [3.1]}).with_columns(pl.col("date").str.to_datetime())
    backend.write(provider_library(MACRO_RISK_PREMIUM_LIBRARY, "fmp"), "RISK_PREMIUM__fmp", risk)
    backend.write(provider_library(MACRO_CALENDAR_LIBRARY, "fmp"), "MACRO_CALENDAR__fmp", calendar)

    assert len(store.read_risk_premium(provider="fmp")) == 1
    assert len(store.read_calendar(provider="fmp")) == 1
