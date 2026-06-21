from __future__ import annotations

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.macro import MacroStore


class FakeBackend:
    def __init__(self) -> None:
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}

    def read(self, library: str, symbol: str) -> pd.DataFrame:
        return self.frames.get((library, symbol), pd.DataFrame())

    def write(self, library: str, symbol: str, frame: pd.DataFrame) -> None:
        self.frames[(library, symbol)] = frame.copy()


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

    economic = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    treasury = pd.DataFrame({"value": [4.0, 4.1]}, index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    backend.write("macro_economic", "GDP__fmp", economic)
    backend.write("macro_treasury", "MACRO__UST_YEAR10__fmp", treasury)
    store._upsert_catalog_state(symbol="GDP", section="macro_economic", provider="fmp", frame=economic)
    store._upsert_catalog_state(
        symbol="macro__ust_year10",
        section="macro_treasury",
        provider="fmp",
        frame=treasury,
    )

    panel = store.read_panel(["GDP", "macro__ust_year10"], provider="fmp")
    assert list(panel.columns) == ["GDP", "macro__ust_year10"]
    assert len(panel) == 2


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

    risk = pd.DataFrame(
        {
            "country": ["United States"],
            "total_equity_risk_premium": [5.0],
            "country_risk_premium": [0.0],
        },
        index=pd.to_datetime(["2024-06-01"]),
    )
    risk.index.name = "as_of"
    calendar = pd.DataFrame(
        {"country": ["US"], "event": ["CPI"], "actual": [3.1]},
        index=pd.to_datetime(["2024-06-01"]),
    )
    calendar.index.name = "date"
    backend.write("macro_risk_premium", "RISK_PREMIUM__fmp", risk)
    backend.write("macro_calendar", "MACRO_CALENDAR__fmp", calendar)

    assert len(store.read_risk_premium(provider="fmp")) == 1
    assert len(store.read_calendar(provider="fmp")) == 1