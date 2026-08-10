from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.market_prices import MarketPricesStore
from quant_warehouse.warehouse.sections import CRYPTO_PRICES_SECTION


class FakeBackend:
    def __init__(self) -> None:
        self.frames: dict[tuple[str, str], pl.DataFrame] = {}

    def read(self, library: str, symbol: str, **kwargs) -> pl.DataFrame:
        return self.frames.get((library, symbol), pl.DataFrame())

    def write(self, library: str, symbol: str, frame: pl.DataFrame) -> None:
        self.frames[(library, symbol)] = frame.clone()


def test_market_prices_store_ingests_normalized_frame(tmp_path, monkeypatch):
    home = tmp_path / "qw"
    config = WarehouseConfig(
        home=home,
        arctic_uri=f"lmdb://{home / 'arctic'}",
        catalog_path=home / "catalog.sqlite",
    )
    backend = FakeBackend()
    catalog = CatalogStore(config.catalog_path)
    store = MarketPricesStore(config=config, backend=backend, catalog=catalog)

    raw = pl.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
        }
    ).with_columns(pl.Series("date", [datetime(2024, 1, 2), datetime(2024, 1, 3)]))

    def _fake_fetch(section, *, symbol, provider, **kwargs):
        assert section == CRYPTO_PRICES_SECTION
        assert symbol == "BTCUSD"
        return raw

    monkeypatch.setattr(
        "quant_warehouse.warehouse.market_prices.fetch_dataframe",
        _fake_fetch,
    )

    stats = store.refresh("BTCUSD", section=CRYPTO_PRICES_SECTION, provider="fmp", full_refresh=True)
    assert stats["rows"] == 2
    state = catalog.get(symbol="BTCUSD", section=CRYPTO_PRICES_SECTION, provider="fmp")
    assert state is not None
    assert state.min_date == "2024-01-02"
    assert state.max_date == "2024-01-03"
