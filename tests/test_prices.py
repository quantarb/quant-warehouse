from pathlib import Path

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import clip_to_min_historical_date, normalize_prices
from quant_warehouse.warehouse.sections import FUND_PRICES_LIBRARY, MIN_HISTORICAL_DATE
from quant_warehouse.warehouse.backend import ArcticBackend
from quant_warehouse.warehouse.merge import merge_upsert
from quant_warehouse.warehouse.prices import (
    PRICES_LIBRARY,
    PricesStore,
    price_library_for_adjustment,
)
from quant_warehouse.warehouse.storage import provider_library


def test_clip_to_min_historical_date_drops_pre_floor_rows():
    frame = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["1899-12-31", "1900-01-01", "1900-01-02"]),
    )
    clipped = clip_to_min_historical_date(frame, min_date=MIN_HISTORICAL_DATE)
    assert clipped.index.min().strftime("%Y-%m-%d") == MIN_HISTORICAL_DATE
    assert len(clipped) == 2


def test_normalize_prices_clips_to_equity_ipo_floor():
    raw = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        },
        index=pd.to_datetime(["1980-12-01", "1980-12-12", "1980-12-13"]),
    )
    raw.index.name = "date"
    out = normalize_prices(raw, provider="fmp", min_date="1980-12-12")
    assert out.index.min().strftime("%Y-%m-%d") == "1980-12-12"
    assert len(out) == 2


def test_normalize_prices_from_index():
    raw = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    raw.index.name = "date"
    out = normalize_prices(raw, provider="yfinance")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert out.index.name == "date"


def test_normalize_prices_preserves_raw_and_adjusted_close():
    raw = pd.DataFrame(
        {
            "date": ["2023-01-03"],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "adj_close": [101.0],
            "volume": [1000],
        }
    )

    out = normalize_prices(raw, provider="yfinance")

    assert out.loc["2023-01-03", "close"] == 105.0
    assert out.loc["2023-01-03", "adj_close"] == 101.0


def test_prices_store_gap_fill_and_read(tmp_path: Path):
    config = WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    backend = ArcticBackend(config.arctic_uri)
    catalog = CatalogStore(config.catalog_path)
    store = PricesStore(config, backend=backend, catalog=catalog)

    first = pd.DataFrame(
        {"close": [100.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    first.index.name = "date"
    store.backend.write(provider_library(PRICES_LIBRARY, "yfinance"), "AAPL__yfinance", first)
    catalog.upsert(
        symbol="AAPL",
        section="prices",
        provider="yfinance",
        min_date="2024-01-01",
        max_date="2024-01-01",
        row_count=1,
        columns_present=["close"],
    )

    assert store._gap_fill_start("AAPL", "yfinance") == "2023-12-27"

    second = pd.DataFrame(
        {"close": [101.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    second.index.name = "date"
    merged = normalize_prices(second, provider="yfinance")
    existing = store.backend.read(provider_library(PRICES_LIBRARY, "yfinance"), "AAPL__yfinance")
    store.backend.write(
        provider_library(PRICES_LIBRARY, "yfinance"),
        "AAPL__yfinance",
        merge_upsert(existing, merged),
    )

    out = store.read("AAPL", provider="yfinance")
    assert len(out) == 2
    assert out.loc["2024-01-02", "close"] == 101.0


def test_prices_store_requests_dividend_adjusted_prices(tmp_path: Path, monkeypatch):
    config = WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    backend = ArcticBackend(config.arctic_uri)
    catalog = CatalogStore(config.catalog_path)
    store = PricesStore(config, backend=backend, catalog=catalog)
    raw = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    raw.index.name = "date"
    calls = []

    def _fake_fetch(section, *, symbol, provider, **kwargs):
        calls.append((section, symbol, provider, kwargs))
        return raw

    monkeypatch.setattr("quant_warehouse.warehouse.prices.fetch_dataframe", _fake_fetch)

    stats = store.refresh("AAPL", providers=("yfinance",), full_refresh=True)

    assert stats["yfinance"]["rows"] == 1
    assert stats["yfinance"]["library"] == provider_library(PRICES_LIBRARY, "yfinance")
    assert calls == [
        (
            "prices",
            "AAPL",
            "yfinance",
            {"adjustment": "splits_and_dividends"},
        )
    ]
    assert backend.read(provider_library(PRICES_LIBRARY, "yfinance"), "AAPL__yfinance") is not None
    assert backend.read(PRICES_LIBRARY, "AAPL__yfinance") is None


def test_prices_store_can_store_unadjusted_prices_separately(tmp_path: Path, monkeypatch):
    config = WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    backend = ArcticBackend(config.arctic_uri)
    catalog = CatalogStore(config.catalog_path)
    store = PricesStore(config, backend=backend, catalog=catalog)
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "adj_close": [95.5],
            "volume": [1000],
        }
    )
    calls = []

    def _fake_fetch(section, *, symbol, provider, **kwargs):
        calls.append((section, symbol, provider, kwargs))
        return raw

    monkeypatch.setattr("quant_warehouse.warehouse.prices.fetch_dataframe", _fake_fetch)

    stats = store.refresh(
        "AAPL",
        providers=("yfinance",),
        full_refresh=True,
        adjustment="unadjusted",
    )
    raw_library = price_library_for_adjustment("unadjusted")
    physical_library = provider_library(raw_library, "yfinance")

    assert stats["yfinance"]["rows"] == 1
    assert stats["yfinance"]["library"] == physical_library
    assert calls == [
        (
            "prices",
            "AAPL",
            "yfinance",
            {"adjustment": "unadjusted"},
        )
    ]
    state = catalog.get(symbol="AAPL", section="prices_unadjusted", provider="yfinance")
    assert state is not None
    assert store.read("AAPL", provider="yfinance").empty
    out = store.read("AAPL", provider="yfinance", adjustment="unadjusted")
    assert out.loc["2024-01-02", "close"] == 100.5
    assert out.loc["2024-01-02", "adj_close"] == 95.5
    assert backend.read(physical_library, "AAPL__yfinance") is not None


def test_prices_store_read_falls_back_to_fund_price_library(tmp_path: Path):
    config = WarehouseConfig(
        home=tmp_path / "home",
        arctic_uri=f"lmdb://{tmp_path / 'arctic'}",
        catalog_path=tmp_path / "catalog.sqlite",
    )
    backend = ArcticBackend(config.arctic_uri)
    catalog = CatalogStore(config.catalog_path)
    store = PricesStore(config, backend=backend, catalog=catalog)
    frame = pd.DataFrame(
        {"close": [10.0, 10.5]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    frame.index.name = "date"

    backend.write(provider_library(FUND_PRICES_LIBRARY, "yfinance"), "VTSAX__yfinance", frame)

    out = store.read("VTSAX", provider="yfinance")

    assert len(out) == 2
    assert out.loc["2024-01-02", "close"] == 10.5
