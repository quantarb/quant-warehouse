from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.ingest.screener_fetch import (
    ScreenerQuery,
    exchange_matches_filters,
    fetch_equity_screener,
    screener_record_to_profile_payload,
)
from quant_warehouse.refresh.screener import resolve_universe_from_catalog, screen_universe_to_catalog
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import MIN_HISTORICAL_DATE


def test_listing_date_extracts_first_stock_price_date():
    from quant_warehouse.catalog.listing_date import listing_date_from_record

    assert listing_date_from_record({"first_stock_price_date": "1980-12-12"}) == "1980-12-12"


def test_equity_historical_floor_uses_max_of_1900_and_ipo():
    from quant_warehouse.catalog.listing_date import equity_historical_floor

    assert equity_historical_floor(ipo_date="1980-12-12").isoformat() == "1980-12-12"
    assert equity_historical_floor(ipo_date="1850-01-01").isoformat() == MIN_HISTORICAL_DATE
    assert equity_historical_floor(ipo_date=None).isoformat() == MIN_HISTORICAL_DATE


def test_exchange_matches_filters_accepts_openbb_aliases():
    assert exchange_matches_filters("NMS", ("NASDAQ",))
    assert exchange_matches_filters("NYSE", ("NYSE",))
    assert not exchange_matches_filters("BUE", ("NASDAQ", "NYSE"))


def test_screener_record_to_profile_payload_normalizes_fields():
    payload = screener_record_to_profile_payload(
        {
            "symbol": "aapl",
            "companyName": "Apple Inc.",
            "marketCap": 3_000_000_000_000,
            "exchangeShortName": "NASDAQ",
            "ipoDate": "1980-12-12",
        }
    )
    assert payload["symbol"] == "AAPL"
    assert payload["market_cap"] == 3_000_000_000_000
    assert payload["ipoDate"] == "1980-12-12"


@patch("quant_warehouse.ingest.screener_fetch._fetch_openbb_screener")
def test_fetch_equity_screener_uses_openbb_only(mock_openbb):
    mock_openbb.return_value = pd.DataFrame(
        [{"symbol": "MSFT", "name": "Microsoft", "market_cap": 2_000_000_000_000, "exchange": "NASDAQ"}]
    )
    frame, source = fetch_equity_screener(ScreenerQuery(provider="fmp", mktcap_min=10_000_000_000, limit=5))
    assert source == "openbb:fmp"
    assert list(frame["symbol"]) == ["MSFT"]


def test_screen_universe_to_catalog_upserts_profiles(tmp_path: Path):
    warehouse = Warehouse()
    warehouse.catalog = CatalogStore(tmp_path / "catalog.sqlite")

    frame = pd.DataFrame(
        [
            {
                "symbol": "DONE",
                "name": "Done Company",
                "market_cap": 50_000_000_000,
                "sector": "Technology",
                "industry": "Software",
                "exchange": "NASDAQ",
                "country": "US",
                "ipoDate": "2010-01-01",
            }
        ]
    )
    with patch("quant_warehouse.refresh.screener.fetch_equity_screener", return_value=(frame, "fmp")):
        symbols, source = screen_universe_to_catalog(
            warehouse,
            ScreenerQuery(provider="fmp", limit=10),
        )

    assert symbols == ("DONE",)
    profile = warehouse.catalog.get_profile(symbol="DONE", provider="fmp")
    assert profile is not None
    assert profile.sector == "Technology"
    assert profile.market_cap == 50_000_000_000
    assert profile.ipo_date == "2010-01-01"


def test_query_symbol_profiles_matches_united_states_country(tmp_path: Path):
    store = CatalogStore(tmp_path / "catalog.sqlite")
    store.upsert_profile(
        symbol="USCO",
        provider="yfinance",
        source_provider="yfinance",
        payload={
            "symbol": "USCO",
            "name": "US Company",
            "market_cap": 20_000_000_000,
            "exchange": "NMS",
            "country": "United States",
            "sector": "Technology",
            "industry": "Software",
        },
    )
    rows = store.query_symbol_profiles(
        provider="yfinance",
        min_market_cap=10_000_000_000,
        country="US",
        exchanges=("NASDAQ",),
    )
    assert [row.symbol for row in rows] == ["USCO"]


def test_resolve_universe_from_catalog_filters_market_cap_and_exchange(tmp_path: Path):
    store = CatalogStore(tmp_path / "catalog.sqlite")
    store.upsert_profile(
        symbol="BIG",
        provider="fmp",
        source_provider="fmp_screener",
        payload={
            "symbol": "BIG",
            "name": "Big Co",
            "market_cap": 50_000_000_000,
            "exchange": "NASDAQ",
            "country": "US",
            "sector": "Technology",
            "industry": "Software",
        },
    )
    store.upsert_profile(
        symbol="SMALL",
        provider="fmp",
        source_provider="fmp_screener",
        payload={
            "symbol": "SMALL",
            "name": "Small Co",
            "market_cap": 1_000_000_000,
            "exchange": "NASDAQ",
            "country": "US",
        },
    )

    warehouse = Warehouse()
    warehouse.catalog = store
    symbols = resolve_universe_from_catalog(
        warehouse,
        provider="fmp",
        min_market_cap=10_000_000_000,
        exchanges=("NASDAQ",),
    )
    assert symbols == ("BIG",)
