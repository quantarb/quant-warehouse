from pathlib import Path

from quant_warehouse.catalog.store import CatalogStore


def test_symbol_profile_roundtrip(tmp_path: Path):
    store = CatalogStore(tmp_path / "catalog.sqlite")
    store.upsert_profile(
        symbol="aapl",
        provider="yfinance",
        source_provider="yfinance",
        payload={
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry_category": "Consumer Electronics",
            "market_cap": 3_000_000_000_000,
            "beta": 1.2,
            "cik": None,
            "first_stock_price_date": "1980-12-12",
        },
    )
    row = store.get_profile(symbol="AAPL", provider="yfinance")
    assert row is not None
    assert row.source_provider == "yfinance"
    assert row.company_name == "Apple Inc."
    assert row.sector == "Technology"
    assert row.industry == "Consumer Electronics"
    assert row.market_cap == 3_000_000_000_000
    assert row.ipo_date == "1980-12-12"