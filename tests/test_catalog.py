from pathlib import Path

from quant_warehouse.catalog.store import CatalogStore


def test_catalog_upsert_and_list(tmp_path: Path):
    store = CatalogStore(tmp_path / "catalog.sqlite")
    store.upsert(
        symbol="aapl",
        section="prices",
        provider="fmp",
        min_date="2020-01-01",
        max_date="2024-12-31",
        row_count=100,
        columns_present=["open", "close"],
    )
    row = store.get(symbol="AAPL", section="prices", provider="fmp")
    assert row is not None
    assert row.row_count == 100
    assert row.columns_present == ("close", "open")

    rows = store.list_symbol("AAPL")
    assert len(rows) == 1
    assert rows[0].provider == "fmp"