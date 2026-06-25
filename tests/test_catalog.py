from pathlib import Path
import sqlite3

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


def test_catalog_connect_retries_transient_open_failure(tmp_path: Path, monkeypatch):
    store = CatalogStore(tmp_path / "catalog.sqlite")
    store.upsert(
        symbol="aapl",
        section="prices",
        provider="fmp",
        min_date="2020-01-01",
        max_date="2024-12-31",
        row_count=100,
        columns_present=["close"],
    )

    original_connect = sqlite3.connect
    attempts = 0

    def flaky_connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", flaky_connect)

    row = store.get(symbol="AAPL", section="prices", provider="fmp")
    assert row is not None
    assert row.row_count == 100
    assert attempts == 2
