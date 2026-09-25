from pathlib import Path
import sqlite3
from datetime import datetime, timedelta, timezone

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


def test_recently_attempted_symbols_filters_scope_sections_and_time(tmp_path: Path):
    store = CatalogStore(tmp_path / "catalog.sqlite")
    for symbol, section in (("AAPL", "income_quarter"), ("MSFT", "prices"), ("NVDA", "income_quarter")):
        store.upsert(
            symbol=symbol,
            section=section,
            provider="fmp",
            min_date="2020-01-01",
            max_date="2024-12-31",
            row_count=1,
            columns_present=["value"],
        )
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE section_state SET last_fetched_at=? WHERE symbol='NVDA'",
            (old_timestamp,),
        )

    attempted = store.recently_attempted_symbols(
        symbols=["aapl", "msft", "nvda", "missing"],
        sections=["income_quarter"],
        provider="fmp",
        since=datetime.now(timezone.utc) - timedelta(hours=1),
    )

    assert attempted == {"AAPL"}


def test_recently_attempted_symbols_includes_attempt_without_stored_rows(tmp_path: Path):
    store = CatalogStore(tmp_path / "catalog.sqlite")
    store.record_refresh_attempt(symbol="aapl", section="income_quarter", provider="fmp")

    attempted = store.recently_attempted_symbols(
        symbols=["AAPL", "MSFT"],
        sections=["income_quarter"],
        provider="fmp",
        since=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    assert attempted == {"AAPL"}
