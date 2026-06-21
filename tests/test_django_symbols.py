from pathlib import Path

from quant_warehouse.ingest.django_symbols import django_is_etf, list_django_symbols


def test_django_symbol_asset_class_filters(optimal_trader_db: Path | None = None):
    db = Path("/home/jlee153232/PycharmProjects/optimal_trader/db.sqlite3")
    if not db.exists():
        return

    equity = list_django_symbols(db, asset_class="equity", require_prices=True, limit=5000)
    etfs = list_django_symbols(db, asset_class="etf", require_prices=True, limit=5000)

    assert "AAPL" in equity
    assert "SPY" in etfs
    assert "SPY" not in equity
    assert "AAPL" not in etfs
    assert django_is_etf(db, "SPY")
    assert not django_is_etf(db, "AAPL")