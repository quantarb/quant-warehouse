from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.migrate.backfill_thetadata_options import (
    _options_range_cached,
    list_arctic_fmp_underlyings,
    list_catalog_price_symbols,
    resolve_backfill_symbols,
)
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.prices import list_arctic_price_underlyings, parse_symbol_provider_key


def test_parse_symbol_provider_key() -> None:
    assert parse_symbol_provider_key("AAPL__fmp") == ("AAPL", "fmp")
    assert parse_symbol_provider_key("invalid") is None


def test_list_arctic_price_underlyings_filters_provider() -> None:
    class _Backend:
        def list_symbols(self, library: str) -> list[str]:
            assert library == "fmp_equity_prices"
            return ["AAPL__fmp", "MSFT__fmp", "SPY__yfinance"]

    assert list_arctic_price_underlyings(_Backend(), provider="fmp") == ["AAPL", "MSFT"]


def test_list_catalog_price_symbols_returns_symbols_with_prices(tmp_path: Path, monkeypatch) -> None:
    store = CatalogStore(tmp_path / "catalog.sqlite")
    store.upsert(
        symbol="AAPL",
        section="prices",
        provider="fmp",
        min_date="2020-01-01",
        max_date="2025-01-01",
        row_count=100,
        columns_present=["close"],
    )
    store.upsert(
        symbol="MSFT",
        section="prices",
        provider="fmp",
        min_date="2020-01-01",
        max_date="2025-01-01",
        row_count=0,
        columns_present=["close"],
    )
    wh = Warehouse()
    monkeypatch.setattr(wh, "catalog", store)
    assert list_catalog_price_symbols(wh) == ["AAPL"]


def test_list_arctic_fmp_underlyings_reads_prices_backend(monkeypatch) -> None:
    wh = Warehouse()

    class _Backend:
        def list_symbols(self, library: str) -> list[str]:
            return ["AAPL__fmp", "QQQ__fmp"]

    monkeypatch.setattr(wh.prices, "backend", _Backend())
    assert list_arctic_fmp_underlyings(wh) == ["AAPL", "QQQ"]


def test_resolve_backfill_symbols_defaults_to_arctic_fmp(monkeypatch) -> None:
    wh = Warehouse()
    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.list_arctic_fmp_underlyings",
        lambda _wh: ["AAPL", "MSFT"],
    )
    assert resolve_backfill_symbols(wh) == ["AAPL", "MSFT"]


def test_resolve_backfill_symbols_explicit_override() -> None:
    wh = Warehouse()
    assert resolve_backfill_symbols(wh, symbols=["aapl", "msft"]) == ["AAPL", "MSFT"]


def test_resolve_backfill_symbols_filters_non_us_by_default() -> None:
    wh = Warehouse()
    resolved = resolve_backfill_symbols(wh, symbols=["AAPL", "600031.SS"], us_only=True)
    assert resolved == ["AAPL"]


def test_options_range_cached_delegates_to_arctic_range_cache(monkeypatch) -> None:
    calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def _fake_range_cached(symbol, start_date, end_date):
        calls.append((symbol, start_date, end_date))
        return pd.Timestamp(end_date).normalize() == pd.Timestamp("2025-01-06")

    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.option_chain_range_cached",
        _fake_range_cached,
    )

    assert _options_range_cached("AAPL", pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-06"))
    assert not _options_range_cached("AAPL", pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-07"))
    assert calls == [
        ("AAPL", pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-06")),
        ("AAPL", pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-07")),
    ]
