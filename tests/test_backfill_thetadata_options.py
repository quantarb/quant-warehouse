from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.migrate.backfill_thetadata_options import (
    _options_range_cached,
    backfill_thetadata_options_for_oracle_trades,
    list_arctic_fmp_underlyings,
    list_catalog_price_symbols,
    list_market_cap_symbols,
    normalize_oracle_trade_windows,
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


def test_list_market_cap_symbols_orders_largest_first_and_requires_prices(tmp_path: Path, monkeypatch) -> None:
    wh = Warehouse()
    store = CatalogStore(tmp_path / "catalog.sqlite")
    for symbol, market_cap in (("SMALL", 20_000_000_000), ("AAPL", 3_000_000_000_000), ("MSFT", 2_000_000_000_000)):
        store.upsert_profile(
            symbol=symbol,
            provider="fmp",
            source_provider="fmp_screener",
            payload={
                "symbol": symbol,
                "name": symbol,
                "market_cap": market_cap,
                "exchange": "NASDAQ",
                "country": "US",
                "is_etf": False,
                "is_fund": False,
            },
        )
    monkeypatch.setattr(wh, "catalog", store)
    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.list_arctic_fmp_underlyings",
        lambda _wh: ["AAPL", "SMALL"],
    )

    assert list_market_cap_symbols(wh, min_market_cap=10_000_000_000) == ["AAPL", "SMALL"]


def test_resolve_backfill_symbols_market_cap_source_applies_tier_filters(monkeypatch) -> None:
    wh = Warehouse()
    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.list_market_cap_symbols",
        lambda _wh, **kwargs: ["AAPL", "MSFT", "NVDA"],
    )

    resolved = resolve_backfill_symbols(
        wh,
        source="market-cap",
        min_market_cap=1_000_000_000_000,
        limit=2,
    )

    assert resolved == ["AAPL", "MSFT"]


def test_resolve_backfill_symbols_explicit_override() -> None:
    wh = Warehouse()
    assert resolve_backfill_symbols(wh, symbols=["aapl", "msft"]) == ["AAPL", "MSFT"]


def test_resolve_backfill_symbols_filters_non_us_by_default() -> None:
    wh = Warehouse()
    resolved = resolve_backfill_symbols(wh, symbols=["AAPL", "600031.SS"], us_only=True)
    assert resolved == ["AAPL"]


def test_options_range_cached_delegates_to_arctic_range_cache(monkeypatch) -> None:
    calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def _fake_range_cached(symbol, start_date, end_date, **kwargs):
        assert "required_columns" in kwargs
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


def test_normalize_oracle_trade_windows_sorts_most_recent_entry_first() -> None:
    trades = pd.DataFrame(
        [
            {"trade_id": "old", "symbol": "msft", "entry_date": "2024-01-02", "exit_date": "2024-01-05"},
            {"trade_id": "new_b", "symbol": "goog", "entry_date": "2024-03-01", "exit_date": "2024-03-04"},
            {"trade_id": "new_a", "symbol": "aapl", "entry_date": "2024-03-01", "exit_date": "2024-03-05"},
            {"trade_id": "bad", "symbol": "aapl", "entry_date": "2024-04-01", "exit_date": "2024-03-01"},
        ]
    )

    windows = normalize_oracle_trade_windows(trades, max_trades=2)

    assert list(windows["trade_id"]) == ["new_a", "new_b"]
    assert list(windows["symbol"]) == ["AAPL", "GOOG"]
    assert list(windows["entry_date"]) == [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01")]


def test_normalize_oracle_trade_windows_sorts_globally_across_symbols() -> None:
    trades = pd.DataFrame(
        [
            {"trade_id": "aapl_old", "symbol": "AAPL", "entry_date": "2024-01-02", "exit_date": "2024-01-05"},
            {"trade_id": "aapl_new", "symbol": "AAPL", "entry_date": "2024-03-01", "exit_date": "2024-03-05"},
            {"trade_id": "msft_middle", "symbol": "MSFT", "entry_date": "2024-02-15", "exit_date": "2024-02-20"},
            {"trade_id": "goog_newest", "symbol": "GOOG", "entry_date": "2024-03-15", "exit_date": "2024-03-18"},
        ]
    )

    windows = normalize_oracle_trade_windows(trades)

    assert list(windows["trade_id"]) == ["goog_newest", "aapl_new", "msft_middle", "aapl_old"]


def test_normalize_oracle_trade_windows_deduplicates_k_variants_before_limit() -> None:
    trades = pd.DataFrame(
        [
            {"trade_id": "aapl_k1", "symbol": "AAPL", "entry_date": "2024-03-01", "exit_date": "2024-03-05", "k": 1},
            {"trade_id": "aapl_k2", "symbol": "AAPL", "entry_date": "2024-03-01", "exit_date": "2024-03-05", "k": 2},
            {"trade_id": "msft_k1", "symbol": "MSFT", "entry_date": "2024-02-15", "exit_date": "2024-02-20", "k": 1},
        ]
    )

    windows = normalize_oracle_trade_windows(trades, max_trades=2)

    assert list(windows["trade_id"]) == ["aapl_k1", "msft_k1"]
    assert len(windows) == 2


def test_backfill_thetadata_options_for_oracle_trades_skips_cached_windows(monkeypatch) -> None:
    cached_calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    download_calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    class _Catalog:
        def upsert(self, **kwargs):
            pass

    class _Warehouse:
        catalog = _Catalog()

    def _fake_cached_summary(symbol, start_date, end_date, **kwargs):
        cached_calls.append((symbol, pd.Timestamp(start_date), pd.Timestamp(end_date)))
        if symbol == "AAPL":
            return {
                pd.Timestamp("2024-02-01"),
                pd.Timestamp("2024-02-02"),
                pd.Timestamp("2024-02-05"),
            }, 10
        return set(), 0

    def _fake_download(symbol, start_date, end_date, **kwargs):
        download_calls.append((symbol, pd.Timestamp(start_date), pd.Timestamp(end_date)))
        return {
            "symbol": symbol,
            "start_date": pd.Timestamp(start_date).date().isoformat(),
            "end_date": pd.Timestamp(end_date).date().isoformat(),
            "snapshot_days": 2,
            "contracts_total": 10,
            "cached_days": 0,
            "existing_cached_days": 0,
            "stale_cached_days": 0,
            "fetched_rows": 10,
            "cached_only": False,
            "paths": [f"arctic://thetadata_derivatives_options_eod/{symbol}"],
            "spec": {},
        }

    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.option_chain_cached_date_summary",
        _fake_cached_summary,
    )
    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.download_option_snapshots_for_range",
        _fake_download,
    )

    summary = backfill_thetadata_options_for_oracle_trades(
        pd.DataFrame(
            [
                {"trade_id": "older", "symbol": "MSFT", "entry_date": "2024-01-02", "exit_date": "2024-01-05"},
                {"trade_id": "newer", "symbol": "AAPL", "entry_date": "2024-02-01", "exit_date": "2024-02-05"},
            ]
        ),
        warehouse=_Warehouse(),
        request_sleep=0.0,
    )

    assert summary["trade_windows_requested"] == 2
    assert summary["trade_windows_skipped"] == 1
    assert [row["trade_id"] for row in summary["results"]] == ["newer", "older"]
    assert cached_calls == [
        ("AAPL", pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-05")),
        ("MSFT", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05")),
    ]
    assert download_calls == [
        ("MSFT", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")),
        ("MSFT", pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05")),
    ]
    assert summary["results"][0]["snapshot_mode"] == "entry_exit"
    assert summary["results"][1]["snapshot_dates"] == ["2024-01-02", "2024-01-05"]


def test_backfill_thetadata_options_for_oracle_trades_skips_current_or_future_endpoints(monkeypatch) -> None:
    cached_calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    download_calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    class _Catalog:
        def upsert(self, **kwargs):
            pass

    class _Warehouse:
        catalog = _Catalog()

    def _fake_cached_summary(symbol, start_date, end_date, **kwargs):
        cached_calls.append((symbol, pd.Timestamp(start_date), pd.Timestamp(end_date)))
        return set(), 0

    def _fake_download(symbol, start_date, end_date, **kwargs):
        download_calls.append((symbol, pd.Timestamp(start_date), pd.Timestamp(end_date)))
        return {
            "symbol": symbol,
            "start_date": pd.Timestamp(start_date).date().isoformat(),
            "end_date": pd.Timestamp(end_date).date().isoformat(),
            "snapshot_days": 1,
            "contracts_total": 5,
            "cached_days": 0,
            "existing_cached_days": 0,
            "stale_cached_days": 0,
            "fetched_rows": 5,
            "cached_only": False,
            "paths": [f"arctic://thetadata_derivatives_options_eod/{symbol}"],
            "spec": {},
        }

    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.option_chain_cached_date_summary",
        _fake_cached_summary,
    )
    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.download_option_snapshots_for_range",
        _fake_download,
    )

    summary = backfill_thetadata_options_for_oracle_trades(
        pd.DataFrame(
            [
                {"trade_id": "future", "symbol": "AAPL", "entry_date": "2099-01-02", "exit_date": "2099-01-05"},
            ]
        ),
        warehouse=_Warehouse(),
        request_sleep=0.0,
    )

    assert summary["trade_windows_requested"] == 1
    assert summary["trade_windows_skipped"] == 1
    assert cached_calls == [
        ("AAPL", pd.Timestamp("2099-01-02"), pd.Timestamp("2099-01-05")),
    ]
    assert download_calls == []
    assert {row["reason"] for row in summary["results"][0]["date_results"]} == {"current_or_future_eod_snapshot"}


def test_backfill_thetadata_options_for_oracle_trades_removes_symbol_after_empty_download(monkeypatch) -> None:
    download_calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    class _Catalog:
        def upsert(self, **kwargs):
            pass

    class _Warehouse:
        catalog = _Catalog()

    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.option_chain_cached_date_summary",
        lambda *args, **kwargs: (set(), 0),
    )

    def _fake_download(symbol, start_date, end_date, **kwargs):
        download_calls.append((symbol, pd.Timestamp(start_date), pd.Timestamp(end_date)))
        return {
            "symbol": symbol,
            "start_date": pd.Timestamp(start_date).date().isoformat(),
            "end_date": pd.Timestamp(end_date).date().isoformat(),
            "snapshot_days": 0,
            "contracts_total": 0,
            "cached_days": 0,
            "existing_cached_days": 0,
            "stale_cached_days": 0,
            "fetched_rows": 0,
            "cached_only": False,
            "paths": [],
            "spec": {},
        }

    monkeypatch.setattr(
        "quant_warehouse.migrate.backfill_thetadata_options.download_option_snapshots_for_range",
        _fake_download,
    )

    summary = backfill_thetadata_options_for_oracle_trades(
        pd.DataFrame(
            [
                {"trade_id": "newer", "symbol": "BRK-A", "entry_date": "2024-03-01", "exit_date": "2024-03-05"},
                {"trade_id": "older", "symbol": "BRK-A", "entry_date": "2024-02-01", "exit_date": "2024-02-05"},
            ]
        ),
        warehouse=_Warehouse(),
        request_sleep=0.0,
    )

    assert download_calls == [("BRK-A", pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-01"))]
    reasons = [
        date_result.get("reason")
        for row in summary["results"]
        for date_result in row["date_results"]
        if date_result.get("reason")
    ]
    assert reasons == [
        "empty_symbol_after_probe",
        "empty_symbol_after_probe",
        "empty_symbol_after_probe",
        "empty_symbol_after_probe",
    ]
    assert summary["trade_windows_skipped"] == 2
    assert summary["download_spec"]["empty_symbol_probe_limit"] == 1
