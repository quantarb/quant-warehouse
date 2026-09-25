from __future__ import annotations

from types import SimpleNamespace
from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.migrate import backfill_missing_fmp
from quant_warehouse.migrate.backfill_missing_fmp import backfill_missing_fmp_historical


def test_backfill_missing_fmp_uses_explicit_symbol_lists(monkeypatch, tmp_path):
    calls: dict[str, list[str]] = {}
    logs: list[str] = []
    warehouse = SimpleNamespace(
        config=SimpleNamespace(catalog_path=tmp_path / "catalog.sqlite"),
        catalog=SimpleNamespace(),
    )

    monkeypatch.setattr(
        backfill_missing_fmp,
        "macro_backfill_needs_update",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        backfill_missing_fmp,
        "_catalog_symbols",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("catalog symbols should not be read")),
    )

    def fake_refresh_prices(_warehouse, symbols, **_kwargs):
        calls["prices"] = list(symbols)
        return [{"status": "skipped_fresh"} for _ in symbols]

    def fake_refresh_fundamentals(_warehouse, symbols, **_kwargs):
        calls["fundamentals"] = list(symbols)
        return [{"status": "skipped_fresh"} for _ in symbols]

    def fake_refresh_nport(_warehouse, symbols, **_kwargs):
        calls["nport"] = list(symbols)
        return [{"status": "skipped_fresh"} for _ in symbols]

    monkeypatch.setattr(
        backfill_missing_fmp,
        "_symbols_needing_price_refresh",
        lambda _warehouse, symbols, **_kwargs: list(symbols),
    )
    monkeypatch.setattr(
        backfill_missing_fmp,
        "_symbols_needing_fundamental_refresh",
        lambda _warehouse, symbols, **_kwargs: list(symbols),
    )
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_prices", fake_refresh_prices)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_fundamentals", fake_refresh_fundamentals)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_nport_disclosure", fake_refresh_nport)

    summary = backfill_missing_fmp_historical(
        warehouse=warehouse,
        include_macro=False,
        include_prices=True,
        equity_symbols=("aapl", "AAPL", "msft"),
        etf_symbols=(),
        progress_logger=logs.append,
    )

    assert calls["prices"] == ["AAPL", "MSFT"]
    assert calls["fundamentals"] == ["AAPL", "MSFT"]
    assert calls["nport"] == []
    assert any("scoped equity symbols (2): AAPL, MSFT" in message for message in logs)
    assert any("scoped ETF symbols (0): (none)" in message for message in logs)
    assert summary["equity_prices"]["total"] == 2
    assert summary["equity"]["total"] == 2


def test_backfill_missing_fmp_prefilters_fresh_symbols(monkeypatch, tmp_path):
    calls: dict[str, list[str]] = {}
    logs: list[str] = []
    warehouse = SimpleNamespace(
        config=SimpleNamespace(catalog_path=tmp_path / "catalog.sqlite"),
        catalog=SimpleNamespace(),
    )

    monkeypatch.setattr(backfill_missing_fmp, "macro_backfill_needs_update", lambda *args, **kwargs: False)
    monkeypatch.setattr(backfill_missing_fmp, "_catalog_symbols", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        backfill_missing_fmp,
        "_symbols_needing_price_refresh",
        lambda _warehouse, symbols, **_kwargs: ["MSFT"],
    )
    monkeypatch.setattr(
        backfill_missing_fmp,
        "_symbols_needing_fundamental_refresh",
        lambda _warehouse, symbols, **_kwargs: ["AAPL"],
    )

    def fake_refresh_prices(_warehouse, symbols, **_kwargs):
        calls["prices"] = list(symbols)
        return [{"status": "updated"} for _ in symbols]

    def fake_refresh_fundamentals(_warehouse, symbols, **_kwargs):
        calls["fundamentals"] = list(symbols)
        return [{"status": "updated"} for _ in symbols]

    def fake_refresh_nport(_warehouse, symbols, **_kwargs):
        calls["nport"] = list(symbols)
        return []

    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_prices", fake_refresh_prices)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_fundamentals", fake_refresh_fundamentals)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_nport_disclosure", fake_refresh_nport)

    summary = backfill_missing_fmp_historical(
        warehouse=warehouse,
        include_macro=False,
        include_prices=True,
        equity_symbols=("AAPL", "MSFT", "NVDA"),
        etf_symbols=(),
        progress_logger=logs.append,
    )

    assert calls["prices"] == ["MSFT"]
    assert calls["fundamentals"] == ["AAPL"]
    assert summary["equity_prices"]["total"] == 1
    assert summary["equity"]["total"] == 1
    assert any("refreshing equity prices for 1 stale symbols (3 scoped; 2 filtered)" in message for message in logs)
    assert any("refreshing equity fundamentals for 1 stale symbols (3 scoped; 2 filtered)" in message for message in logs)


def test_backfill_uses_same_date_guard_for_every_dataset(monkeypatch, tmp_path):
    received: list[float | None] = []
    warehouse = SimpleNamespace(
        config=SimpleNamespace(catalog_path=tmp_path / "catalog.sqlite"),
        catalog=SimpleNamespace(),
    )

    def record_macro(*_args, **kwargs):
        received.append(kwargs["skip_recent_hours"])
        return False

    def record_symbols(_warehouse, _symbols, **kwargs):
        received.append(kwargs["skip_recent_hours"])
        return []

    def record_refresh(_warehouse, _symbols=(), **kwargs):
        received.append(kwargs["skip_recent_hours"])
        return []

    monkeypatch.setattr(backfill_missing_fmp, "macro_backfill_needs_update", record_macro)
    monkeypatch.setattr(backfill_missing_fmp, "_symbols_needing_price_refresh", record_symbols)
    monkeypatch.setattr(backfill_missing_fmp, "_symbols_needing_fundamental_refresh", record_symbols)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_prices", record_refresh)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_fundamentals", record_refresh)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_nport_disclosure", record_refresh)

    summary = backfill_missing_fmp_historical(
        warehouse=warehouse,
        equity_symbols=("AAPL",),
        etf_symbols=(),
        skip_if_fetched_today=True,
    )

    assert received and all(value is None for value in received)
    assert summary["skip_if_fetched_today"] is True


def test_same_date_guard_filters_a_symbol_after_any_fundamental_attempt(tmp_path):
    catalog = CatalogStore(tmp_path / "catalog.sqlite")
    catalog.upsert(
        symbol="AAPL",
        section="dividends",
        provider="fmp",
        min_date="2020-01-01",
        max_date="2026-09-01",
        row_count=10,
        columns_present=["dividend"],
    )
    warehouse = SimpleNamespace(catalog=catalog)

    symbols = backfill_missing_fmp._symbols_needing_fundamental_refresh(
        warehouse,
        ["AAPL", "MSFT"],
        provider="fmp",
        sections=["income", "dividends"],
        period="quarter",
        staleness_days=90,
        skip_recent_hours=None,
    )

    assert symbols == ["MSFT"]
