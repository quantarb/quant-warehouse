from __future__ import annotations

from types import SimpleNamespace

from quant_warehouse.migrate import backfill_missing_fmp
from quant_warehouse.migrate.backfill_missing_fmp import backfill_missing_fmp_historical


def test_backfill_missing_fmp_uses_explicit_symbol_lists(monkeypatch, tmp_path):
    calls: dict[str, list[str]] = {}
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

    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_prices", fake_refresh_prices)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_fundamentals", fake_refresh_fundamentals)
    monkeypatch.setattr(backfill_missing_fmp, "refresh_universe_nport_disclosure", fake_refresh_nport)

    summary = backfill_missing_fmp_historical(
        warehouse=warehouse,
        include_macro=False,
        include_prices=True,
        equity_symbols=("aapl", "AAPL", "msft"),
        etf_symbols=(),
    )

    assert calls["prices"] == ["AAPL", "MSFT"]
    assert calls["fundamentals"] == ["AAPL", "MSFT"]
    assert calls["nport"] == []
    assert summary["equity_prices"]["total"] == 2
    assert summary["equity"]["total"] == 2
