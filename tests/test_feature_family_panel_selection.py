from datetime import datetime

import polars as pl

from quant_warehouse.research_tools import feature_family_eval as module


def test_feature_panel_pushes_requested_sources_into_symbol_build_and_filters_context(monkeypatch):
    received = []

    def fake_symbol(_warehouse, symbol, _config, *, strategy_sources=None, observation_dates=None, **kwargs):
        received.append((strategy_sources, [] if observation_dates is None else list(observation_dates)))
        frame = pl.DataFrame({
            "symbol": [symbol],
            "date": [datetime(2026, 7, 10)],
            "wanted_feature": [1.0],
            "unused_feature": [2.0],
        })
        specs = [
            module.FeatureSpec("wanted_feature", "wanted", "fmp", "x", "higher_is_better"),
            module.FeatureSpec("unused_feature", "unused", "fmp", "y", "higher_is_better"),
        ]
        return frame, specs, {"symbol": symbol, "status": "ok"}

    monkeypatch.setattr(module, "_build_symbol_fundamental_panel", fake_symbol)
    monkeypatch.setattr(module, "_add_time_calendar_features", lambda _panel: [])
    monkeypatch.setattr(module, "_add_macro_context_features", lambda _wh, _panel, _config: [])
    monkeypatch.setattr(module, "_add_cross_symbol_context_features", lambda _wh, _panel, _config: [])

    panel, metadata, _diagnostics, _timings = module.build_fundamental_feature_panel(
        ["AAPL"], module.FamilyEvaluationConfig(), warehouse=object(),
        strategy_sources=("fmp.wanted",),
        observation_dates=pl.DataFrame([{"symbol": "AAPL", "date": datetime(2026, 7, 10)}]),
    )

    assert received == [({"fmp.wanted"}, [datetime(2026, 7, 10)])]
    assert list(panel.columns) == ["symbol", "date", "wanted_feature"]
    assert metadata.select(["source", "family"]).to_dicts() == [{"source": "fmp", "family": "wanted"}]
