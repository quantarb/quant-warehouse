import pandas as pd

from quant_warehouse.research_tools import feature_family_eval as module


def test_feature_panel_pushes_requested_sources_into_symbol_build_and_filters_context(monkeypatch):
    received = []

    def fake_symbol(_warehouse, symbol, _config, *, strategy_sources=None, observation_dates=None):
        received.append((strategy_sources, [] if observation_dates is None else list(observation_dates)))
        frame = pd.DataFrame({
            "symbol": [symbol],
            "date": [pd.Timestamp("2026-07-10")],
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
        observation_dates=pd.DataFrame([{"symbol": "AAPL", "date": "2026-07-10"}]),
    )

    assert received == [({"fmp.wanted"}, [pd.Timestamp("2026-07-10")])]
    assert list(panel.columns) == ["symbol", "date", "wanted_feature"]
    assert metadata[["source", "family"]].to_dict("records") == [{"source": "fmp", "family": "wanted"}]
