import pandas as pd

from quant_warehouse.research_tools import feature_family_eval as module
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet


def test_feature_panel_pushes_requested_sources_into_symbol_build_and_filters_context(monkeypatch):
    received = []

    def fake_symbol(_warehouse, symbol, _config, *, strategy_sources=None, observation_dates=None):
        received.append((strategy_sources, [] if observation_dates is None else list(observation_dates)))
        frame = pd.DataFrame(
            {
                "symbol": [symbol],
                "date": [pd.Timestamp("2026-07-10")],
                "wanted_feature": [1.0],
                "unused_feature": [2.0],
            }
        )
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
        ["AAPL"],
        module.FamilyEvaluationConfig(),
        warehouse=object(),
        strategy_sources=("fmp.wanted",),
        observation_dates=pd.DataFrame([{"symbol": "AAPL", "date": "2026-07-10"}]),
    )

    assert received == [({"fmp.wanted"}, [pd.Timestamp("2026-07-10")])]
    assert list(panel.columns) == ["symbol", "date", "wanted_feature"]
    assert metadata[["source", "family"]].to_dict("records") == [
        {"source": "fmp", "family": "wanted"}
    ]


def test_technical_panel_builds_requested_family_and_only_observation_date(monkeypatch):
    class Warehouse:
        def read_prices(self, symbol, provider):
            return pd.DataFrame(
                {"open": [1, 1], "high": [2, 2], "low": [1, 1], "close": [2, 2], "volume": [10, 10]},
                index=pd.to_datetime(["2026-07-09", "2026-07-10"]),
            )

    received = []

    def fake_ta(symbol, prices, *, families=None):
        received.append(families)
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2026-07-09", "2026-07-10"]), [symbol]],
            names=["date", "symbol"],
        )
        return {
            "technical_momentum": BuiltFeatureSet(
                df=pd.DataFrame({"ta_momentum__rsi": [40.0, 60.0]}, index=index),
                feature_cols=["ta_momentum__rsi"],
            )
        }

    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.fmp.feature_engineering.ta_classic_technical.build_price_ta_classic_feature_families",
        fake_ta,
    )
    panel, metadata, _diagnostics, _timings = module.build_technical_feature_panel(
        ["AAPL"],
        module.FamilyEvaluationConfig(start_date="2026-07-01"),
        strategy_sources=("fmp.technical_momentum",),
        observation_dates=pd.DataFrame([{"symbol": "AAPL", "date": "2026-07-10"}]),
        warehouse=Warehouse(),
    )

    assert received == [{"technical_momentum"}]
    assert panel[["symbol", "date"]].to_dict("records") == [
        {"symbol": "AAPL", "date": pd.Timestamp("2026-07-10")}
    ]
    assert metadata["family"].tolist() == ["technical_momentum"]


def test_technical_panel_processes_symbols_in_parallel_deterministically(monkeypatch):
    calls = []

    class FakeExecutor:
        def __init__(self, max_workers):
            calls.append(("workers", max_workers))

        def map(self, function, tasks):
            calls.append(("symbols", [task[0] for task in tasks]))
            return map(function, tasks)

        def shutdown(self, wait):
            calls.append(("shutdown", wait))

    def fake_worker(task):
        symbol = task[0]
        return (
            pd.DataFrame(
                {"symbol": [symbol], "date": [pd.Timestamp("2026-07-10")], "feature": [1.0]}
            ),
            [module.FeatureSpec("feature", "technical_momentum", "fmp", "feature", "unknown")],
            {"symbol": symbol, "status": "ok"},
        )

    monkeypatch.setattr(module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(module, "_build_symbol_technical_panel_worker", fake_worker)

    panel, metadata, diagnostics, _timings = module.build_technical_feature_panel(
        ["MSFT", "AAPL"],
        module.FamilyEvaluationConfig(),
        strategy_sources=("fmp.technical_momentum",),
        max_workers=8,
    )

    assert calls == [
        ("workers", 2),
        ("symbols", ["MSFT", "AAPL"]),
        ("shutdown", True),
    ]
    assert panel["symbol"].tolist() == ["AAPL", "MSFT"]
    assert metadata["feature"].tolist() == ["feature"]
    assert diagnostics["symbol"].tolist() == ["MSFT", "AAPL"]
