from __future__ import annotations

import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    LabelBuildSpec,
    add_action_labels,
    add_binary_classification_labels,
    add_rank_regression_labels,
    build_label_panel,
    build_oracle_labels,
    build_trade_results,
    generate_optimal_events,
)


def _price_frame(offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    lows = [10 + offset, 14 + offset, 9 + offset, 7 + offset, 13 + offset]
    highs = [11 + offset, 15 + offset, 10 + offset, 8 + offset, 14 + offset]
    closes = [10.5 + offset, 14.5 + offset, 9.5 + offset, 7.5 + offset, 13.5 + offset]
    return pd.DataFrame(
        {
            "adj_low": lows,
            "adj_high": highs,
            "adj_close": closes,
            "close": closes,
            "volume": [100, 110, 120, 130, 140],
        },
        index=index,
    )


def test_label_build_spec_from_mapping_parses_percent_profit() -> None:
    spec = LabelBuildSpec.from_mapping(
        {
            "k_params": {"ME": [1, "2", 2], "YE": ""},
            "min_profit_pct": "5",
            "buy_execution": "adj_high",
        }
    )

    assert spec.k_params == {"M": [1, 2]}
    assert spec.min_profit_pct == 0.05
    assert spec.buy_execution == "adj_high"


def test_generate_optimal_events_and_label_helpers() -> None:
    events = generate_optimal_events(
        _price_frame(),
        {"ME": [2]},
        min_profit_pct=0.05,
        buy_execution="adj_high",
        sell_execution="adj_low",
        short_execution="adj_low",
        cover_execution="adj_high",
        price_col="close",
    )

    assert list(events["event"]) == ["entry", "exit", "entry", "entry", "exit", "exit"]
    assert list(events["side"]) == ["long", "long", "short", "long", "short", "long"]
    assert events["trade_id"].nunique() == 3

    actions = add_action_labels(events)
    binary = add_binary_classification_labels(events, use_sample_weight=False)
    ranked = add_rank_regression_labels(binary)

    assert list(actions["label"]) == ["buy", "sell", "short", "buy", "cover", "sell"]
    assert list(binary["target"]) == [1, 0, 0, 1, 1, 0]
    assert "rank_y" in ranked.columns
    assert ranked["rank_y"].notna().all()


def test_build_trade_results_and_oracle_labels_from_price_frames() -> None:
    spec = LabelBuildSpec(
        k_params={"ME": [2]},
        min_profit_pct=0.05,
        buy_execution="adj_high",
        sell_execution="adj_low",
        short_execution="adj_low",
        cover_execution="adj_high",
        trade_dedup_mode="exact",
    )

    generated = build_trade_results(["AAPL"], spec=spec, price_frames={"AAPL": _price_frame()})
    result = build_oracle_labels(["AAPL"], spec=spec, price_frames={"AAPL": _price_frame()})

    assert len(generated.completed_trades) == 3
    assert [row["side"] for row in generated.completed_trades] == ["long", "long", "short"]
    assert len(result.label_rows) == 6
    assert result.statistics["trade_stats"]["total_trades"] == 3
    assert result.statistics["trade_stats"]["symbols_count"] == 1


def test_build_trade_results_batches_multiple_symbols() -> None:
    spec = LabelBuildSpec(
        k_params={"ME": [2]},
        min_profit_pct=0.05,
        buy_execution="adj_high",
        sell_execution="adj_low",
        short_execution="adj_low",
        cover_execution="adj_high",
        trade_dedup_mode="exact",
    )

    generated = build_trade_results(
        ["AAPL", "MSFT"],
        spec=spec,
        price_frames={"AAPL": _price_frame(), "MSFT": _price_frame(offset=1.0)},
    )

    assert len(generated.completed_trades) == 6
    assert {row["symbol"] for row in generated.completed_trades} == {"AAPL", "MSFT"}


def test_build_label_panel_parallel_matches_sequential() -> None:
    frames = {"AAPL": _price_frame(), "MSFT": _price_frame(offset=1.0)}
    kwargs = {
        "k_params": {"ME": [2]},
        "solver_mode": "period_top_k",
        "execution_params": {
            "min_profit_pct": 0.05,
            "buy_execution": "adj_high",
            "sell_execution": "adj_low",
            "short_execution": "adj_low",
            "cover_execution": "adj_high",
            "price_col": "close",
        },
        "weighting": {"use_sample_weight": False},
        "add_rank_labels": True,
        "deduplicate": True,
    }

    sequential = build_label_panel(frames, max_workers=1, **kwargs)
    parallel = build_label_panel(frames, max_workers=2, **kwargs)

    pd.testing.assert_frame_equal(
        sequential.sort_index().sort_index(axis=1),
        parallel.sort_index().sort_index(axis=1),
    )
    assert set(sequential.index.get_level_values("symbol")) == {"AAPL", "MSFT"}
    assert {"label", "target", "rank_y", "trade_id"}.issubset(sequential.columns)

