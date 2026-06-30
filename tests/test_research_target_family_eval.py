from __future__ import annotations

import numpy as np
import pandas as pd

from quant_warehouse.research_tools.target_family_eval import (
    BinaryTargetConfig,
    _mark_oracle_trade_entries,
    _price_base_panel,
    build_event_target_panel,
    evaluate_feature_target_matrix,
    summarize_binary_targets,
)


def test_binary_target_config_defaults_match_optimal_trader_execution_prices() -> None:
    config = BinaryTargetConfig()

    assert config.oracle_trade_long_entry_price_col == "high"
    assert config.oracle_trade_long_exit_price_col == "low"
    assert config.oracle_trade_short_entry_price_col == "low"
    assert config.oracle_trade_short_exit_price_col == "high"


def test_build_event_target_panel_aligns_events_and_forward_windows() -> None:
    feature_panel = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "family__feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["A"],
            "event_date": [pd.Timestamp("2024-01-03")],
            "event_family": ["congress"],
            "event_type": ["congress_buy"],
            "event_side": [1],
            "mirror_event_type": ["congress_sell"],
            "actor_type": ["House"],
            "actor_name": ["Example"],
            "source": ["unit"],
            "strength": ["small"],
            "raw_json": ["{}"],
        }
    )
    config = BinaryTargetConfig(event_families=("congress",), event_windows=(2,))

    target_panel, metadata = build_event_target_panel(feature_panel, events, config)

    assert set(metadata["target"]) == {
        "target_event_on__congress_buy",
        "target_event_on__congress_sell",
        "target_event_next_2d__congress_buy",
        "target_event_next_2d__congress_sell",
    }
    assert target_panel.loc[target_panel["date"].eq(pd.Timestamp("2024-01-03")), "target_event_on__congress_buy"].item() == 1
    assert target_panel.loc[target_panel["date"].eq(pd.Timestamp("2024-01-02")), "target_event_next_2d__congress_buy"].item() == 1
    assert target_panel.loc[target_panel["date"].eq(pd.Timestamp("2024-01-03")), "target_event_next_2d__congress_buy"].item() == 0


def test_evaluate_feature_target_matrix_reports_usable_pairs() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    feature_panel = pd.DataFrame(
        {
            "symbol": ["A"] * 6 + ["B"] * 6,
            "date": dates.tolist() * 2,
            "family_a__dense": np.arange(12, dtype=float),
            "family_a__sparse": [np.nan, 1, np.nan, 2, np.nan, 3] * 2,
            "family_b__dense": np.arange(12, dtype=float)[::-1],
        }
    )
    feature_metadata = pd.DataFrame(
        [
            {
                "feature": "family_a__dense",
                "family": "family_a",
                "source": "unit",
                "source_column": "dense",
                "expected_direction": "higher_is_better",
            },
            {
                "feature": "family_a__sparse",
                "family": "family_a",
                "source": "unit",
                "source_column": "sparse",
                "expected_direction": "higher_is_better",
            },
            {
                "feature": "family_b__dense",
                "family": "family_b",
                "source": "unit",
                "source_column": "dense",
                "expected_direction": "lower_is_better",
            },
        ]
    )
    target_panel = pd.DataFrame(
        {
            "symbol": ["A"] * 6 + ["B"] * 6,
            "date": dates.tolist() * 2,
            "target_event_on__congress_buy": [0, 1, 0, 1, 0, 0] * 2,
        }
    )
    target_metadata = pd.DataFrame(
        [{"target": "target_event_on__congress_buy", "target_family": "event", "target_type": "binary"}]
    )

    matrix, merged = evaluate_feature_target_matrix(
        feature_panel,
        feature_metadata,
        target_panel,
        target_metadata,
        min_rows=1,
        min_positive_rows=1,
        min_feature_coverage=0.5,
    )
    summary = summarize_binary_targets(target_panel, target_metadata)

    assert not merged.empty
    assert set(matrix["feature_family"]) == {"family_a", "family_b"}
    assert set(matrix["status"]) == {"usable"}
    assert summary.loc[0, "positive_rows"] == 4


def test_mark_oracle_trade_entries_creates_sparse_entry_targets() -> None:
    prices = {
        "A": pd.DataFrame(
            {"high": [101.0, 107.0], "low": [99.0, 104.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
    }
    target_panel = _price_base_panel(prices)
    target_panel["target_oracle_trade_entry__YE_k1_long"] = 0
    target_panel["target_oracle_trade_entry__YE_k1_short"] = 0
    target_panel["target_oracle_trade_entry__YE_k1_any"] = 0
    trade = {
        "side": "long",
        "entry_row": pd.Series({"high": 101.0}, name=pd.Timestamp("2024-01-01")),
        "exit_row": pd.Series({"low": 104.0}, name=pd.Timestamp("2024-01-02")),
    }

    _mark_oracle_trade_entries(
        target_panel,
        {"A": [trade]},
        long_col="target_oracle_trade_entry__YE_k1_long",
        short_col="target_oracle_trade_entry__YE_k1_short",
        any_col="target_oracle_trade_entry__YE_k1_any",
    )

    first = target_panel.loc[target_panel["date"].eq(pd.Timestamp("2024-01-01"))].iloc[0]
    second = target_panel.loc[target_panel["date"].eq(pd.Timestamp("2024-01-02"))].iloc[0]
    assert first["target_oracle_trade_entry__YE_k1_long"] == 1
    assert first["target_oracle_trade_entry__YE_k1_any"] == 1
    assert first["target_oracle_trade_entry__YE_k1_short"] == 0
    assert second["target_oracle_trade_entry__YE_k1_any"] == 0
