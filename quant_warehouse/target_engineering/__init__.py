"""Target engineering utilities for labels and oracle trade targets."""

from quant_warehouse.target_engineering.labels import (
    add_action_labels,
    add_binary_classification_labels,
    add_rank_regression_labels,
    build_label_panel,
    build_oracle_labels,
    build_trade_results,
    deduplicate_labels,
    generate_optimal_events,
)
from quant_warehouse.target_engineering.optimal_trades import build_optimal_trade_labels
from quant_warehouse.target_engineering.returns import (
    build_cross_sectional_rank_labels,
    build_forward_return_labels,
)
from quant_warehouse.target_engineering.operations import (
    apply_trade_deduplication,
    build_label_rows_from_completed_trades,
    build_label_statistics,
    trade_return_pct,
)
from quant_warehouse.target_engineering.specs import (
    LabelBuildSpec,
    OracleLabelResult,
    TradeGenerationResult,
    parse_k_list,
)
from quant_warehouse.target_engineering.strategy_solver import (
    Trade,
    solve_joint_trade_sequence_by_frequency,
    solve_joint_trades_by_frequency,
    solve_longs_by_frequency,
    solve_optimal_joint_trade_sequence_generic,
    solve_optimal_joint_trades_generic,
    solve_optimal_trades_generic,
    solve_shorts_by_frequency,
    solve_trades_by_frequency,
)

__all__ = [
    "LabelBuildSpec",
    "OracleLabelResult",
    "Trade",
    "TradeGenerationResult",
    "add_action_labels",
    "add_binary_classification_labels",
    "add_rank_regression_labels",
    "apply_trade_deduplication",
    "build_label_panel",
    "build_label_rows_from_completed_trades",
    "build_label_statistics",
    "build_cross_sectional_rank_labels",
    "build_forward_return_labels",
    "build_oracle_labels",
    "build_optimal_trade_labels",
    "build_trade_results",
    "deduplicate_labels",
    "generate_optimal_events",
    "parse_k_list",
    "solve_joint_trade_sequence_by_frequency",
    "solve_joint_trades_by_frequency",
    "solve_longs_by_frequency",
    "solve_optimal_joint_trade_sequence_generic",
    "solve_optimal_joint_trades_generic",
    "solve_optimal_trades_generic",
    "solve_shorts_by_frequency",
    "solve_trades_by_frequency",
    "trade_return_pct",
]
