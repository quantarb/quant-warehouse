"""FMP-specific target engineering."""

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs import (
    EVENT_PAIR_COLUMNS,
    EVENT_PAIR_LIBRARY,
    EVENT_PAIR_SECTION,
    EVENT_PAIR_TAXONOMY,
    EventPairLoadResult,
    EventPairStore,
    build_event_pairs_from_historical_data,
    fetch_fmp_event_pair_family,
    fetch_fmp_event_pairs,
    get_event_side,
    get_mirror_event_type,
    normalize_event_pairs,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.labels import (
    add_action_labels,
    add_binary_classification_labels,
    add_rank_regression_labels,
    build_label_panel,
    build_oracle_labels,
    build_trade_results,
    deduplicate_labels,
    generate_optimal_events,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.operations import (
    apply_trade_deduplication,
    build_label_rows_from_completed_trades,
    build_label_statistics,
    trade_return_pct,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.returns import (
    build_cross_sectional_rank_labels,
    build_forward_return_labels,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.specs import (
    LabelBuildSpec,
    OracleLabelResult,
    TradeGenerationResult,
    parse_k_list,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.strategy_solver import (
    Trade,
    solve_longs_by_frequency,
    solve_optimal_trades_generic,
    solve_shorts_by_frequency,
    solve_side_trades_by_frequency_batched_multi_k,
    solve_trades_by_frequency,
)

__all__ = [
    "EVENT_PAIR_COLUMNS",
    "EVENT_PAIR_LIBRARY",
    "EVENT_PAIR_SECTION",
    "EVENT_PAIR_TAXONOMY",
    "EventPairLoadResult",
    "EventPairStore",
    "LabelBuildSpec",
    "OracleLabelResult",
    "Trade",
    "TradeGenerationResult",
    "add_action_labels",
    "add_binary_classification_labels",
    "add_rank_regression_labels",
    "apply_trade_deduplication",
    "build_event_pairs_from_historical_data",
    "build_cross_sectional_rank_labels",
    "build_forward_return_labels",
    "build_label_panel",
    "build_label_rows_from_completed_trades",
    "build_label_statistics",
    "build_oracle_labels",
    "build_trade_results",
    "deduplicate_labels",
    "fetch_fmp_event_pair_family",
    "fetch_fmp_event_pairs",
    "generate_optimal_events",
    "get_event_side",
    "get_mirror_event_type",
    "normalize_event_pairs",
    "parse_k_list",
    "solve_longs_by_frequency",
    "solve_optimal_trades_generic",
    "solve_shorts_by_frequency",
    "solve_side_trades_by_frequency_batched_multi_k",
    "solve_trades_by_frequency",
    "trade_return_pct",
]
