"""ThetaData-specific target engineering."""

from quant_warehouse.platforms.data_providers.thetadata.options import (
    OPTIONS_THETADATA_EOD_LIBRARY,
    ThetaDataDownloadSpec,
    download_option_snapshots_for_range,
    load_cached_snapshots_for_trade_window,
    load_thetadata_option_snapshots,
    normalize_thetadata_option_chain,
    option_chain_range_cached,
    option_chain_snapshots_cached,
    read_option_chain_arctic,
    write_option_chain_arctic,
)
from quant_warehouse.platforms.data_providers.thetadata.target_engineering.option_dataset import (
    OptionMlDatasetResult,
    OptionMlDatasetSpec,
    build_option_ml_dataset,
    save_option_ml_dataset,
)
from quant_warehouse.platforms.data_providers.thetadata.target_engineering.option_labels import (
    OptionLabelResult,
    OptionLabelSpec,
    build_option_label_panel,
    build_option_labels,
    compute_return_covariance_matrix,
    solve_long_only_mean_variance_weights,
    solve_mean_variance_weights,
)
from quant_warehouse.platforms.data_providers.thetadata.target_engineering.options import (
    build_option_best_return_labels,
    build_option_mean_variance_labels,
    build_option_return_rank_labels,
)

__all__ = [
    "OPTIONS_THETADATA_EOD_LIBRARY",
    "OptionLabelResult",
    "OptionLabelSpec",
    "OptionMlDatasetResult",
    "OptionMlDatasetSpec",
    "ThetaDataDownloadSpec",
    "build_option_best_return_labels",
    "build_option_label_panel",
    "build_option_labels",
    "build_option_mean_variance_labels",
    "build_option_ml_dataset",
    "build_option_return_rank_labels",
    "compute_return_covariance_matrix",
    "download_option_snapshots_for_range",
    "load_cached_snapshots_for_trade_window",
    "load_thetadata_option_snapshots",
    "normalize_thetadata_option_chain",
    "option_chain_range_cached",
    "option_chain_snapshots_cached",
    "read_option_chain_arctic",
    "save_option_ml_dataset",
    "solve_long_only_mean_variance_weights",
    "solve_mean_variance_weights",
    "write_option_chain_arctic",
]
