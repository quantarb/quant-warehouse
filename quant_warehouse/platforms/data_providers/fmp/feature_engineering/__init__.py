"""FMP-specific feature engineering."""

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.fundamentals import (
    broadcast_fundamentals_to_daily,
    fetch_fundamentals_data,
    section_prefix,
    warehouse_section_for_legacy_key,
    warehouse_section_to_indexed_frame,
    warehouse_section_to_payload_rows,
)
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.fundamental_features import (
    build_event_features,
    build_fundamental_change_features,
    build_ownership_features,
    build_statement_quality_features,
    build_ttm_financial_statement_features,
)
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.broadcast import broadcast_asof_to_target_index
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.recipe import recipe_hash
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import (
    BuiltFeatureSet,
    FeatureBuildSpec,
    FeatureToggleSpec,
    RepresentationEmbeddingSpec,
)
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.ta_classic_technical import (
    TA_CLASSIC_FAMILY_PREFIXES,
    build_price_ta_classic_feature_families,
)
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.technical import (
    FeaturesResult,
    build_price_technical_features,
    compute_features_worldclass,
)
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.time_features import (
    TimeFeatureConfig,
    build_time_calendar_features,
    build_time_features,
)

__all__ = [
    "BuiltFeatureSet",
    "FeatureBuildSpec",
    "FeatureToggleSpec",
    "FeaturesResult",
    "RepresentationEmbeddingSpec",
    "TA_CLASSIC_FAMILY_PREFIXES",
    "TimeFeatureConfig",
    "broadcast_asof_to_target_index",
    "broadcast_fundamentals_to_daily",
    "build_event_features",
    "build_fundamental_change_features",
    "build_ownership_features",
    "build_price_ta_classic_feature_families",
    "build_price_technical_features",
    "build_statement_quality_features",
    "build_time_calendar_features",
    "build_time_features",
    "build_ttm_financial_statement_features",
    "compute_features_worldclass",
    "fetch_fundamentals_data",
    "recipe_hash",
    "section_prefix",
    "warehouse_section_for_legacy_key",
    "warehouse_section_to_indexed_frame",
    "warehouse_section_to_payload_rows",
]
