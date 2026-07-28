"""FMP data-provider integration package."""

from quant_warehouse.platforms.data_providers.fmp.synthetic_options import (
    FMP_SYNTHETIC_OPTION_SOURCE,
    FMP_SYNTHETIC_UNDERLYING_PRICE_BASIS,
    FmpSyntheticOptionSpec,
    black_scholes_greeks,
    black_scholes_price,
    build_fmp_synthetic_option_chain,
    build_realized_vol_series,
    option_intrinsic_value,
    price_fmp_synthetic_contract,
    read_fmp_synthetic_option_chain,
    settle_fmp_synthetic_option_exit,
)
from quant_warehouse.platforms.data_providers.fmp.related_assets import (
    RELATED_OHLCV_COLUMNS,
    RELATED_SECURITY_CLASSES,
    build_related_asset_panel,
    classify_related_security,
    discover_related_instruments,
    fetch_related_adjusted_ohlcv,
    parse_related_maturity_date,
)

__all__ = [
    "FMP_SYNTHETIC_OPTION_SOURCE",
    "FMP_SYNTHETIC_UNDERLYING_PRICE_BASIS",
    "FmpSyntheticOptionSpec",
    "black_scholes_greeks",
    "black_scholes_price",
    "build_fmp_synthetic_option_chain",
    "build_realized_vol_series",
    "option_intrinsic_value",
    "price_fmp_synthetic_contract",
    "read_fmp_synthetic_option_chain",
    "settle_fmp_synthetic_option_exit",
    "RELATED_OHLCV_COLUMNS",
    "RELATED_SECURITY_CLASSES",
    "build_related_asset_panel",
    "classify_related_security",
    "discover_related_instruments",
    "fetch_related_adjusted_ohlcv",
    "parse_related_maturity_date",
]
