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
]
