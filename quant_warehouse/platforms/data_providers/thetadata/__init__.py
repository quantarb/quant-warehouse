"""ThetaData data-provider integration package."""

from quant_warehouse.platforms.data_providers.thetadata.settlement import (
    OptionQuote,
    OptionSettlement,
    iter_option_exit_lookup_dates,
    lookup_contract_quote,
    option_intrinsic_value,
    settle_option_exit,
    strike_spot_scale_matches,
)

__all__ = [
    "OptionQuote",
    "OptionSettlement",
    "iter_option_exit_lookup_dates",
    "lookup_contract_quote",
    "option_intrinsic_value",
    "settle_option_exit",
    "strike_spot_scale_matches",
]
