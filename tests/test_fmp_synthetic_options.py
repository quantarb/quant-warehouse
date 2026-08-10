from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from quant_warehouse.platforms.data_providers.fmp import (
    FMP_SYNTHETIC_OPTION_SOURCE,
    FMP_SYNTHETIC_UNDERLYING_PRICE_BASIS,
    FmpSyntheticOptionSpec,
    build_fmp_synthetic_option_chain,
    option_intrinsic_value,
    price_fmp_synthetic_contract,
    read_fmp_synthetic_option_chain,
    settle_fmp_synthetic_option_exit,
)


def _prices() -> pl.DataFrame:
    return pl.DataFrame({"date": [datetime(2025, 1, 2), datetime(2025, 1, 3), datetime(2025, 1, 6), datetime(2025, 1, 7)], "close": [100.0, 101.0, 102.0, 104.0]})


def _spec() -> FmpSyntheticOptionSpec:
    return FmpSyntheticOptionSpec(
        tenor_days=(30,),
        strike_multipliers=(1.0,),
        realized_vol_window=2,
        vol_floor=0.20,
        vol_cap=0.20,
        premium_floor=0.0,
        spread_bps=0.0,
        min_spread=0.0,
    )


def test_build_fmp_synthetic_option_chain_uses_adjusted_price_basis() -> None:
    chain = build_fmp_synthetic_option_chain(
        _prices(),
        symbol="aapl",
        start_date="2025-01-03",
        end_date="2025-01-03",
        spec=_spec(),
    )

    assert len(chain) == 2
    assert set(chain["option_type"]) == {"call", "put"}
    assert chain.select(["snapshot_date", "contract_symbol"]).unique().height == chain.height
    assert set(chain["option_source"]) == {FMP_SYNTHETIC_OPTION_SOURCE}
    assert set(chain["underlying_price_basis"]) == {FMP_SYNTHETIC_UNDERLYING_PRICE_BASIS}
    assert chain["underlying_price"].to_list() == [101.0, 101.0]
    assert chain["moneyness"].to_list() == pytest.approx([0.0, 0.0])
    assert chain.select(pl.all_horizontal([pl.col(c).is_not_null() for c in ["delta", "gamma", "theta", "vega", "rho", "iv"]])).to_series().all()


def test_read_fmp_synthetic_option_chain_projects_requested_columns() -> None:
    frame = read_fmp_synthetic_option_chain(
        "AAPL",
        start_date="2025-01-03",
        end_date="2025-01-03",
        columns=["snapshot_date", "contract_symbol", "missing_col"],
        spec=_spec(),
        prices=_prices(),
    )

    assert frame.columns == ["snapshot_date", "contract_symbol", "missing_col"]
    assert frame["missing_col"].is_null().all()


def test_price_fmp_synthetic_contract_settles_intrinsic_at_expiration() -> None:
    quote = price_fmp_synthetic_contract(
        symbol="AAPL",
        snapshot_date="2025-01-07",
        option_type="call",
        strike=101.0,
        expiration="2025-01-07",
        spec=_spec(),
        prices=_prices(),
    )

    assert quote is not None
    assert quote["mid"] == pytest.approx(3.0)
    assert quote["bid"] == pytest.approx(3.0)
    assert quote["ask"] == pytest.approx(3.0)


def test_settle_fmp_synthetic_option_exit_uses_quote_before_expiration_and_intrinsic_at_expiration() -> None:
    before_expiration = settle_fmp_synthetic_option_exit(
        symbol="AAPL",
        contract_symbol="FMPBS_AAPL_C_20250107_1010000",
        option_type="call",
        strike=101.0,
        expiration="2025-01-07",
        equity_exit_date="2025-01-06",
        spec=_spec(),
        prices=_prices(),
    )
    at_expiration = settle_fmp_synthetic_option_exit(
        symbol="AAPL",
        contract_symbol="FMPBS_AAPL_C_20250107_1010000",
        option_type="call",
        strike=101.0,
        expiration="2025-01-07",
        equity_exit_date="2025-01-10",
        spec=_spec(),
        prices=_prices(),
    )

    assert before_expiration is not None
    assert before_expiration.price_source == "contract_quote"
    assert at_expiration is not None
    assert at_expiration.price_source == "expiration_intrinsic"
    assert at_expiration.mid == pytest.approx(
        option_intrinsic_value(option_type="call", strike=101.0, underlying_price=104.0)
    )
