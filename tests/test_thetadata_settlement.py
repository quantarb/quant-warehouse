from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from quant_warehouse.platforms.data_providers.thetadata.settlement import (
    iter_option_exit_lookup_dates,
    lookup_contract_quote,
    option_intrinsic_value,
    settle_option_exit,
)


def test_settle_option_exit_uses_last_contract_quote_before_expiration() -> None:
    quotes = {
        (datetime(2021, 2, 4), "GOOG_put_20210205_2015"): (0.10, 1.20, 0.65),
    }

    def quote_loader(symbol: str, date: datetime, contract_symbol: str):
        assert symbol == "GOOG"
        return quotes.get((date.replace(hour=0, minute=0, second=0, microsecond=0), contract_symbol))

    settlement = settle_option_exit(
        symbol="GOOG",
        contract_symbol="GOOG_put_20210205_2015",
        option_type="put",
        strike=2015.0,
        expiration=datetime(2021, 2, 5),
        equity_exit_date=datetime(2021, 2, 16),
        quote_loader=quote_loader,
        underlying_close_loader=lambda symbol, date: 104.05,
        entry_date=datetime(2021, 2, 3),
        exit_lookback_days=3,
    )

    assert settlement is not None
    assert settlement.snapshot_date == datetime(2021, 2, 4)
    assert settlement.price_source == "last_contract_quote"
    assert settlement.bid == 0.10
    assert settlement.ask == 1.20
    assert settlement.mid == 0.65


def test_settle_option_exit_blocks_scale_mismatched_intrinsic_fallback() -> None:
    settlement = settle_option_exit(
        symbol="GOOG",
        contract_symbol="GOOG_put_20210205_2015",
        option_type="put",
        strike=2015.0,
        expiration=datetime(2021, 2, 5),
        equity_exit_date=datetime(2021, 2, 16),
        quote_loader=lambda symbol, date, contract_symbol: None,
        underlying_close_loader=lambda symbol, date: 104.05,
        exit_lookback_days=3,
    )

    assert settlement is None


def test_settle_option_exit_allows_same_scale_intrinsic_fallback() -> None:
    settlement = settle_option_exit(
        symbol="MU",
        contract_symbol="MU_call_20221021_55",
        option_type="call",
        strike=55.0,
        expiration=datetime(2022, 10, 21),
        equity_exit_date=datetime(2022, 11, 1),
        quote_loader=lambda symbol, date, contract_symbol: None,
        underlying_close_loader=lambda symbol, date: 56.25,
        exit_lookback_days=3,
    )

    assert settlement is not None
    assert settlement.snapshot_date == datetime(2022, 10, 21)
    assert settlement.price_source == "expiration_intrinsic"
    assert settlement.bid == 1.25
    assert settlement.ask == 1.25
    assert settlement.mid == 1.25


def test_lookup_contract_quote_fills_missing_bid_ask_from_mid() -> None:
    quote = lookup_contract_quote(
        pl.DataFrame(
            {
                "contract_symbol": ["AAPL_call_20250117_100"],
                "bid": [None],
                "ask": [None],
                "mid": [4.2],
            }
        ),
        "AAPL_call_20250117_100",
    )

    assert quote == (4.2, 4.2, 4.2)


def test_iter_option_exit_lookup_dates_includes_weekend_target_then_prior_business_days() -> None:
    dates = iter_option_exit_lookup_dates(datetime(2025, 1, 11), 2)

    assert dates == (
        datetime(2025, 1, 11), datetime(2025, 1, 10), datetime(2025, 1, 9),
    )


def test_option_intrinsic_value_requires_matching_price_scale() -> None:
    assert option_intrinsic_value(option_type="put", strike=2015.0, underlying_price=104.05) is None
    assert option_intrinsic_value(option_type="put", strike=105.0, underlying_price=104.05) == pytest.approx(0.95)
