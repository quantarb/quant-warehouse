from __future__ import annotations

import pandas as pd
import pytest

from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import (
    build_option_contract_features,
    option_ranker_feature_columns,
)


def test_build_option_contract_features_adds_liquidity_greeks_and_iv() -> None:
    chain = pd.DataFrame(
        {
            "underlying_symbol": ["AAPL", "AAPL"],
            "snapshot_date": ["2025-01-02", "2025-01-02"],
            "expiration": ["2025-02-21", "2025-02-21"],
            "option_type": ["call", "call"],
            "strike": [100.0, 105.0],
            "bid": [4.8, 2.7],
            "ask": [5.2, 3.3],
            "delta": [0.55, 0.40],
            "gamma": [0.03, 0.04],
            "theta": [-0.05, -0.04],
            "vega": [0.20, 0.18],
            "iv": [0.30, 0.34],
            "volume": [100, 20],
            "open_interest": [1000, 250],
        }
    )

    result = build_option_contract_features(chain, underlying_price=100.0, target_dte=45)

    assert result.family_name == "option-historical-price-eod"
    assert result.endpoint_name == "option_history_greeks_eod"
    assert result.source_asset_class == "option"
    assert result.presence is not None
    assert "contract_static" in result.family_cols
    assert "liquidity" in result.family_cols
    assert "greeks" in result.family_cols
    assert "iv_surface" in result.family_cols
    assert result.df.loc[0, "dte"] == 50
    assert result.df.loc[0, "dte_gap"] == 5
    assert result.df.loc[0, "moneyness"] == 0.0
    assert result.df.loc[0, "spread_pct"] == pytest.approx(0.08)
    assert result.df.loc[0, "abs_delta"] == 0.55
    assert result.df.loc[0, "theta_to_mid"] == pytest.approx(-0.01)
    assert "iv_expiration_z" in result.feature_cols


def test_option_ranker_feature_columns_prefers_available_greeks() -> None:
    frame = pd.DataFrame(
        {
            "dte": [30],
            "delta": [0.5],
            "abs_delta": [0.5],
            "theta_to_mid": [-0.01],
            "all_nan": [None],
            "realized_holding_days": [20],
            "realized_underlying_trade_return": [0.10],
            "planned_holding_days": [30],
            "equity_signal_score": [0.75],
        }
    )

    cols = option_ranker_feature_columns(frame)

    assert cols == ["dte", "delta", "abs_delta", "theta_to_mid"]
    assert "realized_holding_days" not in cols
    assert "realized_underlying_trade_return" not in cols
    assert "planned_holding_days" not in cols
    assert "equity_signal_score" not in cols


def test_build_option_contract_features_does_not_impute_missing_vendor_greeks() -> None:
    chain = pd.DataFrame(
        {
            "underlying_symbol": ["AAPL"],
            "snapshot_date": ["2025-01-02"],
            "expiration": ["2025-02-01"],
            "option_type": ["call"],
            "strike": [100.0],
            "bid": [2.20],
            "ask": [2.3743012561],
            "volume": [100],
            "open_interest": [1000],
        }
    )

    result = build_option_contract_features(
        chain,
        underlying_price=100.0,
        compute_model_greeks=True,
    )
    row = result.df.iloc[0]

    assert "iv" not in result.df.columns
    assert "iv_model_source" not in result.df.columns
    assert "greeks_model_source" not in result.df.columns
    assert "greeks" not in result.family_cols
    assert "iv_surface" not in result.family_cols
    assert "delta" not in option_ranker_feature_columns(result.df)
    assert row["dte"] == 30
    assert row["spread_pct"] == pytest.approx((2.3743012561 - 2.20) / ((2.20 + 2.3743012561) / 2.0))
