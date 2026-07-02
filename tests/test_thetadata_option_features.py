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


def test_build_option_contract_features_computes_missing_black_scholes_greeks() -> None:
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

    result = build_option_contract_features(chain, underlying_price=100.0)
    row = result.df.iloc[0]

    assert row["iv"] == pytest.approx(0.20, abs=1e-3)
    assert row["delta"] == pytest.approx(0.5114, abs=1e-3)
    assert row["gamma"] == pytest.approx(0.0695, abs=1e-3)
    assert row["theta"] == pytest.approx(-0.0381, abs=1e-3)
    assert row["vega"] == pytest.approx(11.4326, abs=1e-3)
    assert row["rho"] == pytest.approx(4.0156, abs=1e-3)
    assert row["iv_model_source"] == "black_scholes_implied"
    assert row["greeks_model_source"] == "black_scholes"
    assert "greeks" in result.family_cols
    assert "iv_surface" in result.family_cols
    assert "delta" in option_ranker_feature_columns(result.df)
