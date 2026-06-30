from __future__ import annotations

import pandas as pd

import numpy as np

from quant_warehouse.platforms.data_providers.thetadata.target_engineering import (
    OptionLabelSpec,
    build_option_label_panel,
    build_option_labels,
    compute_return_covariance_matrix,
    solve_long_only_mean_variance_weights,
    solve_mean_variance_weights,
)
from quant_warehouse.platforms.data_providers.thetadata.target_engineering.option_labels import _build_trade_window_price_panel


def _option_snapshot(snapshot_date: str, returns: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for contract_symbol, (bid, ask) in returns.items():
        option_type = "call" if "C" in contract_symbol else "put"
        strike = float(contract_symbol.split("_")[-1])
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "underlying_symbol": "AAPL",
                "contract_symbol": contract_symbol,
                "expiration": "2024-02-16",
                "strike": strike,
                "option_type": option_type,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "volume": 10,
                "open_interest": 20,
            }
        )
    return pd.DataFrame(rows)


def _trades(*, entry_px: float = 100.0, exit_px: float = 115.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": "T1",
                "symbol": "AAPL",
                "entry_date": "2024-01-02",
                "exit_date": "2024-01-05",
                "trade_return": 0.15,
                "entry_px": entry_px,
                "exit_px": exit_px,
            }
        ]
    )


def test_build_option_labels_ranks_contracts_within_trade() -> None:
    trades = _trades()
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {
                "AAPL_C_100": (1.0, 1.2),
                "AAPL_C_110": (0.6, 0.8),
                "AAPL_P_90": (1.5, 1.7),
            },
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {
                "AAPL_C_100": (1.7, 1.9),
                "AAPL_C_110": (1.3, 1.5),
                "AAPL_P_90": (1.0, 1.1),
            },
        ),
    }

    result = build_option_labels(trades, snapshots)
    df = pd.DataFrame(result.option_rows)

    assert set(df["contract_symbol"]) == {"AAPL_C_100", "AAPL_C_110", "AAPL_P_90"}
    assert df.loc[df["contract_symbol"] == "AAPL_C_110", "option_return_pct"].iloc[0] > df.loc[
        df["contract_symbol"] == "AAPL_C_100", "option_return_pct"
    ].iloc[0]
    assert df.loc[df["contract_symbol"] == "AAPL_C_110", "rank_y"].iloc[0] == 1.0
    assert df.sort_values("rank_y", ascending=False)["contract_symbol"].iloc[0] == "AAPL_C_110"
    assert df["rank_y"].max() == 1.0
    assert result.statistics["trade_stats"]["trades"] == 1
    assert result.statistics["trade_stats"]["contracts"] == 3


def test_build_option_labels_uses_prior_snapshot_fallback() -> None:
    trades = _trades()
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {
                "AAPL_C_100": (1.0, 1.2),
                "AAPL_C_110": (0.6, 0.8),
            },
        ),
        pd.Timestamp("2024-01-04"): _option_snapshot(
            "2024-01-04",
            {
                "AAPL_C_100": (1.6, 1.8),
                "AAPL_C_110": (1.0, 1.2),
            },
        ),
    }

    spec = OptionLabelSpec(entry_quote_col="ask", exit_quote_col="bid")
    result = build_option_labels(trades, snapshots, spec=spec)
    df = pd.DataFrame(result.option_rows)

    assert len(df) == 2
    assert set(df["contract_symbol"]) == {"AAPL_C_100", "AAPL_C_110"}
    assert df["exit_snapshot_date"].astype(str).nunique() == 1
    assert df["exit_snapshot_date"].astype(str).iloc[0].startswith("2024-01-04")


def test_build_option_label_panel_returns_dataframe() -> None:
    trades = _trades()
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {
                "AAPL_C_100": (1.0, 1.2),
                "AAPL_C_110": (0.6, 0.8),
            },
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {
                "AAPL_C_100": (1.7, 1.9),
                "AAPL_C_110": (1.3, 1.5),
            },
        ),
    }

    panel = build_option_label_panel(trades, snapshots)

    assert not panel.empty
    assert {"trade_id", "contract_symbol", "rank_y", "option_return_pct"}.issubset(panel.columns)
    assert panel.iloc[0]["trade_id"] == "T1"


def test_solve_long_only_mean_variance_weights_no_short_selling() -> None:
    weights = solve_long_only_mean_variance_weights(
        [0.10, -0.05, 0.20],
        [0.04, 0.04, 0.01],
        risk_aversion=1.0,
        eligible=[True, True, True],
    )
    assert np.all(weights >= 0.0)
    assert weights[1] == 0.0
    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert weights[2] > weights[0]


def test_build_option_labels_mean_variance_includes_equity_and_zeroes_worthless() -> None:
    trades = _trades(entry_px=100.0, exit_px=115.0)
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {
                "AAPL_C_100": (1.0, 1.2),
                "AAPL_C_110": (0.6, 0.8),
                "AAPL_P_90": (1.5, 1.7),
            },
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {
                "AAPL_C_100": (1.7, 1.9),
                "AAPL_C_110": (0.005, 0.005),
                "AAPL_P_90": (1.0, 1.1),
            },
        ),
    }

    spec = OptionLabelSpec(label_method="mean_variance", include_equity=True, worthless_exit_threshold=0.01)
    result = build_option_labels(trades, snapshots, spec=spec)
    df = pd.DataFrame(result.option_rows)

    assert "AAPL_EQUITY" in set(df["contract_symbol"])
    worthless = df.loc[df["contract_symbol"] == "AAPL_C_110"].iloc[0]
    assert bool(worthless["expires_worthless"])
    assert worthless["mv_weight"] == 0.0
    assert worthless["label"] == 0.0
    assert abs(float(df["mv_weight"].sum()) - 1.0) < 1e-9
    assert (df["mv_weight"] >= 0.0).all()
    equity = df.loc[df["contract_symbol"] == "AAPL_EQUITY"].iloc[0]
    assert equity["option_return_pct"] == 0.15
    assert equity["mv_weight"] > 0.0


def test_build_option_labels_mean_variance_sets_label_to_weight() -> None:
    trades = _trades(entry_px=100.0, exit_px=110.0)
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {"AAPL_C_100": (1.0, 1.2)},
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {"AAPL_C_100": (1.7, 1.9)},
        ),
    }
    spec = OptionLabelSpec(label_method="mean_variance", include_equity=True)
    result = build_option_labels(trades, snapshots, spec=spec)
    df = pd.DataFrame(result.option_rows)
    assert (df["label"] == df["mv_weight"]).all()


def test_solve_mean_variance_weights_allows_short_selling() -> None:
    weights = solve_mean_variance_weights(
        [0.10, -0.05, 0.20],
        [0.04, 0.04, 0.01],
        risk_aversion=1.0,
        eligible=[True, True, True],
        long_only=False,
    )
    assert weights[1] < 0.0
    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert weights[2] > weights[0]


def test_build_option_labels_mean_variance_short_selling_can_short_losers() -> None:
    trades = _trades(entry_px=100.0, exit_px=90.0)
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {"AAPL_C_100": (2.0, 2.2), "AAPL_P_90": (1.0, 1.2)},
        ),
        pd.Timestamp("2024-01-03"): _option_snapshot(
            "2024-01-03",
            {"AAPL_C_100": (1.8, 2.0), "AAPL_P_90": (1.2, 1.4)},
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {"AAPL_C_100": (0.5, 0.7), "AAPL_P_90": (2.0, 2.2)},
        ),
    }
    spec = OptionLabelSpec(
        label_method="mean_variance",
        include_equity=False,
        allow_short_selling=True,
        covariance_shrinkage=0.0,
    )
    result = build_option_labels(trades, snapshots, spec=spec)
    df = pd.DataFrame(result.option_rows)
    call = df.loc[df["contract_symbol"] == "AAPL_C_100"].iloc[0]
    put = df.loc[df["contract_symbol"] == "AAPL_P_90"].iloc[0]
    assert call["option_return_pct"] < 0.0
    assert put["option_return_pct"] > 0.0
    assert call["rank_y"] < put["rank_y"]
    assert put["mv_weight"] > call["mv_weight"]
    assert abs(float(df["mv_weight"].sum()) - 1.0) < 1e-9


def test_diversified_mean_variance_caps_single_name_weight() -> None:
    trades = _trades(entry_px=100.0, exit_px=115.0)
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {"AAPL_C_100": (1.0, 1.2), "AAPL_C_110": (0.5, 0.7), "AAPL_P_90": (1.5, 1.7)},
        ),
        pd.Timestamp("2024-01-03"): _option_snapshot(
            "2024-01-03",
            {"AAPL_C_100": (1.1, 1.3), "AAPL_C_110": (0.55, 0.75), "AAPL_P_90": (1.6, 1.8)},
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {"AAPL_C_100": (1.7, 1.9), "AAPL_C_110": (1.3, 1.5), "AAPL_P_90": (1.0, 1.1)},
        ),
    }
    spec = OptionLabelSpec.diversified_mean_variance(include_equity=False)
    result = build_option_labels(trades, snapshots, spec=spec)
    df = pd.DataFrame(result.option_rows)
    # With only three eligible legs, the cap relaxes to 1/3 so the budget can still sum to one.
    assert df["mv_weight"].max() <= (1.0 / 3.0) + 1e-6
    assert (df["mv_weight"] > 0).sum() >= 2
    assert abs(float(df["mv_weight"].sum()) - 1.0) < 1e-9


def test_hedged_mean_variance_caps_gross_exposure() -> None:
    weights = solve_mean_variance_weights(
        [2.0, -1.0, 0.5],
        [0.04, 0.04, 0.01],
        risk_aversion=1.0,
        eligible=[True, True, True],
        long_only=False,
        max_weight=0.10,
        max_gross_exposure=2.0,
        return_shrinkage=0.5,
    )
    assert float(np.abs(weights).sum()) <= 2.0 + 1e-6


def test_mean_variance_weights_higher_rank_than_lower() -> None:
    trades = _trades(entry_px=100.0, exit_px=115.0)
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {"AAPL_C_100": (1.0, 1.2), "AAPL_C_110": (0.5, 0.7), "AAPL_P_90": (1.5, 1.7)},
        ),
        pd.Timestamp("2024-01-03"): _option_snapshot(
            "2024-01-03",
            {"AAPL_C_100": (1.1, 1.3), "AAPL_C_110": (0.55, 0.75), "AAPL_P_90": (1.6, 1.8)},
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {"AAPL_C_100": (1.7, 1.9), "AAPL_C_110": (1.3, 1.5), "AAPL_P_90": (1.0, 1.1)},
        ),
    }
    spec = OptionLabelSpec(
        label_method="mean_variance",
        include_equity=False,
        covariance_shrinkage=0.0,
        risk_aversion=1.0,
    )
    result = build_option_labels(trades, snapshots, spec=spec)
    df = pd.DataFrame(result.option_rows)

    top = df.loc[df["rank_y"].idxmax(), "contract_symbol"]
    bottom = df.loc[df["rank_y"].idxmin(), "contract_symbol"]
    assert (
        df.loc[df["contract_symbol"] == top, "mv_weight"].iloc[0]
        > df.loc[df["contract_symbol"] == bottom, "mv_weight"].iloc[0]
    )
    assert abs(float(df["mv_weight"].sum()) - 1.0) < 1e-9


def test_compute_return_covariance_matrix_uses_option_time_series() -> None:
    returns = pd.DataFrame(
        {
            "AAPL_C_100": [0.10, 0.12, 0.11],
            "AAPL_C_110": [0.20, 0.24, 0.22],
            "AAPL_P_90": [0.01, -0.02, 0.03],
        }
    )
    cov = compute_return_covariance_matrix(returns, shrinkage=0.0)
    assert cov.shape == (3, 3)
    assert cov[0, 1] > cov[0, 2]
    assert cov[1, 0] == cov[0, 1]


def test_hybrid_blends_rank_and_normalized_return_for_mv_mu() -> None:
    trades = _trades(entry_px=100.0, exit_px=115.0)
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {"AAPL_C_100": (1.0, 1.2), "AAPL_C_110": (0.5, 0.7), "AAPL_P_90": (1.5, 1.7)},
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {"AAPL_C_100": (1.7, 1.9), "AAPL_C_110": (1.3, 1.5), "AAPL_P_90": (1.0, 1.1)},
        ),
    }

    rank_spec = OptionLabelSpec(label_method="mean_variance", include_equity=False, covariance_shrinkage=0.0)
    hybrid_spec = OptionLabelSpec(
        label_method="hybrid",
        include_equity=False,
        covariance_shrinkage=0.0,
        hybrid_rank_weight=0.5,
    )
    return_spec = OptionLabelSpec(
        label_method="hybrid",
        include_equity=False,
        covariance_shrinkage=0.0,
        hybrid_rank_weight=0.0,
    )

    rank_df = pd.DataFrame(build_option_labels(trades, snapshots, spec=rank_spec).option_rows)
    hybrid_df = pd.DataFrame(build_option_labels(trades, snapshots, spec=hybrid_spec).option_rows)
    return_df = pd.DataFrame(build_option_labels(trades, snapshots, spec=return_spec).option_rows)

    assert (hybrid_df["label"] == hybrid_df["mv_weight"]).all()
    expected_mu = 0.5 * rank_df["mv_mu"].to_numpy(dtype=float) + 0.5 * return_df["mv_mu"].to_numpy(dtype=float)
    assert np.allclose(hybrid_df["mv_mu"].to_numpy(dtype=float), expected_mu)
    assert abs(float(hybrid_df["mv_weight"].sum()) - 1.0) < 1e-9
    assert not np.allclose(hybrid_df["mv_weight"].to_numpy(dtype=float), rank_df["mv_weight"].to_numpy(dtype=float))


def test_hybrid_rank_weight_one_matches_mean_variance() -> None:
    trades = _trades(entry_px=100.0, exit_px=115.0)
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {"AAPL_C_100": (1.0, 1.2), "AAPL_C_110": (0.5, 0.7)},
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {"AAPL_C_100": (1.7, 1.9), "AAPL_C_110": (1.3, 1.5)},
        ),
    }
    rank_spec = OptionLabelSpec(label_method="mean_variance", include_equity=False)
    hybrid_spec = OptionLabelSpec(label_method="hybrid", include_equity=False, hybrid_rank_weight=1.0)
    rank_df = pd.DataFrame(build_option_labels(trades, snapshots, spec=rank_spec).option_rows)
    hybrid_df = pd.DataFrame(build_option_labels(trades, snapshots, spec=hybrid_spec).option_rows)
    assert np.allclose(rank_df["mv_weight"].to_numpy(dtype=float), hybrid_df["mv_weight"].to_numpy(dtype=float))


def test_mean_variance_uses_snapshot_covariance_when_time_series_available() -> None:
    trades = _trades(entry_px=100.0, exit_px=115.0)
    snapshots = {
        pd.Timestamp("2024-01-02"): _option_snapshot(
            "2024-01-02",
            {"AAPL_C_100": (1.0, 1.2), "AAPL_C_110": (0.5, 0.7)},
        ),
        pd.Timestamp("2024-01-03"): _option_snapshot(
            "2024-01-03",
            {"AAPL_C_100": (1.1, 1.3), "AAPL_C_110": (0.55, 0.75)},
        ),
        pd.Timestamp("2024-01-04"): _option_snapshot(
            "2024-01-04",
            {"AAPL_C_100": (1.3, 1.5), "AAPL_C_110": (0.65, 0.85)},
        ),
        pd.Timestamp("2024-01-05"): _option_snapshot(
            "2024-01-05",
            {"AAPL_C_100": (1.7, 1.9), "AAPL_C_110": (1.3, 1.5)},
        ),
    }
    spec = OptionLabelSpec(label_method="mean_variance", include_equity=False, covariance_shrinkage=0.0)

    price_panel = _build_trade_window_price_panel(
        snapshots,
        contract_symbols=["AAPL_C_100", "AAPL_C_110"],
        trade=trades.iloc[0].to_dict(),
        entry_dt=pd.Timestamp("2024-01-02"),
        exit_dt=pd.Timestamp("2024-01-05"),
        underlying_symbol="AAPL",
        spec=spec,
    )
    returns = price_panel.pct_change().dropna()
    cov = compute_return_covariance_matrix(returns, shrinkage=0.0)
    assert cov.shape == (2, 2)
    assert cov[0, 1] > 0.0

    result = build_option_labels(trades, snapshots, spec=spec)
    df = pd.DataFrame(result.option_rows)
    diagonal_weights = solve_long_only_mean_variance_weights(
        df["rank_y"].to_numpy(dtype=float),
        (df["entry_quote"].to_numpy(dtype=float) ** 2),
        risk_aversion=spec.risk_aversion,
        eligible=np.ones(len(df), dtype=bool),
    )
    assert abs(float(df["mv_weight"].sum()) - 1.0) < 1e-9
    assert not np.allclose(df["mv_weight"].to_numpy(dtype=float), diagonal_weights)
