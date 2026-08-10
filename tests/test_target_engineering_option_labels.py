from datetime import datetime

import polars as pl
import torch

from quant_warehouse.platforms.data_providers.thetadata.target_engineering import (
    OptionLabelSpec,
    build_option_label_panel,
    build_option_labels,
    compute_return_covariance_matrix,
    solve_long_only_mean_variance_weights,
    solve_mean_variance_weights,
)


def _snapshot(snapshot_date: str, quotes: dict[str, tuple[float, float]]) -> pl.DataFrame:
    return pl.DataFrame([
        {"snapshot_date": snapshot_date, "underlying_symbol": "AAPL", "contract_symbol": symbol, "expiration": "2024-02-16", "strike": float(symbol.split("_")[-1]), "option_type": "call", "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "volume": 10, "open_interest": 20}
        for symbol, (bid, ask) in quotes.items()
    ])


def _trades() -> pl.DataFrame:
    return pl.DataFrame([{"trade_id": "T1", "symbol": "AAPL", "entry_date": "2024-01-02", "exit_date": "2024-01-05", "trade_return": 0.15, "entry_px": 100.0, "exit_px": 115.0}])


def test_build_option_labels_ranks_contracts_within_trade() -> None:
    result = build_option_labels(_trades(), {datetime(2024, 1, 2): _snapshot("2024-01-02", {"AAPL_C_100": (1.0, 1.2), "AAPL_C_110": (0.6, 0.8)}), datetime(2024, 1, 5): _snapshot("2024-01-05", {"AAPL_C_100": (1.7, 1.9), "AAPL_C_110": (1.3, 1.5)})})
    frame = pl.DataFrame(result.option_rows)
    assert frame.height == 2
    assert frame["rank_y"].max() == 1.0
    assert result.statistics["trade_stats"]["trades"] == 1


def test_build_option_label_panel_returns_polars() -> None:
    panel = build_option_label_panel(_trades(), {datetime(2024, 1, 2): _snapshot("2024-01-02", {"AAPL_C_100": (1.0, 1.2)}), datetime(2024, 1, 5): _snapshot("2024-01-05", {"AAPL_C_100": (1.7, 1.9)})})
    assert isinstance(panel, pl.DataFrame)
    assert {"entry_quote", "exit_quote", "option_return_pct", "rank_y"}.issubset(panel.columns)


def test_compute_return_covariance_matrix_uses_option_time_series() -> None:
    returns = pl.DataFrame({"AAPL_C_100": [0.10, 0.12, 0.11], "AAPL_C_110": [0.20, 0.24, 0.22], "AAPL_P_90": [0.01, -0.02, 0.03]})
    cov = compute_return_covariance_matrix(returns, shrinkage=0.0)
    assert cov.shape == (3, 3)
    assert bool(cov[0, 1] > cov[0, 2])
    assert bool(cov[1, 0] == cov[0, 1])


def test_mean_variance_weights_respect_long_only_and_short_modes() -> None:
    expected = torch.tensor([0.2, 0.8], dtype=torch.float64)
    long_weights = solve_long_only_mean_variance_weights(expected, torch.tensor([0.1, 0.2], dtype=torch.float64), risk_aversion=1.0, eligible=torch.ones(2, dtype=torch.bool))
    assert torch.isclose(long_weights.sum(), torch.tensor(1.0, dtype=torch.float64, device=long_weights.device))
    assert bool((long_weights >= 0).all())
    hedged = solve_mean_variance_weights(expected, torch.tensor([0.1, 0.2], dtype=torch.float64), risk_aversion=1.0, long_only=False, max_gross_exposure=2.0)
    assert torch.isclose(hedged.abs().sum(), torch.tensor(2.0, dtype=torch.float64, device=hedged.device))
