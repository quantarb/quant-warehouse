from __future__ import annotations

import pandas as pd

from quant_warehouse.target_engineering import (
    solve_joint_trade_sequence_by_frequency,
    solve_joint_trades_by_frequency,
    solve_optimal_joint_trade_sequence_generic,
    solve_optimal_joint_trades_generic,
    solve_optimal_trades_generic,
    solve_trades_by_frequency,
)


def _frame(values: list[tuple[float, float]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame(values, columns=["low", "high"], index=index)


def test_solve_optimal_trades_generic_long() -> None:
    df = _frame([(10, 11), (8, 9), (12, 13), (7, 8), (15, 16)])

    trades = solve_optimal_trades_generic(df, k=2, side="long", min_profit_pct=0.05)

    assert [(t.entry_row.name, t.exit_row.name, t.entry_price, t.exit_price) for t in trades] == [
        (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05"), 8.0, 15.0),
    ]
    assert [round(t.profit, 6) for t in trades] == [7.0]


def test_solve_optimal_trades_generic_short() -> None:
    df = _frame([(10, 11), (8, 9), (12, 13), (6, 7)])

    trades = solve_optimal_trades_generic(df, k=1, side="short", min_profit_pct=0.10)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.side == "short"
    assert trade.entry_row.name == pd.Timestamp("2024-01-03")
    assert trade.exit_row.name == pd.Timestamp("2024-01-04")
    assert trade.entry_price == 12.0
    assert trade.exit_price == 7.0
    assert trade.profit == 5.0


def test_solve_optimal_joint_trades_generic_can_mix_sides() -> None:
    df = _frame([(10, 11), (14, 15), (9, 10), (7, 8), (13, 14)])

    trades = solve_optimal_joint_trades_generic(df, k=2, min_profit_pct=0.05)

    assert [t.side for t in trades] == ["short", "long"]
    assert [(t.entry_row.name, t.exit_row.name) for t in trades] == [
        (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
        (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")),
    ]
    assert [t.profit for t in trades] == [4.0, 5.0]


def test_solve_joint_trade_sequence_generic_has_no_k_cap() -> None:
    df = _frame([(10, 11), (14, 15), (9, 10), (7, 8), (13, 14)])

    trades = solve_optimal_joint_trade_sequence_generic(df, min_profit_pct=0.05)

    assert [t.side for t in trades] == ["short", "long"]
    assert [(t.entry_row.name, t.exit_row.name) for t in trades] == [
        (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
        (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")),
    ]


def test_solve_trades_by_frequency_accepts_date_column() -> None:
    df = _frame([(10, 11), (8, 9), (12, 13), (7, 8), (15, 16)]).reset_index(names="date")

    trades = solve_trades_by_frequency(df, k=1, freq="ME", side="long", min_profit_pct=0.05)

    assert len(trades) == 1
    assert trades[0]["side"] == "long"
    assert trades[0]["entry_row"].name == pd.Timestamp("2024-01-04")
    assert trades[0]["exit_row"].name == pd.Timestamp("2024-01-05")
    assert trades[0]["period_label"] == "M:2024-01-31"


def test_solve_joint_trades_by_frequency_returns_dict_payloads() -> None:
    df = _frame([(10, 11), (14, 15), (9, 10), (7, 8), (13, 14)])

    trades = solve_joint_trades_by_frequency(df, k=2, freq="ME", min_profit_pct=0.05)

    assert [row["side"] for row in trades] == ["short", "long"]
    assert {row["period_label"] for row in trades} == {"M:2024-01-31"}
    assert trades[0]["entry_price"] == 14.0
    assert trades[0]["exit_price"] == 10.0


def test_solve_joint_trade_sequence_by_frequency_returns_dict_payloads() -> None:
    df = _frame([(10, 11), (14, 15), (9, 10), (7, 8), (13, 14)])

    trades = solve_joint_trade_sequence_by_frequency(df, freq="ME", min_profit_pct=0.05)

    assert [row["side"] for row in trades] == ["short", "long"]
    assert {row["period_label"] for row in trades} == {"M:2024-01-31"}
