from __future__ import annotations

import polars as pl
from datetime import datetime, timedelta

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    solve_optimal_trades_generic,
    solve_trades_by_frequency,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.strategy_solver import solve_side_trades_by_frequency_batched_multi_k


def _frame(values: list[tuple[float, float]]) -> pl.DataFrame:
    dates = pl.Series("date", [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))])
    return pl.DataFrame(values, schema=["low", "high"]).with_columns(dates.alias("date"))


def test_solve_optimal_trades_generic_long() -> None:
    df = _frame([(10, 11), (8, 9), (12, 13), (7, 8), (15, 16)])

    trades = solve_optimal_trades_generic(df, k=2, side="long", min_profit_pct=0.05)

    assert [(t.entry_row["date"], t.exit_row["date"], t.entry_price, t.exit_price) for t in trades] == [
        (datetime(2024, 1, 2), datetime(2024, 1, 3), 9.0, 12.0),
        (datetime(2024, 1, 4), datetime(2024, 1, 5), 8.0, 15.0),
    ]
    assert [round(t.profit, 6) for t in trades] == [3.0, 7.0]


def test_solve_optimal_trades_generic_short() -> None:
    df = _frame([(10, 11), (8, 9), (12, 13), (6, 7)])

    trades = solve_optimal_trades_generic(df, k=1, side="short", min_profit_pct=0.10)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.side == "short"
    assert trade.entry_row["date"] == datetime(2024, 1, 3)
    assert trade.exit_row["date"] == datetime(2024, 1, 4)
    assert trade.entry_price == 12.0
    assert trade.exit_price == 7.0
    assert trade.profit == 5.0


def test_solve_trades_by_frequency_accepts_date_column() -> None:
    df = _frame([(10, 11), (8, 9), (12, 13), (7, 8), (15, 16)])

    trades = solve_trades_by_frequency(df, k=1, freq="ME", side="long", min_profit_pct=0.05)

    assert len(trades) == 1
    assert trades[0]["side"] == "long"
    assert trades[0]["entry_row"]["date"] == datetime(2024, 1, 4)
    assert trades[0]["exit_row"]["date"] == datetime(2024, 1, 5)
    assert trades[0]["period_label"] == "M:2024-01-01"


def test_solve_side_trades_by_frequency_batched_multi_k_solves_sides_independently() -> None:
    frames = {
        "AAA": _frame([(10, 11), (14, 15), (9, 10), (7, 8), (13, 14)]),
        "BBB": _frame([(20, 21), (18, 19), (24, 25), (16, 17), (26, 27)]),
    }

    cpu = solve_side_trades_by_frequency_batched_multi_k(
        frames,
        ks=(1, 2),
        freq="ME",
        min_profit_pct=0.05,
    )

    assert set(cpu) == {1, 2}
    assert set(cpu[1]) == {"AAA", "BBB"}
    assert all(row["period_label"] == "M:2024-01-01" for rows in cpu[2].values() for row in rows)
    assert sum(len(rows) for rows in cpu[2].values()) >= sum(len(rows) for rows in cpu[1].values())
    assert {row["side"] for rows in cpu[2].values() for row in rows} == {"long", "short"}
