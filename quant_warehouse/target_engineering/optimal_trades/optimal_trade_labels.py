from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def build_optimal_trade_labels(
    prices: pd.DataFrame,
    horizons: Sequence[int],
    price_col: str = "close",
    symbol_col: str = "symbol",
    date_col: str = "date",
    allow_short: bool = True,
) -> pd.DataFrame:
    """Compute the best current-close entry trade over each future window.

    Entry is the current row's close. Exits are future closes from t + 1 through
    t + horizon, inclusive.
    """

    if prices is None or prices.empty:
        return pd.DataFrame()
    _require_columns(prices, [symbol_col, date_col, price_col], ctx="build_optimal_trade_labels")
    horizon_values = _normalize_horizons(horizons)

    df = prices[[symbol_col, date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[symbol_col, date_col]).sort_values([symbol_col, date_col])

    rows: list[dict[str, Any]] = []
    for symbol, group in df.groupby(symbol_col, sort=False):
        group = group.sort_values(date_col).reset_index(drop=True)
        dates = group[date_col].to_numpy()
        values = group[price_col].to_numpy(dtype=float)
        for idx, entry_price in enumerate(values):
            for horizon in horizon_values:
                future_start = idx + 1
                future_end = min(idx + horizon + 1, len(group))
                future_prices = values[future_start:future_end]
                future_dates = dates[future_start:future_end]
                row = {
                    symbol_col: symbol,
                    date_col: pd.Timestamp(dates[idx]),
                    "horizon": int(horizon),
                    "target_name": f"optimal_trade_{horizon}d",
                    "entry_date": pd.Timestamp(dates[idx]),
                    "entry_price": float(entry_price) if np.isfinite(entry_price) else np.nan,
                    "optimal_entry_date": pd.Timestamp(dates[idx]),
                    "optimal_exit_date": pd.NaT,
                    "optimal_side": "hold",
                    "optimal_return": 0.0,
                    "long_best_return": np.nan,
                    "short_best_return": np.nan,
                    "target_value": 0.0,
                }
                if future_prices.size == 0 or not np.isfinite(entry_price) or entry_price <= 0:
                    rows.append(row)
                    continue

                long_returns = (future_prices / entry_price) - 1.0
                long_idx = int(np.nanargmax(long_returns))
                long_best = float(long_returns[long_idx])
                short_best = np.nan
                short_idx = long_idx
                if allow_short:
                    short_returns = (entry_price / future_prices) - 1.0
                    short_idx = int(np.nanargmax(short_returns))
                    short_best = float(short_returns[short_idx])

                side = "hold"
                best_return = 0.0
                exit_idx = long_idx
                if long_best > 0.0:
                    side = "long"
                    best_return = long_best
                    exit_idx = long_idx
                if allow_short and np.isfinite(short_best) and short_best > best_return:
                    side = "short"
                    best_return = short_best
                    exit_idx = short_idx

                row.update(
                    {
                        "optimal_exit_date": pd.Timestamp(future_dates[exit_idx]) if side != "hold" else pd.NaT,
                        "optimal_side": side,
                        "optimal_return": float(best_return),
                        "long_best_return": long_best,
                        "short_best_return": short_best,
                        "target_value": float(best_return),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows).sort_values([symbol_col, date_col, "horizon"], ignore_index=True)


def _normalize_horizons(horizons: Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in horizons or []:
        value = int(raw)
        if value <= 0:
            raise ValueError("horizons must contain positive integers")
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
