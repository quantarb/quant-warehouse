from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_forward_return_labels(
    prices: pd.DataFrame,
    horizons: Sequence[int],
    price_col: str = "close",
    symbol_col: str = "symbol",
    date_col: str = "date",
    log_return: bool = False,
) -> pd.DataFrame:
    """Build supervised labels from the return between t and t + horizon."""

    if prices is None or prices.empty:
        return pd.DataFrame()
    _require_columns(prices, [symbol_col, date_col, price_col], ctx="build_forward_return_labels")
    horizon_values = _normalize_horizons(horizons)

    df = prices[[symbol_col, date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[symbol_col, date_col]).sort_values([symbol_col, date_col])

    frames: list[pd.DataFrame] = []
    grouped = df.groupby(symbol_col, sort=False, group_keys=False)
    for horizon in horizon_values:
        out = df[[symbol_col, date_col]].copy()
        future = grouped[price_col].shift(-horizon)
        current = df[price_col]
        if log_return:
            out["target_value"] = np.log(future / current)
            target_name = f"forward_log_return_{horizon}d"
        else:
            out["target_value"] = (future / current) - 1.0
            target_name = f"forward_return_{horizon}d"
        out["horizon"] = int(horizon)
        out["target_name"] = target_name
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)[
        [symbol_col, date_col, "horizon", "target_name", "target_value"]
    ].sort_values([symbol_col, date_col, "horizon"], ignore_index=True)


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
