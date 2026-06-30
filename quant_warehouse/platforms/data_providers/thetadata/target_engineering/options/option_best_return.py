from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

METADATA_COLS = ("option_type", "strike", "expiration", "dte", "moneyness")


def build_option_best_return_labels(
    option_returns: pd.DataFrame,
    group_cols: Sequence[str] = ("underlying_symbol", "date"),
    option_symbol_col: str = "option_symbol",
    entry_price_col: str = "entry_price",
    exit_price_col: str = "exit_price",
) -> pd.DataFrame:
    """Select the best realized-return option contract per group."""

    if option_returns is None or option_returns.empty:
        return pd.DataFrame()
    group_cols = tuple(group_cols)
    _require_columns(
        option_returns,
        [*group_cols, option_symbol_col, entry_price_col, exit_price_col],
        ctx="build_option_best_return_labels",
    )

    df = option_returns.copy()
    df["option_return"] = _realized_return(df, entry_price_col, exit_price_col)
    idx = df.groupby(list(group_cols), dropna=False)["option_return"].idxmax()
    best = df.loc[idx].copy().sort_values(list(group_cols)).reset_index(drop=True)
    best["best_option_symbol"] = best[option_symbol_col]
    best["best_option_return"] = best["option_return"]
    best["target_name"] = "option_best_return"
    best["target_value"] = best["best_option_return"]
    keep = [*group_cols, "target_name", "target_value", "best_option_symbol", "best_option_return"]
    keep.extend([col for col in METADATA_COLS if col in best.columns])
    return best[keep]


def _realized_return(df: pd.DataFrame, entry_col: str, exit_col: str) -> pd.Series:
    entry = pd.to_numeric(df[entry_col], errors="coerce")
    exit_ = pd.to_numeric(df[exit_col], errors="coerce")
    return (exit_ / entry) - 1.0


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
