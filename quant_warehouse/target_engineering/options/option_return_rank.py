from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def build_option_return_rank_labels(
    option_returns: pd.DataFrame,
    group_cols: Sequence[str] = ("underlying_symbol", "date"),
    option_symbol_col: str = "option_symbol",
    entry_price_col: str = "entry_price",
    exit_price_col: str = "exit_price",
    pct: bool = True,
) -> pd.DataFrame:
    """Rank option contracts by realized return within each group."""

    if option_returns is None or option_returns.empty:
        return pd.DataFrame()
    group_cols = tuple(group_cols)
    _require_columns(
        option_returns,
        [*group_cols, option_symbol_col, entry_price_col, exit_price_col],
        ctx="build_option_return_rank_labels",
    )

    out = option_returns.copy()
    out["option_return"] = _realized_return(out, entry_price_col, exit_price_col)
    out["option_return_rank"] = out.groupby(list(group_cols), dropna=False)["option_return"].rank(
        method="average",
        ascending=False,
    )
    if pct:
        out["option_return_percentile"] = out.groupby(list(group_cols), dropna=False)["option_return"].rank(
            method="average",
            pct=True,
        )
        out["target_value"] = out["option_return_percentile"]
        out["target_name"] = "option_return_percentile"
    else:
        out["target_value"] = out["option_return_rank"]
        out["target_name"] = "option_return_rank"
    return out.sort_values([*group_cols, "option_return_rank"], ignore_index=True)


def _realized_return(df: pd.DataFrame, entry_col: str, exit_col: str) -> pd.Series:
    entry = pd.to_numeric(df[entry_col], errors="coerce")
    exit_ = pd.to_numeric(df[exit_col], errors="coerce")
    return (exit_ / entry) - 1.0


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
