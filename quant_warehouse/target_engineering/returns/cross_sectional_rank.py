from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def build_cross_sectional_rank_labels(
    forward_returns: pd.DataFrame,
    value_col: str = "target_value",
    date_col: str = "date",
    horizon_col: str = "horizon",
    symbol_col: str = "symbol",
    pct: bool = True,
) -> pd.DataFrame:
    """Rank symbols by future return for each date and horizon."""

    if forward_returns is None or forward_returns.empty:
        return pd.DataFrame()
    _require_columns(
        forward_returns,
        [symbol_col, date_col, horizon_col, value_col],
        ctx="build_cross_sectional_rank_labels",
    )

    out = forward_returns.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    group_cols = [date_col, horizon_col]
    out["rank"] = out.groupby(group_cols, dropna=False)[value_col].rank(method="average", ascending=False)
    if pct:
        out["rank_pct"] = out.groupby(group_cols, dropna=False)[value_col].rank(method="average", pct=True)
        out["target_value"] = out["rank_pct"]
        target_name = "cross_sectional_return_rank_pct"
    else:
        out["target_value"] = out["rank"]
        target_name = "cross_sectional_return_rank"
    out["target_name"] = out.get("target_name", target_name)
    out["target_name"] = target_name
    return out.sort_values([date_col, horizon_col, "rank"], ignore_index=True)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
