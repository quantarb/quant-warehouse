from __future__ import annotations

from collections.abc import Sequence
import polars as pl


def build_cross_sectional_rank_labels(
    forward_returns: Frame,
    value_col: str = "target_value",
    date_col: str = "date",
    horizon_col: str = "horizon",
    symbol_col: str = "symbol",
    pct: bool = True,
    ) -> pl.DataFrame:
    """Rank symbols by future return using Polars window expressions."""
    if forward_returns is None or forward_returns.is_empty():
        return pl.DataFrame()
    _require_columns(forward_returns, [symbol_col, date_col, horizon_col, value_col], ctx="build_cross_sectional_rank_labels")
    out = forward_returns
    groups = [date_col, horizon_col]
    out = out.with_columns(pl.col(value_col).cast(pl.Float64, strict=False).alias(value_col)).with_columns(
        pl.col(value_col).rank(method="average", descending=True).over(groups).alias("rank")
    )
    if pct:
        out = out.with_columns((pl.col("rank") / pl.len().over(groups)).alias("rank_pct"), (pl.col("rank") / pl.len().over(groups)).alias("target_value"), pl.lit("cross_sectional_return_rank_pct").alias("target_name"))
    else:
        out = out.with_columns(pl.col("rank").alias("target_value"), pl.lit("cross_sectional_return_rank").alias("target_name"))
    out = out.sort([date_col, horizon_col, "rank"])
    return out


def _require_columns(df: pl.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
