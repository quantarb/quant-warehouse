from __future__ import annotations

from collections.abc import Sequence
import polars as pl


def build_option_return_rank_labels(
    option_returns: pl.DataFrame,
    group_cols: Sequence[str] = ("underlying_symbol", "date"),
    option_symbol_col: str = "option_symbol",
    entry_price_col: str = "entry_price",
    exit_price_col: str = "exit_price",
    pct: bool = True,
    ) -> pl.DataFrame:
    """Rank option contracts by realized return using Polars window expressions."""
    if option_returns is None or option_returns.is_empty():
        return pl.DataFrame()
    group_cols = tuple(group_cols)
    _require_columns(option_returns, [*group_cols, option_symbol_col, entry_price_col, exit_price_col], ctx="build_option_return_rank_labels")
    out = option_returns
    groups = list(group_cols)
    out = out.with_columns(
        (pl.col(exit_price_col).cast(pl.Float64, strict=False) / pl.col(entry_price_col).cast(pl.Float64, strict=False) - 1.0).alias("option_return")
    ).with_columns(
        pl.col("option_return").rank(method="average", descending=True).over(groups).alias("option_return_rank")
    )
    if pct:
        out = out.with_columns(
            (pl.col("option_return_rank") / pl.len().over(groups)).alias("option_return_percentile"),
            pl.lit("option_return_percentile").alias("target_name"),
            pl.col("option_return_rank").truediv(pl.len().over(groups)).alias("target_value"),
        )
    else:
        out = out.with_columns(pl.lit("option_return_rank").alias("target_name"), pl.col("option_return_rank").alias("target_value"))
    return out.sort([*groups, "option_return_rank"])


def _require_columns(df: pl.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    available = df.columns
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
