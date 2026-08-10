from __future__ import annotations

from collections.abc import Sequence
import polars as pl

METADATA_COLS = ("option_type", "strike", "expiration", "dte", "moneyness")

def build_option_best_return_labels(
    option_returns: pl.DataFrame,
    group_cols: Sequence[str] = ("underlying_symbol", "date"),
    option_symbol_col: str = "option_symbol",
    entry_price_col: str = "entry_price",
    exit_price_col: str = "exit_price",
    ) -> pl.DataFrame:
    """Select the best realized-return option contract per group."""
    if option_returns is None or option_returns.is_empty():
        return pl.DataFrame()
    groups = list(group_cols)
    _require_columns(option_returns, [*groups, option_symbol_col, entry_price_col, exit_price_col], ctx="build_option_best_return_labels")
    out = option_returns
    out = out.with_columns((pl.col(exit_price_col).cast(pl.Float64, strict=False) / pl.col(entry_price_col).cast(pl.Float64, strict=False) - 1.0).alias("option_return"))
    out = out.sort("option_return", descending=True).group_by(groups, maintain_order=True).first()
    out = out.with_columns([
        pl.col(option_symbol_col).alias("best_option_symbol"),
        pl.col("option_return").alias("best_option_return"),
        pl.lit("option_best_return").alias("target_name"),
        pl.col("option_return").alias("target_value"),
    ])
    keep = [*groups, "target_name", "target_value", "best_option_symbol", "best_option_return"]
    keep.extend([column for column in METADATA_COLS if column in out.columns])
    out = out.select(keep)
    return out


def _require_columns(df: pl.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
