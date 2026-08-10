from __future__ import annotations

from datetime import timedelta
from typing import Sequence

import polars as pl


def _datetime_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    return ((pl.col(column).str.to_datetime(strict=False, time_zone="UTC") if frame.schema[column] == pl.String
             else pl.col(column).cast(pl.Datetime, strict=False)).dt.replace_time_zone(None))


def asof_join_pit(*, left: pl.DataFrame, right: pl.DataFrame, on: str = "date",
                  by: Sequence[str] | None = ("symbol",), direction: str = "backward",
                  tolerance: timedelta | None = None, allow_exact_matches: bool = True) -> pl.DataFrame:
    """Point-in-time safe Polars as-of join preserving left row cardinality."""
    if left is None or right is None:
        raise ValueError("left and right must be non-null Polars dataframes")
    if left.is_empty() or right.is_empty():
        return left.clone()
    if on not in left.columns or on not in right.columns:
        raise ValueError(f"Both dataframes must include '{on}' column.")
    by_cols = [column for column in (by or ()) if column]
    for column in by_cols:
        if column not in left.columns or column not in right.columns:
            raise ValueError(f"Grouping column '{column}' must exist in both dataframes.")
    left_pl = left.with_columns(_datetime_expr(left, on).alias(on))
    right_pl = right.with_columns(_datetime_expr(right, on).alias(on))
    for column in by_cols:
        left_pl = left_pl.with_columns(pl.col(column).cast(pl.String, strict=False).alias(column))
        right_pl = right_pl.with_columns(pl.col(column).cast(pl.String, strict=False).alias(column))
    kwargs: dict[str, object] = {"on": on, "strategy": direction, "allow_exact_matches": allow_exact_matches}
    if by_cols:
        kwargs["by"] = by_cols
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    return left_pl.sort([*by_cols, on]).join_asof(right_pl.sort([*by_cols, on]), **kwargs)


def broadcast_asof_to_target_index(*, sparse_df: pl.DataFrame, target_index: pl.DataFrame,
                                    on: str = "date", by: Sequence[str] | None = ("symbol",)) -> pl.DataFrame:
    """Broadcast sparse rows onto an explicit Polars target panel."""
    if sparse_df is None or sparse_df.is_empty():
        return target_index.head(0)
    if on not in sparse_df.columns or on not in target_index.columns:
        raise ValueError(f"Both sparse_df and target_index must include '{on}'.")
    by_cols = [column for column in (by or ()) if column]
    for column in by_cols:
        if column not in sparse_df.columns or column not in target_index.columns:
            raise ValueError(f"Grouping column '{column}' must exist in both frames.")
    return asof_join_pit(left=target_index, right=sparse_df, on=on, by=by_cols or None, direction="backward")
