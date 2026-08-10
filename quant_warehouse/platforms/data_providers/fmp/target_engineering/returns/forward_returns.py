from __future__ import annotations

from collections.abc import Sequence
import polars as pl


def build_forward_return_labels(
    prices: Frame,
    horizons: Sequence[int],
    price_col: str = "close",
    symbol_col: str = "symbol",
    date_col: str = "date",
    log_return: bool = False,
    ) -> pl.DataFrame:
    """Build forward-return labels with Polars window shifts."""
    if prices is None or prices.is_empty():
        return pl.DataFrame()
    _require_columns(prices, [symbol_col, date_col, price_col], ctx="build_forward_return_labels")
    values = _normalize_horizons(horizons)
    df = prices
    date_expr = pl.col(date_col).str.to_datetime(strict=False) if df.schema[date_col] == pl.String else pl.col(date_col).cast(pl.Datetime, strict=False)
    df = df.select([symbol_col, date_col, price_col]).with_columns([
        pl.col(symbol_col).cast(pl.String), date_expr.alias(date_col), pl.col(price_col).cast(pl.Float64, strict=False).alias(price_col),
    ]).drop_nulls([symbol_col, date_col]).sort([symbol_col, date_col])
    frames: list[pl.DataFrame] = []
    for horizon in values:
        future = pl.col(price_col).shift(-horizon).over(symbol_col)
        target = (future / pl.col(price_col)).log() if log_return else (future / pl.col(price_col) - 1.0)
        name = f"forward_log_return_{horizon}d" if log_return else f"forward_return_{horizon}d"
        frames.append(df.with_columns(target.alias("target_value"), pl.lit(horizon).alias("horizon"), pl.lit(name).alias("target_name")).select([symbol_col, date_col, "horizon", "target_name", "target_value"]))
    out = pl.concat(frames, how="vertical_relaxed").sort([symbol_col, date_col, "horizon"])
    return out


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


def _require_columns(df: pl.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
