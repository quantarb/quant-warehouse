"""Raw preferred-security features for an issuer-level feature family.

Preferred securities are kept separate from the issuer's common stock and
from other security classes.  The input may contain several preferred issues
for one issuer.  We aggregate only the raw daily OHLCV fields by date so the
result can be joined to a normal ``(date, issuer_symbol)`` feature panel.

This module deliberately does not create returns, ranks, or prediction
targets.  Those belong in target engineering and can be added later without
changing the raw feature family.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet


PREFERRED_RAW_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def build_security_class_features(
    issuer_symbol: str,
    security_class: str,
    prices: pl.DataFrame,
    *,
    raw_columns: Sequence[str] = PREFERRED_RAW_COLUMNS,
) -> BuiltFeatureSet:
    """Build one isolated raw feature family for an issuer/security class."""
    built = build_preferred_stock_features(issuer_symbol, prices, raw_columns=raw_columns)
    if built.df.is_empty():
        return built
    prefix = str(security_class).strip().lower()
    renamed = {column: column.replace("preferred__", f"{prefix}__", 1) for column in built.feature_cols}
    frame = built.df.rename(renamed)
    return BuiltFeatureSet(
        df=frame,
        feature_cols=[renamed[column] for column in built.feature_cols],
        family_name=f"{prefix}-historical-price-eod",
        endpoint_name="prices",
        source_asset_class=prefix,
    )


def build_preferred_stock_features(
    issuer_symbol: str,
    preferred_prices: pl.DataFrame,
    *,
    raw_columns: Sequence[str] = PREFERRED_RAW_COLUMNS,
) -> BuiltFeatureSet:
    """Build a separate raw preferred-stock feature family for one issuer.

    ``preferred_prices`` must contain ``date`` (as a column or DatetimeIndex)
    and ``symbol``.  Each row is one preferred issue/day.  When multiple
    preferred issues exist on a date, numeric raw fields are averaged and the
    issue count is retained so the aggregation is explicit.
    """

    if preferred_prices is None or preferred_prices.is_empty():
        return BuiltFeatureSet(df=pl.DataFrame(), feature_cols=[])
    frame = preferred_prices
    if "date" not in frame.columns:
        raise ValueError("preferred_prices must contain a date column")
    if "symbol" not in frame.columns:
        raise ValueError("preferred_prices must contain a symbol column")

    columns = [column for column in raw_columns if column in frame.columns]
    if not columns:
        raise ValueError(f"preferred_prices has none of the requested raw columns: {list(raw_columns)}")
    frame = frame.with_columns(
        (pl.col("date").str.to_datetime(strict=False) if frame.schema["date"] == pl.String else pl.col("date").cast(pl.Datetime, strict=False)).dt.truncate("1d").alias("date"),
        pl.col("symbol").cast(pl.String).str.strip_chars().str.to_uppercase().alias("symbol"),
        *[pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in columns],
    ).drop_nulls("date")
    out = frame.group_by("date", maintain_order=True).agg(
        *[pl.col(column).mean().alias(f"preferred__{column}_mean") for column in columns],
        pl.col("symbol").n_unique().cast(pl.Float64).alias("preferred__issue_count"),
    ).with_columns(
        pl.lit(1.0).alias("preferred__has_data"),
        pl.lit(str(issuer_symbol).strip().upper()).alias("symbol"),
    ).sort("date")
    feature_cols = [column for column in out.columns if column.startswith("preferred__")]
    return BuiltFeatureSet(df=out, feature_cols=feature_cols, family_name="preferred-historical-price-eod", endpoint_name="prices", source_asset_class="preferred")
