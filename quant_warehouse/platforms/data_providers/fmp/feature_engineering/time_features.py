from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Any, Optional

import polars as pl

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet


@dataclass(frozen=True)
class TimeFeatureConfig:
    include_day_of_week_one_hot: bool = True
    include_month_one_hot: bool = True
    prefix: str = ""


def _date_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    return pl.col(column).str.to_datetime(strict=False) if frame.schema[column] == pl.String else pl.col(column).cast(pl.Datetime, strict=False)


def _target_frame(target_index: pl.DataFrame | None, start_date: str | None, end_date: str | None) -> pl.DataFrame:
    if target_index is not None:
        if "date" not in target_index.columns:
            raise ValueError("target_index must contain a date column")
        return target_index.with_columns(_date_expr(target_index, "date").dt.truncate("1d").alias("date")).drop_nulls("date")
    if start_date is None or end_date is None:
        raise ValueError("Provide both start_date and end_date when target_index is not set.")
    dates = pl.date_range(date.fromisoformat(start_date[:10]), date.fromisoformat(end_date[:10]), interval="1d", eager=True)
    return pl.DataFrame({"date": dates})


def _section_event_frame(symbol_obj: Any, section_key: str) -> pl.DataFrame:
    _ = symbol_obj, section_key
    return pl.DataFrame({"date": pl.Series([], dtype=pl.Datetime), "symbol": pl.Series([], dtype=pl.String)})


def _payload_ipo_date(payload: Any) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    value = next((payload.get(key) for key in ("ipoDate", "ipo_date", "listingDate", "listing_date", "date") if payload.get(key)), None)
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def build_time_features(*, start_date: Optional[str] = None, end_date: Optional[str] = None, target_index: Optional[pl.DataFrame] = None, config: Optional[TimeFeatureConfig] = None) -> pl.DataFrame:
    cfg = config or TimeFeatureConfig()
    frame = _target_frame(target_index, start_date, end_date)
    prefix = str(cfg.prefix or "")
    date_col = pl.col("date")
    out = frame.with_columns(
        date_col.dt.weekday().alias(f"{prefix}day_of_week"),
        date_col.dt.day().alias(f"{prefix}day_of_month"),
        date_col.dt.ordinal_day().alias(f"{prefix}day_of_year"),
        date_col.dt.week().alias(f"{prefix}week_of_year"),
        date_col.dt.month().alias(f"{prefix}month"),
        date_col.dt.quarter().alias(f"{prefix}quarter"),
        (date_col.dt.day() == 1).alias(f"{prefix}is_month_start"),
        ((date_col + pl.duration(days=1)).dt.month() != date_col.dt.month()).alias(f"{prefix}is_month_end"),
    )
    out = out.with_columns(
        ((date_col.dt.month().is_in([1, 4, 7, 10])) & (date_col.dt.day() == 1)).alias(f"{prefix}is_quarter_start"),
        (((date_col + pl.duration(days=1)).dt.month().is_in([1, 4, 7, 10])) & ((date_col + pl.duration(days=1)).dt.day() == 1)).alias(f"{prefix}is_quarter_end"),
    )
    if cfg.include_day_of_week_one_hot:
        for day in range(1, 8):
            out = out.with_columns((pl.col(f"{prefix}day_of_week") == day).cast(pl.Int8).alias(f"{prefix}is_day_{day}"))
    if cfg.include_month_one_hot:
        for month in range(1, 13):
            out = out.with_columns((pl.col(f"{prefix}month") == month).cast(pl.Int8).alias(f"{prefix}is_month_{month}"))
    return out


def build_time_calendar_features(symbol_obj: Any, target_index: pl.DataFrame, config: Optional[TimeFeatureConfig] = None) -> BuiltFeatureSet:
    cfg = config or TimeFeatureConfig(prefix="time__")
    out = build_time_features(target_index=target_index, config=cfg)
    prefix = str(cfg.prefix or "")
    ipo_date = _payload_ipo_date(getattr(symbol_obj, "payload", None))
    if ipo_date is None:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(f"{prefix}days_after_ipo"))
    else:
        out = out.with_columns(((pl.col("date") - pl.lit(ipo_date)).dt.total_days().cast(pl.Float64)).alias(f"{prefix}days_after_ipo")).with_columns(pl.when(pl.col(f"{prefix}days_after_ipo") < 0).then(None).otherwise(pl.col(f"{prefix}days_after_ipo")).alias(f"{prefix}days_after_ipo"))
    feature_cols = [column for column in out.columns if column != "date" and column != "symbol"]
    return BuiltFeatureSet(df=out, feature_cols=feature_cols, family_name="time_calendar", endpoint_name="calendar", source_asset_class="issuer")
