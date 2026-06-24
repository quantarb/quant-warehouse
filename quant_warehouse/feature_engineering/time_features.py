from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TimeFeatureConfig:
    include_day_of_week_one_hot: bool = True
    include_month_one_hot: bool = True
    prefix: str = ""


@dataclass(frozen=True)
class BuiltFeatureSet:
    df: pd.DataFrame
    feature_cols: list[str]


def _extract_dates(target_index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(target_index, pd.MultiIndex):
        names = target_index.names or []
        if "date" not in names:
            raise ValueError("target_index MultiIndex must contain a 'date' level.")
        dts = pd.to_datetime(target_index.get_level_values("date"), errors="coerce")
        return pd.DatetimeIndex(dts)

    if isinstance(target_index, pd.DatetimeIndex):
        return pd.DatetimeIndex(pd.to_datetime(target_index, errors="coerce"))

    dts = pd.to_datetime(target_index, errors="coerce")
    return pd.DatetimeIndex(dts)


def _extract_symbols(target_index: pd.Index) -> pd.Index | None:
    if not isinstance(target_index, pd.MultiIndex):
        return None
    names = target_index.names or []
    if "symbol" not in names:
        return None
    return pd.Index(target_index.get_level_values("symbol").astype(str).str.upper())


def _event_distances_for_symbol_dates(
    target_dates: pd.DatetimeIndex,
    target_symbols: pd.Index | None,
    events: pd.DataFrame | pd.Series | pd.DatetimeIndex | Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    before = np.full(len(target_dates), np.nan, dtype=float)
    after = np.full(len(target_dates), np.nan, dtype=float)
    if isinstance(events, pd.DataFrame):
        if events.empty:
            return before, after
        event_frame = events.copy()
        if "date" not in event_frame.columns or "symbol" not in event_frame.columns:
            return before, after
    else:
        event_dates = pd.to_datetime(pd.Series(list(events)), errors="coerce").dropna()
        if event_dates.empty:
            return before, after
        event_frame = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(event_dates).normalize(),
                "symbol": target_symbols[0] if target_symbols is not None and len(target_symbols) else "",
            }
        )

    if target_symbols is None:
        return before, after

    target_ns = target_dates.values.astype("datetime64[ns]")
    symbols = pd.Index(target_symbols.astype(str).str.upper())
    for symbol, event_group in event_frame.groupby("symbol", sort=False):
        mask = symbols == str(symbol).upper()
        if not mask.any():
            continue
        event_dates = pd.DatetimeIndex(event_group["date"]).sort_values().unique().values.astype("datetime64[ns]")
        if len(event_dates) == 0:
            continue

        symbol_dates = target_ns[mask]
        next_pos = np.searchsorted(event_dates, symbol_dates, side="left")
        valid_next = next_pos < len(event_dates)
        target_locs = np.flatnonzero(mask)
        if valid_next.any():
            next_dates = event_dates[next_pos[valid_next]]
            before[target_locs[valid_next]] = (
                next_dates - symbol_dates[valid_next]
            ).astype("timedelta64[D]").astype(float)

        prev_pos = np.searchsorted(event_dates, symbol_dates, side="right") - 1
        valid_prev = prev_pos >= 0
        if valid_prev.any():
            prev_dates = event_dates[prev_pos[valid_prev]]
            after[target_locs[valid_prev]] = (
                symbol_dates[valid_prev] - prev_dates
            ).astype("timedelta64[D]").astype(float)
    return before, after


def _section_event_frame(symbol_obj, section_key: str, target_index: pd.MultiIndex) -> pd.DataFrame:
    _ = symbol_obj, section_key, target_index
    return pd.DataFrame(columns=["date", "symbol"])


def _payload_ipo_date(payload: Any) -> pd.Timestamp | None:
    value = None
    if isinstance(payload, dict):
        value = (
            payload.get("ipoDate")
            or payload.get("ipo_date")
            or payload.get("listingDate")
            or payload.get("listing_date")
            or payload.get("date")
        )
    return pd.Timestamp(value).normalize() if value is not None else None


def _add_event_distance_columns(
    out: pd.DataFrame,
    *,
    dti: pd.DatetimeIndex,
    symbols: pd.Index | None,
    event_frame: pd.DataFrame,
    event_name: str,
    prefix: str,
) -> list[str]:
    until, since = _event_distances_for_symbol_dates(dti, symbols, event_frame)
    since_col = f"{prefix}days_since_{event_name}"
    until_col = f"{prefix}days_until_{event_name}"
    out[since_col] = since
    out[until_col] = until
    return [since_col, until_col]


def build_time_features(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_index: Optional[pd.Index] = None,
    config: Optional[TimeFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Build numeric calendar features from daily dates.

    If target_index is provided, output index matches target_index
    (supports DatetimeIndex or MultiIndex with a 'date' level).
    Otherwise, output is a daily DatetimeIndex from start_date..end_date.
    """
    cfg = config or TimeFeatureConfig()

    if target_index is None:
        if start_date is None or end_date is None:
            raise ValueError("Provide both start_date and end_date when target_index is not set.")
        out_index = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D")
        dti = pd.DatetimeIndex(out_index)
    else:
        out_index = target_index
        dti = _extract_dates(target_index)

    if dti.isna().any():
        raise ValueError("Date index contains invalid/NaT values; cannot build time features.")

    prefix = str(cfg.prefix or "")
    out = pd.DataFrame(index=out_index)
    out[f"{prefix}day_of_week"] = np.asarray(dti.dayofweek, dtype=np.int8)
    out[f"{prefix}day_of_month"] = np.asarray(dti.day, dtype=np.int8)
    out[f"{prefix}day_of_year"] = np.asarray(dti.dayofyear, dtype=np.int16)
    out[f"{prefix}week_of_year"] = np.asarray(dti.isocalendar().week, dtype=np.int16)
    out[f"{prefix}month"] = np.asarray(dti.month, dtype=np.int8)
    out[f"{prefix}quarter"] = np.asarray(dti.quarter, dtype=np.int8)
    out[f"{prefix}is_month_start"] = np.asarray(dti.is_month_start, dtype=np.int8)
    out[f"{prefix}is_month_end"] = np.asarray(dti.is_month_end, dtype=np.int8)
    out[f"{prefix}is_quarter_start"] = np.asarray(dti.is_quarter_start, dtype=np.int8)
    out[f"{prefix}is_quarter_end"] = np.asarray(dti.is_quarter_end, dtype=np.int8)

    if cfg.include_day_of_week_one_hot:
        names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        for i, name in enumerate(names):
            out[f"{prefix}is_{name}"] = (out[f"{prefix}day_of_week"] == i).astype(np.int8)

    if cfg.include_month_one_hot:
        for m in range(1, 13):
            out[f"{prefix}is_month_{m}"] = (out[f"{prefix}month"] == m).astype(np.int8)

    return out


def build_time_calendar_features(symbol_obj, target_index: pd.MultiIndex, config: Optional[TimeFeatureConfig] = None) -> BuiltFeatureSet:
    cfg = config or TimeFeatureConfig(prefix="time__")
    out = build_time_features(target_index=target_index, config=cfg)
    dti = _extract_dates(target_index)
    symbols = _extract_symbols(target_index)
    prefix = str(cfg.prefix or "")
    feature_cols = list(out.columns)

    for section_key, event_name in (
        ("earnings", "earnings"),
        ("dividends", "dividend"),
        ("splits", "stock_split"),
    ):
        event_frame = _section_event_frame(symbol_obj, section_key, target_index)
        added = _add_event_distance_columns(
            out,
            dti=dti,
            symbols=symbols,
            event_frame=event_frame,
            event_name=event_name,
            prefix=prefix,
        )
        feature_cols.extend(added)

    ipo_date = _payload_ipo_date(getattr(symbol_obj, "payload", None))
    ipo_col = f"{prefix}days_after_ipo"
    if ipo_date is None:
        out[ipo_col] = np.nan
    else:
        out[ipo_col] = (dti - ipo_date).days.astype(float)
        out.loc[out[ipo_col] < 0.0, ipo_col] = np.nan
    feature_cols.append(ipo_col)

    out = out.replace([np.inf, -np.inf], np.nan)
    feature_cols = [
        col for col in dict.fromkeys(feature_cols)
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col])
    ]
    return BuiltFeatureSet(df=out, feature_cols=feature_cols)
