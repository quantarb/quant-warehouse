from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from quant_warehouse.warehouse.sections import MIN_HISTORICAL_DATE

INDEX_CANDIDATES = (
    "period_ending",
    "date",
    "as_of",
    "ex_dividend_date",
    "payment_date",
    "record_date",
    "announcement_date",
    "filing_date",
    "accepted_date",
    "report_date",
    "split_date",
    "transaction_date",
    "published_date",
    "disclosure_date",
    "ipo_date",
)
PRICE_INDEX_CANDIDATES = ("date",)
STANDARD_PRICE_COLUMNS = frozenset(
    {"open", "high", "low", "close", "volume", "adj_open", "adj_high", "adj_low", "adj_close"}
)
PRICE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open", "adj_open", "adjopen"),
    "high": ("high", "adj_high", "adjhigh"),
    "low": ("low", "adj_low", "adjlow"),
    "close": ("close", "adj_close", "adjclose", "adj_close_price"),
    "volume": ("volume", "vol"),
    "adj_open": ("adj_open", "adjopen"),
    "adj_high": ("adj_high", "adjhigh"),
    "adj_low": ("adj_low", "adjlow"),
    "adj_close": ("adj_close", "adjclose"),
}
METADATA_COLUMNS = {
    "symbol",
    "cik",
    "link",
    "final_link",
    "reported_currency",
    "currency",
    "fiscal_year",
    "calendar_year",
    "period",
}

PANEL_DIMENSION_COLUMNS = frozenset(
    {
        "business_line",
        "region",
        "cusip",
        "isin",
        "lei",
        "name",
        "title",
        "symbol",
        "fiscal_period",
    }
)


def symbol_provider_key(symbol: str, provider: str) -> str:
    return f"{symbol.strip().upper()}__{provider.strip().lower()}"


def clip_to_min_historical_date(
    frame: pd.DataFrame,
    *,
    min_date: str = MIN_HISTORICAL_DATE,
) -> pd.DataFrame:
    """Drop rows indexed before the warehouse historical floor."""
    if frame.empty:
        return frame
    floor = pd.Timestamp(min_date)
    if isinstance(frame.index, pd.DatetimeIndex):
        clipped = frame[frame.index >= floor]
        clipped.index = pd.DatetimeIndex(clipped.index)
        return clipped.sort_index()
    for column in INDEX_CANDIDATES:
        if column not in frame.columns:
            continue
        dates = pd.to_datetime(frame[column], errors="coerce")
        return frame.loc[dates >= floor].copy()
    return frame


def _pick_index_column(df: pd.DataFrame) -> str | None:
    for col in INDEX_CANDIDATES:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _to_snake(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return name.lower().strip()


def _reset_temporal_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if any(col in out.columns for col in INDEX_CANDIDATES):
        return out
    if isinstance(out.index, pd.DatetimeIndex) or out.index.name in INDEX_CANDIDATES:
        out = out.reset_index()
        index_name = out.columns[0]
        if index_name not in INDEX_CANDIDATES:
            out = out.rename(columns={index_name: "period_ending"})
    return out


def normalize_vendor_frame(
    df: pd.DataFrame,
    *,
    provider: str,
    vendor_only_prefix: str | None = None,
    min_date: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = _reset_temporal_index(df)
    index_col = _pick_index_column(out)
    if index_col is None:
        return pd.DataFrame()

    out[index_col] = pd.to_datetime(out[index_col], errors="coerce")
    out = out.dropna(subset=[index_col])
    out = out.sort_values(index_col)

    rename: dict[str, str] = {}
    for col in out.columns:
        if col == index_col:
            continue
        base = _to_snake(col)
        if base in METADATA_COLUMNS:
            continue
        if vendor_only_prefix:
            rename[col] = f"{vendor_only_prefix}__{base}"
        else:
            rename[col] = base

    out = out.rename(columns=rename)
    keep = [index_col] + [rename[c] for c in rename]
    out = out[keep]

    for col in out.columns:
        if col == index_col:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.drop_duplicates(subset=[index_col], keep="last")
    out = out.set_index(index_col)
    out.index.name = index_col
    return clip_to_min_historical_date(out, min_date=min_date or MIN_HISTORICAL_DATE)


def normalize_panel_frame(
    df: pd.DataFrame,
    *,
    provider: str,
    vendor_only_prefix: str | None = None,
    min_date: str | None = None,
) -> pd.DataFrame:
    """Normalize repeated cross-sections keyed by a filing/as-of date."""
    if df.empty:
        return df

    out = _reset_temporal_index(df)
    index_col = _pick_index_column(out)
    if index_col is None:
        return pd.DataFrame()

    out[index_col] = pd.to_datetime(out[index_col], errors="coerce")
    out = out.dropna(subset=[index_col]).sort_values(index_col)

    rename: dict[str, str] = {}
    for col in out.columns:
        if col == index_col:
            continue
        base = _to_snake(col)
        if base in METADATA_COLUMNS:
            continue
        if vendor_only_prefix:
            rename[col] = f"{vendor_only_prefix}__{base}"
        else:
            rename[col] = base

    out = out.rename(columns=rename)
    keep = [index_col] + [rename[c] for c in rename]
    out = out[keep]

    for col in out.columns:
        if col == index_col or col in PANEL_DIMENSION_COLUMNS:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    dedupe_cols = [index_col, *(
        key for key in (
            "business_line",
            "region",
            "cusip",
            "isin",
            "lei",
            "name",
            "title",
            "symbol",
        )
        if key in out.columns
    )]
    out = out.drop_duplicates(subset=dedupe_cols, keep="last")
    out = out.set_index(index_col)
    out.index.name = index_col
    out.index = pd.DatetimeIndex(out.index)
    return clip_to_min_historical_date(out, min_date=min_date or MIN_HISTORICAL_DATE)


def coerce_object_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime.date values in object columns to timestamps for Arctic writes."""
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype != object:
            continue
        sample = out[column].dropna()
        if sample.empty:
            continue
        value = sample.iloc[0]
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def normalize_dated_snapshot_frame(df: pd.DataFrame, *, section: str) -> pd.DataFrame:
    """Normalize cross-sectional snapshots into a dated panel for Arctic storage."""
    out = normalize_snapshot_frame(df)
    if out.empty:
        return out

    as_of_name = "as_of"
    if section == "etf_holdings" and "updated" in out.columns:
        out["updated"] = pd.to_datetime(out["updated"], errors="coerce")
        out = out.dropna(subset=["updated"])
        out = out.set_index("updated")
        out.index.name = as_of_name
    else:
        as_of = pd.Timestamp.utcnow().normalize()
        out[as_of_name] = as_of
        out = out.set_index(as_of_name)

    out.index = pd.DatetimeIndex(out.index)
    dedupe_cols = [as_of_name, *(
        key
        for key in (
            "cusip",
            "isin",
            "name",
            "sector",
            "country",
            "symbol",
            "title",
            "cik",
        )
        if key in out.reset_index().columns
    )]
    reset = out.reset_index()
    reset = reset.drop_duplicates(subset=dedupe_cols, keep="last")
    out = reset.set_index(as_of_name)
    out.index = pd.DatetimeIndex(out.index)
    return out.sort_index()


def normalize_etf_composition_frame(df: pd.DataFrame, *, section: str) -> pd.DataFrame:
    return normalize_dated_snapshot_frame(df, section=section)


def normalize_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cross-sectional frames (holdings, sectors) without a time index."""
    if df.empty:
        return df

    out = df.copy()
    rename: dict[str, str] = {}
    for col in out.columns:
        if col in METADATA_COLUMNS:
            continue
        rename[col] = _to_snake(col)

    out = out.rename(columns=rename)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="ignore")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)


def _resolve_price_column(name: str) -> str | None:
    base = _to_snake(name)
    for canonical, aliases in PRICE_COLUMN_ALIASES.items():
        if base in aliases:
            return canonical
    return None


def normalize_prices(df: pd.DataFrame, *, provider: str, min_date: str | None = None) -> pd.DataFrame:
    """Normalize vendor OHLCV frames to a shared daily schema indexed by date."""
    if df.empty:
        return df

    out = df.copy()
    if not any(col in out.columns for col in PRICE_INDEX_CANDIDATES):
        if isinstance(out.index, pd.DatetimeIndex) or out.index.name in PRICE_INDEX_CANDIDATES:
            out = out.reset_index()
            index_name = out.columns[0]
            if index_name != "date":
                out = out.rename(columns={index_name: "date"})

    index_col = _pick_index_column(out)
    if index_col is None:
        return pd.DataFrame()

    out[index_col] = pd.to_datetime(out[index_col], errors="coerce")
    out = out.dropna(subset=[index_col]).sort_values(index_col)

    rename: dict[str, str] = {}
    for col in out.columns:
        if col == index_col or col in METADATA_COLUMNS:
            continue
        canonical = _resolve_price_column(col)
        if canonical is None:
            rename[col] = f"{provider}__{_to_snake(col)}"
        else:
            rename[col] = canonical

    out = out.rename(columns=rename)
    seen: set[str] = set()
    ordered_keep: list[str] = []
    for col in [index_col] + [rename[c] for c in rename]:
        if col not in seen:
            ordered_keep.append(col)
            seen.add(col)
    out = out[ordered_keep]

    for col in out.columns:
        if col == index_col:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.drop_duplicates(subset=[index_col], keep="last")
    out = out.set_index(index_col)
    out.index.name = "date"
    return clip_to_min_historical_date(out, min_date=min_date or MIN_HISTORICAL_DATE)