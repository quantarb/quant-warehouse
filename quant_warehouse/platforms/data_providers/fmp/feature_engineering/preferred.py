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

import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet


PREFERRED_RAW_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def build_security_class_features(
    issuer_symbol: str,
    security_class: str,
    prices: pd.DataFrame,
    *,
    raw_columns: Sequence[str] = PREFERRED_RAW_COLUMNS,
) -> BuiltFeatureSet:
    """Build one isolated raw feature family for an issuer/security class."""
    built = build_preferred_stock_features(issuer_symbol, prices, raw_columns=raw_columns)
    if built.df.empty:
        return built
    prefix = str(security_class).strip().lower()
    renamed = {column: column.replace("preferred__", f"{prefix}__", 1) for column in built.feature_cols}
    frame = built.df.rename(columns=renamed)
    return BuiltFeatureSet(
        df=frame,
        feature_cols=[renamed[column] for column in built.feature_cols],
        family_name=f"{prefix}-historical-price-eod",
        endpoint_name="prices",
        source_asset_class=prefix,
    )


def build_preferred_stock_features(
    issuer_symbol: str,
    preferred_prices: pd.DataFrame,
    *,
    raw_columns: Sequence[str] = PREFERRED_RAW_COLUMNS,
) -> BuiltFeatureSet:
    """Build a separate raw preferred-stock feature family for one issuer.

    ``preferred_prices`` must contain ``date`` (as a column or DatetimeIndex)
    and ``symbol``.  Each row is one preferred issue/day.  When multiple
    preferred issues exist on a date, numeric raw fields are averaged and the
    issue count is retained so the aggregation is explicit.
    """

    if preferred_prices is None or preferred_prices.empty:
        return BuiltFeatureSet(df=pd.DataFrame(), feature_cols=[])

    frame = preferred_prices.copy()
    if "date" not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        else:
            raise ValueError("preferred_prices must contain a date column or DatetimeIndex")
    if "symbol" not in frame.columns:
        raise ValueError("preferred_prices must contain a symbol column")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()

    columns = [column for column in raw_columns if column in frame.columns]
    if not columns:
        raise ValueError(f"preferred_prices has none of the requested raw columns: {list(raw_columns)}")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    grouped = frame.groupby("date", sort=True)
    out = grouped[columns].mean().rename(columns={column: f"preferred__{column}_mean" for column in columns})
    out["preferred__issue_count"] = grouped["symbol"].nunique().astype(float)
    out["preferred__has_data"] = 1.0
    out["symbol"] = str(issuer_symbol).strip().upper()
    out = out.reset_index().set_index(["date", "symbol"]).sort_index()
    feature_cols = [column for column in out.columns if column.startswith("preferred__")]
    return BuiltFeatureSet(
        df=out[feature_cols],
        feature_cols=feature_cols,
        family_name="preferred-historical-price-eod",
        endpoint_name="prices",
        source_asset_class="preferred",
    )
