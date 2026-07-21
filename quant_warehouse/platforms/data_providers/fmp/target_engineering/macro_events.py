"""Macro event normalization and company-response target engineering.

Macro events are global observations.  They are stored once per release and
joined to company prices only when constructing the company-specific response
target; callers should not broadcast the raw event row into the source table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


MACRO_RESPONSE_CLASSES = (
    "strong_negative",
    "negative",
    "neutral",
    "positive",
    "strong_positive",
)


@dataclass(frozen=True)
class MacroEventSpec:
    """Configuration for company-specific macro response labels."""

    horizons: tuple[int, ...] = (1, 5, 20)
    minimum_cross_section: int = 5
    require_actual: bool = True


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or "unknown_event"


def normalize_macro_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize FMP economic-calendar rows into one row per release."""
    if events is None or events.empty:
        return pd.DataFrame(
            columns=[
                "macro_event_id", "date", "country", "currency", "event_type",
                "impact", "previous", "estimate", "actual", "unit", "surprise",
                "surprise_pct",
            ]
        )
    frame = events.copy()
    if "date" not in frame.columns or "event" not in frame.columns:
        raise ValueError("macro events require date and event columns")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=["date"]).copy()
    for column in ("previous", "estimate", "actual", "change", "changePercentage"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("country", "currency", "impact", "unit"):
        if column not in frame.columns:
            frame[column] = ""
    frame["country"] = frame["country"].fillna("").astype(str).str.upper()
    frame["currency"] = frame["currency"].fillna("").astype(str).str.upper()
    frame["event_type"] = frame["event"].map(_slug)
    frame["impact"] = frame["impact"].fillna("").astype(str).str.lower()
    frame["unit"] = frame["unit"].fillna("").astype(str)
    actual = pd.to_numeric(frame.get("actual", pd.Series(np.nan, index=frame.index)), errors="coerce")
    estimate = pd.to_numeric(frame.get("estimate", pd.Series(np.nan, index=frame.index)), errors="coerce")
    frame["surprise"] = actual - estimate
    estimate = estimate.abs()
    frame["surprise_pct"] = frame["surprise"] / estimate.replace(0, np.nan)
    identity = ["date", "country", "currency", "event_type"]
    frame["macro_event_id"] = frame[identity].astype(str).agg("|".join, axis=1)
    columns = [
        "macro_event_id", "date", "country", "currency", "event", "event_type",
        "impact", "previous", "estimate", "actual", "unit", "surprise", "surprise_pct",
    ]
    return frame[[column for column in columns if column in frame.columns]].sort_values("date").reset_index(drop=True)


def _macro_target_name(row: pd.Series) -> str:
    """Return a stable binary target name for one calendar release.

    US interest-rate decisions are directionalized from actual minus previous
    so rate cuts, hikes, and holds remain independent event labels. Other
    releases retain a country/event-type target, allowing new FMP event names
    without expanding a restrictive taxonomy.
    """
    country = str(row.get("country", "")).lower()
    event_type = str(row.get("event_type", "unknown_event"))
    is_rate_decision = country == "us" and (
        "interest_rate_decision" in event_type
        or "federal_funds_rate" in event_type
        or "fed_funds_rate" in event_type
    )
    if is_rate_decision:
        actual = pd.to_numeric(pd.Series([row.get("actual")]), errors="coerce").iloc[0]
        previous = pd.to_numeric(pd.Series([row.get("previous")]), errors="coerce").iloc[0]
        if pd.notna(actual) and pd.notna(previous):
            if actual < previous:
                return "fed_rate_cut"
            if actual > previous:
                return "fed_rate_hike"
            return "fed_rate_hold"
        return "fed_rate_decision"
    return f"macro_{country or 'global'}_{event_type}"


def build_macro_event_targets(events: pd.DataFrame) -> pd.DataFrame:
    """Create one dynamic binary event target row per macro release.

    The returned ``target_name`` values are suitable for shared independent
    event heads, e.g. ``fed_rate_cut``, ``fed_rate_hike``,
    ``macro_us_cpi_yoy``, or ``macro_eu_gdp_growth_qoq``.
    """
    macro = normalize_macro_events(events)
    if macro.empty:
        macro["target_name"] = pd.Series(dtype="string")
        macro["target_column"] = pd.Series(dtype="string")
        return macro
    macro = macro.copy()
    macro["target_name"] = macro.apply(_macro_target_name, axis=1)
    macro["target_column"] = "is_" + macro["target_name"]
    return macro


def build_macro_event_label_panel(
    tokens: pd.DataFrame,
    events: pd.DataFrame,
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Add dynamic date-level binary macro labels to token rows.

    Macro releases are global, so a release label is broadcast to every
    symbol token on that release date. The event response labels remain a
    separate company-specific target produced by ``build_macro_response_labels``.
    """
    if date_column not in tokens.columns:
        raise ValueError(f"tokens must contain {date_column!r}")
    out = tokens.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce").dt.normalize()
    macro = build_macro_event_targets(events)
    if macro.empty:
        return out
    dates = macro[["date", "target_column"]].drop_duplicates().assign(value=1.0)
    wide = dates.pivot_table(index="date", columns="target_column", values="value", fill_value=0.0)
    wide.index.name = date_column
    wide = wide.reset_index()
    return out.merge(wide, on=date_column, how="left").fillna({column: 0.0 for column in wide.columns if column != date_column})


def build_macro_response_labels(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    spec: MacroEventSpec | None = None,
    *,
    symbol_column: str = "symbol",
    date_column: str = "date",
    price_column: str = "close",
) -> pd.DataFrame:
    """Build shared five-class company responses for each macro release.

    Response classes are cross-sectional quintile classes for each release and
    horizon.  The class semantics are shared across every macro event type;
    event type and surprise remain inputs to the model rather than separate
    prediction heads.
    """
    spec = spec or MacroEventSpec()
    required = {symbol_column, date_column, price_column}
    if not required.issubset(prices.columns):
        raise ValueError(f"prices must contain {sorted(required)}")
    macro = normalize_macro_events(events)
    if macro.empty:
        return pd.DataFrame()
    if spec.require_actual:
        macro = macro.loc[macro.actual.notna()].copy()
    if macro.empty:
        return pd.DataFrame()
    panel = prices[[symbol_column, date_column, price_column]].copy()
    panel[date_column] = pd.to_datetime(panel[date_column], errors="coerce").dt.normalize()
    panel[symbol_column] = panel[symbol_column].astype(str).str.upper()
    panel[price_column] = pd.to_numeric(panel[price_column], errors="coerce")
    panel = panel.dropna().sort_values([symbol_column, date_column])
    outputs: list[pd.DataFrame] = []
    for horizon in spec.horizons:
        future = panel[[symbol_column, date_column, price_column]].copy()
        future["event_date"] = future[date_column]
        future["future_date"] = future.groupby(symbol_column)[date_column].shift(-int(horizon))
        future["future_price"] = future.groupby(symbol_column)[price_column].shift(-int(horizon))
        future = future.drop(columns=[date_column, price_column])
        current = panel.rename(columns={date_column: "event_date", price_column: "event_price"})
        joined = current.merge(future, on=[symbol_column, "event_date"], how="left")
        joined["forward_return"] = joined["future_price"] / joined["event_price"] - 1.0
        joined = joined.merge(macro, left_on="event_date", right_on="date", how="inner")
        joined = joined.dropna(subset=["forward_return"])
        if joined.empty:
            continue
        grouped = joined.groupby("macro_event_id")["forward_return"]
        counts = grouped.transform("count")
        ranks = grouped.rank(method="first", pct=True)
        joined["response_class"] = np.select(
            [ranks <= 0.2, ranks <= 0.4, ranks <= 0.6, ranks <= 0.8],
            [0, 1, 2, 3],
            default=4,
        ).astype("int8")
        joined.loc[counts < spec.minimum_cross_section, "response_class"] = 2
        joined["horizon"] = int(horizon)
        outputs.append(joined)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True).sort_values(["event_date", "macro_event_id", symbol_column, "horizon"]).reset_index(drop=True)
