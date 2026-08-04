"""Build fund and institutional purchase target-family events."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

FUND_ACTIVITY_TARGET_FAMILY = "fund_activity"
FUND_ACTIVITY_TARGET_FAMILIES = (
    "fund_activity.etf_buy",
    "fund_activity.mutual_fund_buy",
    "fund_activity.institutional_buy",
    "fund_activity.add",
    "fund_activity.reduce",
    "fund_activity.exit",
)
HOLDER_ACTIVITY_TARGET_FAMILIES = (
    "holder_activity.buy",
    "holder_activity.add",
    "holder_activity.reduce",
    "holder_activity.exit",
)


def _date_column(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_datetime(frame[name], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index.normalize(), index=frame.index)
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def _numeric(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _event_rows(
    frame: pd.DataFrame,
    *,
    family: str,
    symbol_column: str = "symbol",
    text_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if frame.empty or symbol_column not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["symbol"] = out[symbol_column].astype(str).str.strip().str.upper()
    out["event_date"] = _date_column(out, ("date", "as_of", "updated", "disclosure_date"))
    out = out.loc[out["symbol"].ne("") & out["event_date"].notna()].copy()
    if out.empty:
        return pd.DataFrame()
    out["date"] = out["event_date"]
    out["target_family"] = family
    out["signal_value"] = _numeric(out, ("signal_value", "value", "weight", "shares"))
    keep = ["symbol", "date", "event_date", "target_family", "signal_value"]
    keep.extend(column for column in text_columns if column in out.columns)
    return out[keep]


def build_institutional_activity_events(summary: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate institutional activity events from FMP summaries."""
    if summary.empty:
        return pd.DataFrame()
    summary = summary.rename(
        columns={
            "newPositions": "new_positions",
            "increasedPositions": "increased_positions",
            "reducedPositions": "reduced_positions",
            "closedPositions": "closed_positions",
            "investorsHolding": "investors_holding",
            "numberOf13fShares": "number_of_13f_shares",
            "totalInvested": "total_invested",
        }
    )
    outputs: list[pd.DataFrame] = []
    for family, columns in (
        ("fund_activity.institutional_buy", ("new_positions", "increased_positions")),
        ("fund_activity.add", ("new_positions", "increased_positions")),
        ("fund_activity.reduce", ("reduced_positions",)),
        ("fund_activity.exit", ("closed_positions",)),
    ):
        values = sum((_numeric(summary, (column,)) for column in columns), pd.Series(0.0, index=summary.index))
        source = summary.copy()
        source["signal_value"] = (values > 0).astype("float32")
        source = source.loc[source["signal_value"].gt(0)]
        rows = _event_rows(source, family=family, text_columns=("cik",))
        if not rows.empty:
            outputs.append(rows)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def build_fund_holding_activity_events(
    holdings: pd.DataFrame,
    *,
    fund_type: str = "etf",
    fund_column: str = "fund_symbol",
    security_column: str = "symbol",
) -> pd.DataFrame:
    """Infer fund add/reduce/exit events from dated holding snapshots."""
    required = {fund_column, security_column}
    if holdings.empty or not required.issubset(holdings.columns):
        return pd.DataFrame()
    work = holdings.copy()
    work["fund_id"] = work[fund_column].astype(str).str.strip().str.upper()
    work["symbol"] = work[security_column].astype(str).str.strip().str.upper()
    work["date"] = _date_column(work, ("date", "as_of", "updated", "disclosure_date"))
    work["position"] = _numeric(work, ("shares", "units", "value", "weight"))
    work = work.loc[work["fund_id"].ne("") & work["symbol"].ne("") & work["date"].notna()]
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values(["fund_id", "symbol", "date"])
    group = work.groupby(["fund_id", "symbol"], sort=False, group_keys=False)
    work["previous_position"] = group["position"].shift(1).fillna(0.0)
    work["delta"] = work["position"] - work["previous_position"]
    type_name = str(fund_type).strip().lower().replace("-", "_")
    buy_family = f"fund_activity.{type_name}_buy" if type_name in {"etf", "mutual_fund"} else "fund_activity.institutional_buy"
    rows: list[pd.DataFrame] = []
    for family, mask in (
        (buy_family, work["delta"].gt(0)),
        ("fund_activity.add", work["delta"].gt(0) & work["previous_position"].gt(0)),
        ("fund_activity.reduce", work["delta"].lt(0) & work["position"].gt(0)),
        ("fund_activity.exit", work["position"].eq(0) & work["previous_position"].gt(0)),
    ):
        selected = work.loc[mask].copy()
        if selected.empty:
            continue
        selected["signal_value"] = selected["delta"].abs().astype("float32")
        rows.append(_event_rows(selected, family=family, text_columns=("fund_id",)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_holder_activity_events(analytics: pd.DataFrame) -> pd.DataFrame:
    """Create holder-level events from FMP extract-analytics rows."""
    if analytics.empty or "symbol" not in analytics.columns:
        return pd.DataFrame()
    work = analytics.copy().rename(columns={
        "investorName": "investor_name",
        "changeInSharesNumber": "change_in_shares",
        "isNew": "is_new",
        "isSoldOut": "is_sold_out",
    })
    work["holder_id"] = work.get("cik", "").astype(str).str.strip()
    work["holder_name"] = work.get("investor_name", "").astype(str).str.strip()
    change = pd.to_numeric(work.get("change_in_shares", 0), errors="coerce").fillna(0.0)
    is_new = work.get("is_new", False).astype(str).str.lower().isin({"true", "1", "yes"})
    is_exit = work.get("is_sold_out", False).astype(str).str.lower().isin({"true", "1", "yes"})
    work["signal_value"] = change.abs().astype("float32")
    rows: list[pd.DataFrame] = []
    for family, mask in (
        ("holder_activity.buy", is_new | change.gt(0)),
        ("holder_activity.add", change.gt(0) & ~is_new),
        ("holder_activity.reduce", change.lt(0) & ~is_exit),
        ("holder_activity.exit", is_exit),
    ):
        selected = work.loc[mask].copy()
        if not selected.empty:
            rows.append(_event_rows(selected, family=family, text_columns=("holder_id", "holder_name")))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


__all__ = [
    "FUND_ACTIVITY_TARGET_FAMILY",
    "FUND_ACTIVITY_TARGET_FAMILIES",
    "HOLDER_ACTIVITY_TARGET_FAMILIES",
    "build_fund_holding_activity_events",
    "build_holder_activity_events",
    "build_institutional_activity_events",
]
