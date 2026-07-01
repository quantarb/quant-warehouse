from __future__ import annotations

from typing import Any

import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_pair_schema import EVENT_PAIR_COLUMNS
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_pair_taxonomy import get_event_side, get_mirror_event_type


def normalize_event_pairs(
    raw_events: pd.DataFrame,
    *,
    event_family: str,
    event_type_col: str,
    symbol_col: str,
    event_date_col: str,
    source: str,
    actor_type_col: str | None = None,
    actor_name_col: str | None = None,
    actor_role_col: str | None = None,
    actor_chamber_col: str | None = None,
    actor_firm_col: str | None = None,
    actor_title_col: str | None = None,
    strength_col: str | None = None,
    transaction_shares_col: str | None = None,
    transaction_price_col: str | None = None,
    transaction_value_col: str | None = None,
    reported_date_col: str | None = None,
    raw_json_col: str | None = None,
) -> pd.DataFrame:
    """Normalize observed mirrored event-pair rows on exact event dates only."""

    if raw_events is None or raw_events.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    required = [event_type_col, symbol_col, event_date_col]
    for optional in (
        actor_type_col,
        actor_name_col,
        actor_role_col,
        actor_chamber_col,
        actor_firm_col,
        actor_title_col,
        strength_col,
        transaction_shares_col,
        transaction_price_col,
        transaction_value_col,
        reported_date_col,
        raw_json_col,
    ):
        if optional:
            required.append(optional)
    _require_columns(raw_events, required, ctx="normalize_event_pairs")

    rows: list[dict[str, Any]] = []
    family = str(event_family).strip().lower()
    for _, row in raw_events.iterrows():
        event_type = str(row[event_type_col]).strip().lower()
        event_side = get_event_side(family, event_type)
        event_date = pd.to_datetime(row[event_date_col], errors="coerce")
        if pd.isna(event_date):
            continue
        reported_date = pd.to_datetime(_optional_value(row, reported_date_col), errors="coerce")
        if pd.isna(reported_date):
            reported_date_value = None
            disclosure_lag_days = None
        else:
            reported_date_value = reported_date.normalize()
            disclosure_lag_days = int((reported_date_value - event_date.normalize()).days)
        rows.append(
            {
                "symbol": str(row[symbol_col]).strip().upper(),
                "event_date": event_date.normalize(),
                "event_family": family,
                "event_type": event_type,
                "event_side": event_side,
                "mirror_event_type": get_mirror_event_type(family, event_type),
                "actor_type": _optional_value(row, actor_type_col),
                "actor_name": _optional_value(row, actor_name_col),
                "actor_role": _optional_value(row, actor_role_col),
                "actor_chamber": _optional_value(row, actor_chamber_col),
                "actor_firm": _optional_value(row, actor_firm_col),
                "actor_title": _optional_value(row, actor_title_col),
                "source": source,
                "strength": _optional_value(row, strength_col),
                "transaction_shares": _optional_value(row, transaction_shares_col),
                "transaction_price": _optional_value(row, transaction_price_col),
                "transaction_value": _optional_value(row, transaction_value_col),
                "reported_date": reported_date_value,
                "disclosure_lag_days": disclosure_lag_days,
                "raw_json": _optional_value(row, raw_json_col),
            }
        )
    if not rows:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    return pd.DataFrame(rows, columns=EVENT_PAIR_COLUMNS).sort_values(
        ["symbol", "event_date", "event_type"],
        ignore_index=True,
    )


def _optional_value(row: pd.Series, column: str | None) -> Any:
    if not column:
        return None
    return row[column]


def _require_columns(df: pd.DataFrame, columns: list[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
