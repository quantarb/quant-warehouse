"""Issuer-level labels for FMP corporate events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

CORPORATE_EVENT_COLUMNS = (
    "is_symbol_change",
    "is_delisted",
    "is_merger_acquisition",
    "is_ma_acquirer",
    "is_ma_target",
)


def _first(row: Mapping, names: tuple[str, ...]) -> object:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return value
    return None


def _symbols(row: Mapping, names: tuple[str, ...]) -> list[str]:
    value = _first(row, names)
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else str(value).replace(";", ",").split(",")
    return [str(item).strip().upper() for item in values if str(item).strip()]


def _all_symbols(row: Mapping, names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for name in names:
        values.extend(_symbols(row, (name,)))
    return list(dict.fromkeys(values))


def build_corporate_event_label_panel(
    token_dates: pd.DataFrame | pd.Series | Iterable,
    events: Iterable[Mapping] | pd.DataFrame,
    *,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create same-day issuer labels from raw FMP corporate-event rows."""
    if isinstance(token_dates, pd.DataFrame):
        dates = pd.to_datetime(token_dates["date"], errors="coerce").dt.normalize()
    elif isinstance(token_dates, pd.Series):
        dates = pd.to_datetime(token_dates, errors="coerce").dt.normalize()
    else:
        dates = pd.to_datetime(list(token_dates), errors="coerce").normalize()
    date_set = set(dates.dropna())
    wanted = {str(value).strip().upper() for value in symbols} if symbols is not None else None
    records = events.to_dict("records") if isinstance(events, pd.DataFrame) else list(events)
    rows: list[dict] = []
    for raw in records:
        row = {str(key): value for key, value in dict(raw).items()}
        kind = str(_first(row, ("corporate_event_type", "event_type", "type")) or "").lower()
        event_date = _first(row, ("date", "eventDate", "event_date", "transactionDate", "transaction_date", "delistedDate"))
        parsed_date = pd.to_datetime(event_date, errors="coerce")
        if pd.isna(parsed_date) or parsed_date.normalize() not in date_set:
            continue
        parsed_date = parsed_date.normalize()
        symbol_labels: list[tuple[str, dict[str, float]]] = []
        if kind in {"symbol_change", "symbol-change"}:
            symbols_for_event = _all_symbols(row, ("symbol", "ticker", "oldSymbol", "oldTicker", "newSymbol", "newTicker"))
            symbol_labels = [(symbol, {"is_symbol_change": 1.0}) for symbol in symbols_for_event]
        elif kind in {"delisted", "delisted_companies", "delisting"}:
            symbols_for_event = _symbols(row, ("symbol", "ticker", "oldSymbol", "oldTicker"))
            symbol_labels = [(symbol, {"is_delisted": 1.0}) for symbol in symbols_for_event]
        else:
            acquirers = _symbols(row, ("acquirerSymbol", "acquirerTicker", "acquiringSymbol", "acquiringTicker", "symbol", "ticker"))
            targets = _symbols(row, ("targetSymbol", "targetTicker", "targetedSymbol", "targetedTicker"))
            symbol_labels = (
                [(symbol, {"is_merger_acquisition": 1.0, "is_ma_acquirer": 1.0}) for symbol in acquirers]
                + [(symbol, {"is_merger_acquisition": 1.0, "is_ma_target": 1.0}) for symbol in targets]
            )
        for symbol, labels in symbol_labels:
            if not symbol or (wanted is not None and symbol not in wanted):
                continue
            rows.append({"date": parsed_date, "symbol": symbol, **labels})
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", *CORPORATE_EVENT_COLUMNS])
    out = pd.DataFrame(rows).groupby(["date", "symbol"], as_index=False).max()
    for column in CORPORATE_EVENT_COLUMNS:
        if column not in out:
            out[column] = 0.0
    return out[["date", "symbol", *CORPORATE_EVENT_COLUMNS]].fillna(0.0)
