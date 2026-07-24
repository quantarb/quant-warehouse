"""Historical index-constituent event targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


INDEX_NAMES = ("sp500", "nasdaq", "dowjones")
SUB_SECTOR_TARGET_COLUMN = "sub_sector_target"
EVENT_COLUMNS = tuple(
    column
    for index_name in INDEX_NAMES
    for column in (f"is_{index_name}_add", f"is_{index_name}_remove")
)


def _first_value(row: Mapping, names: tuple[str, ...]) -> object:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return value
    return None


def build_historical_constituent_event_label_panel(
    token_dates: pd.DataFrame | pd.Series | Iterable,
    events: Iterable[Mapping] | pd.DataFrame,
    *,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create same-day add/remove labels keyed by date and symbol.

    FMP has returned several field shapes over time. This parser supports both
    rows containing addedTicker/removedTicker and rows containing a single
    symbol plus an action, as well as explicit dateAdded/dateRemoved fields.
    """
    if isinstance(token_dates, pd.DataFrame):
        dates = pd.to_datetime(token_dates["date"], errors="coerce").dt.normalize()
    elif isinstance(token_dates, pd.Series):
        dates = pd.to_datetime(token_dates, errors="coerce").dt.normalize()
    else:
        dates = pd.to_datetime(list(token_dates), errors="coerce").normalize()
    date_frame = pd.DataFrame({"date": dates}).dropna().drop_duplicates()
    wanted_symbols = {str(symbol).strip().upper() for symbol in symbols} if symbols is not None else None
    rows: list[dict] = []
    records = events.to_dict("records") if isinstance(events, pd.DataFrame) else list(events)
    for raw in records:
        row = {str(key): value for key, value in dict(raw).items()}
        index_name = str(_first_value(row, ("index", "indexName", "index_name")) or "sp500").lower()
        if index_name not in INDEX_NAMES:
            continue
        prefix = f"is_{index_name}_"
        common_date = _first_value(row, ("date", "eventDate", "event_date"))
        changes = [
            (
                _first_value(row, ("addedTicker", "addedSymbol", "added_symbol", "added")),
                _first_value(row, ("dateAdded", "date_added", "addedDate")) or common_date,
                f"{prefix}add",
            ),
            (
                _first_value(row, ("removedTicker", "removedSymbol", "removed_symbol", "removed")),
                _first_value(row, ("dateRemoved", "date_removed", "removedDate")) or common_date,
                f"{prefix}remove",
            ),
        ]
        single_symbol = _first_value(row, ("symbol", "ticker"))
        action = str(_first_value(row, ("action", "type", "event")) or "").lower()
        if single_symbol is not None:
            if _first_value(row, ("dateAdded", "date_added", "addedDate")) is not None:
                changes.append((single_symbol, _first_value(row, ("dateAdded", "date_added", "addedDate")), f"{prefix}add"))
            elif any(token in action for token in ("add", "includ", "join")):
                changes.append((single_symbol, common_date, f"{prefix}add"))
            elif any(token in action for token in ("remov", "exclud", "drop", "leave")):
                changes.append((single_symbol, common_date, f"{prefix}remove"))
        for symbol, event_date, label in changes:
            if symbol is None or event_date is None:
                continue
            normalized_symbol = str(symbol).strip().upper()
            if not normalized_symbol or (wanted_symbols is not None and normalized_symbol not in wanted_symbols):
                continue
            parsed_date = pd.to_datetime(event_date, errors="coerce")
            if pd.isna(parsed_date):
                continue
            rows.append({"date": parsed_date.normalize(), "symbol": normalized_symbol, label: 1.0})
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", *EVENT_COLUMNS])
    events_frame = pd.DataFrame(rows).groupby(["date", "symbol"], as_index=False).max()
    events_frame = events_frame.loc[events_frame.date.isin(date_frame.date)].copy()
    for column in EVENT_COLUMNS:
        if column not in events_frame:
            events_frame[column] = 0.0
    return events_frame[["date", "symbol", *EVENT_COLUMNS]].fillna(0.0)


def build_historical_sub_sector_target_panel(
    token_dates: pd.DataFrame | pd.Series | Iterable,
    observations: Iterable[Mapping] | pd.DataFrame,
    *,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build an as-of sub-sector classification target panel.

    FMP returns this classification as ``subSector`` on current constituent
    rows, while historical membership rows are sparse and do not consistently
    include it.  The classification is therefore carried forward from the
    recorded ``dateFirstAdded`` (or event date) to later token dates for the
    same symbol.  Dates before the first known classification remain ``-1``.
    The integer mapping is deterministic over the observed sub-sector names.
    """
    if isinstance(token_dates, pd.DataFrame):
        token_frame = token_dates[["date", "symbol"]].copy()
    else:
        dates = token_dates if isinstance(token_dates, pd.Series) else list(token_dates)
        token_frame = pd.DataFrame({"date": dates})
        token_frame["symbol"] = ""
    token_frame["date"] = pd.to_datetime(token_frame["date"], errors="coerce").dt.normalize()
    token_frame["symbol"] = token_frame["symbol"].astype(str).str.strip().str.upper()
    token_frame = token_frame.dropna(subset=["date"])
    wanted = {str(symbol).strip().upper() for symbol in symbols} if symbols is not None else None
    if wanted is not None:
        token_frame = token_frame.loc[token_frame.symbol.isin(wanted)].copy()

    records = observations.to_dict("records") if isinstance(observations, pd.DataFrame) else list(observations)
    rows: list[dict[str, object]] = []
    for raw in records:
        row = {str(key): value for key, value in dict(raw).items()}
        symbol = _first_value(row, (
            "symbol", "ticker", "addedTicker", "addedSymbol", "added_symbol",
            "removedTicker", "removedSymbol", "removed_symbol",
        ))
        sub_sector = _first_value(row, (
            "subSector", "sub_sector", "subsector", "subSectorName",
        ))
        observed_date = _first_value(row, (
            "date", "eventDate", "event_date", "dateAdded", "date_added",
            "dateFirstAdded",
        ))
        if symbol is None or sub_sector is None or observed_date is None:
            continue
        symbol = str(symbol).strip().upper()
        sub_sector = str(sub_sector).strip()
        if not symbol or not sub_sector or (wanted is not None and symbol not in wanted):
            continue
        parsed_date = pd.to_datetime(observed_date, errors="coerce")
        if pd.isna(parsed_date):
            continue
        rows.append({"symbol": symbol, "date": parsed_date.normalize(), "sub_sector": sub_sector})
    if not rows or token_frame.empty:
        return pd.DataFrame(columns=["date", "symbol", SUB_SECTOR_TARGET_COLUMN])

    source = pd.DataFrame(rows).sort_values(["symbol", "date"])
    source = source.drop_duplicates(["symbol", "date"], keep="last")
    classes = {value: index for index, value in enumerate(sorted(source.sub_sector.unique()))}
    source[SUB_SECTOR_TARGET_COLUMN] = source.sub_sector.map(classes).astype("int64")
    source_by_symbol = {
        symbol: group.sort_values("date")
        for symbol, group in source.groupby("symbol", sort=False)
    }
    output: list[pd.DataFrame] = []
    for symbol, group in token_frame.groupby("symbol", sort=False):
        current = group.sort_values("date").copy()
        history = source_by_symbol.get(symbol)
        current[SUB_SECTOR_TARGET_COLUMN] = -1
        if history is not None and not history.empty:
            positions = history.date.searchsorted(current.date, side="right") - 1
            valid = positions >= 0
            current.loc[valid, SUB_SECTOR_TARGET_COLUMN] = history.iloc[positions[valid]][SUB_SECTOR_TARGET_COLUMN].to_numpy()
        output.append(current)
    return pd.concat(output, ignore_index=True)[["date", "symbol", SUB_SECTOR_TARGET_COLUMN]]
