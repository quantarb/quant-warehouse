"""Historical index-constituent event targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime

import polars as pl

INDEX_NAMES = ("sp500", "nasdaq", "dowjones")
SUB_SECTOR_TARGET_COLUMN = "sub_sector_target"
EVENT_COLUMNS = tuple(column for index_name in INDEX_NAMES for column in (f"is_{index_name}_add", f"is_{index_name}_remove"))


def _records(value: object) -> list[dict]:
    if isinstance(value, pl.DataFrame):
        return value.to_dicts()
    return [dict(row) for row in value] if value is not None else []


def _first_value(row: Mapping, names: tuple[str, ...]) -> object:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return value
    return None


def _date(value: object) -> datetime | None:
    if value is None: return None
    if isinstance(value, datetime): return value.replace(hour=0, minute=0, second=0, microsecond=0)
    if isinstance(value, date): return datetime.combine(value, datetime.min.time())
    try: return datetime.fromisoformat(str(value)[:10])
    except ValueError: return None


def _token_frame(token_dates: object) -> pl.DataFrame:
    if isinstance(token_dates, pl.DataFrame):
        out = token_dates
    else:
        values = list(token_dates or [])
        out = pl.DataFrame({"date": values})
    if "date" not in out.columns: return pl.DataFrame()
    if "symbol" not in out.columns: out = out.with_columns(pl.lit("").alias("symbol"))
    return out.select(["date", "symbol"]).with_columns([
        pl.col("date").cast(pl.Datetime, strict=False).dt.truncate("1d"),
        pl.col("symbol").cast(pl.String).str.strip_chars().str.to_uppercase(),
    ]).drop_nulls("date")


def build_historical_constituent_event_label_panel(token_dates: object, events: object, *, symbols: Iterable[str] | None = None) -> pl.DataFrame:
    tokens = _token_frame(token_dates)
    wanted = {str(symbol).strip().upper() for symbol in symbols} if symbols is not None else None
    allowed_dates = set(tokens["date"].to_list()) if not tokens.is_empty() else set()
    rows: list[dict] = []
    for raw in _records(events):
        row = {str(key): value for key, value in raw.items()}
        index_name = str(_first_value(row, ("index", "indexName", "index_name")) or "sp500").lower()
        if index_name not in INDEX_NAMES: continue
        prefix = f"is_{index_name}_"; common = _first_value(row, ("date", "eventDate", "event_date"))
        changes = [(_first_value(row, ("addedTicker", "addedSymbol", "added_symbol", "added")), _first_value(row, ("dateAdded", "date_added", "addedDate")) or common, f"{prefix}add"),
                   (_first_value(row, ("removedTicker", "removedSymbol", "removed_symbol", "removed")), _first_value(row, ("dateRemoved", "date_removed", "removedDate")) or common, f"{prefix}remove")]
        symbol = _first_value(row, ("symbol", "ticker")); action = str(_first_value(row, ("action", "type", "event")) or "").lower()
        if symbol is not None:
            if _first_value(row, ("dateAdded", "date_added", "addedDate")) is not None: changes.append((symbol, _first_value(row, ("dateAdded", "date_added", "addedDate")), f"{prefix}add"))
            elif any(token in action for token in ("add", "includ", "join")): changes.append((symbol, common, f"{prefix}add"))
            elif any(token in action for token in ("remov", "exclud", "drop", "leave")): changes.append((symbol, common, f"{prefix}remove"))
        for symbol, event_date, label in changes:
            parsed = _date(event_date); normalized = str(symbol or "").strip().upper()
            if parsed is not None and normalized and (wanted is None or normalized in wanted) and parsed in allowed_dates:
                rows.append({"date": parsed, "symbol": normalized, label: 1.0})
    if not rows: return pl.DataFrame(schema={"date": pl.Datetime, "symbol": pl.String, **{column: pl.Float64 for column in EVENT_COLUMNS}})
    out = pl.DataFrame(rows).group_by(["date", "symbol"]).max()
    for column in EVENT_COLUMNS:
        if column not in out.columns: out = out.with_columns(pl.lit(0.0).alias(column))
    return out.select(["date", "symbol", *EVENT_COLUMNS]).fill_null(0.0).sort(["date", "symbol"])


def build_historical_sub_sector_target_panel(token_dates: object, observations: object, *, symbols: Iterable[str] | None = None) -> pl.DataFrame:
    tokens = _token_frame(token_dates)
    wanted = {str(symbol).strip().upper() for symbol in symbols} if symbols is not None else None
    rows: list[dict] = []
    for raw in _records(observations):
        symbol = _first_value(raw, ("symbol", "ticker", "addedTicker", "addedSymbol", "added_symbol", "removedTicker", "removedSymbol", "removed_symbol"))
        sector = _first_value(raw, ("subSector", "sub_sector", "subsector", "subSectorName"))
        parsed = _date(_first_value(raw, ("date", "eventDate", "event_date", "dateAdded", "date_added", "dateFirstAdded")))
        symbol = str(symbol or "").strip().upper(); sector = str(sector or "").strip()
        if parsed is not None and symbol and sector and (wanted is None or symbol in wanted): rows.append({"symbol": symbol, "date": parsed, "sub_sector": sector})
    if tokens.is_empty() or not rows: return pl.DataFrame(schema={"date": pl.Datetime, "symbol": pl.String, SUB_SECTOR_TARGET_COLUMN: pl.Int64})
    source = pl.DataFrame(rows).sort(["symbol", "date"]).unique(["symbol", "date"], keep="last")
    classes = {value: index for index, value in enumerate(sorted(source["sub_sector"].unique().to_list()))}
    source = source.with_columns(pl.col("sub_sector").replace(classes).cast(pl.Int64).alias(SUB_SECTOR_TARGET_COLUMN)).drop("sub_sector")
    result = tokens.join_asof(source.sort("date"), on="date", by="symbol", strategy="backward").with_columns(pl.col(SUB_SECTOR_TARGET_COLUMN).fill_null(-1))
    return result.select(["date", "symbol", SUB_SECTOR_TARGET_COLUMN]).sort(["date", "symbol"])
