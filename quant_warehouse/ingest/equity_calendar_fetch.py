from __future__ import annotations

import calendar
from datetime import date, timedelta

import polars as pl

from quant_warehouse.ingest.credentials import configure_openbb_credentials
from quant_warehouse.ingest.normalize import clip_to_min_historical_date
from quant_warehouse.warehouse.sections import MIN_HISTORICAL_DATE

CALENDAR_DATE_COLUMNS: dict[str, str] = {
    "equity_calendar_earnings": "report_date",
    "equity_calendar_dividend": "ex_dividend_date",
    "equity_calendar_splits": "date",
    "equity_calendar_ipo": "ipo_date",
}
CALENDAR_ROUTES = {
    "equity_calendar_earnings": "equity.calendar.earnings",
    "equity_calendar_dividend": "equity.calendar.dividend",
    "equity_calendar_splits": "equity.calendar.splits",
    "equity_calendar_ipo": "equity.calendar.ipo",
}


def _date_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    return (pl.col(column).str.to_datetime(strict=False)
            if frame.schema[column] == pl.String
            else pl.col(column).cast(pl.Datetime, strict=False))


def normalize_equity_calendar_frame(
    raw: pl.DataFrame, *, section: str, min_date: str = MIN_HISTORICAL_DATE
) -> pl.DataFrame:
    if raw is None or raw.is_empty():
        return pl.DataFrame()
    date_col = CALENDAR_DATE_COLUMNS.get(section, "date")
    if date_col not in raw.columns:
        return pl.DataFrame()
    out = raw.with_columns(_date_expr(raw, date_col).alias(date_col)).drop_nulls(date_col).sort(date_col)
    protected = {date_col, "symbol", "name", "exchange", "actions", "split_type", "frequency"}
    date_like = {column for column, dtype in out.schema.items() if dtype in {pl.Date, pl.Datetime}}
    numeric = [column for column in out.columns if column not in protected and column not in date_like and out.schema[column].is_numeric()]
    if numeric:
        out = out.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in numeric)
    return clip_to_min_historical_date(out, min_date=min_date)


def fetch_equity_calendar_chunk(
    section: str, *, provider: str = "fmp", start_date: str, end_date: str
) -> pl.DataFrame:
    route = CALENDAR_ROUTES.get(section)
    if route is None:
        raise ValueError(f"Unknown equity calendar section: {section}")
    configure_openbb_credentials()
    from openbb import obb
    obj = obb
    for part in route.split("."):
        obj = getattr(obj, part)
    try:
        result = obj(start_date=start_date[:10], end_date=end_date[:10], provider=str(provider).lower())
        return normalize_equity_calendar_frame(result.to_polars(), section=section)
    except Exception:
        return pl.DataFrame()


def fetch_equity_calendar_range(
    section: str, *, provider: str = "fmp", start_date: str | None = None, end_date: str | None = None
) -> pl.DataFrame:
    start = date.fromisoformat((start_date or MIN_HISTORICAL_DATE)[:10])
    end = date.fromisoformat((end_date or date.today().isoformat())[:10])
    frames: list[pl.DataFrame] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        month_end = date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1])
        chunk_end = min(month_end, end)
        chunk = fetch_equity_calendar_chunk(
            section, provider=provider, start_date=cursor.isoformat(), end_date=chunk_end.isoformat()
        )
        if not chunk.is_empty():
            frames.append(chunk)
        cursor = (month_end + timedelta(days=1)).replace(day=1)
    if not frames:
        return pl.DataFrame()
    date_col = CALENDAR_DATE_COLUMNS.get(section, "date")
    return pl.concat(frames, how="diagonal_relaxed").unique(date_col, keep="last").sort(date_col)
