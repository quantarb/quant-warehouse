from __future__ import annotations

import pandas as pd

from quant_warehouse.ingest.credentials import configure_openbb_credentials
from quant_warehouse.ingest.normalize import clip_to_min_historical_date, coerce_object_dates
from quant_warehouse.warehouse.sections import MIN_HISTORICAL_DATE

CALENDAR_DATE_COLUMNS: dict[str, str] = {
    "equity_calendar_earnings": "report_date",
    "equity_calendar_dividend": "ex_dividend_date",
    "equity_calendar_splits": "date",
    "equity_calendar_ipo": "ipo_date",
}

CALENDAR_ROUTES: dict[str, str] = {
    "equity_calendar_earnings": "equity.calendar.earnings",
    "equity_calendar_dividend": "equity.calendar.dividend",
    "equity_calendar_splits": "equity.calendar.splits",
    "equity_calendar_ipo": "equity.calendar.ipo",
}


def normalize_equity_calendar_frame(
    raw: pd.DataFrame,
    *,
    section: str,
    min_date: str = MIN_HISTORICAL_DATE,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    date_col = CALENDAR_DATE_COLUMNS.get(section, "date")
    if date_col not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.name or "").lower() in {
            "date",
            date_col,
        }:
            frame = frame.reset_index()
            index_name = frame.columns[0]
            if index_name != date_col:
                frame = frame.rename(columns={index_name: date_col})
        else:
            return pd.DataFrame()

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    for column in frame.columns:
        if column in {date_col, "symbol", "name", "exchange", "actions", "split_type", "frequency"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="ignore")

    frame = coerce_object_dates(frame)
    frame = frame.set_index(date_col)
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = date_col
    return clip_to_min_historical_date(frame, min_date=min_date)


def fetch_equity_calendar_chunk(
    section: str,
    *,
    provider: str = "fmp",
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    provider_name = str(provider or "fmp").strip().lower()
    route = CALENDAR_ROUTES.get(section)
    if route is None:
        raise ValueError(f"Unknown equity calendar section: {section}")

    configure_openbb_credentials()
    from openbb import obb

    parts = route.split(".")
    obj = obb
    for part in parts:
        obj = getattr(obj, part)
    try:
        result = obj(
            start_date=str(start_date)[:10],
            end_date=str(end_date)[:10],
            provider=provider_name,
        )
        return normalize_equity_calendar_frame(result.to_df(), section=section)
    except Exception:
        return pd.DataFrame()


def fetch_equity_calendar_range(
    section: str,
    *,
    provider: str = "fmp",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date or MIN_HISTORICAL_DATE)
    end = pd.Timestamp(end_date or pd.Timestamp.utcnow().tz_convert("America/New_York").date())
    frames: list[pd.DataFrame] = []
    cursor = start.to_period("M").to_timestamp()
    while cursor <= end:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        chunk_end = min(month_end, end)
        chunk = fetch_equity_calendar_chunk(
            section,
            provider=provider,
            start_date=cursor.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
        )
        if not chunk.empty:
            frames.append(chunk)
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return clip_to_min_historical_date(combined)