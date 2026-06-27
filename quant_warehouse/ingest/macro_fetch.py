from __future__ import annotations

import re
from typing import Any

import pandas as pd

from quant_warehouse.ingest.credentials import configure_openbb_credentials
from quant_warehouse.ingest.normalize import clip_to_min_historical_date
from quant_warehouse.warehouse.sections import MIN_HISTORICAL_DATE


def _records_to_frame(records: Any, *, value_column: str = "value") -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    if "date" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if value_column not in frame.columns:
        numeric_cols = [
            column
            for column in frame.columns
            if column != "date" and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not numeric_cols:
            return pd.DataFrame()
        value_column = numeric_cols[0]
    out = frame[["date", value_column]].rename(columns={value_column: "value"}).set_index("date")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return clip_to_min_historical_date(out.dropna(subset=["value"]))


def fetch_economic_indicator_series(
    name: str,
    *,
    provider: str = "fmp",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    provider_name = str(provider or "fmp").strip().lower()
    if provider_name != "fmp":
        raise ValueError(f"Unsupported macro economic provider: {provider_name}")
    configure_openbb_credentials()
    from openbb import obb

    kwargs: dict[str, Any] = {"symbol": str(name).strip(), "provider": "fmp"}
    if start_date:
        kwargs["start_date"] = str(start_date)[:10]
    if end_date:
        kwargs["end_date"] = str(end_date)[:10]
    result = obb.economy.indicators(**kwargs)
    return _records_to_frame(result.to_df())


def _normalize_treasury_column_name(column: str) -> str:
    name = str(column).strip()
    if name == "date":
        return name
    if name.startswith("macro__ust_"):
        return name[len("macro__ust_") :]
    normalized = re.sub(r"_+", "_", name).strip("_")
    if "_" in normalized:
        head, tail = normalized.split("_", 1)
        if head in {"month", "year"} and tail.isdigit():
            return f"{head}{tail}"
    return normalized


def _treasury_wide_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if "date" not in frame.columns:
        index_name = str(frame.index.name or "").strip().lower()
        if isinstance(frame.index, pd.DatetimeIndex) or index_name in {"date", "period_ending"}:
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        else:
            return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    rename_map = {
        column: _normalize_treasury_column_name(column)
        for column in frame.columns
        if column != "date"
    }
    frame = frame.rename(columns=rename_map)
    numeric_cols = [column for column in frame.columns if column != "date"]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return clip_to_min_historical_date(frame.set_index("date"))


def fetch_treasury_rates_wide(
    *,
    provider: str = "fmp",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    provider_name = str(provider or "fmp").strip().lower()
    if provider_name != "fmp":
        raise ValueError(f"Unsupported macro treasury provider: {provider_name}")
    configure_openbb_credentials()
    from openbb import obb

    kwargs: dict[str, Any] = {"provider": "fmp"}
    if start_date:
        kwargs["start_date"] = str(start_date)[:10]
    if end_date:
        kwargs["end_date"] = str(end_date)[:10]
    result = obb.fixedincome.government.treasury_rates(**kwargs)
    return _treasury_wide_frame(result.to_df())


def treasury_series_code(column: str) -> str:
    normalized = _normalize_treasury_column_name(column)
    return f"macro__ust_{normalized}"


def yield_curve_series_code(column: str) -> str:
    normalized = _normalize_treasury_column_name(column)
    return f"macro__yc_{normalized}"


def _yield_curve_wide_from_long(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if "date" not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.name or "").lower() == "date":
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        else:
            return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if "maturity" not in frame.columns or "rate" not in frame.columns:
        return pd.DataFrame()
    frame["maturity"] = frame["maturity"].map(_normalize_treasury_column_name)
    frame["rate"] = pd.to_numeric(frame["rate"], errors="coerce")
    wide = (
        frame.pivot_table(index="date", columns="maturity", values="rate", aggfunc="last")
        .sort_index()
    )
    wide.index = pd.DatetimeIndex(wide.index)
    return clip_to_min_historical_date(wide)


def fetch_yield_curve_snapshot(
    date: str,
    *,
    provider: str = "fmp",
) -> pd.DataFrame:
    provider_name = str(provider or "fmp").strip().lower()
    if provider_name != "fmp":
        raise ValueError(f"Unsupported yield curve provider: {provider_name}")
    configure_openbb_credentials()
    from openbb import obb

    result = obb.fixedincome.government.yield_curve(date=str(date)[:10], provider="fmp")
    return _yield_curve_wide_from_long(result.to_df())


def fetch_yield_curve_history(
    *,
    provider: str = "fmp",
    start_date: str | None = None,
    end_date: str | None = None,
    existing_dates: set[pd.Timestamp] | None = None,
    step_days: int = 1,
) -> pd.DataFrame:
    provider_name = str(provider or "fmp").strip().lower()
    if provider_name != "fmp":
        raise ValueError(f"Unsupported yield curve provider: {provider_name}")
    start = pd.Timestamp(start_date or MIN_HISTORICAL_DATE)
    end = pd.Timestamp(end_date or pd.Timestamp.utcnow().tz_convert("America/New_York").date())
    known = {pd.Timestamp(value).normalize() for value in (existing_dates or set())}
    step = max(1, int(step_days))
    business_days = pd.bdate_range(start=start, end=end)[::step]
    frames: list[pd.DataFrame] = []
    for day in business_days:
        normalized = pd.Timestamp(day).normalize()
        if normalized in known:
            continue
        try:
            snapshot = fetch_yield_curve_snapshot(normalized.strftime("%Y-%m-%d"), provider=provider_name)
            if not snapshot.empty:
                frames.append(snapshot)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return clip_to_min_historical_date(combined)


def normalize_calendar_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if "date" not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.name or "").lower() == "date":
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        else:
            return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    for column in frame.columns:
        if column == "date":
            continue
        if column in {"consensus", "previous", "actual", "change", "change_percent"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "date"
    return clip_to_min_historical_date(frame)


def fetch_economy_calendar(
    *,
    provider: str = "fmp",
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    provider_name = str(provider or "fmp").strip().lower()
    if provider_name != "fmp":
        raise ValueError(f"Unsupported macro calendar provider: {provider_name}")
    configure_openbb_credentials()
    from openbb import obb

    try:
        result = obb.economy.calendar(
            start_date=str(start_date)[:10],
            end_date=str(end_date)[:10],
            provider="fmp",
        )
        return normalize_calendar_frame(result.to_df())
    except Exception:
        return pd.DataFrame()


def fetch_economy_calendar_range(
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
        chunk = fetch_economy_calendar(
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


def normalize_risk_premium_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy().reset_index(drop=True)
    for column in ("total_equity_risk_premium", "country_risk_premium"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "country" not in frame.columns:
        return pd.DataFrame()
    frame = frame.drop_duplicates(subset=["country"], keep="last")
    frame = frame.set_index("country")
    frame.index.name = "country"
    return frame


def fetch_risk_premium_snapshot(*, provider: str = "fmp") -> pd.DataFrame:
    provider_name = str(provider or "fmp").strip().lower()
    if provider_name != "fmp":
        raise ValueError(f"Unsupported risk premium provider: {provider_name}")
    configure_openbb_credentials()
    from openbb import obb

    result = obb.economy.risk_premium(provider="fmp")
    return normalize_risk_premium_frame(result.to_df())


__all__ = [
    "fetch_economic_indicator_series",
    "fetch_economy_calendar",
    "fetch_economy_calendar_range",
    "fetch_risk_premium_snapshot",
    "fetch_treasury_rates_wide",
    "fetch_yield_curve_history",
    "fetch_yield_curve_snapshot",
    "normalize_calendar_frame",
    "normalize_risk_premium_frame",
    "treasury_series_code",
    "yield_curve_series_code",
]
