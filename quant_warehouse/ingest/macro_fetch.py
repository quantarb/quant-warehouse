from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from quant_warehouse.ingest.credentials import configure_openbb_credentials
from quant_warehouse.ingest.normalize import clip_to_min_historical_date
from quant_warehouse.warehouse.sections import MIN_HISTORICAL_DATE

def _as_polars(value: Any) -> pl.DataFrame:
    if value is None:
        return pl.DataFrame()
    if isinstance(value, pl.DataFrame):
        return value
    if hasattr(value, "to_polars"):
        return value.to_polars()
    if isinstance(value, list):
        return pl.DataFrame(value)
    return pl.DataFrame(value)


def _date_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    expr = pl.col(column)
    if frame.schema[column] == pl.String:
        expr = expr.str.to_datetime(strict=False, time_zone="UTC")
    else:
        expr = expr.cast(pl.Datetime, strict=False)
    return expr.dt.replace_time_zone(None).dt.truncate("1d")


def _records_to_frame(records: Any, *, value_column: str = "value") -> pl.DataFrame:
    frame = _as_polars(records)
    if frame.is_empty() or "date" not in frame.columns:
        return pl.DataFrame()
    frame = frame.with_columns(_date_expr(frame, "date").alias("date")).drop_nulls("date")
    if value_column not in frame.columns:
        candidates = [c for c in frame.columns if c != "date" and frame.schema[c].is_numeric()]
        if not candidates:
            return pl.DataFrame()
        value_column = candidates[0]
    return clip_to_min_historical_date(
        frame.select(["date", pl.col(value_column).cast(pl.Float64, strict=False).alias("value")]).drop_nulls("value")
    )


def fetch_economic_indicator_series(name: str, *, provider: str = "fmp", start_date: str | None = None,
                                    end_date: str | None = None) -> pl.DataFrame:
    if str(provider or "fmp").lower() != "fmp":
        raise ValueError(f"Unsupported macro economic provider: {provider}")
    configure_openbb_credentials()
    from openbb import obb
    kwargs: dict[str, Any] = {"symbol": str(name).strip(), "provider": "fmp"}
    if start_date: kwargs["start_date"] = str(start_date)[:10]
    if end_date: kwargs["end_date"] = str(end_date)[:10]
    result = obb.economy.indicators(**kwargs)
    return _records_to_frame(result.to_polars())


def _normalize_treasury_column_name(column: str) -> str:
    name = str(column).strip()
    if name == "date": return name
    if name.startswith("macro__ust_"): return name[len("macro__ust_"):]
    normalized = re.sub(r"_+", "_", name).strip("_")
    if "_" in normalized:
        head, tail = normalized.split("_", 1)
        if head in {"month", "year"} and tail.isdigit(): return f"{head}{tail}"
    return normalized


def _treasury_wide_frame(raw: Any) -> pl.DataFrame:
    frame = _as_polars(raw)
    if frame.is_empty() or "date" not in frame.columns: return pl.DataFrame()
    frame = frame.with_columns(_date_expr(frame, "date").alias("date")).drop_nulls("date")
    rename = {c: _normalize_treasury_column_name(c) for c in frame.columns if c != "date"}
    frame = frame.rename(rename)
    numeric = [c for c in frame.columns if c != "date"]
    return clip_to_min_historical_date(frame.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric])
                                       .unique("date", keep="last").sort("date"))


def fetch_treasury_rates_wide(*, provider: str = "fmp", start_date: str | None = None,
                              end_date: str | None = None) -> pl.DataFrame:
    if str(provider or "fmp").lower() != "fmp": raise ValueError(f"Unsupported macro treasury provider: {provider}")
    configure_openbb_credentials()
    from openbb import obb
    kwargs: dict[str, Any] = {"provider": "fmp"}
    if start_date: kwargs["start_date"] = str(start_date)[:10]
    if end_date: kwargs["end_date"] = str(end_date)[:10]
    return _treasury_wide_frame(obb.fixedincome.government.treasury_rates(**kwargs).to_polars())


def treasury_series_code(column: str) -> str: return f"macro__ust_{_normalize_treasury_column_name(column)}"
def yield_curve_series_code(column: str) -> str: return f"macro__yc_{_normalize_treasury_column_name(column)}"


def _yield_curve_wide_from_long(raw: Any) -> pl.DataFrame:
    frame = _as_polars(raw)
    if frame.is_empty() or not {"date", "maturity", "rate"}.issubset(frame.columns): return pl.DataFrame()
    frame = frame.with_columns([
        _date_expr(frame, "date").alias("date"),
        pl.col("maturity").cast(pl.String).map_elements(_normalize_treasury_column_name, return_dtype=pl.String),
        pl.col("rate").cast(pl.Float64, strict=False),
    ]).drop_nulls(["date", "rate"])
    return clip_to_min_historical_date(frame.pivot(on="maturity", index="date", values="rate", aggregate_function="last").sort("date"))


def fetch_yield_curve_snapshot(date: str, *, provider: str = "fmp") -> pl.DataFrame:
    if str(provider or "fmp").lower() != "fmp": raise ValueError(f"Unsupported yield curve provider: {provider}")
    configure_openbb_credentials()
    from openbb import obb
    return _yield_curve_wide_from_long(obb.fixedincome.government.yield_curve(date=str(date)[:10], provider="fmp").to_polars())


def _days(start: datetime, end: datetime) -> list[datetime]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1) if (start + timedelta(days=i)).weekday() < 5]


def fetch_yield_curve_history(*, provider: str = "fmp", start_date: str | None = None, end_date: str | None = None,
                              existing_dates: set[datetime] | None = None, step_days: int = 1) -> pl.DataFrame:
    start = datetime.fromisoformat(str(start_date or MIN_HISTORICAL_DATE)[:10]); end = datetime.fromisoformat(str(end_date or date.today())[:10])
    known = {value.replace(hour=0, minute=0, second=0, microsecond=0) for value in existing_dates or set()}
    frames = [fetch_yield_curve_snapshot(day.strftime("%Y-%m-%d"), provider=provider)
              for day in _days(start, end)[::max(1, int(step_days))] if day not in known]
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed").unique("date", keep="last").sort("date") if frames else pl.DataFrame()


def normalize_calendar_frame(raw: Any) -> pl.DataFrame:
    frame = _as_polars(raw)
    if frame.is_empty() or "date" not in frame.columns: return pl.DataFrame()
    aliases = {"consensus": "estimate", "importance": "impact", "change_percent": "changePercentage"}
    frame = frame.rename({source: target for source, target in aliases.items() if source in frame.columns and target not in frame.columns})
    frame = frame.with_columns(_date_expr(frame, "date").alias("date")).drop_nulls("date")
    numeric = [c for c in ("estimate", "previous", "actual", "change", "changePercentage") if c in frame.columns]
    return clip_to_min_historical_date(frame.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric]).sort("date"))


def fetch_economy_calendar(*, provider: str = "fmp", start_date: str, end_date: str,
                           ) -> pl.DataFrame:
    if str(provider or "fmp").lower() != "fmp": raise ValueError(f"Unsupported macro calendar provider: {provider}")
    configure_openbb_credentials()
    from openbb import obb
    result = obb.economy.calendar(start_date=str(start_date)[:10], end_date=str(end_date)[:10], provider="fmp")
    return normalize_calendar_frame(result.to_polars())


def fetch_economy_calendar_range(*, provider: str = "fmp", start_date: str | None = None, end_date: str | None = None,
                                 ) -> pl.DataFrame:
    start = datetime.fromisoformat(str(start_date or MIN_HISTORICAL_DATE)[:10]); end = datetime.fromisoformat(str(end_date or date.today())[:10])
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=89), end)
        frame = fetch_economy_calendar(provider=provider, start_date=cursor.strftime("%Y-%m-%d"), end_date=chunk_end.strftime("%Y-%m-%d"))
        if not frame.is_empty(): frames.append(frame)
        cursor = chunk_end + timedelta(days=1)
    return pl.concat(frames, how="diagonal_relaxed").sort("date") if frames else pl.DataFrame()


def normalize_risk_premium_frame(raw: Any) -> pl.DataFrame:
    frame = _as_polars(raw)
    if frame.is_empty() or "country" not in frame.columns: return pl.DataFrame()
    numeric = [c for c in ("total_equity_risk_premium", "country_risk_premium") if c in frame.columns]
    return frame.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric]).unique("country", keep="last")


def fetch_risk_premium_snapshot(*, provider: str = "fmp") -> pl.DataFrame:
    if str(provider or "fmp").lower() != "fmp": raise ValueError(f"Unsupported risk premium provider: {provider}")
    configure_openbb_credentials()
    from openbb import obb
    return normalize_risk_premium_frame(obb.economy.risk_premium(provider="fmp").to_polars())
