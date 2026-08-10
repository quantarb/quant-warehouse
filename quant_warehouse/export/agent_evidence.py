from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl


def build_agent_evidence(
    warehouse: Any,
    symbol: str,
    as_of_date: str,
    *,
    price_providers: tuple[str, ...] = ("fmp", "yfinance"),
    lookback_days: int = 260,
    news_lookback_days: int = 7,
    refresh_news_if_missing: bool = True,
) -> dict[str, Any]:
    """Build a compact point-in-time packet from warehouse-owned data."""
    ticker = str(symbol).strip().upper()
    as_of = datetime.fromisoformat(as_of_date).replace(hour=0, minute=0, second=0, microsecond=0)
    start = as_of - timedelta(days=int(lookback_days))
    prices = pl.DataFrame()
    price_provider = ""
    for provider in price_providers:
        candidate = warehouse.read_prices(
            ticker,
            provider=provider,
            start=start.strftime("%Y-%m-%d"),
            end=as_of.strftime("%Y-%m-%d"),
        )
        if candidate is not None and not candidate.is_empty():
            prices = candidate
            price_provider = provider
            break

    packet: dict[str, Any] = {
        "symbol": ticker,
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "price_provider": price_provider,
        "price_summary": _price_summary(prices),
        "fundamentals": {},
        "news": [],
    }
    for section in ("income", "balance", "cash", "metrics", "ratios"):
        frame = warehouse.read_fundamentals(ticker, section=section, provider="fmp", end=as_of)
        if frame is not None and not frame.is_empty():
            packet["fundamentals"][section] = _latest_record(frame)

    news_start = as_of - timedelta(days=int(news_lookback_days))
    if refresh_news_if_missing and hasattr(warehouse.news, "ensure_date"):
        warehouse.news.ensure_date(ticker, as_of, provider="fmp")
    news = warehouse.read_news(
        ticker,
        provider="fmp",
        start=news_start.strftime("%Y-%m-%d"),
        end=(as_of + timedelta(days=1) - timedelta(microseconds=1)).isoformat(),
    )
    if news is not None and not news.is_empty():
        rows = news.sort("published_at", descending=True).head(20)
        for row in rows.iter_rows(named=True):
            packet["news"].append(
                {
                    "published_at": _json_value(row.get("published_at")),
                    "title": _json_value(row.get("title")),
                    "source": _json_value(row.get("source")),
                    "excerpt": _json_value(row.get("excerpt")),
                }
            )
    packet["sufficient"] = bool(packet["price_summary"])
    packet["missing"] = [
        name
        for name, present in (
            ("prices", bool(packet["price_summary"])),
            ("fundamentals", bool(packet["fundamentals"])),
            ("news", bool(packet["news"])),
        )
        if not present
    ]
    packet["roles"] = {
        "market": {
            "sufficient": bool(packet["price_summary"]),
            "price_provider": packet["price_provider"],
            "price_summary": packet["price_summary"],
        },
        "fundamentals": {
            "sufficient": bool(packet["fundamentals"]),
            "fundamentals": packet["fundamentals"],
        },
        "news": {"sufficient": bool(packet["news"]), "news": packet["news"]},
        "social": {
            "sufficient": False,
            "social": [],
            "missing": ["point_in_time_social_data"],
        },
    }
    return packet


def _price_summary(frame: pl.DataFrame) -> dict[str, Any]:
    if frame is None or frame.is_empty():
        return {}
    out = frame.sort("date")
    close_column = next((name for name in ("close", "Close", "adj_close") if name in out), None)
    if close_column is None:
        return {}
    close = out.select(pl.col(close_column).cast(pl.Float64, strict=False)).to_series().drop_nulls()
    if close.is_empty():
        return {}
    latest = out.select("date").tail(1).item()
    result: dict[str, Any] = {
        "latest_date": latest.strftime("%Y-%m-%d") if hasattr(latest, "strftime") else str(latest),
        "close": float(close[-1]),
        "observations": close.len(),
    }
    for days in (5, 20, 60):
        if len(close) > days:
            result[f"return_{days}d"] = float(close[-1] / close[-days - 1] - 1.0)
        if len(close) >= days:
            result[f"sma_{days}"] = float(close.tail(days).mean())
    if len(close) > 20:
        result["volatility_20d"] = float(close.pct_change().tail(20).std())
    for column in ("open", "high", "low", "volume"):
        if column in out:
            value = out.select(pl.col(column).cast(pl.Float64, strict=False)).to_series().drop_nulls()
            if not value.is_empty():
                result[column] = float(value[-1])
    return result


def _latest_record(frame: pl.DataFrame) -> dict[str, Any]:
    ordered = frame.sort("date") if "date" in frame.columns else frame
    row = ordered.tail(1).to_dicts()[0]
    record = {str(key): _json_value(value) for key, value in row.items()}
    date_column = "date" if "date" in ordered.columns else ordered.columns[0]
    record["period"] = row[date_column].strftime("%Y-%m-%d") if hasattr(row[date_column], "strftime") else str(row[date_column])
    return {key: value for key, value in record.items() if value not in (None, "")}


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
