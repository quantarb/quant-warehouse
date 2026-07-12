from __future__ import annotations

from typing import Any

import pandas as pd


def build_agent_evidence(
    warehouse: Any,
    symbol: str,
    as_of_date: str,
    *,
    price_providers: tuple[str, ...] = ("fmp", "yfinance"),
    lookback_days: int = 260,
    news_lookback_days: int = 7,
) -> dict[str, Any]:
    """Build a compact point-in-time packet from warehouse-owned data."""
    ticker = str(symbol).strip().upper()
    as_of = pd.Timestamp(as_of_date).normalize()
    start = as_of - pd.Timedelta(days=int(lookback_days))
    prices = pd.DataFrame()
    price_provider = ""
    for provider in price_providers:
        candidate = warehouse.read_prices(
            ticker,
            provider=provider,
            start=start.strftime("%Y-%m-%d"),
            end=as_of.strftime("%Y-%m-%d"),
        )
        if candidate is not None and not candidate.empty:
            prices = candidate.copy()
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
        if frame is not None and not frame.empty:
            packet["fundamentals"][section] = _latest_record(frame)

    news_start = as_of - pd.Timedelta(days=int(news_lookback_days))
    news = warehouse.read_news(
        ticker,
        provider="fmp",
        start=news_start.strftime("%Y-%m-%d"),
        end=(as_of + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).isoformat(),
    )
    if news is not None and not news.empty:
        rows = news.sort_index(ascending=False).head(20).reset_index()
        for _, row in rows.iterrows():
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
    return packet


def _price_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    out = frame.sort_index().copy()
    close_column = next((name for name in ("close", "Close", "adj_close") if name in out), None)
    if close_column is None:
        return {}
    close = pd.to_numeric(out[close_column], errors="coerce").dropna()
    if close.empty:
        return {}
    latest = close.index[-1]
    result: dict[str, Any] = {
        "latest_date": pd.Timestamp(latest).strftime("%Y-%m-%d"),
        "close": float(close.iloc[-1]),
        "observations": int(len(close)),
    }
    for days in (5, 20, 60):
        if len(close) > days:
            result[f"return_{days}d"] = float(close.iloc[-1] / close.iloc[-days - 1] - 1.0)
        if len(close) >= days:
            result[f"sma_{days}"] = float(close.tail(days).mean())
    if len(close) > 20:
        result["volatility_20d"] = float(close.pct_change().tail(20).std())
    for column in ("open", "high", "low", "volume"):
        if column in out:
            value = pd.to_numeric(out[column], errors="coerce").dropna()
            if not value.empty:
                result[column] = float(value.iloc[-1])
    return result


def _latest_record(frame: pd.DataFrame) -> dict[str, Any]:
    row = frame.sort_index().iloc[-1]
    record = {str(key): _json_value(value) for key, value in row.items()}
    record["period"] = pd.Timestamp(frame.sort_index().index[-1]).strftime("%Y-%m-%d")
    return {key: value for key, value in record.items() if value not in (None, "")}


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
