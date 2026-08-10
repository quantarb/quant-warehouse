from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import polars as pl

from quant_warehouse.warehouse.api import Warehouse


@dataclass(frozen=True)
class SecurityContextSpec:
    provider: str = "fmp"
    start_date: str | None = None
    end_date: str | None = None
    volatility_window: int = 20
    liquidity_window: int = 20
    market_cap_edges: tuple[float, ...] = (2e9, 10e9, 50e9, 200e9)
    market_cap_labels: tuple[str, ...] = ("micro", "small", "mid", "large", "mega")
    liquidity_edges: tuple[float, ...] = (1e6, 10e6, 50e6, 250e6)
    liquidity_labels: tuple[str, ...] = ("illiquid", "low", "medium", "high", "very_high")
    volatility_edges: tuple[float, ...] = (0.15, 0.30, 0.50)
    volatility_labels: tuple[str, ...] = ("low", "normal", "high", "extreme")


def build_security_context_panel(
    symbols: Sequence[str], *, spec: SecurityContextSpec | None = None,
    warehouse: Warehouse | None = None,
) -> pl.DataFrame:
    config = spec or SecurityContextSpec()
    store = warehouse or Warehouse()
    parts: list[pl.DataFrame] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            continue
        prices = store.read_prices(symbol, provider=config.provider, start=config.start_date,
                                   end=config.end_date, output_format="polars")
        if prices is not None and not prices.is_empty():
            part = _symbol_context(store, symbol, prices, config)
            if not part.is_empty():
                parts.append(part)
    return pl.concat(parts, how="diagonal_relaxed").sort(["date", "symbol"]) if parts else _empty_context_panel()


def _symbol_context(warehouse: Warehouse, symbol: str, prices: pl.DataFrame,
                    spec: SecurityContextSpec) -> pl.DataFrame:
    if "date" not in prices.columns or "close" not in prices.columns:
        return pl.DataFrame()
    date_expr = (pl.col("date").str.to_datetime(strict=False) if prices.schema["date"] == pl.String
                 else pl.col("date").cast(pl.Datetime, strict=False))
    work = (prices.with_columns(
        date_expr.dt.truncate("1d"),
        pl.col("close").cast(pl.Float64, strict=False),
        (pl.col("volume").cast(pl.Float64, strict=False) if "volume" in prices.columns
         else pl.lit(None, dtype=pl.Float64).alias("volume")),
    ).drop_nulls("date").sort("date").unique("date", keep="last"))
    out = work.select(["date", "close", "volume"]).with_columns(
        pl.lit(symbol).alias("symbol"),
        (pl.col("close").pct_change().rolling_std(window_size=spec.volatility_window,
            min_periods=max(2, spec.volatility_window // 2)) * sqrt(252.0)).alias("realized_volatility"),
        (pl.col("close") * pl.col("volume")).rolling_mean(window_size=spec.liquidity_window,
            min_periods=max(1, spec.liquidity_window // 2)).alias("adv_dollars"),
    )
    try:
        mcap = warehouse.read_fundamentals(symbol, section="historical_market_cap", provider=spec.provider,
                                            start=spec.start_date, end=spec.end_date, output_format="polars")
    except (KeyError, TypeError, ValueError):
        mcap = pl.DataFrame()
    if not mcap.is_empty() and "market_cap" in mcap.columns and "date" in mcap.columns:
        mcap_date_expr = (pl.col("date").str.to_datetime(strict=False) if mcap.schema["date"] == pl.String
                          else pl.col("date").cast(pl.Datetime, strict=False))
        mcap = (mcap.select(mcap_date_expr.dt.truncate("1d"),
                            pl.col("market_cap").cast(pl.Float64, strict=False))
                .drop_nulls("date").sort("date").unique("date", keep="last"))
        out = out.sort("date").join_asof(mcap, on="date", strategy="backward")
    if "market_cap" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("market_cap"))
    try:
        profile = warehouse.read_profile(symbol, provider=spec.provider)
    except (KeyError, TypeError, ValueError):
        profile = None
    return out.with_columns(
        _bucket_expr("market_cap", spec.market_cap_edges, spec.market_cap_labels).alias("market_cap_bucket"),
        _bucket_expr("adv_dollars", spec.liquidity_edges, spec.liquidity_labels).alias("liquidity_bucket"),
        _bucket_expr("realized_volatility", spec.volatility_edges, spec.volatility_labels).alias("volatility_regime"),
        pl.col("date").dt.year().alias("calendar_year"), pl.col("date").dt.quarter().alias("calendar_quarter"),
        pl.col("date").dt.month().alias("calendar_month"), pl.col("date").dt.weekday().alias("day_of_week"),
        (pl.col("date").dt.month_end() == pl.col("date")).alias("is_month_end"),
        ((pl.col("date").dt.month().is_in([3, 6, 9, 12])) & (pl.col("date").dt.month_end() == pl.col("date"))).alias("is_quarter_end"),
        ((pl.col("date").dt.month() == 12) & (pl.col("date").dt.month_end() == pl.col("date"))).alias("is_year_end"),
        pl.lit(_profile_value(profile, "sector")).alias("sector"), pl.lit(_profile_value(profile, "industry")).alias("industry"),
        pl.lit(_profile_value(profile, "exchange")).alias("exchange"), pl.lit(_profile_value(profile, "country")).alias("country"),
        pl.lit(_profile_value(profile, "fetched_at")).alias("classification_observed_at"),
        pl.lit("latest_known_applied_historically").alias("classification_temporality"),
        pl.lit("security_context_v1").alias("context_recipe_id"),
    ).drop(["close", "volume"])


def _bucket_expr(column: str, edges: tuple[float, ...], labels: tuple[str, ...]) -> pl.Expr:
    if len(labels) != len(edges) + 1:
        raise ValueError("bucket labels must contain exactly len(edges) + 1 values")
    expr: pl.Expr = pl.lit(labels[-1])
    for edge, label in reversed(list(zip(edges, labels[:-1]))):
        expr = pl.when(pl.col(column) < edge).then(pl.lit(label)).otherwise(expr)
    return pl.when(pl.col(column).is_null()).then(pl.lit(None, dtype=pl.String)).otherwise(expr)


def _profile_value(profile, name: str) -> str | None:
    value = None if profile is None else getattr(profile, name, None)
    text = str(value or "").strip()
    return text or None


def _empty_context_panel() -> pl.DataFrame:
    return pl.DataFrame(schema={"symbol": pl.String, "date": pl.Datetime, "sector": pl.String,
        "industry": pl.String, "calendar_year": pl.Int32, "market_cap": pl.Float64,
        "market_cap_bucket": pl.String, "adv_dollars": pl.Float64, "liquidity_bucket": pl.String,
        "realized_volatility": pl.Float64, "volatility_regime": pl.String,
        "classification_observed_at": pl.String, "classification_temporality": pl.String,
        "context_recipe_id": pl.String})
