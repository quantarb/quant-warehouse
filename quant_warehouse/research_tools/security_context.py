from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

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
    symbols: Sequence[str],
    *,
    spec: SecurityContextSpec | None = None,
    warehouse: Warehouse | None = None,
) -> pd.DataFrame:
    """Build attribution dimensions without making them model features implicitly."""

    config = spec or SecurityContextSpec()
    store = warehouse or Warehouse()
    parts = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            continue
        prices = store.read_prices(
            symbol,
            provider=config.provider,
            start=config.start_date,
            end=config.end_date,
        )
        if prices is None or prices.empty:
            continue
        part = _symbol_context(store, symbol, prices, config)
        if not part.empty:
            parts.append(part)
    if not parts:
        return _empty_context_panel()
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ["date", "symbol"], kind="stable"
    ).reset_index(drop=True)


def _symbol_context(
    warehouse: Warehouse,
    symbol: str,
    prices: pd.DataFrame,
    spec: SecurityContextSpec,
) -> pd.DataFrame:
    work = prices.copy()
    dates = pd.to_datetime(work.index, errors="coerce").normalize()
    valid = dates.notna()
    work = work.loc[valid].copy()
    work.index = dates[valid]
    work = work.loc[~work.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame({"symbol": symbol, "date": work.index})
    close = pd.to_numeric(work.get("close"), errors="coerce")
    volume = pd.to_numeric(work.get("volume"), errors="coerce")
    returns = close.pct_change(fill_method=None)
    out["realized_volatility"] = returns.rolling(
        int(spec.volatility_window), min_periods=max(2, int(spec.volatility_window) // 2)
    ).std(ddof=1).mul(np.sqrt(252.0)).to_numpy()
    dollar_volume = close.mul(volume)
    out["adv_dollars"] = dollar_volume.rolling(
        int(spec.liquidity_window), min_periods=max(1, int(spec.liquidity_window) // 2)
    ).mean().to_numpy()

    market_cap = _historical_market_cap(warehouse, symbol, out["date"], spec.provider)
    out["market_cap"] = market_cap.to_numpy()
    out["market_cap_bucket"] = _bucket(
        out["market_cap"], spec.market_cap_edges, spec.market_cap_labels
    )
    out["liquidity_bucket"] = _bucket(
        out["adv_dollars"], spec.liquidity_edges, spec.liquidity_labels
    )
    out["volatility_regime"] = _bucket(
        out["realized_volatility"], spec.volatility_edges, spec.volatility_labels
    )
    _add_calendar_dimensions(out)
    _add_latest_profile_dimensions(out, warehouse, symbol, spec.provider)
    out["context_recipe_id"] = "security_context_v1"
    return out


def _historical_market_cap(
    warehouse: Warehouse,
    symbol: str,
    dates: pd.Series,
    provider: str,
) -> pd.Series:
    try:
        frame = warehouse.read_fundamentals(
            symbol,
            section="historical_market_cap",
            provider=provider,
            start=dates.min().strftime("%Y-%m-%d"),
            end=dates.max().strftime("%Y-%m-%d"),
        )
    except (KeyError, TypeError, ValueError):
        frame = pd.DataFrame()
    if frame is None or frame.empty or "market_cap" not in frame.columns:
        return pd.Series(np.nan, index=dates.index, dtype=float)
    values = pd.to_numeric(frame["market_cap"], errors="coerce")
    values.index = pd.to_datetime(frame.index, errors="coerce").normalize()
    values = values.loc[values.index.notna()].sort_index()
    target = pd.DatetimeIndex(dates)
    return pd.Series(values.reindex(target, method="ffill").to_numpy(), index=dates.index)


def _add_latest_profile_dimensions(
    frame: pd.DataFrame,
    warehouse: Warehouse,
    symbol: str,
    provider: str,
) -> None:
    try:
        profile = warehouse.read_profile(symbol, provider=provider)
    except (KeyError, TypeError, ValueError):
        profile = None
    frame["sector"] = _profile_value(profile, "sector")
    frame["industry"] = _profile_value(profile, "industry")
    frame["exchange"] = _profile_value(profile, "exchange")
    frame["country"] = _profile_value(profile, "country")
    frame["classification_observed_at"] = _profile_value(profile, "fetched_at")
    frame["classification_temporality"] = "latest_known_applied_historically"


def _profile_value(profile, name: str) -> str | None:
    value = None if profile is None else getattr(profile, name, None)
    text = str(value or "").strip()
    return text or None


def _add_calendar_dimensions(frame: pd.DataFrame) -> None:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    frame["calendar_year"] = dates.dt.year.astype("Int64")
    frame["calendar_quarter"] = dates.dt.quarter.astype("Int64")
    frame["calendar_month"] = dates.dt.month.astype("Int64")
    frame["day_of_week"] = dates.dt.dayofweek.astype("Int64")
    frame["is_month_end"] = dates.dt.is_month_end
    frame["is_quarter_end"] = dates.dt.is_quarter_end
    frame["is_year_end"] = dates.dt.is_year_end


def _bucket(values: pd.Series, edges: tuple[float, ...], labels: tuple[str, ...]) -> pd.Series:
    if len(labels) != len(edges) + 1:
        raise ValueError("bucket labels must contain exactly len(edges) + 1 values")
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, *[float(edge) for edge in edges], np.inf],
        labels=list(labels),
        right=False,
    ).astype("string")


def _empty_context_panel() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "date",
            "sector",
            "industry",
            "calendar_year",
            "market_cap",
            "market_cap_bucket",
            "adv_dollars",
            "liquidity_bucket",
            "realized_volatility",
            "volatility_regime",
            "classification_observed_at",
            "classification_temporality",
            "context_recipe_id",
        ]
    )
