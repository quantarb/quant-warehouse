"""Export a point-in-time Quant-Fleet replay for Fleetcraft.

The warehouse remains the source of truth. This produces a small, browser-safe
JSON artifact so Godot never needs to import Python, pandas, or ArcticDB.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quant_warehouse import Warehouse


FAMILIES = {
    "price_action": ("PRICE ACTION", "#4de1ff", ("return_5d", "range_pct", "log_volume")),
    "fundamentals": ("FUNDAMENTALS", "#8df58d", ("net_margin", "eps", "revenue_growth")),
    "ratios": ("RATIOS", "#f4b860", ("current_ratio", "debt_to_equity", "roe")),
    "market_scale": ("MARKET SCALE", "#c89bff", ("log_market_cap", "close", "vwap")),
    "targets": ("TARGETS", "#ff6b7a", ("forward_return_20d", "event_score", "event_intensity")),
}


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        series = _numeric(frame, name)
        if not series.empty and series.notna().any():
            return series
    return pd.Series(index=frame.index, dtype=float)


def _align(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=dates)
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out = out[~out.index.isna()]
    return out.reindex(dates, method="ffill")


def _mad_scale(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    if not math.isfinite(mad) or mad < 1e-9:
        mad = float(np.nanstd(values)) or 1.0
    return center, mad


def _project(series: pd.Series, center: float, scale: float) -> list[float | None]:
    projected: list[float | None] = []
    for value in series:
        if pd.isna(value):
            projected.append(None)
        else:
            projected.append(round(float(np.tanh((float(value) - center) / (scale * 3.0))), 6))
    return projected


def _feature_frames(warehouse: Warehouse, symbol: str, start: str, end: str) -> dict[str, pd.DataFrame]:
    prices = warehouse.read_prices(symbol, provider="fmp", start=start, end=end)
    dates = pd.DatetimeIndex(prices.index).tz_localize(None)
    close = _numeric(prices, "close")
    volume = _numeric(prices, "volume")
    price_features = pd.DataFrame(index=dates)
    price_features["return_5d"] = close.pct_change(5)
    price_features["range_pct"] = (_numeric(prices, "high") - _numeric(prices, "low")) / close
    price_features["log_volume"] = np.log1p(volume)
    price_features["close"] = close
    price_features["vwap"] = _column(prices, "fmp__vwap", "close")

    income = _align(warehouse.read_fundamentals(symbol, section="income", provider="fmp", start=start, end=end), dates)
    growth = _align(warehouse.read_fundamentals(symbol, section="income_growth", provider="fmp", start=start, end=end), dates)
    fundamentals = pd.DataFrame(index=dates)
    net_income = _column(income, "bottom_line_net_income", "net_income")
    revenue = _column(income, "revenue", "revenue_total")
    fundamentals["net_margin"] = net_income / revenue.replace(0, np.nan)
    fundamentals["eps"] = _column(income, "eps", "eps_diluted")
    fundamentals["revenue_growth"] = _column(growth, "growth_revenue", "revenue_growth")

    ratios_frame = _align(warehouse.read_fundamentals(symbol, section="ratios", provider="fmp", start=start, end=end), dates)
    ratios = pd.DataFrame(index=dates)
    ratios["current_ratio"] = _column(ratios_frame, "current_ratio")
    ratios["debt_to_equity"] = _column(ratios_frame, "debt_to_equity")
    ratios["roe"] = _column(ratios_frame, "return_on_equity", "roe")

    market_cap = _align(warehouse.read_fundamentals(symbol, section="historical_market_cap", provider="fmp", start=start, end=end), dates)
    market_scale = pd.DataFrame(index=dates)
    market_scale["log_market_cap"] = np.log1p(_column(market_cap, "market_cap"))
    market_scale["close"] = close
    market_scale["vwap"] = price_features["vwap"]

    # Event-pair targets are provider-owned storage, but intentionally are not
    # exposed as ordinary fundamentals by the warehouse API.
    events = warehouse.backend.read("fmp_target_event_pairs", f"{symbol}__fmp")
    if events is None:
        events = pd.DataFrame()
    else:
        events = events.loc[
            (pd.to_datetime(events.index) >= pd.Timestamp(start))
            & (pd.to_datetime(events.index) <= pd.Timestamp(end))
        ]
    event_frame = _align(events, dates)
    event_score = pd.Series(0.0, index=dates)
    if not event_frame.empty:
        side = event_frame.get("event_side", pd.Series(index=dates, dtype=object)).astype(str).str.lower()
        event_score = side.map(lambda value: 1.0 if value in {"buy", "upgrade", "beat", "raise", "positive"} else -1.0 if value else 0.0).fillna(0.0)
    targets = pd.DataFrame(index=dates)
    targets["forward_return_20d"] = close.shift(-20) / close - 1.0
    targets["event_score"] = event_score
    targets["event_intensity"] = event_score.abs()
    return {
        "price_action": price_features,
        "fundamentals": fundamentals,
        "ratios": ratios,
        "market_scale": market_scale,
        "targets": targets,
    }


def build_replay(symbol: str, start: str, end: str) -> dict[str, object]:
    warehouse = Warehouse()
    frames = _feature_frames(warehouse, symbol, start, end)
    dates = next(iter(frames.values())).index
    units_by_family: dict[str, dict[str, object]] = {}
    for family_id, (label, color, axes) in FAMILIES.items():
        frame = frames[family_id]
        units_by_family[family_id] = {
            "label": label,
            "color": color,
            "axes": list(axes),
            "projection": {
                axis: {"center": _mad_scale(_numeric(frame, axis))[0], "scale": _mad_scale(_numeric(frame, axis))[1]}
                for axis in axes
            },
        }

    snapshots = []
    last_raw_by_family: dict[str, list[float | None] | None] = {family_id: None for family_id in FAMILIES}
    last_updated_by_family: dict[str, str | None] = {family_id: None for family_id in FAMILIES}
    for date in dates:
        date_key = date.strftime("%Y-%m-%d")
        units = []
        for family_id, (_, _, axes) in FAMILIES.items():
            frame = frames[family_id]
            values = []
            raw_values = []
            for axis in axes:
                center = units_by_family[family_id]["projection"][axis]["center"]
                scale = units_by_family[family_id]["projection"][axis]["scale"]
                value = frame.loc[date, axis] if date in frame.index else np.nan
                raw_values.append(None if pd.isna(value) else round(float(value), 8))
                values.append(_project(pd.Series([value]), center, scale)[0])
            has_observation = any(value is not None for value in raw_values)
            if has_observation and last_raw_by_family[family_id] != raw_values:
                last_raw_by_family[family_id] = raw_values.copy()
                last_updated_by_family[family_id] = date_key
                updated = True
            else:
                updated = False
            units.append({"id": family_id, "family": family_id, "label": units_by_family[family_id]["label"], "color": units_by_family[family_id]["color"], "axes": list(axes), "values": values, "raw_values": raw_values, "updated": updated, "last_updated": last_updated_by_family[family_id]})
        snapshots.append({"date": date_key, "units": units})

    return {
        "schema_version": 1,
        "dataset_id": f"quant-fleet:{symbol}:{start}:{end}",
        "symbol": symbol,
        "provider": "fmp",
        "start_date": start,
        "end_date": end,
        "coordinate_semantics": {"x": "feature axis 1", "y": "feature axis 2 / elevation", "z": "feature axis 3", "time": "date snapshot"},
        "families": units_by_family,
        "snapshots": snapshots,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2021-01-31")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_replay(args.symbol.upper(), args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.output} ({len(payload['snapshots'])} snapshots)")


if __name__ == "__main__":
    main()
