"""Export a point-in-time Quant-Fleet replay as a browser-safe JSON artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl
import torch

from quant_warehouse import Warehouse


FAMILIES = {
    "price_action": ("PRICE ACTION", "#4de1ff", ("return_5d", "range_pct", "log_volume")),
    "fundamentals": ("FUNDAMENTALS", "#8df58d", ("net_margin", "eps", "revenue_growth")),
    "ratios": ("RATIOS", "#f4b860", ("current_ratio", "debt_to_equity", "roe")),
    "market_scale": ("MARKET SCALE", "#c89bff", ("log_market_cap", "close", "vwap")),
    "targets": ("TARGETS", "#ff6b7a", ("forward_return_20d", "event_score", "event_intensity")),
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _numeric(frame: pl.DataFrame, name: str) -> pl.Series:
    if name not in frame.columns:
        return pl.Series(name, [None] * frame.height, dtype=pl.Float64)
    return frame[name].cast(pl.Float64, strict=False).fill_nan(None).fill_null(None)


def _column(frame: pl.DataFrame, *names: str) -> pl.Series:
    for name in names:
        series = _numeric(frame, name)
        if series.len() and series.is_not_null().any():
            return series
    return pl.Series("value", [None] * frame.height, dtype=pl.Float64)


def _dates(frame: pl.DataFrame) -> pl.Series:
    if "date" in frame.columns:
        expr = pl.col("date")
    elif "period_ending" in frame.columns:
        expr = pl.col("period_ending")
    else:
        return pl.Series("date", [], dtype=pl.Datetime)
    if frame.schema[expr.meta.output_name()] == pl.String:
        expr = expr.str.to_datetime(strict=False)
    else:
        expr = expr.cast(pl.Datetime, strict=False)
    return frame.select(expr.dt.replace_time_zone(None).alias("date"))["date"]


def _align(frame: pl.DataFrame, dates: pl.Series) -> pl.DataFrame:
    base = pl.DataFrame({"date": dates})
    if frame is None or frame.is_empty():
        return base
    source = frame.with_columns(_dates(frame)).sort("date")
    return base.join_asof(source, on="date", strategy="backward")


def _mad_scale(series: pl.Series) -> tuple[float, float]:
    values = torch.tensor(series.drop_nulls().to_list(), dtype=torch.float64, device=DEVICE)
    if values.numel() == 0:
        return 0.0, 1.0
    center = float(torch.nanmedian(values))
    mad = float(torch.nanmedian(torch.abs(values - center)))
    if not math.isfinite(mad) or mad < 1e-9:
        mad = float(torch.nanstd(values)) or 1.0
    return center, mad


def _project(value: object, center: float, scale: float) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(float(torch.tanh(torch.tensor((value - center) / (scale * 3.0), device=DEVICE))), 6)


def _feature_frames(warehouse: Warehouse, symbol: str, start: str, end: str) -> dict[str, pl.DataFrame]:
    prices = warehouse.read_prices(symbol, provider="fmp", start=start, end=end)
    if prices.is_empty():
        return {family: pl.DataFrame() for family in FAMILIES}
    dates = _dates(prices)
    close, volume = _numeric(prices, "close"), _numeric(prices, "volume")
    price_features = pl.DataFrame({"date": dates, "return_5d": close.pct_change(5), "range_pct": (_numeric(prices, "high") - _numeric(prices, "low")) / close, "log_volume": volume.log1p(), "close": close, "vwap": _column(prices, "fmp__vwap", "close")})
    income = _align(warehouse.read_fundamentals(symbol, section="income", provider="fmp", start=start, end=end), dates)
    growth = _align(warehouse.read_fundamentals(symbol, section="income_growth", provider="fmp", start=start, end=end), dates)
    net_income, revenue = _column(income, "bottom_line_net_income", "net_income"), _column(income, "revenue", "revenue_total")
    fundamentals = pl.DataFrame({"date": dates, "net_margin": net_income / revenue, "eps": _column(income, "eps", "eps_diluted"), "revenue_growth": _column(growth, "growth_revenue", "revenue_growth")})
    ratios_frame = _align(warehouse.read_fundamentals(symbol, section="ratios", provider="fmp", start=start, end=end), dates)
    ratios = pl.DataFrame({"date": dates, "current_ratio": _column(ratios_frame, "current_ratio"), "debt_to_equity": _column(ratios_frame, "debt_to_equity"), "roe": _column(ratios_frame, "return_on_equity", "roe")})
    market_cap = _align(warehouse.read_fundamentals(symbol, section="historical_market_cap", provider="fmp", start=start, end=end), dates)
    market_scale = pl.DataFrame({"date": dates, "log_market_cap": _column(market_cap, "market_cap").log1p(), "close": close, "vwap": price_features["vwap"]})
    targets = pl.DataFrame({"date": dates, "forward_return_20d": close.shift(-20) / close - 1.0, "event_score": pl.Series("event_score", [0.0] * len(dates)), "event_intensity": pl.Series("event_intensity", [0.0] * len(dates))})
    return {"price_action": price_features, "fundamentals": fundamentals, "ratios": ratios, "market_scale": market_scale, "targets": targets}


def build_replay(symbol: str, start: str, end: str) -> dict[str, object]:
    frames = _feature_frames(Warehouse(), symbol, start, end)
    dates = frames["price_action"]["date"] if not frames["price_action"].is_empty() else pl.Series("date", [], dtype=pl.Datetime)
    units_by_family: dict[str, dict[str, object]] = {}
    for family_id, (label, color, axes) in FAMILIES.items():
        frame = frames[family_id]
        units_by_family[family_id] = {"label": label, "color": color, "axes": list(axes), "projection": {axis: {"center": _mad_scale(_numeric(frame, axis))[0], "scale": _mad_scale(_numeric(frame, axis))[1]} for axis in axes}}
    snapshots = []
    last_raw = {family_id: None for family_id in FAMILIES}
    last_updated = {family_id: None for family_id in FAMILIES}
    for timestamp in dates:
        date_key = timestamp.strftime("%Y-%m-%d")
        units = []
        for family_id, (_, _, axes) in FAMILIES.items():
            frame = frames[family_id]
            row = frame.filter(pl.col("date") == timestamp)
            raw_values, values = [], []
            for axis in axes:
                value = row.item(0, axis) if row.height and axis in row.columns else None
                projection = units_by_family[family_id]["projection"][axis]
                raw_values.append(None if value is None else round(float(value), 8))
                values.append(_project(value, projection["center"], projection["scale"]))
            updated = any(value is not None for value in raw_values) and last_raw[family_id] != raw_values
            if updated:
                last_raw[family_id], last_updated[family_id] = raw_values.copy(), date_key
            units.append({"id": family_id, "family": family_id, "label": units_by_family[family_id]["label"], "color": units_by_family[family_id]["color"], "axes": list(axes), "values": values, "raw_values": raw_values, "updated": updated, "last_updated": last_updated[family_id]})
        snapshots.append({"date": date_key, "units": units})
    return {"schema_version": 1, "dataset_id": f"quant-fleet:{symbol}:{start}:{end}", "symbol": symbol, "provider": "fmp", "start_date": start, "end_date": end, "coordinate_semantics": {"x": "feature axis 1", "y": "feature axis 2 / elevation", "z": "feature axis 3", "time": "date snapshot"}, "families": units_by_family, "snapshots": snapshots}


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
