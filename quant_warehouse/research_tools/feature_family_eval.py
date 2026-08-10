from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import polars as pl

from quant_warehouse.ingest.screener_fetch import ScreenerQuery, fetch_equity_screener
from quant_warehouse.warehouse.api import Warehouse


@dataclass(frozen=True)
class FamilyEvaluationConfig:
    provider: str = "fmp"
    market_cap_min: int = 1_000_000_000_000
    country: str = "US"
    exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")
    screen_limit: int = 5_000
    start_date: str = "2018-01-01"
    end_date: str | None = None
    filing_lag_days: int = 0
    horizons: tuple[int, ...] = (20, 60, 120)
    min_observations: int = 120
    max_features_per_family: int | None = None


@dataclass(frozen=True)
class FeatureSpec:
    feature: str
    family: str
    source: str
    source_column: str
    expected_direction: str


def _dates(frame: pl.DataFrame) -> pl.DataFrame:
    expr = pl.col("date").str.to_datetime(strict=False) if frame.schema["date"] == pl.String else pl.col("date").cast(pl.Datetime, strict=False)
    return frame.with_columns(expr.dt.truncate("1d").alias("date"))


def _records(frame: object) -> list[dict]:
    return frame.to_dicts() if isinstance(frame, pl.DataFrame) else [dict(row) for row in frame or []]


def screen_fmp_equity_universe(config: FamilyEvaluationConfig, *, warehouse: Warehouse | None = None, required_sections: Iterable[str] = ()) -> tuple[tuple[str, ...], pl.DataFrame, pl.DataFrame, str]:
    frame, source = fetch_equity_screener(ScreenerQuery(provider=config.provider, mktcap_min=config.market_cap_min, country=config.country, exchanges=config.exchanges, is_etf=False, is_fund=False, is_active=True, all_share_classes=False, limit=config.screen_limit))
    frame = frame if isinstance(frame, pl.DataFrame) else pl.DataFrame(frame)
    if frame.is_empty(): raise RuntimeError("OpenBB/FMP screener returned no symbols for the configured filters.")
    frame = frame.with_columns(pl.col("symbol").cast(pl.String).str.strip_chars().str.to_uppercase())
    if "market_cap" in frame.columns: frame = frame.filter(pl.col("market_cap").cast(pl.Float64, strict=False) >= config.market_cap_min)
    frame = frame.unique("symbol")
    eligibility = frame.select(["symbol"]).with_columns(pl.lit(True).alias("eligible"), pl.lit("ok").alias("reason"))
    symbols = tuple(eligibility.filter(pl.col("eligible"))["symbol"].to_list())
    return symbols, frame, eligibility, source


def _is_supported_equity_record(symbol: str, record: dict[str, object]) -> tuple[bool, str]:
    if _truthy(record.get("is_etf")) or _clean(record.get("quote_type")) == "etf": return False, "asset_class: etf"
    if _truthy(record.get("is_fund")) or _clean(record.get("quote_type")) in {"fund", "mutualfund", "mutual_fund"}: return False, "asset_class: fund"
    if len(str(symbol).strip()) == 5 and str(symbol).upper().endswith("X"): return False, "asset_class: fund_symbol_pattern"
    return True, "ok"


def _clean(value: object) -> str: return "" if value is None else str(value).strip().lower().replace(" ", "_").replace("-", "_")
def _truthy(value: object) -> bool: return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def build_fundamental_feature_panel(symbols: Iterable[str], config: FamilyEvaluationConfig, *, warehouse: Warehouse | None = None, strategy_sources: Iterable[str] | None = None, observation_dates: pl.DataFrame | None = None, broadcast_to_target: bool = True, fundamental_period: str | None = None, family_suffix: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, float]]:
    started = perf_counter(); wanted = {str(value).strip() for value in strategy_sources or () if str(value).strip()}; frames: list[pl.DataFrame] = []; specs: list[FeatureSpec] = []; diagnostics: list[dict] = []
    for symbol in symbols:
        dates = None if observation_dates is None else observation_dates.filter(pl.col("symbol") == str(symbol).upper())["date"]
        frame, symbol_specs, diagnostic = _build_symbol_fundamental_panel(warehouse or Warehouse(), str(symbol).upper(), config, strategy_sources=wanted or None, observation_dates=dates, broadcast_to_target=broadcast_to_target, fundamental_period=fundamental_period, family_suffix=family_suffix)
        diagnostics.append(diagnostic); specs.extend(symbol_specs)
        if frame is not None and not frame.is_empty(): frames.append(frame)
    if not frames: raise RuntimeError("No feature frames were built for the requested symbols.")
    panel = pl.concat(frames, how="diagonal_relaxed").sort(["date", "symbol"])
    specs.extend(_add_time_calendar_features(panel)); specs.extend(_add_macro_context_features(warehouse or Warehouse(), panel, config)); specs.extend(_add_cross_symbol_context_features(warehouse or Warehouse(), panel, config))
    if wanted:
        specs = [spec for spec in specs if f"{spec.source}.{spec.family}" in wanted]
        columns = [spec.feature for spec in specs if spec.feature in panel.columns]
        panel = panel.select([c for c in ("symbol", "date", *columns) if c in panel.columns])
    metadata = pl.DataFrame([spec.__dict__ for spec in specs]).unique().sort(["family", "feature"]) if specs else pl.DataFrame()
    return panel, metadata, pl.DataFrame(diagnostics), {"raw_panel_build_seconds": perf_counter() - started}


def build_technical_feature_panel(symbols: Iterable[str], config: FamilyEvaluationConfig, *, strategy_sources: Iterable[str], observation_dates: pl.DataFrame | None = None, warehouse: Warehouse | None = None, max_workers: int = 1) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, float]]:
    frames: list[pl.DataFrame] = []; specs: list[FeatureSpec] = []; diagnostics: list[dict] = []; started = perf_counter()
    for symbol in symbols:
        frame, symbol_specs, diagnostic = _build_symbol_technical_panel(warehouse or Warehouse(), str(symbol).upper(), config, requested=set(strategy_sources), wanted_ta=set(), observation_dates=None)
        if frame is not None and not frame.is_empty(): frames.append(frame); specs.extend(symbol_specs)
        diagnostics.append(diagnostic)
    if not frames: raise RuntimeError("No technical feature frames were built for the requested symbols and dates.")
    return pl.concat(frames, how="diagonal_relaxed").sort(["date", "symbol"]), pl.DataFrame([spec.__dict__ for spec in specs]).unique(), pl.DataFrame(diagnostics), {"technical_panel_build_seconds": perf_counter() - started}


def cap_features_by_quality(panel: pl.DataFrame, metadata: pl.DataFrame, *, max_features: int) -> tuple[list[str], pl.DataFrame, pl.DataFrame]:
    rows: list[dict] = []
    for row in metadata.to_dicts():
        feature = row["feature"]; values = panel[feature].cast(pl.Float64, strict=False) if feature in panel.columns else pl.Series([], dtype=pl.Float64)
        rows.append({**row, "observations": values.drop_nulls().len(), "selected": False})
    quality = pl.DataFrame(rows)
    selected: list[str] = []
    for family in quality["family"].unique().to_list():
        names = quality.filter(pl.col("family") == family).sort("observations", descending=True).head(max_features)["feature"].to_list(); selected.extend(names)
    quality = quality.with_columns(pl.col("feature").is_in(selected).alias("selected"))
    return selected, quality.filter(pl.col("selected")), quality


def evaluate_feature_families(panel: pl.DataFrame, metadata: pl.DataFrame, *, horizons: tuple[int, ...] = (20, 60, 120), min_observations: int = 120, include_spreads: bool = True) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, float]:
    started = perf_counter(); rows: list[dict] = []
    for meta in metadata.to_dicts():
        feature = meta["feature"]
        if feature not in panel.columns: continue
        count = panel[feature].drop_nulls().len()
        for horizon in horizons: rows.append({**meta, "horizon": horizon, "observations": count, "score": 0.0})
    results = pl.DataFrame(rows); summary = results.group_by("family").agg(pl.col("score").mean().alias("score")) if not results.is_empty() else pl.DataFrame(); best = results.sort("score", descending=True).group_by("horizon").first() if not results.is_empty() else pl.DataFrame(); stable = results
    return results, summary, best, stable, perf_counter() - started


def _add_time_calendar_features(panel: pl.DataFrame) -> list[FeatureSpec]:
    return [FeatureSpec("time_calendar__day_of_week", "time_calendar", "derived", "date", "neutral"), FeatureSpec("time_calendar__month", "time_calendar", "derived", "date", "neutral")]


def _add_macro_context_features(warehouse: Warehouse, panel: pl.DataFrame, config: FamilyEvaluationConfig) -> list[FeatureSpec]:
    return [FeatureSpec("economic_indicators__gdp", "economic_indicators", "fmp", "GDP", "neutral"), FeatureSpec("treasury_rates__year10", "treasury_rates", "fmp", "macro__ust_year10", "neutral")]


def _add_cross_symbol_context_features(warehouse: Warehouse, panel: pl.DataFrame, config: FamilyEvaluationConfig) -> list[FeatureSpec]:
    return [FeatureSpec("sector_performance__return", "sector_performance", "derived", "close", "neutral"), FeatureSpec("industry_performance__return", "industry_performance", "derived", "close", "neutral"), FeatureSpec("sector_pe__mcap_to_net_income", "sector_pe", "derived", "mcap_to_net_income", "lower_is_better"), FeatureSpec("industry_pe__mcap_to_net_income", "industry_pe", "derived", "mcap_to_net_income", "lower_is_better")]


def _build_symbol_fundamental_panel(warehouse: Warehouse, symbol: str, config: FamilyEvaluationConfig, *, strategy_sources: set[str] | None = None, observation_dates: pl.Series | None = None, broadcast_to_target: bool = True, fundamental_period: str | None = None, family_suffix: str | None = None) -> tuple[pl.DataFrame, list[FeatureSpec], dict]:
    return pl.DataFrame(), [], {"symbol": symbol, "status": "empty"}


def _build_symbol_technical_panel(warehouse: Warehouse, symbol: str, config: FamilyEvaluationConfig, *, requested: set[str], wanted_ta: set[str], observation_dates: pl.Series | None) -> tuple[pl.DataFrame | None, list[FeatureSpec], dict]:
    return None, [], {"symbol": symbol, "status": "empty"}
