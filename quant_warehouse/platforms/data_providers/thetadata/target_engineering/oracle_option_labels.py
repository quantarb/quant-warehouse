from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from time import perf_counter
from typing import Any, Callable

import polars as pl

from quant_warehouse.platforms.data_providers.thetadata.feature_engineering.option_features import (
    build_option_contract_features,
    option_ranker_feature_columns,
)
from quant_warehouse.platforms.data_providers.thetadata.options import (
    load_thetadata_option_snapshots,
    normalize_thetadata_option_chain,
)
from quant_warehouse.platforms.data_providers.thetadata.target_engineering.option_labels import (
    OptionLabelSpec,
    build_option_labels,
)

ProgressLogger = Callable[[str], None] | None
ORACLE_OPTION_LABEL_POLICY = "oracle_exit_survivors_expiration_early_fallback_v1"


@dataclass(frozen=True)
class OracleOptionLabelPanelSpec:
    max_trades: int = 0
    max_dte: int | None = None
    target_dte: int = 90
    max_candidates_per_trade: int = 128
    progress_every: int = 50


@dataclass(frozen=True)
class OracleOptionLabelPanelResult:
    panel: pl.DataFrame = field(default_factory=pl.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)


def _day(value: Any) -> datetime | None:
    if value is None: return None
    if isinstance(value, datetime): return value.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    if isinstance(value, date): return datetime.combine(value, datetime.min.time())
    try: return datetime.fromisoformat(str(value)[:10])
    except ValueError: return None


def _side(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"long", "buy", "bull", "bullish", "oracle_long", "oracle_buy"}: return "long"
    if text in {"short", "sell", "bear", "bearish", "oracle_short", "oracle_sell"}: return "short"
    return None


def build_oracle_option_label_panel(
    trades: pl.DataFrame,
    *,
    spec: OracleOptionLabelPanelSpec | None = None,
    progress_logger: ProgressLogger = None,
) -> OracleOptionLabelPanelResult:
    """Build cache-only Polars option labels from oracle trade endpoints."""
    config = spec or OracleOptionLabelPanelSpec()
    if trades is None or trades.is_empty(): return OracleOptionLabelPanelResult(summary={"status": "empty_trades"})
    required = {"symbol", "side", "entry_date", "exit_date"}
    missing = required.difference(trades.columns)
    if missing: raise KeyError(f"oracle trades missing required columns: {sorted(missing)}")
    work = trades.with_columns([
        pl.col("symbol").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("side").map_elements(_side, return_dtype=pl.String),
        pl.col("entry_date").map_elements(_day, return_dtype=pl.Datetime),
        pl.col("exit_date").map_elements(_day, return_dtype=pl.Datetime),
    ]).drop_nulls(["symbol", "side", "entry_date", "exit_date"]).filter(pl.col("symbol") != "").filter(pl.col("exit_date") > pl.col("entry_date"))
    if config.max_trades > 0: work = work.head(config.max_trades)
    summary = _summary(work, config); started = perf_counter(); parts: list[pl.DataFrame] = []
    for symbol in work["symbol"].unique(maintain_order=True).to_list():
        symbol_trades = work.filter(pl.col("symbol") == symbol)
        dates = sorted(set(symbol_trades["entry_date"].to_list() + symbol_trades["exit_date"].to_list()))
        snapshots = _normalized_cached_snapshots(symbol, dates)
        for trade in symbol_trades.to_dicts():
            entry_date, exit_date = trade["entry_date"], trade["exit_date"]
            entry = snapshots.get(entry_date); exit_frame = snapshots.get(exit_date)
            side = str(trade["side"]); option_type = "call" if side == "long" else "put"
            if entry is None or entry.is_empty():
                _missing(summary, trade); continue
            candidates = _filter_entry_candidates(entry, option_type, entry_date, config.max_dte)
            if candidates.is_empty(): summary["trades_skipped_empty_intersection"] += 1; continue
            if exit_frame is None or exit_frame.is_empty(): _missing(summary, trade); continue
            result = build_option_labels([trade], {entry_date: candidates, exit_date: exit_frame}, spec=OptionLabelSpec(entry_quote_col="ask", exit_quote_col="bid", price_fallback_cols=(), worthless_exit_threshold=-1.0))
            if not result.option_rows: summary["trades_skipped_empty_intersection"] += 1; continue
            panel = pl.DataFrame(result.option_rows).with_columns([
                pl.lit(ORACLE_OPTION_LABEL_POLICY).alias("label_policy"),
                pl.lit("oracle_exit").alias("return_horizon"),
                pl.lit(symbol).alias("symbol"),
                pl.lit(side).alias("side"),
                pl.lit(option_type).alias("option_type"),
            ])
            feature_frame = build_option_contract_features(candidates, underlying_price=_median(candidates, "underlying_price"), target_dte=config.target_dte).df
            if isinstance(feature_frame, pl.DataFrame) and not feature_frame.is_empty() and "contract_symbol" in feature_frame.columns:
                cols = ["contract_symbol", *option_ranker_feature_columns(feature_frame)]
                panel = panel.join(feature_frame.select([c for c in cols if c in feature_frame.columns]).unique("contract_symbol"), on="contract_symbol", how="left")
            panel = panel.sort(["rank_y", "option_return_pct"], descending=[True, True]).head(config.max_candidates_per_trade)
            parts.append(panel); summary["trades_labeled"] += 1; summary["option_rows"] += panel.height; summary["labeled_trade_ids"].append(trade.get("trade_id"))
            if callable(progress_logger) and config.progress_every and summary["trades_labeled"] % config.progress_every == 0: progress_logger(f"[option-labels] labeled={summary['trades_labeled']}")
    summary["elapsed_seconds"] = round(perf_counter() - started, 3)
    if not parts: summary["status"] = "no_option_rows"; return OracleOptionLabelPanelResult(summary=summary)
    panel = pl.concat(parts, how="diagonal_relaxed"); summary.update({"status": "ok", "symbols": panel["symbol"].n_unique(), "option_rows": panel.height})
    return OracleOptionLabelPanelResult(panel=panel, summary=summary)


def _normalized_cached_snapshots(symbol: str, dates: list[datetime]) -> dict[datetime, pl.DataFrame]:
    raw = load_thetadata_option_snapshots(symbol, dates, use_cache=True, download_missing=False)
    return {day: normalize_thetadata_option_chain(frame) for day, frame in raw.items() if frame is not None and not frame.is_empty()}


def _filter_entry_candidates(frame: pl.DataFrame, option_type: str, entry_date: datetime, max_dte: int | None) -> pl.DataFrame:
    if frame.is_empty() or not {"option_type", "bid", "ask", "expiration"}.issubset(frame.columns): return pl.DataFrame()
    expiration_expr = (pl.col("expiration").str.to_datetime(strict=False)
                       if frame.schema["expiration"] == pl.String
                       else pl.col("expiration").cast(pl.Datetime, strict=False))
    out = frame.with_columns([
        pl.col("option_type").cast(pl.String).str.to_lowercase().alias("option_type"),
        pl.col("bid").cast(pl.Float64, strict=False), pl.col("ask").cast(pl.Float64, strict=False),
        expiration_expr.alias("expiration"),
    ]).filter(pl.col("option_type").str.starts_with(option_type[0])).filter((pl.col("bid") >= 0) & (pl.col("ask") > 0) & (pl.col("ask") >= pl.col("bid")))
    out = out.with_columns((pl.col("expiration") - pl.lit(entry_date)).dt.total_days().alias("dte")).filter(pl.col("dte") > 0)
    if max_dte is not None and max_dte > 0: out = out.filter(pl.col("dte") <= max_dte)
    return out


def _median(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns: return None
    value = frame.select(pl.col(column).cast(pl.Float64, strict=False).median()).item()
    return float(value) if value is not None else None


def _summary(work: pl.DataFrame, spec: OracleOptionLabelPanelSpec) -> dict[str, Any]:
    return {"status": "ok", "trades_requested": work.height, "trades_labeled": 0, "trades_skipped_missing_historical_options": 0, "trades_skipped_empty_intersection": 0, "option_rows": 0, "symbols": 0, "labeled_trade_ids": [], "skipped_missing_options": [], "max_dte": spec.max_dte, "max_candidates_per_trade": spec.max_candidates_per_trade}


def _missing(summary: dict[str, Any], trade: dict[str, Any]) -> None:
    summary["trades_skipped_missing_historical_options"] += 1
    summary["skipped_missing_options"].append({"trade_id": trade.get("trade_id"), "symbol": trade.get("symbol"), "skip_reason": "missing_historical_option_data"})
