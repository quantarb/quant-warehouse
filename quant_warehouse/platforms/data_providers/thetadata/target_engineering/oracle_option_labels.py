from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class OracleOptionLabelPanelSpec:
    """Configuration for labels built from oracle equity trade endpoints."""

    max_trades: int = 0
    max_dte: int | None = None
    target_dte: int = 90
    progress_every: int = 50


@dataclass(frozen=True)
class OracleOptionLabelPanelResult:
    """Candidate panel plus coverage and skip diagnostics."""

    panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)


def build_oracle_option_label_panel(
    trades: pd.DataFrame,
    *,
    spec: OracleOptionLabelPanelSpec | None = None,
    progress_logger: ProgressLogger = None,
) -> OracleOptionLabelPanelResult:
    """Build cache-only option labels for oracle trade entry and exit dates.

    Candidates are filtered at entry by option side, executable bid/ask, and
    maximum DTE. Returns use entry ask and exit bid for contracts present at
    both endpoints. Missing endpoint chains skip the entire oracle trade.
    """

    config = spec or OracleOptionLabelPanelSpec()
    if trades is None or trades.empty:
        return OracleOptionLabelPanelResult(summary={"status": "empty_trades"})

    work = trades.copy()
    required = {"symbol", "side", "entry_date", "exit_date"}
    missing = required.difference(work.columns)
    if missing:
        raise KeyError(f"oracle trades missing required columns: {sorted(missing)}")
    work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    work["side"] = work["side"].map(_normalize_side)
    work["entry_date"] = pd.to_datetime(work["entry_date"], errors="coerce").dt.normalize()
    work["exit_date"] = pd.to_datetime(work["exit_date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["symbol", "side", "entry_date", "exit_date"])
    work = work.loc[work["symbol"].ne("") & work["exit_date"].gt(work["entry_date"])].copy()
    if config.max_trades > 0:
        work = work.head(int(config.max_trades)).copy()

    label_spec = OptionLabelSpec(
        entry_quote_col="ask",
        exit_quote_col="bid",
        price_fallback_cols=(),
        sort_descending=True,
        label_method="rank",
        worthless_exit_threshold=-1.0,
    )
    summary = _initial_summary(work, config)
    panel_parts: list[pd.DataFrame] = []
    started = perf_counter()

    for symbol, symbol_trades in work.groupby("symbol", sort=False):
        needed_dates = sorted(
            {
                *symbol_trades["entry_date"].dropna().tolist(),
                *symbol_trades["exit_date"].dropna().tolist(),
            }
        )
        snapshots = _normalized_cached_snapshots(symbol, needed_dates)
        for trade in symbol_trades.to_dict("records"):
            side = str(trade["side"])
            option_type = "call" if side == "long" else "put"
            entry_date = pd.Timestamp(trade["entry_date"]).normalize()
            exit_date = pd.Timestamp(trade["exit_date"]).normalize()
            entry_chain = snapshots.get(entry_date)
            exit_chain = snapshots.get(exit_date)
            entry_ok = entry_chain is not None and not entry_chain.empty
            exit_ok = exit_chain is not None and not exit_chain.empty
            if not entry_ok or not exit_ok:
                _record_missing_endpoint(summary, trade, side, entry_date, exit_date, entry_ok, exit_ok)
                continue

            entry_candidates = _filter_entry_candidates(
                entry_chain,
                option_type=option_type,
                entry_date=entry_date,
                exit_date=exit_date,
                max_dte=config.max_dte,
            )
            if entry_candidates.empty:
                summary["trades_skipped_empty_intersection"] += 1
                _append_limited(
                    summary,
                    "skipped_empty_entry_filters",
                    {
                        "trade_id": trade.get("trade_id"),
                        "symbol": symbol,
                        "side": side,
                        "entry_date": entry_date.date().isoformat(),
                        "reason": "no_entry_candidates_covering_oracle_exit",
                    },
                )
                continue

            label_result = build_option_labels(
                [trade],
                {entry_date: entry_candidates, exit_date: exit_chain},
                spec=label_spec,
            )
            label_frame = pd.DataFrame(label_result.option_rows)
            if label_frame.empty:
                summary["trades_skipped_empty_intersection"] += 1
                continue

            featured = build_option_contract_features(
                entry_candidates,
                underlying_price=_median_underlying_price(entry_candidates),
                target_dte=int(config.target_dte),
            ).df
            panel_part = _panel_rows(
                label_frame,
                trade=trade,
                side=side,
                option_type=option_type,
                entry_features=featured,
                target_dte=config.target_dte,
            )
            if panel_part.empty:
                summary["trades_skipped_empty_intersection"] += 1
                continue

            panel_parts.append(panel_part)
            summary["trades_labeled"] += 1
            summary["option_rows"] += int(len(panel_part))
            summary["labeled_trade_ids"].append(trade.get("trade_id"))
            _log_progress(summary, total=len(work), symbol=symbol, spec=config, logger=progress_logger)

    summary["elapsed_seconds"] = round(perf_counter() - started, 3)
    if not panel_parts:
        summary["status"] = "no_option_rows"
        return OracleOptionLabelPanelResult(summary=summary)

    panel = pd.concat(panel_parts, ignore_index=True, sort=False)
    panel["rank_y"] = panel.groupby("trade_id")["option_return"].rank(
        method="average", pct=True, ascending=True
    )
    summary.update(
        {
            "symbols": int(panel["symbol"].nunique()),
            "option_rows": int(len(panel)),
            "long_call_rows": int(
                ((panel["side"] == "long") & panel["option_type"].astype(str).str.startswith("c")).sum()
            ),
            "short_put_rows": int(
                ((panel["side"] == "short") & panel["option_type"].astype(str).str.startswith("p")).sum()
            ),
        }
    )
    return OracleOptionLabelPanelResult(panel=panel.reset_index(drop=True), summary=summary)


def _normalize_side(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"long", "buy", "bull", "bullish", "oracle_long", "oracle_buy"}:
        return "long"
    if text in {"short", "sell", "bear", "bearish", "oracle_short", "oracle_sell"}:
        return "short"
    return None


def _normalized_cached_snapshots(symbol: str, dates: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.DataFrame]:
    raw = load_thetadata_option_snapshots(symbol, dates, use_cache=True, download_missing=False)
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    for date, frame in raw.items():
        try:
            normalized = normalize_thetadata_option_chain(frame)
        except (KeyError, TypeError, ValueError):
            continue
        if normalized is not None:
            snapshots[pd.Timestamp(date).normalize()] = normalized
    return snapshots


def _filter_entry_candidates(
    chain: pd.DataFrame,
    *,
    option_type: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    max_dte: int | None,
) -> pd.DataFrame:
    if chain is None or chain.empty or not {"option_type", "bid", "ask", "expiration"}.issubset(chain.columns):
        return pd.DataFrame()
    out = chain.copy()
    out["option_type"] = out["option_type"].astype(str).str.lower().str.strip()
    out = out.loc[out["option_type"].str.startswith(str(option_type)[0])].copy()
    out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out = out.loc[out["bid"].ge(0) & out["ask"].gt(0) & out["ask"].ge(out["bid"])].copy()
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.normalize()
    out["dte"] = (out["expiration"] - pd.Timestamp(entry_date).normalize()).dt.days
    keep = out["dte"].gt(0) & out["expiration"].ge(pd.Timestamp(exit_date).normalize())
    if max_dte is not None and int(max_dte) > 0:
        keep &= out["dte"].le(int(max_dte))
    return out.loc[keep].reset_index(drop=True)


def _panel_rows(
    labels: pd.DataFrame,
    *,
    trade: dict[str, Any],
    side: str,
    option_type: str,
    entry_features: pd.DataFrame,
    target_dte: int,
) -> pd.DataFrame:
    contract_col = _first_column(labels, "contract_symbol", "contract_symbol_entry")
    if contract_col is None:
        return pd.DataFrame()
    expiration_col = _first_column(labels, "expiration", "expiration_entry")
    strike_col = _first_column(labels, "strike", "strike_entry")
    type_col = _first_column(labels, "option_type", "option_type_entry")
    count = len(labels)
    panel = pd.DataFrame(index=labels.index)
    panel["trade_id"] = labels.get("trade_id", pd.Series(trade.get("trade_id"), index=labels.index))
    panel["symbol"] = str(trade.get("symbol") or "").upper()
    panel["side"] = side
    panel["equity_signal_side"] = side
    panel["option_type"] = labels[type_col].astype(str).str.lower() if type_col else option_type
    panel["option_action"] = "buy_call" if side == "long" else "buy_put"
    panel["entry_date"] = _date_series(labels, "trade_entry_date", trade.get("entry_date"), count)
    panel["equity_exit_date"] = _date_series(labels, "trade_exit_date", trade.get("exit_date"), count)
    panel["option_exit_date"] = _date_series(labels, "exit_snapshot_date", pd.NaT, count)
    panel["contract_symbol"] = labels[contract_col].astype(str)
    panel["expiration"] = pd.to_datetime(labels[expiration_col], errors="coerce").dt.normalize() if expiration_col else pd.NaT
    panel["strike"] = pd.to_numeric(labels[strike_col], errors="coerce") if strike_col else np.nan
    panel["entry_ask"] = _numeric_column(labels, "ask_entry", "entry_quote")
    panel["exit_bid"] = _numeric_column(labels, "bid_exit", "exit_quote")
    for output, sources in {
        "entry_quote": ("entry_quote",),
        "exit_quote": ("exit_quote",),
        "entry_bid": ("bid_entry",),
        "exit_ask": ("ask_exit",),
        "entry_mid": ("mid_entry",),
        "exit_mid": ("mid_exit",),
    }.items():
        panel[output] = _numeric_column(labels, *sources)
    panel["freq"] = trade.get("freq")
    panel["k"] = trade.get("k")
    panel["hold_days"] = trade.get("hold_days")
    panel["ret_dec"] = trade.get("ret_dec")
    panel["realized_underlying_trade_return"] = pd.to_numeric(trade.get("ret_dec"), errors="coerce")
    panel["realized_holding_days"] = _numeric_column(labels, "trade_duration_days")
    panel["entry_price"] = panel["entry_ask"]
    panel["exit_price"] = panel["exit_bid"]
    panel["return_denominator"] = panel["entry_price"]
    panel["option_pnl"] = panel["exit_price"] - panel["entry_price"]
    panel["option_return"] = np.where(
        panel["entry_price"].gt(0) & panel["exit_price"].notna(),
        panel["option_pnl"] / panel["entry_price"],
        np.nan,
    )
    panel["pricing_convention"] = "buy_ask_sell_bid_entry_exit_only"

    if entry_features is not None and not entry_features.empty and "contract_symbol" in entry_features.columns:
        feature_cols = option_ranker_feature_columns(entry_features)
        extras = [
            col for col in ("bid", "ask", "mid", "underlying_price", "snapshot_date", "spread")
            if col in entry_features.columns and col not in feature_cols
        ]
        features = entry_features[["contract_symbol", *feature_cols, *extras]].copy()
        features["contract_symbol"] = features["contract_symbol"].astype(str)
        panel = panel.merge(features.drop_duplicates("contract_symbol"), on="contract_symbol", how="left")
    if "dte" in panel.columns and "dte_gap" not in panel.columns:
        panel["dte_gap"] = (pd.to_numeric(panel["dte"], errors="coerce") - float(target_dte)).abs()
    panel["fixed_near_atm_score"] = _fixed_near_atm_score(panel)
    panel = panel.dropna(subset=["trade_id", "symbol", "entry_date", "option_return", "contract_symbol"])
    panel = panel.loc[panel["entry_ask"].gt(0) & panel["exit_bid"].ge(0)].copy()
    panel["_occ_like"] = panel["contract_symbol"].str.match(r"^[A-Z]+\d{6}[CP]\d+$", na=False)
    panel = panel.sort_values(["trade_id", "_occ_like"], ascending=[True, False], kind="stable")
    dedupe = [col for col in ("trade_id", "option_type", "expiration", "strike") if col in panel.columns]
    if len(dedupe) >= 3:
        panel = panel.drop_duplicates(dedupe, keep="first")
    return panel.drop(columns="_occ_like").reset_index(drop=True)


def _fixed_near_atm_score(panel: pd.DataFrame) -> pd.Series:
    terms: list[pd.Series] = []
    for col in ("dte_gap", "abs_moneyness", "spread_pct"):
        if col not in panel.columns:
            continue
        values = pd.to_numeric(panel[col], errors="coerce")
        scale = values.abs().median()
        if pd.notna(scale) and scale > 0:
            values = values / float(scale)
        terms.append(values.fillna(values.max() if values.notna().any() else 0.0))
    if not terms:
        return pd.Series(np.nan, index=panel.index)
    return -sum(terms)


def _first_column(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _numeric_column(frame: pd.DataFrame, *names: str) -> pd.Series:
    name = _first_column(frame, *names)
    if name is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _date_series(frame: pd.DataFrame, column: str, fallback: Any, count: int) -> pd.Series:
    values = frame[column] if column in frame.columns else pd.Series([fallback] * count, index=frame.index)
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _median_underlying_price(frame: pd.DataFrame) -> float | None:
    if "underlying_price" not in frame.columns:
        return None
    values = pd.to_numeric(frame["underlying_price"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.loc[values.gt(0)]
    return None if values.empty else float(values.median())


def _initial_summary(work: pd.DataFrame, spec: OracleOptionLabelPanelSpec) -> dict[str, Any]:
    return {
        "status": "ok",
        "trades_requested": int(len(work)),
        "trades_labeled": 0,
        "trades_skipped_missing_historical_options": 0,
        "trades_skipped_empty_intersection": 0,
        "option_rows": 0,
        "symbols": 0,
        "skipped_missing_options": [],
        "labeled_trade_ids": [],
        "elapsed_seconds": 0.0,
        "thetadata_mode": "arctic_cache_only_download_missing=False",
        "skip_policy": "skip_oracle_trade_if_entry_or_exit_option_chain_missing",
        "pricing_convention": "buy_ask_entry_sell_bid_exit_no_intermediate_marks",
        "entry_filters": "option_type + bid/ask + expiration_covers_oracle_exit + optional_max_dte",
        "exit_policy": "unfiltered_existence_match_only",
        "max_dte": None if spec.max_dte is None else int(spec.max_dte),
    }


def _record_missing_endpoint(
    summary: dict[str, Any],
    trade: dict[str, Any],
    side: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_ok: bool,
    exit_ok: bool,
) -> None:
    summary["trades_skipped_missing_historical_options"] += 1
    summary["skipped_missing_options"].append(
        {
            "trade_id": trade.get("trade_id"),
            "symbol": trade.get("symbol"),
            "side": side,
            "entry_date": entry_date.date().isoformat(),
            "exit_date": exit_date.date().isoformat(),
            "k": trade.get("k"),
            "freq": trade.get("freq"),
            "entry_option_data": bool(entry_ok),
            "exit_option_data": bool(exit_ok),
            "skip_reason": "missing_historical_option_data",
        }
    )


def _append_limited(summary: dict[str, Any], key: str, row: dict[str, Any], limit: int = 50) -> None:
    rows = summary.setdefault(key, [])
    if len(rows) < limit:
        rows.append(row)


def _log_progress(
    summary: dict[str, Any],
    *,
    total: int,
    symbol: str,
    spec: OracleOptionLabelPanelSpec,
    logger: ProgressLogger,
) -> None:
    done = (
        summary["trades_labeled"]
        + summary["trades_skipped_missing_historical_options"]
        + summary["trades_skipped_empty_intersection"]
    )
    if callable(logger) and spec.progress_every > 0 and done % int(spec.progress_every) == 0:
        logger(
            f"[option-labels] done={done}/{total} labeled={summary['trades_labeled']} "
            f"rows={summary['option_rows']} "
            f"skipped_missing_options={summary['trades_skipped_missing_historical_options']} "
            f"empty_intersection={summary['trades_skipped_empty_intersection']} symbol={symbol}"
        )
