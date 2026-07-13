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
ORACLE_OPTION_LABEL_POLICY = "oracle_exit_survivors_expiration_early_fallback_v1"


@dataclass(frozen=True)
class OracleOptionLabelPanelSpec:
    """Configuration for labels built from oracle equity trade endpoints."""

    max_trades: int = 0
    max_dte: int | None = None
    target_dte: int = 90
    max_candidates_per_trade: int = 128
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
    """Build cache-only option labels from oracle entries and available exits.

    Candidates are filtered at entry by option side, executable bid/ask, and
    maximum DTE. Contracts surviving the oracle horizon use the oracle-exit
    bid; contracts expiring earlier use their expiration-date bid. Missing
    applicable quotes retain expiration closeness in the same rank space.
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
        entry_candidate_cache: dict[tuple[pd.Timestamp, str], pd.DataFrame] = {}
        entry_feature_cache: dict[tuple[pd.Timestamp, str], pd.DataFrame] = {}
        # Resolve all early-expiration dates in one ArcticDB range read per
        # symbol. Per-trade reads repeatedly scanned the same symbol history.
        early_expiration_dates: set[pd.Timestamp] = set()
        for trade in symbol_trades.to_dict("records"):
            side = str(trade["side"])
            option_type = "call" if side == "long" else "put"
            entry_date = pd.Timestamp(trade["entry_date"]).normalize()
            exit_date = pd.Timestamp(trade["exit_date"]).normalize()
            entry_chain = snapshots.get(entry_date)
            if entry_chain is None or entry_chain.empty:
                continue
            entry_key = (entry_date, option_type)
            candidates = entry_candidate_cache.get(entry_key)
            if candidates is None:
                candidates = _filter_entry_candidates(
                    entry_chain,
                    option_type=option_type,
                    entry_date=entry_date,
                    max_dte=config.max_dte,
                )
                entry_candidate_cache[entry_key] = candidates
            expirations = pd.to_datetime(candidates.get("expiration"), errors="coerce")
            early_expiration_dates.update(
                pd.Timestamp(value).normalize()
                for value in expirations.loc[expirations.lt(exit_date)].dropna().unique()
            )
        missing_early_dates = sorted(date for date in early_expiration_dates if date not in snapshots)
        if missing_early_dates:
            snapshots.update(_normalized_cached_snapshots(symbol, missing_early_dates))

        for trade in symbol_trades.to_dict("records"):
            side = str(trade["side"])
            option_type = "call" if side == "long" else "put"
            entry_date = pd.Timestamp(trade["entry_date"]).normalize()
            exit_date = pd.Timestamp(trade["exit_date"]).normalize()
            entry_chain = snapshots.get(entry_date)
            entry_ok = entry_chain is not None and not entry_chain.empty
            oracle_exit_chain = snapshots.get(exit_date)
            exit_ok = oracle_exit_chain is not None and not oracle_exit_chain.empty
            if not entry_ok:
                _record_missing_endpoint(summary, trade, side, entry_date, exit_date, entry_ok, exit_ok)
                continue

            entry_key = (entry_date, option_type)
            entry_candidates = entry_candidate_cache.get(entry_key)
            if entry_candidates is None:
                entry_candidates = _filter_entry_candidates(
                    entry_chain,
                    option_type=option_type,
                    entry_date=entry_date,
                    max_dte=config.max_dte,
                )
                entry_candidate_cache[entry_key] = entry_candidates
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
                        "reason": "no_entry_candidates_after_type_bid_ask_optional_max_dte",
                    },
                )
                continue

            candidate_expirations = pd.to_datetime(
                entry_candidates["expiration"], errors="coerce"
            ).dt.normalize()
            early_candidates = entry_candidates.loc[candidate_expirations.lt(exit_date)].copy()
            surviving_candidates = entry_candidates.loc[candidate_expirations.ge(exit_date)].copy()

            expiration_exit_chain, expiration_exit_dates = _contract_expiration_exits(
                early_candidates, snapshots
            )

            expiration_label_frame = pd.DataFrame()
            if not expiration_exit_chain.empty:
                label_result = build_option_labels(
                    [trade],
                    {entry_date: early_candidates, exit_date: expiration_exit_chain},
                    spec=label_spec,
                )
                expiration_label_frame = pd.DataFrame(label_result.option_rows)

            oracle_label_frame = pd.DataFrame()
            if exit_ok and not surviving_candidates.empty:
                oracle_result = build_option_labels(
                    [trade],
                    {entry_date: surviving_candidates, exit_date: oracle_exit_chain},
                    spec=label_spec,
                )
                oracle_label_frame = pd.DataFrame(oracle_result.option_rows)

            featured = entry_feature_cache.get(entry_key)
            if featured is None:
                featured = build_option_contract_features(
                    entry_candidates,
                    underlying_price=_median_underlying_price(entry_candidates),
                    target_dte=int(config.target_dte),
                ).df
                entry_feature_cache[entry_key] = featured
            expiration_part = _panel_rows(
                expiration_label_frame,
                trade=trade,
                side=side,
                option_type=option_type,
                entry_features=featured,
                target_dte=config.target_dte,
            )
            if not expiration_part.empty:
                expiration_part["option_exit_date"] = expiration_part["contract_symbol"].map(
                    expiration_exit_dates
                )
                expiration_part["days_before_oracle_exit"] = (
                    exit_date - pd.to_datetime(expiration_part["option_exit_date"], errors="coerce")
                ).dt.days
                expiration_part["return_horizon"] = "contract_expiration"
            oracle_part = _panel_rows(
                oracle_label_frame,
                trade=trade,
                side=side,
                option_type=option_type,
                entry_features=featured,
                target_dte=config.target_dte,
            )
            if not oracle_part.empty:
                oracle_part["return_horizon"] = "oracle_exit"
            realized_contracts = set(
                pd.concat(
                    [
                        expiration_part.get("contract_symbol", pd.Series(dtype=str)),
                        oracle_part.get("contract_symbol", pd.Series(dtype=str)),
                    ],
                    ignore_index=True,
                ).astype(str)
            )
            expired_part = _expiration_closeness_rows(
                entry_candidates.loc[
                    ~entry_candidates["contract_symbol"].astype(str).isin(realized_contracts)
                ].copy(),
                trade=trade,
                side=side,
                option_type=option_type,
                entry_features=featured,
                target_dte=config.target_dte,
            )
            if not expired_part.empty:
                expired_part["return_horizon"] = "expiration_closeness_fallback"
            panel_part = pd.concat(
                [expired_part, oracle_part, expiration_part], ignore_index=True, sort=False
            )
            if panel_part.empty:
                summary["trades_skipped_empty_intersection"] += 1
                continue

            panel_part = _unified_behavior_rank(panel_part)
            panel_part["label_policy"] = ORACLE_OPTION_LABEL_POLICY
            summary["option_rows_before_candidate_bound"] += int(len(panel_part))
            panel_part = _select_diverse_labeled_candidates(
                panel_part, int(config.max_candidates_per_trade)
            )
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
    summary.update(
        {
            "symbols": int(panel["symbol"].nunique()),
            "option_rows": int(len(panel)),
            "realized_return_rows": int(panel["label_basis"].eq("realized_exit_return").sum()),
            "expiration_closeness_rows": int(panel["label_basis"].eq("expiration_closeness").sum()),
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

    def normalize(item: tuple[pd.Timestamp, pd.DataFrame]) -> tuple[pd.Timestamp, pd.DataFrame | None]:
        date, frame = item
        try:
            normalized = normalize_thetadata_option_chain(frame)
        except (KeyError, TypeError, ValueError):
            normalized = None
        return pd.Timestamp(date).normalize(), normalized

    for date, normalized in map(normalize, raw.items()):
        if normalized is not None:
            snapshots[date] = normalized
    return snapshots


def _contract_expiration_exits(
    entry_candidates: pd.DataFrame,
    snapshots: dict[pd.Timestamp, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    """Select each entry contract's quote on its own expiration date."""

    parts: list[pd.DataFrame] = []
    candidates = entry_candidates[["contract_symbol", "expiration"]].copy()
    candidates["contract_symbol"] = candidates["contract_symbol"].astype(str)
    candidates["expiration"] = pd.to_datetime(candidates["expiration"], errors="coerce").dt.normalize()
    for expiration, expected in candidates.dropna().groupby("expiration", sort=False):
        date = pd.Timestamp(expiration).normalize()
        chain = snapshots.get(date)
        if chain is None or chain.empty or "contract_symbol" not in chain:
            continue
        wanted = set(expected["contract_symbol"])
        part = chain.loc[chain["contract_symbol"].astype(str).isin(wanted)].copy()
        part["_contract_expiration_date"] = date
        parts.append(part)
    if not parts:
        return pd.DataFrame(), {}
    combined = pd.concat(parts, ignore_index=True, sort=False)
    if "contract_symbol" not in combined.columns:
        return pd.DataFrame(), {}
    combined["contract_symbol"] = combined["contract_symbol"].astype(str)
    combined["bid"] = pd.to_numeric(combined.get("bid"), errors="coerce")
    combined = combined.dropna(subset=["contract_symbol", "_contract_expiration_date", "bid"])
    combined = combined.loc[combined["bid"].ge(0)].drop_duplicates("contract_symbol", keep="last")
    exit_dates = dict(
        zip(combined["contract_symbol"], combined["_contract_expiration_date"], strict=False)
    )
    return combined.drop(columns="_contract_expiration_date"), exit_dates


def _filter_entry_candidates(
    chain: pd.DataFrame,
    *,
    option_type: str,
    entry_date: pd.Timestamp,
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
    keep = out["dte"].gt(0)
    if max_dte is not None and int(max_dte) > 0:
        keep &= out["dte"].le(int(max_dte))
    return out.loc[keep].reset_index(drop=True)


def _expiration_closeness_rows(
    entry_candidates: pd.DataFrame,
    *,
    trade: dict[str, Any],
    side: str,
    option_type: str,
    entry_features: pd.DataFrame,
    target_dte: int,
) -> pd.DataFrame:
    exit_date = pd.Timestamp(trade.get("exit_date")).normalize()
    fallback = entry_candidates.copy()
    fallback["expiration"] = pd.to_datetime(fallback["expiration"], errors="coerce").dt.normalize()
    if fallback.empty:
        return pd.DataFrame()
    panel = pd.DataFrame(index=fallback.index)
    panel["trade_id"] = trade.get("trade_id")
    panel["symbol"] = str(trade.get("symbol") or "").upper()
    panel["side"] = side
    panel["equity_signal_side"] = side
    panel["option_type"] = fallback["option_type"].astype(str).str.lower()
    panel["option_action"] = "buy_call" if side == "long" else "buy_put"
    panel["entry_date"] = pd.Timestamp(trade.get("entry_date")).normalize()
    panel["equity_exit_date"] = exit_date
    panel["option_exit_date"] = pd.NaT
    panel["contract_symbol"] = fallback["contract_symbol"].astype(str)
    panel["expiration"] = fallback["expiration"]
    panel["strike"] = pd.to_numeric(fallback.get("strike"), errors="coerce")
    panel["entry_bid"] = pd.to_numeric(fallback.get("bid"), errors="coerce")
    panel["entry_ask"] = pd.to_numeric(fallback.get("ask"), errors="coerce")
    panel["entry_mid"] = pd.to_numeric(fallback.get("mid"), errors="coerce")
    panel["exit_bid"] = np.nan
    panel["option_return"] = np.nan
    panel["label_basis"] = "expiration_closeness"
    panel["days_before_oracle_exit"] = (exit_date - panel["expiration"]).dt.days
    panel["freq"] = trade.get("freq")
    panel["k"] = trade.get("k")
    if entry_features is not None and not entry_features.empty and "contract_symbol" in entry_features:
        feature_cols = option_ranker_feature_columns(entry_features)
        extras = [
            col
            for col in ("bid", "ask", "mid", "underlying_price", "snapshot_date", "spread")
            if col in entry_features.columns and col not in feature_cols
        ]
        features = entry_features[["contract_symbol", *feature_cols, *extras]].copy()
        features["contract_symbol"] = features["contract_symbol"].astype(str)
        panel = panel.merge(features.drop_duplicates("contract_symbol"), on="contract_symbol", how="left")
    if "dte" in panel.columns and "dte_gap" not in panel.columns:
        panel["dte_gap"] = (pd.to_numeric(panel["dte"], errors="coerce") - float(target_dte)).abs()
    panel["fixed_near_atm_score"] = _fixed_near_atm_score(panel)
    return panel.reset_index(drop=True)


def _unified_behavior_rank(panel: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _trade_id, group in panel.groupby("trade_id", sort=False):
        work = group.copy()
        realized = work["label_basis"].eq("realized_exit_return")
        expired_count = int((~realized).sum())
        expired_value = -pd.to_numeric(
            work["days_before_oracle_exit"], errors="coerce"
        ).abs()
        realized_value = pd.to_numeric(work["option_return"], errors="coerce")
        ordinal = pd.Series(index=work.index, dtype=float)
        ordinal.loc[~realized] = expired_value.loc[~realized].rank(method="average", ascending=True)
        ordinal.loc[realized] = expired_count + realized_value.loc[realized].rank(
            method="average", ascending=True
        )
        work["rank_y"] = ordinal / float(len(work))
        parts.append(work)
    return pd.concat(parts, ignore_index=True, sort=False)


def _select_diverse_labeled_candidates(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Bound stored rows after full-chain ranking while preserving broad coverage."""

    if limit <= 0 or len(frame) <= limit:
        return frame.copy()
    work = frame.copy()
    work["_source_order"] = np.arange(len(work))
    core_size = max(1, limit // 4)
    core = work.sort_values(
        ["fixed_near_atm_score", "_source_order"],
        ascending=[False, True],
        kind="stable",
    ).head(core_size)
    remaining = work.drop(index=core.index)
    coverage_columns = [
        column for column in ("dte", "moneyness", "spread_pct", "rank_y") if column in remaining
    ]
    for column in coverage_columns:
        remaining[column] = pd.to_numeric(remaining[column], errors="coerce")
    remaining = remaining.sort_values(
        [*coverage_columns, "_source_order"],
        kind="stable",
        na_position="last",
    )
    coverage_size = min(limit - len(core), len(remaining))
    positions = np.linspace(0, len(remaining) - 1, num=coverage_size, dtype=int)
    selected = pd.concat([core, remaining.iloc[positions]], ignore_index=True, sort=False)
    return selected.drop(columns=["_source_order"])


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
    panel["label_basis"] = "realized_exit_return"
    panel["days_before_oracle_exit"] = 0
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
        "option_rows_before_candidate_bound": 0,
        "symbols": 0,
        "skipped_missing_options": [],
        "labeled_trade_ids": [],
        "elapsed_seconds": 0.0,
        "thetadata_mode": "arctic_cache_only_download_missing=False",
        "skip_policy": "skip_oracle_trade_only_if_entry_option_chain_missing",
        "pricing_convention": "buy_ask_entry_sell_oracle_exit_bid_for_survivors_or_expiration_bid_for_early_expiry",
        "entry_filters": "option_type + bid/ask + optional_max_dte",
        "exit_policy": "single_rank_space: oracle_exit_return_for_survivors_then_contract_expiration_return_for_early_expiry_then_expiration_closeness",
        "label_policy": ORACLE_OPTION_LABEL_POLICY,
        "max_dte": None if spec.max_dte is None else int(spec.max_dte),
        "max_candidates_per_trade": int(spec.max_candidates_per_trade),
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
