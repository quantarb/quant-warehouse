from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_warehouse.target_engineering.operations import (
    apply_trade_deduplication,
    build_label_rows_from_completed_trades,
    build_label_statistics,
    trade_return_pct,
)
from quant_warehouse.target_engineering.specs import (
    LabelBuildSpec,
    OracleLabelResult,
    TradeGenerationResult,
)
from quant_warehouse.target_engineering.strategy_solver import (
    solve_joint_trade_sequence_by_frequency,
    solve_joint_trades_by_frequency_batched,
    solve_joint_trades_by_frequency,
)


def normalize_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow-normalized frame with lowercase string column names."""

    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def add_binary_classification_labels(
    events: pd.DataFrame,
    *,
    use_sample_weight: bool = True,
    r_clip: float = 0.10,
    alpha: float = 4.0,
    horizon_balance: bool = True,
    horizon_balance_mode: str = "mass",
    entry_only_weighting: bool = True,
    horizon_factor_cap: float | None = 3.0,
) -> pd.DataFrame:
    """Convert per-event rows into binary long-vs-short labels."""

    if events is None or len(events) == 0:
        return pd.DataFrame()

    ev = normalize_label_frame(events)
    _require_columns(ev, ["event", "side", "horizon"], ctx="add_binary_classification_labels")
    base_cols = ["event", "side", "horizon"]
    extra_cols = ["trade_return"] if "trade_return" in ev.columns else []
    out = ev[base_cols + extra_cols].copy()

    is_long_entry = (out["side"] == "long") & (out["event"] == "entry")
    is_short_exit = (out["side"] == "short") & (out["event"] == "exit")
    out["target"] = (is_long_entry | is_short_exit).astype(int)
    if use_sample_weight and "trade_return" in out.columns:
        returns = pd.to_numeric(out["trade_return"], errors="coerce").fillna(0.0).to_numpy()
        clipped = np.clip(returns, 0.0, float(r_clip))
        denom = float(r_clip) if float(r_clip) > 0 else 1.0
        out["sample_weight"] = (1.0 + float(alpha) * (clipped / denom)).astype(float)
        is_entry = out["event"] == "entry"
        if entry_only_weighting:
            out.loc[~is_entry, "sample_weight"] = 1.0
        if horizon_balance:
            if horizon_balance_mode not in {"mass", "count"}:
                raise ValueError("horizon_balance_mode must be 'mass' or 'count'")
            if horizon_balance_mode == "count":
                denom_series = out.loc[is_entry].groupby(["side", "horizon"]).size().astype(float)
            else:
                denom_series = out.loc[is_entry].groupby(["side", "horizon"])["sample_weight"].sum().astype(float)
            inv = 1.0 / denom_series
            inv = (inv / inv.mean()).clip(lower=1.0)
            if horizon_factor_cap is not None:
                inv = inv.clip(upper=float(horizon_factor_cap))
            entry_keys = pd.MultiIndex.from_arrays(
                [out.loc[is_entry, "side"], out.loc[is_entry, "horizon"]],
                names=["side", "horizon"],
            )
            out.loc[is_entry, "sample_weight"] *= entry_keys.map(inv).to_numpy(dtype=float)

    keep = ["target", "side", "horizon"]
    if "trade_return" in out.columns:
        keep.append("trade_return")
    if "sample_weight" in out.columns:
        keep.append("sample_weight")
    return out[keep].sort_index()


def add_action_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Convert per-event rows into explicit trading action labels."""

    if events is None or len(events) == 0:
        return pd.DataFrame()

    ev = normalize_label_frame(events)
    _require_columns(ev, ["event", "side", "horizon"], ctx="add_action_labels")
    base_cols = ["event", "side", "horizon"]
    extra_cols = ["trade_return"] if "trade_return" in ev.columns else []
    out = ev[base_cols + extra_cols].copy()

    conditions = [
        (out["side"] == "long") & (out["event"] == "entry"),
        (out["side"] == "long") & (out["event"] == "exit"),
        (out["side"] == "short") & (out["event"] == "entry"),
        (out["side"] == "short") & (out["event"] == "exit"),
    ]
    out["label"] = np.select(conditions, ["buy", "sell", "short", "cover"], default="unknown")
    out["market_position"] = out["label"].map({"buy": 0, "short": 0, "sell": 1, "cover": -1}).astype(int)
    keep = ["label", "market_position", "side", "horizon"]
    if "trade_return" in out.columns:
        keep.append("trade_return")
    return out[keep].sort_index()


def add_rank_regression_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Add global percentile-rank regression targets from `trade_return`."""

    if labels is None or len(labels) == 0:
        return pd.DataFrame() if labels is None else labels.copy()
    df = normalize_label_frame(labels)
    _require_columns(df, ["trade_return"], ctx="add_rank_regression_labels")
    ret = pd.to_numeric(df["trade_return"], errors="coerce")
    if "target" in df.columns:
        target = pd.to_numeric(df["target"], errors="coerce")
        df["side_profit"] = ret.where(target == 1, -ret).astype(float)
    else:
        df["side_profit"] = ret.astype(float)
    df["rank_y"] = ret.rank(method="average", pct=True)
    return df


def generate_optimal_events(
    df_daily: pd.DataFrame,
    k_params: Mapping[str, int | Sequence[int]],
    *,
    solver_mode: str = "period_top_k",
    price_col: str = "close",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    min_profit_pct: float = 0.01,
    buy_execution: str | None = None,
    sell_execution: str | None = None,
    short_execution: str | None = None,
    cover_execution: str | None = None,
) -> pd.DataFrame:
    """Generate entry/exit event rows from a daily price frame."""

    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    df = normalize_label_frame(df_daily)
    px = _get_price_series(df, price_col=price_col)
    if not px.index.is_unique:
        px = px.groupby(level=0).last()

    rows: list[dict[str, Any]] = []
    trade_counter = 0

    def _safe_loc_price(ts: pd.Timestamp) -> float:
        if ts not in px.index:
            prev = px.index[px.index <= ts]
            if len(prev) == 0:
                raise KeyError(f"No price available on or before {ts}")
            ts = prev[-1]
        return float(px.loc[ts])

    for freq, k_value in k_params.items():
        ks = [k_value] if isinstance(k_value, int) else list(k_value)
        if solver_mode == "period_sequence":
            trades = solve_joint_trade_sequence_by_frequency(
                df,
                freq=freq,
                min_profit_pct=min_profit_pct,
                long_entry_price_col=buy_execution,
                long_exit_price_col=sell_execution,
                short_entry_price_col=short_execution,
                short_exit_price_col=cover_execution,
            )
            for trade in trades:
                trade_counter += 1
                rows.extend(
                    _event_rows(
                        trade,
                        freq=freq,
                        k=0,
                        trade_counter=trade_counter,
                        price_at=_safe_loc_price,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                    )
                )
            continue

        for k in ks:
            trades = solve_joint_trades_by_frequency(
                df,
                k=int(k),
                freq=freq,
                min_profit_pct=min_profit_pct,
                long_entry_price_col=buy_execution,
                long_exit_price_col=sell_execution,
                short_entry_price_col=short_execution,
                short_exit_price_col=cover_execution,
            )
            for trade in trades:
                trade_counter += 1
                rows.extend(
                    _event_rows(
                        trade,
                        freq=freq,
                        k=int(k),
                        trade_counter=trade_counter,
                        price_at=_safe_loc_price,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                    )
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def build_label_panel(
    daily_by_symbol: Mapping[str, pd.DataFrame],
    *,
    k_params: Mapping[str, int | Sequence[int]],
    execution_params: Mapping[str, Any] | None = None,
    weighting: Mapping[str, Any] | None = None,
    solver_mode: str = "period_top_k",
    add_rank_labels: bool = True,
    deduplicate: bool = True,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Build a combined label panel, optionally parallelized by symbol."""

    execution = dict(execution_params or {})
    weighting_params = dict(weighting or {})
    tasks = [
        (symbol, frame, dict(k_params), execution, weighting_params, solver_mode)
        for symbol, frame in daily_by_symbol.items()
        if frame is not None and not frame.empty
    ]

    all_label_frames: list[pd.DataFrame] = []
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_build_one_symbol_labels, task): task[0] for task in tasks}
            for future in as_completed(futures):
                symbol, result, error = future.result()
                if error:
                    if error != "no events produced":
                        print(f"[build_label_panel] {symbol}: {error}")
                    continue
                all_label_frames.append(result)
    else:
        for task in tasks:
            symbol, result, error = _build_one_symbol_labels(task)
            if error:
                if error != "no events produced":
                    print(f"[build_label_panel] {symbol}: {error}")
                continue
            all_label_frames.append(result)

    if not all_label_frames:
        return pd.DataFrame()
    full_labels = pd.concat(all_label_frames, ignore_index=True).set_index(["date", "symbol"]).sort_index()
    if deduplicate:
        full_labels = deduplicate_labels(full_labels)
    if add_rank_labels:
        full_labels = add_rank_regression_labels(full_labels)
    return full_labels


def build_trade_results(
    symbols: Sequence[str],
    *,
    spec: LabelBuildSpec,
    price_frames: Mapping[str, pd.DataFrame],
    progress_callback: Callable[..., None] | None = None,
) -> TradeGenerationResult:
    """Build raw oracle trade candidates from supplied price frames."""

    trade_rows: list[dict[str, Any]] = []
    completed_trades: list[dict[str, Any]] = []
    normalized_symbols = [str(sym).strip().upper() for sym in list(symbols or []) if str(sym).strip()]
    total_symbols = len(normalized_symbols)
    if callable(progress_callback):
        progress_callback(completed=0, total=total_symbols, current_symbol="")

    can_batch = (
        spec.solver_mode == "period_top_k"
        and len(normalized_symbols) > 1
        and bool(price_frames)
    )

    if can_batch:
        symbol_frames = {
            symbol: (price_frames.get(symbol) if price_frames.get(symbol) is not None else price_frames.get(symbol.lower()))
            for symbol in normalized_symbols
        }
        for freq, ks in spec.k_params.items():
            for k in ks:
                batch_results = solve_joint_trades_by_frequency_batched(
                    symbol_frames,
                    k=int(k),
                    freq=freq,
                    min_profit_pct=spec.min_profit_pct,
                    long_entry_price_col=spec.buy_execution,
                    long_exit_price_col=spec.sell_execution,
                    short_entry_price_col=spec.short_execution,
                    short_exit_price_col=spec.cover_execution,
                )
                for symbol in normalized_symbols:
                    _append_completed(
                        symbol,
                        freq,
                        int(k),
                        batch_results.get(symbol, []),
                        trade_rows,
                        completed_trades,
                    )
        if callable(progress_callback):
            progress_callback(completed=total_symbols, total=total_symbols, current_symbol="")
        return TradeGenerationResult(trade_rows=trade_rows, completed_trades=completed_trades)

    for idx, symbol in enumerate(normalized_symbols, start=1):
        if callable(progress_callback):
            progress_callback(completed=max(0, idx - 1), total=total_symbols, current_symbol=symbol)
        frame = price_frames.get(symbol)
        if frame is None:
            frame = price_frames.get(symbol.lower())
        daily_prices = _slice_dates(frame, spec.start_date, spec.end_date)
        if daily_prices.empty:
            if callable(progress_callback):
                progress_callback(completed=idx, total=total_symbols, current_symbol=symbol)
            continue
        for freq, ks in spec.k_params.items():
            if spec.solver_mode == "period_sequence":
                trades = solve_joint_trade_sequence_by_frequency(
                    daily_prices,
                    freq=freq,
                    min_profit_pct=spec.min_profit_pct,
                    long_entry_price_col=spec.buy_execution,
                    long_exit_price_col=spec.sell_execution,
                    short_entry_price_col=spec.short_execution,
                    short_exit_price_col=spec.cover_execution,
                )
                _append_completed(symbol, freq, 0, trades, trade_rows, completed_trades)
                continue
            for k in ks:
                trades = solve_joint_trades_by_frequency(
                    daily_prices,
                    k=int(k),
                    freq=freq,
                    min_profit_pct=spec.min_profit_pct,
                    long_entry_price_col=spec.buy_execution,
                    long_exit_price_col=spec.sell_execution,
                    short_entry_price_col=spec.short_execution,
                    short_exit_price_col=spec.cover_execution,
                )
                _append_completed(symbol, freq, int(k), trades, trade_rows, completed_trades)
        if callable(progress_callback):
            progress_callback(completed=idx, total=total_symbols, current_symbol=symbol)
    return TradeGenerationResult(trade_rows=trade_rows, completed_trades=completed_trades)


def build_oracle_labels(
    symbols: Sequence[str],
    *,
    spec: LabelBuildSpec,
    price_frames: Mapping[str, pd.DataFrame],
    progress_callback: Callable[..., None] | None = None,
) -> OracleLabelResult:
    """Build canonical label rows and summary statistics from oracle trades."""

    generated = build_trade_results(symbols, spec=spec, price_frames=price_frames, progress_callback=progress_callback)
    _, completed = apply_trade_deduplication(generated.trade_rows, generated.completed_trades, mode=spec.trade_dedup_mode)
    label_rows = build_label_rows_from_completed_trades(completed)
    return OracleLabelResult(
        label_rows=label_rows,
        statistics=build_label_statistics(label_rows),
        completed_trades=completed,
    )


def deduplicate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one signal per date/symbol/side/action, preferring highest return."""

    if df.empty:
        return df
    idx_names = list(df.index.names)
    tmp = df.reset_index()
    subset = ["date", "symbol", "side"]
    if "label" in tmp.columns:
        subset.append("label")
    if "trade_return" in tmp.columns:
        tmp = tmp.sort_values("trade_return", ascending=False)
    unique = tmp.drop_duplicates(subset=subset, keep="first")
    return unique.set_index(idx_names).sort_index()


def _build_one_symbol_labels(args: tuple[Any, ...]) -> tuple[str, pd.DataFrame | None, str | None]:
    symbol, df_daily, k_params, execution, weighting, solver_mode = args
    try:
        events = generate_optimal_events(
            df_daily=df_daily,
            k_params=k_params,
            solver_mode=solver_mode,
            price_col=str(execution.get("price_col") or execution.get("sell_execution") or "close"),
            fee_bps=float(execution.get("fee_bps") or 0.0),
            slippage_bps=float(execution.get("slippage_bps") or 0.0),
            min_profit_pct=float(execution.get("min_profit_pct") or 0.01),
            buy_execution=execution.get("buy_execution"),
            sell_execution=execution.get("sell_execution"),
            short_execution=execution.get("short_execution"),
            cover_execution=execution.get("cover_execution"),
        )
        if events.empty:
            return (symbol, None, "no events produced")

        actions = add_action_labels(events)
        labels = add_binary_classification_labels(events, **weighting)
        labels["label"] = actions["label"]
        labels["market_position"] = actions["market_position"]
        labels["symbol"] = symbol
        for column in ("event", "trade_id", "entry_date", "exit_date", "entry_px", "exit_px", "trade_duration_days"):
            if column in events.columns:
                labels[column] = events[column]
        if "trade_duration_days" in labels.columns and "hold_days" not in labels.columns:
            labels["hold_days"] = labels["trade_duration_days"]
        return (symbol, labels.reset_index(), None)
    except Exception as exc:
        return (symbol, None, f"{type(exc).__name__}: {exc}")


def _event_rows(
    trade: Mapping[str, Any],
    *,
    freq: str,
    k: int,
    trade_counter: int,
    price_at: Callable[[pd.Timestamp], float],
    fee_bps: float,
    slippage_bps: float,
) -> list[dict[str, Any]]:
    side = str(trade.get("side") or "").strip().lower()
    if side not in {"long", "short"}:
        return []
    entry_dt = pd.Timestamp(trade["entry_row"].name)
    exit_dt = pd.Timestamp(trade["exit_row"].name)
    entry_px = price_at(entry_dt)
    exit_px = price_at(exit_dt)
    gross_r = (exit_px - entry_px) / entry_px if side == "long" else (entry_px - exit_px) / entry_px
    net_r = gross_r - 2.0 * (float(fee_bps) + float(slippage_bps)) / 10000.0
    payload = {
        "side": side,
        "horizon": f"{freq}_k{k}" if k else freq,
        "trade_id": f"{side}:{freq}:k{k}:{trade_counter}",
        "entry_date": entry_dt,
        "exit_date": exit_dt,
        "entry_px": float(entry_px),
        "exit_px": float(exit_px),
        "trade_duration_days": int((exit_dt - entry_dt).days),
        "trade_return": float(net_r),
    }
    return [{"date": entry_dt, "event": "entry", **payload}, {"date": exit_dt, "event": "exit", **payload}]


def _append_completed(
    symbol: str,
    freq: str,
    k: int,
    trades: Sequence[Mapping[str, Any]],
    trade_rows: list[dict[str, Any]],
    completed_trades: list[dict[str, Any]],
) -> None:
    for trade in trades:
        side = str(trade.get("side") or "").strip().lower()
        if side not in {"long", "short"}:
            continue
        entry_dt = pd.Timestamp(trade["entry_row"].name)
        exit_dt = pd.Timestamp(trade["exit_row"].name)
        entry_px = float(trade["entry_price"])
        exit_px = float(trade["exit_price"])
        ret_dec = trade_return_pct(side, entry_px, exit_px)
        row = {
            "symbol": symbol,
            "side": side,
            "freq": freq,
            "k": int(k),
            "entry_date": entry_dt.strftime("%Y-%m-%d"),
            "exit_date": exit_dt.strftime("%Y-%m-%d"),
            "entry_px": f"{entry_px:,.4f}",
            "exit_px": f"{exit_px:,.4f}",
            "ret_pct": f"{ret_dec * 100:.2f}%",
        }
        trade_rows.append(row)
        completed_trades.append(
            {
                **row,
                "ret_dec": ret_dec,
                "hold_days": int((exit_dt - entry_dt).days),
            }
        )


def _get_price_series(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    col_map = {str(col).lower(): col for col in df.columns}
    requested = str(price_col).lower()
    if requested in col_map:
        return df[col_map[requested]]
    for fallback in ("close", "adj_close", "adjclose", "price", "adj_low", "low"):
        if fallback in col_map:
            return df[col_map[fallback]]
    raise ValueError(f"Could not find a usable price column. Available: {list(df.columns)}")


def _slice_dates(df: pd.DataFrame | None, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" not in out.columns:
            raise ValueError("Price frames must have a DatetimeIndex or a 'date' column")
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date")
    out = out.sort_index()
    if start_date:
        out = out.loc[out.index >= pd.Timestamp(start_date)]
    if end_date:
        out = out.loc[out.index <= pd.Timestamp(end_date)]
    return out[~out.index.duplicated(keep="last")]


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
