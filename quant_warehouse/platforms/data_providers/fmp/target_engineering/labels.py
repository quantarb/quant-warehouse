from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

import polars as pl

from quant_warehouse.platforms.data_providers.fmp.target_engineering.operations import (
    apply_trade_deduplication,
    build_label_rows_from_completed_trades,
    build_label_statistics,
    trade_return_pct,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.specs import (
    LabelBuildSpec,
    OracleLabelResult,
    TradeGenerationResult,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.strategy_solver import (
    solve_side_trades_by_frequency_batched_multi_k,
    solve_trades_by_frequency,
)


def normalize_label_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return a shallow-normalized frame with lowercase string column names."""

    if isinstance(df, pl.DataFrame):
        return df.rename({col: str(col).strip().lower() for col in df.columns})
    return df.rename({col: str(col).strip().lower() for col in df.columns})


def add_binary_classification_labels(
    events: pl.DataFrame,
    *,
    use_sample_weight: bool = True,
    r_clip: float = 0.10,
    alpha: float = 4.0,
    horizon_balance: bool = True,
    horizon_balance_mode: str = "mass",
    entry_only_weighting: bool = True,
    horizon_factor_cap: float | None = 3.0,
) -> pl.DataFrame:
    """Convert per-event rows into binary long-vs-short labels."""

    if events is None or len(events) == 0:
        return pl.DataFrame()

    ev = normalize_label_frame(events)
    if isinstance(ev, pl.DataFrame):
        _require_columns(ev, ["event", "side", "horizon"], ctx="add_binary_classification_labels")
        extra = ["trade_return"] if "trade_return" in ev.columns else []
        out = ev.select(["event", "side", "horizon"] + extra).with_columns(
            ((pl.col("side") == "long") & (pl.col("event") == "entry")
             | ((pl.col("side") == "short") & (pl.col("event") == "exit")))
            .cast(pl.Int8)
            .alias("target")
        )
        if use_sample_weight and "trade_return" in out.columns:
            out = out.with_columns(
                (1.0 + float(alpha) * pl.col("trade_return").cast(pl.Float64, strict=False).fill_null(0.0).clip(0.0, float(r_clip)) / (float(r_clip) if float(r_clip) > 0 else 1.0)).alias("sample_weight")
            )
            is_entry = (pl.col("event") == "entry")
            if entry_only_weighting:
                out = out.with_columns(pl.when(is_entry).then(pl.col("sample_weight")).otherwise(1.0).alias("sample_weight"))
            if horizon_balance:
                if horizon_balance_mode not in {"mass", "count"}:
                    raise ValueError("horizon_balance_mode must be 'mass' or 'count'")
                entries = out.filter(is_entry)
                denom_col = "sample_weight" if horizon_balance_mode == "mass" else "target"
                grouped = entries.group_by(["side", "horizon"]).agg(
                    pl.col(denom_col).sum().alias("denom") if horizon_balance_mode == "mass" else pl.len().alias("denom")
                ).with_columns((1.0 / pl.col("denom")).alias("raw_factor"))
                mean_factor = grouped.select(pl.col("raw_factor").mean()).item()
                grouped = grouped.with_columns((pl.col("raw_factor") / mean_factor).clip(lower_bound=1.0, upper_bound=horizon_factor_cap).alias("factor"))
                out = out.join(grouped.select(["side", "horizon", "factor"]), on=["side", "horizon"], how="left")
                out = out.with_columns(pl.when(is_entry).then(pl.col("sample_weight") * pl.col("factor")).otherwise(pl.col("sample_weight")).alias("sample_weight")).drop("factor")
        keep = ["target", "side", "horizon"] + extra + (["sample_weight"] if "sample_weight" in out.columns else [])
        return out.select(keep)
def add_action_labels(events: pl.DataFrame) -> pl.DataFrame:
    """Convert per-event rows into explicit trading action labels."""

    if events is None or len(events) == 0:
        return pl.DataFrame()

    ev = normalize_label_frame(events)
    if isinstance(ev, pl.DataFrame):
        _require_columns(ev, ["event", "side", "horizon"], ctx="add_action_labels")
        keep = ["event", "side", "horizon"]
        if "trade_return" in ev.columns:
            keep.append("trade_return")
        return (
            ev.select(keep)
            .with_columns(
                pl.when((pl.col("side") == "long") & (pl.col("event") == "entry"))
                .then(pl.lit("buy"))
                .when((pl.col("side") == "long") & (pl.col("event") == "exit"))
                .then(pl.lit("sell"))
                .when((pl.col("side") == "short") & (pl.col("event") == "entry"))
                .then(pl.lit("short"))
                .when((pl.col("side") == "short") & (pl.col("event") == "exit"))
                .then(pl.lit("cover"))
                .otherwise(pl.lit("unknown"))
                .alias("label")
            )
            .with_columns(
                pl.when(pl.col("label").is_in(["buy", "short"]))
                .then(pl.lit(0))
                .when(pl.col("label") == "sell")
                .then(pl.lit(1))
                .when(pl.col("label") == "cover")
                .then(pl.lit(-1))
                .otherwise(pl.lit(0))
                .alias("market_position")
            )
            .select(["label", "market_position", "side", "horizon"] + (["trade_return"] if "trade_return" in ev.columns else []))
        )
def add_rank_regression_labels(labels: pl.DataFrame) -> pl.DataFrame:
    """Add global percentile-rank regression targets from `trade_return`."""

    if labels is None or len(labels) == 0:
        return pl.DataFrame() if labels is None else labels.clone()
    df = normalize_label_frame(labels)
    if isinstance(df, pl.DataFrame):
        _require_columns(df, ["trade_return"], ctx="add_rank_regression_labels")
        out = df.with_columns(pl.col("trade_return").cast(pl.Float64, strict=False).alias("_return"))
        if "target" in out.columns:
            out = out.with_columns(
                pl.when(pl.col("target").cast(pl.Int64, strict=False) == 1)
                .then(pl.col("_return"))
                .otherwise(-pl.col("_return"))
                .alias("side_profit")
            )
        else:
            out = out.with_columns(pl.col("_return").alias("side_profit"))
        out = out.with_columns(pl.col("_return").rank(method="average").truediv(pl.len()).alias("rank_y"))
        return out.drop("_return")
    _require_columns(df, ["trade_return"], ctx="add_rank_regression_labels")
    ret = df["trade_return"].cast(pl.Float64, strict=False)
    if "target" in df.columns:
        target = df["target"].cast(pl.Float64, strict=False)
        df["side_profit"] = ret.where(target == 1, -ret).astype(float)
    else:
        df["side_profit"] = ret.astype(float)
    df["rank_y"] = ret.rank(method="average", pct=True)
    return df


def generate_optimal_events(
    df_daily: pl.DataFrame,
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
) -> pl.DataFrame:
    """Generate entry/exit event rows from a daily price frame."""

    if df_daily is None or df_daily.is_empty():
        return pl.DataFrame()
    df = normalize_label_frame(df_daily)
    px = _get_price_series(df, price_col=price_col)
    rows: list[dict[str, Any]] = []
    trade_counter = 0

    def _safe_loc_price(ts: Any) -> float:
        available = px.filter(pl.col("date") <= ts).sort("date")
        if available.is_empty():
                raise KeyError(f"No price available on or before {ts}")
        return float(available["price"][available.height - 1])

    for freq, k_value in k_params.items():
        ks = [k_value] if isinstance(k_value, int) else list(k_value)
        if solver_mode == "period_sequence":
            raise ValueError("period_sequence mixed-side oracle solver was removed; use period_top_k side-specific labels")

        for k in ks:
            trades = []
            trades.extend(
                solve_trades_by_frequency(
                    df,
                    k=int(k),
                    freq=freq,
                    side="long",
                    min_profit_pct=min_profit_pct,
                    entry_price_col=buy_execution,
                    exit_price_col=sell_execution,
                )
            )
            trades.extend(
                solve_trades_by_frequency(
                    df,
                    k=int(k),
                    freq=freq,
                    side="short",
                    min_profit_pct=min_profit_pct,
                    entry_price_col=short_execution,
                    exit_price_col=cover_execution,
                )
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
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("date")


def build_label_panel(
    daily_by_symbol: Mapping[str, pl.DataFrame],
    *,
    k_params: Mapping[str, int | Sequence[int]],
    execution_params: Mapping[str, Any] | None = None,
    weighting: Mapping[str, Any] | None = None,
    solver_mode: str = "period_top_k",
    add_rank_labels: bool = True,
    deduplicate: bool = True,
    max_workers: int = 1,
) -> pl.DataFrame:
    """Build a combined label panel, optionally parallelized by symbol."""

    execution = dict(execution_params or {})
    weighting_params = dict(weighting or {})
    tasks = [
        (symbol, frame, dict(k_params), execution, weighting_params, solver_mode)
        for symbol, frame in daily_by_symbol.items()
        if frame is not None and not frame.is_empty()
    ]

    all_label_frames: list[pl.DataFrame] = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
        return pl.DataFrame()
    full_labels = pl.concat(all_label_frames, how="diagonal_relaxed").sort(["date", "symbol"])
    if deduplicate:
        full_labels = deduplicate_labels(full_labels)
    if add_rank_labels:
        full_labels = add_rank_regression_labels(full_labels)
    return full_labels


def build_trade_results(
    symbols: Sequence[str],
    *,
    spec: LabelBuildSpec,
    price_frames: Mapping[str, pl.DataFrame],
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
            batch_by_k = solve_side_trades_by_frequency_batched_multi_k(
                symbol_frames,
                ks=ks,
                freq=freq,
                min_profit_pct=spec.min_profit_pct,
                long_entry_price_col=spec.buy_execution,
                long_exit_price_col=spec.sell_execution,
                short_entry_price_col=spec.short_execution,
                short_exit_price_col=spec.cover_execution,
            )
            for k in ks:
                batch_results = batch_by_k.get(int(k), {})
                for symbol in normalized_symbols:
                    _append_completed(symbol, freq, int(k), batch_results.get(symbol, []), trade_rows, completed_trades)
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
        if daily_prices.is_empty():
            if callable(progress_callback):
                progress_callback(completed=idx, total=total_symbols, current_symbol=symbol)
            continue
        for freq, ks in spec.k_params.items():
            if spec.solver_mode == "period_sequence":
                raise ValueError("period_sequence mixed-side oracle solver was removed; use period_top_k side-specific labels")
            for k in ks:
                trades = []
                trades.extend(
                    solve_trades_by_frequency(
                        daily_prices,
                        k=int(k),
                        freq=freq,
                        side="long",
                        min_profit_pct=spec.min_profit_pct,
                        entry_price_col=spec.buy_execution,
                        exit_price_col=spec.sell_execution,
                    )
                )
                trades.extend(
                    solve_trades_by_frequency(
                        daily_prices,
                        k=int(k),
                        freq=freq,
                        side="short",
                        min_profit_pct=spec.min_profit_pct,
                        entry_price_col=spec.short_execution,
                        exit_price_col=spec.cover_execution,
                    )
                )
                _append_completed(symbol, freq, int(k), trades, trade_rows, completed_trades)
        if callable(progress_callback):
            progress_callback(completed=idx, total=total_symbols, current_symbol=symbol)
    return TradeGenerationResult(trade_rows=trade_rows, completed_trades=completed_trades)


def build_oracle_labels(
    symbols: Sequence[str],
    *,
    spec: LabelBuildSpec,
    price_frames: Mapping[str, pl.DataFrame],
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


def deduplicate_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Keep one signal per date/symbol/side/action, preferring highest return."""

    if df.is_empty():
        return df
    subset = ["date", "symbol", "side"] + (["label"] if "label" in df.columns else [])
    return df.sort("trade_return", descending=True) .unique(subset, keep="first") if "trade_return" in df.columns else df.unique(subset, keep="first")


def _build_one_symbol_labels(args: tuple[Any, ...]) -> tuple[str, pl.DataFrame | None, str | None]:
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
        if events.is_empty():
            return (symbol, None, "no events produced")

        actions = add_action_labels(events)
        labels = add_binary_classification_labels(events, **weighting)
        labels = labels.with_columns(events["date"], actions["label"], actions["market_position"], pl.lit(symbol).alias("symbol"))
        for column in ("event", "trade_id", "entry_date", "exit_date", "entry_px", "exit_px", "trade_duration_days"):
            if column in events.columns:
                labels = labels.with_columns(events[column])
        if "trade_duration_days" in labels.columns and "hold_days" not in labels.columns:
            labels = labels.with_columns(pl.col("trade_duration_days").alias("hold_days"))
        return (symbol, labels, None)
    except Exception as exc:
        return (symbol, None, f"{type(exc).__name__}: {exc}")


def _event_rows(
    trade: Mapping[str, Any],
    *,
    freq: str,
    k: int,
    trade_counter: int,
    price_at: Callable[[Any], float],
    fee_bps: float,
    slippage_bps: float,
) -> list[dict[str, Any]]:
    side = str(trade.get("side") or "").strip().lower()
    if side not in {"long", "short"}:
        return []
    entry_dt = trade["entry_row"].get("date")
    exit_dt = trade["exit_row"].get("date")
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
        entry_dt = trade["entry_row"].get("date")
        exit_dt = trade["exit_row"].get("date")
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


def _get_price_series(df: pl.DataFrame, price_col: str = "close") -> pl.DataFrame:
    col_map = {str(col).lower(): col for col in df.columns}
    requested = str(price_col).lower()
    if requested in col_map:
        return df.select(["date", pl.col(col_map[requested]).cast(pl.Float64, strict=False).alias("price")]).sort("date")
    for fallback in ("close", "adj_close", "adjclose", "price", "adj_low", "low"):
        if fallback in col_map:
            return df.select(["date", pl.col(col_map[fallback]).cast(pl.Float64, strict=False).alias("price")]).sort("date")
    raise ValueError(f"Could not find a usable price column. Available: {list(df.columns)}")


def _slice_dates(df: pl.DataFrame | None, start_date: str | None, end_date: str | None) -> pl.DataFrame:
    if df is None or df.is_empty(): return pl.DataFrame()
    if "date" not in df.columns: raise ValueError("Price frames must have a date column")
    expr = pl.col("date").str.to_datetime(strict=False) if df.schema["date"] == pl.String else pl.col("date").cast(pl.Datetime, strict=False)
    out = df.with_columns(expr.dt.truncate("1d").alias("date")).drop_nulls("date").sort("date").unique("date", keep="last")
    if start_date: out = out.filter(pl.col("date") >= datetime.fromisoformat(start_date[:10]))
    if end_date: out = out.filter(pl.col("date") <= datetime.fromisoformat(end_date[:10]))
    return out


def _require_columns(df: pl.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
