from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import polars as pl

Side = Literal["long", "short"]
Frame = pl.DataFrame


@dataclass
class Trade:
    side: Side
    entry_row: dict[str, object]
    exit_row: dict[str, object]
    entry_price: float
    exit_price: float
    profit: float
    period_label: Optional[str] = None


def _resolve_freq(freq: str) -> Tuple[str, str]:
    freq_map = {"W": "W", "M": "ME", "ME": "ME", "QE": "QE", "YE": "YE"}
    label_freq_map = {"W": "W", "M": "M", "ME": "M", "QE": "Q", "YE": "Y"}
    return freq_map.get(freq, freq), label_freq_map.get(freq, "M")


def _pick_price_cols(side: Side, entry_price_col: Optional[str], exit_price_col: Optional[str]) -> Tuple[str, str]:
    if side == "long":
        return (entry_price_col or "high", exit_price_col or "low")
    return (entry_price_col or "low", exit_price_col or "high")


def _profit_pct(side: Side, entry: float, exit: float) -> float:
    if entry <= 0:
        return 0.0
    return (exit - entry) / entry if side == "long" else (entry - exit) / entry

def _solve_one_side_torch(entry_prices: Sequence[float], exit_prices: Sequence[float], k: int, min_profit_pct: float):
    """Torch DP solver used for the numerical strategy kernel."""
    ep = torch.as_tensor(entry_prices, dtype=torch.float64, device=DEVICE)
    xp = torch.as_tensor(exit_prices, dtype=torch.float64, device=DEVICE)
    n = int(ep.numel())
    cash_val = torch.zeros(k + 1, dtype=torch.float64, device=DEVICE)
    hold_val = torch.full((k + 1,), -torch.inf, dtype=torch.float64, device=DEVICE)
    hold_entry_day = torch.full((k + 1,), -1, dtype=torch.int64, device=DEVICE)
    hold_entry_px = torch.zeros(k + 1, dtype=torch.float64, device=DEVICE)
    cash_action = torch.zeros((n, k + 1), dtype=torch.int64, device=DEVICE)
    cash_entry_day = torch.zeros((n, k + 1), dtype=torch.int64, device=DEVICE)
    for i in range(n):
        for t in range(1, k + 1):
            if float(hold_val[t]) > float(-torch.inf):
                denom = float(abs(hold_entry_px[t]))
                pct = (float(xp[i]) - float(hold_entry_px[t])) / denom if denom > 0.0 else 0.0
                if pct >= min_profit_pct:
                    candidate = hold_val[t] + xp[i]
                    if float(candidate) > float(cash_val[t]) + 1e-12:
                        cash_val[t] = candidate
                        cash_action[i, t] = 1
                        cash_entry_day[i, t] = hold_entry_day[t]
        for t in range(1, k + 1):
            candidate = cash_val[t - 1] - ep[i]
            if float(candidate) > float(hold_val[t]):
                hold_val[t] = candidate
                hold_entry_day[t] = i
                hold_entry_px[t] = ep[i]
    best_t = int(torch.argmax(cash_val).item())
    trades: list[tuple[int, int]] = []
    t, i = best_t, n - 1
    while t > 0 and i >= 0 and len(trades) < k:
        if int(cash_action[i, t]) == 0:
            i -= 1
            continue
        entry_i = int(cash_entry_day[i, t])
        if entry_i < i:
            trades.append((entry_i, i))
            t -= 1
            i = entry_i - 1
        else:
            i -= 1
    trades.reverse()
    return torch.tensor(trades, dtype=torch.int64, device=DEVICE).tolist(), len(trades)


def _solve_one_side_all_k_torch(entry_prices: Sequence[float], exit_prices: Sequence[float], max_k: int, min_profit_pct: float):
    """Solve every transaction count with one shared dynamic-program pass.

    The previous implementation reran the complete ``n x k`` DP once for
    every requested k.  The state for k is independent but shares the same
    per-date observations, so one pass through the largest k produces all
    requested reconstructions without changing the recurrence or tie-breaks.
    """

    max_k = int(max_k)
    if max_k <= 0 or not entry_prices or not exit_prices:
        return [[[] for _ in range(max(0, max_k))] for _ in range(max(0, max_k) + 1)], [0] * (max(0, max_k) + 1)

    ep = torch.as_tensor(entry_prices, dtype=torch.float64, device=DEVICE)
    xp = torch.as_tensor(exit_prices, dtype=torch.float64, device=DEVICE)
    n = int(ep.numel())
    cash_val = torch.zeros(max_k + 1, dtype=torch.float64, device=DEVICE)
    hold_val = torch.full((max_k + 1,), -torch.inf, dtype=torch.float64, device=DEVICE)
    hold_entry_day = torch.full((max_k + 1,), -1, dtype=torch.int64, device=DEVICE)
    hold_entry_px = torch.zeros(max_k + 1, dtype=torch.float64, device=DEVICE)
    cash_action = torch.zeros((n, max_k + 1), dtype=torch.int64, device=DEVICE)
    cash_entry_day = torch.zeros((n, max_k + 1), dtype=torch.int64, device=DEVICE)

    for i in range(n):
        held = hold_val[1:]
        denom = torch.abs(hold_entry_px[1:])
        pct = torch.where(denom > 0.0, (xp[i] - hold_entry_px[1:]) / denom, torch.zeros_like(denom))
        candidate_cash = held + xp[i]
        update_cash = (
            torch.isfinite(held)
            & (pct >= min_profit_pct)
            & (candidate_cash > cash_val[1:] + 1e-12)
        )
        cash_val[1:] = torch.where(update_cash, candidate_cash, cash_val[1:])
        cash_action[i, 1:] = update_cash.to(torch.int64)
        cash_entry_day[i, 1:] = torch.where(update_cash, hold_entry_day[1:], cash_entry_day[i, 1:])

        candidate_hold = cash_val[:-1] - ep[i]
        update_hold = candidate_hold > hold_val[1:]
        hold_val[1:] = torch.where(update_hold, candidate_hold, hold_val[1:])
        hold_entry_day[1:] = torch.where(update_hold, torch.full_like(hold_entry_day[1:], i), hold_entry_day[1:])
        hold_entry_px[1:] = torch.where(update_hold, torch.ones_like(hold_entry_px[1:]) * ep[i], hold_entry_px[1:])

    def reconstruct(target_k: int) -> list[list[int]]:
        t = int(target_k)
        i = n - 1
        trades: list[tuple[int, int]] = []
        while t > 0 and i >= 0 and len(trades) < target_k:
            if int(cash_action[i, t]) == 0:
                i -= 1
                continue
            entry_i = int(cash_entry_day[i, t])
            if entry_i < i:
                trades.append((entry_i, i))
                t -= 1
                i = entry_i - 1
            else:
                i -= 1
        trades.reverse()
        return [[int(entry), int(exit)] for entry, exit in trades]

    results: list[list[list[int]]] = [[] for _ in range(max_k + 1)]
    counts: list[int] = [0] * (max_k + 1)
    for k in range(1, max_k + 1):
        trades = reconstruct(k)
        results[k] = trades
        counts[k] = len(trades)
    return results, counts


def _solve_one_side_all_k_torch_batch(
    price_sequences: Sequence[tuple[Sequence[float], Sequence[float]]],
    max_k: int,
    min_profit_pct: float,
) -> list[tuple[list[list[list[int]]], list[int]]]:
    """Run the shared DP for a bounded batch of variable-length sequences."""

    if not price_sequences:
        return []
    max_k = int(max_k)
    lengths = [min(len(entry), len(exit_)) for entry, exit_ in price_sequences]
    batch_size = len(lengths)
    max_n = max(lengths, default=0)
    if max_k <= 0 or max_n == 0:
        return [([], [0] * (max(0, max_k) + 1)) for _ in lengths]

    padded_entry = [
        [float(value) for value in entry[:length]] + [0.0] * (max_n - length)
        for (entry, _), length in zip(price_sequences, lengths)
    ]
    padded_exit = [
        [float(value) for value in exit_[:length]] + [0.0] * (max_n - length)
        for (_, exit_), length in zip(price_sequences, lengths)
    ]
    ep = torch.tensor(padded_entry, dtype=torch.float64, device=DEVICE)
    xp = torch.tensor(padded_exit, dtype=torch.float64, device=DEVICE)
    lengths_t = torch.tensor(lengths, dtype=torch.int64, device=DEVICE)
    cash_val = torch.zeros((batch_size, max_k + 1), dtype=torch.float64, device=DEVICE)
    hold_val = torch.full((batch_size, max_k + 1), -torch.inf, dtype=torch.float64, device=DEVICE)
    hold_entry_day = torch.full((batch_size, max_k + 1), -1, dtype=torch.int64, device=DEVICE)
    hold_entry_px = torch.zeros((batch_size, max_k + 1), dtype=torch.float64, device=DEVICE)
    cash_action = torch.zeros((batch_size, max_n, max_k + 1), dtype=torch.int64, device=DEVICE)
    cash_entry_day = torch.zeros((batch_size, max_n, max_k + 1), dtype=torch.int64, device=DEVICE)

    for i in range(max_n):
        active = lengths_t > i
        held = hold_val[:, 1:]
        denom = torch.abs(hold_entry_px[:, 1:])
        pct = torch.where(denom > 0.0, (xp[:, i, None] - hold_entry_px[:, 1:]) / denom, torch.zeros_like(denom))
        candidate_cash = held + xp[:, i, None]
        update_cash = (
            active[:, None]
            & torch.isfinite(held)
            & (pct >= min_profit_pct)
            & (candidate_cash > cash_val[:, 1:] + 1e-12)
        )
        cash_val[:, 1:] = torch.where(update_cash, candidate_cash, cash_val[:, 1:])
        cash_action[:, i, 1:] = update_cash.to(torch.int64)
        cash_entry_day[:, i, 1:] = torch.where(update_cash, hold_entry_day[:, 1:], cash_entry_day[:, i, 1:])

        candidate_hold = cash_val[:, :-1] - ep[:, i, None]
        update_hold = active[:, None] & (candidate_hold > hold_val[:, 1:])
        hold_val[:, 1:] = torch.where(update_hold, candidate_hold, hold_val[:, 1:])
        hold_entry_day[:, 1:] = torch.where(
            update_hold,
            torch.full_like(hold_entry_day[:, 1:], i),
            hold_entry_day[:, 1:],
        )
        hold_entry_px[:, 1:] = torch.where(
            update_hold,
            torch.ones_like(hold_entry_px[:, 1:]) * ep[:, i, None],
            hold_entry_px[:, 1:],
        )

    results: list[tuple[list[list[list[int]]], list[int]]] = []
    for batch_index, length in enumerate(lengths):
        per_k: list[list[list[int]]] = [[] for _ in range(max_k + 1)]
        counts = [0] * (max_k + 1)
        for k in range(1, max_k + 1):
            t = k
            i = length - 1
            trades: list[tuple[int, int]] = []
            while t > 0 and i >= 0 and len(trades) < k:
                if int(cash_action[batch_index, i, t]) == 0:
                    i -= 1
                    continue
                entry_i = int(cash_entry_day[batch_index, i, t])
                if entry_i < i:
                    trades.append((entry_i, i))
                    t -= 1
                    i = entry_i - 1
                else:
                    i -= 1
            trades.reverse()
            per_k[k] = [[int(entry), int(exit_)] for entry, exit_ in trades]
            counts[k] = len(trades)
        results.append((per_k, counts))
    return results


def solve_optimal_trades_generic(
    df: Frame,
    k: int,
    side: Side = "long",
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
    min_profit_pct: float = 0.01,
) -> List[Trade]:
    if k <= 0 or df is None or len(df) < 2:
        return []

    entry_col, exit_col = _pick_price_cols(side, entry_price_col, exit_price_col)
    col_map = {str(c).lower(): c for c in df.columns}

    def _resolve_col(col: str) -> str:
        key = str(col).lower()
        if key in col_map:
            return col_map[key]
        raise ValueError(f"Missing column '{col}' (needed by solver)")

    entry_col = _resolve_col(entry_col)
    exit_col = _resolve_col(exit_col)

    entry_prices = df[entry_col].cast(pl.Float64, strict=False).to_list()
    exit_prices = df[exit_col].cast(pl.Float64, strict=False).to_list()

    if side == "short":
        ep = [-value for value in entry_prices]
        xp = [-value for value in exit_prices]
    else:
        ep = entry_prices
        xp = exit_prices

    # Torch owns the numerical DP kernel; Trade retains row-oriented metadata
    # for callers that need the selected entry and exit observations.
    trades_arr, n_trades = _solve_one_side_torch(
        ep, xp, k=k, min_profit_pct=min_profit_pct,
    )

    out: List[Trade] = []
    for idx in range(n_trades):
        entry_i = int(trades_arr[idx][0])
        exit_i = int(trades_arr[idx][1])
        raw_entry = float(entry_prices[entry_i])
        raw_exit = float(exit_prices[exit_i])
        profit_pct = _profit_pct(side, raw_entry, raw_exit)
        if profit_pct < float(min_profit_pct):
            continue
        profit = raw_exit - raw_entry if side == "long" else raw_entry - raw_exit
        out.append(
            Trade(
                side=side,
                entry_row=df.row(entry_i, named=True),
                exit_row=df.row(exit_i, named=True),
                entry_price=raw_entry,
                exit_price=raw_exit,
                profit=profit,
            )
        )
    return out

def solve_optimal_trades_all_k_generic(
    df: Frame,
    ks: Sequence[int],
    side: Side = "long",
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
    min_profit_pct: float = 0.01,
) -> dict[int, List[Trade]]:
    normalized_ks = tuple(dict.fromkeys(int(k) for k in ks if int(k) > 0))
    if not normalized_ks:
        return {}
    if df is None or len(df) < 2:
        return {k: [] for k in normalized_ks}

    max_k = max(normalized_ks)
    entry_col, exit_col = _pick_price_cols(side, entry_price_col, exit_price_col)
    col_map = {str(c).lower(): c for c in df.columns}

    def _resolve_col(col: str) -> str:
        key = str(col).lower()
        if key in col_map:
            return col_map[key]
        raise ValueError(f"Missing column '{col}' (needed by solver)")

    entry_col = _resolve_col(entry_col)
    exit_col = _resolve_col(exit_col)

    entry_prices = df[entry_col].cast(pl.Float64, strict=False).to_list()
    exit_prices = df[exit_col].cast(pl.Float64, strict=False).to_list()
    if side == "short":
        ep = [-value for value in entry_prices]
        xp = [-value for value in exit_prices]
    else:
        ep = entry_prices
        xp = exit_prices

    trades_by_k, counts = _solve_one_side_all_k_torch(
        ep,
        xp,
        max_k=max_k,
        min_profit_pct=float(min_profit_pct),
    )

    out: dict[int, List[Trade]] = {}
    for k in normalized_ks:
        trades: List[Trade] = []
        for idx in range(int(counts[k])):
            entry_i = int(trades_by_k[k][idx][0])
            exit_i = int(trades_by_k[k][idx][1])
            if entry_i < 0 or exit_i < 0:
                continue
            raw_entry = float(entry_prices[entry_i])
            raw_exit = float(exit_prices[exit_i])
            profit_pct = _profit_pct(side, raw_entry, raw_exit)
            if profit_pct < float(min_profit_pct):
                continue
            profit = raw_exit - raw_entry if side == "long" else raw_entry - raw_exit
            trades.append(
                Trade(
                    side=side,
                    entry_row=df.row(entry_i, named=True),
                    exit_row=df.row(exit_i, named=True),
                    entry_price=raw_entry,
                    exit_price=raw_exit,
                    profit=profit,
                )
            )
        out[k] = trades
    return out


def solve_trades_by_frequency(
    df: Frame,
    k: int,
    freq: str = "QE",
    side: Side = "long",
    min_profit_pct: float = 0.01,
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
) -> List[Dict]:
    if df is None or df.is_empty():
        return []
    if "date" not in df.columns:
        raise ValueError("solve_trades_by_frequency requires a 'date' column")
    _, label_freq = _resolve_freq(freq)
    dfi = df.with_columns(pl.col("date").cast(pl.Datetime, strict=False).alias("date")).drop_nulls("date").sort("date")
    dfi = dfi.with_columns(pl.col("date").map_elements(lambda value: _period_start(value, freq), return_dtype=pl.Datetime).alias("_period"))
    all_trades: List[Dict] = []
    for period_group in dfi.partition_by("_period", maintain_order=True):
        if period_group.height < 2:
            continue
        period = period_group["_period"][0]
        period_label = f"{label_freq}:{period.date()}"
        group = period_group.drop("_period")
        trades = solve_optimal_trades_generic(
            group,
            k=k,
            side=side,
            min_profit_pct=min_profit_pct,
            entry_price_col=entry_price_col,
            exit_price_col=exit_price_col,
        )
        for trade in trades:
            all_trades.append(
                {
                    "side": trade.side,
                    "entry_row": trade.entry_row,
                    "exit_row": trade.exit_row,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "profit": trade.profit,
                    "period_label": period_label,
                }
            )
    return all_trades


def solve_longs_by_frequency(
    df: Frame,
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
) -> List[Dict]:
    return solve_trades_by_frequency(
        df,
        k=k,
        freq=freq,
        side="long",
        min_profit_pct=min_profit_pct,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
    )


def solve_shorts_by_frequency(
    df: Frame,
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
) -> List[Dict]:
    return solve_trades_by_frequency(
        df,
        k=k,
        freq=freq,
        side="short",
        min_profit_pct=min_profit_pct,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
    )


def _resolve_frame_column(df: Frame, col: str) -> str:
    col_map = {str(c).lower(): c for c in df.columns}
    key = str(col).lower()
    if key in col_map:
        return col_map[key]
    raise ValueError(f"Missing column '{col}' (needed by solver)")


def _normalize_frame_for_batch(df: Frame) -> Frame:
    if df is None or df.is_empty(): return pl.DataFrame()
    if "date" not in df.columns: raise ValueError("Batch solver requires a 'date' column")
    return df.with_columns(pl.col("date").cast(pl.Datetime, strict=False).alias("date")).drop_nulls("date").sort("date").unique("date", keep="last")


def _period_label(period: Any, label_freq: str) -> str:
    return f"{label_freq}:{period.date()}" if isinstance(period, datetime) else str(period)


def _period_start(value: datetime, freq: str) -> datetime:
    if freq == "W": return value - timedelta(days=value.weekday())
    if freq in {"M", "ME"}: return value.replace(day=1)
    if freq in {"QE", "Q"}: return value.replace(month=((value.month - 1) // 3) * 3 + 1, day=1)
    if freq in {"YE", "Y"}: return value.replace(month=1, day=1)
    return value


def solve_side_trades_by_frequency_batched_multi_k(
    price_frames: Mapping[str, Frame],
    ks: Sequence[int],
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    sides: Sequence[Side] = ("long", "short"),
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
) -> dict[int, dict[str, List[Dict]]]:
    normalized_ks = tuple(dict.fromkeys(int(k) for k in ks if int(k) > 0))
    if not price_frames or not normalized_ks:
        return {}

    freq_resolved, label_freq = _resolve_freq(freq)
    symbols = [
        str(symbol).strip().upper()
        for symbol, frame in price_frames.items()
        if str(symbol).strip() and frame is not None and not frame.is_empty()
    ]
    results: dict[int, dict[str, List[Dict]]] = {
        k: {symbol: [] for symbol in symbols}
        for k in normalized_ks
    }
    period_buckets: dict[str, list[tuple[str, Frame]]] = {}
    period_order: dict[str, datetime] = {}
    normalized_sides = tuple(dict.fromkeys(side for side in sides if side in {"long", "short"}))
    if not normalized_sides:
        return results

    for symbol, frame in price_frames.items():
        symbol_name = str(symbol).strip().upper()
        if frame is None or frame.is_empty() or not symbol_name:
            continue
        dfi = _normalize_frame_for_batch(frame)
        if dfi.is_empty():
            continue
        dfi = dfi.with_columns(pl.col("date").map_elements(lambda value: _period_start(value, freq), return_dtype=pl.Datetime).alias("_period"))
        for group in dfi.partition_by("_period", maintain_order=True):
            if group.height < 2:
                continue
            period = group["_period"][0]
            label = _period_label(period, label_freq)
            if label not in period_buckets:
                period_buckets[label] = []
                try:
                    period_order[label] = period
                except Exception:
                    period_order[label] = datetime.min
            period_buckets[label].append((symbol_name, group.drop("_period")))

    if not period_buckets:
        return results

    le_col_hint = long_entry_price_col or "high"
    lx_col_hint = long_exit_price_col or "low"
    se_col_hint = short_entry_price_col or "low"
    sx_col_hint = short_exit_price_col or "high"

    task_frames: list[Frame] = []
    task_symbols: list[str] = []
    task_labels: list[str] = []
    task_sides: list[Side] = []
    task_columns: list[tuple[str, str]] = []

    for label in sorted(period_buckets, key=lambda item: period_order.get(item, datetime.min)):
        for symbol, group in period_buckets[label]:
            if "long" in normalized_sides:
                task_symbols.append(symbol)
                task_frames.append(group)
                task_labels.append(label)
                task_sides.append("long")
                task_columns.append((_resolve_frame_column(group, le_col_hint), _resolve_frame_column(group, lx_col_hint)))
            if "short" in normalized_sides:
                task_symbols.append(symbol)
                task_frames.append(group)
                task_labels.append(label)
                task_sides.append("short")
                task_columns.append((_resolve_frame_column(group, se_col_hint), _resolve_frame_column(group, sx_col_hint)))

    if not task_frames:
        return results

    batch_size = 256
    for side in normalized_sides:
        side_indices = [index for index, task_side in enumerate(task_sides) if task_side == side]
        for batch_start in range(0, len(side_indices), batch_size):
            batch_indices = side_indices[batch_start : batch_start + batch_size]
            sequences: list[tuple[list[float], list[float]]] = []
            for task_idx in batch_indices:
                group = task_frames[task_idx]
                entry_col, exit_col = task_columns[task_idx]
                entry_prices = group[entry_col].cast(pl.Float64, strict=False).to_list()
                exit_prices = group[exit_col].cast(pl.Float64, strict=False).to_list()
                if side == "short":
                    entry_prices = [-value for value in entry_prices]
                    exit_prices = [-value for value in exit_prices]
                sequences.append((entry_prices, exit_prices))

            batch_results = _solve_one_side_all_k_torch_batch(
                sequences,
                max_k=max(normalized_ks),
                min_profit_pct=float(min_profit_pct),
            )
            for task_idx, (trades_by_k, counts) in zip(batch_indices, batch_results):
                symbol = task_symbols[task_idx]
                group = task_frames[task_idx]
                entry_col, exit_col = task_columns[task_idx]
                raw_entry_prices = group[entry_col].cast(pl.Float64, strict=False).to_list()
                raw_exit_prices = group[exit_col].cast(pl.Float64, strict=False).to_list()
                for k in normalized_ks:
                    result_for_k = results.setdefault(int(k), {symbol: [] for symbol in symbols})
                    for entry_i, exit_i in trades_by_k[k][: int(counts[k])]:
                        raw_entry = float(raw_entry_prices[entry_i])
                        raw_exit = float(raw_exit_prices[exit_i])
                        profit = raw_exit - raw_entry if side == "long" else raw_entry - raw_exit
                        result_for_k.setdefault(symbol, []).append(
                            {
                                "side": side,
                                "entry_row": group.row(entry_i, named=True),
                                "exit_row": group.row(exit_i, named=True),
                                "entry_price": raw_entry,
                                "exit_price": raw_exit,
                                "profit": profit,
                                "period_label": task_labels[task_idx],
                            }
                        )

    return results
