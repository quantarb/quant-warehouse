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
    results = torch.full((max_k + 1, max_k, 2), -1, dtype=torch.int64, device=DEVICE)
    counts = torch.zeros(max_k + 1, dtype=torch.int64, device=DEVICE)
    for k in range(1, max_k + 1):
        trades, count = _solve_one_side_torch(entry_prices, exit_prices, k, min_profit_pct)
        if count:
            results[k, :count] = torch.as_tensor(trades, dtype=torch.int64, device=DEVICE)
        counts[k] = count
    return results.tolist(), counts.tolist()


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

    for task_idx, symbol in enumerate(task_symbols):
        group = task_frames[task_idx]
        side = task_sides[task_idx]
        entry_col, exit_col = task_columns[task_idx]
        trades_by_k = solve_optimal_trades_all_k_generic(
            group,
            ks=normalized_ks,
            side=side,
            min_profit_pct=min_profit_pct,
            entry_price_col=entry_col,
            exit_price_col=exit_col,
        )
        for k, trades in trades_by_k.items():
            result_for_k = results.setdefault(int(k), {symbol: [] for symbol in symbols})
            for trade in trades:
                result_for_k.setdefault(symbol, []).append(
                    {
                        "side": trade.side,
                        "entry_row": trade.entry_row,
                        "exit_row": trade.exit_row,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "profit": trade.profit,
                        "period_label": task_labels[task_idx],
                    }
                )

    return results
