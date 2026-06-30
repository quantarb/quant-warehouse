from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator when numba is not available."""
        def _decorator(fn):
            return fn
        return _decorator

Side = Literal["long", "short"]


@dataclass
class Trade:
    side: Side
    entry_row: pd.Series
    exit_row: pd.Series
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
    if side == "long":
        return (exit - entry) / entry
    return (entry - exit) / entry


def _resolve_required_col(df: pd.DataFrame, col: str) -> str:
    col_map = {str(c).lower(): c for c in df.columns}
    key = str(col).lower()
    if key in col_map:
        return col_map[key]
    raise ValueError(f"Missing column '{col}' (needed by solver)")


@njit
def _solve_one_side_numba(entry_prices, exit_prices, k, min_profit_pct):
    """Numba-accelerated one-side DP solver core.

    Returns (trades_arr, n_trades) where trades_arr is (entry_idx, exit_idx) pairs
    in chronological order, and n_trades is the count.
    """
    n = len(entry_prices)

    cash_val = np.zeros(k + 1, dtype=np.float64)
    hold_val = np.full(k + 1, -np.inf, dtype=np.float64)
    hold_entry_day = np.full(k + 1, -1, dtype=np.int32)
    hold_entry_px = np.zeros(k + 1, dtype=np.float64)
    cash_action = np.zeros((n, k + 1), dtype=np.int32)
    cash_entry_day = np.zeros((n, k + 1), dtype=np.int32)

    for i in range(n):
        ep = entry_prices[i]
        xp = exit_prices[i]

        for t in range(1, k + 1):
            hv = hold_val[t]
            if hv > -np.inf:
                entry_denom = hold_entry_px[t] if hold_entry_px[t] > 0.0 else -hold_entry_px[t]
                pct = (xp - hold_entry_px[t]) / entry_denom if entry_denom > 0.0 else 0.0
                if pct >= min_profit_pct:
                    cand_cash = hv + xp
                    if cand_cash > cash_val[t] + 1e-12:
                        cash_val[t] = cand_cash
                        cash_action[i, t] = 1
                        cash_entry_day[i, t] = hold_entry_day[t]

        for t in range(1, k + 1):
            cand_hold = cash_val[t - 1] - ep
            if cand_hold > hold_val[t]:
                hold_val[t] = cand_hold
                hold_entry_day[t] = i
                hold_entry_px[t] = ep

    best_t = 0
    best_val = 0.0
    for t in range(k + 1):
        if cash_val[t] > best_val + 1e-12:
            best_val = cash_val[t]
            best_t = t

    # Back-tracking returns most recent first.
    trades_out = np.zeros((k, 2), dtype=np.int32)
    n_trades = 0
    t = best_t
    i = n - 1
    while t > 0 and i >= 0 and n_trades < k:
        if cash_action[i, t] == 0:
            i -= 1
            continue

        entry_idx = cash_entry_day[i, t]
        if entry_idx < i:
            trades_out[n_trades, 0] = entry_idx
            trades_out[n_trades, 1] = i
            n_trades += 1
            t -= 1
            i = entry_idx - 1
        else:
            i -= 1

    # Reverse to chronological order.
    result = np.zeros((n_trades, 2), dtype=np.int32)
    for idx in range(n_trades):
        result[idx, 0] = trades_out[n_trades - 1 - idx, 0]
        result[idx, 1] = trades_out[n_trades - 1 - idx, 1]
    return result, n_trades


def solve_optimal_trades_generic(
    df: pd.DataFrame,
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

    entry_prices = df[entry_col].astype(float).values
    exit_prices = df[exit_col].astype(float).values

    if side == "short":
        ep = -entry_prices
        xp = -exit_prices
    else:
        ep = entry_prices
        xp = exit_prices

    # --- Try numba-accelerated path ---
    if _HAS_NUMBA:
        trades_arr, n_trades = _solve_one_side_numba(
            ep.astype(np.float64), xp.astype(np.float64),
            k=k, min_profit_pct=min_profit_pct,
        )

        out: List[Trade] = []
        for idx in range(n_trades):
            entry_i = int(trades_arr[idx, 0])
            exit_i = int(trades_arr[idx, 1])
            raw_entry = float(entry_prices[entry_i])
            raw_exit = float(exit_prices[exit_i])
            profit_pct = _profit_pct(side, raw_entry, raw_exit)
            if profit_pct < float(min_profit_pct):
                continue
            if side == "long":
                profit = raw_exit - raw_entry
                entry_price = raw_entry
                exit_price = raw_exit
            else:
                profit = raw_entry - raw_exit
                entry_price = raw_entry
                exit_price = raw_exit
            out.append(
                Trade(
                    side=side,
                    entry_row=df.iloc[entry_i],
                    exit_row=df.iloc[exit_i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    profit=profit,
                )
            )
        return out

    # --- Pure-Python fallback ---
    n = len(df)

    cash = [0.0] * (k + 1)
    hold = [float("-inf")] * (k + 1)
    hold_entry_day = [-1] * (k + 1)
    hold_entry_px = [0.0] * (k + 1)
    cash_action = [[0] * (k + 1) for _ in range(n)]
    cash_entry_day = [[0] * (k + 1) for _ in range(n)]

    for i in range(n):
        ep_i = float(ep[i])
        xp_i = float(xp[i])
        for trade_count in range(1, k + 1):
            hold_value = hold[trade_count]
            if hold_value > float("-inf"):
                entry_denom = abs(hold_entry_px[trade_count])
                profit_pct = ((xp_i - hold_entry_px[trade_count]) / entry_denom) if entry_denom > 0 else 0.0
                if profit_pct >= float(min_profit_pct):
                    cand_cash = hold_value + xp_i
                    if cand_cash > cash[trade_count] + 1e-12:
                        cash[trade_count] = cand_cash
                        cash_action[i][trade_count] = 1
                        cash_entry_day[i][trade_count] = hold_entry_day[trade_count]

        for trade_count in range(1, k + 1):
            cand_hold = cash[trade_count - 1] - ep_i
            if cand_hold > hold[trade_count]:
                hold[trade_count] = cand_hold
                hold_entry_day[trade_count] = i
                hold_entry_px[trade_count] = ep_i

    trades_idx: List[Tuple[int, int]] = []
    trade_count = max(range(k + 1), key=lambda value: cash[value])
    i = n - 1
    while trade_count > 0 and i >= 0:
        if cash_action[i][trade_count] == 0:
            i -= 1
            continue

        entry_idx = cash_entry_day[i][trade_count]
        if entry_idx < i:
            trades_idx.append((entry_idx, i))
            i = entry_idx - 1
            trade_count -= 1
        else:
            i -= 1

    trades_idx.reverse()
    out: List[Trade] = []
    for entry_i, exit_i in trades_idx:
        raw_entry = float(df[entry_col].iloc[entry_i])
        raw_exit = float(df[exit_col].iloc[exit_i])
        profit_pct = _profit_pct(side, raw_entry, raw_exit)
        if profit_pct < float(min_profit_pct):
            continue
        if side == "long":
            profit = raw_exit - raw_entry
            entry_price = raw_entry
            exit_price = raw_exit
        else:
            profit = raw_entry - raw_exit
            entry_price = raw_entry
            exit_price = raw_exit
        out.append(
            Trade(
                side=side,
                entry_row=df.iloc[entry_i],
                exit_row=df.iloc[exit_i],
                entry_price=entry_price,
                exit_price=exit_price,
                profit=profit,
            )
        )
    return out


def solve_trades_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    side: Side = "long",
    min_profit_pct: float = 0.01,
    entry_price_col: Optional[str] = None,
    exit_price_col: Optional[str] = None,
) -> List[Dict]:
    if df is None or df.empty:
        return []
    freq_resolved, label_freq = _resolve_freq(freq)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            dfi = df.copy()
            dfi["date"] = pd.to_datetime(dfi["date"], errors="coerce")
            dfi = dfi.set_index("date")
        else:
            raise ValueError("solve_trades_by_frequency requires a DatetimeIndex or a 'date' column")
    else:
        dfi = df

    dfi = dfi.sort_index()
    all_trades: List[Dict] = []
    for period, group in dfi.groupby(pd.Grouper(freq=freq_resolved)):
        if group is None or len(group) < 2:
            continue
        try:
            ts = period.to_timestamp() if hasattr(period, "to_timestamp") else pd.to_datetime(period)
            period_label = f"{label_freq}:{ts.date()}"
        except Exception:
            period_label = str(period)
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
    df: pd.DataFrame,
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
    df: pd.DataFrame,
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


def _resolve_frame_column(df: pd.DataFrame, col: str) -> str:
    col_map = {str(c).lower(): c for c in df.columns}
    key = str(col).lower()
    if key in col_map:
        return col_map[key]
    raise ValueError(f"Missing column '{col}' (needed by solver)")


def _normalize_frame_for_batch(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" not in out.columns:
            raise ValueError("Batch solver requires a DatetimeIndex or a 'date' column")
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date")
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def _period_label(period: Any, label_freq: str) -> str:
    try:
        ts = period.to_timestamp() if hasattr(period, "to_timestamp") else pd.to_datetime(period)
        return f"{label_freq}:{ts.date()}"
    except Exception:
        return str(period)


def solve_side_trades_by_frequency_batched_multi_k(
    price_frames: Mapping[str, pd.DataFrame],
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
        if str(symbol).strip() and frame is not None and not frame.empty
    ]
    results: dict[int, dict[str, List[Dict]]] = {
        k: {symbol: [] for symbol in symbols}
        for k in normalized_ks
    }
    period_buckets: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    period_order: dict[str, pd.Timestamp] = {}
    normalized_sides = tuple(dict.fromkeys(side for side in sides if side in {"long", "short"}))
    if not normalized_sides:
        return results

    for symbol, frame in price_frames.items():
        symbol_name = str(symbol).strip().upper()
        if frame is None or frame.empty or not symbol_name:
            continue
        dfi = _normalize_frame_for_batch(frame)
        if dfi.empty:
            continue
        for period, group in dfi.groupby(pd.Grouper(freq=freq_resolved)):
            if group is None or len(group) < 2:
                continue
            label = _period_label(period, label_freq)
            if label not in period_buckets:
                period_buckets[label] = []
                try:
                    period_order[label] = period.to_timestamp() if hasattr(period, "to_timestamp") else pd.to_datetime(period)
                except Exception:
                    period_order[label] = pd.Timestamp.min
            period_buckets[label].append((symbol_name, group))

    if not period_buckets:
        return results

    le_col_hint = long_entry_price_col or "high"
    lx_col_hint = long_exit_price_col or "low"
    se_col_hint = short_entry_price_col or "low"
    sx_col_hint = short_exit_price_col or "high"

    task_frames: list[pd.DataFrame] = []
    task_symbols: list[str] = []
    task_labels: list[str] = []
    task_sides: list[Side] = []
    task_columns: list[tuple[str, str]] = []

    for label in sorted(period_buckets, key=lambda item: period_order.get(item, pd.Timestamp.min)):
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

    for k in normalized_ks:
        result_for_k = results.setdefault(k, {symbol: [] for symbol in symbols})
        for task_idx, symbol in enumerate(task_symbols):
            group = task_frames[task_idx]
            side = task_sides[task_idx]
            entry_col, exit_col = task_columns[task_idx]
            trades = solve_optimal_trades_generic(
                group,
                k=int(k),
                side=side,
                min_profit_pct=min_profit_pct,
                entry_price_col=entry_col,
                exit_price_col=exit_col,
            )
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
