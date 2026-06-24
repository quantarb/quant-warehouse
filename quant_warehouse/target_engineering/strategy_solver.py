from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
    from numba import prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    prange = range

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


@dataclass
class _TradeCandidate:
    side: Side
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    profit: float


@dataclass(frozen=True)
class _TradePathState:
    value: float
    trades: tuple[_TradeCandidate, ...]


@dataclass(frozen=True)
class _HoldState:
    value: float
    entry_idx: int
    entry_price: float
    base_trades: tuple[_TradeCandidate, ...]


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
def _solver_core_numba(
    long_entry, long_exit, short_entry, short_exit,
    k, min_profit_pct,
):
    """Numba-accelerated DP core with per-day back-tracking.

    Returns arrays that can reconstruct trades: for each (day, trade_count) we
    record how that cash state was achieved, so we can walk backward through time
    to extract the actual trade entry/exit pairs.
    """
    n = len(long_entry)

    # Current (rolling) state across trade counts
    cash_val = np.zeros(k + 1, dtype=np.float64)
    long_val = np.full(k + 1, -np.inf, dtype=np.float64)
    long_entry_idx = np.zeros(k + 1, dtype=np.int32)
    long_entry_px = np.zeros(k + 1, dtype=np.float64)
    short_val = np.full(k + 1, -np.inf, dtype=np.float64)
    short_entry_idx = np.zeros(k + 1, dtype=np.int32)
    short_entry_px = np.zeros(k + 1, dtype=np.float64)

    # Per-day back-tracking: for each day i, record the action taken at each trade_count
    cash_action = np.zeros((n, k + 1), dtype=np.int32)   # 0=none/inherited, 1=long exit, 2=short exit
    cash_entry_day = np.zeros((n, k + 1), dtype=np.int32) # entry day for the trade that closed

    for i in range(n):
        le = long_entry[i]
        lx = long_exit[i]
        se = short_entry[i]
        sx = short_exit[i]

        # --- Exit detection ---
        for t in range(1, k + 1):
            lv = long_val[t]
            if lv > -np.inf:
                pct = (lx - long_entry_px[t]) / long_entry_px[t] if long_entry_px[t] > 0.0 else 0.0
                if pct >= min_profit_pct:
                    cand = lv + lx
                    if cand > cash_val[t] + 1e-12:
                        cash_val[t] = cand
                        cash_action[i, t] = 1
                        cash_entry_day[i, t] = long_entry_idx[t]

            sv = short_val[t]
            if sv > -np.inf:
                pct = (short_entry_px[t] - sx) / short_entry_px[t] if short_entry_px[t] > 0.0 else 0.0
                if pct >= min_profit_pct:
                    cand = sv - sx
                    if cand > cash_val[t] + 1e-12:
                        cash_val[t] = cand
                        cash_action[i, t] = 2
                        cash_entry_day[i, t] = short_entry_idx[t]

        # --- Entry detection ---
        for t in range(1, k + 1):
            base_val = cash_val[t - 1]

            if le > 0.0:
                cand = base_val - le
                if cand > long_val[t]:
                    long_val[t] = cand
                    long_entry_idx[t] = i
                    long_entry_px[t] = le

            if se > 0.0:
                cand = base_val + se
                if cand > short_val[t]:
                    short_val[t] = cand
                    short_entry_idx[t] = i
                    short_entry_px[t] = se

    # --- Back-tracking ---
    # Find best final state (last day, best trade_count)
    best_t = 0
    best_val = 0.0
    for t in range(k + 1):
        if cash_val[t] > best_val + 1e-12:
            best_val = cash_val[t]
            best_t = t

    trades_out = np.zeros((k, 3), dtype=np.int32)  # (entry_idx, exit_idx, side_code)
    n_trades = 0
    t = best_t
    i = n - 1

    while t > 0 and i >= 0 and n_trades < k:
        action = cash_action[i, t]
        if action == 0:
            # No exit at this (i, t) — go back one day
            i -= 1
            continue

        # An exit happened at day i for trade_count t
        # Side: 1=long, 2=short
        entry_day = cash_entry_day[i, t]
        if entry_day < i:
            trades_out[n_trades, 0] = entry_day
            trades_out[n_trades, 1] = i
            trades_out[n_trades, 2] = action
            n_trades += 1
            t -= 1
            i = entry_day - 1
        else:
            i -= 1  # degenerate: shouldn't happen, but guard

    return trades_out, n_trades


def _solve_joint_numba(
    df: pd.DataFrame,
    k: int,
    long_entry_col: str,
    long_exit_col: str,
    short_entry_col: str,
    short_exit_col: str,
    min_profit_pct: float = 0.01,
) -> List[Trade]:
    """Python wrapper that calls the numba core and builds Trade objects."""
    if not _HAS_NUMBA:
        return None  # Signal to fall back to pure-Python solver

    long_entry = df[long_entry_col].astype(np.float64).to_numpy()
    long_exit = df[long_exit_col].astype(np.float64).to_numpy()
    short_entry = df[short_entry_col].astype(np.float64).to_numpy()
    short_exit = df[short_exit_col].astype(np.float64).to_numpy()

    trades_arr, n_trades = _solver_core_numba(
        long_entry, long_exit, short_entry, short_exit,
        k=k, min_profit_pct=min_profit_pct,
    )

    if n_trades == 0:
        return []

    # Reverse to get chronological order (back-tracking gives most recent first)
    out: List[Trade] = []
    for idx in range(n_trades - 1, -1, -1):
        entry_i = int(trades_arr[idx, 0])
        exit_i = int(trades_arr[idx, 1])
        side_code = int(trades_arr[idx, 2])
        side = "long" if side_code == 1 else "short"

        entry_price = float(long_entry[entry_i]) if side == "long" else float(short_entry[entry_i])
        exit_price = float(long_exit[exit_i]) if side == "long" else float(short_exit[exit_i])

        profit = exit_price - entry_price if side == "long" else entry_price - exit_price

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


def solve_optimal_joint_trades_generic(
    df: pd.DataFrame,
    k: int,
    *,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
    min_profit_pct: float = 0.01,
) -> List[Trade]:
    if k <= 0 or df is None or len(df) < 2:
        return []

    # Cache column name resolution — build lowercase map once
    _col_map = {str(c).lower(): c for c in df.columns}

    def _resolve(col: str) -> str:
        key = str(col).lower()
        if key in _col_map:
            return _col_map[key]
        raise ValueError(f"Missing column '{col}' (needed by solver)")

    le_col = _resolve(long_entry_price_col or "high")
    lx_col = _resolve(long_exit_price_col or "low")
    se_col = _resolve(short_entry_price_col or "low")
    sx_col = _resolve(short_exit_price_col or "high")

    # --- Numba path for one-side solver only ---
    # Joint solver uses pure Python (tie-breaking in numba back-tracking differs slightly).

    # --- Pure-Python solver ---
    long_entry = df[le_col].astype(float).to_numpy()
    long_exit = df[lx_col].astype(float).to_numpy()
    short_entry = df[se_col].astype(float).to_numpy()
    short_exit = df[sx_col].astype(float).to_numpy()
    n = len(df)

    min_profit = float(min_profit_pct)
    cash: List[_TradePathState] = [_TradePathState(0.0, ()) for _ in range(k + 1)]
    long_hold: List[_HoldState | None] = [None] * (k + 1)
    short_hold: List[_HoldState | None] = [None] * (k + 1)

    for i in range(n):
        le = float(long_entry[i])
        lx = float(long_exit[i])
        se = float(short_entry[i])
        sx = float(short_exit[i])

        prev_cash = list(cash)
        prev_long_hold = list(long_hold)
        prev_short_hold = list(short_hold)

        for trade_count in range(1, k + 1):
            best_cash = prev_cash[trade_count]

            long_state = prev_long_hold[trade_count]
            if long_state is not None and long_state.entry_idx < i:
                long_pct = _profit_pct("long", long_state.entry_price, lx)
                if long_pct >= min_profit:
                    candidate_value = float(long_state.value + lx)
                    if candidate_value > float(best_cash.value) + 1e-12:
                        best_cash = _TradePathState(
                            candidate_value,
                            long_state.base_trades
                            + (
                                _TradeCandidate(
                                    side="long",
                                    entry_idx=int(long_state.entry_idx),
                                    exit_idx=i,
                                    entry_price=float(long_state.entry_price),
                                    exit_price=float(lx),
                                    profit=float(lx - long_state.entry_price),
                                ),
                            ),
                        )

            short_state = prev_short_hold[trade_count]
            if short_state is not None and short_state.entry_idx < i:
                short_pct = _profit_pct("short", short_state.entry_price, sx)
                if short_pct >= min_profit:
                    candidate_value = float(short_state.value - sx)
                    if candidate_value > float(best_cash.value) + 1e-12:
                        best_cash = _TradePathState(
                            candidate_value,
                            short_state.base_trades
                            + (
                                _TradeCandidate(
                                    side="short",
                                    entry_idx=int(short_state.entry_idx),
                                    exit_idx=i,
                                    entry_price=float(short_state.entry_price),
                                    exit_price=float(sx),
                                    profit=float(short_state.entry_price - sx),
                                ),
                            ),
                        )

            cash[trade_count] = best_cash

        for trade_count in range(1, k + 1):
            base_state = prev_cash[trade_count - 1]

            if le > 0:
                candidate_hold_value = float(base_state.value - le)
                current_long = prev_long_hold[trade_count]
                if current_long is None or candidate_hold_value > float(current_long.value) + 1e-12:
                    long_hold[trade_count] = _HoldState(
                        value=candidate_hold_value,
                        entry_idx=i,
                        entry_price=float(le),
                        base_trades=base_state.trades,
                    )
                else:
                    long_hold[trade_count] = current_long

            if se > 0:
                candidate_hold_value = float(base_state.value + se)
                current_short = prev_short_hold[trade_count]
                if current_short is None or candidate_hold_value > float(current_short.value) + 1e-12:
                    short_hold[trade_count] = _HoldState(
                        value=candidate_hold_value,
                        entry_idx=i,
                        entry_price=float(se),
                        base_trades=base_state.trades,
                    )
                else:
                    short_hold[trade_count] = current_short

    chosen = list(max(cash, key=lambda state: float(state.value)).trades)
    if not chosen:
        return []

    out: List[Trade] = []
    for candidate in chosen:
        out.append(
            Trade(
                side=candidate.side,
                entry_row=df.iloc[candidate.entry_idx],
                exit_row=df.iloc[candidate.exit_idx],
                entry_price=float(candidate.entry_price),
                exit_price=float(candidate.exit_price),
                profit=float(candidate.profit),
            )
        )
    return out


def solve_optimal_joint_trade_sequence_generic(
    df: pd.DataFrame,
    *,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
    min_profit_pct: float = 0.01,
) -> List[Trade]:
    """Find the best long/short action sequence through time without a per-period k cap."""

    if df is None or len(df) < 2:
        return []

    # Cache column name resolution — build lowercase map once
    _col_map = {str(c).lower(): c for c in df.columns}
    def _resolve(col: str) -> str:
        key = str(col).lower()
        if key in _col_map:
            return _col_map[key]
        raise ValueError(f"Missing column '{col}' (needed by solver)")

    le_col = _resolve(long_entry_price_col or "high")
    lx_col = _resolve(long_exit_price_col or "low")
    se_col = _resolve(short_entry_price_col or "low")
    sx_col = _resolve(short_exit_price_col or "high")

    long_entry = df[le_col].astype(float).to_numpy()
    long_exit = df[lx_col].astype(float).to_numpy()
    short_entry = df[se_col].astype(float).to_numpy()
    short_exit = df[sx_col].astype(float).to_numpy()
    n = len(df)

    min_profit = float(min_profit_pct)
    cash = _TradePathState(0.0, ())
    long_hold: _HoldState | None = None
    short_hold: _HoldState | None = None

    for i in range(n):
        le = float(long_entry[i])
        lx = float(long_exit[i])
        se = float(short_entry[i])
        sx = float(short_exit[i])

        prev_cash = cash
        prev_long_hold = long_hold
        prev_short_hold = short_hold

        best_cash = prev_cash

        if prev_long_hold is not None and prev_long_hold.entry_idx < i:
            long_pct = _profit_pct("long", prev_long_hold.entry_price, lx)
            if long_pct >= min_profit:
                candidate_value = float(prev_long_hold.value + lx)
                if candidate_value > float(best_cash.value) + 1e-12:
                    best_cash = _TradePathState(
                        candidate_value,
                        prev_long_hold.base_trades
                        + (
                            _TradeCandidate(
                                side="long",
                                entry_idx=int(prev_long_hold.entry_idx),
                                exit_idx=i,
                                entry_price=float(prev_long_hold.entry_price),
                                exit_price=float(lx),
                                profit=float(lx - prev_long_hold.entry_price),
                            ),
                        ),
                    )

        if prev_short_hold is not None and prev_short_hold.entry_idx < i:
            short_pct = _profit_pct("short", prev_short_hold.entry_price, sx)
            if short_pct >= min_profit:
                candidate_value = float(prev_short_hold.value - sx)
                if candidate_value > float(best_cash.value) + 1e-12:
                    best_cash = _TradePathState(
                        candidate_value,
                        prev_short_hold.base_trades
                        + (
                            _TradeCandidate(
                                side="short",
                                entry_idx=int(prev_short_hold.entry_idx),
                                exit_idx=i,
                                entry_price=float(prev_short_hold.entry_price),
                                exit_price=float(sx),
                                profit=float(prev_short_hold.entry_price - sx),
                            ),
                        ),
                    )

        cash = best_cash

        if le > 0:
            candidate_hold_value = float(prev_cash.value - le)
            if prev_long_hold is None or candidate_hold_value > float(prev_long_hold.value) + 1e-12:
                long_hold = _HoldState(
                    value=candidate_hold_value,
                    entry_idx=i,
                    entry_price=float(le),
                    base_trades=prev_cash.trades,
                )
            else:
                long_hold = prev_long_hold

        if se > 0:
            candidate_hold_value = float(prev_cash.value + se)
            if prev_short_hold is None or candidate_hold_value > float(prev_short_hold.value) + 1e-12:
                short_hold = _HoldState(
                    value=candidate_hold_value,
                    entry_idx=i,
                    entry_price=float(se),
                    base_trades=prev_cash.trades,
                )
            else:
                short_hold = prev_short_hold

    chosen = list(cash.trades)
    if not chosen:
        return []

    out: List[Trade] = []
    for candidate in chosen:
        out.append(
            Trade(
                side=candidate.side,
                entry_row=df.iloc[candidate.entry_idx],
                exit_row=df.iloc[candidate.exit_idx],
                entry_price=float(candidate.entry_price),
                exit_price=float(candidate.exit_price),
                profit=float(candidate.profit),
            )
        )
    return out


@njit
def _solve_one_side_numba(entry_prices, exit_prices, k, min_profit_pct):
    """Numba-accelerated one-side DP solver core.

    Returns (trades_arr, n_trades) where trades_arr is (entry_idx, exit_idx) pairs
    in chronological order, and n_trades is the count.
    """
    n = len(entry_prices)

    cash = np.zeros(k + 1, dtype=np.float64)
    hold = np.full(k + 1, -np.inf, dtype=np.float64)
    # Track per-day state for back-tracking
    cash_val = np.zeros((n, k + 1), dtype=np.float64)
    # Track when each hold[t] was entered (day index)
    hold_entry_day = np.full(k + 1, -1, dtype=np.int32)

    for i in range(n):
        ep = entry_prices[i]
        xp = exit_prices[i]
        for t in range(1, k + 1):
            cand_hold = cash[t - 1] - ep
            if cand_hold > hold[t]:
                hold[t] = cand_hold
                hold_entry_day[t] = i
            cand_cash = hold[t] + xp
            if cand_cash > cash[t]:
                cash[t] = cand_cash
        for t in range(k + 1):
            cash_val[i, t] = cash[t]

    # Back-tracking: find trades using hold_entry_day
    trades_out = np.zeros((k, 2), dtype=np.int32)
    n_trades = 0
    t = k
    i = n - 1
    while t > 0 and i >= 1:
        if cash_val[i, t] == cash_val[i - 1, t]:
            i -= 1
            continue

        # Cash increased at day i => exit happened here
        # Find the entry: this is the day when hold[t] was last entered
        entry_idx = hold_entry_day[t]
        if entry_idx < 0 or entry_idx >= i:
            # hold_entry_day might be stale if hold[t] was entered after the last exit
            i -= 1
            continue

        trades_out[n_trades, 0] = entry_idx
        trades_out[n_trades, 1] = i
        n_trades += 1
        i = entry_idx - 1
        t -= 1

    # Reverse to chronological order
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
    min_profit_pct: float = 0.05,
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
    cash_val = [[0.0] * (k + 1) for _ in range(n)]

    for i in range(n):
        ep_i = float(ep[i])
        xp_i = float(xp[i])
        for trade_count in range(1, k + 1):
            cand_hold = cash[trade_count - 1] - ep_i
            if cand_hold > hold[trade_count]:
                hold[trade_count] = cand_hold
                hold_entry_day[trade_count] = i
            cand_cash = hold[trade_count] + xp_i
            if cand_cash > cash[trade_count]:
                cash[trade_count] = cand_cash
        for trade_count in range(k + 1):
            cash_val[i][trade_count] = cash[trade_count]

    trades_idx: List[Tuple[int, int]] = []
    trade_count = k
    i = n - 1
    while trade_count > 0 and i >= 1:
        if i > 0 and abs(cash_val[i][trade_count] - cash_val[i - 1][trade_count]) < 1e-12:
            i -= 1
            continue

        entry_idx = hold_entry_day[trade_count]
        if entry_idx < 0 or entry_idx >= i:
            i -= 1
            continue

        if entry_idx < i:
            trades_idx.append((entry_idx, i))
        i = entry_idx - 1
        trade_count -= 1

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


def solve_joint_trades_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
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
            raise ValueError("solve_joint_trades_by_frequency requires a DatetimeIndex or a 'date' column")
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
        trades = solve_optimal_joint_trades_generic(
            group,
            k=k,
            min_profit_pct=min_profit_pct,
            long_entry_price_col=long_entry_price_col,
            long_exit_price_col=long_exit_price_col,
            short_entry_price_col=short_entry_price_col,
            short_exit_price_col=short_exit_price_col,
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


def solve_joint_trade_sequence_by_frequency(
    df: pd.DataFrame,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
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
            raise ValueError("solve_joint_trade_sequence_by_frequency requires a DatetimeIndex or a 'date' column")
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
        trades = solve_optimal_joint_trade_sequence_generic(
            group,
            min_profit_pct=min_profit_pct,
            long_entry_price_col=long_entry_price_col,
            long_exit_price_col=long_exit_price_col,
            short_entry_price_col=short_entry_price_col,
            short_exit_price_col=short_exit_price_col,
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


def solve_trades_by_frequency(
    df: pd.DataFrame,
    k: int,
    freq: str = "QE",
    side: Side = "long",
    min_profit_pct: float = 0.02,
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


@njit(parallel=True)
def _solve_joint_numba_batch(
    long_entry_2d,
    long_exit_2d,
    short_entry_2d,
    short_exit_2d,
    lengths,
    k,
    min_profit_pct,
):
    n_symbols = len(lengths)
    trades_out = np.full((n_symbols, k, 3), -1, dtype=np.int32)
    trade_counts = np.zeros(n_symbols, dtype=np.int32)
    for s in prange(n_symbols):
        n = int(lengths[s])
        if n < 2:
            continue
        trades_arr, n_trades = _solver_core_numba(
            long_entry_2d[s, :n],
            long_exit_2d[s, :n],
            short_entry_2d[s, :n],
            short_exit_2d[s, :n],
            k,
            min_profit_pct,
        )
        if n_trades > k:
            n_trades = k
        trade_counts[s] = n_trades
        for t in range(n_trades):
            trades_out[s, t, 0] = trades_arr[t, 0]
            trades_out[s, t, 1] = trades_arr[t, 1]
            trades_out[s, t, 2] = trades_arr[t, 2]
    return trades_out, trade_counts


def solve_joint_trades_by_frequency_batched(
    price_frames: Mapping[str, pd.DataFrame],
    k: int,
    freq: str = "QE",
    min_profit_pct: float = 0.01,
    long_entry_price_col: Optional[str] = None,
    long_exit_price_col: Optional[str] = None,
    short_entry_price_col: Optional[str] = None,
    short_exit_price_col: Optional[str] = None,
) -> dict[str, List[Dict]]:
    if not price_frames:
        return {}

    freq_resolved, label_freq = _resolve_freq(freq)
    results: dict[str, List[Dict]] = {
        str(symbol).strip().upper(): []
        for symbol, frame in price_frames.items()
        if frame is not None and not frame.empty
    }
    period_buckets: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    period_order: dict[str, pd.Timestamp] = {}

    for symbol, frame in price_frames.items():
        symbol_name = str(symbol).strip().upper()
        if frame is None or frame.empty:
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

    for label in sorted(period_buckets, key=lambda item: period_order.get(item, pd.Timestamp.min)):
        entries = period_buckets[label]
        if not entries:
            continue
        max_len = max(len(group) for _, group in entries)
        n_symbols = len(entries)
        long_entry = np.zeros((n_symbols, max_len), dtype=np.float64)
        long_exit = np.zeros((n_symbols, max_len), dtype=np.float64)
        short_entry = np.zeros((n_symbols, max_len), dtype=np.float64)
        short_exit = np.zeros((n_symbols, max_len), dtype=np.float64)
        lengths = np.zeros(n_symbols, dtype=np.int32)
        grouped_frames: list[pd.DataFrame] = []
        grouped_symbols: list[str] = []
        resolved_columns: list[tuple[str, str, str, str]] = []

        for idx, (symbol, group) in enumerate(entries):
            grouped_symbols.append(symbol)
            grouped_frames.append(group)
            lengths[idx] = len(group)
            le_col = _resolve_frame_column(group, le_col_hint)
            lx_col = _resolve_frame_column(group, lx_col_hint)
            se_col = _resolve_frame_column(group, se_col_hint)
            sx_col = _resolve_frame_column(group, sx_col_hint)
            resolved_columns.append((le_col, lx_col, se_col, sx_col))
            long_entry[idx, : lengths[idx]] = group[le_col].astype(np.float64).to_numpy()
            long_exit[idx, : lengths[idx]] = group[lx_col].astype(np.float64).to_numpy()
            short_entry[idx, : lengths[idx]] = group[se_col].astype(np.float64).to_numpy()
            short_exit[idx, : lengths[idx]] = group[sx_col].astype(np.float64).to_numpy()

        if _HAS_NUMBA:
            trades_out, trade_counts = _solve_joint_numba_batch(
                long_entry,
                long_exit,
                short_entry,
                short_exit,
                lengths,
                int(k),
                float(min_profit_pct),
            )
        else:
            trades_out = np.full((n_symbols, k, 3), -1, dtype=np.int32)
            trade_counts = np.zeros(n_symbols, dtype=np.int32)
            for idx, group in enumerate(grouped_frames):
                trades = solve_optimal_joint_trades_generic(
                    group,
                    k=k,
                    min_profit_pct=min_profit_pct,
                    long_entry_price_col=le_col_hint,
                    long_exit_price_col=lx_col_hint,
                    short_entry_price_col=se_col_hint,
                    short_exit_price_col=sx_col_hint,
                )
                trade_counts[idx] = min(len(trades), k)
                for t_idx, trade in enumerate(trades[:k]):
                    side_code = 1 if trade.side == "long" else 2
                    entry_idx = int(group.index.get_indexer([trade.entry_row.name])[0])
                    exit_idx = int(group.index.get_indexer([trade.exit_row.name])[0])
                    trades_out[idx, t_idx, 0] = entry_idx
                    trades_out[idx, t_idx, 1] = exit_idx
                    trades_out[idx, t_idx, 2] = side_code

        for idx, symbol in enumerate(grouped_symbols):
            count = int(trade_counts[idx])
            if count <= 0:
                continue
            group = grouped_frames[idx]
            le_col, lx_col, se_col, sx_col = resolved_columns[idx]
            for t_idx in range(count):
                entry_idx = int(trades_out[idx, t_idx, 0])
                exit_idx = int(trades_out[idx, t_idx, 1])
                side_code = int(trades_out[idx, t_idx, 2])
                if entry_idx < 0 or exit_idx < 0:
                    continue
                side = "long" if side_code == 1 else "short"
                entry_price = float(group[le_col].iloc[entry_idx]) if side == "long" else float(group[se_col].iloc[entry_idx])
                exit_price = float(group[lx_col].iloc[exit_idx]) if side == "long" else float(group[sx_col].iloc[exit_idx])
                profit = exit_price - entry_price if side == "long" else entry_price - exit_price
                results.setdefault(symbol, []).append(
                    {
                        "side": side,
                        "entry_row": group.iloc[entry_idx],
                        "exit_row": group.iloc[exit_idx],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "profit": profit,
                        "period_label": label,
                    }
                )

    return results
