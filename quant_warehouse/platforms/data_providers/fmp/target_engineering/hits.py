"""Sparse HITS targets for independent long and short strategies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HitsLabelSpec:
    """Configuration for the supported sparse HITS target variant.

    The implementation intentionally uses the research variant that proved
    most useful: nonnegative clipped returns, one graph per symbol/calendar
    year, and sparse top/bottom tails for model training.  Long and short
    scores are produced independently.
    """

    max_hold: int = 120
    iterations: int = 50
    tail_quantile: float = 0.20
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        if self.max_hold <= 0:
            raise ValueError("max_hold must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not 0.0 < self.tail_quantile <= 0.5:
            raise ValueError("tail_quantile must be in (0, 0.5]")


def build_hits_labels(
    price_frames: Mapping[str, pd.DataFrame],
    *,
    spec: HitsLabelSpec | None = None,
    edge_weight_mode: str = "return",
    progress_callback: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Build sparse long/short HITS labels from per-symbol price frames.

    Each symbol-year is a separate directed graph.  For a long graph, the
    edge from entry ``i`` to later exit ``j`` is ``low[j] / high[i] - 1``.
    For a short graph it is ``low[i] / high[j] - 1``.  Negative returns are
    clipped to zero because HITS requires nonnegative edge weights.

    The returned frame contains scores and ``*_tail`` flags.  A model should
    train on the tail rows for its own score and use the corresponding hub
    score for entries and authority score for exits.  No future-year data is
    used to construct a prior symbol-year graph.
    """

    cfg = spec or HitsLabelSpec()
    if edge_weight_mode not in {"return", "inverse_holding_time"}:
        raise ValueError("edge_weight_mode must be 'return' or 'inverse_holding_time'")
    symbols = [str(symbol).strip().upper() for symbol in price_frames if str(symbol).strip()]
    rows: list[pd.DataFrame] = []
    total = len(symbols)
    for completed, symbol in enumerate(symbols, start=1):
        frame = price_frames.get(symbol)
        if frame is None:
            frame = price_frames.get(symbol.lower())
        if frame is None or frame.empty:
            continue
        normalized = _normalize_prices(frame, cfg)
        if normalized.empty:
            continue
        symbol_rows: list[pd.DataFrame] = []
        for _, year_frame in normalized.groupby(normalized["date"].dt.year, sort=True):
            scores = _build_symbol_year_scores(year_frame, cfg, edge_weight_mode=edge_weight_mode)
            if scores is not None:
                scores.insert(0, "symbol", symbol)
                symbol_rows.append(scores)
        if symbol_rows:
            rows.extend(symbol_rows)
        if callable(progress_callback):
            progress_callback(completed=completed, total=total, current_symbol=symbol)

    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.concat(rows, ignore_index=True)[_OUTPUT_COLUMNS]


def build_hold_timing_hits_labels(
    price_frames: Mapping[str, pd.DataFrame],
    *,
    hold_days: tuple[int, ...] = (5, 20, 60, 120),
    spec: HitsLabelSpec | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Build HITS targets for several maximum holding-time horizons.

    The same historical price frames are converted into independent
    symbol-year graphs for each horizon.  A ``*_5d`` target describes the
    fast graph, while ``*_120d`` describes the slower graph.  Speed graphs
    retain only positive-return edges and weight them by inverse holding
    time, ``1 / days_between(entry, exit)``.
    """
    horizons = tuple(dict.fromkeys(int(days) for days in hold_days))
    if not horizons or any(days <= 0 for days in horizons):
        raise ValueError("hold_days must contain positive integers")
    base_spec = spec or HitsLabelSpec()
    panels: list[pd.DataFrame] = []
    for horizon in horizons:
        horizon_spec = replace(base_spec, max_hold=horizon)
        panel = build_hits_labels(
            price_frames,
            spec=horizon_spec,
            edge_weight_mode="inverse_holding_time",
            progress_callback=progress_callback,
        )
        keep = ["symbol", "date"]
        rename: dict[str, str] = {}
        for side in ("long", "short"):
            for score in ("hub", "authority"):
                source = f"{side}_{score}"
                target = f"{source}_{horizon}d"
                keep.append(source)
                rename[source] = target
        panels.append(panel[keep].rename(columns=rename))
    if not panels:
        return pd.DataFrame(columns=["symbol", "date"])
    out = panels[0]
    for panel in panels[1:]:
        out = out.merge(panel, on=["symbol", "date"], how="outer")
    return out


def build_inverse_holding_time_hits_labels(
    price_frames: Mapping[str, pd.DataFrame],
    *,
    spec: HitsLabelSpec | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Build one speed graph with the return graph's topology.

    Positive-return entry/exit pairs retain their graph edge, but the edge
    weight is ``1 / holding_days``.  The default maximum holding period is
    inherited from ``HitsLabelSpec`` and is not split into separate horizons.
    """
    return build_hits_labels(
        price_frames,
        spec=spec,
        edge_weight_mode="inverse_holding_time",
        progress_callback=progress_callback,
    )


def build_return_and_speed_hits_labels(
    price_frames: Mapping[str, pd.DataFrame],
    *,
    spec: HitsLabelSpec | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Build return and speed HITS targets from one shared edge topology.

    Each symbol-year constructs the valid directed date-pair set once.  The
    same positive-return edges then receive two independent weights:
    realized return and ``1 / holding_days``.  The score families remain
    separate so return and speed are not collapsed into one objective.
    """
    cfg = spec or HitsLabelSpec()
    symbols = [str(symbol).strip().upper() for symbol in price_frames if str(symbol).strip()]
    rows: list[pd.DataFrame] = []
    total = len(symbols)
    for completed, symbol in enumerate(symbols, start=1):
        frame = price_frames.get(symbol)
        if frame is None:
            frame = price_frames.get(symbol.lower())
        if frame is None or frame.empty:
            continue
        normalized = _normalize_prices(frame, cfg)
        for _, year_frame in normalized.groupby(normalized["date"].dt.year, sort=True):
            scores = _build_symbol_year_return_speed_scores(year_frame, cfg)
            if scores is not None:
                scores.insert(0, "symbol", symbol)
                rows.append(scores)
        if callable(progress_callback):
            progress_callback(completed=completed, total=total, current_symbol=symbol)
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "date", "long_hub", "long_authority", "short_hub", "short_authority",
            "long_hub_tail", "long_authority_tail", "short_hub_tail", "short_authority_tail",
            "speed_long_hub", "speed_long_authority", "speed_short_hub", "speed_short_authority",
            "speed_long_hub_tail", "speed_long_authority_tail", "speed_short_hub_tail", "speed_short_authority_tail",
        ])
    return pd.concat(rows, ignore_index=True)


_OUTPUT_COLUMNS = [
    "symbol", "date",
    "long_hub", "long_authority", "short_hub", "short_authority",
    "long_hub_tail", "long_authority_tail", "short_hub_tail", "short_authority_tail",
]


def _normalize_prices(frame: pd.DataFrame, spec: HitsLabelSpec) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    required = {"date", "high", "low"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"HITS price frame is missing columns: {sorted(missing)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out = out.dropna(subset=["date", "high", "low"])
    out = out.loc[(out["high"] > 0) & (out["low"] > 0)]
    if spec.start_date is not None:
        out = out.loc[out["date"] >= pd.Timestamp(spec.start_date).normalize()]
    if spec.end_date is not None:
        out = out.loc[out["date"] <= pd.Timestamp(spec.end_date).normalize()]
    return out[["date", "high", "low"]].sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _build_symbol_year_scores(
    frame: pd.DataFrame,
    spec: HitsLabelSpec,
    *,
    edge_weight_mode: str = "return",
) -> pd.DataFrame | None:
    frame = frame.reset_index(drop=True)
    n = len(frame)
    if n < 2:
        return None
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    index = np.arange(n)
    valid = np.triu(np.ones((n, n), dtype=bool), 1)
    valid &= (index[None, :] - index[:, None]) <= spec.max_hold

    returns, valid, holding_days = _build_edge_channels(high, low, spec.max_hold)
    output: dict[str, Any] = {"date": frame["date"].to_numpy()}
    for side, future_returns in returns.items():
        if edge_weight_mode == "return":
            weights = np.where(valid, np.maximum(future_returns, 0.0), 0.0)
        elif edge_weight_mode == "inverse_holding_time":
            weights = np.zeros_like(future_returns, dtype=float)
            eligible = valid & (future_returns > 0.0)
            np.divide(1.0, holding_days, out=weights, where=eligible)
        else:
            raise ValueError("edge_weight_mode must be 'return' or 'inverse_holding_time'")
        if not np.any(weights > 0):
            output[f"{side}_hub"] = np.zeros(n, dtype=float)
            output[f"{side}_authority"] = np.zeros(n, dtype=float)
            output[f"{side}_hub_tail"] = np.zeros(n, dtype=bool)
            output[f"{side}_authority_tail"] = np.zeros(n, dtype=bool)
            continue
        hub = np.ones(n, dtype=float)
        authority = np.ones(n, dtype=float)
        for _ in range(spec.iterations):
            authority = weights.T @ hub
            authority /= np.linalg.norm(authority) or 1.0
            hub = weights @ authority
            hub /= np.linalg.norm(hub) or 1.0
        hub = hub / (hub.max() or 1.0)
        authority = authority / (authority.max() or 1.0)
        output[f"{side}_hub"] = hub
        output[f"{side}_authority"] = authority
        output[f"{side}_hub_tail"] = _tail_mask(hub, spec.tail_quantile)
        output[f"{side}_authority_tail"] = _tail_mask(authority, spec.tail_quantile)
    return pd.DataFrame(output)


def _build_edge_channels(
    high: np.ndarray,
    low: np.ndarray,
    max_hold: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Construct the date-pair topology and all reusable edge quantities once."""
    n = len(high)
    index = np.arange(n)
    valid = np.triu(np.ones((n, n), dtype=bool), 1)
    holding_days = index[None, :] - index[:, None]
    valid &= holding_days <= max_hold
    returns = {
        "long": low[None, :] / high[:, None] - 1.0,
        "short": low[:, None] / high[None, :] - 1.0,
    }
    return returns, valid, holding_days


def _score_weight_matrix(
    returns: np.ndarray,
    valid: np.ndarray,
    holding_days: np.ndarray,
    iterations: int,
    mode: str,
    tail_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mode == "return":
        weights = np.where(valid, np.maximum(returns, 0.0), 0.0)
    elif mode == "inverse_holding_time":
        weights = np.zeros_like(returns, dtype=float)
        np.divide(1.0, holding_days, out=weights, where=valid & (returns > 0.0))
    else:
        raise ValueError("edge weight mode must be 'return' or 'inverse_holding_time'")
    n = len(returns)
    if not np.any(weights > 0):
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    hub = np.ones(n, dtype=float)
    authority = np.ones(n, dtype=float)
    for _ in range(iterations):
        authority = weights.T @ hub
        authority /= np.linalg.norm(authority) or 1.0
        hub = weights @ authority
        hub /= np.linalg.norm(hub) or 1.0
    hub = hub / (hub.max() or 1.0)
    authority = authority / (authority.max() or 1.0)
    return hub, authority, _tail_mask(hub, tail_quantile), _tail_mask(authority, tail_quantile)


def _build_symbol_year_return_speed_scores(frame: pd.DataFrame, spec: HitsLabelSpec) -> pd.DataFrame | None:
    frame = frame.reset_index(drop=True)
    if len(frame) < 2:
        return None
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    returns, valid, holding_days = _build_edge_channels(high, low, spec.max_hold)
    output: dict[str, Any] = {"date": frame["date"].to_numpy()}
    for side in ("long", "short"):
        for prefix, mode in (("", "return"), ("speed_", "inverse_holding_time")):
            hub, authority, hub_tail, authority_tail = _score_weight_matrix(
                returns[side], valid, holding_days, spec.iterations, mode, spec.tail_quantile
            )
            output[f"{prefix}{side}_hub"] = hub
            output[f"{prefix}{side}_authority"] = authority
            output[f"{prefix}{side}_hub_tail"] = hub_tail
            output[f"{prefix}{side}_authority_tail"] = authority_tail
    return pd.DataFrame(output)


def _tail_mask(values: np.ndarray, quantile: float) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first", pct=True).to_numpy()
    return (ranks <= quantile) | (ranks >= 1.0 - quantile)
