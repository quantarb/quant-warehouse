"""Sparse HITS targets for independent long and short strategies."""

from __future__ import annotations

import polars as pl

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Mapping

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    price_frames: Mapping[str, pl.DataFrame],
    *,
    spec: HitsLabelSpec | None = None,
    edge_weight_mode: str = "return",
    progress_callback: Callable[..., Any] | None = None,
) -> pl.DataFrame:
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
    rows: list[pl.DataFrame] = []
    total = len(symbols)
    for completed, symbol in enumerate(symbols, start=1):
        frame = price_frames.get(symbol)
        if frame is None:
            frame = price_frames.get(symbol.lower())
        if frame is None or frame.is_empty():
            continue
        normalized = _normalize_prices(frame, cfg)
        if normalized.is_empty():
            continue
        symbol_rows: list[pl.DataFrame] = []
        for year_frame in normalized.with_columns(pl.col("date").dt.year().alias("_year")).partition_by("_year", maintain_order=True):
            year_frame = year_frame.drop("_year")
            scores = _build_symbol_year_scores(year_frame, cfg, edge_weight_mode=edge_weight_mode)
            if scores is not None:
                scores = scores.with_columns(pl.lit(symbol).alias("symbol")).select(["symbol", *[c for c in scores.columns if c != "symbol"]])
                symbol_rows.append(scores)
        if symbol_rows:
            rows.extend(symbol_rows)
        if callable(progress_callback):
            progress_callback(completed=completed, total=total, current_symbol=symbol)

    if not rows:
        return pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS})
    return pl.concat(rows, how="diagonal_relaxed").select(_OUTPUT_COLUMNS)


def build_hold_timing_hits_labels(
    price_frames: Mapping[str, pl.DataFrame],
    *,
    hold_days: tuple[int, ...] = (5, 20, 60, 120),
    spec: HitsLabelSpec | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> pl.DataFrame:
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
    panels: list[pl.DataFrame] = []
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
        panels.append(panel.select(keep).rename(rename))
    if not panels:
        return pl.DataFrame(schema={"symbol": pl.String, "date": pl.Datetime})
    out = panels[0]
    for panel in panels[1:]:
        out = out.join(panel, on=["symbol", "date"], how="full", coalesce=True)
    return out


def build_inverse_holding_time_hits_labels(
    price_frames: Mapping[str, pl.DataFrame],
    *,
    spec: HitsLabelSpec | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> pl.DataFrame:
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
    price_frames: Mapping[str, pl.DataFrame],
    *,
    spec: HitsLabelSpec | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> pl.DataFrame:
    """Build return and speed HITS targets from one shared edge topology.

    Each symbol-year constructs the valid directed date-pair set once.  The
    same positive-return edges then receive two independent weights:
    realized return and ``1 / holding_days``.  The score families remain
    separate so return and speed are not collapsed into one objective.
    """
    cfg = spec or HitsLabelSpec()
    symbols = [str(symbol).strip().upper() for symbol in price_frames if str(symbol).strip()]
    rows: list[pl.DataFrame] = []
    total = len(symbols)
    for completed, symbol in enumerate(symbols, start=1):
        frame = price_frames.get(symbol)
        if frame is None:
            frame = price_frames.get(symbol.lower())
        if frame is None or frame.is_empty():
            continue
        normalized = _normalize_prices(frame, cfg)
        for year_frame in normalized.with_columns(pl.col("date").dt.year().alias("_year")).partition_by("_year", maintain_order=True):
            year_frame = year_frame.drop("_year")
            scores = _build_symbol_year_return_speed_scores(year_frame, cfg)
            if scores is not None:
                scores = scores.with_columns(pl.lit(symbol).alias("symbol")).select(["symbol", *[c for c in scores.columns if c != "symbol"]])
                rows.append(scores)
        if callable(progress_callback):
            progress_callback(completed=completed, total=total, current_symbol=symbol)
    if not rows:
        return pl.DataFrame(schema={column: pl.Null for column in [
            "symbol", "date", "long_hub", "long_authority", "short_hub", "short_authority",
            "long_hub_tail", "long_authority_tail", "short_hub_tail", "short_authority_tail",
            "speed_long_hub", "speed_long_authority", "speed_short_hub", "speed_short_authority",
            "speed_long_hub_tail", "speed_long_authority_tail", "speed_short_hub_tail", "speed_short_authority_tail",
        ]})
    return pl.concat(rows, how="diagonal_relaxed")


_OUTPUT_COLUMNS = [
    "symbol", "date",
    "long_hub", "long_authority", "short_hub", "short_authority",
    "long_hub_tail", "long_authority_tail", "short_hub_tail", "short_authority_tail",
]


def _normalize_prices(frame: pl.DataFrame, spec: HitsLabelSpec) -> pl.DataFrame:
    out = frame.rename({column: str(column).strip().lower() for column in frame.columns})
    required = {"date", "high", "low"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"HITS price frame is missing columns: {sorted(missing)}")
    date_expr = pl.col("date").str.to_datetime(strict=False) if out.schema["date"] == pl.String else pl.col("date").cast(pl.Datetime, strict=False)
    out = out.with_columns([date_expr.dt.truncate("1d").alias("date"), pl.col("high").cast(pl.Float64, strict=False), pl.col("low").cast(pl.Float64, strict=False)]).drop_nulls(["date", "high", "low"]).filter((pl.col("high") > 0) & (pl.col("low") > 0))
    if spec.start_date is not None:
        out = out.filter(pl.col("date") >= datetime.fromisoformat(spec.start_date[:10]))
    if spec.end_date is not None:
        out = out.filter(pl.col("date") <= datetime.fromisoformat(spec.end_date[:10]))
    return out.select(["date", "high", "low"]).sort("date").unique("date", keep="last")


def _build_symbol_year_scores(
    frame: pl.DataFrame,
    spec: HitsLabelSpec,
    *,
    edge_weight_mode: str = "return",
) -> pl.DataFrame | None:
    n = frame.height
    if n < 2:
        return None
    high = torch.tensor(frame["high"].to_list(), dtype=torch.float64, device=DEVICE)
    low = torch.tensor(frame["low"].to_list(), dtype=torch.float64, device=DEVICE)
    returns, valid, holding_days = _build_edge_channels_torch(high, low, spec.max_hold)
    output: dict[str, Any] = {"date": frame["date"].to_list()}
    for side, future_returns in returns.items():
        if edge_weight_mode == "return":
            weights = torch.where(valid, future_returns.clamp_min(0.0), torch.zeros_like(future_returns))
        elif edge_weight_mode == "inverse_holding_time":
            weights = torch.zeros_like(future_returns)
            eligible = valid & (future_returns > 0.0)
            weights = torch.where(eligible, 1.0 / holding_days.clamp_min(1.0), weights)
        else:
            raise ValueError("edge_weight_mode must be 'return' or 'inverse_holding_time'")
        if not bool(torch.any(weights > 0)):
            output[f"{side}_hub"] = [0.0] * n
            output[f"{side}_authority"] = [0.0] * n
            output[f"{side}_hub_tail"] = [False] * n
            output[f"{side}_authority_tail"] = [False] * n
            continue
        hub = torch.ones(n, dtype=torch.float64, device=DEVICE)
        authority = torch.ones(n, dtype=torch.float64, device=DEVICE)
        for _ in range(spec.iterations):
            authority = weights.T @ hub
            authority = authority / torch.linalg.vector_norm(authority).clamp_min(1.0)
            hub = weights @ authority
            hub = hub / torch.linalg.vector_norm(hub).clamp_min(1.0)
        hub = hub / hub.max().clamp_min(1.0)
        authority = authority / authority.max().clamp_min(1.0)
        output[f"{side}_hub"] = hub.tolist()
        output[f"{side}_authority"] = authority.tolist()
        output[f"{side}_hub_tail"] = _tail_mask_torch(hub, spec.tail_quantile).tolist()
        output[f"{side}_authority_tail"] = _tail_mask_torch(authority, spec.tail_quantile).tolist()
    return pl.DataFrame(output)


def _build_edge_channels_torch(
    high: torch.Tensor,
    low: torch.Tensor,
    max_hold: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build HITS graph channels without materializing NumPy matrices."""
    n = int(high.numel())
    index = torch.arange(n, dtype=torch.int64, device=DEVICE)
    holding_days = index[None, :] - index[:, None]
    valid = torch.triu(torch.ones((n, n), dtype=torch.bool, device=DEVICE), diagonal=1)
    valid = valid & (holding_days <= max_hold)
    returns = {
        "long": low[None, :] / high[:, None] - 1.0,
        "short": low[:, None] / high[None, :] - 1.0,
    }
    return returns, valid, holding_days.to(torch.float64)


def _build_symbol_year_return_speed_scores(frame: pl.DataFrame, spec: HitsLabelSpec) -> pl.DataFrame | None:
    if frame.height < 2:
        return None
    high = torch.tensor(frame["high"].to_list(), dtype=torch.float64, device=DEVICE)
    low = torch.tensor(frame["low"].to_list(), dtype=torch.float64, device=DEVICE)
    returns, valid, holding_days = _build_edge_channels_torch(high, low, spec.max_hold)
    output: dict[str, Any] = {"date": frame["date"].to_list()}
    for side in ("long", "short"):
        for prefix, mode in (("", "return"), ("speed_", "inverse_holding_time")):
            hub, authority, hub_tail, authority_tail = _score_weight_matrix_torch(
                returns[side], valid, holding_days, spec.iterations, mode, spec.tail_quantile
            )
            output[f"{prefix}{side}_hub"] = hub.detach().cpu().tolist()
            output[f"{prefix}{side}_authority"] = authority.detach().cpu().tolist()
            output[f"{prefix}{side}_hub_tail"] = hub_tail.detach().cpu().tolist()
            output[f"{prefix}{side}_authority_tail"] = authority_tail.detach().cpu().tolist()
    return pl.DataFrame(output)


def _tail_mask_torch(values: torch.Tensor, quantile: float) -> torch.Tensor:
    """Torch equivalent of the stable first-rank tail selection."""
    n = int(values.numel())
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(n, dtype=torch.float64, device=DEVICE)
    ranks[order] = (torch.arange(n, dtype=torch.float64, device=DEVICE) + 1.0) / n
    return (ranks <= quantile) | (ranks >= 1.0 - quantile)


def _score_weight_matrix_torch(
    returns: torch.Tensor,
    valid: torch.Tensor,
    holding_days: torch.Tensor,
    iterations: int,
    mode: str,
    tail_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "return":
        weights = torch.where(valid, returns.clamp_min(0.0), torch.zeros_like(returns))
    elif mode == "inverse_holding_time":
        eligible = valid & (returns > 0.0)
        weights = torch.where(eligible, 1.0 / holding_days.clamp_min(1.0), torch.zeros_like(returns))
    else:
        raise ValueError("edge weight mode must be 'return' or 'inverse_holding_time'")
    n = int(returns.shape[0])
    if not bool(torch.any(weights > 0)):
        zero = torch.zeros(n, dtype=torch.float64, device=DEVICE)
        false = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        return zero, zero, false, false
    hub = torch.ones(n, dtype=torch.float64, device=DEVICE)
    authority = torch.ones(n, dtype=torch.float64, device=DEVICE)
    for _ in range(iterations):
        authority = (weights.T @ hub) / torch.linalg.vector_norm(weights.T @ hub).clamp_min(1.0)
        hub = (weights @ authority) / torch.linalg.vector_norm(weights @ authority).clamp_min(1.0)
    hub = hub / hub.max().clamp_min(1.0)
    authority = authority / authority.max().clamp_min(1.0)
    return hub, authority, _tail_mask_torch(hub, tail_quantile), _tail_mask_torch(authority, tail_quantile)
