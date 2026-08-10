from __future__ import annotations

from collections.abc import Sequence
import polars as pl
import torch


def build_option_mean_variance_labels(
    option_candidates: pl.DataFrame,
    group_cols: Sequence[str] = ("underlying_symbol", "date"),
    expected_return_col: str = "expected_return",
    risk_col: str = "risk",
    risk_aversion: float = 1.0,
    max_weight: float | None = None,
    long_only: bool = True,
    ) -> pl.DataFrame:
    """Build diagonal mean-variance labels with Polars grouping."""
    if option_candidates is None or option_candidates.is_empty():
        return pl.DataFrame()
    groups = tuple(group_cols)
    _require_columns(option_candidates, [*groups, expected_return_col, risk_col], ctx="build_option_mean_variance_labels")
    out = option_candidates.with_row_index("_row_id").with_columns(
        pl.col(expected_return_col).cast(pl.Float64, strict=False),
        pl.col(risk_col).cast(pl.Float64, strict=False),
    ).with_columns(
        (pl.col(expected_return_col) - float(risk_aversion) * pl.col(risk_col)).alias("mv_score")
    ).with_columns(
        pl.col("mv_score").rank(method="ordinal", descending=True).over(list(groups)).alias("mv_rank")
    ).with_columns(
        (pl.col("mv_rank") == 1).alias("mv_selected"),
        pl.lit(0.0).alias("mv_weight"),
    )
    parts: list[pl.DataFrame] = []
    for _, group in out.group_by(list(groups), maintain_order=True):
        weights = _weights_from_scores(group["mv_score"].to_list(), max_weight=max_weight, long_only=long_only)
        parts.append(group.with_columns(pl.Series("mv_weight", weights.tolist())))
    out = pl.concat(parts, how="vertical_relaxed").with_columns(
        pl.lit("option_mean_variance").alias("target_name"),
        pl.col("mv_weight").alias("target_value"),
    ).sort([*groups, "mv_rank"]).drop("_row_id")
    return out


def _weights_from_scores(scores: list[float], *, max_weight: float | None, long_only: bool) -> torch.Tensor:
    values = torch.nan_to_num(torch.as_tensor(scores, dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() == 0:
        return values
    if long_only:
        raw = torch.clamp(values, min=0.0)
        if float(raw.sum()) <= 0.0:
            raw = torch.zeros_like(values)
            raw[int(torch.argmax(values))] = 1.0
    else:
        shifted = values - values.min()
        raw = shifted if float(shifted.sum()) > 0.0 else torch.ones_like(values)
    weights = raw / raw.sum()
    if max_weight is not None:
        cap = float(max_weight)
        if cap <= 0.0:
            raise ValueError("max_weight must be positive when provided")
        weights = _apply_long_only_cap(weights, cap)
    return weights


def _apply_long_only_cap(weights: torch.Tensor, cap: float) -> torch.Tensor:
    if weights.numel() == 0 or cap >= 1.0:
        return weights
    capped = torch.minimum(weights, torch.tensor(cap, dtype=weights.dtype))
    for _ in range(len(weights) + 1):
        remainder = 1.0 - float(capped.sum())
        if remainder <= 1e-12:
            break
        room = capped < cap - 1e-12
        if not bool(room.any()):
            break
        base = weights * room
        if float(base.sum()) <= 0:
            capped[room] += remainder / float(room.sum())
        else:
            capped[room] += remainder * (base[room] / base[room].sum())
        capped = torch.minimum(capped, torch.tensor(cap, dtype=weights.dtype))
    total = float(capped.sum())
    return capped / total if total > 0.0 else capped


def _require_columns(df: pl.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
