from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_option_mean_variance_labels(
    option_candidates: pd.DataFrame,
    group_cols: Sequence[str] = ("underlying_symbol", "date"),
    expected_return_col: str = "expected_return",
    risk_col: str = "risk",
    risk_aversion: float = 1.0,
    max_weight: float | None = None,
    long_only: bool = True,
) -> pd.DataFrame:
    """Build diagonal mean-variance option labels from candidate rows."""

    if option_candidates is None or option_candidates.empty:
        return pd.DataFrame()
    group_cols = tuple(group_cols)
    _require_columns(
        option_candidates,
        [*group_cols, expected_return_col, risk_col],
        ctx="build_option_mean_variance_labels",
    )

    out = option_candidates.copy()
    out[expected_return_col] = pd.to_numeric(out[expected_return_col], errors="coerce")
    out[risk_col] = pd.to_numeric(out[risk_col], errors="coerce")
    out["mv_score"] = out[expected_return_col] - float(risk_aversion) * out[risk_col]
    out["mv_rank"] = out.groupby(list(group_cols), dropna=False)["mv_score"].rank(method="first", ascending=False)
    out["mv_selected"] = out["mv_rank"] == 1
    out["mv_weight"] = 0.0

    for _, idx in out.groupby(list(group_cols), dropna=False).groups.items():
        scores = out.loc[idx, "mv_score"].astype(float)
        weights = _weights_from_scores(scores.to_numpy(dtype=float), max_weight=max_weight, long_only=long_only)
        out.loc[idx, "mv_weight"] = weights

    out["target_name"] = "option_mean_variance"
    out["target_value"] = out["mv_weight"]
    return out.sort_values([*group_cols, "mv_rank"], ignore_index=True)


def _weights_from_scores(
    scores: np.ndarray,
    *,
    max_weight: float | None,
    long_only: bool,
) -> np.ndarray:
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if len(scores) == 0:
        return scores
    if long_only:
        raw = np.clip(scores, 0.0, None)
        if raw.sum() <= 0.0:
            raw = np.zeros_like(scores, dtype=float)
            raw[int(np.argmax(scores))] = 1.0
    else:
        shifted = scores - scores.min()
        raw = shifted if shifted.sum() > 0.0 else np.ones_like(scores, dtype=float)
    weights = raw / raw.sum()
    if max_weight is not None:
        cap = float(max_weight)
        if cap <= 0.0:
            raise ValueError("max_weight must be positive when provided")
        weights = _apply_long_only_cap(weights, cap)
    return weights


def _apply_long_only_cap(weights: np.ndarray, cap: float) -> np.ndarray:
    if len(weights) == 0 or cap >= 1.0:
        return weights
    capped = np.minimum(weights, cap)
    for _ in range(len(weights) + 1):
        remainder = 1.0 - float(capped.sum())
        if remainder <= 1e-12:
            break
        room = capped < cap - 1e-12
        if not room.any():
            break
        base = weights * room
        if base.sum() <= 0:
            capped[room] += remainder / float(room.sum())
        else:
            capped[room] += remainder * (base[room] / base[room].sum())
        capped = np.minimum(capped, cap)
    total = float(capped.sum())
    return capped / total if total > 0.0 else capped


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, ctx: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")
