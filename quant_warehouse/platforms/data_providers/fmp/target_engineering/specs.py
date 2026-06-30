from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def parse_k_list(raw: Any) -> list[int]:
    if raw in (None, ""):
        return []
    values = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    out: list[int] = []
    seen: set[int] = set()
    for part in values:
        text = str(part).strip()
        if not text:
            continue
        try:
            value = int(text)
        except Exception:
            continue
        if value > 0 and value not in seen:
            seen.add(value)
            out.append(value)
    return out


@dataclass(frozen=True)
class LabelBuildSpec:
    """Typed oracle-label generation configuration."""

    k_params: dict[str, list[int]] = field(default_factory=lambda: {"YE": [1]})
    solver_mode: str = "period_top_k"
    min_profit_pct: float = 0.01
    buy_execution: str = "adj_high"
    sell_execution: str = "adj_low"
    short_execution: str = "adj_low"
    cover_execution: str = "adj_high"
    trade_dedup_mode: str = "exact"
    start_date: str | None = None
    end_date: str | None = None

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None = None) -> LabelBuildSpec:
        raw = dict(source or {})
        k_params_raw = raw.get("k_params")
        if isinstance(k_params_raw, dict):
            k_params = {
                "W": parse_k_list(k_params_raw.get("W")),
                "M": parse_k_list(k_params_raw.get("M") or k_params_raw.get("ME")),
                "QE": parse_k_list(k_params_raw.get("QE")),
                "YE": parse_k_list(k_params_raw.get("YE")),
            }
        else:
            k_params = {
                "W": parse_k_list(raw.get("k_w_list")),
                "M": parse_k_list(raw.get("k_m_list")),
                "QE": parse_k_list(raw.get("k_qe_list")),
                "YE": parse_k_list(raw.get("k_ye_list")),
            }
        k_params = {freq: ks for freq, ks in k_params.items() if ks} or {"YE": [1]}
        return cls(
            k_params=k_params,
            solver_mode=_solver_mode(raw.get("solver_mode")),
            min_profit_pct=_min_profit_decimal(raw.get("min_profit_pct")),
            buy_execution=str(raw.get("buy_execution") or "adj_high"),
            sell_execution=str(raw.get("sell_execution") or "adj_low"),
            short_execution=str(raw.get("short_execution") or "adj_low"),
            cover_execution=str(raw.get("cover_execution") or "adj_high"),
            trade_dedup_mode=str(raw.get("trade_dedup_mode") or "exact"),
            start_date=str(raw.get("label_start_date") or raw.get("start_date") or "").strip() or None,
            end_date=str(raw.get("label_end_date") or raw.get("end_date") or "").strip() or None,
        )


@dataclass(frozen=True)
class TradeGenerationResult:
    """Oracle trade rows plus normalized completed trade payloads."""

    trade_rows: list[dict[str, Any]] = field(default_factory=list)
    completed_trades: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class OracleLabelResult:
    """Canonical label rows and diagnostics from oracle trade generation."""

    label_rows: list[dict[str, Any]]
    statistics: dict[str, Any]
    completed_trades: list[dict[str, Any]]


def _min_profit_decimal(value: Any, default_points: float = 1.0) -> float:
    try:
        points = float(value if value not in (None, "") else default_points)
    except Exception:
        points = default_points
    return max(0.0, points) / 100.0


def _solver_mode(value: Any) -> str:
    text = str(value or "period_top_k").strip().lower() or "period_top_k"
    if text not in {"period_top_k", "period_sequence"}:
        return "period_top_k"
    return text
