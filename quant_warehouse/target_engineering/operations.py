from __future__ import annotations

import math
from typing import Any


def trade_return_pct(side: str, entry_px: float, exit_px: float) -> float:
    """Return decimal trade PnL for long or short executions."""

    if not entry_px:
        return 0.0
    if str(side).strip().lower() == "long":
        return (float(exit_px) - float(entry_px)) / float(entry_px)
    return (float(entry_px) - float(exit_px)) / float(entry_px)


def apply_trade_deduplication(
    trade_rows: list[dict[str, Any]],
    completed_trades: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate trade candidates while preserving deterministic ordering."""

    mode_value = str(mode or "exact").strip().lower()
    if mode_value not in {"exact", "entry_date"}:
        return trade_rows, completed_trades

    if mode_value == "exact":
        kept_rows: list[dict[str, Any]] = []
        kept_completed: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for row, completed in zip(trade_rows, completed_trades):
            key = (
                str(row.get("symbol") or ""),
                str(row.get("side") or ""),
                str(row.get("entry_date") or ""),
                str(row.get("exit_date") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            kept_rows.append(row)
            kept_completed.append(completed)
        return kept_rows, kept_completed

    best_by_key: dict[tuple[str, str, str], tuple[int, float]] = {}
    for idx, completed in enumerate(completed_trades):
        key = (
            str(completed.get("symbol") or ""),
            str(completed.get("side") or ""),
            str(completed.get("entry_date") or ""),
        )
        ret_dec = float(completed.get("ret_dec") or 0.0)
        prev = best_by_key.get(key)
        if prev is None or ret_dec > prev[1]:
            best_by_key[key] = (idx, ret_dec)
    keep_indices = {idx for idx, _ in best_by_key.values()}
    return (
        [row for idx, row in enumerate(trade_rows) if idx in keep_indices],
        [row for idx, row in enumerate(completed_trades) if idx in keep_indices],
    )


def build_label_rows_from_completed_trades(completed_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert completed trades into canonical event/action label rows."""

    label_rows: list[dict[str, Any]] = []
    for row in completed_trades:
        side = str(row.get("side") or "").strip().lower()
        if side not in {"long", "short"}:
            continue
        ret = _float(row.get("ret_dec"))
        hold_days = int(_float(row.get("hold_days")))
        symbol = str(row.get("symbol") or "").strip().upper()
        entry_date = str(row.get("entry_date") or "")[:10]
        exit_date = str(row.get("exit_date") or "")[:10]
        if not symbol or not entry_date or not exit_date:
            continue
        freq = str(row.get("freq") or "").strip()
        k = int(_float(row.get("k")))
        horizon = str(row.get("horizon") or "").strip() or (f"{freq}_k{k}" if freq and k > 0 else freq)
        trade_id = str(row.get("trade_id") or "").strip() or (
            f"T|{symbol}|E{entry_date.replace('-', '')}|X{exit_date.replace('-', '')}|S{side}|H{horizon or 'NA'}"
        )
        direction_label = 1 if side == "long" else 0
        entry_action = "buy" if side == "long" else "short"
        exit_action = "sell" if side == "long" else "cover"
        common = {
            "symbol": symbol,
            "trade_id": trade_id,
            "direction_label": direction_label,
            "trade_return": round(ret, 8),
            "hold_days": hold_days,
            "trade_duration_days": hold_days,
            "side": side,
            "freq": freq,
            "k": k,
            "horizon": horizon,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_px": row.get("entry_px") or "",
            "exit_px": row.get("exit_px") or "",
            "ret_pct": f"{ret * 100.0:.2f}%",
        }
        label_rows.append(
            {
                "date": entry_date,
                "event": "entry",
                "label": entry_action,
                "action_label": entry_action,
                "market_position": 1 if side == "long" else -1,
                **common,
            }
        )
        label_rows.append(
            {
                "date": exit_date,
                "event": "exit",
                "label": exit_action,
                "action_label": exit_action,
                "market_position": 0,
                **common,
            }
        )
    return label_rows


def build_label_statistics(label_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate label/trade statistics used by callers and reports."""

    if not label_rows:
        return {"trade_stats": {}, "grouped_trade_stats": [], "symbol_grouped_trade_stats": []}

    per_trade_rows: list[dict[str, Any]] = []
    seen_trade_ids: set[str] = set()
    for row in label_rows:
        trade_id = str(row.get("trade_id") or "").strip()
        if not trade_id or trade_id in seen_trade_ids:
            continue
        seen_trade_ids.add(trade_id)
        per_trade_rows.append(row)
    if not per_trade_rows:
        per_trade_rows = [row for row in label_rows if str(row.get("event") or "").lower() != "exit"] or list(label_rows)

    returns: list[float] = []
    long_returns: list[float] = []
    short_returns: list[float] = []
    symbol_bucket: dict[str, list[float]] = {}
    symbol_holds: dict[str, list[float]] = {}
    grouped_bucket: dict[tuple[str, str, int], list[float]] = {}
    grouped_holds: dict[tuple[str, str, int], list[float]] = {}
    for row in per_trade_rows:
        ret = _float(row.get("trade_return"))
        side = str(row.get("side") or "").strip().lower()
        if side not in {"long", "short"}:
            side = "long" if int(_float(row.get("direction_label"))) == 1 else "short"
        freq = str(row.get("freq") or "D1").strip() or "D1"
        k = int(_float(row.get("k"))) or 1
        hold_days = _float(row.get("hold_days"))
        symbol = str(row.get("symbol") or "").strip().upper()
        returns.append(ret)
        if side == "long":
            long_returns.append(ret)
        else:
            short_returns.append(ret)
        if symbol:
            symbol_bucket.setdefault(symbol, []).append(ret)
            symbol_holds.setdefault(symbol, []).append(hold_days)
        key = (side, freq, k)
        grouped_bucket.setdefault(key, []).append(ret)
        grouped_holds.setdefault(key, []).append(hold_days)

    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    trade_stats = {
        "total_trades": int(len(returns)),
        "symbols_count": int(len(symbol_bucket)),
        "long_trades": int(len(long_returns)),
        "short_trades": int(len(short_returns)),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "breakeven_trades": int(len([value for value in returns if value == 0])),
        "win_rate_pct": round((len(wins) / float(len(returns))) * 100.0, 4) if returns else 0.0,
        "loss_rate_pct": round((len(losses) / float(len(returns))) * 100.0, 4) if returns else 0.0,
        "avg_return_pct": round((_mean(returns) or 0.0) * 100.0, 4),
        "median_return_pct": round((_median(returns) or 0.0) * 100.0, 4),
    }

    grouped_trade_stats = [
        _stats_row(side, freq, k, values, grouped_holds.get((side, freq, k), []))
        for (side, freq, k), values in sorted(grouped_bucket.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    ]
    symbol_grouped_trade_stats = [
        {
            **_stats_row("mixed", "mixed", 0, symbol_bucket[symbol], symbol_holds.get(symbol, [])),
            "symbol": symbol,
        }
        for symbol in sorted(symbol_bucket)
    ]
    return {
        "trade_stats": trade_stats,
        "grouped_trade_stats": grouped_trade_stats,
        "symbol_grouped_trade_stats": symbol_grouped_trade_stats,
    }


def _stats_row(side: str, freq: str, k: int, values: list[float], hold_values: list[float]) -> dict[str, Any]:
    mean_v = _mean(values) or 0.0
    std_v = _std_sample(values)
    hold_mean = _mean(hold_values) or 0.0
    hold_std = _std_sample(hold_values)
    sharpe = (mean_v / std_v) if std_v and abs(std_v) > 1e-12 else None
    return {
        "side": side,
        "freq": freq,
        "k": int(k),
        "trades": int(len(values)),
        "trade_return_mean_pct": round(mean_v * 100.0, 4),
        "trade_return_std_pct": round(std_v * 100.0, 4) if std_v is not None else None,
        "trade_duration_mean": round(float(hold_mean), 4),
        "trade_duration_std": round(float(hold_std), 4) if hold_std is not None else None,
        "sharpe": round(sharpe, 6) if sharpe is not None else None,
    }


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _std_sample(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = _mean(values)
    if mean_value is None:
        return None
    var = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    return float(math.sqrt(var))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    count = len(sorted_values)
    mid = count // 2
    if count % 2 == 1:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)
