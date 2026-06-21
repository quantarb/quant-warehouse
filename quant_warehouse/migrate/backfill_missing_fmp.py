from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.refresh.planner import macro_backfill_needs_update
from quant_warehouse.refresh.universe import (
    refresh_universe_fundamentals,
    refresh_universe_macro,
    refresh_universe_nport_disclosure,
    refresh_universe_prices,
)
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import (
    FMP_HISTORICAL_EQUITY_SECTIONS,
    MIN_HISTORICAL_DATE,
    fundamental_period_for_section,
    normalize_fundamental_period,
)

ProgressLogger = Callable[[str], None] | None


def _catalog_symbols(catalog_path: Path, *, section: str, provider: str) -> list[str]:
    with sqlite3.connect(catalog_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT symbol
            FROM section_state
            WHERE section = ? AND provider = ?
            ORDER BY symbol
            """,
            (section, provider),
        ).fetchall()
    return [str(row[0]).strip().upper() for row in rows if row and row[0]]


def backfill_missing_fmp_historical(
    *,
    warehouse: Warehouse | None = None,
    config: WarehouseConfig | None = None,
    equity_sections: Sequence[str] | None = None,
    equity_provider: str = "fmp",
    etf_provider: str = "fmp",
    period: str = "quarter",
    nport_start_year: int = 2019,
    include_macro: bool | None = None,
    include_prices: bool = True,
    force_macro: bool = False,
    macro_start_date: str = MIN_HISTORICAL_DATE,
    max_equity_symbols: int | None = None,
    max_etf_symbols: int | None = None,
    staleness_days: int = 90,
    skip_recent_hours: float = 24.0,
    max_workers: int = 8,
    progress_logger: ProgressLogger = None,
) -> dict[str, object]:
    warehouse = warehouse or Warehouse(config=config)
    cfg = warehouse.config
    section_list = list(equity_sections or FMP_HISTORICAL_EQUITY_SECTIONS)
    started_at = datetime.now(timezone.utc).isoformat()
    normalized_period = normalize_fundamental_period(period)
    section_periods = {
        section: fundamental_period_for_section(section, preferred=normalized_period) or "n/a"
        for section in section_list
    }
    macro_start_text = str(macro_start_date or MIN_HISTORICAL_DATE)[:10]
    summary: dict[str, object] = {
        "started_at": started_at,
        "equity_sections": section_list,
        "equity_provider": equity_provider,
        "etf_provider": etf_provider,
        "preferred_period": normalized_period,
        "section_periods": section_periods,
        "macro_start_date": macro_start_text,
        "include_prices": include_prices,
        "staleness_days": staleness_days,
        "skip_recent_hours": skip_recent_hours,
        "max_workers": max(1, int(max_workers)),
    }

    should_refresh_macro = force_macro
    if not should_refresh_macro:
        if include_macro is False:
            should_refresh_macro = False
        else:
            should_refresh_macro = macro_backfill_needs_update(
                warehouse.catalog,
                provider=equity_provider,
                history_start_date=pd.Timestamp(macro_start_text).date(),
                skip_recent_hours=skip_recent_hours,
            )

    if should_refresh_macro:
        if callable(progress_logger):
            progress_logger(f"Backfill: refreshing macro series via FMP from {macro_start_text}")
        if force_macro:
            summary["macro"] = warehouse.macro.refresh(
                provider=equity_provider,
                start_date=macro_start_text,
                full_refresh=True,
            )
        else:
            summary["macro"] = refresh_universe_macro(
                warehouse,
                provider=equity_provider,
                macro_start_date=macro_start_text,
                skip_recent_hours=skip_recent_hours,
                progress_logger=progress_logger,
            )
    else:
        summary["macro"] = {"status": "skipped_complete"}

    equity_symbols = _catalog_symbols(cfg.catalog_path, section="prices", provider=equity_provider)
    if max_equity_symbols is not None:
        equity_symbols = equity_symbols[: max(0, int(max_equity_symbols))]

    if include_prices:
        if callable(progress_logger):
            progress_logger(
                f"Backfill: refreshing equity prices for {len(equity_symbols):,} symbols via FMP"
            )
        summary["equity_prices"] = _summarize_results(
            refresh_universe_prices(
                warehouse,
                equity_symbols,
                providers=[equity_provider],
                backfill_skip=True,
                price_start_date=MIN_HISTORICAL_DATE,
                skip_recent_hours=skip_recent_hours,
                max_workers=max_workers,
                progress_logger=progress_logger,
            )
        )

    if callable(progress_logger):
        progress_logger(
            f"Backfill: refreshing equity fundamentals for {len(equity_symbols):,} symbols | "
            f"sections={','.join(section_list)} | preferred_period={normalized_period}"
        )
    equity_results = refresh_universe_fundamentals(
        warehouse,
        equity_symbols,
        sections=section_list,
        providers=[equity_provider],
        period=normalized_period,
        staleness_days=staleness_days,
        skip_recent_hours=skip_recent_hours,
        backfill_skip=True,
        max_workers=max_workers,
        progress_logger=progress_logger,
    )
    summary["equity"] = _summarize_results(equity_results)

    etf_symbols = _catalog_symbols(cfg.catalog_path, section="etf_prices", provider=etf_provider)
    if max_etf_symbols is not None:
        etf_symbols = etf_symbols[: max(0, int(max_etf_symbols))]

    if include_prices and etf_symbols:
        if callable(progress_logger):
            progress_logger(
                f"Backfill: refreshing ETF prices for {len(etf_symbols):,} symbols via FMP"
            )
        summary["etf_prices"] = _summarize_results(
            refresh_universe_prices(
                warehouse,
                etf_symbols,
                providers=[etf_provider],
                etf_symbols=set(etf_symbols),
                backfill_skip=True,
                price_start_date=MIN_HISTORICAL_DATE,
                skip_recent_hours=skip_recent_hours,
                max_workers=max_workers,
                progress_logger=progress_logger,
            )
        )

    if callable(progress_logger):
        progress_logger(
            f"Backfill: refreshing ETF N-PORT history for {len(etf_symbols):,} symbols "
            f"from {nport_start_year}"
        )
    etf_results = refresh_universe_nport_disclosure(
        warehouse,
        etf_symbols,
        provider=etf_provider,
        start_year=nport_start_year,
        staleness_days=staleness_days,
        skip_recent_hours=skip_recent_hours,
        max_workers=max_workers,
        progress_logger=progress_logger,
    )
    summary["etf_nport_disclosure"] = _summarize_results(etf_results)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def _summarize_results(results: Sequence[dict[str, object]]) -> dict[str, int]:
    counts = {"updated": 0, "empty": 0, "skipped_fresh": 0, "error": 0, "other": 0}
    for row in results:
        status = str(row.get("status") or "other")
        if status not in counts:
            counts["other"] += 1
        else:
            counts[status] += 1
    counts["total"] = len(results)
    return counts


def write_backfill_log(summary: dict[str, object], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")