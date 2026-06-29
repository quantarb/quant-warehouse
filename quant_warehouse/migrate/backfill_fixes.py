from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.platforms.data_providers.fmp.sections import FMP_HISTORICAL_ETF_SECTIONS
from quant_warehouse.refresh.universe import refresh_universe_fundamentals
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import EQUITY_CALENDAR_SECTIONS

ProgressLogger = Callable[[str], None] | None


def _etf_symbols(catalog_path: Path) -> list[str]:
    with sqlite3.connect(catalog_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM etf_profile WHERE provider='fmp' ORDER BY symbol"
        ).fetchall()
    return [str(row[0]).strip().upper() for row in rows if row and row[0]]


def _etf_symbols_missing_section(catalog_path: Path, section: str) -> list[str]:
    with sqlite3.connect(catalog_path) as conn:
        rows = conn.execute(
            """
            SELECT p.symbol
            FROM etf_profile p
            LEFT JOIN section_state s
              ON p.symbol = s.symbol
             AND s.section = ?
             AND s.provider = 'fmp'
             AND s.row_count > 0
            WHERE p.provider = 'fmp' AND s.symbol IS NULL
            ORDER BY p.symbol
            """,
            (section,),
        ).fetchall()
    return [str(row[0]).strip().upper() for row in rows if row and row[0]]


def backfill_calendar_and_etf_composition(
    *,
    config: WarehouseConfig | None = None,
    calendar_start_date: str = "2005-01-01",
    full_refresh_earnings: bool = False,
    calendar_sections: Sequence[str] | None = None,
    include_calendars: bool = True,
    include_etf_composition: bool = True,
    etf_retry_missing_holdings: bool = False,
    max_workers: int = 8,
    progress_logger: ProgressLogger = None,
) -> dict[str, object]:
    warehouse = Warehouse(config=config)
    started_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, object] = {
        "started_at": started_at,
        "calendar_start_date": calendar_start_date,
        "full_refresh_earnings": full_refresh_earnings,
        "include_calendars": include_calendars,
        "include_etf_composition": include_etf_composition,
        "etf_retry_missing_holdings": etf_retry_missing_holdings,
        "max_workers": max(1, int(max_workers)),
    }

    if include_calendars:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-fixes: refreshing equity calendars from {calendar_start_date}"
            )
        calendar_stats: dict[str, dict[str, object]] = {}
        for section in list(calendar_sections or EQUITY_CALENDAR_SECTIONS):
            try:
                full_refresh = section == "equity_calendar_earnings" and full_refresh_earnings
                calendar_stats[section] = warehouse.equity_calendar.refresh_section(
                    section,
                    provider="fmp",
                    start_date=calendar_start_date,
                    full_refresh=full_refresh,
                )
            except Exception as exc:
                calendar_stats[section] = {"status": "error", "error": str(exc)}
        summary["equity_calendars"] = calendar_stats

    etf_symbols = (
        _etf_symbols_missing_section(warehouse.config.catalog_path, "etf_holdings")
        if etf_retry_missing_holdings
        else _etf_symbols(warehouse.config.catalog_path)
    )
    composition_sections = [
        section
        for section in FMP_HISTORICAL_ETF_SECTIONS
        if section != "etf_nport_disclosure"
    ]
    if include_etf_composition and etf_symbols and composition_sections:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-fixes: refreshing ETF composition for {len(etf_symbols):,} symbols | "
                f"sections={','.join(composition_sections)}"
            )
        results = refresh_universe_fundamentals(
            warehouse,
            etf_symbols,
            sections=composition_sections,
            providers=["fmp"],
            etf_symbols=set(etf_symbols),
            backfill_skip=True,
            max_workers=max_workers,
            progress_logger=progress_logger,
        )
        counts = {"updated": 0, "empty": 0, "skipped_fresh": 0, "error": 0, "other": 0, "total": len(results)}
        for row in results:
            status = str(row.get("status") or "other")
            if status in counts:
                counts[status] += 1
            else:
                counts["other"] += 1
        summary["etf_composition"] = counts

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def write_backfill_log(summary: dict[str, object], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
