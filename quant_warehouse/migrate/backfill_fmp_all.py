from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.etf_universe import fetch_etf_universe
from quant_warehouse.migrate.backfill_missing_fmp import (
    _catalog_symbols,
    _summarize_results,
    backfill_missing_fmp_historical,
)
from quant_warehouse.refresh.parallel import run_symbol_workers
from quant_warehouse.refresh.planner import backfill_fundamental_needs_update
from quant_warehouse.refresh.universe import (
    refresh_universe_fundamentals,
    refresh_universe_nport_disclosure,
    refresh_universe_prices,
    refresh_universe_profiles,
)
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import (
    EQUITY_CALENDAR_SECTIONS,
    FMP_ALL_EQUITY_SECTIONS,
    FMP_HISTORICAL_ETF_SECTIONS,
    MIN_HISTORICAL_DATE,
)

ProgressLogger = Callable[[str], None] | None

TRANSCRIPT_SECTION = "transcript"


def backfill_fmp_all(
    *,
    config: WarehouseConfig | None = None,
    equity_provider: str = "fmp",
    etf_provider: str = "fmp",
    period: str = "quarter",
    calendar_start_date: str = "2005-01-01",
    nport_start_year: int = 2019,
    transcript_start_year: int = 2005,
    include_macro: bool = False,
    include_prices: bool = True,
    include_profiles: bool = True,
    include_calendars: bool = True,
    include_transcripts: bool = False,
    include_etf_universe: bool = True,
    skip_equity_core: bool = False,
    max_equity_symbols: int | None = None,
    max_etf_symbols: int | None = None,
    staleness_days: int = 90,
    skip_recent_hours: float = 24.0,
    request_sleep_seconds: float = 0.05,
    max_workers: int = 8,
    progress_logger: ProgressLogger = None,
) -> dict[str, object]:
    """Comprehensive OpenBB/FMP backfill for equities, ETFs, and mutual funds."""
    warehouse = Warehouse(config=config)
    cfg = warehouse.config
    started_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, object] = {
        "started_at": started_at,
        "equity_provider": equity_provider,
        "etf_provider": etf_provider,
        "period": period,
        "calendar_start_date": calendar_start_date,
        "include_macro": include_macro,
        "include_prices": include_prices,
        "include_profiles": include_profiles,
        "include_calendars": include_calendars,
        "include_transcripts": include_transcripts,
        "include_etf_universe": include_etf_universe,
        "skip_equity_core": skip_equity_core,
        "max_workers": max(1, int(max_workers)),
    }

    if skip_equity_core:
        if callable(progress_logger):
            progress_logger("Backfill-all: skipping equity core phase (prices + fundamentals)")
        summary["core"] = {"status": "skipped_equity_core"}
    else:
        equity_sections = list(FMP_ALL_EQUITY_SECTIONS)
        core_summary = backfill_missing_fmp_historical(
            warehouse=warehouse,
            equity_sections=equity_sections,
            equity_provider=equity_provider,
            etf_provider=etf_provider,
            period=period,
            nport_start_year=nport_start_year,
            include_macro=include_macro,
            include_prices=include_prices,
            macro_start_date=MIN_HISTORICAL_DATE,
            max_equity_symbols=max_equity_symbols,
            max_etf_symbols=None,
            staleness_days=staleness_days,
            skip_recent_hours=skip_recent_hours,
            max_workers=max_workers,
            progress_logger=progress_logger,
        )
        summary["core"] = core_summary

    equity_symbols = _catalog_symbols(cfg.catalog_path, section="prices", provider=equity_provider)
    if max_equity_symbols is not None:
        equity_symbols = equity_symbols[: max(0, int(max_equity_symbols))]

    if include_profiles and equity_symbols:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-all: refreshing FMP equity profiles for {len(equity_symbols):,} symbols"
            )
        summary["equity_profiles"] = _summarize_results(
            refresh_universe_profiles(
                warehouse,
                equity_symbols,
                providers=[equity_provider],
                refresh_days=staleness_days,
                max_symbols=max_equity_symbols,
                max_workers=max_workers,
                progress_logger=progress_logger,
            )
        )

    if include_transcripts and equity_symbols:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-all: refreshing earnings transcripts for {len(equity_symbols):,} symbols "
                f"from {transcript_start_year}"
            )
        def _refresh_transcript(symbol: str) -> list[dict[str, object]]:
            needs_refresh, reason = backfill_fundamental_needs_update(
                warehouse.catalog,
                symbol,
                TRANSCRIPT_SECTION,
                equity_provider,
                staleness_days=staleness_days,
                skip_recent_hours=skip_recent_hours,
            )
            if not needs_refresh:
                return [
                    {
                        "symbol": symbol,
                        "section": TRANSCRIPT_SECTION,
                        "status": "skipped_fresh",
                        "reason": reason,
                    }
                ]
            try:
                stats = warehouse.fundamentals.refresh_transcripts(
                    symbol,
                    provider=equity_provider,
                    start_year=transcript_start_year,
                )
                rows = int(stats.get("rows") or 0)
                return [
                    {
                        "symbol": symbol,
                        "section": TRANSCRIPT_SECTION,
                        "status": "updated" if rows > 0 else "empty",
                        "reason": reason,
                        "rows": rows,
                        "fetched_periods": int(stats.get("fetched_periods") or 0),
                    }
                ]
            except Exception as exc:
                return [
                    {
                        "symbol": symbol,
                        "section": TRANSCRIPT_SECTION,
                        "status": "error",
                        "reason": reason,
                        "error": str(exc),
                    }
                ]

        transcript_results = run_symbol_workers(
            equity_symbols,
            _refresh_transcript,
            max_workers=max_workers,
            progress_logger=progress_logger,
            progress_label="transcript backfill",
        )
        summary["transcripts"] = _summarize_results(transcript_results)

    if include_calendars:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-all: refreshing equity calendars from {calendar_start_date}"
            )
        calendar_stats: dict[str, dict[str, object]] = {}
        for section in EQUITY_CALENDAR_SECTIONS:
            try:
                calendar_stats[section] = warehouse.equity_calendar.refresh_section(
                    section,
                    provider=equity_provider,
                    start_date=calendar_start_date,
                )
            except Exception as exc:
                calendar_stats[section] = {"status": "error", "error": str(exc)}
            time.sleep(request_sleep_seconds)
        summary["equity_calendars"] = calendar_stats

    etf_symbols = set(_catalog_symbols(cfg.catalog_path, section="etf_prices", provider=etf_provider))
    if include_etf_universe:
        if callable(progress_logger):
            progress_logger("Backfill-all: fetching FMP ETF/mutual-fund universe via etf.search")
        try:
            discovered = fetch_etf_universe(provider=etf_provider)
            summary["etf_universe_discovered"] = len(discovered)
            etf_symbols.update(discovered)
        except Exception as exc:
            summary["etf_universe_discovered"] = 0
            summary["etf_universe_error"] = str(exc)

    etf_symbol_list = sorted(etf_symbols)
    if max_etf_symbols is not None:
        etf_symbol_list = etf_symbol_list[: max(0, int(max_etf_symbols))]

    if include_prices and etf_symbol_list:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-all: refreshing ETF/mutual-fund prices for {len(etf_symbol_list):,} symbols"
            )
        summary["etf_prices_expanded"] = _summarize_results(
            refresh_universe_prices(
                warehouse,
                etf_symbol_list,
                providers=[etf_provider],
                etf_symbols=set(etf_symbol_list),
                backfill_skip=True,
                price_start_date=MIN_HISTORICAL_DATE,
                skip_recent_hours=skip_recent_hours,
                max_workers=max_workers,
                progress_logger=progress_logger,
            )
        )

    if include_profiles and etf_symbol_list:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-all: refreshing FMP ETF profiles for {len(etf_symbol_list):,} symbols"
            )
        summary["etf_profiles"] = _summarize_results(
            refresh_universe_profiles(
                warehouse,
                etf_symbol_list,
                providers=[etf_provider],
                etf_symbols=set(etf_symbol_list),
                refresh_days=staleness_days,
                max_workers=max_workers,
                progress_logger=progress_logger,
            )
        )

    etf_fundamental_sections = [
        section
        for section in FMP_HISTORICAL_ETF_SECTIONS
        if section != "etf_nport_disclosure"
    ]
    if etf_symbol_list and etf_fundamental_sections:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-all: refreshing ETF composition for {len(etf_symbol_list):,} symbols | "
                f"sections={','.join(etf_fundamental_sections)}"
            )
        summary["etf_composition"] = _summarize_results(
            refresh_universe_fundamentals(
                warehouse,
                etf_symbol_list,
                sections=etf_fundamental_sections,
                providers=[etf_provider],
                etf_symbols=set(etf_symbol_list),
                staleness_days=staleness_days,
                skip_recent_hours=skip_recent_hours,
                backfill_skip=True,
                max_workers=max_workers,
                progress_logger=progress_logger,
            )
        )

    if etf_symbol_list:
        if callable(progress_logger):
            progress_logger(
                f"Backfill-all: refreshing ETF N-PORT for {len(etf_symbol_list):,} symbols "
                f"from {nport_start_year}"
            )
        nport_results = refresh_universe_nport_disclosure(
            warehouse,
            etf_symbol_list,
            provider=etf_provider,
            start_year=nport_start_year,
            staleness_days=staleness_days,
            skip_recent_hours=skip_recent_hours,
            max_workers=max_workers,
            progress_logger=progress_logger,
        )
        summary["etf_nport_expanded"] = _summarize_results(nport_results)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def write_backfill_log(summary: dict[str, object], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")