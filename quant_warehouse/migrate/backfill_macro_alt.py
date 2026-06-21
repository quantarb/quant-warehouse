from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.market_prices import refresh_market_price_universe
from quant_warehouse.warehouse.sections import (
    CRYPTO_PRICES_SECTION,
    CURRENCY_PRICES_SECTION,
    DEFAULT_CRYPTO_SYMBOLS,
    DEFAULT_CURRENCY_SYMBOLS,
    DEFAULT_ECONOMIC_SERIES,
    DEFAULT_INDEX_SYMBOLS,
    INDEX_PRICES_SECTION,
    MIN_HISTORICAL_DATE,
)

ProgressLogger = Callable[[str], None] | None


def _summarize_results(results: Sequence[dict[str, object]]) -> dict[str, int]:
    counts = {"updated": 0, "empty": 0, "error": 0, "other": 0}
    for row in results:
        if row.get("status") == "error" or row.get("error"):
            counts["error"] += 1
            continue
        rows = int(row.get("rows") or row.get("fetched_rows") or 0)
        if rows > 0:
            counts["updated"] += 1
        else:
            counts["empty"] += 1
    counts["total"] = len(results)
    return counts


def backfill_fmp_macro_alt(
    *,
    config: WarehouseConfig | None = None,
    provider: str = "fmp",
    macro_start_date: str = MIN_HISTORICAL_DATE,
    economic_series: Sequence[str] | None = None,
    include_treasury_rates: bool = True,
    include_yield_curve: bool = True,
    include_calendar: bool = True,
    include_risk_premium: bool = True,
    include_crypto: bool = True,
    include_currency: bool = True,
    include_index: bool = True,
    crypto_symbols: Sequence[str] | None = None,
    currency_symbols: Sequence[str] | None = None,
    index_symbols: Sequence[str] | None = None,
    yield_curve_step_days: int = 5,
    progress_logger: ProgressLogger = None,
) -> dict[str, object]:
    warehouse = Warehouse(config=config)
    provider_name = str(provider or "fmp").strip().lower()
    start_text = str(macro_start_date or MIN_HISTORICAL_DATE)[:10]
    series_list = list(economic_series or DEFAULT_ECONOMIC_SERIES)
    summary: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider_name,
        "macro_start_date": start_text,
        "economic_series": series_list,
    }

    if callable(progress_logger):
        progress_logger(
            f"Backfill macro-alt: refreshing {len(series_list)} economic indicators from {start_text}"
        )
    summary["economic"] = warehouse.macro.refresh(
        economic_series=series_list,
        include_treasury_rates=include_treasury_rates,
        include_yield_curve=False,
        include_calendar=False,
        include_risk_premium=False,
        provider=provider_name,
        start_date=start_text,
        full_refresh=False,
    )

    if include_risk_premium:
        if callable(progress_logger):
            progress_logger("Backfill macro-alt: refreshing country risk premium snapshot")
        summary["risk_premium"] = warehouse.macro.refresh_risk_premium(provider=provider_name)

    if include_calendar:
        if callable(progress_logger):
            progress_logger(f"Backfill macro-alt: refreshing macro calendar from {start_text}")
        summary["calendar"] = warehouse.macro.refresh_calendar(
            provider=provider_name,
            start_date=start_text,
            full_refresh=False,
        )

    if include_yield_curve:
        if callable(progress_logger):
            progress_logger(
                f"Backfill macro-alt: refreshing yield curve history from {start_text} "
                f"(may take several minutes)"
            )
        summary["yield_curve"] = warehouse.macro.refresh_yield_curve_history(
            provider=provider_name,
            start_date=start_text,
            full_refresh=False,
            step_days=max(1, int(yield_curve_step_days)),
        )

    end_date = datetime.now(timezone.utc).date().isoformat()
    if include_crypto:
        symbols = list(crypto_symbols or DEFAULT_CRYPTO_SYMBOLS)
        if callable(progress_logger):
            progress_logger(f"Backfill macro-alt: refreshing {len(symbols)} crypto symbols")
        crypto_results = refresh_market_price_universe(
            warehouse.market_prices,
            symbols,
            section=CRYPTO_PRICES_SECTION,
            provider=provider_name,
            start_date=start_text,
            end_date=end_date,
        )
        summary["crypto_prices"] = _summarize_results(crypto_results)

    if include_currency:
        symbols = list(currency_symbols or DEFAULT_CURRENCY_SYMBOLS)
        if callable(progress_logger):
            progress_logger(f"Backfill macro-alt: refreshing {len(symbols)} FX pairs")
        currency_results = refresh_market_price_universe(
            warehouse.market_prices,
            symbols,
            section=CURRENCY_PRICES_SECTION,
            provider=provider_name,
            start_date=start_text,
            end_date=end_date,
        )
        summary["currency_prices"] = _summarize_results(currency_results)

    if include_index:
        symbols = list(index_symbols or DEFAULT_INDEX_SYMBOLS)
        if callable(progress_logger):
            progress_logger(f"Backfill macro-alt: refreshing {len(symbols)} index symbols")
        index_results = refresh_market_price_universe(
            warehouse.market_prices,
            symbols,
            section=INDEX_PRICES_SECTION,
            provider=provider_name,
            start_date=start_text,
            end_date=end_date,
        )
        summary["index_prices"] = _summarize_results(index_results)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def write_backfill_log(summary: dict[str, object], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")