from __future__ import annotations

from datetime import date
from typing import Callable, Iterable, Sequence

import pandas as pd

from quant_warehouse.refresh.planner import (
    backfill_fundamental_needs_update,
    catalog_price_max_date,
    expected_latest_price_date,
    fetch_kwargs_from_plan,
    fundamental_refresh_needs_update,
    historical_fetch_plan,
    macro_refresh_needs_update,
    nport_disclosure_needs_update,
    price_backfill_needs_update,
    price_refresh_needs_update,
    profile_refresh_needs_update,
    symbol_has_fresh_prices,
)
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import (
    DEFAULT_ECONOMIC_SERIES,
    MACRO_TREASURY_SECTION,
    TREASURY_BUNDLE_SYMBOL,
)
from quant_warehouse.warehouse.sections import (
    EQUITY_FUNDAMENTAL_SECTIONS,
    ETF_FUNDAMENTAL_SECTIONS,
    MIN_HISTORICAL_DATE,
)


ProgressLogger = Callable[[str], None] | None


def refresh_universe_prices(
    warehouse: Warehouse,
    symbols: Sequence[str],
    *,
    providers: Sequence[str],
    target_end_date: date | None = None,
    etf_symbols: set[str] | None = None,
    skip_recent_hours: float = 24.0,
    backfill_skip: bool = False,
    price_start_date: str = MIN_HISTORICAL_DATE,
    max_symbols: int | None = None,
    progress_logger: ProgressLogger = None,
) -> list[dict[str, object]]:
    target_end_date = target_end_date or expected_latest_price_date()
    end_text = target_end_date.isoformat()
    start_text = str(price_start_date or MIN_HISTORICAL_DATE)[:10]
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if max_symbols is not None:
        normalized = normalized[: max(0, int(max_symbols))]
    etf_set = {str(symbol).strip().upper() for symbol in (etf_symbols or set())}
    results: list[dict[str, object]] = []
    total = len(normalized)

    provider_list = [str(provider).strip().lower() for provider in providers if str(provider).strip()]

    for index, symbol in enumerate(normalized, start=1):
        is_etf = symbol in etf_set
        price_section = "etf_prices" if is_etf else "prices"
        asset_class = "etf" if is_etf else "equity"

        if symbol_has_fresh_prices(
            warehouse.catalog,
            symbol,
            provider_list,
            target_end_date=target_end_date,
            section=price_section,
        ):
            fresh_provider = next(
                (
                    provider
                    for provider in provider_list
                    if (catalog_price_max_date(warehouse.catalog, symbol, provider, section=price_section) or date.min)
                    >= target_end_date
                ),
                provider_list[0] if provider_list else "",
            )
            results.append(
                {
                    "symbol": symbol,
                    "provider": fresh_provider,
                    "asset_class": asset_class,
                    "status": "skipped_fresh",
                    "reason": "fresh",
                }
            )
            if callable(progress_logger) and (index == 1 or index % 25 == 0 or index == total):
                progress_logger(f"Warehouse price refresh progress: {index:,}/{total:,} symbols processed")
            continue

        symbol_reached_target = False
        for provider in provider_list:
            if backfill_skip:
                needs_refresh, reason = price_backfill_needs_update(
                    warehouse.catalog,
                    symbol,
                    provider,
                    target_end_date=target_end_date,
                    skip_recent_hours=skip_recent_hours,
                    section=price_section,
                    is_etf=is_etf,
                )
            else:
                needs_refresh, reason = price_refresh_needs_update(
                    warehouse.catalog,
                    symbol,
                    provider,
                    target_end_date=target_end_date,
                    skip_recent_hours=skip_recent_hours,
                    section=price_section,
                )
            provider_max = catalog_price_max_date(warehouse.catalog, symbol, provider, section=price_section)
            if not needs_refresh and provider_max is not None and provider_max >= target_end_date:
                results.append(
                    {
                        "symbol": symbol,
                        "provider": provider,
                        "asset_class": asset_class,
                        "status": "skipped_fresh",
                        "reason": reason,
                    }
                )
                symbol_reached_target = True
                break
            if not needs_refresh:
                continue
            try:
                symbol_start = (
                    warehouse.catalog.equity_historical_start(symbol) if not is_etf else start_text
                )
                plan = historical_fetch_plan(
                    warehouse.catalog,
                    symbol,
                    price_section,
                    provider,
                    target_end_date=target_end_date,
                    skip_recent_hours=skip_recent_hours,
                    is_etf=is_etf,
                )
                refresh_kwargs: dict[str, object] = {
                    "providers": [provider],
                    **fetch_kwargs_from_plan(plan, default_start=symbol_start),
                }
                if is_etf:
                    stats = warehouse.etf.refresh_prices(symbol, **refresh_kwargs)
                else:
                    stats = warehouse.refresh_prices(symbol, **refresh_kwargs)
                provider_stats = dict(stats.get(provider) or {})
                refreshed_max = catalog_price_max_date(warehouse.catalog, symbol, provider, section=price_section)
                reached_target = refreshed_max is not None and refreshed_max >= target_end_date
                status = "updated" if reached_target else "still_stale"
                results.append(
                    {
                        "symbol": symbol,
                        "provider": provider,
                        "asset_class": asset_class,
                        "status": status,
                        "reason": reason,
                        "fetch_mode": plan.mode,
                        "rows": int(provider_stats.get("rows") or 0),
                        "fetched_rows": int(provider_stats.get("fetched_rows") or 0),
                        "max_date": provider_stats.get("max_date"),
                        "fetch_start": provider_stats.get("fetch_start"),
                    }
                )
                if reached_target:
                    symbol_reached_target = True
                    break
            except Exception as exc:
                results.append(
                    {
                        "symbol": symbol,
                        "provider": provider,
                        "asset_class": asset_class,
                        "status": "error",
                        "reason": reason,
                        "error": str(exc),
                    }
                )
        if callable(progress_logger) and (index == 1 or index % 25 == 0 or index == total):
            progress_logger(f"Warehouse price refresh progress: {index:,}/{total:,} symbols processed")
    return results


def refresh_universe_fundamentals(
    warehouse: Warehouse,
    symbols: Sequence[str],
    *,
    sections: Sequence[str],
    providers: Sequence[str],
    period: str = "quarter",
    etf_symbols: set[str] | None = None,
    staleness_days: int = 90,
    skip_recent_hours: float = 24.0,
    force_sections: frozenset[str] | None = None,
    backfill_skip: bool = False,
    max_symbols: int | None = None,
    progress_logger: ProgressLogger = None,
) -> list[dict[str, object]]:
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if max_symbols is not None:
        normalized = normalized[: max(0, int(max_symbols))]
    etf_set = {str(symbol).strip().upper() for symbol in (etf_symbols or set())}
    allowed_equity = set(EQUITY_FUNDAMENTAL_SECTIONS)
    allowed_etf = set(ETF_FUNDAMENTAL_SECTIONS)
    results: list[dict[str, object]] = []
    total = len(normalized)
    start_text = str(MIN_HISTORICAL_DATE)[:10]

    for index, symbol in enumerate(normalized, start=1):
        is_etf = symbol in etf_set
        allowed = allowed_etf if is_etf else allowed_equity
        target_sections = [section for section in sections if section in allowed]
        if not target_sections:
            results.append(
                {
                    "symbol": symbol,
                    "status": "skipped_no_sections",
                    "asset_class": "etf" if is_etf else "equity",
                }
            )
            continue
        for provider in providers:
            sections_to_refresh: list[str] = []
            section_plans: dict[str, object] = {}
            for section in target_sections:
                if backfill_skip:
                    needs_refresh, reason = backfill_fundamental_needs_update(
                        warehouse.catalog,
                        symbol,
                        section,
                        provider,
                        staleness_days=staleness_days,
                        skip_recent_hours=skip_recent_hours,
                        is_etf=is_etf,
                    )
                else:
                    needs_refresh, reason = fundamental_refresh_needs_update(
                        warehouse.catalog,
                        symbol,
                        section,
                        provider,
                        staleness_days=staleness_days,
                        skip_recent_hours=skip_recent_hours,
                    )
                plan = historical_fetch_plan(
                    warehouse.catalog,
                    symbol,
                    section,
                    provider,
                    staleness_days=staleness_days,
                    skip_recent_hours=skip_recent_hours,
                    is_etf=is_etf,
                )
                if force_sections and section in force_sections:
                    needs_refresh = True
                    reason = "forced_period_backfill"
                if not needs_refresh:
                    results.append(
                        {
                            "symbol": symbol,
                            "section": section,
                            "provider": provider,
                            "status": "skipped_fresh",
                            "reason": reason,
                            "fetch_mode": plan.mode,
                        }
                    )
                    continue
                sections_to_refresh.append(section)
                section_plans[section] = plan

            for section in sections_to_refresh:
                plan = section_plans[section]
                try:
                    default_start = (
                        warehouse.catalog.equity_historical_start(symbol) if not is_etf else start_text
                    )
                    refresh_kwargs = fetch_kwargs_from_plan(plan, default_start=default_start)
                    if is_etf:
                        stats = warehouse.etf.refresh_fundamentals(
                            symbol,
                            sections=[section],
                            providers=[provider],
                            period=period,
                            **refresh_kwargs,
                        )
                    else:
                        stats = warehouse.refresh_fundamentals(
                            symbol,
                            sections=[section],
                            providers=[provider],
                            period=period,
                            **refresh_kwargs,
                        )
                    key = f"{section}:{provider}"
                    results.append(
                        {
                            "symbol": symbol,
                            "section": section,
                            "provider": provider,
                            "status": "updated",
                            "reason": plan.reason,
                            "fetch_mode": plan.mode,
                            "rows": int(stats.get(key) or 0),
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "symbol": symbol,
                            "section": section,
                            "provider": provider,
                            "status": "error",
                            "reason": plan.reason,
                            "fetch_mode": plan.mode,
                            "error": str(exc),
                        }
                    )
        if callable(progress_logger) and (index == 1 or index % 25 == 0 or index == total):
            progress_logger(f"Warehouse fundamental refresh progress: {index:,}/{total:,} symbols processed")
    return results


def refresh_universe_macro(
    warehouse: Warehouse,
    *,
    economic_series: Sequence[str] | None = None,
    include_treasury_rates: bool = True,
    provider: str = "fmp",
    target_end_date: date | None = None,
    macro_start_date: str = MIN_HISTORICAL_DATE,
    skip_recent_hours: float = 24.0,
    progress_logger: ProgressLogger = None,
) -> list[dict[str, object]]:
    target_end_date = target_end_date or expected_latest_price_date()
    end_text = target_end_date.isoformat()
    start_text = str(macro_start_date or MIN_HISTORICAL_DATE)[:10]
    history_start = pd.Timestamp(start_text).date()
    provider_name = str(provider or "fmp").strip().lower()
    results: list[dict[str, object]] = []
    series_list = list(economic_series or DEFAULT_ECONOMIC_SERIES)
    full_history_reasons = {"missing", "below_min_historical_date", "incomplete_early_history"}

    for series_name in series_list:
        needs_refresh, reason = macro_refresh_needs_update(
            warehouse.catalog,
            series_name,
            "macro_economic",
            provider_name,
            target_end_date=target_end_date,
            history_start_date=history_start,
            skip_recent_hours=skip_recent_hours,
        )
        if not needs_refresh:
            results.append(
                {
                    "dataset": "macro_economic",
                    "series": series_name,
                    "status": "skipped_fresh",
                    "reason": reason,
                }
            )
            continue
        try:
            fetch_start = start_text if reason in full_history_reasons else None
            stats = warehouse.macro.refresh_economic_series(
                series_name,
                provider=provider_name,
                start_date=fetch_start,
                end_date=end_text,
                full_refresh=False,
            )
            results.append(
                {
                    "dataset": "macro_economic",
                    "series": series_name,
                    "status": "updated",
                    "reason": reason,
                    "rows": int(stats.get("rows") or 0),
                    "max_date": stats.get("max_date"),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "dataset": "macro_economic",
                    "series": series_name,
                    "status": "error",
                    "reason": reason,
                    "error": str(exc),
                }
            )

    if include_treasury_rates:
        needs_refresh, reason = macro_refresh_needs_update(
            warehouse.catalog,
            TREASURY_BUNDLE_SYMBOL,
            MACRO_TREASURY_SECTION,
            provider_name,
            target_end_date=target_end_date,
            history_start_date=history_start,
            skip_recent_hours=skip_recent_hours,
        )
        if not needs_refresh:
            results.append(
                {
                    "dataset": "macro_treasury",
                    "series": TREASURY_BUNDLE_SYMBOL,
                    "status": "skipped_fresh",
                    "reason": reason,
                }
            )
        else:
            try:
                fetch_start = start_text if reason in full_history_reasons else None
                stats = warehouse.macro.refresh_treasury_rates(
                    provider=provider_name,
                    start_date=fetch_start,
                    end_date=end_text,
                    full_refresh=False,
                )
                results.append(
                    {
                        "dataset": "macro_treasury",
                        "series": TREASURY_BUNDLE_SYMBOL,
                        "status": "updated",
                        "reason": reason,
                        "series_count": int(stats.get("series_count") or 0),
                        "max_date": stats.get("max_date"),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "dataset": "macro_treasury",
                        "series": TREASURY_BUNDLE_SYMBOL,
                        "status": "error",
                        "reason": reason,
                        "error": str(exc),
                    }
                )

    if callable(progress_logger):
        updated = sum(1 for row in results if row.get("status") == "updated")
        skipped = sum(1 for row in results if row.get("status") == "skipped_fresh")
        progress_logger(f"Warehouse macro refresh complete | updated {updated:,} | skipped {skipped:,}")
    return results


def refresh_universe_profiles(
    warehouse: Warehouse,
    symbols: Sequence[str],
    *,
    providers: Sequence[str],
    etf_symbols: set[str] | None = None,
    refresh_days: int = 30,
    max_symbols: int | None = None,
    progress_logger: ProgressLogger = None,
) -> list[dict[str, object]]:
    normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if max_symbols is not None:
        normalized = normalized[: max(0, int(max_symbols))]
    etf_set = {str(symbol).strip().upper() for symbol in (etf_symbols or set())}
    results: list[dict[str, object]] = []
    total = len(normalized)

    for index, symbol in enumerate(normalized, start=1):
        is_etf = symbol in etf_set
        for provider in providers:
            needs_refresh, reason = profile_refresh_needs_update(
                warehouse.catalog,
                symbol,
                provider,
                refresh_days=refresh_days,
                is_etf=is_etf,
            )
            if not needs_refresh:
                results.append(
                    {
                        "symbol": symbol,
                        "provider": provider,
                        "status": "skipped_fresh",
                        "reason": reason,
                    }
                )
                continue
            try:
                if is_etf:
                    stats = warehouse.etf.refresh_profile(symbol, provider=provider)
                else:
                    stats = warehouse.refresh_profile(symbol, provider=provider)
                results.append(
                    {
                        "symbol": symbol,
                        "provider": provider,
                        "status": "updated",
                        "reason": reason,
                        "fields_populated": int(stats.get("fields_populated") or 0),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "symbol": symbol,
                        "provider": provider,
                        "status": "error",
                        "reason": reason,
                        "error": str(exc),
                    }
                )
        if callable(progress_logger) and (index == 1 or index % 25 == 0 or index == total):
            progress_logger(f"Warehouse profile refresh progress: {index:,}/{total:,} symbols processed")
    return results