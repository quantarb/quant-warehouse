from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from quant_warehouse.ingest.django_symbols import list_django_symbols
from quant_warehouse.ingest.providers import validate_price_provider
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.ingest.providers import validate_fundamental_provider
from quant_warehouse.warehouse.sections import (
    ETF_FUNDAMENTAL_SECTIONS,
    ETF_PRICES_SECTION,
    EQUITY_FUNDAMENTAL_SECTIONS,
    EQUITY_PRICES_SECTION,
)


def refresh_equity_profiles_from_django_db(
    db_path: Path | str,
    *,
    providers: Sequence[str],
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    target_symbols = _resolve_symbols(
        db_path,
        symbols=symbols,
        asset_class="equity",
        require_prices=False,
        limit=limit,
        offset=offset,
    )
    warehouse = Warehouse()
    provider_list = [validate_price_provider(provider) for provider in providers]
    stats: list[dict[str, object]] = []
    total = len(target_symbols)
    for index, symbol in enumerate(target_symbols, start=1):
        for provider in provider_list:
            if skip_existing and warehouse.catalog.get_profile(symbol=symbol, provider=provider) is not None:
                stats.append({"symbol": symbol, "provider_requested": provider, "skipped": True})
                continue
            try:
                stats.append(warehouse.profiles.refresh(symbol, provider=provider))
            except Exception as exc:
                stats.append(
                    {"symbol": symbol, "provider_requested": provider, "error": str(exc)},
                )
        if index % 25 == 0 or index == total:
            print(f"[refresh-equity-profiles] {index}/{total} last={symbol}", file=sys.stderr, flush=True)
    return stats


def refresh_etf_profiles_from_django_db(
    db_path: Path | str,
    *,
    providers: Sequence[str],
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    target_symbols = _resolve_symbols(
        db_path,
        symbols=symbols,
        asset_class="etf",
        require_prices=False,
        limit=limit,
        offset=offset,
    )
    warehouse = Warehouse()
    provider_list = [validate_price_provider(provider) for provider in providers]
    stats: list[dict[str, object]] = []
    total = len(target_symbols)
    for index, symbol in enumerate(target_symbols, start=1):
        for provider in provider_list:
            if skip_existing and warehouse.catalog.get_etf_profile(symbol=symbol, provider=provider) is not None:
                stats.append({"symbol": symbol, "provider_requested": provider, "skipped": True})
                continue
            try:
                stats.append(warehouse.etf.refresh_profile(symbol, provider=provider))
            except Exception as exc:
                stats.append(
                    {"symbol": symbol, "provider_requested": provider, "error": str(exc)},
                )
        if index % 25 == 0 or index == total:
            print(f"[refresh-etf-profiles] {index}/{total} last={symbol}", file=sys.stderr, flush=True)
    return stats


def refresh_equity_yfinance_prices_from_django_db(
    db_path: Path | str,
    *,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    start_date: str | None = "1980-01-01",
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    return _refresh_yfinance_prices(
        db_path,
        asset_class="equity",
        symbols=symbols,
        limit=limit,
        offset=offset,
        start_date=start_date,
        skip_existing=skip_existing,
        label="refresh-equity-yfinance-prices",
    )


def refresh_etf_yfinance_prices_from_django_db(
    db_path: Path | str,
    *,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    start_date: str | None = "1980-01-01",
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    return _refresh_yfinance_prices(
        db_path,
        asset_class="etf",
        symbols=symbols,
        limit=limit,
        offset=offset,
        start_date=start_date,
        skip_existing=skip_existing,
        label="refresh-etf-yfinance-prices",
    )


def _resolve_symbols(
    db_path: Path | str,
    *,
    symbols: Sequence[str] | None,
    asset_class: str,
    require_prices: bool,
    limit: int | None,
    offset: int,
) -> list[str]:
    if symbols:
        return [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    return list_django_symbols(
        Path(db_path),
        asset_class=asset_class,  # type: ignore[arg-type]
        require_prices=require_prices,
        limit=limit,
        offset=offset,
    )


def _refresh_yfinance_prices(
    db_path: Path | str,
    *,
    asset_class: str,
    symbols: Sequence[str] | None,
    limit: int | None,
    offset: int,
    start_date: str | None,
    skip_existing: bool,
    label: str,
) -> list[dict[str, object]]:
    warehouse = Warehouse()
    target_symbols = _resolve_symbols(
        db_path,
        symbols=symbols,
        asset_class=asset_class,
        require_prices=True,
        limit=limit,
        offset=offset,
    )
    section = EQUITY_PRICES_SECTION if asset_class == "equity" else ETF_PRICES_SECTION
    stats: list[dict[str, object]] = []
    total = len(target_symbols)
    for index, symbol in enumerate(target_symbols, start=1):
        if skip_existing:
            existing = warehouse.catalog.get(symbol=symbol, section=section, provider="yfinance")
            if existing is not None and int(existing.row_count) > 0:
                stats.append(
                    {
                        "symbol": symbol,
                        "provider": "yfinance",
                        "skipped": True,
                        "rows": int(existing.row_count),
                    }
                )
                continue

        fetch_start = start_date
        if asset_class == "equity":
            fmp_state = warehouse.catalog.get(symbol=symbol, section=EQUITY_PRICES_SECTION, provider="fmp")
        else:
            fmp_state = warehouse.catalog.get(symbol=symbol, section=ETF_PRICES_SECTION, provider="fmp")
        if fetch_start is None and fmp_state is not None and fmp_state.min_date:
            fetch_start = fmp_state.min_date

        try:
            if asset_class == "equity":
                result = warehouse.refresh_prices(
                    symbol,
                    providers=["yfinance"],
                    start_date=fetch_start,
                    full_refresh=True,
                )
            else:
                result = warehouse.etf.refresh_prices(
                    symbol,
                    providers=["yfinance"],
                    start_date=fetch_start,
                    full_refresh=True,
                )
            yf = result.get("yfinance", {})
            stats.append({"symbol": symbol, "provider": "yfinance", "skipped": False, **yf})
        except Exception as exc:
            stats.append({"symbol": symbol, "provider": "yfinance", "error": str(exc)})

        if index % 25 == 0 or index == total:
            last = stats[-1] if stats else {}
            print(
                f"[{label}] {index}/{total} last={symbol} rows={last.get('rows', last.get('error', 'skipped'))}",
                file=sys.stderr,
                flush=True,
            )
    return stats


def refresh_equity_fundamentals_from_django_db(
    db_path: Path | str,
    *,
    providers: Sequence[str],
    sections: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    skip_existing: bool = True,
    period: str = "annual",
) -> list[dict[str, object]]:
    target_symbols = _resolve_symbols(
        db_path,
        symbols=symbols,
        asset_class="equity",
        require_prices=True,
        limit=limit,
        offset=offset,
    )
    warehouse = Warehouse()
    provider_list = [validate_fundamental_provider(provider) for provider in providers]
    section_list = list(sections or EQUITY_FUNDAMENTAL_SECTIONS)
    stats: list[dict[str, object]] = []
    total = len(target_symbols)
    for index, symbol in enumerate(target_symbols, start=1):
        if skip_existing and _has_all_fundamental_sections(
            warehouse,
            symbol=symbol,
            sections=section_list,
            providers=provider_list,
        ):
            stats.append({"symbol": symbol, "skipped": True})
            continue
        try:
            rows = warehouse.refresh_fundamentals(
                symbol,
                sections=section_list,
                providers=provider_list,
                period=period,
            )
            stats.append({"symbol": symbol, "skipped": False, "rows": rows})
        except Exception as exc:
            stats.append({"symbol": symbol, "error": str(exc)})
        if index % 10 == 0 or index == total:
            print(f"[refresh-equity-fundamentals] {index}/{total} last={symbol}", file=sys.stderr, flush=True)
    return stats


def refresh_etf_fundamentals_from_django_db(
    db_path: Path | str,
    *,
    providers: Sequence[str],
    sections: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    skip_existing: bool = True,
    period: str = "annual",
) -> list[dict[str, object]]:
    target_symbols = _resolve_symbols(
        db_path,
        symbols=symbols,
        asset_class="etf",
        require_prices=True,
        limit=limit,
        offset=offset,
    )
    warehouse = Warehouse()
    provider_list = [validate_fundamental_provider(provider) for provider in providers]
    section_list = list(sections or ETF_FUNDAMENTAL_SECTIONS)
    stats: list[dict[str, object]] = []
    total = len(target_symbols)
    for index, symbol in enumerate(target_symbols, start=1):
        if skip_existing and _has_all_fundamental_sections(
            warehouse,
            symbol=symbol,
            sections=section_list,
            providers=provider_list,
        ):
            stats.append({"symbol": symbol, "skipped": True})
            continue
        try:
            rows = warehouse.etf.refresh_fundamentals(
                symbol,
                sections=section_list,
                providers=provider_list,
                period=period,
            )
            stats.append({"symbol": symbol, "skipped": False, "rows": rows})
        except Exception as exc:
            stats.append({"symbol": symbol, "error": str(exc)})
        if index % 10 == 0 or index == total:
            print(f"[refresh-etf-fundamentals] {index}/{total} last={symbol}", file=sys.stderr, flush=True)
    return stats


def _has_all_fundamental_sections(
    warehouse: Warehouse,
    *,
    symbol: str,
    sections: Sequence[str],
    providers: Sequence[str],
) -> bool:
    for section in sections:
        for provider in providers:
            state = warehouse.catalog.get(symbol=symbol, section=section, provider=provider)
            if state is None or int(state.row_count) <= 0:
                return False
    return True