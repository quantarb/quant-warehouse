from __future__ import annotations

import sys
from pathlib import Path

from quant_warehouse.ingest.django_symbols import django_etf_symbol_set
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.ingest.providers import PRICE_PROVIDERS
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import (
    EQUITY_PRICES_LIBRARY,
    EQUITY_PRICES_SECTION,
    EQUITY_PROFILE_SECTION,
    ETF_PRICES_LIBRARY,
    ETF_PRICES_SECTION,
    ETF_PROFILE_SECTION,
)


def separate_etfs_from_equity(
    db_path: Path | str,
    *,
    symbols: set[str] | None = None,
) -> list[dict[str, object]]:
    db_path = Path(db_path).expanduser().resolve()
    warehouse = Warehouse()
    etf_symbols = symbols or django_etf_symbol_set(db_path)
    stats: list[dict[str, object]] = []
    total = len(etf_symbols)

    for index, symbol in enumerate(sorted(etf_symbols), start=1):
        moved_prices: list[str] = []
        for provider in PRICE_PROVIDERS:
            storage_symbol = symbol_provider_key(symbol, provider)
            state = warehouse.catalog.get(symbol=symbol, section=EQUITY_PRICES_SECTION, provider=provider)
            frame = warehouse.backend.read(EQUITY_PRICES_LIBRARY, storage_symbol)
            if frame is not None and not frame.empty:
                warehouse.backend.write(ETF_PRICES_LIBRARY, storage_symbol, frame)
                warehouse.backend.delete(EQUITY_PRICES_LIBRARY, storage_symbol)
                moved_prices.append(provider)
            if state is not None:
                warehouse.catalog.upsert(
                    symbol=symbol,
                    section=ETF_PRICES_SECTION,
                    provider=provider,
                    min_date=state.min_date,
                    max_date=state.max_date,
                    row_count=state.row_count,
                    columns_present=state.columns_present,
                )
                warehouse.catalog.delete_section(
                    symbol=symbol,
                    section=EQUITY_PRICES_SECTION,
                    provider=provider,
                )

        moved_profiles: list[str] = []
        for row in warehouse.catalog.list_profiles(symbol):
            warehouse.catalog.upsert_etf_profile(
                symbol=symbol,
                provider=row.provider,
                source_provider=row.source_provider,
                payload=row.payload,
            )
            warehouse.catalog.delete_profile(symbol=symbol, provider=row.provider)
            moved_profiles.append(row.provider)

        purged_equity: list[str] = []
        for provider in PRICE_PROVIDERS:
            storage_symbol = symbol_provider_key(symbol, provider)
            if warehouse.backend.has_symbol(EQUITY_PRICES_LIBRARY, storage_symbol):
                warehouse.backend.delete(EQUITY_PRICES_LIBRARY, storage_symbol)
                purged_equity.append(f"prices:{provider}")
            if warehouse.catalog.get(symbol=symbol, section=EQUITY_PRICES_SECTION, provider=provider):
                warehouse.catalog.delete_section(
                    symbol=symbol,
                    section=EQUITY_PRICES_SECTION,
                    provider=provider,
                )
        if warehouse.catalog.get_profile(symbol=symbol, provider="yfinance") or any(
            warehouse.catalog.list_profiles(symbol)
        ):
            for row in warehouse.catalog.list_profiles(symbol):
                warehouse.catalog.delete_profile(symbol=symbol, provider=row.provider)
                purged_equity.append(f"profile:{row.provider}")

        stats.append(
            {
                "symbol": symbol,
                "moved_price_providers": moved_prices,
                "moved_profile_providers": moved_profiles,
                "purged_equity": purged_equity,
            }
        )
        if index % 25 == 0 or index == total:
            print(
                f"[separate-etfs] {index}/{total} last={symbol} "
                f"prices={moved_prices} profiles={moved_profiles}",
                file=sys.stderr,
                flush=True,
            )

    return stats