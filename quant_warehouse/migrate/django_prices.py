from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from quant_warehouse.ingest.django_prices import django_fmp_prices_frame
from quant_warehouse.ingest.django_symbols import django_is_etf, list_django_symbols
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import ETF_PRICES_SECTION, EQUITY_PRICES_SECTION


def migrate_django_fmp_prices(
    db_path: Path | str,
    *,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    db_path = Path(db_path).expanduser().resolve()
    warehouse = Warehouse()
    target_symbols = (
        [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        if symbols
        else list_django_symbols(db_path, asset_class="all", require_prices=True, limit=limit, offset=offset)
    )

    stats: list[dict[str, object]] = []
    total = len(target_symbols)
    for index, symbol in enumerate(target_symbols, start=1):
        is_etf = django_is_etf(db_path, symbol)
        section = ETF_PRICES_SECTION if is_etf else EQUITY_PRICES_SECTION
        if skip_existing:
            existing = warehouse.catalog.get(symbol=symbol, section=section, provider="fmp")
            if existing is not None and int(existing.row_count) > 0:
                stats.append(
                    {
                        "symbol": symbol,
                        "asset_class": "etf" if is_etf else "equity",
                        "skipped": True,
                        "rows": int(existing.row_count),
                    }
                )
                continue

        frame = django_fmp_prices_frame(db_path, symbol)
        if frame.empty:
            stats.append({"symbol": symbol, "rows": 0, "error": "no django price rows"})
            continue

        if is_etf:
            result = warehouse.etf.ingest_prices_frame(symbol, provider="fmp", frame=frame, merge=False)
        else:
            result = warehouse.prices.ingest_frame(symbol, provider="fmp", frame=frame, merge=False)
        stats.append(
            {
                "symbol": symbol,
                "asset_class": "etf" if is_etf else "equity",
                "skipped": False,
                **result,
            }
        )
        if index % 25 == 0 or index == total:
            print(
                f"[migrate-django-prices] {index}/{total} last={symbol} "
                f"class={'etf' if is_etf else 'equity'} rows={result.get('rows')}",
                file=sys.stderr,
                flush=True,
            )
    return stats