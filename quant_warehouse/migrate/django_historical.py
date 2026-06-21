from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from quant_warehouse.ingest.django_historical import (
    django_historical_frame,
    django_section_keys,
    list_django_symbols_for_section,
    warehouse_section_for_django,
)
from quant_warehouse.warehouse.api import Warehouse

DEFAULT_PROVIDER = "fmp"


def migrate_django_historical(
    db_path: Path | str,
    *,
    section_keys: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    """Copy Django fmp_symbolsectionhistorical rows into per-section Arctic libraries."""
    db_path = Path(db_path).expanduser().resolve()
    warehouse = Warehouse()
    target_sections = list(section_keys or django_section_keys())
    stats: list[dict[str, object]] = []

    for django_section in target_sections:
        wh_section = warehouse_section_for_django(django_section)
        if symbols:
            section_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        else:
            section_symbols = list_django_symbols_for_section(
                db_path,
                django_section,
                limit=limit,
                offset=offset,
            )

        total = len(section_symbols)
        label = f"migrate-{django_section}"
        for index, symbol in enumerate(section_symbols, start=1):
            if skip_existing:
                existing = warehouse.catalog.get(
                    symbol=symbol,
                    section=wh_section,
                    provider=DEFAULT_PROVIDER,
                )
                if existing is not None and int(existing.row_count) > 0:
                    stats.append(
                        {
                            "symbol": symbol,
                            "django_section": django_section,
                            "section": wh_section,
                            "skipped": True,
                            "rows": int(existing.row_count),
                        }
                    )
                    continue

            try:
                frame = django_historical_frame(db_path, symbol, django_section)
                if frame.empty:
                    stats.append(
                        {
                            "symbol": symbol,
                            "django_section": django_section,
                            "section": wh_section,
                            "rows": 0,
                            "error": "no django historical rows",
                        }
                    )
                    continue

                result = warehouse.fundamentals.ingest_frame(
                    symbol,
                    section=wh_section,
                    provider=DEFAULT_PROVIDER,
                    frame=frame,
                    merge=False,
                )
                stats.append(
                    {
                        "symbol": symbol,
                        "django_section": django_section,
                        "section": wh_section,
                        "skipped": False,
                        **result,
                    }
                )
            except Exception as exc:
                stats.append(
                    {
                        "symbol": symbol,
                        "django_section": django_section,
                        "section": wh_section,
                        "rows": 0,
                        "error": str(exc),
                    }
                )
                continue
            if index % 25 == 0 or index == total:
                last = stats[-1] if stats else {}
                print(
                    f"[{label}] {index}/{total} last={symbol} "
                    f"rows={last.get('rows', last.get('error', 'skipped'))}",
                    file=sys.stderr,
                    flush=True,
                )

    return stats