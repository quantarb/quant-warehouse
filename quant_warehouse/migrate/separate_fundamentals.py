from __future__ import annotations

import re
from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import (
    EQUITY_FUNDAMENTAL_SECTIONS,
    LEGACY_FUNDAMENTALS_LIBRARY,
    fundamental_library,
)

_PREFIX_RE = re.compile(r"^[a-z0-9]+__")


def separate_legacy_fundamentals(
    *,
    symbols: Sequence[str] | None = None,
    sections: Sequence[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    """Move merged `fundamentals` library rows into per-route libraries using catalog column lists."""
    warehouse = Warehouse()
    target_sections = list(sections or EQUITY_FUNDAMENTAL_SECTIONS)
    results: list[dict[str, object]] = []

    if symbols is None:
        symbols = _symbols_with_legacy_fundamentals(warehouse.catalog)

    for symbol in symbols:
        symbol = symbol.strip().upper()
        for section in target_sections:
            for provider in _providers_for_section(warehouse.catalog, symbol=symbol, section=section):
                migrated = _migrate_symbol_section(
                    warehouse,
                    symbol=symbol,
                    section=section,
                    provider=provider,
                    dry_run=dry_run,
                )
                if migrated["rows"] > 0 or migrated.get("skipped"):
                    results.append(migrated)

    return results


def _symbols_with_legacy_fundamentals(catalog: CatalogStore) -> list[str]:
    with catalog._connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT symbol FROM section_state
            WHERE section IN ({})
            ORDER BY symbol
            """.format(",".join("?" * len(EQUITY_FUNDAMENTAL_SECTIONS))),
            tuple(EQUITY_FUNDAMENTAL_SECTIONS),
        ).fetchall()
    return [row["symbol"] for row in rows]


def _providers_for_section(
    catalog: CatalogStore,
    *,
    symbol: str,
    section: str,
) -> list[str]:
    rows = catalog.list_symbol(symbol)
    return sorted({row.provider for row in rows if row.section == section})


def _migrate_symbol_section(
    warehouse: Warehouse,
    *,
    symbol: str,
    section: str,
    provider: str,
    dry_run: bool,
) -> dict[str, object]:
    state = warehouse.catalog.get(symbol=symbol, section=section, provider=provider)
    if state is None or not state.columns_present:
        return {"symbol": symbol, "section": section, "provider": provider, "rows": 0, "skipped": True}

    storage_symbol = symbol_provider_key(symbol, provider)
    legacy = warehouse.backend.read(LEGACY_FUNDAMENTALS_LIBRARY, storage_symbol)
    if legacy is None or legacy.empty:
        return {"symbol": symbol, "section": section, "provider": provider, "rows": 0, "skipped": True}

    target_library = fundamental_library(section)
    existing = warehouse.backend.read(target_library, storage_symbol)
    if existing is not None and not existing.empty:
        return {"symbol": symbol, "section": section, "provider": provider, "rows": 0, "skipped": True}

    columns = _resolve_legacy_columns(legacy, state.columns_present, provider=provider)
    if not columns:
        return {"symbol": symbol, "section": section, "provider": provider, "rows": 0, "error": "no_columns"}

    frame = legacy[columns].copy()
    frame = _strip_provider_prefix(frame, provider=provider)
    if not dry_run and not frame.empty:
        warehouse.backend.write(target_library, storage_symbol, frame)

    return {
        "symbol": symbol,
        "section": section,
        "provider": provider,
        "rows": len(frame),
        "library": target_library,
        "dry_run": dry_run,
    }


def _resolve_legacy_columns(
    legacy: pd.DataFrame,
    catalog_columns: Sequence[str],
    *,
    provider: str,
) -> list[str]:
    available = set(legacy.columns)
    resolved: list[str] = []
    for column in catalog_columns:
        if column in available:
            resolved.append(column)
            continue
        prefixed = f"{provider}__{column}"
        if prefixed in available:
            resolved.append(prefixed)
    return resolved


def _strip_provider_prefix(frame: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    rename: dict[str, str] = {}
    prefix = f"{provider}__"
    for column in frame.columns:
        if column.startswith(prefix):
            rename[column] = column[len(prefix) :]
        elif _PREFIX_RE.match(column):
            rename[column] = column.split("__", 1)[1]
    out = frame.rename(columns=rename)
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
    return out