from __future__ import annotations

import re
from typing import Sequence

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.sections import (
    EQUITY_FUNDAMENTAL_SECTIONS,
    LEGACY_FUNDAMENTALS_LIBRARY,
    fundamental_library,
)
from quant_warehouse.warehouse.storage import provider_library

_PREFIX_RE = re.compile(r"^[a-z0-9]+__")


def separate_legacy_fundamentals(
    *,
    symbols: Sequence[str] | None = None,
    sections: Sequence[str] | None = None,
    dry_run: bool = False,
    delete_verified_legacy: bool = False,
) -> list[dict[str, object]]:
    """Move merged `fundamentals` rows into provider-scoped per-route libraries."""
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
                    delete_verified_legacy=delete_verified_legacy,
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
    delete_verified_legacy: bool,
) -> dict[str, object]:
    state = warehouse.catalog.get(symbol=symbol, section=section, provider=provider)
    if state is None or not state.columns_present:
        return {"symbol": symbol, "section": section, "provider": provider, "rows": 0, "skipped": True}

    storage_symbol = symbol_provider_key(symbol, provider)
    legacy = warehouse.backend.read(LEGACY_FUNDAMENTALS_LIBRARY, storage_symbol)
    if legacy is None or legacy.empty:
        return {"symbol": symbol, "section": section, "provider": provider, "rows": 0, "skipped": True}

    target_library = provider_library(fundamental_library(section), provider)
    existing = warehouse.backend.read(target_library, storage_symbol)
    if existing is not None and not existing.empty:
        if not dry_run and delete_verified_legacy:
            columns = _resolve_legacy_columns(legacy, state.columns_present, provider=provider)
            frame = _strip_provider_prefix(legacy[columns].copy(), provider=provider) if columns else pd.DataFrame()
            if not frame.empty:
                _assert_verified_copy(frame, existing)
                deleted = warehouse.backend.delete(LEGACY_FUNDAMENTALS_LIBRARY, storage_symbol)
                return {
                    "symbol": symbol,
                    "section": section,
                    "provider": provider,
                    "rows": len(existing),
                    "library": target_library,
                    "skipped": True,
                    "reason": "already_migrated",
                    "deleted_legacy": deleted,
                }
        return {"symbol": symbol, "section": section, "provider": provider, "rows": 0, "skipped": True}

    columns = _resolve_legacy_columns(legacy, state.columns_present, provider=provider)
    if not columns:
        return {"symbol": symbol, "section": section, "provider": provider, "rows": 0, "error": "no_columns"}

    frame = legacy[columns].copy()
    frame = _strip_provider_prefix(frame, provider=provider)
    deleted = False
    if not dry_run and not frame.empty:
        warehouse.backend.write(target_library, storage_symbol, frame)
        copied = warehouse.backend.read(target_library, storage_symbol)
        _assert_verified_copy(frame, copied)
        if delete_verified_legacy:
            deleted = warehouse.backend.delete(LEGACY_FUNDAMENTALS_LIBRARY, storage_symbol)

    return {
        "symbol": symbol,
        "section": section,
        "provider": provider,
        "rows": len(frame),
        "library": target_library,
        "dry_run": dry_run,
        "deleted_legacy": deleted,
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


def _assert_verified_copy(source_frame: pd.DataFrame, copied_frame: pd.DataFrame | None) -> None:
    if copied_frame is None or copied_frame.empty:
        raise RuntimeError("copied frame is empty")
    if len(source_frame) != len(copied_frame):
        raise RuntimeError(f"row count mismatch: source={len(source_frame)} copied={len(copied_frame)}")
    if list(source_frame.columns) != list(copied_frame.columns):
        raise RuntimeError("column mismatch")
    if _min_date_text(source_frame) != _min_date_text(copied_frame):
        raise RuntimeError("min date mismatch")
    if _max_date_text(source_frame) != _max_date_text(copied_frame):
        raise RuntimeError("max date mismatch")


def _min_date_text(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    return frame.index.min().strftime("%Y-%m-%d")


def _max_date_text(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    return frame.index.max().strftime("%Y-%m-%d")
