from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.platforms.data_providers.thetadata.options import (
    OPTIONS_THETADATA_EOD_LIBRARY,
    OPTIONS_THETADATA_PROVIDER,
)
from quant_warehouse.warehouse.backend import ArcticBackend
from quant_warehouse.warehouse.prices import parse_symbol_provider_key
from quant_warehouse.warehouse.storage import provider_library

PROVIDER_SPECIFIC_LEGACY_LIBRARIES: dict[str, str] = {
    OPTIONS_THETADATA_EOD_LIBRARY: OPTIONS_THETADATA_PROVIDER,
}


@dataclass(frozen=True)
class ProviderMigrationRow:
    provider: str
    source_uri: str
    source_library: str
    source_symbol: str
    target_uri: str
    target_library: str
    target_symbol: str
    rows: int
    columns: int
    min_date: str | None
    max_date: str | None
    status: str
    deleted_legacy: bool = False
    error: str | None = None


def migrate_legacy_provider_storage(
    provider: str,
    *,
    libraries: Sequence[str] | None = None,
    dry_run: bool = True,
    delete_verified_legacy: bool = False,
    limit: int | None = None,
    config: WarehouseConfig | None = None,
) -> list[ProviderMigrationRow]:
    """Move legacy shared Arctic symbols into a provider-isolated Arctic root."""

    provider_name = str(provider).strip().lower()
    if not provider_name:
        raise ValueError("provider is required")

    config = config or WarehouseConfig.from_env()
    config.ensure_dirs()
    source = ArcticBackend(config.arctic_uri)
    target_uri = config.provider_arctic_uri(provider_name)
    target = None if dry_run else ArcticBackend(target_uri)
    source_libraries = list(libraries or _safe_list_libraries(source))
    rows: list[ProviderMigrationRow] = []
    planned_or_copied = 0

    for source_library in source_libraries:
        for source_symbol in _safe_list_symbols(source, source_library):
            match = _legacy_symbol_match(source_library, source_symbol, provider_name)
            if match is None:
                continue
            if limit is not None and planned_or_copied >= int(limit):
                return rows

            target_symbol = match
            target_library = provider_library(source_library, provider_name)
            try:
                frame = source.read(source_library, source_symbol)
                if frame is None or frame.empty:
                    rows.append(
                        _row(
                            provider_name,
                            config,
                            source_library,
                            source_symbol,
                            target_uri,
                            target_library,
                            target_symbol,
                            pd.DataFrame(),
                            status="skipped_empty",
                        )
                    )
                    continue

                deleted = False
                if not dry_run:
                    if target is None:
                        raise RuntimeError("target backend was not initialized")
                    target.write(target_library, target_symbol, frame)
                    copied = target.read(target_library, target_symbol)
                    _assert_verified_copy(frame, copied)
                    if delete_verified_legacy:
                        deleted = source.delete(source_library, source_symbol)

                rows.append(
                    _row(
                        provider_name,
                        config,
                        source_library,
                        source_symbol,
                        target_uri,
                        target_library,
                        target_symbol,
                        frame,
                        status="planned" if dry_run else "copied",
                        deleted_legacy=deleted,
                    )
                )
                planned_or_copied += 1
            except Exception as exc:
                rows.append(
                    _row(
                        provider_name,
                        config,
                        source_library,
                        source_symbol,
                        target_uri,
                        target_library,
                        target_symbol,
                        pd.DataFrame(),
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return rows


def summarize_provider_migration(
    rows: Sequence[ProviderMigrationRow],
    *,
    provider: str,
    dry_run: bool,
) -> dict[str, object]:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.status] = statuses.get(row.status, 0) + 1
    return {
        "provider": str(provider).strip().lower(),
        "dry_run": bool(dry_run),
        "items": int(len(rows)),
        "rows_total": int(sum(row.rows for row in rows)),
        "deleted_legacy": int(sum(1 for row in rows if row.deleted_legacy)),
        "statuses": statuses,
        "results": [asdict(row) for row in rows],
    }


def write_provider_migration_log(summary: dict[str, object], *, log_path: str | Path) -> Path:
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def _legacy_symbol_match(source_library: str, source_symbol: str, provider: str) -> str | None:
    library_provider = PROVIDER_SPECIFIC_LEGACY_LIBRARIES.get(source_library)
    if library_provider == provider:
        return str(source_symbol).strip().upper()

    parsed = parse_symbol_provider_key(source_symbol)
    if parsed is None:
        return None
    symbol, symbol_provider = parsed
    if symbol_provider != provider:
        return None
    return f"{symbol}__{provider}"


def _safe_list_libraries(backend: ArcticBackend) -> list[str]:
    try:
        return sorted(str(name) for name in backend._arctic.list_libraries())
    except Exception:
        return []


def _safe_list_symbols(backend: ArcticBackend, library: str) -> list[str]:
    try:
        if library not in backend._arctic.list_libraries():
            return []
        return [str(symbol) for symbol in backend.list_symbols(library)]
    except Exception:
        return []


def _row(
    provider: str,
    config: WarehouseConfig,
    source_library: str,
    source_symbol: str,
    target_uri: str,
    target_library: str,
    target_symbol: str,
    frame: pd.DataFrame,
    *,
    status: str,
    deleted_legacy: bool = False,
    error: str | None = None,
) -> ProviderMigrationRow:
    return ProviderMigrationRow(
        provider=provider,
        source_uri=config.arctic_uri,
        source_library=source_library,
        source_symbol=source_symbol,
        target_uri=target_uri,
        target_library=target_library,
        target_symbol=target_symbol,
        rows=int(len(frame)),
        columns=0 if frame is None else int(len(frame.columns)),
        min_date=_min_date_text(frame),
        max_date=_max_date_text(frame),
        status=status,
        deleted_legacy=bool(deleted_legacy),
        error=error,
    )


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
