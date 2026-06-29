from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.warehouse.backend import ArcticBackend
from quant_warehouse.warehouse.prices import PRICES_LIBRARY, parse_symbol_provider_key
from quant_warehouse.warehouse.sections import ALL_FUNDAMENTAL_SECTIONS, ETF_PRICES_LIBRARY, FUND_PRICES_LIBRARY
from quant_warehouse.warehouse.storage import provider_library

PROVIDER = "yfinance"

LEGACY_LIBRARIES: tuple[str, ...] = (
    PRICES_LIBRARY,
    ETF_PRICES_LIBRARY,
    *(f"fundamental_{section}" for section in sorted(ALL_FUNDAMENTAL_SECTIONS) if not section.startswith("etf_")),
    *(section for section in sorted(ALL_FUNDAMENTAL_SECTIONS) if section.startswith("etf_")),
)


@dataclass(frozen=True)
class MigrationRow:
    source_uri: str
    source_library: str
    source_symbol: str
    target_uri: str
    target_library: str
    target_symbol: str
    rows: int
    min_date: str | None
    max_date: str | None
    status: str
    error: str | None = None
    deleted_legacy: bool = False


def migrate_yfinance_storage(
    *,
    libraries: Sequence[str] = LEGACY_LIBRARIES,
    dry_run: bool = True,
    limit: int | None = None,
    skip_catalog_funds: bool = True,
    delete_verified_legacy: bool = False,
    config: WarehouseConfig | None = None,
) -> list[MigrationRow]:
    config = config or WarehouseConfig.from_env()
    config.ensure_dirs()
    catalog = CatalogStore(config.catalog_path)
    source = ArcticBackend(config.arctic_uri)
    target_uri = config.provider_arctic_uri(PROVIDER)
    target = None if dry_run else ArcticBackend(target_uri)
    rows: list[MigrationRow] = []
    planned_or_copied = 0

    for source_library in libraries:
        for source_symbol in _safe_list_symbols(source, source_library):
            if limit is not None and planned_or_copied >= int(limit):
                return rows
            parsed = parse_symbol_provider_key(source_symbol)
            if parsed is None:
                continue
            symbol, provider = parsed
            if provider != PROVIDER:
                continue

            target_library = _target_library_for_symbol(catalog, source_library, symbol, provider=PROVIDER)
            target_symbol = symbol_provider_key(symbol, PROVIDER)
            try:
                if skip_catalog_funds and _is_equity_fundamental_library(source_library) and _should_skip_equity_route_symbol(
                    catalog,
                    symbol,
                ):
                    rows.append(
                        _row(
                            config,
                            source_library,
                            source_symbol,
                            target_uri,
                            target_library,
                            target_symbol,
                            frame=pd.DataFrame(),
                            status="skipped_catalog_fund_or_etf",
                        )
                    )
                    continue
                frame = source.read(source_library, source_symbol)
                if frame is None or frame.empty:
                    rows.append(
                        _row(
                            config,
                            source_library,
                            source_symbol,
                            target_uri,
                            target_library,
                            target_symbol,
                            frame=pd.DataFrame(),
                            status="skipped_empty",
                        )
                    )
                    continue
                if not dry_run:
                    if target is None:
                        raise RuntimeError("target backend was not initialized")
                    target.write(target_library, target_symbol, frame)
                    if delete_verified_legacy:
                        copied = target.read(target_library, target_symbol)
                        _assert_verified_copy(frame, copied)
                        deleted_legacy = source.delete(source_library, source_symbol)
                    else:
                        deleted_legacy = False
                else:
                    deleted_legacy = False
                rows.append(
                    _row(
                        config,
                        source_library,
                        source_symbol,
                        target_uri,
                        target_library,
                        target_symbol,
                        frame=frame,
                        status="planned" if dry_run else "copied",
                        deleted_legacy=deleted_legacy,
                    )
                )
                planned_or_copied += 1
            except Exception as exc:
                rows.append(
                    _row(
                        config,
                        source_library,
                        source_symbol,
                        target_uri,
                        target_library,
                        target_symbol,
                        frame=pd.DataFrame(),
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return rows


def summarize(rows: Sequence[MigrationRow], *, dry_run: bool) -> dict[str, object]:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.status] = statuses.get(row.status, 0) + 1
    return {
        "provider": PROVIDER,
        "dry_run": bool(dry_run),
        "items": int(len(rows)),
        "rows_total": int(sum(row.rows for row in rows)),
        "statuses": statuses,
        "results": [asdict(row) for row in rows],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-off migration: copy legacy yfinance Arctic symbols into provider-isolated storage."
    )
    parser.add_argument("--libraries", default="", help="Comma-separated legacy libraries. Default: all yfinance candidates.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-funds",
        action="store_true",
        help=(
            "Also migrate fund/ETF-like symbols from equity fundamental libraries. "
            "Price rows are always redirected to etf_prices or fund_prices."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Write copied frames. Default is dry-run only.")
    parser.add_argument(
        "--delete-verified-legacy",
        action="store_true",
        help="After writing, delete each legacy yfinance symbol only if the copied frame verifies.",
    )
    parser.add_argument("--log", default="~/.quant-warehouse/logs/migrate-yfinance-provider-storage.json")
    args = parser.parse_args(argv)

    libraries = tuple(_parse_csv(args.libraries)) if args.libraries else LEGACY_LIBRARIES
    dry_run = not bool(args.apply)
    rows = migrate_yfinance_storage(
        libraries=libraries,
        dry_run=dry_run,
        limit=args.limit,
        skip_catalog_funds=not bool(args.include_funds),
        delete_verified_legacy=bool(args.delete_verified_legacy),
    )
    summary = summarize(rows, dry_run=dry_run)
    log_path = Path(args.log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0 if int(summary["statuses"].get("error", 0)) == 0 else 1


def _row(
    config: WarehouseConfig,
    source_library: str,
    source_symbol: str,
    target_uri: str,
    target_library: str,
    target_symbol: str,
    *,
    frame: pd.DataFrame,
    status: str,
    error: str | None = None,
    deleted_legacy: bool = False,
) -> MigrationRow:
    return MigrationRow(
        source_uri=config.arctic_uri,
        source_library=source_library,
        source_symbol=source_symbol,
        target_uri=target_uri,
        target_library=target_library,
        target_symbol=target_symbol,
        rows=int(len(frame)),
        min_date=_min_date_text(frame),
        max_date=_max_date_text(frame),
        status=status,
        error=error,
        deleted_legacy=bool(deleted_legacy),
    )


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_list_symbols(backend: ArcticBackend, library: str) -> list[str]:
    try:
        if library not in backend._arctic.list_libraries():
            return []
        return backend.list_symbols(library)
    except Exception:
        return []


def _is_equity_fundamental_library(library: str) -> bool:
    return str(library).startswith("fundamental_")


def _target_library_for_symbol(catalog: CatalogStore, source_library: str, symbol: str, *, provider: str) -> str:
    if source_library == PRICES_LIBRARY:
        pooled_type = _pooled_vehicle_type(catalog, symbol)
        if pooled_type == "etf":
            return provider_library(ETF_PRICES_LIBRARY, provider)
        if pooled_type == "fund":
            return provider_library(FUND_PRICES_LIBRARY, provider)
    return provider_library(source_library, provider)


def _should_skip_equity_route_symbol(catalog: CatalogStore, symbol: str) -> bool:
    return _pooled_vehicle_type(catalog, symbol) is not None


def _pooled_vehicle_type(catalog: CatalogStore, symbol: str) -> str | None:
    profiles = catalog.list_profiles(symbol)
    profiles.extend(catalog.list_etf_profiles(symbol))
    for profile in profiles:
        payload = {str(key).lower(): value for key, value in dict(profile.payload or {}).items()}
        if _truthy(payload.get("is_etf")) or _truthy(payload.get("isetf")):
            return "etf"
        if _truthy(payload.get("is_fund")) or _truthy(payload.get("isfund")):
            return "fund"
        quote_type = str(payload.get("quote_type") or payload.get("quotetype") or "").strip().lower()
        if quote_type == "etf":
            return "etf"
        if quote_type in {"mutualfund", "mutual_fund", "fund"}:
            return "fund"
        instrument_type = str(payload.get("type") or payload.get("instrument_type") or "").strip().lower()
        if instrument_type == "etf":
            return "etf"
        if instrument_type in {"mutualfund", "mutual_fund", "fund"}:
            return "fund"
        if payload.get("fund_family") not in (None, ""):
            return "fund"
    if _looks_like_mutual_fund_symbol(symbol):
        return "fund"
    return None


def _looks_like_mutual_fund_symbol(symbol: str) -> bool:
    text = str(symbol or "").strip().upper()
    return len(text) == 5 and text.endswith("X") and text.isalpha()


def _assert_verified_copy(source_frame: pd.DataFrame, copied_frame: pd.DataFrame | None) -> None:
    if copied_frame is None or copied_frame.empty:
        raise RuntimeError("copied frame is empty")
    if len(source_frame) != len(copied_frame):
        raise RuntimeError(f"row count mismatch: source={len(source_frame)} copied={len(copied_frame)}")
    if _min_date_text(source_frame) != _min_date_text(copied_frame):
        raise RuntimeError("min date mismatch")
    if _max_date_text(source_frame) != _max_date_text(copied_frame):
        raise RuntimeError("max date mismatch")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _min_date_text(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    return frame.index.min().strftime("%Y-%m-%d")


def _max_date_text(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    return frame.index.max().strftime("%Y-%m-%d")


if __name__ == "__main__":
    raise SystemExit(main())
