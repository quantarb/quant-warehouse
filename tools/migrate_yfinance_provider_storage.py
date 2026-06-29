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
from quant_warehouse.warehouse.sections import ALL_FUNDAMENTAL_SECTIONS, ETF_PRICES_LIBRARY
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


def migrate_yfinance_storage(
    *,
    libraries: Sequence[str] = LEGACY_LIBRARIES,
    dry_run: bool = True,
    limit: int | None = None,
    skip_catalog_funds: bool = True,
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

            target_library = provider_library(source_library, PROVIDER)
            target_symbol = symbol_provider_key(symbol, PROVIDER)
            try:
                if skip_catalog_funds and _is_equity_route_library(source_library) and _should_skip_equity_route_symbol(
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
        help="Also migrate symbols cataloged as ETF/fund into equity-route libraries. Default skips them.",
    )
    parser.add_argument("--apply", action="store_true", help="Write copied frames. Default is dry-run only.")
    parser.add_argument("--log", default="~/.quant-warehouse/logs/migrate-yfinance-provider-storage.json")
    args = parser.parse_args(argv)

    libraries = tuple(_parse_csv(args.libraries)) if args.libraries else LEGACY_LIBRARIES
    dry_run = not bool(args.apply)
    rows = migrate_yfinance_storage(
        libraries=libraries,
        dry_run=dry_run,
        limit=args.limit,
        skip_catalog_funds=not bool(args.include_funds),
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


def _is_equity_route_library(library: str) -> bool:
    return library == PRICES_LIBRARY or str(library).startswith("fundamental_")


def _should_skip_equity_route_symbol(catalog: CatalogStore, symbol: str) -> bool:
    return _catalog_marks_fund(catalog, symbol) or _looks_like_mutual_fund_symbol(symbol)


def _catalog_marks_fund(catalog: CatalogStore, symbol: str) -> bool:
    profiles = catalog.list_profiles(symbol)
    profiles.extend(catalog.list_etf_profiles(symbol))
    for profile in profiles:
        payload = {str(key).lower(): value for key, value in dict(profile.payload or {}).items()}
        if _truthy(payload.get("is_etf")) or _truthy(payload.get("isetf")):
            return True
        if _truthy(payload.get("is_fund")) or _truthy(payload.get("isfund")):
            return True
        quote_type = str(payload.get("quote_type") or payload.get("quotetype") or "").strip().lower()
        if quote_type in {"etf", "mutualfund", "mutual_fund", "fund"}:
            return True
        instrument_type = str(payload.get("type") or payload.get("instrument_type") or "").strip().lower()
        if instrument_type in {"etf", "mutualfund", "mutual_fund", "fund"}:
            return True
        if payload.get("fund_family") not in (None, ""):
            return True
    return False


def _looks_like_mutual_fund_symbol(symbol: str) -> bool:
    text = str(symbol or "").strip().upper()
    return len(text) == 5 and text.endswith("X") and text.isalpha()


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
