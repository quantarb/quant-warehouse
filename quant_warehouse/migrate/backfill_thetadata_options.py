from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.target_engineering.thetadata_loader import (
    ThetaDataDownloadSpec,
    download_option_snapshots_for_range,
    option_chain_range_cached,
)
from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.warehouse.prices import list_arctic_price_underlyings

ProgressLogger = Callable[[str], None] | None
SymbolSource = Literal["arctic-fmp", "catalog"]
OPTION_SECTION = "options_eod"
OPTION_PROVIDER = "thetadata"
_NON_US_SUFFIXES = (".SZ", ".SS", ".HK", ".TO", ".L", ".PA", ".DE", ".AX", ".KS", ".TW", ".T")


def _is_us_option_symbol(symbol: str) -> bool:
    text = str(symbol).strip().upper()
    if not text:
        return False
    return not any(text.endswith(suffix) for suffix in _NON_US_SUFFIXES)


def _filter_us_symbols(symbols: Sequence[str]) -> list[str]:
    return [symbol for symbol in symbols if _is_us_option_symbol(symbol)]


def list_catalog_price_symbols(
    warehouse: Warehouse,
    *,
    providers: Sequence[str] = ("fmp",),
) -> list[str]:
    """Return warehouse catalog symbols that already have stored equity price history."""

    symbols: set[str] = set()
    for provider in providers:
        for state in warehouse.catalog.list_section("prices", provider=str(provider).strip().lower()):
            if int(state.row_count) > 0 and str(state.symbol).strip():
                symbols.add(str(state.symbol).strip().upper())
    return sorted(symbols)


def list_arctic_fmp_underlyings(warehouse: Warehouse) -> list[str]:
    """Return FMP underlyings stored in the Arctic prices library."""

    return list_arctic_price_underlyings(warehouse.prices.backend, provider="fmp")


def resolve_backfill_symbols(
    warehouse: Warehouse,
    *,
    source: SymbolSource = "arctic-fmp",
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    us_only: bool = True,
) -> list[str]:
    """Resolve the symbol batch from explicit args or warehouse storage."""

    if symbols:
        resolved = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    elif source == "catalog":
        resolved = list_catalog_price_symbols(warehouse, providers=("fmp",))
    else:
        resolved = list_arctic_fmp_underlyings(warehouse)

    if offset and not symbols:
        resolved = resolved[offset:]
    if limit is not None and not symbols:
        resolved = resolved[: max(0, int(limit))]

    if us_only:
        resolved = _filter_us_symbols(resolved)
    return resolved


def _business_days(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return [ts.normalize() for ts in pd.date_range(start, end, freq="B")]


def _options_range_cached(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> bool:
    return option_chain_range_cached(symbol, start_date, end_date)


def _upsert_options_catalog_state(
    warehouse: Warehouse,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    snapshot_days: int,
    contracts_total: int,
) -> None:
    warehouse.catalog.upsert(
        symbol=symbol,
        section=OPTION_SECTION,
        provider=OPTION_PROVIDER,
        min_date=start_date,
        max_date=end_date,
        row_count=int(contracts_total),
        columns_present=("bid", "ask", "mid", "snapshot_date", "contract_symbol", "data_interval"),
    )


def backfill_thetadata_options(
    *,
    warehouse: Warehouse | None = None,
    config: WarehouseConfig | None = None,
    symbols: Sequence[str] | None = None,
    source: SymbolSource = "arctic-fmp",
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    max_dte: int = 60,
    strike_range: int = 10,
    limit: int | None = None,
    offset: int = 0,
    skip_existing: bool = True,
    overwrite: bool = False,
    request_sleep: float = 1.0,
    us_only: bool = True,
    progress_logger: ProgressLogger = None,
) -> dict[str, object]:
    """Download daily ThetaData EOD option chains for FMP underlyings in Arctic."""

    warehouse = warehouse or Warehouse(config=config)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date or datetime.now(timezone.utc).date()).normalize()
    if end < start:
        raise ValueError(f"end_date {end.date()} must be on or after start_date {start.date()}")

    target_symbols = resolve_backfill_symbols(
        warehouse,
        source=source,
        symbols=symbols,
        limit=limit,
        offset=offset,
        us_only=us_only,
    )
    download_spec = ThetaDataDownloadSpec(max_dte=max_dte, strike_range=strike_range)
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, object]] = []
    total = len(target_symbols)

    for index, symbol in enumerate(target_symbols, start=1):
        row: dict[str, object] = {"symbol": symbol}
        try:
            if skip_existing and not overwrite and _options_range_cached(symbol, start, end):
                row.update({"skipped": True, "reason": "cached_range"})
                results.append(row)
                if callable(progress_logger):
                    progress_logger(f"[thetadata-options] {index}/{total} skipped cached {symbol}")
                continue

            manifest = download_option_snapshots_for_range(
                symbol,
                start,
                end,
                spec=download_spec,
                overwrite=overwrite,
            )
            _upsert_options_catalog_state(
                warehouse,
                symbol=symbol,
                start_date=str(manifest["start_date"]),
                end_date=str(manifest["end_date"]),
                snapshot_days=int(manifest.get("snapshot_days") or 0),
                contracts_total=int(manifest.get("contracts_total") or 0),
            )
            row.update({"skipped": False, **manifest})
            results.append(row)
            if callable(progress_logger):
                progress_logger(
                    f"[thetadata-options] {index}/{total} {symbol} "
                    f"days={manifest.get('snapshot_days')} contracts={manifest.get('contracts_total')}"
                )
        except Exception as exc:
            row.update({"skipped": False, "error": str(exc)})
            results.append(row)
            if callable(progress_logger):
                progress_logger(f"[thetadata-options] {index}/{total} error {symbol}: {exc}")

        if request_sleep > 0 and index < total:
            time.sleep(float(request_sleep))

    completed = [row for row in results if not row.get("error")]
    skipped = [row for row in results if row.get("skipped")]
    failed = [row for row in results if row.get("error")]
    return {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "symbols_requested": total,
        "symbols_completed": len(completed),
        "symbols_skipped": len(skipped),
        "symbols_failed": len(failed),
        "us_only": us_only,
        "download_spec": {
            "data_interval": download_spec.data_interval,
            "max_dte": download_spec.max_dte,
            "strike_range": download_spec.strike_range,
            "require_bid_ask": download_spec.require_bid_ask,
        },
        "storage_backend": "arctic",
        "results": results,
    }


def write_backfill_log(summary: dict[str, object], *, log_path: str | Path) -> Path:
    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
