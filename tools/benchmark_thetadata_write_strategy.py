from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from quant_warehouse.config import WarehouseConfig
from quant_warehouse.platforms.data_providers.thetadata.options import (
    OPTIONS_THETADATA_EOD_LIBRARY,
    option_chain_storage_symbol,
    read_option_chain_arctic,
    split_snapshots_by_date,
    write_option_chain_arctic,
)
from quant_warehouse.warehouse.backend import open_backend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark per-day vs per-symbol ArcticDB writes using real cached ThetaData rows.",
    )
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--max-days", type=int, default=120)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def _temp_config(root: Path) -> WarehouseConfig:
    return WarehouseConfig(
        home=root,
        arctic_uri=f"lmdb://{root / 'arctic'}",
        catalog_path=root / "catalog.sqlite",
    )


def _read_source(symbol: str, max_days: int) -> pd.DataFrame:
    source = read_option_chain_arctic(symbol)
    if source.empty:
        raise SystemExit(f"No cached option rows found for {symbol}. Download one symbol first.")
    snapshot_dates = pd.to_datetime(source["snapshot_date"], errors="coerce").dt.normalize()
    selected_days = sorted(snapshot_dates.dropna().unique())[: max(1, int(max_days))]
    return source.loc[snapshot_dates.isin(selected_days)].copy()


def _bench_per_day(symbol: str, frame: pd.DataFrame, root: Path) -> tuple[float, int, int]:
    backend = open_backend(_temp_config(root))
    snapshots = split_snapshots_by_date(frame)
    start = time.perf_counter()
    writes = 0
    for _ts, snapshot in sorted(snapshots.items()):
        write_option_chain_arctic(symbol, snapshot, backend=backend, merge=True)
        writes += 1
    elapsed = time.perf_counter() - start
    stored = backend.read(OPTIONS_THETADATA_EOD_LIBRARY, option_chain_storage_symbol(symbol))
    return elapsed, 0 if stored is None else len(stored), writes


def _bench_per_symbol(symbol: str, frame: pd.DataFrame, root: Path) -> tuple[float, int]:
    backend = open_backend(_temp_config(root))
    start = time.perf_counter()
    write_option_chain_arctic(symbol, frame, backend=backend, merge=True)
    elapsed = time.perf_counter() - start
    stored = backend.read(OPTIONS_THETADATA_EOD_LIBRARY, option_chain_storage_symbol(symbol))
    return elapsed, 0 if stored is None else len(stored)


def main() -> int:
    args = _parse_args()
    symbol = str(args.symbol).strip().upper()
    frame = _read_source(symbol, args.max_days)
    days = int(pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize().nunique())
    rows = int(len(frame))

    temp_root = Path(tempfile.mkdtemp(prefix=f"qw_thetadata_write_bench_{symbol}_"))
    try:
        per_day_root = temp_root / "per_day"
        per_symbol_root = temp_root / "per_symbol"
        per_day_seconds, per_day_rows, per_day_writes = _bench_per_day(symbol, frame, per_day_root)
        per_symbol_seconds, per_symbol_rows = _bench_per_symbol(symbol, frame, per_symbol_root)
        result = {
            "symbol": symbol,
            "days": days,
            "rows": rows,
            "per_day_seconds": per_day_seconds,
            "per_day_rows": per_day_rows,
            "per_day_writes": per_day_writes,
            "per_symbol_seconds": per_symbol_seconds,
            "per_symbol_rows": per_symbol_rows,
            "per_symbol_writes": 1,
            "speedup": per_day_seconds / per_symbol_seconds if per_symbol_seconds else None,
            "temp_root": str(temp_root) if args.keep_temp else None,
        }
        print(json.dumps(result, indent=2))
        return 0
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
