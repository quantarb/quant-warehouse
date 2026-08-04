#!/usr/bin/env python3
"""Materialize raw warehouse sections into partitioned Arrow/Parquet data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_warehouse.warehouse.api import Warehouse
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.warehouse.storage import provider_library

DEFAULT_SKIP = {"profile", "news", "options_eod", "filings", "company_news"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_") or "unknown"


def _library_candidates(section: str, provider: str) -> list[str]:
    suffixes = ("_quarter", "_annual", "_ttm")
    base, period = str(section), None
    for suffix in suffixes:
        if base.endswith(suffix):
            base, period = base[: -len(suffix)], suffix[1:]
            break
    fundamental = f"fundamental_{base}" + (f"_{period}" if period else "")
    return list(dict.fromkeys([provider_library(fundamental, provider), provider_library(section, provider)]))


def _read_section(warehouse: Warehouse, symbol: str, state, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    frame = None
    for library in _library_candidates(state.section, state.provider):
        frame = warehouse.backend.read(
            library, symbol_provider_key(symbol, state.provider),
            date_range=(start, end),
            columns=[str(column) for column in state.columns_present] or None,
        )
        if frame is not None and not frame.empty:
            break
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        dates = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
        frame["date"] = dates.normalize()
    elif "date" not in frame.columns:
        frame = frame.reset_index(names="date")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["date"].between(start, end)].copy()
    if frame.empty:
        return None
    frame["symbol"] = symbol
    frame["section"] = str(state.section)
    frame["provider"] = str(state.provider)
    return frame.reset_index(drop=True)


def materialize(symbols: list[str], output: Path, start: pd.Timestamp, end: pd.Timestamp, skip: set[str]) -> dict[str, object]:
    warehouse = Warehouse()
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    sections_seen: set[str] = set()
    for symbol in symbols:
        for state in warehouse.catalog.list_symbol(symbol):
            if state.section in skip or state.provider != "fmp" or state.row_count <= 0:
                continue
            frame = _read_section(warehouse, symbol, state, start, end)
            if frame is None:
                continue
            sections_seen.add(state.section)
            for year, year_frame in frame.groupby(frame["date"].dt.year, sort=True):
                destination = output / f"symbol={_safe_name(symbol)}" / f"year={int(year)}" / f"section={_safe_name(state.section)}.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(year_frame, preserve_index=False), destination, compression="zstd", use_dictionary=True)
                written.append(str(destination))
    manifest = {"format": "parquet", "partitioning": ["symbol", "year", "section"], "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"), "symbols": symbols, "sections": sorted(sections_seen), "files": written}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"symbols": len(symbols), "sections": len(sections_seen), "files": len(written), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--start", default="1962-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--output", type=Path, default=Path("artifacts/raw-arrow"))
    parser.add_argument("--include", default="", help="Comma-separated sections to force-include")
    args = parser.parse_args()
    symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()]
    skip = DEFAULT_SKIP.difference({value.strip() for value in args.include.split(",") if value.strip()})
    print(json.dumps(materialize(symbols, args.output, pd.Timestamp(args.start), pd.Timestamp(args.end), skip), indent=2))


if __name__ == "__main__":
    main()
