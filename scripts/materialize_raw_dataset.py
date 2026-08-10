#!/usr/bin/env python3
"""Materialize raw warehouse sections into partitioned Arrow/Parquet data."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import polars as pl
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


def _read_section(warehouse: Warehouse, symbol: str, state, start: datetime, end: datetime) -> pl.DataFrame | None:
    frame = None
    for library in _library_candidates(state.section, state.provider):
        frame = warehouse.backend.read(
            library, symbol_provider_key(symbol, state.provider),
            date_range=(start, end),
            columns=[str(column) for column in state.columns_present] or None,
        )
        if frame is not None and not frame.is_empty():
            break
    if frame is None or frame.is_empty():
        return None
    frame = frame.clone()
    if "date" not in frame.columns:
        raise ValueError(f"Stored section {state.section} has no date column")
    frame = frame.with_columns(
        pl.col("date").cast(pl.Datetime, strict=False).dt.replace_time_zone(None).dt.truncate("1d")
    ).filter(pl.col("date").is_between(start, end, closed="both"))
    if frame.is_empty():
        return None
    return frame.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(str(state.section)).alias("section"),
        pl.lit(str(state.provider)).alias("provider"),
    )


def materialize(symbols: list[str], output: Path, start: datetime, end: datetime, skip: set[str]) -> dict[str, object]:
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
            for year_frame in frame.with_columns(pl.col("date").dt.year().alias("_year")).partition_by("_year", as_dict=False, maintain_order=True):
                year = year_frame["_year"][0]
                destination = output / f"symbol={_safe_name(symbol)}" / f"year={int(year)}" / f"section={_safe_name(state.section)}.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(year_frame.drop("_year").to_arrow(), destination, compression="zstd", use_dictionary=True)
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
    print(json.dumps(materialize(symbols, args.output, datetime.fromisoformat(args.start), datetime.fromisoformat(args.end), skip), indent=2))


if __name__ == "__main__":
    main()
