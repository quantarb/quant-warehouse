from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from quant_warehouse.ingest.django_prices import _parse_payload
from quant_warehouse.ingest.normalize import normalize_vendor_frame
from quant_warehouse.warehouse.sections import DJANGO_HISTORICAL_SECTION_KEYS, DJANGO_HISTORICAL_SECTION_MAP


def django_section_keys() -> tuple[str, ...]:
    return DJANGO_HISTORICAL_SECTION_KEYS


def warehouse_section_for_django(section_key: str) -> str:
    key = str(section_key).strip()
    mapped = DJANGO_HISTORICAL_SECTION_MAP.get(key)
    if mapped is None:
        raise ValueError(f"Unknown django historical section_key: {section_key}")
    return mapped


def iter_django_historical_payloads(
    db_path: Path | str,
    symbol: str,
    section_key: str,
) -> Iterator[dict]:
    db_path = Path(db_path).expanduser().resolve()
    symbol = symbol.strip().upper()
    section_key = str(section_key).strip()
    query = """
        SELECT h.record_date, h.payload
        FROM fmp_symbolsectionhistorical h
        JOIN fmp_symbol s ON s.id = h.symbol_id
        WHERE h.section_key = ? AND s.symbol = ?
        ORDER BY h.record_date
    """
    with sqlite3.connect(db_path) as conn:
        for record_date, payload in conn.execute(query, (section_key, symbol)):
            record = _parse_payload(payload)
            if not record:
                continue
            if "date" not in record and record_date:
                record["date"] = str(record_date)[:10]
            yield record


def payloads_to_historical_frame(payloads: Iterable[dict]) -> pd.DataFrame:
    rows = [dict(item) for item in payloads if isinstance(item, dict)]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def django_historical_frame(db_path: Path | str, symbol: str, section_key: str) -> pd.DataFrame:
    raw = payloads_to_historical_frame(
        iter_django_historical_payloads(db_path, symbol, section_key),
    )
    if raw.empty:
        return raw
    return normalize_vendor_frame(raw, provider="fmp", vendor_only_prefix=None)


def list_django_symbols_for_section(
    db_path: Path | str,
    section_key: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[str]:
    db_path = Path(db_path).expanduser().resolve()
    query = """
        SELECT DISTINCT s.symbol
        FROM fmp_symbolsectionhistorical h
        JOIN fmp_symbol s ON s.id = h.symbol_id
        WHERE h.section_key = ?
        ORDER BY s.symbol
    """
    params: list[object] = [str(section_key).strip()]
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [str(row[0]).strip().upper() for row in rows if str(row[0]).strip()]