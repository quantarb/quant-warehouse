from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from quant_warehouse.ingest.django_symbols import list_django_symbols
from quant_warehouse.ingest.normalize import normalize_prices

DJANGO_PRICE_SECTION = "prices_div_adj"


def iter_django_price_payloads(db_path: Path, symbol: str) -> Iterator[dict]:
    db_path = Path(db_path).expanduser().resolve()
    symbol = symbol.strip().upper()
    query = """
        SELECT h.record_date, h.payload
        FROM fmp_symbolsectionhistorical h
        JOIN fmp_symbol s ON s.id = h.symbol_id
        WHERE h.section_key = ? AND s.symbol = ?
        ORDER BY h.record_date
    """
    with sqlite3.connect(db_path) as conn:
        for record_date, payload in conn.execute(query, (DJANGO_PRICE_SECTION, symbol)):
            record = _parse_payload(payload)
            if not record:
                continue
            if "date" not in record and record_date:
                record["date"] = str(record_date)[:10]
            yield record


def payloads_to_price_frame(payloads: Iterable[dict]) -> pd.DataFrame:
    rows = [dict(item) for item in payloads if isinstance(item, dict)]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def django_fmp_prices_frame(db_path: Path, symbol: str) -> pd.DataFrame:
    raw = payloads_to_price_frame(iter_django_price_payloads(db_path, symbol))
    if raw.empty:
        return raw
    return normalize_prices(raw, provider="fmp")


def _parse_payload(payload: object) -> dict | None:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str) and payload.strip():
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None