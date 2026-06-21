from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

AssetClass = Literal["equity", "etf", "all"]

_DJANGO_ETF_PREDICATE = """
    COALESCE(json_extract(s.payload, '$.isEtf'), json_extract(s.payload, '$.isETF'), 0)
    IN (1, '1', 'true', 'True')
"""


def list_django_symbols(
    db_path: Path | str,
    *,
    asset_class: AssetClass = "equity",
    require_prices: bool = True,
    limit: int | None = None,
    offset: int = 0,
) -> list[str]:
    db_path = Path(db_path).expanduser().resolve()
    clauses: list[str] = []
    params: list[object] = []

    if require_prices:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM fmp_symbolsectionhistorical h
                WHERE h.symbol_id = s.id AND h.section_key = 'prices_div_adj'
            )
            """
        )

    if asset_class == "equity":
        clauses.append(f"NOT ({_DJANGO_ETF_PREDICATE})")
    elif asset_class == "etf":
        clauses.append(f"({_DJANGO_ETF_PREDICATE})")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT DISTINCT s.symbol FROM fmp_symbol s {where} ORDER BY s.symbol"
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [str(row[0]).strip().upper() for row in rows if str(row[0]).strip()]


def django_etf_symbol_set(db_path: Path | str) -> set[str]:
    return set(list_django_symbols(db_path, asset_class="etf", require_prices=False))


def django_is_etf(db_path: Path | str, symbol: str) -> bool:
    db_path = Path(db_path).expanduser().resolve()
    symbol = symbol.strip().upper()
    query = f"""
        SELECT 1
        FROM fmp_symbol s
        WHERE s.symbol = ? AND ({_DJANGO_ETF_PREDICATE})
        LIMIT 1
    """
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(query, (symbol,)).fetchone()
    return row is not None