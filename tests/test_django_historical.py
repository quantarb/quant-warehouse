import json
import sqlite3
from pathlib import Path

import pandas as pd

from quant_warehouse.ingest.django_historical import (
    django_historical_frame,
    warehouse_section_for_django,
)
from quant_warehouse.migrate.django_historical import migrate_django_historical


def _seed_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE fmp_symbol (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                payload TEXT
            );
            CREATE TABLE fmp_symbolsectionhistorical (
                id INTEGER PRIMARY KEY,
                symbol_id INTEGER NOT NULL,
                section_key TEXT NOT NULL,
                record_key TEXT NOT NULL,
                record_date TEXT,
                payload TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO fmp_symbol (id, symbol, payload) VALUES (1, 'AAPL', ?)",
            (json.dumps({"isEtf": False}),),
        )
        rows = [
            (
                "income_statement",
                "k1",
                "2024-09-28",
                {
                    "date": "2024-09-28",
                    "symbol": "AAPL",
                    "revenue": 100.0,
                    "grossProfit": 40.0,
                    "fiscalYear": "2024",
                    "period": "Q4",
                },
            ),
            (
                "income_statement",
                "k2",
                "2023-09-30",
                {
                    "date": "2023-09-30",
                    "symbol": "AAPL",
                    "revenue": 90.0,
                    "grossProfit": 35.0,
                    "fiscalYear": "2023",
                    "period": "Q4",
                },
            ),
        ]
        for section_key, record_key, record_date, payload in rows:
            conn.execute(
                """
                INSERT INTO fmp_symbolsectionhistorical
                (symbol_id, section_key, record_key, record_date, payload)
                VALUES (1, ?, ?, ?, ?)
                """,
                (section_key, record_key, record_date, json.dumps(payload)),
            )


def test_django_historical_frame_normalizes_fmp_payload(tmp_path: Path):
    db_path = tmp_path / "db.sqlite3"
    _seed_db(db_path)
    frame = django_historical_frame(db_path, "AAPL", "income_statement")
    assert len(frame) == 2
    assert "revenue" in frame.columns
    assert "gross_profit" in frame.columns
    assert frame.loc[pd.Timestamp("2024-09-28"), "revenue"] == 100.0


def test_ingest_frame_preserves_datetime_index_for_metrics(tmp_path: Path, monkeypatch):
    qw_home = tmp_path / "qw"
    monkeypatch.setenv("QW_HOME", str(qw_home))
    monkeypatch.setenv("QW_ARCTIC_URI", f"lmdb://{qw_home / 'arctic'}")
    monkeypatch.setenv("QW_CATALOG_PATH", str(qw_home / "catalog.sqlite"))

    from quant_warehouse.warehouse.api import Warehouse

    wh = Warehouse()
    frame = pd.DataFrame(
        {"pe_ratio": [25.0, 26.0]},
        index=pd.to_datetime(["2023-12-31", "2024-12-31"]),
    )
    frame.index.name = "period_ending"
    result = wh.fundamentals.ingest_frame("AAPL", section="metrics", provider="fmp", frame=frame, merge=False)
    assert result["rows"] == 2
    out = wh.read_fundamentals("AAPL", section="metrics", provider="fmp")
    assert len(out) == 2


def test_migrate_django_historical_to_arctic(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "db.sqlite3"
    _seed_db(db_path)
    qw_home = tmp_path / "qw"
    monkeypatch.setenv("QW_HOME", str(qw_home))
    monkeypatch.setenv("QW_ARCTIC_URI", f"lmdb://{qw_home / 'arctic'}")
    monkeypatch.setenv("QW_CATALOG_PATH", str(qw_home / "catalog.sqlite"))

    stats = migrate_django_historical(db_path, symbols=["AAPL"], section_keys=["income_statement"])
    assert stats[0]["rows"] == 2
    assert warehouse_section_for_django("income_statement") == "income"

    from quant_warehouse.warehouse.api import Warehouse

    wh = Warehouse()
    out = wh.read_fundamentals("AAPL", section="income", provider="fmp")
    assert len(out) == 2
    assert out.loc[pd.Timestamp("2024-09-28"), "revenue"] == 100.0