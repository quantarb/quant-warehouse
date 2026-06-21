from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from quant_warehouse.catalog.country import country_matches_filter, normalize_country_code
from quant_warehouse.catalog.listing_date import equity_historical_floor_text, listing_date_from_record


@dataclass(frozen=True)
class SectionState:
    symbol: str
    section: str
    provider: str
    min_date: str | None
    max_date: str | None
    row_count: int
    columns_present: tuple[str, ...]
    last_fetched_at: str | None


@dataclass(frozen=True)
class SymbolProfile:
    symbol: str
    provider: str
    source_provider: str
    fetched_at: str
    company_name: str | None
    exchange: str | None
    country: str | None
    sector: str | None
    industry: str | None
    market_cap: float | None
    beta: float | None
    cik: str | None
    ipo_date: str | None
    payload: dict[str, object]


class CatalogStore:
    def __init__(self, db_path: Path, *, storage_lock: threading.RLock | None = None) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_lock = storage_lock
        self._init_schema()

    def _storage_guard(self):
        return self._storage_lock or nullcontext()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS section_state (
                    symbol TEXT NOT NULL,
                    section TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    min_date TEXT,
                    max_date TEXT,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    columns_json TEXT NOT NULL DEFAULT '[]',
                    last_fetched_at TEXT,
                    PRIMARY KEY (symbol, section, provider)
                );
                CREATE TABLE IF NOT EXISTS symbol_profile (
                    symbol TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_provider TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    company_name TEXT,
                    exchange TEXT,
                    country TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    beta REAL,
                    cik TEXT,
                    ipo_date TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (symbol, provider)
                );
                CREATE TABLE IF NOT EXISTS etf_profile (
                    symbol TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_provider TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    company_name TEXT,
                    exchange TEXT,
                    country TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    beta REAL,
                    cik TEXT,
                    ipo_date TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (symbol, provider)
                );
                """
            )
            self._ensure_profile_columns(conn)

    def upsert(
        self,
        *,
        symbol: str,
        section: str,
        provider: str,
        min_date: str | None,
        max_date: str | None,
        row_count: int,
        columns_present: Iterable[str],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        columns = sorted({str(c) for c in columns_present if str(c)})
        with self._storage_guard():
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO section_state (
                        symbol, section, provider, min_date, max_date,
                        row_count, columns_json, last_fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, section, provider) DO UPDATE SET
                        min_date=excluded.min_date,
                        max_date=excluded.max_date,
                        row_count=excluded.row_count,
                        columns_json=excluded.columns_json,
                        last_fetched_at=excluded.last_fetched_at
                    """,
                    (
                        symbol.upper(),
                        section,
                        provider,
                        min_date,
                        max_date,
                        int(row_count),
                        json.dumps(columns),
                        now,
                    ),
                )

    def get(self, *, symbol: str, section: str, provider: str) -> SectionState | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM section_state
                WHERE symbol=? AND section=? AND provider=?
                """,
                (symbol.upper(), section, provider),
            ).fetchone()
        if row is None:
            return None
        return SectionState(
            symbol=row["symbol"],
            section=row["section"],
            provider=row["provider"],
            min_date=row["min_date"],
            max_date=row["max_date"],
            row_count=int(row["row_count"]),
            columns_present=tuple(json.loads(row["columns_json"] or "[]")),
            last_fetched_at=row["last_fetched_at"],
        )

    def upsert_profile(
        self,
        *,
        symbol: str,
        provider: str,
        source_provider: str,
        payload: dict[str, object],
    ) -> None:
        self._upsert_profile_row(
            table="symbol_profile",
            section="profile",
            symbol=symbol,
            provider=provider,
            source_provider=source_provider,
            payload=payload,
            name_keys=("name", "company_name", "legal_name"),
        )

    def upsert_etf_profile(
        self,
        *,
        symbol: str,
        provider: str,
        source_provider: str,
        payload: dict[str, object],
    ) -> None:
        self._upsert_profile_row(
            table="etf_profile",
            section="etf_profile",
            symbol=symbol,
            provider=provider,
            source_provider=source_provider,
            payload=payload,
            name_keys=("name", "fund_name", "company_name", "legal_name"),
        )

    def _upsert_profile_row(
        self,
        *,
        table: str,
        section: str,
        symbol: str,
        provider: str,
        source_provider: str,
        payload: dict[str, object],
        name_keys: tuple[str, ...],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        record = dict(payload or {})
        company_name = _first_text(record, *name_keys)
        exchange = _first_text(record, "stock_exchange", "exchange")
        country = normalize_country_code(_first_text(record, "hq_country", "inc_country", "country"))
        sector = _first_text(record, "sector", "category")
        industry = _first_text(record, "industry_category", "industry", "industry_group", "fund_family")
        market_cap = _first_float(record, "market_cap", "total_assets", "net_assets")
        beta = _first_float(record, "beta")
        cik = _first_text(record, "cik")
        ipo_date = listing_date_from_record(record)
        with self._storage_guard():
            with self._connect() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {table} (
                        symbol, provider, source_provider, fetched_at,
                        company_name, exchange, country, sector, industry,
                        market_cap, beta, cik, ipo_date, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, provider) DO UPDATE SET
                        source_provider=excluded.source_provider,
                        fetched_at=excluded.fetched_at,
                        company_name=excluded.company_name,
                        exchange=excluded.exchange,
                        country=excluded.country,
                        sector=excluded.sector,
                        industry=excluded.industry,
                        market_cap=excluded.market_cap,
                        beta=excluded.beta,
                        cik=excluded.cik,
                        ipo_date=excluded.ipo_date,
                        payload_json=excluded.payload_json
                    """,
                    (
                        symbol.upper(),
                        provider.strip().lower(),
                        source_provider.strip().lower(),
                        now,
                        company_name,
                        exchange,
                        country,
                        sector,
                        industry,
                        market_cap,
                        beta,
                        cik,
                        ipo_date,
                        json.dumps(record, default=str),
                    ),
                )
        self.upsert(
            symbol=symbol,
            section=section,
            provider=provider,
            min_date=None,
            max_date=None,
            row_count=1 if record else 0,
            columns_present=sorted(str(key) for key in record),
        )

    def get_profile(self, *, symbol: str, provider: str) -> SymbolProfile | None:
        return self._get_profile_from_table(
            table="symbol_profile",
            symbol=symbol,
            provider=provider,
        )

    def list_profiles(self, symbol: str) -> list[SymbolProfile]:
        return self._list_profiles_from_table(table="symbol_profile", symbol=symbol)

    def resolve_equity_ipo_date(self, symbol: str) -> str | None:
        """Return the best-known IPO/listing date across stored equity profiles."""
        symbol = symbol.strip().upper()
        for profile in self.list_profiles(symbol):
            if profile.ipo_date:
                return str(profile.ipo_date)[:10]
            ipo_date = listing_date_from_record(profile.payload)
            if ipo_date:
                return ipo_date
        return None

    def equity_historical_start(self, symbol: str) -> str:
        """Effective equity history floor: max(1900-01-01, ipo_date)."""
        return equity_historical_floor_text(ipo_date=self.resolve_equity_ipo_date(symbol))

    def query_symbol_profiles(
        self,
        *,
        provider: str | None = None,
        min_market_cap: float | None = None,
        max_market_cap: float | None = None,
        country: str | None = None,
        exchanges: Sequence[str] | None = None,
        exclude_etf: bool = False,
        exclude_fund: bool = False,
        limit: int | None = None,
    ) -> list[SymbolProfile]:
        clauses: list[str] = []
        params: list[object] = []
        if provider is not None:
            clauses.append("provider=?")
            params.append(str(provider).strip().lower())
        if min_market_cap is not None:
            clauses.append("market_cap >= ?")
            params.append(float(min_market_cap))
        if max_market_cap is not None:
            clauses.append("market_cap <= ?")
            params.append(float(max_market_cap))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM symbol_profile {where_sql} ORDER BY market_cap DESC, symbol"
        query_params = list(params)
        if limit is not None and int(limit) > 0:
            query += " LIMIT ?"
            query_params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, query_params).fetchall()
        profiles = [self._profile_from_row(row) for row in rows]
        if country:
            profiles = [
                profile
                for profile in profiles
                if country_matches_filter(profile.country, filter_country=country)
            ]
        exchange_filters = tuple(str(value).strip().upper() for value in (exchanges or ()) if str(value).strip())
        if exchange_filters:
            from quant_warehouse.ingest.screener_fetch import exchange_matches_filters

            profiles = [
                profile
                for profile in profiles
                if exchange_matches_filters(profile.exchange, exchange_filters)
            ]
        if exclude_etf or exclude_fund:
            filtered: list[SymbolProfile] = []
            for profile in profiles:
                payload = dict(profile.payload or {})
                is_etf = bool(payload.get("is_etf", payload.get("isEtf")))
                is_fund = bool(payload.get("is_fund", payload.get("isFund")))
                if exclude_etf and is_etf:
                    continue
                if exclude_fund and is_fund:
                    continue
                filtered.append(profile)
            profiles = filtered
        return profiles

    def get_etf_profile(self, *, symbol: str, provider: str) -> SymbolProfile | None:
        return self._get_profile_from_table(
            table="etf_profile",
            symbol=symbol,
            provider=provider,
        )

    def list_etf_profiles(self, symbol: str) -> list[SymbolProfile]:
        return self._list_profiles_from_table(table="etf_profile", symbol=symbol)

    def delete_section(self, *, symbol: str, section: str, provider: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM section_state WHERE symbol=? AND section=? AND provider=?",
                (symbol.upper(), section, provider.strip().lower()),
            )

    def delete_profile(self, *, symbol: str, provider: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM symbol_profile WHERE symbol=? AND provider=?",
                (symbol.upper(), provider.strip().lower()),
            )
        self.delete_section(symbol=symbol, section="profile", provider=provider)

    def delete_etf_profile(self, *, symbol: str, provider: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM etf_profile WHERE symbol=? AND provider=?",
                (symbol.upper(), provider.strip().lower()),
            )
        self.delete_section(symbol=symbol, section="etf_profile", provider=provider)

    def _get_profile_from_table(
        self,
        *,
        table: str,
        symbol: str,
        provider: str,
    ) -> SymbolProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE symbol=? AND provider=?",
                (symbol.upper(), provider.strip().lower()),
            ).fetchone()
        if row is None:
            return None
        return self._profile_from_row(row)

    def _list_profiles_from_table(self, *, table: str, symbol: str) -> list[SymbolProfile]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE symbol=? ORDER BY provider",
                (symbol.upper(),),
            ).fetchall()
        return [self._profile_from_row(row) for row in rows]

    @staticmethod
    def _profile_from_row(row: sqlite3.Row) -> SymbolProfile:
        payload = json.loads(row["payload_json"] or "{}")
        if not isinstance(payload, dict):
            payload = {}
        ipo_date = row["ipo_date"] if "ipo_date" in row.keys() else None
        if not ipo_date:
            ipo_date = listing_date_from_record(payload)
        return SymbolProfile(
            symbol=row["symbol"],
            provider=row["provider"],
            source_provider=row["source_provider"],
            fetched_at=row["fetched_at"],
            company_name=row["company_name"],
            exchange=row["exchange"],
            country=row["country"],
            sector=row["sector"],
            industry=row["industry"],
            market_cap=row["market_cap"],
            beta=row["beta"],
            cik=row["cik"],
            ipo_date=ipo_date,
            payload=payload,
        )

    @staticmethod
    def _ensure_profile_columns(conn: sqlite3.Connection) -> None:
        for table in ("symbol_profile", "etf_profile"):
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN ipo_date TEXT")
            except sqlite3.OperationalError:
                pass

    def list_section(self, section: str, *, provider: str | None = None) -> list[SectionState]:
        clauses = ["section=?"]
        params: list[object] = [str(section).strip()]
        if provider is not None:
            clauses.append("provider=?")
            params.append(str(provider).strip().lower())
        query = f"SELECT * FROM section_state WHERE {' AND '.join(clauses)} ORDER BY symbol, provider"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            SectionState(
                symbol=row["symbol"],
                section=row["section"],
                provider=row["provider"],
                min_date=row["min_date"],
                max_date=row["max_date"],
                row_count=int(row["row_count"]),
                columns_present=tuple(json.loads(row["columns_json"] or "[]")),
                last_fetched_at=row["last_fetched_at"],
            )
            for row in rows
        ]

    def list_symbol(self, symbol: str) -> list[SectionState]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM section_state WHERE symbol=? ORDER BY section, provider",
                (symbol.upper(),),
            ).fetchall()
        return [
            SectionState(
                symbol=r["symbol"],
                section=r["section"],
                provider=r["provider"],
                min_date=r["min_date"],
                max_date=r["max_date"],
                row_count=int(r["row_count"]),
                columns_present=tuple(json.loads(r["columns_json"] or "[]")),
                last_fetched_at=r["last_fetched_at"],
            )
            for r in rows
        ]


def _first_text(record: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return None


def _first_float(record: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed:
            return parsed
    return None