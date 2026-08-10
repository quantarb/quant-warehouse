from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.ingest.openbb_fetch import fetch_dataframe
from quant_warehouse.ingest.oracle_news import _normalize_news
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.sections import COMPANY_NEWS_LIBRARY, COMPANY_NEWS_SECTION
from quant_warehouse.warehouse.storage import provider_library, read_provider_frame


def _date_expr(frame: pl.DataFrame, name: str) -> pl.Expr:
    return ((pl.col(name).str.to_datetime(strict=False, time_zone="UTC") if frame.schema[name] == pl.String
             else pl.col(name).cast(pl.Datetime, strict=False)).dt.replace_time_zone(None).dt.truncate("1d"))


class CompanyNewsStore:
    """Provider-scoped, point-in-time company news stored per equity symbol."""

    def __init__(self, config: WarehouseConfig | None = None, *, backend: StorageBackend | None = None,
                 catalog: CatalogStore | None = None) -> None:
        self.config = config or WarehouseConfig.from_env()
        self.config.ensure_dirs()
        self.backend: ArcticBackend = backend or open_backend(self.config)
        self.catalog = catalog or CatalogStore(self.config.catalog_path)

    def import_frame(self, frame: pl.DataFrame, *, provider: str = "fmp") -> dict[str, int]:
        provider_name = str(provider or "fmp").strip().lower()
        incoming = self._prepare(frame, provider=provider_name)
        if incoming.is_empty():
            return {}
        library = provider_library(COMPANY_NEWS_LIBRARY, provider_name)
        counts: dict[str, int] = {}
        for symbol, symbol_frame in incoming.partition_by("symbol", as_dict=True).items():
            symbol = str(symbol[0])
            storage_symbol = symbol_provider_key(symbol, provider_name)
            existing = read_provider_frame(self.backend, base_library=COMPANY_NEWS_LIBRARY,
                                           provider=provider_name, symbol=storage_symbol, output_format="polars")
            merged = self._merge(existing, symbol_frame)
            self.backend.write(library, storage_symbol, merged)
            counts[symbol] = merged.height
            self.catalog.upsert(symbol=symbol, section=COMPANY_NEWS_SECTION, provider=provider_name,
                                min_date=str(merged.get_column("published_at").min())[:10],
                                max_date=str(merged.get_column("published_at").max())[:10], row_count=merged.height,
                                columns_present=merged.columns)
        return counts

    def import_parquet(self, path: str, *, provider: str = "fmp") -> dict[str, int]:
        return self.import_frame(pl.read_parquet(path), provider=provider)

    def ensure_date(self, symbol: str, observation_date: str | datetime, *, provider: str = "fmp",
                    page_limit: int = 250, max_pages: int = 101) -> pl.DataFrame:
        ticker = str(symbol).strip().upper()
        day = datetime.fromisoformat(str(observation_date)[:10])
        existing = self.read(ticker, provider=provider, observation_dates=[day])
        if not existing.is_empty() or str(provider).strip().lower() != "fmp":
            return existing
        frames: list[pl.DataFrame] = []
        day_text = day.strftime("%Y-%m-%d")
        for page in range(max_pages):
            frame = fetch_dataframe("company_news", symbol=ticker, provider="fmp", start_date=day_text,
                                    end_date=day_text, page=page, limit=page_limit)
            if frame is None or frame.is_empty():
                break
            frames.append(frame)
            if frame.height < page_limit:
                break
        if not frames:
            return pl.DataFrame()
        boundaries = pl.DataFrame({"symbol": [ticker], "observation_date": [day], "boundary_kind": ["agent_scoring"]})
        normalized = _normalize_news(pl.concat(frames, how="diagonal_relaxed"), boundaries)
        if normalized.is_empty():
            return pl.DataFrame()
        self.import_frame(normalized, provider="fmp")
        return self.read(ticker, provider="fmp", observation_dates=[day])

    def read(self, symbol: str, *, provider: str = "fmp", start: str | None = None, end: str | None = None,
             observation_dates: Iterable[str | datetime] | None = None,
             output_format: Literal["polars"] = "polars") -> pl.DataFrame:
        if output_format != "polars":
            raise ValueError("CompanyNewsStore.read only supports Polars output")
        provider_name = str(provider or "fmp").strip().lower()
        storage_symbol = symbol_provider_key(symbol.strip().upper(), provider_name)
        frame = read_provider_frame(self.backend, base_library=COMPANY_NEWS_LIBRARY, provider=provider_name,
                                    symbol=storage_symbol, start_date=start, end_date=end, output_format="polars")
        if frame is None or frame.is_empty():
            return pl.DataFrame()
        if observation_dates is not None and "observation_date" in frame.columns:
            wanted = [datetime.fromisoformat(str(value)[:10]) for value in observation_dates]
            frame = frame.filter(_date_expr(frame, "observation_date").is_in(wanted))
        return frame

    @staticmethod
    def _prepare(frame: pl.DataFrame, *, provider: str) -> pl.DataFrame:
        required = {"symbol", "published_at", "observation_date"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing company news columns: {', '.join(missing)}")
        out = frame.with_columns(
            pl.col("symbol").cast(pl.String, strict=False).str.strip_chars().str.to_uppercase(),
            _date_expr(frame, "published_at").alias("published_at"),
            _date_expr(frame, "observation_date").alias("observation_date"),
            pl.lit(provider).alias("provider"),
        ).filter((pl.col("symbol") != "") & pl.col("published_at").is_not_null())
        return out.sort("published_at")

    @staticmethod
    def _merge(existing: pl.DataFrame | None, incoming: pl.DataFrame) -> pl.DataFrame:
        combined = incoming if existing is None or existing.is_empty() else pl.concat([existing, incoming], how="diagonal_relaxed")
        identity = [name for name in ("symbol", "published_at", "url", "title") if name in combined.columns]
        return combined.unique(identity, keep="last").sort([name for name in ("published_at", "symbol") if name in combined.columns])
