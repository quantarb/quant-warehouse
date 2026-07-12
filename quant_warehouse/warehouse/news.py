from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.config import WarehouseConfig
from quant_warehouse.ingest.normalize import symbol_provider_key
from quant_warehouse.warehouse.backend import ArcticBackend, StorageBackend, open_backend
from quant_warehouse.warehouse.sections import COMPANY_NEWS_LIBRARY, COMPANY_NEWS_SECTION
from quant_warehouse.warehouse.storage import provider_library, read_provider_frame


class CompanyNewsStore:
    """Provider-scoped, point-in-time company news stored per equity symbol."""

    def __init__(
        self,
        config: WarehouseConfig | None = None,
        *,
        backend: StorageBackend | None = None,
        catalog: CatalogStore | None = None,
    ) -> None:
        self.config = config or WarehouseConfig.from_env()
        self.config.ensure_dirs()
        self.backend: ArcticBackend = backend or open_backend(self.config)
        self.catalog = catalog or CatalogStore(self.config.catalog_path)

    def import_frame(self, frame: pd.DataFrame, *, provider: str = "fmp") -> dict[str, int]:
        """Upsert a normalized news panel and return stored row counts by symbol."""
        provider_name = str(provider or "fmp").strip().lower()
        incoming = self._prepare(frame, provider=provider_name)
        if incoming.empty:
            return {}

        library = provider_library(COMPANY_NEWS_LIBRARY, provider_name)
        counts: dict[str, int] = {}
        for symbol, symbol_frame in incoming.groupby("symbol", sort=True):
            storage_symbol = symbol_provider_key(symbol, provider_name)
            existing = read_provider_frame(
                self.backend,
                base_library=COMPANY_NEWS_LIBRARY,
                provider=provider_name,
                symbol=storage_symbol,
            )
            merged = self._merge(existing, symbol_frame.drop(columns=["symbol"]))
            self.backend.write(library, storage_symbol, merged)
            counts[symbol] = len(merged)
            self.catalog.upsert(
                symbol=symbol,
                section=COMPANY_NEWS_SECTION,
                provider=provider_name,
                min_date=merged.index.min().strftime("%Y-%m-%d"),
                max_date=merged.index.max().strftime("%Y-%m-%d"),
                row_count=len(merged),
                columns_present=[str(column) for column in merged.columns],
            )
        return counts

    def import_parquet(self, path: str, *, provider: str = "fmp") -> dict[str, int]:
        return self.import_frame(pd.read_parquet(path), provider=provider)

    def read(
        self,
        symbol: str,
        *,
        provider: str = "fmp",
        start: str | None = None,
        end: str | None = None,
        observation_dates: Iterable[str | pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        provider_name = str(provider or "fmp").strip().lower()
        storage_symbol = symbol_provider_key(symbol.strip().upper(), provider_name)
        frame = read_provider_frame(
            self.backend,
            base_library=COMPANY_NEWS_LIBRARY,
            provider=provider_name,
            symbol=storage_symbol,
            start_date=pd.Timestamp(start) if start is not None else None,
            end_date=pd.Timestamp(end) if end is not None else None,
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        out = frame.copy()
        if observation_dates is not None:
            wanted = {pd.Timestamp(value).normalize() for value in observation_dates}
            dates = pd.to_datetime(out["observation_date"], errors="coerce").dt.normalize()
            out = out.loc[dates.isin(wanted)]
        return out

    @staticmethod
    def _prepare(frame: pd.DataFrame, *, provider: str) -> pd.DataFrame:
        required = {"symbol", "published_at", "observation_date"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing company news columns: {', '.join(missing)}")
        out = frame.copy()
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
        out["observation_date"] = pd.to_datetime(out["observation_date"], errors="coerce").dt.normalize()
        if "fetched_at" in out:
            out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce", utc=True).dt.tz_convert(None)
        out = out.loc[out["symbol"].ne("") & out["published_at"].notna()]
        out["provider"] = provider
        out = out.set_index(out["published_at"].dt.tz_convert(None)).drop(columns=["published_at"])
        out.index.name = "published_at"
        return out.sort_index()

    @staticmethod
    def _merge(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
        combined = incoming if existing is None or existing.empty else pd.concat([existing, incoming])
        rows = combined.reset_index()
        identity = [column for column in ("published_at", "url", "title") if column in rows]
        rows = rows.drop_duplicates(identity, keep="last").sort_values(identity)
        rows = rows.set_index("published_at")
        rows.index = pd.DatetimeIndex(rows.index)
        return rows
