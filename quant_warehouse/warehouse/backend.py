from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Literal, Protocol

import pandas as pd

from quant_warehouse.config import WarehouseConfig

StorageKind = Literal["arctic"]


class StorageBackend(Protocol):
    kind: StorageKind

    def read(self, library: str, symbol: str) -> pd.DataFrame | None: ...

    def write(
        self,
        library: str,
        symbol: str,
        df: pd.DataFrame,
        *,
        prune_previous_versions: bool = True,
    ) -> None: ...


class ArcticBackend:
    """ArcticDB time-series store (LMDB or S3)."""

    kind: StorageKind = "arctic"

    def __init__(self, uri: str, *, storage_lock: threading.RLock | None = None) -> None:
        from quant_warehouse.deps import require_arcticdb

        require_arcticdb()
        from arcticdb import Arctic

        self._uri = uri
        self._arctic = Arctic(uri)
        self._storage_lock = storage_lock

    def _library(self, name: str):
        if name not in self._arctic.list_libraries():
            self._arctic.create_library(name)
        return self._arctic.get_library(name)

    def read(self, library: str, symbol: str) -> pd.DataFrame | None:
        with self._storage_guard():
            lib = self._library(library)
            if not lib.has_symbol(symbol):
                return None
            version = lib.read(symbol)
            df = version.data
            if df is None or df.empty:
                return None
            if not isinstance(df.index, pd.DatetimeIndex) and df.index.name in ("date", "period_ending"):
                df.index = pd.to_datetime(df.index, errors="coerce")
            return df.sort_index()

    def write(
        self,
        library: str,
        symbol: str,
        df: pd.DataFrame,
        *,
        prune_previous_versions: bool = True,
    ) -> None:
        if df.empty:
            return
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        with self._storage_guard():
            lib = self._library(library)
            lib.write(symbol, df.sort_index(), prune_previous_versions=prune_previous_versions)

    def _storage_guard(self):
        return self._storage_lock or nullcontext()

    def list_symbols(self, library: str) -> list[str]:
        with self._storage_guard():
            lib = self._library(library)
            return [str(symbol) for symbol in lib.list_symbols()]

    def has_symbol(self, library: str, symbol: str) -> bool:
        lib = self._library(library)
        return bool(lib.has_symbol(symbol))

    def delete(self, library: str, symbol: str) -> bool:
        lib = self._library(library)
        if not lib.has_symbol(symbol):
            return False
        lib.delete(symbol)
        return True


def open_backend(
    config: WarehouseConfig,
    *,
    storage_lock: threading.RLock | None = None,
) -> ArcticBackend:
    config.ensure_dirs()
    return ArcticBackend(config.arctic_uri, storage_lock=storage_lock)
