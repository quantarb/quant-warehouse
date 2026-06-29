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


class ProviderRoutingBackend:
    """Route provider-scoped libraries to provider-specific Arctic roots."""

    kind: StorageKind = "arctic"

    def __init__(
        self,
        config: WarehouseConfig,
        *,
        storage_lock: threading.RLock | None = None,
    ) -> None:
        self.config = config
        self._storage_lock = storage_lock
        self._default = ArcticBackend(config.arctic_uri, storage_lock=storage_lock)
        self._provider_backends: dict[str, ArcticBackend] = {}

    def _backend_for_library(self, library: str) -> ArcticBackend:
        from quant_warehouse.warehouse.storage import provider_from_library

        provider = provider_from_library(library)
        if provider is None:
            return self._default
        backend = self._provider_backends.get(provider)
        if backend is None:
            uri = self.config.provider_arctic_uri(provider)
            if uri.startswith("lmdb://"):
                from pathlib import Path

                Path(uri.removeprefix("lmdb://")).mkdir(parents=True, exist_ok=True)
            backend = ArcticBackend(uri, storage_lock=self._storage_lock)
            self._provider_backends[provider] = backend
        return backend

    def read(self, library: str, symbol: str) -> pd.DataFrame | None:
        return self._backend_for_library(library).read(library, symbol)

    def write(
        self,
        library: str,
        symbol: str,
        df: pd.DataFrame,
        *,
        prune_previous_versions: bool = True,
    ) -> None:
        self._backend_for_library(library).write(
            library,
            symbol,
            df,
            prune_previous_versions=prune_previous_versions,
        )

    def list_symbols(self, library: str) -> list[str]:
        return self._backend_for_library(library).list_symbols(library)

    def has_symbol(self, library: str, symbol: str) -> bool:
        return self._backend_for_library(library).has_symbol(library, symbol)

    def delete(self, library: str, symbol: str) -> bool:
        return self._backend_for_library(library).delete(library, symbol)


def open_backend(
    config: WarehouseConfig,
    *,
    storage_lock: threading.RLock | None = None,
) -> ProviderRoutingBackend:
    config.ensure_dirs()
    return ProviderRoutingBackend(config, storage_lock=storage_lock)
