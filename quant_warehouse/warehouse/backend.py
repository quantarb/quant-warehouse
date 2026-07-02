from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Literal, Protocol

import pandas as pd

from quant_warehouse.config import WarehouseConfig

StorageKind = Literal["arctic"]


class StorageBackend(Protocol):
    kind: StorageKind

    def read(
        self,
        library: str,
        symbol: str,
        *,
        date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame | None: ...

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
        self._libraries: dict[str, object] = {}

    def _library(self, name: str):
        cached = self._libraries.get(name)
        if cached is not None:
            return cached
        if name not in self._arctic.list_libraries():
            self._arctic.create_library(name)
        library = self._arctic.get_library(name)
        self._libraries[name] = library
        return library

    def read(
        self,
        library: str,
        symbol: str,
        *,
        date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame | None:
        with self._storage_guard():
            lib = self._library(library)
            if not lib.has_symbol(symbol):
                return None
            read_kwargs = {}
            if date_range is not None:
                read_kwargs["date_range"] = date_range
            if columns is not None:
                read_kwargs["columns"] = columns
            version = lib.read(symbol, **read_kwargs)
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

    def read(
        self,
        library: str,
        symbol: str,
        *,
        date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame | None:
        return self._backend_for_library(library).read(
            library,
            symbol,
            date_range=date_range,
            columns=columns,
        )

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


_BACKEND_CACHE: dict[tuple[str], ProviderRoutingBackend] = {}
_BACKEND_CACHE_LOCK = threading.RLock()


def open_backend(
    config: WarehouseConfig,
    *,
    storage_lock: threading.RLock | None = None,
) -> ProviderRoutingBackend:
    config.ensure_dirs()
    key = (str(config.arctic_uri),)
    with _BACKEND_CACHE_LOCK:
        backend = _BACKEND_CACHE.get(key)
        if backend is None:
            backend = ProviderRoutingBackend(config, storage_lock=storage_lock)
            _BACKEND_CACHE[key] = backend
        return backend
