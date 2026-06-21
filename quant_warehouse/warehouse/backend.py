from __future__ import annotations

from typing import Literal, Protocol

import pandas as pd

from quant_warehouse.config import WarehouseConfig

StorageKind = Literal["arctic"]


class StorageBackend(Protocol):
    kind: StorageKind

    def read(self, library: str, symbol: str) -> pd.DataFrame | None: ...

    def write(self, library: str, symbol: str, df: pd.DataFrame) -> None: ...


class ArcticBackend:
    """ArcticDB time-series store (LMDB or S3)."""

    kind: StorageKind = "arctic"

    def __init__(self, uri: str) -> None:
        from quant_warehouse.deps import require_arcticdb

        require_arcticdb()
        from arcticdb import Arctic

        self._uri = uri
        self._arctic = Arctic(uri)

    def _library(self, name: str):
        if name not in self._arctic.list_libraries():
            self._arctic.create_library(name)
        return self._arctic.get_library(name)

    def read(self, library: str, symbol: str) -> pd.DataFrame | None:
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

    def write(self, library: str, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        lib = self._library(library)
        lib.write(symbol, df.sort_index())

    def has_symbol(self, library: str, symbol: str) -> bool:
        lib = self._library(library)
        return bool(lib.has_symbol(symbol))

    def delete(self, library: str, symbol: str) -> bool:
        lib = self._library(library)
        if not lib.has_symbol(symbol):
            return False
        lib.delete(symbol)
        return True


def open_backend(config: WarehouseConfig) -> ArcticBackend:
    config.ensure_dirs()
    return ArcticBackend(config.arctic_uri)