from __future__ import annotations

import threading
from io import BytesIO
from contextlib import nullcontext
from typing import Literal, Protocol

import polars as pl
from datetime import datetime

from quant_warehouse.config import WarehouseConfig

StorageKind = Literal["arctic"]
FrameFormat = Literal["polars"]
_EMPTY_COLUMNS_METADATA = "quant_warehouse_empty_columns"
_COLUMN_ORDER_METADATA = "quant_warehouse_column_order"
_EMPTY_PAYLOAD_DETAILS_METADATA = "quant_warehouse_empty_payload_details"
_EMPTY_PAYLOAD_SCHEMA_METADATA = "quant_warehouse_empty_payload_schema"
_REMOVED_EMPTY_METADATA = "quant_warehouse_removed_empty_columns"


def empty_payload_columns(frame: pl.DataFrame, *, key_columns: tuple[str, ...] = (),
                          payload_columns: tuple[str, ...] | None = None) -> list[str]:
    """Find wholly missing payloads; keys and zero-row schemas are never removed.

    Missing means null/NaT, float NaN, or the exact empty string. Whitespace,
    infinity, zero, false, and nested/container values are meaningful values.
    """
    from quant_warehouse.ingest.normalize import INDEX_CANDIDATES, PANEL_DIMENSION_COLUMNS
    from quant_warehouse.warehouse.column_roles import REVIEWED_PAYLOAD_COLUMNS, retained_schema_reason

    allowed = set(REVIEWED_PAYLOAD_COLUMNS if payload_columns is None else payload_columns)
    protected = set(INDEX_CANDIDATES) | set(PANEL_DIMENSION_COLUMNS) | set(key_columns) | {
        "__index__", "contract_symbol", "underlying_symbol", "expiration", "strike", "option_type",
        "id", "cik", "event_id", "security_id", "instrument_id", "company_id",
        "period", "fiscal_year", "calendar_year", "quarter",
        "created_at", "updated_at", "published_at", "quote_timestamp", "snapshot_date", "eod_date",
        "event_date", "timestamp", "declaration_date", "published_time", "updated",
        "maturity_date", "last_updated", "instrument_symbol",
    }
    if not frame.height:
        return []
    expressions = []
    for name, dtype in frame.schema.items():
        if name in protected or name not in allowed or retained_schema_reason(name, dtype):
            continue
        missing = pl.col(name).is_null()
        if dtype.is_float():
            missing = missing | pl.col(name).is_nan()
        elif dtype.base_type() in {pl.String, pl.Categorical, pl.Enum}:
            missing = missing | (pl.col(name).cast(pl.String) == "")
        expressions.append(missing.all().alias(name))
    if not expressions:
        return []
    # Audit all columns in one vectorized operation instead of one eager query
    # per floating/string column. This remains a full-value audit, without sampling.
    return [name for name, empty in frame.select(expressions).row(0, named=True).items() if empty]


class StorageBackend(Protocol):
    kind: StorageKind

    def read(
        self,
        library: str,
        symbol: str,
        *,
        date_range: tuple[datetime | None, datetime | None] | None = None,
        columns: list[str] | None = None,
        output_format: FrameFormat = "polars",
        include_empty_columns: bool = True,
    ) -> pl.DataFrame | None: ...

    def column_non_null_counts(self, library: str, symbol: str) -> dict[str, int]: ...

    def write(
        self,
        library: str,
        symbol: str,
        df: pl.DataFrame,
        *,
        prune_previous_versions: bool = True,
        key_columns: tuple[str, ...] = (),
        payload_columns: tuple[str, ...] | None = None,
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
        date_range: tuple[datetime | None, datetime | None] | None = None,
        columns: list[str] | None = None,
        output_format: FrameFormat = "polars",
        include_empty_columns: bool = True,
    ) -> pl.DataFrame | None:
        with self._storage_guard():
            lib = self._library(library)
            if not lib.has_symbol(symbol):
                return None
            read_kwargs = {}
            stored_metadata = None
            if columns is not None:
                metadata_item = lib.read_metadata(symbol)
                stored_metadata = getattr(metadata_item, "metadata", None) or {}
                empty_columns = (set(stored_metadata.get(_EMPTY_COLUMNS_METADATA, ()))
                                 | set(stored_metadata.get(_REMOVED_EMPTY_METADATA, ())))
                physical_columns = [name for name in columns if name not in empty_columns]
            else:
                physical_columns = None
            filter_dates = date_range is not None and lib.get_description(symbol).sorted != "ASCENDING"
            if date_range is not None:
                if not filter_dates:
                    read_kwargs["date_range"] = date_range
            if columns is not None:
                read_kwargs["columns"] = list(dict.fromkeys([*physical_columns, "date"])) if filter_dates else physical_columns
            read_kwargs["output_format"] = output_format
            try:
                version = lib.read(symbol, **read_kwargs)
            except Exception as exc:
                if _is_missing_symbol_error(exc):
                    return None
                raise
            df = version.data
            if df is None or df.is_empty():
                return None
            stored_metadata = stored_metadata or getattr(version, "metadata", None) or {}
            empty_columns = list(stored_metadata.get(_EMPTY_COLUMNS_METADATA, ()))
            if columns is not None:
                empty_columns = list(dict.fromkeys([*empty_columns,
                                      *stored_metadata.get(_REMOVED_EMPTY_METADATA, ())]))
            # Explicit selections override clean-schema mode for fixed-schema callers.
            requested_empty = [name for name in empty_columns
                               if (columns is not None and name in columns)
                               or (columns is None and include_empty_columns)]
            if requested_empty:
                schema_bytes = stored_metadata.get(_EMPTY_PAYLOAD_SCHEMA_METADATA)
                empty_schema = pl.read_ipc(BytesIO(schema_bytes)).schema if schema_bytes else {}
                df = df.with_columns(pl.lit(None, dtype=empty_schema.get(name, pl.Null)).alias(name)
                                     for name in requested_empty)
            # Older datasets used an unnamed time index. ArcticDB exposes it
            # as __index__; restore that observation date before normalization.
            if "__index__" in df.columns and "date" not in df.columns and isinstance(df.schema["__index__"], pl.Datetime):
                df = df.rename({"__index__": "date"})
            # ArcticDB's Polars reader can expose pandas NaT as the minimum
            # int64 timestamp instead of a null. Restore the missing dates.
            timestamp_columns = [name for name, dtype in df.schema.items()
                                 if isinstance(dtype, pl.Datetime)]
            df = df.with_columns(
                pl.when(pl.col(name).cast(pl.Int64) == -(2**63))
                .then(pl.lit(None, dtype=df.schema[name])).otherwise(pl.col(name)).alias(name)
                for name in timestamp_columns
            )
            if filter_dates:
                start, end = date_range
                if start is not None:
                    df = df.filter(pl.col("date") >= start)
                if end is not None:
                    df = df.filter(pl.col("date") <= end)
                if columns is not None:
                    df = df.select(columns)
            elif columns is not None:
                df = df.select(columns)
            elif stored_metadata.get(_COLUMN_ORDER_METADATA):
                order = [name for name in stored_metadata[_COLUMN_ORDER_METADATA] if name in df.columns]
                df = df.select(order)
            return df.sort("date") if "date" in df.columns else df

    def column_non_null_counts(self, library: str, symbol: str) -> dict[str, int]:
        """Count populated values in the complete stored table, without mutation.

        Time-indexed tables use ArcticDB aggregation; row-indexed tables use
        bounded reads. Warehouse metadata restores physically omitted null columns.
        """
        from arcticdb import QueryBuilder

        with self._storage_guard():
            lib = self._library(library)
            if not lib.has_symbol(symbol):
                raise KeyError(symbol)
            description = lib.get_description(symbol)
            metadata = lib.read_metadata(symbol).metadata or {}
            columns = [column.name for column in description.columns]
            counts = {name: 0 for name in columns}
            if description.row_count and columns:
                if description.index_type == "index" and description.sorted == "ASCENDING":
                    query = QueryBuilder().resample("36500D").agg({name: "count" for name in columns})
                    result = lib.read(symbol, query_builder=query, output_format="polars").data
                    counts.update({name: int(result[name].sum() or 0) for name in columns})
                else:
                    for first in range(0, description.row_count, 100_000):
                        frame = lib.read(symbol, columns=columns,
                            row_range=(first, min(first + 100_000, description.row_count)),
                            output_format="polars").data
                        for name in columns:
                            values = frame[name]
                            if isinstance(values.dtype, pl.Datetime):
                                values = values.filter(values.cast(pl.Int64) != -(2**63))
                            if values.dtype.is_float():
                                values = values.fill_nan(None)
                            counts[name] += len(values) - values.null_count()
            counts.update({column.name: int(description.row_count) for column in description.index})
            counts.update({name: 0 for name in metadata.get(_EMPTY_COLUMNS_METADATA, ())})
            return counts

    def write(
        self,
        library: str,
        symbol: str,
        df: pl.DataFrame,
        *,
        prune_previous_versions: bool = True,
        key_columns: tuple[str, ...] = (),
        payload_columns: tuple[str, ...] | None = None,
    ) -> None:
        if df.is_empty():
            return
        with self._storage_guard():
            lib = self._library(library)
            # ArcticDB's time-series writer still requires an indexed frame.
            # Keep that conversion isolated here; all warehouse callers and
            # all reads remain Polars, so the storage-library adapter never
            # crosses the public API.
            from quant_warehouse.ingest.normalize import INDEX_CANDIDATES

            # All-missing source dates have no inferred Polars dtype. Without
            # an explicit date type pandas emits an object column, forcing
            # ArcticDB to inspect every null and log an inference message.
            # Preserve the column and its missing values as typed timestamps.
            empty_dates = [name for name in INDEX_CANDIDATES
                           if name in df.columns and df[name].null_count() == df.height]
            from quant_warehouse.warehouse.column_roles import REVIEWED_PAYLOAD_COLUMNS
            prior_metadata = lib.read_metadata(symbol).metadata if lib.has_symbol(symbol) else None
            prior_removed = (prior_metadata or {}).get(_REMOVED_EMPTY_METADATA, ()) if isinstance(prior_metadata, dict) else ()
            eligible_payloads = set(REVIEWED_PAYLOAD_COLUMNS if payload_columns is None else payload_columns)
            eligible_payloads.update(prior_removed)
            empty_payload = empty_payload_columns(df, key_columns=key_columns,
                                                  payload_columns=tuple(eligible_payloads))
            # Missing payload values are canonically restored as null. Keep their
            # original dtype/counts as provenance without retaining row-sized data.
            empty_details = {
                name: {
                    "dtype": str(df.schema[name]),
                    "null_count": df[name].null_count(),
                    "nan_count": int(df[name].is_nan().sum()) if df.schema[name].is_float() else 0,
                    "empty_string_count": int((df[name] == "").sum()) if df.schema[name].base_type() in {pl.String, pl.Categorical, pl.Enum} else 0,
                }
                for name in empty_payload
            }
            storage_frame = (
                df.drop(empty_payload)
                .with_columns(pl.col(name).cast(pl.Datetime("ns")) for name in empty_dates)
            )
            write_frame = storage_frame.to_pandas()
            # Partially dated source panels must preserve unknown dates. ArcticDB
            # rejects NaT in its time index, so keep such panels row-indexed.
            if "date" in df.columns and df["date"].null_count() == 0:
                write_frame = write_frame.set_index("date")
                write_frame.index.name = "date"
            retained_metadata = (dict(prior_metadata) if isinstance(prior_metadata, dict)
                                 else {"quant_warehouse_source_metadata": prior_metadata}
                                 if prior_metadata is not None else {})
            removed = retained_metadata.get(_REMOVED_EMPTY_METADATA, ())
            clean_schema = "quant_warehouse_empty_column_cleanup" in retained_metadata
            removed = [name for name in removed if name not in df.columns or name in empty_payload]
            if clean_schema:
                removed = list(dict.fromkeys([*removed, *empty_payload]))
            retained_metadata[_REMOVED_EMPTY_METADATA] = removed
            prior_schema_bytes = retained_metadata.get(_EMPTY_PAYLOAD_SCHEMA_METADATA)
            prior_schema = pl.read_ipc(BytesIO(prior_schema_bytes)).schema if prior_schema_bytes else {}
            empty_schema = {name: dtype for name, dtype in prior_schema.items() if name in removed}
            empty_schema.update({name: df.schema[name] for name in empty_payload})
            retained_details = {name: details for name, details in retained_metadata.get(
                _EMPTY_PAYLOAD_DETAILS_METADATA, {}).items() if name in removed}
            retained_details.update(empty_details)
            lib.write(
                symbol,
                write_frame.sort_index(),
                metadata={
                    **retained_metadata,
                    _EMPTY_COLUMNS_METADATA: [] if clean_schema else empty_payload,
                    _COLUMN_ORDER_METADATA: [n for n in df.columns if n not in removed],
                    _EMPTY_PAYLOAD_DETAILS_METADATA: retained_details,
                    _EMPTY_PAYLOAD_SCHEMA_METADATA: pl.DataFrame(schema=empty_schema).write_ipc(None).getvalue(),
                },
                prune_previous_versions=prune_previous_versions,
            )

    def refresh_version_cache(self, library: str) -> None:
        """Observe cross-process writes before a locked read/merge/write."""
        with self._storage_guard():
            self._library(library)._nvs.version_store.flush_version_map()

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


def _is_missing_symbol_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "nodatafound" in text or "keynotfound" in text or "symbol not found" in text


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

    def refresh_version_cache(self, library: str) -> None:
        self._backend_for_library(library).refresh_version_cache(library)

    def read(
        self,
        library: str,
        symbol: str,
        *,
        date_range: tuple[datetime | None, datetime | None] | None = None,
        columns: list[str] | None = None,
        output_format: FrameFormat = "polars",
        include_empty_columns: bool = True,
    ) -> pl.DataFrame | None:
        return self._backend_for_library(library).read(
            library,
            symbol,
            date_range=date_range,
            columns=columns,
            output_format=output_format,
            include_empty_columns=include_empty_columns,
        )

    def column_non_null_counts(self, library: str, symbol: str) -> dict[str, int]:
        return self._backend_for_library(library).column_non_null_counts(library, symbol)

    def write(
        self,
        library: str,
        symbol: str,
        df: pl.DataFrame,
        *,
        prune_previous_versions: bool = True,
        key_columns: tuple[str, ...] = (),
        payload_columns: tuple[str, ...] | None = None,
    ) -> None:
        self._backend_for_library(library).write(
            library,
            symbol,
            df,
            prune_previous_versions=prune_previous_versions,
            key_columns=key_columns,
            payload_columns=payload_columns,
        )

    def list_symbols(self, library: str) -> list[str]:
        return self._backend_for_library(library).list_symbols(library)

    def has_symbol(self, library: str, symbol: str) -> bool:
        return self._backend_for_library(library).has_symbol(library, symbol)

    def delete(self, library: str, symbol: str) -> bool:
        return self._backend_for_library(library).delete(library, symbol)


_BACKEND_CACHE: dict[tuple[str], ProviderRoutingBackend] = {}
_BACKEND_CACHE_LOCK = threading.RLock()
_DEFAULT_STORAGE_LOCK = threading.RLock()


def open_backend(
    config: WarehouseConfig,
    *,
    storage_lock: threading.RLock | None = None,
) -> ProviderRoutingBackend:
    config.ensure_dirs()
    key = (str(config.arctic_uri),)
    lock = storage_lock or _DEFAULT_STORAGE_LOCK
    with _BACKEND_CACHE_LOCK:
        backend = _BACKEND_CACHE.get(key)
        if backend is None:
            backend = ProviderRoutingBackend(config, storage_lock=lock)
            _BACKEND_CACHE[key] = backend
        return backend
