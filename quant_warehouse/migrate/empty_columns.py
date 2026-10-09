"""Explicit, bounded, version-preserving empty-payload schema migration.

Accept an already-authorized Library; never opens storage or creates a library.
Preparation is read-only and application is a separate explicit operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from io import BytesIO
import hashlib
import pandas as pd
import polars as pl

from quant_warehouse.warehouse.backend import (
    empty_payload_columns,
    _EMPTY_COLUMNS_METADATA,
    _COLUMN_ORDER_METADATA,
    _EMPTY_PAYLOAD_SCHEMA_METADATA,
    _REMOVED_EMPTY_METADATA,
)


@dataclass(frozen=True)
class EmptyColumnPlan:
    symbol: str
    source_version: int
    rows: int
    physical_removed: tuple[str, ...]
    logical_removed: tuple[str, ...]
    original: pd.DataFrame
    cleaned: pd.DataFrame
    metadata: dict
    retained_sha256: str
    index_order_repair: bool = False

    def summary(self) -> dict:
        return dict(
            symbol=self.symbol,
            source_version=self.source_version,
            rows=self.rows,
            physical_removed=list(self.physical_removed),
            logical_removed=list(self.logical_removed),
            retained_sha256=self.retained_sha256,
            index_order_repair=self.index_order_repair,
            dry_run=True,
            storage_savings_bytes=None,
            prune_previous_versions=False,
        )


def _hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(repr([(str(c), str(t)) for c, t in frame.dtypes.items()]).encode())
    digest.update(repr(frame.index.names).encode())
    return digest.hexdigest()


def prepare_empty_column_cleanup(
    library, symbol: str, *, max_rows: int = 100_000, key_columns: tuple[str, ...] = ()
) -> EmptyColumnPlan:
    """Read and validate every value of one pinned version, bounded by max_rows.

    Caller-supplied keys supplement reserved date, identity and dimension fields.
    Empty strings/NaNs/nulls are missing, but zero, false, whitespace and infinity
    are values. No sampling, writes, downloads or version pruning occurs here.
    """
    version = library.read_metadata(symbol).version
    description = library.get_description(symbol, as_of=version)
    if max_rows <= 0 or description.row_count > max_rows:
        raise ValueError(
            f"{symbol}: {description.row_count} rows exceeds bounded audit limit {max_rows}"
        )
    item = library.read(symbol, as_of=version)
    original = item.data
    if not isinstance(original, pd.DataFrame) or not original.columns.is_unique:
        raise ValueError("Requires a DataFrame with unique columns")
    if item.metadata is not None and not isinstance(item.metadata, dict):
        raise ValueError("Non-dictionary metadata requires manual preservation review")
    metadata = deepcopy(item.metadata or {})
    # Normalize missing sentinels only in the inspection frame. The original
    # pandas frame is retained verbatim for hashes, writes and rollback checks.
    frame = pl.from_pandas(original, nan_to_null=True)
    physical = tuple(empty_payload_columns(frame, key_columns=key_columns))
    omitted = metadata.get(_EMPTY_COLUMNS_METADATA, ())
    # Reject conflicting metadata rather than concealing physically populated data.
    if any(name in original.columns and name not in physical for name in omitted):
        raise ValueError("Empty-column metadata conflicts with retained physical fields")
    placeholders = pl.DataFrame({name: [None] for name in omitted if name not in original.columns})
    logical = (
        tuple(empty_payload_columns(placeholders, key_columns=key_columns)) if len(original) else ()
    )
    removed = list(dict.fromkeys([*physical, *logical]))
    cleaned = original.drop(columns=list(physical)).copy()
    if removed and not len(cleaned.columns) and not isinstance(original.index, pd.DatetimeIndex):
        raise ValueError(
            "Removing the final payload without a temporal index requires manual row-representation review"
        )
    sha = _hash(cleaned)
    metadata["quant_warehouse_empty_column_cleanup"] = {
        "source_version": version,
        "original_columns": list(original.columns),
        "original_column_order": list(metadata.get(_COLUMN_ORDER_METADATA, original.columns)),
        "original_index_names": list(original.index.names),
        "original_dtypes": {str(c): str(t) for c, t in original.dtypes.items()},
        "physical_removed": list(physical),
        "logical_removed": list(logical),
        "rows": len(original),
        "retained_sha256": sha,
        "empty_definition": "null/NaT, floating NaN, exact empty string; keys preserved",
    }
    metadata[_EMPTY_COLUMNS_METADATA] = [n for n in omitted if n not in removed]
    index_display_names = [n for n in original.index.names if isinstance(n, str)]
    if isinstance(original.index, pd.DatetimeIndex) and original.index.name is None:
        # Warehouse readers restore legacy unnamed time indexes as observation date.
        index_display_names = ["date"]
    default_order = index_display_names + list(original.columns)
    cleaned_order = [
        n for n in metadata.get(_COLUMN_ORDER_METADATA, default_order) if n not in removed
    ]
    for name in reversed(index_display_names):
        if name not in cleaned_order:
            cleaned_order.insert(0, name)
    index_order_repair = (
        "quant_warehouse_empty_column_cleanup" in (item.metadata or {})
        and list((item.metadata or {}).get(_COLUMN_ORDER_METADATA, [])) != cleaned_order
    )
    if index_order_repair and not removed:
        # Metadata-only date-order repairs preserve the original cleanup record.
        metadata = deepcopy(item.metadata)
    metadata[_COLUMN_ORDER_METADATA] = cleaned_order
    if index_order_repair and not removed:
        return EmptyColumnPlan(
            symbol,
            version,
            len(original),
            physical,
            logical,
            original.copy(),
            cleaned,
            metadata,
            sha,
            index_order_repair,
        )
    metadata[_REMOVED_EMPTY_METADATA] = list(
        dict.fromkeys([*metadata.get(_REMOVED_EMPTY_METADATA, ()), *removed])
    )
    old_schema_bytes = metadata.get(_EMPTY_PAYLOAD_SCHEMA_METADATA)
    empty_schema = dict(pl.read_ipc(BytesIO(old_schema_bytes)).schema) if old_schema_bytes else {}
    empty_schema.update({name: frame.schema[name] for name in physical})
    metadata[_EMPTY_PAYLOAD_SCHEMA_METADATA] = (
        pl.DataFrame(schema=empty_schema).write_ipc(None).getvalue()
    )
    return EmptyColumnPlan(
        symbol,
        version,
        len(original),
        physical,
        logical,
        original.copy(),
        cleaned,
        metadata,
        sha,
        index_order_repair,
    )


def apply_empty_column_cleanup(library, plan: EmptyColumnPlan) -> dict:
    """Explicitly write one cleaned version, retaining original versions/metadata.

    Refuse changed sources or modified plans. Caller must serialize concurrent
    writers: ArcticDB does not expose a compare-and-swap write through this API.
    """
    if library.read_metadata(plan.symbol).version != plan.source_version:
        raise RuntimeError("Source advanced; prepare a fresh plan")
    if _hash(plan.cleaned) != plan.retained_sha256:
        raise RuntimeError("Plan changed after validation")
    actual = library.read(plan.symbol, as_of=plan.source_version).data
    pd.testing.assert_frame_equal(actual, plan.original)
    expected = plan.original.drop(columns=list(plan.physical_removed))
    pd.testing.assert_frame_equal(expected, plan.cleaned)
    if not plan.physical_removed and not plan.logical_removed and not plan.index_order_repair:
        return {**plan.summary(), "dry_run": False, "changed": False}
    if plan.physical_removed:
        written = library.write(
            plan.symbol, plan.cleaned, metadata=plan.metadata, prune_previous_versions=False
        )
    else:
        # Already physically omitted fields need only a new schema metadata version.
        written = library.write_metadata(plan.symbol, plan.metadata, prune_previous_versions=False)
    pd.testing.assert_frame_equal(
        library.read(plan.symbol, as_of=written.version).data,
        expected,
        check_column_type=bool(len(expected.columns)),
    )
    pd.testing.assert_frame_equal(
        library.read(plan.symbol, as_of=plan.source_version).data, plan.original
    )
    return {
        **plan.summary(),
        "dry_run": False,
        "changed": True,
        "new_version": written.version,
        "original_version_retained": True,
    }
