from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LINEAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetLineageManifest:
    """Immutable identity and point-in-time boundary for a prepared dataset."""

    dataset_id: str
    dataset_kind: str
    provider: str
    available_at_cutoff: str
    start_date: str | None
    end_date: str | None
    row_count: int
    symbols: tuple[str, ...]
    columns: tuple[str, ...]
    dtypes: Mapping[str, str]
    content_fingerprint: str
    recipe_id: str
    recipe_fingerprint: str
    source_references: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = LINEAGE_SCHEMA_VERSION

    @property
    def lineage_fingerprint(self) -> str:
        return canonical_fingerprint(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = _jsonable(asdict(self))
        payload["lineage_fingerprint"] = self.lineage_fingerprint
        return payload


def build_dataset_lineage_manifest(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_kind: str,
    provider: str,
    available_at_cutoff: str | pd.Timestamp,
    recipe_id: str,
    recipe: Mapping[str, Any],
    source_references: Mapping[str, str] | None = None,
    symbol_column: str = "symbol",
    date_column: str = "date",
    key_columns: Sequence[str] = ("symbol", "date"),
) -> DatasetLineageManifest:
    """Build a deterministic lineage manifest from the complete prepared frame."""

    if frame is None:
        raise TypeError("frame cannot be None")
    cutoff = pd.Timestamp(available_at_cutoff)
    if pd.isna(cutoff):
        raise ValueError("available_at_cutoff must be a valid timestamp")
    dates = (
        pd.to_datetime(frame[date_column], errors="coerce").dropna()
        if date_column in frame.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff
    if not dates.empty and dates.max() > cutoff_naive:
        raise ValueError("dataset contains dates after available_at_cutoff")
    symbols = (
        tuple(sorted(frame[symbol_column].dropna().astype(str).str.strip().str.upper().unique()))
        if symbol_column in frame.columns
        else ()
    )
    return DatasetLineageManifest(
        dataset_id=str(dataset_id),
        dataset_kind=str(dataset_kind),
        provider=str(provider),
        available_at_cutoff=cutoff.isoformat(),
        start_date=None if dates.empty else dates.min().isoformat(),
        end_date=None if dates.empty else dates.max().isoformat(),
        row_count=int(len(frame)),
        symbols=symbols,
        columns=tuple(str(column) for column in frame.columns),
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        content_fingerprint=dataframe_fingerprint(frame, key_columns=key_columns),
        recipe_id=str(recipe_id),
        recipe_fingerprint=canonical_fingerprint(recipe),
        source_references={str(key): str(value) for key, value in dict(source_references or {}).items()},
    )


def dataframe_fingerprint(frame: pd.DataFrame, *, key_columns: Sequence[str] = ()) -> str:
    """Hash all values, column order, and dtypes deterministically."""

    work = frame.copy(deep=False)
    converted = False
    for column in work.columns:
        if work[column].dtype == "object" and work[column].map(
            lambda value: isinstance(value, (dict, list, tuple, set))
        ).any():
            if not converted:
                work = work.copy(deep=False)
                converted = True
            work[column] = work[column].map(_stable_cell)
    digest = hashlib.sha256()
    digest.update(canonical_json({"columns": list(frame.columns), "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()}}).encode())
    hashes = pd.util.hash_pandas_object(work, index=False, categorize=True).to_numpy(dtype=np.uint64)
    if any(column in frame.columns for column in key_columns):
        hashes.sort()
    digest.update(hashes.tobytes())
    return f"sha256:{digest.hexdigest()}"


def write_dataset_lineage_manifest(manifest: DatasetLineageManifest, path: Path) -> Path:
    """Write once, allowing only an identical manifest at an existing path."""

    output = Path(path)
    payload = manifest.to_dict()
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"refusing to overwrite different lineage manifest: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    return output


def read_dataset_lineage_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "dataset_id",
        "available_at_cutoff",
        "content_fingerprint",
        "recipe_fingerprint",
        "lineage_fingerprint",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"lineage manifest missing fields: {sorted(missing)}")
    claimed = str(payload["lineage_fingerprint"])
    identity = {key: value for key, value in payload.items() if key != "lineage_fingerprint"}
    actual = canonical_fingerprint(identity)
    if claimed != actual:
        raise ValueError(f"lineage manifest fingerprint mismatch: {path}")
    return payload


def canonical_fingerprint(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode()).hexdigest()}"


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return canonical_json(value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    return value
