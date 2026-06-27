from __future__ import annotations

import pandas as pd

from quant_warehouse.ingest.normalize import _coerce_object_strings, coerce_object_dates


def merge_upsert(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge incoming rows onto existing data, keeping the latest value per index."""
    if incoming is None or incoming.empty:
        if existing is None or existing.empty:
            return pd.DataFrame()
        return existing.copy()

    incoming = incoming.copy()
    if existing is None or existing.empty:
        return incoming.sort_index()

    combined = pd.concat([existing, incoming])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def merge_panel_upsert(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge panel rows keyed by date plus a holding identifier column."""
    if incoming is None or incoming.empty:
        if existing is None or existing.empty:
            return pd.DataFrame()
        return existing.copy()

    incoming = incoming.copy()
    if existing is None or existing.empty:
        return incoming.sort_index()

    combined = pd.concat([existing, incoming]).reset_index()
    index_name = incoming.index.name or "as_of"
    dedupe_keys = [
        key
        for key in combined.columns
        if key != index_name
        and (
            pd.api.types.is_object_dtype(combined[key])
            or pd.api.types.is_string_dtype(combined[key])
            or pd.api.types.is_datetime64_any_dtype(combined[key])
        )
    ]
    dedupe_cols = [index_name, *dedupe_keys]
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    combined[index_name] = pd.to_datetime(combined[index_name], errors="coerce")
    combined = combined.dropna(subset=[index_name]).sort_values(dedupe_cols)
    combined = combined.set_index(index_name)
    combined.index.name = index_name
    combined.index = pd.DatetimeIndex(combined.index)
    if combined.index.tz is not None:
        combined.index = combined.index.tz_convert(None)
    combined = coerce_object_dates(combined)
    combined = _coerce_object_strings(combined)
    return combined
