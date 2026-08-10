from __future__ import annotations

import polars as pl

def merge_upsert(existing: pl.DataFrame | None, incoming: pl.DataFrame) -> pl.DataFrame:
    """Merge incoming rows onto existing data, keeping the latest value per index."""
    if incoming.is_empty():
        return existing.clone() if existing is not None else pl.DataFrame()
    if existing is None or existing.is_empty():
        return incoming.sort("date") if "date" in incoming.columns else incoming
    key = "date" if "date" in incoming.columns else incoming.columns[0]
    return pl.concat([existing, incoming], how="diagonal_relaxed").unique(
        key, keep="last", maintain_order=True
    ).sort(key)


def merge_panel_upsert(existing: pl.DataFrame | None, incoming: pl.DataFrame) -> pl.DataFrame:
    """Merge panel rows keyed by date plus a holding identifier column."""
    if incoming.is_empty():
        return existing.clone() if existing is not None else pl.DataFrame()
    if existing is None or existing.is_empty():
        return incoming
    index_name = "date" if "date" in incoming.columns else incoming.columns[0]
    keys = [
        index_name,
        *(
            column
            for column in ("symbol", "contract_symbol", "fund_symbol", "country")
            if column in incoming.columns and column in existing.columns
        ),
    ]
    return pl.concat([existing, incoming], how="diagonal_relaxed").unique(
        keys, keep="last", maintain_order=True
    ).sort(keys)
