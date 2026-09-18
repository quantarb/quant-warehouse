from __future__ import annotations

import polars as pl

def _align_timestamps(existing: pl.DataFrame, incoming: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Use UTC without timezone metadata when combining timestamp sources."""
    frames = []
    for frame in (existing, incoming):
        expressions = []
        for name, dtype in frame.schema.items():
            if isinstance(dtype, pl.Datetime):
                value = pl.col(name)
                if dtype.time_zone:
                    value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
                expressions.append(value.cast(pl.Datetime("ns")).alias(name))
        frames.append(frame.with_columns(expressions))
    return frames[0], frames[1]


def merge_upsert(existing: pl.DataFrame | None, incoming: pl.DataFrame, *, date_column: str | None = None) -> pl.DataFrame:
    """Merge incoming rows onto existing data, keeping the latest value per index."""
    if incoming.is_empty():
        return existing.clone() if existing is not None else pl.DataFrame()
    if existing is None or existing.is_empty():
        return incoming.sort("date") if "date" in incoming.columns else incoming
    key = date_column or ("date" if "date" in incoming.columns else incoming.columns[0])
    existing, incoming = _align_timestamps(existing, incoming)
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
            for column in ("symbol", "contract_symbol", "fund_symbol", "country", "source_event_id", "source_id",
                           "owner_cik", "owner_name", "security_type", "transaction_type", "securities_transacted", "transaction_price",
                           "representative", "owner", "amount", "analyst_name", "analyst_firm", "news_url", "url", "title")
            if column in incoming.columns and column in existing.columns
        ),
    ]
    existing, incoming = _align_timestamps(existing, incoming)
    return pl.concat([existing, incoming], how="diagonal_relaxed").unique(
        keys, keep="last", maintain_order=True
    ).sort(keys)
