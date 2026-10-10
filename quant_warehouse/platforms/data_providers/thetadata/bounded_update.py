"""Bounded existing-schema daily option upserts; caller holds store writer lock."""
from datetime import timedelta
import polars as pl


def update_existing_dates(backend, library, symbol, incoming, merge):
    """Return False for a new symbol; refuse incompatible existing schemas.

    Read only incoming dates and one schema row. ArcticDB update preserves all
    segments outside the complete-day range and retains previous versions.
    """
    actual = backend._backend_for_library(library) if hasattr(backend, '_backend_for_library') else backend
    if not hasattr(actual, '_library'):
        return False
    with actual._storage_guard():
        lib = actual._library(library)
        if not lib.has_symbol(symbol):
            return False
        desc = lib.get_description(symbol)
        if desc.sorted != 'ASCENDING' or desc.index_type != 'index':
            raise ValueError('Bounded options update requires a sorted time index')
        first = incoming['snapshot_date'].min()
        last = incoming['snapshot_date'].max() + timedelta(days=1) - timedelta(microseconds=1)
        template = lib.read(symbol, row_range=(0, 1), output_format='polars').data
        if '__index__' in template.columns:
            template = template.rename({'__index__': 'date'})
        physical = [c for c in template.columns if c != 'date']
        extra = [c for c in incoming.columns if c not in template.columns and incoming[c].null_count() < len(incoming)]
        if extra:
            raise ValueError('Bounded options update requires reviewed schema migration')
        existing = backend.read(library, symbol, date_range=(first, last))
        merged = merge(existing, incoming)
        # Match the existing static physical schema exactly. Missing fields are
        # null; never discard a populated field or change unrelated metadata.
        aligned = merged.select([
            (pl.col(c).cast(template.schema[c], strict=True) if c in merged.columns
             else pl.lit(None, dtype=template.schema[c])).alias(c)
            for c in ['date', *physical]
        ])
        metadata = lib.read_metadata(symbol).metadata
        data = aligned.to_pandas().set_index('date').sort_index()
        data.index.name = 'date'
        lib.update(symbol, data, date_range=(first, last), metadata=metadata,
                   prune_previous_versions=False)
        return True
