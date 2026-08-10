from datetime import datetime

import polars as pl

from quant_warehouse.warehouse.merge import merge_upsert


def test_merge_upsert_appends_new_rows():
    existing = pl.DataFrame({"date": [datetime(2024, 1, 1)], "close": [100.0]})
    incoming = pl.DataFrame({"date": [datetime(2024, 1, 2)], "close": [101.0]})
    merged = merge_upsert(existing, incoming)
    assert len(merged) == 2
    assert merged.filter(pl.col("date") == datetime(2024, 1, 2))["close"][0] == 101.0


def test_merge_upsert_overwrites_duplicate_index():
    existing = pl.DataFrame({"date": [datetime(2024, 1, 1)], "close": [100.0]})
    incoming = pl.DataFrame({"date": [datetime(2024, 1, 1)], "close": [99.0]})
    merged = merge_upsert(existing, incoming)
    assert len(merged) == 1
    assert merged["close"][0] == 99.0
