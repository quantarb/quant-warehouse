import pandas as pd

from quant_warehouse.warehouse.merge import merge_upsert


def test_merge_upsert_appends_new_rows():
    existing = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2024-01-01"]))
    incoming = pd.DataFrame({"close": [101.0]}, index=pd.to_datetime(["2024-01-02"]))
    merged = merge_upsert(existing, incoming)
    assert len(merged) == 2
    assert merged.loc["2024-01-02", "close"] == 101.0


def test_merge_upsert_overwrites_duplicate_index():
    existing = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2024-01-01"]))
    incoming = pd.DataFrame({"close": [99.0]}, index=pd.to_datetime(["2024-01-01"]))
    merged = merge_upsert(existing, incoming)
    assert len(merged) == 1
    assert merged.loc["2024-01-01", "close"] == 99.0