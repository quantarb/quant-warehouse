import pandas as pd

from quant_warehouse.feature_engineering.broadcast import broadcast_asof_to_target_index


def test_broadcast_asof_to_target_index():
    sparse = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "revenue": [100.0, 120.0],
        }
    )
    target = pd.to_datetime(["2024-01-15", "2024-02-01", "2024-05-01"])
    out = broadcast_asof_to_target_index(sparse_df=sparse, target_index=target, on="date", by=None)
    assert len(out) == 3
    assert out.loc["2024-01-15", "revenue"] == 100.0
    assert out.loc["2024-05-01", "revenue"] == 120.0
    assert pd.isna(out.loc["2024-02-01", "revenue"]) is False
