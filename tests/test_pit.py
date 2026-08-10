import polars as pl

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.broadcast import broadcast_asof_to_target_index


def test_broadcast_asof_to_target_index():
    sparse = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-04-01"],
            "revenue": [100.0, 120.0],
        }
    )
    target = pl.DataFrame({"date": ["2024-01-15", "2024-02-01", "2024-05-01"]})
    out = broadcast_asof_to_target_index(sparse_df=sparse, target_index=target, on="date", by=None)
    assert out.height == 3
    assert out[0, "revenue"] == 100.0
    assert out[2, "revenue"] == 120.0
    assert out[1, "revenue"] == 100.0
