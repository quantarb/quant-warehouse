import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    HitsLabelSpec,
    build_hits_labels,
)


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "high": [10, 12, 9, 14, 8, 13, 7, 15],
            "low": [9, 11, 8, 13, 7, 12, 6, 14],
        }
    )


def test_build_hits_labels_returns_independent_long_short_tails() -> None:
    labels = build_hits_labels({"A": _prices()}, spec=HitsLabelSpec(max_hold=4, iterations=5))

    assert len(labels) == 8
    assert set(labels.symbol) == {"A"}
    assert labels[["long_hub", "long_authority", "short_hub", "short_authority"]].ge(0).all().all()
    assert labels[["long_hub_tail", "long_authority_tail", "short_hub_tail", "short_authority_tail"]].dtypes.eq(bool).all()
    assert labels.long_hub_tail.any()
    assert labels.long_authority_tail.any()
    assert labels.short_hub_tail.any()
    assert labels.short_authority_tail.any()


def test_build_hits_labels_respects_year_graph_boundaries() -> None:
    first = _prices()
    second = _prices().assign(date=pd.date_range("2025-01-01", periods=8, freq="D"))
    labels = build_hits_labels({"A": pd.concat([first, second], ignore_index=True)})

    assert len(labels) == 16
    assert labels.groupby(labels.date.dt.year).size().to_dict() == {2024: 8, 2025: 8}
