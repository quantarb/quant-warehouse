from datetime import datetime, timedelta

import polars as pl

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    HitsLabelSpec,
    build_hits_labels,
    build_hold_timing_hits_labels,
    build_return_and_speed_hits_labels,
)


def _prices() -> pl.DataFrame:
    dates = pl.Series("date", [datetime(2024, 1, 1) + timedelta(days=i) for i in range(8)])
    return pl.DataFrame(
        {
            "date": dates,
            "high": [10, 12, 9, 14, 8, 13, 7, 15],
            "low": [9, 11, 8, 13, 7, 12, 6, 14],
        }
    )


def test_build_hits_labels_returns_independent_long_short_tails() -> None:
    labels = build_hits_labels({"A": _prices()}, spec=HitsLabelSpec(max_hold=4, iterations=5))

    assert len(labels) == 8
    assert set(labels["symbol"]) == {"A"}
    assert all(labels.select(pl.col(c).ge(0).all()).item() for c in ["long_hub", "long_authority", "short_hub", "short_authority"])
    assert all(labels.schema[c] == pl.Boolean for c in ["long_hub_tail", "long_authority_tail", "short_hub_tail", "short_authority_tail"])
    assert all(labels[c].any() for c in ["long_hub_tail", "long_authority_tail", "short_hub_tail", "short_authority_tail"])


def test_build_hits_labels_respects_year_graph_boundaries() -> None:
    first = _prices()
    second = _prices().with_columns(pl.Series("date", [datetime(2025, 1, 1) + timedelta(days=i) for i in range(8)]))
    labels = build_hits_labels({"A": pl.concat([first, second])})

    assert len(labels) == 16
    assert labels.group_by(pl.col("date").dt.year()).len().sort("date").to_dicts() == [{"date": 2024, "len": 8}, {"date": 2025, "len": 8}]


def test_build_hold_timing_hits_labels_adds_independent_horizons() -> None:
    labels = build_hold_timing_hits_labels({"A": _prices()}, hold_days=(2, 4))

    assert {"long_hub_2d", "long_authority_2d", "short_hub_4d", "short_authority_4d"}.issubset(labels.columns)
    assert not labels.select(pl.any_horizontal([pl.col("long_hub_2d").is_null(), pl.col("long_hub_4d").is_null()])).to_series().any()


def test_build_return_and_speed_hits_labels_shares_topology() -> None:
    labels = build_return_and_speed_hits_labels({"A": _prices()}, spec=HitsLabelSpec(max_hold=4, iterations=5))

    assert {"long_hub", "short_authority", "speed_long_hub", "speed_short_authority"}.issubset(labels.columns)
    assert len(labels) == 8
