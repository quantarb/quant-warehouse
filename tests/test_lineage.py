from __future__ import annotations

import json

import polars as pl
import pytest

from quant_warehouse.lineage import (
    build_dataset_lineage_manifest,
    dataframe_fingerprint,
    read_dataset_lineage_manifest,
    write_dataset_lineage_manifest,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"symbol": "MSFT", "date": "2024-01-03", "value": 2.0},
            {"symbol": "AAPL", "date": "2024-01-02", "value": 1.0},
        ]
    )


def test_dataframe_fingerprint_is_stable_for_key_sorted_rows():
    frame = _frame()
    assert dataframe_fingerprint(frame, key_columns=("symbol", "date")) == dataframe_fingerprint(
        frame.reverse(), key_columns=("symbol", "date")
    )


def test_dataframe_fingerprint_changes_with_content():
    changed = _frame()
    changed = changed.with_row_index("_row").with_columns(pl.when(pl.col("_row") == 0).then(3.0).otherwise(pl.col("value")).alias("value")).drop("_row")
    assert dataframe_fingerprint(_frame()) != dataframe_fingerprint(changed)


def test_dataset_lineage_round_trip_and_immutable_write(tmp_path):
    manifest = build_dataset_lineage_manifest(
        _frame(),
        dataset_id="features-2024",
        dataset_kind="feature_panel",
        provider="fmp",
        available_at_cutoff="2024-01-04",
        recipe_id="feature-recipe-v1",
        recipe={"families": ["quality", "value"]},
        source_references={"prices": "arctic://prices/fmp"},
    )
    path = write_dataset_lineage_manifest(manifest, tmp_path / "lineage.json")

    loaded = read_dataset_lineage_manifest(path)
    assert loaded["lineage_fingerprint"] == manifest.lineage_fingerprint
    assert loaded["symbols"] == ["AAPL", "MSFT"]
    assert loaded["row_count"] == 2
    assert write_dataset_lineage_manifest(manifest, path) == path

    tampered = json.loads(path.read_text())
    tampered["row_count"] = 3
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        read_dataset_lineage_manifest(path)


def test_dataset_lineage_rejects_data_after_cutoff():
    with pytest.raises(ValueError, match="after available_at_cutoff"):
        build_dataset_lineage_manifest(
            _frame(),
            dataset_id="future",
            dataset_kind="features",
            provider="fmp",
            available_at_cutoff="2024-01-01",
            recipe_id="v1",
            recipe={},
        )
