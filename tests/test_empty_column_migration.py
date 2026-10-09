from datetime import datetime
import pandas as pd
import polars as pl
import pytest
from arcticdb import Arctic
from quant_warehouse.migrate.empty_columns import (
    prepare_empty_column_cleanup,
    apply_empty_column_cleanup,
)
from quant_warehouse.warehouse.backend import ArcticBackend


def _fixture():
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    f = pd.DataFrame(
        {
            "number": [float("nan"), float("nan")],
            "text": ["", None],
            "value": [0.0, float("inf")],
            "flag": [False, False],
            "space": [" ", ""],
            "key": ["", ""],
            "optional_timestamp": [pd.NaT, pd.NaT],
        },
        index=pd.DatetimeIndex([datetime(2024, 1, 1), datetime(2024, 1, 2)], name="date"),
    )
    lib.write("TEST", f, metadata={"audit": {"source": "fixture"}}, prune_previous_versions=False)
    return backend, lib, f


def test_dry_plan_full_validation_and_version_preserving_apply():
    backend, lib, original = _fixture()
    lib.snapshot("baseline")
    plan = prepare_empty_column_cleanup(lib, "TEST", key_columns=("key",))
    assert plan.physical_removed == ("number", "text", "optional_timestamp")
    assert plan.rows == 2 and len(lib.list_versions("TEST")) == 1
    report = apply_empty_column_cleanup(lib, plan)
    assert report["new_version"] == 1 and report["original_version_retained"]
    assert lib.read_metadata("TEST").metadata["audit"] == {"source": "fixture"}
    assert set(lib.list_snapshots()) == {"baseline"}
    pd.testing.assert_frame_equal(lib.read("TEST", as_of="baseline").data, original)
    assert backend.read("events", "TEST").columns == ["date", "value", "flag", "space", "key"]
    selected = backend.read("events", "TEST", columns=["text", "number", "value"])
    assert selected.height == 2 and selected.columns == ["text", "number", "value"]
    assert selected.schema["text"] == pl.String and selected.schema["number"] == pl.Float64
    assert selected["text"].null_count() == 2
    assert not apply_empty_column_cleanup(
        lib, prepare_empty_column_cleanup(lib, "TEST", key_columns=("key",))
    )["changed"]


def test_logical_placeholder_removal_preserves_keys_and_provenance():
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    frame = pl.DataFrame(
        {"date": [datetime(2024, 1, 1)], "value": [2.0], "blank": [""], "symbol": [None]}
    )
    backend.write("events", "TEST", frame, prune_previous_versions=False)
    plan = prepare_empty_column_cleanup(lib, "TEST")
    assert plan.logical_removed == ("blank",) and plan.physical_removed == ()
    apply_empty_column_cleanup(lib, plan)
    assert backend.read("events", "TEST").columns == ["date", "value", "symbol"]
    assert backend.read("events", "TEST", columns=["blank"]).height == 1
    assert (
        lib.read_metadata("TEST").metadata["quant_warehouse_empty_payload_details"]["blank"][
            "empty_string_count"
        ]
        == 1
    )


def test_zero_row_schema_noop_and_bounded_validation():
    _, lib, _ = _fixture()
    with pytest.raises(ValueError, match="bounded audit limit"):
        prepare_empty_column_cleanup(lib, "TEST", max_rows=1)
    lib.write("EMPTY", pd.DataFrame({"number": pd.Series(dtype="float64")}))
    plan = prepare_empty_column_cleanup(lib, "EMPTY")
    assert not plan.physical_removed and not plan.logical_removed
    assert not apply_empty_column_cleanup(lib, plan)["changed"]


def test_changed_source_and_changed_plan_refused():
    _, lib, original = _fixture()
    plan = prepare_empty_column_cleanup(lib, "TEST")
    lib.write("TEST", original, prune_previous_versions=False)
    with pytest.raises(RuntimeError, match="Source advanced"):
        apply_empty_column_cleanup(lib, plan)
    plan = prepare_empty_column_cleanup(lib, "TEST")
    plan.cleaned.loc[plan.cleaned.index[0], "value"] = 999
    with pytest.raises(RuntimeError, match="Plan changed"):
        apply_empty_column_cleanup(lib, plan)


def test_missing_library_or_symbol_never_created():
    lib = Arctic("mem://").create_library("events")
    with pytest.raises(Exception):
        prepare_empty_column_cleanup(lib, "MISSING")
    assert lib.list_symbols() == []


def test_last_empty_payload_removal_retains_index_and_rows():
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    original = pd.DataFrame(
        {"blank": ["", ""]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="date")
    )
    lib.write("TEST", original, prune_previous_versions=False)
    plan = prepare_empty_column_cleanup(lib, "TEST")
    apply_empty_column_cleanup(lib, plan)
    assert lib.read("TEST").data.shape == (2, 0)
    assert backend.read("events", "TEST").height == 2
    assert backend.read("events", "TEST").columns == ["date"]


def test_migrated_refresh_stays_clean_until_removed_field_has_real_values():
    backend, lib, _ = _fixture()
    apply_empty_column_cleanup(lib, prepare_empty_column_cleanup(lib, "TEST", key_columns=("key",)))
    refreshed = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "value": [1.0, 2.0],
            "text": ["", None],
        }
    )
    backend.write("events", "TEST", refreshed, prune_previous_versions=False)
    assert "text" not in backend.read("events", "TEST").columns
    assert backend.read("events", "TEST", columns=["text"]).height == 2
    reconstructed = backend.read("events", "TEST", columns=["number", "optional_timestamp"])
    assert reconstructed.schema == {"number": pl.Float64, "optional_timestamp": pl.Datetime("ns")}
    assert reconstructed.null_count().row(0) == (2, 2)
    refreshed = refreshed.with_columns(pl.Series("text", ["", "real"]))
    backend.write("events", "TEST", refreshed, prune_previous_versions=False)
    assert backend.read("events", "TEST")["text"].to_list() == ["", "real"]


def test_last_payload_without_temporal_index_is_not_migrated():
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    lib.write("TEST", pd.DataFrame({"blank": ["", ""]}), prune_previous_versions=False)
    with pytest.raises(ValueError, match="row-representation review"):
        prepare_empty_column_cleanup(lib, "TEST")
    assert len(lib.list_versions("TEST")) == 1


def test_logical_only_cleanup_uses_metadata_version(monkeypatch):
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    backend.write(
        "events",
        "TEST",
        pl.DataFrame({"date": [datetime(2024, 1, 1)], "value": [2.0], "blank": [""]}),
        prune_previous_versions=False,
    )
    source_metadata = lib.read_metadata("TEST").metadata
    lib.snapshot("before_logical_cleanup")
    plan = prepare_empty_column_cleanup(lib, "TEST")

    def forbidden(*args, **kwargs):
        raise AssertionError("Logical-only cleanup must not rewrite data")

    monkeypatch.setattr(lib, "write", forbidden)
    result = apply_empty_column_cleanup(lib, plan)
    assert result["new_version"] == 1
    assert lib.read_metadata("TEST", as_of=0).metadata == source_metadata
    assert lib.read("TEST", as_of="before_logical_cleanup").metadata == source_metadata
    assert set(lib.list_snapshots()) == {"before_logical_cleanup"}
    assert "blank" not in backend.read("events", "TEST").columns


def test_migrated_unnamed_time_index_remains_visible():
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    frame = pd.DataFrame(
        {"blank": ["", ""], "value": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
    )
    lib.write("TEST", frame, prune_previous_versions=False)
    apply_empty_column_cleanup(lib, prepare_empty_column_cleanup(lib, "TEST"))
    out = backend.read("events", "TEST")
    assert out.columns == ["date", "value"]
    assert out["date"].to_list() == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    pd.testing.assert_frame_equal(lib.read("TEST", as_of=0).data, frame)


def test_existing_clean_schema_missing_unnamed_index_gets_metadata_repair():
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    original = pd.DataFrame(
        {"value": [1.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    )
    metadata = {
        "quant_warehouse_empty_column_cleanup": {"source_version": 0},
        "quant_warehouse_column_order": ["value"],
        "audit": "retained",
    }
    lib.write("TEST", original, metadata=metadata, prune_previous_versions=False)
    plan = prepare_empty_column_cleanup(lib, "TEST")
    assert plan.index_order_repair and not plan.physical_removed and not plan.logical_removed
    result = apply_empty_column_cleanup(lib, plan)
    assert result["changed"] and result["new_version"] == 1
    assert backend.read("events", "TEST").columns == ["date", "value"]
    assert lib.read_metadata("TEST", as_of=0).metadata == metadata
    repaired_metadata = lib.read_metadata("TEST").metadata
    assert repaired_metadata == {**metadata, "quant_warehouse_column_order": ["date", "value"]}


def test_text_columns_with_nan_sentinels_are_inspected_without_data_coercion():
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    frame = pd.DataFrame(
        {
            "mixed_text": [float("nan"), "real"],
            "blank_text": [float("nan"), ""],
            "value": [0.0, 2.0],
        },
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="date"),
    )
    lib.write("TEST", frame, prune_previous_versions=False)
    original = lib.read("TEST").data
    plan = prepare_empty_column_cleanup(lib, "TEST")
    assert plan.physical_removed == ("blank_text",)
    apply_empty_column_cleanup(lib, plan)
    pd.testing.assert_frame_equal(lib.read("TEST").data, original.drop(columns=["blank_text"]))
    pd.testing.assert_frame_equal(lib.read("TEST", as_of=0).data, original)


def test_logical_cleanup_preserves_unnamed_datetime_index_and_explicit_empty_schema():
    backend = ArcticBackend("mem://")
    lib = backend._library("events")
    original = pd.DataFrame(
        {"value": [0.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    )
    metadata = {
        "quant_warehouse_empty_columns": ["blank"],
        "quant_warehouse_column_order": ["blank", "value"],
        "source": {"provider": "fixture"},
    }
    lib.write("TEST", original, metadata=metadata, prune_previous_versions=False)
    lib.snapshot("unnamed_before")
    plan = prepare_empty_column_cleanup(lib, "TEST")
    assert plan.logical_removed == ("blank",) and not plan.physical_removed
    apply_empty_column_cleanup(lib, plan)
    out = backend.read("events", "TEST")
    assert out.columns == ["date", "value"]
    assert out["date"].to_list() == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    selected = backend.read("events", "TEST", columns=["date", "blank", "value"])
    assert selected["blank"].null_count() == 2
    pd.testing.assert_frame_equal(lib.read("TEST").data, original)
    pd.testing.assert_frame_equal(lib.read("TEST", as_of="unnamed_before").data, original)
    assert lib.read_metadata("TEST", as_of=0).metadata == metadata
    assert lib.read_metadata("TEST").metadata["source"] == metadata["source"]
