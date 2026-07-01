from __future__ import annotations

import pandas as pd

from quant_warehouse.research_tools import (
    BinaryTargetConfig,
    EventFeatureDatasetConfig,
    add_fmp_event_context_feature_families,
    build_event_context,
    build_event_feature_text_dataset,
    build_identity_text_dataset,
    event_pair_task_specs,
    fmp_event_context_allowed_feature_families_by_task,
)


def test_event_feature_text_dataset_uses_only_actual_event_rows() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "f1": [1.0, 2.0, 3.0],
            "f2": [1.0, None, 3.0],
            "target_event_on__congress_buy": [1, 0, 0],
            "target_event_on__congress_sell": [0, 0, 1],
        }
    )
    metadata = pd.DataFrame(
        [
            {"source": "fmp", "family": "family_a", "feature": "f1"},
            {"source": "fmp", "family": "family_a", "feature": "f2"},
        ]
    )
    specs = event_pair_task_specs(BinaryTargetConfig(event_families=("congress",)), panel.columns)

    result = build_event_feature_text_dataset(
        panel,
        metadata,
        specs,
        config=EventFeatureDatasetConfig(min_feature_coverage=0.5),
    )

    assert len(result.rows) == 2
    assert set(result.rows["date"]) == set(pd.to_datetime(["2024-01-01", "2024-01-03"]))
    assert set(result.rows["label"]) == {"congress_buy", "congress_sell"}
    assert pd.Timestamp("2024-01-02") not in set(result.rows["date"])


def test_event_feature_text_dataset_drops_orphan_feature_family_without_coverage() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "covered_feature": [1.0, 2.0],
            "orphan_feature": [None, None],
            "target_event_on__earnings_beat": [1, 0],
            "target_event_on__earnings_miss": [0, 1],
        }
    )
    metadata = pd.DataFrame(
        [
            {"source": "fmp", "family": "covered", "feature": "covered_feature"},
            {"source": "fmp", "family": "orphan", "feature": "orphan_feature"},
        ]
    )
    specs = event_pair_task_specs(BinaryTargetConfig(event_families=("earnings",)), panel.columns)

    result = build_event_feature_text_dataset(panel, metadata, specs)

    assert set(result.rows["feature_family"]) == {"covered"}
    assert "orphan" not in set(result.rows["feature_family"])


def test_fmp_event_context_feature_families_are_sparse_by_event_family() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": pd.to_datetime(["2024-01-01"]),
            "target_event_on__congress_buy": [1],
            "target_event_on__congress_sell": [0],
            "target_event_on__analyst_upgrade": [1],
            "target_event_on__analyst_downgrade": [0],
        }
    )
    metadata = pd.DataFrame([{"source": "fmp", "family": "issuer", "feature": "issuer_feature"}])
    events = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "event_family": ["congress", "analyst_rating"],
            "event_type": ["congress_buy", "analyst_upgrade"],
            "actor_name": ["Nancy Pelosi", "Example Analyst"],
            "actor_type": ["house", "analyst"],
            "actor_chamber": ["house", None],
            "actor_firm": [None, "Example Firm"],
            "actor_role": [None, "analyst"],
            "transaction_value": [100000.0, None],
            "reported_date": pd.to_datetime(["2024-01-15", "2024-01-01"]),
            "disclosure_lag_days": [14, 0],
        }
    )

    out_panel, out_metadata = add_fmp_event_context_feature_families(panel, metadata, events)

    assert out_panel.loc[0, "fmp_congress_event_context__actor_name"] == "Nancy Pelosi"
    assert out_panel.loc[0, "fmp_analyst_rating_event_context__actor_firm"] == "Example Firm"
    assert ("fmp_congress_event_context" in set(out_metadata["family"]))
    assert ("fmp_analyst_rating_event_context" in set(out_metadata["family"]))


def test_task_family_allowlist_prevents_cross_event_context_mismatch() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": pd.to_datetime(["2024-01-01"]),
            "target_event_on__congress_buy": [1],
            "target_event_on__congress_sell": [0],
            "target_event_on__analyst_upgrade": [1],
            "target_event_on__analyst_downgrade": [0],
            "fmp_congress_event_context__actor_name": ["Nancy Pelosi"],
            "fmp_analyst_rating_event_context__actor_firm": ["Example Firm"],
        }
    )
    metadata = pd.DataFrame(
        [
            {
                "source": "fmp",
                "family": "fmp_congress_event_context",
                "feature": "fmp_congress_event_context__actor_name",
            },
            {
                "source": "fmp",
                "family": "fmp_analyst_rating_event_context",
                "feature": "fmp_analyst_rating_event_context__actor_firm",
            },
        ]
    )
    specs = event_pair_task_specs(BinaryTargetConfig(event_families=("congress", "analyst_rating")), panel.columns)
    allowed = {
        ("fmp", "fmp_congress_event_context"),
        ("fmp", "fmp_analyst_rating_event_context"),
    }
    by_task = fmp_event_context_allowed_feature_families_by_task(specs, allowed)

    result = build_event_feature_text_dataset(
        panel,
        metadata,
        specs,
        config=EventFeatureDatasetConfig(min_feature_coverage=1.0),
        allowed_feature_families=allowed,
        allowed_feature_families_by_task=by_task,
    )

    congress_rows = result.rows.loc[result.rows["target_task"].eq("event_pair__congress")]
    analyst_rows = result.rows.loc[result.rows["target_task"].eq("event_pair__analyst_rating")]
    assert set(congress_rows["feature_family"]) == {"fmp_congress_event_context"}
    assert set(analyst_rows["feature_family"]) == {"fmp_analyst_rating_event_context"}
    assert "Nancy Pelosi" in congress_rows.iloc[0]["text"]
    assert "Example Firm" not in congress_rows.iloc[0]["text"]


def test_identity_dataset_uses_event_context_only() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "event_family": ["congress", "congress"],
            "event_type": ["congress_buy", "congress_sell"],
            "actor_name": ["Jane", "John"],
            "actor_type": ["house", "senate"],
            "actor_chamber": ["house", "senate"],
            "actor_role": [None, None],
            "actor_firm": [None, None],
            "reported_date": pd.to_datetime(["2024-01-10", "2024-01-12"]),
            "disclosure_lag_days": [9, 9],
            "event_side": [1, -1],
        }
    )
    feature_panel = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "feature": [10.0, 20.0, 30.0],
        }
    )
    metadata = pd.DataFrame([{"source": "fmp", "family": "family", "feature": "feature"}])

    context = build_event_context(events, feature_panel, warehouse=None, event_families=("congress",))
    result = build_identity_text_dataset(
        context,
        metadata,
        [("actor_name", "identity__congress_actor", "task_identity_congress_actor")],
        config=EventFeatureDatasetConfig(min_label_rows=1),
    )

    assert len(context) == 2
    assert {"actor_chamber", "actor_role", "actor_firm", "reported_date", "disclosure_lag_days"}.issubset(context.columns)
    assert set(context["actor_chamber"]) == {"house", "senate"}
    assert set(result.rows["label"]) == {"Jane", "John"}
    assert pd.Timestamp("2024-01-02") not in set(result.rows["date"])
