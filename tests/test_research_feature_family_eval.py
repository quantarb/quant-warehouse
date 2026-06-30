from __future__ import annotations

import numpy as np
import pandas as pd

from quant_warehouse.research_tools.feature_family_eval import (
    _is_supported_equity_record,
    cap_features_by_quality,
    evaluate_feature_families,
)


def test_cap_features_by_quality_limits_each_family_without_targets() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    panel = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "symbol": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "close": np.arange(8, dtype=float) + 10.0,
            "family_a__dense": [1, 2, 3, 4, 2, 4, 6, 8],
            "family_a__sparse": [1, np.nan, np.nan, np.nan, 2, np.nan, np.nan, np.nan],
            "family_b__dense": [5, 4, 3, 2, 6, 5, 4, 3],
            "forward_return_1d": [0.1, 0.2, 0.3, np.nan, 0.0, 0.1, 0.2, np.nan],
        }
    )
    metadata = pd.DataFrame(
        [
            {
                "feature": "family_a__dense",
                "family": "family_a",
                "source": "unit",
                "source_column": "dense",
                "expected_direction": "higher_is_better",
            },
            {
                "feature": "family_a__sparse",
                "family": "family_a",
                "source": "unit",
                "source_column": "sparse",
                "expected_direction": "higher_is_better",
            },
            {
                "feature": "family_b__dense",
                "family": "family_b",
                "source": "unit",
                "source_column": "dense",
                "expected_direction": "lower_is_better",
            },
        ]
    )

    selected, capped_metadata, quality = cap_features_by_quality(panel, metadata, max_features=1)

    assert selected == ["family_a__dense", "family_b__dense"]
    assert capped_metadata.groupby("family").size().max() == 1
    assert quality.loc[quality["feature"].eq("family_a__dense"), "selected"].item() is True
    assert quality.loc[quality["feature"].eq("family_a__sparse"), "selected"].item() is False


def test_evaluate_feature_families_returns_family_summaries() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(["A", "B", "C"]):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 10.0 + date_idx + symbol_idx,
                    "family_a__feature": float(symbol_idx),
                    "family_b__feature": float(2 - symbol_idx),
                    "forward_return_1d": float(symbol_idx) / 100.0,
                }
            )
    panel = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        [
            {
                "feature": "family_a__feature",
                "family": "family_a",
                "source": "unit",
                "source_column": "feature",
                "expected_direction": "higher_is_better",
            },
            {
                "feature": "family_b__feature",
                "family": "family_b",
                "source": "unit",
                "source_column": "feature",
                "expected_direction": "lower_is_better",
            },
        ]
    )

    results, summary, best, stable, seconds = evaluate_feature_families(
        panel,
        metadata,
        horizons=(1,),
        min_observations=1,
        include_spreads=False,
    )

    assert len(results) == 2
    assert set(summary["family"]) == {"family_a", "family_b"}
    assert best.loc[0, "horizon"] == 1
    assert set(stable["family"]) == {"family_a", "family_b"}
    assert seconds >= 0.0


def test_supported_equity_record_rejects_pooled_vehicle_payloads() -> None:
    assert _is_supported_equity_record("SPY", {"is_etf": True}) == (False, "asset_class: etf")
    assert _is_supported_equity_record("VFIAX", {"quote_type": "MUTUALFUND"}) == (False, "asset_class: fund")
    assert _is_supported_equity_record("ABALX", {"is_fund": False}) == (False, "asset_class: fund_symbol_pattern")
    assert _is_supported_equity_record("AAPL", {"is_fund": False, "is_etf": False}) == (True, "ok")
