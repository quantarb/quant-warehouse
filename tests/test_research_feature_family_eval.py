from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
import math

import polars as pl

from quant_warehouse.research_tools.feature_family_eval import (
    FamilyEvaluationConfig,
    _add_cross_symbol_context_features,
    _add_macro_context_features,
    _add_time_calendar_features,
    _is_supported_equity_record,
    cap_features_by_quality,
    evaluate_feature_families,
)


def test_cap_features_by_quality_limits_each_family_without_targets() -> None:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(4)]
    panel = pl.DataFrame(
        {
            "date": dates * 2,
            "symbol": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "close": [float(value) + 10.0 for value in range(8)],
            "family_a__dense": [1, 2, 3, 4, 2, 4, 6, 8],
                "family_a__sparse": [1.0, math.nan, math.nan, math.nan, 2.0, math.nan, math.nan, math.nan],
            "family_b__dense": [5, 4, 3, 2, 6, 5, 4, 3],
            "forward_return_1d": [0.1, 0.2, 0.3, math.nan, 0.0, 0.1, 0.2, math.nan],
        }
    )
    metadata = pl.DataFrame(
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

    assert set(selected) == {"family_a__dense", "family_b__dense"}
    assert capped_metadata.group_by("family").len()["len"].max() == 1
    assert quality.filter(pl.col("feature") == "family_a__dense").item(0, "selected") is True
    assert quality.filter(pl.col("feature") == "family_a__sparse").item(0, "selected") is False


def test_evaluate_feature_families_returns_family_summaries() -> None:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
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
    panel = pl.DataFrame(rows)
    metadata = pl.DataFrame(
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
    assert best.item(0, "horizon") == 1
    assert set(stable["family"]) == {"family_a", "family_b"}
    assert seconds >= 0.0


def test_supported_equity_record_rejects_pooled_vehicle_payloads() -> None:
    assert _is_supported_equity_record("SPY", {"is_etf": True}) == (False, "asset_class: etf")
    assert _is_supported_equity_record("VFIAX", {"quote_type": "MUTUALFUND"}) == (False, "asset_class: fund")
    assert _is_supported_equity_record("ABALX", {"is_fund": False}) == (False, "asset_class: fund_symbol_pattern")
    assert _is_supported_equity_record("AAPL", {"is_fund": False, "is_etf": False}) == (True, "ok")


def test_context_feature_families_are_added_without_vendor_calls() -> None:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(8)]
    rows = []
    for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC"]):
        for date_idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 100.0 + symbol_idx * 10.0 + date_idx,
                    "fmp_daily_mcap_multiple__mcap_to_net_income": 15.0 + symbol_idx,
                    "fmp_daily_mcap_multiple__mcap_to_revenue": 4.0 + symbol_idx,
                }
            )
    panel = pl.DataFrame(rows)

    class FakeWarehouse:
        def read_macro_panel(self, series_codes, *, provider, start=None, end=None):
            index = [datetime(2023, 12, 29) + timedelta(days=i) for i in range(12)]
            data = {
                "GDP": [100.0 + 11.0 * i / (len(index) - 1) for i in range(len(index))],
                "macro__ust_year10": [4.0 + 0.2 * i / (len(index) - 1) for i in range(len(index))],
                "macro__ust_year2": [3.7 + 0.2 * i / (len(index) - 1) for i in range(len(index))],
                "macro__ust_month3": [3.4 + 0.2 * i / (len(index) - 1) for i in range(len(index))],
            }
            return pl.DataFrame({code: data[code] for code in series_codes if code in data}).with_columns(pl.Series("date", index))

        def read_profile(self, symbol, *, provider):
            groups = {
                "AAA": ("Technology", "Software"),
                "BBB": ("Technology", "Hardware"),
                "CCC": ("Energy", "Oil & Gas"),
            }
            sector, industry = groups[str(symbol)]
            return SimpleNamespace(sector=sector, industry=industry)

    specs = []
    specs.extend(_add_time_calendar_features(panel))
    specs.extend(_add_macro_context_features(FakeWarehouse(), panel, FamilyEvaluationConfig()))
    specs.extend(_add_cross_symbol_context_features(FakeWarehouse(), panel, FamilyEvaluationConfig()))

    families = {spec.family for spec in specs}
    assert {
        "time_calendar",
        "economic_indicators",
        "treasury_rates",
        "sector_performance",
        "industry_performance",
        "sector_pe",
        "industry_pe",
    }.issubset(families)
    for family in families:
        feature_cols = [spec.feature for spec in specs if spec.family == family]
        assert feature_cols
        assert all(column in {spec.feature for spec in specs} for column in feature_cols)
