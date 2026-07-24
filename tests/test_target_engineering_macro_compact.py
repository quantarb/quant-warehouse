from __future__ import annotations

import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering.macro_events import (
    build_macro_event_label_panel,
    deduplicate_binary_label_columns,
)


def test_compact_macro_directional_labels_preserve_direction_and_fed_classes() -> None:
    tokens = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"])})
    events = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"] * 4),
            "country": ["US"] * 4,
            "event": [
                "10-Year Note Auction",
                "30-Year Bond Auction",
                "Retail Sales Ex Autos",
                "Federal Funds Rate Decision",
            ],
            "previous": [4.1, 4.2, 100.0, 5.0],
            "actual": [4.2, 4.3, 101.0, 4.75],
        }
    )

    panel = build_macro_event_label_panel(
        tokens, events, directional_only=True, compact_directional=True
    )

    assert panel.loc[0, "is_macro_us_treasury_auction_increase"] == 1.0
    assert panel.loc[0, "is_macro_us_consumer_spending_increase"] == 1.0
    assert panel.loc[0, "is_fed_rate_cut"] == 1.0
    assert not any("10_year" in column or "30_year" in column for column in panel.columns)


def test_macro_event_release_time_joins_to_token_calendar_date() -> None:
    tokens = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"])})
    events = pd.DataFrame(
        {
            "date": ["2025-01-02 20:30:00"],
            "country": ["US"],
            "event": ["CPI"],
            "previous": [3.0],
            "actual": [3.2],
        }
    )

    panel = build_macro_event_label_panel(tokens, events, directional_only=True)

    assert panel.loc[0, "is_macro_us_cpi_increase"] == 1.0


def test_deduplicate_binary_label_columns_keeps_first_identical_vector() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "is_a": [1.0, 0.0],
            "is_b": [1.0, 0.0],
            "is_c": [0.0, 1.0],
        }
    )
    reduced, mapping = deduplicate_binary_label_columns(panel)
    assert list(reduced.columns) == ["date", "is_a", "is_c"]
    assert mapping == {"is_b": "is_a"}
