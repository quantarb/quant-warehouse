import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    MACRO_RESPONSE_CLASSES,
    MacroEventSpec,
    build_macro_event_label_panel,
    build_macro_event_targets,
    build_macro_response_labels,
    normalize_macro_events,
)


def test_normalize_macro_events_preserves_release_and_surprise_fields():
    events = normalize_macro_events(
        pd.DataFrame(
            {
                "date": ["2025-01-01 08:30:00"],
                "country": ["us"],
                "event": ["CPI YoY"],
                "estimate": [2.0],
                "actual": [2.5],
                "impact": ["High"],
                "unit": ["%"],
            }
        )
    )

    row = events.iloc[0]
    assert row.event_type == "cpi_yoy"
    assert row.country == "US"
    assert row.surprise == 0.5
    assert row.surprise_pct == 0.25


def test_macro_response_classes_are_shared_across_event_types():
    dates = pd.date_range("2025-01-01", periods=2)
    prices = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"] * 2,
            "date": [dates[0]] * 3 + [dates[1]] * 3,
            "close": [100.0, 100.0, 100.0, 110.0, 90.0, 100.0],
        }
    )
    events = pd.DataFrame(
        {
            "date": [dates[0]],
            "country": ["US"],
            "event": ["CPI YoY"],
            "estimate": [2.0],
            "actual": [2.5],
        }
    )
    labels = build_macro_response_labels(
        prices,
        events,
        MacroEventSpec(horizons=(1,), minimum_cross_section=1),
    )

    assert set(labels.response_class).issubset(range(len(MACRO_RESPONSE_CLASSES)))
    assert set(labels.symbol) == {"A", "B", "C"}
    assert labels.macro_event_id.nunique() == 1


def test_fed_rate_direction_becomes_independent_binary_event_targets():
    events = pd.DataFrame(
        {
            "date": ["2025-01-29", "2025-03-19", "2025-05-07"],
            "country": ["US"] * 3,
            "event": ["Interest Rate Decision"] * 3,
            "previous": [4.50, 4.25, 4.25],
            "actual": [4.25, 4.25, 4.25],
        }
    )
    targets = build_macro_event_targets(events)
    assert targets.target_name.tolist() == ["fed_rate_cut", "fed_rate_hold", "fed_rate_hold"]
    tokens = pd.DataFrame({"symbol": ["A", "B", "A"], "date": ["2025-01-29", "2025-01-29", "2025-01-30"]})
    panel = build_macro_event_label_panel(tokens, events)
    assert panel.loc[panel.date == "2025-01-29", "is_fed_rate_cut"].eq(1).all()
    assert panel.loc[panel.date == "2025-01-30", "is_fed_rate_cut"].eq(0).all()


def test_recurring_month_suffixes_share_one_macro_target():
    events = pd.DataFrame(
        {
            "date": ["2025-01-15", "2025-02-15"],
            "country": ["US", "US"],
            "event": ["CPI YoY (Dec)", "CPI YoY (Jan)"],
            "actual": [2.5, 2.6],
            "previous": [2.4, 2.5],
        }
    )
    targets = build_macro_event_targets(events)
    assert targets.target_name.tolist() == ["macro_us_cpi_yoy_increase", "macro_us_cpi_yoy_increase"]
