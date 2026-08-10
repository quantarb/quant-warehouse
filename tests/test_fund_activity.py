import polars as pl

from quant_warehouse.research_tools.fund_activity import (
    build_fund_holding_activity_events,
    build_holder_activity_events,
    build_institutional_activity_events,
)


def test_fund_holding_activity_emits_buy_add_reduce_and_exit():
    holdings = pl.DataFrame(
        [
            {"fund_symbol": "F1", "symbol": "AAA", "date": "2024-01-01", "shares": 10},
            {"fund_symbol": "F1", "symbol": "AAA", "date": "2024-04-01", "shares": 15},
            {"fund_symbol": "F1", "symbol": "AAA", "date": "2024-07-01", "shares": 5},
            {"fund_symbol": "F1", "symbol": "AAA", "date": "2024-10-01", "shares": 0},
        ]
    )
    events = build_fund_holding_activity_events(holdings, fund_type="etf")
    assert set(events["target_family"]) == {
        "fund_activity.etf_buy",
        "fund_activity.add",
        "fund_activity.reduce",
        "fund_activity.exit",
    }


def test_institutional_activity_uses_aggregate_position_counts():
    summary = pl.DataFrame(
        [{
            "symbol": "AAA",
            "date": "2024-06-30",
            "new_positions": 2,
            "increased_positions": 3,
            "reduced_positions": 1,
            "closed_positions": 0,
        }]
    )
    events = build_institutional_activity_events(summary)
    assert set(events["target_family"]) == {
        "fund_activity.institutional_buy",
        "fund_activity.add",
        "fund_activity.reduce",
    }


def test_holder_activity_preserves_holder_identity_as_event_metadata():
    analytics = pl.DataFrame([{
        "symbol": "AAA", "date": "2024-09-30", "cik": "0001234567",
        "investorName": "Example Manager", "isNew": True,
        "isSoldOut": False, "changeInSharesNumber": 100,
    }])
    events = build_holder_activity_events(analytics)
    assert set(events["target_family"]) == {"holder_activity.buy"}
    assert events[0, "holder_id"] == "0001234567"
    assert events[0, "holder_name"] == "Example Manager"
