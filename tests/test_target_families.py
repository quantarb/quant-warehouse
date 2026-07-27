from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    ANALYST_RATING_TARGET_FAMILY,
    EARNINGS_REPORT_TARGET_FAMILY,
    INSIDER_TRADING_TARGET_FAMILY,
    get_target_family,
)


def test_insider_trading_target_family_preserves_endpoint_and_labels() -> None:
    family = get_target_family("equity.ownership.insider_trading")

    assert family is INSIDER_TRADING_TARGET_FAMILY
    assert family.event_family == "insider"
    assert family.source_endpoint == "equity.ownership.insider_trading"
    assert family.event_types == ("insider_buy", "insider_sell")
    assert family.target_columns == (
        "target_event_on__insider_buy",
        "target_event_on__insider_sell",
    )
    assert family.label_mode == "sparse_event_presence"


def test_unknown_target_family_is_rejected() -> None:
    try:
        get_target_family("not_registered")
    except ValueError as exc:
        assert "not_registered" in str(exc)
    else:
        raise AssertionError("unknown target family should be rejected")


def test_analyst_rating_target_family_preserves_endpoint() -> None:
    family = get_target_family("equity.estimates.price_target")

    assert family is ANALYST_RATING_TARGET_FAMILY
    assert family.source_endpoint == "equity.estimates.price_target"
    assert family.event_types == (
        "analyst_upgrade", "analyst_downgrade", "price_target_raise", "price_target_cut",
    )
    assert family.label_mode == "sparse_endpoint_records"


def test_earnings_report_target_family_preserves_endpoint() -> None:
    family = get_target_family("equity.calendar.earnings")

    assert family is EARNINGS_REPORT_TARGET_FAMILY
    assert family.source_endpoint == "equity.calendar.earnings"
    assert family.target_columns == (
        "target_event_on__earnings_reported",
        "target_event_on__eps_beat",
        "target_event_on__eps_miss",
        "target_event_on__revenue_beat",
        "target_event_on__revenue_miss",
    )
