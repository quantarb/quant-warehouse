"""Endpoint-backed target-family contracts.

Target families preserve the source endpoint and its native columns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetFamilySpec:
    """Stable contract for one endpoint-backed target family."""

    name: str
    event_family: str
    source_endpoint: str
    label_mode: str = "sparse_event_presence"


INSIDER_TRADING_TARGET_FAMILY = TargetFamilySpec(
    name="equity.ownership.insider_trading",
    event_family="insider",
    source_endpoint="equity.ownership.insider_trading",
)

EARNINGS_REPORT_TARGET_FAMILY = TargetFamilySpec(
    name="equity.calendar.earnings",
    event_family="earnings",
    source_endpoint="equity.calendar.earnings",
)

ANALYST_RATING_TARGET_FAMILY = TargetFamilySpec(
    name="equity.estimates.price_target",
    event_family="analyst_rating",
    source_endpoint="equity.estimates.price_target",
    label_mode="sparse_endpoint_records",
)

STOCK_GRADES_TARGET_FAMILY = TargetFamilySpec(
    name="fmp.grades_historical",
    event_family="stock_grade",
    source_endpoint="fmp.grades_historical",
    label_mode="sparse_endpoint_records",
)

GOVERNMENT_TRADING_TARGET_FAMILY = TargetFamilySpec(
    name="equity.ownership.government_trades",
    event_family="congress",
    source_endpoint="equity.ownership.government_trades",
    label_mode="sparse_endpoint_records",
)

PRICE_TARGET_TARGET_FAMILY = ANALYST_RATING_TARGET_FAMILY

DIVIDEND_TARGET_FAMILY = TargetFamilySpec(
    name="equity.calendar.dividend",
    event_family="dividend",
    source_endpoint="equity.calendar.dividend",
    label_mode="sparse_endpoint_records",
)

SPLIT_TARGET_FAMILY = TargetFamilySpec(
    name="equity.calendar.splits",
    event_family="split",
    source_endpoint="equity.calendar.splits",
    label_mode="sparse_endpoint_records",
)

FILING_TARGET_FAMILY = TargetFamilySpec(
    name="equity.fundamental.filings",
    event_family="filing",
    source_endpoint="equity.fundamental.filings",
    label_mode="sparse_endpoint_records",
)

TARGET_FAMILY_REGISTRY: dict[str, TargetFamilySpec] = {
    INSIDER_TRADING_TARGET_FAMILY.name: INSIDER_TRADING_TARGET_FAMILY,
    EARNINGS_REPORT_TARGET_FAMILY.name: EARNINGS_REPORT_TARGET_FAMILY,
    ANALYST_RATING_TARGET_FAMILY.name: ANALYST_RATING_TARGET_FAMILY,
    STOCK_GRADES_TARGET_FAMILY.name: STOCK_GRADES_TARGET_FAMILY,
    GOVERNMENT_TRADING_TARGET_FAMILY.name: GOVERNMENT_TRADING_TARGET_FAMILY,
    DIVIDEND_TARGET_FAMILY.name: DIVIDEND_TARGET_FAMILY,
    SPLIT_TARGET_FAMILY.name: SPLIT_TARGET_FAMILY,
    FILING_TARGET_FAMILY.name: FILING_TARGET_FAMILY,
}

def get_target_family(name: str) -> TargetFamilySpec:
    """Return a registered target family by its exact OpenBB route name."""

    key = str(name).strip().lower()
    try:
        return TARGET_FAMILY_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported target family: {name!r}") from exc


__all__ = [
    "ANALYST_RATING_TARGET_FAMILY",
    "DIVIDEND_TARGET_FAMILY",
    "EARNINGS_REPORT_TARGET_FAMILY",
    "FILING_TARGET_FAMILY",
    "GOVERNMENT_TRADING_TARGET_FAMILY",
    "INSIDER_TRADING_TARGET_FAMILY",
    "PRICE_TARGET_TARGET_FAMILY",
    "SPLIT_TARGET_FAMILY",
    "STOCK_GRADES_TARGET_FAMILY",
    "TARGET_FAMILY_REGISTRY",
    "TargetFamilySpec",
    "get_target_family",
]
