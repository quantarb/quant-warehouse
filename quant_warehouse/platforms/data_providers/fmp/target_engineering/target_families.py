"""Endpoint-backed target-family contracts.

Target families preserve the source endpoint and the event vocabulary that
produced a label.  They are intentionally small: this module describes the
family contract while the existing event-pair store performs loading and
normalization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetFamilySpec:
    """Stable contract for one endpoint-backed target family."""

    name: str
    event_family: str
    source_endpoint: str
    event_types: tuple[str, ...]
    label_mode: str = "sparse_event_presence"

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(f"target_event_on__{event_type}" for event_type in self.event_types)


INSIDER_TRADING_TARGET_FAMILY = TargetFamilySpec(
    name="equity.ownership.insider_trading",
    event_family="insider",
    source_endpoint="equity.ownership.insider_trading",
    event_types=("insider_buy", "insider_sell"),
)

EARNINGS_REPORT_TARGET_FAMILY = TargetFamilySpec(
    name="equity.calendar.earnings",
    event_family="earnings",
    source_endpoint="equity.calendar.earnings",
    event_types=("earnings_reported", "eps_beat", "eps_miss", "revenue_beat", "revenue_miss"),
)

ANALYST_RATING_TARGET_FAMILY = TargetFamilySpec(
    name="equity.estimates.price_target",
    event_family="analyst_rating",
    source_endpoint="equity.estimates.price_target",
    event_types=("analyst_upgrade", "analyst_downgrade", "price_target_raise", "price_target_cut"),
    label_mode="sparse_endpoint_records",
)

GOVERNMENT_TRADING_TARGET_FAMILY = TargetFamilySpec(
    name="equity.ownership.government_trades",
    event_family="congress",
    source_endpoint="equity.ownership.government_trades",
    event_types=("congressman_buy", "congressman_sell", "senator_buy", "senator_sell"),
    label_mode="sparse_endpoint_records",
)

ANALYST_ESTIMATE_TARGET_FAMILY = TargetFamilySpec(
    name="equity.estimates.historical",
    event_family="analyst_estimate",
    source_endpoint="equity.estimates.historical",
    event_types=("analyst_estimate_raise", "analyst_estimate_cut"),
    label_mode="sparse_endpoint_records",
)

PRICE_TARGET_TARGET_FAMILY = ANALYST_RATING_TARGET_FAMILY

INSTITUTIONAL_OWNERSHIP_TARGET_FAMILY = TargetFamilySpec(
    name="equity.ownership.institutional",
    event_family="institutional",
    source_endpoint="equity.ownership.institutional",
    event_types=(
        "institutional_new_position",
        "institutional_position_increased",
        "institutional_position_reduced",
        "institutional_position_closed",
    ),
    label_mode="sparse_endpoint_records",
)

CAPITAL_ACTION_TARGET_FAMILY = TargetFamilySpec(
    name="equity.fundamental.cash",
    event_family="capital_action",
    source_endpoint="equity.fundamental.cash",
    event_types=("buyback_authorization", "equity_offering"),
    label_mode="sparse_endpoint_records",
)

DIVIDEND_TARGET_FAMILY = TargetFamilySpec(
    name="equity.calendar.dividend",
    event_family="dividend",
    source_endpoint="equity.calendar.dividend",
    event_types=(
        "dividend_declared",
        "dividend_ex_date",
        "dividend_record_date",
        "dividend_payment_date",
        "dividend_increase",
        "dividend_cut",
    ),
    label_mode="sparse_endpoint_records",
)

SPLIT_TARGET_FAMILY = TargetFamilySpec(
    name="equity.calendar.splits",
    event_family="split",
    source_endpoint="equity.calendar.splits",
    event_types=("forward_split", "reverse_split"),
    label_mode="sparse_endpoint_records",
)

PROFILE_TARGET_FAMILY = TargetFamilySpec(
    name="equity.profile",
    event_family="profile",
    source_endpoint="equity.profile",
    event_types=("ipo_trading_started",),
    label_mode="sparse_endpoint_records",
)

FILING_TARGET_FAMILY = TargetFamilySpec(
    name="equity.fundamental.filings",
    event_family="filing",
    source_endpoint="equity.fundamental.filings",
    event_types=("sec_8k_filed", "sec_10q_filed", "sec_10k_filed", "sec_form4_filed"),
    label_mode="sparse_endpoint_records",
)

TARGET_FAMILY_REGISTRY: dict[str, TargetFamilySpec] = {
    INSIDER_TRADING_TARGET_FAMILY.name: INSIDER_TRADING_TARGET_FAMILY,
    EARNINGS_REPORT_TARGET_FAMILY.name: EARNINGS_REPORT_TARGET_FAMILY,
    ANALYST_RATING_TARGET_FAMILY.name: ANALYST_RATING_TARGET_FAMILY,
    GOVERNMENT_TRADING_TARGET_FAMILY.name: GOVERNMENT_TRADING_TARGET_FAMILY,
    ANALYST_ESTIMATE_TARGET_FAMILY.name: ANALYST_ESTIMATE_TARGET_FAMILY,
    INSTITUTIONAL_OWNERSHIP_TARGET_FAMILY.name: INSTITUTIONAL_OWNERSHIP_TARGET_FAMILY,
    CAPITAL_ACTION_TARGET_FAMILY.name: CAPITAL_ACTION_TARGET_FAMILY,
    DIVIDEND_TARGET_FAMILY.name: DIVIDEND_TARGET_FAMILY,
    SPLIT_TARGET_FAMILY.name: SPLIT_TARGET_FAMILY,
    PROFILE_TARGET_FAMILY.name: PROFILE_TARGET_FAMILY,
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
    "ANALYST_ESTIMATE_TARGET_FAMILY",
    "ANALYST_RATING_TARGET_FAMILY",
    "CAPITAL_ACTION_TARGET_FAMILY",
    "DIVIDEND_TARGET_FAMILY",
    "EARNINGS_REPORT_TARGET_FAMILY",
    "FILING_TARGET_FAMILY",
    "GOVERNMENT_TRADING_TARGET_FAMILY",
    "INSTITUTIONAL_OWNERSHIP_TARGET_FAMILY",
    "INSIDER_TRADING_TARGET_FAMILY",
    "PRICE_TARGET_TARGET_FAMILY",
    "PROFILE_TARGET_FAMILY",
    "SPLIT_TARGET_FAMILY",
    "TARGET_FAMILY_REGISTRY",
    "TargetFamilySpec",
    "get_target_family",
]
