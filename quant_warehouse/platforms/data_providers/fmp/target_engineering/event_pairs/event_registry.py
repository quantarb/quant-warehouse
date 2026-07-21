"""Independent FMP event definitions.

Events are observations, not mirrored sides.  For example, ``senator_buy``
and ``senator_sell`` are two independent binary labels and neither is the
complement of the other.  The registry deliberately contains no polarity,
mirror, or pair semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventDefinition:
    """Metadata needed to build one independent binary event target."""

    name: str
    family: str
    description: str


_DEFINITIONS = (
    # Government trades.
    ("congressman_buy", "congress", "House purchase transaction"),
    ("congressman_sell", "congress", "House sale transaction"),
    ("senator_buy", "congress", "Senate purchase transaction"),
    ("senator_sell", "congress", "Senate sale transaction"),
    # Insider trades.
    ("insider_buy", "insider", "Insider acquisition or purchase"),
    ("insider_sell", "insider", "Insider disposition or sale"),
    # Earnings and revisions.
    ("earnings_reported", "earnings", "Earnings report was published"),
    ("eps_beat", "earnings", "Actual EPS exceeded consensus EPS"),
    ("eps_miss", "earnings", "Actual EPS was below consensus EPS"),
    ("revenue_beat", "earnings", "Actual revenue exceeded consensus revenue"),
    ("revenue_miss", "earnings", "Actual revenue was below consensus revenue"),
    ("analyst_estimate_raise", "analyst_estimate", "Dated analyst estimate increased"),
    ("analyst_estimate_cut", "analyst_estimate", "Dated analyst estimate decreased"),
    # Analyst actions.
    ("analyst_upgrade", "analyst_rating", "Analyst rating upgraded"),
    ("analyst_downgrade", "analyst_rating", "Analyst rating downgraded"),
    ("price_target_raise", "price_target", "Analyst price target increased"),
    ("price_target_cut", "price_target", "Analyst price target decreased"),
    # Ownership.
    ("institutional_new_position", "institutional", "Institution reported a new position"),
    ("institutional_position_increased", "institutional", "Institution increased a position"),
    ("institutional_position_reduced", "institutional", "Institution reduced a position"),
    ("institutional_position_closed", "institutional", "Institution closed a position"),
    # Corporate actions.
    ("buyback_authorization", "capital_action", "Issuer authorized a buyback"),
    ("equity_offering", "capital_action", "Issuer announced an equity offering"),
    ("dividend_declared", "dividend", "Dividend declaration date"),
    ("dividend_ex_date", "dividend", "Dividend ex-date"),
    ("dividend_record_date", "dividend", "Dividend record date"),
    ("dividend_payment_date", "dividend", "Dividend payment date"),
    ("dividend_increase", "dividend", "Dividend amount increased"),
    ("dividend_cut", "dividend", "Dividend amount decreased"),
    ("forward_split", "split", "Forward stock split"),
    ("reverse_split", "split", "Reverse stock split"),
    ("ipo_trading_started", "profile", "FMP profile first_stock_price_date"),
    # Filings.
    ("sec_8k_filed", "filing", "SEC 8-K filing"),
    ("sec_10q_filed", "filing", "SEC 10-Q filing"),
    ("sec_10k_filed", "filing", "SEC 10-K filing"),
    ("sec_form4_filed", "filing", "SEC Form 4 filing"),
)

EVENT_REGISTRY = {
    name: EventDefinition(name=name, family=family, description=description)
    for name, family, description in _DEFINITIONS
}

EVENT_FAMILY_TYPES: dict[str, tuple[str, ...]] = {}
for _definition in EVENT_REGISTRY.values():
    EVENT_FAMILY_TYPES.setdefault(_definition.family, ())
    EVENT_FAMILY_TYPES[_definition.family] += (_definition.name,)

EVENT_TYPES = tuple(EVENT_REGISTRY)
EVENT_FAMILIES = tuple(EVENT_FAMILY_TYPES)


def event_types_for_families(families: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return registered event labels for the requested source families."""

    unknown = sorted(set(families) - set(EVENT_FAMILY_TYPES))
    if unknown:
        raise ValueError(f"Unsupported event families: {unknown}")
    return tuple(
        event_type
        for family in families
        for event_type in EVENT_FAMILY_TYPES[family]
    )
