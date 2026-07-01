from __future__ import annotations

EVENT_PAIR_TAXONOMY = {
    "congress": {
        "positive": "congress_buy",
        "negative": "congress_sell",
    },
    "insider": {
        "positive": "insider_buy",
        "negative": "insider_sell",
    },
    "analyst_rating": {
        "positive": "analyst_upgrade",
        "negative": "analyst_downgrade",
    },
    "price_target": {
        "positive": "price_target_raise",
        "negative": "price_target_cut",
    },
    "institutional": {
        "positive": "institutional_add",
        "negative": "institutional_reduce",
    },
    "capital_action": {
        "positive": "buyback_authorization",
        "negative": "equity_offering",
    },
    "dividend": {
        "positive": "dividend_increase",
        "negative": "dividend_cut",
    },
    "split": {
        "positive": "forward_split",
        "negative": "reverse_split",
    },
    "earnings": {
        "positive": "earnings_beat",
        "negative": "earnings_miss",
    },
}


def get_mirror_event_type(event_family: str, event_type: str) -> str:
    """Return the exact opposite event type for a supported event family."""

    pair = _get_pair(event_family)
    event_type = str(event_type).strip().lower()
    if event_type == pair["positive"]:
        return pair["negative"]
    if event_type == pair["negative"]:
        return pair["positive"]
    raise ValueError(f"Unsupported event_type '{event_type}' for event_family '{event_family}'")


def get_event_side(event_family: str, event_type: str) -> int:
    """Return +1 for positive event types and -1 for mirrored negative types."""

    pair = _get_pair(event_family)
    event_type = str(event_type).strip().lower()
    if event_type == pair["positive"]:
        return 1
    if event_type == pair["negative"]:
        return -1
    raise ValueError(f"Unsupported event_type '{event_type}' for event_family '{event_family}'")


def _get_pair(event_family: str) -> dict[str, str]:
    family = str(event_family).strip().lower()
    if family not in EVENT_PAIR_TAXONOMY:
        raise ValueError(f"Unsupported event_family '{event_family}'")
    return EVENT_PAIR_TAXONOMY[family]
