from __future__ import annotations

EVENT_COLUMNS = [
    "symbol",
    "event_date",
    "event_family",
    "event_type",
    "actor_type",
    "actor_name",
    "actor_role",
    "actor_chamber",
    "actor_firm",
    "actor_title",
    "source",
    "strength",
    "transaction_shares",
    "transaction_price",
    "transaction_value",
    "reported_date",
    "disclosure_lag_days",
    "raw_json",
]

# Compatibility name for callers that have not yet migrated their imports.
# The schema itself is generalized and contains no pair fields.
EVENT_PAIR_COLUMNS = EVENT_COLUMNS
