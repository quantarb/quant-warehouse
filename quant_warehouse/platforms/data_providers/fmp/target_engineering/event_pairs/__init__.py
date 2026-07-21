"""Exact mirrored event-pair target utilities."""

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_pair_normalizer import normalize_event_pairs, normalize_events
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_pair_schema import EVENT_COLUMNS, EVENT_PAIR_COLUMNS
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_registry import (
    EVENT_FAMILIES,
    EVENT_FAMILY_TYPES,
    EVENT_REGISTRY,
    EVENT_TYPES,
    event_types_for_families,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.fetch import (
    fetch_fmp_event_pair_family,
    fetch_fmp_event_pairs,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.store import (
    EVENT_PAIR_LIBRARY,
    EVENT_PAIR_SECTION,
    EventPairLoadResult,
    EventPairStore,
    build_event_pairs_from_historical_data,
)

EventStore = EventPairStore
build_events_from_historical_data = build_event_pairs_from_historical_data

__all__ = [
    "EVENT_PAIR_COLUMNS",
    "EVENT_PAIR_LIBRARY",
    "EVENT_PAIR_SECTION",
    "EVENT_COLUMNS",
    "EVENT_FAMILIES",
    "EVENT_FAMILY_TYPES",
    "EVENT_REGISTRY",
    "EVENT_TYPES",
    "EventPairLoadResult",
    "EventPairStore",
    "EventStore",
    "build_event_pairs_from_historical_data",
    "build_events_from_historical_data",
    "fetch_fmp_event_pair_family",
    "fetch_fmp_event_pairs",
    "event_types_for_families",
    "normalize_events",
    "normalize_event_pairs",
]
