"""Historical corporate-event ingestion from FMP stable endpoints."""

from __future__ import annotations

from collections.abc import Iterable

import requests

from quant_warehouse.ingest.credentials import resolve_fmp_api_key

CORPORATE_EVENT_ENDPOINTS = {
    "symbol_change": "symbol-change",
    "delisted": "delisted-companies",
    "merger_acquisition": "mergers-acquisitions-latest",
}


def fetch_fmp_corporate_events(
    event_types: Iterable[str] = tuple(CORPORATE_EVENT_ENDPOINTS),
    *,
    page_limit: int = 100,
    max_pages: int = 100,
    timeout: int = 60,
) -> list[dict]:
    """Fetch raw symbol-change, delisting, and M&A rows from FMP."""
    api_key = resolve_fmp_api_key(required=True)
    events: list[dict] = []
    session = requests.Session()
    for event_type in event_types:
        key = str(event_type).strip().lower()
        endpoint = CORPORATE_EVENT_ENDPOINTS.get(key)
        if endpoint is None:
            raise ValueError(f"Unsupported FMP corporate event type: {event_type}")
        seen_pages: set[str] = set()
        for page in range(max_pages):
            response = session.get(
                f"https://financialmodelingprep.com/stable/{endpoint}",
                params={"page": page, "limit": page_limit, "apikey": api_key},
                timeout=(10, timeout),
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload = payload.get("historical", payload.get("data", payload.get("results", [])))
            if not isinstance(payload, list) or not payload:
                break
            signature = repr(payload)
            if signature in seen_pages:
                break
            seen_pages.add(signature)
            for row in payload:
                if isinstance(row, dict):
                    item = dict(row)
                    item["corporate_event_type"] = key
                    events.append(item)
            if len(payload) < page_limit:
                break
    return events
