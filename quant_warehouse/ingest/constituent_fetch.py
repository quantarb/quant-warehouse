"""FMP historical index-constituent event ingestion."""

from __future__ import annotations

import requests

from quant_warehouse.ingest.credentials import resolve_fmp_api_key

HISTORICAL_CONSTITUENT_ENDPOINTS = {
    "sp500": "historical-sp500-constituent",
    "nasdaq": "historical-nasdaq-constituent",
    "dowjones": "historical-dowjones-constituent",
}


def fetch_index_constituents(
    indexes: tuple[str, ...] = ("sp500", "nasdaq", "dowjones"),
    *,
    timeout: int = 60,
) -> list[dict]:
    """Fetch current FMP index constituents, including sector/sub-sector metadata."""
    api_key = resolve_fmp_api_key(required=True)
    rows: list[dict] = []
    for index_name in indexes:
        endpoint = f"{str(index_name).lower()}-constituent"
        response = requests.get(
            f"https://financialmodelingprep.com/stable/{endpoint}",
            params={"apikey": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("historical", payload.get("data", []))
        if not isinstance(payload, list):
            continue
        for row in payload:
            if isinstance(row, dict):
                item = dict(row)
                item["index"] = str(index_name).lower()
                rows.append(item)
    return rows


def fetch_historical_constituent_events(
    indexes: tuple[str, ...] = ("sp500", "nasdaq", "dowjones"),
    *,
    timeout: int = 60,
) -> list[dict]:
    """Fetch historical constituent changes from FMP's stable endpoints."""
    api_key = resolve_fmp_api_key(required=True)
    events: list[dict] = []
    for index_name in indexes:
        endpoint = HISTORICAL_CONSTITUENT_ENDPOINTS.get(str(index_name).lower())
        if endpoint is None:
            raise ValueError(f"Unsupported historical constituent index: {index_name}")
        response = requests.get(
            f"https://financialmodelingprep.com/stable/{endpoint}",
            params={"apikey": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("historical", payload.get("data", []))
        if not isinstance(payload, list):
            continue
        for row in payload:
            if isinstance(row, dict):
                item = dict(row)
                item["index"] = str(index_name).lower()
                events.append(item)
    return events
