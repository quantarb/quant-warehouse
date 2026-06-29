from __future__ import annotations

from quant_warehouse.catalog.store import CatalogStore
from quant_warehouse.warehouse.prices import PRICES_LIBRARY
from quant_warehouse.warehouse.sections import ETF_PRICES_LIBRARY, FUND_PRICES_LIBRARY
from quant_warehouse.warehouse.storage import provider_library

PROVIDER = "yfinance"


def target_library_for_legacy_symbol(catalog: CatalogStore, source_library: str, symbol: str) -> str:
    """Return the provider-scoped target library for a legacy yfinance symbol."""

    if source_library == PRICES_LIBRARY:
        vehicle_type = pooled_vehicle_type(catalog, symbol)
        if vehicle_type == "etf":
            return provider_library(ETF_PRICES_LIBRARY, PROVIDER)
        if vehicle_type == "fund":
            return provider_library(FUND_PRICES_LIBRARY, PROVIDER)
    return provider_library(source_library, PROVIDER)


def is_equity_fundamental_library(library: str) -> bool:
    return str(library).startswith("fundamental_")


def should_skip_equity_fundamental_symbol(catalog: CatalogStore, symbol: str) -> bool:
    return pooled_vehicle_type(catalog, symbol) is not None


def pooled_vehicle_type(catalog: CatalogStore, symbol: str) -> str | None:
    profiles = catalog.list_profiles(symbol)
    profiles.extend(catalog.list_etf_profiles(symbol))
    for profile in profiles:
        payload = {str(key).lower(): value for key, value in dict(profile.payload or {}).items()}
        if _truthy(payload.get("is_etf")) or _truthy(payload.get("isetf")):
            return "etf"
        if _truthy(payload.get("is_fund")) or _truthy(payload.get("isfund")):
            return "fund"
        quote_type = str(payload.get("quote_type") or payload.get("quotetype") or "").strip().lower()
        if quote_type == "etf":
            return "etf"
        if quote_type in {"mutualfund", "mutual_fund", "fund"}:
            return "fund"
        instrument_type = str(payload.get("type") or payload.get("instrument_type") or "").strip().lower()
        if instrument_type == "etf":
            return "etf"
        if instrument_type in {"mutualfund", "mutual_fund", "fund"}:
            return "fund"
        if payload.get("fund_family") not in (None, ""):
            return "fund"
    if _looks_like_mutual_fund_symbol(symbol):
        return "fund"
    return None


def _looks_like_mutual_fund_symbol(symbol: str) -> bool:
    text = str(symbol or "").strip().upper()
    return len(text) == 5 and text.endswith("X") and text.isalpha()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}
