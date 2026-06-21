from __future__ import annotations

from typing import Callable, Sequence

from quant_warehouse.ingest.screener_fetch import (
    ScreenerQuery,
    fetch_equity_screener,
    screener_record_to_profile_payload,
)
from quant_warehouse.warehouse.api import Warehouse

ProgressLogger = Callable[[str], None] | None


def screen_universe_to_catalog(
    warehouse: Warehouse,
    query: ScreenerQuery,
    *,
    progress_logger: ProgressLogger = None,
) -> tuple[tuple[str, ...], str]:
    frame, source = fetch_equity_screener(query)
    if frame.empty:
        return tuple(), source

    provider_name = str(query.provider or "fmp").strip().lower()
    source_provider = source.replace("openbb:", "", 1) if source.startswith("openbb:") else source
    if source_provider == "fmp":
        source_provider = "fmp_screener"

    records = frame.where(frame.notna(), None).to_dict(orient="records")
    symbols: list[str] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        payload = screener_record_to_profile_payload(raw_record)
        symbol = str(payload.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        warehouse.catalog.upsert_profile(
            symbol=symbol,
            provider=provider_name,
            source_provider=source_provider,
            payload=payload,
        )
        symbols.append(symbol)

    unique_symbols = tuple(dict.fromkeys(symbols))
    if callable(progress_logger):
        progress_logger(
            f"Warehouse screener stored {len(unique_symbols):,} symbols via {source}"
        )
    return unique_symbols, source


def resolve_universe_from_catalog(
    warehouse: Warehouse,
    *,
    provider: str = "fmp",
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
    country: str | None = None,
    exchanges: Sequence[str] | None = None,
    exclude_pooled_vehicles: bool = False,
    limit: int | None = None,
) -> tuple[str, ...]:
    profiles = warehouse.catalog.query_symbol_profiles(
        provider=str(provider or "fmp").strip().lower(),
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        country=country,
        exchanges=tuple(str(value).strip().upper() for value in (exchanges or ()) if str(value).strip()),
        exclude_etf=bool(exclude_pooled_vehicles),
        exclude_fund=bool(exclude_pooled_vehicles),
        limit=limit,
    )
    return tuple(profile.symbol for profile in profiles)