"""FMP discovery and feature construction for securities related to an issuer."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable

import pandas as pd
import requests

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.technical import (
    build_historical_price_eod_features,
)


RELATED_SECURITY_CLASSES = (
    "preferred", "warrant", "unit", "note_bond", "adr", "ordinary", "etf",
)
RELATED_OHLCV_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")


def classify_related_security(name: str, symbol: str, issuer: str) -> str | None:
    """Classify an FMP search result as a non-common security of ``issuer``."""
    text, normalized_symbol = str(name).lower(), str(symbol).strip().upper()
    normalized_issuer = str(issuer).strip().upper()
    if not normalized_symbol or normalized_symbol == normalized_issuer:
        return None
    if "warrant" in text or normalized_symbol.endswith("W") or "-W" in normalized_symbol:
        return "warrant"
    if "unit" in text or normalized_symbol.endswith("U") or "-UN" in normalized_symbol:
        return "unit"
    if any(token in text for token in ("note", "bond", "debenture", "senior debt")):
        return "note_bond"
    if (
        any(token in text for token in ("preferred", "pfd", "pref"))
        or "-P" in normalized_symbol
        or (normalized_symbol.endswith("P") and len(normalized_symbol) > len(normalized_issuer))
    ):
        return "preferred"
    if "depositary" in text or "adr" in text:
        return "adr"
    if "ordinary" in text:
        return "ordinary"
    if " etf" in f" {text}" or text.endswith(" etf"):
        return "etf"
    return None


def parse_related_maturity_date(name: str) -> str | None:
    """Extract an optional expiration/due/maturity date from a security name."""
    match = re.search(
        r"(?:expir(?:es|ing)?|due|matur(?:es|ity))\s*(?:on\s*)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        str(name),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    parsed = pd.to_datetime(match.group(1), errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).strftime("%Y-%m-%d")


def discover_related_instruments(
    issuer: str,
    api_key: str,
    *,
    session: requests.Session | None = None,
) -> tuple[str, list[tuple[str, str, str | None]]]:
    """Discover related FMP symbols and return ``(symbol, class, maturity)``."""
    client = session or requests.Session()
    profile = client.get(
        "https://financialmodelingprep.com/stable/profile",
        params={"symbol": issuer, "apikey": api_key}, timeout=30,
    ).json()
    company_name = profile[0].get("companyName", issuer) if isinstance(profile, list) and profile else issuer
    results = client.get(
        "https://financialmodelingprep.com/stable/search-name",
        params={"query": company_name, "apikey": api_key}, timeout=30,
    ).json()
    candidates = []
    for item in results if isinstance(results, list) else []:
        symbol = str(item.get("symbol") or "").strip().upper()
        security_class = classify_related_security(item.get("name", ""), symbol, issuer)
        if security_class:
            candidates.append((symbol, security_class, parse_related_maturity_date(item.get("name", ""))))
    return str(issuer).strip().upper(), sorted(set(candidates))


def fetch_related_adjusted_ohlcv(
    symbol: str,
    api_key: str,
    *,
    start: str = "1900-01-01",
    end: str = "2025-12-31",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch dividend-adjusted OHLCV using canonical warehouse column names."""
    client = session or requests.Session()
    response = client.get(
        "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted",
        params={"symbol": symbol, "from": start, "to": end, "apikey": api_key}, timeout=60,
    )
    data = response.json()
    frame = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    if frame.empty:
        return frame
    frame = frame.rename(columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"})
    return frame[[column for column in RELATED_OHLCV_COLUMNS if column in frame.columns]]


def _build_related_instrument_frame(
    issuer: str,
    instrument_symbol: str,
    security_class: str,
    maturity: str | None,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    prices = prices.copy()
    prices["symbol"] = instrument_symbol
    built = build_historical_price_eod_features(instrument_symbol, prices)
    if built.df.empty:
        return pd.DataFrame()
    technical = built.df.reset_index().rename(columns={"symbol": "instrument_symbol"})
    technical["date"] = pd.to_datetime(technical["date"], errors="coerce").dt.normalize()
    technical = technical.drop_duplicates(["date", "instrument_symbol"], keep="last")
    raw = prices[[column for column in ("date", "open", "high", "low", "close", "volume") if column in prices]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw.insert(1, "instrument_symbol", instrument_symbol)
    frame = raw.merge(technical, on=["date", "instrument_symbol"], how="left", validate="one_to_one")
    feature_columns = [column for column in frame.columns if column.startswith("px__") or column in {"open", "high", "low", "close", "volume"}]
    frame = frame[["date", "instrument_symbol", *feature_columns]]
    frame = frame.rename(columns={column: f"{security_class}__{column}" for column in feature_columns})
    frame.insert(0, "symbol", issuer)
    frame.insert(2, "asset_class", security_class)
    frame.insert(3, "maturity_date", maturity)
    return frame


def build_related_asset_panel(
    issuers: Iterable[str],
    api_key: str,
    *,
    start: str = "1900-01-01",
    end: str = "2025-12-31",
    discover_workers: int = 6,
    fetch_workers: int = 4,
    progress_logger: Callable[[object], None] = print,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Discover and build adjusted related-security features for issuers."""
    issuer_values = sorted({str(value).strip().upper() for value in issuers if str(value).strip()})
    discovered: dict[str, list[tuple[str, str, str | None]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, discover_workers)) as pool:
        futures = [pool.submit(discover_related_instruments, issuer, api_key) for issuer in issuer_values]
        for future in as_completed(futures):
            issuer, candidates = future.result()
            discovered[issuer] = candidates
    requests_by_symbol = sorted({symbol for values in discovered.values() for symbol, _, _ in values})
    raw: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, fetch_workers)) as pool:
        futures = {pool.submit(fetch_related_adjusted_ohlcv, symbol, api_key, start=start, end=end): symbol for symbol in requests_by_symbol}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                raw[symbol] = future.result()
            except Exception as exc:
                progress_logger({"symbol": symbol, "error": str(exc)})
    frames: list[pd.DataFrame] = []
    for issuer, candidates in discovered.items():
        for instrument_symbol, security_class, maturity in candidates:
            prices = raw.get(instrument_symbol)
            if prices is None or prices.empty:
                continue
            try:
                frame = _build_related_instrument_frame(issuer, instrument_symbol, security_class, maturity, prices)
            except Exception as exc:
                progress_logger({"issuer": issuer, "instrument": instrument_symbol, "error": str(exc)})
                continue
            if not frame.empty:
                frames.append(frame)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        panel = panel.sort_values(["symbol", "asset_class", "instrument_symbol", "date"]).reset_index(drop=True)
    stats = {
        "issuers": len(issuer_values),
        "candidate_symbols": len(requests_by_symbol),
        "rows": len(panel),
        "covered_issuers": int(panel["symbol"].nunique()) if not panel.empty else 0,
        "unique_instruments": int(panel["instrument_symbol"].nunique()) if not panel.empty else 0,
        "classes": sorted(panel["asset_class"].unique().tolist()) if not panel.empty else [],
    }
    return panel, stats


__all__ = [
    "RELATED_SECURITY_CLASSES",
    "RELATED_OHLCV_COLUMNS",
    "classify_related_security",
    "parse_related_maturity_date",
    "discover_related_instruments",
    "fetch_related_adjusted_ohlcv",
    "build_related_asset_panel",
]
