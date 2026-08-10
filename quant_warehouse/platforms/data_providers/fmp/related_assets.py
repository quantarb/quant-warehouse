"""FMP discovery and feature construction for securities related to an issuer."""

from __future__ import annotations

import polars as pl

import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable

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
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"):
        try:
            return datetime.strptime(match.group(1), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


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
) -> pl.DataFrame:
    """Fetch dividend-adjusted OHLCV using canonical warehouse column names."""
    client = session or requests.Session()
    response = client.get(
        "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted",
        params={"symbol": symbol, "from": start, "to": end, "apikey": api_key}, timeout=60,
    )
    data = response.json()
    frame = pl.DataFrame(data) if isinstance(data, list) else pl.DataFrame()
    if frame.is_empty():
        return frame
    frame = frame.rename({"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"})
    return frame.select([column for column in RELATED_OHLCV_COLUMNS if column in frame.columns])


def _build_related_instrument_frame(
    issuer: str,
    instrument_symbol: str,
    security_class: str,
    maturity: str | None,
    prices: pl.DataFrame,
) -> pl.DataFrame:
    prices = prices.with_columns(pl.lit(instrument_symbol).alias("symbol"))
    built = build_historical_price_eod_features(instrument_symbol, prices)
    if built.df.is_empty():
        return pl.DataFrame()
    technical = built.df.rename({"symbol": "instrument_symbol"})
    raw = prices.select([column for column in ("date", "open", "high", "low", "close", "volume") if column in prices.columns]).with_columns(pl.lit(instrument_symbol).alias("instrument_symbol"))
    frame = raw.join(technical, on=["date", "instrument_symbol"], how="left")
    feature_columns = [column for column in frame.columns if column.startswith("px__") or column in {"open", "high", "low", "close", "volume"}]
    frame = frame.select(["date", "instrument_symbol", *feature_columns]).rename({column: f"{security_class}__{column}" for column in feature_columns})
    return frame.with_columns(pl.lit(issuer).alias("symbol"), pl.lit(security_class).alias("asset_class"), pl.lit(maturity).alias("maturity_date")).select(["symbol", "date", "asset_class", "maturity_date", "instrument_symbol", *[column for column in frame.columns if column not in {"date", "instrument_symbol"}]])


def build_related_asset_panel(
    issuers: Iterable[str],
    api_key: str,
    *,
    start: str = "1900-01-01",
    end: str = "2025-12-31",
    discover_workers: int = 6,
    fetch_workers: int = 4,
    progress_logger: Callable[[object], None] = print,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Discover and build adjusted related-security features for issuers."""
    issuer_values = sorted({str(value).strip().upper() for value in issuers if str(value).strip()})
    discovered: dict[str, list[tuple[str, str, str | None]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, discover_workers)) as pool:
        futures = [pool.submit(discover_related_instruments, issuer, api_key) for issuer in issuer_values]
        for future in as_completed(futures):
            issuer, candidates = future.result()
            discovered[issuer] = candidates
    requests_by_symbol = sorted({symbol for values in discovered.values() for symbol, _, _ in values})
    raw: dict[str, pl.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, fetch_workers)) as pool:
        futures = {pool.submit(fetch_related_adjusted_ohlcv, symbol, api_key, start=start, end=end): symbol for symbol in requests_by_symbol}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                raw[symbol] = future.result()
            except Exception as exc:
                progress_logger({"symbol": symbol, "error": str(exc)})
    frames: list[pl.DataFrame] = []
    for issuer, candidates in discovered.items():
        for instrument_symbol, security_class, maturity in candidates:
            prices = raw.get(instrument_symbol)
            if prices is None or prices.is_empty():
                continue
            try:
                frame = _build_related_instrument_frame(issuer, instrument_symbol, security_class, maturity, prices)
            except Exception as exc:
                progress_logger({"issuer": issuer, "instrument": instrument_symbol, "error": str(exc)})
                continue
            if not frame.is_empty():
                frames.append(frame)
    panel = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    if not panel.is_empty():
        panel = panel.sort(["symbol", "asset_class", "instrument_symbol", "date"])
    stats = {
        "issuers": len(issuer_values),
        "candidate_symbols": len(requests_by_symbol),
        "rows": panel.height,
        "covered_issuers": panel["symbol"].n_unique() if not panel.is_empty() else 0,
        "unique_instruments": panel["instrument_symbol"].n_unique() if not panel.is_empty() else 0,
        "classes": sorted(panel["asset_class"].unique().to_list()) if not panel.is_empty() else [],
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
