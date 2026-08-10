from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from quant_warehouse.ingest.credentials import configure_openbb_credentials

US_EXCHANGE_ALIASES: dict[str, frozenset[str]] = {
    "NASDAQ": frozenset({"NASDAQ", "NMS", "NGM", "NCM", "NAS", "XNAS"}),
    "NYSE": frozenset({"NYSE", "NYQ", "NYS", "XNYS"}),
    "AMEX": frozenset({"AMEX", "ASE", "AMX", "XASE"}),
}


@dataclass(frozen=True)
class ScreenerQuery:
    provider: str = "fmp"
    mktcap_min: int | None = None
    mktcap_max: int | None = None
    country: str | None = None
    exchanges: tuple[str, ...] = ()
    sector: str | None = None
    industry: str | None = None
    is_etf: bool | None = None
    is_fund: bool | None = None
    is_active: bool | None = None
    all_share_classes: bool | None = None
    limit: int = 10_000


def _normalize_country(country: str | None) -> str | None:
    text = str(country or "").strip()
    if not text:
        return None
    if text.upper() == "US":
        return "US"
    return text


def _normalize_exchange_token(value: Any) -> str:
    return str(value or "").strip().upper()


def exchange_matches_filters(raw_exchange: Any, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    token = _normalize_exchange_token(raw_exchange)
    if not token:
        return False
    for exchange in allowed:
        normalized = _normalize_exchange_token(exchange)
        aliases = US_EXCHANGE_ALIASES.get(normalized, frozenset())
        if token in aliases or token == normalized:
            return True
        if normalized and normalized in token:
            return True
    return False


def _records_to_frame(records: Any) -> pl.DataFrame:
    if records is None:
        return pl.DataFrame()
    if isinstance(records, list):
        return pl.DataFrame(records)
    if isinstance(records, dict):
        return pl.DataFrame([records])
    return pl.DataFrame()


def _normalize_screener_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    rename_map = {
        "companyName": "name",
        "marketCap": "market_cap",
        "exchangeShortName": "exchange",
        "activelyTrading": "actively_trading",
        "isEtf": "is_etf",
        "isFund": "is_fund",
    }
    existing = {src: dst for src, dst in rename_map.items() if src in frame.columns and dst not in frame.columns}
    out = frame.rename(existing)
    if "exchangeShortName" in frame.columns:
        short = pl.col("exchangeShortName").cast(pl.String, strict=False).str.strip_chars()
        out = out.with_columns(pl.when(short != "").then(short).otherwise(pl.col("exchange")).alias("exchange") if "exchange" in out.columns else short.alias("exchange"))
    if "symbol" in out.columns:
        out = out.with_columns(pl.col("symbol").cast(pl.String, strict=False).str.strip_chars().str.to_uppercase()).filter(pl.col("symbol") != "")
    return out.with_columns(pl.col(column).cast(pl.Float64, strict=False) for column in ("market_cap", "beta") if column in out.columns)


def _fetch_openbb_screener(
    query: ScreenerQuery,
    *,
    output_format: str = "polars",
) -> pl.DataFrame:
    try:
        from openbb import obb
    except ImportError as exc:
        raise ImportError("OpenBB is a required quant-warehouse dependency; reinstall quant-warehouse.") from exc

    configure_openbb_credentials()
    provider = str(query.provider or "fmp").strip().lower()
    kwargs: dict[str, Any] = {"provider": provider, "limit": int(query.limit)}
    if query.mktcap_min is not None:
        kwargs["mktcap_min"] = int(query.mktcap_min)
    if query.mktcap_max is not None:
        kwargs["mktcap_max"] = int(query.mktcap_max)
    if query.country:
        country = _normalize_country(query.country)
        kwargs["country"] = "us" if provider == "yfinance" and country == "US" else country
    if query.sector:
        kwargs["sector"] = str(query.sector).strip()
    if query.industry:
        kwargs["industry"] = str(query.industry).strip()
    if query.is_etf is not None:
        kwargs["is_etf"] = bool(query.is_etf)
    if query.is_fund is not None:
        kwargs["is_fund"] = bool(query.is_fund)
    if query.is_active is not None:
        kwargs["is_active"] = bool(query.is_active)
    if query.all_share_classes is not None:
        kwargs["all_share_classes"] = bool(query.all_share_classes)
    if len(query.exchanges) == 1:
        kwargs["exchange"] = query.exchanges[0].lower() if provider == "yfinance" else query.exchanges[0]

    result = obb.equity.screener(**kwargs)
    raw = result.to_polars()
    frame = _normalize_screener_frame(raw)
    if frame.is_empty():
        return frame
    if query.exchanges:
        exchange_col = "exchange" if "exchange" in frame.columns else None
        if exchange_col is not None:
            allowed = [value for value in frame[exchange_col].to_list() if exchange_matches_filters(value, query.exchanges)]
            frame = frame.filter(pl.col(exchange_col).is_in(allowed))
    return frame


def fetch_equity_screener(
    query: ScreenerQuery,
    *,
    output_format: str = "polars",
) -> tuple[pl.DataFrame, str]:
    """
    Fetch a cross-sectional equity universe.

    OpenBB is the only data adapter. Provider issues should be fixed in the
    OpenBB fork instead of bypassing it here.
    """
    provider = str(query.provider or "fmp").strip().lower()
    frame = _fetch_openbb_screener(query, output_format=output_format)
    return frame, f"openbb:{provider}"


def screener_record_to_profile_payload(record: dict[str, Any]) -> dict[str, object]:
    payload = dict(record or {})
    symbol = str(payload.get("symbol") or "").strip().upper()
    if symbol:
        payload["symbol"] = symbol
    name = payload.get("name") or payload.get("companyName")
    if name not in (None, ""):
        payload.setdefault("name", name)
        payload.setdefault("company_name", name)
    market_cap = payload.get("market_cap", payload.get("marketCap"))
    if market_cap not in (None, ""):
        payload["market_cap"] = market_cap
    exchange = payload.get("exchange") or payload.get("exchangeShortName")
    if exchange not in (None, ""):
        payload["exchange"] = exchange
    for key in ("ipoDate", "ipo_date"):
        if payload.get(key) not in (None, ""):
            payload["ipoDate"] = payload.get(key)
            break
    return payload
