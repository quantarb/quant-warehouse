from __future__ import annotations

import re
from typing import Mapping


_FUND_QUOTE_TYPES = {"fund", "mutualfund", "mutual_fund", "money_market"}
_UNSUPPORTED_NAME = re.compile(
    r"\b(warrants?|rights?|units?|preferred|depositary shares?|debentures?|notes?)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SYMBOL = re.compile(r"-(?:P[A-Z]|WT|WS|W|U|R)$", re.IGNORECASE)


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _optional_bool(record: Mapping[str, object], *names: str) -> bool | None:
    for name in names:
        if name not in record or record[name] in (None, ""):
            continue
        value = record[name]
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return None


def supported_equity_reason(symbol: str, record: Mapping[str, object]) -> tuple[bool, str]:
    """Classify common, active equities supported by training and live refresh."""
    normalized = str(symbol or record.get("symbol") or "").strip().upper()
    if not normalized:
        return False, "unsupported: missing_symbol"
    if _optional_bool(record, "is_etf", "isEtf") is True or _clean(record.get("quote_type")) == "etf":
        return False, "asset_class: etf"
    if (_optional_bool(record, "is_fund", "isFund") is True
            or _clean(record.get("quote_type")) in _FUND_QUOTE_TYPES):
        return False, "asset_class: fund"
    # US mutual funds conventionally use five alphabetic characters ending X.
    if len(normalized) == 5 and normalized.isalpha() and normalized.endswith("X"):
        return False, "asset_class: fund_symbol_pattern"
    active = _optional_bool(
        record,
        "actively_trading", "is_actively_trading", "isActivelyTrading", "is_active", "active",
    )
    if active is False:
        return False, "listing: inactive"
    if _UNSUPPORTED_SYMBOL.search(normalized):
        return False, "asset_class: unsupported_symbol"
    name = str(
        record.get("company_name") or record.get("companyName") or record.get("name") or ""
    ).strip()
    if _UNSUPPORTED_NAME.search(name):
        return False, "asset_class: unsupported_security"
    return True, "ok"
