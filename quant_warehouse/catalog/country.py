from __future__ import annotations

COUNTRY_ALIASES: dict[str, frozenset[str]] = {
    "US": frozenset({"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}),
    "GB": frozenset({"GB", "UK", "UNITED KINGDOM", "GREAT BRITAIN"}),
    "CN": frozenset({"CN", "CHINA", "PEOPLES REPUBLIC OF CHINA"}),
}


def normalize_country_code(value: str | None) -> str | None:
    token = str(value or "").strip().upper()
    if not token:
        return None
    for code, aliases in COUNTRY_ALIASES.items():
        if token in aliases:
            return code
    return token


def country_matches_filter(raw_country: str | None, *, filter_country: str | None) -> bool:
    if not filter_country:
        return True
    normalized_filter = normalize_country_code(filter_country)
    normalized_value = normalize_country_code(raw_country)
    if normalized_filter is None:
        return True
    return normalized_value == normalized_filter