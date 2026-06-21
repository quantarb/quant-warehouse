from __future__ import annotations

from datetime import date
from typing import Any

from quant_warehouse.warehouse.sections import MIN_HISTORICAL_DATE


def _absolute_historical_floor() -> date:
    parsed = parse_listing_date(MIN_HISTORICAL_DATE)
    return parsed or date(1900, 1, 1)


def equity_historical_floor(*, ipo_date: str | date | None = None) -> date:
    """Earliest allowed equity history start: max(1900-01-01, ipo_date)."""
    floor = _absolute_historical_floor()
    ipo = ipo_date if isinstance(ipo_date, date) else parse_listing_date(ipo_date)
    if ipo is not None and ipo > floor:
        return ipo
    return floor


def equity_historical_floor_text(*, ipo_date: str | date | None = None) -> str:
    return equity_historical_floor(ipo_date=ipo_date).isoformat()

LISTING_DATE_KEYS: tuple[str, ...] = (
    "ipodate",
    "ipo",
    "firsttradedate",
    "listingdate",
    "listeddate",
    "firststockpricedate",
)


def parse_listing_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        try:
            from datetime import datetime

            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None


def listing_date_from_record(record: dict[str, object] | None) -> str | None:
    if not isinstance(record, dict):
        return None
    normalized = {str(key).lower().replace("_", ""): value for key, value in record.items()}
    for key in LISTING_DATE_KEYS:
        parsed = parse_listing_date(normalized.get(key))
        if parsed is not None:
            return parsed.isoformat()
    return None