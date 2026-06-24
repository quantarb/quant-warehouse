from __future__ import annotations

import json
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from quant_warehouse.ingest.credentials import resolve_fmp_api_key
from quant_warehouse.target_engineering.event_pairs.event_pair_normalizer import (
    normalize_event_pairs,
)
from quant_warehouse.target_engineering.event_pairs.event_pair_schema import EVENT_PAIR_COLUMNS
from quant_warehouse.target_engineering.event_pairs.event_pair_taxonomy import EVENT_PAIR_TAXONOMY

_FMP_API_V4_BASE = "https://financialmodelingprep.com/api/v4"
_FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
_SUPPORTED_FAMILIES = tuple(EVENT_PAIR_TAXONOMY)


def fetch_fmp_event_pairs(
    symbol: str,
    *,
    event_families: Sequence[str] = _SUPPORTED_FAMILIES,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 0,
    limit: int = 100,
) -> pd.DataFrame:
    """Fetch real FMP mirrored event-pair data for one symbol.

    Supports every family in EVENT_PAIR_TAXONOMY. Families with no literal FMP
    event-side field are derived from vendor-provided deltas or comparable values.
    """

    frames = [
        fetch_fmp_event_pair_family(
            symbol,
            event_family=family,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
        )
        for family in event_families
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "event_date", "event_family", "event_type"],
        ignore_index=True,
    )


def fetch_fmp_event_pair_family(
    symbol: str,
    *,
    event_family: str,
    start_date: str | None = None,
    end_date: str | None = None,
    page: int = 0,
    limit: int = 100,
) -> pd.DataFrame:
    family = str(event_family or "").strip().lower()
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if family == "insider":
        return _fetch_insider_event_pairs(
            symbol,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
        )
    if family == "congress":
        return _fetch_congress_event_pairs(
            symbol,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
        )
    if family == "analyst_rating":
        return _fetch_analyst_rating_event_pairs(symbol, start_date=start_date, end_date=end_date)
    if family == "price_target":
        return _fetch_price_target_event_pairs(symbol, start_date=start_date, end_date=end_date)
    if family == "institutional":
        return _fetch_institutional_event_pairs(symbol, start_date=start_date, end_date=end_date)
    if family == "capital_action":
        return _fetch_capital_action_event_pairs(
            symbol,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
        )
    if family == "dividend":
        return _fetch_dividend_event_pairs(symbol, start_date=start_date, end_date=end_date)
    if family == "split":
        return _fetch_split_event_pairs(symbol, start_date=start_date, end_date=end_date)
    if family == "earnings":
        return _fetch_earnings_event_pairs(symbol, start_date=start_date, end_date=end_date)
    raise ValueError(f"Unsupported FMP event_pair family: {event_family}")


def _fetch_insider_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
    page: int,
    limit: int,
) -> pd.DataFrame:
    records = _fmp_records(
        "insider-trading/search",
        params={"symbol": symbol, "page": int(page), "limit": int(limit)},
    )
    frame = _records_to_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)

    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_type"] = frame.apply(_insider_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(
        frame,
        ("transactionDate", "transaction_date", "filingDate", "filing_date"),
    )
    frame["actor_type"] = _first_present_column(
        frame,
        ("typeOfOwner", "type_of_owner", "relationship", "officerTitle"),
    )
    frame["actor_name"] = _first_present_column(
        frame,
        ("reportingName", "reporting_name", "ownerName", "name"),
    )
    frame["strength"] = _first_present_column(
        frame,
        ("securitiesTransacted", "securities_transacted", "shares", "transactionShares"),
    )
    frame["raw_json"] = _raw_records(frame)
    return normalize_event_pairs(
        frame,
        event_family="insider",
        event_type_col="event_type",
        symbol_col="symbol",
        event_date_col="event_date",
        source="fmp:insider-trading/search",
        actor_type_col="actor_type",
        actor_name_col="actor_name",
        strength_col="strength",
        raw_json_col="raw_json",
    )


def _fetch_congress_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
    page: int,
    limit: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chamber, endpoint in (("senate", "senate-trades"), ("house", "house-trades")):
        records = _fmp_records(
            endpoint,
            params={"symbol": symbol, "page": int(page), "limit": int(limit)},
        )
        frame = _records_to_frame(records)
        if frame.empty:
            continue
        frame["actor_type"] = chamber
        frame["source_endpoint"] = endpoint
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)

    frame = pd.concat(frames, ignore_index=True)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_type"] = frame.apply(_congress_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(
        frame,
        ("transactionDate", "transaction_date", "disclosureDate", "disclosure_date"),
    )
    actor_name = _first_present_column(frame, ("representative", "senator", "firstName", "name"))
    last_name = _first_present_column(frame, ("lastName",))
    frame["actor_name"] = _combine_names(actor_name, last_name)
    frame["strength"] = _first_present_column(frame, ("amount", "amountRange", "assetDescription"))
    frame["raw_json"] = _raw_records(frame)
    return normalize_event_pairs(
        frame,
        event_family="congress",
        event_type_col="event_type",
        symbol_col="symbol",
        event_date_col="event_date",
        source="fmp:congress-trades",
        actor_type_col="actor_type",
        actor_name_col="actor_name",
        strength_col="strength",
        raw_json_col="raw_json",
    )


def _fetch_analyst_rating_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    frame = _records_to_frame(_fmp_records("grades-historical", params={"symbol": symbol}))
    if frame.empty:
        frame = _records_to_frame(_fmp_records("grades", params={"symbol": symbol}))
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)

    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_type"] = frame.apply(_analyst_rating_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(frame, ("date", "publishedDate", "gradingDate"))
    frame["actor_type"] = "analyst"
    frame["actor_name"] = _first_present_column(
        frame,
        ("gradingCompany", "company", "analystCompany", "firm", "analystName"),
    )
    frame["strength"] = _first_present_column(frame, ("newGrade", "newRating", "action"))
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(
        frame,
        event_family="analyst_rating",
        source="fmp:grades-historical",
    )


def _fetch_price_target_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    frame = _records_to_frame(_fmp_records("legacy/price-target", params={"symbol": symbol}))
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(frame, ("publishedDate", "date"))
    frame["target_value"] = pd.to_numeric(
        _first_present_column(frame, ("priceTarget", "price_target", "adjPriceTarget")),
        errors="coerce",
    )
    frame = frame.dropna(subset=["event_date", "target_value"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    group_key = _first_present_column(
        frame,
        ("analystCompany", "analystName", "publisher", "newsPublisher"),
    ).fillna("all")
    frame["_group_key"] = group_key
    frame = frame.sort_values(["_group_key", "event_date"])
    frame["previous_target_value"] = frame.groupby("_group_key")["target_value"].shift(1)
    frame["event_type"] = frame.apply(_price_target_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "analyst"
    frame["actor_name"] = _first_present_column(
        frame,
        ("analystName", "analystCompany", "publisher", "newsPublisher"),
    )
    frame["strength"] = frame["target_value"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(
        frame,
        event_family="price_target",
        source="fmp:price-target",
    )


def _fetch_institutional_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    frame = _records_to_frame(
        _fmp_records(
            "institutional-ownership/symbol-positions-summary",
            params={"symbol": symbol},
        )
    )
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _ensure_symbol(frame, symbol)
    frame = _ensure_quarter_date(frame)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["delta"] = _first_numeric_column(
        frame,
        (
            "changeInShares",
            "sharesChange",
            "sharesHeldChange",
            "investorCountChange",
            "numberOfInvestorsChange",
            "ownershipPercentChange",
        ),
    )
    if frame["delta"].isna().all():
        shares = _first_numeric_column(frame, ("sharesHeld", "shares", "totalShares"))
        frame = frame.sort_values("event_date")
        frame["delta"] = shares.diff()
    frame["event_type"] = frame["delta"].map(_institutional_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "institution"
    frame["actor_name"] = "aggregate"
    frame["strength"] = frame["delta"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(
        frame,
        event_family="institutional",
        source="fmp:institutional-ownership/symbol-positions-summary",
    )


def _fetch_capital_action_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
    page: int,
    limit: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offerings = _records_to_frame(
        _fmp_records("fundraising-search", params={"name": symbol, "page": page, "limit": limit})
    )
    if not offerings.empty:
        offerings = _ensure_symbol(offerings, symbol)
        offerings["event_type"] = "equity_offering"
        offerings["event_date"] = _first_present_column(
            offerings,
            ("filingDate", "date", "acceptedDate"),
        )
        offerings["actor_type"] = "issuer"
        offerings["actor_name"] = _first_present_column(offerings, ("companyName", "name", "issuerName"))
        offerings["strength"] = _first_present_column(
            offerings,
            ("offeringAmount", "maximumOfferingAmount", "amount", "totalOfferingAmount"),
        )
        offerings["raw_json"] = _raw_records(offerings)
        frames.append(offerings)

    buybacks = _records_to_frame(
        _fmp_records("news/press-releases", params={"symbols": symbol, "page": page, "limit": limit})
    )
    if not buybacks.empty:
        buybacks = _ensure_symbol(buybacks, symbol)
        buybacks["event_type"] = buybacks.apply(_capital_press_release_event_type, axis=1)
        buybacks = buybacks.dropna(subset=["event_type"]).copy()
        if not buybacks.empty:
            buybacks["event_date"] = _first_present_column(
                buybacks,
                ("publishedDate", "date", "acceptedDate"),
            )
            buybacks["actor_type"] = "issuer"
            buybacks["actor_name"] = _first_present_column(
                buybacks,
                ("publisher", "site", "symbol", "companyName"),
            )
            buybacks["strength"] = _first_present_column(buybacks, ("title", "text"))
            buybacks["raw_json"] = _raw_records(buybacks)
            frames.append(buybacks)

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    return _normalize_family_frame(
        frame,
        event_family="capital_action",
        source="fmp:fundraising-search+press-releases",
    )


def _fetch_dividend_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    frame = _records_to_frame(_fmp_records("dividends", params={"symbol": symbol}))
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(
        frame,
        ("exDividendDate", "declarationDate", "date", "paymentDate"),
    )
    frame["dividend_value"] = _first_numeric_column(
        frame,
        ("adjDividend", "dividend", "amount", "cashAmount"),
    )
    frame = frame.dropna(subset=["event_date", "dividend_value"]).copy()
    frame = frame.sort_values("event_date")
    frame["previous_dividend_value"] = frame["dividend_value"].shift(1)
    frame["event_type"] = frame.apply(_dividend_event_type, axis=1)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["dividend_value"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="dividend", source="fmp:dividends")


def _fetch_split_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    frame = _records_to_frame(_fmp_records("splits", params={"symbol": symbol}))
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(frame, ("date", "splitDate"))
    frame["split_ratio"] = frame.apply(_split_ratio, axis=1)
    frame["event_type"] = frame["split_ratio"].map(_split_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["split_ratio"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="split", source="fmp:splits")


def _fetch_earnings_event_pairs(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    frame = _records_to_frame(_fmp_records("earnings", params={"symbol": symbol}))
    if frame.empty:
        frame = _records_to_frame(_fmp_records("earnings-calendar", params={"symbol": symbol}))
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame = _filter_dates(frame, start_date=start_date, end_date=end_date)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["event_date"] = _first_present_column(
        frame,
        ("date", "fiscalDateEnding", "reportDate", "reportedDate"),
    )
    actual = _first_numeric_column(frame, ("epsActual", "actualEps", "eps", "reportedEPS"))
    estimated = _first_numeric_column(frame, ("epsEstimated", "estimatedEps", "epsEstimate"))
    frame["surprise"] = actual - estimated
    frame["event_type"] = frame["surprise"].map(_earnings_event_type)
    frame = frame.dropna(subset=["event_type"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_PAIR_COLUMNS)
    frame["actor_type"] = "issuer"
    frame["actor_name"] = symbol
    frame["strength"] = frame["surprise"]
    frame["raw_json"] = _raw_records(frame)
    return _normalize_family_frame(frame, event_family="earnings", source="fmp:earnings")


def _fmp_get_json(endpoint: str, *, params: dict[str, Any]) -> Any:
    api_key = resolve_fmp_api_key(required=True)
    query = urlencode({**params, "apikey": api_key})
    if endpoint.startswith("legacy/"):
        url = f"{_FMP_API_V4_BASE}/{endpoint.removeprefix('legacy/').lstrip('/')}?{query}"
    else:
        url = f"{_FMP_STABLE_BASE}/{endpoint.lstrip('/')}?{query}"
    with urlopen(url, timeout=60.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _fmp_records(endpoint: str, *, params: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _fmp_get_json(endpoint, params=params)
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        return [payload]
    return []


def _records_to_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = list(records)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    return frame


def _filter_dates(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    date_values = _first_present_column(
        frame,
        (
            "event_date",
            "transactionDate",
            "transaction_date",
            "filingDate",
            "filing_date",
            "disclosureDate",
            "disclosure_date",
            "publishedDate",
            "published_date",
            "date",
            "exDividendDate",
            "ex_dividend_date",
            "declarationDate",
            "declaration_date",
            "paymentDate",
            "payment_date",
            "splitDate",
            "split_date",
            "fiscalDateEnding",
            "fiscal_date_ending",
            "reportDate",
            "report_date",
            "reportedDate",
            "reported_date",
            "acceptedDate",
            "accepted_date",
            "period_ending",
        ),
    )
    if date_values.isna().all():
        return pd.DataFrame(columns=frame.columns)
    out = frame.copy()
    event_dates = pd.to_datetime(date_values, errors="coerce")
    out = out.loc[event_dates.notna()].copy()
    event_dates = event_dates.loc[out.index]
    if start_date is not None:
        out = out.loc[event_dates >= pd.Timestamp(start_date)]
        event_dates = event_dates.loc[out.index]
    if end_date is not None:
        out = out.loc[event_dates <= pd.Timestamp(end_date)]
    return out


def _insider_event_type(row: pd.Series) -> str | None:
    transaction = _row_text(row, "transactionType", "transaction_type", "type")
    acquired_disposed = _row_text(row, "acquisitionOrDisposition", "acquiredDisposedCode")
    text = f"{transaction} {acquired_disposed}".lower()
    if _is_buy_text(text):
        return "insider_buy"
    if _is_sell_text(text):
        return "insider_sell"
    return None


def _congress_event_type(row: pd.Series) -> str | None:
    text = _row_text(row, "transactionType", "transaction_type", "type").lower()
    if _is_buy_text(text):
        return "congress_buy"
    if _is_sell_text(text):
        return "congress_sell"
    return None


def _analyst_rating_event_type(row: pd.Series) -> str | None:
    action = _row_text(row, "action", "gradeAction", "grade_action").lower()
    if any(token in action for token in ("upgrade", "upgraded", "raised", "initiated")):
        return "analyst_upgrade"
    if any(token in action for token in ("downgrade", "downgraded", "lowered")):
        return "analyst_downgrade"

    old_rating = _rating_score(_row_text(row, "previousGrade", "previousRating", "oldGrade", "oldRating"))
    new_rating = _rating_score(_row_text(row, "newGrade", "newRating", "grade", "rating"))
    if old_rating is None or new_rating is None or old_rating == new_rating:
        return None
    return "analyst_upgrade" if new_rating > old_rating else "analyst_downgrade"


def _price_target_event_type(row: pd.Series) -> str | None:
    current = row.get("target_value")
    previous = row.get("previous_target_value")
    if pd.isna(current) or pd.isna(previous) or float(current) == float(previous):
        return None
    return "price_target_raise" if float(current) > float(previous) else "price_target_cut"


def _institutional_event_type(value: Any) -> str | None:
    if pd.isna(value) or float(value) == 0.0:
        return None
    return "institutional_add" if float(value) > 0 else "institutional_reduce"


def _capital_press_release_event_type(row: pd.Series) -> str | None:
    text = _row_text(row, "title", "text", "content").lower()
    if any(token in text for token in ("buyback", "repurchase", "share repurchase")):
        return "buyback_authorization"
    return None


def _dividend_event_type(row: pd.Series) -> str | None:
    current = row.get("dividend_value")
    previous = row.get("previous_dividend_value")
    if pd.isna(current) or pd.isna(previous) or float(current) == float(previous):
        return None
    return "dividend_increase" if float(current) > float(previous) else "dividend_cut"


def _split_event_type(value: Any) -> str | None:
    if pd.isna(value) or float(value) == 1.0:
        return None
    return "forward_split" if float(value) > 1.0 else "reverse_split"


def _earnings_event_type(value: Any) -> str | None:
    if pd.isna(value) or float(value) == 0.0:
        return None
    return "earnings_beat" if float(value) > 0 else "earnings_miss"


def _rating_score(text: str) -> int | None:
    value = str(text or "").strip().lower()
    if not value:
        return None
    buckets = (
        (5, ("strong buy", "outperform", "overweight", "buy")),
        (4, ("market perform", "neutral", "hold", "equal weight")),
        (3, ("underperform", "underweight", "sell")),
    )
    for score, tokens in buckets:
        if any(token in value for token in tokens):
            return score
    return None


def _is_buy_text(text: str) -> bool:
    tokens = str(text or "").strip().lower()
    return any(token in tokens for token in ("purchase", "buy", "acquisition", "acquired", " p "))


def _is_sell_text(text: str) -> bool:
    tokens = str(text or "").strip().lower()
    return any(token in tokens for token in ("sale", "sell", "disposition", "disposed", " s "))


def _row_text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return f" {row[column]} "
    return ""


def _first_present_column(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([None] * len(frame), index=frame.index)


def _first_numeric_column(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return pd.to_numeric(_first_present_column(frame, columns), errors="coerce")


def _ensure_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = frame.copy()
    if "symbol" not in out.columns:
        out["symbol"] = symbol
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper().replace({"": symbol})
    return out


def _ensure_quarter_date(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "event_date" not in out.columns:
        out["event_date"] = _first_present_column(
            out,
            (
                "date",
                "period_ending",
                "fillingDate",
                "filingDate",
                "reportedDate",
                "calendarYear",
                "year",
            ),
        )
    return out


def _split_ratio(row: pd.Series) -> float | None:
    ratio = _first_numeric_column(pd.DataFrame([row]), ("splitRatio", "split_ratio", "ratio")).iloc[0]
    if pd.notna(ratio):
        return float(ratio)
    numerator = _first_numeric_column(pd.DataFrame([row]), ("numerator", "splitFrom", "fromFactor")).iloc[0]
    denominator = _first_numeric_column(pd.DataFrame([row]), ("denominator", "splitTo", "toFactor")).iloc[0]
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _normalize_family_frame(frame: pd.DataFrame, *, event_family: str, source: str) -> pd.DataFrame:
    return normalize_event_pairs(
        frame,
        event_family=event_family,
        event_type_col="event_type",
        symbol_col="symbol",
        event_date_col="event_date",
        source=source,
        actor_type_col="actor_type",
        actor_name_col="actor_name",
        strength_col="strength",
        raw_json_col="raw_json",
    )


def _combine_names(first: pd.Series, last: pd.Series) -> pd.Series:
    if last.isna().all():
        return first
    values: list[str | None] = []
    for first_value, last_value in zip(first, last, strict=False):
        parts = [str(value).strip() for value in (first_value, last_value) if pd.notna(value)]
        values.append(" ".join(part for part in parts if part) or None)
    return pd.Series(values, index=first.index)


def _raw_records(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame.to_dict(orient="records"), index=frame.index)
