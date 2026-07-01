from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs.event_pair_normalizer import normalize_event_pairs


def insider_event_type(row: pd.Series) -> str | None:
    transaction = row_text(row, "transactionType", "transaction_type", "type")
    acquired_disposed = row_text(
        row,
        "acquisitionOrDisposition",
        "acquisition_or_disposition",
        "acquiredDisposedCode",
        "acquired_disposed_code",
    )
    text = f"{transaction} {acquired_disposed}".lower()
    if is_buy_text(text):
        return "insider_buy"
    if is_sell_text(text):
        return "insider_sell"
    return None


def congress_event_type(row: pd.Series) -> str | None:
    text = row_text(row, "transactionType", "transaction_type", "type").lower()
    if is_buy_text(text):
        return "congress_buy"
    if is_sell_text(text):
        return "congress_sell"
    return None


def analyst_rating_event_type(row: pd.Series) -> str | None:
    action = row_text(row, "action", "gradeAction", "grade_action").lower()
    if any(token in action for token in ("upgrade", "upgraded", "raised", "initiated")):
        return "analyst_upgrade"
    if any(token in action for token in ("downgrade", "downgraded", "lowered")):
        return "analyst_downgrade"

    old_rating = rating_score(row_text(row, "previousGrade", "previousRating", "oldGrade", "oldRating"))
    new_rating = rating_score(row_text(row, "newGrade", "newRating", "grade", "rating"))
    if old_rating is None or new_rating is None or old_rating == new_rating:
        return None
    return "analyst_upgrade" if new_rating > old_rating else "analyst_downgrade"


def price_target_event_type(row: pd.Series) -> str | None:
    return value_revision_event_type(
        row.get("target_value"),
        row.get("previous_target_value"),
        positive="price_target_raise",
        negative="price_target_cut",
    )


def institutional_event_type(value: Any) -> str | None:
    if pd.isna(value) or float(value) == 0.0:
        return None
    return "institutional_add" if float(value) > 0 else "institutional_reduce"


def capital_press_release_event_type(row: pd.Series) -> str | None:
    text = row_text(row, "title", "text", "content").lower()
    if any(token in text for token in ("buyback", "repurchase", "share repurchase")):
        return "buyback_authorization"
    return None


def dividend_event_type(row: pd.Series) -> str | None:
    return value_revision_event_type(
        row.get("dividend_value"),
        row.get("previous_dividend_value"),
        positive="dividend_increase",
        negative="dividend_cut",
    )


def split_event_type(value: Any) -> str | None:
    if pd.isna(value) or float(value) == 1.0:
        return None
    return "forward_split" if float(value) > 1.0 else "reverse_split"


def earnings_event_type(value: Any) -> str | None:
    if pd.isna(value) or float(value) == 0.0:
        return None
    return "earnings_beat" if float(value) > 0 else "earnings_miss"


def value_revision_event_type(current: Any, previous: Any, *, positive: str, negative: str) -> str | None:
    if pd.isna(current) or pd.isna(previous) or float(current) == float(previous):
        return None
    return positive if float(current) > float(previous) else negative


def rating_score(text: str) -> int | None:
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


def is_buy_text(text: str) -> bool:
    tokens = str(text or "").strip().lower()
    return any(token in tokens for token in ("purchase", "buy", "acquisition", "acquired", " p ", " a "))


def is_sell_text(text: str) -> bool:
    tokens = str(text or "").strip().lower()
    return any(token in tokens for token in ("sale", "sell", "disposition", "disposed", " s ", " d "))


def row_text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return f" {row[column]} "
    return ""


def first_present_column(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns and frame[column].notna().any():
            return frame[column]
    return pd.Series([None] * len(frame), index=frame.index)


def first_numeric_column(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return pd.to_numeric(first_present_column(frame, columns), errors="coerce")


def ensure_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = frame.copy()
    if "symbol" not in out.columns:
        out["symbol"] = symbol
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper().replace({"": symbol})
    return out


def filter_dates(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    date_values = first_present_column(
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
            "as_of",
            "period_ending",
            "periodEnding",
            "fiscalPeriodEnding",
            "fiscal_period_ending",
        ),
    )
    if date_values.isna().all():
        return pd.DataFrame(columns=frame.columns)
    out = frame.copy()
    event_dates = pd.to_datetime(date_values, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    out = out.loc[event_dates.notna()].copy()
    event_dates = event_dates.loc[out.index]
    if start_date is not None:
        out = out.loc[event_dates >= pd.Timestamp(start_date)]
        event_dates = event_dates.loc[out.index]
    if end_date is not None:
        out = out.loc[event_dates <= pd.Timestamp(end_date)]
    return out


def ensure_quarter_date(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "event_date" not in out.columns:
        out["event_date"] = first_present_column(
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


def split_ratio(row: pd.Series) -> float | None:
    ratio = first_numeric_column(pd.DataFrame([row]), ("splitRatio", "split_ratio", "ratio")).iloc[0]
    if pd.notna(ratio):
        return float(ratio)
    numerator = first_numeric_column(pd.DataFrame([row]), ("numerator", "splitFrom", "fromFactor")).iloc[0]
    denominator = first_numeric_column(pd.DataFrame([row]), ("denominator", "splitTo", "toFactor")).iloc[0]
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def normalize_family_frame(frame: pd.DataFrame, *, event_family: str, source: str) -> pd.DataFrame:
    return normalize_event_pairs(
        frame,
        event_family=event_family,
        event_type_col="event_type",
        symbol_col="symbol",
        event_date_col="event_date",
        source=source,
        actor_type_col="actor_type",
        actor_name_col="actor_name",
        actor_role_col="actor_role" if "actor_role" in frame.columns else None,
        actor_chamber_col="actor_chamber" if "actor_chamber" in frame.columns else None,
        actor_firm_col="actor_firm" if "actor_firm" in frame.columns else None,
        actor_title_col="actor_title" if "actor_title" in frame.columns else None,
        strength_col="strength",
        transaction_shares_col="transaction_shares" if "transaction_shares" in frame.columns else None,
        transaction_price_col="transaction_price" if "transaction_price" in frame.columns else None,
        transaction_value_col="transaction_value" if "transaction_value" in frame.columns else None,
        reported_date_col="reported_date" if "reported_date" in frame.columns else None,
        raw_json_col="raw_json",
    )


def combine_names(first: pd.Series, last: pd.Series) -> pd.Series:
    if last.isna().all():
        return first
    values: list[str | None] = []
    for first_value, last_value in zip(first, last, strict=False):
        parts = [str(value).strip() for value in (first_value, last_value) if pd.notna(value)]
        values.append(" ".join(part for part in parts if part) or None)
    return pd.Series(values, index=first.index)


def raw_records(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame.to_dict(orient="records"), index=frame.index)
