from __future__ import annotations

import re
from datetime import datetime

import polars as pl

from quant_warehouse.warehouse.sections import MIN_HISTORICAL_DATE

INDEX_CANDIDATES = (
    "period_ending", "date", "as_of", "ex_dividend_date", "payment_date",
    "record_date", "announcement_date", "filing_date", "accepted_date",
    "report_date", "split_date", "transaction_date", "published_date",
    "disclosure_date", "ipo_date",
)
PRICE_INDEX_CANDIDATES = ("date",)
PRICE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open",), "high": ("high",), "low": ("low",),
    "close": ("close",), "volume": ("volume", "vol"),
    "adj_open": ("adj_open", "adjopen"), "adj_high": ("adj_high", "adjhigh"),
    "adj_low": ("adj_low", "adjlow"),
    "adj_close": ("adj_close", "adjclose", "adj_close_price"),
}
METADATA_COLUMNS = {
    "symbol", "cik", "link", "final_link", "reported_currency", "currency",
    "fiscal_year", "calendar_year", "period",
}
PANEL_DIMENSION_COLUMNS = frozenset({
    "business_line", "region", "cusip", "isin", "lei", "name", "title",
    "symbol", "fiscal_period", "sector", "country", "industry",
})


def symbol_provider_key(symbol: str, provider: str) -> str:
    return f"{symbol.strip().upper()}__{provider.strip().lower()}"


def _to_snake(name: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name)).lower().strip()


def _date_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    if frame.schema[column] == pl.String:
        return pl.col(column).str.to_datetime(strict=False)
    return pl.col(column).cast(pl.Datetime, strict=False)


def _floor_expr(min_date: str | None) -> datetime:
    return datetime.fromisoformat(min_date or MIN_HISTORICAL_DATE)


def _finite_numeric(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    if not columns:
        return frame
    return frame.with_columns(
        pl.when(pl.col(column).cast(pl.Float64, strict=False).is_finite())
        .then(pl.col(column).cast(pl.Float64, strict=False))
        .otherwise(None)
        .alias(column)
        for column in columns
    )


def clip_to_min_historical_date(frame: pl.DataFrame, *, min_date: str = MIN_HISTORICAL_DATE) -> pl.DataFrame:
    """Drop rows before the warehouse floor using an explicit date column."""
    if frame.is_empty():
        return frame
    column = next((name for name in INDEX_CANDIDATES if name in frame.columns), None)
    if column is None:
        return frame
    out = frame.with_columns(_date_expr(frame, column).alias(column)).drop_nulls(column)
    return out.filter(pl.col(column) >= pl.lit(_floor_expr(min_date))).sort(column)


def _pick_index_column(frame: pl.DataFrame) -> str | None:
    for column in INDEX_CANDIDATES:
        if column in frame.columns and frame[column].drop_nulls().len():
            return column
    return None


def _normalize_columns(frame: pl.DataFrame, *, provider: str, prefix: str | None) -> tuple[pl.DataFrame, str] | None:
    if frame.is_empty():
        return frame, "date"
    source = frame.clone()
    index_col = _pick_index_column(source)
    if index_col is None:
        return None
    source = source.with_columns(_date_expr(source, index_col).alias(index_col)).drop_nulls(index_col)
    rename: dict[str, str] = {}
    for column in source.columns:
        if column == index_col or column in METADATA_COLUMNS:
            continue
        base = _to_snake(column)
        rename[column] = f"{prefix}__{base}" if prefix else base
    out = source.rename(rename)
    keep = [index_col, *rename.values()]
    out = out.select([column for column in keep if column in out.columns])
    numeric = [column for column in out.columns if column != index_col and column not in PANEL_DIMENSION_COLUMNS]
    out = _finite_numeric(out, numeric)
    return out.unique([index_col], keep="last", maintain_order=True).sort(index_col), index_col


def normalize_vendor_frame(
    df: pl.DataFrame, *, provider: str, vendor_only_prefix: str | None = None,
    min_date: str | None = None,
) -> pl.DataFrame:
    normalized = _normalize_columns(df, provider=provider, prefix=vendor_only_prefix)
    if normalized is None:
        return pl.DataFrame()
    out, _ = normalized
    return clip_to_min_historical_date(out, min_date=min_date or MIN_HISTORICAL_DATE)


def normalize_panel_frame(
    df: pl.DataFrame, *, provider: str, vendor_only_prefix: str | None = None,
    min_date: str | None = None,
) -> pl.DataFrame:
    normalized = _normalize_columns(df, provider=provider, prefix=vendor_only_prefix)
    if normalized is None:
        return pl.DataFrame()
    out, index_col = normalized
    keys = [index_col, *(column for column in PANEL_DIMENSION_COLUMNS if column in out.columns)]
    return clip_to_min_historical_date(
        out.unique(keys, keep="last", maintain_order=True), min_date=min_date or MIN_HISTORICAL_DATE
    )


def coerce_object_dates(frame: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    for column, dtype in frame.schema.items():
        name = str(column).lower()
        if dtype == pl.Date:
            expressions.append(pl.col(column).cast(pl.Datetime).alias(column))
        elif dtype == pl.String and ("date" in name or name.endswith("_at")):
            expressions.append(pl.col(column).str.to_datetime(strict=False).alias(column))
    return frame.with_columns(expressions) if expressions else frame


def _coerce_object_strings(frame: pl.DataFrame) -> pl.DataFrame:
    return frame


def normalize_dated_snapshot_frame(df: pl.DataFrame, *, section: str) -> pl.DataFrame:
    if df.is_empty():
        return df
    out = normalize_snapshot_frame(df)
    if out.is_empty():
        return out
    if section == "etf_holdings" and "updated" in out.columns:
        out = out.with_columns(_date_expr(out, "updated").alias("as_of")).drop("updated")
    elif section in {"ownership_institutional", "ownership_share_statistics"} and "date" in out.columns:
        out = out.with_columns(_date_expr(out, "date").dt.replace_time_zone(None).dt.truncate("1d").alias("as_of"))
    else:
        out = out.with_columns(pl.lit(datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).alias("as_of"))
    dimensions = [column for column in ("cusip", "isin", "name", "sector", "country", "symbol", "title", "cik") if column in out.columns]
    return out.unique(["as_of", *dimensions], keep="last", maintain_order=True).sort("as_of")


def normalize_etf_composition_frame(df: pl.DataFrame, *, section: str) -> pl.DataFrame:
    return normalize_dated_snapshot_frame(df, section=section)


def normalize_snapshot_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    rename = {column: _to_snake(column) for column in df.columns if column not in METADATA_COLUMNS}
    out = df.rename(rename)
    return _finite_numeric(out, [column for column in out.columns if column not in PANEL_DIMENSION_COLUMNS and out.schema[column].is_numeric()])


def _resolve_price_column(name: str) -> str | None:
    base = _to_snake(name)
    return next((canonical for canonical, aliases in PRICE_COLUMN_ALIASES.items() if base in aliases), None)


def normalize_prices(df: pl.DataFrame, *, provider: str, min_date: str | None = None) -> pl.DataFrame:
    if df.is_empty():
        return df
    index_col = next((column for column in PRICE_INDEX_CANDIDATES if column in df.columns), None)
    if index_col is None:
        return pl.DataFrame()
    out = df.clone().with_columns(_date_expr(df, index_col).alias("date")).drop_nulls("date")
    rename: dict[str, str] = {}
    for column in out.columns:
        if column in {index_col, "date"} or column in METADATA_COLUMNS:
            continue
        rename[column] = _resolve_price_column(column) or f"{provider}__{_to_snake(column)}"
    out = out.rename(rename)
    if index_col != "date" and index_col in out.columns:
        out = out.drop(index_col)
    for raw, adjusted in (("open", "adj_open"), ("high", "adj_high"), ("low", "adj_low"), ("close", "adj_close")):
        if raw not in out.columns and adjusted in out.columns:
            out = out.with_columns(pl.col(adjusted).alias(raw))
    out = _finite_numeric(out, [column for column in out.columns if column != "date"])
    return out.unique("date", keep="last", maintain_order=True).sort("date").filter(
        pl.col("date") >= pl.lit(_floor_expr(min_date or MIN_HISTORICAL_DATE))
    )
