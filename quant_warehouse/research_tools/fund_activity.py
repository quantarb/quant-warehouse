"""Build fund and institutional activity events with Polars frames."""
from __future__ import annotations

from typing import Iterable
import polars as pl

FUND_ACTIVITY_TARGET_FAMILY = "fund_activity"
FUND_ACTIVITY_TARGET_FAMILIES = ("fund_activity.etf_buy", "fund_activity.mutual_fund_buy",
    "fund_activity.institutional_buy", "fund_activity.add", "fund_activity.reduce", "fund_activity.exit")
HOLDER_ACTIVITY_TARGET_FAMILIES = ("holder_activity.buy", "holder_activity.add", "holder_activity.reduce", "holder_activity.exit")


def _date_expr(frame: pl.DataFrame, names: Iterable[str]) -> pl.Expr:
    for name in names:
        if name in frame.columns:
            expr = (pl.col(name).str.to_datetime(strict=False) if frame.schema[name] == pl.String
                    else pl.col(name).cast(pl.Datetime, strict=False))
            return expr.dt.truncate("1d")
    return pl.lit(None, dtype=pl.Datetime)


def _numeric_expr(frame: pl.DataFrame, names: Iterable[str]) -> pl.Expr:
    for name in names:
        if name in frame.columns:
            return pl.col(name).cast(pl.Float64, strict=False).fill_null(0.0)
    return pl.lit(0.0)


def _event_rows(frame: pl.DataFrame, *, family: str, symbol_column: str = "symbol",
                text_columns: tuple[str, ...] = ()) -> pl.DataFrame:
    if frame.is_empty() or symbol_column not in frame.columns:
        return pl.DataFrame()
    out = frame.with_columns(
        pl.col(symbol_column).cast(pl.String, strict=False).str.strip_chars().str.to_uppercase().alias("symbol"),
        _date_expr(frame, ("date", "as_of", "updated", "disclosure_date")).alias("event_date"),
    ).filter((pl.col("symbol") != "") & pl.col("event_date").is_not_null())
    if out.is_empty():
        return pl.DataFrame()
    out = out.with_columns(pl.col("event_date").alias("date"), pl.lit(family).alias("target_family"),
                           _numeric_expr(out, ("signal_value", "value", "weight", "shares")).alias("signal_value"))
    keep = ["symbol", "date", "event_date", "target_family", "signal_value"]
    keep.extend(c for c in text_columns if c in out.columns)
    return out.select(keep)


def build_institutional_activity_events(summary: pl.DataFrame) -> pl.DataFrame:
    if summary.is_empty():
        return pl.DataFrame()
    work = summary.rename({src: dst for src, dst in {
        "newPositions": "new_positions", "increasedPositions": "increased_positions",
        "reducedPositions": "reduced_positions", "closedPositions": "closed_positions",
    }.items() if src in summary.columns and dst not in summary.columns})
    outputs: list[pl.DataFrame] = []
    for family, columns in (("fund_activity.institutional_buy", ("new_positions", "increased_positions")),
                            ("fund_activity.add", ("new_positions", "increased_positions")),
                            ("fund_activity.reduce", ("reduced_positions",)),
                            ("fund_activity.exit", ("closed_positions",))):
        signal = sum((_numeric_expr(work, (column,)) for column in columns), pl.lit(0.0))
        rows = _event_rows(work.with_columns((signal > 0).cast(pl.Float32).alias("signal_value"))
                           .filter(pl.col("signal_value") > 0), family=family, text_columns=("cik",))
        if not rows.is_empty():
            outputs.append(rows)
    return pl.concat(outputs, how="diagonal_relaxed") if outputs else pl.DataFrame()


def build_fund_holding_activity_events(holdings: pl.DataFrame, *, fund_type: str = "etf",
                                       fund_column: str = "fund_symbol", security_column: str = "symbol") -> pl.DataFrame:
    if holdings.is_empty() or not {fund_column, security_column}.issubset(holdings.columns):
        return pl.DataFrame()
    work = holdings.rename({fund_column: "fund_id", security_column: "symbol"}).with_columns(
        pl.col("fund_id").cast(pl.String, strict=False).str.strip_chars().str.to_uppercase(),
        pl.col("symbol").cast(pl.String, strict=False).str.strip_chars().str.to_uppercase(),
        _date_expr(holdings, ("date", "as_of", "updated", "disclosure_date")).alias("date"),
        _numeric_expr(holdings, ("shares", "units", "value", "weight")).alias("position"),
    ).filter((pl.col("fund_id") != "") & (pl.col("symbol") != "") & pl.col("date").is_not_null())
    if work.is_empty():
        return pl.DataFrame()
    work = work.sort(["fund_id", "symbol", "date"]).with_columns(
        pl.col("position").shift(1).over(["fund_id", "symbol"]).fill_null(0.0).alias("previous_position")
    ).with_columns((pl.col("position") - pl.col("previous_position")).alias("delta"))
    type_name = str(fund_type).strip().lower().replace("-", "_")
    buy_family = f"fund_activity.{type_name}_buy" if type_name in {"etf", "mutual_fund"} else "fund_activity.institutional_buy"
    outputs: list[pl.DataFrame] = []
    for family, mask in ((buy_family, pl.col("delta") > 0),
                         ("fund_activity.add", (pl.col("delta") > 0) & (pl.col("previous_position") > 0)),
                         ("fund_activity.reduce", (pl.col("delta") < 0) & (pl.col("position") > 0)),
                         ("fund_activity.exit", (pl.col("position") == 0) & (pl.col("previous_position") > 0))):
        rows = _event_rows(work.filter(mask).with_columns(pl.col("delta").abs().cast(pl.Float32).alias("signal_value")),
                           family=family, text_columns=("fund_id",))
        if not rows.is_empty():
            outputs.append(rows)
    return pl.concat(outputs, how="diagonal_relaxed") if outputs else pl.DataFrame()


def build_holder_activity_events(analytics: pl.DataFrame) -> pl.DataFrame:
    if analytics.is_empty() or "symbol" not in analytics.columns:
        return pl.DataFrame()
    work = analytics.rename({src: dst for src, dst in {
        "investorName": "investor_name", "changeInSharesNumber": "change_in_shares",
        "isNew": "is_new", "isSoldOut": "is_sold_out",
    }.items() if src in analytics.columns and dst not in analytics.columns})
    work = work.with_columns(
        pl.col("symbol").cast(pl.String, strict=False).str.strip_chars().str.to_uppercase(),
        (pl.col("cik").cast(pl.String, strict=False).str.strip_chars() if "cik" in work.columns else pl.lit("")).alias("holder_id"),
        (pl.col("investor_name").cast(pl.String, strict=False).str.strip_chars() if "investor_name" in work.columns else pl.lit("")).alias("holder_name"),
        _numeric_expr(work, ("change_in_shares",)).alias("change_value"),
        (pl.col("is_new").cast(pl.String, strict=False).str.to_lowercase().is_in(["true", "1", "yes"]) if "is_new" in work.columns else pl.lit(False)).alias("is_new_flag"),
        (pl.col("is_sold_out").cast(pl.String, strict=False).str.to_lowercase().is_in(["true", "1", "yes"]) if "is_sold_out" in work.columns else pl.lit(False)).alias("is_exit_flag"),
    )
    change = pl.col("change_value")
    outputs: list[pl.DataFrame] = []
    for family, mask in (("holder_activity.buy", pl.col("is_new_flag") | (change > 0)),
                         ("holder_activity.add", (change > 0) & ~pl.col("is_new_flag")),
                         ("holder_activity.reduce", (change < 0) & ~pl.col("is_exit_flag")),
                         ("holder_activity.exit", pl.col("is_exit_flag"))):
        rows = _event_rows(work.filter(mask).with_columns(change.abs().cast(pl.Float32).alias("signal_value")),
                           family=family, text_columns=("holder_id", "holder_name"))
        if not rows.is_empty():
            outputs.append(rows)
    return pl.concat(outputs, how="diagonal_relaxed") if outputs else pl.DataFrame()
