from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

import polars as pl

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.broadcast import broadcast_asof_to_target_index
from quant_warehouse.platforms.data_providers.fmp.sections import LEGACY_FMP_SECTION_MAP

Frame = pl.DataFrame


SECTION_PREFIXES: dict[str, str] = {
    "key_metrics": "km__",
    "ratios": "rt__",
    "income_statement": "is__",
    "income_statement_ttm": "is_ttm__",
    "income_statement_growth": "isg__",
    "cash_flow": "cf__",
    "cash_flow_ttm": "cf_ttm__",
    "cash_flow_growth": "cfg__",
    "balance_sheet": "bs__",
    "balance_sheet_ttm": "bs_ttm__",
    "balance_sheet_growth": "bsg__",
    "financial_growth": "fg__",
    "earnings": "earn__",
    "analyst_estimates": "ae__",
    "ratings_historical": "rating__",
    "insider_trading": "insider__",
    "positions_summary": "ps__",
}


def section_prefix(section_key: str) -> str:
    return SECTION_PREFIXES.get(str(section_key), f"{section_key}__")


@lru_cache(maxsize=1)
def get_warehouse():
    from quant_warehouse import Warehouse

    return Warehouse()


def warehouse_section_for_legacy_key(section_key: str) -> str | None:
    key = str(section_key).strip()
    return LEGACY_FMP_SECTION_MAP.get(key)


def warehouse_sections_for_legacy_keys(
    legacy_section_keys: Iterable[str],
) -> tuple[str, ...]:
    mapped: list[str] = []
    seen: set[str] = set()
    for legacy_key in legacy_section_keys:
        key = str(legacy_key or "").strip()
        if not key or key == "prices_div_adj":
            continue
        warehouse_key = warehouse_section_for_legacy_key(key)
        if not warehouse_key or warehouse_key in seen:
            continue
        seen.add(warehouse_key)
        mapped.append(warehouse_key)
    return tuple(mapped)


def unsupported_legacy_sections_for_refresh(legacy_section_keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        str(key).strip()
        for key in legacy_section_keys
        if str(key).strip()
        and str(key).strip() != "prices_div_adj"
        and LEGACY_FMP_SECTION_MAP.get(str(key).strip()) is None
    )


def warehouse_sections_for_refresh(legacy_section_keys: Iterable[str]) -> tuple[str, ...]:
    return warehouse_sections_for_legacy_keys(legacy_section_keys)


def load_warehouse_fundamental_frame(
    symbol: str,
    legacy_section_key: str,
    *,
    provider: str = "fmp",
    start_date: str | None = None,
    end_date: str | None = None,
    warehouse=None,
) -> pl.DataFrame:
    section = warehouse_section_for_legacy_key(legacy_section_key)
    if section is None:
        return pl.DataFrame()
    wh = warehouse or get_warehouse()
    read_kwargs = {
        "section": section,
        "provider": str(provider or "fmp").strip().lower(),
        "start": start_date,
        "end": end_date,
    }
    try:
        frame = wh.read_fundamentals(
            str(symbol).strip().upper(),
            **read_kwargs,
            output_format="polars",
        )
    except TypeError:
        # Compatibility for injected legacy warehouse doubles.
        frame = wh.read_fundamentals(str(symbol).strip().upper(), **read_kwargs)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("Warehouse fundamentals must return a Polars DataFrame")
    return frame


def warehouse_section_to_payload_rows(
    symbol: str,
    legacy_section_key: str,
    *,
    prefix: str,
    keep_fields: Iterable[str] | None = None,
    filing_lag_days: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str = "fmp",
    warehouse=None,
) -> list[dict[str, Any]]:
    frame = load_warehouse_fundamental_frame(
        symbol,
        legacy_section_key,
        provider=provider,
        start_date=start_date,
        end_date=end_date,
        warehouse=warehouse,
    )
    if frame is None or frame.is_empty():
        return []

    keep = {str(value).lower().strip() for value in (keep_fields or [])}
    rows: list[dict[str, Any]] = []
    working = frame
    date_col = "date" if "date" in working.columns else working.columns[0]
    if date_col != "date":
        working = working.rename({date_col: "date"})
    working = working.with_columns(pl.col("date").cast(pl.Datetime, strict=False)).drop_nulls("date")
    # Preserve the warehouse observation date exactly.  The argument remains
    # for compatibility with older callers, but feature construction never
    # applies an implicit filing/reporting lag.

    for series in working.iter_rows(named=True):
        ts = series["date"]
        row: dict[str, Any] = {
            "date": ts,
            "symbol": str(symbol).strip().upper(),
        }
        for col, value in series.items():
            if col in {"date", "symbol"}:
                continue
            key = str(col).lower().strip()
            if keep and key not in keep:
                continue
            row[f"{prefix}{key}"] = value
        rows.append(row)
    return rows


def warehouse_section_to_indexed_frame(
    symbol: str,
    legacy_section_key: str,
    *,
    prefix: str,
    keep_fields: Iterable[str] | None = None,
    filing_lag_days: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str = "fmp",
    warehouse=None,
) -> pl.DataFrame:
    raw = load_warehouse_fundamental_frame(
        symbol,
        legacy_section_key,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        warehouse=warehouse,
    )
    if raw is None or raw.is_empty():
        return pl.DataFrame()
    date_column = "date" if "date" in raw.columns else next((column for column in raw.columns if column in {"period_ending", "as_of", "filing_date", "report_date"}), None)
    if date_column is None:
        return pl.DataFrame()
    out = raw.rename({date_column: "date"}) if date_column != "date" else raw
    if "symbol" not in out.columns:
        out = out.with_columns(pl.lit(str(symbol).strip().upper()).alias("symbol"))
    rename = {column: f"{prefix}{str(column).lower().strip()}" for column in out.columns if column not in {"date", "symbol"} and not str(column).startswith(prefix)}
    if rename:
        out = out.rename(rename)
    numeric = [column for column in out.columns if column not in {"date", "symbol"}]
    if numeric:
        out = out.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in numeric)
    return out.unique(["date", "symbol"], keep="last").sort(["date", "symbol"])


def fetch_fundamentals_data(
    symbols: Sequence[str],
    api_key: str = "",
    period: str = "quarter",
    limit: int = 160,
    verbose: bool = True,
    use_filing_lag: bool = False,
    filing_lag_days: int = 0,
    provider: str = "fmp",
) -> pl.DataFrame:
    """Load sparse key-metric and ratio fundamentals from quant-warehouse."""

    del api_key, period, limit
    polars_frames: list[pl.DataFrame] = []
    for sym in symbols:
        symbol = str(sym).strip().upper()
        if not symbol:
            continue
        parts: list[pl.DataFrame] = []
        for section_key, prefix in (("key_metrics", "km__"), ("ratios", "rt__")):
            frame = warehouse_section_to_indexed_frame(
                symbol,
                section_key,
                prefix=prefix,
                filing_lag_days=filing_lag_days if use_filing_lag else 0,
                provider=provider,
            )
            if not frame.is_empty():
                parts.append(frame)
        if not parts:
            continue
        merged = pl.concat(parts, how="diagonal_relaxed")
        if "period" not in merged.columns:
            merged = merged.with_columns(pl.lit(None, dtype=pl.String).alias("period"))
        polars_frames.append(merged)
    if not polars_frames:
        if verbose:
            print("[fundamentals] WARN: No quant-warehouse fundamentals found.")
        return pl.DataFrame()
    out = pl.concat(polars_frames, how="diagonal_relaxed")
    return out.sort([column for column in ("symbol", "date") if column in out.columns])
    if not polars_frames:
        if verbose:
            print("[fundamentals] WARN: No quant-warehouse fundamentals found.")
        return pl.DataFrame()
    return pl.concat(polars_frames, how="diagonal_relaxed").sort(["symbol", "date"])


def broadcast_fundamentals_to_daily(
    fund_df: pl.DataFrame,
    target_daily_index: pl.DataFrame,
) -> pl.DataFrame:
    return broadcast_asof_to_target_index(
        sparse_df=fund_df,
        target_index=target_daily_index,
        on="date",
        by=("symbol",),
    )
