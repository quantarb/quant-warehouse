from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.broadcast import broadcast_asof_to_target_index
from quant_warehouse.platforms.data_providers.fmp.sections import LEGACY_FMP_SECTION_MAP


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
    "grades_historical": "grade__",
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
) -> pd.DataFrame:
    section = warehouse_section_for_legacy_key(legacy_section_key)
    if section is None:
        return pd.DataFrame()
    wh = warehouse or get_warehouse()
    return wh.read_fundamentals(
        str(symbol).strip().upper(),
        section=section,
        provider=str(provider or "fmp").strip().lower(),
        start=start_date,
        end=end_date,
    )


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
    if frame is None or frame.empty:
        return []

    keep = {str(value).lower().strip() for value in (keep_fields or [])}
    rows: list[dict[str, Any]] = []
    working = frame.reset_index()
    date_col = working.columns[0]
    working = working.rename(columns={date_col: "date"})
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])
    if filing_lag_days:
        working["date"] = working["date"] + pd.Timedelta(days=int(filing_lag_days))

    for _, series in working.iterrows():
        ts = pd.Timestamp(series["date"]).normalize()
        if pd.isna(ts):
            continue
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
) -> pd.DataFrame:
    rows = warehouse_section_to_payload_rows(
        symbol,
        legacy_section_key,
        prefix=prefix,
        keep_fields=keep_fields,
        filing_lag_days=filing_lag_days,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        warehouse=warehouse,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in {"date", "symbol"}:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return (
        df.sort_values(["date", "symbol"])
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .set_index(["date", "symbol"])
        .sort_index()
    )


def fetch_fundamentals_data(
    symbols: Sequence[str],
    api_key: str = "",
    period: str = "quarter",
    limit: int = 160,
    verbose: bool = True,
    use_filing_lag: bool = True,
    filing_lag_days: int = 45,
    provider: str = "fmp",
) -> pd.DataFrame:
    """Load sparse key-metric and ratio fundamentals from quant-warehouse."""

    del api_key, period, limit
    dfs_per_symbol: list[pd.DataFrame] = []
    for sym in symbols:
        symbol = str(sym).strip().upper()
        if not symbol:
            continue
        df_km = pd.DataFrame(
            warehouse_section_to_payload_rows(
                symbol,
                "key_metrics",
                prefix="km__",
                filing_lag_days=filing_lag_days if use_filing_lag else 0,
                provider=provider,
            )
        )
        df_rt = pd.DataFrame(
            warehouse_section_to_payload_rows(
                symbol,
                "ratios",
                prefix="rt__",
                filing_lag_days=filing_lag_days if use_filing_lag else 0,
                provider=provider,
            )
        )
        if df_km.empty and df_rt.empty:
            continue
        merge_keys = ["date", "symbol", "period"]
        for frame in (df_km, df_rt):
            if frame.empty:
                continue
            for key in merge_keys:
                if key not in frame.columns:
                    frame[key] = np.nan
            frame["symbol"] = frame["symbol"].astype(str)
        if df_km.empty:
            merged = df_rt
        elif df_rt.empty:
            merged = df_km
        else:
            merged = pd.merge(df_km, df_rt, on=merge_keys, how="outer")
        dfs_per_symbol.append(merged)

    if not dfs_per_symbol:
        if verbose:
            print("[fundamentals] WARN: No quant-warehouse fundamentals found.")
        return pd.DataFrame()
    fund_df = pd.concat(dfs_per_symbol, ignore_index=True)
    if "date" in fund_df.columns and "symbol" in fund_df.columns:
        fund_df = fund_df.sort_values(["symbol", "date"]).set_index(["date", "symbol"])
    return _enforce_numeric_features(fund_df)


def broadcast_fundamentals_to_daily(
    fund_df: pd.DataFrame,
    target_daily_index: pd.Index,
) -> pd.DataFrame:
    return broadcast_asof_to_target_index(
        sparse_df=fund_df,
        target_index=target_daily_index,
        on="date",
        by=("symbol",),
    )


def _enforce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.startswith("km__") or col.startswith("rt__"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
