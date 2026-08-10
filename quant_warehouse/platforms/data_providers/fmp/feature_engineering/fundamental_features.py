from __future__ import annotations
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import polars as pl

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.broadcast import broadcast_asof_to_target_index
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.fundamentals import section_prefix, warehouse_section_to_indexed_frame
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet

Frame = pl.DataFrame
SparseLoader = Callable[..., Frame]


def _target(frame: Frame) -> Frame:
    if frame is None: return pl.DataFrame(schema={"date": pl.Datetime, "symbol": pl.String})
    if not {"date", "symbol"}.issubset(frame.columns): raise ValueError("Target panels must contain date and symbol columns")
    date_expr = pl.col("date").str.to_datetime(strict=False) if frame.schema["date"] == pl.String else pl.col("date").cast(pl.Datetime, strict=False)
    return frame.with_columns([date_expr.dt.truncate("1d").alias("date"), pl.col("symbol").cast(pl.String).str.to_uppercase()]).sort(["symbol", "date"])


def default_sparse_loader(symbol_obj: Any, section_key: str, *, prefix: str, keep_fields: Sequence[str] | None = None, filing_lag_days: int = 45) -> Frame:
    symbol = getattr(symbol_obj, "symbol", symbol_obj)
    return warehouse_section_to_indexed_frame(str(symbol), section_key, prefix=prefix, keep_fields=keep_fields, filing_lag_days=filing_lag_days)


def load_section_payload(symbol_obj: Any, section_key: str, *, prefix: str, keep_fields: Sequence[str] | None = None, filing_lag_days: int = 45, sparse_loader: SparseLoader | None = None) -> Frame:
    return (sparse_loader or default_sparse_loader)(symbol_obj, section_key, prefix=prefix, keep_fields=keep_fields, filing_lag_days=filing_lag_days)


def broadcast_sparse(sparse_df: Frame, target_index: Frame) -> Frame:
    target = _target(target_index)
    if sparse_df is None or sparse_df.is_empty(): return target.head(0)
    return broadcast_asof_to_target_index(sparse_df=sparse_df, target_index=target, on="date", by=("symbol",))


def safe_ratio(a: pl.Series, b: pl.Series) -> pl.Series:
    return a.cast(pl.Float64, strict=False) / b.cast(pl.Float64, strict=False).replace(0.0, None)


def first_existing(df: Frame, candidates: Sequence[str]) -> pl.Series | None:
    for column in candidates:
        if column in df.columns: return df[column].cast(pl.Float64, strict=False)
    return None


def target_dates(target_index: Frame) -> pl.Series: return _target(target_index)["date"]


def daily_price_series(df_prices: Frame | None, target_index: Frame, *, price_col: str = "close") -> pl.Series | None:
    if df_prices is None or df_prices.is_empty() or price_col not in df_prices.columns: return None
    joined = _target(target_index).join(df_prices.select(["date", pl.col(price_col).cast(pl.Float64, strict=False).alias("_price")]), on="date", how="left").sort(["symbol", "date"])
    return joined["_price"].fill_null(strategy="forward")


def days_since_last_event(target_dates_index: pl.Series, event_dates: Sequence[Any]) -> pl.Series:
    events = sorted({_day(value) for value in event_dates if _day(value) is not None})
    return pl.Series([float((value - max((event for event in events if event <= value), default=value)).days) if events and any(event <= value for event in events) else None for value in target_dates_index.to_list()])


def days_since_for_target(target_index: Frame, by_date_values: pl.Series) -> pl.Series: return by_date_values


def build_passthrough_section_features(symbol_obj: Any, target_index: Frame, *, section_key: str, prefix: str, filing_lag_days: int = 45, sparse_loader: SparseLoader | None = None, broadcast_to_target: bool = True) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, section_key, prefix=prefix, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    if sparse is None or sparse.is_empty(): return BuiltFeatureSet(df=pl.DataFrame(), feature_cols=[])
    numeric = [column for column in sparse.columns if column.startswith(prefix) and sparse.schema[column].is_numeric()]
    source = sparse.select(["date", "symbol", *numeric]).sort(["symbol", "date"])
    daily = broadcast_sparse(source, target_index) if broadcast_to_target else source
    return BuiltFeatureSet(df=daily, feature_cols=numeric, family_name=section_key, endpoint_name=section_key, source_asset_class="equity")


def add_daily_price_linked_features(daily: Frame, target_index: Frame, *, df_prices: Frame | None = None, market_cap: pl.Series | None = None, share_count_candidates: Sequence[str] = (), price_denominated: Sequence[tuple[Sequence[str], str]] = (), market_cap_denominated: Sequence[tuple[Sequence[str], str]] = (), negate_market_cap_sources: Sequence[str] = ()) -> tuple[Frame, list[str]]:
    out = daily.clone(); added: list[str] = []; close = daily_price_series(df_prices, target_index)
    if market_cap is None and close is not None and share_count_candidates:
        shares = first_existing(out, share_count_candidates); market_cap = shares * close if shares is not None else None
    def add(candidates: Sequence[str], name: str, denominator: pl.Series | None, negate: bool = False) -> None:
        nonlocal out
        source = first_existing(out, candidates)
        if source is None or denominator is None: return
        value = safe_ratio(-source if negate else source, denominator)
        out = out.with_columns(value.alias(name)); added.append(name)
    for candidates, name in price_denominated: add(candidates, name, close)
    for candidates, name in market_cap_denominated: add(candidates, name, market_cap, name in set(negate_market_cap_sources))
    return out, added


def add_growth_adjusted_valuation_features(daily: Frame, *, valuation_frame: Frame | None = None, specs: Sequence[tuple[Sequence[str], Sequence[str], str]] = ()) -> tuple[Frame, list[str]]:
    if valuation_frame is None or valuation_frame.is_empty(): return daily, []
    out = daily.clone(); added: list[str] = []
    for growth_cols, value_cols, name in specs:
        growth, value = first_existing(out, growth_cols), first_existing(valuation_frame, value_cols)
        if growth is None or value is None: continue
        out = out.with_columns(safe_ratio(value, growth.abs().replace(0.0, None)).alias(name)); added.append(name)
    return out, added


def merge_feature_sets(parts: Sequence[BuiltFeatureSet], target_index: Frame | None = None) -> BuiltFeatureSet:
    frames = [part.df for part in parts if part is not None and isinstance(part.df, pl.DataFrame) and not part.df.is_empty()]
    if not frames: return BuiltFeatureSet(df=_target(target_index) if target_index is not None else pl.DataFrame(), feature_cols=[], family_name="combined")
    keys = [key for key in ("date", "symbol") if all(key in frame.columns for frame in frames)]
    merged = frames[0]
    for frame in frames[1:]: merged = merged.join(frame, on=keys, how="full", coalesce=True, suffix="_right")
    if target_index is not None: merged = _target(target_index).join(merged, on=keys, how="left", coalesce=True)
    columns = [column for part in parts for column in (part.feature_cols if part is not None else []) if column in merged.columns]
    return BuiltFeatureSet(df=merged.sort(keys), feature_cols=list(dict.fromkeys(columns)), family_name="combined", endpoint_name="combined")


def _build_section(symbol_obj: Any, target_index: Frame, section: str, prefix: str, *, df_prices: Frame | None = None, market_cap: pl.Series | None = None, filing_lag_days: int = 45, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    built = build_passthrough_section_features(symbol_obj, target_index, section_key=section, prefix=prefix, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    if built.df.is_empty(): return built
    enriched, linked = add_daily_price_linked_features(built.df, target_index, df_prices=df_prices, market_cap=market_cap, share_count_candidates=(f"{prefix}weightedaverageshsout", f"{prefix}weightedaverageshsoutdil"), market_cap_denominated=(((f"{prefix}revenue",), f"{prefix}revenue_to_mcap_daily"),))
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked], family_name=section, endpoint_name=section, source_asset_class="equity")


def build_key_metrics_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _build_section(symbol_obj, target_index, "key_metrics", "km__", df_prices=df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
def build_ratios_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _build_section(symbol_obj, target_index, "ratios", "rt__", df_prices=df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)


def _statement(symbol_obj: Any, target_index: Frame, section: str, prefix: str, df_prices: Frame | None, market_cap: pl.Series | None, filing_lag_days: int, sparse_loader: SparseLoader | None) -> BuiltFeatureSet: return _build_section(symbol_obj, target_index, section, prefix, df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)


def build_income_statement_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _statement(symbol_obj, target_index, "income_statement", "is__", df_prices, None, filing_lag_days, sparse_loader)
def build_income_statement_ttm_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _statement(symbol_obj, target_index, "income_statement_ttm", "is_ttm__", df_prices, None, filing_lag_days, sparse_loader)
def build_cash_flow_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, market_cap: pl.Series | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _statement(symbol_obj, target_index, "cash_flow", "cf__", df_prices, market_cap, filing_lag_days, sparse_loader)
def build_cash_flow_ttm_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, market_cap: pl.Series | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _statement(symbol_obj, target_index, "cash_flow_ttm", "cf_ttm__", df_prices, market_cap, filing_lag_days, sparse_loader)
def build_balance_sheet_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, market_cap: pl.Series | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _statement(symbol_obj, target_index, "balance_sheet", "bs__", df_prices, market_cap, filing_lag_days, sparse_loader)
def build_balance_sheet_ttm_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, market_cap: pl.Series | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _statement(symbol_obj, target_index, "balance_sheet_ttm", "bs_ttm__", df_prices, market_cap, filing_lag_days, sparse_loader)


def _growth(symbol_obj: Any, target_index: Frame, section: str, prefix: str, valuation_frame: Frame | None, filing_lag_days: int, sparse_loader: SparseLoader | None) -> BuiltFeatureSet: return build_passthrough_section_features(symbol_obj, target_index, section_key=section, prefix=prefix, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
def build_income_statement_growth_features(symbol_obj: Any, target_index: Frame, valuation_frame: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _growth(symbol_obj, target_index, "income_growth", "is_growth__", valuation_frame, filing_lag_days, sparse_loader)
def build_cash_flow_growth_features(symbol_obj: Any, target_index: Frame, valuation_frame: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _growth(symbol_obj, target_index, "cash_growth", "cf_growth__", valuation_frame, filing_lag_days, sparse_loader)
def build_balance_sheet_growth_features(symbol_obj: Any, target_index: Frame, valuation_frame: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return _growth(symbol_obj, target_index, "balance_growth", "bs_growth__", valuation_frame, filing_lag_days, sparse_loader)
def build_financial_growth_features(symbol_obj: Any, target_index: Frame, valuation_frame: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return merge_feature_sets([build_income_statement_growth_features(symbol_obj, target_index, valuation_frame, filing_lag_days, sparse_loader=sparse_loader), build_cash_flow_growth_features(symbol_obj, target_index, valuation_frame, filing_lag_days, sparse_loader=sparse_loader), build_balance_sheet_growth_features(symbol_obj, target_index, valuation_frame, filing_lag_days, sparse_loader=sparse_loader)], target_index)
def build_fundamental_change_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return merge_feature_sets([build_key_metrics_features(symbol_obj, target_index, df_prices, filing_lag_days, sparse_loader=sparse_loader), build_ratios_features(symbol_obj, target_index, df_prices, filing_lag_days, sparse_loader=sparse_loader)], target_index)
def build_statement_quality_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return merge_feature_sets([build_income_statement_features(symbol_obj, target_index, df_prices, filing_lag_days, sparse_loader=sparse_loader), build_cash_flow_features(symbol_obj, target_index, df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader), build_balance_sheet_features(symbol_obj, target_index, df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)], target_index)
def build_ttm_financial_statement_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return merge_feature_sets([build_income_statement_ttm_features(symbol_obj, target_index, df_prices, filing_lag_days, sparse_loader=sparse_loader), build_cash_flow_ttm_features(symbol_obj, target_index, df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader), build_balance_sheet_ttm_features(symbol_obj, target_index, df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)], target_index)
def build_earnings_features(symbol_obj: Any, target_index: Frame, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return build_passthrough_section_features(symbol_obj, target_index, section_key="earnings", prefix="evt__", sparse_loader=sparse_loader)
def build_analyst_estimates_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, market_cap: pl.Series | None = None, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return build_passthrough_section_features(symbol_obj, target_index, section_key="analyst_estimates", prefix="ae__", sparse_loader=sparse_loader)
def build_ratings_historical_features(symbol_obj: Any, target_index: Frame, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return build_passthrough_section_features(symbol_obj, target_index, section_key="ratings_historical", prefix="rating__", sparse_loader=sparse_loader)
def build_event_features(symbol_obj: Any, target_index: Frame, df_prices: Frame | None = None, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return merge_feature_sets([build_earnings_features(symbol_obj, target_index, sparse_loader=sparse_loader), build_analyst_estimates_features(symbol_obj, target_index, df_prices, sparse_loader=sparse_loader), build_ratings_historical_features(symbol_obj, target_index, sparse_loader=sparse_loader)], target_index)
def build_insider_trading_features(symbol_obj: Any, target_index: Frame, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return build_passthrough_section_features(symbol_obj, target_index, section_key="insider_trading", prefix="insider__", sparse_loader=sparse_loader)
def build_positions_summary_features(symbol_obj: Any, target_index: Frame, *, positions_source_loader: Callable[[Any], Frame] | None = None) -> BuiltFeatureSet: return BuiltFeatureSet(df=pl.DataFrame(), feature_cols=[])
def build_ownership_features(symbol_obj: Any, target_index: Frame, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet: return merge_feature_sets([build_insider_trading_features(symbol_obj, target_index, sparse_loader=sparse_loader)], target_index)


def _day(value: Any) -> datetime | None:
    if value is None: return None
    if isinstance(value, datetime): return value.replace(hour=0, minute=0, second=0, microsecond=0)
    try: return datetime.fromisoformat(str(value)[:10])
    except ValueError: return None
