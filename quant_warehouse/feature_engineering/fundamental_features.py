from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quant_warehouse.feature_engineering.fundamentals import (
    section_prefix,
    warehouse_section_to_indexed_frame,
)
from quant_warehouse.feature_engineering.specs import BuiltFeatureSet
from quant_warehouse.feature_engineering.broadcast import broadcast_asof_to_target_index


SparseLoader = Callable[..., pd.DataFrame]
PositionsSourceLoader = Callable[[Any], pd.DataFrame]


def default_sparse_loader(
    symbol_obj: Any,
    section_key: str,
    *,
    prefix: str,
    keep_fields: Sequence[str] | None = None,
    filing_lag_days: int = 45,
) -> pd.DataFrame:
    symbol = getattr(symbol_obj, "symbol", symbol_obj)
    return warehouse_section_to_indexed_frame(
        str(symbol),
        section_key,
        prefix=prefix,
        keep_fields=keep_fields,
        filing_lag_days=filing_lag_days,
    )


def load_section_payload(
    symbol_obj: Any,
    section_key: str,
    *,
    prefix: str,
    keep_fields: Sequence[str] | None = None,
    filing_lag_days: int = 45,
    sparse_loader: SparseLoader | None = None,
) -> pd.DataFrame:
    loader = sparse_loader or default_sparse_loader
    return loader(
        symbol_obj,
        section_key,
        prefix=prefix,
        keep_fields=keep_fields,
        filing_lag_days=filing_lag_days,
    )


def broadcast_sparse(sparse_df: pd.DataFrame, target_index: pd.MultiIndex) -> pd.DataFrame:
    if sparse_df.empty:
        return pd.DataFrame(index=target_index)
    return broadcast_asof_to_target_index(sparse_df=sparse_df, target_index=target_index, on="date", by=("symbol",))


def safe_ratio(a, b):
    if a is None or b is None:
        return np.nan
    if not isinstance(a, pd.Series):
        a = pd.Series(a)
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    denom = pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    numer = pd.to_numeric(a, errors="coerce")
    return numer / denom


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def target_dates(target_index: pd.MultiIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date"))).normalize()


def daily_price_series(
    df_prices: pd.DataFrame | None,
    target_index: pd.MultiIndex,
    *,
    price_col: str = "close",
) -> pd.Series | None:
    if df_prices is None or df_prices.empty or price_col not in df_prices.columns:
        return None
    close = pd.to_numeric(df_prices[price_col], errors="coerce").sort_index()
    if isinstance(close.index, pd.MultiIndex):
        return close.reindex(target_index)
    aligned = close.reindex(target_dates(target_index), method="ffill")
    return pd.Series(aligned.to_numpy(), index=target_index, dtype="float64")


def days_since_last_event(target_dates_index: pd.DatetimeIndex, event_dates: Sequence[Any]) -> pd.Series:
    event_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(event_dates)), errors="coerce").dropna()).normalize().sort_values().unique()
    if len(event_index) == 0:
        return pd.Series(np.nan, index=target_dates_index)
    last_seen = np.searchsorted(event_index, target_dates_index.values.astype("datetime64[ns]"), side="right") - 1
    out = np.full(len(target_dates_index), np.nan, dtype=float)
    valid = last_seen >= 0
    if valid.any():
        prior = event_index[last_seen[valid]]
        out[valid] = (target_dates_index[valid] - prior).days.astype(float)
    return pd.Series(out, index=target_dates_index)


def days_since_for_target(target_index: pd.MultiIndex, by_date_values: pd.Series) -> pd.Series:
    values = by_date_values.reindex(target_dates(target_index))
    return pd.Series(values.to_numpy(), index=target_index)


def build_passthrough_section_features(
    symbol_obj: Any,
    target_index: pd.MultiIndex,
    *,
    section_key: str,
    prefix: str,
    filing_lag_days: int = 45,
    sparse_loader: SparseLoader | None = None,
) -> BuiltFeatureSet:
    sparse = load_section_payload(
        symbol_obj,
        section_key,
        prefix=prefix,
        keep_fields=None,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
    )
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    numeric_cols = [c for c in sparse.columns if c.startswith(prefix) and pd.api.types.is_numeric_dtype(sparse[c])]
    if not numeric_cols:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    daily = broadcast_sparse(sparse[numeric_cols].sort_index(), target_index)
    return BuiltFeatureSet(df=daily, feature_cols=[c for c in daily.columns if c.startswith(prefix)])


def add_daily_price_linked_features(
    daily: pd.DataFrame,
    target_index: pd.MultiIndex,
    *,
    df_prices: pd.DataFrame | None = None,
    market_cap: pd.Series | None = None,
    share_count_candidates: Sequence[str] = (),
    price_denominated: Sequence[tuple[Sequence[str], str]] = (),
    market_cap_denominated: Sequence[tuple[Sequence[str], str]] = (),
    negate_market_cap_sources: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    if daily.empty:
        return daily, []
    out = daily.copy()
    close = daily_price_series(df_prices, target_index)
    if market_cap is None and close is not None and share_count_candidates:
        shares = first_existing(out, share_count_candidates)
        if shares is not None:
            market_cap = shares.reindex(out.index) * close.reindex(out.index)
    elif market_cap is not None:
        market_cap = pd.to_numeric(market_cap, errors="coerce").reindex(out.index)

    added: list[str] = []

    def _add(candidates: Sequence[str], output_col: str, denominator: pd.Series | None, *, negate: bool = False) -> None:
        if denominator is None:
            return
        source = first_existing(out, candidates)
        if source is None:
            return
        values = -source if negate else source
        linked = safe_ratio(values.reindex(out.index), denominator.reindex(out.index)).replace([np.inf, -np.inf], np.nan)
        if linked.notna().any():
            out[output_col] = linked
            added.append(output_col)

    for candidates, output_col in price_denominated:
        _add(candidates, output_col, close)
    negate_set = {str(value).strip() for value in negate_market_cap_sources}
    for candidates, output_col in market_cap_denominated:
        _add(candidates, output_col, market_cap, negate=str(output_col) in negate_set)
    return out, added


def _growth_as_percent(series: pd.Series) -> pd.Series:
    growth = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = growth.dropna().abs()
    if not valid.empty and float(valid.median()) <= 2.0:
        growth = growth * 100.0
    return growth


def add_growth_adjusted_valuation_features(
    daily: pd.DataFrame,
    *,
    valuation_frame: pd.DataFrame | None = None,
    specs: Sequence[tuple[Sequence[str], Sequence[str], str]] = (),
) -> tuple[pd.DataFrame, list[str]]:
    if daily.empty or valuation_frame is None or valuation_frame.empty:
        return daily, []
    out = daily.copy()
    valuation = valuation_frame.reindex(out.index)
    added: list[str] = []
    for growth_candidates, valuation_candidates, output_col in specs:
        growth = first_existing(out, growth_candidates)
        valuation_series = first_existing(valuation, valuation_candidates)
        if growth is None or valuation_series is None:
            continue
        growth_pct = _growth_as_percent(growth).where(lambda s: s > 0.0)
        values = safe_ratio(valuation_series.reindex(out.index), growth_pct.reindex(out.index)).replace([np.inf, -np.inf], np.nan)
        if values.notna().any():
            out[output_col] = values
            added.append(output_col)
    return out, added


def merge_feature_sets(parts: Sequence[BuiltFeatureSet], target_index: pd.MultiIndex) -> BuiltFeatureSet:
    frames = [part.df for part in parts if part is not None and not part.df.empty]
    if not frames:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    merged = pd.concat(frames, axis=1).reindex(target_index)
    if merged.columns.has_duplicates:
        merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]
    cols: list[str] = []
    for part in parts:
        for col in getattr(part, "feature_cols", []) or []:
            if col in merged.columns and col not in cols:
                cols.append(col)
    return BuiltFeatureSet(df=merged, feature_cols=cols)


def build_key_metrics_features(
    symbol_obj: Any,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
    *,
    sparse_loader: SparseLoader | None = None,
) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "key_metrics", prefix="km__", filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    value_cols = [c for c in sparse.columns if c.startswith("km__") and pd.api.types.is_numeric_dtype(sparse[c])]
    if not value_cols:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    daily = broadcast_sparse(sparse[value_cols].sort_index(), target_index)
    daily_market_cap = _infer_daily_market_cap_from_sparse(work, target_index, df_prices, "km__")
    if daily_market_cap is not None:
        daily["km__marketcap"] = daily_market_cap
        free_cf_yield_base = _broadcast_inferred_yield_base(work, "km__marketcap", "km__freecashflowyield", target_index)
        if free_cf_yield_base is not None:
            daily["km__freecashflowyield"] = safe_ratio(free_cf_yield_base, daily_market_cap)
    return BuiltFeatureSet(df=daily, feature_cols=[c for c in daily.columns if c.startswith("km__")])


def _broadcast_inferred_yield_base(work: pd.DataFrame, market_cap_col: str, yield_col: str, target_index: pd.MultiIndex) -> pd.Series | None:
    if market_cap_col not in work.columns or yield_col not in work.columns:
        return None
    inferred = pd.to_numeric(work[market_cap_col], errors="coerce") * pd.to_numeric(work[yield_col], errors="coerce")
    sparse = pd.DataFrame({"date": work["date"], "symbol": work["symbol"], "value": inferred}).dropna(subset=["value"])
    if sparse.empty:
        return None
    daily = broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index)
    return pd.to_numeric(daily.get("value"), errors="coerce")


def _infer_daily_market_cap_from_sparse(work: pd.DataFrame, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None, prefix: str) -> pd.Series | None:
    if df_prices is None or df_prices.empty or "close" not in df_prices.columns:
        return None
    shares_col = None
    for candidate in (f"{prefix}sharesoutstanding", f"{prefix}weightedaverageshsout", f"{prefix}weightedaverageshsoutdil"):
        if candidate in work.columns:
            shares_col = candidate
            break
    if shares_col is None:
        return None
    sparse = pd.DataFrame({"date": work["date"], "symbol": work["symbol"], "shares": pd.to_numeric(work[shares_col], errors="coerce")}).dropna(subset=["shares"])
    if sparse.empty:
        return None
    shares = pd.to_numeric(broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index).get("shares"), errors="coerce")
    close = pd.to_numeric(df_prices["close"], errors="coerce").sort_index().reindex(target_dates(target_index), method="ffill")
    return shares * pd.Series(close.to_numpy(), index=target_index)


def build_ratios_features(
    symbol_obj: Any,
    target_index: pd.MultiIndex,
    df_prices: pd.DataFrame | None = None,
    filing_lag_days: int = 45,
    *,
    sparse_loader: SparseLoader | None = None,
) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "ratios", prefix="rt__", filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    value_cols = [c for c in sparse.columns if c.startswith("rt__") and pd.api.types.is_numeric_dtype(sparse[c])]
    if not value_cols:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    daily = broadcast_sparse(sparse[value_cols].sort_index(), target_index)
    daily_price = daily_price_series(df_prices, target_index)
    if daily_price is not None and df_prices is not None and not df_prices.empty:
        price_on_sparse = pd.Series(
            pd.to_numeric(df_prices["close"], errors="coerce").sort_index().reindex(pd.DatetimeIndex(pd.to_datetime(work["date"])).normalize(), method="ffill").to_numpy(),
            index=work.index,
        )
        for ratio_col in (
            "rt__pricetoearningsratio",
            "rt__pricetobookratio",
            "rt__pricetosalesratio",
            "rt__pricetofreecashflowratio",
            "rt__pricetooperatingcashflowratio",
        ):
            per_share = _broadcast_inferred_per_share(work, price_on_sparse, ratio_col, target_index)
            if per_share is not None:
                daily[ratio_col] = safe_ratio(daily_price, per_share)
        dividend_per_share = _broadcast_existing_series(work, "rt__dividendpershare", target_index)
        if dividend_per_share is not None:
            dividend_yield = safe_ratio(dividend_per_share, daily_price)
            daily["rt__dividendyield"] = dividend_yield
            daily["rt__dividendyieldpercentage"] = dividend_yield * 100.0
    return BuiltFeatureSet(df=daily, feature_cols=[c for c in daily.columns if c.startswith("rt__")])


def _broadcast_inferred_per_share(work: pd.DataFrame, price_on_sparse: pd.Series, ratio_col: str, target_index: pd.MultiIndex) -> pd.Series | None:
    if ratio_col not in work.columns:
        return None
    inferred = safe_ratio(price_on_sparse, pd.to_numeric(work[ratio_col], errors="coerce"))
    sparse = pd.DataFrame({"date": work["date"], "symbol": work["symbol"], "value": inferred}).dropna(subset=["value"])
    if sparse.empty:
        return None
    daily = broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index)
    return pd.to_numeric(daily.get("value"), errors="coerce")


def _broadcast_existing_series(work: pd.DataFrame, value_col: str, target_index: pd.MultiIndex) -> pd.Series | None:
    if value_col not in work.columns:
        return None
    sparse = pd.DataFrame({"date": work["date"], "symbol": work["symbol"], "value": pd.to_numeric(work[value_col], errors="coerce")}).dropna(subset=["value"])
    if sparse.empty:
        return None
    daily = broadcast_sparse(sparse.set_index(["date", "symbol"]).sort_index(), target_index)
    return pd.to_numeric(daily.get("value"), errors="coerce")


def _build_income_statement_features(symbol_obj: Any, target_index: pd.MultiIndex, *, section_key: str, prefix: str, df_prices: pd.DataFrame | None, filing_lag_days: int, sparse_loader: SparseLoader | None) -> BuiltFeatureSet:
    built = build_passthrough_section_features(symbol_obj, target_index, section_key=section_key, prefix=prefix, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    if built.df.empty:
        return built
    enriched, linked_cols = add_daily_price_linked_features(
        built.df,
        target_index,
        df_prices=df_prices,
        share_count_candidates=(f"{prefix}weightedaverageshsoutdil", f"{prefix}weightedaverageshsout"),
        price_denominated=(((f"{prefix}eps", f"{prefix}epsdiluted"), f"{prefix}eps_to_price_daily"),),
        market_cap_denominated=(
            ((f"{prefix}revenue",), f"{prefix}revenue_to_mcap_daily"),
            ((f"{prefix}grossprofit",), f"{prefix}grossprofit_to_mcap_daily"),
            ((f"{prefix}ebitda",), f"{prefix}ebitda_to_mcap_daily"),
            ((f"{prefix}operatingincome",), f"{prefix}operatingincome_to_mcap_daily"),
            ((f"{prefix}netincome",), f"{prefix}netincome_to_mcap_daily"),
        ),
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked_cols])


def build_income_statement_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _build_income_statement_features(symbol_obj, target_index, section_key="income_statement", prefix="is__", df_prices=df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)


def build_income_statement_ttm_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _build_income_statement_features(symbol_obj, target_index, section_key="income_statement_ttm", prefix="is_ttm__", df_prices=df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)


def _build_statement_features(
    symbol_obj: Any,
    target_index: pd.MultiIndex,
    *,
    section_key: str,
    prefix: str,
    df_prices: pd.DataFrame | None,
    market_cap: pd.Series | None,
    filing_lag_days: int,
    sparse_loader: SparseLoader | None,
    market_cap_denominated: Sequence[tuple[Sequence[str], str]],
    negate_market_cap_sources: Sequence[str] = (),
) -> BuiltFeatureSet:
    built = build_passthrough_section_features(symbol_obj, target_index, section_key=section_key, prefix=prefix, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    if built.df.empty:
        return built
    enriched, linked_cols = add_daily_price_linked_features(
        built.df,
        target_index,
        df_prices=df_prices,
        market_cap=market_cap,
        market_cap_denominated=market_cap_denominated,
        negate_market_cap_sources=negate_market_cap_sources,
    )
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *linked_cols])


def build_cash_flow_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _build_statement_features(
        symbol_obj,
        target_index,
        section_key="cash_flow",
        prefix="cf__",
        df_prices=df_prices,
        market_cap=market_cap,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
        market_cap_denominated=(
            (("cf__operatingcashflow", "cf__netcashprovidedbyoperatingactivities"), "cf__operatingcashflow_to_mcap_daily"),
            (("cf__freecashflow",), "cf__freecashflow_to_mcap_daily"),
            (("cf__capitalexpenditure", "cf__capitalexpenditures"), "cf__capex_to_mcap_daily"),
        ),
        negate_market_cap_sources=("cf__capex_to_mcap_daily",),
    )


def build_cash_flow_ttm_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _build_statement_features(
        symbol_obj,
        target_index,
        section_key="cash_flow_ttm",
        prefix="cf_ttm__",
        df_prices=df_prices,
        market_cap=market_cap,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
        market_cap_denominated=(
            (("cf_ttm__operatingcashflow", "cf_ttm__netcashprovidedbyoperatingactivities"), "cf_ttm__operatingcashflow_to_mcap_daily"),
            (("cf_ttm__freecashflow",), "cf_ttm__freecashflow_to_mcap_daily"),
            (("cf_ttm__capitalexpenditure", "cf_ttm__capitalexpenditures"), "cf_ttm__capex_to_mcap_daily"),
        ),
        negate_market_cap_sources=("cf_ttm__capex_to_mcap_daily",),
    )


def build_balance_sheet_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _build_statement_features(
        symbol_obj,
        target_index,
        section_key="balance_sheet",
        prefix="bs__",
        df_prices=df_prices,
        market_cap=market_cap,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
        market_cap_denominated=(
            (("bs__cashandcashequivalents", "bs__cashandshortterminvestments"), "bs__cash_to_mcap_daily"),
            (("bs__totaldebt", "bs__shorttermdebt", "bs__longtermdebt"), "bs__debt_to_mcap_daily"),
            (("bs__netdebt",), "bs__netdebt_to_mcap_daily"),
            (("bs__totalstockholdersequity", "bs__totalequity"), "bs__equity_to_mcap_daily"),
            (("bs__totalassets",), "bs__assets_to_mcap_daily"),
        ),
    )


def build_balance_sheet_ttm_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _build_statement_features(
        symbol_obj,
        target_index,
        section_key="balance_sheet_ttm",
        prefix="bs_ttm__",
        df_prices=df_prices,
        market_cap=market_cap,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
        market_cap_denominated=(
            (("bs_ttm__cashandcashequivalents", "bs_ttm__cashandshortterminvestments"), "bs_ttm__cash_to_mcap_daily"),
            (("bs_ttm__totaldebt", "bs_ttm__shorttermdebt", "bs_ttm__longtermdebt"), "bs_ttm__debt_to_mcap_daily"),
            (("bs_ttm__netdebt",), "bs_ttm__netdebt_to_mcap_daily"),
            (("bs_ttm__totalstockholdersequity", "bs_ttm__totalequity"), "bs_ttm__equity_to_mcap_daily"),
            (("bs_ttm__totalassets",), "bs_ttm__assets_to_mcap_daily"),
        ),
    )


def _growth_builder(symbol_obj: Any, target_index: pd.MultiIndex, *, section_key: str, prefix: str, valuation_frame: pd.DataFrame | None, filing_lag_days: int, sparse_loader: SparseLoader | None, specs: Sequence[tuple[Sequence[str], Sequence[str], str]]) -> BuiltFeatureSet:
    built = build_passthrough_section_features(symbol_obj, target_index, section_key=section_key, prefix=prefix, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    if built.df.empty:
        return built
    enriched, peg_cols = add_growth_adjusted_valuation_features(built.df, valuation_frame=valuation_frame, specs=specs)
    return BuiltFeatureSet(df=enriched, feature_cols=[*built.feature_cols, *peg_cols])


def build_income_statement_growth_features(symbol_obj: Any, target_index: pd.MultiIndex, valuation_frame: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _growth_builder(
        symbol_obj,
        target_index,
        section_key="income_statement_growth",
        prefix="isg__",
        valuation_frame=valuation_frame,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
        specs=(
            (("isg__epsgrowth", "isg__epsdilutedgrowth", "isg__netincomegrowth"), ("rt__pricetoearningsratio",), "isg__earnings_peg_daily"),
            (("isg__revenuegrowth",), ("rt__pricetosalesratio", "km__evtosales"), "isg__sales_growth_valuation_daily"),
            (("isg__grossprofitgrowth",), ("is__grossprofit_to_mcap_daily",), "isg__grossprofit_growth_valuation_daily"),
            (("isg__ebitdagrowth",), ("km__evtoebitda", "is__ebitda_to_mcap_daily"), "isg__ebitda_growth_valuation_daily"),
            (("isg__operatingincomegrowth",), ("is__operatingincome_to_mcap_daily",), "isg__operatingincome_growth_valuation_daily"),
        ),
    )


def build_cash_flow_growth_features(symbol_obj: Any, target_index: pd.MultiIndex, valuation_frame: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _growth_builder(
        symbol_obj,
        target_index,
        section_key="cash_flow_growth",
        prefix="cfg__",
        valuation_frame=valuation_frame,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
        specs=(
            (("cfg__operatingcashflowgrowth", "cfg__netcashprovidedbyoperatingactivitiesgrowth"), ("rt__pricetooperatingcashflowratio", "km__evtooperatingcashflow"), "cfg__operatingcashflow_growth_valuation_daily"),
            (("cfg__freecashflowgrowth",), ("rt__pricetofreecashflowratio", "km__evtofreecashflow"), "cfg__freecashflow_growth_valuation_daily"),
            (("cfg__capitalexpendituregrowth", "cfg__capitalexpendituresgrowth"), ("cf__capex_to_mcap_daily",), "cfg__capex_growth_valuation_daily"),
        ),
    )


def build_balance_sheet_growth_features(symbol_obj: Any, target_index: pd.MultiIndex, valuation_frame: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _growth_builder(
        symbol_obj,
        target_index,
        section_key="balance_sheet_growth",
        prefix="bsg__",
        valuation_frame=valuation_frame,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
        specs=(
            (("bsg__totalstockholdersequitygrowth", "bsg__totalequitygrowth", "bsg__bookvaluepersharegrowth"), ("rt__pricetobookratio", "bs__equity_to_mcap_daily"), "bsg__book_growth_valuation_daily"),
            (("bsg__totalassetsgrowth",), ("bs__assets_to_mcap_daily",), "bsg__assets_growth_valuation_daily"),
            (("bsg__cashandcashequivalentsgrowth", "bsg__cashandshortterminvestmentsgrowth"), ("bs__cash_to_mcap_daily",), "bsg__cash_growth_valuation_daily"),
            (("bsg__totaldebtgrowth", "bsg__netdebtgrowth"), ("bs__debt_to_mcap_daily", "bs__netdebt_to_mcap_daily"), "bsg__debt_growth_valuation_daily"),
        ),
    )


def build_financial_growth_features(symbol_obj: Any, target_index: pd.MultiIndex, valuation_frame: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return _growth_builder(
        symbol_obj,
        target_index,
        section_key="financial_growth",
        prefix="fg__",
        valuation_frame=valuation_frame,
        filing_lag_days=filing_lag_days,
        sparse_loader=sparse_loader,
        specs=(
            (("fg__epsgrowth", "fg__epsdilutedgrowth", "fg__netincomegrowth"), ("rt__pricetoearningsratio",), "fg__earnings_peg_daily"),
            (("fg__revenuegrowth",), ("rt__pricetosalesratio", "km__evtosales"), "fg__sales_growth_valuation_daily"),
            (("fg__ebitdagrowth",), ("km__evtoebitda", "is__ebitda_to_mcap_daily"), "fg__ebitda_growth_valuation_daily"),
            (("fg__freecashflowgrowth",), ("rt__pricetofreecashflowratio", "km__evtofreecashflow"), "fg__freecashflow_growth_valuation_daily"),
            (("fg__operatingcashflowgrowth",), ("rt__pricetooperatingcashflowratio", "km__evtooperatingcashflow"), "fg__operatingcashflow_growth_valuation_daily"),
        ),
    )


def build_fundamental_change_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return merge_feature_sets(
        [
            build_key_metrics_features(symbol_obj, target_index, df_prices=df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader),
            build_ratios_features(symbol_obj, target_index, df_prices=df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader),
        ],
        target_index,
    )


def build_statement_quality_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    income_statement = build_income_statement_features(symbol_obj, target_index, df_prices=df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    market_cap = None
    close = daily_price_series(df_prices, target_index)
    shares = first_existing(income_statement.df, ("is__weightedaverageshsoutdil", "is__weightedaverageshsout"))
    if close is not None and shares is not None:
        market_cap = shares.reindex(target_index) * close.reindex(target_index)
    cash_flow = build_cash_flow_features(symbol_obj, target_index, df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    balance_sheet = build_balance_sheet_features(symbol_obj, target_index, df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    valuation_frame = pd.concat([income_statement.df, cash_flow.df, balance_sheet.df], axis=1)
    if valuation_frame.columns.has_duplicates:
        valuation_frame = valuation_frame.loc[:, ~valuation_frame.columns.duplicated(keep="last")]
    return merge_feature_sets(
        [
            income_statement,
            build_income_statement_growth_features(symbol_obj, target_index, valuation_frame=valuation_frame, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader),
            cash_flow,
            build_cash_flow_growth_features(symbol_obj, target_index, valuation_frame=valuation_frame, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader),
            balance_sheet,
            build_balance_sheet_growth_features(symbol_obj, target_index, valuation_frame=valuation_frame, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader),
            build_financial_growth_features(symbol_obj, target_index, valuation_frame=valuation_frame, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader),
        ],
        target_index,
    )


def build_ttm_financial_statement_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, filing_lag_days: int = 45, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    income_statement = build_income_statement_ttm_features(symbol_obj, target_index, df_prices=df_prices, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader)
    market_cap = None
    close = daily_price_series(df_prices, target_index)
    shares = first_existing(income_statement.df, ("is_ttm__weightedaverageshsoutdil", "is_ttm__weightedaverageshsout"))
    if close is not None and shares is not None:
        market_cap = shares.reindex(target_index) * close.reindex(target_index)
    return merge_feature_sets(
        [
            income_statement,
            build_cash_flow_ttm_features(symbol_obj, target_index, df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader),
            build_balance_sheet_ttm_features(symbol_obj, target_index, df_prices=df_prices, market_cap=market_cap, filing_lag_days=filing_lag_days, sparse_loader=sparse_loader),
        ],
        target_index,
    )


def build_earnings_features(symbol_obj: Any, target_index: pd.MultiIndex, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "earnings", prefix="earn__", filing_lag_days=0, sparse_loader=sparse_loader)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    earn_cols = {str(c).lower(): c for c in work.columns if str(c).startswith("earn__")}

    def _find_col(*patterns: str) -> pd.Series:
        for pattern in patterns:
            for key, col in earn_cols.items():
                if pattern in key:
                    return pd.to_numeric(work[col], errors="coerce")
        return pd.Series(np.nan, index=work.index, dtype=float)

    eps_actual = _find_col("epsactual", "eps__actual", "eps_actual", "eps ")
    if eps_actual.isna().all():
        for key, col in earn_cols.items():
            if key.replace("earn__", "") in ("eps", "epsactual"):
                eps_actual = pd.to_numeric(work[col], errors="coerce")
                break
    eps_estimated = _find_col("epsestimated", "eps_estimated", "epse")
    rev_actual = _find_col("revenueactual", "revenue_actual")
    if rev_actual.isna().all():
        for key, col in earn_cols.items():
            if key.replace("earn__", "") in ("revenue", "revenueactual"):
                rev_actual = pd.to_numeric(work[col], errors="coerce")
                break
    rev_estimated = _find_col("revenueestimated", "revenue_estimated", "revenuee")
    out = work[["date", "symbol"]].copy()
    out["evt__earn_eps_surprise"] = safe_ratio(eps_actual - eps_estimated, eps_estimated.abs())
    out["evt__earn_rev_surprise"] = safe_ratio(rev_actual - rev_estimated, rev_estimated.abs())
    out["evt__earn_beat_flag"] = ((eps_actual >= eps_estimated) & eps_actual.notna() & eps_estimated.notna()).astype(float)
    out["evt__earn_beat_streak_4"] = out.groupby("symbol")["evt__earn_beat_flag"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    raw_cols: list[str] = []
    for col in work.columns:
        if col in ("date", "symbol") or not str(col).startswith("earn__"):
            continue
        converted = pd.to_numeric(work[col], errors="coerce")
        if converted.notna().any():
            out[col] = converted
            raw_cols.append(col)
    daily = broadcast_sparse(out.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["evt__earn_days_since"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    daily = daily.replace([np.inf, -np.inf], np.nan)
    derived_cols = ["evt__earn_eps_surprise", "evt__earn_rev_surprise", "evt__earn_beat_flag", "evt__earn_beat_streak_4", "evt__earn_days_since"]
    return BuiltFeatureSet(df=daily, feature_cols=[c for c in derived_cols if c in daily.columns] + raw_cols)


def build_analyst_estimates_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, market_cap: pd.Series | None = None, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "analyst_estimates", prefix="ae__", filing_lag_days=0, sparse_loader=sparse_loader)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    work["ae__epsavg"] = pd.to_numeric(work.get("ae__epsavg"), errors="coerce")
    work["ae__revenueavg"] = pd.to_numeric(work.get("ae__revenueavg"), errors="coerce")
    out = work[["date", "symbol"]].copy()
    out["evt__ae_eps_avg"] = work["ae__epsavg"]
    out["evt__ae_revenue_avg"] = work["ae__revenueavg"]
    out["evt__ae_eps_rev_qoq"] = work.groupby("symbol")["ae__epsavg"].pct_change()
    out["evt__ae_revenue_rev_qoq"] = work.groupby("symbol")["ae__revenueavg"].pct_change()
    daily = broadcast_sparse(out.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["evt__ae_days_since"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    cols = ["evt__ae_eps_avg", "evt__ae_revenue_avg", "evt__ae_eps_rev_qoq", "evt__ae_revenue_rev_qoq", "evt__ae_days_since"]
    daily, linked_cols = add_daily_price_linked_features(
        daily,
        target_index,
        df_prices=df_prices,
        market_cap=market_cap,
        price_denominated=((("evt__ae_eps_avg",), "ae__forward_eps_to_price_daily"),),
        market_cap_denominated=((("evt__ae_revenue_avg",), "ae__forward_revenue_to_mcap_daily"),),
    )
    return BuiltFeatureSet(df=daily, feature_cols=[*cols, *linked_cols])


def build_ratings_historical_features(symbol_obj: Any, target_index: pd.MultiIndex, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "ratings_historical", prefix="rating__", filing_lag_days=0, sparse_loader=sparse_loader)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    work["rating__overallscore"] = pd.to_numeric(work.get("rating__overallscore"), errors="coerce")
    out = work[["date", "symbol"]].copy()
    out["evt__rating_score"] = work["rating__overallscore"]
    out["evt__rating_score_change"] = work.groupby("symbol")["rating__overallscore"].diff()
    daily = broadcast_sparse(out.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["evt__rating_days_since"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    cols = ["evt__rating_score", "evt__rating_score_change", "evt__rating_days_since"]
    return BuiltFeatureSet(df=daily, feature_cols=cols)


GRADE_COLS = [
    "grade__analystratingsstrongbuy",
    "grade__analystratingsbuy",
    "grade__analystratingshold",
    "grade__analystratingssell",
    "grade__analystratingsstrongsell",
]


def build_grades_historical_features(symbol_obj: Any, target_index: pd.MultiIndex, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "grades_historical", prefix="grade__", filing_lag_days=0, sparse_loader=sparse_loader)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    for col in GRADE_COLS:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    total = sum((work[col].fillna(0.0) for col in GRADE_COLS))
    bullish = work["grade__analystratingsstrongbuy"].fillna(0.0) + work["grade__analystratingsbuy"].fillna(0.0)
    bearish = work["grade__analystratingssell"].fillna(0.0) + work["grade__analystratingsstrongsell"].fillna(0.0)
    out = work[["date", "symbol"]].copy()
    out["evt__grade_bullish_ratio"] = safe_ratio(bullish, total.replace(0.0, np.nan))
    out["evt__grade_bearish_ratio"] = safe_ratio(bearish, total.replace(0.0, np.nan))
    out["evt__grade_net_bullish"] = safe_ratio(bullish - bearish, total.replace(0.0, np.nan))
    daily = broadcast_sparse(out.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["evt__grade_days_since"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    cols = ["evt__grade_bullish_ratio", "evt__grade_bearish_ratio", "evt__grade_net_bullish", "evt__grade_days_since"]
    return BuiltFeatureSet(df=daily, feature_cols=cols)


def build_event_features(symbol_obj: Any, target_index: pd.MultiIndex, df_prices: pd.DataFrame | None = None, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    return merge_feature_sets(
        [
            build_earnings_features(symbol_obj, target_index, sparse_loader=sparse_loader),
            build_analyst_estimates_features(symbol_obj, target_index, df_prices=df_prices, sparse_loader=sparse_loader),
            build_ratings_historical_features(symbol_obj, target_index, sparse_loader=sparse_loader),
            build_grades_historical_features(symbol_obj, target_index, sparse_loader=sparse_loader),
        ],
        target_index,
    )


def build_insider_trading_features(symbol_obj: Any, target_index: pd.MultiIndex, *, sparse_loader: SparseLoader | None = None) -> BuiltFeatureSet:
    sparse = load_section_payload(symbol_obj, "insider_trading", prefix="insider__", filing_lag_days=0, sparse_loader=sparse_loader)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"])
    trade_date = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    price = pd.to_numeric(work.get("insider__price"), errors="coerce")
    shares = pd.to_numeric(work.get("insider__securitiestransacted"), errors="coerce")
    disposition = work.get("insider__acquisitionordisposition")
    sign = disposition.astype(str).str.upper().map({"A": 1.0, "D": -1.0}).fillna(0.0)
    event_df = pd.DataFrame(
        {
            "date": trade_date,
            "signed_value": sign * price.fillna(0.0) * shares.fillna(0.0),
            "buy_count": (sign > 0).astype(float),
            "sell_count": (sign < 0).astype(float),
        }
    ).dropna(subset=["date"])
    event_df = event_df.groupby("date", as_index=False).sum().sort_values("date")
    td = target_dates(target_index)
    daily = pd.DataFrame(index=td).join(event_df.set_index("date"), how="left").fillna(0.0)
    daily["own__insider_net_buy_value_90d"] = daily["signed_value"].rolling(90, min_periods=1).sum()
    daily["own__insider_buy_count_90d"] = daily["buy_count"].rolling(90, min_periods=1).sum()
    daily["own__insider_sell_count_90d"] = daily["sell_count"].rolling(90, min_periods=1).sum()
    daily["own__insider_buy_sell_ratio_90d"] = safe_ratio(daily["own__insider_buy_count_90d"], daily["own__insider_sell_count_90d"].replace(0.0, np.nan))
    daily["own__insider_days_since"] = days_since_last_event(td, event_df["date"])
    daily["symbol"] = str(getattr(symbol_obj, "symbol", symbol_obj)).upper()
    daily = daily.drop(columns=["signed_value", "buy_count", "sell_count"]).reset_index().rename(columns={"index": "date"}).set_index(["date", "symbol"]).sort_index()
    cols = ["own__insider_net_buy_value_90d", "own__insider_buy_count_90d", "own__insider_sell_count_90d", "own__insider_buy_sell_ratio_90d", "own__insider_days_since"]
    return BuiltFeatureSet(df=daily.replace([np.inf, -np.inf], np.nan), feature_cols=cols)


POSITION_PREFIX = "ps__"
POSITIONS_SUMMARY_SECTION_KEY = "positions_summary"
_CANONICAL_POSITION_FIELDS: dict[str, tuple[str, ...]] = {
    "investor_count": ("investor_count", "ps__investor_count", "ps__investorcount", "ps__investorscount", "ps__holdercount", "ps__holderscount", "ps__institutionalholders", "ps__numberofinvestors", "ps__numberofholders"),
    "shares_held": ("shares_held", "ps__shares_held", "ps__sharesheld", "ps__totalshares", "ps__sharecount", "ps__shares", "ps__institutionalshares"),
    "investment_value": ("investment_value", "ps__investment_value", "ps__investmentvalue", "ps__totalinvestmentvalue", "ps__marketvalue", "ps__positionvalue"),
    "ownership_pct": ("ownership_pct", "ps__ownership_pct", "ps__ownershippct", "ps__ownershippercentage", "ps__ownershippercent"),
    "shares_change": ("shares_change", "ps__shares_change", "ps__shareschange", "ps__changeinshares", "ps__sharechange"),
    "investment_change": ("investment_change", "ps__investment_change", "ps__investmentchange", "ps__changeininvestment", "ps__valuechange"),
    "ownership_pct_change": ("ownership_pct_change", "ps__ownership_pct_change", "ps__ownershippctchange", "ps__ownershippercentagechange", "ps__changeinownership"),
    "put_call_ratio": ("put_call_ratio", "ps__put_call_ratio", "ps__putcallratio", "ps__putcall"),
    "call_count": ("call_count", "ps__call_count", "ps__callcount", "ps__calls", "ps__callscount"),
    "put_count": ("put_count", "ps__put_count", "ps__putcount", "ps__puts", "ps__putscount"),
}


def _resolve_numeric_series(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return None


def build_positions_summary_features(
    symbol_obj: Any,
    target_index: pd.MultiIndex,
    *,
    sparse_loader: SparseLoader | None = None,
    positions_source_loader: PositionsSourceLoader | None = None,
) -> BuiltFeatureSet:
    sparse = positions_source_loader(symbol_obj) if positions_source_loader is not None else pd.DataFrame()
    if sparse.empty:
        sparse = load_section_payload(symbol_obj, POSITIONS_SUMMARY_SECTION_KEY, prefix=POSITION_PREFIX, filing_lag_days=0, sparse_loader=sparse_loader)
    if sparse.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work = sparse.reset_index().sort_values(["symbol", "date"]).copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    if work.empty:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    work["symbol"] = str(getattr(symbol_obj, "symbol", symbol_obj)).strip().upper()
    feature_frame = work[["date", "symbol"]].copy()
    numeric_sources: dict[str, pd.Series] = {}
    for canonical_name, candidates in _CANONICAL_POSITION_FIELDS.items():
        series = _resolve_numeric_series(work, candidates)
        if series is None:
            continue
        numeric_sources[canonical_name] = series
        feature_frame[f"{POSITION_PREFIX}{canonical_name}"] = series
    if not numeric_sources:
        return BuiltFeatureSet(df=pd.DataFrame(index=target_index), feature_cols=[])
    for canonical_name in ("investor_count", "shares_held", "investment_value", "ownership_pct", "put_call_ratio"):
        series = numeric_sources.get(canonical_name)
        if series is None:
            continue
        feature_frame[f"{POSITION_PREFIX}{canonical_name}_change"] = series.groupby(work["symbol"]).diff()
        feature_frame[f"{POSITION_PREFIX}{canonical_name}_pct_change"] = series.groupby(work["symbol"]).pct_change().replace([np.inf, -np.inf], np.nan)
    if "investor_count" in numeric_sources and "shares_held" in numeric_sources:
        feature_frame["ps__shares_per_investor"] = safe_ratio(numeric_sources["shares_held"], numeric_sources["investor_count"].replace(0.0, np.nan))
    if "investor_count" in numeric_sources and "investment_value" in numeric_sources:
        feature_frame["ps__investment_per_investor"] = safe_ratio(numeric_sources["investment_value"], numeric_sources["investor_count"].replace(0.0, np.nan))
    if "shares_held" in numeric_sources and "ownership_pct" in numeric_sources:
        feature_frame["ps__shares_ownership_ratio"] = safe_ratio(numeric_sources["shares_held"], numeric_sources["ownership_pct"].replace(0.0, np.nan))
    daily = broadcast_sparse(feature_frame.set_index(["date", "symbol"]).sort_index(), target_index)
    daily["ps__days_since_report"] = days_since_for_target(target_index, days_since_last_event(target_dates(target_index), work["date"]))
    feature_cols = [col for col in daily.columns if str(col).startswith(POSITION_PREFIX)]
    return BuiltFeatureSet(df=daily.replace([np.inf, -np.inf], np.nan), feature_cols=feature_cols)


def build_ownership_features(
    symbol_obj: Any,
    target_index: pd.MultiIndex,
    *,
    sparse_loader: SparseLoader | None = None,
    positions_source_loader: PositionsSourceLoader | None = None,
) -> BuiltFeatureSet:
    return merge_feature_sets(
        [
            build_insider_trading_features(symbol_obj, target_index, sparse_loader=sparse_loader),
            build_positions_summary_features(
                symbol_obj,
                target_index,
                sparse_loader=sparse_loader,
                positions_source_loader=positions_source_loader,
            ),
        ],
        target_index,
    )


__all__ = [
    "add_daily_price_linked_features",
    "add_growth_adjusted_valuation_features",
    "broadcast_sparse",
    "build_analyst_estimates_features",
    "build_balance_sheet_features",
    "build_balance_sheet_growth_features",
    "build_balance_sheet_ttm_features",
    "build_cash_flow_features",
    "build_cash_flow_growth_features",
    "build_cash_flow_ttm_features",
    "build_earnings_features",
    "build_event_features",
    "build_financial_growth_features",
    "build_fundamental_change_features",
    "build_grades_historical_features",
    "build_income_statement_features",
    "build_income_statement_growth_features",
    "build_income_statement_ttm_features",
    "build_insider_trading_features",
    "build_key_metrics_features",
    "build_ownership_features",
    "build_passthrough_section_features",
    "build_positions_summary_features",
    "build_ratios_features",
    "build_ratings_historical_features",
    "daily_price_series",
    "days_since_for_target",
    "days_since_last_event",
    "default_sparse_loader",
    "first_existing",
    "load_section_payload",
    "merge_feature_sets",
    "safe_ratio",
    "section_prefix",
    "target_dates",
]
