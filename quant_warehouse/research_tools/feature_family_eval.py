from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from quant_warehouse.ingest.screener_fetch import ScreenerQuery, fetch_equity_screener
from quant_warehouse.warehouse.api import Warehouse


FMP_REQUIRED_FUNDAMENTAL_SECTIONS = (
    "prices",
    "historical_market_cap",
    "income",
    "balance",
    "cash",
    "ratios",
    "metrics",
    "income_growth",
    "balance_growth",
    "cash_growth",
)


@dataclass(frozen=True)
class FamilyEvaluationConfig:
    provider: str = "fmp"
    market_cap_min: int = 1_000_000_000_000
    country: str = "US"
    exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")
    screen_limit: int = 5_000
    start_date: str = "2018-01-01"
    end_date: str | None = None
    filing_lag_days: int = 45
    horizons: tuple[int, ...] = (20, 60, 120)
    min_observations: int = 120
    max_features_per_family: int | None = None


@dataclass(frozen=True)
class FeatureSpec:
    feature: str
    family: str
    source: str
    source_column: str
    expected_direction: str


def screen_fmp_equity_universe(
    config: FamilyEvaluationConfig,
    *,
    warehouse: Warehouse | None = None,
    required_sections: Iterable[str] = FMP_REQUIRED_FUNDAMENTAL_SECTIONS,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame, str]:
    """Screen FMP equities through OpenBB and keep symbols with local warehouse history."""

    wh = warehouse or Warehouse()
    raw_universe, source = _fetch_screener_with_catalog_fallback(wh, config)
    if raw_universe.empty:
        raise RuntimeError("OpenBB/FMP screener returned no symbols for the configured filters.")
    raw_universe = raw_universe.copy()
    raw_universe["symbol"] = raw_universe["symbol"].astype(str).str.strip().str.upper()
    if "market_cap" in raw_universe.columns:
        raw_universe["market_cap"] = pd.to_numeric(raw_universe["market_cap"], errors="coerce")
        raw_universe = raw_universe.loc[raw_universe["market_cap"].ge(config.market_cap_min)]
    raw_universe = raw_universe.drop_duplicates("symbol")

    rows = []
    for record in raw_universe.to_dict("records"):
        symbol = str(record.get("symbol") or "").strip().upper()
        asset_ok, asset_reason = _is_supported_equity_record(symbol, record)
        if asset_ok:
            ok, reason = _has_required_history(wh, symbol, config.provider, tuple(required_sections))
        else:
            ok, reason = False, asset_reason
        row = {"symbol": symbol, "eligible": ok, "reason": reason}
        if "market_cap" in record:
            row["screen_market_cap"] = record.get("market_cap")
        rows.append(row)
    eligibility = pd.DataFrame(rows)
    symbols = tuple(eligibility.loc[eligibility["eligible"], "symbol"].sort_values())
    if not symbols:
        raise RuntimeError("No screened symbols have the required stored Quant Warehouse history.")
    return symbols, raw_universe, eligibility, source


def _fetch_screener_with_catalog_fallback(
    warehouse: Warehouse,
    config: FamilyEvaluationConfig,
) -> tuple[pd.DataFrame, str]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    exchanges = config.exchanges or ("",)
    for exchange in exchanges:
        query = ScreenerQuery(
            provider=config.provider,
            mktcap_min=config.market_cap_min,
            country=config.country,
            exchanges=(exchange,) if exchange else (),
            is_etf=False,
            is_fund=False,
            is_active=True,
            all_share_classes=False,
            limit=config.screen_limit,
        )
        try:
            frame, source = fetch_equity_screener(query)
        except Exception as exc:
            frame = _catalog_profile_universe(warehouse, config, exchanges=(exchange,) if exchange else ())
            source = f"catalog:{config.provider}:fallback_after_{type(exc).__name__}"
        if frame is not None and not frame.empty:
            frames.append(frame)
            sources.append(source)
    if not frames:
        return pd.DataFrame(), f"openbb:{config.provider}"
    out = pd.concat(frames, ignore_index=True)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out = out.drop_duplicates("symbol")
    return out, "+".join(dict.fromkeys(sources))


def _catalog_profile_universe(
    warehouse: Warehouse,
    config: FamilyEvaluationConfig,
    *,
    exchanges: tuple[str, ...],
) -> pd.DataFrame:
    profiles = warehouse.catalog.query_symbol_profiles(
        provider=config.provider,
        min_market_cap=config.market_cap_min,
        country=config.country,
        exchanges=exchanges,
        exclude_etf=True,
        exclude_fund=True,
        limit=config.screen_limit,
    )
    rows = [
        {
            "symbol": profile.symbol,
            "name": profile.company_name,
            "market_cap": profile.market_cap,
            "exchange": profile.exchange,
            "country": profile.country,
            "sector": profile.sector,
            "industry": profile.industry,
            **_profile_asset_payload(profile.payload),
        }
        for profile in profiles
    ]
    return pd.DataFrame(rows)


def _profile_asset_payload(payload: dict[str, object]) -> dict[str, object]:
    raw = dict(payload or {})
    return {
        "is_etf": raw.get("is_etf", raw.get("isEtf")),
        "is_fund": raw.get("is_fund", raw.get("isFund")),
        "quote_type": raw.get("quote_type", raw.get("quoteType")),
        "instrument_type": raw.get("instrument_type", raw.get("instrumentType", raw.get("type"))),
        "fund_family": raw.get("fund_family", raw.get("fundFamily")),
        "security_type": raw.get("security_type", raw.get("securityType")),
    }


def _is_supported_equity_record(symbol: str, record: dict[str, object]) -> tuple[bool, str]:
    asset_type = _pooled_or_noncommon_equity_type(symbol, record)
    if asset_type is not None:
        return False, f"asset_class: {asset_type}"
    return True, "ok"


def _pooled_or_noncommon_equity_type(symbol: str, record: dict[str, object]) -> str | None:
    payload = {str(key).lower(): value for key, value in dict(record or {}).items()}
    if _truthy(payload.get("is_etf")) or _truthy(payload.get("isetf")):
        return "etf"
    if _truthy(payload.get("is_fund")) or _truthy(payload.get("isfund")):
        return "fund"

    quote_type = _clean_token(payload.get("quote_type") or payload.get("quotetype"))
    instrument_type = _clean_token(payload.get("instrument_type") or payload.get("instrumenttype") or payload.get("type"))
    security_type = _clean_token(payload.get("security_type") or payload.get("securitytype"))
    if quote_type == "etf" or instrument_type == "etf" or security_type == "etf":
        return "etf"
    if quote_type in {"mutualfund", "mutual_fund", "fund"}:
        return "fund"
    if instrument_type in {"mutualfund", "mutual_fund", "fund"}:
        return "fund"
    if security_type in {"mutualfund", "mutual_fund", "fund", "open_end_fund", "closed_end_fund"}:
        return "fund"
    if payload.get("fund_family") not in (None, "") or payload.get("fundfamily") not in (None, ""):
        return "fund"
    if _looks_like_mutual_fund_symbol(symbol):
        return "fund_symbol_pattern"
    return None


def _clean_token(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _looks_like_mutual_fund_symbol(symbol: str) -> bool:
    text = str(symbol or "").strip().upper()
    return len(text) == 5 and text.endswith("X") and text.isalpha()


def build_fundamental_feature_panel(
    symbols: Iterable[str],
    config: FamilyEvaluationConfig,
    *,
    warehouse: Warehouse | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build FMP daily-adjusted and FinanceToolkit-style fundamental feature families."""

    wh = warehouse or Warehouse()
    start = perf_counter()
    frames: list[pd.DataFrame] = []
    all_specs: list[FeatureSpec] = []
    diagnostics: list[dict[str, object]] = []
    for symbol in symbols:
        frame, specs, diag = _build_symbol_fundamental_panel(wh, symbol, config)
        diagnostics.append(diag)
        if not frame.empty:
            frames.append(frame)
            all_specs.extend(specs)
    if not frames:
        raise RuntimeError("No feature frames were built for the requested symbols.")
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    metadata = (
        pd.DataFrame([spec.__dict__ for spec in all_specs])
        .drop_duplicates()
        .sort_values(["family", "feature"])
        .reset_index(drop=True)
    )
    diagnostics_df = pd.DataFrame(diagnostics)
    timings = {"raw_panel_build_seconds": perf_counter() - start}
    return panel, metadata, diagnostics_df, timings


def cap_features_by_quality(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    max_features: int | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Optionally cap each feature family using non-target feature quality metrics."""

    quality_frames: list[pd.DataFrame] = []
    selected: list[str] = []
    for family, family_meta in metadata.groupby("family"):
        rows = []
        for feature in family_meta["feature"].tolist():
            values = pd.to_numeric(panel[feature], errors="coerce")
            coverage = float(values.notna().mean())
            var_by_date = panel[["date", feature]].groupby("date")[feature].var()
            avg_xs_var = float(var_by_date.replace([np.inf, -np.inf], np.nan).mean())
            rows.append({"family": family, "feature": feature, "coverage": coverage, "avg_xs_var": avg_xs_var})
        quality = pd.DataFrame(rows).sort_values(
            ["coverage", "avg_xs_var", "feature"],
            ascending=[False, False, True],
        )
        quality = quality.reset_index(drop=True)
        quality["selected"] = True if max_features is None else quality.index < int(max_features)
        quality_frames.append(quality)
        selected.extend(quality.loc[quality["selected"], "feature"].tolist())
    quality_df = pd.concat(quality_frames, ignore_index=True)
    capped_metadata = (
        metadata.loc[metadata["feature"].isin(selected)]
        .copy()
        .sort_values(["family", "feature"])
        .reset_index(drop=True)
    )
    return selected, capped_metadata, quality_df


def evaluate_feature_families(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    min_observations: int = 120,
    include_spreads: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Evaluate feature families with daily cross-sectional rank IC and optional spreads."""

    start = perf_counter()
    features = metadata["feature"].tolist()
    results = _evaluate_features(
        panel,
        metadata,
        features,
        horizons=horizons,
        min_observations=min_observations,
        include_spreads=include_spreads,
    )
    results = results.merge(metadata, on="feature", how="left")
    summary = (
        results.groupby(["horizon", "source", "family"])
        .agg(
            features=("feature", "nunique"),
            mean_rank_ic=("mean_daily_rank_ic", "mean"),
            median_rank_ic=("mean_daily_rank_ic", "median"),
            median_spread_bps=("spread_bps", "median"),
            positive_ic_share=("mean_daily_rank_ic", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values(["horizon", "mean_rank_ic"], ascending=[True, False])
    )
    best = summary.groupby("horizon").head(1).reset_index(drop=True)
    stable = (
        summary.groupby(["source", "family"])
        .agg(
            horizons=("horizon", "nunique"),
            avg_rank_ic=("mean_rank_ic", "mean"),
            min_rank_ic=("mean_rank_ic", "min"),
            avg_spread_bps=("median_spread_bps", "mean"),
            positive_horizons=("mean_rank_ic", lambda s: int((s > 0).sum())),
            avg_positive_ic_share=("positive_ic_share", "mean"),
            features=("features", "max"),
        )
        .reset_index()
        .sort_values(["positive_horizons", "avg_rank_ic"], ascending=[False, False])
    )
    return results, summary, best, stable, perf_counter() - start


def _has_required_history(
    warehouse: Warehouse,
    symbol: str,
    provider: str,
    required_sections: tuple[str, ...],
) -> tuple[bool, str]:
    for section in required_sections:
        try:
            frame = _read_section(warehouse, symbol, section, provider)
        except Exception as exc:
            return False, f"{section}: {type(exc).__name__}"
        if frame is None or frame.empty:
            return False, f"{section}: empty"
    return True, "ok"


def _read_section(warehouse: Warehouse, symbol: str, section: str, provider: str) -> pd.DataFrame:
    if section == "prices":
        return warehouse.read_prices(symbol, provider=provider)
    return warehouse.read_fundamentals(symbol, section=section, provider=provider)


def _slice_frame(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _align_fundamental(
    frame: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    *,
    filing_lag_days: int,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(index=daily_index)
    sparse = _numeric_frame(frame)
    sparse.index = pd.DatetimeIndex(pd.to_datetime(sparse.index, errors="coerce")).normalize() + pd.Timedelta(
        days=int(filing_lag_days)
    )
    sparse = sparse.loc[sparse.index.notna()].sort_index()
    sparse = sparse.loc[~sparse.index.duplicated(keep="last")]
    return sparse.reindex(daily_index, method="ffill")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0.0, np.nan))


def _statement_direction(column: str) -> str:
    text = column.lower()
    if any(token in text for token in ("debt", "liabilit", "expense", "cost", "tax", "payable", "deficit", "loss", "inventory")):
        return "lower_is_better"
    return "higher_is_better"


def _family_for_ratio_column(column: str) -> str | None:
    text = column.lower()
    if any(token in text for token in ("margin", "return_on", "income_quality", "tax_burden", "interest_burden", "sga_to_revenue")):
        return "ft_ratios_profitability"
    if any(token in text for token in ("current_ratio", "quick_ratio", "cash_ratio", "operating_cash_flow_ratio", "working_capital")):
        return "ft_ratios_liquidity"
    if any(token in text for token in ("debt", "coverage", "equity_multiplier", "liabilities", "solvency")):
        return "ft_ratios_solvency"
    if any(token in text for token in ("turnover", "days_", "cycle", "cash_conversion", "asset_turnover")):
        return "ft_ratios_efficiency"
    if any(
        token in text
        for token in (
            "price_to",
            "ev_to",
            "market_cap",
            "enterprise_value",
            "yield",
            "book_value_per_share",
            "earnings_per_share",
            "revenue_per_share",
            "capex_per_share",
        )
    ):
        return "ft_ratios_valuation"
    return None


def _family_for_metric_column(column: str) -> str | None:
    text = column.lower()
    if any(token in text for token in ("ev_to", "market_cap", "enterprise_value", "graham", "invested_capital")):
        return "ft_ratios_valuation"
    if any(token in text for token in ("return_on", "income_quality", "roic")):
        return "ft_ratios_profitability"
    if any(token in text for token in ("debt", "coverage", "working_capital")):
        return "ft_ratios_solvency"
    return None


def _expected_direction(family: str, column: str) -> str:
    text = column.lower()
    if family in {"ft_ratios_valuation", "ft_ratios_solvency"}:
        if any(token in text for token in ("yield", "cash", "working_capital", "interest_coverage", "current_ratio", "quick_ratio")):
            return "higher_is_better"
        return "lower_is_better"
    return "higher_is_better"


def _add_tangible_book(balance_daily: pd.DataFrame) -> pd.DataFrame:
    if balance_daily.empty or "total_stockholders_equity" not in balance_daily.columns:
        return balance_daily
    out = balance_daily.copy()
    if "goodwill_and_intangible_assets" in out.columns:
        goodwill_intangible = out["goodwill_and_intangible_assets"]
    else:
        goodwill = out["goodwill"] if "goodwill" in out.columns else 0.0
        intangible = out["intangible_assets"] if "intangible_assets" in out.columns else 0.0
        goodwill_intangible = goodwill + intangible
    out["book_equity"] = out["total_stockholders_equity"]
    out["tangible_book"] = out["total_stockholders_equity"] - goodwill_intangible
    return out


def _daily_column(aligned: dict[str, pd.DataFrame], section: str, column: str) -> pd.Series:
    frame = aligned.get(section, pd.DataFrame())
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _add_feature(
    feature_frames: dict[str, pd.Series],
    specs: list[FeatureSpec],
    name: str,
    values: pd.Series,
    *,
    family: str,
    source: str,
    source_column: str,
    expected_direction: str,
) -> None:
    if values is None or values.empty or values.notna().sum() == 0:
        return
    feature = f"{family}__{name}"
    feature_frames[feature] = values
    specs.append(FeatureSpec(feature, family, source, source_column, expected_direction))


def _build_symbol_fundamental_panel(
    warehouse: Warehouse,
    symbol: str,
    config: FamilyEvaluationConfig,
) -> tuple[pd.DataFrame, list[FeatureSpec], dict[str, object]]:
    start = perf_counter()
    inputs = {
        section: _slice_frame(
            _read_section(warehouse, symbol, section, config.provider),
            config.start_date if section in {"prices", "historical_market_cap"} else None,
            config.end_date,
        )
        for section in FMP_REQUIRED_FUNDAMENTAL_SECTIONS
    }
    prices = inputs["prices"]
    mcap = inputs["historical_market_cap"]
    if prices.empty or mcap.empty or "close" not in prices.columns or "market_cap" not in mcap.columns:
        return pd.DataFrame(), [], {"symbol": symbol, "status": "missing_prices_or_mcap"}

    daily_index = pd.DatetimeIndex(prices.index)
    close = pd.to_numeric(prices["close"], errors="coerce")
    daily_mcap = pd.to_numeric(mcap["market_cap"], errors="coerce").reindex(daily_index, method="ffill")
    panel = pd.DataFrame({"date": daily_index, "symbol": symbol, "close": close, "daily_market_cap": daily_mcap}, index=daily_index)
    specs: list[FeatureSpec] = []
    feature_frames: dict[str, pd.Series] = {}
    aligned = {
        section: _align_fundamental(inputs[section], daily_index, filing_lag_days=config.filing_lag_days)
        for section in ("income", "balance", "cash", "ratios", "metrics", "income_growth", "balance_growth", "cash_growth")
    }
    aligned["balance"] = _add_tangible_book(aligned["balance"])
    total_debt = _daily_column(aligned, "balance", "total_debt")
    cash = _daily_column(aligned, "balance", "cash_and_cash_equivalents")
    daily_ev = daily_mcap.add(total_debt.reindex(daily_index), fill_value=0.0).sub(cash.reindex(daily_index), fill_value=0.0)

    _build_daily_adjusted_features(aligned, daily_mcap, daily_ev, feature_frames, specs)
    _build_statement_mcap_features(aligned, daily_mcap, feature_frames, specs)
    _build_financetoolkit_style_features(aligned, feature_frames, specs)

    if not feature_frames:
        return pd.DataFrame(), [], {"symbol": symbol, "status": "no_features"}
    feature_df = pd.DataFrame(feature_frames, index=daily_index)
    panel = pd.concat([panel, feature_df], axis=1)
    for horizon in config.horizons:
        panel[f"forward_return_{horizon}d"] = panel["close"].shift(-horizon) / panel["close"] - 1.0
    return panel.reset_index(drop=True), specs, {
        "symbol": symbol,
        "status": "ok",
        "rows": len(panel),
        "features": len(feature_frames),
        "seconds": perf_counter() - start,
    }


def _build_daily_adjusted_features(
    aligned: dict[str, pd.DataFrame],
    daily_mcap: pd.Series,
    daily_ev: pd.Series,
    feature_frames: dict[str, pd.Series],
    specs: list[FeatureSpec],
) -> None:
    mcap_items = {
        "revenue": ("income", "revenue", "higher_is_better"),
        "gross_profit": ("income", "gross_profit", "higher_is_better"),
        "operating_income": ("income", "operating_income", "higher_is_better"),
        "ebit": ("income", "ebit", "higher_is_better"),
        "ebitda": ("income", "ebitda", "higher_is_better"),
        "net_income": ("income", "net_income", "higher_is_better"),
        "operating_cash_flow": ("cash", "operating_cash_flow", "higher_is_better"),
        "free_cash_flow": ("cash", "free_cash_flow", "higher_is_better"),
        "book_equity": ("balance", "book_equity", "higher_is_better"),
        "tangible_book": ("balance", "tangible_book", "higher_is_better"),
        "cash": ("balance", "cash_and_cash_equivalents", "higher_is_better"),
        "cash_and_short_term_investments": ("balance", "cash_and_short_term_investments", "higher_is_better"),
        "total_debt": ("balance", "total_debt", "lower_is_better"),
        "net_debt": ("balance", "net_debt", "lower_is_better"),
    }
    for name, (section, column, direction) in mcap_items.items():
        values = _daily_column(aligned, section, column)
        if values.empty:
            continue
        _add_feature(
            feature_frames,
            specs,
            f"{name}_to_mcap",
            _safe_divide(values, daily_mcap),
            family="fmp_daily_mcap_yield",
            source="fmp",
            source_column=f"{section}.{column}",
            expected_direction=direction,
        )
        _add_feature(
            feature_frames,
            specs,
            f"mcap_to_{name}",
            _safe_divide(daily_mcap, values),
            family="fmp_daily_mcap_multiple",
            source="fmp",
            source_column=f"{section}.{column}",
            expected_direction="lower_is_better" if direction == "higher_is_better" else "higher_is_better",
        )

    ev_items = {
        "revenue": ("income", "revenue"),
        "gross_profit": ("income", "gross_profit"),
        "operating_income": ("income", "operating_income"),
        "ebit": ("income", "ebit"),
        "ebitda": ("income", "ebitda"),
        "operating_cash_flow": ("cash", "operating_cash_flow"),
        "free_cash_flow": ("cash", "free_cash_flow"),
    }
    for name, (section, column) in ev_items.items():
        values = _daily_column(aligned, section, column)
        if values.empty:
            continue
        _add_feature(
            feature_frames,
            specs,
            f"{name}_to_ev",
            _safe_divide(values, daily_ev),
            family="fmp_daily_ev_yield",
            source="fmp",
            source_column=f"{section}.{column}",
            expected_direction="higher_is_better",
        )
        _add_feature(
            feature_frames,
            specs,
            f"ev_to_{name}",
            _safe_divide(daily_ev, values),
            family="fmp_daily_ev_multiple",
            source="fmp",
            source_column=f"{section}.{column}",
            expected_direction="lower_is_better",
        )


def _build_statement_mcap_features(
    aligned: dict[str, pd.DataFrame],
    daily_mcap: pd.Series,
    feature_frames: dict[str, pd.Series],
    specs: list[FeatureSpec],
) -> None:
    for section, family in (("income", "fmp_income_mcap"), ("balance", "fmp_balance_mcap"), ("cash", "fmp_cash_mcap")):
        frame = aligned[section]
        for column in frame.columns:
            if column in {"fiscal_period", "accepted_date", "reported_currency", "calendar_year", "period"}:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() == 0:
                continue
            _add_feature(
                feature_frames,
                specs,
                column,
                _safe_divide(values, daily_mcap),
                family=family,
                source="fmp",
                source_column=f"{section}.{column}",
                expected_direction=_statement_direction(column),
            )


def _build_financetoolkit_style_features(
    aligned: dict[str, pd.DataFrame],
    feature_frames: dict[str, pd.Series],
    specs: list[FeatureSpec],
) -> None:
    for section in ("ratios", "metrics"):
        frame = aligned[section]
        for column in frame.columns:
            family = _family_for_ratio_column(column) if section == "ratios" else _family_for_metric_column(column)
            if family is None:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() == 0:
                continue
            _add_feature(
                feature_frames,
                specs,
                column,
                values,
                family=family,
                source="financetoolkit",
                source_column=f"{section}.{column}",
                expected_direction=_expected_direction(family, column),
            )

    for section, family in (
        ("income_growth", "ft_growth_income"),
        ("balance_growth", "ft_growth_balance"),
        ("cash_growth", "ft_growth_cash"),
    ):
        frame = aligned[section]
        for column in frame.columns:
            if not str(column).startswith("growth_"):
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() == 0:
                continue
            _add_feature(
                feature_frames,
                specs,
                column,
                values,
                family=family,
                source="financetoolkit",
                source_column=f"{section}.{column}",
                expected_direction="higher_is_better",
            )


def _rank_2d_nan(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype="float32")
    for i in range(values.shape[0]):
        row = values[i]
        valid = np.isfinite(row)
        count = int(valid.sum())
        if count == 0:
            continue
        order = np.argsort(row[valid], kind="mergesort")
        ranks = np.empty(count, dtype="float32")
        ranks[order] = np.arange(1, count + 1, dtype="float32")
        out[i, np.flatnonzero(valid)] = ranks
    return out


def _mean_center_nan(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=axis, keepdims=True)
    return values - mean


def _evaluate_features(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    features: list[str],
    *,
    horizons: tuple[int, ...],
    min_observations: int,
    include_spreads: bool,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    symbols = sorted(panel["symbol"].dropna().unique())
    feature_values = np.full((len(dates), len(symbols), len(features)), np.nan, dtype="float32")
    for idx, feature in enumerate(features):
        feature_values[:, :, idx] = (
            panel.pivot(index="date", columns="symbol", values=feature)
            .reindex(index=dates, columns=symbols)
            .to_numpy(dtype="float32")
        )
    signs = (
        metadata.set_index("feature")
        .loc[features, "expected_direction"]
        .map({"higher_is_better": 1.0, "lower_is_better": -1.0})
        .to_numpy(dtype="float32")
    )
    feature_scores = feature_values * signs.reshape(1, 1, -1)
    days, n_symbols, n_features = feature_scores.shape
    flat = feature_scores.transpose(0, 2, 1).reshape(days * n_features, n_symbols)
    feature_ranks = _rank_2d_nan(flat).reshape(days, n_features, n_symbols).transpose(0, 2, 1)
    centered_features = _mean_center_nan(feature_ranks, axis=1)
    rows = []
    for horizon in horizons:
        returns = (
            panel.pivot(index="date", columns="symbol", values=f"forward_return_{horizon}d")
            .reindex(index=dates, columns=symbols)
            .to_numpy(dtype="float32")
        )
        return_ranks = _rank_2d_nan(returns)
        centered_returns = _mean_center_nan(return_ranks, axis=1)
        valid = np.isfinite(centered_features) & np.isfinite(centered_returns[:, :, None])
        cf = np.where(valid, centered_features, 0.0)
        cr = np.where(np.isfinite(centered_returns), centered_returns, 0.0)
        numerator = np.einsum("dsf,ds->df", cf, cr)
        denominator = np.sqrt(np.einsum("dsf,dsf->df", cf, cf) * np.einsum("ds,ds->d", cr, cr)[:, None])
        daily_ic = numerator / denominator
        daily_ic[denominator == 0] = np.nan
        for feature_idx, feature in enumerate(features):
            feature_score = feature_scores[:, :, feature_idx]
            valid_pair = np.isfinite(feature_score) & np.isfinite(returns)
            obs = int(valid_pair.sum())
            if obs < min_observations:
                continue
            spreads = []
            if include_spreads:
                for day_idx in range(days):
                    mask = valid_pair[day_idx]
                    if int(mask.sum()) < 10:
                        continue
                    scores = feature_score[day_idx, mask]
                    rets = returns[day_idx, mask]
                    lo = np.nanquantile(scores, 0.2)
                    hi = np.nanquantile(scores, 0.8)
                    spreads.append(float(np.nanmean(rets[scores >= hi]) - np.nanmean(rets[scores <= lo])))
            ic = daily_ic[:, feature_idx]
            rows.append(
                {
                    "feature": feature,
                    "horizon": horizon,
                    "mean_daily_rank_ic": float(np.nanmean(ic)),
                    "median_daily_rank_ic": float(np.nanmedian(ic)),
                    "rank_ic_hit_rate": float(np.nanmean(ic > 0)),
                    "spread_bps": float(np.nanmedian(spreads) * 10000) if spreads else np.nan,
                    "observations": obs,
                }
            )
    return pd.DataFrame(rows)
