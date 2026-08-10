from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import polars as pl

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet


BASE_PRICE_COLS = ("open", "high", "low", "close", "volume")


def historical_price_eod_family(asset_class: str) -> str:
    """Return the canonical EOD price family for an asset class."""
    normalized = str(asset_class).strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("asset_class must not be empty")
    return f"{normalized}-historical-price-eod"


def normalize_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize a Polars price frame to a sorted date column."""
    if df is None or df.is_empty(): return pl.DataFrame()
    if "date" not in df.columns: raise ValueError("Price frames must contain a date column")
    expr = pl.col("date")
    if df.schema["date"] == pl.String: expr = expr.str.to_datetime(strict=False)
    else: expr = expr.cast(pl.Datetime, strict=False)
    return df.with_columns(expr.dt.replace_time_zone(None).dt.truncate("1d").alias("date")).drop_nulls("date").sort("date")


@dataclass(frozen=True)
class FeaturesResult:
    """Daily feature matrix and its usable feature columns."""

    df_daily: pl.DataFrame
    feature_cols: List[str]


def _ensure_dt_index(df: pl.DataFrame) -> pl.DataFrame:
    return normalize_cols(df).unique("date", keep="last")


def _pick_feature_cols(df_daily: pl.DataFrame) -> List[str]:
    cols = []
    for column in df_daily.columns:
        if column in BASE_PRICE_COLS or column == "symbol":
            continue
        if df_daily.schema[column].is_numeric():
            cols.append(column)
    return sorted(cols)


def _sanitize_features(
    df_daily: pl.DataFrame,
    feature_cols: List[str],
    *,
    fill_method: str = "ffill_bfill_zero",
) -> pl.DataFrame:
    out = df_daily.clone()
    if not feature_cols:
        return out

    matrix = out.select(feature_cols).with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in feature_cols])
    if fill_method == "drop_rows":
        return out.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in feature_cols]))
    if fill_method == "zero":
        matrix = matrix.fill_null(0.0)
    else:
        matrix = matrix.fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0)
    return out.drop(feature_cols).hstack(matrix)


def compute_features_worldclass(df: pl.DataFrame) -> pl.DataFrame:
    """Compute the dense OHLCV feature set with Polars."""
    return _compute_features_worldclass_polars(df)


def _compute_features_worldclass_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars-native implementation of the dense OHLCV feature recipe."""
    missing = [column for column in BASE_PRICE_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")
    eps = 1e-12
    out = df.with_columns([pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in BASE_PRICE_COLS])
    close, high, low, open_, vol = (pl.col(column) for column in ("close", "high", "low", "open", "volume"))

    def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
        return pl.when(denominator.abs() > eps).then(numerator / (denominator + eps)).otherwise(None)

    exprs: list[pl.Expr] = [close.pct_change().alias("Ret1d")]
    for window in [2, 3, 5, 10, 20, 21, 63, 126, 189, 252]:
        exprs.append(close.pct_change(window).alias(f"Ret{window}d"))
    for window in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200]:
        sma = close.rolling_mean(window_size=window)
        exprs.extend([sma.alias(f"SMA{window}"), safe_div(close - sma, sma).alias(f"DistSMA{window}"), sma.diff().alias(f"SMASlope{window}")])
    for window in [12, 26, 50]:
        ema = close.ewm_mean(span=window, adjust=False)
        exprs.append(safe_div(close - ema, ema).alias(f"DistEMA{window}"))

    ema12 = close.ewm_mean(span=12, adjust=False)
    ema26 = close.ewm_mean(span=26, adjust=False)
    macd = ema12 - ema26
    signal = macd.ewm_mean(span=9, adjust=False)
    exprs.extend([macd.alias("MACD"), signal.alias("MACDSignal"), (macd - signal).alias("MACDHist")])
    for window in [10, 20, 63]:
        mean = close.rolling_mean(window_size=window)
        std = close.rolling_std(window_size=window)
        upper, lower = mean + 2 * std, mean - 2 * std
        exprs.extend([safe_div(close - mean, std).alias(f"ZClose{window}"), safe_div(close - lower, upper - lower).alias(f"BBPos{window}")])
    prev_close = close.shift(1)
    exprs.extend([
        safe_div(high - low, close).alias("HlRange"),
        safe_div(close - open_, open_).alias("OcChange"),
        safe_div(open_ - prev_close, prev_close).alias("Gap"),
        pl.max_horizontal([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()]).alias("TrueRange"),
    ])
    out = out.with_columns(exprs)
    tr = pl.col("TrueRange")
    for window in [14, 20]:
        atr = tr.rolling_mean(window_size=window)
        out = out.with_columns(safe_div(atr, close).alias(f"ATRPct{window}"))
    ret = pl.col("Ret1d")
    for window in [5, 10, 20, 63]:
        vol_n = ret.rolling_std(window_size=window)
        base_mean = vol_n.rolling_mean(window_size=252)
        base_std = vol_n.rolling_std(window_size=252)
        out = out.with_columns([vol_n.alias(f"Vol{window}"), safe_div(vol_n - base_mean, base_std).alias(f"VolRegimeZ{window}")])
    for window in [10, 20, 55]:
        hh = high.rolling_max(window_size=window)
        ll = low.rolling_min(window_size=window)
        out = out.with_columns([
            (close > hh.shift(1)).cast(pl.Float64).alias(f"BreakoutUp{window}"),
            (close < ll.shift(1)).cast(pl.Float64).alias(f"BreakoutDn{window}"),
            safe_div(close - ll, hh - ll).alias(f"PosInChannel{window}"),
            safe_div(close - hh, hh).alias(f"DistHh{window}"),
            safe_div(close - ll, ll).alias(f"DistLl{window}"),
        ])
    for window in [5, 20, 63]:
        vmean = vol.rolling_mean(window_size=window)
        vstd = vol.rolling_std(window_size=window)
        out = out.with_columns(safe_div(vol - vmean, vstd).alias(f"VolZ{window}"))
    direction = close.diff().sign().fill_null(0.0)
    out = out.with_columns([
        (direction * vol.fill_null(0.0)).cum_sum().alias("OBV"),
        (close * vol).alias("DollarVol"),
    ])
    dollar = pl.col("DollarVol")
    out = out.with_columns([
        safe_div(dollar - dollar.rolling_mean(window_size=20), dollar.rolling_std(window_size=20)).alias("DollarVolZ20"),
        safe_div((close - low) - (high - close), high - low).alias("CLV"),
    ])
    return out.with_columns([
        pl.when(pl.col(column).is_finite()).then(pl.col(column)).otherwise(None).alias(column)
        for column in out.columns
        if out.schema[column] in (pl.Float32, pl.Float64)
    ])


def load_or_compute_features_daily(
    symbol: str,
    *,
    df_prices: pl.DataFrame,
    compute_fn: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None,
    compute_features_fn: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None,
) -> FeaturesResult:
    """Always recompute technical features from the provided Polars prices."""
    if compute_fn is not None and compute_features_fn is not None:
        raise ValueError("Pass only one of compute_fn or compute_features_fn.")
    selected_compute = compute_fn or compute_features_fn or compute_features_worldclass
    normalized = normalize_cols(df_prices)
    missing = [column for column in BASE_PRICE_COLS if column not in normalized.columns]
    if missing: raise ValueError(f"df_prices missing required columns: {missing}")
    daily = selected_compute(normalized)
    if not isinstance(daily, pl.DataFrame): raise TypeError("Technical feature computation must return Polars")
    features = [column for column, dtype in daily.schema.items() if column not in (*BASE_PRICE_COLS, "symbol", "date") and dtype.is_numeric()]
    if features: daily = daily.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_nan(None).alias(c) for c in features])
    return FeaturesResult(df_daily=daily, feature_cols=features)


def build_historical_price_eod_features(symbol: str, df_prices: pl.DataFrame, *, asset_class: str = "equity") -> BuiltFeatureSet:
    """Build the sparse endpoint-native historical-price family."""
    daily = normalize_cols(df_prices)
    if daily.is_empty(): return BuiltFeatureSet(df=pl.DataFrame(), feature_cols=[])
    source_columns = {"eod__adjusted_open": ("adj_open", "open"), "eod__adjusted_high": ("adj_high", "high"), "eod__adjusted_low": ("adj_low", "low"), "eod__adjusted_close": ("adj_close", "close"), "eod__volume": ("volume",)}
    optional_columns = {"eod__vwap": ("vwap",), "eod__change": ("change",), "eod__change_percent": ("change_percent", "change_pct")}
    selected: list[pl.Expr] = []
    feature_cols: list[str] = []
    for output, candidates in {**source_columns, **optional_columns}.items():
        candidate = next((value for value in candidates if value in daily.columns), None)
        if candidate is not None:
            selected.append(pl.col(candidate).cast(pl.Float64, strict=False).alias(output)); feature_cols.append(output)
    if not selected: return BuiltFeatureSet(df=pl.DataFrame(), feature_cols=[])
    out = daily.select([pl.col("date"), pl.lit(str(symbol).strip().upper()).alias("symbol"), *selected]).sort(["date", "symbol"])
    return BuiltFeatureSet(df=out, feature_cols=feature_cols, family_name=historical_price_eod_family(asset_class), endpoint_name="historical-price-eod", source_asset_class="equity")


def _to_snake(value: str) -> str:
    text = str(value).replace("%", "pct")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()
