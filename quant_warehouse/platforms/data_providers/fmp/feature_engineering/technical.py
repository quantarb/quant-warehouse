from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.feature_engineering.specs import BuiltFeatureSet


BASE_PRICE_COLS = ("open", "high", "low", "close", "volume")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common feature frames to a sorted datetime index."""
    if df is None or len(df) == 0:
        return df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.is_monotonic_increasing and df.index.tz is None:
            return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
        out = out.dropna(subset=["date"])
        return out.set_index("date").sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


@dataclass(frozen=True)
class FeaturesResult:
    """Daily feature matrix and its usable feature columns."""

    df_daily: pd.DataFrame
    feature_cols: List[str]


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def _pick_feature_cols(df_daily: pd.DataFrame) -> List[str]:
    cols = []
    for column in df_daily.columns:
        if column in BASE_PRICE_COLS or column == "symbol":
            continue
        if pd.api.types.is_numeric_dtype(df_daily[column]):
            cols.append(column)
    return sorted(cols)


def _sanitize_features(
    df_daily: pd.DataFrame,
    feature_cols: List[str],
    *,
    fill_method: str = "ffill_bfill_zero",
) -> pd.DataFrame:
    out = df_daily.copy()
    if not feature_cols:
        return out

    matrix = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    if fill_method == "drop_rows":
        mask = matrix.notna().all(axis=1)
        return out.loc[mask].copy()
    if fill_method == "zero":
        matrix = matrix.fillna(0.0)
    else:
        matrix = matrix.ffill().bfill().fillna(0.0)
    out[feature_cols] = matrix
    return out


def compute_features_worldclass(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a dense, no-lookahead OHLCV feature set."""
    cuda_result = _compute_features_worldclass_cuda(df)
    if cuda_result is not None:
        return cuda_result

    out = df.copy()
    eps = 1e-12

    def _safe_div(a, b):
        if hasattr(b, "replace"):
            b = b.replace(0, np.nan)
        return a / (b + eps)

    for column in BASE_PRICE_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    open_ = out["open"]
    high = out["high"]
    low = out["low"]
    close = out["close"]
    vol = out["volume"]

    feats: dict[str, pd.Series] = {}
    ret_1d = close.pct_change()
    feats["Ret1d"] = ret_1d
    for window in [2, 3, 5, 10, 20, 21, 63, 126, 189, 252]:
        feats[f"Ret{window}d"] = close.pct_change(window)

    for window in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200]:
        sma = close.rolling(window).mean()
        feats[f"SMA{window}"] = sma
        feats[f"DistSMA{window}"] = _safe_div(close - sma, sma)
        feats[f"SMASlope{window}"] = sma.diff()

    for window in [12, 26, 50]:
        ema = close.ewm(span=window, adjust=False).mean()
        feats[f"DistEMA{window}"] = _safe_div(close - ema, ema)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feats["MACD"] = macd
    feats["MACDSignal"] = signal
    feats["MACDHist"] = macd - signal

    for window in [10, 20, 63]:
        mean = close.rolling(window).mean()
        std = close.rolling(window).std()
        feats[f"ZClose{window}"] = _safe_div(close - mean, std + eps)
        upper = mean + 2 * std
        lower = mean - 2 * std
        feats[f"BBPos{window}"] = _safe_div(close - lower, (upper - lower) + eps)

    feats["HlRange"] = _safe_div(high - low, close)
    feats["OcChange"] = _safe_div(close - open_, open_)
    feats["Gap"] = _safe_div(open_ - close.shift(1), close.shift(1))

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    feats["TrueRange"] = tr

    for window in [14, 20]:
        atr = tr.rolling(window).mean()
        feats[f"ATRPct{window}"] = _safe_div(atr, close)

    for window in [5, 10, 20, 63]:
        vol_n = ret_1d.rolling(window).std()
        feats[f"Vol{window}"] = vol_n
        base_mean = vol_n.rolling(252).mean()
        base_std = vol_n.rolling(252).std()
        feats[f"VolRegimeZ{window}"] = _safe_div(vol_n - base_mean, base_std + eps)

    for window in [10, 20, 55]:
        hh = high.rolling(window).max()
        ll = low.rolling(window).min()
        feats[f"BreakoutUp{window}"] = (close > hh.shift(1)).astype(float)
        feats[f"BreakoutDn{window}"] = (close < ll.shift(1)).astype(float)
        feats[f"PosInChannel{window}"] = _safe_div(close - ll, (hh - ll) + eps)
        feats[f"DistHh{window}"] = _safe_div(close - hh, hh)
        feats[f"DistLl{window}"] = _safe_div(close - ll, ll)

    for window in [5, 20, 63]:
        vmean = vol.rolling(window).mean()
        vstd = vol.rolling(window).std()
        feats[f"VolZ{window}"] = _safe_div(vol - vmean, vstd + eps)

    direction = np.sign(close.diff()).fillna(0.0)
    feats["OBV"] = (direction * vol.fillna(0.0)).cumsum()
    dollar_vol = close * vol
    feats["DollarVol"] = dollar_vol
    feats["DollarVolZ20"] = _safe_div(
        dollar_vol - dollar_vol.rolling(20).mean(),
        dollar_vol.rolling(20).std() + eps,
    )
    feats["CLV"] = _safe_div((close - low) - (high - close), (high - low) + eps)

    feats_df = pd.DataFrame(feats, index=out.index)
    out = pd.concat([out, feats_df], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def _use_cuda_for_frame(df: pd.DataFrame) -> bool:
    raw = os.getenv("QW_FEATURE_ENGINEERING_CUDA", "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "never"}:
        return False
    if raw in {"1", "true", "yes", "on", "always"}:
        return True
    threshold = int(os.getenv("QW_FEATURE_ENGINEERING_CUDA_MIN_ROWS", "50000"))
    return len(df) >= threshold


def _compute_features_worldclass_cuda(df: pd.DataFrame) -> pd.DataFrame | None:
    """Run the dense price feature recipe with cudf when available.

    GPU dataframe startup and host/device transfer overhead is material for small
    per-symbol frames, so the default auto mode only tries cudf on larger inputs.
    Any cudf incompatibility falls back to the pandas implementation.
    """
    if not _use_cuda_for_frame(df):
        return None
    try:
        import cudf  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        gdf = cudf.from_pandas(df.copy())
        eps = 1e-12
        for column in BASE_PRICE_COLS:
            gdf[column] = cudf.to_numeric(gdf[column], errors="coerce")

        open_ = gdf["open"]
        high = gdf["high"]
        low = gdf["low"]
        close = gdf["close"]
        vol = gdf["volume"]

        def _safe_div(a, b):
            return a / (b.replace(0, np.nan) + eps)

        feats = {}
        ret_1d = close.pct_change()
        feats["Ret1d"] = ret_1d
        for window in [2, 3, 5, 10, 20, 21, 63, 126, 189, 252]:
            feats[f"Ret{window}d"] = close.pct_change(window)
        for window in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200]:
            sma = close.rolling(window).mean()
            feats[f"SMA{window}"] = sma
            feats[f"DistSMA{window}"] = _safe_div(close - sma, sma)
            feats[f"SMASlope{window}"] = sma.diff()
        for window in [12, 26, 50]:
            ema = close.ewm(span=window, adjust=False).mean()
            feats[f"DistEMA{window}"] = _safe_div(close - ema, ema)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        feats["MACD"] = macd
        feats["MACDSignal"] = signal
        feats["MACDHist"] = macd - signal

        for window in [10, 20, 63]:
            mean = close.rolling(window).mean()
            std = close.rolling(window).std()
            feats[f"ZClose{window}"] = _safe_div(close - mean, std + eps)
            upper = mean + 2 * std
            lower = mean - 2 * std
            feats[f"BBPos{window}"] = _safe_div(close - lower, (upper - lower) + eps)

        feats["HlRange"] = _safe_div(high - low, close)
        feats["OcChange"] = _safe_div(close - open_, open_)
        feats["Gap"] = _safe_div(open_ - close.shift(1), close.shift(1))

        prev_close = close.shift(1)
        tr = cudf.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        feats["TrueRange"] = tr
        for window in [14, 20]:
            atr = tr.rolling(window).mean()
            feats[f"ATRPct{window}"] = _safe_div(atr, close)
        for window in [5, 10, 20, 63]:
            vol_n = ret_1d.rolling(window).std()
            feats[f"Vol{window}"] = vol_n
            base_mean = vol_n.rolling(252).mean()
            base_std = vol_n.rolling(252).std()
            feats[f"VolRegimeZ{window}"] = _safe_div(vol_n - base_mean, base_std + eps)
        for window in [10, 20, 55]:
            hh = high.rolling(window).max()
            ll = low.rolling(window).min()
            feats[f"BreakoutUp{window}"] = (close > hh.shift(1)).astype("float64")
            feats[f"BreakoutDn{window}"] = (close < ll.shift(1)).astype("float64")
            feats[f"PosInChannel{window}"] = _safe_div(close - ll, (hh - ll) + eps)
            feats[f"DistHh{window}"] = _safe_div(close - hh, hh)
            feats[f"DistLl{window}"] = _safe_div(close - ll, ll)
        for window in [5, 20, 63]:
            vmean = vol.rolling(window).mean()
            vstd = vol.rolling(window).std()
            feats[f"VolZ{window}"] = _safe_div(vol - vmean, vstd + eps)

        close_diff = close.diff()
        direction = (close_diff > 0).astype("float64") - (close_diff < 0).astype("float64")
        direction = direction.fillna(0.0)
        feats["OBV"] = (direction * vol.fillna(0.0)).cumsum()
        dollar_vol = close * vol
        feats["DollarVol"] = dollar_vol
        feats["DollarVolZ20"] = _safe_div(
            dollar_vol - dollar_vol.rolling(20).mean(),
            dollar_vol.rolling(20).std() + eps,
        )
        feats["CLV"] = _safe_div((close - low) - (high - close), (high - low) + eps)

        out = cudf.concat([gdf, cudf.DataFrame(feats, index=gdf.index)], axis=1).to_pandas()
        return out.replace([np.inf, -np.inf], np.nan)
    except Exception:
        return None


def load_or_compute_features_daily(
    symbol: str,
    *,
    df_prices: pd.DataFrame,
    compute_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    compute_features_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> FeaturesResult:
    """Always recompute technical features from the provided prices."""

    if compute_fn is not None and compute_features_fn is not None:
        raise ValueError("Pass only one of compute_fn or compute_features_fn.")
    if compute_fn is None:
        compute_fn = compute_features_fn
    if compute_fn is None:
        compute_fn = compute_features_worldclass

    df_prices_n = normalize_cols(df_prices)
    df_prices_n = _ensure_dt_index(df_prices_n)
    missing = [column for column in BASE_PRICE_COLS if column not in df_prices_n.columns]
    if missing:
        raise ValueError(f"df_prices missing required columns: {missing}")

    df_daily = compute_fn(df_prices_n.copy())
    df_daily = normalize_cols(df_daily)
    df_daily = _ensure_dt_index(df_daily)

    for column in BASE_PRICE_COLS:
        if column not in df_daily.columns:
            df_daily[column] = df_prices_n[column]

    feature_cols = _pick_feature_cols(df_daily)
    df_daily = _sanitize_features(df_daily, feature_cols, fill_method="drop_rows")
    feature_cols = _pick_feature_cols(df_daily)
    return FeaturesResult(df_daily=df_daily, feature_cols=feature_cols)


def build_price_technical_features(symbol: str, df_prices: pd.DataFrame) -> BuiltFeatureSet:
    """Build prefixed price/technical features for a single symbol."""

    if df_prices.empty:
        return BuiltFeatureSet(df=pd.DataFrame(), feature_cols=[])
    df_daily = compute_features_worldclass(df_prices.copy())
    feature_cols = [
        column
        for column in df_daily.columns
        if column not in BASE_PRICE_COLS and column != "symbol" and pd.api.types.is_numeric_dtype(df_daily[column])
    ]
    rename_map = {column: f"px__{_to_snake(column)}" for column in feature_cols}
    out = df_daily[feature_cols].rename(columns=rename_map).copy()
    out["symbol"] = str(symbol).strip().upper()
    out = out.reset_index().rename(columns={out.index.name or "index": "date"}).set_index(["date", "symbol"]).sort_index()
    renamed_feature_cols = [rename_map[column] for column in feature_cols]
    return BuiltFeatureSet(df=out, feature_cols=renamed_feature_cols)


def _to_snake(value: str) -> str:
    text = str(value).replace("%", "pct")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()
