from __future__ import annotations

import contextlib
import logging
import inspect
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from quant_warehouse.feature_engineering.specs import BuiltFeatureSet
from quant_warehouse.feature_engineering.technical import BASE_PRICE_COLS, _ensure_dt_index, _to_snake, normalize_cols


TA_CLASSIC_FAMILY_PREFIXES: dict[str, str] = {
    "technical_candles": "ta_candle__",
    "technical_cycles": "ta_cycle__",
    "technical_math": "ta_math__",
    "technical_momentum": "ta_momentum__",
    "technical_overlap": "ta_overlap__",
    "technical_performance": "ta_performance__",
}


@dataclass(frozen=True)
class TaIndicatorSpec:
    name: str
    fn_name: str
    inputs: tuple[str, ...]
    kwargs: dict[str, object] | None = None
    min_rows: int = 1


def build_price_ta_classic_feature_families(
    symbol: str,
    df_prices: pd.DataFrame,
) -> dict[str, BuiltFeatureSet]:
    """Build split pandas-ta-classic technical feature families for a single symbol."""

    if df_prices.empty:
        return _empty_family_sets()
    ta = _import_pandas_ta_classic()
    prices = _prepare_price_frame(df_prices)
    if prices.empty:
        return _empty_family_sets()

    result: dict[str, BuiltFeatureSet] = {}
    with _suppress_pandas_ta_classic_row_warnings():
        for family_name, specs in _indicator_specs(ta).items():
            columns: dict[str, pd.Series] = {}
            for spec in specs:
                indicator = _compute_indicator(ta, prices, spec)
                if indicator.empty:
                    continue
                for column in indicator.columns:
                    out_col = _feature_column_name(family_name, spec.name, column)
                    columns[out_col] = pd.to_numeric(indicator[column], errors="coerce")
            frame = pd.DataFrame(columns, index=prices.index) if columns else pd.DataFrame(index=prices.index)
            feature_cols = _usable_feature_cols(frame)
            result[family_name] = _to_built_feature_set(symbol, frame, feature_cols)
    return result


def _import_pandas_ta_classic():
    try:
        import pandas_ta_classic as ta
    except ImportError as exc:
        raise ImportError(
            "pandas-ta-classic is required for split technical feature families. "
            "Install it with `pip install pandas-ta-classic`."
        ) from exc
    return ta


@contextlib.contextmanager
def _suppress_pandas_ta_classic_row_warnings():
    loggers = [
        logging.getLogger("pandas_ta_classic.utils._core"),
        logging.getLogger("pandas_ta_classic.overlap.vwap"),
    ]
    previous_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*VWAP volume series is not datetime ordered.*")
            warnings.filterwarnings("ignore", message=".*VWAP price series is not datetime ordered.*")
            warnings.filterwarnings("ignore", message=".*divide by zero encountered in log10.*")
            warnings.filterwarnings("ignore", message=".*overflow encountered in cast.*")
            warnings.filterwarnings("ignore", message=".*invalid value encountered in sqrt.*")
            warnings.filterwarnings("ignore", message=".*divide by zero encountered in divide.*")
            warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
            yield
    finally:
        for logger, previous_level in zip(loggers, previous_levels):
            logger.setLevel(previous_level)


def _empty_family_sets() -> dict[str, BuiltFeatureSet]:
    return {
        family_name: BuiltFeatureSet(df=pd.DataFrame(), feature_cols=[])
        for family_name in TA_CLASSIC_FAMILY_PREFIXES
    }


def _prepare_price_frame(df_prices: pd.DataFrame) -> pd.DataFrame:
    out = normalize_cols(df_prices)
    out = _ensure_dt_index(out)
    missing = [column for column in BASE_PRICE_COLS if column not in out.columns]
    if missing:
        raise ValueError(f"df_prices missing required columns for pandas-ta-classic features: {missing}")
    out = out.loc[:, list(BASE_PRICE_COLS)].copy()
    for column in BASE_PRICE_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.loc[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _compute_indicator(ta, prices: pd.DataFrame, spec: TaIndicatorSpec) -> pd.DataFrame:
    if len(prices) < int(spec.min_rows):
        return pd.DataFrame(index=prices.index)
    fn: Callable[..., object] | None = getattr(ta, spec.fn_name, None)
    if fn is None:
        return pd.DataFrame(index=prices.index)
    special = _compute_special_indicator(ta, prices, spec.fn_name)
    if special is not None:
        return special
    kwargs = dict(spec.kwargs or {})
    call_args = {input_name: prices[input_name] for input_name in spec.inputs if input_name in prices.columns}
    if "open" in call_args:
        call_args["open_"] = call_args.pop("open")
    if spec.fn_name == "log_return" and "close" in call_args:
        close_series = pd.to_numeric(call_args["close"], errors="coerce")
        call_args["close"] = close_series.where(close_series > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        try:
            raw = fn(**call_args, **kwargs)
        except Exception:
            return pd.DataFrame(index=prices.index)
    if raw is None:
        return pd.DataFrame(index=prices.index)
    if isinstance(raw, tuple):
        frames: list[pd.DataFrame] = []
        for part in raw:
            if part is None:
                continue
            if isinstance(part, pd.Series):
                name = str(part.name or spec.name)
                part_df = part.rename(name).to_frame()
            elif isinstance(part, pd.DataFrame):
                part_df = part
            else:
                continue
            frames.append(part_df.reindex(prices.index))
        if not frames:
            return pd.DataFrame(index=prices.index)
        out = pd.concat(frames, axis=1)
        return out.loc[:, ~out.columns.duplicated()]
    if isinstance(raw, pd.Series):
        name = str(raw.name or spec.name)
        return raw.rename(name).to_frame().reindex(prices.index)
    if isinstance(raw, pd.DataFrame):
        return raw.reindex(prices.index)
    return pd.DataFrame(index=prices.index)


def _compute_special_indicator(ta, prices: pd.DataFrame, fn_name: str) -> pd.DataFrame | None:
    close = prices["close"]
    high = prices["high"]
    low = prices["low"]
    open_ = prices["open"]

    if fn_name == "ma":
        raw = ta.ma(name="ema", source=close)
    elif fn_name == "mavp":
        if len(prices) < 30:
            return pd.DataFrame(index=prices.index)
        periods = pd.Series(14, index=prices.index, dtype="float64")
        raw = ta.mavp(close=close, periods=periods, minperiod=2, maxperiod=30)
    elif fn_name == "beta":
        benchmark = close.rolling(20, min_periods=1).mean()
        raw = ta.beta(close=close, benchmark=benchmark, length=20)
    elif fn_name == "correl":
        benchmark = close.rolling(20, min_periods=1).mean()
        raw = ta.correl(close=close, benchmark=benchmark, length=20)
    elif fn_name == "long_run":
        if len(prices) < 26:
            return pd.DataFrame(index=prices.index)
        ema_fast = ta.ema(close, length=12)
        ema_slow = ta.ema(close, length=26)
        raw = ta.long_run(fast=ema_fast, slow=ema_slow, length=20)
    elif fn_name == "short_run":
        if len(prices) < 26:
            return pd.DataFrame(index=prices.index)
        ema_fast = ta.ema(close, length=12)
        ema_slow = ta.ema(close, length=26)
        raw = ta.short_run(fast=ema_fast, slow=ema_slow, length=20)
    elif fn_name == "tsignals":
        if len(prices) < 26:
            return pd.DataFrame(index=prices.index)
        ema_fast = ta.ema(close, length=12)
        ema_slow = ta.ema(close, length=26)
        trend = ema_fast > ema_slow
        raw = ta.tsignals(trend=trend, asbool=False)
    elif fn_name == "xsignals":
        if len(prices) < 26:
            return pd.DataFrame(index=prices.index)
        rsi_14 = ta.rsi(close, length=14)
        raw = ta.xsignals(signal=rsi_14, xa=70, xb=30, above=True, long=True, asbool=False)
    elif fn_name == "cpr":
        raw = ta.cpr(open=open_, high=high, low=low, close=close)
    elif fn_name == "tos_stdevall":
        length = 20
        x = np.arange(length, dtype=float)
        centered_x = x - x.mean()
        denominator = float(np.dot(centered_x, centered_x))
        slope = close.rolling(length, min_periods=length).apply(
            lambda values: float(np.dot(centered_x, values) / denominator),
            raw=True,
        )
        regression = close.rolling(length, min_periods=length).mean() + slope * centered_x[-1]
        stdev = close.rolling(length, min_periods=length).std(ddof=1)
        raw = pd.DataFrame({"TOS_STDEVALL_20_LR": regression}, index=prices.index)
        for multiple in (1, 2, 3):
            raw[f"TOS_STDEVALL_20_L_{multiple}"] = regression - multiple * stdev
            raw[f"TOS_STDEVALL_20_U_{multiple}"] = regression + multiple * stdev
    else:
        return None

    if raw is None:
        return pd.DataFrame(index=prices.index)
    if isinstance(raw, pd.Series):
        name = str(raw.name or fn_name)
        return raw.rename(name).to_frame().reindex(prices.index)
    if isinstance(raw, pd.DataFrame):
        return raw.reindex(prices.index)
    return pd.DataFrame(index=prices.index)


def _feature_column_name(family_name: str, spec_name: str, raw_column: str) -> str:
    prefix = TA_CLASSIC_FAMILY_PREFIXES[family_name]
    raw = _to_snake(raw_column)
    base = _to_snake(spec_name)
    if raw.startswith(base):
        core = raw
    else:
        core = f"{base}_{raw}"
    return f"{prefix}{core}"


def _usable_feature_cols(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    float32_max = np.finfo(np.float32).max
    for column in frame.columns:
        series = frame[column].replace([np.inf, -np.inf], np.nan)
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if series.notna().any():
            cleaned = pd.to_numeric(series.ffill().fillna(0.0), errors="coerce").replace([np.inf, -np.inf], np.nan)
            cleaned = cleaned.clip(lower=-float32_max, upper=float32_max).fillna(0.0)
            frame[column] = cleaned.astype(np.float32)
            cols.append(column)
    return list(dict.fromkeys(cols))


def _to_built_feature_set(symbol: str, frame: pd.DataFrame, feature_cols: list[str]) -> BuiltFeatureSet:
    if frame.empty or not feature_cols:
        return BuiltFeatureSet(df=pd.DataFrame(), feature_cols=[])
    out = frame.loc[:, feature_cols].copy()
    out["symbol"] = str(symbol).strip().upper()
    out = out.reset_index().rename(columns={out.index.name or "index": "date"}).set_index(["date", "symbol"]).sort_index()
    return BuiltFeatureSet(df=out, feature_cols=list(feature_cols))


def _indicator_specs(ta) -> dict[str, tuple[TaIndicatorSpec, ...]]:
    specs: dict[str, list[TaIndicatorSpec]] = {
        "technical_candles": list(_candlestick_specs(ta)),
        "technical_cycles": [
            TaIndicatorSpec("ebsw_40_10", "ebsw", ("close",), {"length": 40, "bars": 10}, min_rows=40),
            TaIndicatorSpec("ht_dcperiod", "ht_dcperiod", ("close",)),
            TaIndicatorSpec("ht_dcphase", "ht_dcphase", ("close",)),
            TaIndicatorSpec("ht_phasor", "ht_phasor", ("close",)),
            TaIndicatorSpec("ht_sine", "ht_sine", ("close",)),
            TaIndicatorSpec("ht_trendmode", "ht_trendmode", ("close",)),
        ],
        "technical_math": [
            TaIndicatorSpec("zscore_20", "zscore", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("zscore_63", "zscore", ("close",), {"length": 63}, min_rows=63),
            TaIndicatorSpec("entropy_20", "entropy", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("stdev_20", "stdev", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("variance_20", "variance", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("skew_20", "skew", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("kurtosis_20", "kurtosis", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("slope_20", "slope", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("mad_20", "mad", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("median_20", "median", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("quantile_20_25", "quantile", ("close",), {"length": 20, "q": 0.25}, min_rows=20),
            TaIndicatorSpec("quantile_20_50", "quantile", ("close",), {"length": 20, "q": 0.5}, min_rows=20),
            TaIndicatorSpec("quantile_20_75", "quantile", ("close",), {"length": 20, "q": 0.75}, min_rows=20),
            TaIndicatorSpec("linregslope_20", "linregslope", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("tos_stdevall_20", "tos_stdevall", ("close",), {"length": 20}, min_rows=20),
        ],
        "technical_momentum": [
            TaIndicatorSpec("rsi_14", "rsi", ("close",), {"length": 14}, min_rows=14),
            TaIndicatorSpec("macd", "macd", ("close",), min_rows=26),
            TaIndicatorSpec("macdext", "macdext", ("close",), min_rows=26),
            TaIndicatorSpec("macdfix", "macdfix", ("close",), min_rows=26),
            TaIndicatorSpec("stoch", "stoch", ("high", "low", "close"), min_rows=14),
            TaIndicatorSpec("stochf", "stochf", ("high", "low", "close"), min_rows=14),
            TaIndicatorSpec("stochrsi", "stochrsi", ("close",), {"length": 14, "rsi_length": 14}, min_rows=14),
            TaIndicatorSpec("cci_20", "cci", ("high", "low", "close"), {"length": 20}, min_rows=20),
            TaIndicatorSpec("roc_10", "roc", ("close",), {"length": 10}, min_rows=10),
            TaIndicatorSpec("mom_10", "mom", ("close",), {"length": 10}, min_rows=10),
            TaIndicatorSpec("willr_14", "willr", ("high", "low", "close"), {"length": 14}, min_rows=14),
            TaIndicatorSpec("ppo", "ppo", ("close",), min_rows=26),
            TaIndicatorSpec("cmo_14", "cmo", ("close",), {"length": 14}, min_rows=14),
            TaIndicatorSpec("bop", "bop", ("open", "high", "low", "close")),
            TaIndicatorSpec("ao", "ao", ("high", "low"), min_rows=34),
            TaIndicatorSpec("adx_14", "adx", ("high", "low", "close"), {"length": 14}, min_rows=14),
            TaIndicatorSpec("atr_14", "atr", ("high", "low", "close"), {"length": 14}, min_rows=14),
            TaIndicatorSpec("mfi_14", "mfi", ("high", "low", "close", "volume"), {"length": 14}, min_rows=14),
            TaIndicatorSpec("obv", "obv", ("close", "volume")),
            TaIndicatorSpec("adosc", "adosc", ("high", "low", "close", "volume"), min_rows=10),
            TaIndicatorSpec("dpo_20", "dpo", ("close",), {"length": 20, "lookahead": False}, min_rows=20),
            TaIndicatorSpec("kst", "kst", ("close",), min_rows=30),
            TaIndicatorSpec("stc", "stc", ("close",), min_rows=50),
            TaIndicatorSpec("tsi", "tsi", ("close",), min_rows=25),
            TaIndicatorSpec("trix", "trix", ("close",), min_rows=18),
            TaIndicatorSpec("trixh", "trixh", ("close",), min_rows=18),
            TaIndicatorSpec("uo", "uo", ("high", "low", "close"), min_rows=28),
            TaIndicatorSpec("ui_14", "ui", ("close",), {"length": 14}, min_rows=14),
            TaIndicatorSpec("natr_14", "natr", ("high", "low", "close"), {"length": 14}, min_rows=14),
            TaIndicatorSpec("pvo", "pvo", ("volume",), min_rows=26),
            TaIndicatorSpec("pvol", "pvol", ("close", "volume")),
        ],
        "technical_overlap": [
            TaIndicatorSpec("sma_10", "sma", ("close",), {"length": 10}, min_rows=10),
            TaIndicatorSpec("sma_20", "sma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("sma_50", "sma", ("close",), {"length": 50}, min_rows=50),
            TaIndicatorSpec("ema_12", "ema", ("close",), {"length": 12}, min_rows=12),
            TaIndicatorSpec("ema_26", "ema", ("close",), {"length": 26}, min_rows=26),
            TaIndicatorSpec("ema_50", "ema", ("close",), {"length": 50}, min_rows=50),
            TaIndicatorSpec("dema_20", "dema", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("tema_20", "tema", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("hma_20", "hma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("wma_20", "wma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("kama_20", "kama", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("alma_20", "alma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("vwma_20", "vwma", ("close", "volume"), {"length": 20}, min_rows=20),
            TaIndicatorSpec("vwap", "vwap", ("high", "low", "close", "volume"), min_rows=2),
            TaIndicatorSpec("bbands_20", "bbands", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("kc_20", "kc", ("high", "low", "close"), {"length": 20}, min_rows=20),
            TaIndicatorSpec("donchian_20", "donchian", ("high", "low"), {"lower_length": 20, "upper_length": 20}, min_rows=20),
            TaIndicatorSpec("supertrend_7_3", "supertrend", ("high", "low", "close"), {"length": 7, "multiplier": 3.0}, min_rows=7),
            TaIndicatorSpec("psar", "psar", ("high", "low", "close")),
            TaIndicatorSpec("rma_20", "rma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("vidya_14", "vidya", ("close",), {"length": 14}, min_rows=14),
            TaIndicatorSpec("zlma_20", "zlma", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("midpoint_20", "midpoint", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("midprice_20", "midprice", ("high", "low"), {"length": 20}, min_rows=20),
            TaIndicatorSpec("amat", "amat", ("close",), {"fast": 8, "slow": 21, "lookback": 2}, min_rows=21),
            TaIndicatorSpec("hl2", "hl2", ("high", "low")),
            TaIndicatorSpec("hlc3", "hlc3", ("high", "low", "close")),
            TaIndicatorSpec("ohlc4", "ohlc4", ("open", "high", "low", "close")),
        ],
        "technical_performance": [
            TaIndicatorSpec("pct_return_1", "percent_return", ("close",), {"length": 1}, min_rows=2),
            TaIndicatorSpec("pct_return_5", "percent_return", ("close",), {"length": 5}, min_rows=5),
            TaIndicatorSpec("pct_return_20", "percent_return", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("log_return_1", "log_return", ("close",), {"length": 1}, min_rows=2),
            TaIndicatorSpec("log_return_5", "log_return", ("close",), {"length": 5}, min_rows=5),
            TaIndicatorSpec("log_return_20", "log_return", ("close",), {"length": 20}, min_rows=20),
            TaIndicatorSpec("drawdown", "drawdown", ("close",)),
        ],
    }

    existing = {spec.fn_name for family_specs in specs.values() for spec in family_specs}
    all_builtin_names = sorted({fn for functions in ta.Category.values() for fn in functions})
    for fn_name in all_builtin_names:
        if fn_name in existing:
            continue
        family_name = _family_for_builtin_indicator(ta, fn_name)
        spec = _auto_indicator_spec(ta, fn_name)
        if spec is None:
            continue
        specs[family_name].append(spec)
        existing.add(fn_name)

    return {family_name: tuple(family_specs) for family_name, family_specs in specs.items()}


def _candlestick_specs(ta) -> tuple[TaIndicatorSpec, ...]:
    return (
        TaIndicatorSpec("doji", "cdl_doji", ("open", "high", "low", "close"), min_rows=10),
        TaIndicatorSpec("inside", "cdl_inside", ("open", "high", "low", "close"), {"asbool": False}),
        TaIndicatorSpec("pattern_all", "cdl_pattern", ("open", "high", "low", "close"), {"name": "all"}, min_rows=10),
        TaIndicatorSpec("ha", "ha", ("open", "high", "low", "close")),
        TaIndicatorSpec("candle_z", "cdl_z", ("open", "high", "low", "close"), {"length": 20}, min_rows=20),
    )


def _family_for_builtin_indicator(ta, fn_name: str) -> str:
    for category_name, fn_names in ta.Category.items():
        if fn_name not in fn_names:
            continue
        if category_name == "candles":
            return "technical_candles"
        if category_name == "cycles":
            return "technical_cycles"
        if category_name == "statistics":
            return "technical_math"
        if category_name in {"overlap", "volatility"}:
            return "technical_overlap"
        if category_name == "performance":
            return "technical_performance"
        return "technical_momentum"
    return "technical_momentum"


def _auto_indicator_spec(ta, fn_name: str) -> TaIndicatorSpec | None:
    special_specs: dict[str, TaIndicatorSpec] = {
        "ma": TaIndicatorSpec("ma_ema", "ma", ("close",)),
        "mavp": TaIndicatorSpec("mavp_14", "mavp", ("close",)),
        "beta": TaIndicatorSpec("beta_20", "beta", ("close",)),
        "correl": TaIndicatorSpec("correl_20", "correl", ("close",)),
        "long_run": TaIndicatorSpec("long_run", "long_run", ("close",), min_rows=26),
        "short_run": TaIndicatorSpec("short_run", "short_run", ("close",), min_rows=26),
        "tsignals": TaIndicatorSpec("tsignals", "tsignals", ("close",), min_rows=26),
        "xsignals": TaIndicatorSpec("xsignals", "xsignals", ("close",), min_rows=26),
        "cpr": TaIndicatorSpec("cpr", "cpr", ("open", "high", "low", "close")),
        "ichimoku": TaIndicatorSpec(
            "ichimoku",
            "ichimoku",
            ("high", "low", "close"),
            {"lookahead": False, "include_chikou": False},
            min_rows=52,
        ),
    }
    if fn_name in special_specs:
        return special_specs[fn_name]

    fn = getattr(ta, fn_name, None)
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
    except Exception:
        return None

    inputs: list[str] = []
    for param in sig.parameters.values():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if param.default is not inspect._empty:
            continue
        if param.name in {"open", "open_"}:
            inputs.append("open")
        elif param.name == "high":
            inputs.append("high")
        elif param.name == "low":
            inputs.append("low")
        elif param.name == "close":
            inputs.append("close")
        elif param.name == "volume":
            inputs.append("volume")
        elif param.name in {"fast", "slow", "signal", "trend", "source", "benchmark", "periods", "name", "xa", "xb"}:
            continue
        else:
            return None

    if not inputs:
        inputs = ["close"]

    param_names = {
        param.name
        for param in sig.parameters.values()
        if param.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    min_rows = 1
    if {"fast", "slow"}.issubset(param_names):
        min_rows = 26
    elif {"fast", "slow", "signal"}.issubset(param_names):
        min_rows = 30
    elif {
        "length",
        "period",
        "periods",
        "lower_length",
        "upper_length",
        "high_length",
        "low_length",
        "atr_length",
        "bb_length",
        "kc_length",
        "mom_length",
        "rsi_length",
        "lookback",
    } & param_names:
        min_rows = 20

    return TaIndicatorSpec(_to_snake(fn_name), fn_name, tuple(dict.fromkeys(inputs)), min_rows=min_rows)
