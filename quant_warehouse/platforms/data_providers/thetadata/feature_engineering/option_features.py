from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


GREEK_COLUMNS: tuple[str, ...] = ("delta", "gamma", "theta", "vega", "rho")
IV_COLUMNS: tuple[str, ...] = ("iv", "implied_volatility", "implied_vol")
MIN_IV = 1e-4
MAX_IV = 5.0


@dataclass(frozen=True)
class OptionFeatureSet:
    """ThetaData option features aligned one row per contract snapshot."""

    df: pd.DataFrame
    feature_cols: list[str]
    family_cols: dict[str, list[str]]


def build_option_contract_features(
    chain: pd.DataFrame,
    *,
    underlying_price: float | None = None,
    target_dte: int | None = None,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    compute_model_greeks: bool = True,
) -> OptionFeatureSet:
    """Build contract, liquidity, Greek, and IV features from a ThetaData chain.

    The function preserves the vendor chain columns and appends reusable option
    features. It does not price labels or select contracts.
    """

    if chain is None or chain.empty:
        return OptionFeatureSet(df=pd.DataFrame(), feature_cols=[], family_cols={})

    out = chain.copy()
    out.columns = [str(col).strip() for col in out.columns]
    _ensure_datetime(out, "snapshot_date")
    _ensure_datetime(out, "expiration")
    for col in ("strike", "bid", "ask", "mid", "volume", "open_interest", *GREEK_COLUMNS, *IV_COLUMNS):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "mid" not in out.columns or out["mid"].isna().all():
        if {"bid", "ask"}.issubset(out.columns):
            out["mid"] = (out["bid"] + out["ask"]) / 2.0

    family_cols: dict[str, list[str]] = {}
    contract_cols: list[str] = []
    if {"expiration", "snapshot_date"}.issubset(out.columns):
        out["dte"] = (out["expiration"] - out["snapshot_date"]).dt.days
        contract_cols.append("dte")
        if target_dte is not None:
            out["dte_gap"] = (out["dte"] - int(target_dte)).abs()
            contract_cols.append("dte_gap")
    if underlying_price is not None and np.isfinite(float(underlying_price)) and float(underlying_price) > 0:
        out["underlying_spot_entry"] = float(underlying_price)
        if "strike" in out.columns:
            out["moneyness"] = out["strike"] / float(underlying_price) - 1.0
            out["abs_moneyness"] = out["moneyness"].abs()
            contract_cols.extend(["moneyness", "abs_moneyness"])
    _add_family(family_cols, "contract_static", out, contract_cols)

    liquidity_cols: list[str] = []
    if {"bid", "ask"}.issubset(out.columns):
        out["spread"] = out["ask"] - out["bid"]
        liquidity_cols.append("spread")
        if "mid" in out.columns:
            out["spread_pct"] = out["spread"] / out["mid"].replace(0, np.nan)
            liquidity_cols.append("spread_pct")
    if "volume" in out.columns:
        liquidity_cols.append("volume")
    if "open_interest" in out.columns:
        liquidity_cols.append("open_interest")
    if "volume" in out.columns or "open_interest" in out.columns:
        volume = out["volume"] if "volume" in out.columns else pd.Series(0.0, index=out.index)
        open_interest = out["open_interest"] if "open_interest" in out.columns else pd.Series(0.0, index=out.index)
        out["liquidity_score"] = volume.fillna(0.0) + open_interest.fillna(0.0) / 100.0
        liquidity_cols.append("liquidity_score")
    _add_family(family_cols, "liquidity", out, liquidity_cols)

    if compute_model_greeks:
        _fill_black_scholes_features(
            out,
            risk_free_rate=float(risk_free_rate),
            dividend_yield=float(dividend_yield),
        )

    greek_cols: list[str] = []
    for col in GREEK_COLUMNS:
        if col in out.columns:
            greek_cols.append(col)
            out[f"abs_{col}"] = out[col].abs()
            greek_cols.append(f"abs_{col}")
    if "theta" in out.columns and "mid" in out.columns:
        out["theta_to_mid"] = out["theta"] / out["mid"].replace(0, np.nan)
        greek_cols.append("theta_to_mid")
    if "vega" in out.columns and "mid" in out.columns:
        out["vega_to_mid"] = out["vega"] / out["mid"].replace(0, np.nan)
        greek_cols.append("vega_to_mid")
    _add_family(family_cols, "greeks", out, greek_cols)

    iv_cols: list[str] = []
    iv_source = _first_present(out, IV_COLUMNS)
    if iv_source is not None:
        if iv_source != "iv":
            out["iv"] = out[iv_source]
        iv_cols.append("iv")
        group_cols = [col for col in ("snapshot_date", "underlying_symbol", "option_type", "expiration") if col in out.columns]
        if group_cols:
            grouped = out.groupby(group_cols, dropna=False)["iv"]
            out["iv_expiration_z"] = (out["iv"] - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)
            iv_cols.append("iv_expiration_z")
        if "dte" in out.columns:
            out["iv_times_sqrt_dte"] = out["iv"] * np.sqrt(out["dte"].clip(lower=0) / 365.0)
            iv_cols.append("iv_times_sqrt_dte")
    _add_family(family_cols, "iv_surface", out, iv_cols)

    feature_cols = [col for cols in family_cols.values() for col in cols]
    return OptionFeatureSet(df=out, feature_cols=feature_cols, family_cols=family_cols)


def option_ranker_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return available numeric option features suitable for a selector model."""

    preferred = [
        "dte",
        "dte_gap",
        "moneyness",
        "abs_moneyness",
        "spread_pct",
        "volume",
        "open_interest",
        "liquidity_score",
        "delta",
        "abs_delta",
        "gamma",
        "abs_gamma",
        "theta",
        "abs_theta",
        "vega",
        "abs_vega",
        "rho",
        "abs_rho",
        "theta_to_mid",
        "vega_to_mid",
        "iv",
        "iv_expiration_z",
        "iv_times_sqrt_dte",
    ]
    return [col for col in preferred if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().any()]


def _ensure_datetime(frame: pd.DataFrame, col: str) -> None:
    if col in frame.columns:
        frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.normalize()


def _add_family(families: dict[str, list[str]], name: str, frame: pd.DataFrame, cols: list[str]) -> None:
    usable = [col for col in cols if col in frame.columns]
    if usable:
        families[name] = list(dict.fromkeys(usable))


def _first_present(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for col in columns:
        if col in frame.columns and frame[col].notna().any():
            return col
    return None


def _fill_black_scholes_features(
    frame: pd.DataFrame,
    *,
    risk_free_rate: float,
    dividend_yield: float,
) -> None:
    required = {"underlying_spot_entry", "strike", "dte", "option_type", "mid"}
    if not required.issubset(frame.columns):
        return

    if "iv" not in frame.columns:
        frame["iv"] = np.nan
    frame["iv_model_source"] = np.where(frame["iv"].notna(), "vendor", pd.NA)

    missing_iv = frame["iv"].isna()
    if missing_iv.any():
        inferred = frame.loc[missing_iv].apply(
            lambda row: _infer_black_scholes_iv(
                spot=row.get("underlying_spot_entry"),
                strike=row.get("strike"),
                dte=row.get("dte"),
                option_type=row.get("option_type"),
                price=row.get("mid"),
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            ),
            axis=1,
        )
        frame.loc[missing_iv, "iv"] = inferred
        frame.loc[missing_iv & frame["iv"].notna(), "iv_model_source"] = "black_scholes_implied"

    if frame["iv"].isna().all():
        return

    missing_greeks = frame[list(GREEK_COLUMNS)].isna().any(axis=1) if set(GREEK_COLUMNS).issubset(frame.columns) else pd.Series(True, index=frame.index)
    if not missing_greeks.any():
        return

    computed = frame.loc[missing_greeks].apply(
        lambda row: _black_scholes_greeks(
            spot=row.get("underlying_spot_entry"),
            strike=row.get("strike"),
            dte=row.get("dte"),
            option_type=row.get("option_type"),
            volatility=row.get("iv"),
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        ),
        axis=1,
        result_type="expand",
    )
    if computed.empty:
        return

    computed.columns = list(GREEK_COLUMNS)
    for col in GREEK_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
        fill_mask = frame[col].isna() & computed[col].notna()
        frame.loc[fill_mask, col] = computed.loc[fill_mask, col]

    has_any_computed = computed.notna().any(axis=1)
    frame["greeks_model_source"] = pd.NA
    vendor_mask = frame[list(GREEK_COLUMNS)].notna().any(axis=1) & ~has_any_computed.reindex(frame.index, fill_value=False)
    frame.loc[vendor_mask, "greeks_model_source"] = "vendor"
    frame.loc[has_any_computed.index[has_any_computed], "greeks_model_source"] = "black_scholes"


def _infer_black_scholes_iv(
    *,
    spot: object,
    strike: object,
    dte: object,
    option_type: object,
    price: object,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    values = _coerce_bs_inputs(spot=spot, strike=strike, dte=dte, volatility=0.2)
    if values is None:
        return np.nan
    spot_f, strike_f, years, _ = values
    price_f = _finite_float(price)
    side = _option_side(option_type)
    if price_f is None or side is None or price_f <= 0:
        return np.nan

    intrinsic = max(spot_f - strike_f, 0.0) if side == "call" else max(strike_f - spot_f, 0.0)
    upper = spot_f if side == "call" else strike_f
    if price_f < intrinsic * 0.999 or price_f > upper * 1.001:
        return np.nan

    def objective(vol: float) -> float:
        return _black_scholes_price(
            spot=spot_f,
            strike=strike_f,
            years=years,
            option_type=side,
            volatility=vol,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        ) - price_f

    try:
        low = objective(MIN_IV)
        high = objective(MAX_IV)
        if low == 0:
            return MIN_IV
        if high == 0:
            return MAX_IV
        if low * high > 0:
            return np.nan
        return float(brentq(objective, MIN_IV, MAX_IV, maxiter=100))
    except (ValueError, RuntimeError, OverflowError, FloatingPointError):
        return np.nan


def _black_scholes_greeks(
    *,
    spot: object,
    strike: object,
    dte: object,
    option_type: object,
    volatility: object,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[float, float, float, float, float]:
    values = _coerce_bs_inputs(spot=spot, strike=strike, dte=dte, volatility=volatility)
    side = _option_side(option_type)
    if values is None or side is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    spot_f, strike_f, years, vol = values
    sqrt_t = np.sqrt(years)
    d1 = (
        np.log(spot_f / strike_f)
        + (risk_free_rate - dividend_yield + 0.5 * vol * vol) * years
    ) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    discount_q = np.exp(-dividend_yield * years)
    discount_r = np.exp(-risk_free_rate * years)
    pdf_d1 = norm.pdf(d1)

    if side == "call":
        delta = discount_q * norm.cdf(d1)
        theta = (
            -(spot_f * discount_q * pdf_d1 * vol) / (2.0 * sqrt_t)
            - risk_free_rate * strike_f * discount_r * norm.cdf(d2)
            + dividend_yield * spot_f * discount_q * norm.cdf(d1)
        ) / 365.0
        rho = strike_f * years * discount_r * norm.cdf(d2)
    else:
        delta = discount_q * (norm.cdf(d1) - 1.0)
        theta = (
            -(spot_f * discount_q * pdf_d1 * vol) / (2.0 * sqrt_t)
            + risk_free_rate * strike_f * discount_r * norm.cdf(-d2)
            - dividend_yield * spot_f * discount_q * norm.cdf(-d1)
        ) / 365.0
        rho = -strike_f * years * discount_r * norm.cdf(-d2)

    gamma = discount_q * pdf_d1 / (spot_f * vol * sqrt_t)
    vega = spot_f * discount_q * pdf_d1 * sqrt_t
    return (float(delta), float(gamma), float(theta), float(vega), float(rho))


def _black_scholes_price(
    *,
    spot: float,
    strike: float,
    years: float,
    option_type: str,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    sqrt_t = np.sqrt(years)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * years
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discount_q = np.exp(-dividend_yield * years)
    discount_r = np.exp(-risk_free_rate * years)
    if option_type == "call":
        return float(spot * discount_q * norm.cdf(d1) - strike * discount_r * norm.cdf(d2))
    return float(strike * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1))


def _coerce_bs_inputs(
    *,
    spot: object,
    strike: object,
    dte: object,
    volatility: object,
) -> tuple[float, float, float, float] | None:
    spot_f = _finite_float(spot)
    strike_f = _finite_float(strike)
    dte_f = _finite_float(dte)
    vol_f = _finite_float(volatility)
    if spot_f is None or strike_f is None or dte_f is None or vol_f is None:
        return None
    if spot_f <= 0 or strike_f <= 0 or dte_f <= 0 or vol_f <= 0:
        return None
    return spot_f, strike_f, dte_f / 365.0, vol_f


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _option_side(value: object) -> str | None:
    side = str(value).strip().lower()
    if side in {"c", "call"}:
        return "call"
    if side in {"p", "put"}:
        return "put"
    return None
