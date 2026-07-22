from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GREEK_COLUMNS: tuple[str, ...] = ("delta", "gamma", "theta", "vega", "rho")
IV_COLUMNS: tuple[str, ...] = ("iv", "implied_volatility", "implied_vol")


@dataclass(frozen=True)
class OptionFeatureSet:
    """ThetaData option features aligned one row per contract snapshot."""

    df: pd.DataFrame
    feature_cols: list[str]
    family_cols: dict[str, list[str]]


def filter_option_instrument_rows(
    chain: pd.DataFrame,
    *,
    require_change_percent: bool = True,
) -> pd.DataFrame:
    """Filter executable option rows for downstream instrument modeling.

    This runs after full-chain warehouse reads. ThetaData's reported
    ``change_percent`` is used as the one-day movement availability signal so
    a previous option-chain read is not required. The raw ``change`` field is
    not used because it is commonly zero in stored EOD rows.
    """
    if chain is None or chain.empty:
        return pd.DataFrame() if chain is None else chain.copy()
    out = chain.copy()
    required = {"bid", "ask"}
    if require_change_percent:
        required.add("change_percent")
    if not required.issubset(out.columns):
        return out.iloc[0:0].copy()
    bid = pd.to_numeric(out["bid"], errors="coerce")
    ask = pd.to_numeric(out["ask"], errors="coerce")
    valid = bid.gt(0) & ask.gt(0) & ask.ge(bid)
    if require_change_percent:
        change_percent = pd.to_numeric(out["change_percent"], errors="coerce")
        valid &= change_percent.notna() & change_percent.ne(0)
    return out.loc[valid].copy()


def build_option_contract_features(
    chain: pd.DataFrame,
    *,
    underlying_price: float | None = None,
    target_dte: int | None = None,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    compute_model_greeks: bool = False,
) -> OptionFeatureSet:
    """Build contract, liquidity, Greek, and IV features from a ThetaData chain.

    The function preserves the vendor chain columns and appends reusable option
    features. It does not price labels or select contracts.
    """

    # Compatibility only: ThetaData Greeks/IV are provider data now. Missing
    # values are left missing instead of being imputed with a pricing model.
    _ = risk_free_rate, dividend_yield, compute_model_greeks

    if chain is None or chain.empty:
        return OptionFeatureSet(df=pd.DataFrame(), feature_cols=[], family_cols={})

    out = chain.copy()
    out.columns = [str(col).strip() for col in out.columns]
    _ensure_datetime(out, "snapshot_date")
    _ensure_datetime(out, "expiration")
    for col in ("strike", "bid", "ask", "mid", "volume", "open_interest", *GREEK_COLUMNS, *IV_COLUMNS):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "underlying_price" in out.columns:
        out["underlying_price"] = pd.to_numeric(out["underlying_price"], errors="coerce")
        if underlying_price is None:
            chain_spot = out["underlying_price"].replace([np.inf, -np.inf], np.nan).dropna()
            chain_spot = chain_spot.loc[chain_spot > 0]
            if not chain_spot.empty:
                underlying_price = float(chain_spot.median())

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
