from __future__ import annotations

from dataclasses import dataclass
import math
import polars as pl


GREEK_COLUMNS: tuple[str, ...] = ("delta", "gamma", "theta", "vega", "rho")
IV_COLUMNS: tuple[str, ...] = ("iv", "implied_volatility", "implied_vol")
Frame = pl.DataFrame


@dataclass(frozen=True)
class OptionFeatureSet:
    """Option market fields aligned one row per contract snapshot."""

    df: Frame
    feature_cols: list[str]
    family_cols: dict[str, list[str]]
    family_name: str = "option-historical-price-eod"
    endpoint_name: str = "option_history_greeks_eod"
    source_asset_class: str = "option"
    presence: pl.Series | None = None

    def __post_init__(self) -> None:
        if self.presence is not None or _is_empty(self.df):
            return
        columns = [column for column in self.feature_cols if column in self.df.columns]
        if isinstance(self.df, pl.DataFrame):
            observed = self.df.select(
                pl.any_horizontal([pl.col(column).is_not_null() for column in columns])
                if columns
                else pl.lit(False)
            ).to_series()
        object.__setattr__(self, "presence", observed.cast(bool))


def filter_option_instrument_rows(
    chain: Frame,
    *,
    require_change_percent: bool = True,
) -> Frame:
    """Filter executable option rows using Polars expressions when available."""
    if chain is None:
        return pl.DataFrame()
    work = _to_polars(chain)
    required = {"bid", "ask"} | ({"change_percent"} if require_change_percent else set())
    if not required.issubset(work.columns):
        return work.head(0)
    valid = (
        pl.col("bid").cast(pl.Float64, strict=False).gt(0)
        & pl.col("ask").cast(pl.Float64, strict=False).gt(0)
        & pl.col("ask").cast(pl.Float64, strict=False).ge(pl.col("bid").cast(pl.Float64, strict=False))
    )
    if require_change_percent:
        change = pl.col("change_percent").cast(pl.Float64, strict=False)
        valid &= change.is_not_null() & change.ne(0)
    return work.filter(valid)


def build_option_contract_features(
    chain: Frame,
    *,
    underlying_price: float | None = None,
    target_dte: int | None = None,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    compute_model_greeks: bool = False,
) -> OptionFeatureSet:
    """Build contract, liquidity, Greek, and IV features natively in Polars.

    The input and output are Polars frames.
    """
    _ = risk_free_rate, dividend_yield, compute_model_greeks
    if chain is None or _is_empty(chain):
        return OptionFeatureSet(df=pl.DataFrame(), feature_cols=[], family_cols={})

    out = _to_polars(chain)
    out = out.rename({column: str(column).strip() for column in out.columns})
    date_exprs = []
    for column in ("snapshot_date", "expiration"):
        if column in out.columns:
            source = pl.col(column)
            if out.schema[column] == pl.String:
                source = source.str.to_datetime(strict=False)
            else:
                source = source.cast(pl.Datetime, strict=False)
            date_exprs.append(source.dt.replace_time_zone(None).dt.truncate("1d").alias(column))
    if date_exprs:
        out = out.with_columns(date_exprs)
    numeric = [column for column in ("strike", "bid", "ask", "mid", "volume", "open_interest", *GREEK_COLUMNS, *IV_COLUMNS, "underlying_price") if column in out.columns]
    if numeric:
        out = out.with_columns([pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in numeric])

    if underlying_price is None and "underlying_price" in out.columns:
        spot = out.filter(pl.col("underlying_price").is_finite() & pl.col("underlying_price").gt(0)).select(pl.col("underlying_price").median()).item()
        underlying_price = float(spot) if spot is not None else None
    if "mid" not in out.columns or out.select(pl.col("mid").is_null().all()).item():
        if {"bid", "ask"}.issubset(out.columns):
            out = out.with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"))

    families: dict[str, list[str]] = {}
    contract: list[str] = []
    if {"expiration", "snapshot_date"}.issubset(out.columns):
        out = out.with_columns((pl.col("expiration") - pl.col("snapshot_date")).dt.total_days().cast(pl.Int64).alias("dte"))
        contract.append("dte")
        if target_dte is not None:
            out = out.with_columns((pl.col("dte") - int(target_dte)).abs().alias("dte_gap"))
            contract.append("dte_gap")
    if underlying_price is not None and math.isfinite(underlying_price) and underlying_price > 0 and "strike" in out.columns:
        out = out.with_columns([
            pl.lit(float(underlying_price)).alias("underlying_spot_entry"),
            (pl.col("strike") / float(underlying_price) - 1.0).alias("moneyness"),
        ]).with_columns(pl.col("moneyness").abs().alias("abs_moneyness"))
        contract.extend(["moneyness", "abs_moneyness"])
    _add_family(families, "contract_static", contract)

    liquidity: list[str] = []
    if {"bid", "ask"}.issubset(out.columns):
        out = out.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
        liquidity.append("spread")
        if "mid" in out.columns:
            out = out.with_columns(pl.when(pl.col("mid").eq(0)).then(None).otherwise(pl.col("spread") / pl.col("mid")).alias("spread_pct"))
            liquidity.append("spread_pct")
    for column in ("volume", "open_interest"):
        if column in out.columns:
            liquidity.append(column)
    if {"volume", "open_interest"}.intersection(out.columns):
        volume = pl.col("volume").fill_null(0.0) if "volume" in out.columns else pl.lit(0.0)
        interest = pl.col("open_interest").fill_null(0.0) if "open_interest" in out.columns else pl.lit(0.0)
        out = out.with_columns((volume + interest / 100.0).alias("liquidity_score"))
        liquidity.append("liquidity_score")
    _add_family(families, "liquidity", liquidity)

    greeks: list[str] = []
    greek_exprs = []
    for column in GREEK_COLUMNS:
        if column in out.columns:
            greeks.extend([column, f"abs_{column}"])
            greek_exprs.append(pl.col(column).abs().alias(f"abs_{column}"))
    if "theta" in out.columns and "mid" in out.columns:
        greek_exprs.append(pl.when(pl.col("mid").eq(0)).then(None).otherwise(pl.col("theta") / pl.col("mid")).alias("theta_to_mid"))
        greeks.append("theta_to_mid")
    if "vega" in out.columns and "mid" in out.columns:
        greek_exprs.append(pl.when(pl.col("mid").eq(0)).then(None).otherwise(pl.col("vega") / pl.col("mid")).alias("vega_to_mid"))
        greeks.append("vega_to_mid")
    if greek_exprs:
        out = out.with_columns(greek_exprs)
    _add_family(families, "greeks", greeks)

    iv: list[str] = []
    iv_source = _first_present(out, IV_COLUMNS)
    if iv_source is not None:
        if iv_source != "iv":
            out = out.with_columns(pl.col(iv_source).alias("iv"))
        iv.append("iv")
        groups = [column for column in ("snapshot_date", "underlying_symbol", "option_type", "expiration") if column in out.columns]
        if groups:
            out = out.with_columns([
                pl.col("iv").mean().over(groups).alias("_iv_mean"),
                pl.col("iv").std().over(groups).alias("_iv_std"),
            ]).with_columns(
                ((pl.col("iv") - pl.col("_iv_mean")) / pl.when(pl.col("_iv_std").eq(0)).then(None).otherwise(pl.col("_iv_std"))).alias("iv_expiration_z")
            ).drop(["_iv_mean", "_iv_std"])
            iv.append("iv_expiration_z")
        if "dte" in out.columns:
            out = out.with_columns((pl.col("iv") * (pl.col("dte").clip(lower_bound=0) / 365.0).sqrt()).alias("iv_times_sqrt_dte"))
            iv.append("iv_times_sqrt_dte")
    _add_family(families, "iv_surface", iv)
    feature_cols = [column for columns in families.values() for column in columns]
    return OptionFeatureSet(df=out, feature_cols=feature_cols, family_cols=families)


def option_ranker_feature_columns(frame: Frame) -> list[str]:
    preferred = ["dte", "dte_gap", "moneyness", "abs_moneyness", "spread_pct", "volume", "open_interest", "liquidity_score", "delta", "abs_delta", "gamma", "abs_gamma", "theta", "abs_theta", "vega", "abs_vega", "rho", "abs_rho", "theta_to_mid", "vega_to_mid", "iv", "iv_expiration_z", "iv_times_sqrt_dte"]
    if isinstance(frame, pl.DataFrame):
        return [column for column in preferred if column in frame.columns and frame.select(pl.col(column).cast(pl.Float64, strict=False).is_not_null().any()).item()]
    return [column for column in preferred if column in frame.columns and frame.select(pl.col(column).cast(pl.Float64, strict=False).is_not_null().any()).item()]


def _to_polars(frame: Frame) -> pl.DataFrame:
    return frame


def _is_empty(frame: Frame | None) -> bool:
    return frame is None or frame.is_empty()


def _add_family(families: dict[str, list[str]], name: str, cols: list[str]) -> None:
    if cols:
        families[name] = list(dict.fromkeys(cols))


def _first_present(frame: pl.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns and frame.select(pl.col(column).is_not_null().any()).item():
            return column
    return None
