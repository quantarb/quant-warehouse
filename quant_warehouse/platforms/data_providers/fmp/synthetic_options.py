from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from quant_warehouse import Warehouse
from quant_warehouse.platforms.data_providers.options import OptionSettlement


FMP_SYNTHETIC_OPTION_SOURCE = "black_scholes_synthetic"
FMP_SYNTHETIC_UNDERLYING_PRICE_BASIS = "adjusted"
FMP_SYNTHETIC_OPTION_COLUMNS: tuple[str, ...] = (
    "snapshot_date",
    "underlying_symbol",
    "contract_symbol",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "underlying_price",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "iv",
    "volume",
    "open_interest",
    "data_interval",
    "pricing_model",
    "option_source",
    "underlying_provider",
    "underlying_price_basis",
)


@dataclass(frozen=True)
class FmpSyntheticOptionSpec:
    tenor_days: tuple[int, ...] = (30, 60, 90, 120)
    strike_multipliers: tuple[float, ...] = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
    realized_vol_window: int = 21
    vol_floor: float | None = 0.15
    vol_cap: float | None = 0.80
    annualization: float = 252.0
    day_count: float = 365.0
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    iv_multiplier: float = 1.0
    premium_floor: float = 0.25
    spread_bps: float = 50.0
    min_spread: float = 0.01


def read_fmp_synthetic_option_chain(
    symbol: str,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    columns: Sequence[str] | None = None,
    spec: FmpSyntheticOptionSpec | None = None,
    prices: pd.DataFrame | None = None,
    warehouse: Warehouse | None = None,
) -> pd.DataFrame:
    """Build an adjusted-price Black-Scholes option chain from FMP prices.

    The generated strikes and intrinsic settlement live on the same adjusted
    price basis as FMP feature/target engineering. This is intentionally not a
    ThetaData replacement; it is a separate synthetic provider for experiments
    that need option-like prices on the FMP adjusted scale.
    """

    cfg = spec or FmpSyntheticOptionSpec()
    price_frame = _load_fmp_price_frame(
        symbol,
        start_date=start_date,
        end_date=end_date,
        prices=prices,
        warehouse=warehouse,
        vol_window=cfg.realized_vol_window,
    )
    chain = build_fmp_synthetic_option_chain(
        price_frame,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        spec=cfg,
    )
    if columns is not None:
        requested = [str(column) for column in columns]
        for column in requested:
            if column not in chain.columns:
                chain[column] = pd.NA
        chain = chain.loc[:, requested]
    return chain


def build_fmp_synthetic_option_chain(
    prices: pd.DataFrame,
    *,
    symbol: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    spec: FmpSyntheticOptionSpec | None = None,
) -> pd.DataFrame:
    cfg = spec or FmpSyntheticOptionSpec()
    price_frame = _normalize_price_frame(prices)
    if price_frame.empty:
        return pd.DataFrame(columns=list(FMP_SYNTHETIC_OPTION_COLUMNS))
    start = None if start_date is None else pd.Timestamp(start_date).normalize()
    end = None if end_date is None else pd.Timestamp(end_date).normalize()
    realized_vol = build_realized_vol_series(
        price_frame["close"],
        window=cfg.realized_vol_window,
        vol_floor=cfg.vol_floor,
        vol_cap=cfg.vol_cap,
        annualization=cfg.annualization,
    )
    rows: list[dict[str, object]] = []
    for snapshot_date, spot in price_frame["close"].items():
        snapshot = pd.Timestamp(snapshot_date).normalize()
        if start is not None and snapshot < start:
            continue
        if end is not None and snapshot > end:
            continue
        spot_value = _finite_float(spot)
        if spot_value is None or spot_value <= 0.0:
            continue
        volatility = _finite_float(realized_vol.get(snapshot_date))
        if volatility is None or volatility <= 0.0:
            continue
        iv = max(volatility * float(cfg.iv_multiplier), 1e-8)
        for tenor in cfg.tenor_days:
            dte = max(int(tenor), 1)
            expiration = snapshot + pd.Timedelta(days=dte)
            years = max(float(dte) / float(cfg.day_count), 1.0 / float(cfg.day_count))
            for multiplier in cfg.strike_multipliers:
                strike = float(spot_value) * float(multiplier)
                for option_type in ("call", "put"):
                    mid = black_scholes_price(
                        spot=float(spot_value),
                        strike=strike,
                        years=years,
                        option_type=option_type,
                        volatility=iv,
                        risk_free_rate=float(cfg.risk_free_rate),
                        dividend_yield=float(cfg.dividend_yield),
                    )
                    intrinsic = option_intrinsic_value(
                        option_type=option_type,
                        strike=strike,
                        underlying_price=float(spot_value),
                    )
                    mid = max(float(mid), float(intrinsic), float(cfg.premium_floor))
                    greeks = black_scholes_greeks(
                        spot=float(spot_value),
                        strike=strike,
                        years=years,
                        option_type=option_type,
                        volatility=iv,
                        risk_free_rate=float(cfg.risk_free_rate),
                        dividend_yield=float(cfg.dividend_yield),
                    )
                    bid, ask = synthetic_bid_ask(mid, spread_bps=cfg.spread_bps, min_spread=cfg.min_spread)
                    rows.append(
                        {
                            "snapshot_date": snapshot,
                            "underlying_symbol": str(symbol).upper(),
                            "contract_symbol": fmp_synthetic_contract_symbol(
                                symbol=symbol,
                                option_type=option_type,
                                expiration=expiration,
                                strike=strike,
                            ),
                            "expiration": expiration.normalize(),
                            "strike": float(strike),
                            "option_type": option_type,
                            "bid": bid,
                            "ask": ask,
                            "mid": float(mid),
                            "underlying_price": float(spot_value),
                            "delta": greeks["delta"],
                            "gamma": greeks["gamma"],
                            "theta": greeks["theta"],
                            "vega": greeks["vega"],
                            "rho": greeks["rho"],
                            "iv": float(iv),
                            "volume": np.nan,
                            "open_interest": np.nan,
                            "dte": int(dte),
                            "moneyness": float(strike) / float(spot_value) - 1.0,
                            "abs_moneyness": abs(float(strike) / float(spot_value) - 1.0),
                            "spread": float(ask - bid),
                            "spread_pct": float((ask - bid) / mid) if mid > 0 else np.nan,
                            "data_interval": "eod",
                            "pricing_model": "black_scholes",
                            "option_source": FMP_SYNTHETIC_OPTION_SOURCE,
                            "underlying_provider": "fmp",
                            "underlying_price_basis": FMP_SYNTHETIC_UNDERLYING_PRICE_BASIS,
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=list(FMP_SYNTHETIC_OPTION_COLUMNS))
    out = pd.DataFrame(rows)
    return out.sort_values(["snapshot_date", "contract_symbol"]).reset_index(drop=True)


def settle_fmp_synthetic_option_exit(
    *,
    symbol: str,
    contract_symbol: str,
    option_type: str,
    strike: float,
    expiration: pd.Timestamp | str,
    equity_exit_date: pd.Timestamp | str,
    entry_date: pd.Timestamp | str | None = None,
    spec: FmpSyntheticOptionSpec | None = None,
    prices: pd.DataFrame | None = None,
    warehouse: Warehouse | None = None,
    **_: object,
) -> OptionSettlement | None:
    """Settle an adjusted-price FMP synthetic option on the FMP price basis."""

    _ = contract_symbol, entry_date
    cfg = spec or FmpSyntheticOptionSpec()
    expiration_ts = pd.Timestamp(expiration).normalize()
    equity_exit_ts = pd.Timestamp(equity_exit_date).normalize()
    if pd.isna(expiration_ts) or pd.isna(equity_exit_ts):
        return None
    target_exit = min(expiration_ts, equity_exit_ts)
    quote = price_fmp_synthetic_contract(
        symbol=symbol,
        snapshot_date=target_exit,
        option_type=option_type,
        strike=float(strike),
        expiration=expiration_ts,
        spec=cfg,
        prices=prices,
        warehouse=warehouse,
    )
    if quote is None:
        return None
    source = "expiration_intrinsic" if target_exit >= expiration_ts else "contract_quote"
    return OptionSettlement(
        snapshot_date=target_exit,
        bid=float(quote["bid"]),
        ask=float(quote["ask"]),
        mid=float(quote["mid"]),
        price_source=source,
    )


def price_fmp_synthetic_contract(
    *,
    symbol: str,
    snapshot_date: str | pd.Timestamp,
    option_type: str,
    strike: float,
    expiration: str | pd.Timestamp,
    spec: FmpSyntheticOptionSpec | None = None,
    prices: pd.DataFrame | None = None,
    warehouse: Warehouse | None = None,
) -> dict[str, float] | None:
    cfg = spec or FmpSyntheticOptionSpec()
    snapshot = pd.Timestamp(snapshot_date).normalize()
    expiration_ts = pd.Timestamp(expiration).normalize()
    price_frame = _load_fmp_price_frame(
        symbol,
        start_date=snapshot,
        end_date=snapshot,
        prices=prices,
        warehouse=warehouse,
        vol_window=cfg.realized_vol_window,
    )
    normalized = _normalize_price_frame(price_frame)
    if snapshot not in normalized.index:
        return None
    spot = _finite_float(normalized.loc[snapshot, "close"])
    strike_value = _finite_float(strike)
    if spot is None or strike_value is None or spot <= 0.0 or strike_value <= 0.0:
        return None
    dte = max(int((expiration_ts - snapshot).days), 0)
    intrinsic = option_intrinsic_value(option_type=option_type, strike=strike_value, underlying_price=spot)
    if dte <= 0:
        bid, ask = synthetic_bid_ask(float(intrinsic), spread_bps=0.0, min_spread=0.0)
        return {"bid": bid, "ask": ask, "mid": float(intrinsic), "underlying_price": float(spot), "iv": np.nan}
    realized_vol = build_realized_vol_series(
        normalized["close"],
        window=cfg.realized_vol_window,
        vol_floor=cfg.vol_floor,
        vol_cap=cfg.vol_cap,
        annualization=cfg.annualization,
    )
    vol = _finite_float(realized_vol.loc[snapshot])
    if vol is None or vol <= 0.0:
        return None
    iv = max(vol * float(cfg.iv_multiplier), 1e-8)
    years = max(float(dte) / float(cfg.day_count), 1.0 / float(cfg.day_count))
    mid = black_scholes_price(
        spot=float(spot),
        strike=float(strike_value),
        years=years,
        option_type=option_type,
        volatility=iv,
        risk_free_rate=float(cfg.risk_free_rate),
        dividend_yield=float(cfg.dividend_yield),
    )
    mid = max(float(mid), float(intrinsic), float(cfg.premium_floor))
    bid, ask = synthetic_bid_ask(mid, spread_bps=cfg.spread_bps, min_spread=cfg.min_spread)
    return {"bid": bid, "ask": ask, "mid": float(mid), "underlying_price": float(spot), "iv": float(iv)}


def build_realized_vol_series(
    close: pd.Series,
    *,
    window: int = 21,
    vol_floor: float | None = 0.15,
    vol_cap: float | None = 0.80,
    annualization: float = 252.0,
) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").astype(float)
    rolling_window = max(int(window), 1)
    min_periods = 2 if rolling_window > 1 else 1
    log_returns = np.log(values / values.shift(1)).replace([np.inf, -np.inf], np.nan)
    realized_vol = log_returns.rolling(rolling_window, min_periods=min_periods).std()
    realized_vol = realized_vol * math.sqrt(float(annualization))
    if vol_floor is not None:
        realized_vol = realized_vol.clip(lower=float(vol_floor)).fillna(float(vol_floor))
    if vol_cap is not None:
        realized_vol = realized_vol.clip(upper=float(vol_cap))
    return realized_vol


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    years: float,
    option_type: str,
    volatility: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    if years <= 0.0:
        return option_intrinsic_value(option_type=option_type, strike=strike, underlying_price=spot)
    sqrt_t = math.sqrt(float(years))
    d1 = (
        math.log(float(spot) / float(strike))
        + (float(risk_free_rate) - float(dividend_yield) + 0.5 * float(volatility) ** 2) * float(years)
    ) / (float(volatility) * sqrt_t)
    d2 = d1 - float(volatility) * sqrt_t
    discount_q = math.exp(-float(dividend_yield) * float(years))
    discount_r = math.exp(-float(risk_free_rate) * float(years))
    side = _option_side(option_type)
    if side == "put":
        return float(float(strike) * discount_r * norm.cdf(-d2) - float(spot) * discount_q * norm.cdf(-d1))
    return float(float(spot) * discount_q * norm.cdf(d1) - float(strike) * discount_r * norm.cdf(d2))


def black_scholes_greeks(
    *,
    spot: float,
    strike: float,
    years: float,
    option_type: str,
    volatility: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    if years <= 0.0 or volatility <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return {name: np.nan for name in ("delta", "gamma", "theta", "vega", "rho")}
    sqrt_t = math.sqrt(float(years))
    d1 = (
        math.log(float(spot) / float(strike))
        + (float(risk_free_rate) - float(dividend_yield) + 0.5 * float(volatility) ** 2) * float(years)
    ) / (float(volatility) * sqrt_t)
    d2 = d1 - float(volatility) * sqrt_t
    discount_q = math.exp(-float(dividend_yield) * float(years))
    discount_r = math.exp(-float(risk_free_rate) * float(years))
    pdf_d1 = norm.pdf(d1)
    if _option_side(option_type) == "put":
        delta = discount_q * (norm.cdf(d1) - 1.0)
        theta = (
            -(spot * discount_q * pdf_d1 * volatility) / (2.0 * sqrt_t)
            + risk_free_rate * strike * discount_r * norm.cdf(-d2)
            - dividend_yield * spot * discount_q * norm.cdf(-d1)
        ) / 365.0
        rho = -strike * years * discount_r * norm.cdf(-d2)
    else:
        delta = discount_q * norm.cdf(d1)
        theta = (
            -(spot * discount_q * pdf_d1 * volatility) / (2.0 * sqrt_t)
            - risk_free_rate * strike * discount_r * norm.cdf(d2)
            + dividend_yield * spot * discount_q * norm.cdf(d1)
        ) / 365.0
        rho = strike * years * discount_r * norm.cdf(d2)
    gamma = discount_q * pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * discount_q * pdf_d1 * sqrt_t
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


def synthetic_bid_ask(mid: float, *, spread_bps: float, min_spread: float) -> tuple[float, float]:
    mid_value = max(float(mid), 0.0)
    spread = max(mid_value * float(spread_bps) / 10_000.0, float(min_spread))
    bid = max(mid_value - spread / 2.0, 0.0)
    ask = mid_value + spread / 2.0
    return float(bid), float(ask)


def option_intrinsic_value(*, option_type: str, strike: float, underlying_price: float) -> float:
    if _option_side(option_type) == "put":
        return float(max(float(strike) - float(underlying_price), 0.0))
    return float(max(float(underlying_price) - float(strike), 0.0))


def fmp_synthetic_contract_symbol(
    *,
    symbol: str,
    option_type: str,
    expiration: pd.Timestamp,
    strike: float,
) -> str:
    side = "C" if _option_side(option_type) == "call" else "P"
    strike_token = int(round(float(strike) * 10_000))
    return f"FMPBS_{str(symbol).upper()}_{side}_{pd.Timestamp(expiration).strftime('%Y%m%d')}_{strike_token}"


def _load_fmp_price_frame(
    symbol: str,
    *,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
    prices: pd.DataFrame | None,
    warehouse: Warehouse | None,
    vol_window: int,
) -> pd.DataFrame:
    if prices is not None:
        return prices.copy()
    end = None if end_date is None else pd.Timestamp(end_date).normalize()
    start = None if start_date is None else pd.Timestamp(start_date).normalize()
    fetch_start = start
    if start is not None:
        fetch_start = start - pd.Timedelta(days=max(int(vol_window) * 3, 10))
    store = warehouse or Warehouse()
    return store.read_prices(str(symbol).upper(), provider="fmp", start=fetch_start, end=end)


def _normalize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["close"])
    out = prices.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    if "close" not in out.columns:
        raise ValueError("FMP synthetic options require an adjusted close column named 'close'")
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce")).normalize()
    out = out.loc[out.index.notna()].sort_index()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["close"])


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _option_side(value: object) -> Literal["call", "put"]:
    side = str(value).strip().lower()
    return "put" if side.startswith("p") else "call"
