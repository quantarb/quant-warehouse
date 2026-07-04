from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from quant_warehouse.platforms.data_providers.options import (
    OptionExitPriceSource,
    OptionQuote,
    OptionSettlement,
)


QuoteLoader = Callable[
    [str, pd.Timestamp, str],
    OptionQuote | tuple[float, float, float] | None,
]
UnderlyingCloseLoader = Callable[[str, pd.Timestamp], float | None]


def settle_option_exit(
    *,
    symbol: str,
    contract_symbol: str,
    option_type: str,
    strike: float,
    expiration: pd.Timestamp | str,
    equity_exit_date: pd.Timestamp | str,
    quote_loader: QuoteLoader,
    underlying_close_loader: UnderlyingCloseLoader,
    entry_date: pd.Timestamp | str | None = None,
    exit_lookback_days: int = 7,
    intrinsic_scale_min_ratio: float = 0.2,
    intrinsic_scale_max_ratio: float = 5.0,
) -> OptionSettlement | None:
    """Resolve an option exit quote using real quotes before intrinsic fallback.

    The option chain and the underlying price feed can live on different split
    adjustment scales. To avoid mixing raw option strikes with adjusted equity
    closes, intrinsic settlement is only allowed when strike and spot are on a
    plausible shared scale.
    """

    expiration_ts = pd.Timestamp(expiration).normalize()
    equity_exit_ts = pd.Timestamp(equity_exit_date).normalize()
    if pd.isna(expiration_ts) or pd.isna(equity_exit_ts):
        return None
    target_exit = min(equity_exit_ts, expiration_ts)
    entry_ts = None if entry_date is None else pd.Timestamp(entry_date).normalize()
    symbol = str(symbol).upper()
    contract_symbol = str(contract_symbol)

    for quote_date in iter_option_exit_lookup_dates(target_exit, exit_lookback_days):
        if entry_ts is not None and target_exit > entry_ts and quote_date <= entry_ts:
            continue
        quote = _coerce_quote(quote_loader(symbol, quote_date, contract_symbol), quote_date)
        if quote is None:
            continue
        source: OptionExitPriceSource = (
            "contract_quote" if quote.snapshot_date == target_exit else "last_contract_quote"
        )
        return OptionSettlement(
            snapshot_date=quote.snapshot_date,
            bid=quote.bid,
            ask=quote.ask,
            mid=quote.mid,
            price_source=source,
        )

    if target_exit < expiration_ts:
        return None

    spot = underlying_close_loader(symbol, target_exit)
    intrinsic = option_intrinsic_value(
        option_type=option_type,
        strike=strike,
        underlying_price=spot,
        min_strike_spot_ratio=intrinsic_scale_min_ratio,
        max_strike_spot_ratio=intrinsic_scale_max_ratio,
    )
    if intrinsic is None:
        return None
    return OptionSettlement(
        snapshot_date=target_exit,
        bid=intrinsic,
        ask=intrinsic,
        mid=intrinsic,
        price_source="expiration_intrinsic",
    )


def iter_option_exit_lookup_dates(
    target_exit: pd.Timestamp | str,
    exit_lookback_days: int,
) -> tuple[pd.Timestamp, ...]:
    target = pd.Timestamp(target_exit).normalize()
    if pd.isna(target):
        return tuple()
    lookback = max(0, int(exit_lookback_days))
    dates: list[pd.Timestamp] = [target]
    current = target
    for _ in range(lookback):
        current = pd.Timestamp(current - BDay(1)).normalize()
        if current not in dates:
            dates.append(current)
    return tuple(dates)


def option_intrinsic_value(
    *,
    option_type: str,
    strike: float,
    underlying_price: float | None,
    min_strike_spot_ratio: float = 0.2,
    max_strike_spot_ratio: float = 5.0,
) -> float | None:
    strike_value = _finite_float(strike)
    spot_value = _finite_float(underlying_price)
    if strike_value is None or spot_value is None or strike_value <= 0.0 or spot_value <= 0.0:
        return None
    if not strike_spot_scale_matches(
        strike_value,
        spot_value,
        min_ratio=min_strike_spot_ratio,
        max_ratio=max_strike_spot_ratio,
    ):
        return None
    opt_type = str(option_type).strip().lower()
    if opt_type.startswith("p"):
        return float(max(strike_value - spot_value, 0.0))
    if opt_type.startswith("c"):
        return float(max(spot_value - strike_value, 0.0))
    return None


def strike_spot_scale_matches(
    strike: float,
    underlying_price: float,
    *,
    min_ratio: float = 0.2,
    max_ratio: float = 5.0,
) -> bool:
    strike_value = _finite_float(strike)
    spot_value = _finite_float(underlying_price)
    if strike_value is None or spot_value is None or strike_value <= 0.0 or spot_value <= 0.0:
        return False
    ratio = strike_value / spot_value
    return float(min_ratio) <= ratio <= float(max_ratio)


def lookup_contract_quote(chain: pd.DataFrame, contract_symbol: str) -> tuple[float, float, float] | None:
    if chain is None or chain.empty or "contract_symbol" not in chain.columns:
        return None
    work = chain.loc[chain["contract_symbol"].astype(str).eq(str(contract_symbol))].copy()
    if work.empty:
        return None
    row = work.iloc[-1]
    bid = _finite_float(row.get("bid"))
    ask = _finite_float(row.get("ask"))
    mid = _finite_float(row.get("mid"))
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    if bid is None:
        bid = mid
    if ask is None:
        ask = mid
    if bid is None or ask is None or mid is None:
        return None
    return float(bid), float(ask), float(mid)


def _coerce_quote(
    quote: OptionQuote | tuple[float, float, float] | None,
    snapshot_date: pd.Timestamp,
) -> OptionQuote | None:
    if quote is None:
        return None
    if isinstance(quote, OptionQuote):
        bid = _finite_float(quote.bid)
        ask = _finite_float(quote.ask)
        mid = _finite_float(quote.mid)
        quote_date = pd.Timestamp(quote.snapshot_date).normalize()
    else:
        if len(quote) != 3:
            return None
        bid = _finite_float(quote[0])
        ask = _finite_float(quote[1])
        mid = _finite_float(quote[2])
        quote_date = pd.Timestamp(snapshot_date).normalize()
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    if bid is None:
        bid = mid
    if ask is None:
        ask = mid
    if bid is None or ask is None or mid is None:
        return None
    return OptionQuote(snapshot_date=quote_date, bid=float(bid), ask=float(ask), mid=float(mid))


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out
