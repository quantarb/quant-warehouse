from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from quant_warehouse.ingest.credentials import configure_openbb_credentials, load_shared_env


@dataclass(frozen=True)
class ProviderProbe:
    provider: str
    status: str
    rows: int = 0
    message: str = ""


def _now_utc() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _safe_message(exc: BaseException | str, limit: int = 500) -> str:
    text = str(exc).replace(os.environ.get("FMP_API_KEY", ""), "REDACTED")
    return text[:limit]


def _openbb_yfinance_chain(symbol: str, *, date: str | None = None) -> pd.DataFrame:
    configure_openbb_credentials()
    from openbb import obb

    kwargs: dict[str, Any] = {"symbol": symbol, "provider": "yfinance"}
    if date:
        kwargs["date"] = date
    df = obb.derivatives.options.chains(**kwargs).to_df()
    return pd.DataFrame() if df is None else df.copy()


def _yfinance_chain(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frames: list[pd.DataFrame] = []
    for expiration in ticker.options:
        chain = ticker.option_chain(expiration)
        for option_type, side in (("call", chain.calls), ("put", chain.puts)):
            side = side.copy()
            side["underlying_symbol"] = symbol.upper()
            side["expiration"] = expiration
            side["option_type"] = option_type
            frames.append(side)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.rename(
        columns={
            "contractSymbol": "contract_symbol",
            "lastTradeDate": "last_trade_time",
            "lastPrice": "last_trade_price",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_volatility",
            "inTheMoney": "in_the_money",
        }
    )


def _probe_openbb_intrinio(symbol: str) -> ProviderProbe:
    try:
        configure_openbb_credentials()
        from openbb import obb

        df = obb.derivatives.options.chains(symbol=symbol, provider="intrinio").to_df()
        return ProviderProbe("openbb_intrinio", "ok", 0 if df is None else len(df))
    except Exception as exc:
        return ProviderProbe("openbb_intrinio", "blocked", message=_safe_message(exc))


def _probe_fmp_rest(symbol: str) -> ProviderProbe:
    load_shared_env()
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        return ProviderProbe("fmp_rest", "blocked", message="FMP_API_KEY is not configured")
    url = f"https://financialmodelingprep.com/api/v3/options-chain/{symbol.upper()}"
    try:
        response = requests.get(url, params={"apikey": api_key}, timeout=20)
        rows = 0
        try:
            payload = response.json()
            if isinstance(payload, list):
                rows = len(payload)
        except Exception:
            payload = response.text
        if response.ok and rows:
            return ProviderProbe("fmp_rest", "ok", rows)
        return ProviderProbe("fmp_rest", "blocked", rows, _safe_message(payload))
    except Exception as exc:
        return ProviderProbe("fmp_rest", "error", message=_safe_message(exc))


def _probe_alpha_vantage_rest(symbol: str, historical_date: str | None) -> ProviderProbe:
    load_shared_env()
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY") or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return ProviderProbe("alpha_vantage_rest", "blocked", message="ALPHA_VANTAGE_API_KEY is not configured")
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol.upper(),
        "apikey": api_key,
    }
    if historical_date:
        params["date"] = historical_date
    try:
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        payload = response.json()
        rows = len(payload.get("data", [])) if isinstance(payload, dict) and isinstance(payload.get("data"), list) else 0
        if response.ok and rows:
            return ProviderProbe("alpha_vantage_rest", "ok", rows)
        return ProviderProbe("alpha_vantage_rest", "blocked", rows, _safe_message(payload))
    except Exception as exc:
        return ProviderProbe("alpha_vantage_rest", "error", message=_safe_message(exc))


def _probe_thetadata(symbol: str, historical_date: str | None) -> ProviderProbe:
    load_shared_env()
    if not os.environ.get("THETADATA_API_KEY"):
        return ProviderProbe("thetadata", "blocked", message="THETADATA_API_KEY is not configured")
    try:
        from thetadata import ThetaClient
    except ImportError:
        return ProviderProbe("thetadata", "blocked", message="Install thetadata>=1.0.9")

    try:
        request_date = date.fromisoformat(historical_date or "2024-11-04")
        client = ThetaClient(dataframe_type="pandas", dotenv_path=".env")
        df = client.option_history_eod(
            symbol=symbol.upper(),
            expiration="*",
            start_date=request_date,
            end_date=request_date,
            max_dte=45,
            strike_range=2,
        )
        return ProviderProbe("thetadata", "ok", len(df))
    except Exception as exc:
        return ProviderProbe("thetadata", "error", message=_safe_message(exc))


def _provider_probes(symbol: str, historical_date: str | None) -> list[ProviderProbe]:
    probes: list[ProviderProbe] = []
    for provider, fetcher in (
        ("yfinance", lambda: _yfinance_chain(symbol)),
        ("openbb_yfinance", lambda: _openbb_yfinance_chain(symbol)),
        ("openbb_yfinance_dated", lambda: _openbb_yfinance_chain(symbol, date=historical_date)),
    ):
        if provider.endswith("_dated") and not historical_date:
            continue
        try:
            df = fetcher()
            probes.append(ProviderProbe(provider, "ok", len(df)))
        except Exception as exc:
            probes.append(ProviderProbe(provider, "error", message=_safe_message(exc)))
    probes.append(_probe_openbb_intrinio(symbol))
    probes.append(_probe_fmp_rest(symbol))
    probes.append(_probe_alpha_vantage_rest(symbol, historical_date))
    probes.append(_probe_thetadata(symbol, historical_date))
    for provider in ("polygon", "databento", "orats", "marketdata_app"):
        probes.append(ProviderProbe(provider, "not_tested", message="No credential configured in this environment"))
    return probes


def _normalize_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.date
    out["last_trade_time"] = pd.to_datetime(out.get("last_trade_time"), errors="coerce", utc=True)
    for col in ("strike", "bid", "ask", "volume", "open_interest", "implied_volatility", "last_trade_price"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "underlying_price" not in out:
        out["underlying_price"] = np.nan
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread"] = out["ask"] - out["bid"]
    out["spread_pct_mid"] = np.where(out["mid"] > 0, out["spread"] / out["mid"], np.nan)
    out["notional_open_interest"] = out["open_interest"].fillna(0) * out["strike"].fillna(0) * 100
    out["quoted"] = (out["bid"].fillna(0) > 0) | (out["ask"].fillna(0) > 0)
    out["has_activity"] = (out["volume"].fillna(0) > 0) | (out["open_interest"].fillna(0) > 0)
    return out


def _summaries(chain: pd.DataFrame, asof: str) -> dict[str, pd.DataFrame]:
    df = _normalize_for_eda(chain)
    symbol_summary = (
        df.groupby("underlying_symbol", dropna=False)
        .agg(
            contracts=("contract_symbol", "nunique"),
            expirations=("expiration", "nunique"),
            min_expiration=("expiration", "min"),
            max_expiration=("expiration", "max"),
            total_volume=("volume", "sum"),
            total_open_interest=("open_interest", "sum"),
            quoted_contracts=("quoted", "sum"),
            active_contracts=("has_activity", "sum"),
            median_spread_pct_mid=("spread_pct_mid", "median"),
            median_iv=("implied_volatility", "median"),
            p95_iv=("implied_volatility", lambda s: s.quantile(0.95)),
            notional_open_interest=("notional_open_interest", "sum"),
        )
        .reset_index()
    )
    by_expiration = (
        df.groupby(["underlying_symbol", "expiration", "option_type"], dropna=False)
        .agg(
            contracts=("contract_symbol", "nunique"),
            volume=("volume", "sum"),
            open_interest=("open_interest", "sum"),
            median_iv=("implied_volatility", "median"),
            median_spread_pct_mid=("spread_pct_mid", "median"),
        )
        .reset_index()
    )
    by_moneyness = df.copy()
    if by_moneyness["underlying_price"].notna().any():
        by_moneyness["moneyness"] = by_moneyness["strike"] / by_moneyness["underlying_price"]
    else:
        by_moneyness["moneyness"] = np.nan
    by_moneyness["moneyness_bucket"] = pd.cut(
        by_moneyness["moneyness"],
        bins=[-math.inf, 0.8, 0.95, 1.05, 1.2, math.inf],
        labels=["deep_itm_put_otm_call", "near_below_spot", "atm", "near_above_spot", "far_otm_call_itm_put"],
    )
    moneyness_summary = (
        by_moneyness.groupby(["underlying_symbol", "option_type", "moneyness_bucket"], observed=False, dropna=False)
        .agg(
            contracts=("contract_symbol", "nunique"),
            volume=("volume", "sum"),
            open_interest=("open_interest", "sum"),
            median_iv=("implied_volatility", "median"),
            median_spread_pct_mid=("spread_pct_mid", "median"),
        )
        .reset_index()
    )
    metadata = pd.DataFrame(
        [
            {
                "generated_at_utc": asof,
                "source": "OpenBB derivatives.options.chains(provider='yfinance')",
                "row_count": len(df),
                "symbol_count": df["underlying_symbol"].nunique(dropna=True),
            }
        ]
    )
    return {
        "symbol_summary": symbol_summary,
        "expiration_summary": by_expiration,
        "moneyness_summary": moneyness_summary,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe options data providers and run basic options-chain EDA.")
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA,TSLA")
    parser.add_argument("--historical-date", default="2026-06-18")
    parser.add_argument("--out-dir", default="research/options_eda_output")
    args = parser.parse_args()

    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asof = _now_utc()

    probes = _provider_probes(symbols[0], args.historical_date)
    (out_dir / "provider_probes.json").write_text(
        json.dumps([asdict(probe) for probe in probes], indent=2) + "\n",
        encoding="utf-8",
    )

    frames: list[pd.DataFrame] = []
    fetch_errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            frame = _openbb_yfinance_chain(symbol)
            frame["fetched_at_utc"] = asof
            frames.append(frame)
        except Exception as exc:
            fetch_errors[symbol] = _safe_message(exc)

    if not frames:
        raise RuntimeError(f"No option chains fetched. Errors: {fetch_errors}")

    chain = pd.concat(frames, ignore_index=True)
    normalized = _normalize_for_eda(chain)
    normalized.to_csv(out_dir / "chains_openbb_yfinance.csv", index=False)
    for name, summary in _summaries(chain, asof).items():
        summary.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "fetch_errors.json").write_text(json.dumps(fetch_errors, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(chain), "symbols": symbols, "errors": fetch_errors}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
