from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yfinance as yf

from quant_warehouse.target_engineering import LabelBuildSpec, build_trade_results
from quant_warehouse.target_engineering.option_labels import build_option_labels
from quant_warehouse.target_engineering.thetadata_loader import (
    load_thetadata_option_snapshots,
    normalize_thetadata_option_chain,
)


def _load_price_frame(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No price history returned for {symbol}")
    frame = df.reset_index().rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    return frame.set_index("date").sort_index()


def main() -> None:
    api_key = os.environ.get("THETADATA_API_KEY")
    if not api_key:
        raise SystemExit("THETADATA_API_KEY is required")

    symbol = "AAPL"
    price_frame = _load_price_frame(symbol, "2024-01-01", "2025-01-31")
    spec = LabelBuildSpec(
        k_params={"YE": [1]},
        min_profit_pct=0.01,
        start_date="2024-01-01",
        end_date="2025-01-31",
        buy_execution="high",
        sell_execution="low",
        short_execution="low",
        cover_execution="high",
    )

    trade_result = build_trade_results([symbol], spec=spec, price_frames={symbol: price_frame})
    trade = next(t for t in trade_result.completed_trades if t["side"] == "short")
    entry = pd.Timestamp(trade["entry_date"])
    exit = pd.Timestamp(trade["exit_date"])

    snapshots = load_thetadata_option_snapshots(symbol, [entry, exit], api_key=api_key, max_dte=45, strike_range=10)
    normalized = {ts: normalize_thetadata_option_chain(df) for ts, df in snapshots.items()}
    labels = build_option_labels([trade], normalized)
    label_df = pd.DataFrame(labels.option_rows)

    out_dir = Path("research/options_eda_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / f"thetadata_{symbol.lower()}_{entry.date()}_{exit.date()}_option_labels.csv"
    summary_path = out_dir / f"thetadata_{symbol.lower()}_{entry.date()}_{exit.date()}_option_summary.json"

    label_df.to_csv(label_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(labels.statistics, handle, indent=2, default=str)

    print(f"trade={trade}")
    print(f"labels={len(label_df)} saved={label_path}")
    print(f"summary saved={summary_path}")
    if not label_df.empty:
        cols = [
            col
            for col in [
                "contract_symbol",
                "option_type",
                "expiration",
                "strike",
                "entry_quote",
                "exit_quote",
                "option_return_pct",
                "rank_y",
            ]
            if col in label_df.columns
        ]
        print(label_df.sort_values(["rank_y", "option_return_pct"], ascending=[False, False])[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
