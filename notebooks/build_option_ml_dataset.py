from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_warehouse import Warehouse
from quant_warehouse.target_engineering import LabelBuildSpec, build_trade_results
from quant_warehouse.platforms.data_providers.thetadata.target_engineering.option_dataset import (
    OptionMlDatasetSpec,
    build_option_ml_dataset,
    save_option_ml_dataset,
)
from quant_warehouse.platforms.data_providers.thetadata.options import ThetaDataDownloadSpec


def _load_price_frame(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = Warehouse().read_prices(symbol, provider="fmp", start=start, end=end)
    if df.empty:
        raise RuntimeError(f"No warehouse price history returned for {symbol}")
    frame = df.reset_index().rename(columns={"index": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    return frame.set_index("date").sort_index()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ThetaData option chains for oracle trades and build ML label rows."
    )
    parser.add_argument("--symbols", default="AAPL", help="Comma-separated symbols")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-31")
    parser.add_argument("--side", default=None, help="Optional trade side filter: long or short")
    parser.add_argument("--max-dte", type=int, default=45)
    parser.add_argument("--strike-range", type=int, default=10)
    parser.add_argument("--output", default="notebooks/outputs/option_ml_dataset.parquet")
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    price_frames = {symbol: _load_price_frame(symbol, args.start, args.end) for symbol in symbols}

    spec = LabelBuildSpec(
        k_params={"YE": [1]},
        min_profit_pct=0.01,
        start_date=args.start,
        end_date=args.end,
        buy_execution="high",
        sell_execution="low",
        short_execution="low",
        cover_execution="high",
    )
    trade_result = build_trade_results(symbols, spec=spec, price_frames=price_frames)
    trades = trade_result.completed_trades
    if args.side:
        trades = [trade for trade in trades if str(trade.get("side") or "").lower() == str(args.side).lower()]

    dataset_spec = OptionMlDatasetSpec(
        thetadata=ThetaDataDownloadSpec(max_dte=args.max_dte, strike_range=args.strike_range),
        download_missing=True,
    )
    result = build_option_ml_dataset(trades, dataset_spec=dataset_spec)
    output_path = save_option_ml_dataset(result, args.output, file_format=args.format)

    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result.statistics, indent=2), encoding="utf-8")
    print(f"rows={len(result.rows)} saved={output_path}")
    print(f"summary saved={summary_path}")


if __name__ == "__main__":
    main()
