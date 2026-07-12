from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_warehouse.ingest.oracle_news import (
    refresh_fmp_news_for_oracle_trades,
    select_oracle_trades,
)
from quant_warehouse.warehouse.api import Warehouse


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh FMP news only at oracle trade boundaries")
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--symbols-from-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--no-store", action="store_true", help="Do not import output into Arctic")
    args = parser.parse_args()

    trades = pd.read_parquet(args.trades)
    labels = pd.read_parquet(args.symbols_from_labels, columns=["symbol"])
    selected = select_oracle_trades(
        trades,
        symbols=labels["symbol"].unique(),
        k_values=(1, 2, 3),
        frequency="YE",
        start_date=args.start_date,
        end_date=args.end_date,
    )
    result = refresh_fmp_news_for_oracle_trades(selected, output_path=args.output)
    stored = {}
    if not args.no_store:
        stored = Warehouse().news.import_parquet(str(result.output_path), provider="fmp")
    print(
        f"trades={len(selected)} boundaries={result.boundary_rows} "
        f"requests={result.requests} news_rows={result.news_rows} "
        f"stored_symbols={len(stored)} stored_rows={sum(stored.values())} "
        f"output={result.output_path}"
    )


if __name__ == "__main__":
    main()
