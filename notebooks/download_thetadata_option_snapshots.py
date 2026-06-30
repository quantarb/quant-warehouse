from __future__ import annotations

import argparse
import json

from quant_warehouse.platforms.data_providers.thetadata.options import (
    ThetaDataDownloadSpec,
    download_option_snapshots_for_range,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache daily ThetaData EOD option chains (bid/ask required)."
    )
    parser.add_argument("--symbol", required=True, help="Underlying symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--max-dte", type=int, default=60)
    parser.add_argument("--strike-range", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = ThetaDataDownloadSpec(max_dte=args.max_dte, strike_range=args.strike_range)
    manifest = download_option_snapshots_for_range(
        args.symbol,
        args.start,
        args.end,
        spec=spec,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
