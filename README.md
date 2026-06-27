# Quant Warehouse

[![Repository](https://img.shields.io/badge/github-quantarb%2Fquant--warehouse-blue)](https://github.com/quantarb/quant-warehouse)

Multi-vendor **point-in-time** market data and materialized features for ML and backtesting.

Engine-agnostic: consume panels from VectorBT, Zipline, or your own stack. OpenBB is the vendor adapter; ArcticDB is the canonical store for historical time-series data.

## Layers

```text
staging/        optional raw vendor snapshots (audit / replay)
prices/         dense daily OHLCV per symbol
options/        daily ThetaData EOD option chains per underlying
fundamentals/   sparse per symbol__provider (only columns that vendor returns)
features/       dense daily PIT panels keyed by recipe_hash (ML input)
catalog/        SQLite metadata — gap-fill state, columns_present, date ranges
```

## Conda setup

```bash
cd quant-warehouse
conda env create -f environment.yml
conda activate quant-warehouse
```

Or update an existing env:

```bash
conda env update -f environment.yml --prune
```

## Configure

```bash
cp .env.example .env
# set vendor keys such as FMP_API_KEY and optionally THETADATA_API_KEY
export QW_HOME=~/.quant-warehouse
```

The current environment installs `quant-warehouse` with the `[openbb,dev]` extra, which pulls OpenBB packages from the `quantarb/OpenBB` `develop` branch. The warehouse itself reads `QW_HOME`, `QW_ARCTIC_URI`, and `QW_CATALOG_PATH`; by default it stores ArcticDB data under `~/.quant-warehouse/arctic` and metadata in `~/.quant-warehouse/catalog.sqlite`.

## CLI

```bash
quant-warehouse refresh AAPL --sections prices,income --providers fmp,yfinance,sec
quant-warehouse refresh-prices AAPL --providers fmp,yfinance,tiingo --start-date 2020-01-01
quant-warehouse refresh-fundamentals AAPL --sections income,balance,cash --providers fmp,sec
quant-warehouse status AAPL
```

## Python API

```python
from quant_warehouse import Warehouse

wh = Warehouse()
wh.refresh("AAPL", sections=["prices", "income"], providers=["fmp", "yfinance", "sec"])
prices = wh.read_prices("AAPL", provider="fmp", start="2020-01-01")
income = wh.read_fundamentals("AAPL", section="income", provider="fmp")
```

## Design rules

1. **ArcticDB is canonical for historical series** — prices, ETF prices, macro series, fundamentals, event pairs, features, calendars, and ThetaData option chains are stored in ArcticDB libraries.
2. **SQLite is metadata only** — the catalog tracks gap-fill state, columns present, date ranges, and profile metadata.
3. **Parquet/CSV are export artifacts only** — reports and derived ML datasets may be written to files, but historical series loaders must read/write ArcticDB.
4. **Silver fundamentals stay sparse** — `period_ending` index, one Arctic symbol per `TICKER__provider`.
5. **Gold features are daily** — same row count as prices after PIT join + derived columns.
6. **Only store columns a vendor returns** — no empty cross-vendor placeholders.
7. **Gap-fill** — merge on date/period per symbol; catalog tracks ranges and `last_fetched_at`.

## License

MIT
