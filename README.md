# Quant Warehouse

[![Repository](https://img.shields.io/badge/github-quantarb%2Fquant--warehouse-blue)](https://github.com/quantarb/quant-warehouse)

Gets, normalizes, refreshes, and maintains **point-in-time** market and event data from multiple vendors and sources, then materializes provider-specific feature engineering and target engineering datasets for ML research and backtesting systems.

Framework-agnostic data provider: downstream systems such as `quant-orchestrator`, Zipline, VectorBT, or custom research code consume prepared warehouse datasets. This repo does not run model training or backtests. OpenBB is the vendor/source adapter layer; ArcticDB is the canonical store for historical time-series data.

## Storage Layers

```text
staging/        optional raw vendor snapshots (audit / replay)
prices/         dense daily OHLCV per symbol
options/        daily ThetaData EOD option chains per underlying
fundamentals/   sparse per symbol__provider (only columns that vendor returns)
features/       provider-owned dense daily PIT panels keyed by recipe_hash
targets/        provider-owned point-in-time labels
catalog/        SQLite metadata — gap-fill state, columns_present, date ranges
```

## Code Layout

Provider-specific behavior lives under `quant_warehouse/platforms/data_providers/{provider}`. There are intentionally no top-level `quant_warehouse.feature_engineering` or `quant_warehouse.target_engineering` packages.

```text
quant_warehouse/platforms/data_providers/fmp/
    feature_engineering/
        fundamentals.py
        fundamental_features.py
        technical.py
        ta_classic_technical.py
        time_features.py
    target_engineering/
        event_pairs/
        labels.py
        operations.py
        returns/
        specs.py
        strategy_solver.py

quant_warehouse/platforms/data_providers/thetadata/
    options.py
    target_engineering/
        option_dataset.py
        option_labels.py
        options/

quant_warehouse/platforms/data_providers/yfinance/
    storage.py
    migration.py
```

This is deliberate. The current real target and feature semantics are mostly FMP equity/fundamental semantics plus ThetaData option semantics. Shared abstractions will be introduced only after multiple providers prove the common shape.

Computation-library wrappers live separately under `quant_warehouse/platforms/computation_libraries/`. They are implementation backends, not replacements for provider-owned feature or target definitions.

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

The current environment installs `quant-warehouse` with required platform dependencies. OpenBB packages come from the `quantarb/OpenBB` `develop` branch, `pandas-ta-classic` comes from the `quantarb/pandas-ta-classic` `main` branch, and computation libraries without `quantarb` forks use their official packages. The warehouse itself reads `QW_HOME`, `QW_ARCTIC_URI`, and `QW_CATALOG_PATH`; by default it stores legacy/shared ArcticDB data under `~/.quant-warehouse/arctic`, provider-isolated ArcticDB roots under `~/.quant-warehouse/arctic/providers/{provider}`, and metadata in `~/.quant-warehouse/catalog.sqlite`.

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

Provider-owned feature and target code should be imported from the provider package:

```python
from quant_warehouse.platforms.data_providers.fmp.feature_engineering import (
    build_price_technical_features,
    build_ttm_financial_statement_features,
)
from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    LabelBuildSpec,
    build_trade_results,
)
from quant_warehouse.platforms.data_providers.thetadata.target_engineering import (
    build_option_label_panel,
)
```

FMP oracle-trade labels use `LabelBuildSpec` and default to `min_profit_pct=0.01`. Long and short trades are solved as separate top-k problems. The `any` target in research notebooks should be treated as the union of side-specific entry labels, not as a mixed long/short capital allocation solver.

FMP event-pair labels are exact event-date labels only. Congress buy/sell, insider buy/sell, analyst upgrades/downgrades, price target raises/cuts, guidance raises/cuts, and earnings beats/misses should be labeled on the date the event happened. Do not create future-window labels for event pairs; future return horizons and oracle-trade labels are separate target families.

## Design rules

1. **ArcticDB is canonical for historical series** — prices, ETF prices, macro series, fundamentals, event pairs, features, calendars, and ThetaData option chains are stored in ArcticDB libraries.
2. **SQLite is metadata only** — the catalog tracks gap-fill state, columns present, date ranges, and profile metadata.
3. **Parquet/CSV are export artifacts only** — reports and derived ML datasets may be written to files, but historical series loaders must read/write ArcticDB.
4. **Silver fundamentals stay sparse** — `period_ending` index, one Arctic symbol per `TICKER__provider`.
5. **Provider-owned features are daily when appropriate** — FMP equity features are daily after PIT joins; ThetaData option targets/features follow option-chain dates.
6. **Only store columns a vendor returns** — no empty cross-vendor placeholders.
7. **Gap-fill** — merge on date/period per symbol; catalog tracks ranges and `last_fetched_at`.
8. **Provider isolation for new writes** — provider-owned data is written to provider-scoped physical libraries with OpenBB route-family-style dataset names such as `yfinance_equity_prices`, `fmp_etf_prices`, `fmp_equity_fundamental_income`, and `thetadata_derivatives_options_eod`. Provider-scoped libraries are routed to provider-specific ArcticDB roots by default, so heavy ThetaData writes use a different LMDB root than FMP or YFinance. Existing shared libraries remain readable as migration fallbacks.
9. **No premature global feature/target layer** — feature engineering and target engineering belong to the data provider until multiple provider implementations justify a shared abstraction.
10. **FMP oracle trade labels are side-specific** — long and short top-k trades are optimized independently. The mixed long/short joint solver and CUDA oracle solver were removed to avoid ambiguous labels.
11. **FMP event-pair labels are same-day only** — event pairs mark the date the event happened. Do not smear events into future-window event columns.

## License

MIT
