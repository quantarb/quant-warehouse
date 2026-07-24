# Repository Rules

## Provider Boundary

- `quant-warehouse` must treat OpenBB as the only market-data/vendor API adapter.
- Do not add direct vendor API calls in `quant_warehouse` code, including direct FMP REST calls, direct ThetaData SDK calls, or ad hoc `requests`/`urlopen` market-data fetches.
- If a vendor endpoint is incomplete or missing, fix or add it in the `quantarb/OpenBB` fork first, then consume it through `quant_warehouse.ingest.openbb_fetch`.
- Direct vendor fallbacks are forbidden. If an OpenBB route is missing, fix the OpenBB fork first.
- ThetaData option-chain downloads must preserve every contract returned for each requested underlying symbol/date. Do not add or keep provider-side DTE, strike-range, moneyness, expiration, right, bid/ask, minimum-ask, liquidity, volume, open-interest, or contract-level filters in download or storage paths. Universe filters may choose which underlying symbols to backfill, but once a symbol/date is selected the stored chain must remain complete so missing data and downstream bugs can be distinguished.

## Dependency Source Of Truth

- OpenBB dependencies must come from `https://github.com/quantarb/OpenBB.git` on the `develop` branch, not PyPI OpenBB packages and not local editable paths.
- Keep `openbb-thetadata` installed through the OpenBB fork provider package. Do not depend on direct `thetadata` SDK calls from `quant-warehouse` application code.
- `quant-warehouse` itself should be installed from `https://github.com/quantarb/quant-warehouse.git@main` by downstream repos.

## Change Placement

- Fix provider pagination, endpoint coverage, schema normalization, and vendor-specific behavior in the OpenBB fork.
- Keep `quant-warehouse` focused on storage, point-in-time normalization, provider-owned feature engineering, provider-owned target engineering, and route orchestration through OpenBB.
- Put provider-specific feature and target code under `quant_warehouse/platforms/data_providers/{provider}/`.
- Do not recreate top-level `quant_warehouse.feature_engineering` or `quant_warehouse.target_engineering` packages. The current code intentionally keeps FMP equity/fundamental logic under `platforms/data_providers/fmp` and ThetaData option logic under `platforms/data_providers/thetadata`.
- Put computation-library adapters under `quant_warehouse/platforms/computation_libraries/`; those adapters are implementation backends, not canonical feature-family definitions.

## Warehouse Responsibility

- Do not apply reporting, disclosure, or filing lags when constructing model features or aligning feature-family observations. Preserve each source observation date exactly unless the user explicitly requests a lag. Forward-fill is permitted only from that recorded date into later dates.
- Event targets must use exact `(symbol, event_date)` matching. Do not forward-align or smear event labels across a date tolerance window unless the user explicitly requests it.
- Historical backfills must request `1900-01-01` as the warehouse floor. If a provider has a later availability floor, preserve that provider floor and every returned observation; never replace historical dates with the ingestion timestamp.

- Treat `quant-warehouse` as the opinionated persistence layer over the OpenBB fork SDK.
- Use it to request vendor data through OpenBB, normalize schemas, compare requested refreshes against what is already stored, and write point-in-time warehouse datasets.
- Store raw-enough provider datasets first, then filter in downstream feature, label, model, or backtest code after reading from the warehouse. For ThetaData options specifically, warehouse persistence is full-chain-only; selection filters belong in `quant-orchestrator` or in warehouse feature/target builders that operate on already stored full chains.
- Do not put ML model training, backtesting engines, broker integrations, or order submission logic in this repo.
- FMP event-pair targets are exact event-date labels only. Never create future-window event-pair labels. Never use no-event dates as negative examples for event-pair classification. The negative class must be the mirrored negative event on an actual event date. Forward returns and oracle-trade horizons are separate target families; they must not be encoded by smearing event-pair labels across future dates.
- FMP oracle-trade side classifiers must use one buy/sell task across all configured `k` values. Never create separate oracle tasks or models per `k`. Use actual oracle entry dates only: buy/long entries versus sell/short entries. Never use non-entry dates as negative examples, and never train binary event classifiers from the oracle `any` union target.

## Compatibility Policy

- This repo is new and rapidly changing. Do not add new backward-compatibility wrappers, legacy aliases, or duplicate old APIs.
- When a contract changes, update callers and notebooks directly.
- Existing migration helpers and compatibility shims reflect current code state; remove them when the migration path is no longer used instead of extending them.

## Build Vs Buy Policy

- Prefer widely used, actively maintained third-party packages or small forks of proven projects over custom implementations.
- For storage, validation, scheduling helpers, dataframe operations, feature engineering primitives, and target engineering utilities, use battle-tested libraries when they fit the warehouse boundary.
- Build from scratch only when no reliable package fits the requirement or the warehouse needs a small opinionated wrapper around a proven dependency; document that reason in the change.

## Notebook Policy

- Use `notebooks/` for one-off data work only: EDA, warehouse refresh inspection, schema experiments, experimental feature engineering, and experimental target engineering.
- `notebooks/` should contain notebook files only. Do not add standalone `.py` scripts in that directory.
- Keep exploratory, one-off, and notebook-only code inside the notebook itself so people can edit and rerun it interactively.
- Notebooks in this repo must not train ML models, run backtests, simulate portfolios, or submit orders.
- Even notebooks must use OpenBB through `quant-warehouse`; do not call vendor APIs directly.

## Performance Policy

- Prefer vectorized Pandas/NumPy and batched warehouse reads/writes.
- Add GPU/CUDA acceleration only for data transformations where it materially helps. The FMP oracle-trade target path intentionally does not have a CUDA implementation because the measured version was not materially faster than the CPU/Numba path.
- Do not move model training or backtesting into this repo for CUDA reasons; those belong in `quant-orchestrator`.
- The six curated technical feature families are the sole production technical contract. Do not restore an exhaustive/all-indicator mode, expose a mode switch, or add a second bloated technical builder.
- Public feature-panel builders must use the curated technical set by default and support observation-date projection so downstream workflows do not materialize unnecessary full-history panels.

## Git Hygiene

- This workspace is maintained by a single author. Push completed changes directly to the remote branch; do not create or wait on pull requests unless the user explicitly asks for one.
- Use `--force-with-lease` only when intentionally rewriting the current branch history.
