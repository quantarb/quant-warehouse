# Repository Rules

## Provider Boundary

- `quant-warehouse` must treat OpenBB as the only market-data/vendor API adapter.
- Do not add direct vendor API calls in `quant_warehouse` code, including direct FMP REST calls, direct ThetaData SDK calls, or ad hoc `requests`/`urlopen` market-data fetches.
- If a vendor endpoint is incomplete or missing, fix or add it in the `quantarb/OpenBB` fork first, then consume it through `quant_warehouse.ingest.openbb_fetch`.
- Existing direct vendor fallbacks are technical debt. Prefer removing them after the matching OpenBB fork provider route is available and tested.

## Dependency Source Of Truth

- OpenBB dependencies must come from `https://github.com/quantarb/OpenBB.git` on the `develop` branch, not PyPI OpenBB packages and not local editable paths.
- Keep `openbb-thetadata` installed through the OpenBB fork provider package. Do not depend on direct `thetadata` SDK calls from `quant-warehouse` application code.
- `quant-warehouse` itself should be installed from `https://github.com/quantarb/quant-warehouse.git@main` by downstream repos.

## Change Placement

- Fix provider pagination, endpoint coverage, schema normalization, and vendor-specific behavior in the OpenBB fork.
- Keep `quant-warehouse` focused on storage, point-in-time normalization, feature engineering, target engineering, and route orchestration through OpenBB.

## Warehouse Responsibility

- Treat `quant-warehouse` as the opinionated persistence layer over the OpenBB fork SDK.
- Use it to request vendor data through OpenBB, normalize schemas, compare requested refreshes against what is already stored, and write point-in-time warehouse datasets.
- Do not put ML model training, backtesting engines, broker integrations, or order submission logic in this repo.
