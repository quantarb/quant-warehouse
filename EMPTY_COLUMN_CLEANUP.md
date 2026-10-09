# Empty payload schema cleanup

The warehouse omits wholly missing payload columns when writing datasets. Missing means null/NaT, floating NaN, or an exact empty string. Zero, false, infinity, whitespace, nested values, observation dates, index fields, reserved identities/dimensions, and explicitly supplied keys remain meaningful. A zero-row schema never proves emptiness.

`empty_payload_columns(frame, key_columns=())` shares this policy between normal storage writes and the explicit migration. Additional dataset-specific keys must be passed by callers. No vendor API, downloads, training, compaction, or historical version purge is involved.

Unmigrated reads restore their existing omitted-column metadata by default. `ArcticBackend.read(..., include_empty_columns=False)` requests a clean view. Explicit column requests reconstruct known omitted/removed fields using a recorded zero-row Arrow schema, even after migration. Missing values are reconstructed as null with their recorded dtype. Retained historical versions preserve original representations. Migrated defaults omit removed payloads; future refreshes restore a field once it gains a real value.

## Explicit migration

`prepare_empty_column_cleanup(library, symbol, max_rows=100_000, key_columns=())` reads every value of a pinned version and returns a plan without writing. It accepts an existing authorized Library and does not open or create storage. It refuses oversized frames, duplicate column names, non-dictionary metadata, conflicting omitted-column metadata, and removal of the final payload without a temporal index. Inspection-only missing-sentinel normalization does not modify the original pandas frame used for retained-value comparisons and writes.

`apply_empty_column_cleanup(library, plan)` checks the latest version and validates the original frame and retained hash before applying. Physical removals write the cleaned frame; logical removals of already omitted fields write metadata only. Every migration explicitly uses `prune_previous_versions=False`, verifies the original and new versions, and preserves custom metadata. The caller must serialize writers: ArcticDB does not expose a compare-and-swap write through this API. The operational runner also records and verifies all prior versions, deletion flags, and snapshot references.

Unnamed DatetimeIndex observations are exposed as `date` by the reader. Migration column-order metadata must include that observation date. A pure date-order metadata repair preserves every other metadata entry, including existing migration provenance.

## Approved operational pass

The resumable operational pass inventories 263,251 current tables across the legacy store and FMP, ThetaData, and yfinance provider stores. Historical audits only prioritize work; every migrated field is revalidated against all current values. The runner uses a 100,000-row audit bound, serial bounded batches, an exclusive runner lock, a 35 GiB available-space guard, and checks for concurrent warehouse processes. It skips ambiguous or oversized tables and records failures rather than blindly retrying possible writes.

Per-table receipts are fsynced before advancing the atomic durable cursor. Exact manifests distinguish physical payload removal from logical metadata removal. They record source/new versions, retained hashes, indexes, empty-value evidence, exclusions, and prior version/snapshot references. Independently stored legacy/provider symbols are audited separately.

An initial omission in column-order metadata hid observation dates on default reads in 2,625 migrated FMP tables. Stored indexes, rows, dtypes, and nonempty values remained identical to retained source versions. The explicitly approved correction copied existing metadata and prepended only `date`, preserving all other metadata/provenance and history. All 2,625 received independent data, metadata, history, and default-reader verification. Separately, 507 inspection conversion errors occurred before writing; the tested inspection fix recovered them as 73 changed tables and 434 no-ops. The original failed-attempt receipts remain available.

## Validation and storage

Regression coverage checks physical and logical missing payloads, date/index visibility, explicit empty-column reconstruction, keys and meaningful values, refresh behavior, source-version changes, malformed metadata, metadata-only writes, snapshots, and original-value/dtype preservation. Supporting reader, normalization, catalog, storage policy, refresh planner, and provider tests also run in isolated memory or temporary stores.

Old versions remain readable, so schema cleanup does not promise immediate LMDB disk-space recovery. Report logical schema reductions separately from measured filesystem changes. No purge or compaction is authorized.
