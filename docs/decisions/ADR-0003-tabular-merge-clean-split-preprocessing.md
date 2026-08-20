# ADR-0003: Table merge, inspection/cleaning, and split-aware preprocessing boundaries

## Status
Proposed

## Context
The platform needs stable ownership boundaries for:
- N-way tabular merge operations
- dataframe inspection and cleaning commands
- split-aware preprocessing with train/apply semantics that avoids leakage
- output placement across projects, datasets, and artifacts

## Decision

### 1. N-way merge belongs in `research-io`
`research-io` will own generic, backend-neutral table joining and cleaning behavior. Multi-table merge should work as a pure table operation with no knowledge of BIDS layouts or split manifests.

Planned behavior:
- Merge entrypoint supports `N >= 2` table inputs.
- Merge keys are explicit (`on` for each source table pair or explicit per-table mapping).
- In ambiguous cases, a merge without explicit keys is invalid unless a single common candidate key is safely inferable.
- Column collision policy is explicit and deterministic (configurable suffixing/mapping).
- No implicit row matching by DataFrame index; matching is by explicit columns only.
- The operation is pure: it returns merged data and metadata only.
- The command/CLI remains backend-agnostic for supported formats and adapters.

### 2. dataframe inspection and cleaning belongs in `research-io`
Generic, table-level inspection and cleaning functions (examples: previewing value ranges, missingness summaries, duplicate detection, dedupe/drop/normalize operations) are in `research-io` because they are reusable across non-BIDS and BIDS-derived inputs.

These operations are constrained to deterministic, stateless transforms and must:
- accept explicit identifiers where row removal is requested
- preserve provenance metadata describing what changed
- stay independent of split semantics

### 3. raw directory and BIDS discovery belongs in `research-bids`, not `research-io`
`research-io` is a data-agnostic tabular layer and must not resolve BIDS-specific conventions.

Reasons:
- BIDS path semantics, entity parsing, and run-level naming are domain- and dataset-structure concerns, not generic IO semantics.
- Discovery logic changes with acquisition conventions and should not pollute tabular utility APIs.
- Keeping discovery separate allows `research-io` to be used outside BIDS datasets without re-exports or conditional behavior.

### 4. split-aware, leakage-safe fit/apply preprocessing belongs in `research-analysis`
Split-aware fitting is an analysis concern that coordinates dataset partitions and transformation state:
- `fit` is executed only on training rows.
- `transform/apply` is executed on train/validation/test according to the manifest.
- Learned parameters are stored with split provenance.

This logic stays in `research-analysis` because it requires split context and statistical semantics tied to experimental protocol, not to raw table I/O.

### 5. Output staging is explicit by path
- Project-specific split definitions and split metadata: `project/<overlay>/manifests/splits/*`
- Canonical reusable derived tables: `datasets/.../derivatives/*`
- Run-specific or exploratory outputs: `artifacts/tables/*`

### 6. Row dropping must use explicit IDs, never implicit indexes
Dataframe row indexes are backend-specific and unstable after serialization, concatenation, and sort operations.

Therefore:
- all row-removal and row-filtering APIs should receive an explicit subject/item ID or explicit row identifier column,
- if no intrinsic ID exists, a deterministic synthetic row-id is introduced at ingestion,
- and all downstream transforms preserve that identifier column for reproducible joins and auditability.

## Staged implementation plan

1. Finalize this ADR and boundary text in `ARCHITECTURE.md` and `BLUEPRINT.md`.
2. Add/expand `research-io` contracts for N-way merge plus inspection/clean command groups (no split or BIDS discovery dependencies).
3. Finalize `research-bids` discovery APIs to return table descriptors/paths and let orchestration assemble tabular jobs.
4. Add split manifest schema + leakage-safe fit/apply orchestration in `research-analysis`.
5. Implement project-level wiring that routes outputs to:
   - `project/<overlay>/manifests/splits` (split manifests)
   - `datasets/.../derivatives` (reusable derived tables)
   - `artifacts/tables` (transient run outputs)

## Consequences

- Clear ownership of table primitives and preprocessing semantics.
- Fewer hidden couplings between raw dataset layout, splitting policy, and tabular transforms.
- Better reproducibility through explicit row IDs and deterministic output layout.
