# ADR-0017: Materialized MVPA Pattern Tables

## Status

Accepted

This decision supersedes ADR-0016 only where that earlier decision described
`materialized_pattern_table` as wholly deferred. Its schema, scalar planning,
and strict internal loading are implemented here. ADR-0018 subsequently adds
exact-bundle CLI execution and whole-runtime-root transaction safety without
changing this table contract.

**Implementation status (2026-07-21).** The historical decision below
predates the checked-in example. The repository now includes a deterministic
16-row synthetic materialized table, an exact-unit batch, cohort and bundle,
an execution-ready `toy-crossnobis` configuration, a real local CLI lifecycle,
and a normative [v1 producer reference](../materialized-pattern-table-v1.md).
That evidence makes the bounded local materialized-pattern crossnobis path
runnable. It does not add public real-data validation or support for image,
FSL, SPM, mixed-representation, HPC, RDM-export, or publication workflows.

## Context

ADR-0016 defined a backend-neutral pattern-row contract and an explicit MVPA
source-adapter registry. Its first integrated source, `fsl_feat_pe`, discovers
images that the existing runtime later extracts. A second source is needed for
feature vectors that have already been extracted and finalized per ROI, without
forcing generic analysis code to understand FEAT layouts or repeat image
extraction.

A loose delimited-table interface would be unsafe. In particular, a table can
look rectangular while mixing voxel order, ROI definitions, feature spaces,
cross-validation folds, centering states, or incompatible noise estimates. It
can also silently reconstruct combinations that were absent from the exact
analysis-unit batch. Planning must therefore establish identity and portable
provenance without eagerly decoding potentially large vectors, while the later
loader must validate vectors strictly before analysis consumes them.

## Decision

### Use one fixed v1 source contract

The source configuration is configuration-owned and has this shape:

```yaml
pattern_sources:
  - name: prepared_patterns
    backend: materialized_pattern_table
    root_ref: mvpa_inputs
    path: patterns.tsv
    schema_version: research_platform.neuro.mvpa.materialized_pattern_table.v1
```

The schema identifier is exactly
`research_platform.neuro.mvpa.materialized_pattern_table.v1`. The configured
identifier and every row's `schema_version` must agree. Unknown schema versions
are errors rather than invitations to guess aliases or coerce another format.

Each TSV row represents one final ROI feature vector for one exact analysis
unit and one condition. The v1 contract carries:

- a unique `pattern_id`;
- every configured exact-unit key, including required `subject_id`, with
  `session_id`, `task_id`, and BIDS `run_id` present only when applicable;
- required `condition_id` and an optional `cross_validation_label`; when the
  latter is absent, the planner derives it from the exact unit and configured
  cross-validation contract, and when supplied it must match that derivation;
- `pattern_source_name`, `roi_source_name`, and `roi_label`;
- `feature_count`, `voxel_order`, `voxel_index_hash`, `feature_space_id`,
  `roi_definition_id`, and `feature_values`;
- `usable`, `status`, `noise_status`, and `noise_usable`; and
- `mean_centering_applied` and `mean_centering_scope`.

The feature vector is already ROI-final. The adapter does not expand ROI
definitions, apply a mask, change feature order, recenter values, or infer a
feature space. `voxel_index_hash`, `feature_space_id`, and `roi_definition_id`
make alignment reviewable without treating a machine-local mask path as
identity.

Optional scalar metadata can carry event counts, QC reasons, exclusions,
grouping values, warnings, errors, generator or software version, derivation or
holdout labels, and a portable ROI-definition reference. Safe deterministic
exact-unit metadata columns may also be retained. Optional fields do not relax
the required identities or vector checks.

### Join to exact units without expanding them

The adapter joins table rows to the ordered exact-unit rows supplied by the
analysis-bundle resolver. It uses the configured unit-key columns and canonical
stored values; it does not strip BIDS prefixes, synthesize missing dimensions,
or create a Cartesian product. A selected table row must match exactly one
resolved unit; other rows remain audited but do not join. Duplicate pattern
identities, duplicate scientific row identities,
missing key values, ambiguous matches, and incompatible table identities are
errors.

Planner output order is deterministic: resolved unit order first, then
configured condition order, then ROI source and ROI label. Source-table row
order is retained only where those keys do not otherwise distinguish rows.
Session labels are never interpreted as chronological order. Rows in the table
that do not belong to selected units remain auditable through stable unselected
identifiers and counts; they are not silently promoted into the analysis.

The materialized table is a source artifact, not another unit manifest.
Batches, cohorts, and bundles remain authoritative for unit selection.

### Keep planning scalar-only and portable

Planning validates configuration, root availability, the header, selected-row
scalar identities, exact-unit joins, and deterministic ordering. Unselected
rows still satisfy the fixed header and required scalar/key cells, while
selection-specific condition, ROI, CV, centering, QC, and noise compatibility
is deliberately limited to rows that join the resolved units. It records the
SHA-256 digest of the source table bytes, the fixed columns, total, selected,
and unselected row counts, and stable unselected identifiers. Canonical planned
rows use `prepared_features` and portable named-root or relative references.

The planner does not decode `feature_values` or `noise_values`, load images,
compute distances, invoke external tools, or write outputs. Host-local paths
are not canonical references. A plan therefore proves source identity and
selection, not numerical usability.

### Load vectors strictly and all at once

Materialization is a separate internal operation. Before decoding vectors, the
loader verifies that the table still has the digest recorded by planning. It
then materializes the selected rows as backend-neutral mappings and either
returns the complete validated set or fails without returning a partial set.

For every usable pattern row, the loader requires:

- a finite numeric `feature_values` array whose length equals `feature_count`;
- nonempty and mutually compatible feature order, index hash, feature-space,
  and ROI-definition identities;
- the cross-validation label resolved from the exact unit and configured CV
  contract; an optional table `cross_validation_label` must match it, and
  session labels are never treated as an implicit chronological ordering;
- explicit centering state and scope; and
- coherent usability, status, QC, warning, error, exclusion, and reason fields.

The returned mapping preserves `cross_validation_label` for
`research-analysis` row preparation. The analysis layer may select that column
explicitly; its historical run-, session-, subject-, and custom-column defaults
remain unchanged.

Noise requirements follow the configured normalization. Diagonal normalization
requires the complete strict noise contract: `noise_values` must be finite and
strictly positive, with usable state, count, voxel order, index hash, feature
space, ROI identity, `noise_value_kind: variance`, estimation scope, source,
and status consistent with the feature vector. Identity normalization requires
`noise_status: unused` and `noise_usable: false`; vector payloads are not used
for the computation.

### Separate materialization from lifecycle authorization

The adapter can validate the v1 schema, plan the table, and materialize rows for
internal consumers. Those capabilities alone do not authorize output mutation.
As originally accepted, this decision deliberately deferred CLI execution.
ADR-0018 now permits execution only when the same exact plan is joined to a
resolved analysis bundle, the representation-aware runtime is ready, and the
complete compute-before-write transaction passes preflight. `validate`,
`doctor`, `plan`, and default `run` remain non-mutating.

No checked-in materialized table, public project configuration, or executable
project-level MVPA happy path is added. MVPA therefore remains experimental or
external-runtime after ADR-0018. `fsl_feat_pe` retains its external-runtime
planner and tested image-dispatch boundary, but its CLI execution remains
deferred until portable manifest identities are available.
`bids_derivative_pattern_table`, `nilearn_glm`, and `surface_cifti` remain
deferred, and no SPM adapter is registered or claimed.

## Consequences

Positive:

- pre-extracted ROI vectors can enter the shared pattern-row contract without
  coupling analysis code to an imaging backend;
- exact-unit identity, fold identity, ROI identity, centering, feature order,
  and noise compatibility are explicit and testable;
- scalar-only plans stay lightweight while binding later loading to exact
  source bytes;
- portable references and digests make plans reproducible across hosts; and
- existing row-preparation and numerical behavior remain backward compatible.

Tradeoffs:

- producers must emit the fixed v1 schema rather than an arbitrary convenient
  table;
- a changed table must be replanned before loading, even when the change seems
  harmless;
- strict all-or-nothing loading rejects partially usable files instead of
  returning an implicit subset; and
- public-alpha users still need their own generated pattern table, exact-unit
  bundle, and reviewed runtime configuration because no public example is
  checked in.

Rejected alternatives:

- accepting a supplied cross-validation label that disagrees with the
  configured exact-unit derivation, because that can change the intended fold
  contract;
- treating the table as a second subject or exclusion manifest, because exact
  selection already belongs to batches and cohorts;
- decoding vectors during planning, because planning should remain lightweight
  and non-executing;
- arbitrary column mappings, alternate delimiters, shards, and binary formats,
  because v1 needs one reviewable producer contract before adding format or
  discovery complexity;
- accepting aliases for vector identity or centering fields, because implicit
  coercion makes incompatible rows appear comparable; and
- treating loader availability as execution authorization, because ADR-0018
  requires exact-bundle resolution, complete readiness, explicit `--execute`,
  and a failure-safe runtime transaction.
