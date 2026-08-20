# ADR-0016: Backend-Neutral MVPA Pattern-Source Adapters

## Status

Accepted

## Context

MVPA planning needs to translate project-owned unit selection and
backend-specific imaging layouts into pattern rows that generic analysis code
can understand. The first implementation planned FSL FEAT parameter-estimate
images directly. Letting each additional source add another conditional branch
to the core planner would couple shared behavior to FEAT, obscure which sources
are actually available, and make it easy for adapters to reconstruct subject,
session, and run combinations that do not exist.

ADR-0015 established batches as the canonical exact-unit row store, cohorts as
named views, and bundles as references to component configuration. MVPA source
planning should consume those resolved units without adding a second manifest
or another selector system.

## Decision

### Use one explicit adapter registry

`research-neuro` owns one internal registry for MVPA pattern-source adapters.
Each adapter has a stable name and capability/readiness description and
implements two operations:

- validate a source configuration without loading pattern data; and
- plan that source into canonical pattern rows without executing analysis.

Core planning selects adapters only through this registry. Registration is
explicit in the alpha; it does not use dynamic imports, package entry points,
or filesystem discovery. This keeps installed capabilities deterministic,
auditable, and testable while the internal interface is still evolving. A new
adapter can be registered without adding a backend-specific conditional branch
to the core planner.

### Emit one canonical pattern-row contract

Each canonical pattern row represents one actual analysis unit and one
condition. Shared fields contain:

- a deterministic unit identity;
- canonical `subject_id` and optional `session_id`, `task_id`, and BIDS
  `run_id` values;
- an explicit cross-validation label;
- a condition identifier;
- source and backend names;
- a representation kind: `image` or `prepared_features`;
- a portable pattern reference and optional portable noise reference;
- optional event-count and QC/status fields;
- preserved unit metadata; and
- backend-specific metadata in an isolated mapping.

Shared fields do not contain FSL EV indices, parameter-estimate numbers, design
rows, cope mappings, or FEAT layout conventions. Those details belong in the
FSL adapter's backend metadata and in any private compatibility details needed
by the existing runtime.

`image` means the adapter references an image representation that a later
extraction step can read. `prepared_features` means the adapter references an
already materialized feature representation. The representation label does not
assert that the referenced adapter or execution path is available.

Canonical pattern and noise references use named roots or relative paths when
the mapping is truthful. Runtime-resolved local paths may remain in private
planning details required by an existing execution consumer, but they are not
portable publication references.

### Prefer exact resolved bundle units

The preferred planner input is the exact resolved row sequence from an analysis
bundle. Every row requires `subject_id`; session, task, and run identities are
optional. The planner preserves source order, canonical identifiers, and
arbitrary deterministic metadata. It rejects duplicate configured unit keys
and does not create Cartesian combinations. Runtime adapters may derive local
aliases, but they do not replace the stored identity values.

The existing inline subject/session/run selector remains only as the explicit
`legacy_cartesian` compatibility mode. Exact units and legacy selectors cannot
be mixed. Keeping the old behavior visible preserves current configurations
and execution consumers without presenting a Cartesian design as the preferred
scientific contract.

No new unit manifest is introduced. Batches remain the exact-unit row store,
cohorts remain filtered views, and bundles remain configuration references.
Pattern rows are deterministic planner output rather than another user-edited
manifest family.

### Separate schema validity, adapter availability, and readiness

Validation can accept a structurally coherent source for an unavailable
adapter. Doctor reports schema validity, adapter availability, and execution
readiness separately, and returns nonzero when the chosen adapter cannot
execute. Planning remains non-mutating: it does not write outputs, load full
pattern data, invoke an external tool, or execute a statistical analysis.

`fsl_feat_pe` is the only integrated source adapter in this alpha. Its adapter
emits canonical rows while preserving current `condition_pe_rows`, `pe_image`,
and related compatibility fields and execution behavior. FSL EV, PE, design,
cope, and FEAT metadata remains adapter-specific.

The following adapter names describe deferred source schemas rather than
execution-ready implementations:

- `materialized_pattern_table`;
- `bids_derivative_pattern_table`;
- `nilearn_glm`;
- `surface_cifti`.

`materialized_pattern_table` is the next planned implementation. A future SPM
beta-image adapter could map SPM metadata and image references into the same
canonical row contract. This decision does not register, implement, or claim
SPM support.

## Consequences

Positive:

- core planning stays backend-neutral as source types are added;
- exact cross-sectional and irregular longitudinal units reach adapters without
  invented combinations;
- shared consumers receive stable identities and representations without
  learning FSL conventions;
- availability and readiness claims remain truthful for deferred adapters;
- current FSL planning and runtime consumers retain their compatibility fields.

Tradeoffs:

- the internal adapter registry must be updated deliberately for each supported
  source;
- compatibility output temporarily coexists with canonical pattern rows;
- exact-unit callers must choose unit keys that distinguish their real inputs;
- a valid deferred configuration cannot become execution-ready until its
  adapter is implemented.

Rejected alternatives:

- backend-specific conditionals in the core planner, because they duplicate
  dispatch and leak source conventions into shared logic;
- dynamic import or entry-point discovery during the alpha, because it makes
  installed capabilities less predictable before the contract stabilizes;
- a new pattern-unit manifest, because batches and bundles already own exact
  unit identity and selection;
- silently translating legacy lists into the preferred model, because that
  would hide Cartesian compatibility behavior;
- placing FSL EV or PE fields in the shared row, because that would make the
  supposedly generic contract backend-specific.
