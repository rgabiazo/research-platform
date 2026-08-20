# ADR-0015: Exact Analysis Units, Cohort Views, and Analysis Bundles

## Status

Accepted

## Context

Cross-sectional and longitudinal analyses need a stable way to identify the
exact inputs that belong together. Real studies are often irregular: subjects
can have different visits, tasks, acquisitions, or run counts. Constructing a
grid from separate subject, session, task, and run lists can therefore invent
combinations that do not exist.

The workspace already uses `project/<name>/manifests/batches/*.tsv` for tabular
and BIDS input rows. It also has named ROI, extraction, and MVPA configuration
locations. A higher-level analysis contract should reuse those conventions
rather than introducing another unit manifest, copied cohort rows, or a second
exclusion table.

## Decision

### Batches are the exact-unit table

`manifests/batches/<batch>.tsv` is the only canonical project-level analysis
unit row store. Each row is one real analysis or input unit. A neuro unit
requires `subject_id`; `session_id`, `task_id`, and BIDS `run_id` are optional.
Additional deterministic metadata such as cohort, eligibility, QC, exclusion,
timepoint, visit order, acquisition, direction, and adapter-specific fields is
preserved.

Resolvers retain source row order and canonical stored identifiers. They do
not strip BIDS prefixes, synthesize missing entity dimensions, balance groups,
or construct Cartesian combinations. Runtime adapters may derive aliases from
the stored identifiers later, but they must not replace the canonical values.

### Cohorts are named views

`config/cohorts.yaml` defines a cohort as a view over exactly one batch. Include
filters use OR within a column and AND across columns. Explicit exclusion rules
run after inclusion and carry stable rule identifiers plus either fixed reason
text or a configured reason field. Unknown filter columns and missing required
values are errors. Rules that match no included row remain visible in doctor
and plan output instead of disappearing silently.

Included, excluded, and incomplete rows remain auditable. The platform does not
write copied subject lists under `manifests/cohorts/` or separate exclusion-row
manifests.

### Bundles reference selection and component configuration

Plan-only bundle YAML lives at
`config/analysis/bundles/<name>.yaml`. A bundle selects exactly one named cohort
or one batch directly. It cannot mix both forms or contain literal
subject/session/task/run lists.

The bundle declares unit key columns, a subject column, optional longitudinal
fields, named component references, and ordered stages. Component references
point to existing project configuration such as:

- `config/analysis/roi_sets/<name>.yaml`
- `config/analysis/extraction_sets/<name>.yaml`
- `config/analysis/mvpa/<name>.yaml`

Bundle doctor checks reference existence and stage/component consistency, but
does not run domain validators, read images, invoke tools, or execute stages.

### Cross-sectional and longitudinal representation

A cross-sectional batch can contain only `subject_id` and useful metadata. It
does not require a fake session, task, or run.

Longitudinal bundles may configure an occasion column, required occasion
values, an explicit occasion-order column such as `visit_index`, and an
incomplete-case policy:

- `fail`: report the missing occasions and reject the plan;
- `drop`: exclude every unit for an incomplete subject and report each removal;
- `allow`: retain the available units and report the incomplete subject.

Session labels are identifiers, not a chronology. Resolvers never infer visit
order from lexical session ordering. Multiple runs within one occasion remain
separate stored rows and require an appropriately specific unit key.

### Planning and provenance

`rp analysis bundle validate`, `doctor`, and `plan` are non-mutating. Plans
report included and excluded rows, stable reasons, entity counts, incomplete
cases, named components, ordered stages, a SHA-256 digest of the source batch
bytes, a SHA-256 digest of the effective bundle configuration, a deterministic
plan digest, and `executed: false`.

BIDS `run_id` identifies an imaging run within a stored unit. A future bundle
execution will receive a separate `execution_id` or analysis run identifier;
the two identities must not be overloaded.

## Scientific Design Boundary

Scientific selection, exclusion, unit identity, longitudinal completeness, and
stage composition remain in reviewed project configuration. The bundle CLI
therefore does not expose a large subject/session/run/ROI/condition selector
surface. ADR-0009's run-local BIDS selector flags remain useful for operational
subsets of an already defined batch; they do not replace the configuration-owned
scientific design described here.

## Future Adapter Boundary

Future local and SLURM execution must consume the same resolved unit rows.
Pattern-source adapters receive those rows without recreating Cartesian
combinations. FSL-specific parameter-estimate and design metadata remains
adapter-specific. A future SPM adapter could map SPM metadata and images into
the same generic pattern-row contract; this decision does not claim or
implement SPM support.

## Consequences

Positive:

- cross-sectional and irregular longitudinal data share one exact-row model;
- cohort decisions and exclusions remain reviewable without duplicating rows;
- named bundles compose existing domain configurations without owning domain
  execution;
- deterministic digests give later local and SLURM consumers a common input
  identity.

Tradeoffs:

- users must choose unit keys that distinguish every intended unit;
- incomplete longitudinal data requires an explicit policy;
- configuration and batch changes intentionally change plan digests.

Rejected alternatives:

- a second `analysis-units.tsv` family, because it duplicates batches;
- copied cohort or exclusion manifests, because they can drift from the source
  rows and obscure why a row was removed;
- separate subject/session/run lists, because their Cartesian product can
  invent nonexistent inputs;
- a large selector flag surface, because scientific design would become an
  ephemeral invocation rather than reviewed configuration.
