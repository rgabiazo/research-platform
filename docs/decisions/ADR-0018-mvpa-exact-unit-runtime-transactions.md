# ADR-0018: Exact-Unit MVPA Runtime Transactions

## Status

Accepted

This decision extends ADR-0015, ADR-0016, and ADR-0017. It supersedes only
ADR-0017's statement that materialized-pattern CLI execution is deferred.
There is still no checked-in public project-level MVPA execution example.

## Context

The MVPA lifecycle already had configuration validation, backend-neutral
pattern-source planning, FSL image discovery, materialized-pattern-table
planning, strict vector loading, row preparation, distance computation,
summaries, and runtime writers. These pieces did not yet form one coherent
local execution contract.

In particular, the CLI could write image-extraction outputs before later
preparation, distance, or summary failures. The individual writers replace one
file atomically, but sequential writer calls do not make the complete run
atomic. A late collision or serialization error could therefore leave a
runtime directory that looked usable despite representing an unsuccessful
analysis. Materialized patterns also needed the exact units selected by the
authoritative bundle resolver rather than another selector or manifest system.

## Decision

### Hand exact bundle units to MVPA

The preferred lifecycle takes one named analysis bundle through `--bundle`.
The CLI resolves that bundle with the authoritative research-core resolver and
passes its ordered included units, key columns, excluded-unit audit, component
references, and digests into MVPA planning. The bundle's
`components.mvpa_set` must name the requested MVPA set.

`--bundle` is the only scientific-selection flag added to the MVPA lifecycle.
Subject, session, task, run, condition, ROI, feature, and noise choices remain
in batch, cohort, bundle, and MVPA configuration. The lifecycle does not copy
resolved rows into MVPA YAML and does not choose a bundle automatically.
Cross-sectional and irregular longitudinal rows retain their actual order and
combinations; no Cartesian expansion is introduced.

Legacy inline subject, session, and run selectors remain an explicitly
advanced `legacy_cartesian` compatibility path for existing FSL configurations
when no bundle is requested. They cannot be mixed with an exact bundle
selection.

The bundle command family remains plan-only and gains no execution command.
MVPA consumes a bundle resolution; it does not turn the bundle itself into an
execution engine.

### Separate lifecycle meanings

The local commands have distinct contracts:

- `validate` checks the MVPA YAML schema only;
- `doctor` resolves the requested bundle and checks complete execution
  readiness without analysis or writes;
- `plan` produces the deterministic unit, source, adapter, output, collision,
  and provenance plan without decoding vectors or computing distances;
- `run` without `--execute` previews the complete runtime and output contract
  without computation, temporary paths, or writes; and
- `run --execute` is the only authorization for local analysis output
  mutation.

Schema validity, bundle validity, plan validity, materialization readiness, and
execution readiness are reported separately. Planning errors are errors rather
than schema-valid success. Execution cannot begin unless every required
readiness gate passes.

### Dispatch by representation

One runtime facade consumes the exact private execution handle retained by the
plan. Execution handles are excluded from serialized plans, equality, public
provenance, and CLI output. Canonical pattern rows use portable references;
advanced FSL compatibility details remain run-local planning audit and may
contain resolved host paths. Those compatibility details are not portable
publication references and are never copied into a successful-run manifest.

The facade dispatches:

- `image` rows through the existing image and ROI-mask extraction path; and
- `prepared_features` rows through the digest-checked
  `materialized_pattern_table` loader.

Both paths produce the same backend-neutral pattern-row, QC, and provenance
boundary before research-analysis preparation. Materialized rows retain their
canonical `cross_validation_label`; they are not represented as FSL parameter
images, masks, or compatibility rows. Mixed image and prepared-feature sources
remain unsupported in v1. Unknown or deferred representations are not
execution-ready.

The materialized adapter is execution-capable only through this facade and
transaction. Planning and loading the adapter in isolation do not authorize
output mutation. FSL discovery, parameter-estimate mapping, image extraction,
noise values, and numerical behavior remain unchanged. The FSL/image path is
not CLI-execution-ready in this alpha because it cannot yet populate the
portable feature-space, ROI-definition, and noise-scope identities required by
the successful-run manifest. It remains available for planning and tested
representation dispatch rather than using the former sequential-write bypass.

### Compute the complete result before writing

An authorized run first rechecks source digests, materializes or extracts all
patterns, validates QC, prepares every group, computes every requested
distance, and computes summaries in memory. Fatal errors, zero usable
patterns, missing condition coverage, inadequate independent CV partitions,
zero usable distances, or zero summaries fail the complete run before output
staging begins. A successful group cannot be published when another requested
group fails.

This decision changes orchestration and durability, not the crossnobis
formula, condition-pair semantics, feature preparation, noise calculations, or
summary mathematics.

### Own one complete runtime root

Runtime configuration has one v1 collision policy:

```yaml
runtime:
  existing_output: fail
```

`fail` is the default and the only supported value. The configured runtime root
is the transaction unit. Any existing final root or planned destination fails
during read-only preflight, before vector or image loading and before numerical
work. A deliberate rerun uses a different configured artifacts path. Scaffold
`--force` replaces only scaffold YAML; it never authorizes runtime replacement.

Preflight evaluates the complete inventory at once. It rejects duplicate or
lexically aliased destinations, paths outside the runtime root or named root,
existing files and directories, special files, symbolic-link destinations or
parents, invalid parents, representation-specific name conflicts, and the
successful-run manifest collision.

After all computation succeeds, the runtime creates one unique hidden sibling
staging directory on the same filesystem as the final root. Existing writers
serialize their complete outputs into that owned tree. The staged inventory,
TSV headers and row counts, JSON, finite numeric content, hashes, provenance
relationships, and portable path boundary are validated before promotion. The
whole directory is then committed with one atomic rename when the filesystem
supports it.

A writer, serialization, validation, interruption, promotion, or concurrent
destination-claim failure leaves the final root absent and removes all owned
temporary paths. If cleanup itself fails catastrophically, the recovery tree
is retained and its location is reported rather than destroying the last
recoverable copy.

### Commit with a deterministic successful-run manifest

The final commit marker for the implemented materialized-table execution path
is a successful-run manifest inside the runtime root. It records status
`succeeded`; project and MVPA identity; MVPA, bundle, batch, selection, and
source-table digests; adapter and representation; exact included and
excluded unit identities and counts; the CV contract; conditions and pairs;
ROI, feature-space, index, and ROI-definition identities; centering, noise,
threshold, exclusion, and distance-engine contracts; row counts; warnings; an
empty successful error list; and the relative paths and SHA-256 values of the
other runtime outputs.

The manifest does not include its own hash because that would create a
self-referential digest. It is nevertheless part of the validated exact file
inventory and acts as the commit marker. The manifest and portable provenance
contain no absolute roots, private execution handles, temporary paths,
machine-local values, or environment-specific state. They contain no timestamp
unless a future required schema introduces one. Identical materialized-table
inputs written into independent clean roots therefore produce byte-identical
text and JSON output. This decision makes no equivalent determinism claim for
the deferred FSL/image CLI path.

## Consequences

Positive:

- exact bundle units reach MVPA without another manifest or selector system;
- materialized and image sources share one representation facade, while only
  sources satisfying the portable-manifest contract pass CLI readiness;
- late scientific or serialization failures cannot publish a partial run;
- a successful runtime root is distinguishable by its validated commit
  marker; and
- fail-if-exists behavior makes reruns deliberate and preserves prior results.

Tradeoffs:

- v1 cannot resume, replace, or repair an existing runtime root;
- all requested results must fit the existing in-memory computation contract
  before serialization;
- private execution handles must remain attached to the exact in-process plan;
  and
- materialized execution still requires user-supplied prepared vectors because
  no public MVPA project example is checked in; the external image path remains
  plan/dispatch-only until its portable manifest identities are complete.

Rejected alternatives:

- adding subject/session/run selector flags, because scientific selection is
  already owned by batches, cohorts, and bundles;
- replanning or accepting a caller-provided table digest during execution,
  because that would break the binding to the reviewed plan;
- publishing each writer group as it succeeds, because a late failure would
  leave a misleading partial run;
- overwriting or resuming a runtime root, because safe replacement and recovery
  semantics need a separate versioned design; and
- adding bundle execution, because bundles remain generic configuration-owned
  unit resolutions rather than workflow executors.

## 2026-07-20 Implementation Status

The repository now includes a deterministic checked-in public execution
example for this transaction. The `project-example` `toy-crossnobis` bundle
resolves exact units and runs local materialized-pattern crossnobis through the
real `rp` CLI. Its integration verification covers the complete 14-file
transaction, deterministic results, source and output digests, collision
refusal, and portable provenance. Only this bounded local materialized-pattern
path is **Runnable locally**.

The earlier statements in this ADR that no public MVPA example was checked in
and that materialized execution required user-supplied prepared vectors
describe the repository state when ADR-0018 was accepted. They remain above as
historical context and are no longer current status. FSL/image execution,
real-data MVPA, HPC, deferred adapters, RDM/report export, derivative export,
and publication remain experimental, external-runtime, or deferred. The
runtime `distances.tsv` is an RDM-ready pairwise-distance table, not an
exported RDM.
