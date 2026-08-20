# MVPA Crossnobis Workflows

> **Alpha status.** Local materialized-pattern crossnobis is **Runnable
> locally** through the checked-in `project-example` `toy-crossnobis` path.
> FSL/image execution, real-data MVPA, HPC integration, deferred adapters, and
> MVPA export/publication remain **Experimental or external-runtime**.

This page describes the reusable platform contract for MVPA/crossnobis
analysis. Selection and scientific choices remain configuration-owned. Private
or active-study overlays belong outside the public repository; do not weaken
the public overlay allowlist to store them here.

## Checked-in local walkthrough

The public `toy-crossnobis` example consumes the deterministically generated
ROI-final vectors in `datasets/ds-mvpa-example`. Its batch records exactly four
existing units—two invented subjects with two runs each, no artificial session
column—and its table contains 16 rows across two conditions and two ROIs.

From the repository root, direct runtime products to a fresh disposable
directory outside the checkout, using the operating system's temporary
directory and `/tmp` as a portable fallback. Then validate and inspect the
configuration, bundle, source digest, exact-unit plan, and complete output
preview before authorizing the final command:

```bash
TMP_BASE="${TMPDIR:-/tmp}"
export ARTIFACTS_ROOT="$(
  mktemp -d "${TMP_BASE%/}/research-platform-toy-mvpa.XXXXXX"
)"

rp analysis bundle validate toy-crossnobis \
  --project project-example

rp analysis bundle doctor toy-crossnobis \
  --project project-example

rp analysis bundle plan toy-crossnobis \
  --project project-example

rp analysis mvpa validate toy-crossnobis \
  --project project-example

rp analysis mvpa doctor toy-crossnobis \
  --project project-example \
  --bundle toy-crossnobis

rp analysis mvpa plan toy-crossnobis \
  --project project-example \
  --bundle toy-crossnobis

rp analysis mvpa run toy-crossnobis \
  --project project-example \
  --bundle toy-crossnobis

rp analysis mvpa run toy-crossnobis \
  --project project-example \
  --bundle toy-crossnobis \
  --execute
```

The first seven commands are non-mutating. `run --execute` alone authorizes
the failure-safe runtime transaction beneath `ARTIFACTS_ROOT`. A second run at
the same destination fails without modifying the first runtime tree. The
resulting `analysis/prepared-distances/distances.tsv` is an RDM-ready
pairwise-distance table, not an exported RDM. RDM table/figure export,
derivative export, and publication are separate advanced surfaces.

## New-project lifecycle

Start a dependency-light prepared-vector configuration, review its YAML, and
then use one named analysis bundle throughout readiness, planning, and runtime
preview:

```bash
rp analysis mvpa init <mvpa-set> \
  --project <project> \
  --template materialized-crossnobis

rp analysis mvpa validate <mvpa-set> \
  --project <project>

rp analysis mvpa doctor <mvpa-set> \
  --project <project> \
  --bundle <bundle>

rp analysis mvpa plan <mvpa-set> \
  --project <project> \
  --bundle <bundle>

rp analysis mvpa run <mvpa-set> \
  --project <project> \
  --bundle <bundle>

rp analysis mvpa run <mvpa-set> \
  --project <project> \
  --bundle <bundle> \
  --execute
```

The commands intentionally mean different things:

- `init` writes one editable
  `project/<project>/config/analysis/mvpa/<mvpa-set>.yaml` by default;
- `validate` checks only the configuration schema;
- `doctor` resolves the bundle and reports full execution readiness without
  numerical analysis, external tools, temporary paths, or writes;
- `plan` renders the deterministic unit, source, adapter, collision, provenance,
  and output plan without decoding feature or noise vectors;
- `run` without `--execute` previews the complete runtime and output contract
  without computation, temporary paths, or writes; and
- `run --execute` is the only authorization for local analysis output mutation.

`schema_valid`, `bundle_valid`, `plan_valid`,
`ready_for_materialization`, `ready_for_execution`, and `executed` are separate
states. A schema-valid scaffold is not execution-ready, and a planning error
returns a nonzero result rather than being hidden by schema validity.

### Initialization templates

`materialized-crossnobis` is the recommended template. It writes one
structurally valid exact-unit YAML with two neutral editable conditions, one
comparison, the fixed materialized-table v1 declaration, prepared-vector ROI
and feature-space placeholders, and explicit CV, centering, noise, threshold,
runtime-root, and collision defaults. It deliberately remains not ready until
the user supplies a real bundle, named source root, and pattern table.

Use `--dry-run` to preview the YAML without writing. Use `--force` only to
replace an existing scaffold YAML; it has no runtime overwrite meaning.

`fsl-feat-crossnobis` is an advanced external-input image template. It requires
coherent FEAT design metadata, parameter-estimate and noise images, ROI masks,
and the applicable optional runtime. The older `distance-rdm` template name is
retained only as a compatibility alias. Detailed condition, comparison, ROI,
component, engine, noise, and CV authoring options are advanced scaffold
overrides; the reviewed YAML remains the reproducible source of truth.

No template claims SPM, Nilearn, CIFTI, or BIDS-derivative execution support.

## Exact analysis units

The preferred selection source is an
[analysis bundle](decisions/ADR-0015-analysis-batches-cohorts-bundles.md). Its
resolved rows come from one batch manifest under
`project/<name>/manifests/batches/`; the MVPA lifecycle does not introduce
another unit manifest.

Each resolved row represents one combination that actually exists.
`subject_id` is required; `session_id`, `task_id`, and BIDS `run_id` are
optional. Cross-sectional rows do not need artificial dimensions. Irregular
longitudinal rows can contain different visits and run counts per subject. The
resolver preserves canonical stored identifiers, row order, exclusions, and
arbitrary deterministic metadata; neither the bundle nor MVPA planner creates
a Cartesian product.

An MVPA set declares its expected identity contract without copying rows:

```yaml
mvpa_set:
  unit_selection:
    mode: exact_units
    key_columns:
      - subject_id
      - session_id
      - run_id
```

Cross-sectional configurations can use only `subject_id`. Optional dimensions
belong in `key_columns` only when they distinguish real source rows.

The bundle passed through `--bundle` must resolve successfully and its
`components.mvpa_set` value must equal the requested MVPA set. The lifecycle
carries the included and excluded unit audit plus source-batch, effective
bundle, selection-plan, and MVPA-configuration digests into planning and
runtime provenance. It does not infer a bundle when one is omitted and does not
copy bundle rows into MVPA YAML.

The earlier inline subject, session, and run selectors remain only as the
advanced `legacy_cartesian` compatibility mode for existing FSL configurations
when no bundle is requested. Exact units and legacy selectors cannot be mixed.
The analysis-bundle command family itself remains plan-only and has no `run`
command.

## Pattern-source adapters and representations

`research-neuro` owns one explicit adapter registry. Every canonical pattern
row represents one exact unit and one condition. Shared fields carry:

- deterministic unit identity and canonical subject, optional session, task,
  and BIDS run identity;
- an explicit cross-validation label and condition identifier;
- source and backend names plus an `image` or `prepared_features`
  representation kind;
- a portable pattern reference and optional portable noise reference;
- optional event-count and QC/status information; and
- preserved unit metadata with backend-specific metadata isolated separately.

FSL EV indices, parameter-estimate numbers, design rows, cope mappings, and
FEAT conventions stay in FSL-specific metadata and compatibility details.

At execution, one representation-aware facade consumes the private handle from
the exact reviewed plan:

- `image` sources use the existing image and ROI-mask extraction path;
- `prepared_features` sources use the digest-checked materialized-table loader.

Both normalize into the existing backend-neutral preparation boundary.
Materialized rows explicitly preserve `cross_validation_label`; they are never
routed through fake `pe_image`, ROI-mask, or FEAT compatibility fields. Mixed
image and prepared-feature sources are rejected in v1. Unknown or deferred
representations are not execution-ready.

Private execution handles are retained only in memory and are excluded from
serialized plans, equality, CLI output, and portable provenance. Canonical
pattern rows use portable references. Advanced FSL compatibility details in a
run-local plan may still show resolved FEAT, design, image, mask, or event
paths; those details are execution audit material, not portable publication
references. Materialized execution uses the private handle and digest from the
exact plan; it does not replan the table or accept a replacement digest.

`fsl_feat_pe` remains an advanced external-runtime image source. Its discovery,
EV-to-PE mapping, extraction values, compatibility fields, and representation
dispatch remain scientifically unchanged. Local CLI execution is deliberately
not ready in this alpha because the image path does not yet provide the
portable feature-space, ROI-definition, and noise-scope identities required by
the successful-run manifest. Planning can therefore contain run-local host
paths, while no unsafe image run is committed.

These source schemas can be structurally valid but remain deferred:

- `bids_derivative_pattern_table`;
- `nilearn_glm`;
- `surface_cifti`.

There is no SPM adapter. A future adapter could map SPM beta-image metadata into
the same canonical pattern-row contract, but this release neither implements
nor claims SPM support.

## Materialized pattern table v1

The normative producer-facing field, identity, noise, and portability contract
is [Materialized Pattern Table v1](materialized-pattern-table-v1.md). The
summary below explains how that input participates in this lifecycle.

Use this exact source declaration:

```yaml
pattern_sources:
  - name: prepared_patterns
    backend: materialized_pattern_table
    root_ref: mvpa_inputs
    path: patterns.tsv
    schema_version: research_platform.neuro.mvpa.materialized_pattern_table.v1
```

The five fields are fixed in v1. Alternate delimiters, column mappings, globs,
shards, plugins, URI paths, and binary formats are not accepted.

Each TSV row is one final ROI vector for one exact unit and one condition. It
is not an instruction to load an image or apply a mask. The fixed contract
requires schema and pattern identity, exact-unit keys and canonical subject
identity, condition and source identity, ROI source and label, feature count
and values, `voxel_order`, `voxel_index_hash`, `feature_space_id`,
`roi_definition_id`, usability and status, explicit centering state and scope,
and explicit noise status and usability. Session, task, BIDS run, and
`cross_validation_label` remain optional when they are not part of the unit.

Rows join through the configured exact key tuple. The adapter preserves bundle
order and canonical values, never infers identity from filenames, and orders
selected rows by bundle unit, configured condition, ROI source, and ROI label.
Unselected table rows remain visible through audited counts and stable
identities but never enter analysis.

Planning streams scalar metadata, records the exact source-byte SHA-256,
portable reference, columns, and selected/unselected counts, and emits
`prepared_features` rows. It does not decode or retain feature or noise
vectors. The loader later rechecks that digest and either validates the entire
selected set or fails without returning a partial set.

Loaded feature vectors must be nonempty finite numeric arrays matching their
declared count, order, index hash, feature space, and ROI definition. The
loader does not reorder, center, normalize, truncate, pad, or repair them. A
supplied CV label must equal the label derived from the exact unit and configured
CV contract. Diagonal normalization requires finite positive variances with
matching width and identity plus explicit variance, estimation-scope, source,
and usability declarations. Identity normalization records noise as unused.
Usability, QC, exclusions, warnings, and errors remain auditable.

The checked-in `toy-crossnobis` walkthrough uses this exact contract. For real
analyses, users must generate and independently review their own ROI-final
vectors and metadata. See the [producer reference](materialized-pattern-table-v1.md)
and [ADR-0017](decisions/ADR-0017-materialized-mvpa-pattern-tables.md).

## Readiness and compute-before-write

Doctor reports stable checks for configuration and filename identity, bundle
existence and component matching, exact-unit resolution, adapter availability,
source roots and inputs, scalar table digest and coverage, condition and ROI
coverage, feature-space and CV identity, centering, thresholds, normalization
and noise requirements, external inputs where applicable, runtime-root safety,
collisions, and transaction support. It performs no numerical analysis and
writes nothing.

An authorized execution starts only when every required readiness state is
true. It then:

1. rechecks source digests;
2. materializes or extracts the complete pattern-row set;
3. requires usable, scientifically coherent QC;
4. prepares all pattern groups;
5. computes every requested distance;
6. computes all summaries; and
7. requires nonempty groups, adequate independent CV partitions, balanced
   requested-condition coverage, usable distance rows, and summary rows.

Only after every requested group succeeds does output staging begin. A fatal
failure in one group prevents all groups from being committed. The lifecycle
does not change the existing distance, comparison, extraction, or summary
mathematics.

## Runtime collision and transaction contract

The fixed v1 runtime policy is:

```yaml
mvpa_set:
  outputs:
    runtime_root:
      root_ref: artifact_root
      path: .research-platform/mvpa/{mvpa_set}
  runtime:
    existing_output: fail
```

`fail` is the default and only supported value. Any existing final runtime root
or planned destination fails before vector or image loading, numerical work,
or staging. A deliberate rerun uses a different configured artifacts path.
There is no overwrite, replace, resume, retry, or cleanup CLI flag.

Read-only preflight evaluates the complete destination set at once. It rejects
duplicates and lexical aliases, paths outside the configured runtime or named
root, existing files and directories, special files, symbolic-link targets or
parents, unusable parents, representation-name conflicts, and a final manifest
collision.

After computation succeeds, the lifecycle creates one unique hidden sibling
staging directory on the same filesystem as the final root. Existing
serializers write every file into that owned tree. The exact inventory, TSV
headers and row counts, JSON parseability, finite numeric values, provenance
relationships, output hashes, and portable path boundary are validated before
the complete directory is committed with one atomic rename when supported.

Ordinary computation, writer, serialization, validation, interruption,
promotion, or concurrent-destination failures leave the final root absent and
remove all owned temporary paths. If cleanup itself fails catastrophically,
the recovery tree is retained and its location is reported.

### Complete v1 output inventory

The fixed transaction inventory reserves four representation-specific
source-audit files. The image-dispatch inventory is tested, but no current
image adapter passes CLI execution readiness. That reserved image layout is:

```text
neuro/pattern-extraction/patterns.tsv
neuro/pattern-extraction/qc.tsv
neuro/pattern-extraction/provenance.json
neuro/pattern-extraction/vector_metadata.json
```

Prepared-feature sources use:

```text
neuro/pattern-materialization/patterns.tsv
neuro/pattern-materialization/qc.tsv
neuro/pattern-materialization/provenance.json
neuro/pattern-materialization/vector_metadata.json
```

Both representations then write the same nine downstream files:

```text
analysis/prepared-patterns/rows.tsv
analysis/prepared-patterns/qc.tsv
analysis/prepared-patterns/provenance.json
analysis/prepared-distances/distances.tsv
analysis/prepared-distances/qc.tsv
analysis/prepared-distances/provenance.json
analysis/prepared-summaries/summaries.tsv
analysis/prepared-summaries/qc.tsv
analysis/prepared-summaries/provenance.json
```

Those 13 source and analysis files are committed with the top-level marker:

```text
manifest.json
```

`analysis/prepared-distances/distances.tsv` is the runtime's RDM-ready
pairwise-distance table. Producing a separately exported RDM table or figure
still requires the advanced export surfaces and is not part of this local
walkthrough.

For the implemented materialized-table execution path, the manifest records
status `succeeded`, project and MVPA identity, MVPA and
bundle/batch/selection digests, planned and loaded source-table digest,
portable source reference, adapter and representation, selected and excluded unit
audits, CV, conditions and pairs, ROI and feature identities, centering, noise,
threshold and exclusion contracts, distance engine, row counts, warnings, an
empty successful error list, and the relative path and SHA-256 of every other
output.

The manifest does not hash itself because that would be circular, but staged
validation requires it in the exact inventory. It contains no timestamp,
absolute root, private handle, temporary path, or machine-specific value.
Identical materialized-table inputs written into independent clean roots
therefore produce byte-identical text and JSON files. No equivalent
cross-machine determinism claim is made for the deferred FSL/image CLI path.

## Portable and private references

Canonical plan rows and the successful manifest use dataset-relative or named
root references when the mapping is truthful. Runtime-resolved local paths can
remain in private in-memory handles and run-specific audit rows, but must not be
copied into portable provenance or the successful manifest. Published values
are recursively checked for embedded POSIX, Windows, UNC, tilde, and local
`file:` path references.

Project configuration should declare named roots and relative paths, not
workstation paths, user-home references, or environment-specific storage.

## Advanced and external surfaces

The following are advanced rather than beginner lifecycle guidance:

- `legacy_cartesian` selectors;
- FSL FEAT ingestion and manual smoke commands;
- detailed scaffold overrides;
- custom engines, normalization, noise, and CV choices;
- table, figure, RDM, derivative, and publication export command families;
- deferred adapters; and
- HPC integration.

Review command help and the relevant configuration before using these
surfaces. Export and publication commands remain separate from the local
runtime transaction described here.

## Layer boundaries

- `packages/research-analysis` owns reusable pattern preparation, distance,
  summary, table, figure, publication, RDM, and statistical logic.
- `packages/research-neuro` owns pattern-source adapters, representation
  materialization, image extraction, and the local runtime transaction.
- `packages/research-bids` owns BIDS naming and derivative-path helpers.
- `packages/research-core` owns thin CLI orchestration, configuration loading,
  exact bundle resolution, readiness, and plan presentation.
- `project/` overlays stay thin and contain reviewed configuration and exact
  unit manifests only.

## Public-alpha boundary

The project-level integration test exercises the checked-in synthetic table,
exact bundle handoff, 16 planned pattern rows, materialized loading, unchanged
native crossnobis mathematics, the complete 14-file transaction, portable
provenance, collision safety, and deterministic independent runs. That evidence
makes only the bounded local materialized-pattern path **Runnable locally** in
the [capability matrix](capabilities.md).

Image/FSL execution, real imaging inputs, HPC, deferred adapters, custom
scientific configurations, and all RDM/report/export/publication surfaces remain
**Experimental or external-runtime**. No SPM adapter is implemented or claimed.

Any proposal to broaden this boundary must follow the current
[capability matrix](capabilities.md) and the remote design and acceptance
requirements in [ADR-0022](decisions/ADR-0022-headline-hpc-execution-contract.md)
and [ADR-0023](decisions/ADR-0023-hpc-safety-primitives.md), where applicable.
Treat any study-specific MVPA material as private unless it has been
independently rewritten as a deterministic synthetic example.
