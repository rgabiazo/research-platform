# project-example

> **Alpha status — Runnable locally.** Within bounded synthetic examples, the
> verified paths are the toy coordinate-sphere ROI build and extraction, local
> materialized-pattern crossnobis, and package-level toy-memory event mechanics.
> Bundle commands are plan/validation only; advanced imaging and external-tool
> paths remain experimental or external-runtime, and the empty notebook
> directories are scaffold only.

This is a valid thin project overlay backed only by deterministic public toy
data. Its root configuration demonstrates the current tabular project contract;
the sections below cover verified local toy ROI and materialized-pattern
crossnobis paths plus independent reusable event-configuration examples.

Start with the [source-checkout quickstart](../../docs/onboarding/quickstart.md)
for the primary tabular walkthrough, and check the
[capability matrix](../../docs/capabilities.md) before using any configuration
outside the bounded examples documented here.

It should answer:
- Which datasets are used?
- Which cohorts and splits are used?
- Which pipelines are used?
- Which model and visualization configs are used?
- Which compute profile is used?

Shared logic should remain in `packages/`.

Create a new overlay with `rp project init <name>` rather than copying this directory. Then validate it
with `rp config validate --project <name>` before adding project-specific configuration.

## Tabular predictor contract

For the root tabular example, the batch row owns `feature_table` and
`target_column`, while `config/models.yaml` owns the ordered
`models.default.feature_columns` list. Public `rp` workflows do not infer that
list. Identifiers, targets, alternate outcomes, grouping variables, and other
leakage-prone columns must be excluded; changing predictor order changes the
scientific model contract. Validation occurs before run output is created.

## Exact analysis units and plan-only bundles

The batch manifests are the only row stores for exact analysis units.
`manifests/batches/toy_roi_units.tsv` preserves the one exact subject, session,
and task combination supported by the ROI example; its absent `run_id` is not
invented. `manifests/batches/toy_mvpa_units.tsv` preserves four actual
subject/task/run combinations for the materialized MVPA example, with no
artificial session column. `config/cohorts.yaml` defines named views over those
batches, and the corresponding bundle YAML files reference the reviewed ROI or
MVPA component configurations without copying unit rows.

The bundle lifecycle is configuration-first and plan-only:

```bash
rp analysis bundle init <name> --project <project>
rp analysis bundle list --project project-example
rp analysis bundle show toy-roi --project project-example
rp analysis bundle validate toy-roi --project project-example
rp analysis bundle doctor toy-roi --project project-example
rp analysis bundle plan toy-roi --project project-example
```

The checked-in `toy-roi` and `toy-crossnobis` bundles already exist, so use the
generic `init` command only when starting a different bundle. Validation and
planning resolve the exact stored rows, preserve their additional metadata,
and check named component references. Bundle commands do not execute ROI or
MVPA components. The separate local lifecycles below are the executable
examples.

## Privacy boundary

This is one of four checked-in public overlays only. Real-study configuration
must live in a separate private repository or another explicit private
boundary, outside the public `project/` tree. Do not weaken the root project
allowlist.

## Local toy ROI example

This overlay contains a complete, dependency-light ROI example backed by the
deterministic synthetic images in `datasets/ds-roi-example`. The
`config/analysis/roi_sets/toy-spheres.yaml` configuration builds two
coordinate-sphere masks, and
`config/analysis/extraction_sets/toy-values.yaml` extracts values from the toy
value image. No scientific choices or input paths need to be supplied on the
command line.

From the repository root, validate readiness, review each no-write plan, and
then explicitly authorize the two local operations:

```bash
rp analysis roi validate toy-spheres --project project-example
rp analysis roi doctor toy-spheres --project project-example
rp analysis roi build toy-spheres --project project-example
rp analysis roi build toy-spheres --project project-example --execute

rp analysis roi extraction validate toy-values --project project-example
rp analysis roi extraction doctor toy-values --project project-example
rp analysis roi extraction run toy-values --project project-example
rp analysis roi extraction run toy-values --project project-example --execute
```

The checked-in YAML uses the named `roi_example` input root and sends runtime
outputs beneath `artifacts/project-example/toy-roi/`, which is ignored and is
not a canonical dataset derivative. Set `ARTIFACTS_ROOT` to a fresh disposable
directory before running the sequence when you want the outputs somewhere
else. The safe default collision policy makes a second execution into the same
destination fail before changing the first result. Treat the QC table as a
run-specific audit: it may record resolved local input and mask paths and must
not be promoted as a public derivative.

For a new project, begin from the generic scaffold rather than copying this
example:

```bash
rp analysis roi init <name> --project <project> --template coordinate_sphere
```

The generated YAML is the reproducible configuration contract. Review and
replace its placeholder image, coordinates, entities, and runtime output policy
for the new project before execution.

## Local toy materialized-MVPA example

The `toy-crossnobis` configuration is a complete local prepared-vector example
backed by the deterministic synthetic table in `datasets/ds-mvpa-example`. The
exact batch contains two invented subjects and two actual runs per subject,
with `subject_id`, `task_id`, and `run_id` as its identity columns and no
session column. The materialized table contains the corresponding 16 rows:
one exact unit × two conditions × two ROIs. These are ROI-final vectors,
not images awaiting masking.

From the repository root, point `ARTIFACTS_ROOT` at a fresh disposable
directory outside the checkout, using the operating system's temporary
directory and `/tmp` as a portable fallback. Validate and review the bundle
and MVPA plans, preview the no-write runtime, and only then authorize
execution:

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

The first seven commands are non-mutating. The final command writes one
failure-safe 14-file runtime tree beneath the disposable artifacts root. Its
`analysis/prepared-distances/distances.tsv` file is an RDM-ready pairwise-
distance table; it is not an exported RDM. RDM tables, figures, derivative
export, and publication remain separate advanced gates. A second execution
into the same configured destination fails before changing the first result.
Do not check runtime products into the repository.

For a new project, start from the generic configuration scaffold:

```bash
rp analysis mvpa init <name> \
  --project <project> \
  --template materialized-crossnobis
```

The generated YAML owns the configuration contract. Review its
bundle handoff, exact-unit keys, conditions, ROI and feature identities,
cross-validation, centering, noise, thresholds, source root, and runtime
policy. Those scientific choices do not belong in subject, run, condition,
ROI, feature, or noise selector flags.

## Events config

Project-specific event mappings live in `project/project-example/config/events/`.

- reusable BIDS logic belongs in `packages/research-bids`
- study mappings and sidecar metadata belong in project config
- staged previews/manifests belong in `artifacts/...`
- canonical published outputs belong in `datasets/...`

Advanced package-level fixture demonstration:

These commands exercise the lower-level `research-bids` package interface; they
are not the verified minimal-profile walkthrough. The explicit `pandas` backend
requires optional pandas support, and `datasets/...` is a placeholder that must
be replaced with an approved destination before publication.

```bash
python -m research_platform.bids.cli events plan --spec project/project-example/config/events/toy-memory.yaml --source packages/research-bids/tests/fixtures/toy-memory/raw/toy01_visit01_toymemory_2099-01-01.csv --artifact-root artifacts/project-example/events/toy-memory
python -m research_platform.bids.cli events build --spec project/project-example/config/events/toy-memory.yaml --source packages/research-bids/tests/fixtures/toy-memory/raw/toy01_visit01_toymemory_2099-01-01.csv --artifact-root artifacts/project-example/events/toy-memory --backend pandas
python -m research_platform.bids.cli events publish --dataset-root datasets/... --manifest artifacts/project-example/events/toy-memory/manifests/build-manifest.json
```

Version 2 example:

```bash
python -m research_platform.bids.cli events build --spec project/project-example/config/events/toy-memory.v2.yaml --source packages/research-bids/tests/fixtures/toy-memory/raw/toy01_visit01_toymemory_2099-01-01.csv --artifact-root artifacts/project-example/events/toy-memory-v2 --backend pandas
```

Current scope note:
- the reusable BIDS plan/build/publish mechanics live in `packages/research-bids`
- the synthetic `toy-memory` example has both a legacy compatibility config and an explicit generic `version: 2` config
- the same generic compiled-plan engine also powers a minimal non-recognition `simplecues.v2.yaml` example
- the same generic path also covers a broader `simplecues-plus.v2.yaml` proof with multiple runs, optional instruction rows, and row-local scoring

The toy-memory source and expected outputs are deterministically generated fixtures with
invented identifiers, dates, task and stimulus names, timings, conditions, and responses.
They contain no participant-derived or study-derived data.
