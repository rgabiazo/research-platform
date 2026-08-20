# Bring your own data

This guide is the public entry point for using Research Platform with data that
you own or are authorized to process. Start with a source checkout, keep the
project overlay and inputs private, validate paths and configuration, and then
choose one of the currently runnable local routes:

- an explicitly configured tabular model;
- one-entity coordinate-sphere ROI construction plus generic 3D-NIfTI
  extraction;
- exact-unit materialized-pattern crossnobis.

The checked-in projects and datasets are deterministic synthetic examples.
They are evidence for the software paths; they are not templates for the
scientific appropriateness of a real analysis. Configuration validation checks
software contracts, not study design, data quality, anatomical validity, or
statistical suitability.

## 1. Install from a source checkout

The public-alpha installation contract is a source checkout. Research Platform
packages are not currently available from PyPI. Use Git, Bash, and Python 3.11
or 3.12 with `venv` and `pip` support. CI verifies Ubuntu 24.04 x86_64 with
Python 3.11 and 3.12, and macOS 15 ARM64 with Python 3.12. Python 3.13 and newer
are outside the verified contract.

Clone the repository URL supplied by the project, preview the bootstrap, create
the default isolated environment, and activate it:

```bash
read -r -p "Repository URL: " RESEARCH_PLATFORM_REPOSITORY_URL
git clone "$RESEARCH_PLATFORM_REPOSITORY_URL" research-platform
cd research-platform
bash ops/envs/dev/bootstrap.sh --print-plan --profile minimal
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
```

The bootstrap installation step may contact the configured Python package
index or internal mirror for third-party dependencies. Workspace packages are
installed editably from the checkout. `--print-plan` is non-mutating; the
validation, path-inspection, doctor, and planning commands later in this guide
do not install packages. The profiles are:

- `minimal`: the seven runtime packages and their declared dependencies; use
  it for the supported local command surfaces and synthetic examples;
- `dev`: `minimal` plus pytest for repository tests;
- `full`: `dev`, `research-viz`, optional pandas, rsatoolbox, XGBoost,
  notebook, workflow, and visualization dependencies. This profile has not yet
  completed the separate full release-candidate gate;
- `hpc`: a staged scheduler-runtime profile, not the local BYOD default.

Verify the environment without contacting a remote service:

```bash
python --version
python -m pip check
bash ops/envs/dev/smoke-check.sh
rp --version
```

If the interpreter is not Python 3.11 or 3.12, select a supported interpreter
with `PYTHON_BIN` and create a new environment. If an import is missing, compare
the required profile above with `bash ops/envs/dev/bootstrap.sh --help`; do not
patch an unknown base or user environment. To keep the environment outside the
checkout, set `RP_DEV_VENV` consistently for bootstrap, activation, and the
smoke check:

```bash
TMP_BASE="${TMPDIR:-/tmp}"
export RP_DEV_VENV="$(mktemp -d "${TMP_BASE%/}/research-platform-venv.XXXXXX")"
bash ops/envs/dev/bootstrap.sh --profile minimal
source "$RP_DEV_VENV/bin/activate"
python -m pip check
bash ops/envs/dev/smoke-check.sh --venv "$RP_DEV_VENV"
```

Python source edits are visible through the editable installation. Rerun the
same bootstrap profile after dependency or package-metadata changes. To retire
an intentionally disposable environment, deactivate it, confirm that
`RP_DEV_VENV` still names the disposable directory you created, and remove
only that directory. Recreate an environment rather than trying to repair one
whose origin or installed packages are unknown.

## 2. Create a private overlay and external roots

Do not edit `project-example` or either pilot as your real project. Choose
private data and artifact roots first, then create a private-by-default
overlay:

```bash
read -r -p "Private datasets directory (outside checkout): " DATASETS_ROOT
read -r -p "Private artifacts directory (outside checkout): " ARTIFACTS_ROOT
export DATASETS_ROOT ARTIFACTS_ROOT
rp project init private-analysis
rp config validate --project private-analysis
```

The new overlay is `project/private-analysis/`. Only the four named public
overlays are allowlisted in this repository; other `project/*` overlays are
ignored. A real project should also have an explicit private repository or
equivalent access-controlled boundary. Verify the local Git rule without
creating a probe:

```bash
git check-ignore --no-index -v project/private-analysis/project.yaml
git check-ignore --no-index -v artifacts/runs/example/output.json
git check-ignore --no-index -v datasets/private-example/rawdata/input.tsv
```

These checks prove only the named rules. They do not mean that arbitrary files
under every `datasets/<name>/` directory are ignored. Paths outside the
checkout are outside this repository's tracking boundary.

Restore `DATASETS_ROOT`, `ARTIFACTS_ROOT`, and any named-root environment
variables in every fresh shell before validation, planning, or execution.
Changing an environment root changes path resolution; it is not a transparent
relocation of an existing reviewed plan.

### Tabular datasets and artifacts outside the checkout

Keep the same external roots set while scaffolding, validating, planning, or
running a tabular project:

```bash
rp project init tabular-model \
  --project private-tabular \
  --dataset private-tabular \
  --canonical-dataset private-tabular \
  --canonical-features-root tables \
  --batch main
```

Because this private dataset name is not mapped under `WORKSPACE.yaml`
`datasets`, the feature-table path for this example is derived exactly as:

```text
DATASETS_ROOT / canonical_dataset / canonical_features_root / feature_table
```

An explicit workspace dataset mapping takes precedence over `DATASETS_ROOT`.
Otherwise, `canonical_dataset` and `canonical_features_root` come from
`project/private-tabular/config/dataset.yaml`; `feature_table` comes from the
single selected row in `manifests/batches/main.tsv`. `ARTIFACTS_ROOT` owns run
control files and scientific outputs. Inspect the resolved paths before using
them:

```bash
rp config validate --project private-tabular
rp config paths --project private-tabular
```

### Named external analysis roots

ROI and MVPA inputs use `root_ref` plus a safe relative path. Register a
private input root in `project/<name>/config/analysis.yaml`:

```yaml
analysis:
  external_input_roots:
    private_inputs:
      label: private-inputs
      local_root: ${RP_PRIVATE_INPUT_ROOT}
      sync_enabled: false
```

Set and inspect the root before validation:

```bash
read -r -p "Private analysis-input directory: " RP_PRIVATE_INPUT_ROOT
export RP_PRIVATE_INPUT_ROOT
test -d "$RP_PRIVATE_INPUT_ROOT"
rp config validate --project private-analysis
rp config paths --project private-analysis
```

Environment expansion keeps a machine-specific absolute path out of tracked
YAML. Relative declarations follow the established workspace/project
resolution rules; use `rp config paths` to see the result rather than assuming
a base directory. `sync_enabled: false` is appropriate for local private input
unless you deliberately configure synchronization. Omitting `remote_root`
creates no remote capability. Registration resolves a name; it does not copy,
upload, or publish the data.

When named roots are configured, the read-only JSON response adds
`analysis_external_input_roots`, keyed by root name. Each entry reports
`label`, resolved `local_root`, `exists`, and `sync_enabled`; `remote_root`
appears only when declared. Existing path keys are unchanged, and projects
without named roots do not gain this key. The command does not create a missing
root or contact a remote destination. Treat its resolved-path output as
potentially sensitive.

Configuration can itself reveal identifiers and paths. Outputs, QC, logs, and
manifests may also contain identifiers or local path information. Keep them in
the private boundary. Review repository and staged state before any
publication:

```bash
git status --short
git diff --cached --name-only
git diff --cached
```

The overlay keeps ownership explicit:

- `config/dataset.yaml`, `config/preprocessing.yaml`, and
  `config/models.yaml` own tabular configuration;
- `manifests/batches/*.tsv` owns selected tabular rows and exact analysis
  units;
- `config/cohorts.yaml` defines reusable views without copying rows;
- `config/analysis/bundles/*.yaml` binds exact units to named components;
- `config/analysis/roi_sets`, `config/analysis/extraction_sets`, and
  `config/analysis/mvpa` hold domain configurations;
- `ARTIFACTS_ROOT/runs` and configured domain runtime roots own generated
  outputs, never the input directories.

Preserve canonical `subject_id`, optional `session_id`, `task_id`, and BIDS
`run_id` values exactly in manifests and source tables. Do not strip prefixes,
invent optional entities, or construct a Cartesian grid.

## 3. Choose a supported route

| Input you already have | Local route | Profile | Primary configuration |
| --- | --- | --- | --- |
| UTF-8 CSV, TSV, or tab-delimited TXT | Tabular preprocessing, binary logistic training, and evaluation | `minimal` | one-row batch plus `models.default.feature_columns` |
| One 3D reference NIfTI and one matching 3D value NIfTI | coordinate sphere plus generic-NIfTI extraction | `minimal` (including NumPy and nibabel through declared dependencies) | ROI and extraction YAML using one named root |
| ROI-final prepared vectors | materialized-pattern crossnobis | `minimal` | v1 pattern table, exact-unit batch/cohort/bundle, and MVPA YAML |

FSL/image MVPA and real-data neuroimaging pipelines remain experimental or
external-runtime. SPM is unsupported. RDM/report export is deferred. Local HPC
planning and rendering are testable, but remote upload, submission, monitoring,
cancellation, and retrieval have not been live-cluster validated.

## 4. Tabular route

Input may be `.csv`, `.tsv`, or `.txt`; `.tsv` and `.txt` are tab-delimited.
Use a UTF-8 header, unique column names, and stable row order. The table bytes
must not change between reviewed planning and execution.

Edit `manifests/batches/main.tsv` to contain exactly one data row:

```text
feature_table	target_column
features.tsv	binary_target
```

The public tabular commands have no row selector, automatic row iteration, or
Cartesian expansion. The referenced table may contain many records, but its
selected batch must contain one row. For the public logistic path, encode the
binary target as `0` and `1`.

In `config/models.yaml`, replace the example names with a reviewed, ordered,
nonempty predictor list:

```yaml
models:
  default:
    kind: logistic_regression
    feature_columns:
      - predictor_1
      - predictor_2
    learning_rate: 0.2
    iterations: 350
```

Every predictor must exist and contain finite numeric values. Do not include
identifiers, the selected target, alternate outcomes, group variables, or any
leakage-prone field. Predictor order is part of the scientific contract. The
public workflow performs no automatic categorical encoding, imputation,
multiclass conversion, table merging, row iteration, or Cartesian expansion.

Validate and inspect before creating a run:

```bash
rp config validate --project private-tabular
rp config paths --project private-tabular
rp batch show --project private-tabular --batch main
```

Use a distinct durable run ID for each operation. Review each plan, then
execute the same unchanged request once:

```bash
rp run local preprocess tabular \
  --project private-tabular \
  --batch main \
  --run-id private-tabular-preprocess-001 \
  --dry-run
rp run local preprocess tabular \
  --project private-tabular \
  --batch main \
  --run-id private-tabular-preprocess-001 \
  --execute

rp run local train model \
  --project private-tabular \
  --batch main \
  --run-id private-tabular-train-001 \
  --dry-run
rp run local train model \
  --project private-tabular \
  --batch main \
  --run-id private-tabular-train-001 \
  --execute

rp run local evaluate model \
  --project private-tabular \
  --batch main \
  --input-run private-tabular-train-001 \
  --run-id private-tabular-evaluate-001 \
  --dry-run
rp run local evaluate model \
  --project private-tabular \
  --batch main \
  --input-run private-tabular-train-001 \
  --run-id private-tabular-evaluate-001 \
  --execute
```

Training must succeed before evaluation can even be planned. Evaluation
accepts only an unchanged, digest-valid, successful local train transaction for
the same project and batch.

Successful `outputs/` inventories are:

- preprocess: `split.json`, `prep.json`, `features.tsv`, and
  `transaction-manifest.json`;
- train: those first three scientific files, `model.json`, and
  `transaction-manifest.json`;
- evaluate: `evaluation.json` and `transaction-manifest.json`.

Inspect `status.yaml`, logs, the run manifest, predictor provenance, scientific
JSON/TSV, and the transaction manifest. The receipt records exact output sizes
and SHA-256 values. Files become visible together only after staged validation
and atomic publication; `succeeded` means that publication committed.

A run ID is one-shot. A failed, running, completed, malformed, or claimed root
is never reused, and a changed source or configuration invalidates its reviewed
plan. Preserve the old root and choose a new ID. There is no retry, resume,
force, replacement, or overwrite option. If interruption leaves a staging path
or execution claim, treat it as recovery evidence: inspect and preserve it,
and do not delete it merely to force a rerun. If `outputs/` exists while status
is not durably `succeeded` or a claim remains, do not adopt it as a successful
run; downstream evaluation rejects that source. See
[Tabular slice](tabular-slice.md) for the complete integrity contract.

## 5. One-entity 3D-NIfTI ROI route

The runnable generic path builds one coordinate-sphere mask from one 3D
reference NIfTI and extracts values from one matching 3D NIfTI. Use the same
singular subject, task, and space in both configurations; session is optional
under the existing contract. It does not expand subject lists.

Register the input directory as `private_inputs`, then follow the complete
copyable YAML recipe in [ROI workflows](roi-workflows.md). That recipe uses
`root_ref: private_inputs` with relative image paths. Validate, preflight,
preview, and explicitly execute in this order:

```bash
rp config validate --project private-analysis
rp config paths --project private-analysis
rp analysis roi validate private-sphere --project private-analysis
rp analysis roi doctor private-sphere --project private-analysis
rp analysis roi build private-sphere --project private-analysis
rp analysis roi build private-sphere --project private-analysis --execute
rp analysis roi extraction validate private-values --project private-analysis
rp analysis roi extraction doctor private-values --project private-analysis
rp analysis roi extraction run private-values --project private-analysis
rp analysis roi extraction run private-values --project private-analysis --execute
```

The reference and value images require finite 4-by-4 affines. Sphere xyz
coordinates are world millimetres, radius is positive, and the generated mask
must be nonempty and in-grid. The value image must match the mask/reference
first-three-dimensional shape and affine within the documented absolute
tolerance. There is no automatic resampling, transform, or 4D time-series
support in this generic path.

Inspect mask NIfTI files and JSON sidecars, extraction value tables, QC tables,
and provenance. `voxel_count` is the mask size;
`valid_voxel_count` counts finite extracted values. Preserve prior outputs on
collision or failure and choose a new configured artifact destination. FSL,
ANTs, atlas execution, multi-subject expansion, anatomical validation, and SPM
are not implied by this route.

## 6. Materialized-pattern crossnobis route

Use this route only for ROI-final prepared vectors, not images awaiting a mask.
The normative producer schema is
[Materialized pattern table v1](materialized-pattern-table-v1.md). Create one
exact-unit batch row per real subject/task/run combination, a cohort view, an
analysis bundle, and an MVPA set whose source uses the private named root.

Run the bundle and MVPA lifecycle in order:

```bash
rp analysis bundle validate private-crossnobis --project private-analysis
rp analysis bundle doctor private-crossnobis --project private-analysis
rp analysis bundle plan private-crossnobis --project private-analysis
rp analysis mvpa validate private-crossnobis --project private-analysis
rp analysis mvpa doctor private-crossnobis \
  --project private-analysis \
  --bundle private-crossnobis
rp analysis mvpa plan private-crossnobis \
  --project private-analysis \
  --bundle private-crossnobis
rp analysis mvpa run private-crossnobis \
  --project private-analysis \
  --bundle private-crossnobis
rp analysis mvpa run private-crossnobis \
  --project private-analysis \
  --bundle private-crossnobis \
  --execute
```

Planning joins exact canonical unit identities, audits selected and unselected
rows, hashes the source table, and does not decode vectors. Execution rechecks
the digest, loads and validates vectors, computes the complete result in
memory, and publishes one all-or-nothing runtime tree. Review pattern/QC rows,
prepared rows, distances, summaries, provenance, and the successful-run
manifest. `distances.tsv` contains RDM-ready pairwise-distance data; it is not
an exported RDM.

The v1 collision policy is `existing_output: fail`. Preserve a successful or
failed destination and choose a new configured artifacts root for a deliberate
rerun. There is no overwrite, replacement, resume, retry, or cleanup flag. The
checked-in `project-example` `toy-crossnobis` workflow is synthetic evidence;
it is independent of user data and does not establish public real-data
validation.

## 7. Privacy, failures, and unsupported operations

- User data and generated artifacts remain private unless you deliberately
  publish them. Named-root registration alone performs no transfer.
- Never commit credentials, private identifiers, personal paths, or
  participant-derived fixtures. Review staged paths and content before any
  publication.
- A validation success establishes structural consistency, not scientific
  correctness. Read doctor checks, QC, warnings, provenance, and manifests.
- Preserve failed output roots, claims, staging directories, and logs as
  recovery evidence. Do not bypass collision controls by deleting evidence.
- Local tabular transactions and the local materialized-MVPA runtime have
  bounded publication guarantees. Do not project those guarantees onto remote
  execution.
- The current [HPC guide](bids-hpc-slice.md) documents plan-first surfaces.
  Remote operations require external credentials, site configuration, SSH,
  scheduler software, and later live-cluster validation.

The local ROI, bundle, and MVPA doctor commands do not contact remote systems.
Do not generalize that rule to the HPC command family: `rp hpc doctor`
immediately checks SSH connectivity, `rp hpc verify data` immediately contacts
the configured host to check remote paths over SSH when
verification paths exist, and `rp hpc status --live` immediately uses SSH to
run one `squeue` query. None of those three commands needs an `--execute` flag,
and repository coverage mocks their remote boundaries rather than validating a
live cluster. By contrast, `rp hpc status` without `--live` reads only recorded
local state and does not prove the scheduler's current or terminal state.

Use the [capability matrix](capabilities.md) as the final support boundary.
