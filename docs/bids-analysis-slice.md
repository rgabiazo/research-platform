# BIDS Analysis Slice

> **Alpha status — Experimental or external-runtime.** FEAT discovery, FSF
> authoring, runtime planning, and orchestration have automated coverage. Actual
> execution requires user BOLD, EV, and confound inputs plus FSL or a configured
> container. There is no checked-in public executable FEAT project and no live
> cluster validation.

This document describes the phase 1 `analysis bids` slice.

## Scope

Phase 1 adds:

- `rp project init bids-analysis`
- `rp batch discover analysis bids`
- `rp run plan analysis bids`
- `rp run local analysis bids`
- `rp run slurm analysis bids`
- `rp run submit analysis bids`

Phase 1 is intentionally limited to:

- BIDS analysis only
- first-level FSL FEAT only
- local planning/execution and SLURM planning/submission
- the existing run-manifest, runtime-plan, Snakemake, stage, submit, and pull flow

Phase 1 does not include:

- subject-level FFX
- group FEAT
- featquery
- reusable ROI execution or ROI generation
- published-derivative fallback by default

## Boundaries

- `packages/research-core` owns the generic `analysis` action, bundle loading, manifest planning, and CLI surface.
- `packages/research-neuro` owns FEAT-specific selection, FSF rendering, validation, and runtime-plan assembly.
- `packages/research-hpc` continues to own generic SSH, rsync, manifest, and SLURM mechanics.
- `pipelines/analysis-bids` owns generic BIDS analysis orchestration.
- `project/<overlay>/config/analysis.yaml` owns thin analysis defaults, tool selection, input patterns, and stage defaults.
- `project/<overlay>/config/analysis/models/*.yaml` owns reusable GLM specs.

Reusable ROI workflows are documented separately in `docs/roi-workflows.md`.
The independent deterministic generic-NIfTI example now provides a checked-in
local coordinate-sphere build and extraction path, but it remains outside the
first-level FEAT slice and does not provide BOLD or FEAT execution. External
FSL/ANTs, real-data, and HPC-backed ROI paths remain experimental without
changing this FEAT boundary.

## Config shape

Phase 1 scaffolds:

- `config/analysis.yaml`
- `config/analysis/models/task_glm.yaml`
- `config/analysis/groupings/.gitkeep`

The scaffold keeps runtime/tool setup reusable by referencing:

- `analysis.tools.<tool>.runtime_profile`
- `compute.tool_profiles.<profile>`

For FEAT phase 1, the scaffold uses `runtime_profile: fsl`.

For FEAT scaffolds, `rp project init bids-analysis --tool feat` writes a
container-ready `compute.tool_profiles.fsl` by default:

- `compute.tool_profiles.fsl.local.execution_backend: native`
- `compute.tool_profiles.fsl.slurm.execution_backend: apptainer`
- `compute.tool_profiles.fsl.slurm.container.enabled: true`

Generated compatibility scaffolds may include provider-style path fallbacks.
Review them before SLURM execution. A provider-neutral public configuration
uses explicit environment-owned values such as:

- `container.image: ${RP_FSL_CONTAINER_IMAGE:-docker://vnmd/fsl_6.0.7.4:latest}`
- `container.image_name: ${RP_FSL_CONTAINER_IMAGE_NAME:-fsl_6.0.7.4.sif}`
- `container.image_root: ${RP_REMOTE_CONTAINER_ROOT}`

The scaffold intentionally leaves `compute.tool_profiles.fsl.slurm.modules` and
`compute.tool_profiles.fsl.slurm.pre_activate_commands` unset. In the current repo, that is the
safest way to preserve generic site/runtime setup through base `compute.slurm.modules`,
`compute.slurm.pre_activate_commands`, and optional `WORKSPACE.yaml hpc.runtime_defaults`.

The checked-in provider-neutral examples do not choose a module stack, scratch
environment variable, or node-local temporary path. Before SLURM execution,
configure `RP_REMOTE_CONTAINER_ROOT` and any Apptainer cache or temporary
directories for the target site. Optional provider profiles may supply those
values, but they must be reviewed rather than treated as workspace defaults.

## FEAT Scaffold Templates

`rp project init bids-analysis --tool feat` supports scaffold templates. The default is
`--template auto`, which keeps the generic FEAT scaffold unless the inputs clearly point to a known
tool profile.

For fMRIPost-AROMA first-level FEAT inputs, provide the EV root during project creation:

```bash
rp project init bids-analysis \
  --project project-demo-feat \
  --tool feat \
  --template auto \
  --study-root "$BIDS_ROOT" \
  --derivative-root "$AROMA_ROOT" \
  --events-root "$EV_ROOT" \
  --confounds-root "$EV_ROOT" \
  --remote-study-root "$REMOTE_BIDS_ROOT" \
  --remote-derivative-root "$REMOTE_AROMA_ROOT" \
  --remote-events-root "$REMOTE_EV_ROOT" \
  --remote-confounds-root "$REMOTE_EV_ROOT" \
  --task-id exampletask
```

When `--derivative-root` looks like fMRIPost-AROMA and `--events-root` or `--confounds-root` is
provided, `auto` selects `fmripost-aroma-first-level`. You can force that template explicitly:

```bash
rp project init bids-analysis \
  --project project-demo-feat \
  --tool feat \
  --template fmripost-aroma-first-level \
  --study-root "$BIDS_ROOT" \
  --derivative-root "$AROMA_ROOT" \
  --events-root "$EV_ROOT"
```

The fMRIPost-AROMA template generates:

- non-aggressive AROMA BOLD input patterns with a preprocessed-BOLD fallback
- external EV and confound roots under `analysis.external_input_roots`
- `*_desc-confounds_noGSR.txt` confound patterns
- `*_desc-<ev>_events.txt` EV patterns
- first-level settings suitable for pre-normalized derivative inputs

For DeepPrep BOLD inputs in T1w space, use the DeepPrep template:

```bash
rp project init bids-analysis \
  --project project-demo-feat-t1w \
  --tool feat \
  --template deepprep-t1w-first-level \
  --study-root "$BIDS_ROOT" \
  --derivative-root "$DEEPPREP_BOLD_ROOT" \
  --events-root "$EV_ROOT" \
  --confounds-root "$EV_ROOT" \
  --remote-study-root "$REMOTE_BIDS_ROOT" \
  --remote-derivative-root "$REMOTE_DEEPPREP_BOLD_ROOT" \
  --remote-events-root "$REMOTE_EV_ROOT" \
  --remote-confounds-root "$REMOTE_EV_ROOT"
```

The DeepPrep T1w template generates `*_space-T1w_desc-preproc_bold.nii.gz` BOLD patterns, external
EV/confound roots, noGSR confound patterns, and first-level settings for already-preprocessed inputs
with registration, motion correction, slice timing, BET, normalization, and smoothing off.

## Design configure

Generated first-level analysis projects can be adjusted without hand-editing YAML:

```bash
rp analysis design configure first-level \
  --project project-demo-feat-t1w \
  --bold-space T1w \
  --bold-desc preproc \
  --confounds-root "$EV_ROOT" \
  --confounds-pattern "desc-confounds_noGSR.txt" \
  --tr 1.0 \
  --hpf 100 \
  --smooth-mm 0 \
  --norm off \
  --motion-correction off \
  --slice-timing off \
  --bet off \
  --prewhiten on
```

The command writes the same `config/analysis.yaml` fields consumed by current FEAT first-level
runtime planning. Future BIDS analysis adapters can reuse the same config shape when they support
these first-level settings.

## Empty EV policy

First-level FEAT models describe the intended EVs for a task, but whether an EV has rows is a
per-run data property. FEAT analysis projects can declare how empty EV text files are handled under
the first-level validation block:

```yaml
analysis:
  stages:
    first_level:
      validation:
        empty_ev_policy: as_zero
```

Supported values:

- `as_zero`: an existing empty EV file is rendered as FEAT's `Empty (all zeros)` shape. The EV's
  temporal derivative is disabled for that run, and no custom EV file is attached for that EV.
- `fail`: an existing empty EV file is treated as a validation error.

Missing EV files remain controlled by `allow_missing_evs`; malformed non-empty EV files always fail
validation. A run command can override the project default for one plan or submission:

```bash
rp run plan analysis bids \
  --project project-demo-bids-feat \
  --stage first_level \
  --empty-ev-policy fail
```

The resolved policy is written into the run manifest so remote SLURM execution uses the same behavior
as local planning.

## Output descriptors

First-level outputs use the selected run's BIDS-like stem by default:

```text
sub-001_ses-01_task-exampletask_dir-AP_run-01.feat
```

Projects can optionally add a BIDS-style `desc` label to generated FEAT and FSF output names:

```yaml
analysis:
  stages:
    first_level:
      outputs:
        desc: modelA
```

This renders output stems such as:

```text
sub-001_ses-01_task-exampletask_dir-AP_run-01_desc-modelA.feat
sub-001_ses-01_task-exampletask_dir-AP_run-01_desc-modelA.fsf
```

The run commands also accept a one-off override:

```bash
rp run submit analysis bids \
  --project project-demo-bids-feat \
  --stage first_level \
  --output-desc modelA
```

That command renders the local stage and submission plans without contacting the remote system. Review
the reported files and commands, then repeat it with `--execute` to authorize remote staging and scheduler
submission.

The label must contain only letters and numbers, matching the usual BIDS label style. The resolved
descriptor is recorded in the run manifest so local planning and remote SLURM execution agree on
output names.

## Subject filters

BIDS run commands can filter an existing batch manifest without creating a separate project-level
subset TSV:

```bash
rp run plan analysis bids \
  --project project-demo-bids-feat \
  --stage first_level \
  --batch feat_first_level \
  --subject-id 001
```

For BIDS rows, shorthand labels such as `001` are matched the same way as full labels such as
`sub-001`. The filtered batch is written under the run directory, and the run manifest records the
source batch, source row count, filtered row count, and normalized filters. This keeps the original
project batch unchanged while preserving exactly what was submitted.

The same selector is available for HPC data verification. This remotely
read-only command immediately contacts the configured host over SSH; it is not
a local doctor or plan command and may require host-key acceptance,
authentication, or MFA:

```bash
rp hpc verify data \
  --project project-demo-bids-feat \
  --batch feat_first_level \
  --subject-id 001 \
  --profile <profile> \
  --role login
```

Inspect the returned remote check report; command completion alone does not
prove data or infrastructure readiness.

For SLURM-backed runs, the run-local filtered TSV is staged with the rest of the
run bundle. The non-execute form below makes no remote call, but it can write a
reported local stage plan and status under the run root. Its local stage plan
should show the file under `hpc/stage/inputs/`:

```bash
rp hpc stage \
  --run-id <run-id> \
  --profile <profile> \
  --role login

find "artifacts/runs/<run-id>/hpc/stage" -maxdepth 3 -type f
```

If the remote job reports a missing `artifacts/runs/<run-id>/inputs/*-filtered.tsv`, the run bundle
was not staged correctly and the job should be resubmitted after fixing staging.

## External analysis input roots

Phase 1.1 adds a generic analysis-scoped way to declare external input roots that live outside the
workspace and outside the main BIDS derivative root.

Use `analysis.external_input_roots` for reusable named roots:

```yaml
analysis:
  external_input_roots:
    evs:
      local_root: ${STUDY_EV_ROOT:-}
      remote_root: ${STUDY_REMOTE_EV_ROOT:-}
      sync_enabled: true

  inputs:
    evs:
      root_ref: evs
      patterns:
        - "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt"
```

Rules:

- `analysis.inputs.*.root_ref` takes precedence over a direct `root`.
- local validation and `rp run plan analysis bids` stay lenient when the external root's `remote_root` env var is unset.
- `rp run slurm analysis bids` and `rp run submit analysis bids` stay strict for selected external analysis roots that are outside the workspace and need a remote destination.
- external analysis roots participate in the generic HPC `data_roots` flow, so they can be verified, synced, and staged without FEAT-specific core logic.
- run manifests render execution-ready analysis root paths for SLURM, so remote FEAT planning does not rely on the original local absolute paths.

## Phase 2A FSL runtime backend

Phase 2A keeps `research-core` generic and adds a reusable FSL runtime backend in
`packages/research-neuro/src/research_platform/neuro/fsl/`.

The FEAT first-level runtime now supports:

- `execution_backend: native`
- `execution_backend: apptainer`
- `execution_backend: singularity`

Runtime config stays under the existing tool-profile contract. The following
is a reviewed provider-neutral example override, not a verbatim representation
of every generated compatibility scaffold:

```yaml
compute:
  tool_profiles:
    fsl:
      local:
        execution_backend: native
        environment:
          FSLOUTPUTTYPE: NIFTI_GZ

      slurm:
        execution_backend: apptainer
        environment:
          FSLOUTPUTTYPE: NIFTI_GZ
        container:
          enabled: true
          backend: apptainer
          image: ${RP_FSL_CONTAINER_IMAGE:-docker://vnmd/fsl_6.0.7.4:latest}
          pull_mode: if_missing
          image_name: ${RP_FSL_CONTAINER_IMAGE_NAME:-fsl_6.0.7.4.sif}
          image_root: ${RP_REMOTE_CONTAINER_ROOT}
```

Container behavior:

- if `container.image` is a prebuilt `.sif`, FEAT runs it directly
- if `container.image` is `docker://...` and `pull_mode: if_missing`, the runtime creates `image_root`, acquires a simple lock, pulls once to a temporary `.sif`, atomically renames it, and reuses it on later runs
- no separate manual `apptainer pull` step is required in the normal `rp run slurm analysis bids` / `rp run submit analysis bids` flow
- cluster-specific module loading still belongs in base `compute.slurm.modules` or scaffold-time
  `WORKSPACE.yaml hpc.runtime_defaults`, not in the FEAT tool profile

Current FEAT workflow order for Slurm-backed runs:

- `bids_analysis_plan`
- `bids_analysis_container_prep`
- `bids_analysis_unit` fan-out

This means FEAT unit jobs no longer materialize `docker://...` images inline. The one-time prep step
builds or reuses the shared `.sif`, writes `_container-ready.txt`, and only then allows per-run FEAT
jobs to start.

FEAT runtime outputs now preserve subject/session structure when those labels are available in the
selected batch rows. For example, a first-level run writes:

- `outputs/fsl_feat/sub-001/ses-01/sub-001_ses-01_task-exampletask_dir-AP_run-01.feat`
- `outputs/fsf/sub-001/ses-01/sub-001_ses-01_task-exampletask_dir-AP_run-01.fsf`

When a run has no session label, omit the `ses-*` segment rather than creating an empty directory.

## FEAT container troubleshooting

On shared HPC filesystems, the one-time Apptainer build can still feel slow because unpacking and SIF
creation are metadata-heavy and I/O-heavy. In practice this often looks like a long wait in
`bids_analysis_container_prep` even though FEAT itself has not started yet.

Advanced operator guidance, requiring review against the target site's
documentation and preserving failed evidence:

- first successful build wins: later FEAT runs reuse the existing `.sif`
- if a prep job fails, preserve its logs, temporary paths, lock paths, and
  scheduler evidence for diagnosis; do not delete or adopt them merely to
  force a rerun
- if the prep job is still making progress but is close to its walltime limit, increase prep runtime
  before increasing memory
- use more CPU only as a secondary tuning pass; the bottleneck is often shared-filesystem I/O rather
  than CPU saturation

When operators want to warm the image before launching a full FEAT run, an
out-of-band prebuild is a valid workflow. This optional operator example is
provider-neutral in its placeholders only: choose scheduler resources and make
Apptainer available using the target site's documented method before running
it. This direct scheduler/container example is not automatic `rp` behavior and
has no live-cluster validation.

```bash
salloc --cpus-per-task=2 --mem=32G --time=02:00:00

: "${RP_REMOTE_CONTAINER_ROOT:?Set RP_REMOTE_CONTAINER_ROOT for this site}"
TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${TMP_BASE%/}/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMP_BASE%/}/apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RP_REMOTE_CONTAINER_ROOT/fsl"
cd "$RP_REMOTE_CONTAINER_ROOT/fsl"
command -v apptainer >/dev/null || {
  echo "Make Apptainer available using this site's documented method." >&2
  exit 1
}
apptainer pull fsl_6.0.7.4.sif docker://vnmd/fsl_6.0.7.4:latest
apptainer exec "$RP_REMOTE_CONTAINER_ROOT/fsl/fsl_6.0.7.4.sif" fslversion
```

That manual warm step is optional. The normal `rp run ...` flow already performs the same container
prep inside `bids_analysis_container_prep`, but warming the image ahead of time avoids spending FEAT
run walltime on the first image build.

Automatic identity binds:

- workspace root
- dataset root
- derivative root
- selected analysis external input roots
- run output root

The binds use the same host and container path, so FEAT FSF files do not need path rewriting.

Non-interactive FEAT runs are headless by default:

- `FSL_FEAT_WATCH=0`
- `BROWSER=false`

This suppresses watcher/browser launch behavior while still generating the normal FEAT report files.

## Phase 2 model authoring

Phase 2 adds a user-friendly model authoring CLI under `rp analysis model` while keeping model YAML
files as the source of truth.

Phase 2 commands:

- `rp analysis model init <name>`
- `rp analysis model copy <source> <dest>`
- `rp analysis model show <name>`
- `rp analysis model validate <name>`
- `rp analysis model validate --all`

Phase 2 is intentionally limited to FEAT first-level model authoring only. It authors:

- `model.ev_order`
- `model.derivative_on`
- `model.nonconvolved`
- `model.contrasts`

Phase 2 does not move higher-level FEAT settings like `mode`, `z_thresh`, or `p_thresh` into the
model-authoring CLI.

Canonical non-interactive FEAT init:

```bash
rp analysis model init example_glm \
  --project project-example-feat \
  --ev-order condition_a condition_b condition_c instruction \
  --derivative-on condition_a condition_b condition_c \
  --nonconvolved instruction \
  --contrast "task_gt_baseline:1,1,1,0" \
  --contrast "condition_c_gt_others:-1,-1,2,0"
```

Interactive FEAT init:

```bash
rp analysis model init example_glm \
  --project project-example-feat \
  --interactive
```

Other model lifecycle helpers:

```bash
rp analysis model copy example_glm example_glm_v2 \
  --project project-example-feat

rp analysis model show example_glm \
  --project project-example-feat \
  --format summary

rp analysis model show example_glm \
  --project project-example-feat \
  --format yaml

rp analysis model validate example_glm \
  --project project-example-feat

rp analysis model validate --all \
  --project project-example-feat
```

Rules:

- `--tool` is optional when it can be inferred from `analysis.defaults.tool`.
- `--interactive` is explicit; the CLI does not auto-enter wizard mode.
- incomplete non-interactive FEAT init fails with a message that points users to `--interactive` or `--template`.
- model files live under `project/<overlay>/config/analysis/models/<name>.yaml`.
- the filename stem must match `model.name`.
- `rp run ... analysis bids --model <name>` remains unchanged and continues to resolve the YAML model by name.

## Planning example with external inputs

```bash
rp project init bids-analysis \
  --project project-demo-bids-feat \
  --study-root /data/studies/demo-bids \
  --derivative-root /data/studies/demo-bids/derivatives/fmriprep \
  --tool feat

rp batch discover analysis bids \
  --project project-demo-bids-feat \
  --stage first_level

rp run plan analysis bids \
  --project project-demo-bids-feat \
  --stage first_level \
  --run-id feat-plan

rp run slurm analysis bids \
  --project project-demo-bids-feat \
  --stage first_level \
  --run-id feat-slurm
```

## Outputs

- canonical data stays in `datasets/`
- ephemeral run outputs stay in `artifacts/runs/<run_id>/`
- phase 1 FEAT outputs are written under the run output root and are not published back by default

## See also

- `docs/how-to/run-feat-first-level-on-slurm.md`
- `docs/how-to/hpc-troubleshooting.md`
- `docs/decisions/ADR-0009-run-local-filtered-batches.md`
