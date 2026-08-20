# Quickstart

This alpha workflow installs the platform from a source checkout. Its PEP 440
distribution version is `0.1.0a1`; its future Git tag will be `v0.1.0a1`, and
its future GitHub release title will be `Research Platform 0.1.0 Alpha 1`.
The workspace packages come from this checkout and are not currently available
from PyPI.

## Prerequisites

Before starting, install:

- Git
- Bash
- Python 3.11 or 3.12 with `venv` and `pip` support

CI verifies Ubuntu 24.04 x86_64 with Python 3.11 and 3.12, and macOS 15
ARM64 with Python 3.12. Python 3.13 and newer are not currently part of the
verified public-alpha contract.

The first bootstrap must be able to reach a Python package index or a configured
internal package source or mirror for third-party dependencies. The repository
does not bundle those dependencies. Later checks can run offline once the
environment is populated.

Choose the smallest profile that fits the work:

- `minimal` installs the seven runtime packages and declared dependencies for
  the supported local command surfaces and synthetic examples;
- `dev` adds pytest for repository tests;
- `full` adds `research-viz` and optional pandas, rsatoolbox, XGBoost,
  notebook, workflow, and visualization dependencies, but remains subject to
  the later full release-candidate gate;
- `hpc` is the staged scheduler-runtime profile, not the local quickstart.

## Install the source checkout

Set `RESEARCH_PLATFORM_REPOSITORY_URL` to the repository URL you were given,
then clone the repository and enter its root:

```bash
read -r -p "Repository URL: " RESEARCH_PLATFORM_REPOSITORY_URL
git clone "$RESEARCH_PLATFORM_REPOSITORY_URL" research-platform
cd research-platform
```

Previewing the bootstrap is optional and makes no changes:

```bash
bash ops/envs/dev/bootstrap.sh --print-plan --profile minimal
```

Create the minimal editable environment and activate it:

```bash
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
```

The environment path defaults to `.venv`. Set `RP_DEV_VENV` before both the
bootstrap and smoke check if you need a different path, and activate that same
environment with `source "$RP_DEV_VENV/bin/activate"` instead of the default
activation command above.

## Verify the installation

Run the installation smoke check, then validate the public tabular overlay:

```bash
bash ops/envs/dev/smoke-check.sh
python -m pip check
rp --version
rp config validate --project project-pilot-tabular
```

The smoke check verifies imports and the installed public command entry points.
It does not contact a remote system.

Workspace packages are editable, so Python source changes are visible from the
same checkout. Rerun the same bootstrap profile after dependency or package-
metadata changes. If the interpreter is outside 3.11 or 3.12, or the
environment's origin is unknown, select a supported `PYTHON_BIN` and create a
fresh virtual environment rather than repairing the old one. After deactivating
an intentionally disposable environment, remove only the exact environment
directory you created.

## Inspect and plan the toy workflow

Inspect the deterministic synthetic batch:

```bash
rp batch show \
  --project project-pilot-tabular \
  --batch toy_binary_logreg
```

The selected batch row owns `feature_table` and `target_column`. Before
planning, review the ordered predictor list at
`project/project-pilot-tabular/config/models.yaml` under
`models.default.feature_columns`. Public `rp` tabular workflows do not infer
predictors: identifiers, targets, alternate outcomes, grouping variables, and
other leakage-prone columns must not appear in this list. Changing its order
changes the scientific model contract. The workflow validates the configured
predictors against the selected table before creating run output.

Render the preprocessing workflow as a local dry-run plan:

```bash
rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id quickstart-toy-preprocess \
  --dry-run
```

Planning atomically reserves this safe, single-segment run ID and writes
`run-manifest.yaml`, `execute.sh`, `status.yaml`, and supporting plan files under
`artifacts/runs/quickstart-toy-preprocess/`. Its state is `planned`; it does not
execute the workflow or contact a remote system. Review the reported command,
inputs, output paths, and execution script together. You can inspect the saved
local state without contacting a scheduler:

```bash
rp hpc status --run-id quickstart-toy-preprocess
```

When the plan is correct, repeat the same scientific request in local mode and
add explicit execution authorization:

```bash
rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id quickstart-toy-preprocess \
  --execute
```

This transition is allowed once only when the stored plan identity and reviewed
script still match the new request. It preserves the original plan creation
time. If project configuration, data, predictors, settings, resources, or the
script changed after review, plan again with a new run ID. Repeated plans and
running, failed, succeeded, submitted, malformed, or claimed roots are rejected
without replacement. There is no resume, retry, overwrite, replace, or force
option for a run.

The smoke check above is read-only. The plan command writes its manifest and
supporting files beneath `artifacts/`, but creates neither `outputs/` nor a
staging directory. An authorized local tabular run computes the complete output
set in one owned same-filesystem staging directory, validates it, writes
`outputs/transaction-manifest.json`, and atomically publishes the directory
without replacement. The final outputs become visible together, and
`state: succeeded` is written only after publication succeeds.

If a producer, validation, source-integrity check, or pre-promotion collision
fails, no final `outputs/` directory is published. Preserve a failed or
interrupted run for inspection, including its status, logs, and any staging or
stale sibling-claim recovery evidence, then use a new run ID. There is no
resume, retry, overwrite, replace, force, automatic adoption, or stale-claim
cleanup operation.

Evaluation is stricter than a path lookup: its `--input-run` must name a
successful local `train model` transaction for the same project and batch, with
an intact reviewed plan, exact train inventory, valid transaction manifest, and
unchanged recorded bytes. A planned, failed, remote-only, claimed, legacy, or
modified training run cannot be used. If the training transaction changes
after evaluation planning, preserve the rejected evaluation plan and choose a
new run ID. These local guarantees do not cover SLURM or remote execution. See
`docs/tabular-slice.md` for the complete contract.

Use the analysis package through its implemented module entry point, for
example:

```bash
python -m research_platform.analysis.cli --help
```

There is no separate analysis console alias.

## Keep private work outside the public overlay boundary

Exactly four public examples are checked in:

- `project/project-template`
- `project/project-example`
- `project/project-pilot-bids`
- `project/project-pilot-tabular`

All other `project/*` overlays are ignored by default. Keep real overlays in a
separate private repository or another explicit private boundary. Do not weaken
the root allowlist to admit a private overlay. Keep confidential datasets out of
Git, local credentials under ignored `secrets/`, and generated run outputs under
`artifacts/`.

Next, read `ARCHITECTURE.md`, `ROADMAP.md`, and the slice guide relevant to your
work. For private overlays and user-owned inputs, continue with the
[bring-your-own-data guide](../byod.md).
