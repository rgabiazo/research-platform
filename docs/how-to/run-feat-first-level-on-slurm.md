# Run First-Level FEAT on SLURM

> **Alpha support — Experimental or external-runtime.** Discovery, FSF
> rendering, runtime planning, and dry-run safety are tested. Execution requires
> user BOLD, EV, and confound inputs plus FSL or a configured container and
> site. It has not been validated against a live cluster for this release
> candidate.

This guide covers the reusable BIDS first-level FSL FEAT workflow on a generic SLURM cluster. It
keeps local data paths in project configuration, remote paths in local target configuration, and run
outputs under `artifacts/runs/<run-id>/`.

## Prerequisites and environment

- The project has `config/analysis.yaml` and a first-level model under `config/analysis/models/`.
- Preprocessed BOLD data, events, and confounds are available locally and remotely.
- The selected SLURM target can run the configured FSL backend.

Use synthetic labels while validating the workflow:

```bash
export PROJECT="<project>"
export MODEL="<model>"
export BATCH="synthetic-first-level"
export TASK="exampletask"
export SUBJECT="synthetic01"
export HPC_LOGIN_USER="<cluster-user>"
export HPC_LOGIN_HOST="<login-host>"

export LOCAL_BIDS_ROOT="/path/to/synthetic-bids"
export LOCAL_DERIV_ROOT="/path/to/synthetic-bids/derivatives/preprocessed"
export LOCAL_EVENTS_ROOT="/path/to/synthetic-bids/derivatives/events"

export REMOTE_BIDS_ROOT="<remote-bids-root>"
export REMOTE_DERIV_ROOT="<remote-derivative-root>"
export REMOTE_EVENTS_ROOT="<remote-events-root>"
export RP_REMOTE_WORKSPACE_ROOT="<remote-workspace-root>"
export RP_REMOTE_ARTIFACTS_ROOT="<remote-artifacts-root>"
export RP_REMOTE_CONTAINER_ROOT="<remote-container-root>"
```

Configure and activate a local target if needed:

```bash
rp hpc setup \
  --target target-a \
  --host "$HPC_LOGIN_HOST" \
  --user "$HPC_LOGIN_USER" \
  --remote-workspace-root "$RP_REMOTE_WORKSPACE_ROOT" \
  --remote-artifacts-root "$RP_REMOTE_ARTIFACTS_ROOT" \
  --remote-container-root "$RP_REMOTE_CONTAINER_ROOT"

rp hpc validate --target target-a
rp hpc target use target-a
```

`rp hpc setup` writes ignored local defaults and an SSH-profile template. It
makes no network call and proves neither credentials nor host, scheduler,
runtime, storage, or data readiness. The default provider-neutral `generic`
template assumes no provider, authentication method, module stack, scratch path, container
runtime, account, partition, or software version. Alliance SSH support remains
an optional `--template alliance` integration requiring site review, not
live-provider validation.

`rp hpc validate` writes nothing and contacts nothing. It validates local
target/profile structure, project overrides, remote-root syntax, placeholders,
and the required `atomic_no_replace` declaration. That policy is declared, not
remotely verified. Keep real hosts, users, optional identity paths, accounts,
partitions, module names, and container/cache roots in `secrets/hpc/`.

`rp hpc init` remains only a legacy, backward-compatible Alliance-oriented
local-default helper.

## Scaffold or inspect the project

For an fMRIPost-AROMA-backed input derivative:

```bash
rp project init bids-analysis \
  --project "$PROJECT" \
  --tool feat \
  --template fmripost-aroma-first-level \
  --study-root "$LOCAL_BIDS_ROOT" \
  --derivative-root "$LOCAL_DERIV_ROOT" \
  --events-root "$LOCAL_EVENTS_ROOT" \
  --confounds-root "$LOCAL_EVENTS_ROOT" \
  --remote-study-root "$REMOTE_BIDS_ROOT" \
  --remote-derivative-root "$REMOTE_DERIV_ROOT" \
  --remote-events-root "$REMOTE_EVENTS_ROOT" \
  --remote-confounds-root "$REMOTE_EVENTS_ROOT" \
  --task-id "$TASK"
```

Use `--template deepprep-t1w-first-level` when the configured BOLD input is a DeepPrep T1w-space
derivative. Adjust design settings through `rp analysis design configure first-level` or project
configuration, not one-off edits to generated FSFs.

After the overlay exists, repeat offline validation with its target override:

```bash
rp hpc validate --target target-a --project "$PROJECT"
```

Only after offline validation, the following **SSH-active** readiness check
immediately loads the configured profile and contacts its host. It may prompt
for host-key acceptance, authentication, or MFA, and has only mocked repository
evidence:

```bash
rp hpc doctor --project "$PROJECT"
```

## Validate and discover

```bash
rp config validate --project "$PROJECT"
rp analysis model validate "$MODEL" --project "$PROJECT"

rp batch discover analysis bids \
  --project "$PROJECT" \
  --stage first_level \
  --batch "$BATCH" \
  --task-id "$TASK"
```

Review the selected manifest before planning or syncing data.

## Run a local preflight

Render the run plan and FSFs without submitting:

```bash
export PLAN_RUN_ID="feat-preflight"

rp run plan analysis bids \
  --project "$PROJECT" \
  --stage first_level \
  --model "$MODEL" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --output-desc modelA \
  --run-id "$PLAN_RUN_ID"

python pipelines/analysis-bids/scripts/run_bids_analysis.py \
  --run-manifest "artifacts/runs/$PLAN_RUN_ID/run-manifest.yaml" \
  --plan-only
```

Inspect the filtered batch and a representative synthetic FSF:

```bash
cat "artifacts/runs/$PLAN_RUN_ID/inputs/$BATCH-filtered.tsv"

FSF="artifacts/runs/$PLAN_RUN_ID/outputs/fsf/sub-synthetic01/ses-01/sub-synthetic01_ses-01_task-exampletask_run-01_desc-modelA.fsf"
grep -n 'evs_real\|evtitle1\|shape1\|convolve1\|custom1' "$FSF"
```

Check TR, volume count, confounds, EV ordering, empty-EV policy, contrasts, and registration settings.

## Sync and verify remote inputs

```bash
rp hpc sync workspace
rp hpc sync project --project "$PROJECT"
rp hpc sync data --project "$PROJECT" --batch "$BATCH" --selected-only
```

Review every rendered source, destination, exclusion, and local plan path. Then repeat the same commands
with explicit execution authorization:

```bash
rp hpc sync workspace --execute
rp hpc sync project --project "$PROJECT" --execute
rp hpc sync data --project "$PROJECT" --batch "$BATCH" --selected-only --execute
```

The non-execute forms may write reported local plans, manifests, or status
files. They do not validate credentials or remote infrastructure. The
`--execute` forms cross the external rsync/SSH boundary.

**SSH-active read-only verification:** this command immediately connects to
the configured host and checks remote paths. It modifies no remote data, but
still requires credentials, connectivity, and the external site:

```bash
rp hpc verify data \
  --project "$PROJECT" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --task-id "$TASK"
```

Proceed only after verification reports every expected remote input.

## Render a SLURM bundle

```bash
export CHECK_ID="feat-slurm-check"

rp run slurm analysis bids \
  --project "$PROJECT" \
  --stage first_level \
  --model "$MODEL" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --output-desc modelA \
  --run-id "$CHECK_ID"

rp hpc stage --run-id "$CHECK_ID"
```

Inspect `artifacts/runs/$CHECK_ID/submit.sbatch` and the staged filtered batch. Target-provided account,
partition, QoS, constraint, and node directives should appear only when configured. Sites that reject
per-job memory directives can set `compute.slurm.omit_mem_directive: true` in the relevant project or
local target configuration.

## Submit and monitor

```bash
export RUN_ID="feat-${TASK}-synthetic01"

rp run submit analysis bids \
  --project "$PROJECT" \
  --stage first_level \
  --model "$MODEL" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --output-desc modelA \
  --run-id "$RUN_ID"
```

Inspect the rendered stage and submission plans and the local files reported by the command. Once they
match the reviewed SLURM bundle, repeat the submission with explicit authorization:

```bash
rp run submit analysis bids \
  --project "$PROJECT" \
  --stage first_level \
  --model "$MODEL" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --output-desc modelA \
  --run-id "$RUN_ID" \
  --execute
```

After explicit submission, use the platform status command to monitor the
recorded local state without invoking a subprocess:

```bash
rp hpc status --run-id "$RUN_ID"
```

That result does not prove scheduler state. To authorize an immediate SSH
connection and one `squeue` query, use:

```bash
rp hpc status --run-id "$RUN_ID" --live
```

Inspect the returned scheduler payload, including `checked` and `ok`. Empty
`squeue` output is reported as `not-found-or-completed`; it is ambiguous, not
proof of completion or success. The command does not run `sacct` or reconcile
terminal accounting.

For advanced scheduler-native troubleshooting, an operator can inspect the
queue and remote log directly:

```bash
ssh "$HPC_LOGIN_USER@$HPC_LOGIN_HOST" "squeue -j <job-id>"
ssh "$HPC_LOGIN_USER@$HPC_LOGIN_HOST" \
  "tail -n 200 $RP_REMOTE_ARTIFACTS_ROOT/runs/$RUN_ID/logs/slurm.err"
```

The expected order is the analysis planner, optional container preparation, and
one first-level unit per selected BIDS run. Advanced operators can use `squeue`
and a separately invoked site accounting command such as `sacct` for
scheduler-native child-job checks when needed. These direct SSH commands are
optional operator examples requiring site review; `rp hpc status --live` does
not run `sacct` automatically.

## Retrieve outputs

After independently establishing terminal scheduler success, review and
authorize retrieval:

```bash
rp hpc pull --run-id "$RUN_ID" --execute
```

Pulled outputs land under `artifacts/runs/$RUN_ID/hpc/pulled/outputs/`. Publishing them into a
canonical derivative remains a separate, explicit action.

The executed pull uses merge-oriented `rsync -az`. It does not establish
scheduler success, atomically promote a complete result, create an end-to-end
digest receipt, prevent prior destination content from being merged or
replaced, or guarantee interrupted-transfer recovery. Retrieve into an
appropriate private destination, preserve failed local and remote evidence,
and inspect the files. Use a new run ID when collision policy requires it;
never delete claims or recovery material merely to force a rerun.
