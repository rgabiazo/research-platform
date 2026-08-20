# Run DeepPrep on SLURM

> **Alpha support — Experimental or external-runtime.** Adapter behavior,
> planning, and dry-run safety are tested. This guide requires user raw BIDS
> data, a FreeSurfer license, Nextflow, a container runtime, and a configured
> site. It has not been validated against a live cluster for this release
> candidate.

This guide covers the reusable raw-BIDS DeepPrep workflow on a generic SLURM cluster. Project and
connection details stay in project configuration and local, untracked files under `secrets/`.

## Prerequisites

- A valid raw BIDS dataset is available locally.
- A FreeSurfer license is available locally and at a configured remote path.
- Compute nodes can run Apptainer, either directly or through an optional site module.
- The project uses the `deepprep` preprocessing adapter.

Use neutral values while trying the workflow:

```bash
export PROJECT="<project>"
export BATCH="synthetic-deepprep"
export TASK="exampletask"
export SUBJECT="synthetic01"
export HPC_LOGIN_USER="<cluster-user>"
export HPC_LOGIN_HOST="<login-host>"

export LOCAL_BIDS_ROOT="/path/to/synthetic-bids"
export REMOTE_BIDS_ROOT="<remote-bids-root>"
export FS_LICENSE_FILE="secrets/freesurfer/license.txt"
export FS_LICENSE_REMOTE="<remote-license-path>"

export RP_REMOTE_WORKSPACE_ROOT="<remote-workspace-root>"
export RP_REMOTE_ARTIFACTS_ROOT="<remote-artifacts-root>"
export RP_REMOTE_CONTAINER_ROOT="<remote-container-root>"
```

## Configure a local target

Create the local starter configuration and keep it untracked:

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

`rp hpc setup` writes local defaults and an SSH-profile template under
`secrets/`. It makes no network call and does not test credentials, host
reachability, the scheduler, storage, runtime, or data. The default
provider-neutral `generic` template assumes no provider, authentication
method, module stack, scratch
path, container runtime, account, partition, or software version. The explicit
`--template alliance` form is an optional provider integration requiring site
review, not live-provider validation.

`rp hpc validate` writes nothing and contacts nothing. It verifies the local
target/profile structure, project override, remote-root syntax, placeholders,
and required `atomic_no_replace` policy declaration. That policy is declared,
not remotely verified. Site modules, accounts, partitions, container/cache
roots, and identity-file references belong in `secrets/hpc/` only when the
reviewed site requires them.

`rp hpc init` remains a legacy, backward-compatible Alliance-oriented
local-default helper; it is not the beginner entry point.

## Scaffold and configure the project

Create a raw-BIDS DeepPrep project when one does not already exist:

```bash
rp project init bids-preprocess \
  --project "$PROJECT" \
  --tool deepprep \
  --study-root "$LOCAL_BIDS_ROOT" \
  --remote-study-root "$REMOTE_BIDS_ROOT" \
  --task-id "$TASK"
```

After the overlay exists, repeat offline validation with its target override:

```bash
rp hpc validate --target target-a --project "$PROJECT"
```

Only after offline validation, the following **SSH-active** readiness check
immediately loads the selected profile and contacts its host. It may prompt for
host-key acceptance, authentication, or MFA, and has only mocked repository
evidence:

```bash
rp hpc doctor --project "$PROJECT"
```

Keep runtime choices in `project/$PROJECT/config/preprocessing.yaml`. A typical synthetic example
sets `bold_task_type: exampletask`, the desired output spaces, and the local and remote FreeSurfer
license paths. Keep container and Nextflow versions pinned in project compute configuration.

If compute nodes cannot pull images or the Nextflow launcher, pre-stage them from a node with network
access using the configured remote container and cache roots:

```bash
rp hpc container prepare --project "$PROJECT"
```

Review the rendered remote command and destination, then authorize the preparation explicitly:

```bash
rp hpc container prepare --project "$PROJECT" --execute
```

Do not add site module commands to the public root `WORKSPACE.yaml`. Put optional module and directory
setup in the selected local target or in the project compute config when it is genuinely project-wide.

## Choose resources

DeepPrep fan-out uses the project compute policy. For example:

```yaml
compute:
  policy:
    presets:
      deepprep:
        cpus: 4
        ram_gb: 32
        threads: 4
        n_jobs: 10
```

`cpus` and `ram_gb` are per-unit requests, `threads` controls each DeepPrep unit, and `n_jobs`
limits concurrent units. Start with a small `n_jobs` value and increase it only after a successful
trial run.

## Validate and discover

```bash
rp config validate --project "$PROJECT"

rp batch discover bids \
  --project "$PROJECT" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --task-id "$TASK"
```

Review `project/$PROJECT/manifests/batches/$BATCH.tsv` before continuing.

## Sync and verify

Plan transfers first, then execute the reviewed plan:

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

The non-execute forms may write reported local plan, manifest, or status
files; they do not prove remote reachability or readiness. The `--execute`
forms cross the external rsync/SSH boundary.

**SSH-active read-only verification:** the next command immediately contacts
the configured host and inspects remote roots and expected files. It does not
modify remote data, but it requires credentials and reachable infrastructure:

```bash
rp hpc verify data \
  --project "$PROJECT" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --task-id "$TASK"
```

Copy the FreeSurfer license separately to `FS_LICENSE_REMOTE`; never track the license.

## Render a SLURM plan

```bash
export CHECK_ID="deepprep-${TASK}-slurm-check"

rp run slurm preprocess bids \
  --project "$PROJECT" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --task-id "$TASK" \
  --run-id "$CHECK_ID"
```

Inspect `artifacts/runs/$CHECK_ID/submit.sbatch`, the filtered batch, requested resources, and the
generated Snakemake command before submission.

## Submit and monitor

Plan one explicitly synthetic submission first:

```bash
export RUN_ID="deepprep-${TASK}-synthetic01"

rp run submit preprocess bids \
  --project "$PROJECT" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --task-id "$TASK" \
  --run-id "$RUN_ID"
```

Inspect the rendered stage and submission plans and the local files reported by the command. Once they
match the reviewed SLURM bundle, repeat the submission with explicit authorization:

```bash
rp run submit preprocess bids \
  --project "$PROJECT" \
  --batch "$BATCH" \
  --subject-id "$SUBJECT" \
  --task-id "$TASK" \
  --run-id "$RUN_ID" \
  --execute
```

After that unit fans out successfully, omit `--subject-id` to submit all selected rows in the batch.
Use the platform status command to monitor the recorded controller and child jobs:

```bash
rp hpc status --run-id "$RUN_ID"
```

That form reads recorded local state only. It does not prove scheduler state.
To authorize an immediate SSH connection and one `squeue` query, use:

```bash
rp hpc status --run-id "$RUN_ID" --live
```

Inspect the returned scheduler payload, including `checked` and `ok`. Empty
`squeue` output is `not-found-or-completed`, an ambiguous value that is not
proof of completion or success. This command does not run `sacct` or reconcile
terminal accounting.

For advanced scheduler-native troubleshooting, an operator can inspect the
queue and remote log directly:

```bash
ssh "$HPC_LOGIN_USER@$HPC_LOGIN_HOST" "squeue -u $HPC_LOGIN_USER"
ssh "$HPC_LOGIN_USER@$HPC_LOGIN_HOST" \
  "tail -n 200 $RP_REMOTE_ARTIFACTS_ROOT/runs/$RUN_ID/logs/slurm.err"
```

Those direct SSH commands are optional advanced operator examples requiring
site review; they are not automatic `rp` behavior. If the job disappears from
`squeue`, use the site's accounting interface separately (for example,
`sacct` where available) before interpreting a terminal result.

## Retrieve outputs

After independently establishing terminal scheduler success, review and
authorize retrieval:

```bash
rp hpc pull --run-id "$RUN_ID" --execute
```

For a focused retrieval, add a relative `--subpath` under the run output directory. Pulled files land
under `artifacts/runs/$RUN_ID/hpc/pulled/`; publishing into a canonical derivative remains a separate,
explicit step.

The executed pull uses merge-oriented `rsync -az`. It does not establish
scheduler success, atomically promote a complete result, create an end-to-end
digest receipt, prevent prior destination content from being merged or
replaced, or guarantee interrupted-transfer recovery. Retrieve into an
appropriate private destination, preserve failed local and remote evidence,
and inspect the files. Use a new run ID when the established collision policy
requires one; do not delete claims or recovery material to force a rerun.
