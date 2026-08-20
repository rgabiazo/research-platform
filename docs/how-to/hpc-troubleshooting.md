# HPC Troubleshooting

> **Alpha support — Experimental or external-runtime.** These checks require a
> user-configured SSH target, scheduler, remote runtime, and data. No live
> cluster troubleshooting path was validated for this release candidate. Start
> with local recorded state from `rp hpc status --run-id <run-id>`. Use
> `--live` only when you intend to contact the configured host. Direct `ssh`,
> `squeue`, and `sacct` examples below are advanced operator diagnostics, not
> commands that `rp` runs automatically.

This page collects common checks for `rp hpc` and SLURM-backed runs.

## Remote Data Verification Fails

**SSH-active command:** `rp hpc verify data` immediately connects to the
configured host and checks configured remote roots and expected files. It is
remotely read-only, but it requires credentials, host reachability, and any
host-key or MFA interaction. Repository coverage mocks this boundary; it is
not live-cluster evidence.

Run:

```bash
rp hpc verify data \
  --project <project> \
  --batch <batch> \
  --subject-id 001 \
  --profile <profile> \
  --role login
```

If remote roots are missing, sync data:

```bash
rp hpc sync data \
  --project <project> \
  --profile <profile> \
  --role login \
  --execute
```

If a broad sync drops with `rsync: Broken pipe`, preserve its logs and inspect
both endpoints before deciding whether to retry. A repeated merge-oriented
rsync may skip matching files and continue copying, but it is not a
transaction, an integrity receipt, or guaranteed interrupted-transfer
recovery. If only one configured root is missing, review a targeted transfer
plan for that root before authorizing it.

## Recorded State Versus Live Scheduler State

This command reads only the local run manifest and recorded status and invokes
no subprocess:

```bash
rp hpc status --run-id <run-id>
```

It does not prove current or terminal scheduler state. Adding `--live`
immediately loads the configured SSH profile, connects over SSH, and runs one
`squeue` query:

```bash
rp hpc status --run-id <run-id> --live
```

Inspect the returned scheduler payload, including `checked` and `ok`. Empty
`squeue` output is reported as `not-found-or-completed`; that is intentionally
ambiguous and is not evidence of successful completion. `rp hpc status --live`
does not run `sacct` or reconcile terminal accounting.

## Job Is Pending

Check the queue:

```bash
ssh <user>@<login-host> "squeue -j <job-id>"
```

`ST=PD` with `(Priority)` means the scheduler is waiting for priority or resources. This is not a
run failure.

## Job Is Running

Check the controller log:

```bash
ssh <user>@<login-host> \
  "tail -n 200 <configured-remote-artifacts-root>/runs/<run-id>/logs/slurm.err"
```

For `analysis bids` FEAT runs, the controller usually submits child jobs. The controller log prints
their job ids and log paths.

## Snakemake Child Logs

A child job log path looks like:

```text
<remote-run-root>/work/.snakemake/slurm_logs/rule_<rule>/<unit-id>/<job-id>.log
```

For example:

```bash
ssh <user>@<login-host> \
  "tail -n 200 <configured-remote-artifacts-root>/runs/<run-id>/work/.snakemake/slurm_logs/rule_bids_analysis_unit/<unit-id>/<job-id>.log"
```

If the file is not present yet, the child job may still be pending.

## Check Final State

An advanced operator can separately run the site's accounting command after a
job disappears from `squeue`. For a SLURM site with `sacct`, for example:

```bash
ssh <user>@<login-host> \
  "sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS"
```

Successful jobs should show:

```text
COMPLETED 0:0
```

This is a manual operator check. The current `rp hpc status --live` command
does not run `sacct`, and scheduler availability and retention policy are
site-specific. This is a separate advanced operator command.

## Request Cancellation

`rp hpc cancel` records a local `cancel-requested` state and, when enough
metadata exists, renders a possible `scancel` command. It has no `--execute`
option, invokes no SSH or scheduler process, and does not cancel or confirm
cancellation of a remote job:

```bash
rp hpc cancel --run-id <run-id>
```

Preserve the recorded request and scheduler evidence. An advanced operator
must separately apply and verify any site-approved scheduler action. The
command does not run `scancel`; local `cancel-requested` state is not proof of
remote cancellation.

## Missing Run-Local Filtered Batch

Symptom:

```text
FileNotFoundError: .../artifacts/runs/<run-id>/inputs/<batch>-filtered.tsv
```

Cause: the run was planned with a subject filter, but the run-local `inputs/` file was not staged to
the remote run directory.

Check local staging:

```bash
rp hpc stage \
  --run-id <run-id> \
  --profile <profile> \
  --role login

find "artifacts/runs/<run-id>/hpc/stage" -maxdepth 3 -type f
```

Expected:

```text
artifacts/runs/<run-id>/hpc/stage/inputs/<batch>-filtered.tsv
```

If the file is missing, fix staging locally and submit a fresh run. Failed remote run directories can
usually be left in place because each submission uses a separate run id.

## Container Prep Takes A Long Time

For FEAT runs with Apptainer, the first run may spend time in `bids_analysis_container_prep` while
building or reusing the FSL image. This is normal on shared filesystems. Later runs should reuse the
same `.sif`.

If container preparation fails with temporary or lock state, preserve the
failed logs, remote paths, and lock evidence. Review site policy and use a new
configured temporary path or run identity where the established collision
policy requires it. Do not delete lock or recovery material merely to force a
retry.

If container preparation fails with a registry timeout from a compute node,
an operator may pre-stage a reviewed `.sif` from a site-approved host and set
`container.image` to a path beneath the configured remote container root with
`container.pull_mode: never`. Container versions, paths, and preparation hosts
are site-specific choices. `rp hpc container prepare --project <project>
--execute` crosses the remote boundary and runs the configured preparation
command; repository tests do not validate that operation on a live site.

If DeepPrep starts but fails while downloading Nextflow, the compute node likely cannot reach
`www.nextflow.io`. Keep the DeepPrep runtime profile configured with `nextflow.enabled: true`, then
rerun `rp hpc container prepare --project <project> --execute`. The prepare command stages both the
SIF and the configured Nextflow jar under the configured remote roots so submitted jobs can copy it into each run's
`WorkDir/nextflow` before DeepPrep starts.

If DeepPrep reaches `bold_get_bold_file_in_bids` and reports missing `sub-*` outputs, check the
selected raw BIDS inputs rather than the container. DeepPrep intersects matching BOLD runs with raw
T1w subjects during combined anatomical plus BOLD preprocessing, so selected data sync and verify
must include the subject's T1w files as well as the task BOLD files.

## Pull Outputs

Only after independently checking terminal scheduler state should an operator
review a retrieval plan and authorize it:

```bash
rp hpc pull \
  --run-id <run-id> \
  --profile <profile> \
  --role login \
  --execute
```

Pulled files land under:

```text
artifacts/runs/<run-id>/hpc/pulled/
```

`rp hpc pull --execute` performs merge-oriented `rsync -az`. It does not prove
that the scheduler job succeeded, query terminal scheduler accounting, or
publish a result atomically. The retrieved tree is not atomically promoted or
published. It does not produce
an end-to-end digest receipt, protect prior destination content from merging
or replacement, or guarantee interrupted-transfer recovery. Retrieve into an
appropriate private destination, preserve failed transfer evidence, and
inspect the resulting files before any separate publication step. Retrieval
does not automatically promote files into canonical data or derivatives.
