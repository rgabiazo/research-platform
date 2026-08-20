# Minimal BIDS/HPC Slice

> **Alpha status — Plan/validation only for the public example.** Public
> configuration, manifest rendering, SLURM bundles, and dry-run safety are
> tested. The explicit execute paths are **Experimental or external-runtime**:
> the checked-in BIDS dataset has no executable imaging payload, and no live SSH
> host, scheduler, remote runtime, or cluster execution was validated for this
> release candidate.

This repository includes a safe orchestration slice for adapter-driven BIDS preprocessing and analysis. The
original concrete path is fMRIPost-AROMA; DeepPrep now uses the same submit, sync, verify, status, and pull flow
while consuming raw BIDS inputs directly.

- project overlay: `project/project-pilot-bids`
- pipeline: `pipelines/preprocess-bids`
- tools: `fmripost_aroma`, `deepprep`
- input derivative: required for derivative-backed tools such as `fmripost_aroma`; not required for raw-BIDS tools such as `deepprep`
- batch manifests: `project/project-pilot-bids/manifests/batches/*.tsv`

The slice uses a config-loaded adapter:

- `research-core` loads `preprocessing.tool_adapter` and stays orchestration-only
- `research-neuro` owns DeepPrep/fMRIPost-AROMA-specific validation, discovery, selectors, and runtime metadata
- `research-hpc` owns stage, pull, and submit execution behavior

The workspace CLI stays orchestration-only:

- `rp project init bids-preprocess`
- `rp config validate`
- `rp config show`
- `rp config paths`
- `rp batch list`
- `rp batch show`
- `rp batch discover bids`
- `rp run plan preprocess bids`
- `rp run local preprocess bids`
- `rp run slurm preprocess bids`
- `rp run submit preprocess bids [--execute] [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc setup` (canonical local starter)
- `rp hpc validate` (subprocess-free, write-free offline configuration validation)
- `rp hpc init` (legacy Alliance-oriented local-default helper)
- `rp hpc doctor` (immediate SSH connectivity check)
- `rp hpc sync workspace [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc sync project --project <project> [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc sync data --project <project> [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc verify data --project <project> [--batch <batch>] [--subject-id <subject>] [--session-id <session>] [--task-id <task>] [--run-id <row-run-id>] [--profile <profile>] [--role <role>] [--config <path>]` (immediately checks configured remote paths over SSH)
- `rp hpc stage --run-id <run_id> [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc stage --run-id <run_id> --execute [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc status --run-id <run_id>`
- `rp hpc status --run-id <run_id> --live` (immediately uses SSH for one `squeue` query)
- `rp hpc pull --run-id <run_id> [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc pull --run-id <run_id> --subpath <relative-subpath> --destination <local-destination> --execute [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc pull --run-id <run_id> --execute [--profile <profile>] [--role <role>] [--config <path>]`
- `rp hpc cancel --run-id <run_id>`

Install the public `rp` command from the source checkout using the canonical
[quickstart](onboarding/quickstart.md):

```bash
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
rp --help
```

## HPC command boundaries

- `rp hpc setup` is the canonical beginner entry point. It writes ignored
  local starter configuration under `secrets/` and makes no network call. It
  does not test credentials, host reachability, scheduler, runtime, storage, or
  data readiness. The default `generic` template records only the explicit
  host, user, target, and remote-root values and assumes no provider,
  authentication method, module stack, scratch path, container runtime,
  account, partition, or software version.
- `rp hpc validate` is the required next step. It invokes no subprocess,
  writes nothing, contacts nothing, and validates only local configuration.
  The required `promotion.mode: atomic_no_replace` value is a declared policy,
  not proof that the remote filesystem supports it.
- Alliance/MFA behavior remains available only through explicit
  `--template alliance` selection as an optional provider integration
  requiring site review. No provider has been live validated.
- `rp hpc init` remains only as a legacy, backward-compatible
  Alliance-oriented local-default helper.
- `rp hpc doctor` immediately loads the selected profile and checks SSH
  connectivity. Host-key acceptance, interactive authentication, or MFA may
  be required. The command has mocked boundary coverage, not live-cluster
  validation.
- `rp hpc verify data` is remotely read-only but immediately checks configured
  remote paths over SSH when verification paths exist.
- `rp hpc status` without `--live` reads recorded local manifest/status state
  only. `--live` immediately uses SSH for one `squeue` query; it never runs
  `sacct`, and `not-found-or-completed` is ambiguous rather than success.
- `rp hpc cancel` has no `--execute` form. It records a local cancellation
  request and may render `scancel`; it does not run the command or confirm
  cancellation.
- `rp hpc pull --execute` uses merge-oriented `rsync -az` without scheduler
  terminal-state proof, atomic publication, digest attestation, or guaranteed
  interrupted-transfer recovery.

Operational notes:

- `rp run plan ...` only writes the run manifest and status.
- `WORKSPACE.yaml` can provide scaffold-time HPC runtime presets for `compute.slurm.modules` and `compute.slurm.pre_activate_commands`, but the scaffolded project `config/compute.yaml` remains the runtime source of truth after project creation.
- `rp batch discover bids` delegates deterministic row discovery to the configured adapter and writes TSV rows with `subject_id`, `session_id`, `task_id`, and `run_id`.
- BIDS plans record adapter-selected row selectors in `selection`. Derivative-backed preprocessors discover derivative rows; raw-BIDS preprocessors discover rows from `participants.tsv` and `sub-*` folders.
- `rp run local ...` writes the same contract and can execute only when `--execute` is supplied.
- `rp run slurm ...` writes the run manifest and renders `submit.sbatch` but does not submit.
- `rp run submit preprocess bids` reuses the same manifest contract and renders local stage and submission plans without remote calls; `--execute` explicitly stages and submits them.
- The `--execute` form of `rp run submit preprocess bids` stages the bootstrap payload needed for repo bootstrap on a fresh remote, including `ops/` and `packages/research-hpc`, so it no longer requires a separate `rp hpc sync workspace` just to make bootstrap assets available.
- `compute.slurm.modules` and `compute.slurm.pre_activate_commands` remain the only runtime mechanism for remote environment setup in this slice.
- Planned SLURM manifests can record generic `hpc.connection` metadata for a named SSH profile; stage/pull/submit resolve that profile inside `research-hpc` instead of depending on `hpc.ssh_host` alone.
- `rp hpc bootstrap --run-id <run_id>` writes an opt-in remote bootstrap plan when the SLURM compute config declares `compute.slurm.bootstrap.enabled: true`.
- `rp hpc bootstrap --run-id <run_id> --execute` explicitly runs that bootstrap plan over SSH; it is never run automatically.
- Explicit bootstrap resolves `hpc.connection` first and falls back to legacy `hpc.ssh_host` only when no profile metadata is present.
- `rp hpc verify data` is remotely read-only, not local-only: when verification paths exist it immediately uses SSH to check configured remote roots and, for adapter-backed BIDS projects, the exact expected remote input files for selected or discovered rows. It requires credentials and reachability even though it has no `--execute` flag.
- Generic non-BIDS or adapter-less projects can opt into Layer A remote-root checks by declaring `project.yaml` `hpc.data_roots` entries with `local_path` and `remote_root`.
- `analysis bids` projects can also declare `analysis.external_input_roots` and reference them from `analysis.inputs.*.root_ref`; those roots are merged into the same generic `data_roots` model used by `rp hpc sync data`, `rp hpc verify data`, and SLURM run staging.
- Local validation remains lenient when an analysis external root's `remote_root` env var is unset, but `rp run slurm analysis bids` and `rp run submit analysis bids` fail early if a selected external analysis root outside the workspace has no remote destination.
- FEAT-backed `analysis bids` runs can now use `compute.tool_profiles.<profile>.{local,slurm}.execution_backend` with `native`, `apptainer`, or `singularity`.
- Newly scaffolded `rp project init bids-analysis --tool feat` projects now get a container-ready `compute.tool_profiles.fsl` by default, while cluster-specific module loading still stays in base `compute.slurm.modules` / `compute.slurm.pre_activate_commands` and optional `WORKSPACE.yaml hpc.runtime_defaults`.
- When the selected FSL runtime profile uses a container backend and `container.image` is a `docker://...` URI with `pull_mode: if_missing`, the FEAT runtime materializes the `.sif` automatically during unit execution and reuses it on later runs.
- Containerized FEAT units automatically identity-bind the workspace root, dataset root, derivative root, selected analysis external input roots, and run output root, so the normal HPC flow does not require hand-maintained bind lists or FEAT-specific path rewriting.
- Non-interactive FEAT execution is headless by default in the FSL layer, so HPC runs do not try to open a browser while still producing normal FEAT reports.
- Layer A still runs when unrelated project validation issues are present.
- If the selected or default batch is missing or empty, adapter-backed Layer B can fall back to adapter discovery.
- Layer B skips cleanly when adapter prerequisites such as a remote derivative root or resolvable local derivative inputs are missing.
- `rp hpc stage/pull/cancel` operate on a planned run and write local plan/status files under `artifacts/runs/<run_id>/hpc/`. Stage and pull require `--execute` for remote operations. Cancel has no `--execute` option: it records local `cancel-requested` state and may render a proposed `scancel` command, but it never runs `scancel` or confirms cancellation.
- `rp hpc status --run-id <run_id>` reports only the local status recorded for that run and invokes no subprocess. It does not prove scheduler state.
- `rp hpc status --run-id <run_id> --live` immediately loads the SSH profile, contacts the host, and runs one `squeue` query. It does not run `sacct`, reconcile terminal accounting, or prove output completeness. Empty `squeue` is reported as the deliberately ambiguous `not-found-or-completed`; inspect the scheduler payload's `checked` and `ok` values rather than treating a successful CLI invocation as job success.
- `rp hpc stage --execute` and `rp hpc pull --execute` make execution explicit and generic.
- `rp hpc stage`, `rp hpc pull`, and both plan and execute forms of `rp run submit preprocess bids` can use the same named SSH profile defaults as `rp hpc sync workspace` and `rp hpc sync data`.
- After `rp hpc setup`, the common pull path is `rp hpc pull --run-id <run_id> --execute`.
- If `rp hpc pull` omits both `--subpath` and `--destination`, it pulls the full remote run root into `artifacts/runs/<run_id>/hpc/pulled`.
- If `rp hpc pull` supplies `--subpath` but omits `--destination`, it pulls into `artifacts/runs/<run_id>/hpc/pulled/<subpath>`.
- `rp hpc pull --execute` performs merge-oriented `rsync -az`. It does not first prove scheduler success, atomically promote a complete result, provide an end-to-end digest receipt, prevent merging with prior destination content, or guarantee interrupted-transfer recovery.
- `rp hpc sync data` can sync project-declared external local BIDS roots only when the project config also declares explicit remote destinations.
- `rp hpc sync data` preserves overlapping configured data roots as distinct sync targets when a nested root also declares its own remote destination.
- Workspace-aware `rp` commands walk upward to find `WORKSPACE.yaml`, so you can invoke them from the repo root or any repository subdirectory.
- Runtime planning artifacts stay in `artifacts/runs/<run_id>/outputs/`; adapter metadata chooses tool-specific plan and command-script filenames.
- Apptainer-backed preprocessing runs materialize shared container images through a one-time workflow-managed prep step before subject-level fan-out.
- Publish-back remains planning-only in this pass.

Beginner onboarding example:

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    rp project init bids-preprocess \
      --project project-demo-bids \
      --study-root /data/studies/demo-bids \
      --derivative-root /data/studies/demo-bids-derivatives/deepprep-bold \
      --tool fmripost_aroma \
      --remote-study-root /remote/studies/demo-bids \
      --remote-derivative-root /remote/studies/demo-bids/derivatives/deepprep-bold
```

Optional scaffold-time runtime preset example in `WORKSPACE.yaml`:

```yaml
hpc:
  runtime_defaults:
    default: generic-slurm
    catalog:
      generic-slurm:
        slurm:
          modules: []
          pre_activate_commands: []
```

This provider-neutral preset intentionally loads no module stack and defines no
site scratch convention. Add modules, activation commands, cache roots, and
temporary directories only in an optional site configuration after checking
that site's documentation.

That scaffold creates:

- `project/project-demo-bids/project.yaml`
- `project/project-demo-bids/config/dataset.yaml`
- `project/project-demo-bids/config/compute.yaml`
- `project/project-demo-bids/config/preprocessing.yaml`
- `project/project-demo-bids/manifests/batches/default.tsv`

## Plan-first HPC operator flow

The commands in this section map the intended operator sequence. Planning is
locally testable; execution requires external infrastructure and is not a
publicly verified end-to-end cluster walkthrough.

For a first run on a fresh remote, the beginner-facing sequence is:

1. Create or select local HPC defaults with `rp hpc setup`.
2. Run the local-only `rp hpc validate`; correct every error before continuing.
3. Deliberately cross the network boundary with the SSH-active
   `rp hpc doctor` only when credentials and host access are ready.
4. Sync workspace, project, and data as needed for your project.
5. Render the stage and submission plan with `rp run submit ...`.
6. Review it, then repeat with `--execute` to authorize remote staging and scheduler submission.
7. Inspect recorded local state with `rp hpc status`; use the SSH-active `--live` query only when you intend to contact the cluster.
8. Establish terminal state separately, then retrieve into a reviewed private destination with `rp hpc pull --execute` and inspect the returned files.

After local HPC setup, `rp hpc stage`, `rp hpc pull`, and `rp run submit preprocess bids` can usually omit
`--profile`, `--role`, and `--config`. Those defaults resolve from the selected local target, matching env
vars, and the default `secrets/hpc/ssh-profiles.yaml` lookup path.

Provider-neutral local starter:

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc setup \
      --target target-a \
      --user <ssh-user> \
      --host <login-host> \
      --remote-workspace-root /remote/workspace \
      --remote-artifacts-root /remote/workspace/artifacts \
      --ssh-config secrets/hpc/ssh-profiles.yaml \
      --targets-config secrets/hpc/targets.yaml \
      --set-default

env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    rp hpc validate --target target-a --project project-demo-bids

# SSH-active: contacts the selected host immediately.
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    rp hpc doctor --project project-demo-bids

env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc sync workspace

env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc sync project --project project-demo-bids

env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc sync data --project project-demo-bids
```

`rp hpc setup` writes ignored local defaults and an SSH-profile template. It
makes no network call and proves no credentials, reachability, scheduler,
storage, runtime, or data readiness. Generic setup does not inherit workspace
runtime defaults and writes no container root, module, account, or partition
unless the operator supplies it explicitly. Offline validation reports the
promotion policy as declared, not remotely verified; it does not prove
authentication, remote path existence, atomic promotion support, installed
software, quota, or data readiness.

Review the reported local plan files, sources, exclusions, and remote
destinations. Planning may write local manifests, plans, or status files, but
invokes no remote subprocess. Then repeat only the required commands with
explicit authorization:

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc sync workspace --execute

env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc sync project --project project-demo-bids --execute

env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc sync data --project project-demo-bids --execute
```

Use only the sync steps you need. The `--execute` form of `rp run submit preprocess bids` stages its own
run payload for a fresh remote, so `rp hpc sync workspace` is about broader workspace bootstrap rather than
a required submit pre-step for every run.

Minimal pull after setup:

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc pull --run-id demo-submit --execute
```

That command uses merge-oriented `rsync -az` to pull the full remote run
directory into `artifacts/runs/demo-submit/hpc/pulled`. It does not establish
terminal scheduler success or transactionally promote and attest a complete
result. If you
provide `--subpath` but omit `--destination`, the default local destination becomes
`artifacts/runs/demo-submit/hpc/pulled/<subpath>`.

Clearly labeled fMRIPost-AROMA example:

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp run submit preprocess bids --project project-demo-bids --discover --run-id aroma-demo
```

Inspect the rendered stage and submission plans and every reported local file. Then authorize the same
submission explicitly:

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp run submit preprocess bids --project project-demo-bids --discover --run-id aroma-demo --execute

env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc status --run-id aroma-demo
```

This status is recorded local state only. Establish terminal scheduler state
separately before retrieval; `rp hpc status --live` performs one immediate SSH
`squeue` query, but empty output is ambiguous and it does not run `sacct`.

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    rp hpc pull \
      --run-id aroma-demo \
      --subpath outputs/fmripost_aroma \
      --destination /tmp/rp-slice-downloads/fmripost_aroma \
      --execute
```

The high-level pull command still requires `--execute` before any transfer
happens. Preserve failed remote and local evidence, choose a reviewed private
destination, and inspect the retrieved inventory; retrying `rsync` is not a
transactional recovery guarantee.

Clearly labeled DeepPrep example:

```bash
rp project init bids-preprocess \
  --project project-demo-deepprep \
  --study-root "$STUDY_BIDS_ROOT" \
  --tool deepprep \
  --task-id exampletask \
  --remote-study-root "$STUDY_REMOTE_BIDS_ROOT"

rp batch discover bids \
  --project project-demo-deepprep \
  --batch exampletask \
  --subject-id 001 \
  --task-id exampletask
```

The next check is remotely read-only but immediately contacts the configured
host over SSH. Run it only when credentials, host-key policy, and any required
interactive authentication or MFA are ready:

```bash
rp hpc verify data \
  --project project-demo-deepprep \
  --batch exampletask \
  --subject-id 001
```

After reviewing the remote verification report, render the local submission
plan:

```bash
rp run submit preprocess bids \
  --project project-demo-deepprep \
  --batch exampletask \
  --subject-id 001 \
  --task-id exampletask \
  --run-id deepprep-demo-exampletask
```

Review the reported local plans, then repeat with explicit authorization:

```bash
rp run submit preprocess bids \
  --project project-demo-deepprep \
  --batch exampletask \
  --subject-id 001 \
  --task-id exampletask \
  --run-id deepprep-demo-exampletask \
  --execute
```

After separately establishing terminal scheduler state, retrieve into a
reviewed private destination. Retrieval is a merge-oriented transfer, not
transactional publication:

```bash
rp hpc pull \
  --run-id deepprep-demo-exampletask \
  --subpath outputs/deepprep_units/sub-001-exampletask \
  --execute
```

Local example:

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    RP_TEMPLATEFLOW_HOME=/tmp/rp-slice-templateflow \
    rp run local preprocess bids --run-id demo-local --execute
```

HPC/script-generation example:

```bash
env PATH="/tmp/research-platform-venv/bin:$PATH" \
    RESEARCH_PLATFORM_ROOT="$PWD" \
    ARTIFACTS_ROOT=/tmp/rp-slice-artifacts \
    RP_HPC_PROFILE=target-a \
    RESEARCH_HPC_SSH_CONFIG=secrets/hpc/ssh-profiles.yaml \
    RP_REMOTE_WORKSPACE_ROOT=/remote/workspace \
    RP_REMOTE_ARTIFACTS_ROOT=/remote/workspace/artifacts \
    rp run slurm preprocess bids --run-id demo-slurm
```

That SLURM generation step writes `artifacts/runs/demo-slurm/submit.sbatch` and records the planned runtime helper artifacts in the run manifest:

- `artifacts/runs/demo-slurm/outputs/fmripost-aroma-plan.json`
- `artifacts/runs/demo-slurm/outputs/run-fmripost-aroma.sh`

Command inventory, not a local verification script:

The test, validation, setup, and non-execute plan forms below are locally
testable, though some write reported local control files. Every `--execute`
form is external-runtime behavior. `rp hpc verify data` is also SSH-active even
without `--execute`. Do not run this block end to end as a beginner checklist;
use only reviewed commands appropriate to the configured site.

```bash
python3 packages/research-core/tests/unit/test_cli_slice.py
python3 packages/research-hpc/tests/unit/test_slice_helpers.py
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp config validate
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" rp project init bids-preprocess --project project-demo-bids --study-root /data/studies/demo-bids --derivative-root /data/studies/demo-bids-derivatives/deepprep-bold --tool fmripost_aroma
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp batch discover bids --batch demo_discovered
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp run plan preprocess bids --run-id demo-plan
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" rp hpc setup --target target-a --user <ssh-user> --host <login-host> --remote-workspace-root /remote/workspace --remote-artifacts-root /remote/workspace/artifacts --ssh-config secrets/hpc/ssh-profiles.yaml --targets-config secrets/hpc/targets.yaml --set-default
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" rp hpc validate --target target-a --project project-demo-bids
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc sync workspace
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc sync workspace --execute
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc sync project --project project-demo-bids
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc sync project --project project-demo-bids --execute
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts RP_REMOTE_WORKSPACE_ROOT=/remote/workspace RP_REMOTE_ARTIFACTS_ROOT=/remote/workspace/artifacts rp run submit preprocess bids --project project-demo-bids --discover --run-id demo-submit
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts RP_REMOTE_WORKSPACE_ROOT=/remote/workspace RP_REMOTE_ARTIFACTS_ROOT=/remote/workspace/artifacts rp run submit preprocess bids --project project-demo-bids --discover --run-id demo-submit --execute
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc sync data --project project-demo-bids
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc sync data --project project-demo-bids --execute
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc verify data --project project-demo-bids --profile target-a
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc status --run-id demo-submit
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc pull --run-id demo-submit --execute
env PATH="/tmp/research-platform-venv/bin:$PATH" RESEARCH_PLATFORM_ROOT="$PWD" ARTIFACTS_ROOT=/tmp/rp-slice-artifacts rp hpc pull --run-id demo-submit --subpath outputs/fmripost_aroma --destination /tmp/rp-slice-downloads/fmripost_aroma --execute
```
