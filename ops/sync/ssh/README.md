# SSH profiles for user-guided and automation-safe workflows

This directory documents the preferred SSH profile pattern used by `research-hpc`.

## Recommended setup

Create a local config under `secrets/` so site-specific connection details stay untracked:

```bash
rp hpc setup \
  --target <target-name> \
  --host <login-host> \
  --user <ssh-user> \
  --remote-workspace-root <remote-workspace-root>

rp hpc validate --target <target-name>
```

The setup command defaults to a provider-neutral `generic` template. It
creates one named target without deleting unrelated targets or profiles. The
profile defaults to the target, the role defaults to `login`, and the artifacts
root defaults to `<remote-workspace-root>/artifacts`. It generates no provider,
MFA policy, robot or other additional role, multiplexing, module, scratch,
container, account, partition, or software assumption.

Setup makes no network call. `rp hpc validate` is the required next step and is
subprocess-free, write-free, and network-free. It checks the local target,
profile, remote-root, scheduler-field, placeholder, and promotion-policy
contracts. The required `atomic_no_replace` promotion mode is
**declared, not remotely verified**. Neither command proves credentials, host
reachability, authentication, scheduler authorization, remote paths, runtime
readiness, storage quota, data readiness, or remote promotion capability.

In a noninteractive shell, an omitted target, host, user, or absolute workspace
root fails before any file is written. Existing selected target/profile
entries are collision-protected; `--force` replaces only those selected
entries and preserves unrelated configuration. Both configuration destinations
for high-level `rp hpc setup` must remain beneath `secrets/`. On POSIX it
requires `0700` for newly created private directories and `0600` for private
configuration files. It rejects public scaffold exceptions, symlink,
non-regular, and hard-linked destinations and fails before content mutation if
the private file mode cannot be established and verified.

`rp hpc init` remains a legacy, backward-compatible Alliance-oriented
local-default helper; use `rp hpc setup` for beginner onboarding.

If you need a lower-level starting point, either copy
`ops/sync/ssh/config.example` beneath `secrets/` or use:

```bash
research-hpc ssh init-config \
  --template generic \
  --profile <profile-name> \
  --host <login-host> \
  --user <ssh-user> \
  --output secrets/hpc/ssh-profiles.yaml
```

The lower-level command also defaults to `generic`. Replace every example
placeholder before validation.
Its caller-selected `--output` is not confined to `secrets/` and does not
provide the high-level repository-placement boundary. Its private writer does
reject symlink, special-file, and hard-linked destinations; on POSIX it creates
new private parent directories with mode `0700` and creates or secures the
output file with mode `0600` before writing. Keep the caller-selected location
private and untracked.

Offline validation never reads identity-file or known-hosts-file contents. For
declared references it checks controls and unresolved placeholders, but not
path existence or regular-file type. Confirm those properties privately before
an SSH-active command.

For multi-cluster work, also copy the target example into local secrets:

```bash
mkdir -p secrets/hpc
cp ops/sync/ssh/targets.example.yaml secrets/hpc/targets.yaml
```

Edit `secrets/hpc/targets.yaml` with real profile names and remote roots. Add
accounts, partitions, container roots, or other site settings only when they
are explicitly required and reviewed.
This file is local-only; project YAML should keep using environment placeholders such as
`${RP_REMOTE_WORKSPACE_ROOT:-}` and project-specific remote root variables.

## Recommended first-time commands

The canonical order is setup, offline validation, and only then an
explicitly SSH-active check:

```bash
rp hpc setup --target <target-name> --host <login-host> --user <ssh-user> --remote-workspace-root <remote-workspace-root>
rp hpc validate --target <target-name>
# SSH-active: contacts the configured host immediately.
rp hpc doctor --project <project-name>
```

You can inspect the locally configured profile without contacting its host:

```bash
research-hpc ssh list --config secrets/hpc/ssh-profiles.yaml
research-hpc ssh show --profile <profile-name> --role login --config secrets/hpc/ssh-profiles.yaml
```

The following lower-level connectivity check is **SSH-active**: it immediately
contacts the configured host and may require host-key acceptance,
authentication, or MFA. Repository tests mock this boundary:

```bash
research-hpc ssh check --profile <profile-name> --role login --config secrets/hpc/ssh-profiles.yaml --mode auto
```

For MFA-backed Alliance systems, select `--template alliance` explicitly. It
is an optional provider integration requiring site review, not evidence of
live provider compatibility. After offline validation, an operator may warm a
configured connection before multi-step sync, submit, or pull commands:

```bash
rp hpc connect
```

Start transfers in planning mode:

```bash
research-hpc rsync push \
  --profile <profile-name> \
  --role login \
  --config secrets/hpc/ssh-profiles.yaml \
  --source ./project \
  --destination <remote-workspace-root>/project \
  --mode interactive
```

Review the rendered command and destination, then repeat it with `--execute` to authorize rsync.

## Target defaults

`rp hpc target` selects a target without repeating connection and remote-root
flags. The `rp hpc cluster` spelling remains available for backward
compatibility:

```bash
rp hpc target list
rp hpc target show <target> --project <project>
rp hpc target use <target>
```

These target inspection commands are local. **SSH-active:** `rp hpc doctor
--project <project>` immediately loads the selected SSH profile and checks
connectivity; it may prompt for host-key acceptance, authentication, or MFA
and is not a local-only validation command.

After `rp hpc target use <target>`, existing commands resolve the target's SSH profile, role, remote
workspace/artifact roots, project-specific remote data roots, and optional SLURM site directives:

```bash
rp hpc sync workspace
rp hpc sync project --project <project>
rp hpc sync data --project <project>
```

Review the reported local plan files and remote destinations, then authorize the required transfers:

```bash
rp hpc sync workspace --execute
rp hpc sync project --project <project> --execute
rp hpc sync data --project <project> --execute
rp hpc connect
rp run submit analysis bids --project <project> --stage first_level --model <model> --batch <batch>
```

The final command renders the local stage and submission plans. Review those plans, then repeat it with
`--execute` to authorize remote staging and scheduler submission.

Non-execute planning may write reported local manifests, plans, or status
files. It does not test credentials or remote infrastructure. The sync
`--execute` forms and `rp hpc connect` cross the SSH/rsync boundary.

**SSH-active, remotely read-only:** when verification paths exist, this
command immediately connects to the configured host and inspects remote roots
and expected files:

```bash
rp hpc verify data --project <project> --batch <batch>
```

It requires credentials and reachable infrastructure even though it does not
modify remote data.

## Status, cancellation, and retrieval boundaries

- `rp hpc status --run-id <run-id>` reads only recorded local state and
  invokes no subprocess. It does not prove scheduler state.
- `rp hpc status --run-id <run-id> --live` immediately invokes SSH and one
  `squeue` query. Empty output is `not-found-or-completed`, which is ambiguous,
  not successful completion. Inspect the returned `checked` and `ok` fields.
  The command does not run `sacct` or reconcile terminal accounting.
- `rp hpc cancel --run-id <run-id>` has no `--execute` option. It records a
  local `cancel-requested` state and may render `scancel`; it invokes no SSH or
  scheduler subprocess and does not confirm cancellation.
- `rp hpc pull --run-id <run-id> --execute` performs merge-oriented
  `rsync -az`. It does not establish terminal scheduler success, atomically
  publish a complete result, provide an end-to-end digest receipt, protect
  existing destination content from merging or replacement, or guarantee
  interrupted-transfer recovery. It does not automatically promote retrieved
  files into canonical data or derivatives.

Operators should inspect terminal accounting separately where their site
provides it, retrieve into an appropriate private destination, preserve failed
local and remote evidence, and use a new run ID where established collision
policy requires it. Do not delete claims or recovery material simply to force
a rerun.

Command-line `--profile`, `--role`, and SSH `--config` arguments still override target-provided
connection defaults for one-off expert runs.

## Role guidance

- Generic setup creates only the `login` role.
- Add another role only for an explicitly reviewed provider or automation
  design with separately managed credentials.
- Do not reuse a personal interactive key as an automation credential unless
  your site explicitly supports that model.

## Notebook guidance

Notebook work should start from a login role and then move into a scheduler-backed interactive allocation. Treat the login node as an entry point, not as the place to run long-lived notebook kernels.

## Host-key guidance

`research-hpc ssh check` reports host-key mismatch guidance but does not modify `known_hosts`. If you see a stale or changed host key, verify it out-of-band and update your local `known_hosts` before retrying.

## Manual fallback for advanced users

If you prefer to work directly with native tools:

- use `research-hpc ssh print --profile <profile-name> --role <login|robot> --config <path>` to inspect the exact SSH command
- run raw `ssh` only after confirming the resolved target, user, and options
- use `research-hpc rsync push|pull ...` to plan without launching rsync, then repeat the reviewed command with `--execute`
- keep site-specific `~/.ssh/config`, `ssh`, `rsync`, `salloc`, and `jupyter` commands local rather than copying machine-specific values into tracked files
