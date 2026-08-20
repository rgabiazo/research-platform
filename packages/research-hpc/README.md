# research-hpc

> **Alpha status — plan-first and provider-neutral.** Local configuration,
> offline validation, target inspection, command rendering, and non-execute
> planning are **Plan/validation only**. SSH-active checks, transfers,
> submission, live status, and retrieval are **Experimental or
> external-runtime** and have not been accepted against a live provider.

`research-hpc` owns reusable target/profile, SSH, rsync, SLURM, and remote
lifecycle mechanics. It does not establish credentials, provider readiness,
scientific validity, or a complete remote transaction merely by rendering a
command.

## Ownership and boundaries

This package owns:

- provider-neutral local target and SSH-profile configuration;
- offline target/profile validation and local inspection;
- SSH, rsync, and SLURM command construction;
- local planning for synchronization, staging, submission, status,
  cancellation requests, and retrieval;
- claims, receipts, quarantine, recovery, and no-overwrite publication
  mechanics where implemented;
- reusable remote-safety primitives where explicitly implemented.

It does not own:

- private credentials, accounts, partitions, hosts, or provider policy;
- scientific workflow meaning, BIDS or neuroimaging semantics, or project
  configuration;
- proof that a remote path, scheduler, runtime, dataset, or tool is ready;
- universal compatibility with arbitrary clusters;
- a complete, live-validated remote lifecycle or end-to-end receipt contract.

The top-level `rp hpc` surface is the guided workspace interface.
`research-hpc` exposes lower-level package commands for advanced inspection and
explicitly authorized operations.

## Install from a source checkout

The alpha packages are not published on PyPI. From the repository root, use a
supported Python 3.11 or 3.12 interpreter:

```bash
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
research-hpc --help
```

`research-hpc` declares no third-party runtime dependency. External commands
such as SSH, rsync, and scheduler tools are user- and site-supplied and are
used only by the surfaces that explicitly cross those boundaries.

## Provider-neutral first-time flow

Start with the high-level `rp hpc setup` command:

```bash
rp hpc setup \
  --target <target-name> \
  --host <login-host> \
  --user <ssh-user> \
  --remote-workspace-root <remote-workspace-root>
```

Setup writes ignored local defaults and an SSH-profile template beneath
`secrets/`. It makes no network call, tests no credential, and proves no host
reachability, scheduler behaviour, runtime availability, storage, or data
readiness. The default `generic` template assumes no provider, authentication
method, account, partition, module stack, scratch path, container runtime, or
software version.

The profile defaults to the target, the role defaults to `login`, and the
remote artifacts root defaults below the supplied remote workspace root.
Noninteractive generic setup requires the target, host, user, and absolute
remote workspace root before writing.

Next, run the distinct offline check:

```bash
rp hpc validate --target <target-name>
```

`rp hpc validate` is subprocess-free, write-free, and network-free. It checks
local target/profile linkage, placeholders, remote-root syntax, declared
scheduler structure, and `promotion.mode: atomic_no_replace`. That promotion
policy is **declared, not remotely verified**. Validation does not inspect
credentials, host reachability, authentication, scheduler/account
authorization, remote paths, runtime, storage, quota, data readiness, or
promotion capability.

The full offline form accepts explicit local selections:

```bash
rp hpc validate \
  --target <target-name> \
  --targets-config secrets/hpc/targets.yaml \
  --profile <profile-name> \
  --role login \
  --ssh-config secrets/hpc/ssh-profiles.yaml \
  --project <project-name> \
  --json
```

Omit `--project` to validate only the target/profile pair. Adding it checks the
existing project overlay and selected target override without discovery,
scientific execution, or external-tool checks.

Only after reviewing local validation should an operator cross the remote
boundary.

**SSH-active:** `rp hpc doctor` immediately loads the selected SSH profile and
checks connectivity. It may require host-key acceptance, authentication, or
MFA:

```bash
rp hpc doctor --project <project-name>
```

Repository tests mock that boundary; no live cluster is validated. Alliance is
an optional provider integration selected explicitly with
`--template alliance`. It requires site review and is not provider validation.
The legacy `rp hpc init` command remains only a backward-compatible,
Alliance-oriented helper and is not the beginner entry point.

## Local configuration safety

The high-level setup surface confines its configuration destinations beneath
`secrets/`. It rejects conflicting selected entries unless `--force` is
explicitly supplied; forced replacement is limited to those entries and
preserves unrelated configuration.

High-level `rp hpc setup` rejects destinations outside its private boundary,
existing public scaffold exceptions, symlink or non-regular files, and
hard-linked destinations. On POSIX systems, it creates private directories
with mode `0700` and private configuration files with mode `0600`. It fails
before content mutation if those modes cannot be established and verified.

The generic SSH profile needs only an explicit host and user:

```yaml
profiles:
  target-name:
    host: <replace-with-real-ssh-host>
    user: <replace-with-real-ssh-user>
```

Place real values only beneath ignored `secrets/`. Configuration records
private-key and known-hosts paths, never key contents. Offline validation never
reads identity-file or known-hosts-file contents and does not establish path
existence or regular-file type. Perform those checks privately before an
SSH-active operation.

## Lower-level package interface

For a lower-level provider-neutral SSH starter:

```bash
research-hpc ssh init-config \
  --template generic \
  --profile <profile-name> \
  --host <login-host> \
  --user <ssh-user> \
  --output secrets/hpc/ssh-profiles.yaml
```

The low-level command also defaults to `generic`. Unlike the high-level
surface, its caller-selected output is not confined to `secrets/`. Its private
writer rejects symlink, special-file, and hard-linked destinations; on POSIX it
creates private parents with mode `0700` and secures the output with mode
`0600` before writing. Prefer an untracked path beneath `secrets/`.

Inspect the resolved profile locally:

```bash
research-hpc ssh list --config secrets/hpc/ssh-profiles.yaml
research-hpc ssh show \
  --profile <profile-name> \
  --role login \
  --config secrets/hpc/ssh-profiles.yaml
```

**SSH-active:** the following command immediately checks the configured host
and may prompt for host-key, authentication, or MFA handling:

```bash
research-hpc ssh check \
  --profile <profile-name> \
  --role login \
  --config secrets/hpc/ssh-profiles.yaml \
  --mode auto
```

A lower-level rsync command renders a transfer by default:

```bash
research-hpc rsync push \
  --profile <profile-name> \
  --role login \
  --config secrets/hpc/ssh-profiles.yaml \
  --source ./project \
  --destination <remote-workspace-root>/project \
  --mode interactive
```

Review the rendered command before repeating an applicable transfer with
`--execute`. An execute form crosses the remote boundary.

## Planning and remote-contact boundaries

Project-aware planning remains under `rp hpc`:

```bash
rp hpc sync workspace \
  --profile <profile-name> \
  --role login \
  --config secrets/hpc/ssh-profiles.yaml

rp hpc sync project \
  --project <project-name> \
  --profile <profile-name> \
  --role login \
  --config secrets/hpc/ssh-profiles.yaml

rp hpc sync data \
  --project <project-name> \
  --profile <profile-name> \
  --role login \
  --config secrets/hpc/ssh-profiles.yaml

rp hpc notebook plan \
  --project <project-name> \
  --profile <profile-name> \
  --role login \
  --config secrets/hpc/ssh-profiles.yaml
```

These forms do not authorize a remote transfer. They may write reported local
manifests, plan files, or status records beneath the artifact boundary. Review
the reported sources, destinations, commands, and local files before adding
`--execute` to a supported operation.

**SSH-active:** remote data verification is read-only but is not offline
validation. It contacts configured paths over SSH:

```bash
rp hpc verify data --project <project-name>
```

It does not prove scheduler, runtime, software, quota, or full dataset
readiness.

## Status and cancellation

Default status reads recorded local state and invokes no subprocess:

```bash
rp hpc status --run-id <run-id>
```

Recorded state is not proof of the scheduler's current or terminal state.

**SSH-active:** adding `--live` immediately loads the selected SSH profile and
runs one `squeue` query:

```bash
rp hpc status --run-id <run-id> --live
```

Inspect the returned payload's `checked` and `ok` fields. Empty output is the
ambiguous `not-found-or-completed`, not proof of successful completion. The
command does not run `sacct` or reconcile terminal accounting.

Cancellation is currently a local request record:

```bash
rp hpc cancel --run-id <run-id>
```

It records `cancel-requested` and may render a possible `scancel` command. It
has no `--execute` option, invokes no SSH or scheduler subprocess, and does not
run `scancel`. The local record is not proof of remote cancellation or
confirmed cancel behaviour.

## Retrieval

After independently verifying terminal scheduler state, an operator may
authorize run-manifest-based retrieval:

```bash
rp hpc pull --run-id <run-id> --execute
```

A focused retrieval can specify a run-relative source and explicit local
destination:

```bash
rp hpc pull \
  --run-id <run-id> \
  --subpath <relative-subpath> \
  --destination <local-destination> \
  --execute
```

The current pull is merge-oriented `rsync -az`. It does not prove that the
scheduler job succeeded, verify a complete remote result, or reconcile
terminal scheduler state. It does not atomically publish a complete retrieved
tree, attest an end-to-end digest, protect existing destination content from
merging or replacement, guarantee interrupted-transfer recovery, or issue a
complete lifecycle receipt.

Retrieval does not automatically promote files into canonical datasets or
derivatives. Preserve failure evidence, inspect retrieved bytes, and use a new
run identity when the established collision policy requires it.

## Current lifecycle limitations

Local tests cover configuration, validation, rendering, no-execute behaviour,
and mocked crossings of SSH, rsync, and scheduler boundaries. They do not
constitute live-provider evidence.

The complete remote lifecycle remains unsupported. In particular, the current
surface does not provide all of the following as one accepted transaction:

- live readiness and provider validation;
- verified staging and immutable remote input identity;
- scheduler submission with terminal accounting reconciliation;
- confirmed remote cancellation;
- verified complete retrieval and atomic local promotion;
- complete, digest-bound receipts and recovery evidence.

The safety helpers described by
[ADR-0023](../../docs/decisions/ADR-0023-hpc-safety-primitives.md) are
foundational library primitives. They do not make existing YAML, manifest,
workflow, transfer, submission, or retrieval writers transactional, and no
complete remote runtime consumes them end to end.

The headline execution contract in
[ADR-0022](../../docs/decisions/ADR-0022-headline-hpc-execution-contract.md)
remains incomplete. Provider-neutral configuration means the package avoids
embedding one site's values; it does not claim compatibility with arbitrary
clusters, schedulers, authentication systems, storage policies, or scientific
tools.

Further reading:

- [Capability matrix](../../docs/capabilities.md)
- [BIDS and HPC planning](../../docs/bids-hpc-slice.md)
- [HPC troubleshooting](../../docs/how-to/hpc-troubleshooting.md)
- [Architecture and package ownership](../../ARCHITECTURE.md)
- [ADR-0022: headline HPC execution contract](../../docs/decisions/ADR-0022-headline-hpc-execution-contract.md)
- [ADR-0023: HPC safety primitives](../../docs/decisions/ADR-0023-hpc-safety-primitives.md)
