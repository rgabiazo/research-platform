# ADR-0020: Tabular run identity and execution claims

## Status

Accepted

## Context

Run roots share the workspace-global `artifacts/runs/<run-id>/` namespace. The
previous tabular lifecycle created directories permissively and rewrote control
files when an ID was reused. A later plan could therefore replace the command a
user had reviewed, a BIDS or foreign workflow could collide with a tabular run,
and concurrent execution requests could both proceed. A missing executable or
interrupted launch could also leave misleading state.

The public workflow intentionally supports reviewing a plan and then executing
that exact plan once under the same ID. That narrow transition must remain
possible without turning an ID into an overwrite, retry, or resume mechanism.

## Decision

### Safe identity and shared admission

Every explicit or generated run ID is validated before path construction. It is
one safe, nonempty filesystem name, not a path. Absolute paths, separators,
traversal, `.`, `..`, controls, and values escaping the configured runs root are
rejected without normalization or filesystem writes.

A fresh planner atomically reserves an absent run root. An existing file,
symbolic link, special file, empty or foreign directory, malformed control tree,
or root owned by another workflow is never replaced. This minimum shared
admission boundary prevents tabular and BIDS workflows from overwriting each
other without changing the existing run-directory layout.

### Bind execution to the reviewed plan

Each scoped tabular manifest stores a plan-identity schema version and a
canonical SHA-256 digest. The identity binds the run ID, slice, project,
workflow action and target, selected batch row, dataset and input table, ordered
predictor contract, preprocessing and model or analysis settings, evaluation or
analysis input-run identity, normalized resources, expected outputs, complete
rendered command sequence, and exact `execute.sh` bytes. A corresponding remote
plan also binds its SLURM script and submission material.

Creation and status timestamps, the supported planning-to-authorization mode
change, scheduler job IDs, and scheduler observations are excluded. Execution
regenerates the requested plan in memory and verifies both its identity and the
persisted reviewed script. It does not silently rewrite a changed plan. An
allowed transition preserves the original creation time and identity.

### Admit only one-shot transitions

A fresh plan ends in `planned` without an execution claim or subprocess. A
fresh direct local execution may transition from new to `running`, then to
`succeeded` or `failed`. An intact, unclaimed `planned` root may transition once
to local execution only when it represents either:

- `rp run plan ...` followed by the exact corresponding
  `rp run local ... --execute`; or
- `rp run local ... --dry-run` followed by the exact corresponding
  `rp run local ... --execute`.

Remote tabular analysis similarly permits one exact
`rp run submit analysis tabular ...` plan to transition to the same request with
`--execute`, retaining truthful staged, submitted, and failure states.

A repeated plan, second execution, or root in a running, terminal, remote,
cancellation-related, malformed, unknown, or claimed state is rejected before
mutation or subprocess execution. Rejections preserve every existing byte and
filesystem entry and direct the operator to inspect the run and choose a new ID.
No overwrite, resume, retry, replace, force, stale-claim deletion, or takeover
flag is introduced.

### Claim execution atomically

Before any execution-time mutation, local subprocess, stage, or submission, the
process acquires an exclusive hidden sibling claim. It then re-reads and
revalidates the root, manifest, status, plan identity, and script to close the
check/use race. Exactly one concurrent claimant can proceed.

Claim ownership includes the filesystem identity of the object created by that
process. Orderly cleanup removes only that same object. A foreign replacement or
stale claim is retained and causes later requests to fail closed; recovery is a
manual inspection followed by a new run ID, not automatic claim breaking.

Before a local subprocess begins, status is already `running`. A normal nonzero
return, unavailable executable, caught launch error, or interruption records a
truthful failure where safe rather than leaving a misleading running state.

### Keep scientific-output transactions separate

This decision protects run admission, reviewed-plan identity, state changes,
and single-process ownership. It does not stage the complete scientific output
set, validate a successful output inventory, promote outputs atomically, roll
back partial scientific files, or validate upstream-run success and output
digests. Those are separate lifecycle gates.

## Consequences

- A run ID is a durable, one-shot scientific execution identity rather than an
  overwriteable directory name.
- Users retain the review-then-execute workflow without allowing configuration
  or script drift between those phases.
- Concurrent local or remote execution authorization has one owner, while stale
  claims remain visible recovery evidence.
- Failed and interrupted runs are preserved for diagnosis and deliberately
  replaced by a new ID.
- Atomic execution ownership must not be described as a whole-run scientific
  output transaction.

Rejected alternatives include namespacing or migrating existing roots, adding
overwrite/resume/retry flags, treating timestamps as scientific identity,
automatically deleting stale claims, and partially implementing output rollback
inside this control-plane gate.
