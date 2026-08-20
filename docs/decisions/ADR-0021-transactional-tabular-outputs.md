# ADR-0021: Transactional local tabular outputs

## Status

Accepted

## Context

ADR-0020 made a run ID durable, bound execution to an exact reviewed plan, and
gave one process atomic execution ownership. It deliberately did not protect
the scientific output set. Local preprocessing, training, evaluation, or
configured tabular analysis could therefore expose sequentially written or
partially validated files before the run became successful. Evaluation also
located training inputs by path rather than requiring proof that they came from
an intact successful local transaction.

Control-plane identity, execution ownership, scientific-output publication,
output integrity, and downstream consumption are related but distinct
contracts. A public alpha needs each boundary to fail closed without implying
that remote execution or crash recovery is solved.

## Decision

### Plan the complete output transaction

Every supported local tabular plan stores a versioned transaction plan before
its reviewed plan identity is calculated. It binds the run and workflow, the
final `outputs/` directory, exact logical outputs and portable relative names,
content types, future transaction-manifest location, and an
`existing_output: fail` policy. The reviewed command sequence uses a stable
logical output root, so a random runtime staging name does not change the plan
identity.

Plans and dry-runs create no final output or staging directory. A reviewed plan
can execute only while both remain absent. Planned roots created under an older
contract or containing even an empty output directory are preserved and require
a new run ID rather than migration or adoption.

### Keep execution ownership and output ownership separate

The ADR-0020 sibling claim remains the sole authorization claim. After acquiring
it, execution revalidates the reviewed plan, source digests, controls, script,
and filesystem admission, then persists `running`. Only then does it exclusively
create one hidden staging directory beneath the run root, on the same filesystem
as final `outputs/`. The process records that staging object's filesystem
identity and removes it only if it still owns the same object.

Every reviewed scientific command writes into this owned staging root. There is
no per-file publication and no final-directory merge. Random staging details
are runtime-only and never enter public provenance.

### Validate, attest, and atomically publish

Before publication, the complete staged tree must exactly match the planned
workflow inventory. Every entry is confined, regular, nonsymlinked, and of its
declared content type; JSON and TSV structure, finite values, row widths and
counts, ordered predictors, target, split identities, and cross-file provenance
are checked without recomputing scientific statistics.

The platform then writes
`research_platform.core.tabular_output_transaction.v1` inside staging. It binds
the run, workflow, reviewed plan identity, and one portable record per
scientific file containing logical name, relative path, content type, byte size,
SHA-256, and TSV row count and ordered columns where applicable. It excludes
absolute and staging paths, credentials, unnecessary timestamps, and a
recursive digest of itself.

After source revalidation and appropriate flushing, a supported atomic
no-replace directory rename publishes the complete staged directory as the
previously absent `outputs/`. Linux uses a no-replace rename primitive; macOS
uses its exclusive rename boundary. An unavailable safe primitive or a foreign
concurrent destination fails closed. `succeeded` is persisted only after the
atomic publication completes.

### Require verified upstream transactions

Evaluation accepts only an unclaimed, `succeeded`, local `train model` run for
the same project and batch. Its controls, reviewed identity and script, exact
train inventory, transaction-manifest binding, file confinement, sizes, and
digests must all remain valid. A planned, running, failed, submitted,
remote-only, legacy, malformed, claimed, mismatched, or modified source is not
an evaluation input.

The evaluation plan identity binds the upstream plan identity, the source
transaction-manifest SHA-256, all source records, and the exact consumed split,
feature-table, and model digests. Orchestration revalidates the upstream
transaction before execution and promotion. The low-level consumer additionally
opens each input once, hashes it, compares the expected digest, and parses those
same bytes, closing the path-check/reopen gap while preserving direct-call
behavior when expected digests are not supplied.

Training similarly reads staged feature bytes but records the portable final
`outputs/features.tsv` reference in model provenance rather than a transient
staging path.

### Fail with truthful recovery evidence

An ordinary child, validation, or pre-promotion failure publishes no final
output directory. The process removes only its owned staging object, durably
records `failed`, and releases its own execution claim. A cleanup failure,
interruption, uncertain promotion, or terminal-status persistence failure keeps
the claim and exact recovery paths visible. If publication succeeded but the
success status could not be persisted, outputs remain committed and the last
confirmed durable state is `running`. After a replacement followed by a flush
failure, `status.yaml` may read `succeeded` while its durability is uncertain;
the retained claim still makes evaluation refuse the source.

Status updates use atomic same-directory control-file replacement. This does not
authorize replacement of a scientific output directory.

## Consequences

- Successful local tabular outputs become visible as one validated directory,
  and `succeeded` now attests that publication completed.
- Exact byte digests make evaluation depend on the reviewed, unchanged training
  transaction rather than merely familiar filenames.
- Transaction manifests provide portable integrity evidence without changing
  scientific output schemas or numerical behavior.
- Failed pre-promotion executions expose no final scientific output tree, while
  ambiguous failures preserve rather than conceal recovery evidence.
- SLURM and remote execution remain experimental or plan-only and receive no
  transaction or attestation claim from this decision.
- Hard-kill and power-loss recovery, stale-claim removal, retry, resume,
  overwrite, replacement, legacy receipt backfilling, output adoption,
  cryptographic signing, live-cluster validation, the broader BYOD guide, and
  RDM/report work remain deferred.

Rejected alternatives include sequential per-file promotion, `os.replace`,
copy-and-delete publication, adopting pre-existing output trees, trusting only
paths or timestamps, reopening inputs after digest validation, silently
backfilling legacy receipts, and broadening this local gate into remote runtime
support.
