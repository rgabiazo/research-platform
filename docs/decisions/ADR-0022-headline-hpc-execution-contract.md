# ADR-0022: Headline HPC execution contract

## Status

Accepted architecture; implementation is incomplete

No live cluster has been validated. The current public classification remains
plan-first and experimental/external-runtime, specifically **Experimental or
external-runtime**. This decision accepts a
future conditional claim; it does not make that claim true today. The claim is
forbidden until every mandatory implementation gate, the fake-remote
acceptance, and a separately authorized live-cluster acceptance have passed.
Private neuroimaging integration evidence cannot substitute for the mandatory
synthetic lifecycle acceptance.

This ADR defines an architecture and evidence boundary. It does not implement
HPC execution, receipts, live validation, or claim promotion.

## Context

Research Platform can currently write local HPC starter configuration, inspect
targets and profiles, render SLURM plans, and record local planning state. Its
explicit SSH, transfer, submission, live-status, and retrieval paths have
mocked command-boundary evidence but no live-cluster validation. Local
`cancel-requested` state is not remote cancellation, one `squeue` query is not
terminal accounting, and merge-oriented retrieval is not transactional
publication. Those boundaries are documented in the capability matrix and
must not be promoted by this ADR.

A public-alpha headline needs a narrower, testable lifecycle with truthful
states, immutable identities, versioned receipts, exclusive ownership, and
reproducible evidence. The contract must work with private configuration while
keeping the acceptance workload synthetic and the publishable record free of
private infrastructure or participant information.

## Decision

### Accept only one narrow future headline

The maximum headline accepted by this decision is:

> Plan-first SLURM execution, monitoring, cancellation, and verified retrieval
> for a deterministic synthetic tabular workload using the same
> private-overlay/external-root topology as BYOD, validated end to end on one
> documented cluster environment.

This is a conditional future statement. It may be used only after H1 through
H11 and their acceptance evidence are complete and a dedicated H12 claim
promotion gate approves the exact wording. Documentation must name only the
environment, workload, Python version, and limits actually proven.

The mandatory live smoke uses:

- a disposable private-style project overlay;
- a synthetic external tabular input root;
- no participant or private study data;
- unique owned local and remote namespaces;
- a source snapshot bound to an exact Git commit and a canonical
  source/release payload inventory with a SHA-256 tree digest;
- one supported Python version available on the validated cluster environment;
- deterministic tabular preprocessing, training, and evaluation;
- a harmless successful scheduler canary;
- a deliberate nonzero-exit scheduler canary;
- a separate bounded waiting job for cancellation;
- transactional, verified retrieval;
- collision and interruption/recovery evidence; and
- sanitized publishable evidence.

### Preserve explicit exclusions

The headline does not imply:

- universal SLURM compatibility;
- general Alliance or Nibi compatibility, or compatibility with any other
  provider not actually tested;
- validation on more than the actually tested environment;
- arbitrary future workflow support;
- real-data neuroimaging validation or raw BIDS-to-analysis execution;
- FSL, ANTs, SPM, DeepPrep, FEAT, or fMRIPost-AROMA validation;
- containerized neuroimaging validation;
- remote ROI or MVPA execution unless separately accepted;
- support for multiple providers;
- PyPI installation;
- unattended credential or MFA handling; or
- automatic replacement or destructive recovery, including adoption of
  foreign state.

Future tools may reuse this lifecycle only after their own runtime, dependency,
input, output, privacy, and acceptance contracts are implemented and
validated. An optional private DeepPrep, fMRIPost-AROMA, or FEAT integration
series may occur only after H11. It is out of scope here, cannot replace H11,
and cannot broaden the public claim without another decision and reproducible
evidence.

Optional private neuroimaging integration is not a substitute for the
synthetic H11 acceptance.

### Use converging prerequisites, not a rigid setup sequence

Runtime readiness and data staging are independently owned prerequisites:

```text
planned
  +-- runtime-planned -> provisioning -> runtime-ready
  `-- transfer-planned -> staging -> staged

runtime-ready + staged
  -> submitting
  -> submitted
  -> queued or running
  -> scheduler-completed
  -> remote-output-verified
  -> retrieval-staging
  -> retrieved
```

Runtime artifacts may need staging before readiness can be established.
Submission requires every applicable successful readiness and stage receipt;
the implementation must not encode an unnecessarily rigid ordering between
the two branches.

The permitted completion label is `scheduler-completed`. Scheduler completion
is not scientific success: it is evidence about the allocation, not proof that
the workflow succeeded or produced valid outputs.

Cancellation has its own confirmed lifecycle:

```text
queued or running
  -> cancel-requested
  -> cancel-submitted
  -> scheduler-cancelled
```

`cancel-uncertain` records an accepted, disconnected, ambiguous, or otherwise
unreconciled cancellation attempt. A cancellation race may truthfully end in
`scheduler-completed` or `scheduler-failed`; either outcome does not satisfy
cancellation acceptance and may not be rewritten as cancelled.

The future implementation must preserve these distinct failed or unresolved
outcomes, even if later code chooses different exact labels:

```text
stage-failed
provisioning-failed
runtime-not-ready
submission-failed
scheduler-failed
scheduler-timeout
scheduler-out-of-memory
scheduler-preempted
scheduler-node-failure
scheduler-accounting-pending
cancel-uncertain
remote-execution-failed
remote-receipt-missing
remote-output-invalid
retrieval-failed
retrieval-invalid
```

### Enforce state invariants

1. State transitions are monotonic and atomically persisted.
2. Scheduler completion is not scientific-output success.
3. `COMPLETED` without a valid remote execution-success receipt cannot become
   `remote-output-verified`.
4. A controlled scientific failure produces a failure receipt and never a
   success receipt.
5. Abrupt node loss may leave a missing or uncertain receipt and must never be
   inferred as success.
6. Retrieval cannot become successful merely because a transfer command
   returned zero.
7. Missing, delayed, ambiguous, malformed, or conflicting evidence fails
   closed or remains explicitly unresolved.
8. A terminal failure cannot transition back to running or successful.
9. A cancellation request or accepted `scancel` command is not cancellation
   confirmation.
10. Scheduler-accounting delay is represented explicitly rather than guessed.
11. Local and remote receipts bind the same run, plan, source, payload, stage,
    runtime, scheduler, and retrieval identities as applicable.
12. No command silently merges with, replaces, or overwrites a prior canonical
    run.
13. Exact-match reuse or resume is distinct from replacement.
14. Claims and receipts have exclusive ownership and are created atomically.
15. Cleanup is separately authorized and never required to conceal a failed
    operation.

### Bind both repository and exportable source identity

Every reviewed plan and applicable receipt binds both:

- the originating Git commit when one is available; and
- a canonical source/release payload inventory with a SHA-256 tree digest.

The payload digest is mandatory. A history-free public export has different
Git history. Commit identity alone cannot prove equivalence between
the private source tree and the exported release payload. The canonical
inventory definition, path ordering, file-type policy, byte hashing, and tree
digest algorithm must be versioned and deterministic.

### Define six versioned receipt families

Later implementation gates must define machine-readable schemas for:

1. **Runtime readiness receipts**: the provisioned runtime and its verified
   capabilities.
2. **Managed transfer/staging receipts**: the source, verified remote incoming
   tree, and promoted staged payload.
3. **Scheduler submission receipts**: exclusive submission ownership and the
   validated allocation ID.
4. **Scheduler observation and terminal accounting receipts**: raw and
   normalized active and terminal scheduler evidence.
5. **Remote execution success or controlled failure receipts**: atomically
   finalized workflow outcomes.
6. **Local retrieval and validation receipts**: the verified remote source,
   local staging tree, promotion, and final inventory.

Each schema must bind, when applicable:

- receipt schema version;
- run ID and operation ID;
- reviewed plan digest;
- source commit;
- canonical source/payload digest and its inventory;
- stage inventory and tree digest;
- runtime identity and package version, including coordinated package
  identity;
- target/profile identity;
- scheduler job ID;
- raw and normalized scheduler state;
- scheduler exit code, reason, and timestamps;
- exact output inventory with byte sizes and SHA-256 digests;
- previous receipt digests and prerequisite-receipt digests; and
- creation and finalization timestamps.

All portable receipt paths are safe relative paths. Private ignored
operational receipts may retain target, profile, and path identity needed for
safe reconciliation. Tracked documentation, fixtures, tests, and sanitized
publishable evidence must never contain credentials, tokens, private keys,
passphrases, personal absolute paths, private usernames or hostnames, account
or allocation identifiers, participant identifiers, private study names,
participant-derived values, or hashes of private data.

Full operational receipts remain private. Publishable evidence is a separately
sanitized projection that retains enough receipt-chain, synthetic-input,
result, and environment identity to support the bounded claim without exposing
private infrastructure.

### Require provider-neutral setup and offline validation

H1 must replace provider assumptions in the generic default with:

- a genuinely generic SSH/SLURM target template;
- explicit private host and user configuration;
- no Alliance, Nibi, MFA, module, scratch, container, account, partition, or
  version assumption in that default;
- provider examples only as explicitly selected integrations requiring site
  review;
- Alliance/MFA behavior only as an optional provider integration; and
- a subprocess-free offline validation command or mode.

Offline validation must fail in a controlled way for placeholders, unsafe
roots, incomplete profiles, unsupported promotion capabilities, and invalid
scheduler configuration. It is configuration validation, not live readiness.

### Require a remote-runtime readiness receipt

The versioned readiness receipt proves:

- a supported Python version;
- an isolated virtual environment with user-site packages disabled;
- the expected `rp` version and coordinated package identity;
- a successful `pip check`;
- the required workflow driver and SLURM executor are available;
- configured remote roots are accessible;
- workload-specific dependencies are ready;
- container runtime and immutable image identity when applicable; and
- no unplanned package-index or outbound-network dependency exists.

Planning, provisioning, staging, and readiness validation remain distinct
operations.

### Make transfer, staging, and recovery transactional

Managed transfer must provide:

- an exact source inventory with SHA-256 file and tree digests;
- a regular-file-only policy with symlink, device, socket, FIFO, and
  unsafe-path rejection;
- source-stability checks before and after transfer;
- a unique remote incoming directory and exclusive remote operation claim;
- remote inventory and digest verification;
- a capability check for same-filesystem atomic no-replace promotion;
- an unsupported failure when safe promotion is unavailable, with no unsafe
  overwrite fallback;
- a stage receipt bound to the reviewed plan;
- submission rejection without matching stage and readiness receipts; and
- preservation of foreign, incomplete, or conflicting remote state, with no
  implicit merge, overwrite, replacement, or force behavior.

Transfer usability requires one explicit safe same-identity resume/recovery
operation. Resume is allowed only when operation ID, direction, source and
destination identity, reviewed plan digest, source inventory/tree digest, and
owned incoming or retrieval-staging namespace all match. A changed source or
foreign staging tree is rejected; it is never reinterpreted as a resumable
operation.

### Make submission receipt-bound and duplicate-safe

Submission requires matching successful stage and readiness receipts,
exclusive ownership, and `sbatch --parsable` or an equivalently stable
machine-readable job-ID contract. It validates exactly one numeric allocation
ID and rejects empty, malformed, or conflicting identifiers.

An atomic submission receipt protects against duplicate submission after an
interruption or disconnect. A run cannot become `submitted` without a valid
job ID and receipt. When command outcome and receipt persistence cannot be
reconciled, the run uses an explicit unresolved state; the platform must not
guess or issue another submission.

### Reconcile active and terminal scheduler evidence

Monitoring uses `squeue` for active state and `sacct`, or one explicitly
documented equivalent, for terminal accounting. It uses exact allocation-row
selection rather than `.batch` or `.extern` children and normalizes
queued, running, completed, failed, cancelled, timeout, out-of-memory,
preempted, and node-failure states.

Receipts preserve the raw scheduler state, exit code, reason, and timestamps.
Accounting delay becomes `scheduler-accounting-pending`. SSH,
scheduler-command, and parser failures return nonzero and preserve evidence.
Observation may be separated from atomic persistence where needed, but the
result provides idempotent monotonic reconciliation.

### Execute and confirm cancellation explicitly

Cancellation remains plan/render-only by default. Execution requires explicit
execution authorization such as `--execute`, explicit job-ID confirmation, and
validation of run ownership, target identity, and job identity. It invokes
exactly one intended `scancel` invocation.

An accepted command records `cancel-submitted`, not `cancelled`. Only terminal
scheduler accounting can confirm `scheduler-cancelled`.
Already-terminal jobs, races, missing IDs, SSH failures, ambiguous
disconnects, and confirmation failures retain their truthful outcome;
unreconcilable results become `cancel-uncertain`. No successful-cancellation
claim is permitted without terminal scheduler evidence.

### Finalize remote success or controlled failure atomically

The remote wrapper atomically finalizes exactly one of:

- a successful execution receipt after workflow success and complete output
  validation; or
- a controlled-failure receipt that preserves failure identity and
  diagnostics.

A success receipt requires reviewed staged-identity and runtime-readiness
matches, workflow exit success, a complete expected output inventory, safe
relative regular-file paths, byte sizes and SHA-256 digests, and atomic
finalization. Scheduler completion without this receipt remains only
`scheduler-completed`. Abrupt failure may leave `remote-receipt-missing` or
another unresolved state; it is never silently promoted to success.

### Retrieve through verified staging and no-replace promotion

Normal result retrieval requires scheduler-completed evidence and a matching
valid remote execution-success receipt. Failed or uncertain runs use a
separate recovery/quarantine retrieval path that cannot be promoted as normal
results.

Retrieval uses a unique local sibling staging directory, exclusive ownership,
exact inventory, size, and digest validation, source-stability checks, and
atomic no-replace promotion into an absent destination. Exact-match idempotent
reuse is allowed only after independent verification. Interrupted retrieval
may resume only through the explicit same-identity recovery contract. This is
a same-identity interrupted-retrieval resume, never replacement. Collisions
fail closed. There is no merge, overwrite, replacement, or success claim based
only on transfer exit status.

## Implementation gates

Each gate is a separate commit with focused local verification and green CI
before the next begins:

- **H0:** architecture and evidence contract; accept this decision without
  changing the current public capability claim.
- **H1:** provider-neutral setup; generic target configuration and
  subprocess-free offline validation.
- **H2:** shared safety primitives, delivered through four separately committed
  and CI-green subgates:
  - **H2a:** contracts, canonical encoding, and portable lexical paths;
  - **H2b:** descriptor-anchored regular-file inventories and SHA-256 tree
    digests;
  - **H2c:** atomic no-replace publication and exclusive claims; and
  - **H2d:** receipt-envelope foundations and complete cross-platform H2
    acceptance.
  All four subgates collectively satisfy H2. H3 cannot begin until H2d is
  complete and green. Partial H2 implementation does not broaden the current
  **Experimental or external-runtime** classification. The frozen H2 contract
  is recorded in
  [`ADR-0023`](ADR-0023-hpc-safety-primitives.md).
- **H3:** remote runtime readiness; provisioning and a versioned readiness
  receipt.
- **H4:** transactional upload/staging; verified no-replace promotion and
  explicit same-identity recovery.
- **H5:** duplicate-safe submission; receipt-bound scheduler submission with
  one validated allocation ID.
- **H6:** terminal reconciliation; `squeue` plus `sacct` observation,
  accounting delay, and monotonic terminal-state reconciliation.
- **H7:** executed cancellation; explicit `scancel` authorization followed by
  scheduler-confirmed cancellation or a truthful uncertain/race outcome.
- **H8:** remote execution receipts; atomically finalized success and
  controlled-failure receipts.
- **H9:** transactional retrieval; verified no-replace normal retrieval,
  explicit same-identity recovery, and separate recovery/quarantine retrieval.
- **H10:** fake-remote acceptance; deterministic end-to-end acceptance across
  success, failure, cancellation, collisions, and interruption/recovery
  without external infrastructure.
- **H11:** live-cluster acceptance; a separately authorized synthetic-only
  run on one documented cluster environment.
- **H12:** claim promotion and release gate; review sanitized evidence, run
  the complete release-candidate suite, freeze the proven boundary, update
  classifications in a dedicated gate, and create the history-free export
  only under separate authorization.

An optional private neuroimaging integration series may occur after H11 and
before H12. It is not a substitute for H11 and is not a release blocker unless
a corresponding neuroimaging capability is intentionally added to the alpha
claim.

## H11 live-cluster acceptance

H11 runs in a unique owned namespace and must prove all of the following:

1. Private target configuration remains untracked.
2. Credentials and MFA remain external to the repository and receipts.
3. The source commit and canonical payload digest are recorded.
4. The runtime-readiness receipt passes.
5. A deterministic, non-sensitive transfer inventory is verified remotely.
6. A collision at the same destination fails closed.
7. An interrupted upload resumes only through the explicit matching recovery
   operation or is safely rejected with evidence preserved.
8. Exactly one successful canary is submitted.
9. Exactly one numeric job ID is recorded.
10. Duplicate submission is prevented.
11. Active state is observed through `squeue`.
12. Terminal completion and `0:0` are reconciled through `sacct`.
13. Scheduler completion is not called scientific success.
14. A deliberate nonzero-exit job becomes scheduler or remote failure.
15. A separate bounded waiting job is planned for cancellation without remote
    mutation.
16. Explicit cancellation invokes exactly one intended `scancel`.
17. Cancellation is confirmed through terminal accounting.
18. A cancellation race is represented truthfully.
19. Deterministic tabular preprocessing, training, and evaluation complete.
20. The remote success receipt contains the complete expected inventory.
21. Normal retrieval validates every declared byte size and SHA-256 digest.
22. Invalid or incomplete output cannot promote.
23. Interrupted retrieval uses explicit matching recovery or fails safely.
24. A failed run's diagnostics can be retrieved only through the
    recovery/quarantine path.
25. Local and remote receipt chains match.
26. Publishable evidence is sanitized.
27. No private data, private-data hashes, credentials, usernames, hostnames,
    account identifiers, or participant identifiers enter tracked files.
28. Cleanup is reviewed and separately authorized after evidence preservation.

The live record must include the exact tested commit, canonical payload
inventory and digest, selected supported Python version, target environment
description at the least-identifying useful granularity, sanitized receipt
chain, and results for success, failure, cancellation, collision,
interruption/recovery, invalid-output rejection, and verified retrieval. Full
operational receipts remain private.

## Claim-promotion rule

HPC cannot become a headline capability until:

- H1 through H10 are committed and CI-green;
- H11 passes on one documented environment;
- the exact tested commit and canonical payload digest are recorded;
- success, failure, cancellation, collision, interruption/recovery,
  invalid-output rejection, and verified-retrieval evidence exist;
- complete operational receipts are retained privately;
- sanitized evidence receives privacy and technical review;
- public documentation names only the proven provider and workload boundary;
  and
- current experimental classifications are changed only by the dedicated H12
  claim-promotion gate.

Until then, local planning evidence and mocked execution tests remain useful
engineering evidence but cannot be described as live execution validation.

## Consequences

- The current plan-first and experimental/external-runtime classification is
  unchanged.
- A future headline is bounded to one deterministic synthetic tabular
  lifecycle and one documented environment.
- Scheduler state, scientific success, transfer success, and publication are
  separate facts connected by versioned receipts.
- The source tree remains identifiable across a future history-free export.
- Safe transfer and retrieval require no-replace promotion plus explicit,
  exact-identity recovery rather than implicit retry or merge.
- Cancellation becomes an authorized and terminally confirmed operation, not
  a local request or command rendering.
- Private infrastructure details remain available for reconciliation without
  entering publishable evidence.
- Supporting another provider or workflow requires its own bounded contract
  and evidence.

Rejected alternatives include promoting the current mocked boundary; treating
`squeue` disappearance as success; equating scheduler `COMPLETED` with
scientific success; relying on a Git commit without a payload digest; accepting
transfer exit zero as verified staging or retrieval; merge-oriented canonical
publication; overwrite, force, or foreign-state adoption; implicit retry;
automatic stale-claim removal; unconfirmed cancellation; publishing private
receipts; using private neuroimaging data as the mandatory acceptance workload;
and claiming provider-wide compatibility from a single environment.
