# ADR-0023: HPC safety primitives

## Status

Accepted contract; H2a canonical/path foundations, H2b descriptor-anchored
regular-file inventories/tree digests, and H2c1 atomic no-replace publication
implemented by this working snapshot. Hosted H2c1 acceptance and H2c2-H2d
remain pending.

This decision freezes the four top-level H2 boundaries and the detailed H2a,
H2b, H2d, and H2c1 contracts. Detailed H2c2 semantics remain separately
pending. H2 is not complete until H2a, H2b, H2c, and H2d are separately
committed, verified, and CI-green. H3 may not begin before H2d is complete and
green. Partial H2 work changes no public capability claim: HPC remains
plan-first and **Experimental or external-runtime**.

The existing YAML, manifest, tabular, ROI, MVPA, publication, and run-lifecycle
writers are not migrated by this decision. No local or remote runtime workflow
consumes the H2a/H2b/H2c1 primitives.

## Context

The future lifecycle in
[`ADR-0022`](ADR-0022-headline-hpc-execution-contract.md) needs one
dependency-safe authority for portable identity, descriptor-anchored
inventories, immutable publication, exclusive ownership, and receipt
canonicalization. `research-core` depends on the standard-library-only
`research-hpc` package, so the authority belongs under
`research_platform.hpc.safety`; importing core or neuro transaction code would
reverse that dependency.

Existing domain transactions remain useful reference evidence, but their
schemas and rollback rules are domain-specific. Existing `_yaml.py`,
`manifest.py`, status writers, and other `os.replace` helpers permit
replacement and do not satisfy immutable no-replace publication.

## Decision

### Deliver H2 in four bounded subgates

- **H2a — contracts, canonical encoding, and portable paths (implemented):**
  freeze and implement only the canonical JSON, domain-separated digest, and
  lexical managed-payload path foundations.
- **H2b — inventories and tree digests (implemented):** implement
  descriptor-anchored regular-file inventories, mutation checks, and the
  frozen tree digest.
- **H2c — publication and ownership (pending):** deliver the existing H2c gate
  through two internally ordered, separately committed and hosted-CI-green
  implementation gates:
  - **H2c1 — atomic no-replace publication:** implemented by this working
    snapshot; hosted acceptance remains pending until the eventual commit's CI
    succeeds.
  - **H2c2 — exclusive claims:** a separately reviewed pending gate whose
    detailed API, acquisition, release, durability, and recovery semantics are
    frozen only after H2c1 is implemented and hosted-CI-green.
  H2c remains pending and incomplete until both H2c1 and H2c2 pass.
- **H2d — receipt foundations and acceptance (pending):** implement the common
  receipt envelope and complete the cross-platform H2 acceptance suite.

This internal H2c split does not alter ADR-0022's
H2a -> H2b -> H2c -> H2d -> H3 order. Each implementation gate requires
focused verification and green CI before the next. H2d and H3 remain blocked,
and no runtime workflow consumes the implemented H2c1 primitives.

### Canonical JSON v1

The schema identifier is:

```text
research_platform.hpc.canonical_json.v1
```

Accepted values are exactly JSON `null`, booleans, signed 64-bit integers,
Unicode strings, arrays, and objects represented by string-keyed mappings.
Booleans are checked before integers. Floats, `Decimal`, NaN, Infinity, bytes,
tuples, non-string mapping keys, and other objects are rejected without
conversion.

Canonical bytes are UTF-8 with no BOM, leading or trailing whitespace, or
trailing newline. Serialization uses compact separators and
`ensure_ascii=False`. Object keys are ordered by their UTF-8 bytes. Every key
and string value must contain only valid Unicode scalar values assigned in the
Unicode 3.2 repertoire and already be NFC according to the frozen
`unicodedata.ucd_3_2_0` database. Characters whose Unicode 3.2 category is
`Cn`, surrogates, and non-NFC input are rejected rather than normalized.
Interpreter-default Unicode data does not participate in canonical
acceptance. Supporting a broader Unicode repertoire requires a future
canonical schema version. Parsing rejects duplicate keys, invalid UTF-8,
unsupported values, and out-of-range integers, then reserializes the value and
requires exact byte equality. Thus whitespace, alternate escapes, alternate
key order, and other noncanonical encodings are rejected.

The immutable default limits are:

```text
maximum canonical document bytes: 16 MiB
maximum container nesting depth: 64
maximum total array elements plus object members: 100,000
maximum UTF-8 bytes per key or string value: 1 MiB
integer range: -(2**63) through 2**63 - 1
```

A root container has depth one; scalar roots have depth zero. Limits affect
acceptance, never the bytes for an accepted value. Caller-supplied limits may
only narrow these frozen maxima. Before `json.loads` materializes a value, a
bounded lexical preflight enforces depth, aggregate member/element count,
decoded string-byte size, integer range, and valid JSON string escapes without
building a list, mapping, or decoded value tree. The implementation has no
filesystem, process, network, clock, random, or environment dependency.

### Digest identifiers and framing

The frozen identities are:

```text
raw file and canonical-inventory byte hash: sha256
tree digest: research_platform.hpc.sha256_tree.v1
receipt-envelope digest: research_platform.hpc.sha256_receipt_envelope.v1
```

The versioned domains are exactly:

```python
b"research-platform:hpc:regular-file-tree:v1\0"
b"research-platform:hpc:receipt-envelope:v1\0"
```

Both versioned digests use:

```text
SHA256(domain || uint64be(len(canonical_body)) || canonical_body)
```

The length is an unsigned eight-byte big-endian integer. An outer digest
wrapper is never part of its own hashed body. H2a implements only this generic
domain-separated SHA-256 operation; it does not create a tree inventory or
receipt.

### Managed-payload portable relative path v1

The schema identifier is:

```text
research_platform.hpc.portable_relative_path.v1
```

This is an ASCII-only managed-payload protocol for POSIX Linux and macOS. It
does not claim general Unicode-path or Windows support.

A valid value is a nonempty `str` containing one or more `/`-separated
components. Each component contains only `A-Z`, `a-z`, `0-9`, `.`, `_`, or
`-`, is at most 255 ASCII bytes, and is not exactly `.` or `..`. The complete
path is at most 4095 ASCII bytes. Leading-dot names and otherwise valid names
ending in `..` are accepted. Supplied spelling is preserved exactly.

The contract rejects absolute, drive-like, and UNC forms; every backslash;
empty, repeated, leading, or trailing separators; exact `.` and `..`
components; NUL, controls, whitespace, non-ASCII text, unsupported
punctuation, and byte values. These protocol caps do not guarantee that every
concrete host path fits a filesystem's native limits.

A declared file-path set rejects exact duplicates, case-insensitive aliases,
and file-versus-directory prefix collisions. Alias detection applies at every
component prefix, so `A/x.txt` with `a/y.txt` is invalid. `a` with `a/b.txt`
is also invalid. Shared directory prefixes with identical spelling are valid.
The one canonical ordering is ascending ASCII/UTF-8 path bytes. H2a performs
no filesystem lookup, normalization, or case conversion of accepted spelling.

### H2b regular-file inventory contract

H2b implements the schema:

```text
research_platform.hpc.regular_file_inventory.v1
```

H2b owns a context-managed trusted-root opener. It receives an absolute,
lexically normalized path and opens it component-by-component from `/` using
directory descriptors and no-follow behavior. A symlink or non-directory at
any root or ancestor component is rejected. The opener returns a pinned
trusted-root handle containing the root descriptor and its device/inode
identity as private closure-held state. The supported handle API exposes the
device/inode identity and open/closed state, but not the raw owned descriptor.

The frozen H2b entry points are
`open_trusted_root(absolute_path)` and
`scan_regular_file_inventory(trusted_root, *, scope, excluded_prefixes=())`.
`open_trusted_root()` is a context-manager factory: calling it opens no
filesystem descriptor, and path validation plus descriptor opening occur only
on context entry. Direct `TrustedRoot` construction is unsupported and
rejected. Only that factory can create a trusted handle; the yielded exact
handle pins the validated descriptor until explicit `TrustedRoot.close()` or
context exit. Either path permanently retires the authority. Context exit does
not retry closing a handle already retired by explicit close; exceptional
context exit still closes or retires any live authority. Missing, foreign, or
close-failure descriptor states remain permanently retired, and foreign state
is never closed through the retired handle. Inventory admission requires that
exact live handle authority and rechecks its pinned descriptor identity.

Inventory traversal accepts that trusted-root handle and performs every
descendant enumeration and open descriptor-relatively. `Path.resolve()`,
string-joined traversal, and symlink-following canonicalization are forbidden.
Descendant device crossings are rejected. A caller that already possesses a
descriptor cannot claim root/ancestor-path validation unless it uses a
separately named API whose supplied descriptor is explicitly documented as
the trust boundary.

The canonical inventory body is exactly an object with:

- `schema_version`: the identifier above;
- `scope`: a safe synthetic or operation-defined identifier;
- `excluded_prefixes`: canonical, byte-sorted portable component prefixes;
- `files`: records sorted by portable path bytes.

`scope` is an exact string, 1 through 128 ASCII bytes, matching
`[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. It is preserved exactly; subclasses and
values requiring normalization are rejected.

Each file record has exactly:

```text
path
size_bytes
sha256
executable
```

`sha256` is the lowercase raw SHA-256 of exact file bytes. `executable` is
true when any POSIX execute bit is set. Timestamps, full mode, device, inode,
and link count are excluded from payload identity; they are retained only for
race and stability checks.

H2b admits at most the frozen H2a 100,000 observed container entries and
16 MiB of accumulated portable-path bytes per anchored pass. The completed
canonical body must also satisfy every H2a canonical limit. File sizes are
nonnegative signed-64 integers; hashing streams the pinned descriptor rather
than materializing file contents.

The canonical inventory bytes are the canonical JSON bytes of that body. Its
raw `sha256` is the canonical-inventory digest. The tree digest applies the
frozen tree domain and length framing to those same bytes. A digest wrapper,
if stored beside the body, is not part of the body.

Every included entry must be a regular file with link count exactly one. The
trusted-root opener rejects root and ancestor symlinks; descriptor-relative
traversal rejects descendant and broken symlinks, hard-linked files, FIFOs,
sockets, devices, and other special entries. A file is opened once for
hashing, with descriptor identity and metadata checked before and after
reading. Enumeration is repeated to detect additions, deletions, replacement,
and membership drift visible during the anchored passes. This detects
observable mutation; it is not an atomic filesystem snapshot.

Directories are not inventory records. An included logical directory,
including the root, must contain at least one included regular file after
excluded subtrees are pruned; empty logical directories are rejected.
Exclusions are explicit portable component prefixes, never inferred from
`.gitignore`. They are canonical and byte-sorted; overlapping, redundant,
exact-aliased, or case-aliased exclusions are rejected.
Each exclusion must be an exact validated portable-path value. A declared
prefix may be absent. When its boundary exists, H2b opens and verifies that
boundary descriptor-relatively as a same-device real directory, then prunes
it without traversing its contents. Symlinked, broken, regular-file, or
special-entry boundaries fail closed.

Source and release scopes must be named explicitly. H2 test fixtures use
synthetic scope names and must not pre-empt H12 by calling themselves
`public_release_payload_v1`. H4 still inventories before and after transfer.
Final release/export membership is H12 work.

H2b golden vectors freeze the body bytes, raw inventory digest, and tree digest
across different roots and creation orders. Cross-platform acceptance remains
required on Python 3.11 and 3.12, Linux, and macOS.

### H2c shared threat boundary

Any H2c implementation must operate only beneath an exact live `TrustedRoot`
representing the publication or claim parent. That pinned parent must be owned
by the effective user, provide owner read, write, and search access, not be
group- or other-writable, and retain its admitted device/inode identity.

Every H2c guarantee that relies on exclusive mutation authority—not only
deletion—requires a caller-controlled parent and cooperative writers. The
checks above do not prevent a malicious or uncooperative same-UID process, an
ACL-authorized writer, or unrestricted in-process memory or descriptor
manipulation from changing staged content, adding case aliases, replacing
entries, or manipulating cleanup targets.

Within that boundary, H2c remains responsible for descriptor-relative,
no-follow operation; symlink and path-race rejection; exact identity
validation; coordination among compliant Research Platform contenders; crash
residue; detection of observable replacement or mutation; and preservation of
foreign, replaced, ambiguous, or unexpected recovery evidence. POSIX provides
no portable compare-inode-and-unlink or compare-inode-and-rmdir operation.
H2c detects observable drift but does not claim prevention or complete
detection against a writer with equivalent mutation authority. Native
exact-name no-replace behavior is supplied by the operating-system publication
syscall, not by the cooperative alias scan.

### H2c1 no-replace publication contract

H2c1 distinguishes exclusive staging creation, completed regular-file
publication, and completed-directory publication. Remote transfer promotion
remains H4.

#### Public authority and population surface

The H2c1 module-level entry points are exactly:

```text
open_exclusive_staged_file
open_exclusive_staged_directory
publish_completed_file
publish_completed_directory
cleanup_owned_staging
```

Their exact signatures are:

```python
open_exclusive_staged_file(
    trusted_root: TrustedRoot,
    *,
    destination: PortableRelativePath,
) -> AbstractContextManager[StagedFileHandle]

open_exclusive_staged_directory(
    trusted_root: TrustedRoot,
    *,
    destination: PortableRelativePath,
) -> AbstractContextManager[StagedDirectoryHandle]

publish_completed_file(
    trusted_root: TrustedRoot,
    staging: StagedFileHandle,
) -> PublicationResult

publish_completed_directory(
    trusted_root: TrustedRoot,
    staging: StagedDirectoryHandle,
) -> PublicationResult

cleanup_owned_staging(
    trusted_root: TrustedRoot,
    staging: StagedFileHandle | StagedDirectoryHandle,
) -> StagingCleanupResult
```

The exact live `TrustedRoot` supplied to `publish_completed_file`,
`publish_completed_directory`, or `cleanup_owned_staging` must be the same
authority bound to the staging handle. Every authority and portable-path
argument, and every staging-handle argument, uses exact-type admission;
hostile subclasses are rejected.

The public authority, value, and failure names are exactly:

```text
PublicationState
StagingState
StagingCleanupState
PublicationResult
StagingCleanupResult
PublicationEntryIdentity
NamespaceEvidence
PublicationRecoveryEvidence
StagingCleanupRecoveryEvidence
StagedFileHandle
StagedDirectoryHandle
StagingLifecycleError
StagingAuthorityError
PublicationError
PublicationValidationError
PublicationCapabilityError
PublicationCollisionError
StagingAdmissionError
PublicationDurabilityError
PublicationOutcomeUncertainError
PublicationNamespaceConflictError
PublicationNamespaceUncertainError
StagingCleanupError
DescriptorRetirementObservation
DescriptorRetirementIdentity
DescriptorRetirementRecord
DescriptorRetirementEvidence
DescriptorRetirementError
```

The five descriptor-retirement names in that complete frozen surface are
implemented and exported by this working snapshot. Hosted acceptance remains
pending until the eventual commit's CI succeeds.

Every handle is exact and opaque. Every result, identity, and evidence value
is exact, immutable, and read-only. Direct construction, subclass admission,
copying, and serialization are rejected. Public failure objects are catchable
through the frozen hierarchy below, and their transport fields are read-only.
The exact public enum ABI is:

```python
class PublicationState(str, Enum):
    NOT_COMMITTED = "not_committed"
    COMMITTED_DURABLE = "committed_durable"
    COMMITTED_DURABILITY_UNCERTAIN = "committed_durability_uncertain"
    COMMIT_OUTCOME_UNCERTAIN = "commit_outcome_uncertain"

class StagingState(str, Enum):
    OPEN = "open"
    SEALED = "sealed"
    NOT_COMMITTED = "not_committed"
    PUBLISHED = "published"
    DISCARDED = "discarded"
    RETIRED = "retired"

class StagingCleanupState(str, Enum):
    NOT_DISCARDED = "not_discarded"
    DISCARDED_DURABLE = "discarded_durable"
    DISCARDED_DURABILITY_UNCERTAIN = "discarded_durability_uncertain"
    DISCARD_OUTCOME_UNCERTAIN = "discard_outcome_uncertain"
```

The exact descriptor-retirement observation ABI is:

```python
class DescriptorRetirementObservation(str, Enum):
    CLOSED = "closed"
    ALREADY_ABSENT = "already_absent"
    FOREIGN_PRESERVED = "foreign_preserved"
    UNINSPECTABLE = "uninspectable"
    CLOSE_OUTCOME_UNCERTAIN = "close_outcome_uncertain"
```

The exact read-only lifecycle observations are:

```text
StagedFileHandle.state -> StagingState
StagedDirectoryHandle.state -> StagingState
```

These properties are observational only. They expose no descriptor, host
path, or filesystem authority. The handles expose only these population and
sealing operations:

```text
StagedFileHandle.write(chunk: bytes) -> None
StagedFileHandle.seal(*, executable: bool = False) -> None

StagedDirectoryHandle.mkdir(path: PortableRelativePath) -> None
StagedDirectoryHandle.write_file(
    path: PortableRelativePath,
    chunks: Iterable[bytes],
    *,
    executable: bool = False,
) -> None
StagedDirectoryHandle.seal(*, scope: str) -> None
```

Every chunk must be exact `bytes`. Streaming writes handle partial writes and
`EINTR`, reject zero progress, and keep total file sizes within the signed-64
range. Directory entries remain bounded by the frozen H2a/H2b item and
portable-path limits. Every supplied path is an exact
`PortableRelativePath`; directory parents are created only by an explicit
`mkdir` operation.

No raw descriptor or host path is exposed. No caller-created entry is adopted,
and no callback or boolean may assert ownership. H2c creates regular files as
exact mode `0600` or `0700` and directories as exact mode `0700`; it does not
trust `umask`, and it sets and verifies final modes. Executable identity in
this foundation changes only the owner execute bit.

The two `open_exclusive_staged_*` entry points return private context-manager
objects; the context managers themselves are not public authorities.
Their creation is lazy: calling either factory opens or creates nothing.
Context entry performs, in order:

```text
1. validate exact arguments, the parent authority, and required capabilities
2. attempt exclusive creation of the deterministic staging reservation
3. verify the opened and still-named staging identity
4. yield a handle only after stable staging admission
```

The first authoritative destination and case-alias scan occurs at `seal`,
while the reservation is held and a cleanup-capable handle has already been
yielded. If the deterministic reservation already exists before this
invocation creates anything, context entry raises `PublicationCollisionError`
with `NOT_COMMITTED`, the validated destination, and `evidence=None`; it
performs no mutation. If stable admission fails after this invocation creates
the reservation but before yielding, it preserves the entry, permanently
retires the provisional authority, returns no handle, and raises
`StagingAdmissionError`. That error carries `NOT_COMMITTED`, the deterministic
staging leaf, whether an entry may remain, and bounded evidence when a stable
provisional identity was obtained. It may carry `evidence=None` only when no
stable provisional identity could be obtained. Uncertain pre-yield residue is
never deleted automatically. Non-null pre-yield admission evidence records
destination and namespace observations as `not_attempted` and parent `fsync`
as `not_attempted`; its source fields describe only the safely observed
provisional staging identity.

If the seal-time scan finds an existing exact destination or differently
spelled case alias, the handle enters `NOT_COMMITTED`,
`PublicationCollisionError` is raised with non-null bounded evidence, and the
handle retains its immutable ledger and one cleanup authorization. Its
admitted handle descriptor batch remains pending, and the collision error is
delivered without consuming that batch.

The two private `NOT_COMMITTED` retirement-batch origins are exactly:

```text
seal_time_collision:
    public_state: NOT_COMMITTED
    handle_retirement_batch: pending
    immutable_ledger: preserved
    cleanup_authorization: preserved

terminal_publication_proven_not_committed:
    public_state: NOT_COMMITTED
    handle_retirement_batch: consumed
    operation_retirement_batch: consumed
    consumption_precedes: PublicationError delivery
    immutable_ledger: preserved
    cleanup_authorization: preserved
```

For a seal-time collision, context exit before cleanup runs the frozen
three-branch pending-batch matrix below, consumes the pending batch exactly
once, and preserves state, ledger, and cleanup authorization. If
`cleanup_owned_staging()` passes its pre-attempt lifecycle and bound-root
admission before context exit, it consumes and detaches the pending handle
batch exactly once before opening fresh cleanup-operation descriptors. Any
old-batch retirement anomaly remains available for the frozen cleanup error
composition. A pre-attempt lifecycle or authority rejection leaves the batch
pending and consumes no cleanup authorization. Once cleanup is admitted,
context exit performs no later descriptor inspection, close, or retirement
attempt.

When `publish_completed_file()` or `publish_completed_directory()` reaches a
proven `NOT_COMMITTED` outcome, its terminal publication path consumes the
complete handle and operation retirement batch before delivering its
`PublicationError`. The handle remains `NOT_COMMITTED`, its immutable ledger
and one cleanup authorization remain, and any retirement anomaly attaches to
that exact publication error under the frozen private-provenance rule.
Context exit performs no descriptor inspection, close, or retirement attempt
for that already-consumed batch. Without a body exception it returns normally;
with a body exception it propagates that exception unchanged because no
second retirement operation or new retirement anomaly occurs. Later cleanup
remains available through fresh descriptors and full ledger/namespace
revalidation.

Every private `NOT_COMMITTED` retirement batch changes privately and
irreversibly from `pending` to `consumed`; no batch may be consumed twice. The
marker is private, immutable to callers, non-exported, nonserializable, and
carries no filesystem authority. It adds no public state, property, enum,
result field, evidence field, or API. Both origins expose only the existing
public `NOT_COMMITTED` lifecycle state and preserve identical ledger and
cleanup authorization; the only difference visible in context behavior is
whether retirement remains pending. Context exit consults only private
lifecycle provenance and never guesses from descriptor numbers.
Descriptor-number reuse cannot recreate a consumed batch. Cleanup always uses
fresh descriptors through the same exact live `TrustedRoot`.

Normal or exceptional context exit from `OPEN` or `SEALED` is abandonment: it
runs the frozen descriptor-retirement protocol once, enters `RETIRED`, and
preserves the staging entry as recovery evidence. Context exit from
`NOT_COMMITTED` follows the private origin partition above: it runs retirement
only for a pending seal-time batch and performs no second retirement for a
terminal-publication batch already consumed. It never performs cleanup.
Context exit after any terminal operation performs no second terminal
operation.

Handles never auto-publish or auto-delete through context exit or garbage
collection. Finalization may perform only the same best-effort descriptor
retirement and must preserve the staging entry; it never publishes or deletes.
Correctness never depends on garbage collection.

Every terminal publish or cleanup path retires all live descriptor authority
owned by the staging handle. `PUBLISHED` and `DISCARDED` remain the truthful
lifecycle labels after that descriptor retirement. A publication attempt
proven `NOT_COMMITTED` logically retires its live descriptor authority and
permanently loses publication authority, but retains only its immutable
staging identity and ledger as authorization for the one permitted cleanup
attempt; cleanup reopens and revalidates the entry descriptor-relatively
through the same exact `TrustedRoot`.

A proven commit remains `PUBLISHED` after descriptor retirement even when
publication-parent durability is uncertain, a namespace-conflict failure is
raised, or namespace verification remains uncertain. A proven durable cleanup
remains `DISCARDED`. Abandoned `OPEN` or `SEALED` handles, handles whose
staging authority becomes foreign, contradictory, or uninspectable,
commit-outcome-uncertain handles, and all failed or non-durable cleanup
handles are permanently `RETIRED`. A proven `NOT_COMMITTED` handle retains
only its one cleanup authorization. Context exit
after any terminal operation performs no second publish, cleanup, or other
filesystem mutation or second descriptor-retirement attempt. The terminal
retirement batch detaches every owned descriptor slot and consumes every
retirement attempt before the terminal result or failure is delivered, so
context exit has no residual descriptor authority to inspect or close.
Neither a terminal nor a retired handle can be revived by reuse of a former
descriptor number.

#### Direct-sibling staging topology

One pinned publication parent contains both direct sibling leaf entries:

```text
<publication-parent>/<destination-leaf>
<publication-parent>/.rp-stage-v1-<alias-digest>
```

The destination is exactly one `PortableRelativePath` component. H2c1 creates
the staging entry internally: a regular file uses descriptor-relative
exclusive creation; a directory uses descriptor-relative exclusive `mkdir`
followed by a no-follow open. A caller-provided existing path, descriptor,
boolean ownership assertion, or arbitrary filesystem entry is never accepted
as staging authority.

The staging authority is an exact, opaque, nonconstructible, noncopyable
lifecycle handle. It binds the originating exact `TrustedRoot`, parent
device/inode, destination spelling and alias key, staging device/inode/type,
original ownership evidence, and lifecycle state. Its raw descriptor and host
path are not exposed through the supported API. Only H2c-owned
descriptor-relative operations populate and seal the staging entry; only a
sealed entry may be published.

The staging lifecycle states are:

```text
OPEN
SEALED
NOT_COMMITTED
PUBLISHED
DISCARDED
RETIRED
```

Exclusive creation enters `OPEN`; successful writes and directory creation
remain `OPEN`; successful sealing enters `SEALED`; a publication attempt
proven not committed enters `NOT_COMMITTED`; a proven commit enters
`PUBLISHED`; successful durable cleanup enters `DISCARDED`; and foreign,
replaced, contradictory, abandoned, or uninspectable authority enters
`RETIRED`. Cleanup failure or partial cleanup also enters `RETIRED`.

Population and sealing operations are permitted only in `OPEN`; sealing ends
all content mutation. Publication is permitted only from `SEALED`, and each
handle gets at most one publication attempt. `NOT_COMMITTED` cannot publish
again but remains eligible for cleanup. Cleanup is permitted only from:

```text
OPEN
SEALED
NOT_COMMITTED
```

`PUBLISHED`, `DISCARDED`, and `RETIRED` are terminal. Repeated publication or
cleanup after a terminal state is rejected without filesystem mutation. An
operation that is invalid for the current lifecycle state raises
`StagingLifecycleError` before filesystem access, consumes no publication or
cleanup attempt, and leaves the handle unchanged. Exact-type violations remain
`TypeError`. The error reports the unchanged current `StagingState`; an
already `PUBLISHED` handle therefore remains and reports `PUBLISHED`, never
`NOT_COMMITTED`. A terminal entry point first performs non-I/O exact-type and
object-identity validation, including its recorded live/retired state, that
the supplied `TrustedRoot` is the authority bound to the handle; this
bound-root validation precedes lifecycle admission and does not rewrite
lifecycle state. Full descriptor liveness and identity validation occurs only
after lifecycle admission.

Every terminal publish or cleanup path retires the handle's descriptor
authority. Context exit after a terminal operation performs no second publish,
cleanup, filesystem mutation, descriptor inspection, close, or retirement
attempt. The terminal result or failure is delivered only after the retirement
batch has consumed every owned descriptor attempt. Abandoned `OPEN` or
`SEALED` handles, foreign or
commit-outcome-uncertain handles, and all failed or non-durable cleanup
handles are permanently retired. A terminal publication proven
`NOT_COMMITTED` is the sole exception among terminal publication and cleanup
outcomes: its complete terminal retirement batch is consumed, but its one
cleanup authorization remains. A seal-time collision is not a terminal
publication; its admitted handle batch remains pending until an admitted
cleanup, context exit, or finalization consumes it. Finalization follows the
already frozen unraisable-error and cleanup-authorization-loss rules. No
terminal or retired handle can be revived through descriptor-number reuse. No
terminal operation is silently retried. These entry points, handles, results,
states, and private batch provenance are frozen here and implemented and
exported by this working H2c1 snapshot. Hosted acceptance remains pending.

#### Cooperative case-alias reservation

The frozen staging domain is:

```python
PUBLICATION_STAGING_DOMAIN = b"research-platform:hpc:publication-staging:v1\0"
```

The reservation key is the lowercase ASCII destination leaf encoded as exact
ASCII bytes. The alias digest is the H2a domain-separated SHA-256:

```text
SHA256(
  PUBLICATION_STAGING_DOMAIN
  || uint64be(len(reservation_key))
  || reservation_key
)
```

The resulting sibling name is:

```text
.rp-stage-v1-<64-lowercase-hex-digest>
```

Thus `Foo` and `foo` use the same cooperative reservation on Linux and macOS.
Before rename, the staging entry is the cooperative case-alias reservation.
The implementation acquires it by exclusive staging creation. The first
authoritative descriptor-relative parent enumeration occurs during `seal`,
while the reservation is held; an existing exact destination or differently
spelled case alias then causes the admitted handle to enter `NOT_COMMITTED`.
Every compliant Research Platform publisher must use this reservation.

A successful native rename atomically moves the staged inode to the
destination, so the staging leaf is then expected to be absent and the
destination becomes the cooperative namespace authority. Another compliant
contender may subsequently recreate the staging leaf, but it must enumerate
the parent and reject the existing exact destination or differently spelled
alias before attempting publication. Post-publication validation checks the
destination identity and alias set; it does not require the old staging name
to remain present. A foreign newly created staging entry present at the first
authoritative post-call source observation does not change a commit already
proven by the original inode at the destination.

The complete `.rp-stage-v1-` prefix is reserved for H2c internals. A requested
destination whose lowercase ASCII spelling begins with that prefix is
rejected. Scanning alone does not prevent an arbitrary external writer from
introducing an alias, and the accepted destination spelling is never
normalized or rewritten.

#### Native no-replace backends

The only H2c1 native backends and flags are:

```text
Linux:
int renameat2(
  int olddirfd,
  const char *oldpath,
  int newdirfd,
  const char *newpath,
  unsigned int flags
)
RENAME_NOREPLACE = 1

macOS:
int renameatx_np(
  int fromfd,
  const char *from,
  int tofd,
  const char *to,
  unsigned int flags
)
RENAME_EXCL = 4
```

The implementation uses `ctypes.CDLL(None, use_errno=True)`, `ctypes.c_int`
for descriptors and the return value, `ctypes.c_char_p` for the exact ASCII
leaf names, and `ctypes.c_uint` for flags. Source and destination parent
descriptors are anchored. Capability and argument validation occurs before
staging writes where possible; filesystem support is determined by the actual
native publication operation. Errno is captured immediately, and anchored
source and destination evidence is inspected after success or failure.

H2c1 explicitly prohibits `os.rename`, `os.replace`, overwrite-capable
fallback, destination unlinking, check-then-rename, copy-and-delete, merging,
architecture-specific raw syscall-number fallback, `/proc/self/fd`, and
pathname re-resolution. `EINVAL` may be classified as unsupported only after
arguments, flags, descriptors, and names are proven valid and post-call
evidence proves `NOT_COMMITTED`.

#### Staging ledger, durability, and staged-tree validation

Canonical inventory equality alone cannot detect replacement by a different
inode containing identical bytes. H2c1 therefore maintains a noncanonical
internal ledger for every staged file and directory. Each ledger entry retains
the safety metadata required for stability checks, including device, inode,
entry type, link count, effective ownership, complete relevant mode, size, and
stability timestamps where available.

Before and after every file or directory flush, H2c1 compares the opened
descriptor with the ledger and compares the still-named parent entry with the
same ledger. Same-content inode replacement and any unexpected change to
device, type, link count, ownership, mode, size, or stability metadata are
rejected. H2b canonical inventory bytes remain the payload identity, but
canonical equality is never used alone as mutation authority.

H2c v1 uses standard successful `fsync` semantics on Linux and macOS.
`COMMITTED_DURABLE` means:

> Every protocol-required regular-file, staged-directory, and
> publication-parent `fsync` returned successfully, with required identity
> checks also passing.

For this durability definition, the required identity checks are the
destination, pinned publication-parent, and flush-target checks that prove the
original staged identity was committed and the required objects were flushed.
Post-commit source-reservation and case-alias observations are namespace
verification transported separately. Their anomaly may raise
`PublicationNamespaceConflictError` or
`PublicationNamespaceUncertainError` while the already proven durability
state remains `COMMITTED_DURABLE`.

That is protocol/OS-filesystem durability evidence. It is not a promise
against physical power loss, controller or drive-cache behavior, dishonest
storage firmware, network-filesystem behavior, or storage failure after the
sync operation returns. H2c v1 does not require or invoke `F_FULLFSYNC`; a
stronger physical-flush profile requires a future protocol version or
separately frozen capability contract.

If a required directory `fsync` is known to be unsupported, publication fails
closed before commit. If that failure is discovered only after a proven
rename, the result is committed with durability uncertainty.

Completed regular-file publication performs, in order:

1. a complete bounded streaming write with explicit partial-write, zero-write,
   and interruption handling;
2. sealing with exact mode, identity, stability, and ledger verification;
3. ledger comparison before and after file `fsync`;
4. immediate pre-publication source, destination, parent, ledger, and alias
   revalidation;
5. the native no-replace rename; and
6. publication-parent `fsync`.

Completed-directory publication admits only same-device real directories and
single-link regular files. It rejects symlinks, hard links, FIFOs, sockets,
devices, other special entries, aliases, and empty included directories. It
then performs, in order:

1. an H2b-equivalent bounded inventory plus a complete safety ledger before
   flushing;
2. ledger comparison before and after each regular-file flush;
3. ledger comparison before and after deepest-first directory flushes,
   excluding the staged root;
4. separate ledger comparison before and after the staged-root flush;
5. an identical canonical reinventory and an identical safety ledger;
6. immediate pre-publication source, destination, parent, ledger, and alias
   revalidation;
7. the native no-replace rename; and
8. publication-parent `fsync`.

The directory protocol detects observable mutation; it does not claim an
atomic directory snapshot.

#### Publication result and error mapping

The exact immutable normal-result fields are:

```text
PublicationResult:
    state: PublicationState
    destination: PortableRelativePath
    destination_identity: PublicationEntryIdentity
    namespace_evidence: NamespaceEvidence

StagingCleanupResult:
    state: StagingCleanupState
    staging: PortableRelativePath
    discarded_identity: PublicationEntryIdentity
    namespace_evidence: NamespaceEvidence
```

A `PublicationResult` exists only with `COMMITTED_DURABLE`; its destination is
an exact one-component portable path and its destination identity equals the
admitted staging identity. Its namespace evidence is the complete
`no_conflict` form defined below. A `StagingCleanupResult` exists only with
`DISCARDED_DURABLE`; its staging leaf is exact, its discarded identity is the
admitted staging identity proven removed, and its namespace evidence is also
complete and conflict-free.

The exact immutable entry-identity and bounded-evidence fields are:

```text
PublicationEntryIdentity:
    device: int
    inode: int
    entry_type: Literal["regular_file", "directory", "symlink", "fifo", "socket", "character_device", "block_device", "other"]
    link_count: int
    owner_uid: int
    mode: int
    size_bytes: int

NamespaceEvidence:
    namespace_observation: Literal["not_attempted", "no_conflict", "complete_conflict", "bounded_conflict", "uninspectable"]
    conflicting_aliases: tuple[PortableRelativePath, ...]
    conflicting_alias_count: int | None
    aliases_complete: bool

PublicationRecoveryEvidence:
    staging_identity: PublicationEntryIdentity
    source_observation: Literal["not_attempted", "exact", "absent", "foreign", "replaced", "contradictory", "uninspectable"]
    observed_source_identity: PublicationEntryIdentity | None
    destination_observation: Literal["not_attempted", "exact", "absent", "foreign", "replaced", "contradictory", "uninspectable"]
    observed_destination_identity: PublicationEntryIdentity | None
    namespace_evidence: NamespaceEvidence
    parent_fsync: Literal["not_attempted", "succeeded", "failed", "uncertain"]
    native_errno: int | None

StagingCleanupRecoveryEvidence:
    staging_identity: PublicationEntryIdentity
    root_observation: Literal["exact", "owned_partial", "absent", "foreign", "replaced", "contradictory", "malformed", "uninspectable"]
    observed_root_identity: PublicationEntryIdentity | None
    remaining_expected_entries: int | None
    namespace_evidence: NamespaceEvidence
    parent_fsync: Literal["not_attempted", "succeeded", "failed", "uncertain"]
    native_errno: int | None
```

All numeric fields require exact `int`, never `bool`. Device, inode, and owner
identities are from `0` through `2**64 - 1`; link count is from `1` through
`2**63 - 1`; complete mode is from `0` through `0o177777`; and size is from
`0` through `2**63 - 1`. An original staged or durably published identity has
entry type `regular_file` or `directory`; observed foreign identities use the
complete finite entry-type list without opening a symlink or special entry.

An observed identity is present only when descriptor-relative inspection
obtained a stable identity for that named entry; `not_attempted`, absent, or
uninspectable observations carry `None`. `not_attempted` is permitted only in
pre-yield `StagingAdmissionError` evidence; every post-admission terminal
publication evidence forbids it. `native_errno` is `None` or an exact integer
from `1` through `2**31 - 1`. `remaining_expected_entries` is `None` unless
the original staging root and known expected ledger residue are proven;
otherwise it is an exact integer from `0` through the frozen 100,000-item
bound.

For publication evidence, `exact` requires the observed identity to equal the
staging identity; `absent`, `uninspectable`, and `contradictory`
require `None`; and stable `foreign` or `replaced` observations require a
non-null identity unequal to the staging identity. For cleanup evidence,
`exact` likewise requires the observed root to equal the staging identity;
`absent`, `uninspectable`, and `contradictory` require `None`; and stable
`foreign` or `replaced` observations require a non-null unequal identity.
`malformed` carries the stable observed identity when descriptor-relative
metadata is available and otherwise carries `None`.

Root identity and membership state are represented only by `exact`,
`owned_partial`, `absent`, `foreign`, `replaced`, `contradictory`,
`malformed`, or `uninspectable`. Sibling case aliases are represented only by
`NamespaceEvidence`, orthogonally to the root observation. An original
unchanged root with a conflicting alias uses root observation `exact` plus
conflict namespace evidence; an owned partial root with a conflicting alias
uses `owned_partial` plus conflict namespace evidence. A foreign, replaced,
malformed, or uninspectable root remains represented independently of any
simultaneously observed alias evidence.

`owned_partial` carries the current observed root identity. Its stable root
authority is equality of device, inode, entry type, and owner UID with the
original staging identity. Its mode must remain exactly equal to the admitted
safe staging-root mode; cleanup authorizes no `chmod`. Only link count, size,
and applicable stability metadata may differ, and only when the difference is
explained by removals this cleanup attempt has already proven. Every remaining
entry must be an expected ledger member; every removed entry must be proven
absent; and no unexpected entry may exist. Its
`remaining_expected_entries` is the exact bounded count of current expected
ledger members.

`NamespaceEvidence` is immutable and bounded. Its forms are exactly:

- `not_attempted`: empty alias tuple, count `None`, and `aliases_complete`
  exact `False`;
- `no_conflict`: empty alias tuple, count `0`, and `aliases_complete` exact
  `True`;
- `complete_conflict`: a nonempty canonical alias tuple of at most 100,000
  entries, an exact count equal to its length, and `aliases_complete` exact
  `True`;
- `bounded_conflict`: exactly 100,000 canonical alias entries, count `None`,
  and `aliases_complete` exact `False`; and
- `uninspectable`: only the zero through 100,000 safely observed canonical
  alias entries, count `None`, and `aliases_complete` exact `False`.

Every alias is an exact one-component `PortableRelativePath`, exact-spelling
unique, different from the destination or staging spelling,
lowercase-ASCII-equivalent to it, and sorted by portable-path bytes. The
evidence contains no raw descriptor, host path, credential, caller- or
OS-supplied free-form text, mutable mapping, or unbounded collection. The
staging identity is distinct from later observations, so same-content inode
replacement cannot be disguised as the original authority.

The exact immutable descriptor-retirement evidence schema is:

```text
DescriptorRetirementIdentity:
    device: int
    inode: int
    entry_type: Literal[
        "regular_file",
        "directory",
        "symlink",
        "fifo",
        "socket",
        "character_device",
        "block_device",
        "other",
    ]
    owner_uid: int

DescriptorRetirementRecord:
    ordinal: int
    role: Literal[
        "traversal_entry",
        "traversal_directory",
        "traversal_parent",
        "operation_staging",
        "operation_parent",
        "handle_staging",
        "handle_parent",
    ]
    observation: DescriptorRetirementObservation
    close_attempted: bool
    admitted_identity: DescriptorRetirementIdentity | None
    observed_identity: DescriptorRetirementIdentity | None
    error_errno: int | None

DescriptorRetirementEvidence:
    records: tuple[DescriptorRetirementRecord, ...]
```

The seven descriptor roles have these exact acquisition-purpose meanings:

```text
traversal_entry:
    transient operation-local descriptor opened for one included regular-file leaf beneath a staged directory during population, validation, flushing, or cleanup
traversal_directory:
    transient operation-local descriptor opened for one descendant directory beneath the staged root during population, validation, flushing, or cleanup; excludes the top-level staged-root descriptor
traversal_parent:
    transient operation-local descriptor acquired specifically as the stable immediate-parent authority for creating, verifying, or removing a descendant traversal entry during population, validation, flushing, or cleanup; excludes publication-parent descriptors
operation_staging:
    fresh operation-local descriptor reopened for the top-level staged file or staged-directory root during publication or cleanup; distinct from the descriptor retained from admission
operation_parent:
    fresh operation-local descriptor for the top-level publication parent used by a terminal publication or cleanup operation
handle_staging:
    staging file/root descriptor acquired during staging admission and intended for the public handle; it receives this role immediately upon acquisition, remains provisional before yield, is retained by the handle after successful admission, and keeps this role if pre-yield admission fails
handle_parent:
    publication-parent descriptor acquired during staging admission and intended for the public handle; it receives this role immediately upon acquisition, remains provisional before yield, is retained by the handle after successful admission, and keeps this role if pre-yield admission fails
```

During staged-directory population, `write_file()` uses `traversal_entry` for
the descriptor returned by exclusive child-file creation, `mkdir()` uses
`traversal_directory` for the no-follow descriptor opened for the newly
created descendant directory, and either operation uses `traversal_parent` for
a separately acquired stable immediate-parent descriptor.

Population roles are assigned at descriptor acquisition according to
acquisition purpose, before successful ledger admission is known. Role
assignment alone does not prove successful ledger inclusion or authorize
cleanup of a newly created namespace entry.

This population coverage broadens only the acquisition-purpose meaning of the
three existing traversal roles. It adds no descriptor role, role ordering,
public state, enum member, property, result or evidence field, API, lifecycle
transition, or descriptor bound.

Every owned descriptor maps to exactly one role, and one descriptor can never
generate multiple records. Role is fixed by acquisition purpose and never
changes because the descriptor is later used for another check. A top-level
operation descriptor retains its `operation_*` role even when used to begin
traversal. A descendant directory serving as the parent of its children
remains `traversal_directory` unless a separate descriptor was acquired
specifically as `traversal_parent`. Absent roles emit no record. Acquisition
and unwinding must be structured so one retirement batch never owns two
descriptors with the same role.

Role is determined by acquisition purpose, not by whether a public handle was
ultimately yielded. Descriptor-retirement evidence for a failed pre-yield
admission may therefore contain `handle_staging`, `handle_parent`, or both.
Those provisional admission descriptors never use `operation_staging`,
`operation_parent`, or a traversal role. Each still maps to exactly one role
and one record; the frozen role order, not acquisition order, determines their
evidence-tuple order. The existing admission/abandonment bound of two
descriptors is unchanged. No public handle is returned after any admission
failure. When stable admission fails after this invocation created the
reservation, the exact `StagingAdmissionError` remains primary and receives
retirement evidence only under the frozen private-provenance rules; other
pre-yield failures preserve their already frozen exact subtype.

One evidence value is complete for one retirement batch. Descriptors already
verifiably retired before that batch are excluded. Records are immutable and
ordered by the frozen role order above; each role occurs at most once. A role
cannot be reused until the prior descriptor authority for that role is
consumed. `ordinal` is an exact zero-based integer equal to the record's tuple
position. Evidence contains from one through seven records, and at least one
record is not `CLOSED`. Evidence created after a filesystem outcome is already
proven contains at most four records drawn only from `operation_staging`,
`operation_parent`, `handle_staging`, and `handle_parent`.

Every numeric field is exact `int`, never `bool`. Device, inode, and UID range
from `0` through `2**64 - 1`. `error_errno` is `None` or an exact integer from
`1` through `2**31 - 1`. Direct-construction bypass, hostile subclasses,
mutation, copying, deep-copying, and serialization are rejected as they are
for other H2c1 evidence authorities. Retirement evidence exposes no raw
descriptor, host path, credential, mutable mapping, unbounded collection, or
free-form operating-system text.

Observation consistency is exact:

- `CLOSED` requires `close_attempted=True`, a non-null admitted identity, an
  observed identity equal to it, and `error_errno=None`.
- `ALREADY_ABSENT` requires `close_attempted=False`, no observed identity, and
  the pre-close `EBADF` as its bounded errno. Its admitted identity may be
  `None` only for a pre-admission descriptor that never obtained stable
  identity.
- `FOREIGN_PRESERVED` is valid only through one of the two mutually exclusive
  branches frozen below.
- `UNINSPECTABLE` requires `close_attempted=False` and no observed identity.
  An `OSError` contributes its bounded errno; a non-`OSError` contributes
  `None`.
- `CLOSE_OUTCOME_UNCERTAIN` requires `close_attempted=True`, a non-null
  admitted identity, and an equal pre-close observed identity. An `OSError`
  contributes its bounded errno; a non-`OSError` contributes `None`.

The two `FOREIGN_PRESERVED` branches are exactly:

```text
private_generation_mismatch:
    observation = FOREIGN_PRESERVED
    close_attempted = False
    admitted_identity = locally_snapshotted_admitted_identity
    observed_identity = None
    error_errno = None

matching_generation_unequal_stable_identity:
    observation = FOREIGN_PRESERVED
    close_attempted = False
    admitted_identity = non-null
    observed_identity = non-null
    observed_identity != admitted_identity
    error_errno = None
```

For a private-generation mismatch, `admitted_identity` equals the locally
snapshotted admitted identity and may be `None` only for a pre-admission
descriptor that never obtained stable identity. The descriptor is neither
inspected nor closed. For a matching generation plus unequal stable `fstat`
identity, both identities are non-null and unequal, and the descriptor is not
closed. If the generation matches but no admitted identity exists, or if no
stable comparison can be made, the record is `UNINSPECTABLE`, never the
unequal-identity branch.

The exact catchable failure hierarchy is:

```text
StagingLifecycleError -> Exception
StagingAuthorityError -> Exception
PublicationError -> Exception
PublicationValidationError -> PublicationError
PublicationCapabilityError -> PublicationError
PublicationCollisionError -> PublicationError
StagingAdmissionError -> PublicationError
PublicationDurabilityError -> PublicationError
PublicationOutcomeUncertainError -> PublicationError
PublicationNamespaceConflictError -> PublicationError
PublicationNamespaceUncertainError -> PublicationError
StagingCleanupError -> Exception
DescriptorRetirementError -> Exception
```

The exact read-only failure transport fields are:

```text
StagingLifecycleError:
    state: StagingState
    operation: Literal["write", "mkdir", "write_file", "seal", "publish", "cleanup"]
    retirement_evidence: DescriptorRetirementEvidence | None

StagingAuthorityError:
    state: StagingState
    operation: Literal["publish", "cleanup"]
    retirement_evidence: DescriptorRetirementEvidence | None

PublicationError:
    state: PublicationState
    evidence: PublicationRecoveryEvidence | None
    destination: PortableRelativePath | None
    retirement_evidence: DescriptorRetirementEvidence | None

StagingAdmissionError:
    staging: PortableRelativePath
    entry_may_remain: bool

StagingCleanupError:
    state: StagingCleanupState
    evidence: StagingCleanupRecoveryEvidence
    staging: PortableRelativePath
    retirement_evidence: DescriptorRetirementEvidence | None

DescriptorRetirementError:
    state: StagingState
    operation: Literal[
        "write",
        "mkdir",
        "write_file",
        "seal",
        "publish",
        "cleanup",
        "context_exit",
        "finalization",
    ]
    destination: PortableRelativePath
    staging: PortableRelativePath
    terminal_result: PublicationResult | StagingCleanupResult | None
    retirement_evidence: DescriptorRetirementEvidence
```

`DescriptorRetirementError` destination and staging are exact one-component
`PortableRelativePath` values.

The only valid `DescriptorRetirementError` cross-field tuples are:

```text
otherwise_successful_publication:
    operation: "publish"
    state: PUBLISHED
    terminal_result: exact PublicationResult
    terminal_result.state: COMMITTED_DURABLE

otherwise_successful_cleanup:
    operation: "cleanup"
    state: DISCARDED
    terminal_result: exact StagingCleanupResult
    terminal_result.state: DISCARDED_DURABLE

population_or_sealing:
    operation: Literal["write", "mkdir", "write_file", "seal"]
    state: RETIRED
    terminal_result: None

context_exit:
    operation: "context_exit"
    state: Literal[RETIRED, NOT_COMMITTED]
    terminal_result: None

finalization:
    operation: "finalization"
    state: RETIRED
    terminal_result: None
```

No other combination is valid. `terminal_result` is an exact
`PublicationResult` if and only if the otherwise-successful publication tuple
applies, and it is an exact `StagingCleanupResult` if and only if the
otherwise-successful cleanup tuple applies; otherwise it is `None`.
Population or sealing uses the standalone form only when the operation
otherwise had no primary H2c1 error.

Every pre-yield admission failure remains an exact `StagingAdmissionError`;
any retirement anomaly attaches through its inherited `retirement_evidence`.
There is no standalone admission-stage `DescriptorRetirementError`. A
`publish` operation cannot carry `NOT_COMMITTED`, uncertainty, or a null
result through `DescriptorRetirementError`; those outcomes retain their
existing `PublicationError` subtype with attached retirement evidence. A
`cleanup` operation cannot carry cleanup uncertainty through
`DescriptorRetirementError`; it retains `StagingCleanupError` with attached
retirement evidence.

The `context_exit` plus `NOT_COMMITTED` tuple is valid only for the
seal-time-collision origin while its private handle-retirement batch is still
pending. An already-consumed terminal-publication origin cannot create a new
`DescriptorRetirementError` at context exit.

The nullable retirement-evidence field is `None` when no retirement anomaly
occurred. Retirement evidence may be attached to an existing error only when
all of these conditions hold:

1. its exact runtime type is one of `StagingLifecycleError`,
   `StagingAuthorityError`, `PublicationError`,
   `PublicationValidationError`, `PublicationCapabilityError`,
   `PublicationCollisionError`, `StagingAdmissionError`,
   `PublicationDurabilityError`, `PublicationOutcomeUncertainError`,
   `PublicationNamespaceConflictError`,
   `PublicationNamespaceUncertainError`, or `StagingCleanupError`, not a
   hostile subclass;
2. H2c1's private internal error allocator created it;
3. its private non-exported provenance binds it to the exact staging handle or
   provisional staging context currently being retired;
4. its `retirement_evidence` field is still `None`; and
5. the private provenance remains inaccessible through the supported public
   API and cannot itself authorize filesystem mutation.

When all conditions hold, H2c1 populates the internal backing slot exactly
once, re-raises the same error object, and preserves its exact subtype and all
existing evidence. Callers cannot assign the slot, and no second attachment is
permitted.

Otherwise H2c1 does not mutate the body exception. It preserves that exception
as the first member of one ordered `BaseExceptionGroup` and creates one
`DescriptorRetirementError` as the second member. This fallback applies to an
error from another handle or context, a caller-created exact H2c1 error
lacking matching provenance, a hostile subclass, an error whose retirement
field is already non-null, and every arbitrary non-H2c1 `BaseException`.
Earlier retirement evidence is never overwritten, merged, or discarded.

Pre-admission validation, capability, or reservation-collision failures may
carry `evidence=None`. A failed staging admission may carry `None` only when
no stable provisional identity could be obtained. Destination is non-null
once exact `PortableRelativePath` validation succeeds; before that validation
it may be `None`. Every post-admission publication failure carries non-null
bounded recovery evidence and the exact destination. Validated destination
identity is exposed through `evidence.observed_destination_identity`;
exception messages are not authoritative evidence. Terminal lifecycle
rejection is represented by `StagingLifecycleError`, not by a publication or
cleanup outcome.

`StagingAdmissionError` always carries `NOT_COMMITTED`; its staging field is
the deterministic one-component staging leaf and `entry_may_remain` is an
exact `bool`. `StagingLifecycleError.operation` is exactly one member of its
frozen `Literal` vocabulary. A different, closed, retired, or nonmatching
exact `TrustedRoot` supplied to a terminal operation raises
`StagingAuthorityError` before filesystem access, consumes no attempt, and
leaves the staging handle unchanged. Exact-type violations remain `TypeError`.
`StagingLifecycleError` and `StagingAuthorityError` are pre-attempt
invocation-admission rejections, not publication or cleanup outcome failures;
both are outside the post-admission publication-evidence rule above.

`PublicationValidationError`, `PublicationCapabilityError`, and
`PublicationCollisionError` carry `NOT_COMMITTED`; capability and collision
remain distinguishable. `PublicationDurabilityError` carries
`NOT_COMMITTED` for a proven precommit sync failure or
`COMMITTED_DURABILITY_UNCERTAIN` after a proven commit.
`PublicationOutcomeUncertainError` carries `COMMIT_OUTCOME_UNCERTAIN`.
Other definite precommit failures may use `PublicationError` with
`NOT_COMMITTED`.

`PublicationNamespaceConflictError` carries only `COMMITTED_DURABLE` or
`COMMITTED_DURABILITY_UNCERTAIN` and represents known `complete_conflict` or
`bounded_conflict` evidence. `PublicationNamespaceUncertainError` carries only
the same two committed states. It represents either an `uninspectable`
post-commit namespace scan or any proven-commit post-commit source/namespace
verification anomaly from matrix case 1b or 1c that is not represented by a
known alias conflict. It may therefore carry `no_conflict` alias evidence
alongside anomalous source evidence. Commit/durability identity and
namespace-scan completeness are transported separately. Namespace failure
never changes a proven commit to `COMMIT_OUTCOME_UNCERTAIN`. A normal
`PublicationResult` requires both `COMMITTED_DURABLE` and complete
`no_conflict` namespace evidence with no source or namespace anomaly.

Every `StagingCleanupError` carries exactly one of `NOT_DISCARDED`,
`DISCARDED_DURABILITY_UNCERTAIN`, or `DISCARD_OUTCOME_UNCERTAIN`, together with
its bounded evidence and exact staging leaf. Only durable success returns
normally; no uncertain or non-durable outcome is converted into a normal
result.

The frozen publication states remain exactly:

```text
NOT_COMMITTED
COMMITTED_DURABLE
COMMITTED_DURABILITY_UNCERTAIN
COMMIT_OUTCOME_UNCERTAIN
```

Only `COMMITTED_DURABLE` returns as normal publication success. Every
post-admission publication failure carries its exact state and non-null bounded
recovery evidence. Collision and unsupported-capability errors carry
`NOT_COMMITTED`.

Publication outcome observations begin with the first anchored observation
after the native publication call returns or fails. Pre-call ledger, source,
destination, parent, and alias checks are preconditions; they are not the
temporal baseline for post-call `foreign` versus `replaced`. Pre-call identity
drift fails before the native call under the ledger and precondition rules and
is not labeled as a post-call outcome.

The post-call observation vocabulary is exact:

- `absent` means the name is authoritatively absent;
- `foreign` means a stable unequal identity is present at the first
  authoritative post-call outcome observation of that name; and
- `replaced` means identity drift between required post-call outcome
  observations after the first post-call state was recorded.

The source/destination outcome priority is exhaustive:

```text
1. post-call destination=original-staged-identity -> commit proven
   a. first post-call source=absent-or-foreign -> source state is consistent with commit
   b. later post-call source=replaced-or-contradictory-or-uninspectable -> post-commit namespace-verification failure
   c. post-call source=original-staged-identity -> contradictory single-link evidence and post-commit namespace-verification failure
2. post-call destination-not-original and source=exact and destination=absent -> NOT_COMMITTED
3. post-call destination-not-original and source=exact and destination=foreign -> collision and NOT_COMMITTED
4. post-call destination-not-original and every other absent, foreign, replaced, contradictory, or uninspectable combination -> COMMIT_OUTCOME_UNCERTAIN
```

Destination identity has the highest priority: the destination carrying the
original staged identity proves commit. Source replacement, contradiction, or
uninspectability after that point does not erase the commit; it produces a
post-commit namespace-verification failure. The impossible observation in
which both names authoritatively carry the original single-link identity is
contradictory. A foreign staging reservation present at the first
authoritative post-call source observation does not erase a commit proven by
the original identity at the destination. If that first post-call observation
records source absence or another stable state and a later required outcome
observation sees a different entry, the later observation is `replaced` and
produces the already frozen verification anomaly.

`EEXIST`, `ENOTEMPTY`, `EINVAL`, `EXDEV`, `EINTR`, or any other errno alone
never establishes the outcome. When the native call may have committed, H2c1
attempts publication-parent `fsync` even if later namespace or identity
inspection is anomalous. Commit evidence and sync evidence are carried
separately.

A differently spelled alias appearing after the original staged identity
reaches the destination does not make the rename occurrence uncertain. Known
complete or bounded conflict raises
`PublicationNamespaceConflictError`; a post-commit namespace scan that cannot
be completed reliably, or a matrix case 1b or 1c source/namespace anomaly not
represented by a known alias conflict, raises
`PublicationNamespaceUncertainError`. Each carries its bounded namespace
evidence and the proven publication state:

```text
COMMITTED_DURABLE
COMMITTED_DURABILITY_UNCERTAIN
```

Successful publication-parent `fsync` carries `COMMITTED_DURABLE`; failed
publication-parent `fsync` carries `COMMITTED_DURABILITY_UNCERTAIN`. Neither
entry is rolled back.

#### Explicit staging cleanup

Useful staged files and directories are expected to be nonempty. Cleanup may
remove an exact unchanged staged regular file or an exact staged directory tree
matching its complete owned ledger.

Directory cleanup:

1. revalidates the exact live root, publication-parent identity, and complete
   expected ledger;
2. rejects every unexpected or additional entry, every expected entry already
   absent before this cleanup proves its removal, every replaced, special,
   unsafe-mode, or otherwise malformed membership state, and every conflicting
   or incomplete namespace observation before deletion begins;
3. deletes only expected files and directories descriptor-relatively,
   deepest-first;
4. performs immediate identity revalidation before every removal;
5. stops at the first anomaly; and
6. flushes every affected directory and the publication parent as required.

The frozen cleanup states are:

```text
NOT_DISCARDED
DISCARDED_DURABLE
DISCARDED_DURABILITY_UNCERTAIN
DISCARD_OUTCOME_UNCERTAIN
```

Only `DISCARDED_DURABLE` returns normally. `NOT_DISCARDED` applies only when
the root observation is `exact` or `owned_partial`, stable root authority is
proven, every remaining member is known expected ledger residue, every removed
member is proven absent, and alias enumeration is complete and conflict-free.
Stable root authority compares device, inode, entry type, and owner UID; mere
existence of the staging leaf is not ownership evidence. For
`owned_partial`, mutable link-count and size changes must be explained only by
removals this cleanup attempt has already proven.

Cleanup `malformed` includes the original stable root containing any
unexpected or additional entry, an expected entry already absent before this
cleanup proves its removal, a special entry, or an unsafe mode or membership
state. Such evidence maps to `DISCARD_OUTCOME_UNCERTAIN`. A same-named but
replaced, foreign, contradictory, malformed, or uninspectable root also maps
to `DISCARD_OUTCOME_UNCERTAIN`. Simultaneously observed sibling aliases remain
encoded only in the separate namespace evidence.

A staging root proven removed followed by successful publication-parent
`fsync` is `DISCARDED_DURABLE`; proven removal followed by parent-`fsync`
failure is `DISCARDED_DURABILITY_UNCERTAIN`. Known partial expected residue
under the proven original root remains `NOT_DISCARDED` only through the
`owned_partial` observation and exact bounded remaining-entry count.
Nonempty or incomplete alias evidence, or uninspectable alias enumeration,
maps to `DISCARD_OUTCOME_UNCERTAIN`. Only complete `no_conflict` namespace
evidence permits `NOT_DISCARDED`; stable foreign or replaced roots remain
uncertain, and a leaf name by itself never proves ownership.

Any non-durable or failed cleanup permanently retires the handle. There is no
cleanup retry or handle reconstruction. Committed publication, publication
durability uncertainty, and commit-outcome uncertainty remain ineligible for
staging cleanup and preserve recovery evidence.

#### Descriptor retirement, error composition, and finalization

Descriptor retirement proceeds in this exact order:

```text
1. establish and preserve the filesystem outcome and lifecycle state
2. under the private authority lock, snapshot each owned descriptor's role, admitted identity, and private per-acquisition generation
3. before inspection or close, detach every raw descriptor slot, permanently consume its retirement attempt, and compare-and-remove only its matching ownership-registry generation
4. process every independently owned descriptor in the frozen deterministic role order and continue after anomalies
5. generation mismatch -> FOREIGN_PRESERVED with observed_identity=None; do not inspect or close
6. matching generation without an admitted identity -> UNINSPECTABLE; otherwise perform one pre-close fstat
7. pre-close EBADF -> ALREADY_ABSENT; do not close
8. other failure to obtain a stable comparison -> UNINSPECTABLE; do not close
9. matching generation plus unequal stable identity -> FOREIGN_PRESERVED; do not close
10. exact generation and equal identity -> invoke close exactly once
11. close returns -> CLOSED
12. close raises -> CLOSE_OUTCOME_UNCERTAIN
13. never retry close
14. never inspect or act on that descriptor number after close returns or raises
15. context exit and finalization cannot retry consumed retirement
16. continue retiring every other independently owned descriptor after an anomaly
17. the exact live TrustedRoot is borrowed and is never closed, retired, or adopted by H2c1
```

Logical H2c1 authority becomes permanently unavailable before kernel cleanup.
A generation mismatch is foreign state and is preserved without descriptor
inspection and with `observed_identity=None`. A matching generation without
an admitted identity, or without a stable comparison, is `UNINSPECTABLE`.
Otherwise it receives exactly one pre-close `fstat`; a stable unequal identity
is `FOREIGN_PRESERVED`, while pre-close `EBADF` is already-absent evidence.
Any exception from `close`, including `EBADF`, `EINTR`, or `EIO`, is an
uncertain close outcome; close is never blindly retried, and no post-close
operation uses that descriptor number. All remaining independently owned
descriptors are processed after one anomaly.

The descriptor bounds are exact:

- admission or abandonment owns at most two descriptors;
- a proven publication has at most three descriptors requiring terminal
  retirement;
- a proven cleanup has at most four descriptors requiring terminal retirement;
- nested traversal or cleanup has at most seven simultaneously owned H2c1
  descriptors; and
- native rename owns no descriptor.

The exact live `TrustedRoot` is borrowed throughout and is excluded from those
bounds. Private acquisition generations detect H2c-managed same-inode
descriptor reuse. POSIX metadata cannot distinguish unrestricted external
same-number, same-inode ABA from the original open description; unrestricted
raw descriptor manipulation remains outside the frozen threat boundary.
Detectably foreign descriptors are preserved. A close anomaly may leave
physical descriptor lifetime uncertain, but logical H2c1 authority remains
permanently retired and cannot be revived.

Descriptor retirement is orthogonal to the already established publication,
cleanup, namespace, durability, and lifecycle evidence. It never rewrites an
already established `PublicationState`, `StagingCleanupState`, or
`StagingState` solely because retirement later fails:

- durable publication remains `COMMITTED_DURABLE` and the handle remains
  `PUBLISHED`;
- committed durability uncertainty remains unchanged and `PUBLISHED`;
- namespace conflict or namespace uncertainty remains its original exact
  subtype and `PUBLISHED`;
- commit-outcome uncertainty remains unchanged and `RETIRED`;
- durable cleanup remains `DISCARDED_DURABLE` and the handle remains
  `DISCARDED`;
- cleanup durability or outcome uncertainty retains its already frozen
  outcome and lifecycle; and
- a proven `NOT_COMMITTED` outcome remains `NOT_COMMITTED`.

Error precedence is exact. A matching internally created exact H2c1 exception
with empty retirement evidence remains the same catchable subtype; its
publication, cleanup, collision, durability, namespace, and recovery evidence
remains unchanged, and retirement evidence attaches orthogonally exactly once.
Every exception that fails the private provenance conditions remains
unmodified and uses the ordered `BaseExceptionGroup` fallback frozen above.

An otherwise-normal durable publication raises `DescriptorRetirementError`
carrying the exact `PublicationResult` that would otherwise have returned. An
otherwise-normal durable cleanup raises the same error carrying its exact
`StagingCleanupResult`. Those normal result objects remain unchanged and
return only after every required owned descriptor has been successfully and
verifiably retired.

A population or sealing retirement anomaly with no primary H2c1 error uses
the standalone `DescriptorRetirementError` tuple frozen above. Normal
abandonment with a retirement anomaly uses `operation="context_exit"` and
`state=RETIRED`. Pre-yield admission never raises a standalone
`DescriptorRetirementError`; its exact `StagingAdmissionError` remains primary
and receives retirement evidence only through matching private provenance.
Finalization uses the frozen `finalization` tuple and can report its error only
through Python's unraisable-exception mechanism. Multiple descriptor anomalies
aggregate into one bounded retirement-evidence value.

After either private origin's retirement batch is consumed, a proven
`NOT_COMMITTED` handle retains its single cleanup authorization despite
verified closure, an already-absent descriptor, foreign reuse, an
uninspectable descriptor, or an uncertain close result. Publication cannot be
retried.

The pending seal-time-collision `NOT_COMMITTED` context-exit matrix is exactly:

```text
anomaly_free_without_body_exception:
    returns: normally
    state: NOT_COMMITTED
    immutable_ledger: preserved
    cleanup_authorization: preserved
    pending_handle_retirement_batch: consumed_once

retirement_anomaly_without_body_exception:
    error: DescriptorRetirementError
    error.state: NOT_COMMITTED
    error.operation: "context_exit"
    error.terminal_result: None
    state: NOT_COMMITTED
    immutable_ledger: preserved
    cleanup_authorization: preserved
    cleanup: not_performed
    retirement_retry: forbidden

retirement_anomaly_with_body_exception:
    composition: provenance_aware_attachment_or_ordered_BaseExceptionGroup
    body_exception: primary
    state: NOT_COMMITTED
    immutable_ledger: preserved
    cleanup_authorization: preserved
    cleanup: not_performed
    retirement_retry: forbidden
```

An anomaly-free exit with a body exception propagates that body exception
unchanged; the normal return above applies only when no body exception exists.
For a retirement anomaly with a body exception, the provenance-aware
attachment or grouping rule preserves the body exception as primary. A
`NOT_COMMITTED` context exit never silently suppresses a retirement anomaly.
It performs no cleanup and permits no descriptor-retirement retry.

The already-consumed terminal-publication `NOT_COMMITTED` context-exit
behavior is exactly:

```text
without_body_exception:
    returns: normally
    descriptor_inspection: not_performed
    close: not_performed
    retirement_attempt: not_performed

with_body_exception:
    propagation: body_exception_unchanged
    descriptor_inspection: not_performed
    close: not_performed
    retirement_attempt: not_performed
```

Later cleanup opens fresh descriptors through the same exact live
`TrustedRoot`, revalidates the complete immutable ledger and namespace, and
never reuses, revives, retries, or adopts retired descriptor state. Cleanup
authorization is consumed when cleanup begins, and any failed or partial
cleanup permanently retires the handle.

Context and finalization behavior is exact:

- `OPEN` or `SEALED` abandonment preserves staging, enters `RETIRED`, and
  consumes retirement once. A body exception remains primary under the error
  composition above.
- `NOT_COMMITTED` context exit consults only private lifecycle provenance. A
  pending seal-time-collision batch follows the exact three-branch matrix
  above, preserving state, immutable ledger, and one cleanup authorization
  while consuming that batch exactly once. An already-consumed
  terminal-publication batch follows the exact no-second-retirement behavior
  above.
- `PUBLISHED` and `DISCARDED` context exit performs no second publication,
  cleanup, descriptor inspection, or close attempt.
- `RETIRED` context exit is a no-op.
- garbage collection never publishes, deletes, or retries descriptor close;
  finalization permanently detaches remaining logical authority.
- collection of a `NOT_COMMITTED` handle loses its cleanup authorization
  because reconstruction and adoption are forbidden.
- descriptor-number reuse cannot revive a handle or trigger another close.

### H2c2 exclusive-claim contract

The claim schema is:

```text
research_platform.hpc.exclusive_claim.v1
```

H2c2 will retain an immutable canonical claim record binding schema version,
run identity, operation identity, purpose, reviewed-plan SHA-256, owner token,
and creation time. Canonical bytes are built before mutation.
Descriptor-relative exclusive `mkdir` remains the atomic ownership point.
Ownership and release must bind exact owner, record, claim-parent, and
claim-directory identity.

A missing, partial, malformed, stale, foreign, replaced, aliased, or nonempty
claim blocks and remains recovery evidence. There is no automatic stale
deletion, adoption, renewal, replacement, expiry, or forced cleanup.
Deterministic competing-claim tests use barriers rather than timing sleeps and
must prove exactly one compliant winner.

Detailed H2c2 acquisition, release, durability, tombstone, retry, and recovery
semantics remain pending a separate preimplementation freeze after H2c1 is
implemented and hosted-CI-green. In particular:

- claim acquisition depends on H2c1's final publication and result authority;
- a known partial release must be distinguished from a genuinely uncertain
  release;
- parent durability failure after `rmdir` requires a separately frozen
  cross-process recovery or tombstone decision;
- a handle cannot require an admitted `claim.json` identity unless acquisition
  records its full device/inode/type/link/mode fingerprint; and
- exact builder and acquisition signatures and canonical digest serialization
  require the focused H2c2 design gate.

This H2c1 contract does not freeze H2c2 public API names, a claim-directory
derivation algorithm, claim-release states, an exact acquisition or release
sequence, or retry and recovery behavior. H2c2 remains pending, unimplemented,
and unexported.

### H2d receipt-envelope foundation

The generic envelope schema and digest algorithm are:

```text
research_platform.hpc.receipt_envelope.v1
research_platform.hpc.sha256_receipt_envelope.v1
```

No production receipt family is registered by H2. Callers pass an immutable
mapping from `(family, family_version)` to strict family validators; there is
no mutable global registry. Tests may inject one synthetic validator. The
generic validator treats the family payload as opaque only after that strict
family validator accepts it. `family_version` is a positive integer from 1
through `2**31 - 1`.

Identifiers are 1-128 ASCII characters: the first is alphanumeric and the
rest are alphanumeric, `.`, `_`, or `-`. SHA-256 values are exactly 64
lowercase hexadecimal characters. A source commit, when present, is exactly
40 or 64 lowercase hexadecimal characters. UTC timestamps are exactly
`YYYY-MM-DDTHH:MM:SS.ffffffZ`, must identify a real UTC instant, and use no
offset spelling. Finalization cannot precede creation.

The canonical envelope body has exactly:

```text
schema_version
family
family_version
run_id
operation_id
reviewed_plan_digest
source_commit
inventory
target_identity
receipt_dependencies
created_at
finalized_at
payload
```

`reviewed_plan_digest` is a strict `sha256` identity. `source_commit` and
`inventory` are nullable in the generic envelope; later family validators
decide when they are mandatory. An inventory identity contains exactly its
scope, inventory schema, versioned tree-digest identity, and raw canonical
inventory-byte digest. `target_identity` is either null or contains exactly
the target, profile, and role identifiers—never credentials.

Receipt dependencies use one canonical list. Each item contains exactly a
relation, family, family version, and receipt-digest identity. `relation` is
exactly `prior` or `prerequisite`; `family_version` is a positive integer from
1 through `2**31 - 1`. Ordering is the ascending tuple of relation UTF-8
bytes, family UTF-8 bytes, family version numerically, digest-algorithm UTF-8
bytes, and digest-value ASCII bytes. One receipt digest may appear only once
across the entire list regardless of relation. Duplicate references are
rejected. A dependency equal to the envelope's own computed digest is a direct
self-reference and is rejected. Transitive cycle detection requires a later
receipt resolver.

The receipt document has exactly:

```text
envelope
receipt_digest
```

The digest is computed from canonical envelope-body bytes with the frozen
receipt domain and length framing. The outer `receipt_digest` is not part of
those bytes and is forbidden as an envelope-body field. A same-named field
inside a family payload is governed by that family's strict validator rather
than by an unsafe generic recursive rule.

Parsing uses canonical JSON v1 and exact known shapes. Unsupported schema,
family, or family versions; duplicate JSON keys; unknown envelope keys; NaN
or Infinity; malformed digests; unsafe paths; duplicate receipt dependencies;
direct self-reference; and family-validator rejection all fail closed.
Timestamps and nonces are injectable in tests. Serialization is deterministic
for fixed values, not across independently created receipts.

Receipt publication uses H2c atomic no-replace publication. Private
operational receipts and a later sanitized evidence projection remain
separate. H2d implements no H3-H9 receipt-family payload, sanitized-evidence
scope, or release/export scope.

## Gate and evidence boundary

H2a contains no inventory scanner, tree-inventory implementation, no-replace
publisher, claim, or receipt envelope. H2b adds descriptor-anchored
regular-file inventories and canonical tree digests. Its repeated anchored
passes detect observable mutation; they are not an atomic filesystem snapshot.
H2c1 no-replace publication is implemented by this working snapshot, but
hosted acceptance remains pending until the eventual commit's CI succeeds.
H2c2 exclusive claims and H2d receipt envelopes/complete H2 acceptance remain
pending. H2c is incomplete until both of its internal gates are committed and
hosted-CI-green. H2 is therefore incomplete and H3 remains blocked. The
detailed H2c2 contract remains separately pending.

None of the implemented H2a/H2b/H2c1 foundations provides runtime
provisioning, transfer, scheduler submission or reconciliation, executed
cancellation, remote execution receipts, retrieval, fake-remote acceptance,
live-cluster validation, claim promotion, export, tagging, or release
publication.

No readiness, stage, scheduler, cancellation, remote-execution, or retrieval
receipt exists today. No fake-remote or live-cluster validation has occurred.
Current workflows continue to use their existing behavior until a later
explicit migration gate.

### Cross-platform H2 acceptance

Implemented H2b tests use only synthetic temporary roots outside the checkout.
The inventory matrix covers absolute and traversing names, backslashes,
controls, case and Unicode aliases, root/ancestor/descendant/broken symlinks,
hard links, FIFO and socket entries, safely testable devices, descendant device
boundaries, and concurrent addition, deletion, replacement, same-size
mutation, and directory replacement. Frozen vectors prove identical inventory
bytes and digests across roots and creation order. Hosted Python 3.11/3.12,
Linux, and macOS results remain part of the complete cross-platform acceptance
evidence rather than an atomic-snapshot claim.

The implemented H2c1 publication matrix injects failure before write, after
file sync, during publication, and after publication before parent sync. It
proves locally exact collision preservation, unsupported-capability failure,
committed versus uncertain outcomes, no outside-root access, and preservation
of foreign or replaced recovery evidence. Hosted Linux and macOS acceptance
remains pending until the eventual commit's CI succeeds.

Pending H2c2 claim tests use deterministic barriers rather than sleeps to prove
one winner and preserve malformed, partial, stale, foreign, and replaced
recovery evidence. Its detailed acquisition, release, durability, retry,
tombstone, and recovery test matrix is frozen only by the later H2c2
preimplementation gate. Pending H2d receipt tests freeze canonical bytes and
digest vectors and reject duplicate keys, unknown shapes, unsupported
versions, nonfinite values, malformed identities, unsafe paths, duplicate
dependencies, direct self-reference, and unregistered families.

The complete safety suite is required on hosted Linux with Python 3.11 and
3.12 and macOS ARM with Python 3.12. No H2 CI test contacts a remote system or
creates repository-local state.

## Consequences

- Canonical bytes, portable lexical names, versioned digest framing, and the
  later H2 contracts cannot drift independently.
- The standard-library-only `research-hpc` package remains the dependency-safe
  owner; `research-core` and `research-neuro` are not imported.
- The ASCII path protocol intentionally trades general filename support for
  cross-platform alias safety in managed payloads.
- H2a, H2b, and H2c1 create no remote or workflow capability and change no
  current runtime. No workflow consumes their canonical paths, inventories,
  tree digests, or no-replace publication authority.
- H3 remains blocked until H2d completes all H2 acceptance and CI is green.
