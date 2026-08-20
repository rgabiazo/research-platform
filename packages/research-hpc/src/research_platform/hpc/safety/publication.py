"""Descriptor-relative atomic no-replace publication foundations.

This module implements ADR-0023 H2c1 only.  It owns staging creation and
population, publishes through the platform's native no-replace rename
primitive, and preserves bounded evidence for every terminal outcome.  It
does not implement claims, receipts, transfer, or runtime integration.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import ctypes
import errno
import hashlib
import itertools
import os
import stat
import sys
import threading
from typing import Iterable, Iterator, Literal
import weakref

from .canonical import (
    MAX_CANONICAL_CONTAINER_ITEMS,
    MAX_CANONICAL_DOCUMENT_BYTES,
    MAX_SIGNED_64,
    domain_separated_sha256,
)
from .inventory import (
    InventorySafetyError,
    InventoryValidationError,
    RegularFileInventory,
    RegularFileRecord,
    TrustedRoot,
    TrustedRootError,
    _build_regular_file_inventory,
    _open_fresh_trusted_root_directory,
    _require_exact_live_root,
    _validate_scope,
)
from .paths import (
    PortablePathError,
    PortableRelativePath,
    portable_path_sort_key,
    require_distinct_file_paths,
)


PUBLICATION_STAGING_DOMAIN = (
    b"research-platform:hpc:publication-staging:v1\0"
)
_STAGING_PREFIX = ".rp-stage-v1-"
_PRIVATE_MODE_CREATION_LOCK = threading.Lock()
_OPEN_DESCRIPTOR_IDENTITIES: dict[int, "_DescriptorRegistration"] = {}
_OPEN_DESCRIPTOR_IDENTITIES_LOCK = threading.Lock()
_DESCRIPTOR_GENERATIONS = itertools.count(1)
_ERROR_PROVENANCE = threading.local()
_HASH_CHUNK_BYTES = 1024 * 1024
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4
_MAX_EVIDENCE_ALIASES = MAX_CANONICAL_CONTAINER_ITEMS
_MAX_ENTRY_ID = 2**64 - 1
_MAX_LINK_COUNT = 2**63 - 1
_MAX_ERRNO = 2**31 - 1

_DescriptorRole = Literal[
    "traversal_entry",
    "traversal_directory",
    "traversal_parent",
    "operation_staging",
    "operation_parent",
    "handle_staging",
    "handle_parent",
]


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


class DescriptorRetirementObservation(str, Enum):
    CLOSED = "closed"
    ALREADY_ABSENT = "already_absent"
    FOREIGN_PRESERVED = "foreign_preserved"
    UNINSPECTABLE = "uninspectable"
    CLOSE_OUTCOME_UNCERTAIN = "close_outcome_uncertain"


class _NamespaceObservation(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    NO_CONFLICT = "no_conflict"
    COMPLETE_CONFLICT = "complete_conflict"
    BOUNDED_CONFLICT = "bounded_conflict"
    UNINSPECTABLE = "uninspectable"


class _OpaqueFrozenValue:
    __slots__ = ()

    def __new__(cls, *args: object, **kwargs: object):
        raise TypeError(f"{cls.__name__} values are created by H2c1 operations")

    def __copy__(self):
        raise TypeError(f"{type(self).__name__} values cannot be copied")

    def __deepcopy__(self, memo: object):
        raise TypeError(f"{type(self).__name__} values cannot be copied")

    def __reduce__(self):
        raise TypeError(f"{type(self).__name__} values cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        raise TypeError(f"{type(self).__name__} values cannot be serialized")


@dataclass(frozen=True, init=False, slots=True)
class PublicationEntryIdentity(_OpaqueFrozenValue):
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
    link_count: int
    owner_uid: int
    mode: int
    size_bytes: int

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("PublicationEntryIdentity cannot be subclassed")


@dataclass(frozen=True, init=False, slots=True)
class NamespaceEvidence(_OpaqueFrozenValue):
    namespace_observation: Literal[
        "not_attempted",
        "no_conflict",
        "complete_conflict",
        "bounded_conflict",
        "uninspectable",
    ]
    conflicting_aliases: tuple[PortableRelativePath, ...]
    conflicting_alias_count: int | None
    aliases_complete: bool

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("NamespaceEvidence cannot be subclassed")


@dataclass(frozen=True, init=False, slots=True)
class PublicationRecoveryEvidence(_OpaqueFrozenValue):
    staging_identity: PublicationEntryIdentity
    source_observation: Literal[
        "not_attempted",
        "exact",
        "absent",
        "foreign",
        "replaced",
        "contradictory",
        "uninspectable",
    ]
    observed_source_identity: PublicationEntryIdentity | None
    destination_observation: Literal[
        "not_attempted",
        "exact",
        "absent",
        "foreign",
        "replaced",
        "contradictory",
        "uninspectable",
    ]
    observed_destination_identity: PublicationEntryIdentity | None
    namespace_evidence: NamespaceEvidence
    parent_fsync: Literal[
        "not_attempted",
        "succeeded",
        "failed",
        "uncertain",
    ]
    native_errno: int | None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("PublicationRecoveryEvidence cannot be subclassed")


@dataclass(frozen=True, init=False, slots=True)
class StagingCleanupRecoveryEvidence(_OpaqueFrozenValue):
    staging_identity: PublicationEntryIdentity
    root_observation: Literal[
        "exact",
        "owned_partial",
        "absent",
        "foreign",
        "replaced",
        "contradictory",
        "malformed",
        "uninspectable",
    ]
    observed_root_identity: PublicationEntryIdentity | None
    remaining_expected_entries: int | None
    namespace_evidence: NamespaceEvidence
    parent_fsync: Literal[
        "not_attempted",
        "succeeded",
        "failed",
        "uncertain",
    ]
    native_errno: int | None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "StagingCleanupRecoveryEvidence cannot be subclassed"
        )


@dataclass(frozen=True, init=False, slots=True)
class PublicationResult(_OpaqueFrozenValue):
    state: PublicationState
    destination: PortableRelativePath
    destination_identity: PublicationEntryIdentity
    namespace_evidence: NamespaceEvidence

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("PublicationResult cannot be subclassed")


@dataclass(frozen=True, init=False, slots=True)
class StagingCleanupResult(_OpaqueFrozenValue):
    state: StagingCleanupState
    staging: PortableRelativePath
    discarded_identity: PublicationEntryIdentity
    namespace_evidence: NamespaceEvidence

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("StagingCleanupResult cannot be subclassed")


@dataclass(frozen=True, init=False, slots=True)
class DescriptorRetirementIdentity(_OpaqueFrozenValue):
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

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("DescriptorRetirementIdentity cannot be subclassed")


@dataclass(frozen=True, init=False, slots=True)
class DescriptorRetirementRecord(_OpaqueFrozenValue):
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

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("DescriptorRetirementRecord cannot be subclassed")


@dataclass(frozen=True, init=False, slots=True)
class DescriptorRetirementEvidence(_OpaqueFrozenValue):
    records: tuple[DescriptorRetirementRecord, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("DescriptorRetirementEvidence cannot be subclassed")


class StagingLifecycleError(Exception):
    __slots__ = (
        "_state",
        "_operation",
        "_retirement_evidence",
        "_pending_retirement_evidence",
        "_h2c1_provenance",
    )

    def __init__(
        self,
        state: StagingState,
        operation: Literal[
            "write",
            "mkdir",
            "write_file",
            "seal",
            "publish",
            "cleanup",
        ],
    ) -> None:
        if type(state) is not StagingState:
            raise TypeError("lifecycle-error state must be exact StagingState")
        if type(operation) is not str:
            raise TypeError("lifecycle-error operation must be exact str")
        if operation not in {
            "write",
            "mkdir",
            "write_file",
            "seal",
            "publish",
            "cleanup",
        }:
            raise ValueError("invalid lifecycle-error operation")
        self._state = state
        self._operation = operation
        self._retirement_evidence = None
        self._pending_retirement_evidence = None
        self._h2c1_provenance = _current_error_provenance()
        super().__init__(
            f"staging operation {operation!r} is invalid in state {state.value!r}"
        )

    @property
    def state(self) -> StagingState:
        return self._state

    @property
    def operation(
        self,
    ) -> Literal[
        "write",
        "mkdir",
        "write_file",
        "seal",
        "publish",
        "cleanup",
    ]:
        return self._operation

    @property
    def retirement_evidence(
        self,
    ) -> DescriptorRetirementEvidence | None:
        return self._retirement_evidence


class StagingAuthorityError(Exception):
    __slots__ = (
        "_state",
        "_operation",
        "_retirement_evidence",
        "_pending_retirement_evidence",
        "_h2c1_provenance",
    )

    def __init__(
        self,
        state: StagingState,
        operation: Literal["publish", "cleanup"],
    ) -> None:
        if type(state) is not StagingState:
            raise TypeError("authority-error state must be exact StagingState")
        if type(operation) is not str:
            raise TypeError("authority-error operation must be exact str")
        if operation not in {"publish", "cleanup"}:
            raise ValueError("invalid authority-error operation")
        self._state = state
        self._operation = operation
        self._retirement_evidence = None
        self._pending_retirement_evidence = None
        self._h2c1_provenance = _current_error_provenance()
        super().__init__(
            f"the supplied trusted root is not the staging authority for "
            f"{operation!r}"
        )

    @property
    def state(self) -> StagingState:
        return self._state

    @property
    def operation(self) -> Literal["publish", "cleanup"]:
        return self._operation

    @property
    def retirement_evidence(
        self,
    ) -> DescriptorRetirementEvidence | None:
        return self._retirement_evidence


class PublicationError(Exception):
    __slots__ = (
        "_state",
        "_evidence",
        "_destination",
        "_retirement_evidence",
        "_pending_retirement_evidence",
        "_h2c1_provenance",
    )

    def __init__(
        self,
        message: str,
        *,
        state: PublicationState,
        evidence: PublicationRecoveryEvidence | None,
        destination: PortableRelativePath | None,
    ) -> None:
        if type(message) is not str:
            raise TypeError("publication-error message must be exact str")
        if type(state) is not PublicationState:
            raise TypeError(
                "publication-error state must be exact PublicationState"
            )
        if (
            evidence is not None
            and type(evidence) is not PublicationRecoveryEvidence
        ):
            raise TypeError(
                "publication-error evidence must be exact or None"
            )
        if (
            destination is not None
            and type(destination) is not PortableRelativePath
        ):
            raise TypeError(
                "publication-error destination must be exact or None"
            )
        if evidence is not None and destination is None:
            raise ValueError(
                "publication recovery evidence requires a destination"
            )
        if (
            evidence is not None
            and destination is not None
            and len(destination.parts) != 1
        ):
            raise ValueError(
                "post-admission publication destination must be one component"
            )
        _validate_publication_error_fields(
            self,
            state=state,
            evidence=evidence,
            destination=destination,
        )
        self._state = state
        self._evidence = evidence
        self._destination = destination
        self._retirement_evidence = None
        self._pending_retirement_evidence = None
        self._h2c1_provenance = _current_error_provenance(
            caller_depth=(
                3 if type(self) is StagingAdmissionError else 2
            )
        )
        super().__init__(message)

    @property
    def state(self) -> PublicationState:
        return self._state

    @property
    def evidence(self) -> PublicationRecoveryEvidence | None:
        return self._evidence

    @property
    def destination(self) -> PortableRelativePath | None:
        return self._destination

    @property
    def retirement_evidence(
        self,
    ) -> DescriptorRetirementEvidence | None:
        return self._retirement_evidence


class PublicationValidationError(PublicationError):
    pass


class PublicationCapabilityError(PublicationError):
    pass


class PublicationCollisionError(PublicationError):
    pass


class StagingAdmissionError(PublicationError):
    __slots__ = ("_staging", "_entry_may_remain")

    def __init__(
        self,
        message: str,
        *,
        state: PublicationState,
        evidence: PublicationRecoveryEvidence | None,
        destination: PortableRelativePath | None,
        staging: PortableRelativePath,
        entry_may_remain: bool,
    ) -> None:
        if state is not PublicationState.NOT_COMMITTED:
            raise ValueError(
                "staging-admission errors must be not-committed"
            )
        if (
            type(destination) is not PortableRelativePath
            or len(destination.parts) != 1
        ):
            raise TypeError(
                "staging-admission destination must be one exact component"
            )
        if (
            type(staging) is not PortableRelativePath
            or len(staging.parts) != 1
        ):
            raise TypeError(
                "staging-admission path must be one exact component"
            )
        if type(entry_may_remain) is not bool:
            raise TypeError("entry_may_remain must be exactly bool")
        self._staging = staging
        self._entry_may_remain = entry_may_remain
        super().__init__(
            message,
            state=state,
            evidence=evidence,
            destination=destination,
        )

    @property
    def staging(self) -> PortableRelativePath:
        return self._staging

    @property
    def entry_may_remain(self) -> bool:
        return self._entry_may_remain


class PublicationDurabilityError(PublicationError):
    pass


class PublicationOutcomeUncertainError(PublicationError):
    pass


class PublicationNamespaceConflictError(PublicationError):
    pass


class PublicationNamespaceUncertainError(PublicationError):
    pass


def _validate_publication_error_fields(
    error: PublicationError,
    *,
    state: PublicationState,
    evidence: PublicationRecoveryEvidence | None,
    destination: PortableRelativePath | None,
) -> None:
    def require_committed_evidence() -> PublicationRecoveryEvidence:
        if evidence is None or destination is None:
            raise ValueError(
                "committed publication errors require bounded evidence"
            )
        expected_syncs = (
            {"succeeded"}
            if state is PublicationState.COMMITTED_DURABLE
            else {"failed", "uncertain"}
        )
        if evidence.parent_fsync not in expected_syncs:
            raise ValueError(
                "committed publication state contradicts parent fsync"
            )
        if (
            evidence.source_observation == "exact"
            or evidence.destination_observation in {"absent", "foreign"}
        ):
            raise ValueError(
                "committed publication evidence is a definite precommit shape"
            )
        return evidence

    if isinstance(error, PublicationNamespaceConflictError):
        valid_states = {
            PublicationState.COMMITTED_DURABLE,
            PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
        }
        if state not in valid_states:
            raise ValueError(
                "namespace-conflict errors require a committed state"
            )
        committed = require_committed_evidence()
        if committed.namespace_evidence.namespace_observation not in {
            "complete_conflict",
            "bounded_conflict",
        }:
            raise ValueError(
                "namespace-conflict errors require conflict evidence"
            )
        return
    if isinstance(error, PublicationNamespaceUncertainError):
        valid_states = {
            PublicationState.COMMITTED_DURABLE,
            PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
        }
        if state not in valid_states:
            raise ValueError(
                "namespace-uncertain errors require a committed state"
            )
        committed = require_committed_evidence()
        namespace_observation = (
            committed.namespace_evidence.namespace_observation
        )
        source_or_destination_anomaly = (
            committed.source_observation not in {"absent", "foreign"}
            or committed.destination_observation != "exact"
        )
        if not (
            namespace_observation == "uninspectable"
            or (
                namespace_observation == "no_conflict"
                and source_or_destination_anomaly
            )
        ):
            raise ValueError(
                "namespace-uncertain errors require uncertain namespace "
                "or post-commit identity evidence"
            )
        return
    if isinstance(error, PublicationOutcomeUncertainError):
        if state is not PublicationState.COMMIT_OUTCOME_UNCERTAIN:
            raise ValueError(
                "outcome-uncertain errors require uncertain commit state"
            )
        if evidence is None or destination is None:
            raise ValueError(
                "outcome-uncertain errors require bounded recovery evidence"
            )
        if evidence.destination_observation == "exact" or (
            evidence.source_observation == "exact"
            and evidence.destination_observation in {"absent", "foreign"}
        ):
            raise ValueError(
                "outcome-uncertain evidence has a definite outcome"
            )
        return
    if isinstance(error, PublicationDurabilityError):
        if state not in {
            PublicationState.NOT_COMMITTED,
            PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
        }:
            raise ValueError(
                "durability errors require a frozen non-durable state"
            )
        if evidence is None or destination is None:
            raise ValueError(
                "durability errors require bounded recovery evidence"
            )
        if state is PublicationState.NOT_COMMITTED:
            valid = (
                evidence.source_observation == "exact"
                and evidence.destination_observation
                not in {"exact", "foreign"}
                and evidence.namespace_evidence.namespace_observation
                not in {"complete_conflict", "bounded_conflict"}
            )
        else:
            valid = (
                evidence.source_observation in {"absent", "foreign"}
                and evidence.destination_observation == "exact"
                and evidence.namespace_evidence.namespace_observation
                == "no_conflict"
                and evidence.parent_fsync in {"failed", "uncertain"}
            )
        if not valid:
            raise ValueError(
                "durability-error evidence contradicts its outcome"
            )
        return
    if isinstance(error, PublicationCollisionError):
        if state is not PublicationState.NOT_COMMITTED:
            raise ValueError(
                "collision errors must be not-committed"
            )
        if destination is not None and len(destination.parts) != 1:
            raise ValueError(
                "collision destination must be one component"
            )
        if evidence is not None and not (
            evidence.source_observation == "exact"
            and evidence.destination_observation != "exact"
            and (
                evidence.destination_observation == "foreign"
                or evidence.namespace_evidence.namespace_observation
                in {"complete_conflict", "bounded_conflict"}
            )
        ):
            raise ValueError(
                "collision evidence does not describe a stable collision"
            )
        return
    if state is not PublicationState.NOT_COMMITTED:
        raise ValueError(
            "definite precommit publication errors must be not-committed"
        )
    if type(error) is PublicationError:
        if evidence is None or destination is None:
            raise ValueError(
                "generic publication errors require definite outcome evidence"
            )
        if not (
            evidence.source_observation == "exact"
            and evidence.destination_observation == "absent"
        ):
            raise ValueError(
                "generic publication error evidence is not definite"
            )


class StagingCleanupError(Exception):
    __slots__ = (
        "_state",
        "_evidence",
        "_staging",
        "_retirement_evidence",
        "_pending_retirement_evidence",
        "_h2c1_provenance",
    )

    def __init__(
        self,
        message: str,
        *,
        state: StagingCleanupState,
        evidence: StagingCleanupRecoveryEvidence,
        staging: PortableRelativePath,
    ) -> None:
        if type(message) is not str:
            raise TypeError("cleanup-error message must be exact str")
        if type(state) is not StagingCleanupState:
            raise TypeError(
                "cleanup-error state must be exact StagingCleanupState"
            )
        if type(evidence) is not StagingCleanupRecoveryEvidence:
            raise TypeError("cleanup-error evidence must be exact")
        if (
            type(staging) is not PortableRelativePath
            or len(staging.parts) != 1
        ):
            raise TypeError(
                "cleanup-error staging path must be one exact component"
            )
        if state not in {
            StagingCleanupState.NOT_DISCARDED,
            StagingCleanupState.DISCARDED_DURABILITY_UNCERTAIN,
            StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN,
        }:
            raise ValueError(
                "cleanup errors require a frozen non-durable outcome"
            )
        no_conflict = (
            evidence.namespace_evidence.namespace_observation == "no_conflict"
            and evidence.namespace_evidence.aliases_complete
        )
        if state is StagingCleanupState.NOT_DISCARDED:
            valid_evidence = (
                evidence.root_observation in {"exact", "owned_partial"}
                and no_conflict
            )
        elif (
            state
            is StagingCleanupState.DISCARDED_DURABILITY_UNCERTAIN
        ):
            valid_evidence = (
                evidence.root_observation == "absent"
                and no_conflict
                and evidence.parent_fsync in {"failed", "uncertain"}
            )
        else:
            # Public evidence cannot encode whether an absent root was
            # removed by this cleanup attempt or was already missing.  The
            # private deletion journal establishes that distinction before
            # this error is allocated.
            valid_evidence = True
        if not valid_evidence:
            raise ValueError(
                "cleanup error state contradicts its recovery evidence"
            )
        self._state = state
        self._evidence = evidence
        self._staging = staging
        self._retirement_evidence = None
        self._pending_retirement_evidence = None
        self._h2c1_provenance = _current_error_provenance()
        super().__init__(message)

    @property
    def state(self) -> StagingCleanupState:
        return self._state

    @property
    def evidence(self) -> StagingCleanupRecoveryEvidence:
        return self._evidence

    @property
    def staging(self) -> PortableRelativePath:
        return self._staging

    @property
    def retirement_evidence(
        self,
    ) -> DescriptorRetirementEvidence | None:
        return self._retirement_evidence


class DescriptorRetirementError(Exception):
    __slots__ = (
        "_state",
        "_operation",
        "_destination",
        "_staging",
        "_terminal_result",
        "_retirement_evidence",
    )

    def __init__(
        self,
        *,
        state: StagingState,
        operation: Literal[
            "write",
            "mkdir",
            "write_file",
            "seal",
            "publish",
            "cleanup",
            "context_exit",
            "finalization",
        ],
        destination: PortableRelativePath,
        staging: PortableRelativePath,
        terminal_result: PublicationResult | StagingCleanupResult | None,
        retirement_evidence: DescriptorRetirementEvidence,
    ) -> None:
        _validate_descriptor_retirement_error(
            state=state,
            operation=operation,
            destination=destination,
            staging=staging,
            terminal_result=terminal_result,
            retirement_evidence=retirement_evidence,
        )
        self._state = state
        self._operation = operation
        self._destination = destination
        self._staging = staging
        self._terminal_result = terminal_result
        self._retirement_evidence = retirement_evidence
        super().__init__(
            f"descriptor retirement was not fully verifiable during "
            f"{operation!r}"
        )

    @property
    def state(self) -> StagingState:
        return self._state

    @property
    def operation(
        self,
    ) -> Literal[
        "write",
        "mkdir",
        "write_file",
        "seal",
        "publish",
        "cleanup",
        "context_exit",
        "finalization",
    ]:
        return self._operation

    @property
    def destination(self) -> PortableRelativePath:
        return self._destination

    @property
    def staging(self) -> PortableRelativePath:
        return self._staging

    @property
    def terminal_result(
        self,
    ) -> PublicationResult | StagingCleanupResult | None:
        return self._terminal_result

    @property
    def retirement_evidence(self) -> DescriptorRetirementEvidence:
        return self._retirement_evidence

    def __copy__(self):
        raise TypeError("DescriptorRetirementError values cannot be copied")

    def __deepcopy__(self, memo: object):
        raise TypeError("DescriptorRetirementError values cannot be copied")

    def __reduce__(self):
        raise TypeError(
            "DescriptorRetirementError values cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int):
        raise TypeError(
            "DescriptorRetirementError values cannot be serialized"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("DescriptorRetirementError cannot be subclassed")


class _RequiredFsyncError(OSError):
    """Internal marker for a protocol-required precommit fsync failure."""


class _CreatedEntryRegistrationError(RuntimeError):
    """A regular-file entry was created but its descriptor was not admitted."""

    def __init__(
        self,
        message: str,
        *,
        retirement_records: tuple[DescriptorRetirementRecord, ...],
    ) -> None:
        self.retirement_records = retirement_records
        self.retirement_evidence = _retirement_evidence_from_records(
            retirement_records
        )
        super().__init__(message)


class _DescriptorAdmissionFailure(RuntimeError):
    """A fresh descriptor could not become a stable private authority."""

    def __init__(
        self,
        *,
        retirement_records: tuple[DescriptorRetirementRecord, ...],
    ) -> None:
        self.retirement_records = retirement_records
        self.retirement_evidence = _retirement_evidence_from_records(
            retirement_records
        )
        super().__init__("fresh descriptor admission failed")


class _DescriptorRetirementAnomaly(Exception):
    """Internal bounded retirement evidence awaiting public transport."""

    def __init__(
        self,
        evidence: DescriptorRetirementEvidence,
        *,
        primary: BaseException | None = None,
    ) -> None:
        self.evidence = evidence
        self.primary = primary
        super().__init__("one or more descriptor retirements were anomalous")


class _CallerChunkFailure(Exception):
    """Preserve a caller iterator failure across owned-entry retirement."""

    def __init__(self, primary: BaseException) -> None:
        self.primary = primary
        super().__init__("caller-provided chunk iteration failed")


class _IncompleteStagedTreeError(PublicationValidationError):
    """An owned staged tree is valid so far but not yet sealable."""


@dataclass(frozen=True, slots=True)
class _LedgerIdentity:
    public: PublicationEntryIdentity
    group_id: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _TreeAuthority:
    root: _LedgerIdentity
    entries: tuple[tuple[PortableRelativePath, _LedgerIdentity], ...]
    inventory: RegularFileInventory | None


@dataclass(frozen=True, slots=True)
class _DescriptorRegistration:
    generation: int
    identity: DescriptorRetirementIdentity | None
    role: _DescriptorRole


@dataclass(frozen=True, slots=True)
class _OwnedDescriptor:
    descriptor: int
    generation: int
    role: _DescriptorRole
    admitted_identity: DescriptorRetirementIdentity | None


@dataclass(slots=True)
class _StagingAuthorityState:
    trusted_root: TrustedRoot
    backend: _NativeBackend
    parent_device: int
    parent_inode: int
    destination: PortableRelativePath
    staging: PortableRelativePath
    kind: Literal["regular_file", "directory"]
    lifecycle: StagingState
    root_identity: _LedgerIdentity
    tree: _TreeAuthority
    parent_descriptor: int
    staging_descriptor: int
    parent_generation: int | None = None
    staging_generation: int | None = None
    parent_retirement_identity: DescriptorRetirementIdentity | None = None
    staging_retirement_identity: DescriptorRetirementIdentity | None = None
    retirement_batch_pending: bool = True
    error_provenance: object = field(
        default_factory=object,
        repr=False,
        compare=False,
    )
    size_bytes: int = 0
    publication_attempted: bool = False
    cleanup_attempted: bool = False
    active_operation: Literal[
        "write",
        "mkdir",
        "write_file",
        "seal",
        "publish",
        "cleanup",
    ] | None = None
    operation_lock: object = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )


@dataclass(slots=True)
class _ContextExitOrigin:
    explicit_exit_started: bool = False


class _StagedHandle:
    __slots__ = ("__weakref__",)

    def __new__(cls, *args: object, **kwargs: object):
        raise TypeError(
            f"{cls.__name__} handles are created by H2c1 staging contexts"
        )

    @property
    def state(self) -> StagingState:
        state = _staging_state(self)
        if state is None:
            return StagingState.RETIRED
        with state.operation_lock:  # type: ignore[attr-defined]
            return state.lifecycle

    def __copy__(self):
        raise TypeError(f"{type(self).__name__} handles cannot be copied")

    def __deepcopy__(self, memo: object):
        raise TypeError(f"{type(self).__name__} handles cannot be copied")

    def __reduce__(self):
        raise TypeError(f"{type(self).__name__} handles cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        raise TypeError(f"{type(self).__name__} handles cannot be serialized")


class StagedFileHandle(_StagedHandle):
    __slots__ = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("StagedFileHandle cannot be subclassed")

    def write(self, chunk: bytes) -> None:
        with _locked_handle_state(
            self,
            StagedFileHandle,
            operation="write",
        ):
            _write_staged_file(self, chunk)

    def seal(self, *, executable: bool = False) -> None:
        with _locked_handle_state(
            self,
            StagedFileHandle,
            operation="seal",
        ):
            _seal_staged_file(self, executable=executable)


class StagedDirectoryHandle(_StagedHandle):
    __slots__ = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("StagedDirectoryHandle cannot be subclassed")

    def mkdir(self, path: PortableRelativePath) -> None:
        with _locked_handle_state(
            self,
            StagedDirectoryHandle,
            operation="mkdir",
        ):
            _mkdir_staged_directory(self, path)

    def write_file(
        self,
        path: PortableRelativePath,
        chunks: Iterable[bytes],
        *,
        executable: bool = False,
    ) -> None:
        with _locked_handle_state(
            self,
            StagedDirectoryHandle,
            operation="write_file",
        ):
            _write_staged_directory_file(
                self,
                path,
                chunks,
                executable=executable,
            )

    def seal(self, *, scope: str) -> None:
        with _locked_handle_state(
            self,
            StagedDirectoryHandle,
            operation="seal",
        ):
            _seal_staged_directory(self, scope=scope)


def _create_staging_registry():
    live: weakref.WeakKeyDictionary[
        _StagedHandle, _StagingAuthorityState
    ] = weakref.WeakKeyDictionary()
    registry_lock = threading.Lock()

    def state_for(handle: object) -> _StagingAuthorityState | None:
        if type(handle) not in {StagedFileHandle, StagedDirectoryHandle}:
            return None
        with registry_lock:
            return live.get(handle)

    def allocate(
        handle_type: type[StagedFileHandle] | type[StagedDirectoryHandle],
        state: _StagingAuthorityState,
    ) -> StagedFileHandle | StagedDirectoryHandle:
        handle = object.__new__(handle_type)
        with registry_lock:
            live[handle] = state
        weakref.finalize(handle, _finalize_staging_state, state)
        return handle

    return state_for, allocate


_staging_state, _staging_allocator = _create_staging_registry()
del _create_staging_registry


def _finalize_staging_state(state: _StagingAuthorityState) -> None:
    with state.operation_lock:  # type: ignore[attr-defined]
        if state.lifecycle in {
            StagingState.OPEN,
            StagingState.SEALED,
            StagingState.NOT_COMMITTED,
        }:
            state.lifecycle = StagingState.RETIRED
        evidence = _retire_state_descriptors(
            state,
            raise_on_anomaly=False,
        )
        if evidence is not None:
            raise DescriptorRetirementError(
                state=StagingState.RETIRED,
                operation="finalization",
                destination=state.destination,
                staging=state.staging,
                terminal_result=None,
                retirement_evidence=evidence,
            )


def _validate_uint(name: str, value: object, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{name} is outside its frozen integer range")
    return value


def _validate_host_uint(name: str, raw: object) -> int:
    if type(raw) is not int:
        raise ValueError(f"{name} is outside its frozen integer range")
    value = raw if raw >= 0 else raw + 2**64
    return _validate_uint(name, value, _MAX_ENTRY_ID)


def _entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "other"


_RETIREMENT_ROLE_ORDER = (
    "traversal_entry",
    "traversal_directory",
    "traversal_parent",
    "operation_staging",
    "operation_parent",
    "handle_staging",
    "handle_parent",
)
_POST_OUTCOME_RETIREMENT_ROLES = frozenset(
    {
        "operation_staging",
        "operation_parent",
        "handle_staging",
        "handle_parent",
    }
)
_EXACT_H2C1_ERROR_TYPES: tuple[type[BaseException], ...] = (
    StagingLifecycleError,
    StagingAuthorityError,
    PublicationError,
    PublicationValidationError,
    PublicationCapabilityError,
    PublicationCollisionError,
    StagingAdmissionError,
    PublicationDurabilityError,
    PublicationOutcomeUncertainError,
    PublicationNamespaceConflictError,
    PublicationNamespaceUncertainError,
    StagingCleanupError,
)


def _raw_error_provenance() -> object | None:
    return getattr(_ERROR_PROVENANCE, "owner", None)


def _current_error_provenance(
    *,
    caller_depth: int = 2,
) -> object | None:
    owner = _raw_error_provenance()
    if owner is None:
        return None
    try:
        caller = sys._getframe(caller_depth)
    except (AttributeError, ValueError):
        return None
    if caller.f_globals is not globals():
        return None
    return owner


@contextmanager
def _error_provenance_scope(owner: object) -> Iterator[None]:
    previous = _raw_error_provenance()
    _ERROR_PROVENANCE.owner = owner
    try:
        yield
    finally:
        if previous is None:
            try:
                del _ERROR_PROVENANCE.owner
            except AttributeError:
                pass
        else:
            _ERROR_PROVENANCE.owner = previous


@contextmanager
def _suspend_error_provenance() -> Iterator[None]:
    previous = _raw_error_provenance()
    try:
        try:
            del _ERROR_PROVENANCE.owner
        except AttributeError:
            pass
        yield
    finally:
        if previous is not None:
            _ERROR_PROVENANCE.owner = previous


def _allocate_h2c1_error(
    owner: object,
    error_type: type[BaseException],
    *args: object,
    **kwargs: object,
) -> BaseException:
    if error_type not in _EXACT_H2C1_ERROR_TYPES:
        raise TypeError("private allocator requires an exact H2c1 error type")
    with _error_provenance_scope(owner):
        return error_type(*args, **kwargs)


def _make_retirement_identity(
    result: os.stat_result,
) -> DescriptorRetirementIdentity:
    if type(result) is not os.stat_result:
        raise TypeError("retirement identity requires exact stat_result")

    value = object.__new__(DescriptorRetirementIdentity)
    object.__setattr__(
        value,
        "device",
        _validate_host_uint("device", result.st_dev),
    )
    object.__setattr__(
        value,
        "inode",
        _validate_host_uint("inode", result.st_ino),
    )
    object.__setattr__(value, "entry_type", _entry_type(result.st_mode))
    object.__setattr__(
        value,
        "owner_uid",
        _validate_host_uint("owner_uid", result.st_uid),
    )
    return value


def _make_retirement_record(
    *,
    ordinal: int,
    role: str,
    observation: DescriptorRetirementObservation,
    close_attempted: bool,
    admitted_identity: DescriptorRetirementIdentity | None,
    observed_identity: DescriptorRetirementIdentity | None,
    error_errno: int | None,
) -> DescriptorRetirementRecord:
    if type(ordinal) is not int or ordinal < 0 or ordinal > 6:
        raise ValueError("retirement ordinal is outside its frozen range")
    if type(role) is not str:
        raise TypeError("descriptor-retirement role must be exact str")
    if role not in _RETIREMENT_ROLE_ORDER:
        raise ValueError("invalid descriptor-retirement role")
    if type(observation) is not DescriptorRetirementObservation:
        raise TypeError("retirement observation must be exact")
    if type(close_attempted) is not bool:
        raise TypeError("close_attempted must be exactly bool")
    if admitted_identity is not None and (
        type(admitted_identity) is not DescriptorRetirementIdentity
    ):
        raise TypeError("admitted retirement identity must be exact")
    if observed_identity is not None and (
        type(observed_identity) is not DescriptorRetirementIdentity
    ):
        raise TypeError("observed retirement identity must be exact")
    if error_errno is not None and (
        type(error_errno) is not int
        or error_errno < 1
        or error_errno > _MAX_ERRNO
    ):
        raise ValueError("retirement errno is outside its frozen range")

    if observation is DescriptorRetirementObservation.CLOSED:
        valid = (
            close_attempted
            and admitted_identity is not None
            and observed_identity == admitted_identity
            and error_errno is None
        )
    elif observation is DescriptorRetirementObservation.ALREADY_ABSENT:
        valid = (
            not close_attempted
            and observed_identity is None
            and error_errno == errno.EBADF
        )
    elif observation is DescriptorRetirementObservation.FOREIGN_PRESERVED:
        valid = (
            not close_attempted
            and error_errno is None
            and (
                observed_identity is None
                or (
                    admitted_identity is not None
                    and observed_identity != admitted_identity
                )
            )
        )
    elif observation is DescriptorRetirementObservation.UNINSPECTABLE:
        valid = (
            not close_attempted
            and observed_identity is None
        )
    else:
        valid = (
            close_attempted
            and admitted_identity is not None
            and observed_identity == admitted_identity
        )
    if not valid:
        raise ValueError("descriptor-retirement observation is inconsistent")

    value = object.__new__(DescriptorRetirementRecord)
    object.__setattr__(value, "ordinal", ordinal)
    object.__setattr__(value, "role", role)
    object.__setattr__(value, "observation", observation)
    object.__setattr__(value, "close_attempted", close_attempted)
    object.__setattr__(value, "admitted_identity", admitted_identity)
    object.__setattr__(value, "observed_identity", observed_identity)
    object.__setattr__(value, "error_errno", error_errno)
    return value


def _make_retirement_evidence(
    records: tuple[DescriptorRetirementRecord, ...],
    *,
    post_outcome: bool = False,
) -> DescriptorRetirementEvidence:
    if type(post_outcome) is not bool:
        raise TypeError("post_outcome must be exactly bool")
    if type(records) is not tuple or not 1 <= len(records) <= 7:
        raise ValueError("retirement evidence requires one through seven records")
    roles: list[str] = []
    for index, record in enumerate(records):
        if type(record) is not DescriptorRetirementRecord:
            raise TypeError("retirement evidence records must be exact")
        if record.ordinal != index:
            raise ValueError("retirement record ordinal must match tuple position")
        roles.append(record.role)
    if len(set(roles)) != len(roles):
        raise ValueError("each retirement role may occur at most once")
    if roles != sorted(roles, key=_RETIREMENT_ROLE_ORDER.index):
        raise ValueError("retirement records must use frozen role order")
    if all(
        record.observation is DescriptorRetirementObservation.CLOSED
        for record in records
    ):
        raise ValueError("retirement evidence requires at least one anomaly")
    if post_outcome and (
        len(records) > 4
        or any(role not in _POST_OUTCOME_RETIREMENT_ROLES for role in roles)
    ):
        raise ValueError("post-outcome retirement evidence exceeds its bound")
    value = object.__new__(DescriptorRetirementEvidence)
    object.__setattr__(value, "records", records)
    return value


def _combine_retirement_evidence(
    *values: DescriptorRetirementEvidence | None,
    post_outcome: bool = False,
) -> DescriptorRetirementEvidence | None:
    records = [
        record
        for value in values
        if value is not None
        for record in value.records
    ]
    if not records:
        return None
    ordered = sorted(
        records,
        key=lambda record: _RETIREMENT_ROLE_ORDER.index(record.role),
    )
    rebuilt = tuple(
        _make_retirement_record(
            ordinal=index,
            role=record.role,
            observation=record.observation,
            close_attempted=record.close_attempted,
            admitted_identity=record.admitted_identity,
            observed_identity=record.observed_identity,
            error_errno=record.error_errno,
        )
        for index, record in enumerate(ordered)
    )
    return _make_retirement_evidence(rebuilt, post_outcome=post_outcome)


def _validate_descriptor_retirement_error(
    *,
    state: StagingState,
    operation: str,
    destination: PortableRelativePath,
    staging: PortableRelativePath,
    terminal_result: PublicationResult | StagingCleanupResult | None,
    retirement_evidence: DescriptorRetirementEvidence,
) -> None:
    if type(state) is not StagingState:
        raise TypeError("retirement error state must be exact StagingState")
    if type(operation) is not str:
        raise TypeError("retirement error operation must be exact str")
    if (
        type(destination) is not PortableRelativePath
        or len(destination.parts) != 1
        or type(staging) is not PortableRelativePath
        or len(staging.parts) != 1
    ):
        raise TypeError("retirement error paths must be exact one-component paths")
    if type(retirement_evidence) is not DescriptorRetirementEvidence:
        raise TypeError("retirement error evidence must be exact")
    if operation == "publish":
        valid = (
            state is StagingState.PUBLISHED
            and type(terminal_result) is PublicationResult
            and terminal_result.state is PublicationState.COMMITTED_DURABLE
        )
        checked = _make_retirement_evidence(
            retirement_evidence.records,
            post_outcome=True,
        )
        if len(checked.records) > 3:
            raise ValueError(
                "publication retirement evidence exceeds three descriptors"
            )
    elif operation == "cleanup":
        valid = (
            state is StagingState.DISCARDED
            and type(terminal_result) is StagingCleanupResult
            and terminal_result.state
            is StagingCleanupState.DISCARDED_DURABLE
        )
        _make_retirement_evidence(
            retirement_evidence.records,
            post_outcome=True,
        )
    elif operation in {"write", "mkdir", "write_file", "seal"}:
        valid = (
            state is StagingState.RETIRED and terminal_result is None
        )
    elif operation == "context_exit":
        valid = (
            state in {StagingState.RETIRED, StagingState.NOT_COMMITTED}
            and terminal_result is None
        )
        if (
            len(retirement_evidence.records) > 2
            or any(
                record.role not in {"handle_staging", "handle_parent"}
                for record in retirement_evidence.records
            )
        ):
            raise ValueError(
                "context-exit retirement evidence exceeds handle authority"
            )
    elif operation == "finalization":
        valid = (
            state is StagingState.RETIRED and terminal_result is None
        )
        if (
            len(retirement_evidence.records) > 2
            or any(
                record.role not in {"handle_staging", "handle_parent"}
                for record in retirement_evidence.records
            )
        ):
            raise ValueError(
                "finalization retirement evidence exceeds handle authority"
            )
    else:
        valid = False
    if not valid:
        raise ValueError("invalid descriptor-retirement error tuple")


def _attach_retirement_evidence(
    error: BaseException,
    owner: object,
    evidence: DescriptorRetirementEvidence,
) -> bool:
    if type(error) not in _EXACT_H2C1_ERROR_TYPES:
        return False
    if getattr(error, "_h2c1_provenance", None) is not owner:
        return False
    if getattr(error, "_retirement_evidence", None) is not None:
        return False
    object.__setattr__(error, "_retirement_evidence", evidence)
    return True


def _queue_retirement_evidence(
    error: BaseException,
    owner: object,
    evidence: DescriptorRetirementEvidence,
) -> bool:
    if type(error) not in _EXACT_H2C1_ERROR_TYPES:
        return False
    if getattr(error, "_h2c1_provenance", None) is not owner:
        return False
    if getattr(error, "_retirement_evidence", None) is not None:
        return False
    pending = getattr(error, "_pending_retirement_evidence", None)
    combined = _combine_retirement_evidence(pending, evidence)
    object.__setattr__(
        error,
        "_pending_retirement_evidence",
        combined,
    )
    return True


def _internal_admission_retirement_records(
    error: BaseException | None,
) -> tuple[DescriptorRetirementRecord, ...]:
    current = error
    seen: set[int] = set()
    seen_batches: set[int] = set()
    records: list[DescriptorRetirementRecord] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current) in {
            _DescriptorAdmissionFailure,
            _CreatedEntryRegistrationError,
        }:
            batch = getattr(current, "retirement_records", ())
            if id(batch) not in seen_batches:
                records.extend(batch)
                seen_batches.add(id(batch))
        current = current.__cause__
    return tuple(records)


def _internal_admission_retirement_evidence(
    error: BaseException | None,
) -> DescriptorRetirementEvidence | None:
    return _retirement_evidence_from_records(
        _internal_admission_retirement_records(error)
    )


def _retire_with_primary_error(
    state: _StagingAuthorityState,
    error: BaseException,
    *,
    operation: Literal[
        "write",
        "mkdir",
        "write_file",
        "seal",
        "context_exit",
    ] = "context_exit",
    additional: tuple[_OwnedDescriptor, ...] = (),
    post_outcome: bool = False,
    prior_evidence: DescriptorRetirementEvidence | None = None,
    prior_includes_primary_admission: bool = False,
) -> None:
    pending_evidence = getattr(
        error,
        "_pending_retirement_evidence",
        None,
    )
    current_records = _retire_state_descriptor_records(
        state,
        additional=additional,
    )
    evidence = _retirement_evidence_from_records(
        (
            *(
                pending_evidence.records
                if pending_evidence is not None
                else ()
            ),
            *(
                ()
                if (
                    prior_includes_primary_admission
                    or pending_evidence is not None
                )
                else _internal_admission_retirement_records(error)
            ),
            *(prior_evidence.records if prior_evidence is not None else ()),
            *current_records,
        ),
        post_outcome=post_outcome,
    )
    if type(error) in _EXACT_H2C1_ERROR_TYPES:
        object.__setattr__(
            error,
            "_pending_retirement_evidence",
            None,
        )
    if evidence is not None:
        if not _attach_retirement_evidence(
            error,
            state.error_provenance,
            evidence,
        ):
            retirement_error = DescriptorRetirementError(
                state=(
                    state.lifecycle
                    if state.lifecycle
                    in {StagingState.RETIRED, StagingState.NOT_COMMITTED}
                    else StagingState.RETIRED
                ),
                operation=operation,
                destination=state.destination,
                staging=state.staging,
                terminal_result=None,
                retirement_evidence=evidence,
            )
            raise BaseExceptionGroup(
                "H2c1 operation and descriptor retirement both failed",
                [error, retirement_error],
            ) from None
    raise error


def _retire_with_terminal_result(
    state: _StagingAuthorityState,
    operation: Literal["publish", "cleanup"],
    result: PublicationResult | StagingCleanupResult,
    *,
    additional: tuple[_OwnedDescriptor, ...] = (),
    prior_evidence: DescriptorRetirementEvidence | None = None,
) -> PublicationResult | StagingCleanupResult:
    current_records = _retire_state_descriptor_records(
        state,
        additional=additional,
    )
    evidence = _retirement_evidence_from_records(
        (
            *(prior_evidence.records if prior_evidence is not None else ()),
            *current_records,
        ),
        post_outcome=True,
    )
    if evidence is not None:
        raise DescriptorRetirementError(
            state=state.lifecycle,
            operation=operation,
            destination=state.destination,
            staging=state.staging,
            terminal_result=result,
            retirement_evidence=evidence,
        )
    return result


def _raise_after_terminal_result_failure(
    state: _StagingAuthorityState,
    *,
    operation: Literal["publish", "cleanup"],
    primary: BaseException,
    terminal_result: PublicationResult | StagingCleanupResult,
    additional: tuple[_OwnedDescriptor, ...],
    prior_evidence: DescriptorRetirementEvidence | None = None,
) -> None:
    """Consume a proven terminal batch without masking result-build failure."""

    current_records = _retire_state_descriptor_records(
        state,
        additional=additional,
    )
    evidence = _retirement_evidence_from_records(
        (
            *(prior_evidence.records if prior_evidence is not None else ()),
            *current_records,
        ),
        post_outcome=True,
    )
    if evidence is not None:
        retirement_error = DescriptorRetirementError(
            state=state.lifecycle,
            operation=operation,
            destination=state.destination,
            staging=state.staging,
            terminal_result=terminal_result,
            retirement_evidence=evidence,
        )
        raise BaseExceptionGroup(
            "terminal result construction and descriptor retirement both "
            "failed",
            [primary, retirement_error],
        ) from None
    raise primary


def _compose_population_retirement(
    state: _StagingAuthorityState,
    *,
    operation: Literal["write", "mkdir", "write_file", "seal"],
    primary: BaseException | None,
    candidates: tuple[_OwnedDescriptor, ...],
) -> BaseException | None:
    retirement = _retirement_evidence_from_records(
        (
            *_internal_admission_retirement_records(primary),
            *(
                _retire_owned_descriptor_records(candidates)
                if candidates
                else ()
            ),
        )
    )
    if retirement is None:
        return primary
    state.lifecycle = StagingState.RETIRED
    if type(primary) in {
        _DescriptorAdmissionFailure,
        _CreatedEntryRegistrationError,
    }:
        primary = PublicationValidationError(
            "population descriptor could not be admitted safely",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    return _DescriptorRetirementAnomaly(
        retirement,
        primary=primary,
    )


def _preserve_population_primary(error: BaseException) -> bool:
    return (
        type(error) in _EXACT_H2C1_ERROR_TYPES
        or type(error) is _IncompleteStagedTreeError
        or isinstance(
            error,
            (
                TypeError,
                ValueError,
                DescriptorRetirementError,
                BaseExceptionGroup,
            ),
        )
    )


def _raise_population_failure(
    state: _StagingAuthorityState,
    failure: BaseException,
    *,
    operation: Literal["mkdir", "write_file", "seal"],
    message: str,
) -> None:
    primary = failure
    prior_includes_primary_admission = False
    preserve_caller_primary = False
    if type(failure) is _DescriptorRetirementAnomaly:
        prior_evidence = failure.evidence
        primary = failure.primary
        prior_includes_primary_admission = True
    else:
        prior_evidence = _internal_admission_retirement_evidence(failure)
    if type(primary) is _CallerChunkFailure:
        primary = primary.primary
        preserve_caller_primary = True
    state.lifecycle = StagingState.RETIRED
    if type(primary) in {
        _DescriptorAdmissionFailure,
        _CreatedEntryRegistrationError,
    }:
        primary = PublicationValidationError(
            "population descriptor could not be admitted safely",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    if primary is None:
        current_records = _retire_state_descriptor_records(state)
        evidence = _retirement_evidence_from_records(
            (
                *(
                    prior_evidence.records
                    if prior_evidence is not None
                    else ()
                ),
                *current_records,
            )
        )
        if evidence is None:
            raise RuntimeError(
                "descriptor-retirement anomaly carried no evidence"
            )
        raise DescriptorRetirementError(
            state=StagingState.RETIRED,
            operation=operation,
            destination=state.destination,
            staging=state.staging,
            terminal_result=None,
            retirement_evidence=evidence,
        )
    if (
        prior_includes_primary_admission
        or preserve_caller_primary
        or _preserve_population_primary(primary)
    ):
        _retire_with_primary_error(
            state,
            primary,
            operation=operation,
            prior_evidence=prior_evidence,
            prior_includes_primary_admission=(
                prior_includes_primary_admission
            ),
        )
    error = PublicationValidationError(
        message,
        state=PublicationState.NOT_COMMITTED,
        evidence=None,
        destination=state.destination,
    )
    _retire_with_primary_error(
        state,
        error,
        operation=operation,
        prior_evidence=prior_evidence,
    )


def _make_entry_identity(result: os.stat_result) -> PublicationEntryIdentity:
    device = _validate_host_uint("device", result.st_dev)
    inode = _validate_host_uint("inode", result.st_ino)
    link_count = _validate_uint(
        "link_count",
        result.st_nlink,
        _MAX_LINK_COUNT,
    )
    if link_count == 0:
        raise ValueError("link_count must be positive")
    owner_uid = _validate_host_uint("owner_uid", result.st_uid)
    mode = _validate_uint("mode", result.st_mode, 0o177777)
    size_bytes = _validate_uint("size_bytes", result.st_size, MAX_SIGNED_64)
    value = object.__new__(PublicationEntryIdentity)
    object.__setattr__(value, "device", device)
    object.__setattr__(value, "inode", inode)
    object.__setattr__(value, "entry_type", _entry_type(result.st_mode))
    object.__setattr__(value, "link_count", link_count)
    object.__setattr__(value, "owner_uid", owner_uid)
    object.__setattr__(value, "mode", mode)
    object.__setattr__(value, "size_bytes", size_bytes)
    return value


def _make_ledger_identity(result: os.stat_result) -> _LedgerIdentity:
    return _LedgerIdentity(
        public=_make_entry_identity(result),
        group_id=_validate_host_uint("group_id", result.st_gid),
        modified_ns=result.st_mtime_ns,
        changed_ns=result.st_ctime_ns,
    )


def _make_namespace_evidence(
    observation: _NamespaceObservation,
    *,
    reference: PortableRelativePath | None = None,
    aliases: tuple[PortableRelativePath, ...] = (),
) -> NamespaceEvidence:
    if type(observation) is not _NamespaceObservation:
        raise TypeError("namespace observation must be internal authority")
    if type(aliases) is not tuple:
        raise TypeError("namespace aliases must be an exact tuple")
    if reference is not None and (
        type(reference) is not PortableRelativePath
        or len(reference.parts) != 1
    ):
        raise TypeError(
            "namespace reference must be one exact component or None"
        )
    for alias in aliases:
        if type(alias) is not PortableRelativePath or len(alias.parts) != 1:
            raise TypeError("namespace aliases must be one-component paths")
    if aliases:
        if type(reference) is not PortableRelativePath:
            raise TypeError("alias evidence requires an exact reference path")
        reference_key = reference.value.lower()
        if any(
            alias.value == reference.value
            or alias.value.lower() != reference_key
            for alias in aliases
        ):
            raise ValueError(
                "namespace aliases must be differently spelled case aliases"
            )
    ordered = tuple(sorted(aliases, key=portable_path_sort_key))
    if ordered != aliases or len({alias.value for alias in aliases}) != len(
        aliases
    ):
        raise ValueError("namespace aliases must be unique and sorted")

    if observation is _NamespaceObservation.NOT_ATTEMPTED:
        if aliases:
            raise ValueError("not-attempted namespace evidence has no aliases")
        count: int | None = None
        complete = False
    elif observation is _NamespaceObservation.NO_CONFLICT:
        if aliases:
            raise ValueError("no-conflict namespace evidence has no aliases")
        count = 0
        complete = True
    elif observation is _NamespaceObservation.COMPLETE_CONFLICT:
        if not aliases or len(aliases) > _MAX_EVIDENCE_ALIASES:
            raise ValueError("complete conflict requires bounded aliases")
        count = len(aliases)
        complete = True
    elif observation is _NamespaceObservation.BOUNDED_CONFLICT:
        if len(aliases) != _MAX_EVIDENCE_ALIASES:
            raise ValueError("bounded conflict requires the frozen alias cap")
        count = None
        complete = False
    else:
        if len(aliases) > _MAX_EVIDENCE_ALIASES:
            raise ValueError("uninspectable namespace evidence is bounded")
        count = None
        complete = False

    value = object.__new__(NamespaceEvidence)
    object.__setattr__(value, "namespace_observation", observation.value)
    object.__setattr__(value, "conflicting_aliases", aliases)
    object.__setattr__(value, "conflicting_alias_count", count)
    object.__setattr__(value, "aliases_complete", complete)
    return value


def _make_publication_evidence(
    *,
    staging_identity: PublicationEntryIdentity,
    source_observation: str,
    observed_source_identity: PublicationEntryIdentity | None,
    destination_observation: str,
    observed_destination_identity: PublicationEntryIdentity | None,
    namespace_evidence: NamespaceEvidence,
    parent_fsync: str,
    native_errno: int | None,
    allow_not_attempted: bool = False,
) -> PublicationRecoveryEvidence:
    observations = {
        "not_attempted",
        "exact",
        "absent",
        "foreign",
        "replaced",
        "contradictory",
        "uninspectable",
    }
    if type(allow_not_attempted) is not bool:
        raise TypeError("allow_not_attempted must be exactly bool")
    if type(staging_identity) is not PublicationEntryIdentity:
        raise TypeError("staging identity must be exact")
    if staging_identity.entry_type not in {"regular_file", "directory"}:
        raise ValueError("staging identity must be a file or directory")
    if type(source_observation) is not str:
        raise TypeError("source observation must be exact str")
    if type(destination_observation) is not str:
        raise TypeError("destination observation must be exact str")
    if source_observation not in observations:
        raise ValueError("invalid source observation")
    if destination_observation not in observations:
        raise ValueError("invalid destination observation")
    if not allow_not_attempted and (
        source_observation == "not_attempted"
        or destination_observation == "not_attempted"
    ):
        raise ValueError("post-admission evidence cannot be not-attempted")
    _validate_observed_identity(
        source_observation,
        observed_source_identity,
        staging_identity,
    )
    _validate_observed_identity(
        destination_observation,
        observed_destination_identity,
        staging_identity,
    )
    if type(namespace_evidence) is not NamespaceEvidence:
        raise TypeError("namespace evidence must be exact")
    if type(parent_fsync) is not str:
        raise TypeError("parent fsync observation must be exact str")
    if parent_fsync not in {
        "not_attempted",
        "succeeded",
        "failed",
        "uncertain",
    }:
        raise ValueError("invalid parent fsync observation")
    if not allow_not_attempted and parent_fsync == "not_attempted":
        raise ValueError("post-admission evidence must observe parent fsync")
    if native_errno is not None:
        if (
            type(native_errno) is not int
            or native_errno < 1
            or native_errno > _MAX_ERRNO
        ):
            raise ValueError("native errno is outside its frozen range")
    return _allocate_publication_evidence(
        staging_identity=staging_identity,
        source_observation=source_observation,
        observed_source_identity=observed_source_identity,
        destination_observation=destination_observation,
        observed_destination_identity=observed_destination_identity,
        namespace_evidence=namespace_evidence,
        parent_fsync=parent_fsync,
        native_errno=native_errno,
    )


def _allocate_publication_evidence(
    *,
    staging_identity: PublicationEntryIdentity,
    source_observation: str,
    observed_source_identity: PublicationEntryIdentity | None,
    destination_observation: str,
    observed_destination_identity: PublicationEntryIdentity | None,
    namespace_evidence: NamespaceEvidence,
    parent_fsync: str,
    native_errno: int | None,
) -> PublicationRecoveryEvidence:
    """Allocate evidence after the caller has established its invariants."""

    value = object.__new__(PublicationRecoveryEvidence)
    object.__setattr__(value, "staging_identity", staging_identity)
    object.__setattr__(value, "source_observation", source_observation)
    object.__setattr__(
        value,
        "observed_source_identity",
        observed_source_identity,
    )
    object.__setattr__(
        value,
        "destination_observation",
        destination_observation,
    )
    object.__setattr__(
        value,
        "observed_destination_identity",
        observed_destination_identity,
    )
    object.__setattr__(value, "namespace_evidence", namespace_evidence)
    object.__setattr__(value, "parent_fsync", parent_fsync)
    object.__setattr__(value, "native_errno", native_errno)
    return value


def _validate_observed_identity(
    observation: str,
    observed: PublicationEntryIdentity | None,
    original: PublicationEntryIdentity,
) -> None:
    if observation == "exact":
        if observed != original:
            raise ValueError("exact observation must equal staging identity")
    elif observation in {
        "not_attempted",
        "absent",
        "contradictory",
        "uninspectable",
    }:
        if observed is not None:
            raise ValueError(f"{observation} observation must not carry identity")
    elif observation in {"foreign", "replaced"}:
        if (
            type(observed) is not PublicationEntryIdentity
            or observed == original
        ):
            raise ValueError(
                f"{observation} observation requires unequal identity"
            )


def _make_cleanup_evidence(
    *,
    staging_identity: PublicationEntryIdentity,
    root_observation: str,
    observed_root_identity: PublicationEntryIdentity | None,
    remaining_expected_entries: int | None,
    namespace_evidence: NamespaceEvidence,
    parent_fsync: str,
    native_errno: int | None,
) -> StagingCleanupRecoveryEvidence:
    if type(staging_identity) is not PublicationEntryIdentity:
        raise TypeError("cleanup staging identity must be exact")
    if staging_identity.entry_type not in {"regular_file", "directory"}:
        raise ValueError(
            "cleanup staging identity must be a file or directory"
        )
    if type(root_observation) is not str:
        raise TypeError("cleanup root observation must be exact str")
    if observed_root_identity is not None and (
        type(observed_root_identity) is not PublicationEntryIdentity
    ):
        raise TypeError("observed cleanup identity must be exact or None")
    if root_observation not in {
        "exact",
        "owned_partial",
        "absent",
        "foreign",
        "replaced",
        "contradictory",
        "malformed",
        "uninspectable",
    }:
        raise ValueError("invalid cleanup root observation")
    if root_observation == "exact" and observed_root_identity != staging_identity:
        raise ValueError("exact cleanup observation requires staging identity")
    if root_observation in {"absent", "contradictory", "uninspectable"}:
        if observed_root_identity is not None:
            raise ValueError("cleanup observation must not carry identity")
    if root_observation in {"foreign", "replaced"}:
        if (
            type(observed_root_identity) is not PublicationEntryIdentity
            or observed_root_identity == staging_identity
        ):
            raise ValueError("foreign cleanup observation needs unequal identity")
    if root_observation == "owned_partial":
        if type(observed_root_identity) is not PublicationEntryIdentity:
            raise ValueError("owned-partial observation requires root identity")
        if (
            remaining_expected_entries is None
            or type(remaining_expected_entries) is not int
            or remaining_expected_entries < 0
            or remaining_expected_entries > MAX_CANONICAL_CONTAINER_ITEMS
        ):
            raise ValueError("owned-partial evidence needs bounded residue count")
    elif root_observation != "exact" and remaining_expected_entries is not None:
        raise ValueError("residue count is only valid for owned authority")
    if root_observation == "exact" and remaining_expected_entries is not None:
        if (
            type(remaining_expected_entries) is not int
            or remaining_expected_entries < 0
            or remaining_expected_entries > MAX_CANONICAL_CONTAINER_ITEMS
        ):
            raise ValueError("cleanup residue count is outside its bound")
    if type(namespace_evidence) is not NamespaceEvidence:
        raise TypeError("namespace evidence must be exact")
    if type(parent_fsync) is not str:
        raise TypeError("cleanup parent fsync must be exact str")
    if parent_fsync not in {
        "not_attempted",
        "succeeded",
        "failed",
        "uncertain",
    }:
        raise ValueError("invalid parent fsync observation")
    if native_errno is not None:
        if (
            type(native_errno) is not int
            or native_errno < 1
            or native_errno > _MAX_ERRNO
        ):
            raise ValueError("native errno is outside its frozen range")
    value = object.__new__(StagingCleanupRecoveryEvidence)
    object.__setattr__(value, "staging_identity", staging_identity)
    object.__setattr__(value, "root_observation", root_observation)
    object.__setattr__(
        value,
        "observed_root_identity",
        observed_root_identity,
    )
    object.__setattr__(
        value,
        "remaining_expected_entries",
        remaining_expected_entries,
    )
    object.__setattr__(value, "namespace_evidence", namespace_evidence)
    object.__setattr__(value, "parent_fsync", parent_fsync)
    object.__setattr__(value, "native_errno", native_errno)
    return value


def _make_publication_result(
    state: PublicationState,
    destination: PortableRelativePath,
    identity: PublicationEntryIdentity,
    namespace: NamespaceEvidence,
) -> PublicationResult:
    if type(state) is not PublicationState:
        raise TypeError("publication result state must be exact")
    if state is not PublicationState.COMMITTED_DURABLE:
        raise ValueError("normal publication result must be durable")
    if (
        type(destination) is not PortableRelativePath
        or len(destination.parts) != 1
    ):
        raise TypeError(
            "publication result destination must be one exact component"
        )
    if type(identity) is not PublicationEntryIdentity:
        raise TypeError("publication result identity must be exact")
    if identity.entry_type not in {"regular_file", "directory"}:
        raise ValueError(
            "publication result identity must be a file or directory"
        )
    if type(namespace) is not NamespaceEvidence:
        raise TypeError("publication result namespace must be exact")
    if (
        namespace.namespace_observation != "no_conflict"
        or not namespace.aliases_complete
    ):
        raise ValueError("normal publication result needs complete no-conflict")
    return _allocate_publication_result(
        state,
        destination,
        identity,
        namespace,
    )


def _allocate_publication_result(
    state: PublicationState,
    destination: PortableRelativePath,
    identity: PublicationEntryIdentity,
    namespace: NamespaceEvidence,
) -> PublicationResult:
    """Allocate a result after the caller has established its invariants."""

    value = object.__new__(PublicationResult)
    object.__setattr__(value, "state", state)
    object.__setattr__(value, "destination", destination)
    object.__setattr__(value, "destination_identity", identity)
    object.__setattr__(value, "namespace_evidence", namespace)
    return value


def _make_cleanup_result(
    staging: PortableRelativePath,
    identity: PublicationEntryIdentity,
    namespace: NamespaceEvidence,
) -> StagingCleanupResult:
    if type(staging) is not PortableRelativePath or len(staging.parts) != 1:
        raise TypeError(
            "cleanup result staging must be one exact component"
        )
    if type(identity) is not PublicationEntryIdentity:
        raise TypeError("cleanup result identity must be exact")
    if identity.entry_type not in {"regular_file", "directory"}:
        raise ValueError(
            "cleanup result identity must be a file or directory"
        )
    if type(namespace) is not NamespaceEvidence:
        raise TypeError("cleanup result namespace must be exact")
    if (
        namespace.namespace_observation != "no_conflict"
        or not namespace.aliases_complete
    ):
        raise ValueError("normal cleanup result needs complete no-conflict")
    return _allocate_cleanup_result(staging, identity, namespace)


def _allocate_cleanup_result(
    staging: PortableRelativePath,
    identity: PublicationEntryIdentity,
    namespace: NamespaceEvidence,
) -> StagingCleanupResult:
    """Allocate a result after the caller has established its invariants."""

    value = object.__new__(StagingCleanupResult)
    object.__setattr__(
        value,
        "state",
        StagingCleanupState.DISCARDED_DURABLE,
    )
    object.__setattr__(value, "staging", staging)
    object.__setattr__(value, "discarded_identity", identity)
    object.__setattr__(value, "namespace_evidence", namespace)
    return value


@dataclass(frozen=True, slots=True)
class _NativeBackend:
    function: object
    flag: int
    symbol: str


def _load_native_backend() -> _NativeBackend:
    if sys.platform.startswith("linux"):
        symbol = "renameat2"
        flag = _RENAME_NOREPLACE
    elif sys.platform == "darwin":
        symbol = "renameatx_np"
        flag = _RENAME_EXCL
    else:
        raise PublicationCapabilityError(
            "native atomic no-replace publication is unsupported",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )
    try:
        library = ctypes.CDLL(None, use_errno=True)
        function = getattr(library, symbol)
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
    except Exception as exc:
        raise PublicationCapabilityError(
            "native atomic no-replace publication backend is unavailable",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        ) from exc
    return _NativeBackend(function=function, flag=flag, symbol=symbol)


def _native_no_replace(
    backend: _NativeBackend,
    parent_descriptor: int,
    source: str,
    destination: str,
) -> tuple[int, int | None]:
    source_bytes = source.encode("ascii")
    destination_bytes = destination.encode("ascii")
    ctypes.set_errno(0)
    result = backend.function(
        parent_descriptor,
        source_bytes,
        parent_descriptor,
        destination_bytes,
        backend.flag,
    )
    captured = ctypes.get_errno()
    if result == 0:
        return result, None
    if captured < 1 or captured > _MAX_ERRNO:
        return result, None
    return result, captured


def _require_publication_capabilities() -> _NativeBackend:
    required_flags = ("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK", "O_DIRECTORY")
    if any(type(getattr(os, name, None)) is not int for name in required_flags):
        raise PublicationCapabilityError(
            "required POSIX descriptor flags are unavailable",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", ())
    supports_fd = getattr(os, "supports_fd", ())
    for function in (os.open, os.stat, os.mkdir, os.unlink, os.rmdir):
        if function not in supports_dir_fd:
            raise PublicationCapabilityError(
                "required descriptor-relative POSIX operation is unavailable",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=None,
            )
    if os.stat not in supports_follow_symlinks:
        raise PublicationCapabilityError(
            "required no-follow stat operation is unavailable",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )
    if os.scandir not in supports_fd:
        raise PublicationCapabilityError(
            "descriptor-based directory enumeration is unavailable",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )
    return _load_native_backend()


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
        | getattr(os, "O_NOCTTY", 0)
    )


def _write_file_flags() -> int:
    return (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
        | getattr(os, "O_NOCTTY", 0)
    )


def _read_file_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
        | getattr(os, "O_NOCTTY", 0)
    )


def _open_publication_parent(
    trusted_root: TrustedRoot,
    *,
    role: Literal["operation_parent", "handle_parent"],
) -> int:
    descriptor = _open_fresh_trusted_root_directory(trusted_root)
    return _admit_opened_descriptor(descriptor, role=role)


def _open_directory_at(
    name: str,
    parent_descriptor: int,
    *,
    role: Literal[
        "traversal_directory",
        "traversal_parent",
        "operation_staging",
        "handle_staging",
    ],
) -> int:
    descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
    return _admit_opened_descriptor(descriptor, role=role)


def _open_read_file_at(
    name: str,
    parent_descriptor: int,
    *,
    role: Literal["traversal_entry", "operation_staging"],
) -> int:
    descriptor = os.open(name, _read_file_flags(), dir_fd=parent_descriptor)
    return _admit_opened_descriptor(descriptor, role=role)


def _create_file_at(
    name: str,
    parent_descriptor: int,
    mode: int,
    *,
    role: Literal["traversal_entry", "handle_staging"],
) -> int:
    descriptor = os.open(
        name,
        _write_file_flags(),
        mode,
        dir_fd=parent_descriptor,
    )
    try:
        return _admit_opened_descriptor(descriptor, role=role)
    except _DescriptorAdmissionFailure as exc:
        raise _CreatedEntryRegistrationError(
            "created entry descriptor could not be admitted",
            retirement_records=exc.retirement_records,
        ) from exc


def _mkdir_at(name: str, parent_descriptor: int, mode: int) -> None:
    # POSIX mkdir applies the process umask before returning and provides no
    # descriptor for a newly created directory.  Serializing H2c1 directory
    # creation under a private lock lets this foundation request the exact
    # private mode without a path-following chmod fallback.  Unrestricted
    # in-process mutation is outside ADR-0023's threat boundary.
    with _PRIVATE_MODE_CREATION_LOCK:
        # The temporary value remains private even for unrelated cooperative
        # creation: it never grants group or other permissions.
        previous_umask = os.umask(0o077)
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_descriptor)
        finally:
            os.umask(previous_umask)


def _stat_at(name: str, parent_descriptor: int) -> os.stat_result:
    return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)


def _fstat_descriptor(descriptor: int) -> os.stat_result:
    return os.fstat(descriptor)


def _list_names(descriptor: int) -> Iterator[str]:
    with os.scandir(descriptor) as entries:
        for entry in entries:
            yield entry.name


def _directory_has_any_name(descriptor: int) -> bool:
    """Probe emptiness without materializing a late hostile enumeration."""

    names = iter(_list_names(descriptor))
    try:
        try:
            next(names)
        except StopIteration:
            return False
        return True
    finally:
        close = getattr(names, "close", None)
        if close is not None:
            close()


def _write_descriptor(descriptor: int, chunk: bytes) -> int:
    return os.write(descriptor, chunk)


def _read_descriptor(descriptor: int, size: int) -> bytes:
    return os.read(descriptor, size)


def _fchmod_descriptor(descriptor: int, mode: int) -> None:
    os.fchmod(descriptor, mode)


def _fsync_descriptor(descriptor: int) -> None:
    os.fsync(descriptor)


def _required_fsync(descriptor: int, description: str) -> None:
    try:
        _fsync_descriptor(descriptor)
    except OSError as exc:
        number = exc.errno if type(exc.errno) is int else errno.EIO
        raise _RequiredFsyncError(
            number,
            f"required {description} fsync failed",
        ) from exc


def _unlink_at(name: str, parent_descriptor: int) -> None:
    os.unlink(name, dir_fd=parent_descriptor)


def _rmdir_at(name: str, parent_descriptor: int) -> None:
    os.rmdir(name, dir_fd=parent_descriptor)


def _close_descriptor(descriptor: int) -> None:
    with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
        registration = _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor)
    try:
        os.close(descriptor)
    except BaseException:
        with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
            if _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor) is registration:
                _OPEN_DESCRIPTOR_IDENTITIES.pop(descriptor, None)
        raise
    else:
        with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
            if _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor) is registration:
                _OPEN_DESCRIPTOR_IDENTITIES.pop(descriptor, None)


def _register_fresh_descriptor(
    descriptor: int,
    *,
    role: _DescriptorRole,
) -> _DescriptorRegistration:
    with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
        provisional = _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor)
    if provisional is None or provisional.role != role:
        raise RuntimeError(
            "fresh descriptor lacks its provisional acquisition registration"
        )
    observed = os.fstat(descriptor)
    registration = _DescriptorRegistration(
        generation=provisional.generation,
        identity=_make_retirement_identity(observed),
        role=role,
    )
    with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
        if _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor) is not provisional:
            raise RuntimeError(
                "fresh descriptor registration changed during admission"
            )
        _OPEN_DESCRIPTOR_IDENTITIES[descriptor] = registration
    return registration


def _admit_opened_descriptor(
    descriptor: int,
    *,
    role: _DescriptorRole,
) -> int:
    """Register one just-opened descriptor or retire it before propagating."""

    provisional = _DescriptorRegistration(
        generation=next(_DESCRIPTOR_GENERATIONS),
        identity=None,
        role=role,
    )
    with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
        _OPEN_DESCRIPTOR_IDENTITIES[descriptor] = provisional
    try:
        _register_fresh_descriptor(descriptor, role=role)
    except BaseException as exc:
        with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
            current = _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor)
        owned = _OwnedDescriptor(
            descriptor=descriptor,
            generation=provisional.generation,
            role=role,
            admitted_identity=(
                current.identity
                if current is not None
                and current.generation == provisional.generation
                else None
            ),
        )
        retirement_records = _retire_owned_descriptor_records((owned,))
        raise _DescriptorAdmissionFailure(
            retirement_records=retirement_records,
        ) from exc
    return descriptor


def _registered_descriptor_matches(
    descriptor: int,
    current: os.stat_result,
    expected: PublicationEntryIdentity | None,
) -> bool:
    observed = _make_retirement_identity(current)
    with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
        admitted = _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor)
    if admitted is None or admitted.identity != observed:
        return False
    if expected is None:
        return True
    return observed == _retirement_identity_from_publication_identity(
        expected
    )


def _require_parent_security(
    trusted_root: TrustedRoot,
    parent_descriptor: int,
) -> _LedgerIdentity:
    pinned = _require_exact_live_root(trusted_root)
    opened = _fstat_descriptor(parent_descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_dev != pinned.st_dev
        or opened.st_ino != pinned.st_ino
    ):
        raise TrustedRootError("publication parent identity changed")
    if opened.st_uid != os.geteuid():
        raise PublicationValidationError(
            "publication parent must be owned by the effective user",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )
    permissions = stat.S_IMODE(opened.st_mode)
    if permissions & 0o700 != 0o700 or permissions & 0o022:
        raise PublicationValidationError(
            "publication parent requires owner rwx and no group/other write",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )
    return _make_ledger_identity(opened)


def _validate_destination(value: object) -> PortableRelativePath:
    if type(value) is not PortableRelativePath:
        raise TypeError("destination must be exactly PortableRelativePath")
    if len(value.parts) != 1:
        raise PublicationValidationError(
            "publication destination must be one path component",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=value,
        )
    if value.value.lower().startswith(_STAGING_PREFIX):
        raise PublicationValidationError(
            "publication destination uses the reserved staging prefix",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=value,
        )
    return value


def _staging_leaf(destination: PortableRelativePath) -> PortableRelativePath:
    key = destination.value.lower().encode("ascii")
    digest = domain_separated_sha256(PUBLICATION_STAGING_DOMAIN, key)
    return PortableRelativePath.parse(f"{_STAGING_PREFIX}{digest.value}")


def _same_authority(
    observed: PublicationEntryIdentity,
    expected: PublicationEntryIdentity,
) -> bool:
    return (
        observed.device == expected.device
        and observed.inode == expected.inode
        and observed.entry_type == expected.entry_type
        and observed.owner_uid == expected.owner_uid
    )


def _require_exact_ledger(
    result: os.stat_result,
    expected: _LedgerIdentity,
    *,
    message: str,
) -> None:
    if _make_ledger_identity(result) != expected:
        raise PublicationValidationError(
            message,
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )


def _retirement_identity_from_publication_identity(
    identity: PublicationEntryIdentity,
) -> DescriptorRetirementIdentity:
    value = object.__new__(DescriptorRetirementIdentity)
    object.__setattr__(value, "device", identity.device)
    object.__setattr__(value, "inode", identity.inode)
    object.__setattr__(value, "entry_type", identity.entry_type)
    object.__setattr__(value, "owner_uid", identity.owner_uid)
    return value


def _retirement_identity_from_parent_state(
    state: _StagingAuthorityState,
) -> DescriptorRetirementIdentity:
    def normalized(value: int) -> int:
        return value if value >= 0 else value + 2**64

    value = object.__new__(DescriptorRetirementIdentity)
    object.__setattr__(value, "device", normalized(state.parent_device))
    object.__setattr__(value, "inode", normalized(state.parent_inode))
    object.__setattr__(value, "entry_type", "directory")
    object.__setattr__(value, "owner_uid", normalized(os.geteuid()))
    return value


def _owned_descriptor(
    descriptor: int,
    role: _DescriptorRole | None,
    *,
    expected: PublicationEntryIdentity | None = None,
) -> _OwnedDescriptor:
    if descriptor < 0:
        raise ValueError("owned descriptor must be nonnegative")
    if role is not None and role not in _RETIREMENT_ROLE_ORDER:
        raise ValueError("invalid descriptor-retirement role")
    with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
        registration = _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor)
    if registration is None:
        raise RuntimeError("owned descriptor lacks acquisition registration")
    if role is not None and registration.role != role:
        raise RuntimeError(
            "owned descriptor role differs from its acquisition purpose"
        )
    admitted = registration.identity
    owned = _OwnedDescriptor(
        descriptor=descriptor,
        generation=registration.generation,
        role=registration.role,
        admitted_identity=admitted,
    )
    if expected is not None and admitted != (
        _retirement_identity_from_publication_identity(expected)
    ):
        retirement_records = _retire_owned_descriptor_records((owned,))
        raise _DescriptorAdmissionFailure(
            retirement_records=retirement_records,
        )
    return owned


def _descriptor_generation(descriptor: int) -> int:
    with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
        registration = _OPEN_DESCRIPTOR_IDENTITIES.get(descriptor)
    if registration is None:
        raise RuntimeError("owned descriptor was not registered at acquisition")
    return registration.generation


def _retire_owned_descriptor_records(
    owned: tuple[_OwnedDescriptor, ...],
) -> tuple[DescriptorRetirementRecord, ...]:
    ordered = tuple(
        sorted(owned, key=lambda item: _RETIREMENT_ROLE_ORDER.index(item.role))
    )
    if len({item.role for item in ordered}) != len(ordered):
        raise RuntimeError("internal descriptor retirement roles are duplicated")
    if len({item.descriptor for item in ordered}) != len(ordered):
        raise RuntimeError(
            "one owned descriptor cannot occupy multiple retirement roles"
        )
    if len(ordered) > 7:
        raise RuntimeError("internal descriptor retirement batch is unbounded")
    generation_matches: dict[int, bool] = {}
    with _OPEN_DESCRIPTOR_IDENTITIES_LOCK:
        for item in ordered:
            registered = _OPEN_DESCRIPTOR_IDENTITIES.get(item.descriptor)
            matches = (
                registered is not None
                and registered.generation == item.generation
            )
            generation_matches[item.descriptor] = matches
            if matches:
                del _OPEN_DESCRIPTOR_IDENTITIES[item.descriptor]
    raw: list[
        tuple[
            str,
            DescriptorRetirementObservation,
            bool,
            DescriptorRetirementIdentity | None,
            DescriptorRetirementIdentity | None,
            int | None,
        ]
    ] = []
    for item in ordered:
        if not generation_matches[item.descriptor]:
            raw.append(
                (
                    item.role,
                    DescriptorRetirementObservation.FOREIGN_PRESERVED,
                    False,
                    item.admitted_identity,
                    None,
                    None,
                )
            )
            continue
        if item.admitted_identity is None:
            raw.append(
                (
                    item.role,
                    DescriptorRetirementObservation.UNINSPECTABLE,
                    False,
                    None,
                    None,
                    None,
                )
            )
            continue
        try:
            current = _fstat_descriptor(item.descriptor)
        except OSError as exc:
            observation = (
                DescriptorRetirementObservation.ALREADY_ABSENT
                if exc.errno == errno.EBADF
                else DescriptorRetirementObservation.UNINSPECTABLE
            )
            raw.append(
                (
                    item.role,
                    observation,
                    False,
                    item.admitted_identity,
                    None,
                    (
                        errno.EBADF
                        if observation
                        is DescriptorRetirementObservation.ALREADY_ABSENT
                        else _errno_from_exception(exc)
                    ),
                )
            )
            continue
        except BaseException:
            raw.append(
                (
                    item.role,
                    DescriptorRetirementObservation.UNINSPECTABLE,
                    False,
                    item.admitted_identity,
                    None,
                    None,
                )
            )
            continue
        try:
            observed = _make_retirement_identity(current)
        except BaseException as exc:
            raw.append(
                (
                    item.role,
                    DescriptorRetirementObservation.UNINSPECTABLE,
                    False,
                    item.admitted_identity,
                    None,
                    _errno_from_exception(exc),
                )
            )
            continue
        if observed != item.admitted_identity:
            raw.append(
                (
                    item.role,
                    DescriptorRetirementObservation.FOREIGN_PRESERVED,
                    False,
                    item.admitted_identity,
                    observed,
                    None,
                )
            )
            continue
        try:
            os.close(item.descriptor)
        except BaseException as exc:
            raw.append(
                (
                    item.role,
                    DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
                    True,
                    item.admitted_identity,
                    observed,
                    _errno_from_exception(exc),
                )
            )
        else:
            raw.append(
                (
                    item.role,
                    DescriptorRetirementObservation.CLOSED,
                    True,
                    item.admitted_identity,
                    observed,
                    None,
                )
            )
    return tuple(
        _make_retirement_record(
            ordinal=index,
            role=row[0],
            observation=row[1],
            close_attempted=row[2],
            admitted_identity=row[3],
            observed_identity=row[4],
            error_errno=row[5],
        )
        for index, row in enumerate(raw)
    )


def _retirement_evidence_from_records(
    records: tuple[DescriptorRetirementRecord, ...],
    *,
    post_outcome: bool = False,
) -> DescriptorRetirementEvidence | None:
    if not records or all(
        record.observation is DescriptorRetirementObservation.CLOSED
        for record in records
    ):
        return None
    ordered = sorted(
        records,
        key=lambda record: _RETIREMENT_ROLE_ORDER.index(record.role),
    )
    rebuilt = tuple(
        _make_retirement_record(
            ordinal=index,
            role=record.role,
            observation=record.observation,
            close_attempted=record.close_attempted,
            admitted_identity=record.admitted_identity,
            observed_identity=record.observed_identity,
            error_errno=record.error_errno,
        )
        for index, record in enumerate(ordered)
    )
    return _make_retirement_evidence(
        rebuilt,
        post_outcome=post_outcome,
    )


def _retire_owned_descriptors(
    owned: tuple[_OwnedDescriptor, ...],
    *,
    post_outcome: bool = False,
) -> DescriptorRetirementEvidence | None:
    return _retirement_evidence_from_records(
        _retire_owned_descriptor_records(owned),
        post_outcome=post_outcome,
    )


def _close_owned_descriptor(
    descriptor: int,
    expected: PublicationEntryIdentity | None,
    *,
    role: _DescriptorRole | None = None,
) -> None:
    if descriptor < 0:
        return
    evidence = _retire_owned_descriptors(
        (_owned_descriptor(descriptor, role, expected=expected),)
    )
    if evidence is not None:
        raise _DescriptorRetirementAnomaly(evidence)


def _close_fresh_descriptor(
    descriptor: int,
    expected: PublicationEntryIdentity | None,
    *,
    role: Literal[
        "traversal_entry",
        "traversal_directory",
        "traversal_parent",
    ] | None = None,
) -> None:
    """Close one operation-private descriptor, preserving any proven reuse."""

    if descriptor < 0:
        return
    _close_owned_descriptor(descriptor, expected, role=role)


def _raise_after_local_retirement(
    primary: BaseException | None,
    candidates: tuple[_OwnedDescriptor, ...],
    *,
    prior_evidence: DescriptorRetirementEvidence | None = None,
) -> None:
    primary_admission_is_included = False
    if type(primary) is _DescriptorRetirementAnomaly:
        prior_evidence = _combine_retirement_evidence(
            prior_evidence,
            primary.evidence,
        )
        primary = primary.primary
        primary_admission_is_included = True
    retirement = _retirement_evidence_from_records(
        (
            *(prior_evidence.records if prior_evidence is not None else ()),
            *(
                ()
                if primary_admission_is_included
                else _internal_admission_retirement_records(primary)
            ),
            *(
                _retire_owned_descriptor_records(candidates)
                if candidates
                else ()
            ),
        )
    )
    if retirement is not None:
        raise _DescriptorRetirementAnomaly(
            retirement,
            primary=primary,
        ) from primary
    if primary is not None:
        raise primary


def _retire_state_descriptor_records(
    state: _StagingAuthorityState,
    *,
    additional: tuple[_OwnedDescriptor, ...] = (),
) -> tuple[DescriptorRetirementRecord, ...]:
    if not state.retirement_batch_pending and not additional:
        return ()
    staging_descriptor = state.staging_descriptor
    parent_descriptor = state.parent_descriptor
    was_pending = state.retirement_batch_pending
    state.staging_descriptor = -1
    state.parent_descriptor = -1
    state.retirement_batch_pending = False
    owned = list(additional)
    if was_pending and staging_descriptor >= 0:
        owned.append(
            _OwnedDescriptor(
                descriptor=staging_descriptor,
                generation=(
                    state.staging_generation
                    if state.staging_generation is not None
                    else -1
                ),
                role="handle_staging",
                admitted_identity=(
                    state.staging_retirement_identity
                    if state.staging_retirement_identity is not None
                    else _retirement_identity_from_publication_identity(
                        state.root_identity.public
                    )
                ),
            )
        )
    if was_pending and parent_descriptor >= 0:
        owned.append(
            _OwnedDescriptor(
                descriptor=parent_descriptor,
                generation=(
                    state.parent_generation
                    if state.parent_generation is not None
                    else -1
                ),
                role="handle_parent",
                admitted_identity=(
                    state.parent_retirement_identity
                    if state.parent_retirement_identity is not None
                    else _retirement_identity_from_parent_state(state)
                ),
            )
        )
    return _retire_owned_descriptor_records(tuple(owned))


def _retire_state_descriptors(
    state: _StagingAuthorityState,
    *,
    additional: tuple[_OwnedDescriptor, ...] = (),
    post_outcome: bool = False,
    raise_on_anomaly: bool = True,
) -> DescriptorRetirementEvidence | None:
    evidence = _retirement_evidence_from_records(
        _retire_state_descriptor_records(
            state,
            additional=additional,
        ),
        post_outcome=post_outcome,
    )
    if evidence is not None and raise_on_anomaly:
        raise _DescriptorRetirementAnomaly(evidence)
    return evidence


@contextmanager
def _open_staging_context_impl(
    trusted_root: TrustedRoot,
    *,
    destination: PortableRelativePath,
    kind: Literal["regular_file", "directory"],
    allocator: object,
    exit_origin: _ContextExitOrigin,
) -> Iterator[StagedFileHandle | StagedDirectoryHandle]:
    parent_descriptor = -1
    staging_descriptor = -1
    state: _StagingAuthorityState | None = None
    validated_destination: PortableRelativePath | None = None
    staging: PortableRelativePath | None = None
    created = False
    yielded = False
    provisional: _LedgerIdentity | None = None
    staging_close_identity: PublicationEntryIdentity | None = None
    parent_owned: _OwnedDescriptor | None = None
    staging_owned: _OwnedDescriptor | None = None
    admission_provenance = object()
    preyield_retirement: DescriptorRetirementEvidence | None = None
    admission_allocator = _error_provenance_scope(admission_provenance)
    admission_allocator.__enter__()
    admission_allocator_active = True
    escaping_error: BaseException | None = None
    try:
        if type(trusted_root) is not TrustedRoot:
            raise TypeError("trusted_root must be exactly TrustedRoot")
        validated_destination = _validate_destination(destination)
        staging = _staging_leaf(validated_destination)
        try:
            backend = _require_publication_capabilities()
        except PublicationCapabilityError as exc:
            raise PublicationCapabilityError(
                str(exc),
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=validated_destination,
            ) from exc

        try:
            parent_descriptor = _open_publication_parent(
                trusted_root,
                role="handle_parent",
            )
        except TrustedRootError as exc:
            raise PublicationValidationError(
                "trusted publication parent is not live",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=validated_destination,
            ) from exc
        parent_owned = _owned_descriptor(
            parent_descriptor,
            "handle_parent",
        )
        try:
            parent = _require_parent_security(
                trusted_root,
                parent_descriptor,
            )
        except (PublicationError, TrustedRootError, OSError) as exc:
            raise PublicationValidationError(
                str(exc),
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=validated_destination,
            ) from exc
        parent_sync = _safe_parent_fsync(
            parent_descriptor,
            expected_device=parent.public.device,
            expected_inode=parent.public.inode,
        )
        if parent_sync != "succeeded":
            raise PublicationCapabilityError(
                "publication-parent directory fsync is unavailable",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=validated_destination,
            )
        try:
            parent_before_creation = _require_parent_security(
                trusted_root,
                parent_descriptor,
            )
        except (PublicationError, TrustedRootError, OSError) as exc:
            raise PublicationValidationError(
                str(exc),
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=validated_destination,
            ) from exc
        if parent_before_creation != parent:
            raise PublicationValidationError(
                "publication parent changed before staging creation",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=validated_destination,
            )

        try:
            if kind == "regular_file":
                try:
                    staging_descriptor = _create_file_at(
                        staging.value,
                        parent_descriptor,
                        0o600,
                        role="handle_staging",
                    )
                except _CreatedEntryRegistrationError:
                    created = True
                    raise
                created = True
                staging_close_identity = _make_entry_identity(
                    _fstat_descriptor(staging_descriptor)
                )
                try:
                    staging_owned = _owned_descriptor(
                        staging_descriptor,
                        "handle_staging",
                        expected=staging_close_identity,
                    )
                except _DescriptorAdmissionFailure:
                    staging_descriptor = -1
                    raise
                _fchmod_descriptor(staging_descriptor, 0o600)
            else:
                _mkdir_at(staging.value, parent_descriptor, 0o700)
                created = True
                staging_descriptor = _open_directory_at(
                    staging.value,
                    parent_descriptor,
                    role="handle_staging",
                )
                staging_close_identity = _make_entry_identity(
                    _fstat_descriptor(staging_descriptor)
                )
                try:
                    staging_owned = _owned_descriptor(
                        staging_descriptor,
                        "handle_staging",
                        expected=staging_close_identity,
                    )
                except _DescriptorAdmissionFailure:
                    staging_descriptor = -1
                    raise
                _fchmod_descriptor(staging_descriptor, 0o700)
        except FileExistsError as exc:
            raise PublicationCollisionError(
                "deterministic staging reservation already exists",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=validated_destination,
            ) from exc

        provisional = _admit_created_staging(
            parent_descriptor,
            staging_descriptor,
            staging,
            kind=kind,
            trusted_root=trusted_root,
            expected_parent=parent,
        )
        parent_before_yield = _require_parent_security(
            trusted_root,
            parent_descriptor,
        )
        opened_before_yield = _make_ledger_identity(
            _fstat_descriptor(staging_descriptor)
        )
        named_before_yield = _make_ledger_identity(
            _stat_at(staging.value, parent_descriptor)
        )
        parent_after_yield_check = _require_parent_security(
            trusted_root,
            parent_descriptor,
        )
        if (
            not _same_cleanup_directory_authority(
                parent_before_yield,
                parent,
            )
            or parent_after_yield_check != parent_before_yield
            or opened_before_yield != provisional
            or named_before_yield != provisional
        ):
            raise PublicationValidationError(
                "staging authority changed before handle admission",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=validated_destination,
            )
        tree = _TreeAuthority(
            root=provisional,
            entries=(),
            inventory=None,
        )
        state = _StagingAuthorityState(
            trusted_root=trusted_root,
            backend=backend,
            parent_device=parent.public.device,
            parent_inode=parent.public.inode,
            destination=validated_destination,
            staging=staging,
            kind=kind,
            lifecycle=StagingState.OPEN,
            root_identity=provisional,
            tree=tree,
            parent_descriptor=parent_descriptor,
            staging_descriptor=staging_descriptor,
            parent_generation=parent_owned.generation,
            staging_generation=staging_owned.generation,
            parent_retirement_identity=parent_owned.admitted_identity,
            staging_retirement_identity=staging_owned.admitted_identity,
            error_provenance=object(),
        )
        handle_type = (
            StagedFileHandle if kind == "regular_file" else StagedDirectoryHandle
        )
        admitted_handle = [
            allocator(handle_type, state)  # type: ignore[operator]
        ]
        parent_descriptor = -1
        staging_descriptor = -1
        admission_allocator.__exit__(None, None, None)
        admission_allocator_active = False
        yielded = True
        try:
            yield admitted_handle.pop()
        except BaseException as body_error:
            if (
                type(body_error) is GeneratorExit
                and not exit_origin.explicit_exit_started
            ):
                with state.operation_lock:  # type: ignore[attr-defined]
                    if state.lifecycle in {
                        StagingState.OPEN,
                        StagingState.SEALED,
                        StagingState.NOT_COMMITTED,
                    }:
                        state.lifecycle = StagingState.RETIRED
                    retirement = _retire_state_descriptors(
                        state,
                        raise_on_anomaly=False,
                    )
                if retirement is not None:
                    raise DescriptorRetirementError(
                        state=StagingState.RETIRED,
                        operation="finalization",
                        destination=state.destination,
                        staging=state.staging,
                        terminal_result=None,
                        retirement_evidence=retirement,
                    ) from None
                return
            with state.operation_lock:  # type: ignore[attr-defined]
                if state.lifecycle in {StagingState.OPEN, StagingState.SEALED}:
                    state.lifecycle = StagingState.RETIRED
                retirement = _retire_state_descriptors(
                    state,
                    raise_on_anomaly=False,
                )
            if retirement is None:
                raise
            if _attach_retirement_evidence(
                body_error,
                state.error_provenance,
                retirement,
            ):
                raise
            retirement_error = DescriptorRetirementError(
                state=state.lifecycle,
                operation="context_exit",
                destination=state.destination,
                staging=state.staging,
                terminal_result=None,
                retirement_evidence=retirement,
            )
            raise BaseExceptionGroup(
                "context body and descriptor retirement both failed",
                [body_error, retirement_error],
            ) from None
        else:
            with state.operation_lock:  # type: ignore[attr-defined]
                if state.lifecycle in {StagingState.OPEN, StagingState.SEALED}:
                    state.lifecycle = StagingState.RETIRED
                retirement = _retire_state_descriptors(
                    state,
                    raise_on_anomaly=False,
                )
            if retirement is not None:
                raise DescriptorRetirementError(
                    state=state.lifecycle,
                    operation="context_exit",
                    destination=state.destination,
                    staging=state.staging,
                    terminal_result=None,
                    retirement_evidence=retirement,
                )
    except BaseException as exc:
        escaping_error = exc
        if yielded:
            raise
        internal_admission_failure = type(exc) in {
            _DescriptorAdmissionFailure,
            _CreatedEntryRegistrationError,
        }
        if (
            (created or internal_admission_failure)
            and staging is not None
            and validated_destination is not None
        ):
            if state is not None:
                state.lifecycle = StagingState.RETIRED
                preyield_retirement = _retire_state_descriptors(
                    state,
                    raise_on_anomaly=False,
                )
                parent_descriptor = -1
                staging_descriptor = -1
            evidence: PublicationRecoveryEvidence | None = None
            if provisional is not None:
                evidence = _staging_admission_failure_evidence(
                    parent_descriptor=parent_descriptor,
                    staging_descriptor=staging_descriptor,
                    staging=staging,
                    destination=validated_destination,
                    provisional=provisional,
                    failure=exc,
                )
            admission_error = StagingAdmissionError(
                "staging entry could not be admitted safely",
                state=PublicationState.NOT_COMMITTED,
                evidence=evidence,
                destination=validated_destination,
                staging=staging,
                entry_may_remain=created,
            )
            escaping_error = admission_error
            raise admission_error from exc
        raise
    finally:
        if admission_allocator_active:
            admission_allocator.__exit__(None, None, None)
            admission_allocator_active = False
        # A yielded state owns and retires these slots.  Before yield the
        # provisional admission roles are fixed by acquisition purpose.
        if not yielded:
            candidates: list[_OwnedDescriptor] = []
            if staging_descriptor >= 0:
                candidates.append(
                    staging_owned
                    if staging_owned is not None
                    else _owned_descriptor(
                        staging_descriptor,
                        "handle_staging",
                    )
                )
                staging_descriptor = -1
            if parent_descriptor >= 0:
                candidates.append(
                    parent_owned
                    if parent_owned is not None
                    else _owned_descriptor(
                        parent_descriptor,
                        "handle_parent",
                    )
                )
                parent_descriptor = -1
            if candidates:
                local_records = _retire_owned_descriptor_records(
                    tuple(candidates)
                )
            else:
                local_records = ()
            active = escaping_error
            retirement = _retirement_evidence_from_records(
                (
                    *_internal_admission_retirement_records(active),
                    *(
                        preyield_retirement.records
                        if preyield_retirement is not None
                        else ()
                    ),
                    *local_records,
                )
            )
            if retirement is not None:
                if active is not None:
                    if _attach_retirement_evidence(
                        active,
                        admission_provenance,
                        retirement,
                    ):
                        pass
                    elif (
                        staging is not None
                        and validated_destination is not None
                    ):
                        admission_error = _allocate_h2c1_error(
                            admission_provenance,
                            StagingAdmissionError,
                            "staging admission and descriptor retirement "
                            "both failed",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=validated_destination,
                            staging=staging,
                            entry_may_remain=created,
                        )
                        _attach_retirement_evidence(
                            admission_error,
                            admission_provenance,
                            retirement,
                        )
                        raise admission_error from active
                else:
                    if staging is None or validated_destination is None:
                        raise RuntimeError(
                            "pre-yield descriptor retirement lacked "
                            "validated staging identity"
                        )
                    admission_error = _allocate_h2c1_error(
                        admission_provenance,
                        StagingAdmissionError,
                        "staging admission descriptor retirement failed",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=validated_destination,
                        staging=staging,
                        entry_may_remain=created,
                    )
                    _attach_retirement_evidence(
                        admission_error,
                        admission_provenance,
                        retirement,
                    )
                    raise admission_error


def _staging_admission_failure_evidence(
    *,
    parent_descriptor: int,
    staging_descriptor: int,
    staging: PortableRelativePath,
    destination: PortableRelativePath,
    provisional: _LedgerIdentity,
    failure: BaseException,
) -> PublicationRecoveryEvidence:
    source_observation = "uninspectable"
    observed_source: PublicationEntryIdentity | None = None
    try:
        opened = _make_entry_identity(
            _fstat_descriptor(staging_descriptor)
        )
        named_observation, named_identity = _safe_observe_named_entry(
            parent_descriptor,
            staging.value,
            provisional.public,
        )
        if opened == provisional.public and named_observation == "exact":
            source_observation = "exact"
            observed_source = provisional.public
        elif (
            named_observation in {"foreign", "replaced"}
            and named_identity is not None
            and (
                opened == provisional.public
                or opened == named_identity
            )
        ):
            source_observation = "replaced"
            observed_source = named_identity
        elif named_observation == "uninspectable":
            source_observation = "uninspectable"
        else:
            source_observation = "contradictory"
    except BaseException:
        source_observation = "uninspectable"
        observed_source = None
    return _make_publication_evidence(
        staging_identity=provisional.public,
        source_observation=source_observation,
        observed_source_identity=observed_source,
        destination_observation="not_attempted",
        observed_destination_identity=None,
        namespace_evidence=_make_namespace_evidence(
            _NamespaceObservation.NOT_ATTEMPTED,
            reference=destination,
        ),
        parent_fsync="not_attempted",
        native_errno=_errno_from_exception(failure),
        allow_not_attempted=True,
    )


def _bind_staging_entrypoints(allocator: object, context_impl: object):
    class BoundStagingContext:
        __slots__ = (
            "_inner",
            "_exit_origin",
            "_entered",
            "_exited",
            "_handle_ref",
        )

        def __init__(
            self,
            inner: AbstractContextManager[
                StagedFileHandle | StagedDirectoryHandle
            ],
            exit_origin: _ContextExitOrigin,
        ) -> None:
            self._inner = inner
            self._exit_origin = exit_origin
            self._entered = False
            self._exited = False
            self._handle_ref = None

        def __enter__(
            self,
        ) -> StagedFileHandle | StagedDirectoryHandle:
            if self._entered or self._exited:
                raise RuntimeError("staging context cannot be entered twice")
            self._entered = True
            try:
                handle = self._inner.__enter__()
                self._handle_ref = weakref.ref(handle)
                return handle
            except BaseException:
                self._exited = True
                raise

        def __exit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> bool | None:
            if not self._entered:
                self._exited = True
                raise RuntimeError("staging context was not entered")
            if self._exited:
                raise RuntimeError("staging context has already exited")

            def exit_inner() -> bool | None:
                self._exited = True
                self._exit_origin.explicit_exit_started = True
                return self._inner.__exit__(
                    exception_type,
                    exception,
                    traceback,
                )

            handle = (
                self._handle_ref()
                if self._handle_ref is not None
                else None
            )
            state = _staging_state(handle)
            if state is None:
                return exit_inner()
            with state.operation_lock:  # type: ignore[attr-defined]
                if state.active_operation is not None:
                    raise RuntimeError(
                        "staging context cannot exit while a handle operation "
                        "is active"
                    )
                return exit_inner()

    def open_exclusive_staged_file(
        trusted_root: TrustedRoot,
        *,
        destination: PortableRelativePath,
    ) -> AbstractContextManager[StagedFileHandle]:
        """Return a lazy context that exclusively creates one staged file."""

        exit_origin = _ContextExitOrigin()
        return BoundStagingContext(
            context_impl(  # type: ignore[operator]
                trusted_root,
                destination=destination,
                kind="regular_file",
                allocator=allocator,
                exit_origin=exit_origin,
            ),
            exit_origin,
        )

    def open_exclusive_staged_directory(
        trusted_root: TrustedRoot,
        *,
        destination: PortableRelativePath,
    ) -> AbstractContextManager[StagedDirectoryHandle]:
        """Return a lazy context that exclusively creates one staged tree."""

        exit_origin = _ContextExitOrigin()
        return BoundStagingContext(
            context_impl(  # type: ignore[operator]
                trusted_root,
                destination=destination,
                kind="directory",
                allocator=allocator,
                exit_origin=exit_origin,
            ),
            exit_origin,
        )

    return open_exclusive_staged_file, open_exclusive_staged_directory


(
    open_exclusive_staged_file,
    open_exclusive_staged_directory,
) = _bind_staging_entrypoints(
    _staging_allocator,
    _open_staging_context_impl,
)
del _bind_staging_entrypoints
del _open_staging_context_impl
del _staging_allocator


def _admit_created_staging(
    parent_descriptor: int,
    staging_descriptor: int,
    staging: PortableRelativePath,
    *,
    kind: Literal["regular_file", "directory"],
    trusted_root: TrustedRoot,
    expected_parent: _LedgerIdentity,
) -> _LedgerIdentity:
    parent_before = _require_parent_security(
        trusted_root,
        parent_descriptor,
    )
    opened = _fstat_descriptor(staging_descriptor)
    before = _stat_at(staging.value, parent_descriptor)
    after = _stat_at(staging.value, parent_descriptor)
    parent_after = _require_parent_security(
        trusted_root,
        parent_descriptor,
    )
    opened_ledger = _make_ledger_identity(opened)
    if (
        not _same_cleanup_directory_authority(
            parent_before,
            expected_parent,
        )
        or not _same_cleanup_directory_authority(
            parent_after,
            expected_parent,
        )
        or parent_before != parent_after
        or _make_ledger_identity(before) != opened_ledger
        or _make_ledger_identity(after) != opened_ledger
        or opened_ledger.public.device != expected_parent.public.device
        or opened.st_uid != os.geteuid()
        or (kind == "regular_file" and not stat.S_ISREG(opened.st_mode))
        or (kind == "directory" and not stat.S_ISDIR(opened.st_mode))
        or (kind == "regular_file" and opened.st_nlink != 1)
    ):
        raise PublicationValidationError(
            "created staging identity is unsafe or unstable",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )
    expected_mode = 0o600 if kind == "regular_file" else 0o700
    if stat.S_IMODE(opened.st_mode) != expected_mode:
        raise PublicationValidationError(
            "created staging mode could not be established",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=None,
        )
    return opened_ledger


def _errno_from_exception(exc: BaseException) -> int | None:
    if not isinstance(exc, OSError):
        return None
    try:
        value = exc.errno
    except BaseException:
        return None
    if type(value) is int and 1 <= value <= _MAX_ERRNO:
        return value
    return None


def _close_parent_descriptor(
    descriptor: int,
    *,
    expected_device: int | None,
    expected_inode: int | None,
    role: Literal[
        "traversal_directory",
        "traversal_parent",
        "operation_parent",
    ] | None = None,
) -> None:
    if descriptor < 0:
        return
    _close_owned_descriptor(
        descriptor,
        None,
        role=role,
    )


def _require_handle(
    handle: object,
    expected_type: type[StagedFileHandle] | type[StagedDirectoryHandle],
    *,
    operation: Literal[
        "write",
        "mkdir",
        "write_file",
        "seal",
        "publish",
        "cleanup",
    ],
    allowed: tuple[StagingState, ...],
) -> _StagingAuthorityState:
    if type(handle) is not expected_type:
        raise TypeError(f"{operation} requires exactly {expected_type.__name__}")
    state = _staging_state(handle)
    if state is None:
        raise StagingLifecycleError(StagingState.RETIRED, operation)
    if state.lifecycle not in allowed:
        raise StagingLifecycleError(state.lifecycle, operation)
    return state


@contextmanager
def _nonreentrant_handle_operation(
    state: _StagingAuthorityState,
    operation: Literal[
        "write",
        "mkdir",
        "write_file",
        "seal",
        "publish",
        "cleanup",
    ],
) -> Iterator[None]:
    if state.active_operation is not None:
        raise StagingLifecycleError(state.lifecycle, operation)
    state.active_operation = operation
    try:
        yield
    finally:
        state.active_operation = None


@contextmanager
def _locked_handle_state(
    handle: object,
    expected_type: type[StagedFileHandle] | type[StagedDirectoryHandle],
    *,
    operation: Literal[
        "write",
        "mkdir",
        "write_file",
        "seal",
        "publish",
        "cleanup",
    ],
    terminal_authority: tuple[
        object,
        tuple[StagingState, ...],
    ]
    | None = None,
) -> Iterator[_StagingAuthorityState]:
    if terminal_authority is not None:
        trusted_root, _allowed = terminal_authority
        if type(trusted_root) is not TrustedRoot:
            raise TypeError("trusted_root must be exactly TrustedRoot")
    if type(handle) is not expected_type:
        raise TypeError(f"{operation} requires exactly {expected_type.__name__}")
    state = _staging_state(handle)
    if state is None:
        raise StagingLifecycleError(StagingState.RETIRED, operation)
    with state.operation_lock:  # type: ignore[attr-defined]
        if _staging_state(handle) is not state:
            raise StagingLifecycleError(StagingState.RETIRED, operation)
        with _error_provenance_scope(state.error_provenance):
            if terminal_authority is not None:
                trusted_root, allowed = terminal_authority
                if operation not in {"publish", "cleanup"}:
                    raise AssertionError(
                        "terminal authority supplied for a population "
                        "operation"
                    )
                _require_terminal_authority(
                    trusted_root,
                    handle,
                    expected_type,
                    operation=operation,
                    allowed=allowed,
                )
            with _nonreentrant_handle_operation(state, operation):
                try:
                    yield state
                except _DescriptorRetirementAnomaly as anomaly:
                    primary = anomaly.primary
                    if operation in {"publish", "cleanup"}:
                        if (
                            primary is not None
                            and _attach_retirement_evidence(
                                primary,
                                state.error_provenance,
                                anomaly.evidence,
                            )
                        ):
                            raise primary
                        error = StagingAuthorityError(
                            state.lifecycle,
                            operation,
                        )
                        _attach_retirement_evidence(
                            error,
                            state.error_provenance,
                            anomaly.evidence,
                        )
                        if primary is not None:
                            raise BaseExceptionGroup(
                                "terminal operation and descriptor retirement "
                                "both failed",
                                [primary, error],
                            ) from None
                        raise error from anomaly
                    state.lifecycle = StagingState.RETIRED
                    remaining_records = _retire_state_descriptor_records(
                        state
                    )
                    evidence = _retirement_evidence_from_records(
                        (
                            *anomaly.evidence.records,
                            *remaining_records,
                        )
                    )
                    if primary is not None and _attach_retirement_evidence(
                        primary,
                        state.error_provenance,
                        evidence,
                    ):
                        raise primary
                    retirement_error = DescriptorRetirementError(
                        state=StagingState.RETIRED,
                        operation=operation,
                        destination=state.destination,
                        staging=state.staging,
                        terminal_result=None,
                        retirement_evidence=evidence,
                    )
                    if primary is not None:
                        raise BaseExceptionGroup(
                            "staging operation and descriptor retirement both "
                            "failed",
                            [primary, retirement_error],
                        ) from None
                    raise retirement_error from anomaly


def _require_population_authority(
    state: _StagingAuthorityState,
) -> None:
    try:
        _require_exact_live_root(state.trusted_root)
        parent = _require_parent_security(
            state.trusted_root,
            state.parent_descriptor,
        )
        if (
            parent.public.device != state.parent_device
            or parent.public.inode != state.parent_inode
        ):
            raise TrustedRootError("staging parent identity changed")
    except BaseException as exc:
        state.lifecycle = StagingState.RETIRED
        error = PublicationValidationError(
            "staging authority is no longer live and pinned",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
        try:
            _retire_with_primary_error(state, error)
        except BaseException as transported:
            raise transported from exc


def _require_terminal_authority(
    trusted_root: object,
    handle: object,
    expected_type: type[StagedFileHandle] | type[StagedDirectoryHandle],
    *,
    operation: Literal["publish", "cleanup"],
    allowed: tuple[StagingState, ...],
) -> _StagingAuthorityState:
    if type(trusted_root) is not TrustedRoot:
        raise TypeError("trusted_root must be exactly TrustedRoot")
    if type(handle) is not expected_type:
        raise TypeError(f"{operation} requires exactly {expected_type.__name__}")
    state = _staging_state(handle)
    if state is None:
        raise StagingLifecycleError(StagingState.RETIRED, operation)
    if state.trusted_root is not trusted_root or not trusted_root.is_open:
        raise StagingAuthorityError(state.lifecycle, operation)
    if state.lifecycle not in allowed:
        raise StagingLifecycleError(state.lifecycle, operation)
    try:
        _require_exact_live_root(trusted_root)
    except (TrustedRootError, OSError, ValueError) as exc:
        raise StagingAuthorityError(state.lifecycle, operation) from exc
    return state


def _write_staged_file(handle: StagedFileHandle, chunk: bytes) -> None:
    if type(chunk) is not bytes:
        raise TypeError("staged file chunks must be exactly bytes")
    state = _require_handle(
        handle,
        StagedFileHandle,
        operation="write",
        allowed=(StagingState.OPEN,),
    )
    if state.size_bytes > MAX_SIGNED_64 - len(chunk):
        raise ValueError("staged file exceeds the signed-64 size bound")
    _require_population_authority(state)
    previous_size = state.size_bytes
    try:
        _write_all(
            state.staging_descriptor,
            chunk,
            destination=state.destination,
        )
    except BaseException as exc:
        try:
            _reconcile_owned_file_mutation(
                state,
                minimum_size=previous_size,
                maximum_size=previous_size + len(chunk),
                allowed_modes=(stat.S_IMODE(state.root_identity.public.mode),),
            )
        except BaseException as reconciliation_error:
            state.lifecycle = StagingState.RETIRED
            error = PublicationValidationError(
                "failed staged write left unverifiable file authority",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
            try:
                _retire_with_primary_error(state, error)
            except BaseException as transported:
                raise transported from reconciliation_error
        if state.size_bytes != previous_size:
            state.lifecycle = StagingState.RETIRED
            error = PublicationValidationError(
                "partial staged write was preserved and authority retired",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
            try:
                _retire_with_primary_error(state, error)
            except BaseException as transported:
                raise transported from exc
        if isinstance(exc, OSError):
            raise PublicationValidationError(
                "staged file write failed after bounded reconciliation",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            ) from exc
        raise
    _reconcile_owned_file_mutation(
        state,
        minimum_size=previous_size + len(chunk),
        maximum_size=previous_size + len(chunk),
        allowed_modes=(stat.S_IMODE(state.root_identity.public.mode),),
    )


def _write_all(
    descriptor: int,
    chunk: bytes,
    *,
    destination: PortableRelativePath | None = None,
) -> None:
    offset = 0
    while offset < len(chunk):
        try:
            written = _write_descriptor(descriptor, chunk[offset:])
        except InterruptedError:
            continue
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            raise
        if type(written) is not int or written <= 0:
            raise PublicationValidationError(
                "staged write made no progress",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=destination,
            )
        if written > len(chunk) - offset:
            raise PublicationValidationError(
                "staged write exceeded requested bytes",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=destination,
            )
        offset += written


def _reconcile_owned_file_mutation(
    state: _StagingAuthorityState,
    *,
    minimum_size: int,
    maximum_size: int,
    allowed_modes: tuple[int, ...],
) -> None:
    opened = _make_ledger_identity(_fstat_descriptor(state.staging_descriptor))
    named = _make_ledger_identity(
        _stat_at(state.staging.value, state.parent_descriptor)
    )
    if (
        opened != named
        or opened.public.entry_type != "regular_file"
        or opened.public.link_count != 1
        or opened.public.owner_uid != os.geteuid()
        or opened.public.device != state.parent_device
        or opened.public.inode != state.root_identity.public.inode
        or not minimum_size <= opened.public.size_bytes <= maximum_size
        or stat.S_IMODE(opened.public.mode) not in allowed_modes
    ):
        state.lifecycle = StagingState.RETIRED
        error = PublicationValidationError(
            "staged file identity changed during population",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
        _retire_with_primary_error(state, error)
    state.size_bytes = opened.public.size_bytes
    state.root_identity = opened
    state.tree = _TreeAuthority(root=opened, entries=(), inventory=None)


def _seal_staged_file(
    handle: StagedFileHandle,
    *,
    executable: bool,
) -> None:
    if type(executable) is not bool:
        raise TypeError("executable must be exactly bool")
    state = _require_handle(
        handle,
        StagedFileHandle,
        operation="seal",
        allowed=(StagingState.OPEN,),
    )
    _require_population_authority(state)
    mode = 0o700 if executable else 0o600
    try:
        _fchmod_descriptor(state.staging_descriptor, mode)
    except BaseException:
        try:
            _reconcile_owned_file_mutation(
                state,
                minimum_size=state.size_bytes,
                maximum_size=state.size_bytes,
                allowed_modes=(0o600, 0o700),
            )
        except BaseException as reconciliation_error:
            state.lifecycle = StagingState.RETIRED
            error = PublicationValidationError(
                "failed seal left unverifiable file authority",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
            try:
                _retire_with_primary_error(state, error)
            except BaseException as transported:
                raise transported from reconciliation_error
        raise
    _reconcile_owned_file_mutation(
        state,
        minimum_size=state.size_bytes,
        maximum_size=state.size_bytes,
        allowed_modes=(mode,),
    )
    if stat.S_IMODE(state.root_identity.public.mode) != mode:
        raise PublicationValidationError(
            "sealed file mode is not exact",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    _complete_seal(state)


def _mkdir_staged_directory(
    handle: StagedDirectoryHandle,
    path: PortableRelativePath,
) -> None:
    if type(path) is not PortableRelativePath:
        raise TypeError("staged tree paths must be exactly PortableRelativePath")
    state = _require_handle(
        handle,
        StagedDirectoryHandle,
        operation="mkdir",
        allowed=(StagingState.OPEN,),
    )
    _require_new_tree_path(state, path, entry_type="directory")
    _require_population_authority(state)
    try:
        parent_descriptor = _open_tree_parent(state, path)
    except BaseException as exc:
        _raise_population_failure(
            state,
            exc,
            operation="mkdir",
            message="staged directory parent could not be opened safely",
        )
    parent_owned = _owned_descriptor(parent_descriptor, None)
    parent_close_inode: int | None = None
    child_descriptor = -1
    child_owned: _OwnedDescriptor | None = None
    child_close_identity: PublicationEntryIdentity | None = None
    created = False
    created_identity: _LedgerIdentity | None = None
    failure: BaseException | None = None
    try:
        parent_close_inode = _fstat_descriptor(parent_descriptor).st_ino
        _mkdir_at(path.parts[-1], parent_descriptor, 0o700)
        created = True
        child_descriptor = _open_directory_at(
            path.parts[-1],
            parent_descriptor,
            role="traversal_directory",
        )
        child_owned = _owned_descriptor(
            child_descriptor,
            "traversal_directory",
        )
        child_close_identity = _make_entry_identity(
            _fstat_descriptor(child_descriptor)
        )
        _fchmod_descriptor(child_descriptor, 0o700)
        result = _fstat_descriptor(child_descriptor)
        named = _stat_at(path.parts[-1], parent_descriptor)
        if (
            _make_ledger_identity(result) != _make_ledger_identity(named)
            or not stat.S_ISDIR(result.st_mode)
            or _validate_host_uint("device", result.st_dev)
            != state.parent_device
            or result.st_uid != os.geteuid()
            or stat.S_IMODE(result.st_mode) != 0o700
        ):
            raise PublicationValidationError(
                "created staged directory is unsafe or unstable",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        created_identity = _make_ledger_identity(result)
    except BaseException as exc:
        failure = exc
    finally:
        candidates = tuple(
            candidate
            for candidate in (child_owned, parent_owned)
            if candidate is not None
        )
        child_descriptor = -1
        parent_descriptor = -1
        failure = _compose_population_retirement(
            state,
            operation="mkdir",
            primary=failure,
            candidates=candidates,
        )
    if created and failure is None and created_identity is not None:
        try:
            _refresh_directory_tree_after_addition(
                state,
                added_path=path,
                added_identity=created_identity,
            )
        except BaseException as refresh_error:
            _raise_population_failure(
                state,
                refresh_error,
                operation="mkdir",
                message="failed mkdir left unverifiable staged-tree authority",
            )
    elif created:
        _raise_population_failure(
            state,
            (
                failure
                if failure is not None
                else RuntimeError("created directory lacked an admitted identity")
            ),
            operation="mkdir",
            message="failed mkdir left unadmitted staged-tree residue",
        )
    if failure is not None:
        if type(failure) is _DescriptorRetirementAnomaly:
            _raise_population_failure(
                state,
                failure,
                operation="mkdir",
                message="staged directory creation failed safely",
            )
        raise failure


def _write_staged_directory_file(
    handle: StagedDirectoryHandle,
    path: PortableRelativePath,
    chunks: Iterable[bytes],
    *,
    executable: bool,
) -> None:
    if type(path) is not PortableRelativePath:
        raise TypeError("staged tree paths must be exactly PortableRelativePath")
    if type(executable) is not bool:
        raise TypeError("executable must be exactly bool")
    state = _require_handle(
        handle,
        StagedDirectoryHandle,
        operation="write_file",
        allowed=(StagingState.OPEN,),
    )
    _require_new_tree_path(state, path, entry_type="regular_file")
    try:
        with _suspend_error_provenance():
            iterator = iter(chunks)
    except TypeError as exc:
        raise TypeError("chunks must be iterable") from exc
    _require_population_authority(state)

    try:
        parent_descriptor = _open_tree_parent(state, path)
    except BaseException as exc:
        _raise_population_failure(
            state,
            exc,
            operation="write_file",
            message="staged file parent could not be opened safely",
        )
    parent_owned = _owned_descriptor(parent_descriptor, None)
    parent_close_inode: int | None = None
    descriptor = -1
    descriptor_owned: _OwnedDescriptor | None = None
    descriptor_close_identity: PublicationEntryIdentity | None = None
    created = False
    created_identity: _LedgerIdentity | None = None
    size_bytes = 0
    mode = 0o700 if executable else 0o600
    failure: BaseException | None = None
    try:
        parent_close_inode = _fstat_descriptor(parent_descriptor).st_ino
        try:
            descriptor = _create_file_at(
                path.parts[-1],
                parent_descriptor,
                0o600,
                role="traversal_entry",
            )
        except _CreatedEntryRegistrationError:
            created = True
            raise
        created = True
        descriptor_owned = _owned_descriptor(
            descriptor,
            "traversal_entry",
        )
        descriptor_close_identity = _make_entry_identity(
            _fstat_descriptor(descriptor)
        )
        while True:
            try:
                with _suspend_error_provenance():
                    chunk = next(iterator)
            except StopIteration:
                break
            except BaseException as exc:
                raise _CallerChunkFailure(exc) from exc
            if type(chunk) is not bytes:
                raise TypeError("staged file chunks must be exactly bytes")
            if size_bytes > MAX_SIGNED_64 - len(chunk):
                raise ValueError("staged file exceeds the signed-64 size bound")
            _require_population_authority(state)
            _write_all(
                descriptor,
                chunk,
                destination=state.destination,
            )
            size_bytes += len(chunk)
        _require_population_authority(state)
        _fchmod_descriptor(descriptor, mode)
        opened = _fstat_descriptor(descriptor)
        named = _stat_at(path.parts[-1], parent_descriptor)
        if (
            _make_ledger_identity(opened) != _make_ledger_identity(named)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _validate_host_uint("device", opened.st_dev)
            != state.parent_device
            or opened.st_uid != os.geteuid()
            or opened.st_size != size_bytes
            or stat.S_IMODE(opened.st_mode) != mode
        ):
            raise PublicationValidationError(
                "created staged file is unsafe or unstable",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        created_identity = _make_ledger_identity(opened)
    except BaseException as exc:
        failure = exc
    finally:
        candidates = tuple(
            candidate
            for candidate in (descriptor_owned, parent_owned)
            if candidate is not None
        )
        descriptor = -1
        parent_descriptor = -1
        failure = _compose_population_retirement(
            state,
            operation="write_file",
            primary=failure,
            candidates=candidates,
        )
        if created and failure is None and created_identity is not None:
            try:
                _refresh_directory_tree_after_addition(
                    state,
                    added_path=path,
                    added_identity=created_identity,
                )
            except BaseException as refresh_error:
                _raise_population_failure(
                    state,
                    refresh_error,
                    operation="write_file",
                    message=(
                        "failed staged-file write left unverifiable tree "
                        "authority"
                    ),
                )
    if failure is not None:
        if created:
            _raise_population_failure(
                state,
                failure,
                operation="write_file",
                message=(
                    "failed staged-file population left preserved residue"
                ),
            )
        if type(failure) is _DescriptorRetirementAnomaly:
            _raise_population_failure(
                state,
                failure,
                operation="write_file",
                message="staged-file creation failed safely",
            )
        raise failure


def _seal_staged_directory(
    handle: StagedDirectoryHandle,
    *,
    scope: str,
) -> None:
    state = _require_handle(
        handle,
        StagedDirectoryHandle,
        operation="seal",
        allowed=(StagingState.OPEN,),
    )
    if type(scope) is not str:
        raise TypeError("scope must be exactly str")
    try:
        validated_scope = _validate_scope(scope)
    except InventoryValidationError as exc:
        raise PublicationValidationError(
            str(exc),
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        ) from exc
    _require_population_authority(state)
    try:
        tree = _scan_staged_tree(
            state,
            scope=validated_scope,
            hash_files=True,
            require_nonempty=True,
        )
    except _IncompleteStagedTreeError as exc:
        raise PublicationValidationError(
            str(exc),
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        ) from exc
    except BaseException as exc:
        _raise_population_failure(
            state,
            exc,
            operation="seal",
            message=(
                "staged directory could not be inventoried safely at seal"
            ),
        )
    if tree.root != state.tree.root or tree.entries != state.tree.entries:
        state.lifecycle = StagingState.RETIRED
        error = PublicationValidationError(
            "staged directory changed before sealing",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
        _retire_with_primary_error(state, error)
    state.root_identity = tree.root
    state.tree = tree
    _complete_seal(state)


def _require_new_tree_path(
    state: _StagingAuthorityState,
    path: object,
    *,
    entry_type: Literal["regular_file", "directory"],
) -> PortableRelativePath:
    if type(path) is not PortableRelativePath:
        raise TypeError("staged tree paths must be exactly PortableRelativePath")
    if len(state.tree.entries) + 1 > MAX_CANONICAL_CONTAINER_ITEMS:
        raise ValueError("staged tree exceeds the frozen entry bound")
    if entry_type == "regular_file":
        regular_file_count = sum(
            identity.public.entry_type == "regular_file"
            for _existing, identity in state.tree.entries
        )
        maximum_inventory_files = (
            MAX_CANONICAL_CONTAINER_ITEMS - 4
        ) // 5
        if regular_file_count + 1 > maximum_inventory_files:
            raise ValueError(
                "staged tree exceeds the H2b canonical inventory item bound"
            )
    total_path_bytes = sum(
        len(existing.value.encode("ascii"))
        for existing, _identity in state.tree.entries
    ) + len(path.value.encode("ascii"))
    if total_path_bytes > MAX_CANONICAL_DOCUMENT_BYTES:
        raise ValueError("staged tree exceeds the accumulated path-byte bound")

    entries = {existing.value: identity for existing, identity in state.tree.entries}
    parent_parts = path.parts[:-1]
    if parent_parts:
        parent_path = "/".join(parent_parts)
        parent = entries.get(parent_path)
        if parent is None or parent.public.entry_type != "directory":
            raise PublicationValidationError(
                "staged file parents must be created explicitly",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
    for existing, existing_identity in state.tree.entries:
        left = existing.parts
        right = path.parts
        for first, second in zip(left, right):
            if first.lower() == second.lower() and first != second:
                raise PublicationValidationError(
                    "staged tree contains a case-insensitive alias",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            if first.lower() != second.lower():
                break
        if existing.value == path.value:
            raise PublicationValidationError(
                "staged tree entry already exists",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        if len(left) < len(right) and right[: len(left)] == left:
            if existing_identity.public.entry_type != "directory":
                raise PublicationValidationError(
                    "a staged regular file cannot be a parent",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
        if len(right) < len(left) and left[: len(right)] == right:
            raise PublicationValidationError(
                "new staged entry collides with an existing descendant",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
    return path


def _open_tree_parent(
    state: _StagingAuthorityState,
    path: PortableRelativePath,
    *,
    root: _LedgerIdentity | None = None,
    entries: dict[PortableRelativePath, _LedgerIdentity] | None = None,
) -> int:
    expected_root = state.root_identity if root is None else root
    expected_entries = (
        dict(state.tree.entries) if entries is None else entries
    )
    descriptor = -1
    descriptor_owned: _OwnedDescriptor | None = None
    descriptor_identity = expected_root
    child = -1
    child_owned: _OwnedDescriptor | None = None
    child_identity = expected_root
    promoted = -1
    promoted_owned: _OwnedDescriptor | None = None
    promoted_identity = expected_root
    prefix: list[str] = []
    try:
        descriptor = _open_directory_at(
            ".",
            state.staging_descriptor,
            role="traversal_parent",
        )
        descriptor_owned = _owned_descriptor(
            descriptor,
            "traversal_parent",
        )
        opened_root = _make_ledger_identity(
            _fstat_descriptor(descriptor)
        )
        if opened_root != expected_root:
            raise PublicationValidationError(
                "staged root changed while opening a tree parent",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        descriptor_identity = opened_root
        for component in path.parts[:-1]:
            prefix.append(component)
            logical = PortableRelativePath.parse("/".join(prefix))
            expected = expected_entries.get(logical)
            if expected is None or expected.public.entry_type != "directory":
                raise PublicationValidationError(
                    "staged path parent was not explicitly created",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            before = _stat_at(component, descriptor)
            before_identity = _make_ledger_identity(before)
            child = _open_directory_at(
                component,
                descriptor,
                role="traversal_directory",
            )
            child_owned = _owned_descriptor(
                child,
                "traversal_directory",
            )
            child_identity = before_identity
            opened = _fstat_descriptor(child)
            child_identity = _make_ledger_identity(opened)
            after = _stat_at(component, descriptor)
            if (
                before_identity != expected
                or child_identity != expected
                or _make_ledger_identity(after) != expected
            ):
                raise PublicationValidationError(
                    "staged directory parent identity changed",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )

            prior_parent = descriptor_owned
            descriptor = -1
            descriptor_owned = None
            parent_retirement = _retire_owned_descriptors(
                (prior_parent,)
            )
            if parent_retirement is not None:
                raise _DescriptorRetirementAnomaly(parent_retirement)

            promoted = _open_directory_at(
                ".",
                child,
                role="traversal_parent",
            )
            promoted_owned = _owned_descriptor(
                promoted,
                "traversal_parent",
            )
            promoted_identity = _make_ledger_identity(
                _fstat_descriptor(promoted)
            )
            if promoted_identity != child_identity:
                raise PublicationValidationError(
                    "staged directory promotion changed identity",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )

            prior_child = child_owned
            child = -1
            child_owned = None
            child_retirement = _retire_owned_descriptors((prior_child,))
            if child_retirement is not None:
                raise _DescriptorRetirementAnomaly(child_retirement)

            descriptor = promoted
            descriptor_owned = promoted_owned
            descriptor_identity = promoted_identity
            promoted = -1
            promoted_owned = None
        return descriptor
    except BaseException as exc:
        prior_evidence = None
        primary: BaseException | None = exc
        if type(exc) is _DescriptorRetirementAnomaly:
            prior_evidence = exc.evidence
            primary = exc.primary
        candidates = tuple(
            candidate
            for candidate in (
                child_owned,
                promoted_owned,
                descriptor_owned,
            )
            if candidate is not None
        )
        child = promoted = descriptor = -1
        child_owned = promoted_owned = descriptor_owned = None
        _raise_after_local_retirement(
            primary,
            candidates,
            prior_evidence=prior_evidence,
        )
        raise AssertionError("unreachable")


def _refresh_directory_tree_after_addition(
    state: _StagingAuthorityState,
    *,
    added_path: PortableRelativePath,
    added_identity: _LedgerIdentity,
) -> None:
    before = state.tree
    tree = _scan_staged_tree(
        state,
        scope=None,
        hash_files=False,
        require_nonempty=False,
    )
    prior = dict(before.entries)
    observed = dict(tree.entries)
    expected_paths = {*prior, added_path}
    if set(observed) != expected_paths or added_path in prior:
        raise PublicationValidationError(
            "staged-tree population observed unexpected membership",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    if observed[added_path] != added_identity:
        raise PublicationValidationError(
            "new staged-tree entry changed before ledger admission",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    immediate_parent = added_path.parts[:-1]
    for path, expected in prior.items():
        current = observed[path]
        is_immediate_parent = path.parts == immediate_parent
        if (
            is_immediate_parent
            and expected.public.entry_type == "directory"
        ):
            stable = (
                _same_cleanup_directory_authority(current, expected)
                and current.public.mode == expected.public.mode
            )
        else:
            stable = current == expected
        if not stable:
            raise PublicationValidationError(
                "prior staged-tree authority changed during population",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
    if immediate_parent:
        root_stable = tree.root == before.root
    else:
        root_stable = (
            _same_cleanup_directory_authority(tree.root, before.root)
            and tree.root.public.mode == before.root.public.mode
        )
    if not root_stable:
        raise PublicationValidationError(
            "staged-tree root authority changed during population",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    state.root_identity = tree.root
    state.tree = tree


def _open_scanned_directory_path(
    state: _StagingAuthorityState,
    components: tuple[str, ...],
    *,
    root: _LedgerIdentity,
    entries: dict[PortableRelativePath, _LedgerIdentity],
) -> tuple[
    int,
    int,
    _LedgerIdentity,
    _LedgerIdentity | None,
    str | None,
]:
    """Open one admitted staged directory without recursive traversal."""

    descriptor = -1
    descriptor_owned: _OwnedDescriptor | None = None
    descriptor_identity = root
    child = -1
    child_owned: _OwnedDescriptor | None = None
    child_identity = root
    promoted = -1
    promoted_owned: _OwnedDescriptor | None = None
    promoted_identity = root
    try:
        descriptor = _open_directory_at(
            ".",
            state.staging_descriptor,
            role="traversal_parent",
        )
        descriptor_owned = _owned_descriptor(
            descriptor,
            "traversal_parent",
        )
        opened_root = _make_ledger_identity(
            _fstat_descriptor(descriptor)
        )
        if opened_root != root:
            raise PublicationValidationError(
                "staged root changed while reopening a directory",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        descriptor_identity = opened_root
        prefix: list[str] = []
        for component in components:
            prefix.append(component)
            path = PortableRelativePath.parse("/".join(prefix))
            expected = entries.get(path)
            if expected is None or expected.public.entry_type != "directory":
                raise PublicationValidationError(
                    "staged directory was not admitted before traversal",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            before = _make_ledger_identity(
                _stat_at(component, descriptor)
            )
            child = _open_directory_at(
                component,
                descriptor,
                role="traversal_directory",
            )
            child_owned = _owned_descriptor(
                child,
                "traversal_directory",
            )
            child_identity = before
            child_identity = _make_ledger_identity(
                _fstat_descriptor(child)
            )
            named = _make_ledger_identity(
                _stat_at(component, descriptor)
            )
            if (
                before != expected
                or child_identity != expected
                or named != expected
            ):
                raise PublicationValidationError(
                    "staged directory changed while it was reopened",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            prior_parent = descriptor_owned
            descriptor = -1
            descriptor_owned = None
            parent_retirement = _retire_owned_descriptors(
                (prior_parent,)
            )
            if parent_retirement is not None:
                raise _DescriptorRetirementAnomaly(parent_retirement)
            promoted = _open_directory_at(
                ".",
                child,
                role="traversal_parent",
            )
            promoted_owned = _owned_descriptor(
                promoted,
                "traversal_parent",
            )
            promoted_identity = _make_ledger_identity(
                _fstat_descriptor(promoted)
            )
            if promoted_identity != child_identity:
                raise PublicationValidationError(
                    "staged directory promotion changed identity",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            prior_child = child_owned
            child = -1
            child_owned = None
            child_retirement = _retire_owned_descriptors((prior_child,))
            if child_retirement is not None:
                raise _DescriptorRetirementAnomaly(child_retirement)
            descriptor = promoted
            descriptor_owned = promoted_owned
            descriptor_identity = promoted_identity
            promoted = -1
            promoted_owned = None

        result = (descriptor, -1, descriptor_identity, None, None)
        descriptor = -1
        descriptor_owned = None
        return result
    except BaseException as exc:
        prior_evidence = None
        primary: BaseException | None = exc
        if type(exc) is _DescriptorRetirementAnomaly:
            prior_evidence = exc.evidence
            primary = exc.primary
        candidates = tuple(
            candidate
            for candidate in (
                child_owned,
                promoted_owned,
                descriptor_owned,
            )
            if candidate is not None
        )
        child = promoted = descriptor = -1
        child_owned = promoted_owned = descriptor_owned = None
        _raise_after_local_retirement(
            primary,
            candidates,
            prior_evidence=prior_evidence,
        )
        raise AssertionError("unreachable")


def _scan_staged_tree(
    state: _StagingAuthorityState,
    *,
    scope: str | None,
    hash_files: bool,
    require_nonempty: bool,
) -> _TreeAuthority:
    root_before = _make_ledger_identity(
        _fstat_descriptor(state.staging_descriptor)
    )
    root_named = _make_ledger_identity(
        _stat_at(state.staging.value, state.parent_descriptor)
    )
    if (
        root_before != root_named
        or root_before.public.entry_type != "directory"
        or root_before.public.device != state.parent_device
        or root_before.public.owner_uid != os.geteuid()
        or stat.S_IMODE(root_before.public.mode) != 0o700
    ):
        raise PublicationValidationError(
            "staged directory root is unsafe or unstable",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )

    entries: dict[PortableRelativePath, _LedgerIdentity] = {}
    records: list[RegularFileRecord] = []
    observed = 0
    path_bytes = 0

    queue: deque[tuple[str, ...]] = deque([()])
    while queue:
        components = queue.popleft()
        (
            descriptor,
            parent_descriptor,
            expected_directory,
            parent_identity,
            leaf,
        ) = _open_scanned_directory_path(
            state,
            components,
            root=root_before,
            entries=entries,
        )
        descriptor_owned = _owned_descriptor(descriptor, None)
        parent_owned = (
            _owned_descriptor(parent_descriptor, None)
            if parent_descriptor >= 0
            else None
        )
        active: BaseException | None = None
        try:
            before_directory = _make_ledger_identity(
                _fstat_descriptor(descriptor)
            )
            if before_directory != expected_directory:
                raise PublicationValidationError(
                    "staged directory changed before enumeration",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            raw_names = _list_names(descriptor)
            names: list[str] = []
            spellings: dict[str, str] = {}
            try:
                for name in raw_names:
                    observed += 1
                    if observed > MAX_CANONICAL_CONTAINER_ITEMS:
                        raise PublicationValidationError(
                            "staged tree exceeds the frozen entry bound",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=state.destination,
                        )
                    if type(name) is not str or "/" in name:
                        raise PublicationValidationError(
                            "staged tree enumeration returned an unsafe name",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=state.destination,
                        )
                    try:
                        path = PortableRelativePath.parse(
                            "/".join((*components, name))
                        )
                    except PortablePathError as exc:
                        raise PublicationValidationError(
                            "staged tree contains an unsafe portable path",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=state.destination,
                        ) from exc
                    path_bytes += len(path.value.encode("ascii"))
                    if path_bytes > MAX_CANONICAL_DOCUMENT_BYTES:
                        raise PublicationValidationError(
                            "staged tree exceeds the path-byte bound",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=state.destination,
                        )
                    alias = name.lower()
                    if alias in spellings:
                        raise PublicationValidationError(
                            "staged tree contains an exact or case alias",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=state.destination,
                        )
                    spellings[alias] = name
                    names.append(name)
            finally:
                close = getattr(raw_names, "close", None)
                if close is not None:
                    close()

            for name in sorted(names, key=lambda item: item.encode("ascii")):
                path = PortableRelativePath.parse(
                    "/".join((*components, name))
                )
                first = _stat_at(name, descriptor)
                first_ledger = _make_ledger_identity(first)
                if (
                    first_ledger.public.device != state.parent_device
                    or first.st_uid != os.geteuid()
                ):
                    raise PublicationValidationError(
                        "staged entry crosses device or ownership boundary",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=state.destination,
                    )
                if stat.S_ISDIR(first.st_mode):
                    if stat.S_IMODE(first.st_mode) != 0o700:
                        raise PublicationValidationError(
                            "staged directory mode is not exact",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=state.destination,
                        )
                    child = _open_directory_at(
                        name,
                        descriptor,
                        role="traversal_directory",
                    )
                    child_owned = _owned_descriptor(
                        child,
                        "traversal_directory",
                        expected=first_ledger.public,
                    )
                    child_active: BaseException | None = None
                    try:
                        opened = _make_ledger_identity(
                            _fstat_descriptor(child)
                        )
                        named = _make_ledger_identity(
                            _stat_at(name, descriptor)
                        )
                        if opened != first_ledger or named != first_ledger:
                            raise PublicationValidationError(
                                "staged directory changed while opened",
                                state=PublicationState.NOT_COMMITTED,
                                evidence=None,
                                destination=state.destination,
                            )
                        entries[path] = first_ledger
                        queue.append((*components, name))
                    except BaseException as exc:
                        child_active = exc
                    finally:
                        child = -1
                        _raise_after_local_retirement(
                            child_active,
                            (child_owned,),
                        )
                elif stat.S_ISREG(first.st_mode):
                    if (
                        first.st_nlink != 1
                        or stat.S_IMODE(first.st_mode) not in {0o600, 0o700}
                    ):
                        raise PublicationValidationError(
                            "staged regular file metadata is unsafe",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=state.destination,
                        )
                    entries[path] = first_ledger
                    if hash_files:
                        records.append(
                            _hash_staged_file(
                                state,
                                descriptor,
                                name,
                                path,
                                first_ledger,
                            )
                        )
                    else:
                        second = _make_ledger_identity(
                            _stat_at(name, descriptor)
                        )
                        if second != first_ledger:
                            raise PublicationValidationError(
                                "staged regular file changed during inspection",
                                state=PublicationState.NOT_COMMITTED,
                                evidence=None,
                                destination=state.destination,
                            )
                else:
                    raise PublicationValidationError(
                        "staged tree contains a symlink or special entry",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=state.destination,
                    )

            after_directory = _make_ledger_identity(
                _fstat_descriptor(descriptor)
            )
            if after_directory != before_directory:
                raise PublicationValidationError(
                    "staged directory metadata changed during enumeration",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
        except BaseException as exc:
            active = exc
        finally:
            candidates = tuple(
                candidate
                for candidate in (descriptor_owned, parent_owned)
                if candidate is not None
            )
            descriptor = parent_descriptor = -1
            descriptor_owned = parent_owned = None
            _raise_after_local_retirement(active, candidates)
        if components:
            logical_directory = PortableRelativePath.parse(
                "/".join(components)
            )
            verification_parent = _open_tree_parent(
                state,
                logical_directory,
                root=root_before,
                entries=entries,
            )
            verification_parent_owned = _owned_descriptor(
                verification_parent,
                None,
            )
            active = None
            try:
                final_named = _make_ledger_identity(
                    _stat_at(components[-1], verification_parent)
                )
                if final_named != expected_directory:
                    raise PublicationValidationError(
                        "staged directory changed during traversal",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=state.destination,
                    )
            except BaseException as exc:
                active = exc
            finally:
                verification_parent = -1
                _raise_after_local_retirement(
                    active,
                    (verification_parent_owned,),
                )

    root_after = _make_ledger_identity(
        _fstat_descriptor(state.staging_descriptor)
    )
    root_after_named = _make_ledger_identity(
        _stat_at(state.staging.value, state.parent_descriptor)
    )
    if root_after != root_before or root_after_named != root_before:
        raise PublicationValidationError(
            "staged root changed during traversal",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )

    ordered_entries = tuple(
        sorted(entries.items(), key=lambda item: portable_path_sort_key(item[0]))
    )
    if require_nonempty:
        file_paths = {
            path.value
            for path, identity in ordered_entries
            if identity.public.entry_type == "regular_file"
        }
        if not file_paths:
            raise _IncompleteStagedTreeError(
                "staged directory contains no regular files",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        for path, identity in ordered_entries:
            if identity.public.entry_type != "directory":
                continue
            prefix = f"{path.value}/"
            if not any(file_path.startswith(prefix) for file_path in file_paths):
                raise _IncompleteStagedTreeError(
                    "staged directory contains an empty logical directory",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )

    inventory: RegularFileInventory | None = None
    if hash_files:
        if scope is None:
            raise ValueError("hashed staged tree requires a scope")
        ordered_records = tuple(
            sorted(records, key=lambda item: portable_path_sort_key(item.path))
        )
        try:
            require_distinct_file_paths(
                tuple(record.path for record in ordered_records)
            )
            inventory = _build_regular_file_inventory(
                scope=scope,
                excluded_prefixes=(),
                files=ordered_records,
            )
        except (InventorySafetyError, PortablePathError, TypeError, ValueError) as exc:
            raise PublicationValidationError(
                "staged tree cannot form an H2b inventory",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            ) from exc
    return _TreeAuthority(
        root=root_before,
        entries=ordered_entries,
        inventory=inventory,
    )


def _hash_staged_file(
    state: _StagingAuthorityState,
    parent_descriptor: int,
    name: str,
    path: PortableRelativePath,
    expected: _LedgerIdentity,
) -> RegularFileRecord:
    descriptor = _open_read_file_at(
        name,
        parent_descriptor,
        role="traversal_entry",
    )
    descriptor_owned = _owned_descriptor(
        descriptor,
        "traversal_entry",
        expected=expected.public,
    )
    active: BaseException | None = None
    try:
        opened = _make_ledger_identity(_fstat_descriptor(descriptor))
        if opened != expected:
            raise PublicationValidationError(
                "staged file changed while opened for hashing",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        digest = hashlib.sha256()
        remaining = expected.public.size_bytes
        while remaining:
            requested = min(_HASH_CHUNK_BYTES, remaining)
            chunk = _read_descriptor(
                descriptor,
                requested,
            )
            if not chunk:
                raise PublicationValidationError(
                    "staged file ended during hashing",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            if len(chunk) > requested:
                raise PublicationValidationError(
                    "staged file read exceeded the requested bound",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if _read_descriptor(descriptor, 1):
            raise PublicationValidationError(
                "staged file grew during hashing",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        if (
            _make_ledger_identity(_fstat_descriptor(descriptor)) != expected
            or _make_ledger_identity(_stat_at(name, parent_descriptor))
            != expected
        ):
            raise PublicationValidationError(
                "staged file changed during hashing",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        from .canonical import Sha256Digest

        return RegularFileRecord(
            path=path,
            size_bytes=expected.public.size_bytes,
            sha256=Sha256Digest(digest.hexdigest()),
            executable=bool(expected.public.mode & 0o111),
        )
    except BaseException as exc:
        active = exc
    finally:
        descriptor = -1
        _raise_after_local_retirement(active, (descriptor_owned,))


def _collect_namespace_evidence(
    parent_descriptor: int,
    reference: PortableRelativePath,
) -> NamespaceEvidence:
    reference_key = reference.value.lower()
    aliases: dict[str, PortableRelativePath] = {}
    observed_entries = 0
    try:
        names = _list_names(parent_descriptor)
        try:
            for name in names:
                observed_entries += 1
                if type(name) is not str:
                    return _make_namespace_evidence(
                        _NamespaceObservation.UNINSPECTABLE,
                        reference=reference,
                        aliases=tuple(
                            sorted(
                                aliases.values(),
                                key=portable_path_sort_key,
                            )
                        ),
                    )
                if name == reference.value or name.lower() != reference_key:
                    if observed_entries > _MAX_EVIDENCE_ALIASES:
                        observation = (
                            _NamespaceObservation.BOUNDED_CONFLICT
                            if len(aliases) == _MAX_EVIDENCE_ALIASES
                            else _NamespaceObservation.UNINSPECTABLE
                        )
                        return _make_namespace_evidence(
                            observation,
                            reference=reference,
                            aliases=tuple(
                                sorted(
                                    aliases.values(),
                                    key=portable_path_sort_key,
                                )
                            ),
                        )
                    continue
                try:
                    candidate = PortableRelativePath.parse(name)
                except PortablePathError:
                    return _make_namespace_evidence(
                        _NamespaceObservation.UNINSPECTABLE,
                        reference=reference,
                        aliases=tuple(
                            sorted(
                                aliases.values(),
                                key=portable_path_sort_key,
                            )
                        ),
                    )
                if len(candidate.parts) != 1 or name in aliases:
                    return _make_namespace_evidence(
                        _NamespaceObservation.UNINSPECTABLE,
                        reference=reference,
                        aliases=tuple(
                            sorted(
                                aliases.values(),
                                key=portable_path_sort_key,
                            )
                        ),
                    )
                if len(aliases) == _MAX_EVIDENCE_ALIASES:
                    ordered = tuple(
                        sorted(
                            aliases.values(),
                            key=portable_path_sort_key,
                        )
                    )
                    return _make_namespace_evidence(
                        _NamespaceObservation.BOUNDED_CONFLICT,
                        reference=reference,
                        aliases=ordered,
                    )
                aliases[name] = candidate
                if observed_entries > _MAX_EVIDENCE_ALIASES:
                    return _make_namespace_evidence(
                        _NamespaceObservation.UNINSPECTABLE,
                        reference=reference,
                        aliases=tuple(
                            sorted(
                                aliases.values(),
                                key=portable_path_sort_key,
                            )
                        ),
                    )
        finally:
            close = getattr(names, "close", None)
            if close is not None:
                close()
    except OSError:
        return _make_namespace_evidence(
            _NamespaceObservation.UNINSPECTABLE,
            reference=reference,
            aliases=tuple(
                sorted(aliases.values(), key=portable_path_sort_key)
            ),
        )

    ordered = tuple(
        sorted(aliases.values(), key=portable_path_sort_key)
    )
    if ordered:
        return _make_namespace_evidence(
            _NamespaceObservation.COMPLETE_CONFLICT,
            reference=reference,
            aliases=ordered,
        )
    return _make_namespace_evidence(
        _NamespaceObservation.NO_CONFLICT,
        reference=reference,
    )


def _observe_named_entry(
    parent_descriptor: int,
    name: str,
    original: PublicationEntryIdentity,
) -> tuple[str, PublicationEntryIdentity | None]:
    first = _observe_named_entry_once(parent_descriptor, name, original)
    second = _observe_named_entry_once(parent_descriptor, name, original)
    return _merge_postcall_observations(first, second, original)


def _safe_observe_named_entry(
    parent_descriptor: int,
    name: str,
    original: PublicationEntryIdentity,
) -> tuple[str, PublicationEntryIdentity | None]:
    try:
        return _observe_named_entry(parent_descriptor, name, original)
    except BaseException:
        return "uninspectable", None


def _safe_observe_named_entry_once(
    parent_descriptor: int,
    name: str,
    original: PublicationEntryIdentity,
) -> tuple[str, PublicationEntryIdentity | None]:
    try:
        return _observe_named_entry_once(
            parent_descriptor,
            name,
            original,
        )
    except BaseException:
        return "uninspectable", None


def _safe_namespace_evidence(
    parent_descriptor: int,
    reference: PortableRelativePath,
) -> NamespaceEvidence:
    try:
        return _collect_namespace_evidence(parent_descriptor, reference)
    except BaseException:
        return _make_namespace_evidence(
            _NamespaceObservation.UNINSPECTABLE,
            reference=reference,
        )


def _observe_named_entry_once(
    parent_descriptor: int,
    name: str,
    original: PublicationEntryIdentity,
) -> tuple[str, PublicationEntryIdentity | None]:
    listed_before = _exact_name_is_listed(parent_descriptor, name)
    if listed_before is None:
        return "uninspectable", None
    if not listed_before:
        return "absent", None
    try:
        observed = _make_entry_identity(_stat_at(name, parent_descriptor))
    except FileNotFoundError:
        return "uninspectable", None
    except (OSError, ValueError):
        return "uninspectable", None
    listed_after = _exact_name_is_listed(parent_descriptor, name)
    if listed_after is not True:
        return "uninspectable", None
    if observed == original:
        return "exact", original
    return "foreign", observed


def _exact_name_is_listed(
    parent_descriptor: int,
    name: str,
) -> bool | None:
    observed = 0
    try:
        names = _list_names(parent_descriptor)
        try:
            for candidate in names:
                observed += 1
                if observed > MAX_CANONICAL_CONTAINER_ITEMS:
                    return None
                if type(candidate) is not str:
                    return None
                if candidate == name:
                    return True
        finally:
            close = getattr(names, "close", None)
            if close is not None:
                close()
    except (OSError, ValueError):
        return None
    return False


def _merge_postcall_observations(
    first: tuple[str, PublicationEntryIdentity | None],
    second: tuple[str, PublicationEntryIdentity | None],
    original: PublicationEntryIdentity,
) -> tuple[str, PublicationEntryIdentity | None]:
    if first == second:
        return first
    if first[0] == "uninspectable" or second[0] == "uninspectable":
        return "uninspectable", None
    if first[0] == "exact" and second[0] == "exact":
        return "exact", original
    observed = second[1] if second[1] not in {None, original} else first[1]
    if observed in {None, original}:
        return "contradictory", None
    return "replaced", observed


def _observe_parent_fsync(
    parent_descriptor: int,
    *,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> str:
    try:
        before = _fstat_descriptor(parent_descriptor)
    except OSError:
        return "uncertain"
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) & 0o700 != 0o700
        or stat.S_IMODE(before.st_mode) & 0o022
        or (
            expected_device is not None
            and _validate_host_uint("device", before.st_dev)
            != expected_device
        )
        or (
            expected_inode is not None
            and _validate_host_uint("inode", before.st_ino)
            != expected_inode
        )
    ):
        return "uncertain"
    try:
        _fsync_descriptor(parent_descriptor)
    except OSError:
        result = "failed"
    else:
        result = "succeeded"
    try:
        after = _fstat_descriptor(parent_descriptor)
    except OSError:
        return "uncertain"
    if (
        not stat.S_ISDIR(after.st_mode)
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_uid != before.st_uid
        or stat.S_IMODE(after.st_mode) != stat.S_IMODE(before.st_mode)
    ):
        return "uncertain"
    return result


def _safe_parent_fsync(
    parent_descriptor: int,
    *,
    expected_device: int,
    expected_inode: int,
) -> str:
    try:
        return _observe_parent_fsync(
            parent_descriptor,
            expected_device=expected_device,
            expected_inode=expected_inode,
        )
    except BaseException:
        return "uncertain"


def _complete_seal(state: _StagingAuthorityState) -> None:
    destination_observation, destination_identity = _observe_named_entry(
        state.parent_descriptor,
        state.destination.value,
        state.root_identity.public,
    )
    namespace = _collect_namespace_evidence(
        state.parent_descriptor,
        state.destination,
    )
    known_alias = namespace.namespace_observation in {
        "complete_conflict",
        "bounded_conflict",
    }
    stable_collision = destination_observation == "foreign" or known_alias
    if stable_collision:
        parent_fsync = _safe_parent_fsync(
            state.parent_descriptor,
            expected_device=state.parent_device,
            expected_inode=state.parent_inode,
        )
        evidence = _make_publication_evidence(
            staging_identity=state.root_identity.public,
            source_observation="exact",
            observed_source_identity=state.root_identity.public,
            destination_observation=destination_observation,
            observed_destination_identity=destination_identity,
            namespace_evidence=namespace,
            parent_fsync=parent_fsync,
            native_errno=None,
        )
        state.lifecycle = StagingState.NOT_COMMITTED
        raise PublicationCollisionError(
            "publication destination or case alias already exists",
            state=PublicationState.NOT_COMMITTED,
            evidence=evidence,
            destination=state.destination,
        )
    if (
        destination_observation != "absent"
        or namespace.namespace_observation == "uninspectable"
    ):
        parent_fsync = _safe_parent_fsync(
            state.parent_descriptor,
            expected_device=state.parent_device,
            expected_inode=state.parent_inode,
        )
        evidence = _make_publication_evidence(
            staging_identity=state.root_identity.public,
            source_observation="exact",
            observed_source_identity=state.root_identity.public,
            destination_observation=destination_observation,
            observed_destination_identity=destination_identity,
            namespace_evidence=namespace,
            parent_fsync=parent_fsync,
            native_errno=None,
        )
        state.lifecycle = StagingState.RETIRED
        error = PublicationValidationError(
            "publication namespace could not be inspected at seal",
            state=PublicationState.NOT_COMMITTED,
            evidence=evidence,
            destination=state.destination,
        )
        _retire_with_primary_error(state, error)
    state.lifecycle = StagingState.SEALED


def publish_completed_file(
    trusted_root: TrustedRoot,
    staging: StagedFileHandle,
) -> PublicationResult:
    """Publish one sealed staged file through native no-replace rename."""

    if type(trusted_root) is not TrustedRoot:
        raise TypeError("trusted_root must be exactly TrustedRoot")
    with _locked_handle_state(
        staging,
        StagedFileHandle,
        operation="publish",
        terminal_authority=(
            trusted_root,
            (StagingState.SEALED,),
        ),
    ) as state:
        if state.kind != "regular_file":
            raise TypeError("file publication requires a staged-file authority")
        return _publish_completed(state)


def publish_completed_directory(
    trusted_root: TrustedRoot,
    staging: StagedDirectoryHandle,
) -> PublicationResult:
    """Publish one sealed staged directory through native no-replace rename."""

    if type(trusted_root) is not TrustedRoot:
        raise TypeError("trusted_root must be exactly TrustedRoot")
    with _locked_handle_state(
        staging,
        StagedDirectoryHandle,
        operation="publish",
        terminal_authority=(
            trusted_root,
            (StagingState.SEALED,),
        ),
    ) as state:
        if state.kind != "directory":
            raise TypeError(
                "directory publication requires a staged-directory authority"
            )
        return _publish_completed(state)


def _publish_completed(state: _StagingAuthorityState) -> PublicationResult:
    if state.publication_attempted:
        raise StagingLifecycleError(state.lifecycle, "publish")
    parent_descriptor = -1
    operation_parent: _OwnedDescriptor | None = None
    escaping_error: BaseException | None = None
    try:
        try:
            parent_descriptor = _open_publication_parent(
                state.trusted_root,
                role="operation_parent",
            )
            operation_parent = _owned_descriptor(
                parent_descriptor,
                "operation_parent",
            )
            parent = _require_parent_security(
                state.trusted_root,
                parent_descriptor,
            )
        except BaseException as exc:
            authority_error = StagingAuthorityError(
                state.lifecycle,
                "publish",
            )
            admission_retirement = _internal_admission_retirement_evidence(
                exc
            )
            if admission_retirement is not None:
                _attach_retirement_evidence(
                    authority_error,
                    state.error_provenance,
                    admission_retirement,
                )
            raise authority_error from exc
        if (
            parent.public.device != state.parent_device
            or parent.public.inode != state.parent_inode
        ):
            raise StagingAuthorityError(state.lifecycle, "publish")
        state.publication_attempted = True

        try:
            if state.kind == "regular_file":
                _flush_staged_file(state, parent_descriptor)
            else:
                _flush_staged_directory(state, parent_descriptor)
            _require_prepublication_namespace(state, parent_descriptor)
        except PublicationCollisionError:
            raise
        except BaseException as exc:
            _raise_precommit_failure(state, parent_descriptor, exc)

        try:
            _native_result, native_errno = _native_no_replace(
                state.backend,
                parent_descriptor,
                state.staging.value,
                state.destination.value,
            )
        except BaseException as native_failure:
            native_errno = _errno_from_exception(native_failure)

        # Once the native call has been attempted its return path alone cannot
        # establish whether the namespace mutation committed.  Flush the
        # pinned parent immediately, then gather bounded outcome evidence.
        try:
            parent_fsync = _safe_parent_fsync(
                parent_descriptor,
                expected_device=state.parent_device,
                expected_inode=state.parent_inode,
            )
        except BaseException:
            parent_fsync = "uncertain"

        try:
            parent_is_exact = _parent_descriptor_matches(
                state,
                parent_descriptor,
            )
            if not parent_is_exact:
                raise RuntimeError("publication parent is uninspectable")
            source_first = _safe_observe_named_entry_once(
                parent_descriptor,
                state.staging.value,
                state.root_identity.public,
            )
            destination_first = _safe_observe_named_entry_once(
                parent_descriptor,
                state.destination.value,
                state.root_identity.public,
            )
            source_second = _safe_observe_named_entry_once(
                parent_descriptor,
                state.staging.value,
                state.root_identity.public,
            )
            destination_second = _safe_observe_named_entry_once(
                parent_descriptor,
                state.destination.value,
                state.root_identity.public,
            )
        except BaseException:
            source_first = source_second = ("uninspectable", None)
            destination_first = destination_second = ("uninspectable", None)
        source = _merge_postcall_observations(
            source_first,
            source_second,
            state.root_identity.public,
        )
        destination = _merge_postcall_observations(
            destination_first,
            destination_second,
            state.root_identity.public,
        )
        destination_proved_commit = (
            destination_first[0] == "exact"
            or destination_second[0] == "exact"
        )
        if destination_proved_commit and source[0] == "exact":
            source = ("contradictory", None)

        try:
            if not _parent_descriptor_matches(state, parent_descriptor):
                raise RuntimeError("publication parent is uninspectable")
            namespace = _safe_namespace_evidence(
                parent_descriptor,
                state.destination,
            )
        except BaseException:
            namespace = _make_namespace_evidence(
                _NamespaceObservation.UNINSPECTABLE,
                reference=state.destination,
            )
        if destination_proved_commit:
            state.lifecycle = StagingState.PUBLISHED
        elif source[0] == "exact" and destination[0] in {
            "absent",
            "foreign",
        }:
            state.lifecycle = StagingState.NOT_COMMITTED
        else:
            state.lifecycle = StagingState.RETIRED
        committed_state = (
            PublicationState.COMMITTED_DURABLE
            if parent_fsync == "succeeded"
            else PublicationState.COMMITTED_DURABILITY_UNCERTAIN
        )
        try:
            evidence = _make_publication_evidence(
                staging_identity=state.root_identity.public,
                source_observation=source[0],
                observed_source_identity=source[1],
                destination_observation=destination[0],
                observed_destination_identity=destination[1],
                namespace_evidence=namespace,
                parent_fsync=parent_fsync,
                native_errno=native_errno,
            )
        except BaseException as evidence_error:
            fallback_namespace = namespace
            fallback_error_type: type[PublicationError]
            if destination_proved_commit:
                conflict = namespace.namespace_observation in {
                    "complete_conflict",
                    "bounded_conflict",
                }
                source_or_destination_anomaly = (
                    source[0] not in {"absent", "foreign"}
                    or destination[0] != "exact"
                )
                if conflict:
                    fallback_error_type = (
                        PublicationNamespaceConflictError
                    )
                elif (
                    namespace.namespace_observation == "uninspectable"
                    or source_or_destination_anomaly
                ):
                    fallback_error_type = (
                        PublicationNamespaceUncertainError
                    )
                elif parent_fsync != "succeeded":
                    fallback_error_type = PublicationDurabilityError
                else:
                    fallback_error_type = (
                        PublicationNamespaceUncertainError
                    )
                    fallback_namespace = _make_namespace_evidence(
                        _NamespaceObservation.UNINSPECTABLE,
                        reference=state.destination,
                    )
            elif source[0] == "exact" and destination[0] == "absent":
                fallback_error_type = (
                    PublicationCapabilityError
                    if native_errno
                    in {
                        errno.ENOSYS,
                        errno.ENOTSUP,
                        getattr(
                            errno,
                            "EOPNOTSUPP",
                            errno.ENOTSUP,
                        ),
                        errno.EXDEV,
                        errno.EINVAL,
                    }
                    else PublicationError
                )
            elif source[0] == "exact" and destination[0] == "foreign":
                fallback_error_type = PublicationCollisionError
            else:
                fallback_error_type = PublicationOutcomeUncertainError
            fallback_evidence = _allocate_publication_evidence(
                staging_identity=state.root_identity.public,
                source_observation=source[0],
                observed_source_identity=source[1],
                destination_observation=destination[0],
                observed_destination_identity=destination[1],
                namespace_evidence=fallback_namespace,
                parent_fsync=parent_fsync,
                native_errno=native_errno,
            )
            if destination_proved_commit:
                raise fallback_error_type(
                    "publication committed but outcome evidence could not "
                    "be constructed through the strict validator",
                    state=committed_state,
                    evidence=fallback_evidence,
                    destination=state.destination,
                ) from evidence_error
            if source[0] == "exact" and destination[0] == "absent":
                raise fallback_error_type(
                    "native no-replace publication did not commit and "
                    "strict outcome evidence construction failed",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=fallback_evidence,
                    destination=state.destination,
                ) from evidence_error
            if source[0] == "exact" and destination[0] == "foreign":
                raise fallback_error_type(
                    "native no-replace publication preserved a collision "
                    "while strict outcome evidence construction failed",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=fallback_evidence,
                    destination=state.destination,
                ) from evidence_error
            raise fallback_error_type(
                "native publication outcome and strict evidence "
                "construction are uncertain",
                state=PublicationState.COMMIT_OUTCOME_UNCERTAIN,
                evidence=fallback_evidence,
                destination=state.destination,
            ) from evidence_error

        if destination_proved_commit:
            conflict = namespace.namespace_observation in {
                "complete_conflict",
                "bounded_conflict",
            }
            source_or_destination_anomaly = (
                source[0] not in {"absent", "foreign"}
                or destination[0] != "exact"
            )
            if conflict:
                raise PublicationNamespaceConflictError(
                    "publication committed with a conflicting case alias",
                    state=committed_state,
                    evidence=evidence,
                    destination=state.destination,
                )
            if (
                namespace.namespace_observation == "uninspectable"
                or source_or_destination_anomaly
            ):
                raise PublicationNamespaceUncertainError(
                    "publication committed but namespace verification is "
                    "uncertain",
                    state=committed_state,
                    evidence=evidence,
                    destination=state.destination,
                )
            if parent_fsync != "succeeded":
                raise PublicationDurabilityError(
                    "publication committed but parent durability is uncertain",
                    state=PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
                    evidence=evidence,
                    destination=state.destination,
                )
            try:
                result = _make_publication_result(
                    PublicationState.COMMITTED_DURABLE,
                    state.destination,
                    state.root_identity.public,
                    namespace,
                )
            except BaseException as result_error:
                fallback_result = _allocate_publication_result(
                    PublicationState.COMMITTED_DURABLE,
                    state.destination,
                    state.root_identity.public,
                    namespace,
                )
                additional = (
                    (operation_parent,)
                    if operation_parent is not None
                    else ()
                )
                parent_descriptor = -1
                _raise_after_terminal_result_failure(
                    state,
                    operation="publish",
                    primary=result_error,
                    terminal_result=fallback_result,
                    additional=additional,
                )
            parent_descriptor = -1
            return _retire_with_terminal_result(
                state,
                "publish",
                result,
                additional=(
                    (operation_parent,)
                    if operation_parent is not None
                    else ()
                ),
            )  # type: ignore[return-value]

        if source[0] == "exact" and destination[0] == "absent":
            error_type: type[PublicationError]
            if native_errno in {
                errno.ENOSYS,
                errno.ENOTSUP,
                getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
                errno.EXDEV,
                errno.EINVAL,
            }:
                error_type = PublicationCapabilityError
            else:
                error_type = PublicationError
            raise error_type(
                "native no-replace publication did not commit",
                state=PublicationState.NOT_COMMITTED,
                evidence=evidence,
                destination=state.destination,
            )

        if source[0] == "exact" and destination[0] == "foreign":
            raise PublicationCollisionError(
                "native no-replace publication preserved a collision",
                state=PublicationState.NOT_COMMITTED,
                evidence=evidence,
                destination=state.destination,
            )

        raise PublicationOutcomeUncertainError(
            "native publication outcome cannot be reconciled safely",
            state=PublicationState.COMMIT_OUTCOME_UNCERTAIN,
            evidence=evidence,
            destination=state.destination,
        )
    except PublicationError as error:
        escaping_error = error
        if state.publication_attempted and state.lifecycle in {
            StagingState.NOT_COMMITTED,
            StagingState.PUBLISHED,
            StagingState.RETIRED,
        }:
            parent_descriptor = -1
            post_outcome_retirement = (
                getattr(
                    error,
                    "_pending_retirement_evidence",
                    None,
                )
                is None
            )
            _retire_with_primary_error(
                state,
                error,
                additional=(
                    (operation_parent,)
                    if operation_parent is not None
                    else ()
                ),
                post_outcome=post_outcome_retirement,
            )
        raise
    except BaseException as error:
        escaping_error = error
        if state.publication_attempted and state.lifecycle in {
            StagingState.NOT_COMMITTED,
            StagingState.PUBLISHED,
            StagingState.RETIRED,
        }:
            additional = (
                (operation_parent,)
                if (
                    parent_descriptor >= 0
                    and operation_parent is not None
                )
                else ()
            )
            parent_descriptor = -1
            retirement = _retirement_evidence_from_records(
                _retire_state_descriptor_records(
                    state,
                    additional=additional,
                ),
                post_outcome=True,
            )
            if retirement is not None:
                raise _DescriptorRetirementAnomaly(
                    retirement,
                    primary=error,
                ) from error
        raise
    finally:
        if parent_descriptor >= 0:
            candidate = (
                operation_parent
                if operation_parent is not None
                else _owned_descriptor(
                    parent_descriptor,
                    "operation_parent",
                )
            )
            parent_descriptor = -1
            retirement = _retire_owned_descriptors((candidate,))
            if retirement is not None:
                if (
                    escaping_error is not None
                    and _attach_retirement_evidence(
                        escaping_error,
                        state.error_provenance,
                        retirement,
                    )
                ):
                    pass
                elif escaping_error is not None:
                    authority_error = StagingAuthorityError(
                        state.lifecycle,
                        "publish",
                    )
                    _attach_retirement_evidence(
                        authority_error,
                        state.error_provenance,
                        retirement,
                    )
                    raise BaseExceptionGroup(
                        "publication and operation-parent retirement both "
                        "failed",
                        [escaping_error, authority_error],
                    ) from None
                else:
                    authority_error = StagingAuthorityError(
                        state.lifecycle,
                        "publish",
                    )
                    _attach_retirement_evidence(
                        authority_error,
                        state.error_provenance,
                        retirement,
                    )
                    raise authority_error


def _require_prepublication_namespace(
    state: _StagingAuthorityState,
    parent_descriptor: int,
) -> None:
    parent = _require_parent_security(
        state.trusted_root,
        parent_descriptor,
    )
    if (
        parent.public.device != state.parent_device
        or parent.public.inode != state.parent_inode
    ):
        raise StagingAuthorityError(state.lifecycle, "publish")
    if state.kind == "regular_file":
        opened = _make_ledger_identity(
            _fstat_descriptor(state.staging_descriptor)
        )
        named = _make_ledger_identity(
            _stat_at(state.staging.value, parent_descriptor)
        )
        ledger_valid = opened == state.root_identity and named == state.root_identity
    else:
        if state.tree.inventory is None:
            ledger_valid = False
        else:
            observed_tree = _scan_staged_tree(
                state,
                scope=state.tree.inventory.scope,
                hash_files=True,
                require_nonempty=True,
            )
            ledger_valid = observed_tree == state.tree
    if not ledger_valid:
        raise PublicationValidationError(
            "sealed staging ledger changed immediately before publication",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )

    source_observation, source_identity = _observe_named_entry(
        parent_descriptor,
        state.staging.value,
        state.root_identity.public,
    )
    destination_observation, destination_identity = _observe_named_entry(
        parent_descriptor,
        state.destination.value,
        state.root_identity.public,
    )
    namespace = _collect_namespace_evidence(
        parent_descriptor,
        state.destination,
    )
    if (
        source_observation == "exact"
        and destination_observation == "absent"
        and namespace.namespace_observation == "no_conflict"
    ):
        return
    parent_fsync = _safe_parent_fsync(
        parent_descriptor,
        expected_device=state.parent_device,
        expected_inode=state.parent_inode,
    )
    evidence = _make_publication_evidence(
        staging_identity=state.root_identity.public,
        source_observation=source_observation,
        observed_source_identity=source_identity,
        destination_observation=destination_observation,
        observed_destination_identity=destination_identity,
        namespace_evidence=namespace,
        parent_fsync=parent_fsync,
        native_errno=None,
    )
    if source_observation == "exact" and (
        destination_observation == "foreign"
        or namespace.namespace_observation
        in {"complete_conflict", "bounded_conflict"}
    ):
        state.lifecycle = StagingState.NOT_COMMITTED
        raise PublicationCollisionError(
            "publication precondition found a destination collision",
            state=PublicationState.NOT_COMMITTED,
            evidence=evidence,
            destination=state.destination,
        )
    state.lifecycle = StagingState.RETIRED
    raise PublicationValidationError(
        "publication preconditions failed before the native call",
        state=PublicationState.NOT_COMMITTED,
        evidence=evidence,
        destination=state.destination,
    )


def _raise_precommit_failure(
    state: _StagingAuthorityState,
    parent_descriptor: int,
    failure: BaseException,
) -> None:
    local_retirement = None
    primary_failure = failure
    if type(failure) is _DescriptorRetirementAnomaly:
        local_retirement = failure.evidence
        primary_failure = failure.primary or failure
    else:
        local_retirement = _internal_admission_retirement_evidence(
            primary_failure
        )
    authority_was_retired = state.lifecycle is StagingState.RETIRED
    try:
        if not _parent_descriptor_matches(state, parent_descriptor):
            raise RuntimeError("publication parent is uninspectable")
        source_observation, source_identity = _safe_observe_named_entry(
            parent_descriptor,
            state.staging.value,
            state.root_identity.public,
        )
        destination_observation, destination_identity = (
            _safe_observe_named_entry(
                parent_descriptor,
                state.destination.value,
                state.root_identity.public,
            )
        )
        namespace = _safe_namespace_evidence(
            parent_descriptor,
            state.destination,
        )
    except BaseException:
        source_observation, source_identity = "uninspectable", None
        destination_observation, destination_identity = "uninspectable", None
        namespace = _make_namespace_evidence(
            _NamespaceObservation.UNINSPECTABLE,
            reference=state.destination,
        )
    parent_fsync = _safe_parent_fsync(
        parent_descriptor,
        expected_device=state.parent_device,
        expected_inode=state.parent_inode,
    )
    evidence = _make_publication_evidence(
        staging_identity=state.root_identity.public,
        source_observation=source_observation,
        observed_source_identity=source_identity,
        destination_observation=destination_observation,
        observed_destination_identity=destination_identity,
        namespace_evidence=namespace,
        parent_fsync=parent_fsync,
        native_errno=_errno_from_exception(primary_failure),
    )
    if (
        not authority_was_retired
        and source_observation == "exact"
    ):
        state.lifecycle = StagingState.NOT_COMMITTED
        if (
            destination_observation == "foreign"
            or namespace.namespace_observation
            in {"complete_conflict", "bounded_conflict"}
        ):
            error_type: type[PublicationError] = PublicationCollisionError
        elif isinstance(primary_failure, _RequiredFsyncError):
            error_type = PublicationDurabilityError
        else:
            error_type = PublicationValidationError
        error = error_type(
            "publication failed before the native no-replace call",
            state=PublicationState.NOT_COMMITTED,
            evidence=evidence,
            destination=state.destination,
        )
        if local_retirement is not None:
            _queue_retirement_evidence(
                error,
                state.error_provenance,
                local_retirement,
            )
        raise error from primary_failure
    state.lifecycle = StagingState.RETIRED
    error = PublicationValidationError(
        "pre-publication failure left staging authority unsafe",
        state=PublicationState.NOT_COMMITTED,
        evidence=evidence,
        destination=state.destination,
    )
    if local_retirement is not None:
        _queue_retirement_evidence(
            error,
            state.error_provenance,
            local_retirement,
        )
    raise error from primary_failure


def _flush_staged_file(
    state: _StagingAuthorityState,
    parent_descriptor: int,
) -> None:
    before_open = _make_ledger_identity(
        _fstat_descriptor(state.staging_descriptor)
    )
    before_named = _make_ledger_identity(
        _stat_at(state.staging.value, parent_descriptor)
    )
    if (
        before_open != state.root_identity
        or before_named != state.root_identity
    ):
        raise PublicationValidationError(
            "sealed staged file changed before fsync",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    _required_fsync(state.staging_descriptor, "staged regular-file")
    after_open = _make_ledger_identity(
        _fstat_descriptor(state.staging_descriptor)
    )
    after_named = _make_ledger_identity(
        _stat_at(state.staging.value, parent_descriptor)
    )
    if (
        after_open != state.root_identity
        or after_named != state.root_identity
    ):
        raise PublicationValidationError(
            "sealed staged file changed across fsync",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )


def _flush_staged_directory(
    state: _StagingAuthorityState,
    parent_descriptor: int,
) -> None:
    if state.tree.inventory is None:
        raise PublicationValidationError(
            "sealed directory has no H2b inventory identity",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    before = _scan_staged_tree(
        state,
        scope=state.tree.inventory.scope,
        hash_files=True,
        require_nonempty=True,
    )
    if before != state.tree:
        raise PublicationValidationError(
            "sealed staged directory changed before flushing",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )

    for path, identity in before.entries:
        if identity.public.entry_type != "regular_file":
            continue
        parent = _open_tree_parent(state, path)
        parent_owned = _owned_descriptor(parent, None)
        parent_inode: int | None = None
        descriptor = -1
        descriptor_owned: _OwnedDescriptor | None = None
        active: BaseException | None = None
        try:
            parent_inode = _fstat_descriptor(parent).st_ino
            named_before = _make_ledger_identity(
                _stat_at(path.parts[-1], parent)
            )
            descriptor = _open_read_file_at(
                path.parts[-1],
                parent,
                role="traversal_entry",
            )
            descriptor_owned = _owned_descriptor(
                descriptor,
                "traversal_entry",
            )
            opened_before = _make_ledger_identity(
                _fstat_descriptor(descriptor)
            )
            if named_before != identity or opened_before != identity:
                raise PublicationValidationError(
                    "staged file changed before fsync",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            _required_fsync(descriptor, "staged tree regular-file")
            if (
                _make_ledger_identity(_fstat_descriptor(descriptor)) != identity
                or _make_ledger_identity(_stat_at(path.parts[-1], parent))
                != identity
            ):
                raise PublicationValidationError(
                    "staged file changed across fsync",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
        except BaseException as exc:
            active = exc
        finally:
            candidates = tuple(
                candidate
                for candidate in (descriptor_owned, parent_owned)
                if candidate is not None
            )
            descriptor = parent = -1
            descriptor_owned = parent_owned = None
            _raise_after_local_retirement(active, candidates)

    directories = [
        (path, identity)
        for path, identity in before.entries
        if identity.public.entry_type == "directory"
    ]
    directories.sort(
        key=lambda item: (-len(item[0].parts), portable_path_sort_key(item[0]))
    )
    for path, identity in directories:
        parent = _open_tree_parent(state, path)
        parent_owned = _owned_descriptor(parent, None)
        parent_inode: int | None = None
        descriptor = -1
        descriptor_owned: _OwnedDescriptor | None = None
        active = None
        try:
            parent_inode = _fstat_descriptor(parent).st_ino
            descriptor = _open_directory_at(
                path.parts[-1],
                parent,
                role="traversal_directory",
            )
            descriptor_owned = _owned_descriptor(
                descriptor,
                "traversal_directory",
            )
            if (
                _make_ledger_identity(_fstat_descriptor(descriptor)) != identity
                or _make_ledger_identity(_stat_at(path.parts[-1], parent))
                != identity
            ):
                raise PublicationValidationError(
                    "staged directory changed before fsync",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            _required_fsync(descriptor, "staged child-directory")
            if (
                _make_ledger_identity(_fstat_descriptor(descriptor)) != identity
                or _make_ledger_identity(_stat_at(path.parts[-1], parent))
                != identity
            ):
                raise PublicationValidationError(
                    "staged directory changed across fsync",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
        except BaseException as exc:
            active = exc
        finally:
            candidates = tuple(
                candidate
                for candidate in (descriptor_owned, parent_owned)
                if candidate is not None
            )
            descriptor = parent = -1
            descriptor_owned = parent_owned = None
            _raise_after_local_retirement(active, candidates)

    if (
        _make_ledger_identity(_fstat_descriptor(state.staging_descriptor))
        != before.root
        or _make_ledger_identity(
            _stat_at(state.staging.value, parent_descriptor)
        )
        != before.root
    ):
        raise PublicationValidationError(
            "staged root changed before fsync",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    _required_fsync(state.staging_descriptor, "staged root-directory")
    if (
        _make_ledger_identity(_fstat_descriptor(state.staging_descriptor))
        != before.root
        or _make_ledger_identity(
            _stat_at(state.staging.value, parent_descriptor)
        )
        != before.root
    ):
        raise PublicationValidationError(
            "staged root changed across fsync",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )

    after = _scan_staged_tree(
        state,
        scope=state.tree.inventory.scope,
        hash_files=True,
        require_nonempty=True,
    )
    if after != before:
        raise PublicationValidationError(
            "staged directory changed after flushing",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )


def cleanup_owned_staging(
    trusted_root: TrustedRoot,
    staging: StagedFileHandle | StagedDirectoryHandle,
) -> StagingCleanupResult:
    """Remove one exact owned staging entry without touching foreign state."""

    if type(trusted_root) is not TrustedRoot:
        raise TypeError("trusted_root must be exactly TrustedRoot")
    if type(staging) is StagedFileHandle:
        expected_type: type[StagedFileHandle] | type[StagedDirectoryHandle] = (
            StagedFileHandle
        )
    elif type(staging) is StagedDirectoryHandle:
        expected_type = StagedDirectoryHandle
    else:
        raise TypeError("cleanup requires an exact H2c1 staging handle")
    with _locked_handle_state(
        staging,
        expected_type,
        operation="cleanup",
        terminal_authority=(
            trusted_root,
            (
                StagingState.OPEN,
                StagingState.SEALED,
                StagingState.NOT_COMMITTED,
            ),
        ),
    ) as state:
        if state.cleanup_attempted:
            raise StagingLifecycleError(state.lifecycle, "cleanup")
        state.cleanup_attempted = True
        handle_retirement = _retire_state_descriptors(
            state,
            raise_on_anomaly=False,
        )
        return _cleanup_owned_state(
            state,
            prior_retirement=handle_retirement,
        )


def _cleanup_owned_state(
    state: _StagingAuthorityState,
    *,
    prior_retirement: DescriptorRetirementEvidence | None,
) -> StagingCleanupResult:
    parent_descriptor = -1
    staging_descriptor = -1
    operation_parent: _OwnedDescriptor | None = None
    operation_staging: _OwnedDescriptor | None = None
    removed: set[str] = set()
    directory_journal: dict[str, _LedgerIdentity] = {}
    root_was_admitted = False
    root_removed = False
    root_anomaly: Literal["malformed", "uninspectable"] | None = None
    parent_sync = "not_attempted"
    escaping_error: BaseException | None = None
    terminal_result_failure_delivered = False
    try:
        try:
            parent_descriptor = _open_publication_parent(
                state.trusted_root,
                role="operation_parent",
            )
            operation_parent = _owned_descriptor(
                parent_descriptor,
                "operation_parent",
            )
            parent = _require_parent_security(
                state.trusted_root,
                parent_descriptor,
            )
        except BaseException as exc:
            authority_error = StagingAuthorityError(
                state.lifecycle,
                "cleanup",
            )
            admission_retirement = _internal_admission_retirement_evidence(
                exc
            )
            if admission_retirement is not None:
                _queue_retirement_evidence(
                    authority_error,
                    state.error_provenance,
                    admission_retirement,
                )
            raise authority_error from exc
        if (
            parent.public.device != state.parent_device
            or parent.public.inode != state.parent_inode
        ):
            raise StagingAuthorityError(state.lifecycle, "cleanup")
        namespace = _collect_namespace_evidence(
            parent_descriptor,
            state.staging,
        )
        if namespace.namespace_observation != "no_conflict":
            raise PublicationValidationError(
                "staging cleanup namespace is conflicting or incomplete",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )

        root_observation, observed_root = _observe_named_entry(
            parent_descriptor,
            state.staging.value,
            state.root_identity.public,
        )
        directory_authority_candidate = (
            state.kind == "directory"
            and root_observation == "foreign"
            and observed_root is not None
            and _same_authority(
                observed_root,
                state.root_identity.public,
            )
            and observed_root.mode == state.root_identity.public.mode
        )
        if root_observation != "exact" and not directory_authority_candidate:
            raise PublicationValidationError(
                "staging cleanup root is absent, foreign, or unstable",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        if state.kind == "regular_file":
            staging_descriptor = _open_read_file_at(
                state.staging.value,
                parent_descriptor,
                role="operation_staging",
            )
        else:
            staging_descriptor = _open_directory_at(
                state.staging.value,
                parent_descriptor,
                role="operation_staging",
            )
        operation_staging = _owned_descriptor(
            staging_descriptor,
            "operation_staging",
        )
        opened_root = _make_ledger_identity(
            _fstat_descriptor(staging_descriptor)
        )
        named_root = _make_ledger_identity(
            _stat_at(state.staging.value, parent_descriptor)
        )
        root_is_admitted = (
            opened_root == named_root
            and (
                opened_root == state.root_identity
                if state.kind == "regular_file"
                else _same_cleanup_directory_authority(
                    opened_root,
                    state.root_identity,
                )
            )
        )
        if not root_is_admitted:
            raise PublicationValidationError(
                "staging cleanup root changed during admission",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        root_was_admitted = True

        operation_state = _StagingAuthorityState(
            trusted_root=state.trusted_root,
            backend=state.backend,
            parent_device=state.parent_device,
            parent_inode=state.parent_inode,
            destination=state.destination,
            staging=state.staging,
            kind=state.kind,
            lifecycle=state.lifecycle,
            root_identity=opened_root,
            tree=state.tree,
            parent_descriptor=parent_descriptor,
            staging_descriptor=staging_descriptor,
            size_bytes=state.size_bytes,
        )

        if state.kind == "regular_file":
            if not _parent_descriptor_matches(state, parent_descriptor):
                raise PublicationValidationError(
                    "publication parent changed before staged-file cleanup",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            if (
                _make_ledger_identity(
                    _fstat_descriptor(staging_descriptor)
                )
                != state.root_identity
                or _make_ledger_identity(
                    _stat_at(state.staging.value, parent_descriptor)
                )
                != state.root_identity
            ):
                raise PublicationValidationError(
                    "staged file changed immediately before cleanup",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            _unlink_at(state.staging.value, parent_descriptor)
            root_removed = True
            removed.add("")
            removed_observation, _removed_identity = _observe_named_entry(
                parent_descriptor,
                state.staging.value,
                state.root_identity.public,
            )
            if removed_observation != "absent":
                raise OSError(errno.EIO, "staged file removal was not observed")
        else:
            try:
                current = _scan_staged_tree(
                    operation_state,
                    scope=None,
                    hash_files=False,
                    require_nonempty=False,
                )
            except _DescriptorRetirementAnomaly as anomaly:
                root_anomaly = (
                    "malformed"
                    if isinstance(
                        anomaly.primary,
                        PublicationValidationError,
                    )
                    else "uninspectable"
                )
                raise
            except PublicationValidationError:
                root_anomaly = "malformed"
                raise
            except BaseException:
                root_anomaly = "uninspectable"
                raise
            if (
                current.root != state.tree.root
                or current.entries != state.tree.entries
            ):
                root_anomaly = "malformed"
                raise PublicationValidationError(
                    "staged directory ledger changed before cleanup",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            try:
                _delete_staged_tree(
                    operation_state,
                    removed=removed,
                    directory_journal=directory_journal,
                )
            except _DescriptorRetirementAnomaly as anomaly:
                root_anomaly = (
                    "malformed"
                    if isinstance(
                        anomaly.primary,
                        PublicationValidationError,
                    )
                    else "uninspectable"
                )
                raise
            except PublicationValidationError:
                root_anomaly = "malformed"
                raise
            final_tree = _validate_cleanup_residue(
                operation_state,
                original=state.tree,
                removed=removed,
                directory_journal=directory_journal,
            )
            if final_tree.entries:
                raise PublicationValidationError(
                    "staged cleanup left expected tree members",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            opened_final_root = _make_ledger_identity(
                _fstat_descriptor(staging_descriptor)
            )
            named_final_root = _make_ledger_identity(
                _stat_at(state.staging.value, parent_descriptor)
            )
            if (
                opened_final_root != final_tree.root
                or named_final_root != final_tree.root
                or not _same_cleanup_directory_authority(
                    final_tree.root,
                    state.tree.root,
                )
            ):
                root_anomaly = "malformed"
                raise PublicationValidationError(
                    "staged cleanup root changed before removal",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            if not _parent_descriptor_matches(state, parent_descriptor):
                raise PublicationValidationError(
                    "publication parent changed before staged-root cleanup",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            if (
                _make_ledger_identity(
                    _fstat_descriptor(staging_descriptor)
                )
                != final_tree.root
                or _make_ledger_identity(
                    _stat_at(state.staging.value, parent_descriptor)
                )
                != final_tree.root
            ):
                raise PublicationValidationError(
                    "staged root changed immediately before cleanup",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            _rmdir_at(state.staging.value, parent_descriptor)
            root_removed = True
            removed_observation, _removed_identity = _observe_named_entry(
                parent_descriptor,
                state.staging.value,
                state.root_identity.public,
            )
            if removed_observation != "absent":
                raise OSError(
                    errno.EIO,
                    "staged directory removal was not observed",
                )

        final_namespace = (
            _safe_namespace_evidence(
                parent_descriptor,
                state.staging,
            )
            if _parent_descriptor_matches(state, parent_descriptor)
            else _make_namespace_evidence(
                _NamespaceObservation.UNINSPECTABLE,
                reference=state.staging,
            )
        )
        parent_sync = _safe_parent_fsync(
            parent_descriptor,
            expected_device=state.parent_device,
            expected_inode=state.parent_inode,
        )
        if parent_sync != "succeeded":
            raise OSError(
                errno.EIO,
                "publication-parent fsync did not establish durability",
            )
        if final_namespace.namespace_observation != "no_conflict":
            raise PublicationValidationError(
                "staging cleanup completed with conflicting or incomplete "
                "namespace evidence",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )

        state.lifecycle = StagingState.DISCARDED
        try:
            result = _make_cleanup_result(
                state.staging,
                state.root_identity.public,
                final_namespace,
            )
        except BaseException as result_error:
            fallback_result = _allocate_cleanup_result(
                state.staging,
                state.root_identity.public,
                final_namespace,
            )
            additional = tuple(
                candidate
                for candidate in (operation_staging, operation_parent)
                if candidate is not None
            )
            staging_descriptor = -1
            parent_descriptor = -1
            terminal_result_failure_delivered = True
            _raise_after_terminal_result_failure(
                state,
                operation="cleanup",
                primary=result_error,
                terminal_result=fallback_result,
                additional=additional,
                prior_evidence=prior_retirement,
            )
        additional = tuple(
            candidate
            for candidate in (operation_staging, operation_parent)
            if candidate is not None
        )
        staging_descriptor = -1
        parent_descriptor = -1
        return _retire_with_terminal_result(
            state,
            "cleanup",
            result,
            additional=additional,
            prior_evidence=prior_retirement,
        )  # type: ignore[return-value]
    except DescriptorRetirementError as error:
        escaping_error = error
        raise
    except BaseException as exc:
        escaping_error = exc
        if terminal_result_failure_delivered:
            raise
        local_retirement = None
        primary_exc = exc
        if type(exc) is _DescriptorRetirementAnomaly:
            local_retirement = exc.evidence
            primary_exc = exc.primary or exc
        else:
            local_retirement = _combine_retirement_evidence(
                getattr(primary_exc, "_retirement_evidence", None),
                _internal_admission_retirement_evidence(primary_exc),
            )
        state.lifecycle = StagingState.RETIRED
        if parent_sync == "not_attempted":
            parent_sync = _safe_parent_fsync(
                parent_descriptor,
                expected_device=state.parent_device,
                expected_inode=state.parent_inode,
            )
        try:
            evidence_state, evidence = _cleanup_failure_evidence(
                state,
                parent_descriptor,
                staging_descriptor,
                root_was_admitted=root_was_admitted,
                root_removed=root_removed,
                removed=removed,
                directory_journal=directory_journal,
                root_anomaly=root_anomaly,
                parent_fsync=parent_sync,
                native_errno=_errno_from_exception(primary_exc),
                allow_partial_verification=local_retirement is None,
            )
        except _DescriptorRetirementAnomaly as recovery:
            local_retirement = _combine_retirement_evidence(
                local_retirement,
                recovery.evidence,
            )
            evidence_state = StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN
            evidence = _make_cleanup_evidence(
                staging_identity=state.root_identity.public,
                root_observation="uninspectable",
                observed_root_identity=None,
                remaining_expected_entries=None,
                namespace_evidence=_make_namespace_evidence(
                    _NamespaceObservation.UNINSPECTABLE,
                    reference=state.staging,
                ),
                parent_fsync=(
                    parent_sync
                    if parent_sync
                    in {"not_attempted", "succeeded", "failed", "uncertain"}
                    else "uncertain"
                ),
                native_errno=_errno_from_exception(
                    recovery.primary or primary_exc
                ),
            )
        except BaseException:
            evidence_state = StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN
            evidence = _make_cleanup_evidence(
                staging_identity=state.root_identity.public,
                root_observation="uninspectable",
                observed_root_identity=None,
                remaining_expected_entries=None,
                namespace_evidence=_make_namespace_evidence(
                    _NamespaceObservation.UNINSPECTABLE,
                    reference=state.staging,
                ),
                parent_fsync=(
                    parent_sync
                    if parent_sync
                    in {"not_attempted", "succeeded", "failed", "uncertain"}
                    else "uncertain"
                ),
                native_errno=_errno_from_exception(primary_exc),
            )
        if evidence_state is StagingCleanupState.DISCARDED_DURABLE:
            state.lifecycle = StagingState.DISCARDED
            result = _allocate_cleanup_result(
                state.staging,
                state.root_identity.public,
                evidence.namespace_evidence,
            )
            additional = tuple(
                candidate
                for candidate in (operation_staging, operation_parent)
                if candidate is not None
            )
            staging_descriptor = -1
            parent_descriptor = -1
            _raise_after_terminal_result_failure(
                state,
                operation="cleanup",
                primary=primary_exc,
                terminal_result=result,
                additional=additional,
                prior_evidence=_combine_retirement_evidence(
                    prior_retirement,
                    local_retirement,
                    post_outcome=True,
                ),
            )
        cleanup_error = StagingCleanupError(
            "owned staging cleanup did not complete durably",
            state=evidence_state,
            evidence=evidence,
            staging=state.staging,
        )
        if local_retirement is not None:
            _queue_retirement_evidence(
                cleanup_error,
                state.error_provenance,
                local_retirement,
            )
        additional = tuple(
            candidate
            for candidate in (operation_staging, operation_parent)
            if candidate is not None
        )
        staging_descriptor = -1
        parent_descriptor = -1
        try:
            _retire_with_primary_error(
                state,
                cleanup_error,
                additional=additional,
                post_outcome=local_retirement is None,
                prior_evidence=prior_retirement,
            )
        except BaseException as transported:
            raise transported from primary_exc
    finally:
        candidates: list[_OwnedDescriptor] = []
        if staging_descriptor >= 0:
            candidates.append(
                operation_staging
                if operation_staging is not None
                else _owned_descriptor(
                    staging_descriptor,
                    "operation_staging",
                )
            )
            staging_descriptor = -1
        if parent_descriptor >= 0:
            candidates.append(
                operation_parent
                if operation_parent is not None
                else _owned_descriptor(
                    parent_descriptor,
                    "operation_parent",
                )
            )
            parent_descriptor = -1
        if candidates:
            retirement = _retirement_evidence_from_records(
                (
                    *(
                        prior_retirement.records
                        if prior_retirement is not None
                        else ()
                    ),
                    *_retire_owned_descriptor_records(tuple(candidates)),
                )
            )
            if retirement is not None:
                if (
                    escaping_error is not None
                    and _attach_retirement_evidence(
                        escaping_error,
                        state.error_provenance,
                        retirement,
                    )
                ):
                    pass
                else:
                    authority_error = StagingAuthorityError(
                        state.lifecycle,
                        "cleanup",
                    )
                    _attach_retirement_evidence(
                        authority_error,
                        state.error_provenance,
                        retirement,
                    )
                    if escaping_error is not None:
                        raise BaseExceptionGroup(
                            "cleanup and descriptor retirement both failed",
                            [escaping_error, authority_error],
                        ) from None
                    raise authority_error


def _delete_staged_tree(
    state: _StagingAuthorityState,
    *,
    removed: set[str],
    directory_journal: dict[str, _LedgerIdentity],
) -> None:
    original = state.tree
    entries = list(original.entries)
    files = [
        (path, identity)
        for path, identity in entries
        if identity.public.entry_type == "regular_file"
    ]
    directories = [
        (path, identity)
        for path, identity in entries
        if identity.public.entry_type == "directory"
    ]
    files.sort(
        key=lambda item: (-len(item[0].parts), portable_path_sort_key(item[0]))
    )
    directories.sort(
        key=lambda item: (-len(item[0].parts), portable_path_sort_key(item[0]))
    )

    try:
        for path, identity in (*files, *directories):
            current = _validate_cleanup_residue(
                state,
                original=original,
                removed=removed,
                directory_journal=directory_journal,
            )
            current_entries = dict(current.entries)
            current_identity = current_entries.get(path)
            if current_identity is None:
                raise PublicationValidationError(
                    "staged cleanup member disappeared before removal",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=state.destination,
                )
            state.root_identity = current.root
            state.tree = current
            parent = _open_tree_parent(state, path)
            parent_owned = _owned_descriptor(parent, None)
            parent_inode: int | None = None
            opened = -1
            opened_owned: _OwnedDescriptor | None = None
            active: BaseException | None = None
            try:
                parent_inode = _fstat_descriptor(parent).st_ino
                before = _make_ledger_identity(
                    _stat_at(path.parts[-1], parent)
                )
                if before != current_identity:
                    raise PublicationValidationError(
                        "staged cleanup member changed before removal",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=state.destination,
                    )
                if identity.public.entry_type == "regular_file":
                    opened = _open_read_file_at(
                        path.parts[-1],
                        parent,
                        role="traversal_entry",
                    )
                    opened_owned = _owned_descriptor(
                        opened,
                        "traversal_entry",
                    )
                else:
                    opened = _open_directory_at(
                        path.parts[-1],
                        parent,
                        role="traversal_directory",
                    )
                    opened_owned = _owned_descriptor(
                        opened,
                        "traversal_directory",
                    )
                opened_identity = _make_ledger_identity(
                    _fstat_descriptor(opened)
                )
                named_identity = _make_ledger_identity(
                    _stat_at(path.parts[-1], parent)
                )
                if (
                    opened_identity != current_identity
                    or named_identity != current_identity
                ):
                    raise PublicationValidationError(
                        "staged cleanup member changed while opened",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=state.destination,
                    )
                if identity.public.entry_type == "directory":
                    if _directory_has_any_name(opened):
                        raise PublicationValidationError(
                            "staged cleanup directory is unexpectedly nonempty",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=state.destination,
                        )
                final_named = _make_ledger_identity(
                    _stat_at(path.parts[-1], parent)
                )
                if final_named != current_identity:
                    raise PublicationValidationError(
                        "staged cleanup member changed before final removal",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=state.destination,
                    )
                parent_path = (
                    PortableRelativePath.parse(
                        "/".join(path.parts[:-1])
                    )
                    if path.parts[:-1]
                    else None
                )
                parent_expected = (
                    current.root
                    if parent_path is None
                    else current_entries[parent_path]
                )
                if (
                    _make_ledger_identity(
                        _fstat_descriptor(parent)
                    )
                    != parent_expected
                ):
                    raise PublicationValidationError(
                        "staged cleanup parent changed before removal",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=state.destination,
                    )
                if identity.public.entry_type == "regular_file":
                    _unlink_at(path.parts[-1], parent)
                else:
                    _rmdir_at(path.parts[-1], parent)
                removed.add(path.value)
                if _name_exists(parent, path.parts[-1]):
                    raise OSError(
                        errno.EIO,
                        "staged cleanup member removal was not observed",
                    )
                removal_candidates = tuple(
                    candidate
                    for candidate in (opened_owned, parent_owned)
                    if candidate is not None
                )
                opened = parent = -1
                opened_owned = parent_owned = None
                removal_retirement = _retire_owned_descriptors(
                    removal_candidates
                )
                if removal_retirement is not None:
                    raise _DescriptorRetirementAnomaly(
                        removal_retirement
                    )
                parent_after = _fsync_cleanup_directory(
                    state,
                    parent_parts=path.parts[:-1],
                    expected_before_removal=parent_expected,
                    directory_journal=directory_journal,
                )
                directory_journal["/".join(path.parts[:-1])] = parent_after
            except BaseException as exc:
                active = exc
            finally:
                candidates = tuple(
                    candidate
                    for candidate in (opened_owned, parent_owned)
                    if candidate is not None
                )
                opened = parent = -1
                opened_owned = parent_owned = None
                _raise_after_local_retirement(active, candidates)
    finally:
        state.root_identity = original.root
        state.tree = original


def _fsync_cleanup_directory(
    state: _StagingAuthorityState,
    *,
    parent_parts: tuple[str, ...],
    expected_before_removal: _LedgerIdentity,
    directory_journal: dict[str, _LedgerIdentity],
) -> _LedgerIdentity:
    descriptor = -1
    descriptor_owned: _OwnedDescriptor | None = None
    verifier = -1
    verifier_owned: _OwnedDescriptor | None = None
    active: BaseException | None = None
    try:
        if parent_parts:
            expected_entries = dict(state.tree.entries)
            for logical, identity in directory_journal.items():
                if logical:
                    expected_entries[
                        PortableRelativePath.parse(logical)
                    ] = identity
            logical_parent = PortableRelativePath.parse(
                "/".join(parent_parts)
            )
            verifier = _open_tree_parent(
                state,
                logical_parent,
                root=directory_journal.get("", state.tree.root),
                entries=expected_entries,
            )
            verifier_owned = _owned_descriptor(
                verifier,
                "traversal_parent",
            )
            named_before = _make_ledger_identity(
                _stat_at(parent_parts[-1], verifier)
            )
            descriptor = _open_directory_at(
                parent_parts[-1],
                verifier,
                role="traversal_directory",
            )
        else:
            descriptor = state.staging_descriptor
            if descriptor < 0:
                raise StagingAuthorityError(
                    state.lifecycle,
                    "cleanup",
                )
            named_before = _make_ledger_identity(
                _stat_at(state.staging.value, state.parent_descriptor)
            )
        if parent_parts:
            descriptor_owned = _owned_descriptor(
                descriptor,
                "traversal_directory",
            )
        before = _make_ledger_identity(_fstat_descriptor(descriptor))
        if (
            before != named_before
            or not _same_cleanup_directory_authority(
                before,
                expected_before_removal,
            )
        ):
            raise PublicationValidationError(
                "cleanup parent changed before directory fsync",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        directory_journal["/".join(parent_parts)] = before
        _fsync_descriptor(descriptor)
        after = _make_ledger_identity(_fstat_descriptor(descriptor))
        named_after = _make_ledger_identity(
            _stat_at(parent_parts[-1], verifier)
            if parent_parts
            else _stat_at(
                state.staging.value,
                state.parent_descriptor,
            )
        )
        if after != before or named_after != before:
            raise PublicationValidationError(
                "cleanup parent changed across directory fsync",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )
        return after
    except BaseException as exc:
        active = exc
    finally:
        candidates = tuple(
            candidate
            for candidate in (descriptor_owned, verifier_owned)
            if candidate is not None
        )
        descriptor = verifier = -1
        descriptor_owned = verifier_owned = None
        _raise_after_local_retirement(active, candidates)


def _same_cleanup_directory_authority(
    observed: _LedgerIdentity,
    expected: _LedgerIdentity,
) -> bool:
    return (
        observed.public.device == expected.public.device
        and observed.public.inode == expected.public.inode
        and observed.public.entry_type == "directory"
        and expected.public.entry_type == "directory"
        and observed.public.owner_uid == expected.public.owner_uid
        and observed.public.mode == expected.public.mode
        and observed.group_id == expected.group_id
    )


def _validate_cleanup_residue(
    state: _StagingAuthorityState,
    *,
    original: _TreeAuthority,
    removed: set[str],
    directory_journal: dict[str, _LedgerIdentity],
) -> _TreeAuthority:
    current = _scan_staged_tree(
        state,
        scope=None,
        hash_files=False,
        require_nonempty=False,
    )
    expected_entries = {
        path: identity
        for path, identity in original.entries
        if path.value not in removed
    }
    current_entries = dict(current.entries)
    if current_entries.keys() != expected_entries.keys():
        raise PublicationValidationError(
            "staged cleanup membership is not expected ledger residue",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )

    for path, expected in expected_entries.items():
        observed = current_entries[path]
        required = directory_journal.get(path.value, expected)
        valid = observed == required
        if not valid:
            raise PublicationValidationError(
                "staged cleanup residue changed outside owned removals",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=state.destination,
            )

    root_valid = current.root == directory_journal.get("", original.root)
    if not root_valid:
        raise PublicationValidationError(
            "staged cleanup root changed outside owned removals",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=state.destination,
        )
    return current


def _name_exists(parent_descriptor: int, name: str) -> bool:
    try:
        _stat_at(name, parent_descriptor)
    except FileNotFoundError:
        return False
    return True


def _cleanup_failure_evidence(
    state: _StagingAuthorityState,
    parent_descriptor: int,
    staging_descriptor: int,
    *,
    root_was_admitted: bool,
    root_removed: bool,
    removed: set[str],
    directory_journal: dict[str, _LedgerIdentity],
    root_anomaly: Literal["malformed", "uninspectable"] | None,
    parent_fsync: str,
    native_errno: int | None,
    allow_partial_verification: bool,
) -> tuple[StagingCleanupState, StagingCleanupRecoveryEvidence]:
    if not _parent_descriptor_matches(state, parent_descriptor):
        evidence = _make_cleanup_evidence(
            staging_identity=state.root_identity.public,
            root_observation="uninspectable",
            observed_root_identity=None,
            remaining_expected_entries=None,
            namespace_evidence=_make_namespace_evidence(
                _NamespaceObservation.UNINSPECTABLE,
                reference=state.staging,
            ),
            parent_fsync="uncertain",
            native_errno=native_errno,
        )
        return StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN, evidence

    try:
        namespace = _collect_namespace_evidence(
            parent_descriptor,
            state.staging,
        )
    except BaseException:
        namespace = _make_namespace_evidence(
            _NamespaceObservation.UNINSPECTABLE,
            reference=state.staging,
        )
    try:
        root_observation, observed_root = _observe_named_entry(
            parent_descriptor,
            state.staging.value,
            state.root_identity.public,
        )
    except BaseException:
        root_observation, observed_root = "uninspectable", None

    if root_removed and root_observation == "absent":
        evidence = _make_cleanup_evidence(
            staging_identity=state.root_identity.public,
            root_observation="absent",
            observed_root_identity=None,
            remaining_expected_entries=None,
            namespace_evidence=namespace,
            parent_fsync=parent_fsync,
            native_errno=native_errno,
        )
        if namespace.namespace_observation != "no_conflict":
            cleanup_state = StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN
        elif parent_fsync != "succeeded":
            cleanup_state = (
                StagingCleanupState.DISCARDED_DURABILITY_UNCERTAIN
            )
        else:
            cleanup_state = StagingCleanupState.DISCARDED_DURABLE
        return cleanup_state, evidence

    if root_anomaly is not None:
        if root_observation == "uninspectable":
            anomaly_observation = "uninspectable"
            anomaly_identity = None
        elif (
            observed_root is not None
            and not _same_authority(
                observed_root,
                state.root_identity.public,
            )
        ):
            anomaly_observation = (
                "replaced" if root_was_admitted else "foreign"
            )
            anomaly_identity = observed_root
        else:
            anomaly_observation = root_anomaly
            anomaly_identity = (
                observed_root
                if root_anomaly == "malformed"
                and observed_root is not None
                else None
            )
        evidence = _make_cleanup_evidence(
            staging_identity=state.root_identity.public,
            root_observation=anomaly_observation,
            observed_root_identity=anomaly_identity,
            remaining_expected_entries=None,
            namespace_evidence=namespace,
            parent_fsync=parent_fsync,
            native_errno=native_errno,
        )
        return StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN, evidence

    if namespace.namespace_observation != "no_conflict":
        if root_observation == "exact":
            observed_root = state.root_identity.public
        evidence = _make_cleanup_evidence(
            staging_identity=state.root_identity.public,
            root_observation=(
                root_observation
                if root_observation
                in {
                    "exact",
                    "absent",
                    "foreign",
                    "replaced",
                    "contradictory",
                    "uninspectable",
                }
                else "uninspectable"
            ),
            observed_root_identity=observed_root,
            remaining_expected_entries=(
                len(state.tree.entries)
                if root_observation == "exact"
                else None
            ),
            namespace_evidence=namespace,
            parent_fsync=parent_fsync,
            native_errno=native_errno,
        )
        return StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN, evidence

    if (
        allow_partial_verification
        and root_was_admitted
        and observed_root is not None
    ):
        if (
            _same_authority(observed_root, state.root_identity.public)
            and observed_root.mode == state.root_identity.public.mode
        ):
            if state.kind == "regular_file":
                partial = (
                    (state.root_identity.public, 0)
                    if (
                        not removed
                        and _make_ledger_identity(
                            _fstat_descriptor(staging_descriptor)
                        )
                        == state.root_identity
                        and _make_ledger_identity(
                            _stat_at(
                                state.staging.value,
                                parent_descriptor,
                            )
                        )
                        == state.root_identity
                    )
                    else None
                )
            else:
                partial = _verify_owned_partial_residue(
                    state,
                    parent_descriptor,
                    staging_descriptor,
                    removed,
                    directory_journal,
                )
            if partial is not None:
                partial_root, partial_count = partial
                evidence = _make_cleanup_evidence(
                    staging_identity=state.root_identity.public,
                    root_observation=(
                        "owned_partial" if removed else "exact"
                    ),
                    observed_root_identity=partial_root,
                    remaining_expected_entries=partial_count,
                    namespace_evidence=namespace,
                    parent_fsync=parent_fsync,
                    native_errno=native_errno,
                )
                return StagingCleanupState.NOT_DISCARDED, evidence
        root_observation = (
            "replaced"
            if not _same_authority(observed_root, state.root_identity.public)
            else "malformed"
        )

    if root_observation == "foreign" and root_was_admitted:
        root_observation = "replaced"
    if root_observation not in {
        "absent",
        "foreign",
        "replaced",
        "contradictory",
        "malformed",
        "uninspectable",
    }:
        root_observation = "malformed"
    if root_observation in {"malformed"}:
        identity_for_evidence = observed_root
    elif root_observation in {"foreign", "replaced"}:
        identity_for_evidence = observed_root
    else:
        identity_for_evidence = None
    evidence = _make_cleanup_evidence(
        staging_identity=state.root_identity.public,
        root_observation=root_observation,
        observed_root_identity=identity_for_evidence,
        remaining_expected_entries=None,
        namespace_evidence=namespace,
        parent_fsync=parent_fsync,
        native_errno=native_errno,
    )
    return StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN, evidence


def _parent_descriptor_matches(
    state: _StagingAuthorityState,
    parent_descriptor: int,
) -> bool:
    if parent_descriptor < 0:
        return False
    try:
        observed = _fstat_descriptor(parent_descriptor)
    except OSError:
        return False
    return (
        stat.S_ISDIR(observed.st_mode)
        and _validate_host_uint("device", observed.st_dev)
        == state.parent_device
        and _validate_host_uint("inode", observed.st_ino)
        == state.parent_inode
        and observed.st_uid == os.geteuid()
        and stat.S_IMODE(observed.st_mode) & 0o700 == 0o700
        and not stat.S_IMODE(observed.st_mode) & 0o022
    )


def _verify_owned_partial_residue(
    state: _StagingAuthorityState,
    parent_descriptor: int,
    staging_descriptor: int,
    removed: set[str],
    directory_journal: dict[str, _LedgerIdentity],
) -> tuple[PublicationEntryIdentity, int] | None:
    if staging_descriptor < 0:
        return None
    try:
        opened_root = _make_ledger_identity(
            _fstat_descriptor(staging_descriptor)
        )
        named_root = _make_ledger_identity(
            _stat_at(state.staging.value, parent_descriptor)
        )
        if (
            opened_root != named_root
            or not _same_cleanup_directory_authority(
                opened_root,
                state.root_identity,
            )
        ):
            return None
        operation_state = _StagingAuthorityState(
            trusted_root=state.trusted_root,
            backend=state.backend,
            parent_device=state.parent_device,
            parent_inode=state.parent_inode,
            destination=state.destination,
            staging=state.staging,
            kind="directory",
            lifecycle=StagingState.RETIRED,
            root_identity=opened_root,
            tree=state.tree,
            parent_descriptor=parent_descriptor,
            staging_descriptor=staging_descriptor,
        )
        current = _validate_cleanup_residue(
            operation_state,
            original=state.tree,
            removed=removed,
            directory_journal=directory_journal,
        )
    except _DescriptorRetirementAnomaly:
        raise
    except BaseException:
        return None

    return current.root.public, len(current.entries)


__all__ = [
    "DescriptorRetirementObservation",
    "DescriptorRetirementIdentity",
    "DescriptorRetirementRecord",
    "DescriptorRetirementEvidence",
    "DescriptorRetirementError",
    "PublicationState",
    "StagingState",
    "StagingCleanupState",
    "PublicationResult",
    "StagingCleanupResult",
    "PublicationEntryIdentity",
    "NamespaceEvidence",
    "PublicationRecoveryEvidence",
    "StagingCleanupRecoveryEvidence",
    "StagedFileHandle",
    "StagedDirectoryHandle",
    "StagingLifecycleError",
    "StagingAuthorityError",
    "PublicationError",
    "PublicationValidationError",
    "PublicationCapabilityError",
    "PublicationCollisionError",
    "StagingAdmissionError",
    "PublicationDurabilityError",
    "PublicationOutcomeUncertainError",
    "PublicationNamespaceConflictError",
    "PublicationNamespaceUncertainError",
    "StagingCleanupError",
    "open_exclusive_staged_file",
    "open_exclusive_staged_directory",
    "publish_completed_file",
    "publish_completed_directory",
    "cleanup_owned_staging",
]
