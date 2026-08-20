"""Standard-library contracts for H2c1 atomic no-replace publication."""

from __future__ import annotations

import ast
from contextlib import AbstractContextManager, ExitStack, contextmanager
import copy
from dataclasses import fields
from enum import Enum
import errno
import gc
import importlib
import inspect
import os
from pathlib import Path
import pickle
import socket
import stat
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from typing import Iterable, Literal, get_type_hints
import weakref


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc import safety
from research_platform.hpc.safety import (
    DescriptorRetirementError,
    DescriptorRetirementEvidence,
    DescriptorRetirementIdentity,
    DescriptorRetirementObservation,
    DescriptorRetirementRecord,
    NamespaceEvidence,
    PortableRelativePath,
    PublicationCapabilityError,
    PublicationCollisionError,
    PublicationEntryIdentity,
    PublicationError,
    PublicationNamespaceConflictError,
    PublicationNamespaceUncertainError,
    PublicationOutcomeUncertainError,
    PublicationRecoveryEvidence,
    PublicationResult,
    PublicationState,
    PublicationValidationError,
    StagedDirectoryHandle,
    StagedFileHandle,
    StagingAdmissionError,
    StagingAuthorityError,
    StagingCleanupError,
    StagingCleanupRecoveryEvidence,
    StagingCleanupResult,
    StagingCleanupState,
    StagingLifecycleError,
    StagingState,
    cleanup_owned_staging,
    open_exclusive_staged_directory,
    open_exclusive_staged_file,
    open_trusted_root,
    publish_completed_directory,
    publish_completed_file,
    scan_regular_file_inventory,
)


publication_module = importlib.import_module(
    "research_platform.hpc.safety.publication"
)


@contextmanager
def synthetic_publication_root():
    """Yield one lexically canonical, private synthetic publication parent."""

    with tempfile.TemporaryDirectory() as temporary:
        canonical_temporary = Path(os.path.realpath(temporary))
        root = canonical_temporary / "publication-parent"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        yield root


def portable(value: str) -> PortableRelativePath:
    return PortableRelativePath.parse(value)


def staging_leaf(destination: PortableRelativePath) -> str:
    return publication_module._staging_leaf(destination).value


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


def assert_no_entries(test: unittest.TestCase, root: Path) -> None:
    test.assertEqual(tuple(root.iterdir()), ())


def lowest_available_descriptor() -> int:
    """Observe the next POSIX descriptor number without `/proc` or `/dev/fd`."""

    descriptor = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
    os.close(descriptor)
    return descriptor


def synthetic_tree_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    """Return bounded structural bytes and modes for one synthetic fixture."""

    observed: list[tuple[object, ...]] = []

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            relative = (*prefix, entry.name)
            result = entry.stat(follow_symlinks=False)
            entry_mode = stat.S_IMODE(result.st_mode)
            if stat.S_ISREG(result.st_mode):
                payload: bytes | None = Path(entry.path).read_bytes()
                entry_type = "regular_file"
            elif stat.S_ISDIR(result.st_mode):
                payload = None
                entry_type = "directory"
            else:
                payload = None
                entry_type = "special"
            observed.append(
                ("/".join(relative), entry_type, entry_mode, payload)
            )
            if stat.S_ISDIR(result.st_mode):
                visit(Path(entry.path), relative)

    visit(root, ())
    return tuple(observed)


class PublicSurfaceTests(unittest.TestCase):
    def test_package_exports_exact_h2c1_surface(self) -> None:
        ordered = (
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
        )
        required = set(ordered)
        self.assertEqual(tuple(publication_module.__all__), ordered)
        self.assertTrue(required <= set(safety.__all__))
        for name in required:
            with self.subTest(name=name):
                self.assertIs(getattr(safety, name), getattr(publication_module, name))

        for deferred in (
            "ClaimHandle",
            "ClaimRecord",
            "ReceiptEnvelope",
            "build_claim_record",
            "acquire_exclusive_claim",
            "release_exclusive_claim",
        ):
            with self.subTest(deferred=deferred):
                self.assertNotIn(deferred, safety.__all__)
                self.assertFalse(hasattr(safety, deferred))

        # The staging domain is frozen in publication.py but is not part of
        # the exact package-level authority list.
        self.assertNotIn("PUBLICATION_STAGING_DOMAIN", safety.__all__)
        self.assertFalse(hasattr(safety, "PUBLICATION_STAGING_DOMAIN"))

    def test_public_value_field_order_and_runtime_types_are_exact(self) -> None:
        entry_type = Literal[
            "regular_file",
            "directory",
            "symlink",
            "fifo",
            "socket",
            "character_device",
            "block_device",
            "other",
        ]
        namespace_observation = Literal[
            "not_attempted",
            "no_conflict",
            "complete_conflict",
            "bounded_conflict",
            "uninspectable",
        ]
        publication_observation = Literal[
            "not_attempted",
            "exact",
            "absent",
            "foreign",
            "replaced",
            "contradictory",
            "uninspectable",
        ]
        parent_fsync = Literal[
            "not_attempted",
            "succeeded",
            "failed",
            "uncertain",
        ]
        expected_types = {
            PublicationResult: {
                "state": PublicationState,
                "destination": PortableRelativePath,
                "destination_identity": PublicationEntryIdentity,
                "namespace_evidence": NamespaceEvidence,
            },
            StagingCleanupResult: {
                "state": StagingCleanupState,
                "staging": PortableRelativePath,
                "discarded_identity": PublicationEntryIdentity,
                "namespace_evidence": NamespaceEvidence,
            },
            PublicationEntryIdentity: {
                "device": int,
                "inode": int,
                "entry_type": entry_type,
                "link_count": int,
                "owner_uid": int,
                "mode": int,
                "size_bytes": int,
            },
            NamespaceEvidence: {
                "namespace_observation": namespace_observation,
                "conflicting_aliases": tuple[PortableRelativePath, ...],
                "conflicting_alias_count": int | None,
                "aliases_complete": bool,
            },
            PublicationRecoveryEvidence: {
                "staging_identity": PublicationEntryIdentity,
                "source_observation": publication_observation,
                "observed_source_identity": PublicationEntryIdentity | None,
                "destination_observation": publication_observation,
                "observed_destination_identity": (
                    PublicationEntryIdentity | None
                ),
                "namespace_evidence": NamespaceEvidence,
                "parent_fsync": parent_fsync,
                "native_errno": int | None,
            },
            StagingCleanupRecoveryEvidence: {
                "staging_identity": PublicationEntryIdentity,
                "root_observation": Literal[
                    "exact",
                    "owned_partial",
                    "absent",
                    "foreign",
                    "replaced",
                    "contradictory",
                    "malformed",
                    "uninspectable",
                ],
                "observed_root_identity": PublicationEntryIdentity | None,
                "remaining_expected_entries": int | None,
                "namespace_evidence": NamespaceEvidence,
                "parent_fsync": parent_fsync,
                "native_errno": int | None,
            },
            DescriptorRetirementIdentity: {
                "device": int,
                "inode": int,
                "entry_type": entry_type,
                "owner_uid": int,
            },
            DescriptorRetirementRecord: {
                "ordinal": int,
                "role": Literal[
                    "traversal_entry",
                    "traversal_directory",
                    "traversal_parent",
                    "operation_staging",
                    "operation_parent",
                    "handle_staging",
                    "handle_parent",
                ],
                "observation": DescriptorRetirementObservation,
                "close_attempted": bool,
                "admitted_identity": DescriptorRetirementIdentity | None,
                "observed_identity": DescriptorRetirementIdentity | None,
                "error_errno": int | None,
            },
            DescriptorRetirementEvidence: {
                "records": tuple[DescriptorRetirementRecord, ...],
            },
        }
        for value_type, types in expected_types.items():
            with self.subTest(value=value_type.__name__):
                self.assertEqual(
                    tuple(field.name for field in fields(value_type)),
                    tuple(types),
                )
                self.assertEqual(get_type_hints(value_type), types)

    def test_staging_allocator_and_context_implementation_are_not_exposed(
        self,
    ) -> None:
        for name in (
            "_staging_allocator",
            "_open_staging_context_impl",
            "_bind_staging_entrypoints",
            "_STAGING_CREATION_TOKEN",
            "_ROOT_CREATION_TOKEN",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(publication_module, name))

    def test_module_level_signatures_are_exact(self) -> None:
        expected = {
            open_exclusive_staged_file: (
                (
                    ("trusted_root", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    ("destination", inspect.Parameter.KEYWORD_ONLY),
                ),
                {
                    "trusted_root": safety.TrustedRoot,
                    "destination": PortableRelativePath,
                    "return": AbstractContextManager[StagedFileHandle],
                },
            ),
            open_exclusive_staged_directory: (
                (
                    ("trusted_root", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    ("destination", inspect.Parameter.KEYWORD_ONLY),
                ),
                {
                    "trusted_root": safety.TrustedRoot,
                    "destination": PortableRelativePath,
                    "return": AbstractContextManager[
                        StagedDirectoryHandle
                    ],
                },
            ),
            publish_completed_file: (
                (
                    ("trusted_root", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    ("staging", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                ),
                {
                    "trusted_root": safety.TrustedRoot,
                    "staging": StagedFileHandle,
                    "return": PublicationResult,
                },
            ),
            publish_completed_directory: (
                (
                    ("trusted_root", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    ("staging", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                ),
                {
                    "trusted_root": safety.TrustedRoot,
                    "staging": StagedDirectoryHandle,
                    "return": PublicationResult,
                },
            ),
            cleanup_owned_staging: (
                (
                    ("trusted_root", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    ("staging", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                ),
                {
                    "trusted_root": safety.TrustedRoot,
                    "staging": StagedFileHandle | StagedDirectoryHandle,
                    "return": StagingCleanupResult,
                },
            ),
        }
        for function, (parameters, annotations) in expected.items():
            with self.subTest(function=function.__name__):
                signature = inspect.signature(function)
                self.assertEqual(
                    tuple(
                        (parameter.name, parameter.kind)
                        for parameter in signature.parameters.values()
                    ),
                    parameters,
                )
                self.assertEqual(
                    tuple(
                        parameter.default
                        for parameter in signature.parameters.values()
                    ),
                    (inspect.Parameter.empty,) * len(parameters),
                )
                self.assertEqual(get_type_hints(function), annotations)

    def test_handle_method_signatures_are_exact(self) -> None:
        expected = {
            StagedFileHandle.write: (
                (
                    (
                        "self",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                    (
                        "chunk",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                ),
                {"chunk": bytes, "return": type(None)},
            ),
            StagedFileHandle.seal: (
                (
                    (
                        "self",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                    ("executable", inspect.Parameter.KEYWORD_ONLY, False),
                ),
                {"executable": bool, "return": type(None)},
            ),
            StagedDirectoryHandle.mkdir: (
                (
                    (
                        "self",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                    (
                        "path",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                ),
                {"path": PortableRelativePath, "return": type(None)},
            ),
            StagedDirectoryHandle.write_file: (
                (
                    (
                        "self",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                    (
                        "path",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                    (
                        "chunks",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                    ("executable", inspect.Parameter.KEYWORD_ONLY, False),
                ),
                {
                    "path": PortableRelativePath,
                    "chunks": Iterable[bytes],
                    "executable": bool,
                    "return": type(None),
                },
            ),
            StagedDirectoryHandle.seal: (
                (
                    (
                        "self",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.empty,
                    ),
                    (
                        "scope",
                        inspect.Parameter.KEYWORD_ONLY,
                        inspect.Parameter.empty,
                    ),
                ),
                {"scope": str, "return": type(None)},
            ),
        }
        for method, (parameters, annotations) in expected.items():
            with self.subTest(method=method.__qualname__):
                signature = inspect.signature(method)
                self.assertEqual(
                    tuple(
                        (
                            parameter.name,
                            parameter.kind,
                            parameter.default,
                        )
                        for parameter in signature.parameters.values()
                    ),
                    parameters,
                )
                self.assertEqual(get_type_hints(method), annotations)

    def test_enum_abi_is_exact(self) -> None:
        expected = {
            PublicationState: {
                "NOT_COMMITTED": "not_committed",
                "COMMITTED_DURABLE": "committed_durable",
                "COMMITTED_DURABILITY_UNCERTAIN": (
                    "committed_durability_uncertain"
                ),
                "COMMIT_OUTCOME_UNCERTAIN": "commit_outcome_uncertain",
            },
            StagingState: {
                "OPEN": "open",
                "SEALED": "sealed",
                "NOT_COMMITTED": "not_committed",
                "PUBLISHED": "published",
                "DISCARDED": "discarded",
                "RETIRED": "retired",
            },
            StagingCleanupState: {
                "NOT_DISCARDED": "not_discarded",
                "DISCARDED_DURABLE": "discarded_durable",
                "DISCARDED_DURABILITY_UNCERTAIN": (
                    "discarded_durability_uncertain"
                ),
                "DISCARD_OUTCOME_UNCERTAIN": "discard_outcome_uncertain",
            },
            DescriptorRetirementObservation: {
                "CLOSED": "closed",
                "ALREADY_ABSENT": "already_absent",
                "FOREIGN_PRESERVED": "foreign_preserved",
                "UNINSPECTABLE": "uninspectable",
                "CLOSE_OUTCOME_UNCERTAIN": "close_outcome_uncertain",
            },
        }
        for enum_type, members in expected.items():
            with self.subTest(enum=enum_type.__name__):
                self.assertEqual(enum_type.__bases__, (str, Enum))
                self.assertEqual(
                    {name: member.value for name, member in enum_type.__members__.items()},
                    members,
                )

    def test_error_hierarchy_is_exact(self) -> None:
        self.assertEqual(StagingLifecycleError.__bases__, (Exception,))
        self.assertEqual(StagingAuthorityError.__bases__, (Exception,))
        self.assertEqual(PublicationError.__bases__, (Exception,))
        for error in (
            PublicationValidationError,
            PublicationCapabilityError,
            PublicationCollisionError,
            StagingAdmissionError,
            safety.PublicationDurabilityError,
            PublicationOutcomeUncertainError,
            PublicationNamespaceConflictError,
            PublicationNamespaceUncertainError,
        ):
            with self.subTest(error=error.__name__):
                self.assertEqual(error.__bases__, (PublicationError,))
        self.assertEqual(StagingCleanupError.__bases__, (Exception,))
        self.assertEqual(DescriptorRetirementError.__bases__, (Exception,))

    def test_public_error_state_and_evidence_matrix_is_enforced(self) -> None:
        destination = portable("Result")
        staging = portable(".rp-stage-v1-" + "0" * 64)
        identity = publication_module._make_entry_identity(
            os.stat(__file__, follow_symlinks=False)
        )
        foreign = publication_module._make_entry_identity(
            os.stat(os.devnull, follow_symlinks=False)
        )
        no_conflict = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.NO_CONFLICT,
            reference=destination,
        )
        conflict = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.COMPLETE_CONFLICT,
            reference=destination,
            aliases=(portable("result"),),
        )
        uninspectable = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.UNINSPECTABLE,
            reference=destination,
        )

        def evidence(
            *,
            source: str,
            target: str,
            namespace: NamespaceEvidence,
            parent_fsync: str,
        ) -> PublicationRecoveryEvidence:
            source_identity = identity if source == "exact" else None
            target_identity = (
                identity
                if target == "exact"
                else foreign if target == "foreign" else None
            )
            return publication_module._make_publication_evidence(
                staging_identity=identity,
                source_observation=source,
                observed_source_identity=source_identity,
                destination_observation=target,
                observed_destination_identity=target_identity,
                namespace_evidence=namespace,
                parent_fsync=parent_fsync,
                native_errno=None,
            )

        definite = evidence(
            source="exact",
            target="absent",
            namespace=no_conflict,
            parent_fsync="succeeded",
        )
        collision = evidence(
            source="exact",
            target="foreign",
            namespace=no_conflict,
            parent_fsync="succeeded",
        )
        committed = evidence(
            source="absent",
            target="exact",
            namespace=no_conflict,
            parent_fsync="succeeded",
        )
        committed_uncertain = evidence(
            source="absent",
            target="exact",
            namespace=no_conflict,
            parent_fsync="failed",
        )
        committed_conflict = evidence(
            source="absent",
            target="exact",
            namespace=conflict,
            parent_fsync="succeeded",
        )
        committed_uninspectable = evidence(
            source="absent",
            target="exact",
            namespace=uninspectable,
            parent_fsync="succeeded",
        )
        uncertain = evidence(
            source="uninspectable",
            target="uninspectable",
            namespace=uninspectable,
            parent_fsync="uncertain",
        )

        self.assertIsInstance(
            PublicationError(
                "definite",
                state=PublicationState.NOT_COMMITTED,
                evidence=definite,
                destination=destination,
            ),
            PublicationError,
        )
        self.assertIsInstance(
            PublicationCollisionError(
                "collision",
                state=PublicationState.NOT_COMMITTED,
                evidence=collision,
                destination=destination,
            ),
            PublicationCollisionError,
        )
        self.assertIsInstance(
            safety.PublicationDurabilityError(
                "precommit sync",
                state=PublicationState.NOT_COMMITTED,
                evidence=definite,
                destination=destination,
            ),
            safety.PublicationDurabilityError,
        )
        self.assertIsInstance(
            safety.PublicationDurabilityError(
                "commit sync",
                state=PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
                evidence=committed_uncertain,
                destination=destination,
            ),
            safety.PublicationDurabilityError,
        )
        self.assertIsInstance(
            PublicationOutcomeUncertainError(
                "uncertain",
                state=PublicationState.COMMIT_OUTCOME_UNCERTAIN,
                evidence=uncertain,
                destination=destination,
            ),
            PublicationOutcomeUncertainError,
        )
        self.assertIsInstance(
            PublicationNamespaceConflictError(
                "alias",
                state=PublicationState.COMMITTED_DURABLE,
                evidence=committed_conflict,
                destination=destination,
            ),
            PublicationNamespaceConflictError,
        )
        self.assertIsInstance(
            PublicationNamespaceUncertainError(
                "namespace",
                state=PublicationState.COMMITTED_DURABLE,
                evidence=committed_uninspectable,
                destination=destination,
            ),
            PublicationNamespaceUncertainError,
        )

        invalid_publication = (
            (
                PublicationError,
                PublicationState.NOT_COMMITTED,
                None,
                destination,
            ),
            (
                PublicationValidationError,
                PublicationState.COMMITTED_DURABLE,
                None,
                None,
            ),
            (
                PublicationCapabilityError,
                PublicationState.COMMITTED_DURABLE,
                None,
                None,
            ),
            (
                PublicationCollisionError,
                PublicationState.COMMITTED_DURABLE,
                collision,
                destination,
            ),
            (
                PublicationCollisionError,
                PublicationState.NOT_COMMITTED,
                committed_conflict,
                destination,
            ),
            (
                safety.PublicationDurabilityError,
                PublicationState.COMMIT_OUTCOME_UNCERTAIN,
                uncertain,
                destination,
            ),
            (
                safety.PublicationDurabilityError,
                PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
                committed,
                destination,
            ),
            (
                PublicationOutcomeUncertainError,
                PublicationState.NOT_COMMITTED,
                uncertain,
                destination,
            ),
            (
                PublicationOutcomeUncertainError,
                PublicationState.COMMIT_OUTCOME_UNCERTAIN,
                committed,
                destination,
            ),
            (
                PublicationNamespaceConflictError,
                PublicationState.NOT_COMMITTED,
                committed_conflict,
                destination,
            ),
            (
                PublicationNamespaceConflictError,
                PublicationState.COMMITTED_DURABLE,
                collision,
                destination,
            ),
            (
                PublicationNamespaceUncertainError,
                PublicationState.COMMITTED_DURABLE,
                committed,
                destination,
            ),
            (
                PublicationNamespaceUncertainError,
                PublicationState.COMMITTED_DURABLE,
                committed_conflict,
                destination,
            ),
            (
                PublicationNamespaceUncertainError,
                PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
                committed_uninspectable,
                destination,
            ),
        )
        for error_type, state, recovery, path in invalid_publication:
            with (
                self.subTest(error=error_type.__name__, state=state.value),
                self.assertRaises(ValueError),
            ):
                error_type(
                    "invalid",
                    state=state,
                    evidence=recovery,
                    destination=path,
                )

        cleanup_namespace = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.NO_CONFLICT,
            reference=staging,
        )
        exact_cleanup = publication_module._make_cleanup_evidence(
            staging_identity=identity,
            root_observation="exact",
            observed_root_identity=identity,
            remaining_expected_entries=0,
            namespace_evidence=cleanup_namespace,
            parent_fsync="not_attempted",
            native_errno=None,
        )
        absent_failed = publication_module._make_cleanup_evidence(
            staging_identity=identity,
            root_observation="absent",
            observed_root_identity=None,
            remaining_expected_entries=None,
            namespace_evidence=cleanup_namespace,
            parent_fsync="failed",
            native_errno=errno.EIO,
        )
        absent_succeeded = publication_module._make_cleanup_evidence(
            staging_identity=identity,
            root_observation="absent",
            observed_root_identity=None,
            remaining_expected_entries=None,
            namespace_evidence=cleanup_namespace,
            parent_fsync="succeeded",
            native_errno=None,
        )
        self.assertIsInstance(
            StagingCleanupError(
                "residue",
                state=StagingCleanupState.NOT_DISCARDED,
                evidence=exact_cleanup,
                staging=staging,
            ),
            StagingCleanupError,
        )
        self.assertIsInstance(
            StagingCleanupError(
                "durability",
                state=(
                    StagingCleanupState.DISCARDED_DURABILITY_UNCERTAIN
                ),
                evidence=absent_failed,
                staging=staging,
            ),
            StagingCleanupError,
        )
        for cleanup_state, cleanup_evidence in (
            (StagingCleanupState.DISCARDED_DURABLE, absent_succeeded),
            (StagingCleanupState.NOT_DISCARDED, absent_succeeded),
            (
                StagingCleanupState.DISCARDED_DURABILITY_UNCERTAIN,
                exact_cleanup,
            ),
        ):
            with (
                self.subTest(cleanup_state=cleanup_state.value),
                self.assertRaises(ValueError),
            ):
                StagingCleanupError(
                    "invalid",
                    state=cleanup_state,
                    evidence=cleanup_evidence,
                    staging=staging,
                )
        self.assertIsInstance(
            StagingCleanupError(
                "preexisting absence cannot be proven as deletion",
                state=StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN,
                evidence=absent_succeeded,
                staging=staging,
            ),
            StagingCleanupError,
        )

        # A nested exact portable path remains valid evidence for the
        # pre-admission validation error that rejects it.
        nested = portable("nested/result")
        validation = PublicationValidationError(
            "nested",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=nested,
        )
        self.assertIs(validation.destination, nested)

    def test_private_value_builders_reject_hostile_or_malformed_inputs(
        self,
    ) -> None:
        class HostileStr(str):
            pass

        class HostileTuple(tuple):
            pass

        destination = portable("Result")
        staging = portable(".rp-stage-v1-" + "0" * 64)
        identity = publication_module._make_entry_identity(
            os.stat(__file__, follow_symlinks=False)
        )
        special = publication_module._make_entry_identity(
            os.stat(os.devnull, follow_symlinks=False)
        )
        namespace = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.NO_CONFLICT,
            reference=destination,
        )
        cleanup_namespace = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.NO_CONFLICT,
            reference=staging,
        )

        with self.assertRaises(TypeError):
            publication_module._make_namespace_evidence(
                publication_module._NamespaceObservation.COMPLETE_CONFLICT,
                reference=destination,
                aliases=HostileTuple((portable("result"),)),
            )
        retirement_identity = publication_module._make_retirement_identity(
            os.stat(__file__, follow_symlinks=False)
        )
        with self.assertRaises(TypeError):
            publication_module._make_retirement_record(
                ordinal=0,
                role=HostileStr("traversal_entry"),
                observation=(
                    DescriptorRetirementObservation.UNINSPECTABLE
                ),
                close_attempted=False,
                admitted_identity=retirement_identity,
                observed_identity=None,
                error_errno=errno.EIO,
            )

        publication_arguments = {
            "staging_identity": identity,
            "source_observation": "absent",
            "observed_source_identity": None,
            "destination_observation": "absent",
            "observed_destination_identity": None,
            "namespace_evidence": namespace,
            "parent_fsync": "succeeded",
            "native_errno": None,
        }
        for replacement in (
            {"staging_identity": object()},
            {"staging_identity": special},
            {"source_observation": HostileStr("absent")},
            {"destination_observation": HostileStr("absent")},
            {"parent_fsync": HostileStr("succeeded")},
        ):
            arguments = dict(publication_arguments)
            arguments.update(replacement)
            with self.subTest(publication=replacement), self.assertRaises(
                (TypeError, ValueError)
            ):
                publication_module._make_publication_evidence(**arguments)

        cleanup_arguments = {
            "staging_identity": identity,
            "root_observation": "exact",
            "observed_root_identity": identity,
            "remaining_expected_entries": 0,
            "namespace_evidence": cleanup_namespace,
            "parent_fsync": "not_attempted",
            "native_errno": None,
        }
        for replacement in (
            {"staging_identity": object()},
            {"staging_identity": special},
            {"root_observation": HostileStr("exact")},
            {"parent_fsync": HostileStr("not_attempted")},
            {
                "root_observation": "malformed",
                "observed_root_identity": object(),
                "remaining_expected_entries": None,
            },
        ):
            arguments = dict(cleanup_arguments)
            arguments.update(replacement)
            with self.subTest(cleanup=replacement), self.assertRaises(
                (TypeError, ValueError)
            ):
                publication_module._make_cleanup_evidence(**arguments)

        for arguments in (
            (
                PublicationState.COMMITTED_DURABLE,
                object(),
                identity,
                namespace,
            ),
            (
                PublicationState.COMMITTED_DURABLE,
                destination,
                object(),
                namespace,
            ),
            (
                PublicationState.COMMITTED_DURABLE,
                destination,
                special,
                namespace,
            ),
            (
                PublicationState.COMMITTED_DURABLE,
                destination,
                identity,
                object(),
            ),
        ):
            with self.subTest(publication_result=arguments), self.assertRaises(
                (TypeError, ValueError)
            ):
                publication_module._make_publication_result(*arguments)
        for arguments in (
            (object(), identity, cleanup_namespace),
            (staging, object(), cleanup_namespace),
            (staging, special, cleanup_namespace),
            (staging, identity, object()),
        ):
            with self.subTest(cleanup_result=arguments), self.assertRaises(
                (TypeError, ValueError)
            ):
                publication_module._make_cleanup_result(*arguments)

    def test_host_identity_fields_use_unsigned_64_bit_spelling(self) -> None:
        observed = os.stat(os.devnull, follow_symlinks=False)
        entry = publication_module._make_entry_identity(observed)
        retirement = publication_module._make_retirement_identity(observed)
        ledger = publication_module._make_ledger_identity(observed)
        self.assertEqual(entry.device, observed.st_dev % 2**64)
        self.assertEqual(entry.inode, observed.st_ino % 2**64)
        self.assertEqual(entry.owner_uid, observed.st_uid % 2**64)
        self.assertEqual(retirement.device, entry.device)
        self.assertEqual(retirement.inode, entry.inode)
        self.assertEqual(retirement.owner_uid, entry.owner_uid)
        self.assertEqual(ledger.group_id, observed.st_gid % 2**64)

    def test_parent_identity_checks_accept_signed_host_spelling(self) -> None:
        with synthetic_publication_root() as root:
            observed = os.stat(root, follow_symlinks=False)
            fields = list(observed)
            fields[1] = -11
            fields[2] = -7
            signed = os.stat_result(fields)
            with (
                patch.object(
                    publication_module,
                    "_fstat_descriptor",
                    return_value=signed,
                ),
                patch.object(
                    publication_module,
                    "_fsync_descriptor",
                ) as synced,
            ):
                self.assertEqual(
                    publication_module._observe_parent_fsync(
                        91,
                        expected_device=(-7) % 2**64,
                        expected_inode=(-11) % 2**64,
                    ),
                    "succeeded",
                )
            synced.assert_called_once_with(91)

            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=portable("signed-parent"),
                ) as staged:
                    state = publication_module._staging_state(staged)
                    assert state is not None
                    original_device = state.parent_device
                    original_inode = state.parent_inode
                    state.parent_device = (-7) % 2**64
                    state.parent_inode = (-11) % 2**64
                    try:
                        with patch.object(
                            publication_module,
                            "_fstat_descriptor",
                            return_value=signed,
                        ):
                            self.assertTrue(
                                publication_module
                                ._parent_descriptor_matches(state, 92)
                            )
                    finally:
                        state.parent_device = original_device
                        state.parent_inode = original_inode
                    cleanup_owned_staging(trusted, staged)

    def test_authorities_and_values_reject_direct_construction(self) -> None:
        for authority in (
            PublicationResult,
            StagingCleanupResult,
            PublicationEntryIdentity,
            NamespaceEvidence,
            PublicationRecoveryEvidence,
            StagingCleanupRecoveryEvidence,
            DescriptorRetirementIdentity,
            DescriptorRetirementRecord,
            DescriptorRetirementEvidence,
            StagedFileHandle,
            StagedDirectoryHandle,
        ):
            with self.subTest(authority=authority.__name__):
                with self.assertRaises(TypeError):
                    authority()
                with self.assertRaises(TypeError):
                    authority(None)
                with self.assertRaises(TypeError):
                    authority(value=None)

    def test_authorities_and_values_reject_subclassing(self) -> None:
        for authority in (
            PublicationResult,
            StagingCleanupResult,
            PublicationEntryIdentity,
            NamespaceEvidence,
            PublicationRecoveryEvidence,
            StagingCleanupRecoveryEvidence,
            DescriptorRetirementIdentity,
            DescriptorRetirementRecord,
            DescriptorRetirementEvidence,
            StagedFileHandle,
            StagedDirectoryHandle,
        ):
            with self.subTest(authority=authority.__name__):
                with self.assertRaises(TypeError):
                    type(f"Hostile{authority.__name__}", (authority,), {})

    def test_live_handles_are_read_only_noncopyable_and_nonserializable(
        self,
    ) -> None:
        cases = (
            ("file", open_exclusive_staged_file),
            ("directory", open_exclusive_staged_directory),
        )
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                for kind, factory in cases:
                    with self.subTest(kind=kind):
                        with factory(
                            trusted,
                            destination=portable(f"{kind}-authority"),
                        ) as staged:
                            with self.assertRaises((AttributeError, TypeError)):
                                staged.state = StagingState.RETIRED
                            with self.assertRaises(TypeError):
                                copy.copy(staged)
                            with self.assertRaises(TypeError):
                                copy.deepcopy(staged)
                            with self.assertRaises(TypeError):
                                pickle.dumps(staged)
                            cleanup_owned_staging(trusted, staged)


class StagingAdmissionTests(unittest.TestCase):
    def test_preyield_expected_identity_mismatch_retires_both_roles(
        self,
    ) -> None:
        for kind in ("file", "directory"):
            with self.subTest(kind=kind):
                destination = portable(f"admission-mismatch-{kind}")
                with synthetic_publication_root() as root:
                    sentinel = root / "sentinel.bin"
                    sentinel.write_bytes(b"sentinel")
                    foreign_identity = (
                        publication_module._make_entry_identity(
                            sentinel.stat(follow_symlinks=False)
                        )
                    )
                    with open_trusted_root(str(root)) as trusted:
                        baseline_descriptor = lowest_available_descriptor()
                        with (
                            publication_module
                            ._OPEN_DESCRIPTOR_IDENTITIES_LOCK
                        ):
                            baseline_registry = dict(
                                publication_module
                                ._OPEN_DESCRIPTOR_IDENTITIES
                            )
                        real_owned = publication_module._owned_descriptor
                        observed: dict[str, int] = {}

                        def mismatch_staging_identity(
                            descriptor: int,
                            role: str | None,
                            *,
                            expected=None,
                        ):
                            if role in {"handle_parent", "handle_staging"}:
                                observed[role] = descriptor
                            if role == "handle_staging" and expected is not None:
                                return real_owned(
                                    descriptor,
                                    role,
                                    expected=foreign_identity,
                                )
                            return real_owned(
                                descriptor,
                                role,
                                expected=expected,
                            )

                        factory = (
                            open_exclusive_staged_file
                            if kind == "file"
                            else open_exclusive_staged_directory
                        )
                        with (
                            patch.object(
                                publication_module,
                                "_owned_descriptor",
                                side_effect=mismatch_staging_identity,
                            ),
                            self.assertRaises(
                                StagingAdmissionError
                            ) as caught,
                        ):
                            factory(
                                trusted,
                                destination=destination,
                            ).__enter__()
                        self.assertIs(
                            caught.exception.state,
                            PublicationState.NOT_COMMITTED,
                        )
                        self.assertTrue(
                            caught.exception.entry_may_remain
                        )
                        self.assertEqual(
                            set(observed),
                            {"handle_parent", "handle_staging"},
                        )
                        for descriptor in observed.values():
                            with self.assertRaises(OSError) as absent:
                                os.fstat(descriptor)
                            self.assertEqual(
                                absent.exception.errno,
                                errno.EBADF,
                            )
                        with (
                            publication_module
                            ._OPEN_DESCRIPTOR_IDENTITIES_LOCK
                        ):
                            self.assertEqual(
                                publication_module
                                ._OPEN_DESCRIPTOR_IDENTITIES,
                                baseline_registry,
                            )
                        self.assertEqual(
                            lowest_available_descriptor(),
                            baseline_descriptor,
                        )
                    residue = root / staging_leaf(destination)
                    self.assertTrue(
                        residue.is_file()
                        if kind == "file"
                        else residue.is_dir()
                    )

    def test_allocator_failure_remains_preyield_admission_error(
        self,
    ) -> None:
        destination = portable("allocator-failure.bin")
        captured_descriptor = -1
        close_failed = False
        real_create = publication_module._create_file_at
        real_close = os.close

        def capture_created_descriptor(*args, **kwargs) -> int:
            nonlocal captured_descriptor
            captured_descriptor = real_create(*args, **kwargs)
            return captured_descriptor

        def fail_created_close(descriptor: int) -> None:
            nonlocal close_failed
            if descriptor == captured_descriptor and not close_failed:
                close_failed = True
                raise OSError(
                    errno.EIO,
                    "synthetic allocator retirement uncertainty",
                )
            real_close(descriptor)

        with synthetic_publication_root() as root:
            try:
                with open_trusted_root(str(root)) as trusted:
                    with (
                        patch.object(
                            publication_module,
                            "_create_file_at",
                            side_effect=capture_created_descriptor,
                        ),
                        patch.object(
                            publication_module.weakref,
                            "finalize",
                            side_effect=RuntimeError(
                                "synthetic finalizer registration failure"
                            ),
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_created_close,
                        ),
                        self.assertRaises(StagingAdmissionError) as caught,
                    ):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ):
                            self.fail(
                                "allocator failure yielded a public handle"
                            )
                self.assertIs(type(caught.exception), StagingAdmissionError)
                self.assertIs(
                    caught.exception.state,
                    PublicationState.NOT_COMMITTED,
                )
                self.assertIs(caught.exception.destination, destination)
                self.assertEqual(
                    caught.exception.staging,
                    portable(staging_leaf(destination)),
                )
                self.assertTrue(caught.exception.entry_may_remain)
                self.assertIsNotNone(caught.exception.evidence)
                retirement = caught.exception.retirement_evidence
                self.assertIsNotNone(retirement)
                assert retirement is not None
                self.assertEqual(
                    tuple(record.role for record in retirement.records),
                    ("handle_staging", "handle_parent"),
                )
                self.assertIs(
                    retirement.records[0].observation,
                    DescriptorRetirementObservation
                    .CLOSE_OUTCOME_UNCERTAIN,
                )
                self.assertIs(
                    retirement.records[1].observation,
                    DescriptorRetirementObservation.CLOSED,
                )
                self.assertTrue(close_failed)
                self.assertTrue(
                    (root / staging_leaf(destination)).is_file()
                )
            finally:
                if captured_descriptor >= 0:
                    real_close(captured_descriptor)

    def test_failed_admission_uses_provisional_handle_roles_and_primary_error(
        self,
    ) -> None:
        destination = portable("result.bin")
        captured: list[tuple[object, ...]] = []
        real_retire = publication_module._retire_owned_descriptor_records
        real_close = os.close
        close_calls: list[int] = []

        def capture_retirement(owned):
            captured.append(tuple(owned))
            return real_retire(owned)

        def first_close_fails(descriptor: int) -> None:
            close_calls.append(descriptor)
            if len(close_calls) == 1:
                raise OSError(errno.EIO, "synthetic provisional close failure")
            real_close(descriptor)

        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with (
                    patch.object(
                        publication_module,
                        "_admit_created_staging",
                        side_effect=RuntimeError(
                            "synthetic stable-admission failure"
                        ),
                    ),
                    patch.object(
                        publication_module,
                        "_retire_owned_descriptor_records",
                        side_effect=capture_retirement,
                    ),
                    patch.object(
                        publication_module.os,
                        "close",
                        side_effect=first_close_fails,
                    ),
                    self.assertRaises(StagingAdmissionError) as caught,
                ):
                    with open_exclusive_staged_file(
                        trusted,
                        destination=destination,
                    ):
                        self.fail("failed admission yielded a public handle")

            self.assertEqual(len(captured), 1)
            owned = captured[0]
            self.assertEqual(
                tuple(item.role for item in owned),
                ("handle_staging", "handle_parent"),
            )
            self.assertEqual(
                len({item.descriptor for item in owned}),
                len(owned),
            )
            self.assertLessEqual(len(owned), 2)
            self.assertIsNotNone(caught.exception.retirement_evidence)
            self.assertEqual(
                tuple(
                    record.role
                    for record in caught.exception.retirement_evidence.records
                ),
                ("handle_staging", "handle_parent"),
            )
            self.assertIs(
                caught.exception.retirement_evidence.records[0].observation,
                DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
            )
            self.assertTrue(caught.exception.entry_may_remain)
            # The uncertain-close test owns cleanup of the injected descriptor;
            # production correctly never retries it.
            real_close(owned[0].descriptor)

    def test_file_factory_is_lazy(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with patch.object(
                    publication_module,
                    "_open_publication_parent",
                    wraps=publication_module._open_publication_parent,
                ) as opened:
                    context = open_exclusive_staged_file(
                        trusted,
                        destination=destination,
                    )
                    opened.assert_not_called()
                    assert_no_entries(self, root)
                    with context as staged:
                        self.assertIs(staged.state, StagingState.OPEN)
                        self.assertEqual(opened.call_count, 1)
                        cleanup = cleanup_owned_staging(trusted, staged)
                        self.assertIs(
                            cleanup.state,
                            StagingCleanupState.DISCARDED_DURABLE,
                        )
                assert_no_entries(self, root)

    def test_directory_factory_is_lazy(self) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                )
                assert_no_entries(self, root)
                with context as staged:
                    self.assertIs(staged.state, StagingState.OPEN)
                    cleanup_owned_staging(trusted, staged)
                assert_no_entries(self, root)

    def test_unentered_context_exit_is_inert_and_permanently_rejected(
        self,
    ) -> None:
        destination = portable("unentered.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                baseline = lowest_available_descriptor()
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "staging context was not entered",
                ):
                    context.__exit__(None, None, None)
                assert_no_entries(self, root)
                self.assertEqual(lowest_available_descriptor(), baseline)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "staging context cannot be entered twice",
                ):
                    context.__enter__()
                assert_no_entries(self, root)
                self.assertEqual(lowest_available_descriptor(), baseline)

    def test_context_rejects_reentry_and_reexit_without_mutation(self) -> None:
        destination = portable("single-use.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                baseline = lowest_available_descriptor()
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                with self.assertRaisesRegex(
                    RuntimeError,
                    "staging context cannot be entered twice",
                ):
                    context.__enter__()
                cleanup_owned_staging(trusted, staged)
                context.__exit__(None, None, None)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "staging context has already exited",
                ):
                    context.__exit__(None, None, None)
                assert_no_entries(self, root)
                self.assertEqual(lowest_available_descriptor(), baseline)

    def test_failed_context_entry_cannot_be_retried_or_exited(self) -> None:
        destination = portable("failed-entry.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                baseline = lowest_available_descriptor()
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                with (
                    patch.object(
                        publication_module,
                        "_require_publication_capabilities",
                        side_effect=PublicationCapabilityError(
                            "synthetic missing capability",
                            state=PublicationState.NOT_COMMITTED,
                            evidence=None,
                            destination=destination,
                        ),
                    ),
                    self.assertRaises(PublicationCapabilityError),
                ):
                    context.__enter__()
                assert_no_entries(self, root)
                self.assertEqual(lowest_available_descriptor(), baseline)
                for operation in ("enter", "exit"):
                    with (
                        self.subTest(operation=operation),
                        self.assertRaises(RuntimeError),
                    ):
                        if operation == "enter":
                            context.__enter__()
                        else:
                            context.__exit__(None, None, None)
                assert_no_entries(self, root)
                self.assertEqual(lowest_available_descriptor(), baseline)

    def test_same_alias_uses_one_exclusive_reservation(self) -> None:
        upper = portable("Foo")
        lower = portable("foo")
        self.assertEqual(staging_leaf(upper), staging_leaf(lower))
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=upper,
                ) as first:
                    first.write(b"first")
                    with self.assertRaises(PublicationCollisionError) as caught:
                        with open_exclusive_staged_file(
                            trusted,
                            destination=lower,
                        ):
                            self.fail("a second cooperative claimant entered")
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertIsNone(caught.exception.evidence)
                    self.assertEqual(caught.exception.destination, lower)
                    self.assertIs(first.state, StagingState.OPEN)
                    cleanup_owned_staging(trusted, first)
                assert_no_entries(self, root)

    def test_failed_postcreation_admission_preserves_residue_without_handle(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                with (
                    patch.object(
                        publication_module,
                        "_admit_created_staging",
                        side_effect=OSError(
                            errno.EIO,
                            "synthetic postcreation admission failure",
                        ),
                    ),
                    self.assertRaises(StagingAdmissionError) as caught,
                ):
                    context.__enter__()
                self.assertIs(
                    caught.exception.state,
                    PublicationState.NOT_COMMITTED,
                )
                self.assertTrue(caught.exception.entry_may_remain)
                self.assertEqual(
                    caught.exception.staging,
                    portable(staging_leaf(destination)),
                )
                self.assertIsNone(caught.exception.evidence)
                self.assertTrue((root / staging_leaf(destination)).is_file())
                self.assertEqual(
                    tuple(path.name for path in root.iterdir()),
                    (staging_leaf(destination),),
                )

    def test_parent_descriptor_replacement_preserves_foreign_descriptor(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            outside = root.parent / "outside-parent"
            outside.mkdir(mode=0o700)
            outside.chmod(0o700)
            foreign_descriptor = -1
            original_admit = publication_module._admit_created_staging

            def replace_parent_before_admission(
                parent_descriptor: int,
                staging_descriptor: int,
                staging: PortableRelativePath,
                **kwargs: object,
            ):
                nonlocal foreign_descriptor
                os.close(parent_descriptor)
                foreign_descriptor = os.open(
                    outside,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                )
                self.assertEqual(foreign_descriptor, parent_descriptor)
                return original_admit(
                    parent_descriptor,
                    staging_descriptor,
                    staging,
                    **kwargs,
                )

            try:
                with open_trusted_root(str(root)) as trusted:
                    with (
                        patch.object(
                            publication_module,
                            "_admit_created_staging",
                            side_effect=replace_parent_before_admission,
                        ),
                        self.assertRaises(StagingAdmissionError) as caught,
                    ):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ):
                            self.fail("replaced parent admitted a handle")
                    self.assertTrue(caught.exception.entry_may_remain)
                self.assertGreaterEqual(foreign_descriptor, 0)
                self.assertTrue(
                    stat.S_ISDIR(os.fstat(foreign_descriptor).st_mode)
                )
                self.assertEqual(tuple(outside.iterdir()), ())
                self.assertEqual(
                    tuple(path.name for path in root.iterdir()),
                    (staging_leaf(destination),),
                )
            finally:
                if foreign_descriptor >= 0:
                    os.close(foreign_descriptor)

    def test_final_preyield_replacement_evidence_is_not_false_exact(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            original_admit = publication_module._admit_created_staging

            def replace_after_provisional(
                parent_descriptor: int,
                staging_descriptor: int,
                staging: PortableRelativePath,
                **kwargs: object,
            ):
                provisional = original_admit(
                    parent_descriptor,
                    staging_descriptor,
                    staging,
                    **kwargs,
                )
                staged_path = root / staging.value
                staged_path.unlink()
                staged_path.write_bytes(b"replacement")
                staged_path.chmod(0o600)
                return provisional

            with open_trusted_root(str(root)) as trusted:
                with (
                    patch.object(
                        publication_module,
                        "_admit_created_staging",
                        side_effect=replace_after_provisional,
                    ),
                    self.assertRaises(StagingAdmissionError) as caught,
                ):
                    with open_exclusive_staged_file(
                        trusted,
                        destination=destination,
                    ):
                        self.fail("replaced staging entry yielded a handle")
                self.assertTrue(caught.exception.entry_may_remain)
                self.assertIsNotNone(caught.exception.evidence)
                evidence = caught.exception.evidence
                assert evidence is not None
                self.assertNotEqual(evidence.source_observation, "exact")
                self.assertIn(
                    evidence.source_observation,
                    {"replaced", "contradictory", "uninspectable"},
                )
                self.assertEqual(
                    (root / staging_leaf(destination)).read_bytes(),
                    b"replacement",
                )

    def test_o_exclusive_registration_failure_preserves_residue_without_handle(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            original_register = publication_module._register_fresh_descriptor
            registrations = 0
            failed_descriptor = -1

            def fail_created_file_registration(
                descriptor: int,
                *,
                role: str,
            ) -> None:
                nonlocal failed_descriptor, registrations
                registrations += 1
                if registrations == 2:
                    failed_descriptor = descriptor
                    raise RuntimeError(
                        "synthetic post-O_EXCL registration failure"
                    )
                original_register(descriptor, role=role)

            try:
                with open_trusted_root(str(root)) as trusted:
                    with (
                        patch.object(
                            publication_module,
                            "_register_fresh_descriptor",
                            side_effect=fail_created_file_registration,
                        ),
                        self.assertRaises(StagingAdmissionError) as caught,
                    ):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ):
                            self.fail("registration failure yielded a handle")
                    self.assertEqual(registrations, 2)
                    self.assertTrue(caught.exception.entry_may_remain)
                    self.assertIsNone(caught.exception.evidence)
                    retirement = caught.exception.retirement_evidence
                    self.assertIsNotNone(retirement)
                    assert retirement is not None
                    self.assertEqual(
                        tuple(record.role for record in retirement.records),
                        ("handle_staging", "handle_parent"),
                    )
                    self.assertIs(
                        retirement.records[0].observation,
                        DescriptorRetirementObservation.UNINSPECTABLE,
                    )
                    self.assertIs(
                        retirement.records[1].observation,
                        DescriptorRetirementObservation.CLOSED,
                    )
                    residue = root / staging_leaf(destination)
                    self.assertTrue(residue.is_file())
                    self.assertEqual(residue.read_bytes(), b"")
                    self.assertEqual(mode(residue), 0o600)
            finally:
                if failed_descriptor >= 0:
                    try:
                        os.close(failed_descriptor)
                    except OSError as exc:
                        if exc.errno != errno.EBADF:
                            raise

    def test_registration_close_anomaly_attaches_to_admission_error(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            original_register = publication_module._register_fresh_descriptor
            real_close = os.close
            registrations = 0
            failed_descriptor = -1
            close_calls: list[int] = []

            def register_then_fail(
                descriptor: int,
                *,
                role: str,
            ) -> None:
                nonlocal failed_descriptor, registrations
                registrations += 1
                original_register(descriptor, role=role)
                if registrations == 2:
                    failed_descriptor = descriptor
                    raise RuntimeError("synthetic admitted registration failure")

            def fail_staging_close(descriptor: int) -> None:
                close_calls.append(descriptor)
                if descriptor == failed_descriptor:
                    raise OSError(
                        errno.EIO,
                        "synthetic admission close uncertainty",
                    )
                real_close(descriptor)

            try:
                with open_trusted_root(str(root)) as trusted:
                    with (
                        patch.object(
                            publication_module,
                            "_register_fresh_descriptor",
                            side_effect=register_then_fail,
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_staging_close,
                        ),
                        self.assertRaises(StagingAdmissionError) as caught,
                    ):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ):
                            self.fail("failed registration yielded a handle")
                self.assertTrue(caught.exception.entry_may_remain)
                retirement = caught.exception.retirement_evidence
                self.assertIsNotNone(retirement)
                assert retirement is not None
                self.assertEqual(len(retirement.records), 2)
                record = retirement.records[0]
                self.assertEqual(record.role, "handle_staging")
                self.assertIs(
                    record.observation,
                    DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
                )
                self.assertEqual(record.error_errno, errno.EIO)
                self.assertEqual(
                    retirement.records[1].role,
                    "handle_parent",
                )
                self.assertIs(
                    retirement.records[1].observation,
                    DescriptorRetirementObservation.CLOSED,
                )
                self.assertEqual(close_calls.count(failed_descriptor), 1)
                os.fstat(failed_descriptor)
                self.assertTrue((root / staging_leaf(destination)).exists())
            finally:
                if failed_descriptor >= 0:
                    real_close(failed_descriptor)
                self.assertEqual(
                    tuple(path.name for path in root.iterdir()),
                    (staging_leaf(destination),),
                )

    def test_admission_roles_and_private_error_provenance_are_exact(
        self,
    ) -> None:
        for failed_role in ("handle_parent", "handle_staging"):
            with self.subTest(failed_role=failed_role):
                destination = portable(f"{failed_role}.bin")
                caller_error = PublicationValidationError(
                    "caller-created exact error",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=destination,
                )
                with synthetic_publication_root() as root:
                    original_register = (
                        publication_module._register_fresh_descriptor
                    )
                    real_close = os.close
                    failed_descriptor = -1
                    close_calls: list[int] = []

                    def register_then_raise_caller_error(
                        descriptor: int,
                        *,
                        role: str,
                    ) -> None:
                        nonlocal failed_descriptor
                        original_register(descriptor, role=role)
                        if role == failed_role:
                            failed_descriptor = descriptor
                            raise caller_error

                    def fail_admission_close(descriptor: int) -> None:
                        close_calls.append(descriptor)
                        if descriptor == failed_descriptor:
                            raise OSError(
                                errno.EIO,
                                "synthetic provisional close uncertainty",
                            )
                        real_close(descriptor)

                    try:
                        with open_trusted_root(str(root)) as trusted:
                            with (
                                patch.object(
                                    publication_module,
                                    "_register_fresh_descriptor",
                                    side_effect=(
                                        register_then_raise_caller_error
                                    ),
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=fail_admission_close,
                                ),
                                self.assertRaises(
                                    StagingAdmissionError
                                ) as caught,
                            ):
                                with open_exclusive_staged_file(
                                    trusted,
                                    destination=destination,
                                ):
                                    self.fail(
                                        "failed provisional admission "
                                        "yielded a handle"
                                    )
                        self.assertIsNot(caught.exception, caller_error)
                        self.assertIsNone(caller_error.retirement_evidence)
                        self.assertEqual(
                            caught.exception.entry_may_remain,
                            failed_role == "handle_staging",
                        )
                        retirement = caught.exception.retirement_evidence
                        self.assertIsNotNone(retirement)
                        assert retirement is not None
                        self.assertEqual(
                            tuple(
                                record.role
                                for record in retirement.records
                            ),
                            (
                                ("handle_parent",)
                                if failed_role == "handle_parent"
                                else ("handle_staging", "handle_parent")
                            ),
                        )
                        self.assertIs(
                            retirement.records[0].observation,
                            DescriptorRetirementObservation
                            .CLOSE_OUTCOME_UNCERTAIN,
                        )
                        if failed_role == "handle_staging":
                            self.assertIs(
                                retirement.records[1].observation,
                                DescriptorRetirementObservation.CLOSED,
                            )
                        self.assertEqual(
                            close_calls.count(failed_descriptor),
                            1,
                        )
                        cause = caught.exception.__cause__
                        seen_caller = False
                        while cause is not None:
                            if cause is caller_error:
                                seen_caller = True
                                break
                            cause = cause.__cause__
                        self.assertTrue(seen_caller)
                        residue = root / staging_leaf(destination)
                        self.assertEqual(
                            residue.exists(),
                            failed_role == "handle_staging",
                        )
                    finally:
                        if failed_descriptor >= 0:
                            real_close(failed_descriptor)

    def test_reserved_staging_prefix_is_rejected_without_mutation(self) -> None:
        for spelling in (
            ".rp-stage-v1-user",
            ".RP-STAGE-V1-user",
        ):
            with self.subTest(spelling=spelling):
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with self.assertRaises(PublicationValidationError):
                            with open_exclusive_staged_file(
                                trusted,
                                destination=portable(spelling),
                            ):
                                self.fail("reserved destination was admitted")
                    assert_no_entries(self, root)

    def test_nested_destination_is_rejected_without_mutation(self) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with self.assertRaises(PublicationValidationError):
                    with open_exclusive_staged_file(
                        trusted,
                        destination=portable("nested/result.bin"),
                    ):
                        self.fail("nested destination was admitted")
            assert_no_entries(self, root)

    def test_parent_permissions_fail_closed_before_staging(self) -> None:
        with synthetic_publication_root() as root:
            root.chmod(0o770)
            try:
                with open_trusted_root(str(root)) as trusted:
                    with self.assertRaises(PublicationValidationError):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=portable("result.bin"),
                        ):
                            self.fail("group-writable parent was admitted")
            finally:
                root.chmod(0o700)
            assert_no_entries(self, root)

    def test_wrong_argument_types_are_rejected(self) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with self.assertRaises(TypeError):
                    with open_exclusive_staged_file(  # type: ignore[arg-type]
                        trusted,
                        destination="result.bin",
                    ):
                        self.fail("string destination was admitted")
                with self.assertRaises(TypeError):
                    with open_exclusive_staged_file(  # type: ignore[arg-type]
                        object(),
                        destination=portable("result.bin"),
                    ):
                        self.fail("foreign authority was admitted")
            assert_no_entries(self, root)

    def test_closed_trusted_root_fails_entry_without_mutation_or_leak(
        self,
    ) -> None:
        factories = (
            open_exclusive_staged_file,
            open_exclusive_staged_directory,
        )
        for index, factory in enumerate(factories):
            with self.subTest(factory=factory.__name__):
                destination = portable(f"closed-root-{index}")
                with synthetic_publication_root() as root:
                    trusted_context = open_trusted_root(str(root))
                    trusted = trusted_context.__enter__()
                    trusted.close()
                    baseline = lowest_available_descriptor()
                    with self.assertRaises(
                        PublicationValidationError
                    ) as caught:
                        with factory(
                            trusted,
                            destination=destination,
                        ):
                            self.fail("closed trusted root admitted staging")
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertIsNone(caught.exception.evidence)
                    self.assertIs(caught.exception.destination, destination)
                    self.assertEqual(
                        lowest_available_descriptor(),
                        baseline,
                    )
                    assert_no_entries(self, root)
                    trusted_context.__exit__(None, None, None)


class FilePublicationTests(unittest.TestCase):
    @staticmethod
    def _terminal_state_snapshot(state) -> tuple[object, ...]:
        return (
            state.lifecycle,
            id(state.trusted_root),
            state.backend,
            state.parent_device,
            state.parent_inode,
            state.destination,
            state.staging,
            state.kind,
            id(state.root_identity),
            id(state.tree),
            state.parent_descriptor,
            state.staging_descriptor,
            state.parent_generation,
            state.staging_generation,
            state.parent_retirement_identity,
            state.staging_retirement_identity,
            state.retirement_batch_pending,
            id(state.error_provenance),
            state.size_bytes,
            state.publication_attempted,
            state.cleanup_attempted,
            state.active_operation,
        )

    @contextmanager
    def _no_terminal_filesystem_access(self):
        targets = (
            (publication_module, "_require_exact_live_root"),
            (publication_module, "_open_publication_parent"),
            (publication_module, "_fstat_descriptor"),
            (publication_module, "_close_descriptor"),
            (publication_module, "_retire_state_descriptors"),
            (publication_module, "_native_no_replace"),
            (publication_module, "_fsync_descriptor"),
            (publication_module, "_safe_parent_fsync"),
            (publication_module, "_collect_namespace_evidence"),
            (publication_module, "_safe_namespace_evidence"),
            (publication_module, "_require_prepublication_namespace"),
            (publication_module, "_scan_staged_tree"),
            (publication_module, "_unlink_at"),
            (publication_module, "_rmdir_at"),
            (publication_module, "_publish_completed"),
            (publication_module, "_cleanup_owned_state"),
            (publication_module.os, "open"),
            (publication_module.os, "close"),
            (publication_module.os, "stat"),
            (publication_module.os, "fstat"),
            (publication_module.os, "fsync"),
            (publication_module.os, "unlink"),
            (publication_module.os, "rmdir"),
        )
        probes = []
        with ExitStack() as stack:
            for owner, name in targets:
                probes.append(
                    stack.enter_context(
                        patch.object(
                            owner,
                            name,
                            side_effect=AssertionError(
                                f"invalid terminal call reached {name}"
                            ),
                        )
                    )
                )
            yield
        for probe in probes:
            probe.assert_not_called()

    def _assert_invalid_terminal_call(
        self,
        trusted,
        staged: StagedFileHandle,
        operation: Literal["publish", "cleanup"],
        expected_state: StagingState,
    ) -> None:
        state = publication_module._staging_state(staged)
        self.assertIsNotNone(state)
        before = self._terminal_state_snapshot(state)
        with self._no_terminal_filesystem_access():
            with self.assertRaises(StagingLifecycleError) as caught:
                if operation == "publish":
                    publish_completed_file(trusted, staged)
                else:
                    cleanup_owned_staging(trusted, staged)
        self.assertIs(type(caught.exception), StagingLifecycleError)
        self.assertIs(caught.exception.state, expected_state)
        self.assertEqual(caught.exception.operation, operation)
        self.assertEqual(self._terminal_state_snapshot(state), before)
        self.assertIs(staged.state, expected_state)

    def _assert_bound_root_rejected_without_io(
        self,
        trusted,
        staged: StagedFileHandle,
        operation: Literal["publish", "cleanup"],
        expected_state: StagingState,
    ) -> None:
        state = publication_module._staging_state(staged)
        self.assertIsNotNone(state)
        before = self._terminal_state_snapshot(state)
        with self._no_terminal_filesystem_access():
            with self.assertRaises(StagingAuthorityError) as caught:
                if operation == "publish":
                    publish_completed_file(trusted, staged)
                else:
                    cleanup_owned_staging(trusted, staged)
        self.assertIs(type(caught.exception), StagingAuthorityError)
        self.assertIs(caught.exception.state, expected_state)
        self.assertEqual(caught.exception.operation, operation)
        self.assertEqual(self._terminal_state_snapshot(state), before)
        self.assertIs(staged.state, expected_state)

    def test_regular_file_publication_is_durable_and_immutable(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"alpha")
                    staged.write(b"-")
                    staged.write(b"omega")
                    staged.seal()
                    self.assertIs(staged.state, StagingState.SEALED)
                    result = publish_completed_file(trusted, staged)
                    self.assertIs(staged.state, StagingState.PUBLISHED)

                self.assertIs(
                    result.state,
                    PublicationState.COMMITTED_DURABLE,
                )
                self.assertEqual(result.destination, destination)
                self.assertEqual(
                    result.namespace_evidence.namespace_observation,
                    "no_conflict",
                )
                self.assertEqual(
                    result.namespace_evidence.conflicting_aliases,
                    (),
                )
                self.assertEqual(
                    result.namespace_evidence.conflicting_alias_count,
                    0,
                )
                self.assertIs(
                    result.namespace_evidence.aliases_complete,
                    True,
                )
                self.assertEqual((root / destination.value).read_bytes(), b"alpha-omega")
                self.assertEqual(mode(root / destination.value), 0o600)
                self.assertFalse((root / staging_leaf(destination)).exists())
                published = (root / destination.value).stat()
                self.assertEqual(result.destination_identity.inode, published.st_ino)
                self.assertEqual(result.destination_identity.device, published.st_dev)

                for value in (
                    result,
                    result.destination_identity,
                    result.namespace_evidence,
                ):
                    with self.subTest(value=type(value).__name__):
                        with self.assertRaises((AttributeError, TypeError)):
                            value.state = None
                        with self.assertRaises(TypeError):
                            copy.copy(value)
                        with self.assertRaises(TypeError):
                            copy.deepcopy(value)
                        with self.assertRaises(TypeError):
                            pickle.dumps(value)

    def test_terminal_root_exact_type_precedes_handle_type_without_io(
        self,
    ) -> None:
        terminal_calls = (
            publish_completed_file,
            publish_completed_directory,
            cleanup_owned_staging,
        )
        for terminal_call in terminal_calls:
            with self.subTest(terminal_call=terminal_call.__name__):
                with (
                    patch.object(
                        publication_module,
                        "_staging_state",
                        side_effect=AssertionError(
                            "handle registry was consulted"
                        ),
                    ) as state_for,
                    self.assertRaises(TypeError) as caught,
                ):
                    terminal_call(object(), object())  # type: ignore[arg-type]
                state_for.assert_not_called()
                self.assertEqual(
                    str(caught.exception),
                    "trusted_root must be exactly TrustedRoot",
                )

    def test_executable_mode_changes_only_owner_execute(self) -> None:
        destination = portable("program")
        with synthetic_publication_root() as root:
            old_umask = os.umask(0o777)
            try:
                with open_trusted_root(str(root)) as trusted:
                    with open_exclusive_staged_file(
                        trusted,
                        destination=destination,
                    ) as staged:
                        staged.write(b"payload")
                        self.assertEqual(
                            mode(root / staging_leaf(destination)),
                            0o600,
                        )
                        staged.seal(executable=True)
                        self.assertEqual(
                            mode(root / staging_leaf(destination)),
                            0o700,
                        )
                        publish_completed_file(trusted, staged)
            finally:
                os.umask(old_umask)
            self.assertEqual(mode(root / destination.value), 0o700)

    def test_write_rejects_non_bytes_and_zero_progress(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    for invalid in (bytearray(b"x"), memoryview(b"x"), "x"):
                        with self.subTest(invalid=type(invalid).__name__):
                            with self.assertRaises(TypeError):
                                staged.write(invalid)  # type: ignore[arg-type]
                            self.assertIs(staged.state, StagingState.OPEN)
                    with patch.object(
                        publication_module,
                        "_write_descriptor",
                        return_value=0,
                    ):
                        with self.assertRaises(
                            PublicationValidationError
                        ) as caught:
                            staged.write(b"x")
                    self.assertIs(caught.exception.destination, destination)
                    self.assertIs(staged.state, StagingState.OPEN)
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"",
                    )
                    with patch.object(
                        publication_module,
                        "_write_descriptor",
                        return_value=2,
                    ):
                        with self.assertRaises(
                            PublicationValidationError
                        ) as caught:
                            staged.write(b"x")
                    self.assertIs(caught.exception.destination, destination)
                    self.assertIs(staged.state, StagingState.OPEN)
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"",
                    )
                    staged.write(b"recovered")
                    staged.seal()
                    publish_completed_file(trusted, staged)
                self.assertEqual(
                    (root / destination.value).read_bytes(),
                    b"recovered",
                )

    def test_partial_writes_and_eintr_are_retried(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    original = publication_module._write_descriptor
                    calls = 0

                    def interrupted_partial_write(
                        descriptor: int,
                        chunk: bytes,
                    ) -> int:
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            raise InterruptedError(errno.EINTR, "interrupted")
                        return original(descriptor, chunk[:2])

                    with patch.object(
                        publication_module,
                        "_write_descriptor",
                        side_effect=interrupted_partial_write,
                    ):
                        staged.write(b"abcdef")
                    staged.seal()
                    publish_completed_file(trusted, staged)
            self.assertEqual((root / destination.value).read_bytes(), b"abcdef")
            self.assertGreaterEqual(calls, 4)

    def test_partial_write_then_error_retires_top_level_handle(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    original_write = publication_module._write_descriptor
                    calls = 0

                    def partial_then_error(
                        descriptor: int,
                        chunk: bytes,
                    ) -> int:
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            return original_write(descriptor, chunk[:3])
                        raise OSError(
                            errno.EIO,
                            "synthetic write failure after progress",
                        )

                    with patch.object(
                        publication_module,
                        "_write_descriptor",
                        side_effect=partial_then_error,
                    ):
                        with self.assertRaises(
                            PublicationValidationError
                        ) as caught:
                            staged.write(b"abcdef")
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"abc",
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    with self.assertRaises(StagingLifecycleError):
                        staged.seal()
                    with self.assertRaises(StagingLifecycleError):
                        publish_completed_file(trusted, staged)
                    self.assertFalse((root / destination.value).exists())

    def test_signed_64_size_bound_fails_before_oversized_write(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"abc")
                    state = publication_module._staging_state(staged)
                    state.size_bytes = publication_module.MAX_SIGNED_64
                    with self.assertRaises(ValueError):
                        staged.write(b"d")
                    state.size_bytes = 3
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"abc",
                    )
                    cleanup_owned_staging(trusted, staged)

    def test_regular_file_has_one_protocol_file_fsync(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    state = publication_module._staging_state(staged)
                    staged_descriptor = state.staging_descriptor
                    original = publication_module._fsync_descriptor
                    calls: list[int] = []

                    def record_fsync(descriptor: int) -> None:
                        calls.append(descriptor)
                        original(descriptor)

                    with patch.object(
                        publication_module,
                        "_fsync_descriptor",
                        side_effect=record_fsync,
                    ):
                        publish_completed_file(trusted, staged)
        self.assertEqual(calls.count(staged_descriptor), 1)

    def test_seal_collision_is_not_committed_and_cleanup_remains_available(
        self,
    ) -> None:
        destination = portable("result.bin")
        sentinel = b"foreign-destination"
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"staged")
                    (root / destination.value).write_bytes(sentinel)
                    with self.assertRaises(PublicationCollisionError) as caught:
                        staged.seal()
                    self.assertIs(staged.state, StagingState.NOT_COMMITTED)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertIsNotNone(caught.exception.evidence)
                    state = publication_module._staging_state(staged)
                    self.assertTrue(state.retirement_batch_pending)
                    self.assertFalse(state.cleanup_attempted)
                    cleanup = cleanup_owned_staging(trusted, staged)
                    self.assertIs(
                        cleanup.state,
                        StagingCleanupState.DISCARDED_DURABLE,
                    )
                    self.assertIs(staged.state, StagingState.DISCARDED)
                    self.assertFalse(state.retirement_batch_pending)
                    self.assertTrue(state.cleanup_attempted)
                self.assertEqual((root / destination.value).read_bytes(), sentinel)
                self.assertEqual(tuple(path.name for path in root.iterdir()), ("result.bin",))

    def test_seal_case_alias_collision_has_bounded_namespace_evidence(
        self,
    ) -> None:
        destination = portable("Result")
        alias = portable("result")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"staged")
                    (root / alias.value).write_bytes(b"alias")
                    with self.assertRaises(PublicationCollisionError) as caught:
                        staged.seal()
                    evidence = caught.exception.evidence
                    self.assertIsNotNone(evidence)
                    assert evidence is not None
                    namespace = evidence.namespace_evidence
                    self.assertEqual(
                        namespace.namespace_observation,
                        "complete_conflict",
                    )
                    self.assertEqual(namespace.conflicting_aliases, (alias,))
                    self.assertEqual(namespace.conflicting_alias_count, 1)
                    self.assertIs(namespace.aliases_complete, True)
                    self.assertIs(staged.state, StagingState.NOT_COMMITTED)
                    cleanup_owned_staging(trusted, staged)
                self.assertEqual((root / alias.value).read_bytes(), b"alias")

    def test_special_destination_is_rejected_without_opening_it(self) -> None:
        destination = portable("result.bin")
        creators = {
            "symlink": lambda path, sentinel: path.symlink_to(sentinel),
            "fifo": lambda path, _sentinel: os.mkfifo(path, 0o600),
        }
        for kind, create in creators.items():
            with self.subTest(kind=kind):
                with synthetic_publication_root() as root:
                    sentinel = root.parent / f"outside-{kind}.bin"
                    sentinel.write_bytes(b"outside-sentinel")
                    target = root / destination.value
                    create(target, sentinel)
                    opened_files: list[str] = []
                    opened_directories: list[str] = []
                    real_file_open = publication_module._open_read_file_at
                    real_directory_open = publication_module._open_directory_at

                    def record_file(
                        name: str,
                        parent: int,
                        *,
                        role: str,
                    ) -> int:
                        opened_files.append(name)
                        return real_file_open(name, parent, role=role)

                    def record_directory(
                        name: str,
                        parent: int,
                        *,
                        role: str,
                    ) -> int:
                        opened_directories.append(name)
                        return real_directory_open(name, parent, role=role)

                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"owned")
                            with (
                                patch.object(
                                    publication_module,
                                    "_open_read_file_at",
                                    side_effect=record_file,
                                ),
                                patch.object(
                                    publication_module,
                                    "_open_directory_at",
                                    side_effect=record_directory,
                                ),
                                self.assertRaises(PublicationCollisionError),
                            ):
                                staged.seal()
                            self.assertNotIn(destination.value, opened_files)
                            self.assertNotIn(
                                destination.value,
                                opened_directories,
                            )
                            cleanup_owned_staging(trusted, staged)
                    self.assertEqual(sentinel.read_bytes(), b"outside-sentinel")
                    if kind == "symlink":
                        self.assertTrue(target.is_symlink())
                    else:
                        self.assertTrue(stat.S_ISFIFO(target.lstat().st_mode))

    def test_prepublication_file_collision_preserves_both_entries_until_cleanup(
        self,
    ) -> None:
        destination = portable("result.bin")
        sentinel = b"foreign"
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"owned")
                    staged.seal()
                    (root / destination.value).write_bytes(sentinel)
                    with self.assertRaises(PublicationCollisionError) as caught:
                        publish_completed_file(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertIs(staged.state, StagingState.NOT_COMMITTED)
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"owned",
                    )
                    self.assertEqual((root / destination.value).read_bytes(), sentinel)
                    cleanup_owned_staging(trusted, staged)
                self.assertEqual((root / destination.value).read_bytes(), sentinel)

    def test_invalid_lifecycle_calls_fail_before_filesystem_access(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    with self.assertRaises(StagingLifecycleError) as caught:
                        staged.write(b"later")
                    self.assertIs(caught.exception.state, StagingState.SEALED)
                    self.assertEqual(caught.exception.operation, "write")
                    publish_completed_file(trusted, staged)
                    with patch.object(
                        publication_module,
                        "_open_publication_parent",
                    ) as opened:
                        for operation in (
                            lambda: publish_completed_file(trusted, staged),
                            lambda: cleanup_owned_staging(trusted, staged),
                        ):
                            with self.assertRaises(StagingLifecycleError):
                                operation()
                        opened.assert_not_called()
                    self.assertIs(staged.state, StagingState.PUBLISHED)

    def test_invalid_terminal_lifecycle_matrix_is_non_io_and_unchanged(
        self,
    ) -> None:
        with self.subTest(state="open", operation="publish"):
            with synthetic_publication_root() as root:
                with open_trusted_root(str(root)) as trusted:
                    context = open_exclusive_staged_file(
                        trusted,
                        destination=portable("open.bin"),
                    )
                    staged = context.__enter__()
                    staged.write(b"payload")
                    self._assert_invalid_terminal_call(
                        trusted,
                        staged,
                        "publish",
                        StagingState.OPEN,
                    )
                    cleanup_owned_staging(trusted, staged)
                    self.assertFalse(context.__exit__(None, None, None))

        with self.subTest(state="not_committed", operation="publish"):
            with synthetic_publication_root() as root:
                destination = portable("not-committed.bin")
                (root / destination.value).write_bytes(b"foreign")
                with open_trusted_root(str(root)) as trusted:
                    context = open_exclusive_staged_file(
                        trusted,
                        destination=destination,
                    )
                    staged = context.__enter__()
                    staged.write(b"payload")
                    with self.assertRaises(PublicationCollisionError):
                        staged.seal()
                    state = publication_module._staging_state(staged)
                    self.assertTrue(state.retirement_batch_pending)
                    self._assert_invalid_terminal_call(
                        trusted,
                        staged,
                        "publish",
                        StagingState.NOT_COMMITTED,
                    )
                    self.assertTrue(state.retirement_batch_pending)
                    self.assertFalse(state.cleanup_attempted)
                    cleanup_owned_staging(trusted, staged)
                    self.assertFalse(context.__exit__(None, None, None))
                self.assertEqual(
                    (root / destination.value).read_bytes(),
                    b"foreign",
                )

        with self.subTest(state="published"):
            with synthetic_publication_root() as root:
                with open_trusted_root(str(root)) as trusted:
                    context = open_exclusive_staged_file(
                        trusted,
                        destination=portable("published.bin"),
                    )
                    staged = context.__enter__()
                    staged.write(b"payload")
                    staged.seal()
                    publish_completed_file(trusted, staged)
                    for operation in ("publish", "cleanup"):
                        with self.subTest(operation=operation):
                            self._assert_invalid_terminal_call(
                                trusted,
                                staged,
                                operation,
                                StagingState.PUBLISHED,
                            )
                    self.assertFalse(context.__exit__(None, None, None))

        with self.subTest(state="discarded"):
            with synthetic_publication_root() as root:
                with open_trusted_root(str(root)) as trusted:
                    context = open_exclusive_staged_file(
                        trusted,
                        destination=portable("discarded.bin"),
                    )
                    staged = context.__enter__()
                    staged.write(b"payload")
                    cleanup_owned_staging(trusted, staged)
                    for operation in ("publish", "cleanup"):
                        with self.subTest(operation=operation):
                            self._assert_invalid_terminal_call(
                                trusted,
                                staged,
                                operation,
                                StagingState.DISCARDED,
                            )
                    self.assertFalse(context.__exit__(None, None, None))

        with self.subTest(state="retired"):
            with synthetic_publication_root() as root:
                with open_trusted_root(str(root)) as trusted:
                    context = open_exclusive_staged_file(
                        trusted,
                        destination=portable("retired.bin"),
                    )
                    staged = context.__enter__()
                    staged.write(b"recovery")
                    self.assertFalse(context.__exit__(None, None, None))
                    for operation in ("publish", "cleanup"):
                        with self.subTest(operation=operation):
                            self._assert_invalid_terminal_call(
                                trusted,
                                staged,
                                operation,
                                StagingState.RETIRED,
                            )
                self.assertEqual(
                    (root / staging_leaf(portable("retired.bin"))).read_bytes(),
                    b"recovery",
                )

    def test_nonmatching_trusted_root_does_not_consume_attempt(self) -> None:
        destination = portable("result.bin")
        with (
            synthetic_publication_root() as root,
            synthetic_publication_root() as other_root,
        ):
            with (
                open_trusted_root(str(root)) as trusted,
                open_trusted_root(str(other_root)) as other,
            ):
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    with self.assertRaises(StagingAuthorityError) as caught:
                        publish_completed_file(other, staged)
                    self.assertIs(caught.exception.state, StagingState.SEALED)
                    self.assertEqual(caught.exception.operation, "publish")
                    self.assertIs(staged.state, StagingState.SEALED)
                    publish_completed_file(trusted, staged)

    def test_bound_root_identity_and_closed_state_precede_lifecycle(
        self,
    ) -> None:
        with self.subTest(authority="wrong"):
            with (
                synthetic_publication_root() as root,
                synthetic_publication_root() as other_root,
            ):
                with (
                    open_trusted_root(str(root)) as trusted,
                    open_trusted_root(str(other_root)) as other,
                ):
                    context = open_exclusive_staged_file(
                        trusted,
                        destination=portable("wrong-root-published.bin"),
                    )
                    staged = context.__enter__()
                    staged.write(b"payload")
                    staged.seal()
                    publish_completed_file(trusted, staged)
                    for operation in ("publish", "cleanup"):
                        with self.subTest(operation=operation):
                            self._assert_bound_root_rejected_without_io(
                                other,
                                staged,
                                operation,
                                StagingState.PUBLISHED,
                            )
                    self.assertFalse(context.__exit__(None, None, None))

        with self.subTest(authority="closed"):
            with synthetic_publication_root() as root:
                with open_trusted_root(str(root)) as trusted:
                    context = open_exclusive_staged_file(
                        trusted,
                        destination=portable("closed-root-published.bin"),
                    )
                    staged = context.__enter__()
                    staged.write(b"payload")
                    staged.seal()
                    publish_completed_file(trusted, staged)
                    self.assertFalse(context.__exit__(None, None, None))
                    trusted.close()
                    for operation in ("publish", "cleanup"):
                        with self.subTest(operation=operation):
                            self._assert_bound_root_rejected_without_io(
                                trusted,
                                staged,
                                operation,
                                StagingState.PUBLISHED,
                            )

    def test_not_committed_cleanup_root_failure_preserves_authorization(
        self,
    ) -> None:
        destination = portable("pending-cleanup-authority.bin")
        with synthetic_publication_root() as root:
            (root / destination.value).write_bytes(b"foreign")
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"payload")
                with self.assertRaises(PublicationCollisionError):
                    staged.seal()
                state = publication_module._staging_state(staged)
                self.assertIsNotNone(state)
                self.assertIs(
                    state.lifecycle,
                    StagingState.NOT_COMMITTED,
                )
                self.assertTrue(state.retirement_batch_pending)
                self.assertFalse(state.cleanup_attempted)
                before = self._terminal_state_snapshot(state)
                later_operations = (
                    "_open_publication_parent",
                    "_retire_state_descriptors",
                    "_cleanup_owned_state",
                    "_fstat_descriptor",
                    "_fsync_descriptor",
                    "_unlink_at",
                    "_rmdir_at",
                )
                with ExitStack() as stack:
                    require_live = stack.enter_context(
                        patch.object(
                            publication_module,
                            "_require_exact_live_root",
                            side_effect=safety.TrustedRootError(
                                "synthetic full-root integrity failure"
                            ),
                        )
                    )
                    blocked = [
                        stack.enter_context(
                            patch.object(
                                publication_module,
                                name,
                                side_effect=AssertionError(
                                    f"authority rejection reached {name}"
                                ),
                            )
                        )
                        for name in later_operations
                    ]
                    with self.assertRaises(
                        StagingAuthorityError
                    ) as caught:
                        cleanup_owned_staging(trusted, staged)
                require_live.assert_called_once_with(trusted)
                for probe in blocked:
                    probe.assert_not_called()
                self.assertIs(
                    type(caught.exception),
                    StagingAuthorityError,
                )
                self.assertIs(
                    caught.exception.state,
                    StagingState.NOT_COMMITTED,
                )
                self.assertEqual(
                    caught.exception.operation,
                    "cleanup",
                )
                self.assertEqual(
                    self._terminal_state_snapshot(state),
                    before,
                )
                self.assertTrue(state.retirement_batch_pending)
                self.assertFalse(state.cleanup_attempted)

                result = cleanup_owned_staging(trusted, staged)
                self.assertIs(
                    result.state,
                    StagingCleanupState.DISCARDED_DURABLE,
                )
                self.assertFalse(context.__exit__(None, None, None))
            self.assertEqual(
                (root / destination.value).read_bytes(),
                b"foreign",
            )

    def test_open_or_sealed_context_abandonment_retires_without_deleting(
        self,
    ) -> None:
        for seal in (False, True):
            with self.subTest(seal=seal):
                destination = portable("result.bin")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"recovery")
                            if seal:
                                staged.seal()
                        self.assertIs(staged.state, StagingState.RETIRED)
                        self.assertEqual(
                            (root / staging_leaf(destination)).read_bytes(),
                            b"recovery",
                        )
                        with self.assertRaises(StagingLifecycleError):
                            cleanup_owned_staging(trusted, staged)
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"recovery",
                    )

    def test_not_committed_context_exit_preserves_one_cleanup_attempt(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"recovery")
                (root / destination.value).write_bytes(b"foreign")
                with self.assertRaises(PublicationCollisionError):
                    staged.seal()
                context.__exit__(None, None, None)
                self.assertIs(staged.state, StagingState.NOT_COMMITTED)
                cleanup_owned_staging(trusted, staged)
                self.assertIs(staged.state, StagingState.DISCARDED)
                with self.assertRaises(StagingLifecycleError):
                    cleanup_owned_staging(trusted, staged)
                self.assertEqual((root / destination.value).read_bytes(), b"foreign")

    def test_file_and_directory_handle_types_are_not_interchangeable(self) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=portable("file-result"),
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    with self.assertRaises(TypeError):
                        publish_completed_directory(  # type: ignore[arg-type]
                            trusted,
                            staged,
                        )
                    self.assertIs(staged.state, StagingState.SEALED)
                    publish_completed_file(trusted, staged)


class DirectoryPublicationTests(unittest.TestCase):
    def test_seal_stat_to_open_replacement_retires_registered_descriptor(
        self,
    ) -> None:
        for replacement_kind in (
            "regular_file",
            "fifo",
            "directory",
        ):
            with self.subTest(replacement_kind=replacement_kind):
                destination = portable(f"seal-race-{replacement_kind}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with (
                            publication_module
                            ._OPEN_DESCRIPTOR_IDENTITIES_LOCK
                        ):
                            baseline_registry = dict(
                                publication_module
                                ._OPEN_DESCRIPTOR_IDENTITIES
                            )
                        baseline_descriptor = lowest_available_descriptor()
                        context = open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        if replacement_kind == "directory":
                            staged.mkdir(portable("child"))
                            staged.write_file(
                                portable("child/payload.bin"),
                                (b"payload",),
                            )
                        else:
                            staged.write_file(
                                portable("payload.bin"),
                                (b"payload",),
                            )
                        staged_root = root / staging_leaf(destination)
                        real_open_file = (
                            publication_module._open_read_file_at
                        )
                        real_open_directory = (
                            publication_module._open_directory_at
                        )
                        opened_descriptor = -1
                        replaced = False

                        def replace_before_file_open(
                            name: str,
                            parent_descriptor: int,
                            *,
                            role: str,
                        ) -> int:
                            nonlocal opened_descriptor, replaced
                            if name == "payload.bin" and not replaced:
                                replaced = True
                                target = staged_root / "payload.bin"
                                target.unlink()
                                if replacement_kind == "fifo":
                                    os.mkfifo(target, 0o600)
                                else:
                                    target.write_bytes(b"payload")
                                    target.chmod(0o600)
                            opened_descriptor = real_open_file(
                                name,
                                parent_descriptor,
                                role=role,
                            )
                            return opened_descriptor

                        def replace_before_directory_open(
                            name: str,
                            parent_descriptor: int,
                            *,
                            role: str,
                        ) -> int:
                            nonlocal opened_descriptor, replaced
                            if (
                                name == "child"
                                and role == "traversal_directory"
                                and not replaced
                            ):
                                replaced = True
                                child = staged_root / "child"
                                (child / "payload.bin").unlink()
                                child.rmdir()
                                child.mkdir(mode=0o700)
                            opened_descriptor = real_open_directory(
                                name,
                                parent_descriptor,
                                role=role,
                            )
                            return opened_descriptor

                        patch_target = (
                            "_open_directory_at"
                            if replacement_kind == "directory"
                            else "_open_read_file_at"
                        )
                        patch_effect = (
                            replace_before_directory_open
                            if replacement_kind == "directory"
                            else replace_before_file_open
                        )
                        with (
                            patch.object(
                                publication_module,
                                patch_target,
                                side_effect=patch_effect,
                            ),
                            self.assertRaises(PublicationValidationError),
                        ):
                            staged.seal(scope="synthetic-race-v1")
                        self.assertTrue(replaced)
                        self.assertIs(
                            staged.state,
                            StagingState.RETIRED,
                        )
                        self.assertFalse(
                            publication_module._staging_state(
                                staged
                            ).retirement_batch_pending
                        )
                        self.assertFalse(
                            context.__exit__(None, None, None)
                        )
                        self.assertGreaterEqual(opened_descriptor, 0)
                        with self.assertRaises(OSError) as absent:
                            os.fstat(opened_descriptor)
                        self.assertEqual(
                            absent.exception.errno,
                            errno.EBADF,
                        )
                        with (
                            publication_module
                            ._OPEN_DESCRIPTOR_IDENTITIES_LOCK
                        ):
                            self.assertEqual(
                                publication_module
                                ._OPEN_DESCRIPTOR_IDENTITIES,
                                baseline_registry,
                            )
                        self.assertEqual(
                            lowest_available_descriptor(),
                            baseline_descriptor,
                        )

    def test_nonempty_directory_publication_uses_exact_modes(self) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            old_umask = os.umask(0o777)
            try:
                with open_trusted_root(str(root)) as trusted:
                    with open_exclusive_staged_directory(
                        trusted,
                        destination=destination,
                    ) as staged:
                        staged.mkdir(portable("nested"))
                        staged.write_file(
                            portable("nested/data.bin"),
                            (b"alpha", b"beta"),
                        )
                        staged.write_file(
                            portable("program"),
                            (b"run",),
                            executable=True,
                        )
                        stage_root = root / staging_leaf(destination)
                        self.assertEqual(mode(stage_root), 0o700)
                        self.assertEqual(mode(stage_root / "nested"), 0o700)
                        self.assertEqual(
                            mode(stage_root / "nested" / "data.bin"),
                            0o600,
                        )
                        self.assertEqual(mode(stage_root / "program"), 0o700)
                        staged.seal(scope="synthetic-publication-v1")
                        result = publish_completed_directory(trusted, staged)
            finally:
                os.umask(old_umask)

            self.assertIs(result.state, PublicationState.COMMITTED_DURABLE)
            self.assertEqual(
                (root / destination.value / "nested" / "data.bin").read_bytes(),
                b"alphabeta",
            )
            self.assertEqual(
                (root / destination.value / "program").read_bytes(),
                b"run",
            )
            self.assertFalse((root / staging_leaf(destination)).exists())

    def test_sealed_directory_binds_h2b_inventory_and_complete_inode_ledger(
        self,
    ) -> None:
        destination = portable("result-dir")
        scope = "synthetic-ledger-inventory-v1"
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.mkdir(portable("nested"))
                    staged.write_file(
                        portable("nested/alpha.bin"),
                        (b"alpha",),
                    )
                    staged.write_file(
                        portable("beta.bin"),
                        (b"beta",),
                        executable=True,
                    )
                    staged.seal(scope=scope)
                    state = publication_module._staging_state(staged)
                    self.assertIsNotNone(state.tree.inventory)
                    inventory = state.tree.inventory
                    assert inventory is not None
                    staged_root = root / staging_leaf(destination)
                    with open_trusted_root(str(staged_root)) as payload_root:
                        independent = scan_regular_file_inventory(
                            payload_root,
                            scope=scope,
                        )
                    self.assertEqual(
                        inventory.canonical_bytes,
                        independent.canonical_bytes,
                    )
                    self.assertEqual(
                        inventory.canonical_inventory_sha256,
                        independent.canonical_inventory_sha256,
                    )
                    self.assertEqual(
                        inventory.tree_digest,
                        independent.tree_digest,
                    )
                    ledger_paths = tuple(
                        path.value for path, _identity in state.tree.entries
                    )
                    self.assertEqual(
                        ledger_paths,
                        ("beta.bin", "nested", "nested/alpha.bin"),
                    )
                    for path, identity in state.tree.entries:
                        materialized = staged_root.joinpath(*path.parts)
                        observed = materialized.stat(follow_symlinks=False)
                        self.assertEqual(identity.public.device, observed.st_dev)
                        self.assertEqual(identity.public.inode, observed.st_ino)
                    cleanup_owned_staging(trusted, staged)

    def test_portable_tree_deeper_than_python_recursion_limit_seals_iteratively(
        self,
    ) -> None:
        destination = portable("deep-result")
        reduced_limit = 80
        components = tuple(f"d{index:03d}" for index in range(96))
        deepest_file = portable("/".join((*components, "payload.bin")))
        self.assertLess(len(deepest_file.value.encode("ascii")), 4096)
        self.assertGreater(len(components), reduced_limit)

        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    for depth in range(1, len(components) + 1):
                        staged.mkdir(
                            portable("/".join(components[:depth]))
                        )
                    staged.write_file(deepest_file, (b"payload",))
                    original_limit = sys.getrecursionlimit()
                    recursion_failure: RecursionError | None = None
                    try:
                        sys.setrecursionlimit(reduced_limit)
                        try:
                            staged.seal(scope="synthetic-deep-tree-v1")
                        except RecursionError as exc:
                            recursion_failure = exc
                    finally:
                        sys.setrecursionlimit(original_limit)
                    if recursion_failure is not None:
                        self.fail(
                            "descriptor traversal depends on Python recursion "
                            f"depth: {recursion_failure}"
                        )
                    self.assertIs(staged.state, StagingState.SEALED)
                    cleanup_owned_staging(trusted, staged)

    def test_write_file_requires_explicit_parent(self) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    with self.assertRaises(PublicationValidationError):
                        staged.write_file(
                            portable("missing/data.bin"),
                            (b"payload",),
                        )
                    self.assertIs(staged.state, StagingState.OPEN)
                    stage_root = root / staging_leaf(destination)
                    self.assertFalse((stage_root / "missing").exists())
                    cleanup_owned_staging(trusted, staged)

    def test_population_roles_are_assigned_before_ledger_admission(
        self,
    ) -> None:
        cases = (
            ("mkdir", "traversal_directory"),
            ("write_file", "traversal_entry"),
        )
        for operation, child_role in cases:
            with self.subTest(operation=operation):
                destination = portable(f"population-role-{operation}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            original_register = (
                                publication_module._register_fresh_descriptor
                            )
                            original_refresh = (
                                publication_module
                                ._refresh_directory_tree_after_addition
                            )
                            acquisitions: list[
                                tuple[int, int, str]
                            ] = []
                            preledger_acquisitions: tuple[
                                tuple[int, int, str], ...
                            ] = ()

                            def record_registration(
                                descriptor: int,
                                *,
                                role: str,
                            ):
                                registration = original_register(
                                    descriptor,
                                    role=role,
                                )
                                acquisitions.append(
                                    (
                                        descriptor,
                                        registration.generation,
                                        role,
                                    )
                                )
                                return registration

                            def observe_ledger_admission(
                                *args: object,
                                **kwargs: object,
                            ):
                                nonlocal preledger_acquisitions
                                preledger_acquisitions = tuple(acquisitions)
                                return original_refresh(*args, **kwargs)

                            with (
                                patch.object(
                                    publication_module,
                                    "_register_fresh_descriptor",
                                    side_effect=record_registration,
                                ),
                                patch.object(
                                    publication_module,
                                    "_refresh_directory_tree_after_addition",
                                    side_effect=observe_ledger_admission,
                                ),
                            ):
                                if operation == "mkdir":
                                    staged.mkdir(portable("child"))
                                else:
                                    staged.write_file(
                                        portable("child.bin"),
                                        (b"payload",),
                                    )

                            self.assertEqual(
                                tuple(
                                    role
                                    for _descriptor, _generation, role
                                    in preledger_acquisitions
                                ),
                                ("traversal_parent", child_role),
                            )
                            self.assertEqual(
                                len(
                                    {
                                        (descriptor, generation)
                                        for descriptor, generation, _role
                                        in preledger_acquisitions
                                    }
                                ),
                                len(preledger_acquisitions),
                            )
                            self.assertFalse(
                                {
                                    "operation_staging",
                                    "operation_parent",
                                    "handle_staging",
                                    "handle_parent",
                                }
                                & {
                                    role
                                    for _descriptor, _generation, role
                                    in preledger_acquisitions
                                }
                            )
                            cleanup_owned_staging(trusted, staged)

    def test_population_role_does_not_admit_or_cleanup_rejected_entry(
        self,
    ) -> None:
        cases = (
            ("mkdir", "child", "traversal_directory"),
            ("write_file", "child.bin", "traversal_entry"),
        )
        for operation, child_name, child_role in cases:
            with self.subTest(operation=operation):
                destination = portable(f"rejected-population-{operation}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            original_register = (
                                publication_module._register_fresh_descriptor
                            )
                            acquisitions: list[
                                tuple[int, int, str]
                            ] = []

                            def record_registration(
                                descriptor: int,
                                *,
                                role: str,
                            ):
                                registration = original_register(
                                    descriptor,
                                    role=role,
                                )
                                acquisitions.append(
                                    (
                                        descriptor,
                                        registration.generation,
                                        role,
                                    )
                                )
                                return registration

                            with (
                                patch.object(
                                    publication_module,
                                    "_register_fresh_descriptor",
                                    side_effect=record_registration,
                                ),
                                patch.object(
                                    publication_module,
                                    "_refresh_directory_tree_after_addition",
                                    side_effect=RuntimeError(
                                        "synthetic pre-ledger rejection"
                                    ),
                                ),
                                self.assertRaises(
                                    PublicationValidationError
                                ),
                            ):
                                if operation == "mkdir":
                                    staged.mkdir(portable(child_name))
                                else:
                                    staged.write_file(
                                        portable(child_name),
                                        (b"payload",),
                                    )

                            self.assertEqual(
                                tuple(
                                    role
                                    for _descriptor, _generation, role
                                    in acquisitions
                                ),
                                ("traversal_parent", child_role),
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            staged_root = root / staging_leaf(destination)
                            created = staged_root / child_name
                            self.assertTrue(created.exists())
                            if operation == "write_file":
                                self.assertEqual(
                                    created.read_bytes(),
                                    b"payload",
                                )
                            with (
                                patch.object(
                                    publication_module,
                                    "_unlink_at",
                                    side_effect=AssertionError(
                                        "retired role authority unlinked residue"
                                    ),
                                ),
                                patch.object(
                                    publication_module,
                                    "_rmdir_at",
                                    side_effect=AssertionError(
                                        "retired role authority removed residue"
                                    ),
                                ),
                                self.assertRaises(StagingLifecycleError),
                            ):
                                cleanup_owned_staging(trusted, staged)
                        self.assertTrue(
                            (
                                root
                                / staging_leaf(destination)
                                / child_name
                            ).exists()
                        )

    def test_directory_paths_and_chunks_require_exact_authorities(self) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    with self.assertRaises(TypeError):
                        staged.mkdir("nested")  # type: ignore[arg-type]
                    staged.mkdir(portable("nested"))
                    with self.assertRaises(TypeError):
                        staged.write_file(
                            portable("nested/data.bin"),
                            (b"first", bytearray(b"unsafe")),  # type: ignore[arg-type]
                        )
                    self.assertIs(staged.state, StagingState.RETIRED)

    def test_directory_item_bound_rejects_before_second_entry_creation(
        self,
    ) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    with patch.object(
                        publication_module,
                        "MAX_CANONICAL_CONTAINER_ITEMS",
                        1,
                    ):
                        staged.mkdir(portable("one"))
                        with self.assertRaises(ValueError):
                            staged.mkdir(portable("two"))
                    stage_root = root / staging_leaf(destination)
                    self.assertEqual(
                        tuple(path.name for path in stage_root.iterdir()),
                        ("one",),
                    )
                    cleanup_owned_staging(trusted, staged)

    def test_directory_file_bound_matches_h2b_canonical_item_budget(
        self,
    ) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    # Inventory body overhead is four items and every file
                    # contributes five canonical items.
                    with patch.object(
                        publication_module,
                        "MAX_CANONICAL_CONTAINER_ITEMS",
                        9,
                    ):
                        staged.write_file(portable("one.bin"), (b"one",))
                        with self.assertRaises(ValueError):
                            staged.write_file(portable("two.bin"), (b"two",))
                    stage_root = root / staging_leaf(destination)
                    self.assertEqual(
                        tuple(path.name for path in stage_root.iterdir()),
                        ("one.bin",),
                    )
                    cleanup_owned_staging(trusted, staged)

    def test_directory_scope_is_exact_and_lifecycle_precedes_validation(
        self,
    ) -> None:
        class HostileScope(str):
            pass

        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write_file(portable("one.bin"), (b"one",))
                    with self.assertRaises(
                        PublicationValidationError
                    ) as caught:
                        staged.seal(scope="unsafe scope")
                    self.assertIs(caught.exception.destination, destination)
                    self.assertIs(staged.state, StagingState.OPEN)
                    with self.assertRaises(TypeError):
                        staged.seal(scope=HostileScope("safe-scope"))
                    self.assertIs(staged.state, StagingState.OPEN)
                    staged.seal(scope="synthetic-scope-v1")
                    with self.assertRaises(StagingLifecycleError):
                        staged.seal(scope=HostileScope("unsafe scope"))
                    cleanup_owned_staging(trusted, staged)

    def test_empty_included_directory_is_rejected_at_seal(self) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.mkdir(portable("empty"))
                    with self.assertRaises(
                        PublicationValidationError
                    ) as caught:
                        staged.seal(scope="synthetic-publication-v1")
                    self.assertIs(
                        type(caught.exception),
                        PublicationValidationError,
                    )
                    self.assertIs(staged.state, StagingState.OPEN)
                    cleanup_owned_staging(trusted, staged)

    def test_empty_staged_root_uses_exact_public_validation_error(
        self,
    ) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    with self.assertRaises(
                        PublicationValidationError
                    ) as caught:
                        staged.seal(scope="synthetic-publication-v1")
                    self.assertIs(
                        type(caught.exception),
                        PublicationValidationError,
                    )
                    self.assertIs(caught.exception.destination, destination)
                    self.assertIs(staged.state, StagingState.OPEN)
                    cleanup_owned_staging(trusted, staged)

    def test_partial_directory_chunk_failure_retires_without_adoption(
        self,
    ) -> None:
        cases = ("iterator-error", "invalid-later-chunk")
        for failure_kind in cases:
            with self.subTest(failure=failure_kind):
                destination = portable("result-dir")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            def chunks():
                                yield b"accepted"
                                if failure_kind == "iterator-error":
                                    raise RuntimeError(
                                        "synthetic chunks iterator failure"
                                    )
                                yield bytearray(b"invalid")

                            expected_error = (
                                RuntimeError
                                if failure_kind == "iterator-error"
                                else TypeError
                            )
                            with self.assertRaises(expected_error):
                                staged.write_file(
                                    portable("partial.bin"),
                                    chunks(),
                                )
                            staged_root = root / staging_leaf(destination)
                            self.assertEqual(
                                (staged_root / "partial.bin").read_bytes(),
                                b"accepted",
                            )
                            self.assertEqual(
                                publication_module._staging_state(
                                    staged
                                ).tree.entries,
                                (),
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            with self.assertRaises(StagingLifecycleError):
                                staged.seal(
                                    scope="synthetic-partial-chunks-v1"
                                )
                            with self.assertRaises(StagingLifecycleError):
                                publish_completed_directory(
                                    trusted,
                                    staged,
                                )

    def test_chunk_generator_extra_file_is_rejected_without_adoption(
        self,
    ) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged_root = root / staging_leaf(destination)

                    def chunks():
                        yield b"owned"
                        extra = staged_root / "extra.bin"
                        extra.write_bytes(b"foreign")
                        extra.chmod(0o600)

                    with self.assertRaises(
                        PublicationValidationError
                    ) as caught:
                        staged.write_file(
                            portable("intended.bin"),
                            chunks(),
                        )
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertEqual(
                        publication_module._staging_state(staged).tree.entries,
                        (),
                    )
                    self.assertEqual(
                        (staged_root / "intended.bin").read_bytes(),
                        b"owned",
                    )
                    self.assertEqual(
                        (staged_root / "extra.bin").read_bytes(),
                        b"foreign",
                    )
                    with self.assertRaises(StagingLifecycleError):
                        staged.seal(scope="synthetic-extra-file-v1")

    def test_nested_addition_rejects_unexplained_ancestor_metadata_change(
        self,
    ) -> None:
        for changed_ancestor in ("root", "higher"):
            with self.subTest(changed_ancestor=changed_ancestor):
                destination = portable("result-dir")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.mkdir(portable("higher"))
                            staged.mkdir(portable("higher/immediate"))
                            state = publication_module._staging_state(staged)
                            before_entries = state.tree.entries
                            staged_root = root / staging_leaf(destination)
                            ancestor = (
                                staged_root
                                if changed_ancestor == "root"
                                else staged_root / "higher"
                            )

                            def chunks():
                                yield b"payload"
                                observed = ancestor.stat(
                                    follow_symlinks=False
                                )
                                os.utime(
                                    ancestor,
                                    ns=(
                                        observed.st_atime_ns,
                                        observed.st_mtime_ns + 10_000_000,
                                    ),
                                    follow_symlinks=False,
                                )

                            with self.assertRaises(
                                PublicationValidationError
                            ):
                                staged.write_file(
                                    portable(
                                        "higher/immediate/new.bin"
                                    ),
                                    chunks(),
                                )
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            self.assertEqual(
                                publication_module._staging_state(
                                    staged
                                ).tree.entries,
                                before_entries,
                            )
                            self.assertEqual(
                                (
                                    staged_root
                                    / "higher"
                                    / "immediate"
                                    / "new.bin"
                                ).read_bytes(),
                                b"payload",
                            )

    def test_population_parent_drift_retires_before_mutation(self) -> None:
        for operation in ("mkdir", "write_file"):
            with self.subTest(operation=operation):
                destination = portable(f"result-{operation}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.mkdir(portable("parent"))
                            staged_root = root / staging_leaf(destination)
                            parent = staged_root / "parent"
                            parent.chmod(0o710)
                            target = (
                                parent / "child"
                                if operation == "mkdir"
                                else parent / "item.bin"
                            )
                            with self.assertRaises(
                                PublicationValidationError
                            ):
                                if operation == "mkdir":
                                    staged.mkdir(
                                        portable("parent/child")
                                    )
                                else:
                                    staged.write_file(
                                        portable("parent/item.bin"),
                                        (b"payload",),
                                    )
                            self.assertFalse(target.exists())
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            with self.assertRaises(StagingLifecycleError):
                                staged.seal(
                                    scope="synthetic-parent-drift-v1"
                                )

    def test_special_staged_entries_are_rejected_before_open(self) -> None:
        destination = portable("result-dir")
        creators = {
            "symlink": lambda path, sentinel: path.symlink_to(sentinel),
            "fifo": lambda path, _sentinel: os.mkfifo(path, 0o600),
        }
        for kind, create in creators.items():
            with self.subTest(kind=kind):
                with synthetic_publication_root() as root:
                    sentinel = root.parent / f"outside-staged-{kind}.bin"
                    sentinel.write_bytes(b"outside-sentinel")
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged_root = root / staging_leaf(destination)
                            special = staged_root / "special"
                            create(special, sentinel)
                            opened_files: list[str] = []
                            opened_directories: list[str] = []
                            real_file_open = publication_module._open_read_file_at
                            real_directory_open = (
                                publication_module._open_directory_at
                            )

                            def record_file(
                                name: str,
                                parent: int,
                                *,
                                role: str,
                            ) -> int:
                                opened_files.append(name)
                                return real_file_open(
                                    name,
                                    parent,
                                    role=role,
                                )

                            def record_directory(
                                name: str,
                                parent: int,
                                *,
                                role: str,
                            ) -> int:
                                opened_directories.append(name)
                                return real_directory_open(
                                    name,
                                    parent,
                                    role=role,
                                )

                            with (
                                patch.object(
                                    publication_module,
                                    "_open_read_file_at",
                                    side_effect=record_file,
                                ),
                                patch.object(
                                    publication_module,
                                    "_open_directory_at",
                                    side_effect=record_directory,
                                ),
                                self.assertRaises(PublicationValidationError),
                            ):
                                staged.seal(scope="synthetic-special-v1")
                            self.assertNotIn("special", opened_files)
                            self.assertNotIn("special", opened_directories)
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            with self.assertRaises(StagingLifecycleError):
                                staged.seal(
                                    scope="synthetic-special-retry-v1"
                                )
                    self.assertEqual(sentinel.read_bytes(), b"outside-sentinel")

    def test_every_forbidden_staged_entry_is_rejected_without_opening_it(
        self,
    ) -> None:
        """Cover every H2c1 special-entry class, including a real AF_UNIX node."""

        for kind in (
            "hardlink",
            "symlink",
            "broken-symlink",
            "fifo",
            "socket",
            "simulated-device",
        ):
            with self.subTest(kind=kind):
                destination = portable(f"tree-{kind}")
                with synthetic_publication_root() as root:
                    sentinel = root.parent / f"outside-{kind}.bin"
                    sentinel.write_bytes(b"outside-sentinel")
                    endpoint: socket.socket | None = None
                    simulated_entry_mode: int | None = None
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged_root = root / staging_leaf(destination)
                            special = staged_root / "special"
                            if kind == "hardlink":
                                os.link(sentinel, special)
                            elif kind == "symlink":
                                special.symlink_to(sentinel)
                            elif kind == "broken-symlink":
                                special.symlink_to(root.parent / "absent")
                            elif kind == "fifo":
                                os.mkfifo(special, 0o600)
                            elif kind == "socket":
                                endpoint = socket.socket(
                                    socket.AF_UNIX,
                                    socket.SOCK_STREAM,
                                )
                                original_directory = os.open(
                                    ".",
                                    os.O_RDONLY | os.O_CLOEXEC,
                                )
                                try:
                                    os.chdir(staged_root)
                                    try:
                                        endpoint.bind("special")
                                    except PermissionError:
                                        endpoint.close()
                                        endpoint = None
                                        if os.environ.get("CI"):
                                            raise
                                        special.write_bytes(
                                            b"simulated-socket"
                                        )
                                        simulated_entry_mode = (
                                            stat.S_IFSOCK | 0o600
                                        )
                                finally:
                                    os.fchdir(original_directory)
                                    os.close(original_directory)
                            else:
                                special.write_bytes(b"device-sentinel")
                                simulated_entry_mode = stat.S_IFCHR | 0o600

                            opened_files: list[str] = []
                            opened_directories: list[str] = []
                            real_file_open = publication_module._open_read_file_at
                            real_directory_open = (
                                publication_module._open_directory_at
                            )
                            real_stat = publication_module._stat_at

                            def record_file(
                                name: str,
                                parent: int,
                                *,
                                role: str,
                            ) -> int:
                                opened_files.append(name)
                                return real_file_open(
                                    name,
                                    parent,
                                    role=role,
                                )

                            def record_directory(
                                name: str,
                                parent: int,
                                *,
                                role: str,
                            ) -> int:
                                opened_directories.append(name)
                                return real_directory_open(
                                    name,
                                    parent,
                                    role=role,
                                )

                            def simulate_device(
                                name: str,
                                parent: int,
                            ) -> os.stat_result:
                                result = real_stat(name, parent)
                                if (
                                    simulated_entry_mode is not None
                                    and name == "special"
                                ):
                                    fields = list(result)
                                    fields[0] = simulated_entry_mode
                                    return os.stat_result(fields)
                                return result

                            try:
                                with (
                                    patch.object(
                                        publication_module,
                                        "_open_read_file_at",
                                        side_effect=record_file,
                                    ),
                                    patch.object(
                                        publication_module,
                                        "_open_directory_at",
                                        side_effect=record_directory,
                                    ),
                                    patch.object(
                                        publication_module,
                                        "_stat_at",
                                        side_effect=simulate_device,
                                    ),
                                    self.assertRaises(
                                        PublicationValidationError
                                    ),
                                ):
                                    staged.seal(
                                        scope="synthetic-special-entry-v1"
                                    )
                            finally:
                                if endpoint is not None:
                                    endpoint.close()

                            self.assertNotIn("special", opened_files)
                            self.assertNotIn("special", opened_directories)
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            with self.assertRaises(StagingLifecycleError):
                                staged.seal(
                                    scope="synthetic-special-retry-v1"
                                )
                    self.assertEqual(
                        sentinel.read_bytes(),
                        b"outside-sentinel",
                    )

    def test_directory_prepublication_collision_preserves_foreign_destination(
        self,
    ) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write_file(portable("owned.bin"), (b"owned",))
                    staged.seal(scope="synthetic-publication-v1")
                    foreign = root / destination.value
                    foreign.mkdir()
                    (foreign / "sentinel.bin").write_bytes(b"foreign")
                    with self.assertRaises(PublicationCollisionError) as caught:
                        publish_completed_directory(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertEqual(
                        (foreign / "sentinel.bin").read_bytes(),
                        b"foreign",
                    )
                    cleanup_owned_staging(trusted, staged)
                self.assertEqual(
                    (foreign / "sentinel.bin").read_bytes(),
                    b"foreign",
                )

    def test_prepublication_ledger_rejects_mutation_and_inode_replacement(
        self,
    ) -> None:
        mutations = {
            "same-size-content": lambda path: path.write_bytes(b"other!!"),
            "same-content-inode": lambda path: (
                path.unlink(),
                path.write_bytes(b"payload"),
            ),
            "mode": lambda path: path.chmod(0o700),
        }
        for kind, mutate in mutations.items():
            with self.subTest(kind=kind):
                destination = portable("result-dir")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write_file(
                                portable("payload.bin"),
                                (b"payload",),
                            )
                            staged.seal(scope="synthetic-ledger-v1")
                            payload = (
                                root
                                / staging_leaf(destination)
                                / "payload.bin"
                            )
                            mutate(payload)
                            with self.assertRaises(
                                PublicationValidationError
                            ) as caught:
                                publish_completed_directory(trusted, staged)
                            self.assertIs(
                                caught.exception.state,
                                PublicationState.NOT_COMMITTED,
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.NOT_COMMITTED,
                            )
                            self.assertFalse(
                                (root / destination.value).exists()
                            )

    def test_cleanup_open_and_sealed_directories(self) -> None:
        for seal in (False, True):
            with self.subTest(seal=seal):
                destination = portable("result-dir")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.mkdir(portable("nested"))
                            staged.write_file(
                                portable("nested/data.bin"),
                                (b"payload",),
                            )
                            if seal:
                                staged.seal(scope="synthetic-publication-v1")
                            result = cleanup_owned_staging(trusted, staged)
                            self.assertIs(
                                result.state,
                                StagingCleanupState.DISCARDED_DURABLE,
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.DISCARDED,
                            )
                    assert_no_entries(self, root)


class NativeBackendTests(unittest.TestCase):
    def test_supported_platform_executes_real_native_file_and_directory_paths(
        self,
    ) -> None:
        self.assertIn(sys.platform, {"darwin", "linux"})
        source = Path(publication_module.__file__).read_text(encoding="utf-8")
        if sys.platform == "darwin":
            self.assertIn("renameatx_np", source)
            self.assertIn("RENAME_EXCL = 4", source)
        else:
            self.assertIn("renameat2", source)
            self.assertIn("RENAME_NOREPLACE = 1", source)

        # These are deliberately unmocked. Both leaf kinds must pass through
        # the selected host's actual libc no-replace backend.
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                real_native = publication_module._native_no_replace
                with patch.object(
                    publication_module,
                    "_native_no_replace",
                    wraps=real_native,
                ) as native:
                    with open_exclusive_staged_file(
                        trusted,
                        destination=portable("file-result"),
                    ) as staged_file:
                        staged_file.write(b"native-file")
                        staged_file.seal()
                        staged_file_stat = (
                            root
                            / staging_leaf(portable("file-result"))
                        ).stat(follow_symlinks=False)
                        file_result = publish_completed_file(
                            trusted,
                            staged_file,
                        )
                    self.assertEqual(native.call_count, 1)
                    with open_exclusive_staged_directory(
                        trusted,
                        destination=portable("directory-result"),
                    ) as staged_directory:
                        staged_directory.write_file(
                            portable("item.bin"),
                            (b"native-directory",),
                        )
                        staged_directory.seal(scope="synthetic-native-v1")
                        staged_directory_stat = (
                            root
                            / staging_leaf(portable("directory-result"))
                        ).stat(follow_symlinks=False)
                        directory_result = publish_completed_directory(
                            trusted,
                            staged_directory,
                        )
                    self.assertEqual(native.call_count, 2)
            self.assertIs(
                file_result.state,
                PublicationState.COMMITTED_DURABLE,
            )
            self.assertIs(
                directory_result.state,
                PublicationState.COMMITTED_DURABLE,
            )
            file_destination = (root / "file-result").stat(
                follow_symlinks=False
            )
            directory_destination = (root / "directory-result").stat(
                follow_symlinks=False
            )
            self.assertEqual(
                (
                    file_result.destination_identity.device,
                    file_result.destination_identity.inode,
                ),
                (
                    staged_file_stat.st_dev % 2**64,
                    staged_file_stat.st_ino % 2**64,
                ),
            )
            self.assertEqual(
                (
                    file_destination.st_dev % 2**64,
                    file_destination.st_ino % 2**64,
                ),
                (
                    staged_file_stat.st_dev % 2**64,
                    staged_file_stat.st_ino % 2**64,
                ),
            )
            self.assertEqual(
                (
                    directory_result.destination_identity.device,
                    directory_result.destination_identity.inode,
                ),
                (
                    staged_directory_stat.st_dev % 2**64,
                    staged_directory_stat.st_ino % 2**64,
                ),
            )
            self.assertEqual(
                (
                    directory_destination.st_dev % 2**64,
                    directory_destination.st_ino % 2**64,
                ),
                (
                    staged_directory_stat.st_dev % 2**64,
                    staged_directory_stat.st_ino % 2**64,
                ),
            )
            self.assertFalse(
                (root / staging_leaf(portable("file-result"))).exists()
            )
            self.assertFalse(
                (
                    root / staging_leaf(portable("directory-result"))
                ).exists()
            )

    def test_real_native_file_collision_preserves_both_inodes(self) -> None:
        destination = portable("file-result")
        sentinel = b"foreign"
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"owned")
                    staged.seal()
                    original_native = publication_module._native_no_replace

                    def collide_then_call_native(
                        backend,
                        parent_descriptor: int,
                        source: str,
                        target: str,
                    ):
                        descriptor = publication_module._create_file_at(
                            target,
                            parent_descriptor,
                            0o600,
                            role="traversal_entry",
                        )
                        try:
                            publication_module._write_all(
                                descriptor,
                                sentinel,
                            )
                        finally:
                            publication_module._close_descriptor(descriptor)
                        return original_native(
                            backend,
                            parent_descriptor,
                            source,
                            target,
                        )

                    with patch.object(
                        publication_module,
                        "_native_no_replace",
                        side_effect=collide_then_call_native,
                    ):
                        with self.assertRaises(
                            PublicationCollisionError
                        ) as caught:
                            publish_completed_file(trusted, staged)
                    self.assertEqual(
                        caught.exception.evidence.native_errno,
                        errno.EEXIST,
                    )
                    self.assertEqual(
                        (root / destination.value).read_bytes(),
                        sentinel,
                    )
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"owned",
                    )
                    cleanup_owned_staging(trusted, staged)
            self.assertEqual((root / destination.value).read_bytes(), sentinel)

    def test_real_native_nonempty_directory_collision_preserves_trees(
        self,
    ) -> None:
        destination = portable("directory-result")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write_file(
                        portable("owned.bin"),
                        (b"owned",),
                    )
                    staged.seal(scope="synthetic-native-collision-v1")
                    original_native = publication_module._native_no_replace

                    def collide_then_call_native(
                        backend,
                        parent_descriptor: int,
                        source: str,
                        target: str,
                    ):
                        publication_module._mkdir_at(
                            target,
                            parent_descriptor,
                            0o700,
                        )
                        directory = publication_module._open_directory_at(
                            target,
                            parent_descriptor,
                            role="traversal_directory",
                        )
                        try:
                            descriptor = publication_module._create_file_at(
                                "foreign.bin",
                                directory,
                                0o600,
                                role="traversal_entry",
                            )
                            try:
                                publication_module._write_all(
                                    descriptor,
                                    b"foreign",
                                )
                            finally:
                                publication_module._close_descriptor(descriptor)
                        finally:
                            publication_module._close_descriptor(directory)
                        return original_native(
                            backend,
                            parent_descriptor,
                            source,
                            target,
                        )

                    with patch.object(
                        publication_module,
                        "_native_no_replace",
                        side_effect=collide_then_call_native,
                    ):
                        with self.assertRaises(
                            PublicationCollisionError
                        ) as caught:
                            publish_completed_directory(trusted, staged)
                    self.assertEqual(
                        caught.exception.evidence.native_errno,
                        errno.EEXIST,
                    )
                    self.assertEqual(
                        (root / destination.value / "foreign.bin").read_bytes(),
                        b"foreign",
                    )
                    self.assertEqual(
                        (
                            root
                            / staging_leaf(destination)
                            / "owned.bin"
                        ).read_bytes(),
                        b"owned",
                    )
                    cleanup_owned_staging(trusted, staged)
            self.assertEqual(
                (root / destination.value / "foreign.bin").read_bytes(),
                b"foreign",
            )

    def test_real_native_collisions_preserve_every_destination_type_without_open(
        self,
    ) -> None:
        """The native backend must preserve, and never open, foreign entries."""

        for kind in (
            "regular",
            "directory",
            "symlink",
            "broken-symlink",
            "fifo",
            "socket",
            "simulated-device",
        ):
            with self.subTest(kind=kind):
                destination = portable(f"native-{kind}")
                with synthetic_publication_root() as root:
                    sentinel = root.parent / f"outside-native-{kind}.bin"
                    sentinel.write_bytes(b"outside-sentinel")
                    endpoint: socket.socket | None = None
                    simulate_special_mode: int | None = None
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"owned")
                            staged.seal()
                            original_native = (
                                publication_module._native_no_replace
                            )
                            real_stat = publication_module._stat_at
                            opened_files: list[str] = []
                            opened_directories: list[str] = []
                            real_file_open = (
                                publication_module._open_read_file_at
                            )
                            real_directory_open = (
                                publication_module._open_directory_at
                            )

                            def record_file(
                                name: str,
                                parent: int,
                                *,
                                role: str,
                            ) -> int:
                                opened_files.append(name)
                                return real_file_open(
                                    name,
                                    parent,
                                    role=role,
                                )

                            def record_directory(
                                name: str,
                                parent: int,
                                *,
                                role: str,
                            ) -> int:
                                opened_directories.append(name)
                                return real_directory_open(
                                    name,
                                    parent,
                                    role=role,
                                )

                            def observe_simulated_device(
                                name: str,
                                parent: int,
                            ) -> os.stat_result:
                                result = real_stat(name, parent)
                                if (
                                    simulate_special_mode is not None
                                    and name == destination.value
                                ):
                                    fields = list(result)
                                    fields[0] = simulate_special_mode
                                    return os.stat_result(fields)
                                return result

                            def collide_then_call_native(
                                backend,
                                parent_descriptor: int,
                                source: str,
                                target: str,
                            ):
                                nonlocal endpoint, simulate_special_mode
                                foreign = root / target
                                if kind in {"regular", "simulated-device"}:
                                    foreign.write_bytes(b"foreign")
                                    if kind == "simulated-device":
                                        simulate_special_mode = (
                                            stat.S_IFCHR | 0o600
                                        )
                                elif kind == "directory":
                                    foreign.mkdir(mode=0o700)
                                    (foreign / "sentinel.bin").write_bytes(
                                        b"foreign"
                                    )
                                elif kind == "symlink":
                                    foreign.symlink_to(sentinel)
                                elif kind == "broken-symlink":
                                    foreign.symlink_to(root.parent / "absent")
                                elif kind == "fifo":
                                    os.mkfifo(foreign, 0o600)
                                else:
                                    endpoint = socket.socket(
                                        socket.AF_UNIX,
                                        socket.SOCK_STREAM,
                                    )
                                    try:
                                        endpoint.bind(str(foreign))
                                    except PermissionError:
                                        endpoint.close()
                                        endpoint = None
                                        if os.environ.get("CI"):
                                            raise
                                        foreign.write_bytes(
                                            b"simulated-socket"
                                        )
                                        simulate_special_mode = (
                                            stat.S_IFSOCK | 0o600
                                        )
                                return original_native(
                                    backend,
                                    parent_descriptor,
                                    source,
                                    target,
                                )

                            try:
                                with (
                                    patch.object(
                                        publication_module,
                                        "_native_no_replace",
                                        side_effect=collide_then_call_native,
                                    ),
                                    patch.object(
                                        publication_module,
                                        "_open_read_file_at",
                                        side_effect=record_file,
                                    ),
                                    patch.object(
                                        publication_module,
                                        "_open_directory_at",
                                        side_effect=record_directory,
                                    ),
                                    patch.object(
                                        publication_module,
                                        "_stat_at",
                                        side_effect=observe_simulated_device,
                                    ),
                                    self.assertRaises(
                                        PublicationCollisionError
                                    ) as caught,
                                ):
                                    publish_completed_file(trusted, staged)
                            finally:
                                if endpoint is not None:
                                    endpoint.close()

                            self.assertIs(
                                caught.exception.state,
                                PublicationState.NOT_COMMITTED,
                            )
                            self.assertNotIn(destination.value, opened_files)
                            self.assertNotIn(
                                destination.value,
                                opened_directories,
                            )
                            foreign = root / destination.value
                            if kind in {"regular", "simulated-device"}:
                                self.assertEqual(
                                    foreign.read_bytes(),
                                    b"foreign",
                                )
                            elif kind == "directory":
                                self.assertEqual(
                                    (foreign / "sentinel.bin").read_bytes(),
                                    b"foreign",
                                )
                            elif kind in {"symlink", "broken-symlink"}:
                                self.assertTrue(foreign.is_symlink())
                            elif kind == "fifo":
                                self.assertTrue(
                                    stat.S_ISFIFO(foreign.lstat().st_mode)
                                )
                            else:
                                if simulate_special_mode is None:
                                    self.assertTrue(
                                        stat.S_ISSOCK(
                                            foreign.lstat().st_mode
                                        )
                                    )
                                else:
                                    self.assertEqual(
                                        foreign.read_bytes(),
                                        b"simulated-socket",
                                    )
                            self.assertEqual(
                                (
                                    root / staging_leaf(destination)
                                ).read_bytes(),
                                b"owned",
                            )
                            cleanup_owned_staging(trusted, staged)
                    self.assertEqual(
                        sentinel.read_bytes(),
                        b"outside-sentinel",
                    )

    def test_missing_native_symbol_fails_before_staging_mutation(self) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with patch.object(
                    publication_module,
                    "_load_native_backend",
                    side_effect=PublicationCapabilityError(
                        "synthetic missing native backend",
                        state=PublicationState.NOT_COMMITTED,
                        evidence=None,
                        destination=None,
                    ),
                ):
                    with self.assertRaises(PublicationCapabilityError):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=portable("result.bin"),
                        ):
                            self.fail("missing backend admitted staging")
                assert_no_entries(self, root)

    def test_native_library_and_abi_setup_fail_before_mutation(self) -> None:
        class RejectSignature:
            @property
            def argtypes(self):
                return None

            @argtypes.setter
            def argtypes(self, value) -> None:
                raise RuntimeError("synthetic ABI assignment failure")

        class SyntheticLibrary:
            pass

        library = SyntheticLibrary()
        symbol = (
            "renameatx_np" if sys.platform == "darwin" else "renameat2"
        )
        setattr(library, symbol, RejectSignature())
        cases = (
            OSError(errno.ENOENT, "synthetic libc load failure"),
            library,
        )
        for index, side_effect in enumerate(cases):
            with self.subTest(case=index):
                destination = portable(f"backend-failure-{index}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        cdll_patch = (
                            patch.object(
                                publication_module.ctypes,
                                "CDLL",
                                side_effect=side_effect,
                            )
                            if isinstance(side_effect, BaseException)
                            else patch.object(
                                publication_module.ctypes,
                                "CDLL",
                                return_value=side_effect,
                            )
                        )
                        with (
                            cdll_patch,
                            patch.object(
                                publication_module,
                                "_open_publication_parent",
                                wraps=(
                                    publication_module
                                    ._open_publication_parent
                                ),
                            ) as opened,
                            self.assertRaises(
                                PublicationCapabilityError
                            ) as caught,
                        ):
                            with open_exclusive_staged_file(
                                trusted,
                                destination=destination,
                            ):
                                self.fail(
                                    "invalid native backend admitted staging"
                                )
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertIsNone(caught.exception.evidence)
                    self.assertIs(caught.exception.destination, destination)
                    opened.assert_not_called()
                    assert_no_entries(self, root)

    def test_missing_nofollow_stat_capability_fails_before_mutation(
        self,
    ) -> None:
        supported = set(os.supports_follow_symlinks)
        supported.discard(os.stat)
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with patch.object(
                    publication_module.os,
                    "supports_follow_symlinks",
                    supported,
                ):
                    with self.assertRaises(
                        PublicationCapabilityError
                    ) as caught:
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ):
                            self.fail(
                                "missing no-follow stat admitted staging"
                            )
                self.assertIs(
                    caught.exception.state,
                    PublicationState.NOT_COMMITTED,
                )
                self.assertIsNone(caught.exception.evidence)
                self.assertIs(caught.exception.destination, destination)
                assert_no_entries(self, root)


class NamespaceEvidenceTests(unittest.TestCase):
    def test_namespace_forms_are_exact_and_immutable(self) -> None:
        forms = {
            publication_module._NamespaceObservation.NOT_ATTEMPTED: (
                "not_attempted",
                (),
                None,
                False,
            ),
            publication_module._NamespaceObservation.NO_CONFLICT: (
                "no_conflict",
                (),
                0,
                True,
            ),
            publication_module._NamespaceObservation.COMPLETE_CONFLICT: (
                "complete_conflict",
                (portable("Foo"),),
                1,
                True,
            ),
            publication_module._NamespaceObservation.UNINSPECTABLE: (
                "uninspectable",
                (portable("Foo"),),
                None,
                False,
            ),
        }
        reference = portable("foo")
        for observation, expected in forms.items():
            aliases = expected[1]
            evidence = publication_module._make_namespace_evidence(
                observation,
                reference=reference,
                aliases=aliases,
            )
            with self.subTest(observation=observation.value):
                self.assertEqual(
                    (
                        evidence.namespace_observation,
                        evidence.conflicting_aliases,
                        evidence.conflicting_alias_count,
                        evidence.aliases_complete,
                    ),
                    expected,
                )
                with self.assertRaises((AttributeError, TypeError)):
                    evidence.aliases_complete = not evidence.aliases_complete
                with self.assertRaises(TypeError):
                    copy.copy(evidence)
                with self.assertRaises(TypeError):
                    pickle.dumps(evidence)

    def test_recovery_evidence_values_are_immutable(self) -> None:
        destination = portable("result")
        staging = portable(".rp-stage-v1-" + "0" * 64)
        identity = publication_module._make_entry_identity(
            os.stat(__file__, follow_symlinks=False)
        )
        destination_namespace = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.NO_CONFLICT,
            reference=destination,
        )
        staging_namespace = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.NO_CONFLICT,
            reference=staging,
        )
        publication_evidence = publication_module._make_publication_evidence(
            staging_identity=identity,
            source_observation="exact",
            observed_source_identity=identity,
            destination_observation="absent",
            observed_destination_identity=None,
            namespace_evidence=destination_namespace,
            parent_fsync="succeeded",
            native_errno=None,
        )
        cleanup_evidence = publication_module._make_cleanup_evidence(
            staging_identity=identity,
            root_observation="exact",
            observed_root_identity=identity,
            remaining_expected_entries=0,
            namespace_evidence=staging_namespace,
            parent_fsync="not_attempted",
            native_errno=None,
        )
        for value, attribute in (
            (publication_evidence, "source_observation"),
            (cleanup_evidence, "root_observation"),
        ):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises((AttributeError, TypeError)):
                    setattr(value, attribute, "uninspectable")
                with self.assertRaises(TypeError):
                    copy.copy(value)
                with self.assertRaises(TypeError):
                    copy.deepcopy(value)
                with self.assertRaises(TypeError):
                    pickle.dumps(value)

    def test_complete_conflicts_are_canonically_sorted(self) -> None:
        reference = portable("result")
        with patch.object(
            publication_module,
            "_list_names",
            return_value=iter(("reSult", "Result", "RESULT")),
        ):
            evidence = publication_module._collect_namespace_evidence(
                -1,
                reference,
            )
        self.assertEqual(evidence.namespace_observation, "complete_conflict")
        self.assertEqual(
            tuple(alias.value for alias in evidence.conflicting_aliases),
            ("RESULT", "Result", "reSult"),
        )
        self.assertEqual(evidence.conflicting_alias_count, 3)
        self.assertIs(evidence.aliases_complete, True)

    def test_interrupted_enumeration_is_bounded_and_uninspectable(self) -> None:
        def names():
            yield "Result"
            raise OSError(errno.EIO, "synthetic enumeration failure")

        with patch.object(
            publication_module,
            "_list_names",
            side_effect=lambda _descriptor: names(),
        ):
            evidence = publication_module._collect_namespace_evidence(
                -1,
                portable("result"),
            )
        self.assertEqual(evidence.namespace_observation, "uninspectable")
        self.assertEqual(
            tuple(alias.value for alias in evidence.conflicting_aliases),
            ("Result",),
        )
        self.assertIsNone(evidence.conflicting_alias_count)
        self.assertIs(evidence.aliases_complete, False)

    def test_alias_overflow_uses_the_frozen_bounded_conflict_form(self) -> None:
        reference_text = "abcdefghijklmnopq"

        def aliases():
            # Seventeen ASCII letters provide 131,072 exact spellings with
            # one lowercase alias key. Skip the exact reference spelling.
            for bitset in range(1, 100_002):
                yield "".join(
                    character.upper() if bitset & (1 << index) else character
                    for index, character in enumerate(reference_text)
                )

        with patch.object(
            publication_module,
            "_list_names",
            side_effect=lambda _descriptor: aliases(),
        ):
            evidence = publication_module._collect_namespace_evidence(
                -1,
                portable(reference_text),
            )
        self.assertEqual(evidence.namespace_observation, "bounded_conflict")
        self.assertEqual(
            len(evidence.conflicting_aliases),
            100_000,
        )
        self.assertIsNone(evidence.conflicting_alias_count)
        self.assertIs(evidence.aliases_complete, False)
        self.assertEqual(
            evidence.conflicting_aliases,
            tuple(
                sorted(
                    evidence.conflicting_aliases,
                    key=lambda item: item.value.encode("ascii"),
                )
            ),
        )


class PublicationOutcomeTests(unittest.TestCase):
    def _sealed_file(
        self,
        root: Path,
        trusted: safety.TrustedRoot,
    ):
        context = open_exclusive_staged_file(
            trusted,
            destination=portable("result.bin"),
        )
        staged = context.__enter__()
        staged.write(b"payload")
        staged.seal()
        identity = publication_module._staging_state(staged).root_identity.public
        return context, staged, identity

    def _foreign_identity(self, root: Path, name: str) -> PublicationEntryIdentity:
        path = root / name
        path.write_bytes(name.encode("ascii"))
        return publication_module._make_entry_identity(
            path.stat(follow_symlinks=False)
        )

    def _patch_publication_observations(
        self,
        observations: tuple[
            tuple[str, PublicationEntryIdentity | None],
            tuple[str, PublicationEntryIdentity | None],
            tuple[str, PublicationEntryIdentity | None],
            tuple[str, PublicationEntryIdentity | None],
        ],
        *,
        native_errno: int | None,
        parent_fsync: str = "succeeded",
        namespace: NamespaceEvidence | None = None,
    ):
        if namespace is None:
            namespace = publication_module._make_namespace_evidence(
                publication_module._NamespaceObservation.NO_CONFLICT
            )
        return (
            patch.object(publication_module, "_flush_staged_file"),
            patch.object(publication_module, "_require_prepublication_namespace"),
            patch.object(
                publication_module,
                "_native_no_replace",
                return_value=(-1 if native_errno else 0, native_errno),
            ),
            patch.object(
                publication_module,
                "_observe_named_entry_once",
                side_effect=observations,
            ),
            patch.object(
                publication_module,
                "_collect_namespace_evidence",
                return_value=namespace,
            ),
            patch.object(
                publication_module,
                "_observe_parent_fsync",
                return_value=parent_fsync,
            ),
        )

    def test_post_native_evidence_builder_failure_preserves_proven_commit(
        self,
    ) -> None:
        class SyntheticEvidenceFailure(Exception):
            pass

        for close_anomaly in (False, True):
            with self.subTest(close_anomaly=close_anomaly):
                destination = portable(
                    f"evidence-builder-{int(close_anomaly)}.bin"
                )
                primary = SyntheticEvidenceFailure(
                    "synthetic strict evidence failure"
                )
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        staged.write(b"payload")
                        staged.seal()
                        state = publication_module._staging_state(staged)
                        assert state is not None
                        owned_descriptor = state.staging_descriptor
                        real_close = os.close

                        def optional_close_failure(number: int) -> None:
                            if (
                                close_anomaly
                                and number == owned_descriptor
                            ):
                                raise OSError(
                                    errno.EIO,
                                    "synthetic terminal close failure",
                                )
                            real_close(number)

                        try:
                            with (
                                patch.object(
                                    publication_module,
                                    "_make_publication_evidence",
                                    side_effect=primary,
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=optional_close_failure,
                                ),
                                self.assertRaises(
                                    PublicationNamespaceUncertainError
                                ) as caught,
                            ):
                                publish_completed_file(trusted, staged)
                            self.assertIs(
                                caught.exception.__cause__,
                                primary,
                            )
                            self.assertIs(
                                caught.exception.state,
                                PublicationState.COMMITTED_DURABLE,
                            )
                            self.assertEqual(
                                caught.exception.evidence
                                .namespace_evidence
                                .namespace_observation,
                                "uninspectable",
                            )
                            self.assertEqual(
                                caught.exception.retirement_evidence
                                is not None,
                                close_anomaly,
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.PUBLISHED,
                            )
                            self.assertFalse(
                                state.retirement_batch_pending
                            )
                            with (
                                patch.object(
                                    publication_module,
                                    "_fstat_descriptor",
                                    side_effect=AssertionError(
                                        "published exit inspected a "
                                        "consumed descriptor"
                                    ),
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=AssertionError(
                                        "published exit retried close"
                                    ),
                                ),
                            ):
                                self.assertFalse(
                                    context.__exit__(None, None, None)
                                )
                        finally:
                            if close_anomaly:
                                real_close(owned_descriptor)
                        self.assertEqual(
                            (root / destination.value).read_bytes(),
                            b"payload",
                        )

    def test_evidence_builder_failure_preserves_capability_classification(
        self,
    ) -> None:
        cases = (
            (errno.ENOSYS, PublicationCapabilityError),
            (errno.EINVAL, PublicationCapabilityError),
            (errno.EIO, PublicationError),
        )
        for native_errno, expected_type in cases:
            with self.subTest(
                native_errno=native_errno,
                expected_type=expected_type.__name__,
            ):
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context, staged, original = self._sealed_file(
                            root,
                            trusted,
                        )
                        primary = RuntimeError(
                            "synthetic strict evidence failure"
                        )
                        observations = (
                            ("exact", original),
                            ("absent", None),
                            ("exact", original),
                            ("absent", None),
                        )
                        patches = self._patch_publication_observations(
                            observations,
                            native_errno=native_errno,
                        )
                        with (
                            patches[0],
                            patches[1],
                            patches[2],
                            patches[3],
                            patches[4],
                            patches[5],
                            patch.object(
                                publication_module,
                                "_make_publication_evidence",
                                side_effect=primary,
                            ),
                            self.assertRaises(expected_type) as caught,
                        ):
                            publish_completed_file(trusted, staged)
                        self.assertIs(
                            type(caught.exception),
                            expected_type,
                        )
                        self.assertIs(
                            caught.exception.__cause__,
                            primary,
                        )
                        self.assertIs(
                            caught.exception.state,
                            PublicationState.NOT_COMMITTED,
                        )
                        self.assertIs(
                            staged.state,
                            StagingState.NOT_COMMITTED,
                        )
                        state = publication_module._staging_state(staged)
                        assert state is not None
                        self.assertFalse(
                            state.retirement_batch_pending
                        )
                        cleanup = cleanup_owned_staging(
                            trusted,
                            staged,
                        )
                        self.assertIs(
                            cleanup.state,
                            StagingCleanupState.DISCARDED_DURABLE,
                        )
                        self.assertFalse(
                            context.__exit__(None, None, None)
                        )

    def test_evidence_builder_failure_preserves_committed_classification(
        self,
    ) -> None:
        cases = (
            (
                PublicationNamespaceConflictError,
                PublicationState.COMMITTED_DURABLE,
                "succeeded",
                publication_module._make_namespace_evidence(
                    publication_module
                    ._NamespaceObservation.COMPLETE_CONFLICT,
                    reference=portable("result.bin"),
                    aliases=(portable("RESULT.BIN"),),
                ),
            ),
            (
                safety.PublicationDurabilityError,
                PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
                "failed",
                publication_module._make_namespace_evidence(
                    publication_module._NamespaceObservation.NO_CONFLICT,
                ),
            ),
        )
        for (
            expected_type,
            expected_state,
            parent_fsync,
            namespace,
        ) in cases:
            with self.subTest(expected_type=expected_type.__name__):
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context, staged, original = self._sealed_file(
                            root,
                            trusted,
                        )
                        primary = RuntimeError(
                            "synthetic strict evidence failure"
                        )
                        observations = (
                            ("absent", None),
                            ("exact", original),
                            ("absent", None),
                            ("exact", original),
                        )
                        patches = self._patch_publication_observations(
                            observations,
                            native_errno=None,
                            parent_fsync=parent_fsync,
                            namespace=namespace,
                        )
                        with (
                            patches[0],
                            patches[1],
                            patches[2],
                            patches[3],
                            patches[4],
                            patches[5],
                            patch.object(
                                publication_module,
                                "_make_publication_evidence",
                                side_effect=primary,
                            ),
                            self.assertRaises(expected_type) as caught,
                        ):
                            publish_completed_file(trusted, staged)
                        self.assertIs(type(caught.exception), expected_type)
                        self.assertIs(
                            caught.exception.__cause__,
                            primary,
                        )
                        self.assertIs(
                            caught.exception.state,
                            expected_state,
                        )
                        self.assertIs(
                            caught.exception.evidence.namespace_evidence,
                            namespace,
                        )
                        self.assertIs(
                            staged.state,
                            StagingState.PUBLISHED,
                        )
                        state = publication_module._staging_state(staged)
                        assert state is not None
                        self.assertFalse(
                            state.retirement_batch_pending
                        )
                        self.assertFalse(
                            context.__exit__(None, None, None)
                        )

    def test_source_exact_destination_absent_is_not_committed_regardless_errno(
        self,
    ) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context, staged, original = self._sealed_file(root, trusted)
                observations = (
                    ("exact", original),
                    ("absent", None),
                    ("exact", original),
                    ("absent", None),
                )
                patches = self._patch_publication_observations(
                    observations,
                    native_errno=errno.EIO,
                )
                with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                    with self.assertRaises(PublicationError) as caught:
                        publish_completed_file(trusted, staged)
                self.assertIs(
                    caught.exception.state,
                    PublicationState.NOT_COMMITTED,
                )
                self.assertIs(staged.state, StagingState.NOT_COMMITTED)
                self.assertEqual(caught.exception.evidence.native_errno, errno.EIO)
                cleanup_owned_staging(trusted, staged)
                context.__exit__(None, None, None)

    def test_destination_exact_proves_commit_despite_native_failure(self) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context, staged, original = self._sealed_file(root, trusted)
                observations = (
                    ("absent", None),
                    ("exact", original),
                    ("absent", None),
                    ("exact", original),
                )
                patches = self._patch_publication_observations(
                    observations,
                    native_errno=errno.EIO,
                )
                with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                    result = publish_completed_file(trusted, staged)
                self.assertIs(
                    result.state,
                    PublicationState.COMMITTED_DURABLE,
                )
                self.assertIs(staged.state, StagingState.PUBLISHED)
                context.__exit__(None, None, None)

    def test_source_exact_stable_foreign_destination_is_collision(self) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context, staged, original = self._sealed_file(root, trusted)
                foreign = self._foreign_identity(root, "foreign.bin")
                observations = (
                    ("exact", original),
                    ("foreign", foreign),
                    ("exact", original),
                    ("foreign", foreign),
                )
                patches = self._patch_publication_observations(
                    observations,
                    native_errno=errno.EEXIST,
                )
                with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                    with self.assertRaises(PublicationCollisionError) as caught:
                        publish_completed_file(trusted, staged)
                self.assertIs(
                    caught.exception.state,
                    PublicationState.NOT_COMMITTED,
                )
                self.assertEqual(
                    caught.exception.evidence.destination_observation,
                    "foreign",
                )
                cleanup_owned_staging(trusted, staged)
                context.__exit__(None, None, None)

    def test_later_source_replacement_does_not_erase_proven_commit(self) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context, staged, original = self._sealed_file(root, trusted)
                first = self._foreign_identity(root, "first.bin")
                second = self._foreign_identity(root, "second.bin")
                observations = (
                    ("foreign", first),
                    ("exact", original),
                    ("foreign", second),
                    ("exact", original),
                )
                patches = self._patch_publication_observations(
                    observations,
                    native_errno=None,
                )
                with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                    with self.assertRaises(
                        PublicationNamespaceUncertainError
                    ) as caught:
                        publish_completed_file(trusted, staged)
                self.assertIs(
                    caught.exception.state,
                    PublicationState.COMMITTED_DURABLE,
                )
                self.assertEqual(
                    caught.exception.evidence.source_observation,
                    "replaced",
                )
                self.assertIs(staged.state, StagingState.PUBLISHED)
                context.__exit__(None, None, None)

    def test_required_errno_matrix_never_substitutes_for_anchored_evidence(
        self,
    ) -> None:
        capability_errnos = tuple(
            dict.fromkeys(
                (
                    errno.ENOSYS,
                    errno.ENOTSUP,
                    getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
                    errno.EINVAL,
                    errno.EXDEV,
                )
            )
        )
        other_errnos = (
            errno.EEXIST,
            errno.ENOTEMPTY,
            errno.EINTR,
            errno.EIO,
        )
        for native_errno in (*capability_errnos, *other_errnos):
            with self.subTest(native_errno=native_errno):
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context, staged, original = self._sealed_file(
                            root,
                            trusted,
                        )
                        observations = (
                            ("exact", original),
                            ("absent", None),
                            ("exact", original),
                            ("absent", None),
                        )
                        patches = self._patch_publication_observations(
                            observations,
                            native_errno=native_errno,
                        )
                        with (
                            patches[0],
                            patches[1],
                            patches[2] as native,
                            patches[3],
                            patches[4],
                            patches[5],
                            self.assertRaises(PublicationError) as caught,
                        ):
                            publish_completed_file(trusted, staged)
                        expected_type = (
                            PublicationCapabilityError
                            if native_errno in capability_errnos
                            else PublicationError
                        )
                        self.assertIs(type(caught.exception), expected_type)
                        self.assertIs(
                            caught.exception.state,
                            PublicationState.NOT_COMMITTED,
                        )
                        self.assertEqual(
                            caught.exception.evidence.native_errno,
                            native_errno,
                        )
                        self.assertEqual(native.call_count, 1)
                        cleanup_owned_staging(trusted, staged)
                        context.__exit__(None, None, None)

        # The same errnos cannot erase a commit proved by the original inode
        # at the destination.
        for native_errno in (*capability_errnos, *other_errnos):
            with self.subTest(
                destination_exact_native_errno=native_errno,
            ):
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context, staged, original = self._sealed_file(
                            root,
                            trusted,
                        )
                        observations = (
                            ("absent", None),
                            ("exact", original),
                            ("absent", None),
                            ("exact", original),
                        )
                        patches = self._patch_publication_observations(
                            observations,
                            native_errno=native_errno,
                        )
                        with (
                            patches[0],
                            patches[1],
                            patches[2] as native,
                            patches[3],
                            patches[4],
                            patches[5],
                        ):
                            result = publish_completed_file(trusted, staged)
                        self.assertIs(
                            result.state,
                            PublicationState.COMMITTED_DURABLE,
                        )
                        self.assertEqual(native.call_count, 1)
                        self.assertIs(
                            staged.state,
                            StagingState.PUBLISHED,
                        )
                        context.__exit__(None, None, None)

    def test_exhaustive_frozen_source_destination_outcome_priority(self) -> None:
        """Exercise every frozen outcome class from the two anchored passes."""

        scenarios = (
            ("commit-source-absent", "absent", "exact", "normal"),
            ("commit-source-foreign", "foreign", "exact", "normal"),
            (
                "commit-source-replaced",
                "replaced",
                "exact",
                "namespace-uncertain",
            ),
            (
                "commit-source-exact",
                "exact",
                "exact",
                "namespace-uncertain",
            ),
            (
                "commit-destination-replaced",
                "absent",
                "destination-replaced-after-exact",
                "namespace-uncertain",
            ),
            ("not-committed", "exact", "absent", "not-committed"),
            ("collision", "exact", "foreign", "collision"),
            (
                "uncertain-both-absent",
                "absent",
                "absent",
                "outcome-uncertain",
            ),
            (
                "uncertain-source-foreign",
                "foreign",
                "absent",
                "outcome-uncertain",
            ),
            (
                "uncertain-destination-replaced",
                "exact",
                "replaced",
                "outcome-uncertain",
            ),
            (
                "uncertain-source-contradictory",
                "contradictory",
                "absent",
                "outcome-uncertain",
            ),
            (
                "uncertain-uninspectable",
                "uninspectable",
                "absent",
                "outcome-uncertain",
            ),
        )
        for label, source_form, destination_form, expected in scenarios:
            with self.subTest(label=label):
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context, staged, original = self._sealed_file(
                            root,
                            trusted,
                        )
                        foreign_one = self._foreign_identity(root, "one.bin")
                        foreign_two = self._foreign_identity(root, "two.bin")
                        exact = ("exact", original)
                        absent = ("absent", None)
                        foreign = ("foreign", foreign_one)
                        uninspectable = ("uninspectable", None)

                        if source_form == "exact":
                            source_pair = (exact, exact)
                        elif source_form == "absent":
                            source_pair = (absent, absent)
                        elif source_form == "foreign":
                            source_pair = (foreign, foreign)
                        elif source_form == "replaced":
                            source_pair = (
                                foreign,
                                ("foreign", foreign_two),
                            )
                        elif source_form == "contradictory":
                            source_pair = (exact, absent)
                        else:
                            source_pair = (
                                uninspectable,
                                uninspectable,
                            )

                        if destination_form == "exact":
                            destination_pair = (exact, exact)
                        elif destination_form == "absent":
                            destination_pair = (absent, absent)
                        elif destination_form == "foreign":
                            destination_pair = (foreign, foreign)
                        elif destination_form == "replaced":
                            destination_pair = (
                                foreign,
                                ("foreign", foreign_two),
                            )
                        else:
                            destination_pair = (exact, foreign)

                        observations = (
                            source_pair[0],
                            destination_pair[0],
                            source_pair[1],
                            destination_pair[1],
                        )
                        patches = self._patch_publication_observations(
                            observations,
                            native_errno=errno.EIO,
                        )
                        managers = (
                            patches[0],
                            patches[1],
                            patches[2],
                            patches[3],
                            patches[4],
                            patches[5],
                        )
                        if expected == "normal":
                            with (
                                managers[0],
                                managers[1],
                                managers[2],
                                managers[3],
                                managers[4],
                                managers[5],
                            ):
                                result = publish_completed_file(
                                    trusted,
                                    staged,
                                )
                            self.assertIs(
                                result.state,
                                PublicationState.COMMITTED_DURABLE,
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.PUBLISHED,
                            )
                        else:
                            error_type: type[PublicationError]
                            expected_state: PublicationState
                            expected_lifecycle: StagingState
                            if expected == "namespace-uncertain":
                                error_type = (
                                    PublicationNamespaceUncertainError
                                )
                                expected_state = (
                                    PublicationState.COMMITTED_DURABLE
                                )
                                expected_lifecycle = StagingState.PUBLISHED
                            elif expected == "collision":
                                error_type = PublicationCollisionError
                                expected_state = (
                                    PublicationState.NOT_COMMITTED
                                )
                                expected_lifecycle = (
                                    StagingState.NOT_COMMITTED
                                )
                            elif expected == "not-committed":
                                error_type = PublicationError
                                expected_state = (
                                    PublicationState.NOT_COMMITTED
                                )
                                expected_lifecycle = (
                                    StagingState.NOT_COMMITTED
                                )
                            else:
                                error_type = (
                                    PublicationOutcomeUncertainError
                                )
                                expected_state = (
                                    PublicationState.COMMIT_OUTCOME_UNCERTAIN
                                )
                                expected_lifecycle = StagingState.RETIRED
                            with (
                                managers[0],
                                managers[1],
                                managers[2],
                                managers[3],
                                managers[4],
                                managers[5],
                                self.assertRaises(error_type) as caught,
                            ):
                                publish_completed_file(trusted, staged)
                            self.assertIs(
                                caught.exception.state,
                                expected_state,
                            )
                            self.assertIs(
                                staged.state,
                                expected_lifecycle,
                            )
                            if expected_lifecycle is StagingState.NOT_COMMITTED:
                                cleanup_owned_staging(trusted, staged)
                        context.__exit__(None, None, None)

    def test_postnative_observation_wrapper_failure_is_typed_and_retires(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    with patch.object(
                        publication_module,
                        "_safe_observe_named_entry_once",
                        side_effect=RuntimeError(
                            "synthetic postnative observation failure"
                        ),
                    ):
                        with self.assertRaises(
                            PublicationOutcomeUncertainError
                        ) as caught:
                            publish_completed_file(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.COMMIT_OUTCOME_UNCERTAIN,
                    )
                    self.assertIsNotNone(caught.exception.evidence)
                    self.assertIs(staged.state, StagingState.RETIRED)
            self.assertEqual((root / destination.value).read_bytes(), b"payload")
            self.assertFalse((root / staging_leaf(destination)).exists())


class PublicationDurabilityTests(unittest.TestCase):
    def test_file_fsync_failure_is_definitely_not_committed(self) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    state = publication_module._staging_state(staged)
                    staged_descriptor = state.staging_descriptor
                    original = publication_module._fsync_descriptor
                    failed = False

                    def fail_file_fsync(descriptor: int) -> None:
                        nonlocal failed
                        if descriptor == staged_descriptor and not failed:
                            failed = True
                            raise OSError(errno.EIO, "synthetic file fsync")
                        original(descriptor)

                    with patch.object(
                        publication_module,
                        "_fsync_descriptor",
                        side_effect=fail_file_fsync,
                    ):
                        with self.assertRaises(
                            safety.PublicationDurabilityError
                        ) as caught:
                            publish_completed_file(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertEqual(
                        caught.exception.evidence.source_observation,
                        "exact",
                    )
                    self.assertEqual(
                        caught.exception.evidence.destination_observation,
                        "absent",
                    )
                    self.assertIs(staged.state, StagingState.NOT_COMMITTED)
                    cleanup_owned_staging(trusted, staged)
            self.assertFalse((root / destination.value).exists())

    def test_directory_file_child_and_root_fsync_fail_before_commit(
        self,
    ) -> None:
        for target_kind in ("regular-file", "child-directory", "staged-root"):
            with self.subTest(target_kind=target_kind):
                destination = portable(f"tree-{target_kind}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.mkdir(portable("nested"))
                            staged.write_file(
                                portable("nested/payload.bin"),
                                (b"payload",),
                            )
                            staged.seal(scope="synthetic-fsync-v1")
                            state = publication_module._staging_state(staged)
                            if target_kind == "staged-root":
                                target_inode = state.tree.root.public.inode
                            else:
                                wanted_type = (
                                    "regular_file"
                                    if target_kind == "regular-file"
                                    else "directory"
                                )
                                target_inode = next(
                                    identity.public.inode
                                    for _path, identity in state.tree.entries
                                    if identity.public.entry_type
                                    == wanted_type
                                )
                            original = publication_module._fsync_descriptor
                            failed = False

                            def fail_selected_fsync(descriptor: int) -> None:
                                nonlocal failed
                                observed = os.fstat(descriptor)
                                if (
                                    observed.st_ino == target_inode
                                    and not failed
                                ):
                                    failed = True
                                    raise OSError(
                                        errno.EIO,
                                        f"synthetic {target_kind} fsync",
                                    )
                                original(descriptor)

                            with (
                                patch.object(
                                    publication_module,
                                    "_fsync_descriptor",
                                    side_effect=fail_selected_fsync,
                                ),
                                patch.object(
                                    publication_module,
                                    "_native_no_replace",
                                ) as native,
                                self.assertRaises(
                                    safety.PublicationDurabilityError
                                ) as caught,
                            ):
                                publish_completed_directory(trusted, staged)
                            self.assertTrue(failed)
                            native.assert_not_called()
                            self.assertIs(
                                caught.exception.state,
                                PublicationState.NOT_COMMITTED,
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.NOT_COMMITTED,
                            )
                            self.assertFalse(
                                (root / destination.value).exists()
                            )
                            cleanup_owned_staging(trusted, staged)

    def test_precommit_evidence_wrapper_failure_is_typed_and_retires(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    with (
                        patch.object(
                            publication_module,
                            "_flush_staged_file",
                            side_effect=OSError(
                                errno.EIO,
                                "synthetic precommit failure",
                            ),
                        ),
                        patch.object(
                            publication_module,
                            "_safe_observe_named_entry",
                            side_effect=RuntimeError(
                                "synthetic evidence observation failure"
                            ),
                        ),
                    ):
                        with self.assertRaises(
                            PublicationValidationError
                        ) as caught:
                            publish_completed_file(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertIsNotNone(caught.exception.evidence)
                    self.assertIs(staged.state, StagingState.RETIRED)
            self.assertFalse((root / destination.value).exists())
            self.assertEqual(
                (root / staging_leaf(destination)).read_bytes(),
                b"payload",
            )

    def test_pre_native_failure_with_stable_collision_keeps_one_cleanup_attempt(
        self,
    ) -> None:
        for collision_kind in ("exact-destination", "case-alias"):
            with self.subTest(collision_kind=collision_kind):
                destination = portable("Result.bin")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"owned")
                            staged.seal()
                            foreign = root / (
                                destination.value
                                if collision_kind == "exact-destination"
                                else destination.value.lower()
                            )
                            foreign.write_bytes(b"foreign")
                            with (
                                patch.object(
                                    publication_module,
                                    "_flush_staged_file",
                                    side_effect=OSError(
                                        errno.EIO,
                                        "synthetic pre-native failure",
                                    ),
                                ),
                                patch.object(
                                    publication_module,
                                    "_native_no_replace",
                                ) as native,
                                self.assertRaises(
                                    PublicationCollisionError
                                ) as caught,
                            ):
                                publish_completed_file(trusted, staged)
                            native.assert_not_called()
                            self.assertIs(
                                caught.exception.state,
                                PublicationState.NOT_COMMITTED,
                            )
                            self.assertIsNotNone(caught.exception.evidence)
                            self.assertIs(
                                staged.state,
                                StagingState.NOT_COMMITTED,
                            )
                            if collision_kind == "exact-destination":
                                self.assertEqual(
                                    caught.exception.evidence
                                    .destination_observation,
                                    "foreign",
                                )
                            else:
                                self.assertEqual(
                                    caught.exception.evidence
                                    .namespace_evidence
                                    .namespace_observation,
                                    "complete_conflict",
                                )
                            result = cleanup_owned_staging(trusted, staged)
                            self.assertIs(
                                result.state,
                                StagingCleanupState.DISCARDED_DURABLE,
                            )
                            self.assertEqual(
                                foreign.read_bytes(),
                                b"foreign",
                            )
                            with patch.object(
                                publication_module,
                                "_open_publication_parent",
                            ) as opened:
                                with self.assertRaises(
                                    StagingLifecycleError
                                ):
                                    cleanup_owned_staging(trusted, staged)
                                opened.assert_not_called()

    def test_proven_commit_parent_fsync_failure_is_durability_uncertain(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    with patch.object(
                        publication_module,
                        "_observe_parent_fsync",
                        return_value="failed",
                    ):
                        with self.assertRaises(
                            safety.PublicationDurabilityError
                        ) as caught:
                            publish_completed_file(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
                    )
                    self.assertEqual(
                        caught.exception.evidence.parent_fsync,
                        "failed",
                    )
                    self.assertIs(staged.state, StagingState.PUBLISHED)
            self.assertEqual((root / destination.value).read_bytes(), b"payload")
            self.assertFalse((root / staging_leaf(destination)).exists())

    def test_postcommit_uninspectable_namespace_keeps_commit_state(self) -> None:
        destination = portable("result.bin")
        uninspectable = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.UNINSPECTABLE
        )
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    with (
                        patch.object(
                            publication_module,
                            "_require_prepublication_namespace",
                        ),
                        patch.object(
                            publication_module,
                            "_collect_namespace_evidence",
                            return_value=uninspectable,
                        ),
                    ):
                        with self.assertRaises(
                            PublicationNamespaceUncertainError
                        ) as caught:
                            publish_completed_file(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.COMMITTED_DURABLE,
                    )
                    self.assertEqual(
                        caught.exception.evidence.namespace_evidence,
                        uninspectable,
                    )
                    self.assertIs(staged.state, StagingState.PUBLISHED)
            self.assertEqual((root / destination.value).read_bytes(), b"payload")

    def test_postcommit_namespace_conflict_preserves_proven_commit(self) -> None:
        destination = portable("result.bin")
        alias = portable("Result.BIN")
        conflict = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.COMPLETE_CONFLICT,
            reference=destination,
            aliases=(alias,),
        )
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    with (
                        patch.object(
                            publication_module,
                            "_require_prepublication_namespace",
                        ),
                        patch.object(
                            publication_module,
                            "_collect_namespace_evidence",
                            return_value=conflict,
                        ),
                    ):
                        with self.assertRaises(
                            PublicationNamespaceConflictError
                        ) as caught:
                            publish_completed_file(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.COMMITTED_DURABLE,
                    )
                    self.assertEqual(
                        (
                            caught.exception.evidence.namespace_evidence
                            .conflicting_aliases
                        ),
                        (alias,),
                    )
                    self.assertEqual(
                        caught.exception.evidence.namespace_evidence,
                        conflict,
                    )
                    self.assertIs(staged.state, StagingState.PUBLISHED)
            self.assertEqual((root / destination.value).read_bytes(), b"payload")
            self.assertFalse((root / staging_leaf(destination)).exists())


class CleanupFailureTests(unittest.TestCase):
    def test_preexisting_missing_staging_root_is_outcome_uncertain(
        self,
    ) -> None:
        destination = portable("missing-before-cleanup")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    (root / staging_leaf(destination)).unlink()
                    with self.assertRaises(StagingCleanupError) as caught:
                        cleanup_owned_staging(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN,
                    )
                    self.assertEqual(
                        caught.exception.evidence.root_observation,
                        "absent",
                    )
                    self.assertIs(
                        staged.state,
                        StagingState.RETIRED,
                    )

    def test_cleanup_accepts_open_sealed_and_not_committed_for_both_kinds(
        self,
    ) -> None:
        for kind in ("file", "directory"):
            for initial_state in ("open", "sealed", "not-committed"):
                with self.subTest(kind=kind, initial_state=initial_state):
                    destination = portable(f"{kind}-{initial_state}")
                    with synthetic_publication_root() as root:
                        with open_trusted_root(str(root)) as trusted:
                            factory = (
                                open_exclusive_staged_file
                                if kind == "file"
                                else open_exclusive_staged_directory
                            )
                            with factory(
                                trusted,
                                destination=destination,
                            ) as staged:
                                if kind == "file":
                                    staged.write(b"payload")
                                else:
                                    staged.write_file(
                                        portable("payload.bin"),
                                        (b"payload",),
                                    )
                                if initial_state == "sealed":
                                    if kind == "file":
                                        staged.seal()
                                    else:
                                        staged.seal(
                                            scope="synthetic-cleanup-v1"
                                        )
                                elif initial_state == "not-committed":
                                    foreign = root / destination.value
                                    if kind == "file":
                                        foreign.write_bytes(b"foreign")
                                    else:
                                        foreign.mkdir(mode=0o700)
                                        (
                                            foreign / "sentinel.bin"
                                        ).write_bytes(b"foreign")
                                    with self.assertRaises(
                                        PublicationCollisionError
                                    ):
                                        if kind == "file":
                                            staged.seal()
                                        else:
                                            staged.seal(
                                                scope="synthetic-cleanup-v1"
                                            )
                                    self.assertIs(
                                        staged.state,
                                        StagingState.NOT_COMMITTED,
                                    )
                                result = cleanup_owned_staging(
                                    trusted,
                                    staged,
                                )
                                self.assertIs(
                                    result.state,
                                    StagingCleanupState.DISCARDED_DURABLE,
                                )
                                for value, attribute in (
                                    (result, "state"),
                                    (
                                        result.discarded_identity,
                                        "inode",
                                    ),
                                    (
                                        result.namespace_evidence,
                                        "aliases_complete",
                                    ),
                                ):
                                    with self.subTest(
                                        value=type(value).__name__,
                                    ):
                                        with self.assertRaises(
                                            (AttributeError, TypeError)
                                        ):
                                            setattr(value, attribute, None)
                                        with self.assertRaises(TypeError):
                                            copy.copy(value)
                                        with self.assertRaises(TypeError):
                                            copy.deepcopy(value)
                                        with self.assertRaises(TypeError):
                                            pickle.dumps(value)
                                self.assertIs(
                                    staged.state,
                                    StagingState.DISCARDED,
                                )
                            self.assertFalse(
                                (root / staging_leaf(destination)).exists()
                            )
                            if initial_state == "not-committed":
                                foreign = root / destination.value
                                if kind == "file":
                                    self.assertEqual(
                                        foreign.read_bytes(),
                                        b"foreign",
                                    )
                                else:
                                    self.assertEqual(
                                        (
                                            foreign / "sentinel.bin"
                                        ).read_bytes(),
                                        b"foreign",
                                    )
                            else:
                                self.assertFalse(
                                    (root / destination.value).exists()
                                )

    def test_cleanup_proven_removal_maps_failed_and_uncertain_parent_sync(
        self,
    ) -> None:
        for kind in ("file", "directory"):
            for parent_sync in ("failed", "uncertain"):
                with self.subTest(kind=kind, parent_sync=parent_sync):
                    destination = portable(f"{kind}-{parent_sync}")
                    with synthetic_publication_root() as root:
                        with open_trusted_root(str(root)) as trusted:
                            factory = (
                                open_exclusive_staged_file
                                if kind == "file"
                                else open_exclusive_staged_directory
                            )
                            with factory(
                                trusted,
                                destination=destination,
                            ) as staged:
                                if kind == "file":
                                    staged.write(b"payload")
                                else:
                                    staged.write_file(
                                        portable("payload.bin"),
                                        (b"payload",),
                                    )
                                with (
                                    patch.object(
                                        publication_module,
                                        "_safe_parent_fsync",
                                        return_value=parent_sync,
                                    ),
                                    self.assertRaises(
                                        StagingCleanupError
                                    ) as caught,
                                ):
                                    cleanup_owned_staging(trusted, staged)
                                self.assertIs(
                                    caught.exception.state,
                                    StagingCleanupState
                                    .DISCARDED_DURABILITY_UNCERTAIN,
                                )
                                self.assertEqual(
                                    caught.exception.evidence.parent_fsync,
                                    parent_sync,
                                )
                                self.assertEqual(
                                    caught.exception.evidence.root_observation,
                                    "absent",
                                )
                                evidence = caught.exception.evidence
                                with self.assertRaises(
                                    (AttributeError, TypeError)
                                ):
                                    evidence.root_observation = "exact"
                                with self.assertRaises(TypeError):
                                    copy.copy(evidence)
                                with self.assertRaises(TypeError):
                                    copy.deepcopy(evidence)
                                with self.assertRaises(TypeError):
                                    pickle.dumps(evidence)
                                self.assertIs(
                                    staged.state,
                                    StagingState.RETIRED,
                                )
                            self.assertFalse(
                                (root / staging_leaf(destination)).exists()
                            )

    def test_removed_file_parent_fsync_failure_is_durability_uncertain(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    with patch.object(
                        publication_module,
                        "_fsync_descriptor",
                        side_effect=OSError(
                            errno.EIO,
                            "synthetic publication-parent fsync failure",
                        ),
                    ):
                        with self.assertRaises(StagingCleanupError) as caught:
                            cleanup_owned_staging(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        StagingCleanupState.DISCARDED_DURABILITY_UNCERTAIN,
                    )
                    self.assertEqual(
                        caught.exception.evidence.root_observation,
                        "absent",
                    )
                    self.assertEqual(
                        caught.exception.evidence.parent_fsync,
                        "failed",
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    with patch.object(
                        publication_module,
                        "_open_publication_parent",
                    ) as opened:
                        with self.assertRaises(StagingLifecycleError):
                            cleanup_owned_staging(trusted, staged)
                        opened.assert_not_called()
                self.assertFalse((root / staging_leaf(destination)).exists())

    def test_proven_durable_removal_preserves_result_builder_failure(
        self,
    ) -> None:
        for kind in ("file", "directory"):
            with self.subTest(kind=kind):
                destination = portable(f"durable-recovery-{kind}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        factory = (
                            open_exclusive_staged_file
                            if kind == "file"
                            else open_exclusive_staged_directory
                        )
                        with factory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            if kind == "file":
                                staged.write(b"payload")
                            else:
                                staged.write_file(
                                    portable("payload.bin"),
                                    (b"payload",),
                                )
                            original_result = (
                                publication_module._make_cleanup_result
                            )
                            result_calls = 0

                            def fail_once_then_build(*args):
                                nonlocal result_calls
                                result_calls += 1
                                if result_calls == 1:
                                    raise RuntimeError(
                                        "synthetic post-durability failure"
                                    )
                                return original_result(*args)

                            with patch.object(
                                publication_module,
                                "_make_cleanup_result",
                                side_effect=fail_once_then_build,
                            ), self.assertRaises(RuntimeError) as caught:
                                cleanup_owned_staging(
                                    trusted,
                                    staged,
                                )
                            self.assertEqual(
                                str(caught.exception),
                                "synthetic post-durability failure",
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.DISCARDED,
                            )
                            self.assertEqual(result_calls, 1)
                    self.assertFalse(
                        (root / staging_leaf(destination)).exists()
                    )

    def test_recovered_durable_cleanup_preserves_primary_and_retirement(
        self,
    ) -> None:
        destination = portable("durable-recovery-retirement")
        primary = OSError(errno.EIO, "synthetic first parent sync failure")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"payload")
                state = publication_module._staging_state(staged)
                assert state is not None
                operation_staging = -1
                real_open = publication_module._open_read_file_at
                real_close = os.close
                sync_calls = 0

                def record_operation_staging(*args, **kwargs):
                    nonlocal operation_staging
                    operation_staging = real_open(*args, **kwargs)
                    return operation_staging

                def fail_then_confirm_sync(*args, **kwargs):
                    nonlocal sync_calls
                    sync_calls += 1
                    if sync_calls == 1:
                        raise primary
                    return "succeeded"

                def fail_operation_staging_close(number: int) -> None:
                    if number == operation_staging:
                        raise OSError(
                            errno.EIO,
                            "synthetic operation close failure",
                        )
                    real_close(number)

                try:
                    with (
                        patch.object(
                            publication_module,
                            "_open_read_file_at",
                            side_effect=record_operation_staging,
                        ),
                        patch.object(
                            publication_module,
                            "_safe_parent_fsync",
                            side_effect=fail_then_confirm_sync,
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_operation_staging_close,
                        ),
                        self.assertRaises(BaseExceptionGroup) as caught,
                    ):
                        cleanup_owned_staging(trusted, staged)
                    self.assertEqual(
                        caught.exception.exceptions[0],
                        primary,
                    )
                    retirement_error = caught.exception.exceptions[1]
                    self.assertIs(
                        type(retirement_error),
                        DescriptorRetirementError,
                    )
                    self.assertEqual(
                        retirement_error.operation,
                        "cleanup",
                    )
                    self.assertIs(
                        retirement_error.terminal_result.state,
                        StagingCleanupState.DISCARDED_DURABLE,
                    )
                    self.assertIs(
                        staged.state,
                        StagingState.DISCARDED,
                    )
                    self.assertFalse(state.retirement_batch_pending)
                    self.assertEqual(sync_calls, 2)
                    self.assertFalse(
                        context.__exit__(None, None, None)
                    )
                finally:
                    if operation_staging >= 0:
                        real_close(operation_staging)
                self.assertFalse(
                    (root / staging_leaf(destination)).exists()
                )

    def test_partial_directory_cleanup_is_known_not_discarded_and_retired(
        self,
    ) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write_file(portable("payload.bin"), (b"payload",))
                    state = publication_module._staging_state(staged)
                    staging_identity = state.root_identity.public
                    original_fsync = publication_module._fsync_descriptor
                    failed = False

                    def fail_after_member_removal(descriptor: int) -> None:
                        nonlocal failed
                        observed = os.fstat(descriptor)
                        is_staging_root = (
                            observed.st_dev == staging_identity.device
                            and observed.st_ino == staging_identity.inode
                        )
                        if is_staging_root and not failed:
                            failed = True
                            raise OSError(
                                errno.EIO,
                                "synthetic partial cleanup fsync failure",
                            )
                        original_fsync(descriptor)

                    with patch.object(
                        publication_module,
                        "_fsync_descriptor",
                        side_effect=fail_after_member_removal,
                    ):
                        with self.assertRaises(StagingCleanupError) as caught:
                            cleanup_owned_staging(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        StagingCleanupState.NOT_DISCARDED,
                    )
                    self.assertEqual(
                        caught.exception.evidence.root_observation,
                        "owned_partial",
                    )
                    self.assertEqual(
                        caught.exception.evidence.remaining_expected_entries,
                        0,
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    staged_root = root / staging_leaf(destination)
                    self.assertTrue(staged_root.is_dir())
                    self.assertEqual(tuple(staged_root.iterdir()), ())

    def test_uninspectable_cleanup_namespace_preserves_staging(self) -> None:
        destination = portable("result.bin")
        namespace = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.UNINSPECTABLE,
            reference=portable(staging_leaf(destination)),
        )
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"recovery")
                    with (
                        patch.object(
                            publication_module,
                            "_collect_namespace_evidence",
                            return_value=namespace,
                        ),
                        patch.object(publication_module, "_unlink_at") as unlink,
                        self.assertRaises(StagingCleanupError) as caught,
                    ):
                        cleanup_owned_staging(trusted, staged)
                    unlink.assert_not_called()
                    self.assertIs(
                        caught.exception.state,
                        StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN,
                    )
                    self.assertEqual(
                        caught.exception.evidence.root_observation,
                        "exact",
                    )
                    self.assertEqual(
                        caught.exception.evidence.namespace_evidence,
                        namespace,
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"recovery",
                    )

    def test_malformed_directory_membership_is_uncertain_before_deletion(
        self,
    ) -> None:
        for anomaly in ("unexpected", "special", "missing"):
            with self.subTest(anomaly=anomaly):
                destination = portable("result-dir")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write_file(
                                portable("expected.bin"),
                                (b"expected",),
                            )
                            staged_root = root / staging_leaf(destination)
                            if anomaly == "unexpected":
                                (staged_root / "unexpected.bin").write_bytes(
                                    b"unexpected"
                                )
                            elif anomaly == "special":
                                os.mkfifo(
                                    staged_root / "unexpected-fifo",
                                    0o600,
                                )
                            else:
                                (staged_root / "expected.bin").unlink()
                            before = synthetic_tree_snapshot(root)
                            with (
                                patch.object(
                                    publication_module,
                                    "_unlink_at",
                                ) as unlink,
                                patch.object(
                                    publication_module,
                                    "_rmdir_at",
                                ) as rmdir,
                                self.assertRaises(
                                    StagingCleanupError
                                ) as caught,
                            ):
                                cleanup_owned_staging(trusted, staged)
                            unlink.assert_not_called()
                            rmdir.assert_not_called()
                            self.assertIs(
                                caught.exception.state,
                                StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN,
                            )
                            self.assertEqual(
                                caught.exception.evidence.root_observation,
                                "malformed",
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            self.assertEqual(
                                synthetic_tree_snapshot(root),
                                before,
                            )

    def test_late_directory_emptiness_probe_consumes_at_most_one_name(
        self,
    ) -> None:
        destination = portable("result-dir")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.mkdir(portable("nested"))
                    staged.write_file(
                        portable("nested/payload.bin"),
                        (b"payload",),
                    )
                    state = publication_module._staging_state(staged)
                    target_inode = (
                        root / staging_leaf(destination) / "nested"
                    ).stat(follow_symlinks=False).st_ino
                    original_list_names = publication_module._list_names
                    target_enumerations = 0
                    late_requests = 0

                    def bounded_late_names():
                        nonlocal late_requests
                        late_requests += 1
                        yield "synthetic-late-entry"
                        late_requests += 1
                        raise AssertionError(
                            "late emptiness probe requested a second name"
                        )

                    def instrumented_names(descriptor: int):
                        nonlocal target_enumerations
                        observed = os.fstat(descriptor)
                        if observed.st_ino == target_inode:
                            target_enumerations += 1
                            if target_enumerations == 3:
                                return bounded_late_names()
                        return original_list_names(descriptor)

                    baseline = lowest_available_descriptor()
                    with (
                        patch.object(
                            publication_module,
                            "_list_names",
                            side_effect=instrumented_names,
                        ),
                        self.assertRaises(PublicationValidationError),
                    ):
                        publication_module._delete_staged_tree(
                            state,
                            removed=set(),
                            directory_journal={},
                        )
                    self.assertEqual(target_enumerations, 3)
                    self.assertEqual(late_requests, 1)
                    self.assertEqual(
                        lowest_available_descriptor(),
                        baseline,
                    )
                    staged_root = root / staging_leaf(destination)
                    self.assertTrue((staged_root / "nested").is_dir())
                    self.assertFalse(
                        (staged_root / "nested" / "payload.bin").exists()
                    )

    def test_final_cleanup_namespace_conflict_is_uncertain_and_preserved(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"owned")
                    state = publication_module._staging_state(staged)
                    alias = root / state.staging.value.upper()
                    original_collect = (
                        publication_module._collect_namespace_evidence
                    )
                    calls = 0

                    def introduce_alias_after_removal(
                        descriptor: int,
                        reference: PortableRelativePath,
                    ) -> NamespaceEvidence:
                        nonlocal calls
                        calls += 1
                        if calls == 2:
                            alias.write_bytes(b"foreign-alias")
                        return original_collect(descriptor, reference)

                    with patch.object(
                        publication_module,
                        "_collect_namespace_evidence",
                        side_effect=introduce_alias_after_removal,
                    ):
                        with self.assertRaises(StagingCleanupError) as caught:
                            cleanup_owned_staging(trusted, staged)
                    self.assertIs(
                        caught.exception.state,
                        StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN,
                    )
                    self.assertIn(
                        caught.exception.evidence.root_observation,
                        {"absent", "foreign"},
                    )
                    namespace = caught.exception.evidence.namespace_evidence
                    self.assertEqual(
                        namespace.namespace_observation,
                        "complete_conflict",
                    )
                    self.assertEqual(
                        tuple(
                            item.value
                            for item in namespace.conflicting_aliases
                        ),
                        (state.staging.value.upper(),),
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertEqual(alias.read_bytes(), b"foreign-alias")
                    exact_names = tuple(path.name for path in root.iterdir())
                    self.assertNotIn(state.staging.value, exact_names)
                    self.assertIn(state.staging.value.upper(), exact_names)

    def test_repeated_cleanup_uncertainty_retires_without_descriptor_growth(
        self,
    ) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                baseline = lowest_available_descriptor()
                for index in range(4):
                    destination = portable(f"result-{index}.bin")
                    with open_exclusive_staged_file(
                        trusted,
                        destination=destination,
                    ) as staged:
                        staged.write(b"payload")
                        with patch.object(
                            publication_module,
                            "_fsync_descriptor",
                            side_effect=OSError(
                                errno.EIO,
                                "synthetic cleanup uncertainty",
                            ),
                        ):
                            with self.assertRaises(StagingCleanupError):
                                cleanup_owned_staging(trusted, staged)
                        self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertEqual(lowest_available_descriptor(), baseline)


class PerHandleSerializationTests(unittest.TestCase):
    @staticmethod
    def _run_thread(
        outcomes: dict[str, object],
        key: str,
        operation,
        started: threading.Event | None = None,
    ) -> None:
        if started is not None:
            started.set()
        try:
            outcomes[key] = operation()
        except BaseException as exc:
            outcomes[key] = exc

    def test_caller_iterator_reentrancy_is_rejected_before_mutation(
        self,
    ) -> None:
        actions = (
            "cleanup",
            "publish",
            "seal",
            "mkdir",
            "write_file",
            "context_exit",
        )
        for phase in ("iter", "next"):
            for action in actions:
                with self.subTest(phase=phase, action=action):
                    destination = portable(f"reentrant-{phase}-{action}")
                    with synthetic_publication_root() as root:
                        with open_trusted_root(str(root)) as trusted:
                            context = open_exclusive_staged_directory(
                                trusted,
                                destination=destination,
                            )
                            staged = context.__enter__()
                            staged.write_file(
                                portable("first.bin"),
                                (b"first",),
                            )
                            state = publication_module._staging_state(staged)
                            assert state is not None
                            captured: list[BaseException] = []

                            def attempt_reentry() -> None:
                                try:
                                    if action == "cleanup":
                                        cleanup_owned_staging(
                                            trusted,
                                            staged,
                                        )
                                    elif action == "publish":
                                        publish_completed_directory(
                                            trusted,
                                            staged,
                                        )
                                    elif action == "seal":
                                        staged.seal(
                                            scope="synthetic-reentrant-v1"
                                        )
                                    elif action == "mkdir":
                                        staged.mkdir(portable("nested"))
                                    elif action == "write_file":
                                        staged.write_file(
                                            portable("nested.bin"),
                                            (b"nested",),
                                        )
                                    else:
                                        context.__exit__(
                                            None,
                                            None,
                                            None,
                                        )
                                except BaseException as error:
                                    captured.append(error)

                            class ReentrantChunks:
                                def __init__(self) -> None:
                                    self.yielded = False

                                def __iter__(self):
                                    if phase == "iter":
                                        attempt_reentry()
                                    return self

                                def __next__(self):
                                    if self.yielded:
                                        raise StopIteration
                                    if phase == "next":
                                        attempt_reentry()
                                    self.yielded = True
                                    return b"late"

                            staged.write_file(
                                portable("late.bin"),
                                ReentrantChunks(),
                            )
                            self.assertEqual(len(captured), 1)
                            if action == "context_exit":
                                self.assertIs(type(captured[0]), RuntimeError)
                                self.assertIn(
                                    "cannot exit while a handle operation "
                                    "is active",
                                    str(captured[0]),
                                )
                            else:
                                self.assertIs(
                                    type(captured[0]),
                                    StagingLifecycleError,
                                )
                                self.assertEqual(
                                    captured[0].operation,  # type: ignore[union-attr]
                                    action,
                                )
                            self.assertIs(
                                state.lifecycle,
                                StagingState.OPEN,
                            )
                            self.assertIsNone(state.active_operation)
                            self.assertFalse(state.publication_attempted)
                            self.assertFalse(state.cleanup_attempted)
                            self.assertEqual(
                                tuple(
                                    path.value
                                    for path, _identity in state.tree.entries
                                ),
                                ("first.bin", "late.bin"),
                            )
                            staged_root = root / staging_leaf(destination)
                            self.assertFalse(
                                (staged_root / "nested").exists()
                            )
                            self.assertFalse(
                                (staged_root / "nested.bin").exists()
                            )
                            cleanup = cleanup_owned_staging(
                                trusted,
                                staged,
                            )
                            self.assertIs(
                                cleanup.state,
                                StagingCleanupState.DISCARDED_DURABLE,
                            )
                            self.assertFalse(
                                context.__exit__(None, None, None)
                            )

    def test_terminal_reentry_validates_bound_root_before_lifecycle(
        self,
    ) -> None:
        for operation in ("publish", "cleanup"):
            with self.subTest(operation=operation):
                destination = portable(f"wrong-root-reentrant-{operation}")
                with (
                    synthetic_publication_root() as root,
                    synthetic_publication_root() as foreign_root,
                ):
                    with (
                        open_trusted_root(str(root)) as trusted,
                        open_trusted_root(str(foreign_root)) as foreign_trusted,
                    ):
                        context = open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        captured: list[BaseException] = []

                        class ReentrantChunks:
                            def __iter__(self):
                                return self

                            def __next__(self):
                                if captured:
                                    raise StopIteration
                                try:
                                    if operation == "publish":
                                        publish_completed_directory(
                                            foreign_trusted,
                                            staged,
                                        )
                                    else:
                                        cleanup_owned_staging(
                                            foreign_trusted,
                                            staged,
                                        )
                                except BaseException as error:
                                    captured.append(error)
                                return b"payload"

                        staged.write_file(
                            portable("payload.bin"),
                            ReentrantChunks(),
                        )
                        self.assertEqual(len(captured), 1)
                        self.assertIs(
                            type(captured[0]),
                            StagingAuthorityError,
                        )
                        self.assertEqual(
                            captured[0].operation,  # type: ignore[union-attr]
                            operation,
                        )
                        self.assertIs(staged.state, StagingState.OPEN)
                        cleanup_owned_staging(trusted, staged)
                        self.assertFalse(context.__exit__(None, None, None))

    def test_competing_publish_calls_cross_native_boundary_exactly_once(
        self,
    ) -> None:
        destination = portable("serialized-publish.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    original_native = publication_module._native_no_replace
                    entered = threading.Barrier(2)
                    release = threading.Barrier(2)
                    second_started = threading.Event()
                    native_calls = 0
                    call_lock = threading.Lock()
                    outcomes: dict[str, object] = {}

                    def blocked_native(*args):
                        nonlocal native_calls
                        with call_lock:
                            native_calls += 1
                        entered.wait(timeout=5)
                        release.wait(timeout=5)
                        return original_native(*args)

                    with patch.object(
                        publication_module,
                        "_native_no_replace",
                        side_effect=blocked_native,
                    ):
                        first = threading.Thread(
                            target=self._run_thread,
                            args=(
                                outcomes,
                                "first",
                                lambda: publish_completed_file(
                                    trusted,
                                    staged,
                                ),
                            ),
                        )
                        first.start()
                        entered.wait(timeout=5)
                        second = threading.Thread(
                            target=self._run_thread,
                            args=(
                                outcomes,
                                "second",
                                lambda: publish_completed_file(
                                    trusted,
                                    staged,
                                ),
                                second_started,
                            ),
                        )
                        second.start()
                        self.assertTrue(second_started.wait(timeout=5))
                        release.wait(timeout=5)
                        first.join(timeout=5)
                        second.join(timeout=5)
                    self.assertFalse(first.is_alive())
                    self.assertFalse(second.is_alive())
                    self.assertEqual(native_calls, 1)
                    self.assertIsInstance(outcomes["first"], PublicationResult)
                    self.assertIsInstance(
                        outcomes["second"],
                        StagingLifecycleError,
                    )
                    self.assertIs(staged.state, StagingState.PUBLISHED)
            self.assertEqual(
                (root / destination.value).read_bytes(),
                b"payload",
            )

    def test_competing_cleanup_calls_mutate_the_namespace_exactly_once(
        self,
    ) -> None:
        destination = portable("serialized-cleanup.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    original_unlink = publication_module._unlink_at
                    entered = threading.Barrier(2)
                    release = threading.Barrier(2)
                    second_started = threading.Event()
                    unlink_calls = 0
                    call_lock = threading.Lock()
                    outcomes: dict[str, object] = {}

                    def blocked_unlink(*args):
                        nonlocal unlink_calls
                        with call_lock:
                            unlink_calls += 1
                        entered.wait(timeout=5)
                        release.wait(timeout=5)
                        return original_unlink(*args)

                    with patch.object(
                        publication_module,
                        "_unlink_at",
                        side_effect=blocked_unlink,
                    ):
                        first = threading.Thread(
                            target=self._run_thread,
                            args=(
                                outcomes,
                                "first",
                                lambda: cleanup_owned_staging(
                                    trusted,
                                    staged,
                                ),
                            ),
                        )
                        first.start()
                        entered.wait(timeout=5)
                        second = threading.Thread(
                            target=self._run_thread,
                            args=(
                                outcomes,
                                "second",
                                lambda: cleanup_owned_staging(
                                    trusted,
                                    staged,
                                ),
                                second_started,
                            ),
                        )
                        second.start()
                        self.assertTrue(second_started.wait(timeout=5))
                        release.wait(timeout=5)
                        first.join(timeout=5)
                        second.join(timeout=5)
                    self.assertFalse(first.is_alive())
                    self.assertFalse(second.is_alive())
                    self.assertEqual(unlink_calls, 1)
                    self.assertIsInstance(
                        outcomes["first"],
                        StagingCleanupResult,
                    )
                    self.assertIsInstance(
                        outcomes["second"],
                        StagingLifecycleError,
                    )
                    self.assertIs(staged.state, StagingState.DISCARDED)
            self.assertFalse((root / staging_leaf(destination)).exists())

    def test_seal_serializes_against_late_population_and_rejects_write(
        self,
    ) -> None:
        destination = portable("serialized-seal.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"original")
                    original_chmod = publication_module._fchmod_descriptor
                    entered = threading.Barrier(2)
                    release = threading.Barrier(2)
                    second_started = threading.Event()
                    chmod_calls = 0
                    call_lock = threading.Lock()
                    outcomes: dict[str, object] = {}

                    def blocked_chmod(*args):
                        nonlocal chmod_calls
                        with call_lock:
                            chmod_calls += 1
                        entered.wait(timeout=5)
                        release.wait(timeout=5)
                        return original_chmod(*args)

                    with patch.object(
                        publication_module,
                        "_fchmod_descriptor",
                        side_effect=blocked_chmod,
                    ):
                        first = threading.Thread(
                            target=self._run_thread,
                            args=(
                                outcomes,
                                "seal",
                                lambda: staged.seal(executable=True),
                            ),
                        )
                        first.start()
                        entered.wait(timeout=5)
                        second = threading.Thread(
                            target=self._run_thread,
                            args=(
                                outcomes,
                                "write",
                                lambda: staged.write(b"-late"),
                                second_started,
                            ),
                        )
                        second.start()
                        self.assertTrue(second_started.wait(timeout=5))
                        release.wait(timeout=5)
                        first.join(timeout=5)
                        second.join(timeout=5)
                    self.assertFalse(first.is_alive())
                    self.assertFalse(second.is_alive())
                    self.assertIsNone(outcomes["seal"])
                    self.assertIsInstance(
                        outcomes["write"],
                        StagingLifecycleError,
                    )
                    self.assertEqual(chmod_calls, 1)
                    self.assertIs(staged.state, StagingState.SEALED)
                    self.assertEqual(
                        (root / staging_leaf(destination)).read_bytes(),
                        b"original",
                    )
                    cleanup_owned_staging(trusted, staged)


class DescriptorRetirementValueTests(unittest.TestCase):
    @staticmethod
    def identity() -> DescriptorRetirementIdentity:
        return publication_module._make_retirement_identity(
            os.stat(os.devnull, follow_symlinks=False)
        )

    @classmethod
    def record(
        cls,
        *,
        ordinal: int = 0,
        role: str = "handle_staging",
        observation: DescriptorRetirementObservation = (
            DescriptorRetirementObservation.UNINSPECTABLE
        ),
        close_attempted: bool = False,
        admitted_identity: DescriptorRetirementIdentity | None = None,
        observed_identity: DescriptorRetirementIdentity | None = None,
        error_errno: int | None = errno.EIO,
    ) -> DescriptorRetirementRecord:
        identity = (
            cls.identity()
            if admitted_identity is None
            and observation
            is not DescriptorRetirementObservation.ALREADY_ABSENT
            else admitted_identity
        )
        return publication_module._make_retirement_record(
            ordinal=ordinal,
            role=role,
            observation=observation,
            close_attempted=close_attempted,
            admitted_identity=identity,
            observed_identity=observed_identity,
            error_errno=error_errno,
        )

    @classmethod
    def evidence(
        cls,
        *,
        role: str = "handle_staging",
        post_outcome: bool = False,
    ) -> DescriptorRetirementEvidence:
        return publication_module._make_retirement_evidence(
            (cls.record(role=role),),
            post_outcome=post_outcome,
        )

    def test_observation_invariants_and_numeric_bounds_are_exact(self) -> None:
        identity = self.identity()
        other = object.__new__(DescriptorRetirementIdentity)
        object.__setattr__(other, "device", identity.device)
        object.__setattr__(other, "inode", identity.inode + 1)
        object.__setattr__(other, "entry_type", identity.entry_type)
        object.__setattr__(other, "owner_uid", identity.owner_uid)
        cases = (
            (
                DescriptorRetirementObservation.CLOSED,
                True,
                identity,
                identity,
                None,
            ),
            (
                DescriptorRetirementObservation.ALREADY_ABSENT,
                False,
                identity,
                None,
                errno.EBADF,
            ),
            (
                DescriptorRetirementObservation.FOREIGN_PRESERVED,
                False,
                identity,
                None,
                None,
            ),
            (
                DescriptorRetirementObservation.FOREIGN_PRESERVED,
                False,
                identity,
                other,
                None,
            ),
            (
                DescriptorRetirementObservation.UNINSPECTABLE,
                False,
                identity,
                None,
                errno.EIO,
            ),
            (
                DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
                True,
                identity,
                identity,
                errno.EINTR,
            ),
        )
        for observation, attempted, admitted, observed, number in cases:
            with self.subTest(observation=observation.value, observed=observed):
                record = publication_module._make_retirement_record(
                    ordinal=0,
                    role="traversal_entry",
                    observation=observation,
                    close_attempted=attempted,
                    admitted_identity=admitted,
                    observed_identity=observed,
                    error_errno=number,
                )
                self.assertIs(record.observation, observation)

        for name, kwargs in (
            ("bool-ordinal", {"ordinal": True}),
            ("large-ordinal", {"ordinal": 7}),
            ("bool-errno", {"error_errno": True}),
            ("large-errno", {"error_errno": 2**31}),
            (
                "closed-without-close",
                {
                    "observation": DescriptorRetirementObservation.CLOSED,
                    "close_attempted": False,
                    "error_errno": None,
                    "observed_identity": identity,
                },
            ),
            (
                "foreign-equal",
                {
                    "observation": (
                        DescriptorRetirementObservation.FOREIGN_PRESERVED
                    ),
                    "error_errno": None,
                    "observed_identity": identity,
                },
            ),
        ):
            arguments = {
                "ordinal": 0,
                "role": "traversal_entry",
                "observation": DescriptorRetirementObservation.UNINSPECTABLE,
                "close_attempted": False,
                "admitted_identity": identity,
                "observed_identity": None,
                "error_errno": errno.EIO,
            }
            arguments.update(kwargs)
            with self.subTest(invalid=name), self.assertRaises(
                (TypeError, ValueError)
            ):
                publication_module._make_retirement_record(**arguments)

        invalid_identity = object.__new__(DescriptorRetirementIdentity)
        object.__setattr__(invalid_identity, "device", -1)
        object.__setattr__(invalid_identity, "inode", 0)
        object.__setattr__(invalid_identity, "entry_type", "regular_file")
        object.__setattr__(invalid_identity, "owner_uid", 0)
        self.assertEqual(identity.device >= 0, True)
        with self.assertRaises(ValueError):
            publication_module._validate_uint(
                "device",
                invalid_identity.device,
                2**64 - 1,
            )

    def test_evidence_order_uniqueness_bounds_and_immutability(self) -> None:
        identity = self.identity()
        roles = (
            "traversal_entry",
            "traversal_directory",
            "traversal_parent",
            "operation_staging",
            "operation_parent",
            "handle_staging",
            "handle_parent",
        )
        records = tuple(
            publication_module._make_retirement_record(
                ordinal=index,
                role=role,
                observation=(
                    DescriptorRetirementObservation.UNINSPECTABLE
                    if index == 0
                    else DescriptorRetirementObservation.CLOSED
                ),
                close_attempted=index != 0,
                admitted_identity=identity,
                observed_identity=identity if index != 0 else None,
                error_errno=errno.EIO if index == 0 else None,
            )
            for index, role in enumerate(roles)
        )
        evidence = publication_module._make_retirement_evidence(records)
        self.assertEqual(
            tuple(record.role for record in evidence.records),
            roles,
        )
        for value in (*records, evidence, identity):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises((AttributeError, TypeError)):
                    value.records = ()
                with self.assertRaises(TypeError):
                    copy.copy(value)
                with self.assertRaises(TypeError):
                    copy.deepcopy(value)
                with self.assertRaises(TypeError):
                    pickle.dumps(value)

        with self.assertRaises(ValueError):
            publication_module._make_retirement_evidence(())
        with self.assertRaises(ValueError):
            publication_module._make_retirement_evidence(records + (records[-1],))
        with self.assertRaises(ValueError):
            publication_module._make_retirement_evidence(
                (self.record(role="handle_staging"), self.record(role="handle_staging"))
            )
        with self.assertRaises(ValueError):
            publication_module._make_retirement_evidence(
                (self.record(role="traversal_entry"),),
                post_outcome=True,
            )

    def test_retirement_error_cross_field_matrix_and_transport(self) -> None:
        destination = portable("result")
        staging = portable(".rp-stage-v1-" + "0" * 64)
        evidence = self.evidence(role="handle_staging", post_outcome=True)
        namespace = publication_module._make_namespace_evidence(
            publication_module._NamespaceObservation.NO_CONFLICT,
            reference=destination,
        )
        entry = publication_module._make_entry_identity(
            os.stat(__file__, follow_symlinks=False)
        )
        publication_result = publication_module._make_publication_result(
            PublicationState.COMMITTED_DURABLE,
            destination,
            entry,
            namespace,
        )
        cleanup_result = publication_module._make_cleanup_result(
            staging,
            entry,
            publication_module._make_namespace_evidence(
                publication_module._NamespaceObservation.NO_CONFLICT,
                reference=staging,
            ),
        )
        valid = (
            (
                StagingState.PUBLISHED,
                "publish",
                publication_result,
            ),
            (
                StagingState.DISCARDED,
                "cleanup",
                cleanup_result,
            ),
            (StagingState.RETIRED, "write", None),
            (StagingState.RETIRED, "mkdir", None),
            (StagingState.RETIRED, "write_file", None),
            (StagingState.RETIRED, "seal", None),
            (StagingState.RETIRED, "context_exit", None),
            (StagingState.NOT_COMMITTED, "context_exit", None),
            (StagingState.RETIRED, "finalization", None),
        )
        for state, operation, result in valid:
            with self.subTest(operation=operation, state=state.value):
                error = DescriptorRetirementError(
                    state=state,
                    operation=operation,
                    destination=destination,
                    staging=staging,
                    terminal_result=result,
                    retirement_evidence=evidence,
                )
                self.assertIs(error.state, state)
                self.assertEqual(error.operation, operation)
                self.assertIs(error.terminal_result, result)
                self.assertIs(error.retirement_evidence, evidence)
                with self.assertRaises((AttributeError, TypeError)):
                    error.state = StagingState.OPEN
                with self.assertRaises(TypeError):
                    copy.copy(error)
                with self.assertRaises(TypeError):
                    copy.deepcopy(error)
                with self.assertRaises(TypeError):
                    pickle.dumps(error)

        for state, operation, result in (
            (StagingState.PUBLISHED, "publish", None),
            (StagingState.NOT_COMMITTED, "publish", publication_result),
            (StagingState.RETIRED, "cleanup", None),
            (StagingState.RETIRED, "staging_admission", None),
        ):
            with self.subTest(invalid=(state.value, operation)):
                with self.assertRaises((TypeError, ValueError)):
                    DescriptorRetirementError(
                        state=state,
                        operation=operation,
                        destination=destination,
                        staging=staging,
                        terminal_result=result,
                        retirement_evidence=evidence,
                    )

        class HostileOperation(str):
            pass

        with self.assertRaises(TypeError):
            DescriptorRetirementError(
                state=StagingState.RETIRED,
                operation=HostileOperation("write"),
                destination=destination,
                staging=staging,
                terminal_result=None,
                retirement_evidence=evidence,
            )
        self.assertEqual(
            get_type_hints(DescriptorRetirementError.operation.fget)[
                "return"
            ].__origin__,
            __import__("typing").Literal,
        )


class DescriptorLifecycleTests(unittest.TestCase):
    def test_expected_identity_binding_retires_mismatched_open_descriptors(
        self,
    ) -> None:
        for replacement_kind in (
            "regular_file",
            "fifo",
            "directory",
        ):
            with self.subTest(replacement_kind=replacement_kind):
                with synthetic_publication_root() as root:
                    target = root / "target"
                    if replacement_kind == "directory":
                        target.mkdir(mode=0o700)
                        original_flags = (
                            os.O_RDONLY
                            | os.O_DIRECTORY
                            | os.O_NOFOLLOW
                            | os.O_CLOEXEC
                        )
                    else:
                        target.write_bytes(b"original")
                        original_flags = (
                            os.O_RDONLY
                            | os.O_NOFOLLOW
                            | os.O_CLOEXEC
                        )
                    original_descriptor = os.open(target, original_flags)
                    try:
                        expected = publication_module._make_entry_identity(
                            target.stat(follow_symlinks=False)
                        )
                        self.assertEqual(
                            publication_module._make_retirement_identity(
                                os.fstat(original_descriptor)
                            ),
                            publication_module
                            ._retirement_identity_from_publication_identity(
                                expected
                            ),
                        )
                        parent = os.open(
                            root,
                            os.O_RDONLY
                            | os.O_DIRECTORY
                            | os.O_NOFOLLOW
                            | os.O_CLOEXEC,
                        )
                        descriptor = -1
                        try:
                            baseline = lowest_available_descriptor()
                            if replacement_kind == "directory":
                                target.rmdir()
                                target.mkdir(mode=0o700)
                            elif replacement_kind == "fifo":
                                target.unlink()
                                os.mkfifo(target, 0o600)
                            else:
                                target.unlink()
                                target.write_bytes(b"replacement")
                            descriptor = (
                                publication_module._open_directory_at(
                                    "target",
                                    parent,
                                    role="traversal_directory",
                                )
                                if replacement_kind == "directory"
                                else publication_module._open_read_file_at(
                                    "target",
                                    parent,
                                    role="traversal_entry",
                                )
                            )
                            self.assertNotEqual(
                                publication_module._make_retirement_identity(
                                    os.fstat(descriptor)
                                ),
                                publication_module
                                ._retirement_identity_from_publication_identity(
                                    expected
                                ),
                            )
                            with (
                                publication_module
                                ._OPEN_DESCRIPTOR_IDENTITIES_LOCK
                            ):
                                self.assertNotIn(
                                    original_descriptor,
                                    publication_module
                                    ._OPEN_DESCRIPTOR_IDENTITIES,
                                )
                            with self.assertRaises(
                                publication_module._DescriptorAdmissionFailure
                            ) as caught:
                                publication_module._owned_descriptor(
                                    descriptor,
                                    (
                                        "traversal_directory"
                                        if replacement_kind == "directory"
                                        else "traversal_entry"
                                    ),
                                    expected=expected,
                                )
                            self.assertEqual(
                                len(caught.exception.retirement_records),
                                1,
                            )
                            self.assertIs(
                                caught.exception.retirement_records[
                                    0
                                ].observation,
                                DescriptorRetirementObservation.CLOSED,
                            )
                            with self.assertRaises(OSError) as absent:
                                os.fstat(descriptor)
                            self.assertEqual(
                                absent.exception.errno,
                                errno.EBADF,
                            )
                            with (
                                publication_module
                                ._OPEN_DESCRIPTOR_IDENTITIES_LOCK
                            ):
                                self.assertNotIn(
                                    descriptor,
                                    publication_module
                                    ._OPEN_DESCRIPTOR_IDENTITIES,
                                )
                                self.assertNotIn(
                                    original_descriptor,
                                    publication_module
                                    ._OPEN_DESCRIPTOR_IDENTITIES,
                                )
                            self.assertEqual(
                                lowest_available_descriptor(),
                                baseline,
                            )
                        finally:
                            if descriptor >= 0:
                                try:
                                    publication_module._close_descriptor(
                                        descriptor
                                    )
                                except OSError as exc:
                                    if exc.errno != errno.EBADF:
                                        raise
                            os.close(parent)
                    finally:
                        os.close(original_descriptor)

    def test_successful_retirement_closes_once_and_removes_registration(
        self,
    ) -> None:
        with synthetic_publication_root() as root:
            (root / "input.bin").write_bytes(b"input")
            parent = os.open(
                root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptor = publication_module._open_read_file_at(
                "input.bin",
                parent,
                role="traversal_entry",
            )
            owned = publication_module._owned_descriptor(
                descriptor,
                "traversal_entry",
            )
            real_fstat = publication_module._fstat_descriptor
            real_close = os.close
            fstats: list[int] = []
            closes: list[int] = []

            def observed_fstat(value: int):
                fstats.append(value)
                return real_fstat(value)

            def observed_close(value: int) -> None:
                closes.append(value)
                real_close(value)

            try:
                with (
                    patch.object(
                        publication_module,
                        "_fstat_descriptor",
                        side_effect=observed_fstat,
                    ),
                    patch.object(
                        publication_module.os,
                        "close",
                        side_effect=observed_close,
                    ),
                ):
                    evidence = publication_module._retire_owned_descriptors(
                        (owned,)
                    )
                self.assertIsNone(evidence)
                self.assertEqual(fstats, [descriptor])
                self.assertEqual(closes, [descriptor])
                with self.assertRaises(OSError) as caught:
                    os.fstat(descriptor)
                self.assertEqual(caught.exception.errno, errno.EBADF)
                with publication_module._OPEN_DESCRIPTOR_IDENTITIES_LOCK:
                    self.assertNotIn(
                        descriptor,
                        publication_module._OPEN_DESCRIPTOR_IDENTITIES,
                    )
            finally:
                os.close(parent)

    def test_generation_mismatch_is_foreign_without_inspection_or_close(
        self,
    ) -> None:
        with synthetic_publication_root() as root:
            (root / "input.bin").write_bytes(b"input")
            parent = os.open(
                root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            old_descriptor = publication_module._open_read_file_at(
                "input.bin",
                parent,
                role="traversal_entry",
            )
            old_owned = publication_module._owned_descriptor(
                old_descriptor,
                "traversal_entry",
            )
            os.close(old_descriptor)
            replacement = publication_module._open_read_file_at(
                "input.bin",
                parent,
                role="traversal_entry",
            )
            self.assertEqual(replacement, old_descriptor)
            try:
                with (
                    patch.object(
                        publication_module,
                        "_fstat_descriptor",
                        side_effect=AssertionError(
                            "generation mismatch inspected foreign descriptor"
                        ),
                    ),
                    patch.object(
                        publication_module.os,
                        "close",
                        side_effect=AssertionError(
                            "generation mismatch closed foreign descriptor"
                        ),
                    ),
                ):
                    evidence = publication_module._retire_owned_descriptors(
                        (old_owned,)
                    )
                self.assertIsNotNone(evidence)
                record = evidence.records[0]
                self.assertIs(
                    record.observation,
                    DescriptorRetirementObservation.FOREIGN_PRESERVED,
                )
                self.assertIsNone(record.observed_identity)
                self.assertFalse(record.close_attempted)
                os.fstat(replacement)
                with publication_module._OPEN_DESCRIPTOR_IDENTITIES_LOCK:
                    registration = (
                        publication_module._OPEN_DESCRIPTOR_IDENTITIES[
                            replacement
                        ]
                    )
                self.assertNotEqual(
                    registration.generation,
                    old_owned.generation,
                )
            finally:
                publication_module._close_descriptor(replacement)
                os.close(parent)

    def test_preclose_absence_and_matching_generation_foreign_reuse(
        self,
    ) -> None:
        for replacement_kind in ("absent", "foreign"):
            with self.subTest(replacement_kind=replacement_kind):
                with synthetic_publication_root() as root:
                    (root / "input.bin").write_bytes(b"input")
                    parent = os.open(
                        root,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                    )
                    descriptor = publication_module._open_read_file_at(
                        "input.bin",
                        parent,
                        role="traversal_entry",
                    )
                    owned = publication_module._owned_descriptor(
                        descriptor,
                        "traversal_entry",
                    )
                    os.close(descriptor)
                    foreign = -1
                    if replacement_kind == "foreign":
                        foreign = os.open(
                            os.devnull,
                            os.O_RDONLY | os.O_CLOEXEC,
                        )
                        self.assertEqual(foreign, descriptor)
                    closes: list[int] = []
                    real_close = os.close
                    try:
                        with patch.object(
                            publication_module.os,
                            "close",
                            side_effect=lambda value: closes.append(value),
                        ):
                            evidence = (
                                publication_module._retire_owned_descriptors(
                                    (owned,)
                                )
                            )
                        self.assertIsNotNone(evidence)
                        record = evidence.records[0]
                        if replacement_kind == "absent":
                            self.assertIs(
                                record.observation,
                                DescriptorRetirementObservation.ALREADY_ABSENT,
                            )
                            self.assertEqual(record.error_errno, errno.EBADF)
                        else:
                            self.assertIs(
                                record.observation,
                                DescriptorRetirementObservation.FOREIGN_PRESERVED,
                            )
                            self.assertIsNotNone(record.observed_identity)
                            self.assertNotEqual(
                                record.observed_identity,
                                record.admitted_identity,
                            )
                            os.fstat(foreign)
                        self.assertEqual(closes, [])
                    finally:
                        if foreign >= 0:
                            real_close(foreign)
                        os.close(parent)

    def test_uninspectable_and_every_close_exception_are_bounded_once(
        self,
    ) -> None:
        failures: tuple[tuple[str, BaseException, int | None], ...] = (
            ("fstat-oserror", OSError(errno.EIO, "synthetic"), errno.EIO),
            ("fstat-nonoserror", RuntimeError("synthetic"), None),
        )
        for name, failure, expected_errno in failures:
            with self.subTest(name=name):
                descriptor = publication_module._admit_opened_descriptor(
                    os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC),
                    role="traversal_entry",
                )
                owned = publication_module._owned_descriptor(
                    descriptor,
                    "traversal_entry",
                )
                close_calls: list[int] = []
                try:
                    with (
                        patch.object(
                            publication_module,
                            "_fstat_descriptor",
                            side_effect=failure,
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=lambda value: close_calls.append(value),
                        ),
                    ):
                        evidence = (
                            publication_module._retire_owned_descriptors(
                                (owned,)
                            )
                        )
                    record = evidence.records[0]
                    self.assertIs(
                        record.observation,
                        DescriptorRetirementObservation.UNINSPECTABLE,
                    )
                    self.assertEqual(record.error_errno, expected_errno)
                    self.assertEqual(close_calls, [])
                finally:
                    os.close(descriptor)

        close_failures: tuple[BaseException, ...] = (
            OSError(errno.EBADF, "synthetic"),
            OSError(errno.EINTR, "synthetic"),
            OSError(errno.EIO, "synthetic"),
            RuntimeError("synthetic"),
        )
        for failure in close_failures:
            with self.subTest(close_failure=type(failure).__name__, errno=getattr(failure, "errno", None)):
                descriptor = publication_module._admit_opened_descriptor(
                    os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC),
                    role="traversal_entry",
                )
                owned = publication_module._owned_descriptor(
                    descriptor,
                    "traversal_entry",
                )
                fstat_calls: list[int] = []
                close_calls: list[int] = []
                real_fstat = publication_module._fstat_descriptor

                def observed_fstat(value: int):
                    fstat_calls.append(value)
                    return real_fstat(value)

                def failed_close(value: int) -> None:
                    close_calls.append(value)
                    raise failure

                try:
                    with (
                        patch.object(
                            publication_module,
                            "_fstat_descriptor",
                            side_effect=observed_fstat,
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=failed_close,
                        ),
                    ):
                        evidence = (
                            publication_module._retire_owned_descriptors(
                                (owned,)
                            )
                        )
                    record = evidence.records[0]
                    self.assertIs(
                        record.observation,
                        DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
                    )
                    self.assertEqual(fstat_calls, [descriptor])
                    self.assertEqual(close_calls, [descriptor])
                    self.assertEqual(
                        record.error_errno,
                        failure.errno if isinstance(failure, OSError) else None,
                    )
                finally:
                    os.close(descriptor)

    def test_batch_continues_after_anomaly_and_rejects_duplicate_descriptor(
        self,
    ) -> None:
        with synthetic_publication_root() as root:
            (root / "input.bin").write_bytes(b"input")
            raw_parent = os.open(
                root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            entry = publication_module._open_read_file_at(
                "input.bin",
                raw_parent,
                role="traversal_entry",
            )
            parent = publication_module._open_directory_at(
                ".",
                raw_parent,
                role="traversal_parent",
            )
            entry_owned = publication_module._owned_descriptor(
                entry,
                "traversal_entry",
            )
            parent_owned = publication_module._owned_descriptor(
                parent,
                "traversal_parent",
            )
            real_close = os.close
            close_calls: list[int] = []

            def close_one_fails(value: int) -> None:
                close_calls.append(value)
                if value == entry:
                    raise OSError(errno.EIO, "synthetic first anomaly")
                real_close(value)

            try:
                with patch.object(
                    publication_module.os,
                    "close",
                    side_effect=close_one_fails,
                ):
                    evidence = publication_module._retire_owned_descriptors(
                        (parent_owned, entry_owned)
                    )
                self.assertEqual(close_calls, [entry, parent])
                self.assertEqual(
                    tuple(record.role for record in evidence.records),
                    ("traversal_entry", "traversal_parent"),
                )
                self.assertIs(
                    evidence.records[0].observation,
                    DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
                )
                self.assertIs(
                    evidence.records[1].observation,
                    DescriptorRetirementObservation.CLOSED,
                )
                with self.assertRaises(OSError):
                    os.fstat(parent)
                os.fstat(entry)
            finally:
                real_close(entry)
                os.close(raw_parent)

        descriptor = publication_module._admit_opened_descriptor(
            os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC),
            role="traversal_entry",
        )
        owned = publication_module._owned_descriptor(
            descriptor,
            "traversal_entry",
        )
        duplicate = publication_module._OwnedDescriptor(
            descriptor=owned.descriptor,
            generation=owned.generation,
            role="traversal_parent",
            admitted_identity=owned.admitted_identity,
        )
        try:
            with (
                patch.object(
                    publication_module,
                    "_fstat_descriptor",
                    side_effect=AssertionError("duplicate descriptor inspected"),
                ),
                patch.object(
                    publication_module.os,
                    "close",
                    side_effect=AssertionError("duplicate descriptor closed"),
                ),
                self.assertRaises(RuntimeError),
            ):
                publication_module._retire_owned_descriptors(
                    (owned, duplicate)
                )
        finally:
            publication_module._close_descriptor(descriptor)

    def test_terminal_descriptor_admission_failures_transport_once(
        self,
    ) -> None:
        cases = (
            ("publish-parent", "operation_parent"),
            ("cleanup-parent", "operation_parent"),
            ("cleanup-staging", "operation_staging"),
        )
        for case, failed_role in cases:
            with self.subTest(case=case):
                destination = portable(f"{case}.bin")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        staged.write(b"payload")
                        if case == "publish-parent":
                            staged.seal()
                        state = publication_module._staging_state(staged)
                        original_register = (
                            publication_module._register_fresh_descriptor
                        )
                        real_close = os.close
                        failed_descriptor = -1
                        role_descriptors: dict[str, int] = {}
                        close_calls: list[int] = []
                        target_close_calls: list[int] = []

                        def register_then_fail(
                            descriptor: int,
                            *,
                            role: str,
                        ) -> None:
                            nonlocal failed_descriptor
                            original_register(descriptor, role=role)
                            role_descriptors[role] = descriptor
                            if role == failed_role:
                                failed_descriptor = descriptor
                                raise RuntimeError(
                                    "synthetic terminal descriptor "
                                    "admission failure"
                                )

                        def fail_target_close(descriptor: int) -> None:
                            close_calls.append(descriptor)
                            if descriptor == failed_descriptor:
                                target_close_calls.append(descriptor)
                                raise OSError(
                                    errno.EIO,
                                    "synthetic terminal close uncertainty",
                                )
                            real_close(descriptor)

                        try:
                            with (
                                patch.object(
                                    publication_module,
                                    "_register_fresh_descriptor",
                                    side_effect=register_then_fail,
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=fail_target_close,
                                ),
                            ):
                                if case == "publish-parent":
                                    with self.assertRaises(
                                        StagingAuthorityError
                                    ) as caught:
                                        publish_completed_file(
                                            trusted,
                                            staged,
                                        )
                                else:
                                    with self.assertRaises(
                                        StagingCleanupError
                                    ) as caught:
                                        cleanup_owned_staging(
                                            trusted,
                                            staged,
                                        )
                            retirement = caught.exception.retirement_evidence
                            self.assertIsNotNone(retirement)
                            assert retirement is not None
                            self.assertIn(
                                failed_role,
                                tuple(
                                    record.role
                                    for record in retirement.records
                                ),
                            )
                            self.assertEqual(
                                len(target_close_calls),
                                1,
                            )
                            if case == "publish-parent":
                                self.assertIs(
                                    staged.state,
                                    StagingState.SEALED,
                                )
                                self.assertFalse(
                                    state.publication_attempted
                                )
                                self.assertTrue(
                                    state.retirement_batch_pending
                                )
                            else:
                                self.assertIs(
                                    staged.state,
                                    StagingState.RETIRED,
                                )
                                self.assertTrue(state.cleanup_attempted)
                                self.assertEqual(
                                    (
                                        root
                                        / staging_leaf(destination)
                                    ).read_bytes(),
                                    b"payload",
                                )
                                with (
                                    patch.object(
                                        publication_module,
                                        "_open_publication_parent",
                                    ) as opened,
                                    self.assertRaises(
                                        StagingLifecycleError
                                    ),
                                ):
                                    cleanup_owned_staging(
                                        trusted,
                                        staged,
                                    )
                                opened.assert_not_called()
                        finally:
                            if failed_descriptor >= 0:
                                real_close(failed_descriptor)

                        if case == "publish-parent":
                            result = publish_completed_file(
                                trusted,
                                staged,
                            )
                            self.assertIs(
                                result.state,
                                PublicationState.COMMITTED_DURABLE,
                            )
                        self.assertFalse(
                            context.__exit__(None, None, None)
                        )

    def test_write_file_retires_local_descriptor_batch_without_masking(
        self,
    ) -> None:
        class CallerIteratorFailure(Exception):
            pass

        for caller_failure in (False, True):
            with self.subTest(caller_failure=caller_failure):
                destination = portable(
                    f"local-batch-{int(caller_failure)}"
                )
                sentinel = CallerIteratorFailure("caller iterator failed")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        original_admit = (
                            publication_module._admit_opened_descriptor
                        )
                        real_close = os.close
                        role_descriptors: dict[str, int] = {}
                        close_calls: list[int] = []

                        def capture_admission(
                            descriptor: int,
                            *,
                            role: str,
                        ) -> int:
                            admitted = original_admit(
                                descriptor,
                                role=role,
                            )
                            if role in {
                                "traversal_entry",
                                "traversal_parent",
                            }:
                                role_descriptors[role] = admitted
                            return admitted

                        def fail_entry_close(descriptor: int) -> None:
                            close_calls.append(descriptor)
                            if descriptor == role_descriptors.get(
                                "traversal_entry"
                            ):
                                raise OSError(
                                    errno.EIO,
                                    "synthetic entry close uncertainty",
                                )
                            real_close(descriptor)

                        def chunks():
                            yield b"payload"
                            if caller_failure:
                                raise sentinel

                        uncertain_descriptor = -1
                        try:
                            with (
                                patch.object(
                                    publication_module,
                                    "_admit_opened_descriptor",
                                    side_effect=capture_admission,
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=fail_entry_close,
                                ),
                            ):
                                if caller_failure:
                                    with self.assertRaises(
                                        BaseExceptionGroup
                                    ) as caught:
                                        staged.write_file(
                                            portable("payload.bin"),
                                            chunks(),
                                        )
                                    self.assertIs(
                                        caught.exception.exceptions[0],
                                        sentinel,
                                    )
                                    retirement_error = (
                                        caught.exception.exceptions[1]
                                    )
                                else:
                                    with self.assertRaises(
                                        DescriptorRetirementError
                                    ) as caught:
                                        staged.write_file(
                                            portable("payload.bin"),
                                            chunks(),
                                        )
                                    retirement_error = caught.exception
                            self.assertIsInstance(
                                retirement_error,
                                DescriptorRetirementError,
                            )
                            self.assertEqual(
                                retirement_error.operation,
                                "write_file",
                            )
                            self.assertIs(
                                retirement_error.state,
                                StagingState.RETIRED,
                            )
                            self.assertEqual(
                                tuple(
                                    record.role
                                    for record
                                    in retirement_error
                                    .retirement_evidence.records
                                ),
                                (
                                    "traversal_entry",
                                    "traversal_parent",
                                    "handle_staging",
                                    "handle_parent",
                                ),
                            )
                            self.assertEqual(
                                close_calls[:2],
                                [
                                    role_descriptors[
                                        "traversal_entry"
                                    ],
                                    role_descriptors[
                                        "traversal_parent"
                                    ],
                                ],
                            )
                            uncertain_descriptor = role_descriptors[
                                "traversal_entry"
                            ]
                            self.assertEqual(
                                close_calls.count(uncertain_descriptor),
                                1,
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            with (
                                patch.object(
                                    publication_module,
                                    "_fstat_descriptor",
                                    side_effect=AssertionError(
                                        "retired context inspected an old "
                                        "descriptor"
                                    ),
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=AssertionError(
                                        "retired context retried a close"
                                    ),
                                ),
                            ):
                                self.assertFalse(
                                    context.__exit__(None, None, None)
                                )
                        finally:
                            if uncertain_descriptor >= 0:
                                real_close(uncertain_descriptor)

    def test_population_and_handle_retirement_anomalies_use_one_batch(
        self,
    ) -> None:
        class CallerIteratorFailure(Exception):
            pass

        destination = portable("combined-retirement-batch")
        sentinel = CallerIteratorFailure("caller iterator failed")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                state = publication_module._staging_state(staged)
                assert state is not None
                handle_staging = state.staging_descriptor
                original_admit = publication_module._admit_opened_descriptor
                real_close = os.close
                traversal_entry = -1
                uncertain_descriptors: set[int] = {handle_staging}
                close_calls: list[int] = []

                def capture_admission(
                    descriptor: int,
                    *,
                    role: str,
                ) -> int:
                    nonlocal traversal_entry
                    admitted = original_admit(descriptor, role=role)
                    if role == "traversal_entry":
                        traversal_entry = admitted
                        uncertain_descriptors.add(admitted)
                    return admitted

                def fail_two_closes(descriptor: int) -> None:
                    close_calls.append(descriptor)
                    if descriptor in uncertain_descriptors:
                        raise OSError(
                            errno.EIO,
                            "synthetic combined close uncertainty",
                        )
                    real_close(descriptor)

                def chunks():
                    yield b"payload"
                    raise sentinel

                try:
                    with (
                        patch.object(
                            publication_module,
                            "_admit_opened_descriptor",
                            side_effect=capture_admission,
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_two_closes,
                        ),
                        self.assertRaises(BaseExceptionGroup) as caught,
                    ):
                        staged.write_file(
                            portable("payload.bin"),
                            chunks(),
                        )
                    self.assertEqual(len(caught.exception.exceptions), 2)
                    self.assertIs(caught.exception.exceptions[0], sentinel)
                    retirement_error = caught.exception.exceptions[1]
                    self.assertIs(
                        type(retirement_error),
                        DescriptorRetirementError,
                    )
                    self.assertEqual(
                        tuple(
                            record.role
                            for record in retirement_error
                            .retirement_evidence.records
                        ),
                        (
                            "traversal_entry",
                            "traversal_parent",
                            "handle_staging",
                            "handle_parent",
                        ),
                    )
                    records = {
                        record.role: record
                        for record in retirement_error
                        .retirement_evidence.records
                    }
                    for role in ("traversal_entry", "handle_staging"):
                        self.assertIs(
                            records[role].observation,
                            DescriptorRetirementObservation
                            .CLOSE_OUTCOME_UNCERTAIN,
                        )
                    for role in ("traversal_parent", "handle_parent"):
                        self.assertIs(
                            records[role].observation,
                            DescriptorRetirementObservation.CLOSED,
                        )
                    self.assertEqual(
                        close_calls.count(traversal_entry),
                        1,
                    )
                    self.assertEqual(close_calls.count(handle_staging), 1)
                    self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertFalse(context.__exit__(None, None, None))
                finally:
                    for descriptor in (traversal_entry, handle_staging):
                        if descriptor >= 0:
                            real_close(descriptor)

    def test_private_error_provenance_attaches_once_and_rejects_foreign_errors(
        self,
    ) -> None:
        owner = object()
        other_owner = object()
        evidence = DescriptorRetirementValueTests.evidence()

        caller_error = PublicationValidationError(
            "caller",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=portable("result"),
        )
        self.assertFalse(
            publication_module._attach_retirement_evidence(
                caller_error,
                owner,
                evidence,
            )
        )
        self.assertIsNone(caller_error.retirement_evidence)

        with publication_module._error_provenance_scope(owner):
            externally_allocated_error = PublicationValidationError(
                "internal",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=portable("result"),
            )
        self.assertFalse(
            publication_module._attach_retirement_evidence(
                externally_allocated_error,
                owner,
                evidence,
            )
        )
        internal_error = publication_module._allocate_h2c1_error(
            owner,
            PublicationValidationError,
            "internal",
            state=PublicationState.NOT_COMMITTED,
            evidence=None,
            destination=portable("result"),
        )
        self.assertIs(type(internal_error), PublicationValidationError)
        self.assertFalse(
            publication_module._attach_retirement_evidence(
                internal_error,
                other_owner,
                evidence,
            )
        )
        self.assertTrue(
            publication_module._attach_retirement_evidence(
                internal_error,
                owner,
                evidence,
            )
        )
        self.assertIs(internal_error.retirement_evidence, evidence)
        self.assertFalse(
            publication_module._attach_retirement_evidence(
                internal_error,
                owner,
                evidence,
            )
        )

        class HostilePublicationError(PublicationValidationError):
            pass

        with publication_module._error_provenance_scope(owner):
            hostile = HostilePublicationError(
                "hostile",
                state=PublicationState.NOT_COMMITTED,
                evidence=None,
                destination=portable("result"),
            )
        self.assertFalse(
            publication_module._attach_retirement_evidence(
                hostile,
                owner,
                evidence,
            )
        )
        self.assertIsNone(hostile.retirement_evidence)

    def test_terminal_error_subtypes_keep_outcomes_during_retirement(
        self,
    ) -> None:
        publication_cases = (
            (PublicationError, PublicationState.NOT_COMMITTED, "definite"),
            (
                PublicationValidationError,
                PublicationState.NOT_COMMITTED,
                "definite",
            ),
            (
                PublicationCapabilityError,
                PublicationState.NOT_COMMITTED,
                "definite",
            ),
            (
                PublicationCollisionError,
                PublicationState.NOT_COMMITTED,
                "collision",
            ),
            (
                safety.PublicationDurabilityError,
                PublicationState.COMMITTED_DURABILITY_UNCERTAIN,
                "committed_uncertain",
            ),
            (
                PublicationOutcomeUncertainError,
                PublicationState.COMMIT_OUTCOME_UNCERTAIN,
                "outcome_uncertain",
            ),
            (
                PublicationNamespaceConflictError,
                PublicationState.COMMITTED_DURABLE,
                "namespace_conflict",
            ),
            (
                PublicationNamespaceUncertainError,
                PublicationState.COMMITTED_DURABLE,
                "namespace_uncertain",
            ),
        )
        for index, (error_type, outcome, evidence_kind) in enumerate(
            publication_cases
        ):
            with self.subTest(error=error_type.__name__):
                destination = portable(f"publication-error-{index}")
                with synthetic_publication_root() as root:
                    foreign_path = root / f"foreign-{index}"
                    foreign_path.write_bytes(b"foreign")
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        state = publication_module._staging_state(staged)
                        assert state is not None
                        identity = state.root_identity.public
                        foreign = publication_module._make_entry_identity(
                            foreign_path.stat(follow_symlinks=False)
                        )
                        if evidence_kind == "namespace_conflict":
                            namespace = (
                                publication_module
                                ._make_namespace_evidence(
                                    publication_module
                                    ._NamespaceObservation
                                    .COMPLETE_CONFLICT,
                                    reference=destination,
                                    aliases=(
                                        portable(
                                            destination.value.swapcase()
                                        ),
                                    ),
                                )
                            )
                        elif evidence_kind in {
                            "namespace_uncertain",
                            "outcome_uncertain",
                        }:
                            namespace = (
                                publication_module
                                ._make_namespace_evidence(
                                    publication_module
                                    ._NamespaceObservation
                                    .UNINSPECTABLE,
                                    reference=destination,
                                )
                            )
                        else:
                            namespace = (
                                publication_module
                                ._make_namespace_evidence(
                                    publication_module
                                    ._NamespaceObservation.NO_CONFLICT,
                                    reference=destination,
                                )
                            )
                        if evidence_kind.startswith("committed_") or (
                            evidence_kind.startswith("namespace_")
                            and evidence_kind != "outcome_uncertain"
                        ):
                            source_observation = "absent"
                            observed_source = None
                            destination_observation = "exact"
                            observed_destination = identity
                            parent_fsync = (
                                "failed"
                                if evidence_kind == "committed_uncertain"
                                else "succeeded"
                            )
                        elif evidence_kind == "collision":
                            source_observation = "exact"
                            observed_source = identity
                            destination_observation = "foreign"
                            observed_destination = foreign
                            parent_fsync = "succeeded"
                        elif evidence_kind == "outcome_uncertain":
                            source_observation = "uninspectable"
                            observed_source = None
                            destination_observation = "uninspectable"
                            observed_destination = None
                            parent_fsync = "uncertain"
                        else:
                            source_observation = "exact"
                            observed_source = identity
                            destination_observation = "absent"
                            observed_destination = None
                            parent_fsync = "succeeded"
                        original_evidence = (
                            publication_module._make_publication_evidence(
                                staging_identity=identity,
                                source_observation=source_observation,
                                observed_source_identity=observed_source,
                                destination_observation=(
                                    destination_observation
                                ),
                                observed_destination_identity=(
                                    observed_destination
                                ),
                                namespace_evidence=namespace,
                                parent_fsync=parent_fsync,
                                native_errno=None,
                            )
                        )
                        error = publication_module._allocate_h2c1_error(
                            state.error_provenance,
                            error_type,
                            "synthetic terminal publication failure",
                            state=outcome,
                            evidence=original_evidence,
                            destination=destination,
                        )
                        self.assertIs(type(error), error_type)
                        state.lifecycle = (
                            StagingState.PUBLISHED
                            if outcome
                            in {
                                PublicationState.COMMITTED_DURABLE,
                                PublicationState
                                .COMMITTED_DURABILITY_UNCERTAIN,
                            }
                            else (
                                StagingState.RETIRED
                                if outcome
                                is PublicationState
                                .COMMIT_OUTCOME_UNCERTAIN
                                else StagingState.NOT_COMMITTED
                            )
                        )
                        expected_lifecycle = state.lifecycle
                        descriptor = state.staging_descriptor
                        real_close = os.close

                        def fail_staging_close(number: int) -> None:
                            if number == descriptor:
                                raise OSError(
                                    errno.EIO,
                                    "synthetic publication retirement failure",
                                )
                            real_close(number)

                        try:
                            with (
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=fail_staging_close,
                                ),
                                self.assertRaises(error_type) as caught,
                            ):
                                publication_module._retire_with_primary_error(
                                    state,
                                    error,
                                    post_outcome=True,
                                )
                            self.assertIs(caught.exception, error)
                            self.assertIs(caught.exception.state, outcome)
                            self.assertIs(
                                caught.exception.evidence,
                                original_evidence,
                            )
                            self.assertIsNotNone(
                                caught.exception.retirement_evidence
                            )
                            self.assertIs(
                                state.lifecycle,
                                expected_lifecycle,
                            )
                            self.assertFalse(
                                state.retirement_batch_pending
                            )
                            self.assertFalse(
                                context.__exit__(None, None, None)
                            )
                        finally:
                            real_close(descriptor)

        cleanup_cases = (
            (
                StagingCleanupState.NOT_DISCARDED,
                "exact",
                "not_attempted",
            ),
            (
                StagingCleanupState.DISCARDED_DURABILITY_UNCERTAIN,
                "absent",
                "failed",
            ),
            (
                StagingCleanupState.DISCARD_OUTCOME_UNCERTAIN,
                "uninspectable",
                "uncertain",
            ),
        )
        for index, (outcome, root_observation, parent_fsync) in enumerate(
            cleanup_cases
        ):
            with self.subTest(cleanup=outcome.value):
                destination = portable(f"cleanup-error-{index}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        state = publication_module._staging_state(staged)
                        assert state is not None
                        identity = state.root_identity.public
                        namespace = (
                            publication_module._make_namespace_evidence(
                                (
                                    publication_module
                                    ._NamespaceObservation.UNINSPECTABLE
                                    if root_observation == "uninspectable"
                                    else publication_module
                                    ._NamespaceObservation.NO_CONFLICT
                                ),
                                reference=state.staging,
                            )
                        )
                        original_evidence = (
                            publication_module._make_cleanup_evidence(
                                staging_identity=identity,
                                root_observation=root_observation,
                                observed_root_identity=(
                                    identity
                                    if root_observation == "exact"
                                    else None
                                ),
                                remaining_expected_entries=(
                                    0
                                    if root_observation == "exact"
                                    else None
                                ),
                                namespace_evidence=namespace,
                                parent_fsync=parent_fsync,
                                native_errno=None,
                            )
                        )
                        error = publication_module._allocate_h2c1_error(
                            state.error_provenance,
                            StagingCleanupError,
                            "synthetic terminal cleanup failure",
                            state=outcome,
                            evidence=original_evidence,
                            staging=state.staging,
                        )
                        self.assertIs(type(error), StagingCleanupError)
                        state.lifecycle = StagingState.RETIRED
                        descriptor = state.staging_descriptor
                        real_close = os.close

                        def fail_staging_close(number: int) -> None:
                            if number == descriptor:
                                raise OSError(
                                    errno.EIO,
                                    "synthetic cleanup retirement failure",
                                )
                            real_close(number)

                        try:
                            with (
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=fail_staging_close,
                                ),
                                self.assertRaises(
                                    StagingCleanupError
                                ) as caught,
                            ):
                                publication_module._retire_with_primary_error(
                                    state,
                                    error,
                                    post_outcome=True,
                                )
                            self.assertIs(caught.exception, error)
                            self.assertIs(caught.exception.state, outcome)
                            self.assertIs(
                                caught.exception.evidence,
                                original_evidence,
                            )
                            self.assertIsNotNone(
                                caught.exception.retirement_evidence
                            )
                            self.assertIs(
                                state.lifecycle,
                                StagingState.RETIRED,
                            )
                            self.assertFalse(
                                state.retirement_batch_pending
                            )
                            self.assertFalse(
                                context.__exit__(None, None, None)
                            )
                        finally:
                            real_close(descriptor)

    def test_terminal_result_builder_failures_retire_once_and_preserve_outcome(
        self,
    ) -> None:
        class SyntheticResultFailure(Exception):
            pass

        for operation in ("publish", "cleanup"):
            for close_anomaly in (False, True):
                with self.subTest(
                    operation=operation,
                    close_anomaly=close_anomaly,
                ):
                    destination = portable(
                        f"{operation}-result-build-{int(close_anomaly)}"
                    )
                    with synthetic_publication_root() as root:
                        with open_trusted_root(str(root)) as trusted:
                            context = open_exclusive_staged_file(
                                trusted,
                                destination=destination,
                            )
                            staged = context.__enter__()
                            staged.write(b"payload")
                            if operation == "publish":
                                staged.seal()
                            state = publication_module._staging_state(staged)
                            assert state is not None
                            owned_descriptor = state.staging_descriptor
                            primary = SyntheticResultFailure(
                                f"synthetic {operation} result failure"
                            )
                            real_close = os.close
                            close_attempts = 0

                            def optional_close_failure(number: int) -> None:
                                nonlocal close_attempts
                                if (
                                    close_anomaly
                                    and number == owned_descriptor
                                ):
                                    close_attempts += 1
                                    raise OSError(
                                        errno.EIO,
                                        "synthetic terminal close failure",
                                    )
                                real_close(number)

                            builder_name = (
                                "_make_publication_result"
                                if operation == "publish"
                                else "_make_cleanup_result"
                            )
                            terminal_call = (
                                publish_completed_file
                                if operation == "publish"
                                else cleanup_owned_staging
                            )
                            try:
                                with (
                                    patch.object(
                                        publication_module,
                                        builder_name,
                                        side_effect=primary,
                                    ),
                                    patch.object(
                                        publication_module.os,
                                        "close",
                                        side_effect=optional_close_failure,
                                    ),
                                ):
                                    if close_anomaly:
                                        with self.assertRaises(
                                            BaseExceptionGroup
                                        ) as caught:
                                            terminal_call(trusted, staged)
                                        self.assertEqual(
                                            len(
                                                caught.exception.exceptions
                                            ),
                                            2,
                                        )
                                        self.assertIs(
                                            caught.exception.exceptions[0],
                                            primary,
                                        )
                                        retirement_error = (
                                            caught.exception.exceptions[1]
                                        )
                                        self.assertIs(
                                            type(retirement_error),
                                            DescriptorRetirementError,
                                        )
                                        self.assertEqual(
                                            retirement_error.operation,
                                            operation,
                                        )
                                        self.assertIs(
                                            retirement_error.terminal_result.state,
                                            (
                                                PublicationState
                                                .COMMITTED_DURABLE
                                                if operation == "publish"
                                                else StagingCleanupState
                                                .DISCARDED_DURABLE
                                            ),
                                        )
                                        self.assertEqual(close_attempts, 1)
                                    else:
                                        with self.assertRaises(
                                            SyntheticResultFailure
                                        ) as caught:
                                            terminal_call(trusted, staged)
                                        self.assertIs(
                                            caught.exception,
                                            primary,
                                        )
                                self.assertIs(
                                    staged.state,
                                    (
                                        StagingState.PUBLISHED
                                        if operation == "publish"
                                        else StagingState.DISCARDED
                                    ),
                                )
                                self.assertFalse(
                                    state.retirement_batch_pending
                                )
                                with (
                                    patch.object(
                                        publication_module,
                                        "_fstat_descriptor",
                                        side_effect=AssertionError(
                                            "terminal exit inspected a "
                                            "consumed descriptor"
                                        ),
                                    ),
                                    patch.object(
                                        publication_module.os,
                                        "close",
                                        side_effect=AssertionError(
                                            "terminal exit retried close"
                                        ),
                                    ),
                                ):
                                    self.assertFalse(
                                        context.__exit__(None, None, None)
                                    )
                            finally:
                                if close_anomaly:
                                    real_close(owned_descriptor)
                            if operation == "publish":
                                self.assertEqual(
                                    (root / destination.value).read_bytes(),
                                    b"payload",
                                )
                            else:
                                self.assertFalse(
                                    (root / staging_leaf(destination)).exists()
                                )

    def test_caller_chunk_failure_stays_primary_with_retirement_anomaly(
        self,
    ) -> None:
        class SyntheticChunkFailure(Exception):
            pass

        for close_anomaly in (False, True):
            with self.subTest(close_anomaly=close_anomaly):
                destination = portable(
                    f"caller-chunk-{int(close_anomaly)}"
                )
                primary = SyntheticChunkFailure("caller chunk failure")

                def chunks():
                    yield b"accepted"
                    raise primary

                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        state = publication_module._staging_state(staged)
                        assert state is not None
                        handle_descriptor = state.staging_descriptor
                        created_descriptor = -1
                        real_create = publication_module._create_file_at
                        real_close = os.close

                        def record_created_descriptor(*args, **kwargs):
                            nonlocal created_descriptor
                            created_descriptor = real_create(*args, **kwargs)
                            return created_descriptor

                        def optional_close_failure(number: int) -> None:
                            if close_anomaly and number in {
                                created_descriptor,
                                handle_descriptor,
                            }:
                                raise OSError(
                                    errno.EIO,
                                    "synthetic population close failure",
                                )
                            real_close(number)

                        try:
                            with (
                                patch.object(
                                    publication_module,
                                    "_create_file_at",
                                    side_effect=record_created_descriptor,
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=optional_close_failure,
                                ),
                            ):
                                if close_anomaly:
                                    with self.assertRaises(
                                        BaseExceptionGroup
                                    ) as caught:
                                        staged.write_file(
                                            portable("partial.bin"),
                                            chunks(),
                                        )
                                    self.assertIs(
                                        caught.exception.exceptions[0],
                                        primary,
                                    )
                                    retirement_error = (
                                        caught.exception.exceptions[1]
                                    )
                                    self.assertIs(
                                        type(retirement_error),
                                        DescriptorRetirementError,
                                    )
                                    self.assertEqual(
                                        tuple(
                                            record.role
                                            for record in (
                                                retirement_error
                                                .retirement_evidence.records
                                            )
                                        ),
                                        (
                                            "traversal_entry",
                                            "traversal_parent",
                                            "handle_staging",
                                            "handle_parent",
                                        ),
                                    )
                                else:
                                    with self.assertRaises(
                                        SyntheticChunkFailure
                                    ) as caught:
                                        staged.write_file(
                                            portable("partial.bin"),
                                            chunks(),
                                        )
                                    self.assertIs(
                                        caught.exception,
                                        primary,
                                    )
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                            self.assertFalse(
                                state.retirement_batch_pending
                            )
                            self.assertFalse(
                                context.__exit__(None, None, None)
                            )
                        finally:
                            if close_anomaly:
                                real_close(created_descriptor)
                                real_close(handle_descriptor)
                        self.assertEqual(
                            (
                                root
                                / staging_leaf(destination)
                                / "partial.bin"
                            ).read_bytes(),
                            b"accepted",
                        )

    def test_user_chunk_callbacks_cannot_inherit_private_error_provenance(
        self,
    ) -> None:
        created: list[PublicationValidationError] = []

        class CallerChunks:
            def __iter__(self):
                return self

            def __next__(self):
                if created:
                    raise StopIteration
                error = PublicationValidationError(
                    "caller-created during iteration",
                    state=PublicationState.NOT_COMMITTED,
                    evidence=None,
                    destination=portable("caller"),
                )
                created.append(error)
                return b"payload"

        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=portable("result"),
                ) as staged:
                    staged.write_file(
                        portable("item.bin"),
                        CallerChunks(),
                    )
                    self.assertEqual(len(created), 1)
                    self.assertIsNone(
                        getattr(created[0], "_h2c1_provenance", None)
                    )
                    cleanup_owned_staging(trusted, staged)

    def test_body_exception_precedes_context_retirement_failure(self) -> None:
        class SyntheticBodyFailure(Exception):
            pass

        destination = portable("abandoned.bin")
        body_error = SyntheticBodyFailure("body")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                real_close = os.close
                owned_descriptor = -1

                def fail_staging_close(descriptor: int) -> None:
                    if descriptor == owned_descriptor:
                        raise OSError(errno.EIO, "synthetic close failure")
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_staging_close,
                        ),
                        self.assertRaises(BaseExceptionGroup) as caught,
                    ):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"payload")
                            owned_descriptor = (
                                publication_module._staging_state(
                                    staged
                                ).staging_descriptor
                            )
                            raise body_error
                    self.assertEqual(len(caught.exception.exceptions), 2)
                    self.assertIs(caught.exception.exceptions[0], body_error)
                    retirement = caught.exception.exceptions[1]
                    self.assertIsInstance(
                        retirement,
                        DescriptorRetirementError,
                    )
                    self.assertEqual(
                        retirement.operation,
                        "context_exit",
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                finally:
                    if owned_descriptor >= 0:
                        real_close(owned_descriptor)

    def test_bound_root_error_keeps_subtype_during_exit_retirement(
        self,
    ) -> None:
        destination = portable("wrong-root-body.bin")
        with (
            synthetic_publication_root() as root,
            synthetic_publication_root() as foreign_root,
        ):
            with (
                open_trusted_root(str(root)) as trusted,
                open_trusted_root(str(foreign_root)) as foreign_trusted,
            ):
                real_close = os.close
                owned_descriptor = -1

                def fail_staging_close(descriptor: int) -> None:
                    if descriptor == owned_descriptor:
                        raise OSError(errno.EIO, "synthetic close failure")
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_staging_close,
                        ),
                        self.assertRaises(
                            StagingAuthorityError
                        ) as caught,
                    ):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"payload")
                            state = publication_module._staging_state(staged)
                            assert state is not None
                            owned_descriptor = state.staging_descriptor
                            cleanup_owned_staging(
                                foreign_trusted,
                                staged,
                            )
                    self.assertIs(
                        type(caught.exception),
                        StagingAuthorityError,
                    )
                    self.assertEqual(
                        caught.exception.operation,
                        "cleanup",
                    )
                    self.assertIsNotNone(
                        caught.exception.retirement_evidence
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                finally:
                    if owned_descriptor >= 0:
                        real_close(owned_descriptor)

    def test_matching_seal_collision_error_keeps_subtype_and_gets_evidence(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            (root / destination.value).write_bytes(b"foreign")
            with open_trusted_root(str(root)) as trusted:
                real_close = os.close
                owned_descriptor = -1

                def fail_staging_close(descriptor: int) -> None:
                    if descriptor == owned_descriptor:
                        raise OSError(errno.EIO, "synthetic close failure")
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_staging_close,
                        ),
                        self.assertRaises(
                            PublicationCollisionError
                        ) as caught,
                    ):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"owned")
                            owned_descriptor = (
                                publication_module._staging_state(
                                    staged
                                ).staging_descriptor
                            )
                            staged.seal()
                    self.assertIs(
                        caught.exception.state,
                        PublicationState.NOT_COMMITTED,
                    )
                    self.assertIsNotNone(
                        caught.exception.retirement_evidence
                    )
                    self.assertIs(staged.state, StagingState.NOT_COMMITTED)
                    state = publication_module._staging_state(staged)
                    self.assertFalse(state.retirement_batch_pending)
                    self.assertFalse(state.cleanup_attempted)
                    real_close(owned_descriptor)
                    owned_descriptor = -1
                    result = cleanup_owned_staging(trusted, staged)
                    self.assertIs(
                        result.state,
                        StagingCleanupState.DISCARDED_DURABLE,
                    )
                finally:
                    if owned_descriptor >= 0:
                        real_close(owned_descriptor)

    def test_terminal_not_committed_consumes_batch_and_exit_is_noop(
        self,
    ) -> None:
        for close_anomaly in (False, True):
            with self.subTest(close_anomaly=close_anomaly):
                destination = portable(
                    f"not-committed-{int(close_anomaly)}.bin"
                )
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        staged.write(b"payload")
                        staged.seal()
                        state = publication_module._staging_state(staged)
                        owned_descriptor = state.staging_descriptor
                        real_close = os.close

                        def optional_close_failure(descriptor: int) -> None:
                            if close_anomaly and descriptor == owned_descriptor:
                                raise OSError(
                                    errno.EIO,
                                    "synthetic terminal close failure",
                                )
                            real_close(descriptor)

                        try:
                            with (
                                patch.object(
                                    publication_module,
                                    "_native_no_replace",
                                    return_value=(-1, errno.EAGAIN),
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=optional_close_failure,
                                ),
                                self.assertRaises(PublicationError) as caught,
                            ):
                                publish_completed_file(trusted, staged)
                            self.assertIs(
                                caught.exception.state,
                                PublicationState.NOT_COMMITTED,
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.NOT_COMMITTED,
                            )
                            self.assertFalse(state.retirement_batch_pending)
                            self.assertEqual(state.staging_descriptor, -1)
                            self.assertEqual(state.parent_descriptor, -1)
                            if close_anomaly:
                                self.assertIsNotNone(
                                    caught.exception.retirement_evidence
                                )
                                real_close(owned_descriptor)
                                owned_descriptor = -1
                            else:
                                self.assertIsNone(
                                    caught.exception.retirement_evidence
                                )

                            with (
                                patch.object(
                                    publication_module,
                                    "_fstat_descriptor",
                                    side_effect=AssertionError(
                                        "consumed exit inspected a descriptor"
                                    ),
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=AssertionError(
                                        "consumed exit closed a descriptor"
                                    ),
                                ),
                            ):
                                self.assertFalse(
                                    context.__exit__(None, None, None)
                                )
                            self.assertIs(
                                staged.state,
                                StagingState.NOT_COMMITTED,
                            )
                            cleanup = cleanup_owned_staging(trusted, staged)
                            self.assertIs(
                                cleanup.state,
                                StagingCleanupState.DISCARDED_DURABLE,
                            )
                        finally:
                            if owned_descriptor >= 0 and close_anomaly:
                                real_close(owned_descriptor)

    def test_pending_not_committed_cleanup_survives_every_retirement_observation(
        self,
    ) -> None:
        cases = (
            ("closed", None),
            (
                "already_absent",
                DescriptorRetirementObservation.ALREADY_ABSENT,
            ),
            (
                "generation_mismatch",
                DescriptorRetirementObservation.FOREIGN_PRESERVED,
            ),
            (
                "foreign_identity",
                DescriptorRetirementObservation.FOREIGN_PRESERVED,
            ),
            (
                "uninspectable",
                DescriptorRetirementObservation.UNINSPECTABLE,
            ),
            (
                "close_uncertain",
                DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
            ),
        )
        for index, (case, expected_observation) in enumerate(cases):
            with self.subTest(case=case):
                destination = portable(f"retirement-cleanup-{index}.bin")
                with synthetic_publication_root() as root:
                    (root / destination.value).write_bytes(b"foreign")
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        staged.write(b"owned")
                        with self.assertRaises(PublicationCollisionError):
                            staged.seal()
                        state = publication_module._staging_state(staged)
                        assert state is not None
                        ledger = state.tree
                        owned_descriptor = state.staging_descriptor
                        replacement_descriptor = -1
                        managed_replacement = False
                        real_close = os.close
                        real_fstat = publication_module._fstat_descriptor
                        close_calls: list[int] = []
                        try:
                            if case == "already_absent":
                                real_close(owned_descriptor)
                            elif case == "generation_mismatch":
                                real_close(owned_descriptor)
                                replacement_descriptor = (
                                    publication_module._open_read_file_at(
                                        state.staging.value,
                                        state.parent_descriptor,
                                        role="traversal_entry",
                                    )
                                )
                                self.assertEqual(
                                    replacement_descriptor,
                                    owned_descriptor,
                                )
                                managed_replacement = True
                            elif case == "foreign_identity":
                                real_close(owned_descriptor)
                                replacement_descriptor = os.open(
                                    os.devnull,
                                    os.O_RDONLY | os.O_CLOEXEC,
                                )
                                self.assertEqual(
                                    replacement_descriptor,
                                    owned_descriptor,
                                )

                            def observed_fstat(number: int):
                                if (
                                    number == owned_descriptor
                                    and case == "generation_mismatch"
                                ):
                                    raise AssertionError(
                                        "generation mismatch inspected "
                                        "foreign descriptor"
                                    )
                                if (
                                    number == owned_descriptor
                                    and case == "uninspectable"
                                ):
                                    raise OSError(
                                        errno.EIO,
                                        "synthetic fstat failure",
                                    )
                                return real_fstat(number)

                            def observed_close(number: int) -> None:
                                close_calls.append(number)
                                if (
                                    number == owned_descriptor
                                    and case == "close_uncertain"
                                ):
                                    raise OSError(
                                        errno.EIO,
                                        "synthetic close uncertainty",
                                    )
                                if (
                                    number == owned_descriptor
                                    and case
                                    in {
                                        "already_absent",
                                        "generation_mismatch",
                                        "foreign_identity",
                                        "uninspectable",
                                    }
                                ):
                                    raise AssertionError(
                                        "unsafe descriptor close"
                                    )
                                real_close(number)

                            with (
                                patch.object(
                                    publication_module,
                                    "_fstat_descriptor",
                                    side_effect=observed_fstat,
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=observed_close,
                                ),
                            ):
                                if expected_observation is None:
                                    self.assertFalse(
                                        context.__exit__(
                                            None,
                                            None,
                                            None,
                                        )
                                    )
                                    retirement = None
                                else:
                                    with self.assertRaises(
                                        DescriptorRetirementError
                                    ) as caught:
                                        context.__exit__(
                                            None,
                                            None,
                                            None,
                                        )
                                    retirement = (
                                        caught.exception
                                        .retirement_evidence
                                    )

                            if expected_observation is not None:
                                assert retirement is not None
                                target = next(
                                    record
                                    for record in retirement.records
                                    if record.role == "handle_staging"
                                )
                                self.assertIs(
                                    target.observation,
                                    expected_observation,
                                )
                                if case == "generation_mismatch":
                                    self.assertIsNone(
                                        target.observed_identity
                                    )
                                if case == "foreign_identity":
                                    self.assertIsNotNone(
                                        target.observed_identity
                                    )
                                    self.assertNotEqual(
                                        target.observed_identity,
                                        target.admitted_identity,
                                    )
                            self.assertEqual(
                                close_calls.count(owned_descriptor),
                                (
                                    1
                                    if case
                                    in {"closed", "close_uncertain"}
                                    else 0
                                ),
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.NOT_COMMITTED,
                            )
                            self.assertFalse(
                                state.retirement_batch_pending
                            )
                            self.assertIs(state.tree, ledger)
                            self.assertFalse(state.cleanup_attempted)
                            result = cleanup_owned_staging(
                                trusted,
                                staged,
                            )
                            self.assertIs(
                                result.state,
                                StagingCleanupState.DISCARDED_DURABLE,
                            )
                            self.assertIs(
                                state.lifecycle,
                                StagingState.DISCARDED,
                            )
                            self.assertEqual(
                                (root / destination.value).read_bytes(),
                                b"foreign",
                            )
                        finally:
                            if replacement_descriptor >= 0:
                                if managed_replacement:
                                    publication_module._close_descriptor(
                                        replacement_descriptor
                                    )
                                else:
                                    real_close(replacement_descriptor)
                            elif case in {
                                "uninspectable",
                                "close_uncertain",
                            }:
                                real_close(owned_descriptor)

    def test_pending_not_committed_body_error_precedes_retirement_anomaly(
        self,
    ) -> None:
        class SyntheticBodyFailure(Exception):
            pass

        destination = portable("pending-not-committed.bin")
        body_error = SyntheticBodyFailure("body")
        with synthetic_publication_root() as root:
            (root / destination.value).write_bytes(b"foreign")
            with open_trusted_root(str(root)) as trusted:
                real_close = os.close
                owned_descriptor = -1
                close_attempts = 0

                def fail_owned_close(descriptor: int) -> None:
                    nonlocal close_attempts
                    if descriptor == owned_descriptor:
                        close_attempts += 1
                        raise OSError(
                            errno.EIO,
                            "synthetic pending close uncertainty",
                        )
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_owned_close,
                        ),
                        self.assertRaises(BaseExceptionGroup) as caught,
                    ):
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"payload")
                            state = publication_module._staging_state(
                                staged
                            )
                            owned_descriptor = state.staging_descriptor
                            with self.assertRaises(
                                PublicationCollisionError
                            ):
                                staged.seal()
                            self.assertTrue(
                                state.retirement_batch_pending
                            )
                            raise body_error
                    self.assertEqual(
                        len(caught.exception.exceptions),
                        2,
                    )
                    self.assertIs(
                        caught.exception.exceptions[0],
                        body_error,
                    )
                    retirement = caught.exception.exceptions[1]
                    self.assertIsInstance(
                        retirement,
                        DescriptorRetirementError,
                    )
                    self.assertIs(
                        retirement.state,
                        StagingState.NOT_COMMITTED,
                    )
                    self.assertEqual(
                        retirement.operation,
                        "context_exit",
                    )
                    self.assertEqual(close_attempts, 1)
                    self.assertIs(
                        staged.state,
                        StagingState.NOT_COMMITTED,
                    )
                    self.assertFalse(state.retirement_batch_pending)
                    self.assertFalse(state.cleanup_attempted)
                    real_close(owned_descriptor)
                    owned_descriptor = -1
                    cleanup = cleanup_owned_staging(trusted, staged)
                    self.assertIs(
                        cleanup.state,
                        StagingCleanupState.DISCARDED_DURABLE,
                    )
                finally:
                    if owned_descriptor >= 0:
                        real_close(owned_descriptor)
            self.assertEqual(
                (root / destination.value).read_bytes(),
                b"foreign",
            )

    def test_consumed_not_committed_body_error_has_no_second_retirement(
        self,
    ) -> None:
        class SyntheticBodyFailure(Exception):
            pass

        destination = portable("consumed-not-committed.bin")
        body_error = SyntheticBodyFailure("body")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with self.assertRaises(SyntheticBodyFailure) as caught:
                    with open_exclusive_staged_file(
                        trusted,
                        destination=destination,
                    ) as staged:
                        staged.write(b"payload")
                        staged.seal()
                        state = publication_module._staging_state(staged)
                        with (
                            patch.object(
                                publication_module,
                                "_native_no_replace",
                                return_value=(-1, errno.EAGAIN),
                            ),
                            self.assertRaises(PublicationError),
                        ):
                            publish_completed_file(trusted, staged)
                        self.assertIs(
                            staged.state,
                            StagingState.NOT_COMMITTED,
                        )
                        self.assertFalse(state.retirement_batch_pending)
                        with (
                            patch.object(
                                publication_module,
                                "_fstat_descriptor",
                                side_effect=AssertionError(
                                    "consumed exit inspected a descriptor"
                                ),
                            ),
                            patch.object(
                                publication_module.os,
                                "close",
                                side_effect=AssertionError(
                                    "consumed exit closed a descriptor"
                                ),
                            ),
                        ):
                            raise body_error
                self.assertIs(caught.exception, body_error)
                self.assertIs(
                    staged.state,
                    StagingState.NOT_COMMITTED,
                )
                self.assertFalse(state.cleanup_attempted)
                cleanup = cleanup_owned_staging(trusted, staged)
                self.assertIs(
                    cleanup.state,
                    StagingCleanupState.DISCARDED_DURABLE,
                )

    def test_nested_registration_failure_keeps_primary_and_retirement(
        self,
    ) -> None:
        for operation in ("seal", "publish", "cleanup"):
            with self.subTest(operation=operation):
                destination = portable(f"nested-registration-{operation}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        staged.write_file(
                            portable("payload.bin"),
                            (b"payload",),
                        )
                        if operation == "publish":
                            staged.seal(
                                scope="synthetic-registration-v1"
                            )
                        original_register = (
                            publication_module._register_fresh_descriptor
                        )
                        real_close = os.close
                        failed_descriptor = -1
                        registration_failed = False
                        close_failed = False

                        def register_then_fail(
                            descriptor: int,
                            *,
                            role: str,
                        ):
                            nonlocal failed_descriptor
                            nonlocal registration_failed
                            registration = original_register(
                                descriptor,
                                role=role,
                            )
                            if (
                                role == "traversal_entry"
                                and not registration_failed
                            ):
                                registration_failed = True
                                failed_descriptor = descriptor
                                raise RuntimeError(
                                    "synthetic nested registration failure"
                                )
                            return registration

                        def fail_first_target_close(descriptor: int) -> None:
                            nonlocal close_failed
                            if (
                                descriptor == failed_descriptor
                                and not close_failed
                            ):
                                close_failed = True
                                raise OSError(
                                    errno.EIO,
                                    "synthetic nested close uncertainty",
                                )
                            real_close(descriptor)

                        try:
                            with (
                                patch.object(
                                    publication_module,
                                    "_register_fresh_descriptor",
                                    side_effect=register_then_fail,
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=fail_first_target_close,
                                ),
                            ):
                                if operation == "seal":
                                    with self.assertRaises(
                                        PublicationValidationError
                                    ) as caught:
                                        staged.seal(
                                            scope=(
                                                "synthetic-registration-v1"
                                            )
                                        )
                                elif operation == "publish":
                                    with self.assertRaises(
                                        PublicationError
                                    ) as caught:
                                        publish_completed_directory(
                                            trusted,
                                            staged,
                                        )
                                else:
                                    with self.assertRaises(
                                        StagingCleanupError
                                    ) as caught:
                                        cleanup_owned_staging(
                                            trusted,
                                            staged,
                                        )
                            retirement = (
                                caught.exception.retirement_evidence
                            )
                            self.assertIsNotNone(retirement)
                            assert retirement is not None
                            roles = tuple(
                                record.role
                                for record in retirement.records
                            )
                            self.assertIn("traversal_entry", roles)
                            self.assertEqual(len(roles), len(set(roles)))
                            self.assertTrue(registration_failed)
                            self.assertTrue(close_failed)
                            if operation == "publish":
                                self.assertIs(
                                    staged.state,
                                    StagingState.NOT_COMMITTED,
                                )
                            else:
                                self.assertIs(
                                    staged.state,
                                    StagingState.RETIRED,
                                )
                        finally:
                            if failed_descriptor >= 0:
                                real_close(failed_descriptor)
                        self.assertFalse(
                            context.__exit__(None, None, None)
                        )
                        if operation == "publish":
                            cleanup = cleanup_owned_staging(
                                trusted,
                                staged,
                            )
                            self.assertIs(
                                cleanup.state,
                                StagingCleanupState.DISCARDED_DURABLE,
                            )

    def test_population_registration_close_anomalies_use_public_transport(
        self,
    ) -> None:
        cases = (
            ("mkdir", "traversal_directory"),
            ("write_file", "traversal_entry"),
        )
        for operation, failed_role in cases:
            with self.subTest(operation=operation):
                destination = portable(f"registration-{operation}")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        original_register = (
                            publication_module._register_fresh_descriptor
                        )
                        real_close = os.close
                        failed_descriptor = -1
                        close_calls: list[int] = []
                        failed_once = False

                        def register_then_fail(
                            descriptor: int,
                            *,
                            role: str,
                        ):
                            nonlocal failed_descriptor, failed_once
                            registration = original_register(
                                descriptor,
                                role=role,
                            )
                            if role == failed_role and not failed_once:
                                failed_once = True
                                failed_descriptor = descriptor
                                raise RuntimeError(
                                    "synthetic population registration failure"
                                )
                            return registration

                        def fail_target_close(descriptor: int) -> None:
                            close_calls.append(descriptor)
                            if descriptor == failed_descriptor:
                                raise OSError(
                                    errno.EIO,
                                    "synthetic population close uncertainty",
                                )
                            real_close(descriptor)

                        try:
                            with (
                                patch.object(
                                    publication_module,
                                    "_register_fresh_descriptor",
                                    side_effect=register_then_fail,
                                ),
                                patch.object(
                                    publication_module.os,
                                    "close",
                                    side_effect=fail_target_close,
                                ),
                                self.assertRaises(
                                    PublicationValidationError
                                ) as caught,
                            ):
                                if operation == "mkdir":
                                    staged.mkdir(portable("child"))
                                else:
                                    staged.write_file(
                                        portable("child.bin"),
                                        (b"payload",),
                                    )
                            self.assertIs(
                                type(caught.exception),
                                PublicationValidationError,
                            )
                            self.assertNotIsInstance(
                                caught.exception,
                                BaseExceptionGroup,
                            )
                            retirement = (
                                caught.exception.retirement_evidence
                            )
                            self.assertIsNotNone(retirement)
                            assert retirement is not None
                            roles = tuple(
                                record.role
                                for record in retirement.records
                            )
                            self.assertIn(failed_role, roles)
                            self.assertIn("traversal_parent", roles)
                            self.assertEqual(
                                roles,
                                tuple(
                                    sorted(
                                        roles,
                                        key=(
                                            publication_module
                                            ._RETIREMENT_ROLE_ORDER.index
                                        ),
                                    )
                                ),
                            )
                            record = next(
                                item
                                for item in retirement.records
                                if item.role == failed_role
                            )
                            self.assertIs(
                                record.observation,
                                DescriptorRetirementObservation
                                .CLOSE_OUTCOME_UNCERTAIN,
                            )
                            self.assertEqual(
                                close_calls.count(failed_descriptor),
                                1,
                            )
                            self.assertIs(
                                staged.state,
                                StagingState.RETIRED,
                            )
                        finally:
                            if failed_descriptor >= 0:
                                real_close(failed_descriptor)
                        with (
                            patch.object(
                                publication_module,
                                "_fstat_descriptor",
                                side_effect=AssertionError(
                                    "context exit retried population retirement"
                                ),
                            ),
                            patch.object(
                                publication_module.os,
                                "close",
                                side_effect=AssertionError(
                                    "context exit retried population close"
                                ),
                            ),
                        ):
                            self.assertFalse(
                                context.__exit__(None, None, None)
                            )

    def test_root_cleanup_fsync_reuses_operation_staging_descriptor(
        self,
    ) -> None:
        destination = portable("root-cleanup-role")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_directory(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write_file(
                        portable("payload.bin"),
                        (b"payload",),
                    )
                    original_open = publication_module._open_directory_at
                    acquisitions: list[tuple[str, str]] = []

                    def record_open(
                        name: str,
                        parent_descriptor: int,
                        *,
                        role: str,
                    ) -> int:
                        acquisitions.append((name, role))
                        return original_open(
                            name,
                            parent_descriptor,
                            role=role,
                        )

                    with patch.object(
                        publication_module,
                        "_open_directory_at",
                        side_effect=record_open,
                    ):
                        result = cleanup_owned_staging(
                            trusted,
                            staged,
                        )
                    self.assertIs(
                        result.state,
                        StagingCleanupState.DISCARDED_DURABLE,
                    )
                    self.assertNotIn(
                        (".", "traversal_directory"),
                        acquisitions,
                    )

    def test_finalization_reports_once_through_unraisable_hook(self) -> None:
        destination = portable("finalized.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                parent = publication_module._open_publication_parent(
                    trusted,
                    role="handle_parent",
                )
                parent_owned = publication_module._owned_descriptor(
                    parent,
                    "handle_parent",
                )
                staging = publication_module._create_file_at(
                    staging_leaf(destination),
                    parent,
                    0o600,
                    role="handle_staging",
                )
                staging_owned = publication_module._owned_descriptor(
                    staging,
                    "handle_staging",
                )
                root_identity = publication_module._make_ledger_identity(
                    publication_module._fstat_descriptor(staging)
                )
                state = publication_module._StagingAuthorityState(
                    trusted_root=trusted,
                    backend=publication_module._require_publication_capabilities(),
                    parent_device=root_identity.public.device,
                    parent_inode=os.fstat(parent).st_ino,
                    destination=destination,
                    staging=portable(staging_leaf(destination)),
                    kind="regular_file",
                    lifecycle=StagingState.OPEN,
                    root_identity=root_identity,
                    tree=publication_module._TreeAuthority(
                        root=root_identity,
                        entries=(),
                        inventory=None,
                    ),
                    parent_descriptor=parent,
                    staging_descriptor=staging,
                    parent_generation=parent_owned.generation,
                    staging_generation=staging_owned.generation,
                    parent_retirement_identity=parent_owned.admitted_identity,
                    staging_retirement_identity=staging_owned.admitted_identity,
                )
                class FinalizationOwner:
                    pass

                handle = FinalizationOwner()
                weakref.finalize(
                    handle,
                    publication_module._finalize_staging_state,
                    state,
                )
                captured: list[object] = []
                real_close = os.close

                def fail_staging_close(descriptor: int) -> None:
                    if descriptor == staging:
                        raise OSError(
                            errno.EIO,
                            "synthetic finalizer close failure",
                        )
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_staging_close,
                        ),
                        patch.object(
                            sys,
                            "unraisablehook",
                            side_effect=lambda value: captured.append(value),
                        ),
                    ):
                        del handle
                        gc.collect()
                    self.assertEqual(len(captured), 1)
                    error = captured[0].exc_value
                    self.assertIsInstance(error, DescriptorRetirementError)
                    self.assertEqual(error.operation, "finalization")
                    self.assertIs(error.state, StagingState.RETIRED)
                    self.assertIs(state.lifecycle, StagingState.RETIRED)
                    self.assertFalse(state.retirement_batch_pending)
                    with (
                        patch.object(
                            publication_module,
                            "_fstat_descriptor",
                            side_effect=AssertionError(
                                "finalization retried descriptor inspection"
                            ),
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=AssertionError(
                                "finalization retried descriptor close"
                            ),
                        ),
                    ):
                        publication_module._finalize_staging_state(state)
                finally:
                    real_close(staging)

    def test_public_handle_finalization_reports_once_without_context_retry(
        self,
    ) -> None:
        destination = portable("public-handle-finalization")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                state = publication_module._staging_state(staged)
                assert state is not None
                descriptor = state.staging_descriptor
                staged_ref = weakref.ref(staged)
                captured: list[object] = []
                real_close = os.close

                def fail_owned_close(number: int) -> None:
                    if number == descriptor:
                        raise OSError(
                            errno.EIO,
                            "synthetic public-handle finalization failure",
                        )
                    real_close(number)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_owned_close,
                        ),
                        patch.object(
                            sys,
                            "unraisablehook",
                            side_effect=lambda value: captured.append(value),
                        ),
                    ):
                        del staged
                        gc.collect()
                    self.assertIsNone(staged_ref())
                    self.assertEqual(len(captured), 1)
                    error = captured[0].exc_value
                    self.assertIs(type(error), DescriptorRetirementError)
                    self.assertEqual(error.operation, "finalization")
                    self.assertIs(error.state, StagingState.RETIRED)
                    self.assertIs(state.lifecycle, StagingState.RETIRED)
                    self.assertFalse(state.retirement_batch_pending)
                    with (
                        patch.object(
                            publication_module,
                            "_fstat_descriptor",
                            side_effect=AssertionError(
                                "context exit retried finalizer inspection"
                            ),
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=AssertionError(
                                "context exit retried finalizer close"
                            ),
                        ),
                    ):
                        self.assertFalse(
                            context.__exit__(None, None, None)
                        )
                finally:
                    real_close(descriptor)

    def test_context_finalization_with_live_handle_never_retries(
        self,
    ) -> None:
        destination = portable("context-finalization")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                state = publication_module._staging_state(staged)
                assert state is not None
                descriptor = state.staging_descriptor
                captured: list[object] = []
                real_close = os.close

                def fail_owned_close(number: int) -> None:
                    if number == descriptor:
                        raise OSError(
                            errno.EIO,
                            "synthetic context finalization failure",
                        )
                    real_close(number)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_owned_close,
                        ),
                        patch.object(
                            sys,
                            "unraisablehook",
                            side_effect=lambda value: captured.append(value),
                        ),
                    ):
                        del context
                        gc.collect()
                    self.assertEqual(len(captured), 1)
                    error = captured[0].exc_value
                    self.assertIs(type(error), DescriptorRetirementError)
                    self.assertEqual(error.operation, "finalization")
                    self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertFalse(state.retirement_batch_pending)
                    with (
                        patch.object(
                            publication_module,
                            "_fstat_descriptor",
                            side_effect=AssertionError(
                                "handle finalizer retried inspection"
                            ),
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=AssertionError(
                                "handle finalizer retried close"
                            ),
                        ),
                    ):
                        del staged
                        gc.collect()
                    self.assertEqual(len(captured), 1)
                finally:
                    real_close(descriptor)

    def test_explicit_generator_exit_remains_body_primary(self) -> None:
        destination = portable("explicit-generator-exit")
        sentinel = GeneratorExit("synthetic body exit")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with self.assertRaises(GeneratorExit) as caught:
                    with open_exclusive_staged_file(
                        trusted,
                        destination=destination,
                    ):
                        raise sentinel
                self.assertIs(caught.exception, sentinel)

        destination = portable("explicit-generator-exit-anomaly")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                state = publication_module._staging_state(staged)
                assert state is not None
                descriptor = state.staging_descriptor
                sentinel = GeneratorExit("synthetic anomalous body exit")
                real_close = os.close

                def fail_owned_close(number: int) -> None:
                    if number == descriptor:
                        raise OSError(
                            errno.EIO,
                            "synthetic explicit-exit close failure",
                        )
                    real_close(number)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_owned_close,
                        ),
                        self.assertRaises(BaseExceptionGroup) as caught,
                    ):
                        context.__exit__(
                            GeneratorExit,
                            sentinel,
                            None,
                        )
                    self.assertEqual(len(caught.exception.exceptions), 2)
                    self.assertIs(
                        caught.exception.exceptions[0],
                        sentinel,
                    )
                    retirement_error = caught.exception.exceptions[1]
                    self.assertIs(
                        type(retirement_error),
                        DescriptorRetirementError,
                    )
                    self.assertEqual(
                        retirement_error.operation,
                        "context_exit",
                    )
                    self.assertIs(state.lifecycle, StagingState.RETIRED)
                    self.assertFalse(state.retirement_batch_pending)
                finally:
                    real_close(descriptor)

    def test_pending_not_committed_collection_preserves_residue_only(
        self,
    ) -> None:
        destination = portable("pending-not-committed-finalization")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"payload")
                (root / destination.value).write_bytes(b"foreign")
                with self.assertRaises(PublicationCollisionError):
                    staged.seal()
                state = publication_module._staging_state(staged)
                assert state is not None
                self.assertIs(
                    state.lifecycle,
                    StagingState.NOT_COMMITTED,
                )
                self.assertTrue(state.retirement_batch_pending)
                staged_ref = weakref.ref(staged)
                captured: list[object] = []
                with patch.object(
                    sys,
                    "unraisablehook",
                    side_effect=lambda value: captured.append(value),
                ):
                    del staged
                    gc.collect()
                self.assertIsNone(staged_ref())
                self.assertEqual(captured, [])
                self.assertIs(state.lifecycle, StagingState.RETIRED)
                self.assertFalse(state.retirement_batch_pending)
                self.assertFalse(state.cleanup_attempted)
                self.assertTrue(
                    (root / staging_leaf(destination)).is_file()
                )
                with (
                    patch.object(
                        publication_module,
                        "_fstat_descriptor",
                        side_effect=AssertionError(
                            "retired collection retried inspection"
                        ),
                    ),
                    patch.object(
                        publication_module.os,
                        "close",
                        side_effect=AssertionError(
                            "retired collection retried close"
                        ),
                    ),
                ):
                    self.assertFalse(context.__exit__(None, None, None))

    def test_consumed_not_committed_collection_is_descriptor_noop(
        self,
    ) -> None:
        destination = portable("consumed-not-committed-finalization")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"payload")
                staged.seal()
                with (
                    patch.object(
                        publication_module,
                        "_native_no_replace",
                        return_value=(-1, errno.EAGAIN),
                    ),
                    self.assertRaises(PublicationError),
                ):
                    publish_completed_file(trusted, staged)
                state = publication_module._staging_state(staged)
                assert state is not None
                self.assertIs(
                    state.lifecycle,
                    StagingState.NOT_COMMITTED,
                )
                self.assertFalse(state.retirement_batch_pending)
                with (
                    patch.object(
                        publication_module,
                        "_fstat_descriptor",
                        side_effect=AssertionError(
                            "consumed collection inspected a descriptor"
                        ),
                    ),
                    patch.object(
                        publication_module.os,
                        "close",
                        side_effect=AssertionError(
                            "consumed collection closed a descriptor"
                        ),
                    ),
                ):
                    del staged
                    gc.collect()
                    self.assertFalse(context.__exit__(None, None, None))
                self.assertIs(state.lifecycle, StagingState.RETIRED)
                self.assertFalse(state.cleanup_attempted)
                self.assertTrue(
                    (root / staging_leaf(destination)).is_file()
                )

    def test_fresh_descriptor_retirement_preserves_reused_foreign_descriptor(
        self,
    ) -> None:
        """Registered provenance, not a caller fstat, controls retirement."""

        with synthetic_publication_root() as root:
            input_file = root / "input.bin"
            input_file.write_bytes(b"input")
            baseline = lowest_available_descriptor()
            parent_descriptor = os.open(
                root,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
            )
            owned_descriptor = -1
            foreign_descriptor = -1
            try:
                owned_descriptor = publication_module._open_read_file_at(
                    "input.bin",
                    parent_descriptor,
                    role="traversal_entry",
                )
                os.close(owned_descriptor)
                foreign_descriptor = os.open(
                    os.devnull,
                    os.O_RDONLY | os.O_CLOEXEC,
                )
                self.assertEqual(foreign_descriptor, owned_descriptor)

                with self.assertRaises(
                    publication_module._DescriptorRetirementAnomaly
                ) as caught:
                    publication_module._close_fresh_descriptor(
                        foreign_descriptor,
                        None,
                    )

                self.assertTrue(
                    stat.S_ISCHR(os.fstat(foreign_descriptor).st_mode)
                )
                self.assertIs(
                    caught.exception.evidence.records[0].observation,
                    DescriptorRetirementObservation.FOREIGN_PRESERVED,
                )
                with publication_module._OPEN_DESCRIPTOR_IDENTITIES_LOCK:
                    self.assertNotIn(
                        foreign_descriptor,
                        publication_module._OPEN_DESCRIPTOR_IDENTITIES,
                    )
            finally:
                if foreign_descriptor >= 0:
                    os.close(foreign_descriptor)
                elif owned_descriptor >= 0:
                    try:
                        publication_module._close_descriptor(
                            owned_descriptor
                        )
                    except OSError as exc:
                        if exc.errno != errno.EBADF:
                            raise
                os.close(parent_descriptor)
            self.assertEqual(lowest_available_descriptor(), baseline)

    def test_persistent_close_failure_is_non_silent_during_abandonment(
        self,
    ) -> None:
        destination = portable("abandoned.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"recovery")
                state = publication_module._staging_state(staged)
                owned_descriptor = state.staging_descriptor
                parent_descriptor = state.parent_descriptor
                real_close = os.close
                close_calls: list[int] = []

                def fail_owned_close(descriptor: int) -> None:
                    close_calls.append(descriptor)
                    if descriptor == owned_descriptor:
                        raise OSError(
                            errno.EIO,
                            "synthetic persistent close failure",
                        )
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_owned_close,
                        ),
                        self.assertRaises(DescriptorRetirementError) as caught,
                    ):
                        context.__exit__(None, None, None)
                    self.assertEqual(caught.exception.operation, "context_exit")
                    self.assertIs(
                        caught.exception.retirement_evidence.records[0].observation,
                        DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertEqual(state.staging_descriptor, -1)
                    self.assertEqual(state.parent_descriptor, -1)
                    self.assertEqual(close_calls.count(owned_descriptor), 1)
                    self.assertEqual(close_calls.count(parent_descriptor), 1)
                    os.fstat(owned_descriptor)
                    with self.assertRaises(OSError) as parent_closed:
                        os.fstat(parent_descriptor)
                    self.assertEqual(parent_closed.exception.errno, errno.EBADF)
                    with self.assertRaises(StagingLifecycleError):
                        staged.write(b"no-retry")
                    self.assertEqual(close_calls.count(owned_descriptor), 1)
                finally:
                    real_close(owned_descriptor)

    def test_persistent_close_failure_preserves_published_terminal_state(
        self,
    ) -> None:
        destination = portable("published.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"payload")
                staged.seal()
                state = publication_module._staging_state(staged)
                owned_descriptor = state.staging_descriptor
                real_close = os.close
                close_calls: list[int] = []

                def fail_owned_close(descriptor: int) -> None:
                    close_calls.append(descriptor)
                    if descriptor == owned_descriptor:
                        raise OSError(
                            errno.EIO,
                            "synthetic persistent close failure",
                        )
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_owned_close,
                        ),
                        self.assertRaises(DescriptorRetirementError) as caught,
                    ):
                        publish_completed_file(trusted, staged)
                    self.assertIsInstance(
                        caught.exception.terminal_result,
                        PublicationResult,
                    )
                    self.assertIs(staged.state, StagingState.PUBLISHED)
                    self.assertEqual(state.staging_descriptor, -1)
                    self.assertEqual(state.parent_descriptor, -1)
                    self.assertEqual(
                        (root / destination.value).read_bytes(),
                        b"payload",
                    )
                    self.assertEqual(close_calls.count(owned_descriptor), 1)
                    os.fstat(owned_descriptor)
                    context.__exit__(None, None, None)
                    self.assertEqual(close_calls.count(owned_descriptor), 1)
                    self.assertIs(staged.state, StagingState.PUBLISHED)
                finally:
                    real_close(owned_descriptor)

    def test_persistent_close_failure_preserves_discarded_terminal_state(
        self,
    ) -> None:
        destination = portable("discarded.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"payload")
                state = publication_module._staging_state(staged)
                owned_descriptor = state.staging_descriptor
                real_close = os.close
                close_calls: list[int] = []

                def fail_owned_close(descriptor: int) -> None:
                    close_calls.append(descriptor)
                    if descriptor == owned_descriptor:
                        raise OSError(
                            errno.EIO,
                            "synthetic persistent close failure",
                        )
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=fail_owned_close,
                        ),
                        self.assertRaises(DescriptorRetirementError) as caught,
                    ):
                        cleanup_owned_staging(trusted, staged)
                    self.assertIsInstance(
                        caught.exception.terminal_result,
                        StagingCleanupResult,
                    )
                    self.assertIs(staged.state, StagingState.DISCARDED)
                    self.assertEqual(state.staging_descriptor, -1)
                    self.assertEqual(state.parent_descriptor, -1)
                    self.assertFalse((root / staging_leaf(destination)).exists())
                    self.assertEqual(close_calls.count(owned_descriptor), 1)
                    os.fstat(owned_descriptor)
                    context.__exit__(None, None, None)
                    self.assertEqual(close_calls.count(owned_descriptor), 1)
                    self.assertIs(staged.state, StagingState.DISCARDED)
                finally:
                    real_close(owned_descriptor)

    def test_close_then_raise_is_uncertain_without_post_close_retry(self) -> None:
        destination = portable("published.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    state = publication_module._staging_state(staged)
                    owned_descriptor = state.staging_descriptor
                    real_close = os.close
                    close_calls: list[int] = []

                    def close_then_raise(descriptor: int) -> None:
                        close_calls.append(descriptor)
                        real_close(descriptor)
                        if descriptor == owned_descriptor:
                            raise OSError(
                                errno.EIO,
                                "synthetic post-close error",
                            )

                    with (
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=close_then_raise,
                        ),
                        self.assertRaises(DescriptorRetirementError) as caught,
                    ):
                        publish_completed_file(trusted, staged)
                    self.assertIs(
                        caught.exception.terminal_result.state,
                        PublicationState.COMMITTED_DURABLE,
                    )
                    self.assertIn(
                        DescriptorRetirementObservation.CLOSE_OUTCOME_UNCERTAIN,
                        tuple(
                            record.observation
                            for record in caught.exception.retirement_evidence.records
                        ),
                    )
                    self.assertIs(staged.state, StagingState.PUBLISHED)
                    self.assertEqual(close_calls.count(owned_descriptor), 1)
                    with self.assertRaises(OSError) as closed:
                        os.fstat(owned_descriptor)
                    self.assertEqual(closed.exception.errno, errno.EBADF)

    def test_close_then_reuse_preserves_foreign_descriptor(self) -> None:
        destination = portable("published.bin")
        foreign_descriptor = -1
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                with open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                ) as staged:
                    staged.write(b"payload")
                    staged.seal()
                    state = publication_module._staging_state(staged)
                    owned_descriptor = state.staging_descriptor
                    real_close = os.close
                    close_calls: list[int] = []

                    def close_then_reuse(descriptor: int) -> None:
                        nonlocal foreign_descriptor
                        close_calls.append(descriptor)
                        real_close(descriptor)
                        if descriptor == owned_descriptor:
                            foreign_descriptor = os.open(
                                os.devnull,
                                os.O_RDONLY | os.O_CLOEXEC,
                            )
                            self.assertEqual(
                                foreign_descriptor,
                                owned_descriptor,
                            )
                            raise OSError(
                                errno.EIO,
                                "synthetic post-close reuse",
                            )

                    try:
                        with (
                            patch.object(
                                publication_module.os,
                                "close",
                                side_effect=close_then_reuse,
                            ),
                            self.assertRaises(
                                DescriptorRetirementError
                            ) as caught,
                        ):
                            publish_completed_file(trusted, staged)
                        self.assertIs(
                            caught.exception.terminal_result.state,
                            PublicationState.COMMITTED_DURABLE,
                        )
                        self.assertIs(staged.state, StagingState.PUBLISHED)
                        self.assertEqual(close_calls.count(owned_descriptor), 1)
                        self.assertTrue(
                            stat.S_ISCHR(
                                os.fstat(foreign_descriptor).st_mode
                            )
                        )
                    finally:
                        if foreign_descriptor >= 0:
                            real_close(foreign_descriptor)

    def test_preclose_fstat_failure_is_non_silent_and_clears_authority(
        self,
    ) -> None:
        destination = portable("abandoned.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                staged.write(b"recovery")
                state = publication_module._staging_state(staged)
                owned_descriptor = state.staging_descriptor
                parent_descriptor = state.parent_descriptor
                real_fstat = publication_module._fstat_descriptor
                real_close = os.close
                close_calls: list[int] = []

                def fail_owned_fstat(descriptor: int):
                    if descriptor == owned_descriptor:
                        raise OSError(
                            errno.EIO,
                            "synthetic descriptor inspection failure",
                        )
                    return real_fstat(descriptor)

                def record_close(descriptor: int) -> None:
                    close_calls.append(descriptor)
                    real_close(descriptor)

                try:
                    with (
                        patch.object(
                            publication_module,
                            "_fstat_descriptor",
                            side_effect=fail_owned_fstat,
                        ),
                        patch.object(
                            publication_module.os,
                            "close",
                            side_effect=record_close,
                        ),
                        self.assertRaises(DescriptorRetirementError) as caught,
                    ):
                        context.__exit__(None, None, None)
                    self.assertIn(
                        DescriptorRetirementObservation.UNINSPECTABLE,
                        tuple(
                            record.observation
                            for record in caught.exception.retirement_evidence.records
                        ),
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertEqual(state.staging_descriptor, -1)
                    self.assertEqual(state.parent_descriptor, -1)
                    self.assertNotIn(owned_descriptor, close_calls)
                    self.assertIn(parent_descriptor, close_calls)
                    os.fstat(owned_descriptor)
                    with self.assertRaises(StagingLifecycleError):
                        staged.write(b"no-revival")
                finally:
                    real_close(owned_descriptor)

    def test_wrapper_registration_failure_closes_every_fresh_descriptor(
        self,
    ) -> None:
        wrappers = (
            "publication_parent",
            "directory",
            "read_file",
            "create_file",
        )
        for wrapper_name in wrappers:
            for partial_registration in (False, True):
                with self.subTest(
                    wrapper=wrapper_name,
                    partial_registration=partial_registration,
                ):
                    with synthetic_publication_root() as root:
                        child_directory = root / "child"
                        child_directory.mkdir(mode=0o700)
                        child_directory.chmod(0o700)
                        child_file = root / "input.bin"
                        child_file.write_bytes(b"input")
                        parent_descriptor = -1
                        opened_descriptor = -1
                        original_register = (
                            publication_module._register_fresh_descriptor
                        )

                        def fail_registration(
                            descriptor: int,
                            *,
                            role: str,
                        ) -> None:
                            nonlocal opened_descriptor
                            opened_descriptor = descriptor
                            if partial_registration:
                                original_register(descriptor, role=role)
                            raise RuntimeError(
                                "synthetic descriptor registration failure"
                            )

                        try:
                            if wrapper_name == "publication_parent":
                                trusted_context = open_trusted_root(str(root))
                                trusted = trusted_context.__enter__()
                                call = lambda: (
                                    publication_module._open_publication_parent(
                                        trusted,
                                        role="operation_parent",
                                    )
                                )
                            else:
                                trusted_context = None
                                parent_descriptor = os.open(
                                    root,
                                    os.O_RDONLY
                                    | os.O_DIRECTORY
                                    | os.O_NOFOLLOW
                                    | os.O_CLOEXEC,
                                )
                                if wrapper_name == "directory":
                                    call = lambda: (
                                        publication_module._open_directory_at(
                                            "child",
                                            parent_descriptor,
                                            role="traversal_directory",
                                        )
                                    )
                                elif wrapper_name == "read_file":
                                    call = lambda: (
                                        publication_module._open_read_file_at(
                                            "input.bin",
                                            parent_descriptor,
                                            role="traversal_entry",
                                        )
                                    )
                                else:
                                    output_name = (
                                        "output-partial.bin"
                                        if partial_registration
                                        else "output.bin"
                                    )
                                    call = lambda: (
                                        publication_module._create_file_at(
                                            output_name,
                                            parent_descriptor,
                                            0o600,
                                            role="traversal_entry",
                                        )
                                    )

                            baseline = lowest_available_descriptor()
                            with (
                                patch.object(
                                    publication_module,
                                    "_register_fresh_descriptor",
                                    side_effect=fail_registration,
                                ),
                                self.assertRaises(RuntimeError) as caught,
                            ):
                                call()
                            self.assertGreaterEqual(opened_descriptor, 0)
                            retirement = caught.exception.retirement_evidence
                            if partial_registration:
                                self.assertIsNone(retirement)
                                self.assertEqual(
                                    lowest_available_descriptor(),
                                    baseline,
                                )
                                with self.assertRaises(OSError) as absent:
                                    os.fstat(opened_descriptor)
                                self.assertEqual(
                                    absent.exception.errno,
                                    errno.EBADF,
                                )
                            else:
                                self.assertIsNotNone(retirement)
                                assert retirement is not None
                                self.assertIs(
                                    retirement.records[0].observation,
                                    DescriptorRetirementObservation.UNINSPECTABLE,
                                )
                                os.fstat(opened_descriptor)
                            with (
                                publication_module
                                ._OPEN_DESCRIPTOR_IDENTITIES_LOCK
                            ):
                                self.assertNotIn(
                                    opened_descriptor,
                                    publication_module
                                    ._OPEN_DESCRIPTOR_IDENTITIES,
                                )
                        finally:
                            if opened_descriptor >= 0:
                                try:
                                    publication_module._close_descriptor(
                                        opened_descriptor
                                    )
                                except OSError as exc:
                                    if exc.errno != errno.EBADF:
                                        raise
                            if parent_descriptor >= 0:
                                os.close(parent_descriptor)
                            if trusted_context is not None:
                                trusted_context.__exit__(None, None, None)

    def test_closed_trusted_root_retires_every_population_surface_without_mutation(
        self,
    ) -> None:
        cases = (
            ("write", "file"),
            ("file-seal", "file"),
            ("mkdir", "directory"),
            ("write-file", "directory"),
            ("directory-seal", "directory"),
        )
        for operation, kind in cases:
            with self.subTest(operation=operation):
                destination = portable(f"{operation}-result")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        factory = (
                            open_exclusive_staged_file
                            if kind == "file"
                            else open_exclusive_staged_directory
                        )
                        context = factory(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        if operation == "file-seal":
                            staged.write(b"payload")
                        elif operation == "directory-seal":
                            staged.write_file(
                                portable("payload.bin"),
                                (b"payload",),
                            )
                        before = synthetic_tree_snapshot(root)
                        trusted.close()
                        with (
                            patch.object(
                                publication_module,
                                "_write_descriptor",
                                side_effect=AssertionError(
                                    "closed-root operation wrote bytes"
                                ),
                            ),
                            patch.object(
                                publication_module,
                                "_mkdir_at",
                                side_effect=AssertionError(
                                    "closed-root operation created a directory"
                                ),
                            ),
                            patch.object(
                                publication_module,
                                "_create_file_at",
                                side_effect=AssertionError(
                                    "closed-root operation created a file"
                                ),
                            ),
                            patch.object(
                                publication_module,
                                "_fchmod_descriptor",
                                side_effect=AssertionError(
                                    "closed-root operation changed a mode"
                                ),
                            ),
                            patch.object(
                                publication_module,
                                "_fsync_descriptor",
                                side_effect=AssertionError(
                                    "closed-root operation flushed state"
                                ),
                            ),
                            self.assertRaises(
                                PublicationValidationError
                            ) as caught,
                        ):
                            if operation == "write":
                                staged.write(b"mutation")
                            elif operation == "file-seal":
                                staged.seal()
                            elif operation == "mkdir":
                                staged.mkdir(portable("new-directory"))
                            elif operation == "write-file":
                                staged.write_file(
                                    portable("new-file.bin"),
                                    (b"mutation",),
                                )
                            else:
                                staged.seal(scope="synthetic-closed-root-v1")
                        self.assertIs(
                            caught.exception.state,
                            PublicationState.NOT_COMMITTED,
                        )
                        self.assertIs(staged.state, StagingState.RETIRED)
                        self.assertEqual(
                            synthetic_tree_snapshot(root),
                            before,
                        )
                        context.__exit__(None, None, None)
                        self.assertIs(staged.state, StagingState.RETIRED)
                        self.assertEqual(
                            synthetic_tree_snapshot(root),
                            before,
                        )

    def test_terminal_root_integrity_failure_is_authority_error_without_attempt(
        self,
    ) -> None:
        cases = ("publish", "cleanup")
        for operation in cases:
            with self.subTest(operation=operation):
                destination = portable(f"{operation}-result.bin")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"payload")
                            if operation == "publish":
                                staged.seal()
                            state = publication_module._staging_state(staged)
                            before = (
                                state.lifecycle,
                                state.publication_attempted,
                                state.cleanup_attempted,
                                state.retirement_batch_pending,
                                state.parent_descriptor,
                                state.staging_descriptor,
                            )
                            with (
                                patch.object(
                                    publication_module,
                                    "_require_exact_live_root",
                                    side_effect=safety.TrustedRootError(
                                        "synthetic integrity failure"
                                    ),
                                ) as require_live,
                                patch.object(
                                    publication_module,
                                    "_open_publication_parent",
                                ) as opened,
                                self.assertRaises(
                                    StagingAuthorityError
                                ) as caught,
                            ):
                                if operation == "publish":
                                    publish_completed_file(trusted, staged)
                                else:
                                    cleanup_owned_staging(trusted, staged)
                            require_live.assert_called_once_with(trusted)
                            opened.assert_not_called()
                            self.assertEqual(
                                caught.exception.operation,
                                operation,
                            )
                            self.assertIs(
                                staged.state,
                                (
                                    StagingState.SEALED
                                    if operation == "publish"
                                    else StagingState.OPEN
                                ),
                            )
                            self.assertFalse(state.publication_attempted)
                            self.assertFalse(state.cleanup_attempted)
                            self.assertEqual(
                                (
                                    state.lifecycle,
                                    state.publication_attempted,
                                    state.cleanup_attempted,
                                    state.retirement_batch_pending,
                                    state.parent_descriptor,
                                    state.staging_descriptor,
                                ),
                                before,
                            )
                            if operation == "publish":
                                publish_completed_file(trusted, staged)
                                self.assertIs(
                                    staged.state,
                                    StagingState.PUBLISHED,
                                )
                            else:
                                cleanup_owned_staging(trusted, staged)
                                self.assertIs(
                                    staged.state,
                                    StagingState.DISCARDED,
                                )

    def test_unentered_context_allocates_no_descriptor(self) -> None:
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                baseline = lowest_available_descriptor()
                context = open_exclusive_staged_file(
                    trusted,
                    destination=portable("result.bin"),
                )
                self.assertEqual(lowest_available_descriptor(), baseline)
                del context
                self.assertEqual(lowest_available_descriptor(), baseline)
                assert_no_entries(self, root)

    def test_repeated_success_cleanup_and_body_failure_do_not_leak(
        self,
    ) -> None:
        class SyntheticBodyFailure(Exception):
            pass

        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                baseline = lowest_available_descriptor()
                for index in range(8):
                    with self.subTest(kind="publish", index=index):
                        destination = portable(f"published-{index}")
                        with open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write(b"payload")
                            staged.seal()
                            publish_completed_file(trusted, staged)
                        self.assertEqual(
                            lowest_available_descriptor(),
                            baseline,
                        )

                    with self.subTest(kind="cleanup", index=index):
                        destination = portable(f"discarded-{index}")
                        with open_exclusive_staged_directory(
                            trusted,
                            destination=destination,
                        ) as staged:
                            staged.write_file(
                                portable("item.bin"),
                                (b"payload",),
                            )
                            cleanup_owned_staging(trusted, staged)
                        self.assertEqual(
                            lowest_available_descriptor(),
                            baseline,
                        )

                    with self.subTest(kind="body-failure", index=index):
                        destination = portable(f"abandoned-{index}")
                        try:
                            with open_exclusive_staged_file(
                                trusted,
                                destination=destination,
                            ) as staged:
                                staged.write(b"recovery")
                                raise SyntheticBodyFailure
                        except SyntheticBodyFailure:
                            pass
                        self.assertIs(staged.state, StagingState.RETIRED)
                        self.assertEqual(
                            lowest_available_descriptor(),
                            baseline,
                        )

    def test_context_retirement_never_closes_reused_foreign_descriptor(
        self,
    ) -> None:
        destination = portable("result.bin")
        with synthetic_publication_root() as root:
            with open_trusted_root(str(root)) as trusted:
                context = open_exclusive_staged_file(
                    trusted,
                    destination=destination,
                )
                staged = context.__enter__()
                state = publication_module._staging_state(staged)
                owned_descriptor = state.staging_descriptor
                os.close(owned_descriptor)
                foreign_descriptor = os.open(
                    os.devnull,
                    os.O_RDONLY | os.O_CLOEXEC,
                )
                self.assertEqual(foreign_descriptor, owned_descriptor)
                try:
                    with self.assertRaises(
                        DescriptorRetirementError
                    ) as caught:
                        context.__exit__(None, None, None)
                    self.assertIn(
                        DescriptorRetirementObservation.FOREIGN_PRESERVED,
                        tuple(
                            record.observation
                            for record in caught.exception.retirement_evidence.records
                        ),
                    )
                    self.assertIs(staged.state, StagingState.RETIRED)
                    self.assertTrue(
                        stat.S_ISCHR(os.fstat(foreign_descriptor).st_mode)
                    )
                    with self.assertRaises(StagingLifecycleError):
                        staged.write(b"revive")
                finally:
                    os.close(foreign_descriptor)

    def test_terminal_publish_and_cleanup_cannot_revive_or_close_reused_fd(
        self,
    ) -> None:
        for operation in ("publish", "cleanup"):
            with self.subTest(operation=operation):
                destination = portable(f"{operation}-result.bin")
                with synthetic_publication_root() as root:
                    with open_trusted_root(str(root)) as trusted:
                        context = open_exclusive_staged_file(
                            trusted,
                            destination=destination,
                        )
                        staged = context.__enter__()
                        staged.write(b"payload")
                        state = publication_module._staging_state(staged)
                        retired_descriptor = state.staging_descriptor
                        if operation == "publish":
                            staged.seal()
                            publish_completed_file(trusted, staged)
                            terminal_state = StagingState.PUBLISHED
                        else:
                            cleanup_owned_staging(trusted, staged)
                            terminal_state = StagingState.DISCARDED
                        self.assertIs(staged.state, terminal_state)
                        foreign_descriptors: list[int] = []
                        while retired_descriptor not in foreign_descriptors:
                            self.assertLess(len(foreign_descriptors), 32)
                            foreign_descriptors.append(
                                os.open(
                                    os.devnull,
                                    os.O_RDONLY | os.O_CLOEXEC,
                                )
                            )
                        try:
                            context.__exit__(None, None, None)
                            for foreign_descriptor in foreign_descriptors:
                                self.assertTrue(
                                    stat.S_ISCHR(
                                        os.fstat(foreign_descriptor).st_mode
                                    )
                                )
                            self.assertIs(staged.state, terminal_state)
                            with self.assertRaises(StagingLifecycleError):
                                if operation == "publish":
                                    publish_completed_file(trusted, staged)
                                else:
                                    cleanup_owned_staging(trusted, staged)
                            for foreign_descriptor in foreign_descriptors:
                                self.assertTrue(
                                    stat.S_ISCHR(
                                        os.fstat(foreign_descriptor).st_mode
                                    )
                                )
                        finally:
                            for foreign_descriptor in reversed(
                                foreign_descriptors
                            ):
                                os.close(foreign_descriptor)


class StaticSafetyBoundaryTests(unittest.TestCase):
    def test_publication_module_uses_no_forbidden_fallback_or_remote_stack(
        self,
    ) -> None:
        path = Path(publication_module.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        forbidden_imports = (
            "subprocess",
            "socket",
            "asyncio",
            "urllib",
            "http",
            "requests",
            "paramiko",
            "random",
            "time",
            "research_platform.core",
            "research_platform.neuro",
            "ops",
        )
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(
            {
                name
                for name in imported
                if any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in forbidden_imports
                )
            }
        )

        forbidden_text = (
            "os.rename(",
            "os.replace(",
            "Path.resolve(",
            ".resolve()",
            "realpath(",
            "/proc/self/fd",
            "subprocess",
            "F_FULLFSYNC",
        )
        for forbidden in forbidden_text:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        raw_normalized_comparisons = (
            "st_dev != state.parent_device",
            "st_dev == state.parent_device",
            "st_ino != state.parent_inode",
            "st_ino == state.parent_inode",
            "st_dev != expected_device",
            "st_ino != expected_inode",
            "st_dev != expected_parent.public.device",
        )
        for comparison in raw_normalized_comparisons:
            with self.subTest(raw_normalized=comparison):
                self.assertNotIn(comparison, source)

        self.assertIn("ctypes.CDLL(None, use_errno=True)", source)
        self.assertIn("renameat2", source)
        self.assertIn("renameatx_np", source)

    def test_no_h2c2_or_h2d_module_exists(self) -> None:
        safety_root = Path(publication_module.__file__).parent
        self.assertFalse((safety_root / "claims.py").exists())
        self.assertFalse((safety_root / "receipts.py").exists())


if __name__ == "__main__":
    unittest.main()
