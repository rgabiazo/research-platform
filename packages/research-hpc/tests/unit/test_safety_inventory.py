"""Contracts for descriptor-anchored regular-file inventories."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
from dataclasses import FrozenInstanceError
import errno
import hashlib
import importlib
import json
import os
import pickle
from pathlib import Path
import stat
import struct
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc.safety.canonical import (
    TREE_DIGEST_ALGORITHM,
    TREE_DIGEST_DOMAIN,
)
from research_platform.hpc.safety.inventory import (
    INVENTORY_SCHEMA,
    InventoryCapabilityError,
    InventoryEntryError,
    InventoryLimitError,
    InventoryMutationError,
    InventoryValidationError,
    RegularFileInventory,
    RegularFileRecord,
    TrustedRoot,
    TrustedRootError,
    open_trusted_root,
    scan_regular_file_inventory,
)
from research_platform.hpc.safety.paths import PortableRelativePath


inventory_module = importlib.import_module(
    "research_platform.hpc.safety.inventory"
)


@contextmanager
def synthetic_root():
    """Yield a symlink-free synthetic root while retaining portable cleanup."""

    with tempfile.TemporaryDirectory() as temporary:
        canonical_temporary = Path(os.path.realpath(temporary))
        root = canonical_temporary / "payload"
        root.mkdir()
        yield canonical_temporary, root


def write_file(root: Path, relative: str, content: bytes, mode: int = 0o600) -> Path:
    destination = root.joinpath(*relative.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    destination.chmod(mode)
    return destination


def scan(
    root: Path,
    *,
    exclusions: tuple[str, ...] = (),
    scope: str = "synthetic-test-v1",
) -> RegularFileInventory:
    excluded = tuple(PortableRelativePath.parse(value) for value in exclusions)
    with open_trusted_root(str(root)) as trusted:
        return scan_regular_file_inventory(
            trusted,
            scope=scope,
            excluded_prefixes=excluded,
        )


def private_trusted_root_descriptor(trusted: TrustedRoot) -> int:
    """Inspect closure-held state only for deterministic lifecycle tests."""

    state = inventory_module._trusted_root_state(trusted)
    if state is None:
        raise AssertionError("trusted root has no live private state")
    return state.descriptor


def altered_stat(
    result: os.stat_result,
    *,
    device: int | None = None,
    mode: int | None = None,
) -> os.stat_result:
    values = list(result)
    if mode is not None:
        values[0] = mode
    if device is not None:
        values[2] = device
    return os.stat_result(values)


class DescriptorRecorder:
    """Track only descriptors opened through the H2b private wrappers."""

    def __init__(self) -> None:
        self.live: Counter[int] = Counter()
        self.opened: list[tuple[str, str, int]] = []
        self.closed: list[int] = []

    @property
    def live_count(self) -> int:
        return sum(self.live.values())

    def record_open(self, kind: str, name: str, descriptor: int) -> int:
        self.live[descriptor] += 1
        self.opened.append((kind, name, descriptor))
        return descriptor

    def record_close(self, descriptor: int) -> None:
        if self.live[descriptor]:
            self.live[descriptor] -= 1
            if not self.live[descriptor]:
                del self.live[descriptor]
        self.closed.append(descriptor)

    def names(self, kind: str) -> tuple[str, ...]:
        return tuple(
            name for opened_kind, name, _descriptor in self.opened
            if opened_kind == kind
        )


@contextmanager
def record_inventory_descriptors():
    recorder = DescriptorRecorder()
    original_absolute = inventory_module._open_absolute_root
    original_directory = inventory_module._open_directory_at
    original_file = inventory_module._open_regular_file_at
    original_close = inventory_module._close_descriptor

    def open_absolute(flags: int) -> int:
        return recorder.record_open(
            "absolute",
            "/",
            original_absolute(flags),
        )

    def open_directory(name: str, parent: int) -> int:
        return recorder.record_open(
            "directory",
            name,
            original_directory(name, parent),
        )

    def open_file(name: str, parent: int) -> int:
        return recorder.record_open(
            "file",
            name,
            original_file(name, parent),
        )

    def close_descriptor(descriptor: int) -> None:
        original_close(descriptor)
        recorder.record_close(descriptor)

    with (
        patch.object(
            inventory_module,
            "_open_absolute_root",
            side_effect=open_absolute,
        ),
        patch.object(
            inventory_module,
            "_open_directory_at",
            side_effect=open_directory,
        ),
        patch.object(
            inventory_module,
            "_open_regular_file_at",
            side_effect=open_file,
        ),
        patch.object(
            inventory_module,
            "_close_descriptor",
            side_effect=close_descriptor,
        ),
    ):
        yield recorder


class TrustedRootTests(unittest.TestCase):
    def assert_retired(self, trusted: TrustedRoot) -> None:
        self.assertIsNone(inventory_module._trusted_root_state(trusted))
        self.assertFalse(trusted.is_open)
        self.assertTrue(trusted.is_closed)
        for attribute in ("device", "inode"):
            with self.subTest(attribute=attribute), self.assertRaises(
                TrustedRootError
            ):
                getattr(trusted, attribute)
        with self.assertRaises(TrustedRootError):
            trusted.close()
        with self.assertRaises(TrustedRootError):
            scan_regular_file_inventory(
                trusted,
                scope="synthetic-test-v1",
            )

    def test_normal_absolute_root_is_pinned_and_closes_on_exit(self) -> None:
        with synthetic_root() as (_temporary, root):
            expected = root.stat()
            self.assertNotIn("descriptor", TrustedRoot.__dict__)
            with open_trusted_root(str(root)) as trusted:
                self.assertIs(type(trusted), TrustedRoot)
                self.assertFalse(hasattr(trusted, "descriptor"))
                self.assertTrue(trusted.is_open)
                self.assertFalse(trusted.is_closed)
                self.assertEqual(trusted.device, expected.st_dev)
                self.assertEqual(trusted.inode, expected.st_ino)
                descriptor = private_trusted_root_descriptor(trusted)
                self.assertEqual(os.fstat(descriptor).st_ino, expected.st_ino)
                self.assertFalse(os.get_inheritable(descriptor))
            self.assertFalse(trusted.is_open)
            self.assertTrue(trusted.is_closed)
            with self.assertRaises(TrustedRootError):
                trusted.close()
            with self.assertRaises(TrustedRootError):
                scan_regular_file_inventory(
                    trusted,
                    scope="synthetic-test-v1",
                )

    def test_root_path_grammar_is_fail_closed(self) -> None:
        class HostileString(str):
            def startswith(self, *_args: object, **_kwargs: object) -> bool:
                return True

            def split(self, *_args: object, **_kwargs: object) -> list[str]:
                return ["safe"]

        invalid: tuple[object, ...] = (
            "relative/path",
            "./relative",
            "../relative",
            "/tmp//payload",
            "/tmp/./payload",
            "/tmp/../payload",
            "/tmp/payload/",
            "/tmp/nul\0payload",
            "/tmp/line\npayload",
            Path("/tmp/payload"),
            HostileString("/tmp/payload"),
        )
        for value in invalid:
            with self.subTest(value=repr(value)):
                with self.assertRaises(TrustedRootError):
                    with open_trusted_root(value):  # type: ignore[arg-type]
                        pass

    def test_root_and_ancestor_symlinks_are_rejected(self) -> None:
        with synthetic_root() as (temporary, root):
            write_file(root, "item.txt", b"x")
            root_link = temporary / "root-link"
            root_link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(TrustedRootError):
                with open_trusted_root(str(root_link)):
                    pass

            real_ancestor = temporary / "real-ancestor"
            child = real_ancestor / "child"
            child.mkdir(parents=True)
            ancestor_link = temporary / "ancestor-link"
            ancestor_link.symlink_to(real_ancestor, target_is_directory=True)
            with self.assertRaises(TrustedRootError):
                with open_trusted_root(str(ancestor_link / "child")):
                    pass

    def test_root_and_ancestor_non_directories_are_rejected(self) -> None:
        with synthetic_root() as (temporary, _root):
            regular = temporary / "regular"
            regular.write_bytes(b"x")
            with self.assertRaises(TrustedRootError):
                with open_trusted_root(str(regular)):
                    pass
            with self.assertRaises(TrustedRootError):
                with open_trusted_root(str(regular / "child")):
                    pass

    def test_intermediate_descriptors_close_after_success_and_failure(self) -> None:
        original_open = inventory_module._open_directory_at
        original_close = inventory_module._close_descriptor

        for fail in (False, True):
            with self.subTest(fail=fail), synthetic_root() as (temporary, root):
                path = root
                if fail:
                    link = temporary / "bad-link"
                    link.symlink_to(root, target_is_directory=True)
                    path = link

                opened: list[int] = []
                closed: list[int] = []

                def recording_open(name: str, parent: int) -> int:
                    descriptor = original_open(name, parent)
                    opened.append(descriptor)
                    return descriptor

                def recording_close(descriptor: int) -> None:
                    closed.append(descriptor)
                    original_close(descriptor)

                with (
                    patch.object(
                        inventory_module,
                        "_open_directory_at",
                        side_effect=recording_open,
                    ),
                    patch.object(
                        inventory_module,
                        "_close_descriptor",
                        side_effect=recording_close,
                    ),
                ):
                    if fail:
                        with self.assertRaises(TrustedRootError):
                            with open_trusted_root(str(path)):
                                pass
                    else:
                        with open_trusted_root(str(path)):
                            pass

                opened_counts = Counter(opened)
                closed_counts = Counter(closed)
                for descriptor, count in opened_counts.items():
                    self.assertGreaterEqual(closed_counts[descriptor], count)

    def test_directory_open_contract_is_nonblocking_and_no_follow(self) -> None:
        flags = inventory_module._directory_open_flags()
        for required in (
            os.O_DIRECTORY,
            os.O_NOFOLLOW,
            os.O_CLOEXEC,
            os.O_NONBLOCK,
        ):
            with self.subTest(required=required):
                self.assertEqual(flags & required, required)

    def test_direct_forged_or_subclassed_handle_is_rejected(self) -> None:
        for args, kwargs in (
            ((), {}),
            ((0, 0, 0), {}),
            ((), {"descriptor": 0, "device": 0, "inode": 0}),
            ((0, 0, 0), {"_token": object()}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(TypeError):
                    TrustedRoot(*args, **kwargs)
        self.assertFalse(hasattr(inventory_module, "_ROOT_CREATION_TOKEN"))

        forged = object.__new__(TrustedRoot)
        with self.assertRaises(TrustedRootError):
            scan_regular_file_inventory(forged, scope="synthetic-test-v1")

        class HostileRoot(TrustedRoot):
            pass

        hostile = object.__new__(HostileRoot)
        with self.assertRaises(TrustedRootError):
            scan_regular_file_inventory(hostile, scope="synthetic-test-v1")

    def test_arbitrary_directory_descriptor_cannot_be_wrapped(self) -> None:
        with synthetic_root() as (_temporary, root):
            descriptor = os.open(
                root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                result = os.fstat(descriptor)
                with self.assertRaises(TypeError):
                    TrustedRoot(
                        descriptor,
                        result.st_dev,
                        result.st_ino,
                        _token=object(),
                    )
                forged = object.__new__(TrustedRoot)
                with self.assertRaises(TrustedRootError):
                    scan_regular_file_inventory(
                        forged,
                        scope="synthetic-test-v1",
                    )
                self.assertEqual(os.fstat(descriptor).st_ino, result.st_ino)
            finally:
                os.close(descriptor)

    def test_unentered_factory_opens_nothing_and_needs_no_cleanup(self) -> None:
        with synthetic_root() as (_temporary, root):
            with patch.object(
                inventory_module,
                "_open_absolute_root",
                wraps=inventory_module._open_absolute_root,
            ) as open_absolute:
                manager = open_trusted_root(str(root))
                open_absolute.assert_not_called()
                del manager
                open_absolute.assert_not_called()

    def test_descriptor_accounting_for_success_failure_and_body_exception(
        self,
    ) -> None:
        with synthetic_root() as (temporary, root):
            write_file(root, "item.txt", b"x")
            bad_link = temporary / "bad-link"
            bad_link.symlink_to(root, target_is_directory=True)

            with record_inventory_descriptors() as descriptors:
                manager = open_trusted_root(str(root))
                self.assertEqual(descriptors.live_count, 0)
                with manager:
                    self.assertEqual(descriptors.live_count, 1)
                self.assertEqual(descriptors.live_count, 0)

                with self.assertRaises(TrustedRootError):
                    with open_trusted_root(str(bad_link)):
                        pass
                self.assertEqual(descriptors.live_count, 0)

                with self.assertRaisesRegex(RuntimeError, "body failure"):
                    with open_trusted_root(str(root)):
                        self.assertEqual(descriptors.live_count, 1)
                        raise RuntimeError("body failure")
                self.assertEqual(descriptors.live_count, 0)

                fifo = root / "pipe"
                os.mkfifo(fifo)
                with open_trusted_root(str(root)) as trusted:
                    with self.assertRaises(InventoryEntryError):
                        scan_regular_file_inventory(
                            trusted,
                            scope="synthetic-test-v1",
                        )
                    self.assertEqual(descriptors.live_count, 1)
                self.assertEqual(descriptors.live_count, 0)

    def test_repeated_success_and_entry_failure_do_not_accumulate_descriptors(
        self,
    ) -> None:
        with synthetic_root() as (temporary, root):
            write_file(root, "item.txt", b"x")
            bad_link = temporary / "bad-link"
            bad_link.symlink_to(root, target_is_directory=True)
            with record_inventory_descriptors() as descriptors:
                for _ in range(20):
                    with open_trusted_root(str(root)):
                        self.assertEqual(descriptors.live_count, 1)
                    self.assertEqual(descriptors.live_count, 0)
                    with self.assertRaises(TrustedRootError):
                        with open_trusted_root(str(bad_link)):
                            pass
                    self.assertEqual(descriptors.live_count, 0)

    def test_explicit_close_inside_context_is_not_closed_twice(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "item.txt", b"x")
            with record_inventory_descriptors() as descriptors:
                with open_trusted_root(str(root)) as trusted:
                    owned = private_trusted_root_descriptor(trusted)
                    closed_before = descriptors.closed.count(owned)
                    self.assertEqual(descriptors.live_count, 1)
                    trusted.close()
                    self.assertEqual(
                        descriptors.closed.count(owned),
                        closed_before + 1,
                    )
                    self.assertEqual(descriptors.live_count, 0)
                    self.assert_retired(trusted)
                    closed_after = descriptors.closed.count(owned)
                self.assertEqual(
                    descriptors.closed.count(owned),
                    closed_after,
                )
                self.assertEqual(descriptors.live_count, 0)

    def test_missing_owned_descriptor_retires_on_context_exit(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "item.txt", b"x")
            manager = open_trusted_root(str(root))
            trusted = manager.__enter__()
            owned = private_trusted_root_descriptor(trusted)
            os.close(owned)

            with self.assertRaises(TrustedRootError):
                manager.__exit__(None, None, None)

            self.assert_retired(trusted)
            with self.assertRaises(OSError):
                os.fstat(owned)

    def test_foreign_descriptor_reuse_is_preserved_and_retires_authority(
        self,
    ) -> None:
        for detector in ("scan", "context-exit"):
            with self.subTest(detector=detector), synthetic_root() as (
                temporary,
                root,
            ):
                write_file(root, "item.txt", b"x")
                foreign = temporary / "foreign"
                foreign.mkdir()
                manager = open_trusted_root(str(root))
                trusted = manager.__enter__()
                owned = private_trusted_root_descriptor(trusted)
                foreign_source = os.open(
                    foreign,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                )
                os.close(owned)
                os.dup2(foreign_source, owned, inheritable=False)
                foreign_identity = os.fstat(owned)
                try:
                    if detector == "scan":
                        with self.assertRaises(TrustedRootError):
                            scan_regular_file_inventory(
                                trusted,
                                scope="synthetic-test-v1",
                            )
                        manager.__exit__(None, None, None)
                    else:
                        with self.assertRaises(TrustedRootError):
                            manager.__exit__(None, None, None)

                    self.assert_retired(trusted)
                    still_foreign = os.fstat(owned)
                    self.assertEqual(
                        (still_foreign.st_dev, still_foreign.st_ino),
                        (foreign_identity.st_dev, foreign_identity.st_ino),
                    )
                finally:
                    os.close(owned)
                    os.close(foreign_source)

    def test_retired_handle_cannot_revive_after_descriptor_number_reuse(
        self,
    ) -> None:
        for reopened_kind in ("original", "foreign"):
            with self.subTest(reopened_kind=reopened_kind), synthetic_root() as (
                temporary,
                root,
            ):
                write_file(root, "item.txt", b"x")
                selected = root
                if reopened_kind == "foreign":
                    selected = temporary / "foreign"
                    selected.mkdir()

                manager = open_trusted_root(str(root))
                trusted = manager.__enter__()
                owned = private_trusted_root_descriptor(trusted)
                os.close(owned)
                with self.assertRaises(TrustedRootError):
                    manager.__exit__(None, None, None)
                self.assert_retired(trusted)

                source = os.open(
                    selected,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                )
                installed = source
                if source != owned:
                    installed = os.dup2(source, owned, inheritable=False)
                try:
                    self.assertEqual(installed, owned)
                    expected = os.fstat(owned)
                    self.assert_retired(trusted)
                    current = os.fstat(owned)
                    self.assertEqual(
                        (current.st_dev, current.st_ino),
                        (expected.st_dev, expected.st_ino),
                    )
                finally:
                    os.close(owned)
                    if source != owned:
                        os.close(source)

    def test_close_failure_retires_without_a_context_exit_retry(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "item.txt", b"x")
            manager = open_trusted_root(str(root))
            trusted = manager.__enter__()
            owned = private_trusted_root_descriptor(trusted)
            original_close = inventory_module._close_descriptor
            attempted = 0

            def fail_owned_close(descriptor: int) -> None:
                nonlocal attempted
                if descriptor == owned:
                    attempted += 1
                    raise OSError(errno.EIO, "synthetic close failure")
                original_close(descriptor)

            try:
                with patch.object(
                    inventory_module,
                    "_close_descriptor",
                    side_effect=fail_owned_close,
                ):
                    with self.assertRaises(TrustedRootError):
                        manager.__exit__(None, None, None)
                    self.assertEqual(attempted, 1)
                self.assert_retired(trusted)
                self.assertEqual(attempted, 1)
            finally:
                os.close(owned)

    def test_repeated_missing_and_foreign_failures_leave_no_live_authority(
        self,
    ) -> None:
        with synthetic_root() as (temporary, root):
            write_file(root, "item.txt", b"x")
            foreign = temporary / "foreign"
            foreign.mkdir()
            retired: list[TrustedRoot] = []
            with record_inventory_descriptors() as descriptors:
                for index in range(20):
                    manager = open_trusted_root(str(root))
                    trusted = manager.__enter__()
                    retired.append(trusted)
                    owned = private_trusted_root_descriptor(trusted)

                    if index % 2:
                        foreign_source = os.open(
                            foreign,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                        )
                    else:
                        foreign_source = None

                    os.close(owned)
                    descriptors.record_close(owned)
                    if foreign_source is not None:
                        os.dup2(
                            foreign_source,
                            owned,
                            inheritable=False,
                        )
                    try:
                        with self.assertRaises(TrustedRootError):
                            manager.__exit__(None, None, None)
                        self.assert_retired(trusted)
                        self.assertEqual(descriptors.live_count, 0)
                        if foreign_source is not None:
                            os.fstat(owned)
                    finally:
                        if foreign_source is not None:
                            os.close(owned)
                            os.close(foreign_source)

            self.assertTrue(retired)
            for trusted in retired:
                self.assert_retired(trusted)

    def test_root_components_replaced_between_stat_and_open_are_rejected(
        self,
    ) -> None:
        for target_kind in ("ancestor", "final"):
            with self.subTest(target_kind=target_kind), synthetic_root() as (
                temporary,
                _root,
            ):
                ancestor = temporary / "admission-ancestor"
                root = ancestor / "payload-root"
                root.mkdir(parents=True)
                write_file(root, "item.txt", b"x")
                changed = False
                target_name = (
                    "admission-ancestor"
                    if target_kind == "ancestor"
                    else "payload-root"
                )

                with record_inventory_descriptors() as descriptors:
                    recording_open = inventory_module._open_directory_at

                    def replace_then_open(name: str, parent: int) -> int:
                        nonlocal changed
                        if name == target_name and not changed:
                            changed = True
                            selected = (
                                ancestor if target_kind == "ancestor" else root
                            )
                            moved = temporary / f"former-{target_kind}"
                            selected.rename(moved)
                            selected.mkdir()
                            if target_kind == "ancestor":
                                (selected / "payload-root").mkdir()
                        return recording_open(name, parent)

                    with patch.object(
                        inventory_module,
                        "_open_directory_at",
                        side_effect=replace_then_open,
                    ):
                        with self.assertRaises(TrustedRootError):
                            with open_trusted_root(str(root)):
                                pass
                    self.assertTrue(changed)
                    self.assertEqual(descriptors.live_count, 0)

    def test_root_component_metadata_change_during_admission_is_rejected(
        self,
    ) -> None:
        for target_kind in ("ancestor", "final"):
            with self.subTest(target_kind=target_kind), synthetic_root() as (
                temporary,
                _root,
            ):
                ancestor = temporary / "metadata-ancestor"
                root = ancestor / "metadata-root"
                root.mkdir(parents=True)
                selected = ancestor if target_kind == "ancestor" else root
                target_name = selected.name
                changed = False

                with record_inventory_descriptors() as descriptors:
                    recording_open = inventory_module._open_directory_at

                    def change_mode_then_open(name: str, parent: int) -> int:
                        nonlocal changed
                        if name == target_name and not changed:
                            changed = True
                            selected.chmod(0o750)
                        return recording_open(name, parent)

                    with patch.object(
                        inventory_module,
                        "_open_directory_at",
                        side_effect=change_mode_then_open,
                    ):
                        with self.assertRaises(TrustedRootError):
                            with open_trusted_root(str(root)):
                                pass
                self.assertTrue(changed)
                self.assertEqual(descriptors.live_count, 0)

    def test_entered_root_remains_pinned_after_path_rename_and_replacement(
        self,
    ) -> None:
        with synthetic_root() as (temporary, root):
            write_file(root, "original.txt", b"original")
            with open_trusted_root(str(root)) as trusted:
                moved = temporary / "pinned-root"
                root.rename(moved)
                root.mkdir()
                write_file(root, "replacement.txt", b"replacement")
                result = scan_regular_file_inventory(
                    trusted,
                    scope="synthetic-test-v1",
                )
            self.assertEqual(
                tuple(record.path.value for record in result.files),
                ("original.txt",),
            )

    def test_handle_copying_and_serialization_fail_without_losing_owner(
        self,
    ) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "item.txt", b"x")
            with open_trusted_root(str(root)) as trusted:
                for operation in (
                    copy.copy,
                    copy.deepcopy,
                    pickle.dumps,
                ):
                    with self.subTest(operation=operation.__name__):
                        with self.assertRaises(TypeError):
                            operation(trusted)
                result = scan_regular_file_inventory(
                    trusted,
                    scope="synthetic-test-v1",
                )
                self.assertEqual(
                    tuple(record.path.value for record in result.files),
                    ("item.txt",),
                )


class InventoryCapabilityTests(unittest.TestCase):
    def _assert_capability_failure_before_open(self, root: Path) -> None:
        with (
            patch.object(
                inventory_module,
                "_open_absolute_root",
                wraps=inventory_module._open_absolute_root,
            ) as open_absolute,
            patch.object(
                inventory_module,
                "_list_directory_names",
                wraps=inventory_module._list_directory_names,
            ) as list_names,
        ):
            with self.assertRaises(InventoryCapabilityError):
                with open_trusted_root(str(root)):
                    pass
        open_absolute.assert_not_called()
        list_names.assert_not_called()

    def test_missing_required_flags_fail_before_open(self) -> None:
        with synthetic_root() as (_temporary, root):
            for name in ("O_NOFOLLOW", "O_DIRECTORY"):
                with self.subTest(name=name), patch.object(os, name, None):
                    self._assert_capability_failure_before_open(root)

    def test_missing_descriptor_relative_open_or_stat_fails_before_open(
        self,
    ) -> None:
        with synthetic_root() as (_temporary, root):
            for function in (os.open, os.stat):
                with self.subTest(function=function.__name__):
                    supported = set(os.supports_dir_fd)
                    supported.discard(function)
                    with patch.object(os, "supports_dir_fd", supported):
                        self._assert_capability_failure_before_open(root)

    def test_missing_descriptor_directory_enumeration_fails_before_open(
        self,
    ) -> None:
        with synthetic_root() as (_temporary, root):
            supported = set(os.supports_fd)
            supported.discard(os.scandir)
            with patch.object(os, "supports_fd", supported):
                self._assert_capability_failure_before_open(root)


class InventoryGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = (
            PACKAGE_ROOT
            / "tests"
            / "fixtures"
            / "safety-v1"
            / "inventory-golden.json"
        )
        cls.fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def _materialize_golden_tree(
        self,
        root: Path,
        *,
        reverse: bool,
        timestamp: int,
    ) -> None:
        records = list(self.fixture["files"])
        if reverse:
            records.reverse()
        for record in records:
            if record["executable"]:
                mode = 0o500 if reverse else 0o700
            else:
                mode = 0o400 if reverse else 0o600
            write_file(
                root,
                record["path"],
                bytes.fromhex(record["content_hex"]),
                mode,
            )
            os.utime(
                root.joinpath(*record["path"].split("/")),
                (timestamp, timestamp),
            )
        excluded = (
            ("logs/tmp/not-included.txt", b"ignored\n"),
            ("cache/not-included.txt", b"ignored\n"),
        )
        if reverse:
            excluded = tuple(reversed(excluded))
        for relative, content in excluded:
            destination = write_file(root, relative, content)
            os.utime(destination, (timestamp + 1, timestamp + 1))

    def _assert_matches_frozen_fixture(
        self,
        result: RegularFileInventory,
    ) -> None:
        expected_bytes = bytes.fromhex(self.fixture["canonical_body_hex"])
        self.assertEqual(INVENTORY_SCHEMA, self.fixture["inventory_schema"])
        self.assertEqual(result.schema_version, INVENTORY_SCHEMA)
        self.assertEqual(result.scope, self.fixture["scope"])
        self.assertEqual(
            len(expected_bytes),
            self.fixture["canonical_body_size_bytes"],
        )
        self.assertEqual(result.canonical_bytes, expected_bytes)
        self.assertEqual(
            result.canonical_inventory_sha256.value,
            self.fixture["canonical_inventory_sha256"],
        )
        self.assertEqual(
            result.tree_digest.value,
            self.fixture["tree_digest_sha256"],
        )
        self.assertEqual(
            TREE_DIGEST_ALGORITHM,
            self.fixture["tree_digest_algorithm"],
        )
        self.assertEqual(
            TREE_DIGEST_DOMAIN.hex(),
            self.fixture["tree_digest_domain_hex"],
        )
        self.assertEqual(
            struct.pack(">Q", len(expected_bytes)).hex(),
            self.fixture["tree_digest_length_prefix_hex"],
        )

        expected_by_path = {
            record["path"]: record for record in self.fixture["files"]
        }
        for record in result.files:
            expected = expected_by_path[record.path.value]
            content = bytes.fromhex(expected["content_hex"])
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                expected["sha256"],
            )
            self.assertEqual(record.sha256.value, expected["sha256"])
            self.assertEqual(record.size_bytes, expected["size_bytes"])
            self.assertIs(record.executable, expected["executable"])

        self.assertEqual(
            hashlib.sha256(expected_bytes).hexdigest(),
            self.fixture["canonical_inventory_sha256"],
        )
        independently_framed = (
            bytes.fromhex(self.fixture["tree_digest_domain_hex"])
            + struct.pack(">Q", len(expected_bytes))
            + expected_bytes
        )
        self.assertEqual(
            hashlib.sha256(independently_framed).hexdigest(),
            self.fixture["tree_digest_sha256"],
        )

    def test_frozen_canonical_body_and_digests_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(os.path.realpath(temporary))
            results: list[RegularFileInventory] = []
            for index, reverse in enumerate((False, True)):
                root = base / f"fixture-root-{index}"
                root.mkdir()
                self._materialize_golden_tree(
                    root,
                    reverse=reverse,
                    timestamp=1_700_000_000 + (index * 1_000),
                )
                exclusions = list(self.fixture["excluded_prefixes"])
                if reverse:
                    exclusions.reverse()
                result = scan(
                    root,
                    exclusions=tuple(exclusions),
                    scope=self.fixture["scope"],
                )
                self._assert_matches_frozen_fixture(result)
                results.append(result)

        self.assertEqual(results[0].canonical_bytes, results[1].canonical_bytes)
        self.assertEqual(
            results[0].canonical_inventory_sha256,
            results[1].canonical_inventory_sha256,
        )
        self.assertEqual(results[0].tree_digest, results[1].tree_digest)

    def test_inventory_values_are_immutable(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "item.txt", b"value")
            result = scan(root)
        for field in (
            "schema_version",
            "scope",
            "excluded_prefixes",
            "files",
            "canonical_bytes",
            "canonical_inventory_sha256",
            "tree_digest",
        ):
            with self.subTest(field=field), self.assertRaises(
                FrozenInstanceError
            ):
                setattr(result, field, getattr(result, field))
        with self.assertRaises(TypeError):
            result.files[0] = result.files[0]  # type: ignore[index]
        self.assertFalse(hasattr(result, "__dict__"))
        self.assertFalse(hasattr(result.files[0], "__dict__"))

    def test_inventory_direct_construction_and_unchecked_factory_are_rejected(
        self,
    ) -> None:
        for args, kwargs in (
            ((), {}),
            (("scope",), {}),
            ((), {"scope": "synthetic-test-v1"}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(TypeError):
                    RegularFileInventory(*args, **kwargs)
        self.assertFalse(hasattr(RegularFileInventory, "_from_validated"))

        with synthetic_root() as (_temporary, root):
            write_file(root, "item.txt", b"value")
            result = scan(root)
        self.assertIs(type(result), RegularFileInventory)
        self.assertTrue(
            all(type(record) is RegularFileRecord for record in result.files)
        )
        decoded = json.loads(result.canonical_bytes)
        self.assertEqual(decoded["schema_version"], result.schema_version)
        self.assertEqual(decoded["scope"], result.scope)
        self.assertEqual(
            decoded["excluded_prefixes"],
            [path.value for path in result.excluded_prefixes],
        )
        self.assertEqual(
            decoded["files"],
            [record._canonical_value() for record in result.files],
        )
        self.assertEqual(
            hashlib.sha256(result.canonical_bytes).hexdigest(),
            result.canonical_inventory_sha256.value,
        )
        self.assertEqual(
            hashlib.sha256(
                TREE_DIGEST_DOMAIN
                + struct.pack(">Q", len(result.canonical_bytes))
                + result.canonical_bytes
            ).hexdigest(),
            result.tree_digest.value,
        )

    def test_cross_root_creation_order_timestamp_and_nonexecute_mode_determinism(
        self,
    ) -> None:
        results: list[RegularFileInventory] = []
        for reverse in (False, True):
            with synthetic_root() as (_temporary, root):
                items = [
                    ("nested/b.txt", b"beta\n", 0o600),
                    ("a.txt", b"alpha\n", 0o700),
                ]
                if reverse:
                    items.reverse()
                for relative, content, mode in items:
                    destination = write_file(root, relative, content, mode)
                    timestamp = 1_700_000_000 + (100 if reverse else 0)
                    os.utime(destination, (timestamp, timestamp))
                if reverse:
                    (root / "nested/b.txt").chmod(0o400)
                    (root / "a.txt").chmod(0o500)
                results.append(scan(root))

        first, second = results
        self.assertEqual(first.canonical_bytes, second.canonical_bytes)
        self.assertEqual(
            first.canonical_inventory_sha256,
            second.canonical_inventory_sha256,
        )
        self.assertEqual(first.tree_digest, second.tree_digest)

    def test_content_path_size_and_execute_identity_change_digests(self) -> None:
        variants = (
            ("item.txt", b"AAAA", 0o600),
            ("item.txt", b"BBBB", 0o600),
            ("renamed.txt", b"AAAA", 0o600),
            ("item.txt", b"AAAA-more", 0o600),
            ("item.txt", b"AAAA", 0o700),
        )
        results: list[RegularFileInventory] = []
        for relative, content, mode in variants:
            with synthetic_root() as (_temporary, root):
                write_file(root, relative, content, mode)
                results.append(scan(root))
        baseline = results[0]
        for changed in results[1:]:
            self.assertNotEqual(baseline.canonical_bytes, changed.canonical_bytes)
            self.assertNotEqual(baseline.tree_digest, changed.tree_digest)

    def test_nested_records_use_portable_byte_order(self) -> None:
        values = ("z.txt", "A.txt", ".hidden", "nested/z.txt", "nested/a.txt")
        with synthetic_root() as (_temporary, root):
            for value in reversed(values):
                write_file(root, value, value.encode("ascii"))
            result = scan(root)
        self.assertEqual(
            tuple(record.path.value for record in result.files),
            tuple(sorted(values, key=lambda value: value.encode("ascii"))),
        )


class InventoryAdmissionTests(unittest.TestCase):
    def test_scope_uses_a_strict_public_safe_identifier(self) -> None:
        class HostileString(str):
            pass

        invalid: tuple[object, ...] = (
            "",
            "-leading",
            "_leading",
            ".leading",
            "white space",
            "café",
            "line\nbreak",
            "a" * 129,
            HostileString("safe"),
            1,
        )
        with synthetic_root() as (_temporary, root):
            write_file(root, "item.txt", b"x")
            with open_trusted_root(str(root)) as trusted:
                for value in invalid:
                    with self.subTest(value=repr(value)):
                        with self.assertRaises(InventoryValidationError):
                            scan_regular_file_inventory(
                                trusted,
                                scope=value,  # type: ignore[arg-type]
                            )

    def test_invalid_and_case_aliased_entry_names_fail_closed(self) -> None:
        invalid_name_sets = (
            ["bad:name"],
            ["café"],
            ["white space"],
            ["line\nbreak"],
            [".", "safe.txt"],
            ["/absolute"],
            ["nested/name"],
            ["../escape"],
            ["A", "a"],
        )
        for names in invalid_name_sets:
            with self.subTest(names=names), synthetic_root() as (_temporary, root):
                write_file(root, "safe.txt", b"x")
                with (
                    open_trusted_root(str(root)) as trusted,
                    patch.object(
                        inventory_module,
                        "_list_directory_names",
                        return_value=names,
                    ),
                ):
                    with self.assertRaises(InventoryEntryError):
                        scan_regular_file_inventory(
                            trusted,
                            scope="synthetic-test-v1",
                        )

    def test_descendant_and_broken_symlinks_fail(self) -> None:
        for broken in (False, True):
            with self.subTest(broken=broken), synthetic_root() as (
                temporary,
                root,
            ):
                write_file(root, "keep.txt", b"x")
                link = root / "link"
                if broken:
                    link.symlink_to(temporary / "missing")
                else:
                    target = temporary / "outside.txt"
                    target.write_bytes(b"outside")
                    link.symlink_to(target)
                with self.assertRaises(InventoryEntryError):
                    scan(root)

    def test_hard_links_fail(self) -> None:
        with synthetic_root() as (_temporary, root):
            original = write_file(root, "original.txt", b"value")
            os.link(original, root / "linked.txt")
            with self.assertRaises(InventoryEntryError):
                scan(root)

    def _assert_special_name_is_never_opened(
        self,
        root: Path,
        *,
        name: str,
        stat_side_effect=None,
    ) -> None:
        original_directory = inventory_module._open_directory_at
        original_file = inventory_module._open_regular_file_at
        directory_names: list[str] = []
        file_names: list[str] = []

        def open_directory(selected: str, parent: int) -> int:
            directory_names.append(selected)
            return original_directory(selected, parent)

        def open_file(selected: str, parent: int) -> int:
            file_names.append(selected)
            return original_file(selected, parent)

        patches = [
            patch.object(
                inventory_module,
                "_open_directory_at",
                side_effect=open_directory,
            ),
            patch.object(
                inventory_module,
                "_open_regular_file_at",
                side_effect=open_file,
            ),
        ]
        if stat_side_effect is not None:
            patches.append(
                patch.object(
                    inventory_module,
                    "_stat_entry",
                    side_effect=stat_side_effect,
                )
            )
        with patches[0], patches[1]:
            if len(patches) == 3:
                with patches[2]:
                    with self.assertRaises(InventoryEntryError):
                        scan(root)
            else:
                with self.assertRaises(InventoryEntryError):
                    scan(root)

        self.assertIn(".", directory_names)
        self.assertIn("normal", directory_names)
        self.assertNotIn(name, directory_names)
        self.assertNotIn(name, file_names)

    def test_fifo_and_unix_socket_fail_without_opening_them(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "normal/item.txt", b"x")
            fifo = root / "zz-pipe"
            os.mkfifo(fifo)
            self._assert_special_name_is_never_opened(
                root,
                name="zz-pipe",
            )

        with synthetic_root() as (_temporary, root):
            write_file(root, "normal/item.txt", b"x")
            endpoint_path = root / "zz-socket"
            endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            simulated = False
            try:
                try:
                    endpoint.bind(str(endpoint_path))
                except OSError as exc:
                    if exc.errno not in {
                        errno.EACCES,
                        errno.EPERM,
                        errno.ENAMETOOLONG,
                    }:
                        raise
                    if os.environ.get("CI"):
                        raise AssertionError(
                            "hosted safety acceptance must exercise a real "
                            "Unix-socket entry"
                        ) from exc
                    simulated = True
                    write_file(root, "zz-socket", b"synthetic")

                if not simulated:
                    self._assert_special_name_is_never_opened(
                        root,
                        name="zz-socket",
                    )
                    return

                original_stat = inventory_module._stat_entry

                def report_socket(name: str, parent: int) -> os.stat_result:
                    result = original_stat(name, parent)
                    if name == "zz-socket":
                        return altered_stat(
                            result,
                            mode=stat.S_IFSOCK | 0o600,
                        )
                    return result

                self._assert_special_name_is_never_opened(
                    root,
                    name="zz-socket",
                    stat_side_effect=report_socket,
                )
            finally:
                endpoint.close()

    def test_simulated_device_and_cross_device_entries_fail(self) -> None:
        for kind in ("device", "cross-device-file", "cross-device-directory"):
            with self.subTest(kind=kind), synthetic_root() as (_temporary, root):
                if kind == "cross-device-directory":
                    write_file(root, "nested/item.txt", b"x")
                    selected = "nested"
                else:
                    write_file(root, "item.txt", b"x")
                    selected = "item.txt"
                original_stat = inventory_module._stat_entry

                def changed(name: str, parent: int) -> os.stat_result:
                    result = original_stat(name, parent)
                    if name != selected:
                        return result
                    if kind == "device":
                        return altered_stat(
                            result,
                            mode=stat.S_IFCHR | 0o600,
                        )
                    return altered_stat(result, device=result.st_dev + 1)

                with patch.object(
                    inventory_module,
                    "_stat_entry",
                    side_effect=changed,
                ):
                    with self.assertRaises(InventoryEntryError):
                        scan(root)

    def test_simulated_device_is_never_passed_to_an_entry_opener(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "normal/item.txt", b"x")
            write_file(root, "zz-device", b"synthetic")
            original_stat = inventory_module._stat_entry

            def report_device(name: str, parent: int) -> os.stat_result:
                result = original_stat(name, parent)
                if name == "zz-device":
                    return altered_stat(
                        result,
                        mode=stat.S_IFCHR | 0o600,
                    )
                return result

            self._assert_special_name_is_never_opened(
                root,
                name="zz-device",
                stat_side_effect=report_device,
            )

    def test_empty_root_and_empty_included_directory_fail(self) -> None:
        with synthetic_root() as (_temporary, root):
            with self.assertRaises(InventoryEntryError):
                scan(root)

        with synthetic_root() as (_temporary, root):
            write_file(root, "keep.txt", b"x")
            (root / "empty").mkdir()
            with self.assertRaises(InventoryEntryError):
                scan(root)

    def test_file_record_limit_fails_during_baseline_before_hashing(self) -> None:
        with synthetic_root() as (_temporary, root):
            for index in range(3):
                write_file(root, f"item-{index}.txt", b"x")
            with (
                patch.object(
                    inventory_module,
                    "MAX_CANONICAL_CONTAINER_ITEMS",
                    14,
                ),
                patch.object(
                    inventory_module,
                    "_open_regular_file_at",
                ) as open_file,
            ):
                with self.assertRaises(InventoryLimitError):
                    scan(root)
            open_file.assert_not_called()

    def test_directory_enumeration_is_bounded_before_entry_materialization(
        self,
    ) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "keep.txt", b"x")
            iterator_closed = False

            def too_many_names(_descriptor: int):
                nonlocal iterator_closed
                try:
                    for index in range(6):
                        yield f"item-{index}.txt"
                finally:
                    iterator_closed = True

            with (
                patch.object(
                    inventory_module,
                    "MAX_CANONICAL_CONTAINER_ITEMS",
                    5,
                ),
                patch.object(
                    inventory_module,
                    "_list_directory_names",
                    side_effect=too_many_names,
                ),
                patch.object(
                    inventory_module,
                    "_stat_entry",
                ) as stat_entry,
            ):
                with self.assertRaises(InventoryLimitError):
                    scan(root)
            self.assertTrue(iterator_closed)
            stat_entry.assert_not_called()


class InventoryExclusionTests(unittest.TestCase):
    def test_excluded_subtree_is_pruned_and_declaration_is_canonical(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "keep.txt", b"keep")
            excluded = root / "private"
            excluded.mkdir()
            (excluded / "broken").symlink_to(excluded / "missing")
            write_file(excluded, "bad:name", b"not scanned")
            result = scan(
                root,
                exclusions=("private", "absent/subtree"),
            )
        self.assertEqual(
            tuple(path.value for path in result.excluded_prefixes),
            ("absent/subtree", "private"),
        )
        self.assertEqual(
            tuple(record.path.value for record in result.files),
            ("keep.txt",),
        )
        decoded = json.loads(result.canonical_bytes)
        self.assertEqual(
            decoded["excluded_prefixes"],
            ["absent/subtree", "private"],
        )

    def test_duplicate_case_aliased_and_overlapping_exclusions_fail(self) -> None:
        cases = (
            ("cache", "cache"),
            ("Cache", "cache"),
            ("cache", "cache/tmp"),
            ("cache/tmp", "cache"),
        )
        for values in cases:
            with self.subTest(values=values), synthetic_root() as (
                _temporary,
                root,
            ):
                write_file(root, "keep.txt", b"x")
                exclusions = tuple(
                    PortableRelativePath.parse(value) for value in values
                )
                with open_trusted_root(str(root)) as trusted:
                    with self.assertRaises(InventoryValidationError):
                        scan_regular_file_inventory(
                            trusted,
                            scope="synthetic-test-v1",
                            excluded_prefixes=exclusions,
                        )

    def test_exclusions_require_exact_portable_authorities(self) -> None:
        class HostilePath(PortableRelativePath):
            pass

        with synthetic_root() as (_temporary, root):
            write_file(root, "keep.txt", b"x")
            with open_trusted_root(str(root)) as trusted:
                for values in (("cache",), (HostilePath("cache"),)):
                    with self.subTest(values=values):
                        with self.assertRaises(InventoryValidationError):
                            scan_regular_file_inventory(
                                trusted,
                                scope="synthetic-test-v1",
                                excluded_prefixes=values,  # type: ignore[arg-type]
                            )

    def test_exclusion_boundary_must_be_a_same_device_real_directory(self) -> None:
        kinds = ("file", "symlink", "broken-symlink", "fifo")
        for kind in kinds:
            with self.subTest(kind=kind), synthetic_root() as (temporary, root):
                write_file(root, "keep.txt", b"x")
                boundary = root / "excluded"
                if kind == "file":
                    boundary.write_bytes(b"x")
                elif kind == "symlink":
                    target = temporary / "outside"
                    target.mkdir()
                    boundary.symlink_to(target, target_is_directory=True)
                elif kind == "broken-symlink":
                    boundary.symlink_to(temporary / "missing")
                elif kind == "fifo":
                    os.mkfifo(boundary)
                with self.assertRaises(InventoryEntryError):
                    scan(root, exclusions=("excluded",))

        with synthetic_root() as (_temporary, root):
            write_file(root, "keep.txt", b"x")
            write_file(root, "excluded", b"synthetic")
            original_stat = inventory_module._stat_entry

            def report_socket(name: str, parent: int) -> os.stat_result:
                result = original_stat(name, parent)
                if name == "excluded":
                    return altered_stat(result, mode=stat.S_IFSOCK | 0o600)
                return result

            with patch.object(
                inventory_module,
                "_stat_entry",
                side_effect=report_socket,
            ):
                with self.assertRaises(InventoryEntryError):
                    scan(root, exclusions=("excluded",))

        with synthetic_root() as (_temporary, root):
            write_file(root, "keep.txt", b"x")
            (root / "excluded").mkdir()
            original_stat = inventory_module._stat_entry

            def report_other_device(name: str, parent: int) -> os.stat_result:
                result = original_stat(name, parent)
                if name == "excluded":
                    return altered_stat(result, device=result.st_dev + 1)
                return result

            with patch.object(
                inventory_module,
                "_stat_entry",
                side_effect=report_other_device,
            ):
                with self.assertRaises(InventoryEntryError):
                    scan(root, exclusions=("excluded",))

    def test_case_alias_of_declared_exclusion_fails(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "keep.txt", b"x")
            (root / "cache").mkdir()
            with self.assertRaises(InventoryEntryError):
                scan(root, exclusions=("Cache",))

    def test_excluding_all_files_still_fails_empty_root_contract(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "excluded/item.txt", b"x")
            with self.assertRaises(InventoryEntryError):
                scan(root, exclusions=("excluded",))

        with synthetic_root() as (_temporary, root):
            write_file(root, "keep.txt", b"x")
            write_file(root, "parent/excluded/item.txt", b"x")
            with self.assertRaises(InventoryEntryError):
                scan(root, exclusions=("parent/excluded",))


class InventoryMutationTests(unittest.TestCase):
    def _assert_flat_mutation_is_detected(self, mutation) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "item.txt", b"AAAA")
            original_list = inventory_module._list_directory_names
            calls = 0

            def mutate_before_hash(descriptor: int) -> list[str]:
                nonlocal calls
                calls += 1
                if calls == 3:
                    mutation(root)
                return original_list(descriptor)

            with patch.object(
                inventory_module,
                "_list_directory_names",
                side_effect=mutate_before_hash,
            ):
                with self.assertRaises(InventoryMutationError):
                    scan(root)

    def test_addition_deletion_and_replacement_are_detected(self) -> None:
        def add(root: Path) -> None:
            write_file(root, "added.txt", b"new")

        def delete(root: Path) -> None:
            (root / "item.txt").unlink()

        def replace(root: Path) -> None:
            destination = root / "item.txt"
            destination.unlink()
            destination.write_bytes(b"AAAA")

        for name, mutation in (
            ("addition", add),
            ("deletion", delete),
            ("replacement", replace),
        ):
            with self.subTest(name=name):
                self._assert_flat_mutation_is_detected(mutation)

    def test_same_size_content_mutation_during_hashing_is_detected(self) -> None:
        with synthetic_root() as (_temporary, root):
            destination = write_file(root, "item.txt", b"AAAA")
            original_read = inventory_module._read_descriptor_chunk
            changed = False

            def mutate_after_read(descriptor: int, size: int) -> bytes:
                nonlocal changed
                chunk = original_read(descriptor, size)
                if not changed and chunk:
                    changed = True
                    destination.write_bytes(b"BBBB")
                return chunk

            with patch.object(
                inventory_module,
                "_read_descriptor_chunk",
                side_effect=mutate_after_read,
            ):
                with self.assertRaises(InventoryMutationError):
                    scan(root)
            self.assertTrue(changed)

    def test_directory_replacement_is_detected(self) -> None:
        with synthetic_root() as (temporary, root):
            write_file(root, "nested/item.txt", b"value")
            original_list = inventory_module._list_directory_names
            calls = 0

            def replace_before_hash(descriptor: int) -> list[str]:
                nonlocal calls
                calls += 1
                if calls == 5:
                    original = root / "nested"
                    moved = temporary / "former-nested"
                    original.rename(moved)
                    replacement = root / "nested"
                    replacement.mkdir()
                    write_file(replacement, "item.txt", b"value")
                return original_list(descriptor)

            with patch.object(
                inventory_module,
                "_list_directory_names",
                side_effect=replace_before_hash,
            ):
                with self.assertRaises(InventoryMutationError):
                    scan(root)

    def test_regular_file_stat_to_open_races_fail_closed(self) -> None:
        for replacement_kind in ("regular", "symlink", "fifo", "special"):
            with self.subTest(replacement_kind=replacement_kind), synthetic_root() as (
                temporary,
                root,
            ):
                destination = write_file(root, "item.txt", b"AAAA")
                sentinel = temporary / "outside-sentinel.txt"
                sentinel.write_bytes(b"outside-sentinel")
                sentinel.chmod(0o640)
                sentinel_before = (
                    sentinel.read_bytes(),
                    stat.S_IMODE(sentinel.stat().st_mode),
                    sentinel.stat().st_ino,
                )
                moved = temporary / f"former-{replacement_kind}.txt"
                changed = False
                race_descriptor: int | None = None
                outside_read = False
                original_fstat = inventory_module._fstat_descriptor
                original_read = inventory_module._read_descriptor_chunk

                with record_inventory_descriptors() as descriptors:
                    recording_open = inventory_module._open_regular_file_at

                    def replace_then_open(name: str, parent: int) -> int:
                        nonlocal changed, race_descriptor
                        if name == "item.txt" and not changed:
                            changed = True
                            destination.rename(moved)
                            if replacement_kind == "regular":
                                destination.write_bytes(b"BBBB")
                            elif replacement_kind == "symlink":
                                destination.symlink_to(sentinel)
                            elif replacement_kind == "fifo":
                                os.mkfifo(destination)
                            else:
                                destination.write_bytes(b"device-placeholder")
                        descriptor = recording_open(name, parent)
                        if name == "item.txt" and changed:
                            race_descriptor = descriptor
                        return descriptor

                    def report_special(descriptor: int) -> os.stat_result:
                        result = original_fstat(descriptor)
                        if (
                            replacement_kind == "special"
                            and descriptor == race_descriptor
                        ):
                            return altered_stat(
                                result,
                                mode=stat.S_IFCHR | 0o600,
                            )
                        return result

                    def reject_outside_read(descriptor: int, size: int) -> bytes:
                        nonlocal outside_read
                        result = os.fstat(descriptor)
                        if (
                            result.st_dev == sentinel.stat().st_dev
                            and result.st_ino == sentinel_before[2]
                        ):
                            outside_read = True
                            raise AssertionError("outside sentinel was read")
                        return original_read(descriptor, size)

                    with (
                        patch.object(
                            inventory_module,
                            "_open_regular_file_at",
                            side_effect=replace_then_open,
                        ),
                        patch.object(
                            inventory_module,
                            "_fstat_descriptor",
                            side_effect=report_special,
                        ),
                        patch.object(
                            inventory_module,
                            "_read_descriptor_chunk",
                            side_effect=reject_outside_read,
                        ),
                        open_trusted_root(str(root)) as trusted,
                    ):
                        with self.assertRaises(InventoryMutationError):
                            scan_regular_file_inventory(
                                trusted,
                                scope="synthetic-test-v1",
                            )
                        self.assertEqual(descriptors.live_count, 1)

                self.assertTrue(changed)
                self.assertFalse(outside_read)
                self.assertEqual(
                    (
                        sentinel.read_bytes(),
                        stat.S_IMODE(sentinel.stat().st_mode),
                        sentinel.stat().st_ino,
                    ),
                    sentinel_before,
                )
                self.assertEqual(descriptors.live_count, 0)

    def test_included_and_excluded_directories_replaced_before_open_fail(
        self,
    ) -> None:
        for boundary_kind in ("included", "excluded"):
            with self.subTest(boundary_kind=boundary_kind), synthetic_root() as (
                temporary,
                root,
            ):
                write_file(root, "keep.txt", b"keep")
                selected = root / boundary_kind
                write_file(selected, "item.txt", b"original")
                moved = temporary / f"former-{boundary_kind}"
                changed = False

                with record_inventory_descriptors() as descriptors:
                    with open_trusted_root(str(root)) as trusted:
                        recording_open = inventory_module._open_directory_at

                        def replace_then_open(name: str, parent: int) -> int:
                            nonlocal changed
                            if name == boundary_kind and not changed:
                                changed = True
                                selected.rename(moved)
                                selected.mkdir()
                                write_file(selected, "item.txt", b"replacement")
                            return recording_open(name, parent)

                        exclusions = (
                            (PortableRelativePath.parse("excluded"),)
                            if boundary_kind == "excluded"
                            else ()
                        )
                        with patch.object(
                            inventory_module,
                            "_open_directory_at",
                            side_effect=replace_then_open,
                        ):
                            with self.assertRaises(InventoryMutationError):
                                scan_regular_file_inventory(
                                    trusted,
                                    scope="synthetic-test-v1",
                                    excluded_prefixes=exclusions,
                                )
                        self.assertEqual(descriptors.live_count, 1)
                self.assertTrue(changed)
                self.assertEqual(
                    (moved / "item.txt").read_bytes(),
                    b"original",
                )
                self.assertEqual(descriptors.live_count, 0)

    def test_post_hash_content_and_execute_mode_mutations_fail_closed(
        self,
    ) -> None:
        for mutation_kind in ("content", "execute"):
            with self.subTest(mutation_kind=mutation_kind), synthetic_root() as (
                _temporary,
                root,
            ):
                destination = write_file(root, "item.txt", b"AAAA", 0o600)
                original_read = inventory_module._read_descriptor_chunk
                changed = False

                def mutate_after_eof_probe(
                    descriptor: int,
                    size: int,
                ) -> bytes:
                    nonlocal changed
                    chunk = original_read(descriptor, size)
                    if not changed and size == 1 and chunk == b"":
                        changed = True
                        if mutation_kind == "content":
                            destination.write_bytes(b"BBBB")
                        else:
                            destination.chmod(0o700)
                    return chunk

                with record_inventory_descriptors() as descriptors:
                    with (
                        open_trusted_root(str(root)) as trusted,
                        patch.object(
                            inventory_module,
                            "_read_descriptor_chunk",
                            side_effect=mutate_after_eof_probe,
                        ),
                    ):
                        with self.assertRaises(InventoryMutationError):
                            scan_regular_file_inventory(
                                trusted,
                                scope="synthetic-test-v1",
                            )
                        self.assertEqual(descriptors.live_count, 1)
                self.assertTrue(changed)
                self.assertEqual(descriptors.live_count, 0)

    def test_excluded_boundary_replacement_between_passes_is_detected(
        self,
    ) -> None:
        for after_pass in (1, 2):
            with self.subTest(after_pass=after_pass), synthetic_root() as (
                temporary,
                root,
            ):
                write_file(root, "keep.txt", b"keep")
                boundary = root / "excluded"
                write_file(boundary, "private.txt", b"private")
                moved = temporary / f"former-excluded-{after_pass}"
                original_capture = inventory_module._capture_tree
                calls = 0
                changed = False

                def capture_then_replace(*args: object, **kwargs: object):
                    nonlocal calls, changed
                    result = original_capture(*args, **kwargs)
                    calls += 1
                    if calls == after_pass:
                        changed = True
                        boundary.rename(moved)
                        boundary.mkdir()
                        write_file(boundary, "replacement.txt", b"replacement")
                    return result

                with record_inventory_descriptors() as descriptors:
                    with (
                        open_trusted_root(str(root)) as trusted,
                        patch.object(
                            inventory_module,
                            "_capture_tree",
                            side_effect=capture_then_replace,
                        ),
                    ):
                        with self.assertRaises(InventoryMutationError):
                            scan_regular_file_inventory(
                                trusted,
                                scope="synthetic-test-v1",
                                excluded_prefixes=(
                                    PortableRelativePath.parse("excluded"),
                                ),
                            )
                        self.assertEqual(descriptors.live_count, 1)
                self.assertTrue(changed)
                self.assertEqual(
                    (moved / "private.txt").read_bytes(),
                    b"private",
                )
                self.assertEqual(descriptors.live_count, 0)

    def test_hashing_streams_and_opens_each_file_exactly_once(self) -> None:
        with synthetic_root() as (_temporary, root):
            write_file(root, "large.bin", b"a" * (1024 * 1024 + 17))
            write_file(root, "nested/small.bin", b"small")
            original_open = inventory_module._open_regular_file_at
            original_read = inventory_module._read_descriptor_chunk
            opened: list[str] = []
            requested_sizes: list[int] = []

            def record_open(name: str, parent: int) -> int:
                opened.append(name)
                return original_open(name, parent)

            def record_read(descriptor: int, size: int) -> bytes:
                requested_sizes.append(size)
                return original_read(descriptor, size)

            with (
                patch.object(
                    inventory_module,
                    "_open_regular_file_at",
                    side_effect=record_open,
                ),
                patch.object(
                    inventory_module,
                    "_read_descriptor_chunk",
                    side_effect=record_read,
                ),
            ):
                result = scan(root)

        self.assertEqual(Counter(opened), Counter({"large.bin": 1, "small.bin": 1}))
        self.assertGreaterEqual(requested_sizes.count(1024 * 1024), 1)
        self.assertIn(17, requested_sizes)
        self.assertEqual(len(result.files), 2)


if __name__ == "__main__":
    unittest.main()
