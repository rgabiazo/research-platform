"""Descriptor-anchored regular-file inventories for managed HPC payloads.

The scanner detects mutations observable across repeated descriptor-anchored
passes.  It does not provide an atomic filesystem snapshot and deliberately
does not implement publication, claims, receipts, transfer, or remote work.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import os
import re
import stat
from typing import Iterable, Iterator

from .canonical import (
    MAX_CANONICAL_CONTAINER_ITEMS,
    MAX_CANONICAL_DOCUMENT_BYTES,
    MAX_SIGNED_64,
    TREE_DIGEST_DOMAIN,
    CanonicalJsonError,
    Sha256Digest,
    canonical_json_bytes,
    domain_separated_sha256,
)
from .paths import (
    PortablePathError,
    PortableRelativePath,
    portable_path_sort_key,
    require_distinct_file_paths,
)


INVENTORY_SCHEMA = "research_platform.hpc.regular_file_inventory.v1"
MAX_INVENTORY_SCOPE_BYTES = 128

_SCOPE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HASH_CHUNK_BYTES = 1024 * 1024


class InventorySafetyError(ValueError):
    """Base class for a rejected H2b filesystem safety operation."""


class InventoryCapabilityError(InventorySafetyError):
    """Raised when the host lacks a required POSIX descriptor primitive."""


class TrustedRootError(InventorySafetyError):
    """Raised when a trusted root cannot be opened or is no longer valid."""


class InventoryValidationError(InventorySafetyError):
    """Raised when an inventory request is structurally invalid."""


class InventoryEntryError(InventoryValidationError):
    """Raised when a payload entry violates the regular-file contract."""


class InventoryLimitError(InventoryValidationError):
    """Raised when bounded inventory state would exceed protocol limits."""


class InventoryMutationError(InventorySafetyError):
    """Raised when an observable mutation prevents an authoritative result."""


@dataclass(frozen=True, slots=True)
class RegularFileRecord:
    """One immutable canonical regular-file identity."""

    path: PortableRelativePath
    size_bytes: int
    sha256: Sha256Digest
    executable: bool

    def __post_init__(self) -> None:
        if type(self.path) is not PortableRelativePath:
            raise TypeError("record path must be exactly PortableRelativePath")
        if (
            type(self.size_bytes) is not int
            or self.size_bytes < 0
            or self.size_bytes > MAX_SIGNED_64
        ):
            raise ValueError(
                "record size_bytes must be a nonnegative signed-64 integer"
            )
        if type(self.sha256) is not Sha256Digest:
            raise TypeError("record sha256 must be exactly Sha256Digest")
        if type(self.executable) is not bool:
            raise TypeError("record executable must be exactly bool")

    def _canonical_value(self) -> dict[str, object]:
        return {
            "path": self.path.value,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256.value,
            "executable": self.executable,
        }


@dataclass(frozen=True, init=False, slots=True)
class RegularFileInventory:
    """One immutable authoritative inventory and its frozen digest identities."""

    schema_version: str
    scope: str
    excluded_prefixes: tuple[PortableRelativePath, ...]
    files: tuple[RegularFileRecord, ...]
    canonical_bytes: bytes
    canonical_inventory_sha256: Sha256Digest
    tree_digest: Sha256Digest

    def __new__(cls, *args: object, **kwargs: object) -> "RegularFileInventory":
        raise TypeError(
            "RegularFileInventory values are created by "
            "scan_regular_file_inventory"
        )


class TrustedRoot:
    """A pinned descriptor authority yielded only by ``open_trusted_root``."""

    __slots__ = ()

    def __new__(cls, *args: object, **kwargs: object) -> "TrustedRoot":
        raise TypeError("TrustedRoot values are created by open_trusted_root")

    @property
    def device(self) -> int:
        """Return the pinned root device identity."""

        state, _current = _require_live_trusted_root(self)
        return state.device

    @property
    def inode(self) -> int:
        """Return the pinned root inode identity."""

        state, _current = _require_live_trusted_root(self)
        return state.inode

    @property
    def is_open(self) -> bool:
        """Report whether this authority remains registered as live."""

        return _trusted_root_state(self) is not None

    @property
    def is_closed(self) -> bool:
        """Report whether this authority has been permanently retired."""

        return _trusted_root_state(self) is None

    def _require_live(self) -> os.stat_result:
        _state, current = _require_live_trusted_root(self)
        return current

    def close(self) -> None:
        """Close this handle's owned descriptor."""

        state = _trusted_root_state(self)
        if state is None:
            raise TrustedRootError("trusted root handle is closed or forged")
        descriptor = state.descriptor
        _retire_trusted_root(self, state)
        try:
            current = _fstat_descriptor(descriptor)
        except OSError as exc:
            raise TrustedRootError(
                "trusted root descriptor was already closed"
            ) from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != state.device
            or current.st_ino != state.inode
        ):
            raise TrustedRootError(
                "trusted root descriptor was replaced; foreign state preserved"
            )
        try:
            _close_descriptor(descriptor)
        except OSError as exc:
            raise TrustedRootError(
                "failed to close trusted root descriptor"
            ) from exc

    def __copy__(self) -> "TrustedRoot":
        raise TypeError("TrustedRoot handles cannot be copied")

    def __deepcopy__(self, memo: object) -> "TrustedRoot":
        raise TypeError("TrustedRoot handles cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("TrustedRoot handles cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("TrustedRoot handles cannot be serialized")


@dataclass(frozen=True, slots=True)
class _TrustedRootState:
    handle: TrustedRoot
    descriptor: int
    device: int
    inode: int


def _create_trusted_root_authority():
    live: dict[int, _TrustedRootState] = {}

    def state_for(handle: object) -> _TrustedRootState | None:
        if type(handle) is not TrustedRoot:
            return None
        state = live.get(id(handle))
        if state is None or state.handle is not handle:
            return None
        return state

    def retire(handle: TrustedRoot, state: _TrustedRootState) -> None:
        current = live.get(id(handle))
        if current is not state or current.handle is not handle:
            raise TrustedRootError("trusted root authority changed during close")
        del live[id(handle)]

    @contextmanager
    def open_trusted_root(
        absolute_path: str,
    ) -> Iterator[TrustedRoot]:
        """Yield one pinned root; validation and opening begin on context entry."""

        descriptor = -1
        handle: TrustedRoot | None = None
        try:
            descriptor, opened = _open_trusted_root_descriptor(absolute_path)
            handle = object.__new__(TrustedRoot)
            state = _TrustedRootState(
                handle=handle,
                descriptor=descriptor,
                device=opened.st_dev,
                inode=opened.st_ino,
            )
            if id(handle) in live:
                raise TrustedRootError(
                    "trusted root authority identity unexpectedly collided"
                )
            live[id(handle)] = state
            descriptor = -1
            yield handle
        finally:
            if handle is not None and state_for(handle) is not None:
                handle.close()
            if descriptor >= 0:
                _close_descriptor(descriptor)

    return open_trusted_root, state_for, retire


(
    open_trusted_root,
    _trusted_root_state,
    _retire_trusted_root,
) = _create_trusted_root_authority()
del _create_trusted_root_authority


def _require_live_trusted_root(
    handle: TrustedRoot,
) -> tuple[_TrustedRootState, os.stat_result]:
    """Return live closure-held state or permanently retire invalid authority."""

    state = _trusted_root_state(handle)
    if state is None:
        raise TrustedRootError("trusted root handle is closed or forged")
    try:
        current = _fstat_descriptor(state.descriptor)
    except OSError as exc:
        _retire_trusted_root(handle, state)
        raise TrustedRootError(
            "trusted root descriptor is no longer available"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_dev != state.device
        or current.st_ino != state.inode
    ):
        _retire_trusted_root(handle, state)
        raise TrustedRootError("trusted root descriptor identity changed")
    return state, current


def _open_fresh_trusted_root_directory(handle: TrustedRoot) -> int:
    """Open a scanner-owned root descriptor without exposing the pinned one."""

    state, _current = _require_live_trusted_root(handle)
    try:
        return _open_directory_at(".", state.descriptor)
    except OSError as exc:
        raise TrustedRootError(
            "unable to open a fresh trusted-root directory stream"
        ) from exc


@dataclass(frozen=True)
class _StatFingerprint:
    device: int
    inode: int
    mode: int
    link_count: int
    user_id: int
    group_id: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _BoundaryIdentity:
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class _TreeSnapshot:
    root: _StatFingerprint
    directories: tuple[tuple[PortableRelativePath, _StatFingerprint], ...]
    files: tuple[tuple[PortableRelativePath, _StatFingerprint], ...]
    exclusions: tuple[tuple[PortableRelativePath, _BoundaryIdentity], ...]
    records: tuple[RegularFileRecord, ...]


@dataclass
class _InventoryBudget:
    maximum_files: int
    accumulated_path_bytes: int
    observed_entries: int = 0
    observed_files: int = 0

    def observe(self, path: PortableRelativePath) -> None:
        self.observed_entries += 1
        if self.observed_entries > MAX_CANONICAL_CONTAINER_ITEMS:
            raise InventoryLimitError(
                "inventory exceeds the maximum observed-entry count"
            )
        self.accumulated_path_bytes += len(path.value.encode("ascii"))
        if self.accumulated_path_bytes > MAX_CANONICAL_DOCUMENT_BYTES:
            raise InventoryLimitError(
                "inventory exceeds the maximum accumulated path bytes"
            )

    def observe_file(self) -> None:
        self.observed_files += 1
        if self.observed_files > self.maximum_files:
            raise InventoryLimitError(
                "inventory exceeds the canonical file-record item budget"
            )


@dataclass
class _ExclusionNode:
    spelling: str | None = None
    terminal: bool = False
    children: dict[str, "_ExclusionNode"] | None = None

    def child_map(self) -> dict[str, "_ExclusionNode"]:
        if self.children is None:
            self.children = {}
        return self.children


@dataclass(frozen=True)
class _ExclusionMatcher:
    root: _ExclusionNode

    def classify(self, path: PortableRelativePath) -> str:
        node = self.root
        for component in path.parts:
            children = node.children
            if not children:
                return "unrelated"
            child = children.get(component.lower())
            if child is None:
                return "unrelated"
            if child.spelling != component:
                raise InventoryEntryError(
                    "payload path is a case-insensitive alias of an exclusion"
                )
            node = child
            if node.terminal:
                return "boundary"
        return "ancestor" if node.children else "unrelated"


@dataclass
class _OpenedDirectory:
    descriptor: int
    parent_descriptor: int | None
    leaf_name: str | None
    fingerprint: _StatFingerprint

    def close(self) -> None:
        errors: list[OSError] = []
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            try:
                _close_descriptor(descriptor)
            except OSError as exc:
                errors.append(exc)
        if self.parent_descriptor is not None and self.parent_descriptor >= 0:
            descriptor = self.parent_descriptor
            self.parent_descriptor = -1
            try:
                _close_descriptor(descriptor)
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise InventorySafetyError(
                "failed to close an inventory directory descriptor"
            ) from errors[0]


def _open_trusted_root_descriptor(
    absolute_path: str,
) -> tuple[int, os.stat_result]:
    """Open and return one validated root descriptor without creating authority."""

    _require_descriptor_capabilities()
    components = _validate_trusted_root_path(absolute_path)
    flags = _directory_open_flags()
    descriptor = -1
    try:
        descriptor = _open_absolute_root(flags)
        root_stat = _fstat_descriptor(descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise TrustedRootError("POSIX root is not a directory")
        admitted_fingerprint = _fingerprint(root_stat)

        for component in components:
            before = _stat_root_component(component, descriptor)
            if not stat.S_ISDIR(before.st_mode):
                raise TrustedRootError(
                    "trusted root path contains a symlink or non-directory"
                )
            try:
                child = _open_directory_at(component, descriptor)
            except OSError as exc:
                raise TrustedRootError(
                    "trusted root path contains a symlink or non-directory"
                ) from exc
            try:
                opened = _fstat_descriptor(child)
                after = _stat_root_component(component, descriptor)
                before_fingerprint = _fingerprint(before)
                opened_fingerprint = _fingerprint(opened)
                after_fingerprint = _fingerprint(after)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or before_fingerprint != opened_fingerprint
                    or opened_fingerprint != after_fingerprint
                ):
                    raise TrustedRootError(
                        "trusted root path changed while it was opened"
                    )
                admitted_fingerprint = opened_fingerprint
            except BaseException:
                _close_descriptor(child)
                raise
            previous = descriptor
            descriptor = child
            _close_descriptor(previous)

        final = _fstat_descriptor(descriptor)
        if (
            not stat.S_ISDIR(final.st_mode)
            or _fingerprint(final) != admitted_fingerprint
        ):
            raise TrustedRootError(
                "trusted root changed before its descriptor was pinned"
            )
        result = (descriptor, final)
        descriptor = -1
        return result
    except InventorySafetyError:
        raise
    except (OSError, ValueError) as exc:
        raise TrustedRootError("unable to open trusted root safely") from exc
    finally:
        if descriptor >= 0:
            _close_descriptor(descriptor)


def scan_regular_file_inventory(
    trusted_root: TrustedRoot,
    *,
    scope: str,
    excluded_prefixes: Iterable[PortableRelativePath] = (),
) -> RegularFileInventory:
    """Return one immutable inventory after three anchored stability passes."""

    _require_descriptor_capabilities()
    root_stat = _require_exact_live_root(trusted_root)
    validated_scope = _validate_scope(scope)
    exclusions = _validate_exclusions(excluded_prefixes)
    matcher = _build_exclusion_matcher(exclusions)
    maximum_files = (
        MAX_CANONICAL_CONTAINER_ITEMS - 4 - len(exclusions)
    ) // 5
    exclusion_path_bytes = sum(
        len(exclusion.value.encode("ascii")) for exclusion in exclusions
    )

    try:
        baseline = _capture_tree(
            trusted_root,
            root_stat=root_stat,
            matcher=matcher,
            hash_files=False,
            expected=None,
            maximum_files=maximum_files,
            initial_path_bytes=exclusion_path_bytes,
        )
    except InventorySafetyError:
        raise
    except OSError as exc:
        raise InventoryMutationError(
            "payload changed or became unavailable during the baseline pass"
        ) from exc
    try:
        hashed = _capture_tree(
            trusted_root,
            root_stat=root_stat,
            matcher=matcher,
            hash_files=True,
            expected=baseline,
            maximum_files=maximum_files,
            initial_path_bytes=exclusion_path_bytes,
        )
        _require_same_tree(baseline, hashed)
        final = _capture_tree(
            trusted_root,
            root_stat=root_stat,
            matcher=matcher,
            hash_files=False,
            expected=baseline,
            maximum_files=maximum_files,
            initial_path_bytes=exclusion_path_bytes,
        )
        _require_same_tree(baseline, final)
    except InventoryMutationError:
        raise
    except InventoryCapabilityError:
        raise
    except TrustedRootError:
        raise
    except InventorySafetyError as exc:
        raise InventoryMutationError(
            "payload changed after the baseline inventory pass"
        ) from exc
    except OSError as exc:
        raise InventoryMutationError(
            "payload became unavailable after the baseline inventory pass"
        ) from exc

    records = tuple(
        sorted(hashed.records, key=lambda record: portable_path_sort_key(record.path))
    )
    try:
        file_paths = require_distinct_file_paths(
            tuple(record.path for record in records)
        )
    except (PortablePathError, TypeError) as exc:
        raise InventoryValidationError(
            "inventory file paths are not collision-free"
        ) from exc
    if file_paths != tuple(record.path for record in records):
        raise InventoryValidationError("inventory file ordering is inconsistent")
    _require_canonical_item_budget(exclusions, records)
    try:
        return _build_regular_file_inventory(
            scope=validated_scope,
            excluded_prefixes=exclusions,
            files=records,
        )
    except CanonicalJsonError as exc:
        raise InventoryLimitError(
            "canonical inventory body exceeds the frozen H2a limits"
        ) from exc


def _require_descriptor_capabilities() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if any(type(getattr(os, name, None)) is not int for name in required_flags):
        raise InventoryCapabilityError(
            "required POSIX no-follow descriptor flags are unavailable"
        )
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", ())
    supports_fd = getattr(os, "supports_fd", ())
    if (
        not callable(getattr(os, "open", None))
        or not callable(getattr(os, "stat", None))
        or not callable(getattr(os, "scandir", None))
        or os.open not in supports_dir_fd
        or os.stat not in supports_dir_fd
        or os.stat not in supports_follow_symlinks
        or os.scandir not in supports_fd
    ):
        raise InventoryCapabilityError(
            "required descriptor-relative POSIX operations are unavailable"
        )


def _validate_trusted_root_path(value: object) -> tuple[str, ...]:
    if type(value) is not str:
        raise TrustedRootError("trusted root path must be exactly a string")
    if not value.startswith("/"):
        raise TrustedRootError("trusted root path must be absolute")
    if "\x00" in value:
        raise TrustedRootError("trusted root path must not contain NUL")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TrustedRootError(
            "trusted root path must not contain control characters"
        )
    if value == "/":
        return ()
    if value.endswith("/") or "//" in value:
        raise TrustedRootError(
            "trusted root path must be lexically normalized"
        )
    components = tuple(value[1:].split("/"))
    if any(component in {"", ".", ".."} for component in components):
        raise TrustedRootError(
            "trusted root path must not contain '.', '..', or empty components"
        )
    return components


def _validate_scope(value: object) -> str:
    if type(value) is not str or _SCOPE_PATTERN.fullmatch(value) is None:
        raise InventoryValidationError(
            "scope must be an ASCII identifier of 1 through "
            f"{MAX_INVENTORY_SCOPE_BYTES} bytes using letters, digits, '.', "
            "'_', or '-'"
        )
    return value


def _validate_exclusions(
    values: Iterable[PortableRelativePath],
) -> tuple[PortableRelativePath, ...]:
    accepted: list[PortableRelativePath] = []
    root = _ExclusionNode()
    accumulated_path_bytes = 0
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise InventoryValidationError(
            "excluded_prefixes must be iterable"
        ) from exc

    for value in iterator:
        if 4 + len(accepted) + 1 > MAX_CANONICAL_CONTAINER_ITEMS:
            raise InventoryLimitError("too many excluded prefixes")
        if type(value) is not PortableRelativePath:
            raise InventoryValidationError(
                "excluded prefixes must be exact PortableRelativePath values"
            )
        accumulated_path_bytes += len(value.value.encode("ascii"))
        if accumulated_path_bytes > MAX_CANONICAL_DOCUMENT_BYTES:
            raise InventoryLimitError(
                "excluded prefixes exceed the accumulated path-byte limit"
            )
        accepted.append(value)
        node = root
        for component in value.parts:
            if node.terminal:
                raise InventoryValidationError(
                    "excluded prefixes must not overlap or be redundant"
                )
            children = node.child_map()
            key = component.lower()
            child = children.get(key)
            if child is None:
                child = _ExclusionNode(spelling=component)
                children[key] = child
            elif child.spelling != component:
                raise InventoryValidationError(
                    "excluded prefixes contain a case-insensitive alias"
                )
            node = child
        if node.terminal:
            raise InventoryValidationError(
                "excluded prefixes contain an exact duplicate"
            )
        if node.children:
            raise InventoryValidationError(
                "excluded prefixes must not overlap or be redundant"
            )
        node.terminal = True

    return tuple(sorted(accepted, key=portable_path_sort_key))


def _build_exclusion_matcher(
    exclusions: tuple[PortableRelativePath, ...],
) -> _ExclusionMatcher:
    root = _ExclusionNode()
    for exclusion in exclusions:
        node = root
        for component in exclusion.parts:
            children = node.child_map()
            node = children.setdefault(
                component.lower(),
                _ExclusionNode(spelling=component),
            )
        node.terminal = True
    return _ExclusionMatcher(root)


def _require_exact_live_root(trusted_root: object) -> os.stat_result:
    if type(trusted_root) is not TrustedRoot:
        raise TrustedRootError(
            "inventory admission requires an exact TrustedRoot handle"
        )
    return trusted_root._require_live()


def _capture_tree(
    trusted_root: TrustedRoot,
    *,
    root_stat: os.stat_result,
    matcher: _ExclusionMatcher,
    hash_files: bool,
    expected: _TreeSnapshot | None,
    maximum_files: int,
    initial_path_bytes: int,
) -> _TreeSnapshot:
    current_root = trusted_root._require_live()
    if _identity(current_root) != _identity(root_stat):
        raise TrustedRootError("trusted root descriptor identity changed")

    root_fingerprint = _fingerprint(current_root)
    if expected is not None and root_fingerprint != expected.root:
        raise InventoryMutationError("trusted root metadata changed")

    expected_directories = dict(expected.directories) if expected else {}
    expected_files = dict(expected.files) if expected else {}
    expected_exclusions = dict(expected.exclusions) if expected else {}

    directories: dict[PortableRelativePath, _StatFingerprint] = {}
    files: dict[PortableRelativePath, _StatFingerprint] = {}
    exclusions: dict[PortableRelativePath, _BoundaryIdentity] = {}
    records: list[RegularFileRecord] = []
    budget = _InventoryBudget(
        maximum_files=maximum_files,
        accumulated_path_bytes=initial_path_bytes,
    )
    verification_budget = _InventoryBudget(
        maximum_files=maximum_files,
        accumulated_path_bytes=initial_path_bytes,
    )
    queue: deque[tuple[str, ...]] = deque([()])

    while queue:
        components = queue.popleft()
        opened = _open_directory_path(
            trusted_root,
            components,
            expected_directories=expected_directories,
            admitted_directories=directories,
        )
        try:
            _scan_open_directory(
                opened,
                components=components,
                trusted_root=trusted_root,
                matcher=matcher,
                hash_files=hash_files,
                expected_directories=expected_directories,
                expected_files=expected_files,
                expected_exclusions=expected_exclusions,
                directories=directories,
                files=files,
                exclusions=exclusions,
                records=records,
                queue=queue,
                budget=budget,
                verification_budget=verification_budget,
            )
        finally:
            opened.close()

    _require_logically_nonempty(directories, files)
    ending_root = trusted_root._require_live()
    if _fingerprint(ending_root) != root_fingerprint:
        raise InventoryMutationError("trusted root metadata changed during scan")

    return _TreeSnapshot(
        root=root_fingerprint,
        directories=tuple(
            sorted(
                directories.items(),
                key=lambda item: portable_path_sort_key(item[0]),
            )
        ),
        files=tuple(
            sorted(
                files.items(),
                key=lambda item: portable_path_sort_key(item[0]),
            )
        ),
        exclusions=tuple(
            sorted(
                exclusions.items(),
                key=lambda item: portable_path_sort_key(item[0]),
            )
        ),
        records=tuple(
            sorted(records, key=lambda item: portable_path_sort_key(item.path))
        ),
    )


def _open_directory_path(
    trusted_root: TrustedRoot,
    components: tuple[str, ...],
    *,
    expected_directories: dict[PortableRelativePath, _StatFingerprint],
    admitted_directories: dict[PortableRelativePath, _StatFingerprint],
) -> _OpenedDirectory:
    current = _open_fresh_trusted_root_directory(trusted_root)
    parent: int | None = None
    leaf: str | None = None
    try:
        current_stat = _fstat_descriptor(current)
        if (
            not stat.S_ISDIR(current_stat.st_mode)
            or current_stat.st_dev != trusted_root.device
            or current_stat.st_ino != trusted_root.inode
        ):
            raise TrustedRootError("fresh root descriptor identity changed")

        prefix: list[str] = []
        for component in components:
            before = _stat_entry(component, current)
            if not stat.S_ISDIR(before.st_mode):
                raise InventoryMutationError(
                    "included directory became a non-directory"
                )
            if before.st_dev != trusted_root.device:
                raise InventoryEntryError(
                    "payload directory crosses the trusted-root device boundary"
                )
            child = _open_directory_at(component, current)
            try:
                opened = _fstat_descriptor(child)
                after = _stat_entry(component, current)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or _fingerprint(before) != _fingerprint(opened)
                    or _fingerprint(opened) != _fingerprint(after)
                ):
                    raise InventoryMutationError(
                        "included directory changed while it was opened"
                    )
                prefix.append(component)
                path = PortableRelativePath.parse("/".join(prefix))
                admitted = admitted_directories.get(path)
                if admitted is None or _fingerprint(opened) != admitted:
                    raise InventoryMutationError(
                        "included directory changed after admission"
                    )
                expected = expected_directories.get(path)
                if expected_directories and expected is None:
                    raise InventoryMutationError(
                        "an unexpected included directory appeared"
                    )
                if expected is not None and _fingerprint(opened) != expected:
                    raise InventoryMutationError(
                        "included directory identity or metadata changed"
                    )
            except BaseException:
                _close_descriptor(child)
                raise
            previous_parent = parent
            parent = current
            current = child
            leaf = component
            if previous_parent is not None:
                _close_descriptor(previous_parent)

        fingerprint = _fingerprint(_fstat_descriptor(current))
        return _OpenedDirectory(current, parent, leaf, fingerprint)
    except BaseException:
        if current >= 0:
            _close_descriptor(current)
        if parent is not None and parent >= 0:
            _close_descriptor(parent)
        raise


def _scan_open_directory(
    opened: _OpenedDirectory,
    *,
    components: tuple[str, ...],
    trusted_root: TrustedRoot,
    matcher: _ExclusionMatcher,
    hash_files: bool,
    expected_directories: dict[PortableRelativePath, _StatFingerprint],
    expected_files: dict[PortableRelativePath, _StatFingerprint],
    expected_exclusions: dict[PortableRelativePath, _BoundaryIdentity],
    directories: dict[PortableRelativePath, _StatFingerprint],
    files: dict[PortableRelativePath, _StatFingerprint],
    exclusions: dict[PortableRelativePath, _BoundaryIdentity],
    records: list[RegularFileRecord],
    queue: deque[tuple[str, ...]],
    budget: _InventoryBudget,
    verification_budget: _InventoryBudget,
) -> None:
    descriptor = opened.descriptor
    before = _fstat_descriptor(descriptor)
    if _fingerprint(before) != opened.fingerprint:
        raise InventoryMutationError("directory changed before enumeration")

    first_names = _validated_directory_names(
        descriptor,
        components,
        budget=budget,
    )
    for name, path in first_names:
        classification = matcher.classify(path)
        entry = _stat_entry(name, descriptor)

        if stat.S_ISLNK(entry.st_mode):
            raise InventoryEntryError("payload contains a symlink")

        if classification == "boundary":
            boundary = _verify_excluded_boundary(
                descriptor,
                name,
                path,
                entry,
                trusted_root=trusted_root,
                expected=expected_exclusions.get(path),
                expected_set=bool(expected_exclusions),
            )
            exclusions[path] = boundary
            continue

        if stat.S_ISDIR(entry.st_mode):
            fingerprint = _verify_directory_entry(
                descriptor,
                name,
                path,
                entry,
                trusted_root=trusted_root,
                expected=expected_directories.get(path),
                expected_set=bool(expected_directories),
            )
            directories[path] = fingerprint
            queue.append((*components, name))
            continue

        if stat.S_ISREG(entry.st_mode):
            budget.observe_file()
            fingerprint, record = _verify_regular_file(
                descriptor,
                name,
                path,
                entry,
                trusted_root=trusted_root,
                hash_file=hash_files,
                expected=expected_files.get(path),
                expected_set=bool(expected_files),
            )
            files[path] = fingerprint
            if record is not None:
                records.append(record)
            continue

        raise InventoryEntryError(
            "payload contains a FIFO, socket, device, or other special entry"
        )

    second_names = _validated_directory_names(
        descriptor,
        components,
        budget=verification_budget,
    )
    if tuple(path.value for _, path in first_names) != tuple(
        path.value for _, path in second_names
    ):
        raise InventoryMutationError(
            "directory membership changed during enumeration"
        )

    after = _fstat_descriptor(descriptor)
    if _fingerprint(after) != _fingerprint(before):
        raise InventoryMutationError("directory metadata changed during enumeration")

    if opened.parent_descriptor is not None and opened.leaf_name is not None:
        still_named = _stat_entry(opened.leaf_name, opened.parent_descriptor)
        if _fingerprint(still_named) != _fingerprint(after):
            raise InventoryMutationError(
                "directory no longer has the same parent entry"
            )


def _validated_directory_names(
    descriptor: int,
    components: tuple[str, ...],
    *,
    budget: _InventoryBudget,
) -> tuple[tuple[str, PortableRelativePath], ...]:
    try:
        raw_names = _list_directory_names(descriptor)
    except OSError as exc:
        raise InventoryMutationError("unable to enumerate payload directory") from exc

    accepted: list[tuple[str, PortableRelativePath]] = []
    spellings: dict[str, str] = {}
    try:
        for name in raw_names:
            if type(name) is not str:
                raise InventoryEntryError(
                    "directory entry name must be a string"
                )
            if "/" in name:
                raise InventoryEntryError(
                    "directory enumeration returned a non-component name"
                )
            logical = "/".join((*components, name))
            try:
                path = PortableRelativePath.parse(logical)
            except PortablePathError as exc:
                raise InventoryEntryError(
                    "payload contains an invalid portable path"
                ) from exc
            budget.observe(path)
            alias = name.lower()
            existing = spellings.get(alias)
            if existing is not None and existing != name:
                raise InventoryEntryError(
                    "payload contains a case-insensitive path alias"
                )
            if existing is not None:
                raise InventoryMutationError(
                    "directory enumeration returned a duplicate entry"
                )
            spellings[alias] = name
            accepted.append((name, path))
    finally:
        close = getattr(raw_names, "close", None)
        if close is not None:
            close()
    return tuple(
        sorted(accepted, key=lambda item: portable_path_sort_key(item[1]))
    )


def _verify_directory_entry(
    parent_descriptor: int,
    name: str,
    path: PortableRelativePath,
    before: os.stat_result,
    *,
    trusted_root: TrustedRoot,
    expected: _StatFingerprint | None,
    expected_set: bool,
) -> _StatFingerprint:
    if before.st_dev != trusted_root.device:
        raise InventoryEntryError(
            "payload directory crosses the trusted-root device boundary"
        )
    try:
        descriptor = _open_directory_at(name, parent_descriptor)
    except OSError as exc:
        raise InventoryMutationError(
            "payload directory changed before it could be opened"
        ) from exc
    try:
        opened = _fstat_descriptor(descriptor)
        after = _stat_entry(name, parent_descriptor)
        fingerprint = _fingerprint(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _fingerprint(before) != fingerprint
            or fingerprint != _fingerprint(after)
        ):
            raise InventoryMutationError(
                "payload directory changed while it was admitted"
            )
        if expected_set and expected is None:
            raise InventoryMutationError(
                "an unexpected payload directory appeared"
            )
        if expected is not None and fingerprint != expected:
            raise InventoryMutationError(
                "payload directory identity or metadata changed"
            )
        return fingerprint
    finally:
        _close_descriptor(descriptor)


def _verify_excluded_boundary(
    parent_descriptor: int,
    name: str,
    path: PortableRelativePath,
    before: os.stat_result,
    *,
    trusted_root: TrustedRoot,
    expected: _BoundaryIdentity | None,
    expected_set: bool,
) -> _BoundaryIdentity:
    if not stat.S_ISDIR(before.st_mode):
        raise InventoryEntryError(
            "an existing excluded boundary must be a real directory"
        )
    if before.st_dev != trusted_root.device:
        raise InventoryEntryError(
            "excluded boundary crosses the trusted-root device boundary"
        )
    try:
        descriptor = _open_directory_at(name, parent_descriptor)
    except OSError as exc:
        raise InventoryMutationError(
            "excluded boundary changed before it could be verified"
        ) from exc
    try:
        opened = _fstat_descriptor(descriptor)
        after = _stat_entry(name, parent_descriptor)
        identity = _boundary_identity(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _boundary_identity(before) != identity
            or identity != _boundary_identity(after)
        ):
            raise InventoryMutationError(
                "excluded boundary changed while it was verified"
            )
        if expected_set and expected is None:
            raise InventoryMutationError(
                "an excluded boundary appeared after the baseline pass"
            )
        if expected is not None and identity != expected:
            raise InventoryMutationError("excluded boundary identity changed")
        return identity
    finally:
        _close_descriptor(descriptor)


def _verify_regular_file(
    parent_descriptor: int,
    name: str,
    path: PortableRelativePath,
    before: os.stat_result,
    *,
    trusted_root: TrustedRoot,
    hash_file: bool,
    expected: _StatFingerprint | None,
    expected_set: bool,
) -> tuple[_StatFingerprint, RegularFileRecord | None]:
    fingerprint = _fingerprint(before)
    _require_regular_file_metadata(before, trusted_root=trusted_root)
    if expected_set and expected is None:
        raise InventoryMutationError("an unexpected payload file appeared")
    if expected is not None and fingerprint != expected:
        raise InventoryMutationError("payload file identity or metadata changed")

    if not hash_file:
        after = _stat_entry(name, parent_descriptor)
        if _fingerprint(after) != fingerprint:
            raise InventoryMutationError(
                "payload file changed while its metadata was inspected"
            )
        return fingerprint, None

    try:
        descriptor = _open_regular_file_at(name, parent_descriptor)
    except OSError as exc:
        raise InventoryMutationError(
            "payload file changed before it could be opened"
        ) from exc
    try:
        opened = _fstat_descriptor(descriptor)
        _require_regular_file_metadata(opened, trusted_root=trusted_root)
        opened_fingerprint = _fingerprint(opened)
        if opened_fingerprint != fingerprint:
            raise InventoryMutationError(
                "payload file changed while it was opened"
            )

        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            requested = min(_HASH_CHUNK_BYTES, remaining)
            chunk = _read_descriptor_chunk(descriptor, requested)
            if not chunk:
                raise InventoryMutationError(
                    "payload file ended before its declared size"
                )
            if len(chunk) > requested:
                raise InventoryMutationError(
                    "payload read exceeded the bounded hash request"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if _read_descriptor_chunk(descriptor, 1):
            raise InventoryMutationError(
                "payload file grew while it was hashed"
            )

        after_open = _fstat_descriptor(descriptor)
        after_named = _stat_entry(name, parent_descriptor)
        if (
            _fingerprint(after_open) != opened_fingerprint
            or _fingerprint(after_named) != opened_fingerprint
        ):
            raise InventoryMutationError(
                "payload file changed while it was hashed"
            )
        record = RegularFileRecord(
            path=path,
            size_bytes=opened.st_size,
            sha256=Sha256Digest(digest.hexdigest()),
            executable=bool(opened.st_mode & 0o111),
        )
        return opened_fingerprint, record
    finally:
        _close_descriptor(descriptor)


def _require_regular_file_metadata(
    result: os.stat_result,
    *,
    trusted_root: TrustedRoot,
) -> None:
    if not stat.S_ISREG(result.st_mode):
        raise InventoryEntryError("payload entry is not a regular file")
    if result.st_dev != trusted_root.device:
        raise InventoryEntryError(
            "payload file crosses the trusted-root device boundary"
        )
    if result.st_nlink != 1:
        raise InventoryEntryError("hard-linked payload files are forbidden")
    if result.st_size < 0 or result.st_size > MAX_SIGNED_64:
        raise InventoryLimitError(
            "payload file size exceeds the signed-64 inventory range"
        )


def _require_logically_nonempty(
    directories: dict[PortableRelativePath, _StatFingerprint],
    files: dict[PortableRelativePath, _StatFingerprint],
) -> None:
    if not files:
        raise InventoryEntryError(
            "included inventory root contains no regular files"
        )
    nonempty_directories: set[str] = set()
    for file_path in files:
        parts = file_path.parts
        for index in range(1, len(parts)):
            nonempty_directories.add("/".join(parts[:index]))
    for directory in directories:
        if directory.value not in nonempty_directories:
            raise InventoryEntryError(
                "included payload directory contains no included regular file"
            )


def _require_same_tree(
    expected: _TreeSnapshot,
    observed: _TreeSnapshot,
) -> None:
    if (
        expected.root != observed.root
        or expected.directories != observed.directories
        or expected.files != observed.files
        or expected.exclusions != observed.exclusions
    ):
        raise InventoryMutationError(
            "payload membership, identity, or metadata changed between passes"
        )


def _require_canonical_item_budget(
    exclusions: tuple[PortableRelativePath, ...],
    records: tuple[RegularFileRecord, ...],
) -> None:
    container_items = 4 + len(exclusions) + (5 * len(records))
    if container_items > MAX_CANONICAL_CONTAINER_ITEMS:
        raise InventoryLimitError(
            "inventory exceeds the canonical container-item limit"
        )


def _build_regular_file_inventory(
    *,
    scope: object,
    excluded_prefixes: object,
    files: object,
) -> RegularFileInventory:
    """Build one authority only after revalidating every canonical input."""

    validated_scope = _validate_scope(scope)
    if type(excluded_prefixes) is not tuple:
        raise TypeError("validated excluded prefixes must be exactly a tuple")
    validated_exclusions = _validate_exclusions(excluded_prefixes)
    if validated_exclusions != excluded_prefixes:
        raise InventoryValidationError(
            "validated excluded prefixes must already be canonically sorted"
        )
    if type(files) is not tuple:
        raise TypeError("validated inventory records must be exactly a tuple")
    for record in files:
        if type(record) is not RegularFileRecord:
            raise TypeError(
                "validated inventory records must be exact RegularFileRecord values"
            )
        try:
            RegularFileRecord(
                path=record.path,
                size_bytes=record.size_bytes,
                sha256=record.sha256,
                executable=record.executable,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise InventoryValidationError(
                "validated inventory contains a malformed file record"
            ) from exc

    typed_files = files
    try:
        ordered_paths = require_distinct_file_paths(
            tuple(record.path for record in typed_files)
        )
    except (AttributeError, PortablePathError, TypeError) as exc:
        raise InventoryValidationError(
            "validated inventory paths are not collision-free"
        ) from exc
    if ordered_paths != tuple(record.path for record in typed_files):
        raise InventoryValidationError(
            "validated inventory records must already be canonically sorted"
        )
    _require_canonical_item_budget(validated_exclusions, typed_files)

    body = {
        "schema_version": INVENTORY_SCHEMA,
        "scope": validated_scope,
        "excluded_prefixes": [
            prefix.value for prefix in validated_exclusions
        ],
        "files": [record._canonical_value() for record in typed_files],
    }
    encoded = canonical_json_bytes(body)
    raw_digest = Sha256Digest(hashlib.sha256(encoded).hexdigest())
    tree_digest = domain_separated_sha256(TREE_DIGEST_DOMAIN, encoded)

    result = object.__new__(RegularFileInventory)
    object.__setattr__(result, "schema_version", INVENTORY_SCHEMA)
    object.__setattr__(result, "scope", validated_scope)
    object.__setattr__(
        result,
        "excluded_prefixes",
        validated_exclusions,
    )
    object.__setattr__(result, "files", typed_files)
    object.__setattr__(result, "canonical_bytes", encoded)
    object.__setattr__(
        result,
        "canonical_inventory_sha256",
        raw_digest,
    )
    object.__setattr__(result, "tree_digest", tree_digest)
    _require_inventory_consistency(result)
    return result


def _require_inventory_consistency(result: RegularFileInventory) -> None:
    if type(result) is not RegularFileInventory:
        raise TypeError("inventory authority must be exact RegularFileInventory")
    body = {
        "schema_version": result.schema_version,
        "scope": result.scope,
        "excluded_prefixes": [
            prefix.value for prefix in result.excluded_prefixes
        ],
        "files": [record._canonical_value() for record in result.files],
    }
    encoded = canonical_json_bytes(body)
    raw_digest = Sha256Digest(hashlib.sha256(encoded).hexdigest())
    tree_digest = domain_separated_sha256(TREE_DIGEST_DOMAIN, encoded)
    if (
        result.schema_version != INVENTORY_SCHEMA
        or result.canonical_bytes != encoded
        or result.canonical_inventory_sha256 != raw_digest
        or result.tree_digest != tree_digest
    ):
        raise InventoryValidationError(
            "constructed inventory bytes and digests are inconsistent"
        )


def _fingerprint(result: os.stat_result) -> _StatFingerprint:
    return _StatFingerprint(
        device=result.st_dev,
        inode=result.st_ino,
        mode=result.st_mode,
        link_count=result.st_nlink,
        user_id=result.st_uid,
        group_id=result.st_gid,
        size=result.st_size,
        modified_ns=result.st_mtime_ns,
        changed_ns=result.st_ctime_ns,
    )


def _identity(result: os.stat_result) -> tuple[int, int, int]:
    return (result.st_dev, result.st_ino, stat.S_IFMT(result.st_mode))


def _boundary_identity(result: os.stat_result) -> _BoundaryIdentity:
    return _BoundaryIdentity(
        device=result.st_dev,
        inode=result.st_ino,
        mode=result.st_mode,
    )


def _directory_open_flags() -> int:
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
    )
    flags |= getattr(os, "O_NOCTTY", 0)
    return flags


def _regular_file_open_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    flags |= getattr(os, "O_NOCTTY", 0)
    return flags


def _open_absolute_root(flags: int) -> int:
    return os.open("/", flags)


def _open_directory_at(name: str, parent_descriptor: int) -> int:
    return os.open(name, _directory_open_flags(), dir_fd=parent_descriptor)


def _open_regular_file_at(name: str, parent_descriptor: int) -> int:
    return os.open(name, _regular_file_open_flags(), dir_fd=parent_descriptor)


def _stat_entry(name: str, parent_descriptor: int) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise InventoryMutationError(
            "payload entry disappeared during inventory"
        ) from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise InventoryEntryError(
                "payload contains a symlink or non-directory boundary"
            ) from exc
        raise


def _stat_root_component(
    name: str,
    parent_descriptor: int,
) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise TrustedRootError(
            "trusted root path is missing, inaccessible, or unsafe"
        ) from exc


def _fstat_descriptor(descriptor: int) -> os.stat_result:
    return os.fstat(descriptor)


def _list_directory_names(descriptor: int) -> Iterator[str]:
    with os.scandir(descriptor) as entries:
        for entry in entries:
            yield entry.name


def _read_descriptor_chunk(descriptor: int, size: int) -> bytes:
    return os.read(descriptor, size)


def _close_descriptor(descriptor: int) -> None:
    os.close(descriptor)
