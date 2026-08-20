"""Lexical portable paths for managed HPC payloads.

This module deliberately validates protocol paths without consulting the host
filesystem. Descriptor-anchored traversal and file-type validation belong to
the later H2b gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PORTABLE_PATH_SCHEMA = "research_platform.hpc.portable_relative_path.v1"
MAX_PORTABLE_PATH_BYTES = 4095
MAX_PORTABLE_COMPONENT_BYTES = 255

_ALLOWED_ASCII = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
)


class PortablePathError(ValueError):
    """Base error for an invalid managed-payload path contract."""


class PortablePathCollisionError(PortablePathError):
    """Raised when a path set contains an alias or file/tree collision."""


@dataclass(frozen=True, order=True)
class PortableRelativePath:
    """One validated ASCII POSIX-relative managed-payload path."""

    value: str

    def __post_init__(self) -> None:
        _validate_portable_relative_path(self.value)

    @classmethod
    def parse(cls, value: object) -> "PortableRelativePath":
        """Validate *value* without normalizing or changing its spelling."""

        if type(value) is not str:
            raise PortablePathError("portable relative paths must be strings")
        return cls(value)

    @property
    def parts(self) -> tuple[str, ...]:
        """Return the already validated POSIX path components."""

        return tuple(self.value.split("/"))


def _validate_portable_relative_path(value: object) -> None:
    if type(value) is not str:
        raise PortablePathError("portable relative paths must be strings")
    if not value:
        raise PortablePathError("portable relative paths must not be empty")
    if value.startswith("/"):
        raise PortablePathError("portable relative paths must not be absolute")
    if "\\" in value:
        raise PortablePathError(
            "portable relative paths must not contain backslashes"
        )
    if len(value) > MAX_PORTABLE_PATH_BYTES:
        raise PortablePathError(
            f"portable relative paths must be at most {MAX_PORTABLE_PATH_BYTES} bytes"
        )

    parts = value.split("/")
    if any(not part for part in parts):
        raise PortablePathError(
            "portable relative paths must not contain empty components"
        )
    for part in parts:
        if part in {".", ".."}:
            raise PortablePathError(
                "portable relative paths must not contain '.' or '..' components"
            )
        if len(part) > MAX_PORTABLE_COMPONENT_BYTES:
            raise PortablePathError(
                "portable relative path components must be at most "
                f"{MAX_PORTABLE_COMPONENT_BYTES} bytes"
            )
        if any(character not in _ALLOWED_ASCII for character in part):
            raise PortablePathError(
                "portable relative path components may contain only "
                "ASCII letters, digits, '.', '_', and '-'"
            )


def portable_path_sort_key(path: PortableRelativePath) -> bytes:
    """Return the frozen bytewise ordering key for a validated path."""

    if type(path) is not PortableRelativePath:
        raise TypeError("portable path sort keys require PortableRelativePath")
    return path.value.encode("ascii")


@dataclass
class _PathTrieNode:
    spelling: str | None = None
    is_file: bool = False
    children: dict[str, "_PathTrieNode"] | None = None

    def child_map(self) -> dict[str, "_PathTrieNode"]:
        if self.children is None:
            self.children = {}
        return self.children


def require_distinct_file_paths(
    paths: Iterable[PortableRelativePath],
) -> tuple[PortableRelativePath, ...]:
    """Validate one collision-free file inventory and return bytewise order.

    Case-alias checks apply to every implicit directory prefix. A declared file
    may not also be an ancestor of another declared file.
    """

    accepted: list[PortableRelativePath] = []
    root = _PathTrieNode()

    for path in paths:
        if type(path) is not PortableRelativePath:
            raise TypeError("file path sets require PortableRelativePath values")
        accepted.append(path)
        node = root
        for index, component in enumerate(path.parts):
            if node.is_file:
                raise PortablePathCollisionError(
                    "a declared file path is an ancestor of another file path"
                )
            alias_key = component.lower()
            children = node.child_map()
            child = children.get(alias_key)
            if child is None:
                child = _PathTrieNode(spelling=component)
                children[alias_key] = child
            elif child.spelling != component:
                raise PortablePathCollisionError(
                    "portable file paths contain a case-insensitive alias"
                )
            node = child
            if index == len(path.parts) - 1:
                if node.is_file:
                    raise PortablePathCollisionError(
                        "portable file paths contain an exact duplicate"
                    )
                if node.children:
                    raise PortablePathCollisionError(
                        "a declared file path is an ancestor of another file path"
                    )
                node.is_file = True

    return tuple(sorted(accepted, key=portable_path_sort_key))
