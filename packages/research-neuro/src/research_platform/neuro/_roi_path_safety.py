"""Host-independent lexical path safety for published ROI derivatives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePath, PurePosixPath, PureWindowsPath
import re
from typing import Any
from urllib.parse import unquote, urlsplit


_DRIVE_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
_NON_FILE_URI_PATTERN = re.compile(
    r"\b(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*):/{2}[^\s\"'<>;,]*",
    flags=re.IGNORECASE,
)
_FILE_URI_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+.-])file:(?:/{1,3}|\\+)",
    flags=re.IGNORECASE,
)
_EMBEDDED_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_$}.])[A-Za-z]:(?:\\+|/)"),
    re.compile(r"(?<![A-Za-z0-9_$}.])(?:\\{2,}|/{2,})(?=[^\\/\s])"),
    re.compile(r"(?<![A-Za-z0-9_$}.])~(?:\\+|/)"),
    re.compile(r"(?<![A-Za-z0-9_$}.])/(?!/)(?=[^/\s])"),
    re.compile(r"(?<![A-Za-z0-9_$}.])\\(?!\\)(?=[^\\\s])"),
)
_SAFE_ROOT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class UnmappedLocalPathError(ValueError):
    """Raised when publishable text contains no truthfully portable path."""


def configured_path_is_unsafe(value: str) -> bool:
    """Return whether a configured path is not safely relative to its root.

    This is a lexical check. It does not expand user or environment references,
    resolve the path, or inspect the filesystem.
    """

    text = str(value).strip()
    if not text:
        return False
    if text == "~" or text.startswith(("/", "\\", "~/", "~\\")):
        return True
    if _DRIVE_ABSOLUTE_PATTERN.match(text) or text.casefold().startswith("file:"):
        return True
    return any(component == ".." for component in re.split(r"[\\/]", text))


def published_text_contains_local_path_reference(text: str) -> bool:
    """Return whether text contains a local absolute or user-root reference."""

    candidate = str(text)
    if _FILE_URI_PATTERN.search(candidate):
        return True
    candidate = _NON_FILE_URI_PATTERN.sub(_mask_non_file_uri, candidate)
    return any(pattern.search(candidate) for pattern in _EMBEDDED_LOCAL_PATH_PATTERNS)


def published_value_local_path_fields(value: Any, *, label: str = "value") -> tuple[str, ...]:
    """Return locations containing local paths without returning unsafe values."""

    fields: list[str] = []
    _collect_local_path_fields(value, label=label, fields=fields)
    return tuple(fields)


def portable_path_reference(
    value: str | PurePath,
    *,
    dataset_root: str | PurePath | None = None,
    named_roots: Mapping[str, str | PurePath] | None = None,
) -> str:
    """Return a truthful portable reference for one complete path value.

    Safe relative paths, named-root references, environment-root references,
    and non-file URIs are preserved. A complete local absolute path is first
    made relative to ``dataset_root`` when possible, then mapped beneath the
    deepest matching named root. Embedded and unmapped local references fail
    without including the unsafe value in the exception message.
    """

    if not isinstance(value, (str, PurePath)):
        raise TypeError("ROI publication path values must be strings or pure paths.")

    original = str(value)
    text = original.strip()
    local = _complete_local_path(text)
    if local is None:
        if _is_complete_non_file_uri(text):
            return original
        if configured_path_is_unsafe(text) or published_text_contains_local_path_reference(text):
            raise UnmappedLocalPathError(
                "Embedded local path references cannot be converted for ROI publication."
            )
        return original
    if _has_parent_component(local):
        raise UnmappedLocalPathError(
            "Parent traversal cannot be converted for ROI publication."
        )

    dataset = _complete_local_path(str(dataset_root).strip()) if dataset_root is not None else None
    if dataset is not None:
        relative = _relative_to(local, dataset)
        if relative is not None and _useful_root(dataset):
            return relative

    candidates: list[tuple[int, str, str]] = []
    for raw_name, raw_root in (named_roots or {}).items():
        name = str(raw_name)
        if not _SAFE_ROOT_NAME_PATTERN.fullmatch(name):
            continue
        root = _complete_local_path(str(raw_root).strip())
        if root is None or not _useful_root(root):
            continue
        relative = _relative_to(local, root)
        if relative is None:
            continue
        candidates.append((_root_depth(root), name, relative))

    if candidates:
        _depth, name, relative = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        return f"root_ref:{name}" if relative == "." else f"root_ref:{name}/{relative}"

    raise UnmappedLocalPathError(
        "Local path reference cannot be published because no configured portable root contains it."
    )


def _collect_local_path_fields(value: Any, *, label: str, fields: list[str]) -> None:
    if isinstance(value, Mapping):
        for index, (key, child) in enumerate(value.items()):
            key_is_unsafe = isinstance(key, (str, PurePath)) and published_text_contains_local_path_reference(str(key))
            if key_is_unsafe:
                fields.append(f"{label}.<mapping-key:{index}>")
            child_label = _mapping_child_label(label, key, index=index, key_is_unsafe=key_is_unsafe)
            _collect_local_path_fields(child, label=child_label, fields=fields)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _collect_local_path_fields(child, label=f"{label}[{index}]", fields=fields)
        return
    if isinstance(value, (str, PurePath)) and published_text_contains_local_path_reference(str(value)):
        fields.append(label)


def _mapping_child_label(label: str, key: Any, *, index: int, key_is_unsafe: bool) -> str:
    if not key_is_unsafe and isinstance(key, str) and _SAFE_FIELD_NAME_PATTERN.fullmatch(key):
        return f"{label}.{key}"
    return f"{label}.<mapping-value:{index}>"


def _mask_non_file_uri(match: re.Match[str]) -> str:
    if match.group("scheme").casefold() == "file":
        return match.group(0)
    return " " * len(match.group(0))


def _is_complete_non_file_uri(value: str) -> bool:
    match = _NON_FILE_URI_PATTERN.fullmatch(value)
    return match is not None and match.group("scheme").casefold() != "file"


def _complete_local_path(value: str) -> PurePosixPath | PureWindowsPath | None:
    if not value:
        return None
    text = _file_uri_path(value) if value.casefold().startswith("file:") else value
    if text is None or text == "~" or text.startswith(("~/", "~\\")):
        return None
    if _looks_like_windows_absolute(text):
        return PureWindowsPath(text)
    if text.startswith("/"):
        return PurePosixPath(text)
    if text.startswith("\\"):
        return PureWindowsPath(text)
    return None


def _file_uri_path(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.casefold() != "file":
        return None
    path = unquote(parsed.path)
    if parsed.netloc and parsed.netloc.casefold() != "localhost":
        return f"//{parsed.netloc}{path}"
    if re.match(r"^/[A-Za-z]:[\\/]", path):
        return path[1:]
    return path or None


def _looks_like_windows_absolute(value: str) -> bool:
    return bool(
        _DRIVE_ABSOLUTE_PATTERN.match(value)
        or value.startswith(("\\\\", "//"))
        or value.casefold().startswith(("\\\\?\\", "//?/"))
    )


def _relative_to(
    path: PurePosixPath | PureWindowsPath,
    root: PurePosixPath | PureWindowsPath,
) -> str | None:
    if type(path) is not type(root) or _has_parent_component(path) or _has_parent_component(root):
        return None
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    rendered = relative.as_posix()
    return "." if rendered in {"", "."} else rendered


def _has_parent_component(path: PurePosixPath | PureWindowsPath) -> bool:
    return any(part == ".." for part in path.parts)


def _useful_root(root: PurePosixPath | PureWindowsPath) -> bool:
    return _root_depth(root) > 0


def _root_depth(root: PurePosixPath | PureWindowsPath) -> int:
    component_depth = len(root.parts) - (1 if root.anchor else 0)
    if isinstance(root, PureWindowsPath) and root.drive.startswith("\\\\"):
        return component_depth + 2
    return component_depth


__all__ = [
    "UnmappedLocalPathError",
    "configured_path_is_unsafe",
    "portable_path_reference",
    "published_text_contains_local_path_reference",
    "published_value_local_path_fields",
]
