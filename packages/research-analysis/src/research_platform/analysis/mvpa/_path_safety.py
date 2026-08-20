"""Host-independent lexical path checks for MVPA publication outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


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
    re.compile(r"(?:^|[=\s\"'(\[,;:])/(?!/)(?=[^/\s])"),
)


def configured_path_is_unsafe(value: str) -> bool:
    """Return whether a configured path is not safely relative to its root.

    The check is lexical: it does not resolve the path, inspect the filesystem,
    expand a user or environment reference, or use host-specific path rules.
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
    """Return whether publishable text embeds a local absolute or user path."""

    candidate = str(text)
    if _FILE_URI_PATTERN.search(candidate):
        return True
    candidate = _NON_FILE_URI_PATTERN.sub(_mask_non_file_uri, candidate)
    return any(pattern.search(candidate) for pattern in _EMBEDDED_LOCAL_PATH_PATTERNS)


def published_value_contains_local_path_reference(value: Any) -> bool:
    """Recursively inspect publishable values, including table keys and cells."""

    if isinstance(value, str):
        return published_text_contains_local_path_reference(value)
    if isinstance(value, Mapping):
        return any(
            published_value_contains_local_path_reference(key)
            or published_value_contains_local_path_reference(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(published_value_contains_local_path_reference(child) for child in value)
    return False


def _mask_non_file_uri(match: re.Match[str]) -> str:
    if match.group("scheme").casefold() == "file":
        return match.group(0)
    return " " * len(match.group(0))
