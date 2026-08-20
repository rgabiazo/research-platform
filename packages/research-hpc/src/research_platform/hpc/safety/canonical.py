"""Strict canonical JSON and domain-separated digest foundations for HPC safety.

This module deliberately implements only the H2a encoding foundation.  Tree
inventories and receipt envelopes are defined by later H2 subgates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import struct
import unicodedata


CANONICAL_JSON_VERSION = "research_platform.hpc.canonical_json.v1"
CANONICAL_UNICODE_VERSION = "3.2.0"
TREE_DIGEST_ALGORITHM = "research_platform.hpc.sha256_tree.v1"
RECEIPT_DIGEST_ALGORITHM = (
    "research_platform.hpc.sha256_receipt_envelope.v1"
)

TREE_DIGEST_DOMAIN = b"research-platform:hpc:regular-file-tree:v1\0"
RECEIPT_DIGEST_DOMAIN = b"research-platform:hpc:receipt-envelope:v1\0"

MAX_CANONICAL_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_CANONICAL_DEPTH = 64
MAX_CANONICAL_CONTAINER_ITEMS = 100_000
MAX_CANONICAL_STRING_BYTES = 1024 * 1024
MIN_SIGNED_64 = -(2**63)
MAX_SIGNED_64 = 2**63 - 1

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_UNICODE_DATABASE = unicodedata.ucd_3_2_0
_JSON_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_JSON_WHITESPACE = frozenset(" \t\r\n")

if _UNICODE_DATABASE.unidata_version != CANONICAL_UNICODE_VERSION:
    raise RuntimeError("Python's frozen Unicode 3.2 database version changed")


class CanonicalJsonError(ValueError):
    """Base class for rejected canonical JSON values or byte documents."""


class CanonicalJsonTypeError(CanonicalJsonError):
    """Raised when a value is outside the canonical JSON type system."""


class CanonicalJsonLimitError(CanonicalJsonError):
    """Raised when a canonical JSON resource limit is exceeded."""


class CanonicalJsonDecodeError(CanonicalJsonError):
    """Raised when input bytes are invalid or are not canonical JSON bytes."""


@dataclass(frozen=True)
class CanonicalJsonLimits:
    """Resource limits that affect acceptance, never accepted canonical bytes."""

    maximum_document_bytes: int = MAX_CANONICAL_DOCUMENT_BYTES
    maximum_depth: int = MAX_CANONICAL_DEPTH
    maximum_container_items: int = MAX_CANONICAL_CONTAINER_ITEMS
    maximum_string_bytes: int = MAX_CANONICAL_STRING_BYTES

    def __post_init__(self) -> None:
        values = {
            "maximum_document_bytes": (
                self.maximum_document_bytes,
                MAX_CANONICAL_DOCUMENT_BYTES,
            ),
            "maximum_depth": (
                self.maximum_depth,
                MAX_CANONICAL_DEPTH,
            ),
            "maximum_container_items": (
                self.maximum_container_items,
                MAX_CANONICAL_CONTAINER_ITEMS,
            ),
            "maximum_string_bytes": (
                self.maximum_string_bytes,
                MAX_CANONICAL_STRING_BYTES,
            ),
        }
        for name, (value, protocol_maximum) in values.items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            minimum = 1 if name == "maximum_document_bytes" else 0
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            if value > protocol_maximum:
                raise ValueError(
                    f"{name} must not exceed the protocol maximum "
                    f"{protocol_maximum}"
                )


DEFAULT_CANONICAL_JSON_LIMITS = CanonicalJsonLimits()


@dataclass(frozen=True, order=True)
class Sha256Digest:
    """A validated lowercase hexadecimal SHA-256 digest."""

    value: str

    def __post_init__(self) -> None:
        if type(self.value) is not str or _SHA256_PATTERN.fullmatch(
            self.value
        ) is None:
            raise ValueError(
                "SHA-256 digest must be exactly 64 lowercase hexadecimal characters"
            )

    def __str__(self) -> str:
        return self.value


@dataclass
class _ValidationState:
    container_items: int = 0


def _require_limits(limits: CanonicalJsonLimits) -> None:
    if type(limits) is not CanonicalJsonLimits:
        raise TypeError("limits must be exactly CanonicalJsonLimits")


def _canonicalize_value(
    value: object,
    *,
    limits: CanonicalJsonLimits,
    state: _ValidationState,
    container_depth: int,
    active_containers: set[int],
) -> object:
    if value is None or type(value) is bool:
        return value

    if type(value) is int:
        if value < MIN_SIGNED_64 or value > MAX_SIGNED_64:
            raise CanonicalJsonTypeError(
                "integers must be within the signed 64-bit range"
            )
        return value

    if type(value) is str:
        _validate_string(value, limits=limits)
        return value

    if type(value) is list:
        return _canonicalize_list(
            value,
            limits=limits,
            state=state,
            container_depth=container_depth,
            active_containers=active_containers,
        )

    if type(value) is dict:
        return _canonicalize_object(
            value,
            limits=limits,
            state=state,
            container_depth=container_depth,
            active_containers=active_containers,
        )

    raise CanonicalJsonTypeError(
        "canonical JSON accepts only null, booleans, signed 64-bit integers, "
        "strings, lists, and dictionaries"
    )


def _validate_string(value: str, *, limits: CanonicalJsonLimits) -> None:
    encoded_bytes = 0
    for character in value:
        try:
            encoded_bytes += len(character.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise CanonicalJsonTypeError(
                "canonical JSON strings must contain only Unicode scalar values"
            ) from exc
        if encoded_bytes > limits.maximum_string_bytes:
            raise CanonicalJsonLimitError(
                "canonical JSON string exceeds maximum_string_bytes"
            )
        if _UNICODE_DATABASE.category(character) == "Cn":
            raise CanonicalJsonTypeError(
                "canonical JSON strings may contain only characters assigned in "
                f"Unicode {CANONICAL_UNICODE_VERSION}"
            )
    if _UNICODE_DATABASE.normalize("NFC", value) != value:
        raise CanonicalJsonTypeError(
            "canonical JSON strings must already use NFC normalization under "
            f"Unicode {CANONICAL_UNICODE_VERSION}"
        )


def _enter_container(
    value: list[object] | dict[str, object],
    *,
    limits: CanonicalJsonLimits,
    state: _ValidationState,
    container_depth: int,
    active_containers: set[int],
) -> int:
    next_depth = container_depth + 1
    if next_depth > limits.maximum_depth:
        raise CanonicalJsonLimitError("canonical JSON exceeds maximum_depth")

    state.container_items += len(value)
    if state.container_items > limits.maximum_container_items:
        raise CanonicalJsonLimitError(
            "canonical JSON exceeds maximum_container_items"
        )

    identity = id(value)
    if identity in active_containers:
        raise CanonicalJsonTypeError(
            "canonical JSON containers must not contain cycles"
        )
    active_containers.add(identity)
    return next_depth


def _canonicalize_list(
    value: list[object],
    *,
    limits: CanonicalJsonLimits,
    state: _ValidationState,
    container_depth: int,
    active_containers: set[int],
) -> list[object]:
    next_depth = _enter_container(
        value,
        limits=limits,
        state=state,
        container_depth=container_depth,
        active_containers=active_containers,
    )
    try:
        return [
            _canonicalize_value(
                item,
                limits=limits,
                state=state,
                container_depth=next_depth,
                active_containers=active_containers,
            )
            for item in value
        ]
    finally:
        active_containers.remove(id(value))


def _canonicalize_object(
    value: dict[str, object],
    *,
    limits: CanonicalJsonLimits,
    state: _ValidationState,
    container_depth: int,
    active_containers: set[int],
) -> dict[str, object]:
    next_depth = _enter_container(
        value,
        limits=limits,
        state=state,
        container_depth=container_depth,
        active_containers=active_containers,
    )
    try:
        ordered: dict[str, object] = {}
        for key in value:
            if type(key) is not str:
                raise CanonicalJsonTypeError(
                    "canonical JSON object keys must be strings"
                )
            _validate_string(key, limits=limits)
        for key in sorted(value, key=lambda item: item.encode("utf-8")):
            ordered[key] = _canonicalize_value(
                value[key],
                limits=limits,
                state=state,
                container_depth=next_depth,
                active_containers=active_containers,
            )
        return ordered
    finally:
        active_containers.remove(id(value))


def canonical_json_bytes(
    value: object,
    *,
    limits: CanonicalJsonLimits = DEFAULT_CANONICAL_JSON_LIMITS,
) -> bytes:
    """Serialize one accepted value to deterministic canonical UTF-8 JSON."""

    _require_limits(limits)
    canonical_value = _canonicalize_value(
        value,
        limits=limits,
        state=_ValidationState(),
        container_depth=0,
        active_containers=set(),
    )
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    encoded_parts: list[bytes] = []
    encoded_size = 0
    try:
        for text_part in encoder.iterencode(canonical_value):
            byte_part = text_part.encode("utf-8")
            encoded_size += len(byte_part)
            if encoded_size > limits.maximum_document_bytes:
                raise CanonicalJsonLimitError(
                    "canonical JSON exceeds maximum_document_bytes"
                )
            encoded_parts.append(byte_part)
    except CanonicalJsonLimitError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CanonicalJsonTypeError(
            "value cannot be represented as canonical JSON"
        ) from exc

    return b"".join(encoded_parts)


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJsonDecodeError(
                f"duplicate canonical JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _reject_json_float(value: str) -> object:
    raise CanonicalJsonTypeError(
        f"canonical JSON does not accept floating-point number {value!r}"
    )


def _reject_json_constant(value: str) -> object:
    raise CanonicalJsonTypeError(
        f"canonical JSON does not accept non-finite value {value!r}"
    )


def _parse_json_integer(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > 19:
        raise CanonicalJsonTypeError(
            "canonical JSON integers must be within the signed 64-bit range"
        )
    return int(value)


@dataclass
class _LexicalFrame:
    kind: str
    state: str


@dataclass
class _LexicalPreflightState:
    container_items: int = 0
    deferred_error: CanonicalJsonError | None = None

    def defer(self, error: CanonicalJsonError) -> None:
        if self.deferred_error is None:
            self.deferred_error = error


def _scan_json_string(
    text: str,
    start: int,
    *,
    limits: CanonicalJsonLimits,
    state: _LexicalPreflightState,
) -> int:
    index = start + 1
    decoded_bytes = 0

    while index < len(text):
        character = text[index]
        if character == '"':
            return index + 1
        if ord(character) < 0x20:
            raise CanonicalJsonDecodeError(
                "JSON strings must escape control characters"
            )
        if character != "\\":
            decoded_bytes += len(character.encode("utf-8"))
            index += 1
        else:
            index += 1
            if index >= len(text):
                raise CanonicalJsonDecodeError("unterminated JSON escape")
            escape = text[index]
            if escape in '"\\/bfnrt':
                decoded_bytes += 1
                index += 1
            elif escape == "u":
                codepoint, index = _scan_json_unicode_escape(text, index)
                if 0xD800 <= codepoint <= 0xDBFF:
                    if text[index : index + 2] == "\\u":
                        low, index = _scan_json_unicode_escape(text, index + 1)
                    else:
                        low = None
                    if low is None or not 0xDC00 <= low <= 0xDFFF:
                        state.defer(
                            CanonicalJsonTypeError(
                                "high-surrogate JSON escapes require a low "
                                "surrogate"
                            )
                        )
                        decoded_bytes += 3
                        if low is not None:
                            if 0xD800 <= low <= 0xDFFF:
                                decoded_bytes += 3
                            else:
                                decoded_bytes += len(chr(low).encode("utf-8"))
                    else:
                        codepoint = (
                            0x10000
                            + ((codepoint - 0xD800) << 10)
                            + (low - 0xDC00)
                        )
                        decoded_bytes += len(chr(codepoint).encode("utf-8"))
                elif 0xDC00 <= codepoint <= 0xDFFF:
                    state.defer(
                        CanonicalJsonTypeError(
                            "low-surrogate JSON escape has no high surrogate"
                        )
                    )
                    decoded_bytes += 3
                else:
                    decoded_bytes += len(chr(codepoint).encode("utf-8"))
            else:
                raise CanonicalJsonDecodeError(
                    f"unsupported JSON escape: \\{escape}"
                )
        if decoded_bytes > limits.maximum_string_bytes:
            state.defer(
                CanonicalJsonLimitError(
                    "canonical JSON string exceeds maximum_string_bytes"
                )
            )

    raise CanonicalJsonDecodeError("unterminated JSON string")


def _scan_json_unicode_escape(text: str, u_index: int) -> tuple[int, int]:
    if text[u_index] != "u":
        raise CanonicalJsonDecodeError("invalid JSON Unicode escape")
    digits_start = u_index + 1
    digits_end = digits_start + 4
    if digits_end > len(text):
        raise CanonicalJsonDecodeError("truncated JSON Unicode escape")
    digits = text[digits_start:digits_end]
    if any(character not in _JSON_HEX_DIGITS for character in digits):
        raise CanonicalJsonDecodeError("invalid JSON Unicode escape")
    return int(digits, 16), digits_end


def _scan_json_integer_token(
    text: str,
    start: int,
    *,
    state: _LexicalPreflightState,
) -> int:
    if text.startswith("-Infinity", start):
        raise CanonicalJsonTypeError(
            "canonical JSON does not accept non-finite values"
        )

    index = start
    negative = text[index] == "-"
    if negative:
        index += 1
        if index >= len(text) or text[index] not in "0123456789":
            raise CanonicalJsonDecodeError("invalid JSON integer")

    digits_start = index
    if text[index] == "0":
        index += 1
        if index < len(text) and text[index] in "0123456789":
            raise CanonicalJsonDecodeError(
                "JSON integers must not contain leading zeroes"
            )
    else:
        if text[index] not in "123456789":
            raise CanonicalJsonDecodeError("invalid JSON integer")
        while index < len(text) and text[index] in "0123456789":
            index += 1

    integer_digits_end = index
    is_float = False
    if index < len(text) and text[index] == ".":
        is_float = True
        index += 1
        if index >= len(text) or text[index] not in "0123456789":
            raise CanonicalJsonDecodeError(
                "JSON fractions require at least one digit"
            )
        while index < len(text) and text[index] in "0123456789":
            index += 1
    if index < len(text) and text[index] in "eE":
        is_float = True
        index += 1
        if index < len(text) and text[index] in "+-":
            index += 1
        if index >= len(text) or text[index] not in "0123456789":
            raise CanonicalJsonDecodeError(
                "JSON exponents require at least one digit"
            )
        while index < len(text) and text[index] in "0123456789":
            index += 1
    if is_float:
        raise CanonicalJsonTypeError(
            "canonical JSON does not accept floating-point numbers"
        )

    maximum = "9223372036854775808" if negative else "9223372036854775807"
    digit_count = integer_digits_end - digits_start
    if digit_count > len(maximum) or (
        digit_count == len(maximum)
        and text[digits_start:integer_digits_end] > maximum
    ):
        state.defer(
            CanonicalJsonTypeError(
                "canonical JSON integers must be within the signed 64-bit range"
            )
        )
    return index


def _scan_json_value(
    text: str,
    index: int,
    *,
    limits: CanonicalJsonLimits,
    stack: list[_LexicalFrame],
    state: _LexicalPreflightState,
) -> int:
    if index >= len(text):
        raise CanonicalJsonDecodeError("missing JSON value")
    character = text[index]

    if character in "[{":
        next_depth = len(stack) + 1
        if next_depth > limits.maximum_depth:
            raise CanonicalJsonLimitError(
                "canonical JSON exceeds maximum_depth"
            )
        if character == "[":
            stack.append(_LexicalFrame("array", "value_or_end"))
        else:
            stack.append(_LexicalFrame("object", "key_or_end"))
        return index + 1
    if character == '"':
        return _scan_json_string(text, index, limits=limits, state=state)
    if text.startswith("null", index):
        return index + 4
    if text.startswith("true", index):
        return index + 4
    if text.startswith("false", index):
        return index + 5
    if text.startswith(("NaN", "Infinity"), index):
        raise CanonicalJsonTypeError(
            "canonical JSON does not accept non-finite values"
        )
    if character == "-" or character in "0123456789":
        return _scan_json_integer_token(text, index, state=state)
    raise CanonicalJsonDecodeError("invalid JSON value")


def _preflight_canonical_json_text(
    text: str,
    *,
    limits: CanonicalJsonLimits,
) -> None:
    """Bound lexical resources without recursive parsing or value allocation."""

    stack: list[_LexicalFrame] = []
    root_state = "value"
    state = _LexicalPreflightState()
    index = 0

    def increment_items() -> None:
        state.container_items += 1
        if state.container_items > limits.maximum_container_items:
            state.defer(
                CanonicalJsonLimitError(
                    "canonical JSON exceeds maximum_container_items"
                )
            )

    while True:
        while index < len(text) and text[index] in _JSON_WHITESPACE:
            index += 1

        if not stack:
            if root_state == "done":
                if index == len(text):
                    if state.deferred_error is not None:
                        raise state.deferred_error
                    return
                raise CanonicalJsonDecodeError(
                    "canonical JSON contains data after the root value"
                )
            if index == len(text):
                raise CanonicalJsonDecodeError(
                    "canonical JSON is missing a root value"
                )
            root_state = "done"
            index = _scan_json_value(
                text,
                index,
                limits=limits,
                stack=stack,
                state=state,
            )
            continue

        frame = stack[-1]
        if index == len(text):
            raise CanonicalJsonDecodeError("unterminated JSON container")

        if frame.kind == "array":
            if frame.state == "value_or_end" and text[index] == "]":
                stack.pop()
                index += 1
                continue
            if frame.state in {"value_or_end", "value"}:
                increment_items()
                frame.state = "comma_or_end"
                index = _scan_json_value(
                    text,
                    index,
                    limits=limits,
                    stack=stack,
                    state=state,
                )
                continue
            if text[index] == ",":
                frame.state = "value"
                index += 1
                continue
            if text[index] == "]":
                stack.pop()
                index += 1
                continue
            raise CanonicalJsonDecodeError(
                "JSON array requires a comma or closing bracket"
            )

        if frame.state == "key_or_end" and text[index] == "}":
            stack.pop()
            index += 1
            continue
        if frame.state in {"key_or_end", "key"}:
            if text[index] != '"':
                raise CanonicalJsonDecodeError(
                    "JSON object keys must be strings"
                )
            index = _scan_json_string(
                text,
                index,
                limits=limits,
                state=state,
            )
            increment_items()
            frame.state = "colon"
            continue
        if frame.state == "colon":
            if text[index] != ":":
                raise CanonicalJsonDecodeError(
                    "JSON object keys must be followed by a colon"
                )
            frame.state = "value"
            index += 1
            continue
        if frame.state == "value":
            frame.state = "comma_or_end"
            index = _scan_json_value(
                text,
                index,
                limits=limits,
                stack=stack,
                state=state,
            )
            continue
        if text[index] == ",":
            frame.state = "key"
            index += 1
            continue
        if text[index] == "}":
            stack.pop()
            index += 1
            continue
        raise CanonicalJsonDecodeError(
            "JSON object requires a comma or closing brace"
        )


def parse_canonical_json_bytes(
    data: bytes,
    *,
    limits: CanonicalJsonLimits = DEFAULT_CANONICAL_JSON_LIMITS,
) -> object:
    """Parse canonical JSON bytes and reject any noncanonical representation."""

    _require_limits(limits)
    if type(data) is not bytes:
        raise CanonicalJsonTypeError("canonical JSON input must be bytes")
    if len(data) > limits.maximum_document_bytes:
        raise CanonicalJsonLimitError(
            "canonical JSON exceeds maximum_document_bytes"
        )
    if data.startswith(b"\xef\xbb\xbf"):
        raise CanonicalJsonDecodeError("canonical JSON must not contain a BOM")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalJsonDecodeError(
            "canonical JSON must be valid UTF-8"
        ) from exc

    _preflight_canonical_json_text(text, limits=limits)

    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_int=_parse_json_integer,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except CanonicalJsonError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CanonicalJsonDecodeError("invalid canonical JSON document") from exc

    canonical = canonical_json_bytes(value, limits=limits)
    if canonical != data:
        raise CanonicalJsonDecodeError(
            "JSON input is valid but does not use canonical byte encoding"
        )
    return value


def domain_separated_sha256(domain: bytes, body: bytes) -> Sha256Digest:
    """Hash ``domain || uint64be(len(body)) || body`` with SHA-256."""

    if type(domain) is not bytes or not domain:
        raise TypeError("domain must be nonempty bytes")
    if type(body) is not bytes:
        raise TypeError("body must be bytes")
    if len(body) > 2**64 - 1:
        raise ValueError("body is too large for a uint64 length prefix")

    digest = hashlib.sha256(
        domain + struct.pack(">Q", len(body)) + body
    ).hexdigest()
    return Sha256Digest(digest)


__all__ = [
    "CANONICAL_JSON_VERSION",
    "CANONICAL_UNICODE_VERSION",
    "DEFAULT_CANONICAL_JSON_LIMITS",
    "MAX_CANONICAL_CONTAINER_ITEMS",
    "MAX_CANONICAL_DEPTH",
    "MAX_CANONICAL_DOCUMENT_BYTES",
    "MAX_CANONICAL_STRING_BYTES",
    "MAX_SIGNED_64",
    "MIN_SIGNED_64",
    "RECEIPT_DIGEST_ALGORITHM",
    "RECEIPT_DIGEST_DOMAIN",
    "TREE_DIGEST_ALGORITHM",
    "TREE_DIGEST_DOMAIN",
    "CanonicalJsonDecodeError",
    "CanonicalJsonError",
    "CanonicalJsonLimitError",
    "CanonicalJsonLimits",
    "CanonicalJsonTypeError",
    "Sha256Digest",
    "canonical_json_bytes",
    "domain_separated_sha256",
    "parse_canonical_json_bytes",
]
