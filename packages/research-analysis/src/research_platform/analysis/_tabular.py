from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def infer_delimiter(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return "\t"
    if suffix == ".csv":
        return ","
    raise ValueError(f"Unsupported tabular format for {path!s}. Use .csv, .tsv, or .txt.")


def read_input_bytes(path: str | Path, *, expected_sha256: str | None = None) -> bytes:
    """Read one input once and optionally verify the bytes before parsing them."""

    input_path = Path(path)
    if expected_sha256 is None:
        with input_path.open("rb") as handle:
            return handle.read()
    if not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters.")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(input_path, flags)
    except OSError as exc:
        raise ValueError(f"Input {input_path} could not be opened as a regular nonsymlink file.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Input {input_path} is not a regular file.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Input {input_path} changed while it was read.")
        try:
            current = os.lstat(input_path)
        except OSError as exc:
            raise ValueError(f"Input {input_path} changed filesystem identity while it was read.") from exc
        if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(f"Input {input_path} changed filesystem identity while it was read.")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256.lower()):
        raise ValueError(f"SHA-256 mismatch for input {input_path}.")
    return payload


def _rows_from_bytes(payload: bytes, *, path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    table_path = Path(path)
    text = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=infer_delimiter(table_path))
    if reader.fieldnames is None:
        raise ValueError(f"Input table {table_path} is missing a header row.")
    return list(reader.fieldnames), [dict(row) for row in reader]


def read_rows(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    payload = read_input_bytes(path, expected_sha256=expected_sha256)
    return _rows_from_bytes(payload, path=path)


def write_rows(path: str | Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=infer_delimiter(table_path))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return table_path


def read_json(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    payload = read_input_bytes(path, expected_sha256=expected_sha256)
    return json.loads(payload.decode("utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path


def resolve_feature_columns(
    fieldnames: list[str],
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> list[str]:
    if feature_columns is None:
        blocked = {target_column, "split_set"}
        return [name for name in fieldnames if name not in blocked]

    explicit_columns = list(feature_columns)
    if not explicit_columns:
        raise ValueError("Explicit feature columns must contain at least one column.")
    if any(not isinstance(name, str) or not name.strip() for name in explicit_columns):
        raise ValueError("Explicit feature columns must contain only non-blank strings.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in explicit_columns:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"Duplicate feature columns are not allowed: {', '.join(duplicates)}")
    if target_column in explicit_columns:
        raise ValueError(f"Target column {target_column!r} cannot also be a feature column.")

    reserved = [name for name in explicit_columns if name == "split_set"]
    if reserved:
        raise ValueError(f"Reserved generated columns cannot be feature columns: {', '.join(reserved)}")

    missing = [name for name in explicit_columns if name not in fieldnames]
    if missing:
        raise ValueError(f"Unknown feature columns: {', '.join(missing)}")
    return explicit_columns


def infer_numeric_feature_columns(
    *,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    target_column: str,
    feature_columns: list[str] | None = None,
) -> list[str]:
    candidates = resolve_feature_columns(fieldnames, target_column=target_column, feature_columns=feature_columns)
    if feature_columns is not None:
        for column in candidates:
            for row_number, row in enumerate(rows, start=1):
                numeric_value(row[column], column=column, row_number=row_number)
        return candidates

    numeric_columns: list[str] = []
    for column in candidates:
        try:
            for row_number, row in enumerate(rows, start=1):
                numeric_value(row[column], column=column, row_number=row_number)
        except ValueError:
            continue
        numeric_columns.append(column)
    if not numeric_columns:
        raise ValueError("No numeric feature columns were available for this slice.")
    return numeric_columns


def format_float(value: float) -> str:
    rendered = f"{value:.10f}".rstrip("0").rstrip(".")
    return rendered if rendered not in {"", "-0"} else "0"


def numeric_value(value: str | None, *, column: str, row_number: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} must be numeric. Invalid value {value!r} at row {row_number}.") from exc
