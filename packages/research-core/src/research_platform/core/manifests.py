"""Reusable helpers for small, exact-row tabular manifests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path
import csv


Row = Mapping[str, str]
Normalizer = Callable[[str], str]


@dataclass(frozen=True)
class ManifestTable:
    """A raw TSV manifest with source columns, row order, and byte digest."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    source_sha256: str


def read_manifest_table(path: str | Path) -> ManifestTable:
    """Read a UTF-8 TSV without expanding or rewriting stored cell values."""

    file_path = Path(path)
    raw = file_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Manifest must be UTF-8 text: {file_path}") from exc

    handle = StringIO(text, newline="")
    reader = csv.DictReader(handle, delimiter="\t")
    fieldnames = tuple(reader.fieldnames or ())
    _validate_manifest_columns(fieldnames)

    rows: list[dict[str, str]] = []
    for row_index, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(f"Manifest row {row_index} contains more values than its header.")
        rows.append({column: row.get(column) or "" for column in fieldnames})
    return ManifestTable(
        columns=fieldnames,
        rows=tuple(rows),
        source_sha256=sha256(raw).hexdigest(),
    )


def write_manifest_table(
    path: str | Path,
    rows: Sequence[Row],
    *,
    columns: Sequence[str] = (),
    preferred_columns: Sequence[str] = (),
) -> Path:
    """Write every manifest column using deterministic ordering and newlines.

    ``preferred_columns`` are emitted first even when every row leaves them
    empty. This preserves the established four-column BIDS discovery contract.
    Explicit source ``columns`` follow, then arbitrary metadata columns in
    first-seen order. No row values are silently discarded.
    """

    fieldnames = _manifest_output_columns(
        rows,
        columns=columns,
        preferred_columns=preferred_columns,
    )
    _validate_manifest_columns(fieldnames)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                column: "" if row.get(column) is None else str(row.get(column, ""))
                for column in fieldnames
            }
        )

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(buffer.getvalue().encode("utf-8"))
    return file_path


def normalize_filter_values(values: object) -> tuple[str, ...]:
    """Return a stable tuple of non-empty string filter values."""

    if values is None:
        return ()
    if isinstance(values, str):
        candidates: Sequence[object] = (values,)
    elif isinstance(values, Sequence):
        candidates = values
    else:
        candidates = (values,)

    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def filter_manifest_rows(
    rows: Sequence[Row],
    filters: Mapping[str, object],
    *,
    normalizers: Mapping[str, Normalizer] | None = None,
) -> list[dict[str, str]]:
    """Filter rows with OR-within-column and AND-across-columns semantics."""

    active_filters = normalized_manifest_filters(filters)
    if not active_filters:
        return [dict(row) for row in rows]

    normalizers = normalizers or {}
    return [
        dict(row)
        for row in rows
        if manifest_row_matches(row, active_filters, normalizers=normalizers)
    ]


def manifest_row_matches(
    row: Row,
    filters: Mapping[str, object],
    *,
    normalizers: Mapping[str, Normalizer] | None = None,
) -> bool:
    """Return whether one row satisfies normalized manifest filters."""

    active_filters = normalized_manifest_filters(filters)
    normalizers = normalizers or {}
    for column, expected_values in active_filters.items():
        normalize = normalizers.get(column, normalize_manifest_identity)
        actual = normalize(str(row.get(column, "")))
        expected = {normalize(value) for value in expected_values}
        if actual not in expected:
            return False
    return True


def normalize_manifest_identity(value: object) -> str:
    """Normalize a manifest value for comparison without rewriting storage.

    Manifest readers and bundle plans retain the source cell verbatim. This
    comparison form only removes surrounding whitespace so filtering, unit
    identity, grouping, and counts apply the same semantics.
    """

    return str(value).strip()


def normalized_manifest_filters(filters: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """Normalize configured filters without changing their column order."""

    return {
        str(column): normalized
        for column, values in filters.items()
        if (normalized := normalize_filter_values(values))
    }


def unknown_manifest_filter_columns(
    columns: Sequence[str],
    filters: Mapping[str, object],
) -> tuple[str, ...]:
    """Return configured filter columns absent from a manifest header."""

    available = set(columns)
    return tuple(str(column) for column in filters if str(column) not in available)


def _manifest_output_columns(
    rows: Sequence[Row],
    *,
    columns: Sequence[str],
    preferred_columns: Sequence[str],
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in (*preferred_columns, *columns):
        column = str(candidate)
        if column not in seen:
            seen.add(column)
            ordered.append(column)
    for row in rows:
        for candidate in row:
            column = str(candidate)
            if column not in seen:
                seen.add(column)
                ordered.append(column)
    return tuple(ordered)


def _validate_manifest_columns(columns: Sequence[str]) -> None:
    if not columns:
        raise ValueError("Manifest must define at least one column.")
    seen: set[str] = set()
    for index, column in enumerate(columns):
        if not column:
            raise ValueError(f"Manifest column {index + 1} must be non-empty.")
        if column != column.strip():
            raise ValueError(f"Manifest column {column!r} must not contain surrounding whitespace.")
        if column in seen:
            raise ValueError(f"Manifest contains duplicate column: {column}.")
        seen.add(column)


__all__ = [
    "ManifestTable",
    "filter_manifest_rows",
    "manifest_row_matches",
    "normalize_filter_values",
    "normalize_manifest_identity",
    "normalized_manifest_filters",
    "read_manifest_table",
    "unknown_manifest_filter_columns",
    "write_manifest_table",
]
