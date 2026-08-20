"""Reusable event-file inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NumericEventFileInspection:
    path: Path
    status: str
    message: str | None = None
    row_count: int = 0


def inspect_numeric_event_file(path: str | Path, *, min_columns: int = 3) -> NumericEventFileInspection:
    """Classify a whitespace-delimited numeric event file.

    Blank lines and comment lines are ignored. A file with no data rows is
    classified as ``empty`` so callers can decide whether that is valid for
    their runtime.
    """

    event_path = Path(path)
    if not event_path.exists():
        return NumericEventFileInspection(path=event_path, status="missing", message=f"File is missing: {event_path}")
    if not event_path.is_file():
        return NumericEventFileInspection(path=event_path, status="invalid", message=f"Path is not a file: {event_path}")

    row_count = 0
    try:
        with event_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                row_count += 1
                tokens = stripped.split()
                if len(tokens) < min_columns:
                    return NumericEventFileInspection(
                        path=event_path,
                        status="invalid",
                        message=f"Line {line_number} has {len(tokens)} columns; expected at least {min_columns}.",
                        row_count=row_count,
                    )
                for token in tokens[:min_columns]:
                    try:
                        float(token)
                    except ValueError:
                        return NumericEventFileInspection(
                            path=event_path,
                            status="invalid",
                            message=f"Line {line_number} contains a non-numeric value: {token!r}.",
                            row_count=row_count,
                        )
    except OSError as exc:
        return NumericEventFileInspection(path=event_path, status="invalid", message=str(exc), row_count=row_count)

    if row_count == 0:
        return NumericEventFileInspection(path=event_path, status="empty", row_count=0)
    return NumericEventFileInspection(path=event_path, status="valid", row_count=row_count)
