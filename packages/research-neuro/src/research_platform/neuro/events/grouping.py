from __future__ import annotations

from dataclasses import dataclass
import csv
from decimal import Decimal
from pathlib import Path
from typing import Any

from .spec import BuildSpec, CompiledRowSet, normalize_session_value, resolve_session, resolve_subject


@dataclass(frozen=True)
class RunGroup:
    subject: str
    session: str | None
    run: int
    phase: CompiledRowSet
    rows: list[dict[str, str]]


def read_source_rows(path: str | Path, *, encoding: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding=encoding) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _required_source_columns(spec: BuildSpec, *, session_override: str | None = None) -> set[str]:
    columns = set(spec.required_source_columns)
    if spec.session_column and session_override is not None:
        columns.discard(spec.session_column)
    return columns


def validate_source_columns(rows: list[dict[str, str]], spec: BuildSpec, *, session_override: str | None = None) -> None:
    if not rows:
        raise ValueError("Source file contains no data rows.")
    available = set(rows[0].keys())
    missing = sorted(column for column in _required_source_columns(spec, session_override=session_override) if column not in available)
    if missing:
        raise ValueError(f"Source file is missing required columns: {', '.join(missing)}.")


def resolve_subject_session(
    rows: list[dict[str, str]],
    spec: BuildSpec,
    *,
    session_override: str | None = None,
) -> tuple[str, str | None]:
    subject: str | None = None
    for row in rows:
        subject_value = row.get(spec.subject_column, "").strip()
        if not subject_value:
            continue
        subject = resolve_subject(subject_value, spec.subject_regex, spec.subject_width)
        break
    if subject is None:
        raise ValueError("Unable to resolve subject from source rows.")

    if session_override is not None:
        return subject, normalize_session_value(session_override, spec.session_width)

    if not spec.session_column or not spec.session_regex:
        return subject, None

    sessions: list[str] = []
    for row in rows:
        session_value = row.get(spec.session_column, "").strip()
        if session_value:
            resolved = resolve_session(session_value, spec.session_regex, spec.session_width)
            if resolved not in sessions:
                sessions.append(resolved)
    if len(sessions) > 1:
        raise ValueError(
            "Mixed sessions found in source rows: "
            + ", ".join(f"ses-{session}" for session in sessions)
            + ". Provide --session or split the source file by session."
        )
    if sessions:
        return subject, sessions[0]
    return subject, None


def resolve_run_groups(
    rows: list[dict[str, str]],
    spec: BuildSpec,
    *,
    session_override: str | None = None,
) -> list[RunGroup]:
    validate_source_columns(rows, spec, session_override=session_override)
    subject, session = resolve_subject_session(rows, spec, session_override=session_override)
    groups: list[RunGroup] = []
    for phase in spec.compiled_plan.row_sets:
        selected = [
            row
            for row in rows
            if all(
                (
                    row.get(str(selector["column"]), "").strip() == str(selector["value"])
                    if selector["operator"] == "equals"
                    else row.get(str(selector["column"]), "").startswith(str(selector["value"]))
                    if selector["operator"] == "starts_with"
                    else bool(row.get(str(selector["column"]), "").strip())
                )
                for selector in phase.selectors
            )
        ]
        if not selected:
            context_summary = phase.context.get("image_prefix") or phase.context.get("selection_label") or "n/a"
            raise ValueError(
                "Invalid run-group reference: "
                f"no source rows matched run={phase.run} phase={phase.phase} condition={phase.condition} "
                f"prefix={context_summary!r}. Check phase_templates, conditions, and source stim_column values."
            )
        selected.sort(key=lambda row: Decimal(row[phase.onset_sort_column]))
        groups.append(RunGroup(subject=subject, session=session, run=phase.run, phase=phase, rows=selected))
    return groups
