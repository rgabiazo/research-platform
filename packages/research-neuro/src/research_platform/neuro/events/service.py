from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .grouping import read_source_rows, resolve_run_groups, resolve_subject_session
from .rows import build_run_rows, validate_compiled_plan_rows
from .spec import BuildSpec, load_build_spec


@dataclass(frozen=True)
class EventsSemanticRun:
    run: int
    rows: list[dict[str, str]]
    row_count: int


@dataclass(frozen=True)
class EventsSemanticResult:
    spec: BuildSpec
    source_path: Path
    subject: str
    session: str | None
    runs: list[EventsSemanticRun]
    warnings: list[str]

    def run_rows_by_run(self) -> dict[int, list[dict[str, str]]]:
        return {run.run: run.rows for run in self.runs}

    def total_row_count(self) -> int:
        return sum(run.row_count for run in self.runs)


SemanticRunRows = EventsSemanticRun
SemanticEventsBuildResult = EventsSemanticResult


class EventsSemanticAPI(Protocol):
    def plan(
        self,
        *,
        spec_path: str | Path,
        source_path: str | Path,
        session: str | None = None,
    ) -> EventsSemanticResult: ...

    def build(
        self,
        *,
        spec_path: str | Path,
        source_path: str | Path,
        session: str | None = None,
        preserve_source_stim_file: bool = False,
    ) -> EventsSemanticResult: ...


def _resolve_semantic_build(
    *,
    spec_path: str | Path,
    source_path: str | Path,
    session: str | None,
    preserve_source_stim_file: bool,
) -> EventsSemanticResult:
    spec = load_build_spec(spec_path)
    resolved_source_path = Path(source_path)
    rows = read_source_rows(resolved_source_path, encoding=spec.source_encoding)
    validate_compiled_plan_rows(rows, spec, session_override=session)
    subject, resolved_session = resolve_subject_session(rows, spec, session_override=session)
    groups = resolve_run_groups(rows, spec, session_override=session)
    run_rows = build_run_rows(groups, spec, preserve_source_stim_file=preserve_source_stim_file)
    runs = [
        EventsSemanticRun(run=run, rows=event_rows, row_count=len(event_rows))
        for run, event_rows in sorted(run_rows.items())
    ]
    return EventsSemanticResult(
        spec=spec,
        source_path=resolved_source_path,
        subject=subject,
        session=resolved_session,
        runs=runs,
        warnings=[],
    )


def plan_semantic_events(
    *,
    spec_path: str | Path,
    source_path: str | Path,
    session: str | None = None,
) -> EventsSemanticResult:
    return _resolve_semantic_build(
        spec_path=spec_path,
        source_path=source_path,
        session=session,
        preserve_source_stim_file=False,
    )


def build_semantic_events(
    *,
    spec_path: str | Path,
    source_path: str | Path,
    session: str | None = None,
    preserve_source_stim_file: bool = False,
) -> EventsSemanticResult:
    return _resolve_semantic_build(
        spec_path=spec_path,
        source_path=source_path,
        session=session,
        preserve_source_stim_file=preserve_source_stim_file,
    )


@dataclass(frozen=True)
class _EventsSemanticFacade:
    """Canonical semantic handoff surface consumed by the BIDS facade."""

    result_type: type[EventsSemanticResult] = EventsSemanticResult
    run_type: type[EventsSemanticRun] = EventsSemanticRun

    def plan(
        self,
        *,
        spec_path: str | Path,
        source_path: str | Path,
        session: str | None = None,
    ) -> EventsSemanticResult:
        return plan_semantic_events(
            spec_path=spec_path,
            source_path=source_path,
            session=session,
        )

    def build(
        self,
        *,
        spec_path: str | Path,
        source_path: str | Path,
        session: str | None = None,
        preserve_source_stim_file: bool = False,
    ) -> EventsSemanticResult:
        return build_semantic_events(
            spec_path=spec_path,
            source_path=source_path,
            session=session,
            preserve_source_stim_file=preserve_source_stim_file,
        )


events_semantic_api: EventsSemanticAPI = _EventsSemanticFacade()
