"""Events build-spec semantics: spec loading, run grouping and row building."""

from .grouping import read_source_rows, resolve_run_groups, resolve_subject_session
from .rows import build_run_rows, validate_compiled_plan_rows
from .service import (
    EventsSemanticAPI,
    EventsSemanticResult,
    EventsSemanticRun,
    SemanticEventsBuildResult,
    SemanticRunRows,
    build_semantic_events,
    events_semantic_api,
    plan_semantic_events,
)
from .spec import BuildSpec, load_build_spec

__all__ = [
    "BuildSpec",
    "EventsSemanticAPI",
    "EventsSemanticResult",
    "EventsSemanticRun",
    "SemanticEventsBuildResult",
    "SemanticRunRows",
    "build_run_rows",
    "build_semantic_events",
    "events_semantic_api",
    "load_build_spec",
    "plan_semantic_events",
    "read_source_rows",
    "resolve_run_groups",
    "resolve_subject_session",
    "validate_compiled_plan_rows",
]
