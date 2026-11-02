"""Task event helpers and lazy re-exports of the BIDS events semantics."""

from .files import NumericEventFileInspection, inspect_numeric_event_file

_SEMANTIC_NAMES = frozenset(
    {
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
    }
)

__all__ = [
    "BuildSpec",
    "EventsSemanticAPI",
    "EventsSemanticResult",
    "EventsSemanticRun",
    "SemanticEventsBuildResult",
    "SemanticRunRows",
    "NumericEventFileInspection",
    "build_run_rows",
    "build_semantic_events",
    "events_semantic_api",
    "inspect_numeric_event_file",
    "load_build_spec",
    "plan_semantic_events",
    "read_source_rows",
    "resolve_run_groups",
    "resolve_subject_session",
    "validate_compiled_plan_rows",
]


def __getattr__(name: str):
    if name in _SEMANTIC_NAMES:
        from research_platform.bids.events import semantics

        return getattr(semantics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
