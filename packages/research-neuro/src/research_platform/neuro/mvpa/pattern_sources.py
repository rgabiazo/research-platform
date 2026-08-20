"""Backend-neutral MVPA pattern-source and exact-unit contracts.

This module is deliberately dependency-light.  It defines the explicit
adapter boundary and turns caller-supplied analysis-bundle rows into ordered
analysis units, but it does not import a backend, inspect the filesystem, load
images, or execute an analysis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import json
import math

from research_platform.neuro._roi_path_safety import published_value_local_path_fields


PATTERN_BACKEND_FSL_FEAT_PE = "fsl_feat_pe"
PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE = "materialized_pattern_table"
PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE = "bids_derivative_pattern_table"
PATTERN_BACKEND_NILEARN_GLM = "nilearn_glm"
PATTERN_BACKEND_SURFACE_CIFTI = "surface_cifti"

ALLOWED_PATTERN_BACKENDS = frozenset(
    {
        PATTERN_BACKEND_FSL_FEAT_PE,
        PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
        PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
        PATTERN_BACKEND_NILEARN_GLM,
        PATTERN_BACKEND_SURFACE_CIFTI,
    }
)

UNIT_SELECTION_EXACT = "exact_units"
UNIT_SELECTION_LEGACY_CARTESIAN = "legacy_cartesian"
ALLOWED_UNIT_SELECTION_MODES = frozenset(
    {UNIT_SELECTION_EXACT, UNIT_SELECTION_LEGACY_CARTESIAN}
)

REPRESENTATION_IMAGE = "image"
REPRESENTATION_PREPARED_FEATURES = "prepared_features"
ALLOWED_REPRESENTATION_KINDS = frozenset(
    {REPRESENTATION_IMAGE, REPRESENTATION_PREPARED_FEATURES}
)


@dataclass(frozen=True)
class UnitSelectionConfig:
    """How an MVPA set obtains its actual analysis units."""

    mode: str = UNIT_SELECTION_LEGACY_CARTESIAN
    key_columns: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedAnalysisUnit:
    """One ordered, exact analysis unit supplied to pattern-source adapters.

    Canonical entity values are retained exactly as supplied.  Backend
    adapters may derive stripped or prefixed template aliases from these
    values, but must not replace the canonical fields in planned pattern rows.
    """

    unit_id: str
    source_row: int
    key_columns: tuple[str, ...]
    subject_id: str
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, ...]:
        """Return the configured unit-key values in configured column order."""

        return tuple(_comparison_text(self.values.get(column)) or "" for column in self.key_columns)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return all deterministic source columns, including extra metadata."""

        return self.values


@dataclass(frozen=True)
class AnalysisUnitResolution:
    """Ordered result of exact-unit or compatibility-mode unit resolution."""

    mode: str
    key_columns: tuple[str, ...]
    units: tuple[ResolvedAnalysisUnit, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether unit identity was resolved without contract errors."""

        return not self.errors


@dataclass(frozen=True)
class PatternSourceAdapterCapabilities:
    """Truthful schema, planning, and execution status for one adapter."""

    status: str
    schema_supported: bool
    planning_supported: bool
    execution_ready: bool
    materialization_supported: bool = False
    representation_kinds: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class PlannedPatternRow:
    """Canonical backend-neutral row for one unit and one condition."""

    unit_id: str
    subject_id: str
    cross_validation_label: str
    condition_id: str
    source_name: str
    backend_name: str
    representation_kind: str
    pattern_reference: str
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    noise_reference: str | None = None
    event_count: int | None = None
    qc_status: str | None = None
    status: str = "planned"
    unit_metadata: Mapping[str, Any] = field(default_factory=dict)
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible row mapping."""

        return {
            "unit_id": self.unit_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "cross_validation_label": self.cross_validation_label,
            "condition_id": self.condition_id,
            "source_name": self.source_name,
            "backend_name": self.backend_name,
            "representation_kind": self.representation_kind,
            "pattern_reference": self.pattern_reference,
            "noise_reference": self.noise_reference,
            "event_count": self.event_count,
            "qc_status": self.qc_status,
            "status": self.status,
            "unit_metadata": _json_safe_mapping(self.unit_metadata),
            "backend_metadata": _json_safe_mapping(self.backend_metadata),
        }


@dataclass(frozen=True)
class PatternSourceAdapterPlan:
    """Backend-neutral envelope returned by every source adapter."""

    adapter_name: str
    status: str
    ready_for_execution: bool
    ready_for_materialization: bool = False
    pattern_rows: tuple[PlannedPatternRow, ...] = ()
    source_rows: tuple[Mapping[str, Any], ...] = ()
    source_summary: Mapping[str, Any] = field(default_factory=dict)
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    compatibility_source_rows: tuple[Any, ...] = ()
    compatibility_condition_rows: tuple[Any, ...] = ()
    input_checks: tuple[Any, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    materialization_handle: Any = field(default=None, repr=False, compare=False)
    executed: bool = False


@dataclass(frozen=True)
class PatternSourceExecutionHandle:
    """Private adapter-owned input retained from one exact discovery plan.

    The opaque payload may contain runtime-local paths and therefore must not
    be serialized, compared, or reconstructed from a public plan mapping.
    """

    source_name: str
    backend_name: str
    representation_kind: str
    payload: Any = field(repr=False, compare=False)


@runtime_checkable
class PatternSourceAdapter(Protocol):
    """Small explicit interface implemented by pattern-source backends."""

    name: str
    status: str
    capabilities: PatternSourceAdapterCapabilities

    def validate_source(self, source: Mapping[str, Any], label: str) -> tuple[str, ...]:
        """Validate one backend-owned source declaration without I/O."""

    def plan_source(
        self,
        *,
        config: Any,
        source: Any,
        units: Sequence[ResolvedAnalysisUnit],
        roots: Mapping[str, str | Path] | None,
        context: Mapping[str, Any] | None,
        raise_on_fail_policy: bool = False,
    ) -> PatternSourceAdapterPlan:
        """Return a plan without loading patterns or invoking external tools."""


@dataclass(frozen=True, init=False)
class PatternSourceAdapterRegistry:
    """Immutable explicit registry with no import or entry-point discovery."""

    _adapters: tuple[PatternSourceAdapter, ...]

    def __init__(self, adapters: Iterable[PatternSourceAdapter] = ()) -> None:
        registered = tuple(adapters)
        names = tuple(adapter.name for adapter in registered)
        duplicates = _duplicates(names)
        if duplicates:
            raise ValueError(
                "Pattern-source adapter names must be unique; duplicate names: "
                + ", ".join(repr(name) for name in duplicates)
                + "."
            )
        if any(not name for name in names):
            raise ValueError("Pattern-source adapter names must be non-empty.")
        object.__setattr__(self, "_adapters", registered)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered adapter names in explicit registration order."""

        return tuple(adapter.name for adapter in self._adapters)

    @property
    def adapters(self) -> tuple[PatternSourceAdapter, ...]:
        """Return the immutable ordered adapter tuple."""

        return self._adapters

    def adapter(self, name: str) -> PatternSourceAdapter | None:
        """Return the named adapter, or ``None`` when it is not registered."""

        return next((adapter for adapter in self._adapters if adapter.name == name), None)

    def require(self, name: str) -> PatternSourceAdapter:
        """Return the named adapter or raise a concise lookup error."""

        adapter = self.adapter(name)
        if adapter is None:
            available = ", ".join(self.names) or "none"
            raise KeyError(f"Unknown pattern-source adapter {name!r}; registered adapters: {available}.")
        return adapter

    def with_adapter(self, adapter: PatternSourceAdapter) -> PatternSourceAdapterRegistry:
        """Return a new registry containing one additional explicit adapter."""

        return PatternSourceAdapterRegistry((*self._adapters, adapter))


def resolve_analysis_units(
    config: Any,
    *,
    exact_units: Sequence[Mapping[str, Any]] | None = None,
    unit_key_columns: Sequence[str] | None = None,
) -> AnalysisUnitResolution:
    """Resolve ordered actual units without inventing entity combinations.

    ``exact_units`` accepts direct row mappings and the bundle resolver's
    ``{"source_row": ..., "values": {...}}`` envelopes.  Exact mode never
    expands rows.  Legacy mode deliberately retains the historical Cartesian
    selector behavior and is labelled accordingly in the result.
    """

    selection = _selection_from_config(config)
    if selection.mode == UNIT_SELECTION_EXACT:
        return _resolve_exact_analysis_units(
            selection,
            exact_units=exact_units,
            unit_key_columns=unit_key_columns,
        )
    return _resolve_legacy_analysis_units(
        config,
        exact_units=exact_units,
        unit_key_columns=unit_key_columns,
    )


def _resolve_exact_analysis_units(
    selection: UnitSelectionConfig,
    *,
    exact_units: Sequence[Mapping[str, Any]] | None,
    unit_key_columns: Sequence[str] | None,
) -> AnalysisUnitResolution:
    configured_keys = selection.key_columns
    supplied_keys = tuple(str(column).strip() for column in (unit_key_columns or ()))
    errors: list[str] = []
    if supplied_keys and supplied_keys != configured_keys:
        errors.append(
            "Exact-unit key columns must match mvpa_set.unit_selection.key_columns in the same order; "
            f"configured {list(configured_keys)!r}, received {list(supplied_keys)!r}."
        )
    if exact_units is None:
        errors.append(
            "Exact-unit mode requires analysis units from the analysis-bundle resolver; "
            "no exact unit rows were supplied."
        )
        return AnalysisUnitResolution(
            mode=UNIT_SELECTION_EXACT,
            key_columns=configured_keys,
            errors=tuple(errors),
        )
    if not exact_units:
        errors.append(
            "Exact-unit mode requires at least one included analysis unit from the analysis-bundle resolver."
        )

    units: list[ResolvedAnalysisUnit] = []
    seen: dict[tuple[str, ...], int] = {}
    for fallback_row, raw_row in enumerate(exact_units, start=1):
        if not isinstance(raw_row, Mapping):
            errors.append(f"Exact unit row {fallback_row} must contain a mapping.")
            continue
        source_row, values = _unit_row_payload(raw_row, fallback_row=fallback_row, errors=errors)
        if values is None:
            continue

        key: list[str] = []
        missing_columns: list[str] = []
        for column in configured_keys:
            value = _comparison_text(values.get(column))
            if value is None:
                missing_columns.append(column)
            else:
                key.append(value)
        if missing_columns:
            errors.append(
                f"Exact unit row {source_row} is missing required key value(s): "
                + ", ".join(missing_columns)
                + "."
            )
            continue

        subject_id = _canonical_entity_text(values.get("subject_id"))
        if subject_id is None:
            errors.append(f"Exact unit row {source_row} is missing required subject_id.")
            continue

        metadata_errors = _exact_unit_metadata_errors(values, source_row=source_row)
        if metadata_errors:
            errors.extend(metadata_errors)
            continue

        key_tuple = tuple(key)
        previous = seen.get(key_tuple)
        if previous is not None:
            rendered = ", ".join(
                f"{column}={value!r}" for column, value in zip(configured_keys, key_tuple)
            )
            errors.append(
                f"Duplicate exact-unit key at source rows {previous} and {source_row}: {rendered}."
            )
            continue
        seen[key_tuple] = source_row

        copied_values = dict(values)
        units.append(
            ResolvedAnalysisUnit(
                unit_id=_unit_id(configured_keys, key_tuple),
                source_row=source_row,
                key_columns=configured_keys,
                subject_id=subject_id,
                session_id=_canonical_entity_text(values.get("session_id")),
                task_id=_canonical_entity_text(values.get("task_id")),
                run_id=_canonical_entity_text(values.get("run_id")),
                values=copied_values,
            )
        )

    return AnalysisUnitResolution(
        mode=UNIT_SELECTION_EXACT,
        key_columns=configured_keys,
        units=tuple(units),
        errors=tuple(errors),
    )


def _resolve_legacy_analysis_units(
    config: Any,
    *,
    exact_units: Sequence[Mapping[str, Any]] | None,
    unit_key_columns: Sequence[str] | None,
) -> AnalysisUnitResolution:
    errors: list[str] = []
    if exact_units is not None:
        errors.append(
            "Exact analysis-unit rows must not be mixed with legacy_cartesian inline selectors."
        )
    if unit_key_columns:
        errors.append(
            "Exact unit key columns must not be supplied with legacy_cartesian inline selectors."
        )
    if errors:
        return AnalysisUnitResolution(
            mode=UNIT_SELECTION_LEGACY_CARTESIAN,
            key_columns=("subject_id", "session_id", "run_id"),
            errors=tuple(errors),
        )

    selector = getattr(config, "selector", None)
    subjects = tuple(getattr(selector, "subjects", ()))
    sessions = tuple(getattr(selector, "sessions", ()))
    runs = tuple(getattr(selector, "runs", ()))
    task_id = _canonical_entity_text(getattr(getattr(config, "entities", None), "task", None))
    units: list[ResolvedAnalysisUnit] = []
    source_row = 0
    for subject_id in subjects:
        for session_id in sessions:
            for run_id in runs:
                source_row += 1
                values = {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "run_id": run_id,
                }
                if task_id is not None:
                    values["task_id"] = task_id
                key = tuple(str(values[column]) for column in ("subject_id", "session_id", "run_id"))
                units.append(
                    ResolvedAnalysisUnit(
                        unit_id=_unit_id(("subject_id", "session_id", "run_id"), key),
                        source_row=source_row,
                        key_columns=("subject_id", "session_id", "run_id"),
                        subject_id=str(subject_id),
                        session_id=str(session_id),
                        task_id=task_id,
                        run_id=str(run_id),
                        values=values,
                    )
                )
    return AnalysisUnitResolution(
        mode=UNIT_SELECTION_LEGACY_CARTESIAN,
        key_columns=("subject_id", "session_id", "run_id"),
        units=tuple(units),
    )


def _selection_from_config(config: Any) -> UnitSelectionConfig:
    selection = getattr(config, "unit_selection", None)
    if isinstance(selection, UnitSelectionConfig):
        return selection
    if isinstance(selection, Mapping):
        return UnitSelectionConfig(
            mode=str(selection.get("mode") or UNIT_SELECTION_LEGACY_CARTESIAN),
            key_columns=tuple(str(column) for column in selection.get("key_columns", ())),
            fields=dict(selection),
        )
    return UnitSelectionConfig()


def _unit_row_payload(
    raw_row: Mapping[str, Any],
    *,
    fallback_row: int,
    errors: list[str],
) -> tuple[int, Mapping[str, Any] | None]:
    if "values" in raw_row:
        values = raw_row.get("values")
        if not isinstance(values, Mapping):
            errors.append(f"Exact unit row {fallback_row}.values must contain a mapping.")
            return fallback_row, None
        source_value = raw_row.get("source_row", fallback_row)
    else:
        values = raw_row
        source_value = raw_row.get("source_row", fallback_row)

    if isinstance(source_value, bool):
        errors.append(f"Exact unit row {fallback_row}.source_row must be a positive integer.")
        return fallback_row, None
    try:
        source_row = int(source_value)
    except (TypeError, ValueError):
        errors.append(f"Exact unit row {fallback_row}.source_row must be a positive integer.")
        return fallback_row, None
    if source_row < 1:
        errors.append(f"Exact unit row {fallback_row}.source_row must be a positive integer.")
        return fallback_row, None
    return source_row, values


def _unit_id(columns: Sequence[str], values: Sequence[str]) -> str:
    payload = [[column, value] for column, value in zip(columns, values)]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "unit-" + sha256(encoded).hexdigest()


def _exact_unit_metadata_errors(
    values: Mapping[str, Any],
    *,
    source_row: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    if published_value_local_path_fields(values, label="exact_unit"):
        errors.append(
            f"Exact unit row {source_row} contains a non-portable local path reference."
        )
    for column, value in values.items():
        if isinstance(value, (Mapping, list, tuple, set)):
            errors.append(
                f"Exact unit row {source_row} metadata column {str(column)!r} must be scalar."
            )
        elif isinstance(value, float) and not math.isfinite(value):
            errors.append(
                f"Exact unit row {source_row} metadata column {str(column)!r} must be finite."
            )
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            errors.append(
                f"Exact unit row {source_row} metadata column {str(column)!r} "
                "must be a JSON-safe scalar."
            )
    return tuple(errors)


def _canonical_entity_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _comparison_text(value: Any) -> str | None:
    text = _canonical_entity_text(value)
    return text.strip() if text is not None else None


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(child) for key, child in value.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ALLOWED_PATTERN_BACKENDS",
    "ALLOWED_REPRESENTATION_KINDS",
    "ALLOWED_UNIT_SELECTION_MODES",
    "AnalysisUnitResolution",
    "PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE",
    "PATTERN_BACKEND_FSL_FEAT_PE",
    "PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE",
    "PATTERN_BACKEND_NILEARN_GLM",
    "PATTERN_BACKEND_SURFACE_CIFTI",
    "PatternSourceAdapter",
    "PatternSourceAdapterCapabilities",
    "PatternSourceAdapterPlan",
    "PatternSourceAdapterRegistry",
    "PatternSourceExecutionHandle",
    "PlannedPatternRow",
    "REPRESENTATION_IMAGE",
    "REPRESENTATION_PREPARED_FEATURES",
    "ResolvedAnalysisUnit",
    "UNIT_SELECTION_EXACT",
    "UNIT_SELECTION_LEGACY_CARTESIAN",
    "UnitSelectionConfig",
    "resolve_analysis_units",
]
