"""Backend-neutral and compatibility plan-only MVPA discovery plans.

Plan objects are JSON-serializable previews only. The unified discovery layer
may delegate to plan-only FEAT PE and ROI source planners, but it does not load
NIfTI files, extract voxel data, run MVPA/crossnobis, execute FSL, invoke CLI
commands, generate reports, orchestrate pipelines, or write outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, fields
from pathlib import Path
from typing import Any

from .config import (
    ConditionPairConfig,
    ExclusionRule,
    MeanCenteringConfig,
    MvpaSetConfig,
    ThresholdSweepConfig,
    parse_mvpa_set_document,
    validate_mvpa_set_document,
)
from .pattern_source_adapters import (
    default_pattern_source_adapter_registry,
    runtime_unit_contexts,
)
from .pattern_sources import (
    AnalysisUnitResolution,
    PatternSourceExecutionHandle,
    PatternSourceAdapterRegistry,
    resolve_analysis_units,
)
from .roi_sources import plan_mvpa_roi_sources


@dataclass(frozen=True)
class PatternSourcePlanRow:
    """One deferred pattern-source discovery row."""

    name: str
    backend: str
    status: str = "deferred"
    reason: str = "adapter_planning_not_attempted"
    root_ref: str | None = None
    path_template: str | None = None
    pattern_template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiSourcePlanRow:
    """One deferred ROI-source row."""

    name: str
    source: str
    status: str = "deferred"
    reason: str = "roi_source_planning_not_attempted"
    roi_set_ref: str | None = None
    root_ref: str | None = None
    path_template: str | None = None
    pattern_template: str | None = None
    mask_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ConditionPlanRow:
    """One configured condition row."""

    id: str
    aliases: tuple[str, ...] = ()
    status: str = "configured"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class DistancePlanRow:
    """One configured distance computation row in the runtime plan."""

    metric: str
    engine: str
    cv_unit: str
    noise_normalization_method: str
    noise_nonpositive_policy: str = "strict"
    min_retained_features: int = 5
    warn_dropped_feature_fraction: float = 0.10
    status: str = "planned"
    reason: str = "configured_for_representation_aware_runtime"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ConditionPairPlanRow:
    """One configured runtime condition pair."""

    id: str
    condition_id_a: str
    condition_id_b: str
    status: str = "configured"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ThresholdSweepPlanRow:
    """One configured runtime threshold sweep."""

    id: str
    min_events: int | None = None
    min_observations: int | None = None
    status: str = "configured"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ExclusionPlanRow:
    """One configured exclusion row."""

    id: str
    reason: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    source_config_field: str = "mvpa_set.exclusions.rules"
    status: str = "configured"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class OutputPreviewRow:
    """A root-ref plus relative path template preview."""

    name: str
    root_ref: str
    relative_path_template: str
    status: str = "preview"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MissingInputPlanRow:
    """A missing-input policy row used when source checks are not attempted."""

    policy: str
    status: str = "not_checked"
    reason: str = "source_checks_not_attempted"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MvpaDiscoveryPlan:
    """JSON-serializable backend-neutral MVPA discovery plan."""

    mvpa_set_name: str | None
    status: str
    schema_valid: bool = True
    ready_for_execution: bool = False
    ready_for_materialization: bool = False
    unit_selection_mode: str = "legacy_cartesian"
    errors: tuple[str, ...] = ()
    pattern_sources: tuple[PatternSourcePlanRow, ...] = ()
    roi_sources: tuple[RoiSourcePlanRow, ...] = ()
    conditions: tuple[ConditionPlanRow, ...] = ()
    distances: tuple[DistancePlanRow, ...] = ()
    condition_pairs: tuple[ConditionPairPlanRow, ...] = ()
    threshold_sweeps: tuple[ThresholdSweepPlanRow, ...] = ()
    exclusions: tuple[ExclusionPlanRow, ...] = ()
    outputs: tuple[OutputPreviewRow, ...] = ()
    missing_inputs: tuple[MissingInputPlanRow, ...] = ()
    event_thresholds: Mapping[str, Any] | None = None
    mean_centering: Mapping[str, Any] = field(default_factory=dict)
    grouping_columns: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    backend_summary: Mapping[str, Any] = field(default_factory=dict)
    adapter_availability: tuple[Mapping[str, Any], ...] = ()
    analysis_units: tuple[Mapping[str, Any], ...] = ()
    pattern_rows: tuple[Mapping[str, Any], ...] = ()
    pattern_source_metadata_rows: tuple[Mapping[str, Any], ...] = ()
    pattern_source_summaries: tuple[Mapping[str, Any], ...] = ()
    pattern_source_provenance: tuple[Mapping[str, Any], ...] = ()
    subjects: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    runs: tuple[str, ...] = ()
    mvpa_set: str | None = None
    pattern_source_rows: tuple[Mapping[str, Any], ...] = ()
    condition_pe_rows: tuple[Mapping[str, Any], ...] = ()
    roi_source_rows: tuple[Mapping[str, Any], ...] = ()
    input_checks: tuple[Mapping[str, Any], ...] = ()
    provenance_rows: tuple[Mapping[str, Any], ...] = ()
    event_threshold_rows: tuple[Mapping[str, Any], ...] = ()
    _execution_handles: tuple[PatternSourceExecutionHandle, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    executed: bool = False

    @property
    def plan_valid(self) -> bool:
        """Return whether schema validation and plan assembly both succeeded."""

        return self.schema_valid and not self.errors and self.status not in {"error", "invalid"}

    @property
    def valid(self) -> bool:
        """Backward-compatible alias for complete plan validity."""

        return self.plan_valid

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["plan_valid"] = self.plan_valid
        payload["valid"] = self.valid
        return payload


def plan_mvpa_discovery(
    document: Mapping[str, Any] | Any,
    context: Mapping[str, Any] | None = None,
    *,
    roots: Mapping[str, str | Path] | None = None,
    roi_sets: Mapping[str, Any] | None = None,
    enable_backend_discovery: bool = True,
    exact_units: Sequence[Mapping[str, Any]] | None = None,
    unit_key_columns: Sequence[str] | None = None,
    adapter_registry: PatternSourceAdapterRegistry | None = None,
) -> MvpaDiscoveryPlan:
    """Return a plan-only MVPA discovery preview or integration plan.

    Invalid configs return ``status="invalid"`` with validation errors. Valid
    configs with no supplied source metadata return a deferred preview. When
    concrete roots or ROI metadata are supplied, the explicit adapter registry
    plans canonical rows alongside compatibility details.
    """

    registry = adapter_registry or default_pattern_source_adapter_registry()
    errors = validate_mvpa_set_document(document, adapter_registry=registry)
    if errors:
        return MvpaDiscoveryPlan(
            mvpa_set_name=_candidate_mvpa_set_name(document),
            status="invalid",
            schema_valid=False,
            ready_for_execution=False,
            errors=tuple(errors),
            context=dict(context or {}),
            backend_summary={"integration_attempted": False},
            mvpa_set=_candidate_mvpa_set_name(document),
            executed=False,
        )

    config = parse_mvpa_set_document(document, adapter_registry=registry)
    unit_resolution = resolve_analysis_units(
        config,
        exact_units=exact_units,
        unit_key_columns=unit_key_columns,
    )
    if not unit_resolution.valid:
        return MvpaDiscoveryPlan(
            mvpa_set_name=config.name,
            status="error",
            schema_valid=True,
            ready_for_execution=False,
            unit_selection_mode=unit_resolution.mode,
            errors=unit_resolution.errors,
            context=dict(context or {}),
            backend_summary={"integration_attempted": False},
            adapter_availability=_adapter_availability(config, registry=registry),
            analysis_units=_analysis_unit_rows(unit_resolution),
            subjects=config.selector.subjects,
            sessions=config.selector.sessions,
            runs=config.selector.runs,
            mvpa_set=config.name,
            executed=False,
        )
    integration_enabled = enable_backend_discovery and (roots is not None or roi_sets is not None)
    if not integration_enabled:
        return _deferred_plan(
            config,
            context=context,
            unit_resolution=unit_resolution,
            registry=registry,
        )

    return _integrated_plan(
        config,
        context=context,
        roots=roots,
        roi_sets=roi_sets,
        unit_resolution=unit_resolution,
        registry=registry,
    )


def _deferred_plan(
    config: MvpaSetConfig,
    *,
    context: Mapping[str, Any] | None,
    unit_resolution: AnalysisUnitResolution,
    registry: PatternSourceAdapterRegistry,
) -> MvpaDiscoveryPlan:
    availability = _adapter_availability(config, registry=registry)
    return MvpaDiscoveryPlan(
        mvpa_set_name=config.name,
        status="deferred",
        schema_valid=True,
        ready_for_execution=False,
        unit_selection_mode=unit_resolution.mode,
        pattern_sources=_pattern_source_rows(config),
        roi_sources=_roi_source_rows(config),
        conditions=tuple(ConditionPlanRow(id=condition.id, aliases=condition.aliases) for condition in config.conditions),
        distances=_distance_rows(config),
        condition_pairs=_condition_pair_rows(config),
        threshold_sweeps=_threshold_sweep_rows(config),
        exclusions=_exclusion_rows(config),
        outputs=tuple(
            OutputPreviewRow(
                name=root.name,
                root_ref=root.root_ref,
                relative_path_template=root.path,
            )
            for root in config.outputs.roots
        ),
        missing_inputs=(MissingInputPlanRow(policy=config.missing_input_policy.policy),),
        event_thresholds=dict(config.event_thresholds.fields) if config.event_thresholds is not None else None,
        mean_centering=_mean_centering_payload(config.mean_centering),
        grouping_columns=config.distance.grouping_columns,
        context=dict(context or {}),
        backend_summary=_backend_summary(
            config=config,
            pattern_statuses={source.name: "deferred" for source in config.pattern_sources},
            roi_statuses={source.name: "deferred" for source in config.roi_sources},
            integration_attempted=False,
            adapter_availability=availability,
        ),
        adapter_availability=availability,
        analysis_units=_analysis_unit_rows(unit_resolution),
        subjects=config.selector.subjects,
        sessions=config.selector.sessions,
        runs=config.selector.runs,
        mvpa_set=config.name,
        event_threshold_rows=_event_threshold_rows(config),
        executed=False,
    )


def _integrated_plan(
    config: MvpaSetConfig,
    *,
    context: Mapping[str, Any] | None,
    roots: Mapping[str, str | Path] | None,
    roi_sets: Mapping[str, Any] | None,
    unit_resolution: AnalysisUnitResolution,
    registry: PatternSourceAdapterRegistry,
) -> MvpaDiscoveryPlan:
    pattern_sources: list[PatternSourcePlanRow] = []
    roi_sources: list[RoiSourcePlanRow] = []
    pattern_rows: list[Mapping[str, Any]] = []
    pattern_source_metadata_rows: list[Mapping[str, Any]] = []
    pattern_source_summaries: list[Mapping[str, Any]] = []
    pattern_source_provenance: list[Mapping[str, Any]] = []
    pattern_source_rows: list[Mapping[str, Any]] = []
    condition_pe_rows: list[Mapping[str, Any]] = []
    roi_source_rows: list[Mapping[str, Any]] = []
    input_checks: list[Mapping[str, Any]] = []
    provenance_rows: list[Mapping[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    deferred_components: list[str] = []
    pattern_statuses: dict[str, str] = {}
    pattern_readiness: dict[str, bool] = {}
    materialization_readiness: dict[str, bool] = {}
    execution_handles: list[PatternSourceExecutionHandle] = []
    event_threshold_rows: list[Mapping[str, Any]] = []
    roi_statuses: dict[str, str] = {}
    adapter_availability = list(_adapter_availability(config, registry=registry))

    for source in config.pattern_sources:
        adapter = registry.require(source.backend)
        source_plan = adapter.plan_source(
            config=config,
            source=source,
            units=unit_resolution.units,
            roots=roots,
            context=context,
            raise_on_fail_policy=False,
        )
        pattern_statuses[source.name] = source_plan.status
        pattern_readiness[source.name] = source_plan.ready_for_execution
        materialization_readiness[source.name] = source_plan.ready_for_materialization
        if source_plan.materialization_handle is not None:
            representation_kinds = tuple(
                dict.fromkeys(row.representation_kind for row in source_plan.pattern_rows)
            )
            if len(representation_kinds) == 1:
                execution_handles.append(
                    PatternSourceExecutionHandle(
                        source_name=source.name,
                        backend_name=source.backend,
                        representation_kind=representation_kinds[0],
                        payload=source_plan.materialization_handle,
                    )
                )
        if not adapter.capabilities.planning_supported or source_plan.status == "deferred":
            deferred_components.append(f"pattern source {source.name!r}")
        pattern_sources.append(
            PatternSourcePlanRow(
                name=source.name,
                backend=source.backend,
                status=source_plan.status,
                reason=adapter.capabilities.reason or "adapter_plan_only",
                root_ref=source.root_ref,
                path_template=source.path,
                pattern_template=source.pattern,
            )
        )
        pattern_rows.extend(row.to_dict() for row in source_plan.pattern_rows)
        pattern_source_metadata_rows.extend(
            {
                **dict(row),
                "source_name": source.name,
                "backend": source.backend,
            }
            for row in source_plan.source_rows
        )
        if source_plan.source_summary:
            source_threshold_rows = source_plan.source_summary.get("event_threshold_rows", ())
            if isinstance(source_threshold_rows, Sequence) and not isinstance(
                source_threshold_rows, (str, bytes)
            ):
                event_threshold_rows.extend(
                    {
                        "source_name": source.name,
                        "backend": source.backend,
                        **dict(row),
                    }
                    for row in source_threshold_rows
                    if isinstance(row, Mapping)
                )
            pattern_source_summaries.append(
                {
                    "source_name": source.name,
                    "backend": source.backend,
                    **dict(source_plan.source_summary),
                }
            )
        if source_plan.source_provenance:
            pattern_source_provenance.append(
                {
                    "source_name": source.name,
                    "backend": source.backend,
                    **dict(source_plan.source_provenance),
                }
            )
        pattern_source_rows.extend(
            _rows_to_dicts(
                source_plan.compatibility_source_rows,
                source_type="pattern_source",
                source_name=source.name,
                planner=adapter.name,
            )
        )
        condition_pe_rows.extend(
            _rows_to_dicts(
                source_plan.compatibility_condition_rows,
                source_type="pattern_source",
                source_name=source.name,
                planner=adapter.name,
            )
        )
        input_checks.extend(
            _rows_to_dicts(
                source_plan.input_checks,
                source_type="pattern_source",
                source_name=source.name,
                planner=adapter.name,
            )
        )
        warnings.extend(_prefix_messages(source_plan.warnings, source_type="pattern source", source_name=source.name))
        errors.extend(_prefix_messages(source_plan.errors, source_type="pattern source", source_name=source.name))

    prepared_features_only = _prepared_features_only(config, registry=registry)
    for source in config.roi_sources:
        if prepared_features_only:
            coverage_validated = (
                bool(pattern_rows)
                and not errors
                and bool(pattern_statuses)
                and all(status in {"valid", "warning"} for status in pattern_statuses.values())
            )
            roi_statuses[source.name] = (
                "validated_in_materialized_table"
                if coverage_validated
                else "materialized_table_validation_failed"
            )
            roi_sources.append(
                RoiSourcePlanRow(
                    name=source.name,
                    source=source.source,
                    status="valid" if coverage_validated else "error",
                    reason=(
                        "roi_coverage_validated_by_materialized_pattern_table"
                        if coverage_validated
                        else "roi_coverage_not_validated_due_to_materialized_source_errors"
                    ),
                    roi_set_ref=source.roi_set_ref,
                    root_ref=source.root_ref,
                    path_template=source.path,
                    pattern_template=source.pattern,
                    mask_count=len(source.masks),
                )
            )
            continue
        source_plan = plan_mvpa_roi_sources(
            config,
            roi_source_name=source.name,
            roots=roots,
            roi_sets=roi_sets,
            context=context,
            unit_contexts=runtime_unit_contexts(config, unit_resolution.units, context=context),
            raise_on_fail_policy=False,
        )
        roi_statuses[source.name] = source_plan.status
        roi_sources.append(
            RoiSourcePlanRow(
                name=source.name,
                source=source.source,
                status=source_plan.status,
                reason="integrated_roi_source_plan_only",
                roi_set_ref=source.roi_set_ref,
                root_ref=source.root_ref,
                path_template=source.path,
                pattern_template=source.pattern,
                mask_count=len(source_plan.rows) if source_plan.rows else len(source.masks),
            )
        )
        roi_source_rows.extend(
            _rows_to_dicts(
                source_plan.rows,
                source_type="roi_source",
                source_name=source.name,
                planner="roi_sources",
            )
        )
        input_checks.extend(
            _rows_to_dicts(
                source_plan.input_checks,
                source_type="roi_source",
                source_name=source.name,
                planner="roi_sources",
            )
        )
        provenance_rows.extend(
            _rows_to_dicts(
                source_plan.provenance_rows,
                source_type="roi_source",
                source_name=source.name,
                planner="roi_sources",
            )
        )
        warnings.extend(_prefix_messages(source_plan.warnings, source_type="ROI source", source_name=source.name))
        errors.extend(_prefix_messages(source_plan.errors, source_type="ROI source", source_name=source.name))

    warnings = _unique_text(warnings)
    errors = _unique_text(errors)
    status = _aggregate_status(
        errors=errors,
        warnings=warnings,
        deferred_components=deferred_components,
        pattern_source_rows=pattern_source_rows,
        condition_pe_rows=condition_pe_rows,
        roi_source_rows=roi_source_rows,
        input_checks=input_checks,
        provenance_rows=provenance_rows,
        canonical_pattern_rows=pattern_rows,
    )
    availability = tuple(
        {
            **row,
            "ready_for_execution": pattern_readiness.get(str(row["source_name"]), False),
            "ready_for_materialization": materialization_readiness.get(
                str(row["source_name"]), False
            ),
        }
        for row in adapter_availability
    )
    ready_for_execution = _ready_for_execution(
        status=status,
        errors=errors,
        pattern_readiness=pattern_readiness,
        materialization_readiness=materialization_readiness,
        prepared_features_only=prepared_features_only,
        pattern_rows=pattern_rows,
        condition_pe_rows=condition_pe_rows,
        roi_source_rows=roi_source_rows,
    )
    ready_for_materialization = _ready_for_materialization(
        status=status,
        errors=errors,
        prepared_features_only=prepared_features_only,
        materialization_readiness=materialization_readiness,
        pattern_rows=pattern_rows,
    )

    return MvpaDiscoveryPlan(
        mvpa_set_name=config.name,
        status=status,
        schema_valid=True,
        ready_for_execution=ready_for_execution,
        ready_for_materialization=ready_for_materialization,
        unit_selection_mode=unit_resolution.mode,
        errors=tuple(errors),
        pattern_sources=tuple(pattern_sources),
        roi_sources=tuple(roi_sources),
        conditions=tuple(ConditionPlanRow(id=condition.id, aliases=condition.aliases) for condition in config.conditions),
        distances=_distance_rows(config),
        condition_pairs=_condition_pair_rows(config),
        threshold_sweeps=_threshold_sweep_rows(config),
        exclusions=_exclusion_rows(config),
        outputs=tuple(
            OutputPreviewRow(
                name=root.name,
                root_ref=root.root_ref,
                relative_path_template=root.path,
            )
            for root in config.outputs.roots
        ),
        missing_inputs=(MissingInputPlanRow(policy=config.missing_input_policy.policy),),
        event_thresholds=dict(config.event_thresholds.fields) if config.event_thresholds is not None else None,
        mean_centering=_mean_centering_payload(config.mean_centering),
        grouping_columns=config.distance.grouping_columns,
        context=dict(context or {}),
        warnings=tuple(warnings),
        backend_summary=_backend_summary(
            config=config,
            pattern_statuses=pattern_statuses,
            roi_statuses=roi_statuses,
            integration_attempted=True,
            adapter_availability=availability,
        ),
        adapter_availability=availability,
        analysis_units=_analysis_unit_rows(unit_resolution),
        pattern_rows=tuple(pattern_rows),
        pattern_source_metadata_rows=tuple(pattern_source_metadata_rows),
        pattern_source_summaries=tuple(pattern_source_summaries),
        pattern_source_provenance=tuple(pattern_source_provenance),
        subjects=config.selector.subjects,
        sessions=config.selector.sessions,
        runs=config.selector.runs,
        mvpa_set=config.name,
        pattern_source_rows=tuple(pattern_source_rows),
        condition_pe_rows=tuple(condition_pe_rows),
        roi_source_rows=tuple(roi_source_rows),
        input_checks=tuple(input_checks),
        provenance_rows=tuple(provenance_rows),
        event_threshold_rows=(
            tuple(event_threshold_rows)
            if event_threshold_rows
            else _event_threshold_rows(config)
        ),
        _execution_handles=tuple(execution_handles),
        executed=False,
    )


def _pattern_source_rows(config: MvpaSetConfig) -> tuple[PatternSourcePlanRow, ...]:
    return tuple(
        PatternSourcePlanRow(
            name=source.name,
            backend=source.backend,
            root_ref=source.root_ref,
            path_template=source.path,
            pattern_template=source.pattern,
        )
        for source in config.pattern_sources
    )


def _roi_source_rows(config: MvpaSetConfig) -> tuple[RoiSourcePlanRow, ...]:
    return tuple(
        RoiSourcePlanRow(
            name=source.name,
            source=source.source,
            roi_set_ref=source.roi_set_ref,
            root_ref=source.root_ref,
            path_template=source.path,
            pattern_template=source.pattern,
            mask_count=len(source.masks),
        )
        for source in config.roi_sources
    )


def _distance_rows(config: MvpaSetConfig) -> tuple[DistancePlanRow, ...]:
    rows: list[DistancePlanRow] = []
    for metric in config.distance.metrics:
        for engine in config.distance.engines:
            rows.append(
                DistancePlanRow(
                    metric=metric,
                    engine=engine,
                    cv_unit=config.distance.cv_unit,
                    noise_normalization_method=config.distance.noise_normalization.method,
                    noise_nonpositive_policy=config.distance.noise_normalization.nonpositive_policy,
                    min_retained_features=config.distance.noise_normalization.min_retained_features,
                    warn_dropped_feature_fraction=config.distance.noise_normalization.warn_dropped_feature_fraction,
                )
            )
    return tuple(rows)


def _condition_pair_rows(config: MvpaSetConfig) -> tuple[ConditionPairPlanRow, ...]:
    return tuple(_condition_pair_row(pair) for pair in config.condition_pairs)


def _condition_pair_row(pair: ConditionPairConfig) -> ConditionPairPlanRow:
    return ConditionPairPlanRow(
        id=pair.id,
        condition_id_a=pair.condition_id_a,
        condition_id_b=pair.condition_id_b,
    )


def _threshold_sweep_rows(config: MvpaSetConfig) -> tuple[ThresholdSweepPlanRow, ...]:
    return tuple(_threshold_sweep_row(sweep) for sweep in config.threshold_sweeps)


def _threshold_sweep_row(sweep: ThresholdSweepConfig) -> ThresholdSweepPlanRow:
    return ThresholdSweepPlanRow(
        id=sweep.id,
        min_events=sweep.min_events,
        min_observations=sweep.min_observations,
    )


def _exclusion_rows(config: MvpaSetConfig) -> tuple[ExclusionPlanRow, ...]:
    return tuple(_exclusion_row(rule) for rule in config.exclusions)


def _exclusion_row(rule: ExclusionRule) -> ExclusionPlanRow:
    return ExclusionPlanRow(
        id=rule.id,
        reason=rule.reason,
        subject_id=rule.subject_id,
        session_id=rule.session_id,
        run_id=rule.run_id,
        source_config_field=rule.source_config_field,
    )


def _mean_centering_payload(mean_centering: MeanCenteringConfig) -> Mapping[str, Any]:
    return {
        "enabled": mean_centering.enabled,
        "scope": mean_centering.scope,
    }


def _candidate_mvpa_set_name(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    payload = document.get("mvpa_set") if isinstance(document.get("mvpa_set"), Mapping) else document
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("name")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _event_threshold_rows(config: MvpaSetConfig) -> tuple[Mapping[str, Any], ...]:
    thresholds = config.event_thresholds
    if thresholds is None:
        return ()

    rows: list[Mapping[str, Any]] = []
    for key in ("min_events_per_condition_per_run", "min_runs_per_condition"):
        value = getattr(thresholds, key)
        if value is not None or key in thresholds.fields:
            rows.append(
                {
                    "threshold": key,
                    "value": value,
                    "status": "not_evaluated",
                    "reason": "event_counts_not_read_phase_2f",
                }
            )
    for key, value in thresholds.fields.items():
        key_text = str(key)
        if key_text in {"min_events_per_condition_per_run", "min_runs_per_condition"}:
            continue
        rows.append(
            {
                "threshold": key_text,
                "value": value,
                "status": "not_evaluated",
                "reason": "event_counts_not_read_phase_2f",
            }
        )
    return tuple(rows)


def _backend_summary(
    *,
    config: MvpaSetConfig,
    pattern_statuses: Mapping[str, str],
    roi_statuses: Mapping[str, str],
    integration_attempted: bool,
    adapter_availability: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    availability_by_source = {
        str(row.get("source_name")): row for row in adapter_availability
    }
    return {
        "integration_attempted": integration_attempted,
        "ready_for_execution": bool(adapter_availability)
        and all(bool(row.get("ready_for_execution")) for row in adapter_availability),
        "ready_for_materialization": bool(adapter_availability)
        and all(
            bool(row.get("ready_for_materialization"))
            for row in adapter_availability
        ),
        "pattern_sources": tuple(
            {
                "name": source.name,
                "backend": source.backend,
                "status": pattern_statuses.get(source.name, "deferred"),
                "integrated": integration_attempted
                and bool(availability_by_source.get(source.name, {}).get("planning_supported")),
                "adapter": dict(availability_by_source.get(source.name, {})),
            }
            for source in config.pattern_sources
        ),
        "roi_sources": tuple(
            {
                "name": source.name,
                "source": source.source,
                "status": roi_statuses.get(source.name, "deferred"),
                "integrated": integration_attempted,
            }
            for source in config.roi_sources
        ),
    }


def _rows_to_dicts(
    rows: Sequence[Any],
    *,
    source_type: str,
    source_name: str,
    planner: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            **_row_to_dict(row),
            "source_type": source_type,
            "source_name": source_name,
            "planner": planner,
        }
        for row in rows
    )


def _row_to_dict(row: Any) -> dict[str, Any]:
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, Mapping) else {"value": _json_safe(value)}
    if is_dataclass(row) and not isinstance(row, type):
        return _json_safe_dataclass(row)
    if isinstance(row, Mapping):
        return {str(key): _json_safe(value) for key, value in row.items()}
    return {"value": _json_safe(row)}


def _prefix_messages(messages: Sequence[str], *, source_type: str, source_name: str) -> tuple[str, ...]:
    return tuple(f"{source_type} {source_name!r}: {message}" for message in messages)


def _aggregate_status(
    *,
    errors: Sequence[str],
    warnings: Sequence[str],
    deferred_components: Sequence[str],
    pattern_source_rows: Sequence[Mapping[str, Any]],
    condition_pe_rows: Sequence[Mapping[str, Any]],
    roi_source_rows: Sequence[Mapping[str, Any]],
    input_checks: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    canonical_pattern_rows: Sequence[Mapping[str, Any]],
) -> str:
    all_rows = (
        tuple(pattern_source_rows)
        + tuple(condition_pe_rows)
        + tuple(roi_source_rows)
        + tuple(input_checks)
        + tuple(provenance_rows)
        + tuple(canonical_pattern_rows)
    )
    if errors or any(_row_status(row) == "error" for row in all_rows):
        return "error"

    executable_rows = tuple(pattern_source_rows) + tuple(roi_source_rows)
    if executable_rows and all(_row_is_skipped_or_excluded(row) for row in executable_rows):
        return "skipped-all"

    warning_statuses = {"warning", "skipped", "excluded", "preview_only", "not_checked"}
    if warnings or any(_row_status(row) in warning_statuses or row.get("excluded") is True for row in all_rows):
        return "warning"

    if deferred_components:
        return "deferred"

    return "valid"


def _adapter_availability(
    config: MvpaSetConfig,
    *,
    registry: PatternSourceAdapterRegistry,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for source in config.pattern_sources:
        adapter = registry.require(source.backend)
        capabilities = adapter.capabilities
        rows.append(
            {
                "source_name": source.name,
                "backend": source.backend,
                "registered": True,
                "status": capabilities.status,
                "available": capabilities.planning_supported,
                "schema_supported": capabilities.schema_supported,
                "planning_supported": capabilities.planning_supported,
                "execution_available": capabilities.execution_ready,
                "materialization_supported": capabilities.materialization_supported,
                "ready_for_execution": False,
                "ready_for_materialization": False,
                "representation_kinds": capabilities.representation_kinds,
                "reason": capabilities.reason,
            }
        )
    return tuple(rows)


def _analysis_unit_rows(resolution: AnalysisUnitResolution) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "unit_id": unit.unit_id,
            "source_row": unit.source_row,
            "key_columns": unit.key_columns,
            "subject_id": unit.subject_id,
            "session_id": unit.session_id,
            "task_id": unit.task_id,
            "run_id": unit.run_id,
            "values": dict(unit.values),
        }
        for unit in resolution.units
    )


def _ready_for_execution(
    *,
    status: str,
    errors: Sequence[str],
    pattern_readiness: Mapping[str, bool],
    materialization_readiness: Mapping[str, bool],
    prepared_features_only: bool,
    pattern_rows: Sequence[Mapping[str, Any]],
    condition_pe_rows: Sequence[Mapping[str, Any]],
    roi_source_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if errors or status not in {"valid", "warning"}:
        return False
    if not pattern_readiness or not all(pattern_readiness.values()):
        return False
    usable_statuses = {"ok", "valid", "warning"}
    if prepared_features_only:
        return (
            bool(materialization_readiness)
            and all(materialization_readiness.values())
            and bool(pattern_rows)
            and any(_row_status(row) in usable_statuses for row in pattern_rows)
        )
    if not pattern_rows or not condition_pe_rows or not roi_source_rows:
        return False
    return (
        any(_row_status(row) in usable_statuses for row in pattern_rows)
        and any(
            _row_status(row) in usable_statuses and bool(row.get("pe_image"))
            for row in condition_pe_rows
        )
        and any(
            _row_status(row) in usable_statuses
            and bool(row.get("mask_path"))
            and row.get("mask_exists") is not False
            for row in roi_source_rows
        )
    )


def _prepared_features_only(
    config: MvpaSetConfig,
    *,
    registry: PatternSourceAdapterRegistry,
) -> bool:
    capabilities = tuple(
        registry.require(source.backend).capabilities
        for source in config.pattern_sources
    )
    representation_kinds = {
        kind for capability in capabilities for kind in capability.representation_kinds
    }
    return (
        bool(capabilities)
        and representation_kinds == {"prepared_features"}
        and all(capability.materialization_supported for capability in capabilities)
    )


def _ready_for_materialization(
    *,
    status: str,
    errors: Sequence[str],
    prepared_features_only: bool,
    materialization_readiness: Mapping[str, bool],
    pattern_rows: Sequence[Mapping[str, Any]],
) -> bool:
    return (
        prepared_features_only
        and not errors
        and status in {"valid", "warning"}
        and bool(materialization_readiness)
        and all(materialization_readiness.values())
        and bool(pattern_rows)
    )


def _row_status(row: Mapping[str, Any]) -> str:
    return str(row.get("status") or "")


def _row_is_skipped_or_excluded(row: Mapping[str, Any]) -> bool:
    return _row_status(row) in {"skipped", "excluded"} or row.get("excluded") is True


def _unique_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return {
        field.name: _json_safe(getattr(value, field.name))
        for field in fields(value)
        if not field.name.startswith("_")
    }


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return str(value)


__all__ = [
    "ConditionPairPlanRow",
    "ConditionPlanRow",
    "DistancePlanRow",
    "ExclusionPlanRow",
    "MissingInputPlanRow",
    "MvpaDiscoveryPlan",
    "OutputPreviewRow",
    "PatternSourcePlanRow",
    "RoiSourcePlanRow",
    "ThresholdSweepPlanRow",
    "plan_mvpa_discovery",
]
