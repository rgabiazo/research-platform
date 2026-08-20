"""Local ROI build and extraction runners.

The module keeps core orchestration thin: generic NIfTI work delegates to the
Phase 2 helpers, and Phase 4 LOSO FLAME1 work delegates to the FSL/LOSO neuro
layers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from string import Formatter
from typing import Any
import csv
import json
import os
import shutil

from research_platform.neuro._roi_runtime_outputs import (
    RoiRuntimeOutput,
    RoiRuntimeOutputError,
    RoiRuntimeOutputTransaction,
    preflight_runtime_outputs,
)
from research_platform.neuro._roi_path_safety import configured_path_is_unsafe

from research_platform.neuro.roi import (
    ExtractionSet,
    RoiDefinition,
    RoiSet,
    parse_extraction_set_document,
    parse_roi_set_document,
    runtime_existing_output_policy,
    validate_roi_sidecar_document,
)


SUPPORTED_BUILD_FAMILIES = frozenset({"coordinate_sphere", "manual_mask", "functional_threshold_map", "loso_group_map"})
DEFERRED_BUILD_FAMILIES = frozenset({"atlas_label", "data_driven_hook"})
SUPPORTED_EXTRACTION_BACKENDS = frozenset({"generic_nifti", "fsl_featquery"})
DEFAULT_EXTRACTION_METRICS = (
    "mean",
    "median",
    "sum",
    "std",
    "min",
    "max",
    "voxel_count",
    "valid_voxel_count",
)
DEFAULT_FEATQUERY_METRICS = ("mean_cope", "roi_voxel_count")
ANALYSIS_VALUES_COLUMN_ORDER = (
    "subject_id",
    "session_id",
    "task_id",
    "model",
    "roi_set",
    "roi_label",
    "roi_desc",
    "roi_family",
    "source_contrast",
    "cope",
    "mean_cope",
    "mean_psc",
    "roi_voxel_count",
    "thresholded_peak",
    "below_threshold_fallback",
    "peak_x_mm",
    "peak_y_mm",
    "peak_z_mm",
    "z_at_peak",
)
VALUES_TABLE_EXCLUDED_COLUMNS = frozenset(
    {
        "feat_dir",
        "roi_mask_path",
        "featquery_output_dir",
        "report_path",
        "backend",
        "featquery_command",
        "usable",
        "qc_flags",
        "warnings",
        "value_map_path",
        "mask_path",
        "metrics",
        "target_name",
        "value_image",
        "featquery_output_name",
        "stats_image",
    }
)
VALUES_TABLE_METRIC_COLUMNS = frozenset(
    {
        "mean",
        "median",
        "sum",
        "std",
        "min",
        "max",
        "voxel_count",
        "valid_voxel_count",
        "mean_cope",
        "mean_psc",
        "median_cope",
        "max_cope",
        "roi_voxel_count",
        "thresholded_peak",
        "below_threshold_fallback",
        "peak_x_mm",
        "peak_y_mm",
        "peak_z_mm",
        "z_at_peak",
    }
)


class UnsupportedRoiRuntimeError(RuntimeError):
    """Raised when a schema-valid ROI config asks for deferred runtime behavior."""


@dataclass(frozen=True)
class RoiExecutionCheck:
    """One stable, JSON-ready ROI execution-readiness finding."""

    check_id: str
    status: str
    message: str
    path: Path | None = None
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "check_id": self.check_id,
                "status": self.status,
                "message": self.message,
                "path": self.path,
                "category": self.category,
            }
        )


@dataclass(frozen=True)
class RoiExecutionPreflight:
    """Read-only readiness result shared by doctor and local executors."""

    ready_for_execution: bool
    checks: tuple[RoiExecutionCheck, ...]
    output_paths: tuple[Path, ...] = ()
    existing_output: str = "fail"

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(check.message for check in self.checks if check.status == "error")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "ready_for_execution": self.ready_for_execution,
                "existing_output": self.existing_output,
                "output_paths": self.output_paths,
                "checks": [check.to_dict() for check in self.checks],
            }
        )


@dataclass(frozen=True)
class RoiExecutionContext:
    """Project-local context needed to resolve ROI config path references."""

    workspace_root: Path
    project_root: Path
    artifacts_root: Path
    project_name: str | None = None
    root_refs: Mapping[str, str | Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        workspace = _resolved_configured_root(self.workspace_root)
        project = _resolved_configured_root(self.project_root)
        artifacts = _resolved_configured_root(self.artifacts_root)
        refs: dict[str, Path] = {
            "workspace_root": workspace,
            "project_root": project,
            "project_config_root": project / "config",
            "analysis_config_root": project / "config" / "analysis",
            "roi_config_root": project / "config" / "analysis",
            "project_roi_root": project / "config" / "analysis",
            "artifacts_root": artifacts,
            "artifact_root": artifacts,
        }
        refs.update({str(name): _resolved_configured_root(value) for name, value in self.root_refs.items()})
        object.__setattr__(self, "workspace_root", workspace)
        object.__setattr__(self, "project_root", project)
        object.__setattr__(self, "artifacts_root", artifacts)
        object.__setattr__(self, "root_refs", refs)

    def resolve_root_ref(self, name: str) -> Path:
        """Resolve a named root reference declared by core/project config."""

        root = self.root_refs.get(name)
        if root is None:
            known = ", ".join(sorted(self.root_refs))
            raise ValueError(f"Unknown ROI path root_ref {name!r}. Known root refs: {known}.")
        return Path(root)


@dataclass(frozen=True)
class RoiBuildAction:
    """One planned or executed ROI mask build."""

    roi_label: str
    family: str
    backend: str
    mask_path: Path
    sidecar_path: Path
    input_paths: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "roi_label": self.roi_label,
                "family": self.family,
                "backend": self.backend,
                "mask_path": self.mask_path,
                "sidecar_path": self.sidecar_path,
                "input_paths": dict(self.input_paths),
                "metadata": dict(self.metadata),
                "result": self.result,
            }
        )


@dataclass(frozen=True)
class RoiBuildPlan:
    """Plan or execution summary for a ROI set."""

    roi_set_name: str
    actions: tuple[RoiBuildAction, ...]
    executed: bool = False
    cleanup: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "roi_set": self.roi_set_name,
                "executed": self.executed,
                "actions": [action.to_dict() for action in self.actions],
                "cleanup": list(self.cleanup),
            }
        )


@dataclass(frozen=True)
class RoiExtractionAction:
    """One planned or executed value-map by ROI-mask extraction."""

    target_name: str
    backend: str
    roi_label: str
    value_map_path: Path
    mask_path: Path
    table_path: Path
    metrics: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "target_name": self.target_name,
                "backend": self.backend,
                "roi_label": self.roi_label,
                "value_map_path": self.value_map_path,
                "mask_path": self.mask_path,
                "table_path": self.table_path,
                "metrics": list(self.metrics),
                "metadata": dict(self.metadata),
                "result": self.result,
            }
        )


@dataclass(frozen=True)
class RoiExtractionPlan:
    """Plan or execution summary for a ROI extraction set."""

    extraction_set_name: str
    roi_set_name: str
    actions: tuple[RoiExtractionAction, ...]
    executed: bool = False
    tables: tuple[Path, ...] = ()
    cleanup: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "extraction_set": self.extraction_set_name,
                "roi_set": self.roi_set_name,
                "executed": self.executed,
                "tables": list(self.tables),
                "actions": [action.to_dict() for action in self.actions],
                "cleanup": list(self.cleanup),
            }
        )


@dataclass(frozen=True)
class _FslFeatqueryContrast:
    source_contrast: str
    cope: str
    desc: str
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class _FslRoiMaskSpec:
    roi_label: str
    mask_path: Path
    roi_desc: str | None = None
    roi_family: str | None = None
    sidecar: Mapping[str, Any] = field(default_factory=dict)


CommandFinder = Callable[[str], str | None]


def load_roi_set_for_execution(document: Mapping[str, Any], *, validate_personal_paths: bool = True) -> RoiSet:
    """Validate and parse an ROI set config document for Phase 3 execution."""

    return parse_roi_set_document(document, validate_personal_paths=validate_personal_paths)


def load_extraction_set_for_execution(
    document: Mapping[str, Any],
    *,
    validate_personal_paths: bool = True,
) -> ExtractionSet:
    """Validate and parse an extraction set config document for Phase 3 execution."""

    return parse_extraction_set_document(document, validate_personal_paths=validate_personal_paths)


def plan_roi_build(
    document: Mapping[str, Any],
    *,
    context: RoiExecutionContext,
    validate_personal_paths: bool = True,
) -> RoiBuildPlan:
    """Validate and plan local ROI mask builds without writing outputs.

    Callers that already checked raw config paths can disable personal-path
    validation while planning against env-resolved runtime paths.
    """

    roi_set = load_roi_set_for_execution(document, validate_personal_paths=validate_personal_paths)
    generic_rois = tuple(roi for roi in roi_set.rois if roi.family != "loso_group_map")
    loso_rois = tuple(roi for roi in roi_set.rois if roi.family == "loso_group_map")
    actions = tuple(_plan_build_action(roi_set, roi, context=context) for roi in generic_rois)
    if loso_rois:
        actions = actions + _plan_loso_build_actions(roi_set, loso_rois, context=context)
    return RoiBuildPlan(roi_set_name=roi_set.name, actions=actions, executed=False)


def preflight_roi_build(
    document: Mapping[str, Any],
    *,
    context: RoiExecutionContext,
    validate_personal_paths: bool = True,
    command_finder: CommandFinder | None = None,
) -> RoiExecutionPreflight:
    """Inspect complete ROI build readiness without creating files or running tools."""

    checks: list[RoiExecutionCheck] = []
    try:
        existing_output = runtime_existing_output_policy(document, payload_key="roi_set")
        roi_set = load_roi_set_for_execution(document, validate_personal_paths=validate_personal_paths)
    except (TypeError, ValueError) as exc:
        checks.append(_execution_check("configuration_valid", "error", str(exc), category="configuration"))
        return _preflight_result(checks, existing_output="fail")

    checks.append(_execution_check("configuration_valid", "ok", "ROI set configuration is structurally valid.", category="configuration"))
    family_errors: list[str] = []
    for roi in roi_set.rois:
        try:
            _require_supported_build_runtime(roi, _build_backend(roi_set, roi))
        except UnsupportedRoiRuntimeError as exc:
            family_errors.append(str(exc))
    if family_errors:
        checks.extend(
            _execution_check("roi_family_supported", "error", message, category="roi_family")
            for message in family_errors
        )
        return _preflight_result(checks, existing_output=existing_output)
    checks.append(
        _execution_check(
            "roi_family_supported",
            "ok",
            "All configured ROI families have a supported local execution path.",
            category="roi_family",
        )
    )

    try:
        plan = plan_roi_build(document, context=context, validate_personal_paths=validate_personal_paths)
    except (ImportError, RuntimeError, ValueError) as exc:
        checks.append(_execution_check(_planning_failure_check_id(exc), "error", str(exc), category="planning"))
        return _preflight_result(checks, existing_output=existing_output)

    outputs = _build_runtime_outputs(plan)
    _append_output_preflight_checks(checks, outputs, existing_output=existing_output)
    _append_nifti_dependency_check(checks, required=bool(plan.actions))
    _append_build_input_checks(checks, plan)
    _append_build_tool_checks(checks, plan, command_finder=command_finder or shutil.which)
    return _preflight_result(
        checks,
        existing_output=existing_output,
        output_paths=tuple(output.destination for output in outputs),
    )


def run_roi_build(
    document: Mapping[str, Any],
    *,
    context: RoiExecutionContext,
    validate_personal_paths: bool = True,
) -> RoiBuildPlan:
    """Execute local ROI mask builds and write sidecars."""

    preflight = preflight_roi_build(
        document,
        context=context,
        validate_personal_paths=validate_personal_paths,
    )
    _require_ready_preflight(preflight, operation="ROI build")
    plan = plan_roi_build(document, context=context, validate_personal_paths=validate_personal_paths)
    executed: list[RoiBuildAction] = []
    roi_set = load_roi_set_for_execution(document, validate_personal_paths=validate_personal_paths)
    loso_cache_state: dict[str, Mapping[str, Any]] = {}
    outputs = _build_runtime_outputs(plan)
    with RoiRuntimeOutputTransaction(outputs, existing_output=preflight.existing_output) as transaction:
        staged_loso_jobs: dict[tuple[str, ...], Mapping[str, Any]] = {}
        staged_actions: list[RoiBuildAction] = []
        for action in plan.actions:
            metadata = dict(action.metadata)
            if action.family == "loso_group_map":
                final_job = _loso_job_metadata(action)
                job_key = _loso_job_staging_key(final_job)
                staged_job = staged_loso_jobs.get(job_key)
                if staged_job is None:
                    staged_job = _stage_loso_job(final_job, transaction=transaction)
                    staged_loso_jobs[job_key] = staged_job
                metadata["loso_group_job"] = staged_job
            staged_actions.append(
                replace(
                    action,
                    mask_path=transaction.candidate_path(action.mask_path),
                    sidecar_path=transaction.candidate_path(action.sidecar_path),
                    metadata=metadata,
                )
            )

        for action, staged_action in zip(plan.actions, staged_actions, strict=True):
            roi = _roi_by_label(roi_set, action.roi_label)
            if action.family == "loso_group_map":
                from research_platform.neuro.roi_loso import execute_loso_build_action

                result = execute_loso_build_action(
                    staged_action,
                    roi,
                    context=context,
                    cache_state=loso_cache_state,
                    provenance_job=_loso_job_metadata(action),
                )
            else:
                result = _execute_build_action(staged_action, roi, context=context)
            executed.append(
                replace(
                    action,
                    result=_build_result_payload(
                        result,
                        mask_path=action.mask_path,
                        sidecar_path=action.sidecar_path,
                    ),
                )
            )
        _validate_staged_build_outputs(plan, transaction)
        transaction.promote()
    executed_actions = tuple(executed)
    cleanup = _publish_loso_roi_build_if_enabled(document, actions=executed_actions, context=context)
    return RoiBuildPlan(roi_set_name=plan.roi_set_name, actions=executed_actions, executed=True, cleanup=cleanup)


def plan_roi_extraction(
    extraction_document: Mapping[str, Any],
    *,
    roi_set_document: Mapping[str, Any] | None = None,
    context: RoiExecutionContext,
    validate_personal_paths: bool = True,
) -> RoiExtractionPlan:
    """Validate and plan local ROI extraction without writing tables.

    Callers that already checked raw config paths can disable personal-path
    validation while planning against env-resolved runtime paths.
    """

    extraction_set = load_extraction_set_for_execution(
        extraction_document,
        validate_personal_paths=validate_personal_paths,
    )
    roi_set = (
        load_roi_set_for_execution(roi_set_document, validate_personal_paths=validate_personal_paths)
        if roi_set_document is not None
        else None
    )
    if extraction_set.roi_set and roi_set is not None and extraction_set.roi_set != roi_set.name:
        raise ValueError(
            f"Extraction set references ROI set {extraction_set.roi_set!r}, "
            f"but loaded ROI set is {roi_set.name!r}."
        )

    actions: list[RoiExtractionAction] = []
    for target in extraction_set.targets:
        actions.extend(_plan_extraction_target(extraction_set, target, roi_set, context=context))
    return RoiExtractionPlan(
        extraction_set_name=extraction_set.name,
        roi_set_name=roi_set.name if roi_set is not None else (extraction_set.roi_set or "<explicit masks>"),
        actions=tuple(actions),
        executed=False,
        tables=_unique_paths(action.table_path for action in actions),
    )


def preflight_roi_extraction(
    extraction_document: Mapping[str, Any],
    *,
    roi_set_document: Mapping[str, Any] | None = None,
    context: RoiExecutionContext,
    validate_personal_paths: bool = True,
    command_finder: CommandFinder | None = None,
) -> RoiExecutionPreflight:
    """Inspect complete ROI extraction readiness without writes or tool execution."""

    checks: list[RoiExecutionCheck] = []
    try:
        existing_output = runtime_existing_output_policy(extraction_document, payload_key="extraction_set")
        extraction_set = load_extraction_set_for_execution(
            extraction_document,
            validate_personal_paths=validate_personal_paths,
        )
        if roi_set_document is not None:
            load_roi_set_for_execution(roi_set_document, validate_personal_paths=validate_personal_paths)
    except (TypeError, ValueError) as exc:
        checks.append(_execution_check("configuration_valid", "error", str(exc), category="configuration"))
        return _preflight_result(checks, existing_output="fail")

    checks.append(
        _execution_check(
            "configuration_valid",
            "ok",
            "ROI extraction configuration is structurally valid.",
            category="configuration",
        )
    )
    unsupported = sorted(
        {
            _extraction_backend(extraction_set, target)
            for target in extraction_set.targets
            if _extraction_backend(extraction_set, target) not in SUPPORTED_EXTRACTION_BACKENDS
        }
    )
    if unsupported:
        checks.append(
            _execution_check(
                "roi_family_supported",
                "error",
                "Unsupported ROI extraction backend(s): " + ", ".join(unsupported) + ".",
                category="backend",
            )
        )
        return _preflight_result(checks, existing_output=existing_output)
    checks.append(
        _execution_check(
            "roi_family_supported",
            "ok",
            "All configured ROI extraction backends have a supported execution path.",
            category="backend",
        )
    )

    try:
        plan = plan_roi_extraction(
            extraction_document,
            roi_set_document=roi_set_document,
            context=context,
            validate_personal_paths=validate_personal_paths,
        )
        outputs = _extraction_runtime_outputs(plan)
    except (ImportError, RuntimeError, ValueError) as exc:
        checks.append(_execution_check(_planning_failure_check_id(exc), "error", str(exc), category="planning"))
        return _preflight_result(checks, existing_output=existing_output)

    _append_output_preflight_checks(checks, outputs, existing_output=existing_output)
    _append_nifti_dependency_check(
        checks,
        required=bool(plan.actions),
    )
    _append_extraction_input_checks(checks, plan)
    _append_extraction_tool_checks(checks, plan, command_finder=command_finder or shutil.which)
    return _preflight_result(
        checks,
        existing_output=existing_output,
        output_paths=tuple(output.destination for output in outputs),
    )


def run_roi_extraction(
    extraction_document: Mapping[str, Any],
    *,
    roi_set_document: Mapping[str, Any] | None = None,
    context: RoiExecutionContext,
    validate_personal_paths: bool = True,
) -> RoiExtractionPlan:
    """Execute local ROI extraction and write configured summary tables."""

    preflight = preflight_roi_extraction(
        extraction_document,
        roi_set_document=roi_set_document,
        context=context,
        validate_personal_paths=validate_personal_paths,
    )
    _require_ready_preflight(preflight, operation="ROI extraction")
    plan = plan_roi_extraction(
        extraction_document,
        roi_set_document=roi_set_document,
        context=context,
        validate_personal_paths=validate_personal_paths,
    )
    rows_by_table: dict[Path, list[dict[str, Any]]] = {}
    executed: list[RoiExtractionAction] = []
    outputs = _extraction_runtime_outputs(plan)
    table_outputs = tuple(output for output in outputs if output.kind == "file")
    with RoiRuntimeOutputTransaction(outputs, existing_output=preflight.existing_output) as transaction:
        featquery_candidates = {
            output.destination: transaction.sibling_candidate_path(output.destination)
            for output in outputs
            if output.kind == "directory"
        }
        for action in plan.actions:
            if action.backend == "fsl_featquery":
                output_dir = Path(str(action.metadata["featquery_output_dir"]))
                row, result = _execute_fsl_featquery_action(
                    action,
                    staged_output_name=featquery_candidates[output_dir].name,
                )
                rows_by_table.setdefault(action.table_path, []).append(row)
                executed.append(replace(action, result=result))
                continue

            from research_platform.neuro.roi_extraction import build_extraction_result, extraction_result_to_row

            value_image = _load_nifti_image(action.value_map_path)
            mask_image = _load_nifti_image(action.mask_path)
            result = build_extraction_result(
                value_image,
                mask_image,
                roi_label=action.roi_label,
                value_desc=_optional_text(action.metadata.get("value_desc")),
                metrics=action.metrics,
                provenance={
                    key: value
                    for key, value in {
                        "target_name": action.target_name,
                        "value_map_path": str(action.value_map_path),
                        "mask_path": str(action.mask_path),
                        **dict(action.metadata),
                    }.items()
                    if value is not None
                },
            )
            row = extraction_result_to_row(result)
            rows_by_table.setdefault(action.table_path, []).append(row)
            executed.append(replace(action, result=dict(result.metrics)))

        written_tables: list[Path] = []
        for table_path, rows in rows_by_table.items():
            candidate = transaction.candidate_path(table_path)
            qc_candidate = transaction.candidate_path(_qc_summary_table_path(table_path))
            staged_tables = _write_extraction_summary_tables(rows, candidate, qc_output_path=qc_candidate)
            expected_candidates = (
                candidate,
                qc_candidate,
            )
            if tuple(staged_tables) != expected_candidates:
                raise RoiRuntimeOutputError("ROI extraction staged an unexpected summary-table destination set.")
            written_tables.extend((table_path, _qc_summary_table_path(table_path)))
        _validate_staged_extraction_outputs(table_outputs, transaction)
        _validate_staged_featquery_outputs(plan, transaction)
        transaction.promote()

    executed_actions = tuple(executed)
    tables = tuple(written_tables)
    cleanup = _publish_loso_featquery_if_enabled(
        extraction_document,
        roi_set_document=roi_set_document,
        actions=executed_actions,
        tables=tables,
        context=context,
    )
    return RoiExtractionPlan(
        extraction_set_name=plan.extraction_set_name,
        roi_set_name=plan.roi_set_name,
        actions=executed_actions,
        executed=True,
        tables=tables,
        cleanup=cleanup,
    )


def _execution_check(
    check_id: str,
    status: str,
    message: str,
    *,
    path: Path | None = None,
    category: str | None = None,
) -> RoiExecutionCheck:
    if status not in {"ok", "warning", "error"}:
        raise ValueError("ROI execution check status must be one of: ok, warning, error.")
    return RoiExecutionCheck(
        check_id=check_id,
        status=status,
        message=str(message),
        path=Path(path) if path is not None else None,
        category=category,
    )


def _preflight_result(
    checks: Sequence[RoiExecutionCheck],
    *,
    existing_output: str,
    output_paths: Sequence[Path] = (),
) -> RoiExecutionPreflight:
    normalized = tuple(checks)
    return RoiExecutionPreflight(
        ready_for_execution=bool(normalized) and not any(check.status == "error" for check in normalized),
        checks=normalized,
        output_paths=tuple(output_paths),
        existing_output=existing_output,
    )


def _require_ready_preflight(preflight: RoiExecutionPreflight, *, operation: str) -> None:
    if preflight.ready_for_execution:
        return
    details = "; ".join(preflight.errors) or "readiness checks did not pass"
    raise RoiRuntimeOutputError(f"{operation} is not ready for execution: {details}")


def _build_runtime_outputs(plan: RoiBuildPlan) -> tuple[RoiRuntimeOutput, ...]:
    outputs: list[RoiRuntimeOutput] = []
    for action in plan.actions:
        outputs.extend(
            (
                RoiRuntimeOutput(action.mask_path, f"ROI mask for {action.roi_label}"),
                RoiRuntimeOutput(action.sidecar_path, f"ROI sidecar for {action.roi_label}"),
            )
        )
    outputs.extend(_loso_runtime_outputs(plan))
    return tuple(outputs)


def _loso_runtime_outputs(plan: RoiBuildPlan) -> tuple[RoiRuntimeOutput, ...]:
    declared: list[RoiRuntimeOutput] = []
    seen_declarations: set[tuple[str, str, Path]] = set()
    for action in plan.actions:
        if action.family != "loso_group_map":
            continue
        job = action.metadata.get("loso_group_job")
        if not isinstance(job, Mapping):
            continue
        producer = str(job.get("cache_key") or "<missing-cache-key>")
        heldout = str(job.get("heldout_subject") or action.roi_label)
        declarations = (
            ("zstat", RoiRuntimeOutput(Path(str(job["zstat_path"])), f"LOSO group zstat for {heldout}")),
            ("sidecar", RoiRuntimeOutput(Path(str(job["sidecar_path"])), f"LOSO group sidecar for {heldout}")),
            (
                "work_dir",
                RoiRuntimeOutput(
                    Path(str(job["work_dir"])),
                    f"LOSO FLAME work directory for {heldout}",
                    kind="directory",
                    required=False,
                ),
            ),
        )
        for slot, output in declarations:
            declaration_key = (producer, slot, output.destination)
            if declaration_key not in seen_declarations:
                declared.append(output)
                seen_declarations.add(declaration_key)
        generated = job.get("generated_group_mask")
        if isinstance(generated, Mapping):
            generated_outputs = (
                (
                    "generated_mask",
                    RoiRuntimeOutput(
                        Path(str(generated["mask_path"])),
                        f"LOSO generated group mask for {heldout}",
                    ),
                ),
                (
                    "generated_sidecar",
                    RoiRuntimeOutput(
                        Path(str(generated["sidecar_path"])),
                        f"LOSO generated group-mask sidecar for {heldout}",
                    ),
                ),
            )
            for slot, output in generated_outputs:
                declaration_key = (producer, slot, output.destination)
                if declaration_key not in seen_declarations:
                    declared.append(output)
                    seen_declarations.add(declaration_key)
    return tuple(sorted(declared, key=lambda output: (str(output.destination), output.category)))


def _loso_job_metadata(action: RoiBuildAction) -> Mapping[str, Any]:
    job = action.metadata.get("loso_group_job")
    if not isinstance(job, Mapping):
        raise RoiRuntimeOutputError(f"LOSO build action {action.roi_label} is missing group-map job metadata.")
    return job


def _loso_job_staging_key(job: Mapping[str, Any]) -> tuple[str, ...]:
    generated = job.get("generated_group_mask")
    generated_mask = generated if isinstance(generated, Mapping) else {}
    return tuple(
        str(value)
        for value in (
            job.get("cache_key"),
            job.get("zstat_path"),
            job.get("sidecar_path"),
            job.get("work_dir"),
            job.get("group_mask_path"),
            generated_mask.get("mask_path"),
            generated_mask.get("sidecar_path"),
        )
    )


def _stage_loso_job(
    job: Mapping[str, Any],
    *,
    transaction: RoiRuntimeOutputTransaction,
) -> Mapping[str, Any]:
    runtime_job = dict(job)
    final_zstat = Path(str(job["zstat_path"]))
    final_sidecar = Path(str(job["sidecar_path"]))
    final_work_dir = Path(str(job["work_dir"]))
    runtime_job["zstat_path"] = str(transaction.candidate_path(final_zstat))
    runtime_job["sidecar_path"] = str(transaction.candidate_path(final_sidecar))
    runtime_job["work_dir"] = str(transaction.candidate_path(final_work_dir))

    final_generated = job.get("generated_group_mask")
    generated_pairs: list[tuple[Path, Path]] = []
    if isinstance(final_generated, Mapping):
        runtime_generated = dict(final_generated)
        final_mask = Path(str(final_generated["mask_path"]))
        final_mask_sidecar = Path(str(final_generated["sidecar_path"]))
        staged_mask = transaction.candidate_path(final_mask)
        staged_mask_sidecar = transaction.candidate_path(final_mask_sidecar)
        runtime_generated["mask_path"] = str(staged_mask)
        runtime_generated["sidecar_path"] = str(staged_mask_sidecar)
        runtime_job["generated_group_mask"] = runtime_generated
        runtime_job["group_mask_path"] = str(staged_mask)
        generated_pairs.extend(((final_mask, staged_mask), (final_mask_sidecar, staged_mask_sidecar)))

    cache_pairs = [
        (final_zstat, Path(str(runtime_job["zstat_path"]))),
        (final_sidecar, Path(str(runtime_job["sidecar_path"]))),
        *generated_pairs,
    ]
    if bool(job.get("cache_reuse")) and all(
        source.is_file() and not source.is_symlink() for source, _candidate in cache_pairs
    ):
        for source, candidate in cache_pairs:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, candidate)
    return runtime_job


def _extraction_runtime_outputs(plan: RoiExtractionPlan) -> tuple[RoiRuntimeOutput, ...]:
    producers: dict[Path, set[str]] = {}
    for action in plan.actions:
        producers.setdefault(action.table_path, set()).add(action.target_name)
    ambiguous = {path: names for path, names in producers.items() if len(names) > 1}
    if ambiguous:
        path, names = sorted(ambiguous.items(), key=lambda item: str(item[0]))[0]
        raise RoiRuntimeOutputError(
            f"ROI extraction targets {', '.join(sorted(names))} produce the same summary-table destination: {path}."
        )

    outputs: list[RoiRuntimeOutput] = []
    for table_path, names in sorted(producers.items(), key=lambda item: str(item[0])):
        target = next(iter(names))
        outputs.extend(
            (
                RoiRuntimeOutput(table_path, f"ROI extraction values table for {target}"),
                RoiRuntimeOutput(_qc_summary_table_path(table_path), f"ROI extraction QC table for {target}"),
            )
        )

    featquery_directories: dict[Path, list[str]] = {}
    for action in plan.actions:
        if action.backend != "fsl_featquery":
            continue
        raw = action.metadata.get("featquery_output_dir")
        if raw is None:
            continue
        path = Path(str(raw))
        featquery_directories.setdefault(path, []).append(f"{action.target_name}/{action.roi_label}")
    for path, labels in sorted(featquery_directories.items(), key=lambda item: str(item[0])):
        if len(labels) > 1:
            raise RoiRuntimeOutputError(
                "ROI extraction produced duplicate featquery output-directory destinations: " + str(path) + "."
            )
        outputs.append(RoiRuntimeOutput(path, f"featquery output directory for {labels[0]}", kind="directory"))
    return tuple(outputs)


def _append_output_preflight_checks(
    checks: list[RoiExecutionCheck],
    outputs: Sequence[RoiRuntimeOutput],
    *,
    existing_output: str,
) -> None:
    if not outputs:
        checks.append(
            _execution_check(
                "output_collision",
                "error",
                "Execution did not plan any runtime outputs; supply executable inputs and output configuration.",
                category="output",
            )
        )
        return
    try:
        preflight_runtime_outputs(outputs, existing_output=existing_output)
    except RoiRuntimeOutputError as exc:
        check_id = _runtime_output_failure_check_id(exc)
        checks.append(_execution_check(check_id, "error", str(exc), category="output"))
        return
    checks.extend(
        (
            _execution_check(
                "configured_root_available",
                "ok",
                "Every runtime destination has an available local staging filesystem.",
                category="output",
            ),
            _execution_check(
                "output_collision",
                "ok",
                "The complete planned runtime output set has no unsafe collision.",
                category="output",
            ),
        )
    )


def _runtime_output_failure_check_id(error: RoiRuntimeOutputError) -> str:
    message = str(error).casefold()
    root_failure_markers = (
        "absolute resolved path",
        "destination parent",
        "staging filesystem",
    )
    if any(marker in message for marker in root_failure_markers):
        return "configured_root_available"
    return "output_collision"


def _planning_failure_check_id(error: Exception) -> str:
    message = str(error).casefold()
    if "root_ref" in message or "root ref" in message or "named root" in message:
        return "configured_root_available"
    return "configuration_valid"


def _append_nifti_dependency_check(checks: list[RoiExecutionCheck], *, required: bool) -> None:
    if not required:
        checks.append(
            _execution_check(
                "python_dependency_available",
                "ok",
                "The configured backend does not require the local NIfTI Python runtime.",
                category="python_dependency",
            )
        )
        return
    try:
        from research_platform.neuro import nifti

        available = nifti.nib is not None
    except ImportError:
        available = False
    checks.append(
        _execution_check(
            "python_dependency_available",
            "ok" if available else "error",
            (
                "The local NIfTI Python runtime is available."
                if available
                else "Local ROI execution requires numpy and nibabel; install an explicit profile that supplies them."
            ),
            category="python_dependency",
        )
    )


def _append_build_tool_checks(
    checks: list[RoiExecutionCheck],
    plan: RoiBuildPlan,
    *,
    command_finder: CommandFinder,
) -> None:
    if not any(action.backend == "fsl_flame1" for action in plan.actions):
        checks.append(
            _execution_check(
                "external_tool_available",
                "ok",
                "The configured ROI build does not require an external command-line tool.",
                category="external_tool",
            )
        )
        return
    configured_paths = {
        _configured_tool_path(action.metadata.get("loso_group_job"))
        for action in plan.actions
        if action.backend == "fsl_flame1"
    }
    for tool in ("fslmerge", "flameo"):
        for configured_path in sorted(configured_paths, key=lambda value: value or ""):
            resolved = _find_configured_tool(command_finder, tool, configured_path)
            checks.append(
                _execution_check(
                    "external_tool_available",
                    "ok" if resolved else "error",
                    (
                        f"Required external tool {tool} is available."
                        if resolved
                        else f"Required external tool {tool} was not found; load or install FSL before execution."
                    ),
                    path=Path(resolved) if resolved else None,
                    category=tool,
                )
            )


def _append_extraction_tool_checks(
    checks: list[RoiExecutionCheck],
    plan: RoiExtractionPlan,
    *,
    command_finder: CommandFinder,
) -> None:
    if not any(action.backend == "fsl_featquery" for action in plan.actions):
        checks.append(
            _execution_check(
                "external_tool_available",
                "ok",
                "The configured ROI extraction does not require an external command-line tool.",
                category="external_tool",
            )
        )
        return
    configured_paths = {
        _configured_tool_path(action.metadata.get("command_plan"))
        for action in plan.actions
        if action.backend == "fsl_featquery"
    }
    for configured_path in sorted(configured_paths, key=lambda value: value or ""):
        resolved = _find_configured_tool(command_finder, "featquery", configured_path)
        checks.append(
            _execution_check(
                "external_tool_available",
                "ok" if resolved else "error",
                (
                    "Required external tool featquery is available."
                    if resolved
                    else "Required external tool featquery was not found; load or install FSL before execution."
                ),
                path=Path(resolved) if resolved else None,
                category="featquery",
            )
        )


def _configured_tool_path(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    backend = payload.get("backend_config") if isinstance(payload.get("backend_config"), Mapping) else payload
    environment = backend.get("environment") if isinstance(backend, Mapping) else None
    if not isinstance(environment, Mapping):
        return None
    value = environment.get("PATH")
    return str(value) if value is not None and str(value).strip() else None


def _find_configured_tool(command_finder: CommandFinder, tool: str, configured_path: str | None) -> str | None:
    if command_finder is shutil.which and configured_path is not None:
        return shutil.which(tool, path=configured_path)
    return command_finder(tool)


def _append_build_input_checks(checks: list[RoiExecutionCheck], plan: RoiBuildPlan) -> None:
    image_cache: dict[Path, Any] = {}
    for action in plan.actions:
        if action.family == "loso_group_map":
            _append_loso_input_checks(checks, action, image_cache=image_cache)
            continue

        loaded: dict[str, Any] = {}
        for name, path in action.input_paths.items():
            if not _append_input_path_check(checks, path, category=name, expect_directory=False):
                continue
            image = _load_preflight_image(checks, path, category=name, cache=image_cache)
            if image is not None:
                loaded[name] = image
        _append_generic_build_geometry_checks(checks, action, loaded)


def _append_extraction_input_checks(checks: list[RoiExecutionCheck], plan: RoiExtractionPlan) -> None:
    image_cache: dict[Path, Any] = {}
    for action in plan.actions:
        if action.backend == "fsl_featquery":
            feat_dir = action.metadata.get("feat_dir")
            if feat_dir is not None:
                _append_input_path_check(
                    checks,
                    Path(str(feat_dir)),
                    category="feat_dir",
                    expect_directory=True,
                )
            value_path = _existing_nifti_path(action.value_map_path)
            value_exists = _append_input_path_check(
                checks,
                value_path,
                category="value_image",
                expect_directory=False,
            )
            mask_exists = _append_input_path_check(
                checks,
                action.mask_path,
                category="roi_mask",
                expect_directory=False,
            )
            value_image = (
                _load_preflight_image(checks, value_path, category="value_image", cache=image_cache)
                if value_exists
                else None
            )
            mask_image = (
                _load_preflight_image(checks, action.mask_path, category="roi_mask", cache=image_cache)
                if mask_exists
                else None
            )
            if value_image is not None:
                _append_3d_image_check(checks, value_image, path=value_path, category="value_image")
            if mask_image is not None:
                _append_binary_mask_check(checks, mask_image, path=action.mask_path, category="roi_mask")
            if value_image is not None and mask_image is not None:
                _append_geometry_check(
                    checks,
                    value_image,
                    mask_image,
                    path=action.mask_path,
                    category="value_image_to_roi_mask",
                )
            continue

        value_exists = _append_input_path_check(
            checks,
            action.value_map_path,
            category="value_image",
            expect_directory=False,
        )
        mask_exists = _append_input_path_check(
            checks,
            action.mask_path,
            category="roi_mask",
            expect_directory=False,
        )
        value_image = (
            _load_preflight_image(checks, action.value_map_path, category="value_image", cache=image_cache)
            if value_exists
            else None
        )
        mask_image = (
            _load_preflight_image(checks, action.mask_path, category="roi_mask", cache=image_cache)
            if mask_exists
            else None
        )
        if mask_image is not None:
            _append_binary_mask_check(checks, mask_image, path=action.mask_path, category="roi_mask")
        if value_image is not None:
            _append_3d_image_check(checks, value_image, path=action.value_map_path, category="value_image")
        if value_image is not None and mask_image is not None:
            _append_geometry_check(
                checks,
                value_image,
                mask_image,
                path=action.mask_path,
                category="value_image_to_roi_mask",
            )


def _append_loso_input_checks(
    checks: list[RoiExecutionCheck],
    action: RoiBuildAction,
    *,
    image_cache: dict[Path, Any],
) -> None:
    job = action.metadata.get("loso_group_job")
    generated_group_mask = isinstance(job, Mapping) and isinstance(job.get("generated_group_mask"), Mapping)
    loaded: list[tuple[str, Path, Any]] = []
    for name, path in action.input_paths.items():
        if name == "loso_zstat" or (name == "group_mask" and generated_group_mask):
            continue
        if not _append_input_path_check(checks, path, category=name, expect_directory=False):
            continue
        image = _load_preflight_image(checks, path, category=name, cache=image_cache)
        if image is None:
            continue
        loaded.append((name, path, image))
        if "mask" in name:
            _append_binary_mask_check(checks, image, path=path, category=name)
        else:
            _append_3d_image_check(checks, image, path=path, category=name)

    reference = next((item for item in loaded if item[0].startswith("cope:")), None)
    if reference is None and loaded:
        reference = loaded[0]
    if reference is None:
        return
    reference_name, _reference_path, reference_image = reference
    for name, path, image in loaded:
        if name == reference_name and path == reference[1]:
            continue
        _append_geometry_check(
            checks,
            reference_image,
            image,
            path=path,
            category=f"{reference_name}_to_{name}",
        )


def _existing_nifti_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    for suffix in (".nii.gz", ".nii"):
        suffixed = Path(f"{candidate}{suffix}")
        if suffixed.is_file():
            return suffixed
    return candidate


def _append_input_path_check(
    checks: list[RoiExecutionCheck],
    path: Path,
    *,
    category: str,
    expect_directory: bool,
) -> bool:
    resolved = Path(path)
    exists = resolved.is_dir() if expect_directory else resolved.is_file()
    kind = "directory" if expect_directory else "file"
    checks.append(
        _execution_check(
            "input_exists",
            "ok" if exists else "error",
            (
                f"Required {category} {kind} exists."
                if exists
                else f"Required {category} {kind} is missing or has the wrong type: {resolved}."
            ),
            path=resolved,
            category=category,
        )
    )
    return exists


def _load_preflight_image(
    checks: list[RoiExecutionCheck],
    path: Path,
    *,
    category: str,
    cache: dict[Path, Any],
) -> Any | None:
    resolved = Path(path)
    if resolved in cache:
        image = cache[resolved]
        checks.append(
            _execution_check(
                "image_readable",
                "ok",
                f"Required {category} image is readable.",
                path=resolved,
                category=category,
            )
        )
        return image
    try:
        image = _load_nifti_image(resolved)
        # Force header/data access now rather than after the first output write.
        image.get_fdata()
    except Exception as exc:
        checks.append(
            _execution_check(
                "image_readable",
                "error",
                f"Required {category} image cannot be read: {resolved} ({type(exc).__name__}).",
                path=resolved,
                category=category,
            )
        )
        return None
    cache[resolved] = image
    checks.append(
        _execution_check(
            "image_readable",
            "ok",
            f"Required {category} image is readable.",
            path=resolved,
            category=category,
        )
    )
    return image


def _append_generic_build_geometry_checks(
    checks: list[RoiExecutionCheck],
    action: RoiBuildAction,
    images: Mapping[str, Any],
) -> None:
    reference_name: str | None = None
    if action.family == "coordinate_sphere":
        reference_name = "reference_image"
    elif action.family == "manual_mask":
        reference_name = "reference_image" if "reference_image" in images else "source_mask"
        if "source_mask" in images:
            _append_binary_mask_check(
                checks,
                images["source_mask"],
                path=action.input_paths["source_mask"],
                category="source_mask",
            )
        if "reference_image" in images and "source_mask" in images:
            _append_geometry_check(
                checks,
                images["reference_image"],
                images["source_mask"],
                path=action.input_paths["source_mask"],
                category="reference_to_source_mask",
            )
    elif action.family == "functional_threshold_map":
        reference_name = "stat_map"
        if "search_mask" in images and "stat_map" in images:
            _append_binary_mask_check(
                checks,
                images["search_mask"],
                path=action.input_paths["search_mask"],
                category="search_mask",
            )
            _append_geometry_check(
                checks,
                images["stat_map"],
                images["search_mask"],
                path=action.input_paths["search_mask"],
                category="stat_map_to_search_mask",
            )

    if reference_name is None or reference_name not in images:
        return
    for name, image in images.items():
        if not name.startswith("coverage_mask:"):
            continue
        _append_binary_mask_check(checks, image, path=action.input_paths[name], category=name)
        _append_geometry_check(
            checks,
            images[reference_name],
            image,
            path=action.input_paths[name],
            category=f"{reference_name}_to_{name}",
        )


def _append_binary_mask_check(
    checks: list[RoiExecutionCheck],
    image: Any,
    *,
    path: Path,
    category: str,
) -> None:
    try:
        from research_platform.neuro.roi_masks import validate_binary_mask

        validate_binary_mask(image.get_fdata(), allow_empty=True, label=category)
    except Exception as exc:
        checks.append(
            _execution_check(
                "image_geometry_compatible",
                "error",
                f"Required {category} is not a readable 3D binary mask: {path} ({type(exc).__name__}).",
                path=path,
                category=category,
            )
        )
        return
    checks.append(
        _execution_check(
            "image_geometry_compatible",
            "ok",
            f"Required {category} is a 3D binary mask.",
            path=path,
            category=category,
        )
    )


def _append_3d_image_check(
    checks: list[RoiExecutionCheck],
    image: Any,
    *,
    path: Path,
    category: str,
) -> None:
    shape = tuple(int(value) for value in image.shape)
    if len(shape) != 3:
        checks.append(
            _execution_check(
                "image_geometry_compatible",
                "error",
                f"Required {category} must be a 3D image: {path}.",
                path=path,
                category=category,
            )
        )
        return
    checks.append(
        _execution_check(
            "image_geometry_compatible",
            "ok",
            f"Required {category} is a 3D image.",
            path=path,
            category=category,
        )
    )


def _append_geometry_check(
    checks: list[RoiExecutionCheck],
    reference_image: Any,
    other_image: Any,
    *,
    path: Path,
    category: str,
) -> None:
    try:
        from research_platform.neuro.nifti import validate_compatible_geometry

        validate_compatible_geometry(reference_image, other_image)
    except Exception as exc:
        checks.append(
            _execution_check(
                "image_geometry_compatible",
                "error",
                f"Required images have incompatible geometry for {category}: {path} ({type(exc).__name__}).",
                path=path,
                category=category,
            )
        )
        return
    checks.append(
        _execution_check(
            "image_geometry_compatible",
            "ok",
            f"Required images have compatible geometry for {category}.",
            path=path,
            category=category,
        )
    )


def _validate_staged_build_outputs(plan: RoiBuildPlan, transaction: RoiRuntimeOutputTransaction) -> None:
    from research_platform.neuro.roi_masks import validate_binary_mask

    for action in plan.actions:
        mask = transaction.candidate_path(action.mask_path)
        sidecar = transaction.candidate_path(action.sidecar_path)
        try:
            image = _load_nifti_image(mask)
            validate_binary_mask(image.get_fdata(), allow_empty=True, label="staged ROI mask")
            reference_path = _build_action_reference_path(action, transaction=transaction)
            if reference_path is not None:
                from research_platform.neuro.nifti import validate_compatible_geometry

                validate_compatible_geometry(_load_nifti_image(reference_path), image)
        except Exception as exc:
            raise RoiRuntimeOutputError(
                f"ROI build staged an unreadable or invalid mask for {action.roi_label}."
            ) from exc
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RoiRuntimeOutputError(f"ROI build staged an invalid JSON sidecar for {action.roi_label}.") from exc
        errors = validate_roi_sidecar_document(payload)
        if errors:
            raise RoiRuntimeOutputError(f"ROI build staged an invalid ROI sidecar for {action.roi_label}: {'; '.join(errors)}")

    validated_jobs: set[tuple[Path, Path]] = set()
    for action in plan.actions:
        if action.family != "loso_group_map":
            continue
        job = action.metadata.get("loso_group_job")
        if not isinstance(job, Mapping):
            raise RoiRuntimeOutputError(f"LOSO build action {action.roi_label} is missing group-map job metadata.")
        zstat_path = Path(str(job["zstat_path"]))
        group_sidecar_path = Path(str(job["sidecar_path"]))
        job_key = (zstat_path, group_sidecar_path)
        if job_key in validated_jobs:
            continue
        validated_jobs.add(job_key)
        staged_zstat_path = transaction.candidate_path(zstat_path)
        staged_group_sidecar_path = transaction.candidate_path(group_sidecar_path)
        try:
            zstat_image = _load_nifti_image(staged_zstat_path)
            zstat_image.get_fdata()
            if len(tuple(zstat_image.shape)) != 3:
                raise ValueError("LOSO zstat must be 3D")
        except Exception as exc:
            raise RoiRuntimeOutputError("ROI build produced an unreadable or non-3D LOSO group zstat.") from exc
        _validate_runtime_json_sidecar(staged_group_sidecar_path, label="LOSO group-map sidecar", portable=True)

        generated = job.get("generated_group_mask")
        group_mask_path = (
            transaction.candidate_path(Path(str(generated["mask_path"])))
            if isinstance(generated, Mapping)
            else Path(str(job["group_mask_path"]))
        )
        try:
            group_mask_image = _load_nifti_image(group_mask_path)
            validate_binary_mask(group_mask_image.get_fdata(), allow_empty=True, label="LOSO group mask")
            from research_platform.neuro.nifti import validate_compatible_geometry

            validate_compatible_geometry(zstat_image, group_mask_image)
        except Exception as exc:
            raise RoiRuntimeOutputError("ROI build produced or used an invalid LOSO group mask.") from exc
        if isinstance(generated, Mapping):
            _validate_runtime_json_sidecar(
                transaction.candidate_path(Path(str(generated["sidecar_path"]))),
                label="LOSO generated group-mask sidecar",
                portable=True,
            )


def _build_action_reference_path(
    action: RoiBuildAction,
    *,
    transaction: RoiRuntimeOutputTransaction | None = None,
) -> Path | None:
    if action.family == "loso_group_map":
        job = action.metadata.get("loso_group_job")
        if not isinstance(job, Mapping) or not job.get("zstat_path"):
            return None
        path = Path(str(job["zstat_path"]))
        return transaction.candidate_path(path) if transaction is not None else path
    for name in ("reference_image", "source_mask", "stat_map"):
        if name in action.input_paths:
            return action.input_paths[name]
    return None


def _validate_staged_extraction_outputs(
    outputs: Sequence[RoiRuntimeOutput],
    transaction: RoiRuntimeOutputTransaction,
) -> None:
    for output in outputs:
        path = transaction.candidate_path(output.destination)
        delimiter = "\t" if output.destination.suffix.lower() == ".tsv" else ","
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                if not reader.fieldnames:
                    raise ValueError("table header is empty")
                list(reader)
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            raise RoiRuntimeOutputError(f"ROI extraction staged an invalid {output.category}.") from exc


def _validate_staged_featquery_outputs(
    plan: RoiExtractionPlan,
    transaction: RoiRuntimeOutputTransaction,
) -> None:
    from research_platform.neuro.fsl.featquery import parse_featquery_report

    for action in plan.actions:
        if action.backend != "fsl_featquery":
            continue
        final_output_dir = Path(str(action.metadata["featquery_output_dir"]))
        staged_output_dir = transaction.candidate_path(final_output_dir)
        report_name = Path(str(action.metadata["report_path"])).name
        staged_report = staged_output_dir / report_name
        try:
            if staged_report.is_symlink() or not staged_report.is_file():
                raise ValueError("featquery report is missing or is not a regular file")
            parse_featquery_report(
                staged_report,
                required_metrics=_featquery_report_required_metrics(action.metrics, action.metadata),
            )
        except Exception as exc:
            raise RoiRuntimeOutputError(
                f"ROI extraction staged an unreadable featquery report for {action.target_name}/{action.roi_label}."
            ) from exc


def _validate_runtime_json_sidecar(path: Path, *, label: str, portable: bool) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RoiRuntimeOutputError(f"ROI build produced an invalid {label}.") from exc
    if not isinstance(payload, Mapping):
        raise RoiRuntimeOutputError(f"ROI build produced an invalid {label}.")
    if portable:
        from research_platform.neuro.roi import validate_portable_provenance_paths

        errors = validate_portable_provenance_paths(payload, label=label)
        if errors:
            raise RoiRuntimeOutputError(f"ROI build produced an invalid {label}: {'; '.join(errors)}")


def _publish_loso_roi_build_if_enabled(
    document: Mapping[str, Any],
    *,
    actions: Sequence[RoiBuildAction],
    context: RoiExecutionContext,
) -> tuple[Mapping[str, Any], ...]:
    if not any(action.family == "loso_group_map" for action in actions):
        return ()
    from research_platform.neuro.roi_cleanup import cleanup_after_loso_roi_build
    from research_platform.neuro.roi_publication import publish_loso_roi_build_result

    publication = publish_loso_roi_build_result(document, actions=actions, context=context)
    return cleanup_after_loso_roi_build(
        document,
        context=context,
        publication_complete=publication.complete,
        publication_root=publication.root,
    )


def _publish_loso_featquery_if_enabled(
    extraction_document: Mapping[str, Any],
    *,
    roi_set_document: Mapping[str, Any] | None,
    actions: Sequence[RoiExtractionAction],
    tables: Sequence[Path],
    context: RoiExecutionContext,
) -> tuple[Mapping[str, Any], ...]:
    if not any(action.backend == "fsl_featquery" for action in actions):
        return ()
    from research_platform.neuro.roi_cleanup import cleanup_after_loso_featquery_extraction
    from research_platform.neuro.roi_publication import publish_loso_featquery_extraction_result

    publication = publish_loso_featquery_extraction_result(
        extraction_document,
        roi_set_document=roi_set_document,
        actions=actions,
        tables=tables,
        context=context,
    )
    return cleanup_after_loso_featquery_extraction(
        extraction_document,
        roi_set_document=roi_set_document,
        context=context,
        publication_complete=publication.complete,
        publication_root=publication.root,
    )


def _plan_build_action(roi_set: RoiSet, roi: RoiDefinition, *, context: RoiExecutionContext) -> RoiBuildAction:
    backend = _build_backend(roi_set, roi)
    _require_supported_build_runtime(roi, backend)
    fields = dict(roi.fields)
    entities = _entities(_roi_set_common_fields(roi_set), fields)
    method_desc = _method_desc(roi_set, roi)
    output_root, pipeline_name = _resolve_output_root(
        fields,
        _roi_set_common_fields(roi_set),
        context=context,
        default_root=_default_roi_derivative_root(context),
    )
    mask_path = _build_roi_mask_path(
        output_root,
        entities=entities,
        roi_label=roi.label,
        method_desc=method_desc,
        pipeline_name=pipeline_name,
    )
    sidecar_path = _build_roi_sidecar_path(mask_path)

    input_paths: dict[str, Path] = {}
    if roi.family == "coordinate_sphere":
        input_paths["reference_image"] = _resolve_required_path(fields, "reference_image", context=context, entities=entities)
    elif roi.family == "manual_mask":
        input_paths["source_mask"] = _resolve_required_path(
            fields,
            "source",
            "source_mask",
            "mask",
            context=context,
            entities=entities,
        )
        reference = _resolve_optional_path(fields, "reference_image", context=context, entities=entities)
        if reference is not None:
            input_paths["reference_image"] = reference
    elif roi.family == "functional_threshold_map":
        input_paths["stat_map"] = _resolve_required_path(
            fields,
            "stat_map",
            "source",
            context=context,
            entities=entities,
        )
        search_mask = _resolve_optional_path(fields, "search_mask", "search_mask_image", context=context, entities=entities)
        if search_mask is not None:
            input_paths["search_mask"] = search_mask

    for name, path in _coverage_mask_paths(fields, context=context, entities=entities).items():
        input_paths[f"coverage_mask:{name}"] = path

    return RoiBuildAction(
        roi_label=roi.label,
        family=roi.family,
        backend=backend,
        mask_path=mask_path,
        sidecar_path=sidecar_path,
        input_paths=input_paths,
        metadata={
            "desc": method_desc,
            "entities": entities,
        },
    )


def _plan_loso_build_actions(
    roi_set: RoiSet,
    rois: Sequence[RoiDefinition],
    *,
    context: RoiExecutionContext,
) -> tuple[RoiBuildAction, ...]:
    for roi in rois:
        _require_supported_build_runtime(roi, _build_backend(roi_set, roi))
    from research_platform.neuro.roi_loso import plan_loso_group_map_build_actions

    return tuple(plan_loso_group_map_build_actions(roi_set, rois, context=context))


def _execute_build_action(action: RoiBuildAction, roi: RoiDefinition, *, context: RoiExecutionContext) -> Any:
    fields = dict(roi.fields)
    coverage_masks = {
        name.split(":", 1)[1]: _load_nifti_image(path)
        for name, path in action.input_paths.items()
        if name.startswith("coverage_mask:")
    }
    provenance = {
        "roi_set": action.metadata.get("entities", {}).get("roi_set"),
        "project": context.project_name,
    }
    provenance = {key: value for key, value in provenance.items() if value is not None}

    if action.family == "coordinate_sphere":
        from research_platform.neuro.roi_builders import build_coordinate_sphere_roi

        return build_coordinate_sphere_roi(
            reference_image=_load_nifti_image(action.input_paths["reference_image"]),
            center_xyz_mm=_number_sequence(_required_field(fields, "coordinate", "center_xyz_mm"), label="coordinate"),
            radius_mm=_number_value(_required_field(fields, "radius_mm", "radius"), label="radius_mm"),
            roi_label=action.roi_label,
            desc=_optional_text(action.metadata.get("desc")),
            output_mask_path=action.mask_path,
            sidecar_path=action.sidecar_path,
            coverage_masks=coverage_masks,
            min_voxels_warn=_optional_int(fields.get("min_voxels_warn")),
            min_voxels_fail=_optional_int(fields.get("min_voxels_fail")),
            fail_on_qc=_bool_value(fields.get("fail_on_qc"), default=True),
            provenance=provenance,
        )

    if action.family == "manual_mask":
        from research_platform.neuro.roi_builders import copy_manual_mask_roi

        return copy_manual_mask_roi(
            source_mask_path=action.input_paths["source_mask"],
            output_mask_path=action.mask_path,
            roi_label=action.roi_label,
            reference_image=_load_nifti_image(action.input_paths["reference_image"])
            if "reference_image" in action.input_paths
            else None,
            sidecar_path=action.sidecar_path,
            desc=_optional_text(action.metadata.get("desc")),
            source_ref=_source_ref(fields, "source", "source_mask", "mask"),
            coverage_masks=coverage_masks,
            min_voxels_warn=_optional_int(fields.get("min_voxels_warn")),
            min_voxels_fail=_optional_int(fields.get("min_voxels_fail")),
            fail_on_qc=_bool_value(fields.get("fail_on_qc"), default=True),
            provenance=provenance,
        )

    if action.family == "functional_threshold_map":
        from research_platform.neuro.roi_builders import build_functional_threshold_map_roi

        return build_functional_threshold_map_roi(
            stat_image=_load_nifti_image(action.input_paths["stat_map"]),
            roi_label=action.roi_label,
            desc=_optional_text(action.metadata.get("desc")),
            output_mask_path=action.mask_path,
            sidecar_path=action.sidecar_path,
            sphere_radius_mm=_number_value(_required_field(fields, "sphere_radius_mm", "radius_mm"), label="sphere_radius_mm"),
            threshold=_number_value(_required_field(fields, "z_threshold", "threshold"), label="z_threshold"),
            search_mask_image=_load_nifti_image(action.input_paths["search_mask"]) if "search_mask" in action.input_paths else None,
            seed_xyz_mm=_optional_number_sequence(fields.get("seed_coordinate"), label="seed_coordinate"),
            search_radius_mm=_optional_number(fields.get("search_radius_mm")),
            allow_below_threshold_fallback=_bool_value(fields.get("allow_below_threshold_fallback"), default=False),
            coverage_masks=coverage_masks,
            min_voxels_warn=_optional_int(fields.get("min_voxels_warn")),
            min_voxels_fail=_optional_int(fields.get("min_voxels_fail")),
            fail_on_qc=_bool_value(fields.get("fail_on_qc"), default=True),
            provenance=provenance,
        )

    raise UnsupportedRoiRuntimeError(f"ROI family {action.family!r} is not executable in local ROI execution.")


def _plan_extraction_target(
    extraction_set: ExtractionSet,
    target: Any,
    roi_set: RoiSet | None,
    *,
    context: RoiExecutionContext,
) -> list[RoiExtractionAction]:
    fields = dict(target.fields)
    common = _extraction_set_common_fields(extraction_set)
    backend = _extraction_backend(extraction_set, target)
    if backend not in SUPPORTED_EXTRACTION_BACKENDS:
        raise UnsupportedRoiRuntimeError(
            f"ROI extraction backend {backend!r} is schema-only/deferred in local ROI execution; "
            "supported backends: generic_nifti, fsl_featquery."
        )
    if backend == "fsl_featquery":
        return _plan_fsl_featquery_target(extraction_set, target, roi_set, context=context)
    if roi_set is None:
        raise ValueError("generic_nifti extraction requires roi_set or roi_set_ref.")

    entities = _entities(_roi_set_common_fields(roi_set), common, fields)
    metrics = tuple(_string_list(fields.get("metrics") or common.get("metrics") or DEFAULT_EXTRACTION_METRICS, label="metrics"))
    _validate_generic_extraction_metrics(metrics)
    value_specs = _value_map_specs(fields)
    mask_specs = _roi_mask_specs(fields, roi_set, context=context, entities=entities)
    output_root, pipeline_name = _resolve_output_root(
        fields,
        common,
        context=context,
        default_root=_default_roi_derivative_root(context),
    )
    table_path = _build_roi_extraction_table_path(
        output_root,
        entities=entities,
        extraction_desc=_table_desc(target),
        pipeline_name=pipeline_name,
        extension=_table_extension(fields, common),
    )

    actions: list[RoiExtractionAction] = []
    for value_index, value_spec in enumerate(value_specs):
        value_map_path = _resolve_path_spec(
            value_spec,
            context=context,
            entities={**entities, "target": target.name, "target_name": target.name, "value_index": value_index},
            label=f"extraction target {target.name} inputs",
        )
        value_desc = _optional_text(_mapping_value(value_spec).get("desc")) or target.desc or target.name
        for roi_label, mask_path in mask_specs:
            actions.append(
                RoiExtractionAction(
                    target_name=target.name,
                    backend=backend,
                    roi_label=roi_label,
                    value_map_path=value_map_path,
                    mask_path=mask_path,
                    table_path=table_path,
                    metrics=metrics,
                    metadata={
                        "subject_id": entities.get("subject_id"),
                        "session_id": entities.get("session_id"),
                        "task_id": entities.get("task_id"),
                        "space": entities.get("space"),
                        "value_desc": value_desc,
                    },
                )
            )
    return actions


def _validate_generic_extraction_metrics(metrics: Sequence[str]) -> None:
    unknown = sorted(set(metrics) - set(DEFAULT_EXTRACTION_METRICS))
    if unknown:
        raise ValueError(f"Unsupported generic_nifti extraction metric(s): {', '.join(unknown)}.")


def _plan_fsl_featquery_target(
    extraction_set: ExtractionSet,
    target: Any,
    roi_set: RoiSet | None,
    *,
    context: RoiExecutionContext,
) -> list[RoiExtractionAction]:
    from research_platform.neuro.fsl.featquery import build_featquery_command_plan

    fields = dict(target.fields)
    common = _extraction_set_common_fields(extraction_set)
    merged = _merge_nested(common, fields)
    metrics = tuple(_string_list(merged.get("metrics") or DEFAULT_FEATQUERY_METRICS, label="metrics"))
    include_percent_signal_change = _featquery_include_percent_signal_change(merged)
    _validate_featquery_metrics(metrics, include_percent_signal_change=include_percent_signal_change)
    feat_dir_specs = _feat_dir_specs(merged)
    output_name_template = _featquery_output_name_template(merged)
    value_image_template = _featquery_value_image_template(merged)
    environment = _featquery_environment(merged)
    explicit_masks = _has_explicit_roi_masks(merged)
    build_actions = (
        ()
        if explicit_masks or roi_set is None
        else plan_roi_build({"roi_set": roi_set.fields}, context=context, validate_personal_paths=False).actions
    )
    output_root, _pipeline_name = _resolve_output_root(
        fields,
        common,
        context=context,
        default_root=_default_roi_derivative_root(context),
    )

    actions: list[RoiExtractionAction] = []
    for base_entities in _fsl_entity_groups(merged):
        for contrast in _featquery_contrasts(merged):
            entities = {
                **base_entities,
                "source_contrast": contrast.source_contrast,
                "contrast": contrast.source_contrast,
                "contrast_id": contrast.source_contrast,
                "contrast_desc": contrast.desc,
                "cope": contrast.cope,
                "cope_number": contrast.cope,
                "extraction_set": extraction_set.name,
                "target": target.name,
                "target_name": target.name,
                "roi_set": roi_set.name if roi_set is not None else extraction_set.roi_set,
            }
            table_path = _featquery_summary_table_path(
                merged,
                output_root=output_root,
                extraction_set=extraction_set,
                target=target,
                entities=entities,
                context=context,
            )
            roi_specs = _fsl_roi_mask_specs(
                merged,
                roi_set=roi_set,
                build_actions=build_actions,
                context=context,
                entities=entities,
            )
            for feat_index, feat_spec in enumerate(feat_dir_specs):
                feat_dir = _resolve_path_spec(
                    feat_spec,
                    context=context,
                    entities={**entities, "feat_index": feat_index},
                    label=f"extraction target {target.name} feat_dir",
                )
                value_image = _render_template(str(value_image_template), entities, label="value_image")
                for roi_spec in roi_specs:
                    roi_entities = {**entities, "roi_label": roi_spec.roi_label}
                    output_name = _render_template(output_name_template, roi_entities, label="featquery_output_name")
                    command_plan = build_featquery_command_plan(
                        feat_dir=feat_dir,
                        roi_mask_path=roi_spec.mask_path,
                        output_name=output_name,
                        value_image=value_image,
                        metrics=metrics,
                        include_percent_signal_change=include_percent_signal_change,
                        environment=environment,
                    )
                    missing_inputs: list[str] = []
                    warnings: list[str] = []
                    if not feat_dir.exists():
                        message = f"FEAT directory is missing: {feat_dir}"
                        if _missing_policy(merged, "feat_dir") == "skip":
                            continue
                        _record_missing("feat_dir", message, merged, missing_inputs=missing_inputs, warnings=warnings)
                    if not roi_spec.mask_path.exists():
                        message = f"ROI mask is missing: {roi_spec.mask_path}"
                        if _missing_policy(merged, "roi_mask") == "skip":
                            continue
                        _record_missing("roi_mask", message, merged, missing_inputs=missing_inputs, warnings=warnings)

                    metadata = {
                        "subject_id": entities.get("subject_id"),
                        "session_id": entities.get("session_id"),
                        "task_id": entities.get("task_id"),
                        "space": entities.get("space"),
                        "direction": entities.get("direction"),
                        "resolution": entities.get("resolution"),
                        "model": entities.get("model"),
                        "roi_set": roi_set.name if roi_set is not None else extraction_set.roi_set,
                        "roi_desc": roi_spec.roi_desc,
                        "roi_family": roi_spec.roi_family,
                        "source_contrast": contrast.source_contrast,
                        "cope": contrast.cope,
                        "feat_dir": str(command_plan.feat_dir),
                        "value_image": command_plan.value_image,
                        "featquery_output_name": command_plan.output_name,
                        "featquery_output_dir": str(command_plan.output_dir),
                        "report_path": str(command_plan.report_path),
                        "command": list(command_plan.command),
                        "command_plan": command_plan.to_dict(),
                        "include_percent_signal_change": command_plan.include_percent_signal_change,
                        "missing": _missing_config(merged),
                        "missing_inputs": missing_inputs,
                        "warnings": warnings,
                        "roi_sidecar": dict(roi_spec.sidecar),
                    }
                    actions.append(
                        RoiExtractionAction(
                            target_name=target.name,
                            backend="fsl_featquery",
                            roi_label=roi_spec.roi_label,
                            value_map_path=command_plan.feat_dir / command_plan.value_image,
                            mask_path=command_plan.roi_mask_path,
                            table_path=table_path,
                            metrics=metrics,
                            metadata=metadata,
                        )
                    )
    return actions


def _execute_fsl_featquery_action(
    action: RoiExtractionAction,
    *,
    staged_output_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from research_platform.neuro.fsl import featquery

    metadata = dict(action.metadata)
    warnings = list(metadata.get("warnings", []))
    missing_inputs = list(metadata.get("missing_inputs", []))
    report = None
    if not missing_inputs:
        command_plan = featquery.build_featquery_command_plan(
            feat_dir=str(metadata["feat_dir"]),
            roi_mask_path=action.mask_path,
            output_name=staged_output_name or str(metadata["featquery_output_name"]),
            value_image=str(metadata["value_image"]),
            metrics=action.metrics,
            include_percent_signal_change=metadata.get("include_percent_signal_change"),
            environment=dict(metadata.get("command_plan", {})).get("environment", {}),
        )
        featquery.execute_featquery_command_plan(command_plan)
        report = featquery.parse_featquery_report(
            command_plan.report_path,
            required_metrics=_featquery_report_required_metrics(action.metrics, metadata),
        )
        if staged_output_name is not None:
            report = replace(report, report_path=Path(str(metadata["report_path"])))
        warnings.extend(report.warnings)
        if "missing_report_values" in report.qc_flags and _missing_policy(metadata, "report_values") == "fail":
            raise ValueError("; ".join(report.warnings) or f"Missing featquery report values: {command_plan.report_path}")

    row = _fsl_featquery_row(action, report=report, warnings=warnings, missing_inputs=missing_inputs)
    result = {
        key: value
        for key, value in {
            "mean_psc": row.get("mean_psc"),
            "mean_cope": row.get("mean_cope"),
            "roi_voxel_count": row.get("roi_voxel_count"),
            "usable": row.get("usable"),
            "qc_flags": row.get("qc_flags"),
            "warnings": row.get("warnings"),
            "report_path": row.get("report_path"),
        }.items()
        if value not in (None, "")
    }
    return row, result


def _validate_featquery_metrics(metrics: Sequence[str], *, include_percent_signal_change: bool | None = None) -> None:
    from research_platform.neuro.fsl.featquery import SUPPORTED_FEATQUERY_METRICS, build_featquery_command_plan

    unknown = sorted(set(metrics) - set(SUPPORTED_FEATQUERY_METRICS))
    if unknown:
        raise ValueError(f"Unsupported featquery metric(s): {', '.join(unknown)}.")
    build_featquery_command_plan(
        feat_dir="validation.feat",
        roi_mask_path="validation_mask.nii.gz",
        output_name="validation",
        metrics=metrics,
        include_percent_signal_change=include_percent_signal_change,
    )


def _fsl_entity_groups(fields: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    subjects = _entity_values(fields, "subject", "subjects", "subject_ids", strip_prefix="sub", required=True)
    sessions = _entity_values(fields, "session", "sessions", "session_ids", strip_prefix="ses", required=False)
    tasks = _entity_values(fields, "task", "tasks", "task_ids", required=True)
    models = _entity_values(fields, "model", "models", "model_ids", required=False)
    scalar_fields = _scalar_template_fields(fields)
    output: list[dict[str, Any]] = []
    for subject in subjects:
        for session in sessions:
            for task in tasks:
                for model in models:
                    entities = dict(scalar_fields)
                    entities.update(
                        {
                            "subject_id": subject,
                            "subject": subject,
                            "subject_dir": f"sub-{subject}",
                            "session_id": session,
                            "session": session,
                            "session_dir": f"ses-{session}" if session else "",
                            "task_id": task,
                            "task": task,
                            "model": model,
                        }
                    )
                    space = _first_text(fields, "space")
                    if space is not None:
                        entities["space"] = space
                    output.append({key: value for key, value in entities.items() if value is not None})
    return tuple(output)


def _entity_values(
    fields: Mapping[str, Any],
    singular: str,
    plural: str,
    aliases: str,
    *,
    strip_prefix: str | None = None,
    required: bool,
) -> tuple[str | None, ...]:
    for key in (plural, aliases, singular, f"{singular}_id"):
        value = fields.get(key)
        values = _optional_list(value)
        if not values:
            continue
        normalized = tuple(_strip_entity_prefix(item, strip_prefix) if strip_prefix else item for item in values)
        return normalized
    if required:
        raise ValueError(f"fsl_featquery extraction requires {singular} or {plural} in config.")
    return (None,)


def _featquery_contrasts(fields: Mapping[str, Any]) -> tuple[_FslFeatqueryContrast, ...]:
    raw = fields.get("contrasts") or fields.get("source_contrasts")
    if isinstance(raw, list) and raw:
        contrasts: list[_FslFeatqueryContrast] = []
        cope_values = _optional_list(fields.get("copes") or fields.get("cope_numbers"))
        for index, item in enumerate(raw):
            if isinstance(item, Mapping):
                contrast_id = _optional_text(item.get("id") or item.get("name") or item.get("source_contrast") or item.get("contrast_id"))
                cope = _optional_text(item.get("cope") or item.get("cope_number"))
                desc = _optional_text(item.get("desc") or item.get("contrast_desc")) or contrast_id
                fields_for_contrast = dict(item)
            else:
                contrast_id = _optional_text(item)
                cope = cope_values[index] if index < len(cope_values) else None
                desc = contrast_id
                fields_for_contrast = {}
            if contrast_id is None or cope is None:
                raise ValueError(f"fsl_featquery contrasts[{index}] must define a source contrast id and cope number.")
            contrasts.append(
                _FslFeatqueryContrast(
                    source_contrast=contrast_id,
                    cope=str(cope),
                    desc=_bids_label(desc or contrast_id),
                    fields=fields_for_contrast,
                )
            )
        return tuple(contrasts)

    contrast_id = _optional_text(fields.get("source_contrast") or fields.get("contrast_id") or fields.get("contrast"))
    cope = _optional_text(fields.get("cope") or fields.get("cope_number"))
    if contrast_id is None or cope is None:
        raise ValueError("fsl_featquery extraction requires contrasts or source_contrast plus cope_number.")
    return (_FslFeatqueryContrast(source_contrast=contrast_id, cope=cope, desc=_bids_label(contrast_id), fields={}),)


def _feat_dir_specs(fields: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = fields.get("feat_dirs", fields.get("feat_dir", fields.get("cope_dirs", fields.get("cope_dir"))))
    inputs = fields.get("inputs")
    if raw is None and isinstance(inputs, Mapping):
        raw = {
            "root_ref": inputs.get("feat_root_ref") or inputs.get("root_ref"),
            "pattern": inputs.get("feat_dir_pattern")
            or inputs.get("feat_dir")
            or inputs.get("cope_dir_pattern")
            or inputs.get("cope_dir")
            or inputs.get("pattern"),
            "path": inputs.get("path"),
        }
    if raw is None:
        raise ValueError("fsl_featquery targets must define feat_dir, feat_dirs, or inputs.feat_dir.")
    if isinstance(raw, list):
        return tuple(_canonical_path_spec(item) for item in raw)
    specs = _patterns_to_specs(raw)
    return tuple(_canonical_path_spec(item) for item in specs)


def _patterns_to_specs(raw: Any) -> tuple[Any, ...]:
    if isinstance(raw, Mapping) and isinstance(raw.get("patterns"), list):
        return tuple({**dict(raw), "pattern": pattern, "patterns": None} for pattern in raw["patterns"])
    return (raw,)


def _canonical_path_spec(raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        return raw
    spec = dict(raw)
    if spec.get("root_ref") is None and spec.get("feat_root_ref") is not None:
        spec["root_ref"] = spec.get("feat_root_ref")
    for source_key in ("feat_dir_pattern", "feat_dir", "cope_dir_pattern", "cope_dir", "mask_pattern", "mask"):
        if spec.get("pattern") is None and spec.get(source_key) is not None:
            spec["pattern"] = spec.get(source_key)
    return spec


def _featquery_value_image_template(fields: Mapping[str, Any]) -> str:
    inputs = fields.get("inputs") if isinstance(fields.get("inputs"), Mapping) else {}
    value = fields.get("value_image") or inputs.get("value_image") or inputs.get("value_image_path") or "stats/cope{cope}"
    return str(value)


def _featquery_output_name_template(fields: Mapping[str, Any]) -> str:
    outputs = fields.get("outputs") if isinstance(fields.get("outputs"), Mapping) else {}
    value = (
        fields.get("featquery_output_name")
        or fields.get("output_name")
        or outputs.get("featquery_output_name")
        or outputs.get("output_name")
        or "fq_{extraction_set}_{roi_label}_{source_contrast}_cope{cope}"
    )
    return str(value)


def _featquery_environment(fields: Mapping[str, Any]) -> dict[str, str]:
    for key in ("environment", "env"):
        if isinstance(fields.get(key), Mapping):
            return {str(name): str(value) for name, value in dict(fields[key]).items()}
    backend_settings = fields.get("backend_settings") or fields.get("fsl") or fields.get("featquery")
    if isinstance(backend_settings, Mapping) and isinstance(backend_settings.get("environment"), Mapping):
        return {str(name): str(value) for name, value in dict(backend_settings["environment"]).items()}
    return {}


def _featquery_include_percent_signal_change(fields: Mapping[str, Any]) -> bool | None:
    for value in _featquery_setting_values(fields, "include_percent_signal_change"):
        return _optional_bool(value, label="include_percent_signal_change")
    return None


def _featquery_setting_values(fields: Mapping[str, Any], setting_name: str) -> tuple[Any, ...]:
    values: list[Any] = []
    if fields.get(setting_name) is not None:
        values.append(fields[setting_name])
    for key in ("featquery", "fsl", "backend_settings", "fsl_featquery"):
        block = fields.get(key)
        if isinstance(block, Mapping) and block.get(setting_name) is not None:
            values.append(block[setting_name])
    return tuple(values)


def _featquery_summary_table_path(
    fields: Mapping[str, Any],
    *,
    output_root: Path,
    extraction_set: ExtractionSet,
    target: Any,
    entities: Mapping[str, Any],
    context: RoiExecutionContext,
) -> Path:
    outputs = fields.get("outputs") if isinstance(fields.get("outputs"), Mapping) else {}
    explicit = outputs.get("summary_table") or outputs.get("summary_table_path")
    if explicit is not None:
        return _resolve_path_spec(explicit, context=context, entities=entities, label="outputs.summary_table")

    from research_platform.bids.roi import build_roi_group_extraction_table_path

    extension = _table_extension(fields, {})
    table_root = output_root / "roi_extract" / extraction_set.name
    return build_roi_group_extraction_table_path(
        table_root,
        session_id=_optional_text(entities.get("session_id")),
        task_id=str(entities["task_id"]),
        extraction_desc=_bids_label(_table_desc(target) or extraction_set.name),
        pipeline_name=None,
        extension=extension,
    )


def _fsl_roi_mask_specs(
    fields: Mapping[str, Any],
    *,
    roi_set: RoiSet | None,
    build_actions: Sequence[RoiBuildAction],
    context: RoiExecutionContext,
    entities: Mapping[str, Any],
) -> tuple[_FslRoiMaskSpec, ...]:
    raw_masks = fields.get("roi_masks", fields.get("masks"))
    requested_labels = _optional_string_set(fields.get("roi_labels"))
    if raw_masks is not None:
        if not isinstance(raw_masks, list) or not raw_masks:
            raise ValueError("roi_masks must contain a non-empty list when declared.")
        specs: list[_FslRoiMaskSpec] = []
        for index, raw in enumerate(raw_masks):
            spec_map = _mapping_value(raw)
            roi_label = _optional_text(spec_map.get("label") or spec_map.get("roi_label"))
            if roi_label is None:
                raise ValueError(f"roi_masks[{index}] must define label or roi_label.")
            if requested_labels is not None and roi_label not in requested_labels:
                continue
            path_spec = _canonical_path_spec(spec_map)
            mask_path = _resolve_path_spec(
                path_spec,
                context=context,
                entities={**entities, "roi_label": roi_label},
                label=f"roi_masks[{index}]",
            )
            sidecar = _load_roi_sidecar(mask_path)
            specs.append(
                _FslRoiMaskSpec(
                    roi_label=roi_label,
                    mask_path=mask_path,
                    roi_desc=_optional_text(spec_map.get("desc")) or _optional_text(sidecar.get("desc")),
                    roi_family=_optional_text(spec_map.get("family")) or _optional_text(sidecar.get("roi_family")),
                    sidecar=sidecar,
                )
            )
        if specs:
            return tuple(specs)
        raise ValueError("No explicit ROI masks matched roi_labels.")

    if roi_set is None:
        raise ValueError("fsl_featquery extraction requires roi_set_ref or explicit roi_masks.")

    if _roi_mask_source(fields) == "roi_set_publication":
        from research_platform.neuro.roi_publication import expected_loso_roi_publication_mask_specs

        specs = []
        published_specs = expected_loso_roi_publication_mask_specs(
            {"roi_set": roi_set.fields},
            actions=build_actions,
            context=context,
        )
        for published in published_specs:
            roi_label = str(published["roi_label"])
            if requested_labels is not None and roi_label not in requested_labels:
                continue
            action = published.get("action")
            if action is not None and not _build_action_matches_entities(action, entities):
                continue
            mask_path = Path(published["mask_path"])
            sidecar = _load_roi_sidecar(mask_path)
            specs.append(
                _FslRoiMaskSpec(
                    roi_label=roi_label,
                    mask_path=mask_path,
                    roi_desc=_optional_text(published.get("roi_desc")) or _optional_text(sidecar.get("desc")),
                    roi_family=_optional_text(published.get("roi_family")) or _optional_text(sidecar.get("roi_family")),
                    sidecar=sidecar,
                )
            )
        if not specs:
            raise ValueError("No published ROI set masks matched the fsl_featquery target entities.")
        return tuple(specs)

    specs = []
    for action in build_actions:
        if requested_labels is not None and action.roi_label not in requested_labels:
            continue
        if not _build_action_matches_entities(action, entities):
            continue
        sidecar = _load_roi_sidecar(action.mask_path)
        specs.append(
            _FslRoiMaskSpec(
                roi_label=action.roi_label,
                mask_path=action.mask_path,
                roi_desc=_optional_text(action.metadata.get("desc")) or _optional_text(sidecar.get("desc")),
                roi_family=action.family,
                sidecar=sidecar,
            )
        )
    if not specs:
        raise ValueError("No ROI set mask outputs matched the fsl_featquery target entities.")
    return tuple(specs)


def _build_action_matches_entities(action: RoiBuildAction, entities: Mapping[str, Any]) -> bool:
    action_entities = action.metadata.get("entities") if isinstance(action.metadata.get("entities"), Mapping) else {}
    for key in ("subject_id", "session_id", "task_id", "model"):
        expected = _optional_text(entities.get(key))
        observed = _optional_text(action_entities.get(key))
        if expected is not None and observed is not None and expected != observed:
            return False
    job = action.metadata.get("loso_group_job")
    if isinstance(job, Mapping):
        contrast = job.get("contrast")
        contrast_id = contrast.get("contrast_id") if isinstance(contrast, Mapping) else None
        expected_contrast = _optional_text(entities.get("source_contrast"))
        if expected_contrast is not None and _optional_text(contrast_id) != expected_contrast:
            return False
    return True


def _load_roi_sidecar(mask_path: Path) -> Mapping[str, Any]:
    from research_platform.bids.roi import build_roi_sidecar_path

    sidecar_path = build_roi_sidecar_path(mask_path)
    if not sidecar_path.exists():
        return {}
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"warnings": [f"ROI sidecar could not be parsed: {sidecar_path}"]}
    return _normalize_roi_sidecar(payload) if isinstance(payload, Mapping) else {}


def _normalize_roi_sidecar(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = payload.get("RuntimeSidecar")
    normalized: dict[str, Any] = dict(runtime) if isinstance(runtime, Mapping) else {}
    normalized.update({key: value for key, value in payload.items() if key != "RuntimeSidecar"})
    aliases = {
        "roi_label": ("roi_label", "ROILabel", "ROI"),
        "roi_family": ("roi_family",),
        "desc": ("desc", "Description"),
        "source_contrast": ("source_contrast", "ContrastName"),
        "cope_number": ("cope_number", "FSLCOPE"),
        "held_out_subject": ("held_out_subject", "HeldOutSubject"),
        "voxel_count": ("voxel_count", "VoxelCount"),
        "qc_flags": ("qc_flags", "QCFlags"),
        "warnings": ("warnings", "Warnings"),
        "fallback_status": ("fallback_status", "FallbackStatus"),
        "loso_peak_coordinate": ("loso_peak_coordinate", "PeakCoordinate"),
        "selected_peak_coordinate": ("selected_peak_coordinate", "PeakCoordinate"),
        "selected_peak_z": ("selected_peak_z", "PeakZStatistic"),
        "selected_peak_stat": ("selected_peak_stat", "PeakZStatistic"),
        "sphere_radius_mm": ("sphere_radius_mm", "SphereRadiusMM"),
        "search_radius_mm": ("search_radius_mm", "SearchRadiusMM"),
        "seed_coordinate": ("seed_coordinate", "SeedCoordinate"),
    }
    for canonical, names in aliases.items():
        if normalized.get(canonical) is not None:
            continue
        for name in names:
            if payload.get(name) is not None:
                normalized[canonical] = payload[name]
                break
    if isinstance(normalized.get("QCFlags"), list) and normalized.get("qc_flags") is None:
        normalized["qc_flags"] = normalized["QCFlags"]
    if isinstance(normalized.get("Warnings"), list) and normalized.get("warnings") is None:
        normalized["warnings"] = normalized["Warnings"]
    return normalized


def _fsl_featquery_row(
    action: RoiExtractionAction,
    *,
    report: Any | None,
    warnings: Sequence[str],
    missing_inputs: Sequence[str],
) -> dict[str, Any]:
    metadata = dict(action.metadata)
    sidecar = metadata.get("roi_sidecar") if isinstance(metadata.get("roi_sidecar"), Mapping) else {}
    report_values = report.to_dict() if report is not None else {}
    blocking_missing_fields = _featquery_blocking_missing_fields(action.metrics, report_values, sidecar)
    qc_flags = _featquery_row_qc_flags(
        report_values,
        blocking_missing_fields=blocking_missing_fields,
        report_missing_policy=_missing_policy(metadata, "report_values"),
        sidecar=sidecar,
    )
    if missing_inputs:
        qc_flags.append("missing_inputs")
    sidecar_warnings = sidecar.get("warnings") if isinstance(sidecar.get("warnings"), list) else []
    all_warnings = [*sidecar_warnings, *warnings]
    fallback_status = _optional_text(sidecar.get("fallback_status"))
    peak_coordinate = sidecar.get("loso_peak_coordinate") or sidecar.get("selected_peak_coordinate")
    peak = peak_coordinate if isinstance(peak_coordinate, Sequence) and not isinstance(peak_coordinate, (str, bytes)) else ()
    sidecar_voxel_count = _sidecar_voxel_count(sidecar)
    report_voxel_count = report_values.get("roi_voxel_count")
    usable = (
        not missing_inputs
        and report is not None
        and "ambiguous_report_values" not in report_values.get("qc_flags", [])
        and not blocking_missing_fields
        and not _roi_sidecar_failed_qc(sidecar)
    )
    if not qc_flags:
        qc_flags.append("pass")
    peak_z = sidecar.get("selected_peak_z") if sidecar.get("selected_peak_z") is not None else sidecar.get("selected_peak_stat")
    return {
        "subject_id": metadata.get("subject_id"),
        "session_id": metadata.get("session_id"),
        "task_id": metadata.get("task_id"),
        "model": metadata.get("model"),
        "roi_set": metadata.get("roi_set"),
        "roi_label": action.roi_label,
        "roi_desc": metadata.get("roi_desc"),
        "roi_family": metadata.get("roi_family"),
        "source_contrast": metadata.get("source_contrast"),
        "cope": metadata.get("cope"),
        "feat_dir": metadata.get("feat_dir"),
        "roi_mask_path": str(action.mask_path),
        "featquery_output_dir": metadata.get("featquery_output_dir"),
        "report_path": metadata.get("report_path"),
        "mean_psc": report_values.get("mean_psc"),
        "mean_cope": report_values.get("mean_cope"),
        "roi_voxel_count": sidecar_voxel_count if sidecar_voxel_count is not None else report_voxel_count,
        "usable": usable,
        "thresholded_peak": fallback_status == "thresholded" if fallback_status else None,
        "below_threshold_fallback": fallback_status == "below_threshold_fallback" if fallback_status else None,
        "peak_x_mm": peak[0] if len(peak) == 3 else None,
        "peak_y_mm": peak[1] if len(peak) == 3 else None,
        "peak_z_mm": peak[2] if len(peak) == 3 else None,
        "z_at_peak": peak_z,
        "backend": action.backend,
        "featquery_command": _json_cell(metadata.get("command")),
        "qc_flags": ";".join(str(flag) for flag in qc_flags),
        "warnings": ";".join(str(item) for item in all_warnings if item),
    }


def _featquery_report_required_metrics(metrics: Sequence[str], metadata: Mapping[str, Any]) -> tuple[str, ...]:
    sidecar = metadata.get("roi_sidecar") if isinstance(metadata.get("roi_sidecar"), Mapping) else {}
    if _sidecar_voxel_count(sidecar) is None:
        return tuple(metrics)
    return tuple(metric for metric in metrics if metric != "roi_voxel_count")


def _featquery_blocking_missing_fields(
    metrics: Sequence[str],
    report_values: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> tuple[str, ...]:
    missing: list[str] = []
    metrics_set = set(metrics)
    if "mean_cope" in metrics_set and report_values.get("mean_cope") is None:
        missing.append("mean_cope")
    if "percent_signal_change" in metrics_set and report_values.get("mean_psc") is None:
        missing.append("mean_psc")
    if "roi_voxel_count" in metrics_set and report_values.get("roi_voxel_count") is None and _sidecar_voxel_count(sidecar) is None:
        missing.append("roi_voxel_count")
    return tuple(missing)


def _featquery_row_qc_flags(
    report_values: Mapping[str, Any],
    *,
    blocking_missing_fields: Sequence[str],
    report_missing_policy: str,
    sidecar: Mapping[str, Any],
) -> list[str]:
    flags = []
    for flag in report_values.get("qc_flags", []):
        if flag == "pass":
            continue
        if flag == "missing_report_values" and not blocking_missing_fields and report_missing_policy != "fail":
            continue
        flags.append(str(flag))
    if _roi_sidecar_failed_qc(sidecar):
        flags.append("roi_qc_failed")
    return flags


def _sidecar_voxel_count(sidecar: Mapping[str, Any]) -> int | None:
    value = sidecar.get("voxel_count")
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _roi_sidecar_failed_qc(sidecar: Mapping[str, Any]) -> bool:
    flags = sidecar.get("qc_flags")
    if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
        return False
    return any(str(flag).startswith("fail") for flag in flags)


def _record_missing(
    key: str,
    message: str,
    fields: Mapping[str, Any],
    *,
    missing_inputs: list[str],
    warnings: list[str],
) -> None:
    policy = _missing_policy(fields, key)
    if policy == "fail":
        raise ValueError(message)
    missing_inputs.append(key)
    warnings.append(message)


def _missing_policy(fields: Mapping[str, Any], key: str) -> str:
    raw = None
    block = fields.get("missing") or fields.get("on_missing") or fields.get("missing_behavior")
    if isinstance(block, Mapping):
        raw = block.get(key) or block.get(f"{key}s") or block.get("default")
    elif block is not None:
        raw = block
    for field_name in (f"on_missing_{key}", f"missing_{key}", f"{key}_missing"):
        if fields.get(field_name) is not None:
            raw = fields[field_name]
    value = _optional_text(raw) or "warn"
    aliases = {"error": "fail", "raise": "fail", "warning": "warn"}
    normalized = aliases.get(value.lower(), value.lower())
    if normalized not in {"warn", "skip", "fail"}:
        raise ValueError("Missing-input behavior must be one of: warn, skip, fail.")
    return normalized


def _missing_config(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    block = fields.get("missing") or fields.get("on_missing") or fields.get("missing_behavior")
    return dict(block) if isinstance(block, Mapping) else {"default": block} if block is not None else {}


def _has_explicit_roi_masks(fields: Mapping[str, Any]) -> bool:
    return fields.get("roi_masks") is not None or fields.get("masks") is not None


def _roi_mask_source(fields: Mapping[str, Any]) -> str:
    raw = fields.get("roi_mask_source")
    if isinstance(raw, Mapping):
        source = _optional_text(raw.get("source"))
    else:
        source = _optional_text(raw)
    return source or "roi_set_runtime"


def _merge_nested(*payloads: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        for key, value in payload.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**dict(merged[key]), **dict(value)}
            else:
                merged[key] = value
    return merged


def _scalar_template_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in fields.items()
        if isinstance(value, (str, int, float, bool)) and value is not None
    }


def _optional_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def _bids_label(value: Any) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError("BIDS-like label value must be defined.")
    label = "".join(character for character in text if character.isalnum())
    if not label:
        raise ValueError(f"Cannot derive a BIDS-like label from {text!r}.")
    return label


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _require_supported_build_runtime(roi: RoiDefinition, backend: str) -> None:
    if roi.family in DEFERRED_BUILD_FAMILIES:
        raise UnsupportedRoiRuntimeError(
            f"ROI family {roi.family!r} is schema-only/deferred in local ROI execution; "
            "supported families: coordinate_sphere, manual_mask, functional_threshold_map, loso_group_map."
        )
    if roi.family not in SUPPORTED_BUILD_FAMILIES:
        raise UnsupportedRoiRuntimeError(f"Unsupported ROI family for local ROI execution: {roi.family!r}.")
    if roi.family == "loso_group_map":
        if backend != "fsl_flame1":
            raise UnsupportedRoiRuntimeError("Phase 4 LOSO ROI execution supports only backend fsl_flame1.")
        return
    supported_backends = {"generic_nifti"} if roi.family != "manual_mask" else {"generic_nifti", "manual"}
    if backend not in supported_backends:
        raise UnsupportedRoiRuntimeError(
            f"ROI backend {backend!r} for family {roi.family!r} is not executable in local ROI execution."
        )


def _build_backend(roi_set: RoiSet, roi: RoiDefinition) -> str:
    if roi.backend:
        return roi.backend
    if roi_set.backend:
        return roi_set.backend
    return "manual" if roi.family == "manual_mask" else "generic_nifti"


def _extraction_backend(extraction_set: ExtractionSet, target: Any) -> str:
    return target.backend or extraction_set.backend or "generic_nifti"


def _roi_by_label(roi_set: RoiSet, roi_label: str) -> RoiDefinition:
    for roi in roi_set.rois:
        if roi.label == roi_label:
            return roi
    raise ValueError(f"ROI label {roi_label!r} was not found in ROI set {roi_set.name!r}.")


def _roi_set_common_fields(roi_set: RoiSet) -> dict[str, Any]:
    return {
        **dict(roi_set.fields),
        "name": roi_set.name,
        "roi_set": roi_set.name,
        "desc": roi_set.desc,
        "backend": roi_set.backend,
        **dict(roi_set.provenance if isinstance(roi_set.provenance, Mapping) else {}),
    }


def _extraction_set_common_fields(extraction_set: ExtractionSet) -> dict[str, Any]:
    return {
        **dict(extraction_set.fields),
        "name": extraction_set.name,
        "extraction_set": extraction_set.name,
        "roi_set": extraction_set.roi_set,
        "backend": extraction_set.backend,
        **dict(extraction_set.provenance if isinstance(extraction_set.provenance, Mapping) else {}),
    }


def _entities(*payloads: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        if isinstance(payload.get("entities"), Mapping):
            merged.update(dict(payload["entities"]))
        if isinstance(payload.get("bids_entities"), Mapping):
            merged.update(dict(payload["bids_entities"]))
        merged.update({key: value for key, value in payload.items() if key not in {"entities", "bids_entities"}})

    subject_id = _first_text(merged, "subject_id", "subject")
    session_id = _first_text(merged, "session_id", "session")
    task_id = _first_text(merged, "task_id", "task")
    space = _first_text(merged, "space")
    direction = _first_text(merged, "direction", "dir")
    resolution = _first_text(merged, "resolution", "res")
    datatype = _first_text(merged, "datatype") or "func"
    if subject_id is None:
        raise ValueError("ROI execution requires a subject_id or subject in config.")
    if task_id is None:
        raise ValueError("ROI execution requires a task_id or task in config.")
    if space is None:
        raise ValueError("ROI execution requires a space in config.")

    normalized_subject = _strip_entity_prefix(subject_id, "sub")
    normalized_session = _strip_entity_prefix(session_id, "ses") if session_id is not None else None
    output = dict(merged)
    output.update(
        {
            "subject_id": normalized_subject,
            "subject": normalized_subject,
            "subject_dir": f"sub-{normalized_subject}",
            "session_id": normalized_session,
            "session": normalized_session,
            "session_dir": f"ses-{normalized_session}" if normalized_session else "",
            "task_id": task_id,
            "task": task_id,
            "space": space,
            "direction": direction,
            "dir": direction,
            "resolution": resolution,
            "res": resolution,
            "datatype": datatype,
        }
    )
    return {key: value for key, value in output.items() if value is not None}


def _strip_entity_prefix(value: str | None, prefix: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    token = f"{prefix}-"
    return text[len(token) :] if text.startswith(token) else text


def _method_desc(roi_set: RoiSet, roi: RoiDefinition) -> str:
    return _optional_text(roi.desc) or _optional_text(roi_set.desc) or roi.label


def _table_desc(target: Any) -> str:
    outputs = target.fields.get("outputs")
    if isinstance(outputs, Mapping):
        table_desc = _optional_text(outputs.get("table_desc")) or _optional_text(outputs.get("desc"))
        if table_desc:
            return table_desc
    return target.desc or target.name


def _table_extension(fields: Mapping[str, Any], common: Mapping[str, Any]) -> str:
    outputs = fields.get("outputs") if isinstance(fields.get("outputs"), Mapping) else common.get("outputs")
    extension = ".tsv"
    if isinstance(outputs, Mapping):
        extension = _optional_text(outputs.get("extension")) or extension
        table_format = _optional_text(outputs.get("format"))
        if table_format:
            if table_format not in {"tsv", "csv"}:
                raise ValueError("extraction outputs.format must be either tsv or csv.")
            extension = f".{table_format}"
    return extension if extension.startswith(".") else f".{extension}"


def _resolve_output_root(
    fields: Mapping[str, Any],
    common: Mapping[str, Any],
    *,
    context: RoiExecutionContext,
    default_root: Path,
) -> tuple[Path, str | None]:
    outputs = fields.get("outputs") if isinstance(fields.get("outputs"), Mapping) else common.get("outputs")
    if isinstance(outputs, Mapping):
        pipeline_name = _optional_text(outputs.get("pipeline_name"))
        root_value = outputs.get("derivative_root") or outputs.get("root") or outputs.get("output_root")
        if root_value is not None:
            return (
                _resolve_runtime_output_root_spec(
                    root_value,
                    context=context,
                    label="outputs.derivative_root",
                ),
                pipeline_name or "roi",
            )
        if outputs.get("root_ref") is not None:
            root = context.resolve_root_ref(str(outputs["root_ref"]))
            subpath = _optional_text(outputs.get("path")) or _optional_text(outputs.get("subpath"))
            relative = _runtime_output_relative_path(subpath, label="outputs.path") if subpath else None
            return (root / relative if relative is not None else root), pipeline_name or "roi"
        if outputs.get("path") is not None:
            relative = _runtime_output_relative_path(outputs["path"], label="outputs.path")
            return context.project_root / relative, pipeline_name or "roi"

    for key in ("derivative_root", "output_root"):
        if fields.get(key) is not None:
            return _resolve_runtime_output_root_spec(fields[key], context=context, label=key), "roi"
        if common.get(key) is not None:
            return _resolve_runtime_output_root_spec(common[key], context=context, label=key), "roi"
    return default_root, "roi"


def _resolve_runtime_output_root_spec(
    spec: Any,
    *,
    context: RoiExecutionContext,
    label: str,
) -> Path:
    if isinstance(spec, (str, Path)):
        text = str(spec).strip()
        if not text:
            raise ValueError(f"{label} must be a non-empty path string or mapping.")
        _reject_project_parent_traversal(text, label=label)
        path = Path(text).expanduser()
        if path.is_absolute():
            return _resolved_configured_root(path)
        # The project root is already a trusted canonical base. Keep the
        # configured suffix lexical so runtime preflight can see any symlink
        # introduced by the output-root declaration.
        return context.project_root / path
    if not isinstance(spec, Mapping):
        raise ValueError(f"{label} must be a path string or mapping.")

    root_ref = _optional_text(spec.get("root_ref"))
    pattern = spec.get("pattern")
    path_value = spec.get("path")
    if pattern is None and path_value is None and root_ref is None:
        raise ValueError(f"{label} must define path, pattern, or root_ref.")

    base = context.resolve_root_ref(root_ref) if root_ref is not None else context.project_root
    relative = pattern if pattern is not None else path_value
    if relative is None:
        return base
    rendered = _render_template(str(relative), {}, label=label)
    if not rendered.strip():
        raise ValueError(f"{label} must define a non-empty path or pattern.")
    if root_ref is not None:
        return base / _runtime_output_relative_path(rendered, label=label)
    _reject_project_parent_traversal(rendered, label=label)
    path = Path(rendered).expanduser()
    return _resolved_configured_root(path if path.is_absolute() else base / path)


def _runtime_output_relative_path(value: Any, *, label: str) -> Path:
    text = str(value).strip()
    if not text or configured_path_is_unsafe(text):
        raise ValueError(
            f"{label} must be a relative path that remains beneath its configured root."
        )
    return Path(text)


def _default_roi_derivative_root(context: RoiExecutionContext) -> Path:
    project = context.project_name or "project"
    return context.artifacts_root / "roi" / project / "derivatives"


def _resolve_required_path(
    fields: Mapping[str, Any],
    *keys: str,
    context: RoiExecutionContext,
    entities: Mapping[str, Any],
) -> Path:
    path = _resolve_optional_path(fields, *keys, context=context, entities=entities)
    if path is None:
        names = ", ".join(keys)
        raise ValueError(f"ROI config must define one of: {names}.")
    return path


def _resolve_optional_path(
    fields: Mapping[str, Any],
    *keys: str,
    context: RoiExecutionContext,
    entities: Mapping[str, Any],
) -> Path | None:
    for key in keys:
        if key in fields and fields[key] is not None:
            return _resolve_path_spec(fields[key], context=context, entities=entities, label=key)
    return None


def _resolve_path_spec(
    spec: Any,
    *,
    context: RoiExecutionContext,
    entities: Mapping[str, Any],
    label: str,
) -> Path:
    if isinstance(spec, (str, Path)):
        return _resolve_project_relative_path(context, _render_template(str(spec), entities, label=label))
    if not isinstance(spec, Mapping):
        raise ValueError(f"{label} must be a path string or mapping.")

    root_ref = _optional_text(spec.get("root_ref"))
    pattern = spec.get("pattern")
    path_value = spec.get("path")
    if pattern is None and path_value is None and root_ref is None:
        raise ValueError(f"{label} must define path, pattern, or root_ref.")

    base = context.resolve_root_ref(root_ref) if root_ref is not None else context.project_root
    relative = pattern if pattern is not None else path_value
    if relative is None:
        return base
    rendered = _render_template(str(relative), entities, label=label)
    if root_ref is not None:
        return base / _runtime_output_relative_path(rendered, label=label)
    _reject_project_parent_traversal(rendered, label=label)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_project_relative_path(context: RoiExecutionContext, value: str) -> Path:
    _reject_project_parent_traversal(value, label="ROI input path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (context.project_root / path).resolve()


def _reject_project_parent_traversal(value: str, *, label: str) -> None:
    if any(component == ".." for component in str(value).strip().replace("\\", "/").split("/")):
        raise ValueError(f"{label} must remain beneath its implicit project root.")


def _resolved_configured_root(value: str | Path) -> Path:
    """Canonicalize a trusted root while preserving a symlink at the root.

    Platform-level aliases above a trusted root (for example a temporary-file
    prefix) are not ROI output configuration. Resolving those aliases avoids
    false positives. A configured root that is itself a symbolic link remains
    lexical so runtime output preflight can reject it.
    """

    lexical = Path(os.path.abspath(str(Path(value).expanduser())))
    if lexical.is_symlink():
        return lexical.parent.resolve(strict=False) / lexical.name
    return lexical.resolve(strict=False)


def _render_template(template: str, values: Mapping[str, Any], *, label: str) -> str:
    names = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
    missing = sorted(name for name in names if name not in values)
    if missing:
        raise ValueError(f"{label} references missing template field(s): {', '.join(missing)}.")
    return template.format(**values)


def _coverage_mask_paths(
    fields: Mapping[str, Any],
    *,
    context: RoiExecutionContext,
    entities: Mapping[str, Any],
) -> dict[str, Path]:
    raw = fields.get("coverage_masks")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("coverage_masks must contain a mapping of names to path specs.")
    return {
        str(name): _resolve_path_spec(spec, context=context, entities=entities, label=f"coverage_masks.{name}")
        for name, spec in raw.items()
    }


def _value_map_specs(fields: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = fields.get("inputs", fields.get("value_map", fields.get("value_maps")))
    if raw is None:
        raise ValueError("generic_nifti extraction targets must define inputs or value_map.")
    if isinstance(raw, Mapping):
        patterns = raw.get("patterns")
        if isinstance(patterns, list):
            return tuple({**dict(raw), "pattern": pattern, "patterns": None} for pattern in patterns)
        return (raw,)
    if isinstance(raw, list):
        return tuple(raw)
    return (raw,)


def _roi_mask_specs(
    fields: Mapping[str, Any],
    roi_set: RoiSet,
    *,
    context: RoiExecutionContext,
    entities: Mapping[str, Any],
) -> tuple[tuple[str, Path], ...]:
    raw_masks = fields.get("roi_masks", fields.get("masks"))
    if raw_masks is not None:
        if not isinstance(raw_masks, list) or not raw_masks:
            raise ValueError("roi_masks must contain a non-empty list when declared.")
        resolved: list[tuple[str, Path]] = []
        for index, spec in enumerate(raw_masks):
            spec_map = _mapping_value(spec)
            roi_label = _optional_text(spec_map.get("label") or spec_map.get("roi_label"))
            if roi_label is None:
                raise ValueError(f"roi_masks[{index}] must define label or roi_label.")
            resolved.append(
                (
                    roi_label,
                    _resolve_path_spec(spec, context=context, entities={**entities, "roi_label": roi_label}, label=f"roi_masks[{index}]"),
                )
            )
        return tuple(resolved)

    requested_labels = _optional_string_set(fields.get("roi_labels"))
    masks: list[tuple[str, Path]] = []
    for roi in roi_set.rois:
        if requested_labels is not None and roi.label not in requested_labels:
            continue
        action = _plan_build_action(roi_set, roi, context=context)
        masks.append((roi.label, action.mask_path))
    if not masks:
        raise ValueError("No ROI masks were selected for extraction.")
    return tuple(masks)


def _mapping_value(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_ref(fields: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key not in fields or fields[key] is None:
            continue
        value = fields[key]
        if isinstance(value, Mapping):
            return _optional_text(value.get("pattern") or value.get("path"))
        return _optional_text(value)
    return None


def _load_nifti_image(path: str | Path) -> Any:
    try:
        from research_platform.neuro.nifti import load_nifti_image
    except ImportError as exc:
        raise RuntimeError(f"ROI execution requires NIfTI runtime dependencies: {exc}") from exc
    return load_nifti_image(path)


def _build_result_payload(
    result: Any,
    *,
    mask_path: Path | None = None,
    sidecar_path: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mask_path": mask_path if mask_path is not None else result.mask_path,
        "sidecar_path": sidecar_path if sidecar_path is not None else result.sidecar_path,
        "voxel_count": result.voxel_count,
        "qc_flags": list(result.qc.qc_flags),
        "warnings": list(result.qc.warnings),
    }
    if result.peak is not None:
        payload["peak"] = result.peak.provenance_fields()
    return _json_ready(payload)


def _build_roi_mask_path(
    derivative_root: Path,
    *,
    entities: Mapping[str, Any],
    roi_label: str,
    method_desc: str,
    pipeline_name: str | None,
) -> Path:
    from research_platform.bids.roi import build_roi_mask_path

    return build_roi_mask_path(
        derivative_root,
        subject_id=str(entities["subject_id"]),
        session_id=_optional_text(entities.get("session_id")),
        task_id=str(entities["task_id"]),
        direction=_optional_text(entities.get("direction")),
        space=str(entities["space"]),
        resolution=_optional_text(entities.get("resolution")),
        roi_label=roi_label,
        method_desc=method_desc,
        pipeline_name=pipeline_name,
        datatype=str(entities.get("datatype", "func")),
    )


def _build_roi_sidecar_path(mask_path: Path) -> Path:
    from research_platform.bids.roi import build_roi_sidecar_path

    return build_roi_sidecar_path(mask_path)


def _build_roi_extraction_table_path(
    derivative_root: Path,
    *,
    entities: Mapping[str, Any],
    extraction_desc: str,
    pipeline_name: str | None,
    extension: str,
) -> Path:
    from research_platform.bids.roi import build_roi_extraction_table_path

    return build_roi_extraction_table_path(
        derivative_root,
        subject_id=str(entities["subject_id"]),
        session_id=_optional_text(entities.get("session_id")),
        task_id=str(entities["task_id"]),
        direction=_optional_text(entities.get("direction")),
        space=_optional_text(entities.get("space")),
        resolution=_optional_text(entities.get("resolution")),
        extraction_desc=extraction_desc,
        pipeline_name=pipeline_name,
        datatype=str(entities.get("datatype", "func")),
        extension=extension,
    )


def _required_field(fields: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if fields.get(key) is not None:
            return fields[key]
    raise ValueError(f"ROI config must define one of: {', '.join(keys)}.")


def _first_text(fields: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _optional_text(fields.get(key))
        if value is not None:
            return value
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number_sequence(value: Any, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numeric values.")
    return tuple(_number_value(item, label=f"{label}[{index}]") for index, item in enumerate(value))


def _optional_number_sequence(value: Any, *, label: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _number_sequence(value, label=label)


def _number_value(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    return _number_value(value, label="number")


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    return int(value)


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Boolean ROI config fields must contain true or false.")


def _optional_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} must be a boolean value.")


def _string_list(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        values = tuple(str(item).strip() for item in value if str(item).strip())
        if values:
            return values
    raise ValueError(f"{label} must contain a non-empty string or list of strings.")


def _optional_string_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    return set(_string_list(value, label="roi_labels"))


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    output: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        output.append(path)
    return tuple(output)


def _write_extraction_summary_tables(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    qc_output_path: str | Path | None = None,
) -> tuple[Path, ...]:
    row_list = [_normalize_summary_row(row) for row in rows]
    if not row_list:
        raise ValueError("At least one extraction row is required.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if output.suffix.lower() == ".tsv" else ","

    values_rows = [row for row in row_list if _included_in_values(row)]
    _write_rows(
        values_rows,
        output,
        fieldnames=_analysis_values_fieldnames(row_list, values_rows),
        delimiter=delimiter,
    )

    qc_rows = [_qc_summary_row(row, included=_included_in_values(row)) for row in row_list]
    qc_output = Path(qc_output_path) if qc_output_path is not None else _qc_summary_table_path(output)
    qc_output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(qc_rows, qc_output, fieldnames=_ordered_fieldnames(qc_rows), delimiter=delimiter)
    return (output, qc_output)


def _write_extraction_rows(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    row_list = [_normalize_summary_row(row) for row in rows]
    if not row_list:
        raise ValueError("At least one extraction row is required.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if output.suffix.lower() == ".tsv" else ","
    _write_rows(row_list, output, fieldnames=_ordered_fieldnames(row_list), delimiter=delimiter)
    return output


def _write_rows(rows: Sequence[Mapping[str, Any]], output: Path, *, fieldnames: Sequence[str], delimiter: str) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ordered_fieldnames(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _analysis_values_fieldnames(all_rows: Sequence[Mapping[str, Any]], values_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    source_rows = values_rows if values_rows else all_rows
    available = _ordered_fieldnames(source_rows)
    fieldnames = [
        column
        for column in ANALYSIS_VALUES_COLUMN_ORDER
        if column in available and _include_analysis_values_column(column, all_rows=all_rows, values_rows=values_rows)
    ]
    for column in available:
        if column in fieldnames or column in ANALYSIS_VALUES_COLUMN_ORDER:
            continue
        if _include_analysis_values_column(column, all_rows=all_rows, values_rows=values_rows):
            fieldnames.append(column)
    return fieldnames


def _include_analysis_values_column(
    column: str,
    *,
    all_rows: Sequence[Mapping[str, Any]],
    values_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if column in VALUES_TABLE_EXCLUDED_COLUMNS:
        return False
    if column in VALUES_TABLE_METRIC_COLUMNS:
        return bool(values_rows) and any(_cell_has_value(row.get(column)) for row in values_rows)
    source_rows = values_rows if values_rows else all_rows
    return any(_cell_has_value(row.get(column)) for row in source_rows)


def _normalize_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "subject_id" in normalized:
        normalized["subject_id"] = _strip_entity_prefix(normalized.get("subject_id"), "sub")
    if "session_id" in normalized:
        normalized["session_id"] = _strip_entity_prefix(normalized.get("session_id"), "ses")
    return normalized


def _included_in_values(row: Mapping[str, Any]) -> bool:
    if "usable" not in row:
        return True
    value = row.get("usable")
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return False
    return True


def _qc_summary_row(row: Mapping[str, Any], *, included: bool) -> dict[str, Any]:
    output = dict(row)
    output["included_in_values"] = "true" if included else "false"
    output["exclude_reason"] = "" if included else _values_exclude_reason(row)
    return output


def _values_exclude_reason(row: Mapping[str, Any]) -> str:
    flags = _optional_text(row.get("qc_flags"))
    if flags and flags != "pass":
        return f"qc_flags={flags}"
    if _cell_has_value(row.get("warnings")):
        return "warnings_present"
    return "usable=false"


def _qc_summary_table_path(values_path: Path) -> Path:
    for suffix in (".tsv", ".csv"):
        ending = f"_values{suffix}"
        if values_path.name.endswith(ending):
            return values_path.with_name(f"{values_path.name[: -len(ending)]}_qc{suffix}")
    if values_path.stem.endswith("_values"):
        return values_path.with_name(f"{values_path.stem[:-7]}_qc{values_path.suffix}")
    return values_path.with_name(f"{values_path.stem}_qc{values_path.suffix}")


def _cell_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items() if item is not None}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
