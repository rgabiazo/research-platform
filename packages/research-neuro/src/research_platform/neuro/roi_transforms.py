"""ROI transform planning plus explicit execution/QC helpers.

This module plans ANTs-backed transforms of existing MNI-space ROI masks into a
subject reference space. It renders argv vectors and preflight rows only: it
does not execute ANTs or FSL, load NIfTI files, create transformed masks, write
QC outputs, extract MVPA patterns, compute distances, or summarize results.

Execution is intentionally exposed through separate ``execute_*`` APIs. Those
APIs run only already-planned ANTs argv vectors through an injectable runner,
then perform actual transformed-mask QC and optional planned JSON writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import json
import math
import os
import re
import shutil
import subprocess

from research_platform.neuro._roi_path_safety import configured_path_is_unsafe


ANTS_APPLY_TRANSFORMS = "antsApplyTransforms"
DEFAULT_DIMENSION = 3
DEFAULT_INTERPOLATION = "NearestNeighbor"
MISSING_POLICY_FAIL = "fail"
MISSING_POLICY_WARN = "warn"
MISSING_POLICY_IGNORE = "ignore"
MISSING_POLICIES = frozenset({MISSING_POLICY_FAIL, MISSING_POLICY_WARN, MISSING_POLICY_IGNORE})

_UNRESOLVED_PLACEHOLDER = re.compile(r"\$\{[^}]+\}|\{[^{}]+\}")
_GLOB_CHARS = frozenset("*?[")
_SOURCE_PATH_KEYS = (
    "source_mask_path",
    "mni_mask_path",
    "roi_mask_path",
    "mask_path",
    "path",
)
_TARGET_PATH_KEYS = (
    "target_reference_path",
    "reference_image_path",
    "reference_path",
    "target_path",
    "ref_path",
    "path",
)
_OUTPUT_PATH_KEYS = (
    "planned_output_mask_path",
    "output_mask_path",
    "transformed_mask_path",
    "output_path",
)
_COVERAGE_PATH_KEYS = (
    "coverage_mask_path",
    "brain_mask_path",
    "coverage_path",
)
_QC_OUTPUT_PATH_KEYS = (
    "planned_qc_path",
    "output_qc_path",
    "qc_path",
    "roi_qc_path",
    "transformed_qc_path",
)
_PROVENANCE_OUTPUT_PATH_KEYS = (
    "planned_provenance_path",
    "output_provenance_path",
    "provenance_path",
    "planned_sidecar_path",
    "output_sidecar_path",
    "sidecar_path",
    "roi_sidecar_path",
)
_IDENTIFIER_KEYS = (
    "subject_id",
    "participant_id",
    "session_id",
    "task_id",
    "run_id",
    "model",
    "contrast_id",
    "roi_label",
)


@dataclass(frozen=True)
class RoiTransformSourceMaskRow:
    """One source ROI mask to transform from MNI space."""

    source_index: int
    source_mask_path: str | None
    source_space: str | None = "MNI"
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    model: str | None = None
    contrast_id: str | None = None
    roi_label: str | None = None
    root_ref: str | None = None
    exists: bool | None = None
    status: str = "planned"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TargetReferenceImageRow:
    """One target anatomical or FEAT/MVPA reference image."""

    source_index: int
    target_reference_path: str | None
    target_space: str | None = "T1w"
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    model: str | None = None
    contrast_id: str | None = None
    roi_label: str | None = None
    root_ref: str | None = None
    exists: bool | None = None
    status: str = "planned"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TransformChainSpecRow:
    """One ordered transform in an ROI transform chain."""

    source_index: int
    order_index: int
    transform_path: str | None
    invert: bool = False
    transform_type: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    model: str | None = None
    contrast_id: str | None = None
    roi_label: str | None = None
    root_ref: str | None = None
    exists: bool | None = None
    status: str = "planned"
    policy: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TransformToolSpecRow:
    """Transform tool configuration used for command planning."""

    tool_name: str = ANTS_APPLY_TRANSFORMS
    executable: str | None = None
    dimension: int = DEFAULT_DIMENSION
    default_interpolation: str = DEFAULT_INTERPOLATION
    lookup_on_path: bool = True
    status: str = "configured"
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TransformToolPreflightRow:
    """Availability check for a configured transform executable."""

    tool_name: str
    requested_executable: str | None
    resolved_executable: str | None
    available: bool
    status: str
    check_method: str
    message: str
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TransformCommandPlanRow:
    """One shell-safe transform command argv preview."""

    source_index: int
    tool_name: str
    argv: tuple[str, ...]
    dimension: int
    interpolation: str
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    model: str | None = None
    contrast_id: str | None = None
    roi_label: str | None = None
    status: str = "planned"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PlannedTransformedRoiOutputRow:
    """One transformed ROI output path planned for future execution."""

    source_index: int
    output_mask_path: str | None
    output_space: str | None = "T1w"
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    model: str | None = None
    contrast_id: str | None = None
    roi_label: str | None = None
    root_ref: str | None = None
    status: str = "planned"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    qc_path: str | None = None
    provenance_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiQcPreviewRow:
    """One preflight or post-execution ROI QC check preview."""

    source_index: int
    check_kind: str
    timing: str
    status: str
    message: str
    path: str | None = None
    expected: str | None = None
    threshold: int | float | str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    model: str | None = None
    contrast_id: str | None = None
    roi_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformProvenanceRow:
    """Plan provenance for root references and transform-chain sources."""

    provenance_kind: str
    status: str
    root_ref: str | None = None
    path: str | None = None
    source_index: int | None = None
    message: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformPlan:
    """Top-level JSON-safe ROI transform plan."""

    status: str
    executed: bool = False
    plan_only: bool = True
    source_masks: tuple[RoiTransformSourceMaskRow, ...] = ()
    target_references: tuple[TargetReferenceImageRow, ...] = ()
    transform_chains: tuple[TransformChainSpecRow, ...] = ()
    tool_specs: tuple[TransformToolSpecRow, ...] = ()
    tool_preflight: tuple[TransformToolPreflightRow, ...] = ()
    command_plans: tuple[TransformCommandPlanRow, ...] = ()
    planned_outputs: tuple[PlannedTransformedRoiOutputRow, ...] = ()
    qc_preview: tuple[RoiQcPreviewRow, ...] = ()
    provenance_rows: tuple[RoiTransformProvenanceRow, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """Return whether the plan has no error-level rows."""

        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


MniToT1wRoiTransformPlan = RoiTransformPlan


@dataclass(frozen=True)
class ExecutableMniToT1wRoiTransformJob:
    """One already-planned transform command that is eligible for execution."""

    source_index: int
    tool_name: str
    argv: tuple[str, ...]
    source_mask_path: str
    target_reference_path: str
    output_mask_path: str
    transform_chain: tuple[Mapping[str, Any], ...] = ()
    interpolation: str = DEFAULT_INTERPOLATION
    dimension: int = DEFAULT_DIMENSION
    coverage_mask_path: str | None = None
    coverage_mask_paths: tuple[str, ...] = ()
    coverage_mask_names: tuple[str, ...] = ()
    coverage_intersection_policy: str = "qc_only"
    qc_path: str | None = None
    provenance_path: str | None = None
    min_voxels_warn: int | None = None
    min_voxels_fail: int | None = None
    coverage_min_overlap_fraction_warn: float | None = None
    coverage_min_overlap_fraction_fail: float | None = None
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    model: str | None = None
    contrast_id: str | None = None
    roi_label: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformCommandExecutionRecord:
    """One attempted transform command execution."""

    source_index: int
    argv: tuple[str, ...]
    status: str
    returncode: int | None = None
    stdout_summary: str | None = None
    stderr_summary: str | None = None
    error: str | None = None
    attempted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TransformedRoiOutputVerificationRow:
    """Post-command existence check for a transformed ROI output."""

    source_index: int
    path: str
    exists: bool
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformGeometryQcRow:
    """Geometry QC comparing transformed ROI and target reference images."""

    source_index: int
    mask_path: str
    reference_path: str
    status: str
    shape_matches: bool | None
    affine_matches: bool | None
    output_shape: tuple[int, ...] = ()
    reference_shape: tuple[int, ...] = ()
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformBinaryMaskQcRow:
    """Binary-mask QC for the transformed ROI image."""

    source_index: int
    mask_path: str
    status: str
    binary_interpretable: bool
    unique_values: tuple[int | float | str, ...] = ()
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformVoxelCountQcRow:
    """Voxel-count and empty/small-mask QC for a transformed ROI."""

    source_index: int
    mask_path: str
    status: str
    voxel_count: int
    empty: bool
    min_voxels_warn: int | None = None
    min_voxels_fail: int | None = None
    qc_flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformCoverageOverlapQcRow:
    """Coverage/brain-mask overlap QC for a transformed ROI."""

    source_index: int
    mask_path: str
    coverage_mask_path: str
    status: str
    overlap_ratio: float
    roi_voxel_count: int
    overlap_voxel_count: int
    coverage_voxel_count: int
    min_overlap_warn: float | None = None
    min_overlap_fail: float | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformCoverageIntersectionQcRow:
    """QC row for applying configured coverage masks to transformed ROIs."""

    source_index: int
    mask_path: str
    status: str
    applied_mask_names: tuple[str, ...]
    applied_mask_paths: tuple[str, ...]
    original_voxel_count: int
    retained_voxel_count: int
    dropped_voxel_count: int
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformJsonWriteRow:
    """One optional planned JSON artifact write."""

    source_index: int
    artifact_kind: str
    path: str
    status: str
    bytes_written: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformExecutionJobRow:
    """Skipped, failed, or completed execution status for one transform job."""

    source_index: int
    status: str
    message: str
    output_mask_path: str | None = None
    qc_path: str | None = None
    provenance_path: str | None = None
    command_status: str | None = None
    qc_status: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiTransformExecutionValidationResult:
    """Side-effect-free validation result for a transform execution plan."""

    status: str
    executable_jobs: tuple[ExecutableMniToT1wRoiTransformJob, ...] = ()
    skipped_jobs: tuple[RoiTransformExecutionJobRow, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


@dataclass(frozen=True)
class RoiTransformExecutionResult:
    """Top-level JSON-safe execution and QC result."""

    status: str
    executed: bool = True
    plan_only: bool = False
    overwrite: bool = False
    validation: RoiTransformExecutionValidationResult | None = None
    executable_jobs: tuple[ExecutableMniToT1wRoiTransformJob, ...] = ()
    command_records: tuple[RoiTransformCommandExecutionRecord, ...] = ()
    output_verifications: tuple[TransformedRoiOutputVerificationRow, ...] = ()
    geometry_qc: tuple[RoiTransformGeometryQcRow, ...] = ()
    binary_mask_qc: tuple[RoiTransformBinaryMaskQcRow, ...] = ()
    voxel_count_qc: tuple[RoiTransformVoxelCountQcRow, ...] = ()
    coverage_intersection_qc: tuple[RoiTransformCoverageIntersectionQcRow, ...] = ()
    coverage_overlap_qc: tuple[RoiTransformCoverageOverlapQcRow, ...] = ()
    json_writes: tuple[RoiTransformJsonWriteRow, ...] = ()
    job_rows: tuple[RoiTransformExecutionJobRow, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


@dataclass(frozen=True)
class _PathState:
    path: str | None
    root_ref: str | None = None
    exists: bool | None = None
    status: str = "preview_only"
    message: str | None = None


@dataclass(frozen=True)
class _PlannedUnit:
    source: RoiTransformSourceMaskRow
    target: TargetReferenceImageRow
    transforms: tuple[TransformChainSpecRow, ...]
    output: PlannedTransformedRoiOutputRow
    coverage_masks: tuple["_CoverageMaskState", ...]
    interpolation: str
    dimension: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class _RoiTransformQcRuntime:
    load_nifti_image: Any
    validate_compatible_geometry: Any
    validate_binary_mask: Any
    check_min_voxel_count: Any
    apply_coverage_masks: Any
    write_roi_nifti_mask: Any


@dataclass(frozen=True)
class _CoverageMaskState:
    name: str
    path: str
    exists: bool | None
    status: str
    message: str | None


def check_transform_tool_availability(
    executable: str | os.PathLike[str] | None = None,
    *,
    tool_name: str = ANTS_APPLY_TRANSFORMS,
    missing_policy: str = MISSING_POLICY_WARN,
) -> TransformToolPreflightRow:
    """Return a plan-only availability row for a transform executable.

    When ``executable`` is omitted, the tool is resolved with ``shutil.which``.
    When it is supplied, that exact configured executable path or command name
    is checked and recorded. No packages are installed and no command is run.
    """

    policy = _normalize_missing_policy(missing_policy)
    requested = _optional_text(executable)
    severity = "error" if policy == MISSING_POLICY_FAIL else "warning"
    if requested is not None:
        if _looks_like_path(requested):
            candidate = Path(requested).expanduser()
            exists = candidate.is_file()
            executable_ok = exists and os.access(candidate, os.X_OK)
            resolved = candidate.as_posix() if exists else requested
            if executable_ok:
                return TransformToolPreflightRow(
                    tool_name=tool_name,
                    requested_executable=requested,
                    resolved_executable=resolved,
                    available=True,
                    status="ok",
                    check_method="configured_path",
                    message=f"Configured {tool_name} executable is available.",
                    severity="info",
                )
            reason = "not executable" if exists else "not found"
            return TransformToolPreflightRow(
                tool_name=tool_name,
                requested_executable=requested,
                resolved_executable=resolved if exists else None,
                available=False,
                status=severity,
                check_method="configured_path",
                message=f"Configured {tool_name} executable was {reason}: {requested}",
                severity=severity,
            )

        resolved = shutil.which(requested)
        if resolved:
            return TransformToolPreflightRow(
                tool_name=tool_name,
                requested_executable=requested,
                resolved_executable=resolved,
                available=True,
                status="ok",
                check_method="configured_command",
                message=f"Configured {tool_name} command is available on PATH.",
                severity="info",
            )
        return TransformToolPreflightRow(
            tool_name=tool_name,
            requested_executable=requested,
            resolved_executable=None,
            available=False,
            status=severity,
            check_method="configured_command",
            message=f"Configured {tool_name} command was not found on PATH: {requested}",
            severity=severity,
        )

    resolved = shutil.which(tool_name)
    if resolved:
        return TransformToolPreflightRow(
            tool_name=tool_name,
            requested_executable=None,
            resolved_executable=resolved,
            available=True,
            status="ok",
            check_method="PATH",
            message=f"{tool_name} is available on PATH.",
            severity="info",
        )
    return TransformToolPreflightRow(
        tool_name=tool_name,
        requested_executable=None,
        resolved_executable=None,
        available=False,
        status=severity,
        check_method="PATH",
        message=f"{tool_name} was not found on PATH. Install ANTs separately or configure an executable path.",
        severity=severity,
    )


def preflight_ants_apply_transforms(
    executable: str | os.PathLike[str] | None = None,
    *,
    missing_policy: str = MISSING_POLICY_WARN,
) -> TransformToolPreflightRow:
    """Return an availability preflight row for ``antsApplyTransforms``."""

    return check_transform_tool_availability(
        executable,
        tool_name=ANTS_APPLY_TRANSFORMS,
        missing_policy=missing_policy,
    )


def build_ants_apply_transforms_argv(
    *,
    input_mask: str | os.PathLike[str],
    reference_image: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    transforms: Sequence[str | os.PathLike[str] | Mapping[str, Any] | TransformChainSpecRow],
    executable: str | os.PathLike[str] = ANTS_APPLY_TRANSFORMS,
    dimension: int = DEFAULT_DIMENSION,
    interpolation: str = DEFAULT_INTERPOLATION,
) -> tuple[str, ...]:
    """Build a shell-safe ``antsApplyTransforms`` argv vector for an ROI mask."""

    argv: list[str] = [
        str(executable),
        "-d",
        str(int(dimension)),
        "-i",
        str(input_mask),
        "-r",
        str(reference_image),
        "-o",
        str(output_path),
        "-n",
        str(interpolation),
    ]
    for transform in transforms:
        path, invert = _transform_path_and_invert(transform)
        if path is None:
            continue
        argv.extend(("-t", _format_transform_arg(path, invert=invert)))
    return tuple(argv)


def validate_mni_to_t1w_roi_transform_document(document: Mapping[str, Any] | Any) -> list[str]:
    """Validate the portable structure of a transform configuration.

    This deliberately does not resolve named roots, inspect the filesystem, or
    look up ANTs.  Those are execution-readiness concerns handled by planning
    and doctor checks.
    """

    if not isinstance(document, Mapping):
        return ["ROI transform set document must contain a mapping."]

    wrapper_keys = ("roi_transform_plan", "roi_transforms", "mni_to_t1w_roi_transforms")
    declared_wrappers = [key for key in wrapper_keys if key in document]
    if len(declared_wrappers) > 1:
        return ["ROI transform set document must define only one transform payload wrapper."]
    if declared_wrappers:
        wrapper = declared_wrappers[0]
        payload = document.get(wrapper)
        if not isinstance(payload, Mapping):
            return [f"{wrapper} must contain a mapping."]
        label = wrapper
    else:
        payload = document
        label = "roi_transform_plan"

    errors: list[str] = []
    for key in ("roots", "defaults", "qc", "tool", "ants", "outputs", "selectors", "selector"):
        if key in payload and payload.get(key) is not None and not isinstance(payload.get(key), Mapping):
            errors.append(f"{label}.{key} must contain a mapping.")

    roots = payload.get("roots")
    if isinstance(roots, Mapping):
        for root_ref, root_path in roots.items():
            if not str(root_ref).strip():
                errors.append(f"{label}.roots keys must be non-empty named-root identifiers.")
            _validate_path_spec(root_path, f"{label}.roots.{root_ref}", errors)

    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), Mapping) else {}
    tool = payload.get("tool") if isinstance(payload.get("tool"), Mapping) else {}
    ants = payload.get("ants") if isinstance(payload.get("ants"), Mapping) else {}
    for section_name, section in ((label, payload), (f"{label}.defaults", defaults)):
        for key in ("missing_policy", "tool_missing_policy"):
            value = section.get(key)
            if value is not None and str(value).strip().lower() not in MISSING_POLICIES:
                errors.append(f"{section_name}.{key} must be one of: {', '.join(sorted(MISSING_POLICIES))}.")

    for section_name, section in (
        (label, payload),
        (f"{label}.defaults", defaults),
        (f"{label}.tool", tool),
        (f"{label}.ants", ants),
    ):
        if "dimension" in section:
            try:
                dimension = int(section.get("dimension"))
            except (TypeError, ValueError):
                errors.append(f"{section_name}.dimension must be a positive integer.")
            else:
                if dimension <= 0:
                    errors.append(f"{section_name}.dimension must be a positive integer.")
        interpolation = section.get("interpolation") or section.get("default_interpolation")
        if interpolation is not None and not _is_nearest_neighbor(interpolation):
            errors.append(
                f"{section_name}.interpolation must use nearest-neighbor interpolation for ROI masks."
            )

    source_key = next(
        (key for key in ("source_masks", "roi_handoff", "roi_build_outputs") if payload.get(key) is not None),
        None,
    )
    if source_key is None:
        errors.append(f"{label} must define source_masks, roi_handoff, or roi_build_outputs.")
        return _unique_text(errors)

    raw_sources = payload.get(source_key)
    source_rows = _validate_transform_row_container(raw_sources, f"{label}.{source_key}", errors)
    target_rows = _validate_transform_row_container(
        payload.get("target_references"),
        f"{label}.target_references",
        errors,
        required=False,
    )
    transform_rows = _validate_transform_row_container(
        payload.get("transform_chains"),
        f"{label}.transform_chains",
        errors,
        required=False,
    )

    outputs = payload.get("outputs")
    needs_common_output = False
    for index, source in enumerate(source_rows):
        source_label = f"{label}.{source_key}[{index}]"
        _validate_declared_path(source, _SOURCE_PATH_KEYS, f"{source_label}.source_mask_path", errors)

        embedded_targets = any(source.get(key) is not None for key in _TARGET_PATH_KEYS)
        if not embedded_targets and not target_rows:
            errors.append(
                f"{source_label} must define a target reference path or use {label}.target_references."
            )
        elif embedded_targets:
            _validate_declared_path(source, _TARGET_PATH_KEYS, f"{source_label}.target_reference_path", errors)

        embedded_transforms = source.get("transforms") or source.get("transform_chain")
        if embedded_transforms is None and not transform_rows:
            errors.append(
                f"{source_label} must define a transform chain or use {label}.transform_chains."
            )
        elif embedded_transforms is not None:
            if not isinstance(embedded_transforms, Sequence) or isinstance(
                embedded_transforms, (str, bytes, bytearray)
            ):
                errors.append(f"{source_label}.transforms must contain a sequence of mappings.")
                embedded_rows = ()
            else:
                embedded_rows = _validate_transform_row_container(
                    embedded_transforms,
                    f"{source_label}.transforms",
                    errors,
                )
            for transform_index, transform in enumerate(embedded_rows):
                _validate_declared_path(
                    transform,
                    ("transform_path", "path", "transform", "filename"),
                    f"{source_label}.transforms[{transform_index}].path",
                    errors,
                )

        has_source_output = any(source.get(key) is not None for key in _OUTPUT_PATH_KEYS)
        outputs = payload.get("outputs")
        has_common_output = isinstance(outputs, Mapping) and bool(outputs)
        if not has_source_output and not has_common_output:
            errors.append(
                f"{source_label} must define a planned output path or use {label}.outputs."
            )
        elif has_source_output:
            _validate_declared_path(source, _OUTPUT_PATH_KEYS, f"{source_label}.planned_output_mask_path", errors)
        else:
            needs_common_output = True

    for index, target in enumerate(target_rows):
        _validate_declared_path(
            target,
            _TARGET_PATH_KEYS,
            f"{label}.target_references[{index}].target_reference_path",
            errors,
        )
    for index, transform in enumerate(transform_rows):
        _validate_declared_path(
            transform,
            ("transform_path", "path", "transform", "filename"),
            f"{label}.transform_chains[{index}].path",
            errors,
        )

    if needs_common_output and isinstance(outputs, Mapping) and outputs:
        mask_spec = outputs.get("mask") if isinstance(outputs.get("mask"), Mapping) else outputs
        _validate_path_spec(mask_spec, f"{label}.outputs", errors)

    return _unique_text(errors)


def plan_mni_to_t1w_roi_transforms(
    transform_config: Mapping[str, Any] | None = None,
    *,
    roots: Mapping[str, str | Path] | None = None,
    source_masks: Sequence[Mapping[str, Any] | RoiTransformSourceMaskRow] | None = None,
    target_references: Sequence[Mapping[str, Any] | TargetReferenceImageRow] | Mapping[str, Any] | None = None,
    transform_chains: Sequence[Mapping[str, Any] | TransformChainSpecRow] | Mapping[str, Any] | None = None,
    roi_handoff: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    roi_build_outputs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    tool: Mapping[str, Any] | TransformToolSpecRow | None = None,
    missing_policy: str | None = None,
    tool_missing_policy: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> RoiTransformPlan:
    """Plan MNI ROI mask transforms into subject T1w/reference space.

    The planner inventories paths, checks resolvable prerequisites, renders ANTs
    argv vectors, and previews QC rows. It never executes commands or writes
    transformed masks, QC outputs, extraction tables, summaries, figures, or
    reports.
    """

    config = _payload_mapping(transform_config)
    roots = roots or _mapping_or_none(config.get("roots"))
    defaults = _mapping_or_empty(config.get("defaults"))
    qc_config = _mapping_or_empty(config.get("qc"))
    policy = _normalize_missing_policy(missing_policy or _optional_text(config.get("missing_policy")) or _optional_text(defaults.get("missing_policy")))
    tool_policy = _normalize_missing_policy(
        tool_missing_policy
        or _optional_text(config.get("tool_missing_policy"))
        or _optional_text(defaults.get("tool_missing_policy"))
        or MISSING_POLICY_WARN
    )
    tool_spec = _tool_spec(tool or _mapping_or_none(config.get("tool")) or _mapping_or_none(config.get("ants")))
    dimension = _int_value(config.get("dimension") or defaults.get("dimension") or tool_spec.dimension, default=DEFAULT_DIMENSION)
    interpolation = (
        _optional_text(config.get("interpolation"))
        or _optional_text(defaults.get("interpolation"))
        or tool_spec.default_interpolation
        or DEFAULT_INTERPOLATION
    )
    outputs_config = _mapping_or_empty(config.get("outputs"))

    raw_sources = _expand_selector_source_rows(config, _collect_source_rows(config, source_masks, roi_handoff, roi_build_outputs))
    raw_targets = _collect_config_rows(target_references if target_references is not None else config.get("target_references"))
    raw_chains = _collect_config_rows(transform_chains if transform_chains is not None else config.get("transform_chains"))

    warnings: list[str] = []
    errors: list[str] = []
    provenance_rows: list[RoiTransformProvenanceRow] = _root_ref_rows(roots)
    tool_specs = (
        TransformToolSpecRow(
            tool_name=tool_spec.tool_name,
            executable=tool_spec.executable,
            dimension=dimension,
            default_interpolation=interpolation,
            lookup_on_path=tool_spec.lookup_on_path,
            status=tool_spec.status,
            message=tool_spec.message,
        ),
    )
    preflight = check_transform_tool_availability(
        tool_spec.executable,
        tool_name=tool_spec.tool_name,
        missing_policy=tool_policy,
    )
    if preflight.status == "error":
        errors.append(preflight.message)
    elif preflight.status == "warning":
        warnings.append(preflight.message)

    if not raw_sources:
        errors.append("No source ROI masks were supplied for MNI-to-reference transform planning.")

    units: list[_PlannedUnit] = []
    for source_index, raw_source in enumerate(raw_sources):
        source_mapping = _row_mapping(raw_source)
        identifiers = _identifiers(source_mapping)
        unit_warnings: list[str] = []
        unit_errors: list[str] = []

        source_state = _resolve_path_from_keys(source_mapping, _SOURCE_PATH_KEYS, roots=roots)
        source_status, source_warnings, source_errors = _status_for_path(
            source_state,
            "source MNI ROI mask",
            policy=policy,
            required=True,
        )
        unit_warnings.extend(source_warnings)
        unit_errors.extend(source_errors)
        source_row = RoiTransformSourceMaskRow(
            source_index=source_index,
            source_mask_path=source_state.path,
            source_space=_optional_text(source_mapping.get("source_space") or source_mapping.get("space")) or "MNI",
            root_ref=source_state.root_ref,
            exists=source_state.exists,
            status=source_status,
            warnings=tuple(source_warnings),
            errors=tuple(source_errors),
            **identifiers,
        )

        target_mapping = _match_row(source_mapping, raw_targets, source_index=source_index)
        target_resolution_mapping = {**source_mapping, **target_mapping}
        target_state = _resolve_target_path(source_mapping, target_resolution_mapping, roots=roots)
        target_status, target_warnings, target_errors = _status_for_path(
            target_state,
            "target reference image",
            policy=policy,
            required=True,
        )
        unit_warnings.extend(target_warnings)
        unit_errors.extend(target_errors)
        target_identifiers = {**identifiers, **_identifiers(target_mapping)}
        target_row = TargetReferenceImageRow(
            source_index=source_index,
            target_reference_path=target_state.path,
            target_space=_optional_text(target_mapping.get("target_space") or source_mapping.get("target_space")) or "T1w",
            root_ref=target_state.root_ref,
            exists=target_state.exists,
            status=target_status,
            warnings=tuple(target_warnings),
            errors=tuple(target_errors),
            **target_identifiers,
        )

        transform_specs = _raw_transform_specs(source_mapping, raw_chains, source_index=source_index)
        transform_rows: list[TransformChainSpecRow] = []
        if not transform_specs:
            message = "No transform chain was supplied; transforms are not invented by the planner."
            unit_errors.append(message)
        for order_index, transform_spec in enumerate(transform_specs):
            raw_transform_mapping = _row_mapping(transform_spec)
            transform_mapping = {**source_mapping, **raw_transform_mapping}
            transform_state = _resolve_transform_path(transform_mapping, roots=roots)
            transform_status, transform_warnings, transform_errors = _status_for_path(
                transform_state,
                f"transform chain item {order_index}",
                policy=policy,
                required=True,
            )
            unit_warnings.extend(transform_warnings)
            unit_errors.extend(transform_errors)
            transform_identifiers = {**identifiers, **_identifiers(raw_transform_mapping)}
            transform_row = TransformChainSpecRow(
                source_index=source_index,
                order_index=order_index,
                transform_path=transform_state.path,
                invert=_bool_value(transform_mapping.get("invert") or transform_mapping.get("inverted")),
                transform_type=_optional_text(transform_mapping.get("type") or transform_mapping.get("transform_type")),
                root_ref=transform_state.root_ref,
                exists=transform_state.exists,
                status=transform_status,
                policy=policy,
                warnings=tuple(transform_warnings),
                errors=tuple(transform_errors),
                **transform_identifiers,
            )
            transform_rows.append(transform_row)

        output_state = _resolve_output_path(source_mapping, outputs_config, roots=roots, identifiers=identifiers)
        qc_state = _resolve_optional_output_artifact_path(
            source_mapping,
            outputs_config,
            keys=_QC_OUTPUT_PATH_KEYS,
            nested_keys=("qc", "quality_control"),
            roots=roots,
            identifiers=identifiers,
        )
        provenance_state = _resolve_optional_output_artifact_path(
            source_mapping,
            outputs_config,
            keys=_PROVENANCE_OUTPUT_PATH_KEYS,
            nested_keys=("provenance", "sidecar"),
            roots=roots,
            identifiers=identifiers,
        )
        output_status, output_warnings, output_errors = _status_for_output(output_state)
        unit_warnings.extend(output_warnings)
        unit_errors.extend(output_errors)
        output_row = PlannedTransformedRoiOutputRow(
            source_index=source_index,
            output_mask_path=output_state.path,
            output_space=target_row.target_space,
            root_ref=output_state.root_ref,
            status=output_status,
            warnings=tuple(output_warnings),
            errors=tuple(output_errors),
            qc_path=qc_state.path,
            provenance_path=provenance_state.path,
            **identifiers,
        )

        coverage_masks: list[_CoverageMaskState] = []
        for coverage_name, coverage_state in _resolve_coverage_paths(source_mapping, qc_config, roots=roots):
            coverage_status, coverage_warnings, coverage_errors = _status_for_path(
                coverage_state,
                f"coverage or brain mask {coverage_name}",
                policy=policy,
                required=False,
            )
            unit_warnings.extend(coverage_warnings)
            unit_errors.extend(coverage_errors)
            if coverage_state.path is not None:
                coverage_masks.append(
                    _CoverageMaskState(
                        name=coverage_name,
                        path=coverage_state.path,
                        exists=coverage_state.exists,
                        status=coverage_status,
                        message=coverage_state.message,
                    )
                )

        provenance_rows.append(
            RoiTransformProvenanceRow(
                provenance_kind="transform_chain",
                source_index=source_index,
                status="planned" if transform_rows else "error",
                message=f"{len(transform_rows)} transform(s) supplied in configured order.",
                fields={
                    "transform_count": len(transform_rows),
                    "inverted_count": sum(1 for row in transform_rows if row.invert),
                },
            )
        )
        units.append(
            _PlannedUnit(
                source=source_row,
                target=target_row,
                transforms=tuple(transform_rows),
                output=output_row,
                coverage_masks=tuple(coverage_masks),
                interpolation=interpolation,
                dimension=dimension,
                warnings=tuple(_unique_text(unit_warnings)),
                errors=tuple(_unique_text(unit_errors)),
            )
        )
        warnings.extend(unit_warnings)
        errors.extend(unit_errors)

    source_rows = tuple(unit.source for unit in units)
    target_rows = tuple(unit.target for unit in units)
    transform_rows = tuple(row for unit in units for row in unit.transforms)
    output_rows = tuple(unit.output for unit in units)
    command_rows = tuple(
        _command_row(
            unit,
            executable=preflight.resolved_executable or tool_spec.executable or tool_spec.tool_name,
            tool_name=tool_spec.tool_name,
        )
        for unit in units
    )
    qc_rows = tuple(row for unit in units for row in _qc_rows(unit, qc_config=qc_config))

    warnings = _unique_text(warnings)
    errors = _unique_text(errors)
    status = _aggregate_status(
        errors=errors,
        warnings=warnings,
        rows=(*source_rows, *target_rows, *transform_rows, *output_rows, *command_rows, *qc_rows, preflight),
    )
    return RoiTransformPlan(
        status=status,
        source_masks=source_rows,
        target_references=target_rows,
        transform_chains=transform_rows,
        tool_specs=tool_specs,
        tool_preflight=(preflight,),
        command_plans=command_rows,
        planned_outputs=output_rows,
        qc_preview=qc_rows,
        provenance_rows=tuple(provenance_rows),
        warnings=tuple(warnings),
        errors=tuple(errors),
        context=dict(context or {}),
        executed=False,
        plan_only=True,
    )


def validate_mni_to_t1w_roi_transform_execution_plan(
    plan: RoiTransformPlan | Mapping[str, Any],
    *,
    overwrite: bool = False,
    min_voxels_warn: int | None = None,
    min_voxels_fail: int | None = None,
    coverage_min_overlap_fraction_warn: float | None = None,
    coverage_min_overlap_fraction_fail: float | None = None,
) -> RoiTransformExecutionValidationResult:
    """Validate an already-planned MNI-to-T1w ROI transform plan for execution.

    Validation is side-effect free: it does not create directories, run ANTs, or
    write JSON. The executor calls this first and refuses to mutate on errors.
    """

    try:
        normalized_plan = _coerce_roi_transform_plan(plan)
    except (TypeError, ValueError) as error:
        return RoiTransformExecutionValidationResult(status="error", errors=(str(error),))

    warnings: list[str] = list(normalized_plan.warnings)
    errors: list[str] = []
    if normalized_plan.status == "error" or not normalized_plan.valid:
        errors.append("ROI transform plan has error status and cannot be executed.")
    if normalized_plan.executed or not normalized_plan.plan_only:
        errors.append("Execution requires a plan-only RoiTransformPlan, not an executed result.")

    jobs, skipped_jobs, selection_warnings, selection_errors = _select_jobs_with_diagnostics(
        normalized_plan,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
        coverage_min_overlap_fraction_warn=coverage_min_overlap_fraction_warn,
        coverage_min_overlap_fraction_fail=coverage_min_overlap_fraction_fail,
    )
    warnings.extend(selection_warnings)
    errors.extend(selection_errors)

    executable_errors: list[str] = []
    overwrite_errors: list[str] = []
    overwrite_errors.extend(
        _complete_transform_destination_errors(normalized_plan, jobs, overwrite=overwrite)
    )
    for job in jobs:
        executable_error = _execution_executable_error(job.argv[0] if job.argv else None)
        if executable_error is not None:
            executable_errors.append(f"source_index {job.source_index}: {executable_error}")

    errors.extend(executable_errors)
    errors.extend(overwrite_errors)
    if not jobs and not errors:
        errors.append("No executable planned MNI-to-T1w ROI transform jobs were selected.")

    status = "error" if errors else "warning" if warnings else "ok"
    return RoiTransformExecutionValidationResult(
        status=status,
        executable_jobs=jobs if status != "error" else (),
        skipped_jobs=tuple(skipped_jobs),
        warnings=tuple(_unique_text(warnings)),
        errors=tuple(_unique_text(errors)),
    )


def select_executable_mni_to_t1w_roi_transform_jobs(
    plan: RoiTransformPlan | Mapping[str, Any],
    *,
    min_voxels_warn: int | None = None,
    min_voxels_fail: int | None = None,
    coverage_min_overlap_fraction_warn: float | None = None,
    coverage_min_overlap_fraction_fail: float | None = None,
) -> tuple[ExecutableMniToT1wRoiTransformJob, ...]:
    """Return executable planned transform jobs without mutating the filesystem."""

    try:
        normalized_plan = _coerce_roi_transform_plan(plan)
    except (TypeError, ValueError):
        return ()
    jobs, _, _, _ = _select_jobs_with_diagnostics(
        normalized_plan,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
        coverage_min_overlap_fraction_warn=coverage_min_overlap_fraction_warn,
        coverage_min_overlap_fraction_fail=coverage_min_overlap_fraction_fail,
    )
    return jobs


def execute_mni_to_t1w_roi_transform_plan(
    plan: RoiTransformPlan | Mapping[str, Any],
    *,
    runner: Any | None = None,
    overwrite: bool = False,
    min_voxels_warn: int | None = None,
    min_voxels_fail: int | None = None,
    coverage_min_overlap_fraction_warn: float | None = None,
    coverage_min_overlap_fraction_fail: float | None = None,
) -> RoiTransformExecutionResult:
    """Execute already-planned MNI-to-T1w ROI transforms and run actual ROI QC.

    This is the explicit mutation gate. It validates the supplied plan first,
    creates only planned parent directories, runs exactly the planned argv
    tuples through ``runner``, verifies transformed outputs, runs ROI QC, and
    writes optional planned QC/provenance JSON artifacts.
    """

    validation = validate_mni_to_t1w_roi_transform_execution_plan(
        plan,
        overwrite=overwrite,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
        coverage_min_overlap_fraction_warn=coverage_min_overlap_fraction_warn,
        coverage_min_overlap_fraction_fail=coverage_min_overlap_fraction_fail,
    )
    if not validation.valid:
        return RoiTransformExecutionResult(
            status="error",
            overwrite=overwrite,
            validation=validation,
            job_rows=validation.skipped_jobs,
            warnings=validation.warnings,
            errors=validation.errors,
        )

    command_runner = runner or _default_transform_runner
    _create_planned_parent_directories(validation.executable_jobs)

    command_records: list[RoiTransformCommandExecutionRecord] = []
    output_verifications: list[TransformedRoiOutputVerificationRow] = []
    geometry_rows: list[RoiTransformGeometryQcRow] = []
    binary_rows: list[RoiTransformBinaryMaskQcRow] = []
    voxel_rows: list[RoiTransformVoxelCountQcRow] = []
    coverage_intersection_rows: list[RoiTransformCoverageIntersectionQcRow] = []
    coverage_rows: list[RoiTransformCoverageOverlapQcRow] = []
    json_write_rows: list[RoiTransformJsonWriteRow] = []
    job_rows: list[RoiTransformExecutionJobRow] = []
    warnings: list[str] = list(validation.warnings)
    errors: list[str] = []

    for job in validation.executable_jobs:
        command_record = _run_transform_command(job, command_runner)
        command_records.append(command_record)
        if command_record.status != "completed":
            message = command_record.error or command_record.stderr_summary or "Transform command failed."
            errors.append(f"source_index {job.source_index}: {message}")
            job_rows.append(
                RoiTransformExecutionJobRow(
                    source_index=job.source_index,
                    status="failed",
                    message=message,
                    output_mask_path=job.output_mask_path,
                    qc_path=job.qc_path,
                    provenance_path=job.provenance_path,
                    command_status=command_record.status,
                    errors=(message,),
                )
            )
            continue

        intersection_rows = _apply_coverage_intersection(job)
        coverage_intersection_rows.extend(intersection_rows)
        if any(row.status == "error" for row in intersection_rows):
            message = "; ".join(row.message or "Coverage-mask intersection failed." for row in intersection_rows if row.status == "error")
            errors.append(f"source_index {job.source_index}: {message}")
            job_rows.append(
                RoiTransformExecutionJobRow(
                    source_index=job.source_index,
                    status="failed",
                    message=message,
                    output_mask_path=job.output_mask_path,
                    qc_path=job.qc_path,
                    provenance_path=job.provenance_path,
                    command_status=command_record.status,
                    errors=(message,),
                )
            )
            continue

        verification = _verify_transformed_output(job)
        output_verifications.append(verification)
        if verification.status == "error":
            errors.append(f"source_index {job.source_index}: {verification.message}")
            job_rows.append(
                RoiTransformExecutionJobRow(
                    source_index=job.source_index,
                    status="failed",
                    message=verification.message,
                    output_mask_path=job.output_mask_path,
                    qc_path=job.qc_path,
                    provenance_path=job.provenance_path,
                    command_status=command_record.status,
                    errors=(verification.message,),
                )
            )
            continue

        qc_result = dict(_run_actual_transform_qc(job))
        qc_result["coverage_intersection_qc"] = tuple(intersection_rows)
        geometry_rows.extend(qc_result["geometry_qc"])
        binary_rows.extend(qc_result["binary_mask_qc"])
        voxel_rows.extend(qc_result["voxel_count_qc"])
        coverage_rows.extend(qc_result["coverage_overlap_qc"])
        qc_status = _qc_status(
            (
                *intersection_rows,
                *qc_result["geometry_qc"],
                *qc_result["binary_mask_qc"],
                *qc_result["voxel_count_qc"],
                *qc_result["coverage_overlap_qc"],
            )
        )
        if qc_status == "warning":
            warnings.append(f"source_index {job.source_index}: transformed ROI QC completed with warnings.")
        elif qc_status == "error":
            errors.append(f"source_index {job.source_index}: transformed ROI QC failed.")

        planned_writes = _write_planned_execution_json(job, command_record, verification, qc_result)
        json_write_rows.extend(planned_writes)
        write_errors = tuple(row.error for row in planned_writes if row.status == "error" and row.error)
        errors.extend(f"source_index {job.source_index}: {error}" for error in write_errors)

        job_status = "failed" if qc_status == "error" or write_errors else "completed"
        job_warning_rows = tuple(
            row.message
            for row in (*qc_result["voxel_count_qc"], *qc_result["coverage_overlap_qc"])
            if getattr(row, "status", None) == "warning" and getattr(row, "message", None)
        )
        job_rows.append(
            RoiTransformExecutionJobRow(
                source_index=job.source_index,
                status=job_status,
                message="Transform execution and ROI QC completed." if job_status == "completed" else "Transform execution completed but QC or JSON writing failed.",
                output_mask_path=job.output_mask_path,
                qc_path=job.qc_path,
                provenance_path=job.provenance_path,
                command_status=command_record.status,
                qc_status=qc_status,
                warnings=job_warning_rows,
                errors=tuple(str(error) for error in write_errors) if write_errors else (),
            )
        )

    status = "error" if errors or any(row.status == "failed" for row in job_rows) else "warning" if warnings else "ok"
    return RoiTransformExecutionResult(
        status=status,
        overwrite=overwrite,
        validation=validation,
        executable_jobs=validation.executable_jobs,
        command_records=tuple(command_records),
        output_verifications=tuple(output_verifications),
        geometry_qc=tuple(geometry_rows),
        binary_mask_qc=tuple(binary_rows),
        voxel_count_qc=tuple(voxel_rows),
        coverage_intersection_qc=tuple(coverage_intersection_rows),
        coverage_overlap_qc=tuple(coverage_rows),
        json_writes=tuple(json_write_rows),
        job_rows=tuple(job_rows),
        warnings=tuple(_unique_text(warnings)),
        errors=tuple(_unique_text(errors)),
    )


def _coerce_roi_transform_plan(raw: RoiTransformPlan | Mapping[str, Any]) -> RoiTransformPlan:
    if isinstance(raw, RoiTransformPlan):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("ROI transform execution input must be a RoiTransformPlan or plan-compatible mapping.")
    payload = _payload_mapping(raw)
    if "command_plans" not in payload and "source_masks" not in payload:
        raise ValueError("ROI transform execution input is not a plan-compatible payload.")
    return RoiTransformPlan(
        status=_optional_text(payload.get("status")) or "error",
        executed=_bool_value(payload.get("executed")),
        plan_only=True if payload.get("plan_only") is None else _bool_value(payload.get("plan_only")),
        source_masks=tuple(_coerce_row(RoiTransformSourceMaskRow, row) for row in _collect_config_rows(payload.get("source_masks"))),
        target_references=tuple(_coerce_row(TargetReferenceImageRow, row) for row in _collect_config_rows(payload.get("target_references"))),
        transform_chains=tuple(_coerce_row(TransformChainSpecRow, row) for row in _collect_config_rows(payload.get("transform_chains"))),
        tool_specs=tuple(_coerce_row(TransformToolSpecRow, row) for row in _collect_config_rows(payload.get("tool_specs"))),
        tool_preflight=tuple(_coerce_row(TransformToolPreflightRow, row) for row in _collect_config_rows(payload.get("tool_preflight"))),
        command_plans=tuple(_coerce_row(TransformCommandPlanRow, row) for row in _collect_config_rows(payload.get("command_plans"))),
        planned_outputs=tuple(_coerce_output_row(row) for row in _collect_config_rows(payload.get("planned_outputs"))),
        qc_preview=tuple(_coerce_row(RoiQcPreviewRow, row) for row in _collect_config_rows(payload.get("qc_preview"))),
        provenance_rows=tuple(_coerce_row(RoiTransformProvenanceRow, row) for row in _collect_config_rows(payload.get("provenance_rows"))),
        warnings=tuple(str(value) for value in _collect_scalar_sequence(payload.get("warnings"))),
        errors=tuple(str(value) for value in _collect_scalar_sequence(payload.get("errors"))),
        context=_mapping_or_empty(payload.get("context")),
    )


def _coerce_row(cls: type[Any], raw: Any) -> Any:
    mapping = dict(_row_mapping(raw))
    allowed = {field.name for field in fields(cls)}
    values = {key: mapping[key] for key in allowed if key in mapping}
    for key in ("warnings", "errors"):
        if key in values:
            values[key] = tuple(str(value) for value in _collect_scalar_sequence(values[key]))
    if "argv" in values:
        values["argv"] = tuple(str(value) for value in _collect_scalar_sequence(values["argv"]))
    return cls(**values)


def _coerce_output_row(raw: Any) -> PlannedTransformedRoiOutputRow:
    mapping = dict(_row_mapping(raw))
    if mapping.get("qc_path") is None:
        for key in _QC_OUTPUT_PATH_KEYS:
            if mapping.get(key) is not None:
                mapping["qc_path"] = mapping[key]
                break
    if mapping.get("provenance_path") is None:
        for key in _PROVENANCE_OUTPUT_PATH_KEYS:
            if mapping.get(key) is not None:
                mapping["provenance_path"] = mapping[key]
                break
    return _coerce_row(PlannedTransformedRoiOutputRow, mapping)


def _collect_scalar_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _select_jobs_with_diagnostics(
    plan: RoiTransformPlan,
    *,
    min_voxels_warn: int | None,
    min_voxels_fail: int | None,
    coverage_min_overlap_fraction_warn: float | None,
    coverage_min_overlap_fraction_fail: float | None,
) -> tuple[tuple[ExecutableMniToT1wRoiTransformJob, ...], tuple[RoiTransformExecutionJobRow, ...], list[str], list[str]]:
    source_by_index = {row.source_index: row for row in plan.source_masks}
    target_by_index = {row.source_index: row for row in plan.target_references}
    outputs_by_index = {row.source_index: row for row in plan.planned_outputs}
    transforms_by_index: dict[int, list[TransformChainSpecRow]] = {}
    for transform in plan.transform_chains:
        transforms_by_index.setdefault(transform.source_index, []).append(transform)
    for rows in transforms_by_index.values():
        rows.sort(key=lambda row: row.order_index)

    coverage_paths = _coverage_paths_by_source_index(plan)
    thresholds = _execution_thresholds_by_source_index(
        plan,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
        coverage_min_overlap_fraction_warn=coverage_min_overlap_fraction_warn,
        coverage_min_overlap_fraction_fail=coverage_min_overlap_fraction_fail,
    )
    jobs: list[ExecutableMniToT1wRoiTransformJob] = []
    skipped: list[RoiTransformExecutionJobRow] = []
    warnings: list[str] = []
    errors: list[str] = []

    if not plan.command_plans:
        errors.append("No planned transform command rows were supplied.")
        return (), (), warnings, errors

    for command in plan.command_plans:
        row_errors: list[str] = []
        source = source_by_index.get(command.source_index)
        target = target_by_index.get(command.source_index)
        output = outputs_by_index.get(command.source_index)
        transforms = tuple(transforms_by_index.get(command.source_index, ()))

        if command.status != "planned":
            row_errors.append(f"Command row is not planned: status={command.status}.")
        if not command.argv:
            row_errors.append("Command row has no argv vector.")
        elif not all(isinstance(arg, str) for arg in command.argv):
            row_errors.append("Command argv must contain only string elements.")
        if not _is_nearest_neighbor(command.interpolation):
            row_errors.append(f"ROI transforms require nearest-neighbor interpolation, got {command.interpolation!r}.")
        if source is None:
            row_errors.append("Missing source mask row for command.")
        if target is None:
            row_errors.append("Missing target reference row for command.")
        if output is None:
            row_errors.append("Missing planned output row for command.")
        if not transforms:
            row_errors.append("Missing ordered transform-chain rows for command.")

        if source is not None:
            row_errors.extend(_existing_file_errors("source mask", source.source_mask_path))
        if target is not None:
            row_errors.extend(_existing_file_errors("target reference", target.target_reference_path))
        for transform in transforms:
            row_errors.extend(_existing_file_errors(f"transform chain item {transform.order_index}", transform.transform_path))
        if output is not None:
            row_errors.extend(_planned_output_path_errors(output.output_mask_path))
        coverage_specs = coverage_paths.get(command.source_index, ())
        for coverage_name, coverage_path in coverage_specs:
            row_errors.extend(_existing_file_errors(f"coverage mask {coverage_name}", coverage_path))

        if command.argv and source is not None and target is not None and output is not None:
            row_errors.extend(_command_argv_consistency_errors(command, source, target, transforms, output))

        if row_errors:
            message = "; ".join(row_errors)
            errors.append(f"source_index {command.source_index}: {message}")
            skipped.append(
                RoiTransformExecutionJobRow(
                    source_index=command.source_index,
                    status="skipped",
                    message=message,
                    output_mask_path=output.output_mask_path if output is not None else None,
                    qc_path=output.qc_path if output is not None else None,
                    provenance_path=output.provenance_path if output is not None else None,
                    command_status=command.status,
                    errors=tuple(row_errors),
                )
            )
            continue

        assert source is not None
        assert target is not None
        assert output is not None
        source_thresholds = thresholds.get(command.source_index, {})
        jobs.append(
            ExecutableMniToT1wRoiTransformJob(
                source_index=command.source_index,
                tool_name=command.tool_name,
                argv=tuple(command.argv),
                source_mask_path=str(source.source_mask_path),
                target_reference_path=str(target.target_reference_path),
                output_mask_path=str(output.output_mask_path),
                transform_chain=tuple(_transform_provenance_row(row) for row in transforms),
                interpolation=command.interpolation,
                dimension=command.dimension,
                coverage_mask_path=coverage_specs[0][1] if coverage_specs else None,
                coverage_mask_paths=tuple(path for _name, path in coverage_specs),
                coverage_mask_names=tuple(name for name, _path in coverage_specs),
                coverage_intersection_policy=str(source_thresholds.get("coverage_intersection_policy") or "qc_only"),
                qc_path=output.qc_path,
                provenance_path=output.provenance_path,
                min_voxels_warn=source_thresholds.get("min_voxels_warn"),
                min_voxels_fail=source_thresholds.get("min_voxels_fail"),
                coverage_min_overlap_fraction_warn=source_thresholds.get("coverage_min_overlap_fraction_warn"),
                coverage_min_overlap_fraction_fail=source_thresholds.get("coverage_min_overlap_fraction_fail"),
                subject_id=command.subject_id,
                session_id=command.session_id,
                task_id=command.task_id,
                run_id=command.run_id,
                model=command.model,
                contrast_id=command.contrast_id,
                roi_label=command.roi_label,
                warnings=command.warnings,
                errors=command.errors,
            )
        )
    return tuple(jobs), tuple(skipped), warnings, errors


def _coverage_paths_by_source_index(plan: RoiTransformPlan) -> dict[int, tuple[tuple[str, str], ...]]:
    paths: dict[int, list[tuple[str, str]]] = {}
    for row in plan.qc_preview:
        if row.check_kind == "coverage_mask_exists" and row.path:
            paths.setdefault(row.source_index, []).append((row.expected or "coverage_mask", row.path))
        elif row.check_kind == "post_execution_overlap_coverage" and row.expected:
            paths.setdefault(row.source_index, []).append(("coverage_mask", row.expected))
    deduped: dict[int, tuple[tuple[str, str], ...]] = {}
    for source_index, values in paths.items():
        seen_paths: set[str] = set()
        ordered: list[tuple[str, str]] = []
        for name, path in values:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            ordered.append((name, path))
        deduped[source_index] = tuple(ordered)
    return deduped


def _execution_thresholds_by_source_index(
    plan: RoiTransformPlan,
    *,
    min_voxels_warn: int | None,
    min_voxels_fail: int | None,
    coverage_min_overlap_fraction_warn: float | None,
    coverage_min_overlap_fraction_fail: float | None,
) -> dict[int, dict[str, Any]]:
    context_qc = _mapping_or_empty(plan.context.get("qc") if isinstance(plan.context, Mapping) else None)
    context_min_warn = context_qc.get("min_voxels_warn")
    if context_min_warn is None:
        context_min_warn = context_qc.get("small_mask_warn_voxels")
    context_min_fail = context_qc.get("min_voxels_fail")
    if context_min_fail is None:
        context_min_fail = context_qc.get("small_mask_fail_voxels")
    context_overlap_warn = context_qc.get("coverage_min_overlap_fraction_warn")
    if context_overlap_warn is None:
        context_overlap_warn = context_qc.get("min_coverage_fraction_warn")
    context_overlap_fail = context_qc.get("coverage_min_overlap_fraction_fail")
    if context_overlap_fail is None:
        context_overlap_fail = context_qc.get("min_coverage_fraction_fail")
    context_coverage_policy = _optional_text(context_qc.get("coverage_intersection_policy")) or "qc_only"
    base = {
        "min_voxels_warn": min_voxels_warn if min_voxels_warn is not None else _optional_int(context_min_warn),
        "min_voxels_fail": min_voxels_fail if min_voxels_fail is not None else _optional_int(context_min_fail),
        "coverage_min_overlap_fraction_warn": coverage_min_overlap_fraction_warn
        if coverage_min_overlap_fraction_warn is not None
        else _optional_float(context_overlap_warn),
        "coverage_min_overlap_fraction_fail": coverage_min_overlap_fraction_fail
        if coverage_min_overlap_fraction_fail is not None
        else _optional_float(context_overlap_fail),
        "coverage_intersection_policy": context_coverage_policy,
    }
    thresholds: dict[int, dict[str, Any]] = {}
    for row in plan.qc_preview:
        current = thresholds.setdefault(row.source_index, dict(base))
        if row.check_kind == "post_execution_small_mask":
            parsed = _parse_threshold_spec(row.threshold, integer=True)
            if current.get("min_voxels_warn") is None and parsed.get("warn") is not None:
                current["min_voxels_warn"] = int(parsed["warn"])
            if current.get("min_voxels_fail") is None and parsed.get("fail") is not None:
                current["min_voxels_fail"] = int(parsed["fail"])
        elif row.check_kind == "post_execution_overlap_coverage":
            parsed = _parse_threshold_spec(row.threshold, integer=False)
            if current.get("coverage_min_overlap_fraction_warn") is None and parsed.get("warn") is not None:
                current["coverage_min_overlap_fraction_warn"] = float(parsed["warn"])
            if current.get("coverage_min_overlap_fraction_fail") is None and parsed.get("fail") is not None:
                current["coverage_min_overlap_fraction_fail"] = float(parsed["fail"])
        elif row.check_kind == "coverage_intersection_policy" and row.expected:
            current["coverage_intersection_policy"] = str(row.expected)
    for command in plan.command_plans:
        thresholds.setdefault(command.source_index, dict(base))
    return thresholds


def _parse_threshold_spec(value: Any, *, integer: bool) -> dict[str, int | float | None]:
    parsed: dict[str, int | float | None] = {"warn": None, "fail": None}
    if value is None:
        return parsed
    if isinstance(value, (int, float)):
        parsed["warn"] = int(value) if integer else float(value)
        return parsed
    text = str(value)
    for label in ("warn", "fail"):
        match = re.search(rf"{label}\s*<\s*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            parsed[label] = int(float(match.group(1))) if integer else float(match.group(1))
    return parsed


def _existing_file_errors(label: str, path: str | None) -> list[str]:
    if path is None:
        return [f"{label} path was not supplied."]
    if not _path_is_checkable(path):
        return [f"{label} path is not concrete enough for execution: {path}"]
    if not Path(path).is_file():
        return [f"{label} file is missing: {path}"]
    return []


def _planned_output_path_errors(path: str | None) -> list[str]:
    if path is None:
        return ["planned transformed output path was not supplied."]
    if not _path_is_checkable(path):
        return [f"planned transformed output path is not concrete enough for execution: {path}"]
    return _destination_path_errors(Path(path), label="planned transformed output", overwrite=True)


def _command_argv_consistency_errors(
    command: TransformCommandPlanRow,
    source: RoiTransformSourceMaskRow,
    target: TargetReferenceImageRow,
    transforms: Sequence[TransformChainSpecRow],
    output: PlannedTransformedRoiOutputRow,
) -> list[str]:
    errors: list[str] = []
    argv = tuple(command.argv)
    if _argv_value(argv, "-i") != source.source_mask_path:
        errors.append("planned argv -i does not match the source mask path.")
    if _argv_value(argv, "-r") != target.target_reference_path:
        errors.append("planned argv -r does not match the target reference path.")
    if _argv_value(argv, "-o") != output.output_mask_path:
        errors.append("planned argv -o does not match the transformed output path.")
    if not _is_nearest_neighbor(_argv_value(argv, "-n")):
        errors.append("planned argv -n must use nearest-neighbor interpolation for ROI masks.")
    planned_transform_args = _argv_values(argv, "-t")
    expected_transform_args = tuple(_format_transform_arg(row.transform_path or "", invert=row.invert) for row in transforms)
    if planned_transform_args != expected_transform_args:
        errors.append("planned argv -t transform chain does not match the ordered transform rows.")
    return errors


def _argv_value(argv: Sequence[str], flag: str) -> str | None:
    try:
        index = tuple(argv).index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _argv_values(argv: Sequence[str], flag: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            values.append(argv[index + 1])
    return tuple(values)


def _is_nearest_neighbor(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("_", "").replace("-", "")
    return text in {"nearestneighbor", "nearestneighbour", "nn"}


def _transform_provenance_row(row: TransformChainSpecRow) -> Mapping[str, Any]:
    return {
        "order_index": row.order_index,
        "transform_path": row.transform_path,
        "invert": row.invert,
        "transform_type": row.transform_type,
        "exists": row.exists,
    }


def _execution_executable_error(executable: str | None) -> str | None:
    if not executable:
        return "planned command is missing an executable."
    if _looks_like_path(executable):
        path = Path(executable).expanduser()
        if not path.is_file():
            return f"planned executable path is missing: {executable}"
        if not os.access(path, os.X_OK):
            return f"planned executable path is not executable: {executable}"
        return None
    if shutil.which(executable):
        return None
    return f"{executable} was not found on PATH and no usable configured executable path was planned."


def _complete_transform_destination_errors(
    plan: RoiTransformPlan,
    jobs: Sequence[ExecutableMniToT1wRoiTransformJob],
    *,
    overwrite: bool,
) -> list[str]:
    output_rows = {row.source_index: row for row in plan.planned_outputs}
    root_paths = {
        str(row.root_ref): Path(str(row.path))
        for row in plan.provenance_rows
        if row.provenance_kind == "root_ref" and row.root_ref and row.path
    }
    destinations: list[tuple[str, Path, Path | None]] = []
    for job in jobs:
        output_row = output_rows.get(job.source_index)
        anchor = root_paths.get(str(output_row.root_ref)) if output_row and output_row.root_ref else None
        for label, raw_path in (
            (f"source_index {job.source_index} transformed output", job.output_mask_path),
            (f"source_index {job.source_index} QC JSON", job.qc_path),
            (f"source_index {job.source_index} provenance JSON", job.provenance_path),
        ):
            if raw_path is None:
                continue
            destinations.append((label, Path(raw_path), anchor))

    by_destination: dict[Path, list[str]] = {}
    for label, path, _anchor in destinations:
        normalized = Path(os.path.abspath(os.path.normpath(str(path))))
        by_destination.setdefault(normalized, []).append(label)

    errors: list[str] = []
    for path, labels in sorted(by_destination.items(), key=lambda item: str(item[0])):
        if len(labels) > 1:
            errors.append(
                "duplicate planned ROI transform destination "
                f"{path}: {', '.join(sorted(labels))}"
            )
    for label, path, anchor in destinations:
        if not _path_is_checkable(str(path)):
            errors.append(f"planned {label} path is not concrete enough for execution: {path}")
            continue
        errors.extend(
            _destination_path_errors(
                path,
                label=f"planned {label}",
                overwrite=overwrite,
                anchor=anchor,
            )
        )
    return errors


def _destination_path_errors(
    path: Path,
    *,
    label: str,
    overwrite: bool,
    anchor: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    absolute_path = Path(os.path.abspath(os.path.normpath(str(path))))
    absolute_anchor = (
        Path(os.path.abspath(os.path.normpath(str(anchor))))
        if anchor is not None
        else None
    )
    if absolute_anchor is not None:
        try:
            absolute_path.relative_to(absolute_anchor)
        except ValueError:
            errors.append(f"{label} escapes its configured output root: {path}")
            absolute_anchor = None

    current = absolute_path
    while True:
        if current.is_symlink():
            errors.append(f"{label} uses a symbolic-link destination or parent: {current}")
            break
        if absolute_anchor is not None:
            if current == absolute_anchor:
                break
        elif current != absolute_path and current.exists():
            break
        if current == current.parent:
            break
        current = current.parent

    nearest = absolute_path.parent
    while not nearest.exists() and not nearest.is_symlink() and nearest != nearest.parent:
        nearest = nearest.parent
    if nearest.is_symlink():
        message = f"{label} uses a symbolic-link destination parent: {nearest}"
        if message not in errors:
            errors.append(message)
    elif not nearest.is_dir():
        errors.append(f"{label} parent is not a directory: {nearest}")

    if path.exists() and not path.is_file():
        errors.append(f"{label} destination is not a regular file: {path}")
    elif path.exists() and not overwrite:
        errors.append(f"{label} already exists and overwrite is false: {path}")
    return errors


def _create_planned_parent_directories(jobs: Sequence[ExecutableMniToT1wRoiTransformJob]) -> None:
    for job in jobs:
        for path in (job.output_mask_path, job.qc_path, job.provenance_path):
            if path is not None:
                Path(path).parent.mkdir(parents=True, exist_ok=True)


def _default_transform_runner(argv: Sequence[str]) -> Any:
    return subprocess.run(
        list(argv),
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_transform_command(job: ExecutableMniToT1wRoiTransformJob, runner: Any) -> RoiTransformCommandExecutionRecord:
    try:
        completed = runner(tuple(job.argv))
    except Exception as error:  # pragma: no cover - specific runner exceptions are caller supplied.
        return RoiTransformCommandExecutionRecord(
            source_index=job.source_index,
            argv=tuple(job.argv),
            status="error",
            error=str(error),
        )
    returncode = _completed_returncode(completed)
    stdout = _completed_stream(completed, "stdout")
    stderr = _completed_stream(completed, "stderr")
    if returncode is None:
        return RoiTransformCommandExecutionRecord(
            source_index=job.source_index,
            argv=tuple(job.argv),
            status="error",
            stdout_summary=_summarize_text(stdout),
            stderr_summary=_summarize_text(stderr),
            error="Runner result did not expose a returncode.",
        )
    return RoiTransformCommandExecutionRecord(
        source_index=job.source_index,
        argv=tuple(job.argv),
        status="completed" if returncode == 0 else "failed",
        returncode=returncode,
        stdout_summary=_summarize_text(stdout),
        stderr_summary=_summarize_text(stderr),
        error=None if returncode == 0 else f"Transform command exited with return code {returncode}.",
    )


def _completed_returncode(completed: Any) -> int | None:
    if isinstance(completed, Mapping):
        value = completed.get("returncode")
    else:
        value = getattr(completed, "returncode", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _completed_stream(completed: Any, name: str) -> str | None:
    if isinstance(completed, Mapping):
        value = completed.get(name)
    else:
        value = getattr(completed, name, None)
    if value is None:
        return None
    return str(value)


def _summarize_text(value: str | None, *, limit: int = 2000) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _verify_transformed_output(job: ExecutableMniToT1wRoiTransformJob) -> TransformedRoiOutputVerificationRow:
    exists = Path(job.output_mask_path).is_file()
    return TransformedRoiOutputVerificationRow(
        source_index=job.source_index,
        path=job.output_mask_path,
        exists=exists,
        status="ok" if exists else "error",
        message="Transformed ROI output exists." if exists else f"Transform command succeeded but output is missing: {job.output_mask_path}",
    )


def _apply_coverage_intersection(job: ExecutableMniToT1wRoiTransformJob) -> tuple[RoiTransformCoverageIntersectionQcRow, ...]:
    if job.coverage_intersection_policy != "apply":
        return ()
    coverage_paths = tuple(job.coverage_mask_paths or ((job.coverage_mask_path,) if job.coverage_mask_path else ()))
    if not coverage_paths:
        return ()
    names = tuple(job.coverage_mask_names or tuple(f"coverage_mask_{index + 1}" for index in range(len(coverage_paths))))
    try:
        runtime = _load_roi_transform_qc_runtime()
        output_image = runtime.load_nifti_image(job.output_mask_path)
        output_mask = runtime.validate_binary_mask(output_image.get_fdata(), allow_empty=True, label="transformed ROI mask")
        coverage_arrays: dict[str, Any] = {}
        for name, path in zip(names, coverage_paths):
            coverage_image = runtime.load_nifti_image(path)
            runtime.validate_compatible_geometry(output_image, coverage_image)
            coverage_arrays[name] = runtime.validate_binary_mask(coverage_image.get_fdata(), allow_empty=True, label=f"{name} coverage mask")
        original_count = _mask_count(output_mask)
        application = runtime.apply_coverage_masks(output_mask, coverage_masks=coverage_arrays)
        retained_count = _mask_count(application.mask)
        runtime.write_roi_nifti_mask(application.mask, output_image, job.output_mask_path)
    except Exception as error:
        return (
            RoiTransformCoverageIntersectionQcRow(
                source_index=job.source_index,
                mask_path=job.output_mask_path,
                status="error",
                applied_mask_names=names,
                applied_mask_paths=coverage_paths,
                original_voxel_count=0,
                retained_voxel_count=0,
                dropped_voxel_count=0,
                message=f"Could not apply configured coverage-mask intersection: {error}",
            ),
        )
    dropped_count = max(0, original_count - retained_count)
    return (
        RoiTransformCoverageIntersectionQcRow(
            source_index=job.source_index,
            mask_path=job.output_mask_path,
            status="ok",
            applied_mask_names=tuple(application.applied_masks),
            applied_mask_paths=coverage_paths,
            original_voxel_count=original_count,
            retained_voxel_count=retained_count,
            dropped_voxel_count=dropped_count,
            message="Configured coverage masks were intersected with the transformed ROI mask.",
        ),
    )


def _run_actual_transform_qc(job: ExecutableMniToT1wRoiTransformJob) -> dict[str, tuple[Any, ...]]:
    try:
        runtime = _load_roi_transform_qc_runtime()
        reference_image = runtime.load_nifti_image(job.target_reference_path)
        output_image = runtime.load_nifti_image(job.output_mask_path)
    except Exception as error:
        return {
            "geometry_qc": (
                RoiTransformGeometryQcRow(
                    source_index=job.source_index,
                    mask_path=job.output_mask_path,
                    reference_path=job.target_reference_path,
                    status="error",
                    shape_matches=None,
                    affine_matches=None,
                    message=f"Could not load transformed ROI or reference image for QC: {error}",
                ),
            ),
            "binary_mask_qc": (),
            "voxel_count_qc": (),
            "coverage_overlap_qc": (),
        }

    geometry_row = _geometry_qc_row(job, runtime, reference_image, output_image)
    try:
        output_data = output_image.get_fdata()
        binary_mask = runtime.validate_binary_mask(output_data, allow_empty=True, label="transformed ROI mask")
        binary_row = RoiTransformBinaryMaskQcRow(
            source_index=job.source_index,
            mask_path=job.output_mask_path,
            status="ok",
            binary_interpretable=True,
            unique_values=_unique_values(output_data),
            message="Transformed ROI mask is binary-interpretable.",
        )
    except Exception as error:
        return {
            "geometry_qc": (geometry_row,),
            "binary_mask_qc": (
                RoiTransformBinaryMaskQcRow(
                    source_index=job.source_index,
                    mask_path=job.output_mask_path,
                    status="error",
                    binary_interpretable=False,
                    unique_values=_unique_values(_safe_get_fdata(output_image)),
                    message=f"Transformed ROI mask is not binary-interpretable: {error}",
                ),
            ),
            "voxel_count_qc": (),
            "coverage_overlap_qc": (),
        }

    voxel_row = _voxel_count_qc_row(job, runtime, binary_mask)
    coverage_rows = _coverage_qc_rows(job, runtime, output_image, binary_mask, voxel_row.voxel_count)
    return {
        "geometry_qc": (geometry_row,),
        "binary_mask_qc": (binary_row,),
        "voxel_count_qc": (voxel_row,),
        "coverage_overlap_qc": coverage_rows,
    }


def _load_roi_transform_qc_runtime() -> _RoiTransformQcRuntime:
    from research_platform.neuro.nifti import load_nifti_image, validate_compatible_geometry
    from research_platform.neuro.roi_masks import apply_coverage_masks, check_min_voxel_count, validate_binary_mask, write_roi_nifti_mask

    return _RoiTransformQcRuntime(
        load_nifti_image=load_nifti_image,
        validate_compatible_geometry=validate_compatible_geometry,
        validate_binary_mask=validate_binary_mask,
        check_min_voxel_count=check_min_voxel_count,
        apply_coverage_masks=apply_coverage_masks,
        write_roi_nifti_mask=write_roi_nifti_mask,
    )


def _geometry_qc_row(
    job: ExecutableMniToT1wRoiTransformJob,
    runtime: _RoiTransformQcRuntime,
    reference_image: Any,
    output_image: Any,
) -> RoiTransformGeometryQcRow:
    output_shape = _image_shape(output_image)
    reference_shape = _image_shape(reference_image)
    shape_matches = bool(output_shape and reference_shape and output_shape == reference_shape)
    affine_matches = _affines_match(output_image, reference_image)
    try:
        runtime.validate_compatible_geometry(reference_image, output_image)
    except Exception as error:
        return RoiTransformGeometryQcRow(
            source_index=job.source_index,
            mask_path=job.output_mask_path,
            reference_path=job.target_reference_path,
            status="error",
            shape_matches=shape_matches,
            affine_matches=affine_matches,
            output_shape=output_shape,
            reference_shape=reference_shape,
            message=f"Transformed ROI geometry does not match target reference: {error}",
        )
    return RoiTransformGeometryQcRow(
        source_index=job.source_index,
        mask_path=job.output_mask_path,
        reference_path=job.target_reference_path,
        status="ok",
        shape_matches=True,
        affine_matches=True,
        output_shape=output_shape,
        reference_shape=reference_shape,
        message="Transformed ROI geometry matches target reference.",
    )


def _voxel_count_qc_row(
    job: ExecutableMniToT1wRoiTransformJob,
    runtime: _RoiTransformQcRuntime,
    binary_mask: Any,
) -> RoiTransformVoxelCountQcRow:
    try:
        voxel_qc = runtime.check_min_voxel_count(
            binary_mask,
            min_voxels_warn=job.min_voxels_warn,
            min_voxels_fail=job.min_voxels_fail,
        )
        voxel_count = int(voxel_qc.voxel_count)
        flags = tuple(str(value) for value in getattr(voxel_qc, "qc_flags", ()))
        row_warnings = tuple(str(value) for value in getattr(voxel_qc, "warnings", ()))
        passed = bool(getattr(voxel_qc, "passed", True))
    except Exception as error:
        return RoiTransformVoxelCountQcRow(
            source_index=job.source_index,
            mask_path=job.output_mask_path,
            status="error",
            voxel_count=0,
            empty=True,
            min_voxels_warn=job.min_voxels_warn,
            min_voxels_fail=job.min_voxels_fail,
            qc_flags=("voxel_count_qc_error",),
            message=f"Could not compute transformed ROI voxel-count QC: {error}",
        )
    empty = voxel_count == 0
    status = "error" if empty or not passed else "warning" if row_warnings else "ok"
    flags = tuple(_unique_text([*flags, *(("empty_mask",) if empty else ())]))
    message = "Transformed ROI mask is empty." if empty else "; ".join(row_warnings) if row_warnings else "Transformed ROI voxel count passed QC."
    return RoiTransformVoxelCountQcRow(
        source_index=job.source_index,
        mask_path=job.output_mask_path,
        status=status,
        voxel_count=voxel_count,
        empty=empty,
        min_voxels_warn=job.min_voxels_warn,
        min_voxels_fail=job.min_voxels_fail,
        qc_flags=flags,
        warnings=row_warnings,
        message=message,
    )


def _coverage_qc_rows(
    job: ExecutableMniToT1wRoiTransformJob,
    runtime: _RoiTransformQcRuntime,
    output_image: Any,
    binary_mask: Any,
    voxel_count: int,
) -> tuple[RoiTransformCoverageOverlapQcRow, ...]:
    if job.coverage_mask_path is None:
        return ()
    try:
        coverage_image = runtime.load_nifti_image(job.coverage_mask_path)
        runtime.validate_compatible_geometry(output_image, coverage_image)
        coverage_mask = runtime.validate_binary_mask(coverage_image.get_fdata(), allow_empty=True, label="coverage mask")
        overlap_count = _mask_overlap_count(binary_mask, coverage_mask)
        coverage_count = _mask_count(coverage_mask)
        ratio = float(overlap_count / voxel_count) if voxel_count > 0 else 0.0
    except Exception as error:
        return (
            RoiTransformCoverageOverlapQcRow(
                source_index=job.source_index,
                mask_path=job.output_mask_path,
                coverage_mask_path=job.coverage_mask_path,
                status="error",
                overlap_ratio=0.0,
                roi_voxel_count=voxel_count,
                overlap_voxel_count=0,
                coverage_voxel_count=0,
                min_overlap_warn=job.coverage_min_overlap_fraction_warn,
                min_overlap_fail=job.coverage_min_overlap_fraction_fail,
                message=f"Could not compute coverage overlap QC: {error}",
            ),
        )
    status = "ok"
    message = "Transformed ROI overlaps the supplied coverage mask."
    if job.coverage_min_overlap_fraction_fail is not None and ratio < job.coverage_min_overlap_fraction_fail:
        status = "error"
        message = f"coverage overlap ratio {ratio:.6g} is below fail threshold {job.coverage_min_overlap_fraction_fail:.6g}."
    elif job.coverage_min_overlap_fraction_warn is not None and ratio < job.coverage_min_overlap_fraction_warn:
        status = "warning"
        message = f"coverage overlap ratio {ratio:.6g} is below warn threshold {job.coverage_min_overlap_fraction_warn:.6g}."
    return (
        RoiTransformCoverageOverlapQcRow(
            source_index=job.source_index,
            mask_path=job.output_mask_path,
            coverage_mask_path=job.coverage_mask_path,
            status=status,
            overlap_ratio=ratio,
            roi_voxel_count=voxel_count,
            overlap_voxel_count=overlap_count,
            coverage_voxel_count=coverage_count,
            min_overlap_warn=job.coverage_min_overlap_fraction_warn,
            min_overlap_fail=job.coverage_min_overlap_fraction_fail,
            message=message,
        ),
    )


def _write_planned_execution_json(
    job: ExecutableMniToT1wRoiTransformJob,
    command_record: RoiTransformCommandExecutionRecord,
    verification: TransformedRoiOutputVerificationRow,
    qc_result: Mapping[str, tuple[Any, ...]],
) -> tuple[RoiTransformJsonWriteRow, ...]:
    rows: list[RoiTransformJsonWriteRow] = []
    qc_payload = {
        "artifact_kind": "mni_to_t1w_roi_transform_qc",
        "job": job.to_dict(),
        "command": command_record.to_dict(),
        "output_verification": verification.to_dict(),
        "geometry_qc": [row.to_dict() for row in qc_result["geometry_qc"]],
        "binary_mask_qc": [row.to_dict() for row in qc_result["binary_mask_qc"]],
        "voxel_count_qc": [row.to_dict() for row in qc_result["voxel_count_qc"]],
        "coverage_intersection_qc": [row.to_dict() for row in qc_result.get("coverage_intersection_qc", ())],
        "coverage_overlap_qc": [row.to_dict() for row in qc_result["coverage_overlap_qc"]],
    }
    if job.qc_path is not None:
        rows.append(_write_json_artifact(job.source_index, "qc", job.qc_path, qc_payload))
    if job.provenance_path is not None:
        provenance_payload = {
            "artifact_kind": "mni_to_t1w_roi_transform_provenance",
            "job": job.to_dict(),
            "argv": list(job.argv),
            "interpolation": job.interpolation,
            "transform_chain": list(job.transform_chain),
            "command": command_record.to_dict(),
            "output_verification": verification.to_dict(),
            "qc_status": _qc_status(
                (
                    *qc_result.get("coverage_intersection_qc", ()),
                    *qc_result["geometry_qc"],
                    *qc_result["binary_mask_qc"],
                    *qc_result["voxel_count_qc"],
                    *qc_result["coverage_overlap_qc"],
                )
            ),
        }
        rows.append(_write_json_artifact(job.source_index, "provenance", job.provenance_path, provenance_payload))
    return tuple(rows)


def _write_json_artifact(source_index: int, artifact_kind: str, path: str, payload: Mapping[str, Any]) -> RoiTransformJsonWriteRow:
    try:
        text = json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n"
        Path(path).write_text(text, encoding="utf-8")
    except Exception as error:
        return RoiTransformJsonWriteRow(
            source_index=source_index,
            artifact_kind=artifact_kind,
            path=path,
            status="error",
            error=str(error),
        )
    return RoiTransformJsonWriteRow(
        source_index=source_index,
        artifact_kind=artifact_kind,
        path=path,
        status="written",
        bytes_written=len(text.encode("utf-8")),
    )


def _qc_status(rows: Sequence[Any]) -> str:
    if any(getattr(row, "status", None) == "error" for row in rows):
        return "error"
    if any(getattr(row, "status", None) == "warning" for row in rows):
        return "warning"
    return "ok"


def _image_shape(image: Any) -> tuple[int, ...]:
    shape = getattr(image, "shape", ())
    try:
        return tuple(int(value) for value in tuple(shape)[:3])
    except (TypeError, ValueError):
        return ()


def _affines_match(left: Any, right: Any, *, atol: float = 1e-5) -> bool | None:
    left_affine = _matrix_values(getattr(left, "affine", None))
    right_affine = _matrix_values(getattr(right, "affine", None))
    if left_affine is None or right_affine is None:
        return None
    return all(abs(a - b) <= atol for left_row, right_row in zip(left_affine, right_affine) for a, b in zip(left_row, right_row))


def _matrix_values(value: Any) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    raw = value.tolist() if hasattr(value, "tolist") else value
    try:
        rows = tuple(tuple(float(cell) for cell in row) for row in raw)
    except (TypeError, ValueError):
        return None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        return None
    return rows


def _safe_get_fdata(image: Any) -> Any:
    try:
        return image.get_fdata()
    except Exception:
        return ()


def _unique_values(data: Any, *, limit: int = 16) -> tuple[int | float | str, ...]:
    if data is None:
        return ()
    raw = data.ravel().tolist() if hasattr(data, "ravel") else data
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raw = (raw,)
    seen: list[int | float | str] = []
    for value in raw:
        normalized = _json_number_or_text(value)
        if normalized not in seen:
            seen.append(normalized)
        if len(seen) >= limit:
            break
    return tuple(sorted(seen, key=str))


def _json_number_or_text(value: Any) -> int | float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    if number.is_integer():
        return int(number)
    return number


def _mask_count(mask: Any) -> int:
    if hasattr(mask, "sum"):
        return int(mask.sum())
    if isinstance(mask, Sequence) and not isinstance(mask, (str, bytes, bytearray)):
        return sum(_mask_count(item) for item in mask)
    return int(bool(mask))


def _mask_overlap_count(left: Any, right: Any) -> int:
    try:
        return int((left & right).sum())
    except Exception:
        return sum(1 for left_value, right_value in zip(_flatten(left), _flatten(right)) if bool(left_value) and bool(right_value))


def _flatten(value: Any) -> tuple[Any, ...]:
    if hasattr(value, "ravel"):
        return tuple(value.ravel().tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened: list[Any] = []
        for child in value:
            flattened.extend(_flatten(child))
        return tuple(flattened)
    return (value,)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tool_spec(raw: Mapping[str, Any] | TransformToolSpecRow | None) -> TransformToolSpecRow:
    if isinstance(raw, TransformToolSpecRow):
        return raw
    fields = _mapping_or_empty(raw)
    return TransformToolSpecRow(
        tool_name=_optional_text(fields.get("tool_name") or fields.get("name") or fields.get("command")) or ANTS_APPLY_TRANSFORMS,
        executable=_optional_text(fields.get("executable") or fields.get("executable_path") or fields.get("path")),
        dimension=_int_value(fields.get("dimension"), default=DEFAULT_DIMENSION),
        default_interpolation=_optional_text(fields.get("interpolation") or fields.get("default_interpolation")) or DEFAULT_INTERPOLATION,
        lookup_on_path=not _bool_value(fields.get("disable_path_lookup")),
    )


def _command_row(unit: _PlannedUnit, *, executable: str, tool_name: str) -> TransformCommandPlanRow:
    row_errors = list(unit.errors)
    row_warnings = list(unit.warnings)
    if unit.source.source_mask_path is None:
        row_errors.append("Cannot render a complete command without a source mask path.")
    if unit.target.target_reference_path is None:
        row_errors.append("Cannot render a complete command without a target reference image path.")
    if unit.output.output_mask_path is None:
        row_errors.append("Cannot render a complete command without a planned output mask path.")
    if not unit.transforms:
        row_errors.append("Cannot render a complete command without at least one supplied transform.")
    if row_errors:
        argv: tuple[str, ...] = ()
        status = "error"
    else:
        argv = build_ants_apply_transforms_argv(
            executable=executable,
            input_mask=str(unit.source.source_mask_path),
            reference_image=str(unit.target.target_reference_path),
            output_path=str(unit.output.output_mask_path),
            transforms=unit.transforms,
            dimension=unit.dimension,
            interpolation=unit.interpolation,
        )
        status = "planned"
    return TransformCommandPlanRow(
        source_index=unit.source.source_index,
        tool_name=tool_name,
        argv=argv,
        dimension=unit.dimension,
        interpolation=unit.interpolation,
        subject_id=unit.source.subject_id,
        session_id=unit.source.session_id,
        task_id=unit.source.task_id,
        run_id=unit.source.run_id,
        model=unit.source.model,
        contrast_id=unit.source.contrast_id,
        roi_label=unit.source.roi_label,
        status=status,
        warnings=tuple(_unique_text(row_warnings)),
        errors=tuple(_unique_text(row_errors)),
    )


def _qc_rows(unit: _PlannedUnit, *, qc_config: Mapping[str, Any]) -> tuple[RoiQcPreviewRow, ...]:
    identifiers = _row_identifiers(unit.source)
    rows: list[RoiQcPreviewRow] = [
        RoiQcPreviewRow(
            source_index=unit.source.source_index,
            check_kind="source_mask_exists",
            timing="preflight",
            status=unit.source.status,
            message=_exists_message("source MNI ROI mask", unit.source.exists),
            path=unit.source.source_mask_path,
            **identifiers,
        ),
        RoiQcPreviewRow(
            source_index=unit.source.source_index,
            check_kind="target_reference_exists",
            timing="preflight",
            status=unit.target.status,
            message=_exists_message("target reference image", unit.target.exists),
            path=unit.target.target_reference_path,
            **identifiers,
        ),
    ]
    for transform in unit.transforms:
        rows.append(
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="transform_exists",
                timing="preflight",
                status=transform.status,
                message=_exists_message(f"transform {transform.order_index}", transform.exists),
                path=transform.transform_path,
                expected=f"order={transform.order_index}; inverted={str(transform.invert).lower()}",
                **identifiers,
            )
        )
    for coverage in unit.coverage_masks:
        rows.append(
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="coverage_mask_exists",
                timing="preflight",
                status=coverage.status,
                message=coverage.message or _exists_message(f"coverage or brain mask {coverage.name}", coverage.exists),
                path=coverage.path,
                expected=coverage.name,
                **identifiers,
            )
        )
    if unit.coverage_masks:
        rows.append(
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="coverage_intersection_policy",
                timing="planned_post_execution",
                status="planned",
                message="After execution, apply or audit configured coverage masks according to policy.",
                path=unit.output.output_mask_path,
                expected=_coverage_intersection_policy(qc_config),
                **identifiers,
            )
        )
    rows.extend(
        [
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="planned_output_path",
                timing="preflight",
                status=unit.output.status,
                message="Transformed ROI output path is planned but not written.",
                path=unit.output.output_mask_path,
                **identifiers,
            ),
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="post_execution_geometry_matches_reference",
                timing="planned_post_execution",
                status="planned",
                message="After execution, verify transformed ROI geometry matches the target reference image.",
                path=unit.output.output_mask_path,
                expected=unit.target.target_reference_path,
                **identifiers,
            ),
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="post_execution_voxel_count",
                timing="planned_post_execution",
                status="planned",
                message="After execution, count non-zero ROI voxels.",
                path=unit.output.output_mask_path,
                **identifiers,
            ),
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="post_execution_empty_mask",
                timing="planned_post_execution",
                status="planned",
                message="After execution, flag an empty transformed ROI mask.",
                path=unit.output.output_mask_path,
                expected=_optional_text(qc_config.get("empty_mask_policy")) or "fail",
                **identifiers,
            ),
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="post_execution_small_mask",
                timing="planned_post_execution",
                status="planned",
                message="After execution, compare voxel count with configured small-mask thresholds.",
                path=unit.output.output_mask_path,
                threshold=_small_mask_threshold(qc_config),
                **identifiers,
            ),
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="interpolation_policy",
                timing="preflight",
                status="ok" if unit.interpolation == DEFAULT_INTERPOLATION else "warning",
                message="ROI mask transform interpolation policy is planned.",
                expected=unit.interpolation,
                **identifiers,
            ),
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="transform_chain_provenance",
                timing="preflight",
                status="planned" if unit.transforms else "error",
                message="Transform chain order and inversion flags are recorded for provenance.",
                expected=" -> ".join(
                    _format_transform_arg(row.transform_path or "", invert=row.invert) for row in unit.transforms
                ),
                **identifiers,
            ),
        ]
    )
    for coverage in unit.coverage_masks:
        rows.append(
            RoiQcPreviewRow(
                source_index=unit.source.source_index,
                check_kind="post_execution_overlap_coverage",
                timing="planned_post_execution",
                status="planned",
                message="After execution, compare transformed ROI overlap with the supplied coverage mask.",
                path=unit.output.output_mask_path,
                expected=coverage.path,
                threshold=_coverage_overlap_threshold(qc_config),
                **identifiers,
            )
        )
    return tuple(rows)


def _small_mask_threshold(qc_config: Mapping[str, Any]) -> int | str | None:
    warn = qc_config.get("min_voxels_warn")
    if warn is None:
        warn = qc_config.get("small_mask_warn_voxels")
    fail = qc_config.get("min_voxels_fail")
    if fail is None:
        fail = qc_config.get("small_mask_fail_voxels")
    if warn is not None and fail is not None:
        return f"warn<{warn}; fail<{fail}"
    if warn is not None:
        return f"warn<{warn}"
    if fail is not None:
        return f"fail<{fail}"
    return None


def _coverage_overlap_threshold(qc_config: Mapping[str, Any]) -> float | str | None:
    warn = qc_config.get("coverage_min_overlap_fraction_warn")
    if warn is None:
        warn = qc_config.get("min_coverage_fraction_warn")
    if warn is None:
        warn = qc_config.get("coverage_threshold")
    fail = qc_config.get("coverage_min_overlap_fraction_fail")
    if fail is None:
        fail = qc_config.get("min_coverage_fraction_fail")
    if warn is not None and fail is not None:
        return f"warn<{warn}; fail<{fail}"
    if warn is not None:
        return warn
    if fail is not None:
        return f"fail<{fail}"
    return None


def _coverage_intersection_policy(qc_config: Mapping[str, Any]) -> str:
    policy = _optional_text(qc_config.get("coverage_intersection_policy")) or "qc_only"
    if policy not in {"qc_only", "apply"}:
        return "qc_only"
    return policy


def _status_for_path(
    state: _PathState,
    label: str,
    *,
    policy: str,
    required: bool,
) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    if state.path is None:
        if not required:
            return "not_applicable", warnings, errors
        message = f"{label} path was not supplied."
        if policy == MISSING_POLICY_FAIL:
            errors.append(message)
            return "error", warnings, errors
        warnings.append(message)
        return "warning", warnings, errors
    if state.exists is True:
        return "ok", warnings, errors
    if state.exists is None:
        return "preview_only", warnings, errors
    message = f"{label} is missing: {state.path}"
    if not required and policy == MISSING_POLICY_IGNORE:
        return "missing", warnings, errors
    if policy == MISSING_POLICY_FAIL:
        errors.append(message)
        return "error", warnings, errors
    warnings.append(message)
    return "warning", warnings, errors


def _status_for_output(state: _PathState) -> tuple[str, list[str], list[str]]:
    if state.path:
        return "planned", [], []
    message = "Planned transformed ROI output path was not supplied."
    return "error", [], [message]


def _resolve_path_from_keys(row: Mapping[str, Any], keys: Sequence[str], *, roots: Mapping[str, str | Path] | None) -> _PathState:
    for key in keys:
        if key in row and row.get(key) is not None:
            return _resolve_path(row.get(key), roots=roots, row=row)
    return _resolve_path(row, roots=roots, row=row, path_keys=keys)


def _resolve_target_path(
    source_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    *,
    roots: Mapping[str, str | Path] | None,
) -> _PathState:
    for row in (target_row, source_row):
        state = _resolve_path_from_keys(row, _TARGET_PATH_KEYS, roots=roots)
        if state.path is not None:
            return state
    return _PathState(path=None, status="missing")


def _resolve_transform_path(transform_row: Mapping[str, Any], *, roots: Mapping[str, str | Path] | None) -> _PathState:
    for key in ("transform_path", "path", "transform", "filename"):
        if key in transform_row and transform_row.get(key) is not None:
            return _resolve_path(transform_row.get(key), roots=roots, row=transform_row)
    return _resolve_path(transform_row, roots=roots, row=transform_row, path_keys=("path", "transform_path"))


def _resolve_output_path(
    source_row: Mapping[str, Any],
    outputs_config: Mapping[str, Any],
    *,
    roots: Mapping[str, str | Path] | None,
    identifiers: Mapping[str, Any],
) -> _PathState:
    for key in _OUTPUT_PATH_KEYS:
        if source_row.get(key) is not None:
            return _resolve_path(source_row.get(key), roots=roots, row=source_row, context=identifiers)
    if outputs_config:
        output_spec = outputs_config.get("mask") if isinstance(outputs_config.get("mask"), Mapping) else outputs_config
        return _resolve_path(output_spec, roots=roots, row=source_row, context=identifiers)
    return _PathState(path=None, status="missing")


def _resolve_optional_output_artifact_path(
    source_row: Mapping[str, Any],
    outputs_config: Mapping[str, Any],
    *,
    keys: Sequence[str],
    nested_keys: Sequence[str],
    roots: Mapping[str, str | Path] | None,
    identifiers: Mapping[str, Any],
) -> _PathState:
    for key in keys:
        if source_row.get(key) is not None:
            return _resolve_path(source_row.get(key), roots=roots, row=source_row, context=identifiers)
    for key in keys:
        if outputs_config.get(key) is not None:
            return _resolve_path(outputs_config.get(key), roots=roots, row=source_row, context=identifiers)
    for nested_key in nested_keys:
        nested = outputs_config.get(nested_key)
        if isinstance(nested, Mapping):
            return _resolve_path(nested, roots=roots, row=source_row, context=identifiers)
    return _PathState(path=None, status="not_applicable")


def _resolve_coverage_paths(
    source_row: Mapping[str, Any],
    qc_config: Mapping[str, Any],
    *,
    roots: Mapping[str, str | Path] | None,
) -> tuple[tuple[str, _PathState], ...]:
    states: list[tuple[str, _PathState]] = []
    for row in (source_row, qc_config):
        for key in _COVERAGE_PATH_KEYS:
            if row.get(key) is not None:
                states.append((key, _resolve_path(row.get(key), roots=roots, row=source_row)))
        raw_multi = row.get("coverage_mask_paths") or row.get("brain_mask_paths")
        if isinstance(raw_multi, Sequence) and not isinstance(raw_multi, (str, bytes, bytearray)):
            for index, spec in enumerate(raw_multi):
                states.append((f"coverage_mask_{index + 1}", _resolve_path(spec, roots=roots, row=source_row)))
        raw_named = row.get("coverage_masks") or row.get("brain_masks")
        if isinstance(raw_named, Mapping):
            for name, spec in raw_named.items():
                states.append((str(name), _resolve_path(spec, roots=roots, row=source_row)))
    deduped: list[tuple[str, _PathState]] = []
    seen: set[tuple[str, str | None]] = set()
    for name, state in states:
        key = (name, state.path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, state))
    return tuple(deduped)


def _resolve_path(
    spec: Any,
    *,
    roots: Mapping[str, str | Path] | None,
    row: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    path_keys: Sequence[str] = ("path", "relative_path", "pattern", "template", "path_template"),
) -> _PathState:
    row = row or {}
    context = {**row, **(context or {})}
    if isinstance(spec, Path):
        return _path_state(spec.as_posix(), root_ref=None)
    if isinstance(spec, str):
        return _path_state(_render_template(spec, context), root_ref=_optional_text(row.get("root_ref")))
    if not isinstance(spec, Mapping):
        return _PathState(path=None, status="missing")

    root_ref = _optional_text(spec.get("root_ref") or row.get("root_ref"))
    literal_root = _optional_text(spec.get("root") or spec.get("base") or row.get("root"))
    raw_path = None
    for key in path_keys:
        if spec.get(key) is not None:
            raw_path = _optional_text(spec.get(key))
            break
    if raw_path is None:
        return _PathState(path=None, root_ref=root_ref, status="missing")
    rendered = _render_template(raw_path, context)
    candidate = Path(rendered)
    if candidate.is_absolute():
        return _path_state(candidate.as_posix(), root_ref=root_ref)
    if root_ref and roots is not None and root_ref in roots:
        return _path_state((Path(roots[root_ref]) / rendered).as_posix(), root_ref=root_ref)
    if literal_root is not None:
        literal = _render_template(literal_root, context)
        return _path_state((Path(literal) / rendered).as_posix(), root_ref=root_ref)
    if root_ref:
        return _PathState(path=f"{root_ref}/{rendered}", root_ref=root_ref, exists=None, status="preview_only")
    return _path_state(rendered, root_ref=root_ref)


def _path_state(path: str, *, root_ref: str | None) -> _PathState:
    if not _path_is_checkable(path):
        return _PathState(path=path, root_ref=root_ref, exists=None, status="preview_only")
    exists = Path(path).exists()
    return _PathState(
        path=path,
        root_ref=root_ref,
        exists=exists,
        status="ok" if exists else "missing",
        message=None if exists else f"Path does not exist: {path}",
    )


def _path_is_checkable(path: str | None) -> bool:
    if not path:
        return False
    if _UNRESOLVED_PLACEHOLDER.search(path):
        return False
    return not any(char in path for char in _GLOB_CHARS)


def _collect_source_rows(
    config: Mapping[str, Any],
    source_masks: Sequence[Mapping[str, Any] | RoiTransformSourceMaskRow] | None,
    roi_handoff: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    roi_build_outputs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[Any, ...]:
    rows: list[Any] = []
    rows.extend(source_masks or ())
    rows.extend(_collect_config_rows(config.get("source_masks")))
    rows.extend(_candidate_mask_rows(roi_handoff))
    rows.extend(_candidate_mask_rows(roi_build_outputs))
    return tuple(rows)


def _expand_selector_source_rows(config: Mapping[str, Any], rows: Sequence[Any]) -> tuple[Any, ...]:
    selectors = _selector_axes(config)
    if not selectors:
        return tuple(rows)

    expanded: list[Mapping[str, Any]] = []
    for row in rows:
        mapping = _row_mapping(row)
        contexts = _selector_contexts(selectors, mapping)
        for context in contexts:
            expanded.append({**context, **mapping})
    return tuple(expanded)


def _selector_axes(config: Mapping[str, Any]) -> Mapping[str, tuple[str, ...]]:
    raw = config.get("selectors")
    if not isinstance(raw, Mapping):
        raw = config.get("selector")
    selector = dict(raw) if isinstance(raw, Mapping) else {}
    for key in ("subjects", "subject", "sessions", "session", "runs", "run", "tasks", "task", "models", "model", "contrasts", "contrast", "contrast_ids", "roi_labels", "labels"):
        if key in config and key not in selector:
            selector[key] = config[key]
    axes = {
        "subjects": _selector_values(selector, "subjects", "subject"),
        "sessions": _selector_values(selector, "sessions", "session"),
        "runs": _selector_values(selector, "runs", "run"),
        "tasks": _selector_values(selector, "tasks", "task"),
        "models": _selector_values(selector, "models", "model"),
        "contrasts": _selector_values(selector, "contrast_ids", "contrasts", "contrast"),
        "roi_labels": _selector_values(selector, "roi_labels", "labels", "roi_label", "label"),
    }
    return {key: values for key, values in axes.items() if values}


def _selector_contexts(selectors: Mapping[str, tuple[str, ...]], row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    axes: list[tuple[Mapping[str, Any], ...]] = []
    if not _row_has_any(row, ("subject_id", "participant_id", "subject", "subject_dir")) and selectors.get("subjects"):
        axes.append(tuple(_subject_values(value) for value in selectors["subjects"]))
    if not _row_has_any(row, ("session_id", "session", "session_dir")) and selectors.get("sessions"):
        axes.append(tuple(_session_values(value) for value in selectors["sessions"]))
    if not _row_has_any(row, ("run_id", "run", "run_dir", "run_entity")) and selectors.get("runs"):
        axes.append(tuple(_run_values(value) for value in selectors["runs"]))
    if not _row_has_any(row, ("task_id", "task")) and selectors.get("tasks"):
        axes.append(tuple(_task_values(value) for value in selectors["tasks"]))
    if not _row_has_any(row, ("model",)) and selectors.get("models"):
        axes.append(tuple({"model": value} for value in selectors["models"]))
    if not _row_has_any(row, ("contrast_id", "contrast")) and selectors.get("contrasts"):
        axes.append(tuple(_contrast_values(value) for value in selectors["contrasts"]))
    if not _row_has_any(row, ("roi_label", "label")) and selectors.get("roi_labels"):
        axes.append(tuple({"roi_label": value} for value in selectors["roi_labels"]))
    if not axes:
        return ({},)

    contexts: tuple[Mapping[str, Any], ...] = ({},)
    for axis in axes:
        contexts = tuple({**left, **right} for left in contexts for right in axis)
    return contexts


def _selector_values(mapping: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    for key in keys:
        values = _string_sequence(mapping.get(key))
        if values:
            return values
    return ()


def _row_has_any(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(_optional_text(row.get(key)) is not None for key in keys)


def _subject_values(value: str) -> Mapping[str, str]:
    subject_id = _strip_entity_prefix(value, "sub")
    subject_dir = value if value.startswith("sub-") else f"sub-{subject_id}"
    return {"subject_id": subject_id, "participant_id": subject_id, "subject": subject_dir, "subject_dir": subject_dir}


def _session_values(value: str) -> Mapping[str, str]:
    session_id = _strip_entity_prefix(value, "ses")
    session_dir = value if value.startswith("ses-") else f"ses-{session_id}"
    return {"session_id": session_id, "session": session_dir, "session_dir": session_dir}


def _run_values(value: str) -> Mapping[str, str]:
    run_id = _strip_entity_prefix(value, "run")
    run_dir = value if value.startswith("run-") else f"run-{run_id}"
    return {"run_id": run_id, "run": run_dir, "run_dir": run_dir, "run_entity": run_dir}


def _task_values(value: str) -> Mapping[str, str]:
    task_id = _strip_entity_prefix(value, "task")
    return {"task_id": task_id, "task": task_id}


def _contrast_values(value: str) -> Mapping[str, str]:
    contrast_id = _strip_entity_prefix(value, "contrast")
    return {"contrast_id": contrast_id, "contrast": contrast_id}


def _strip_entity_prefix(value: str, prefix: str) -> str:
    text = str(value).strip()
    marker = f"{prefix}-"
    return text[len(marker) :] if text.startswith(marker) else text


def _candidate_mask_rows(raw: Any) -> tuple[Mapping[str, Any], ...]:
    candidates = _collect_config_rows(raw)
    rows: list[Mapping[str, Any]] = []
    for row in candidates:
        mapping = _row_mapping(row)
        if any(mapping.get(key) is not None for key in _SOURCE_PATH_KEYS):
            rows.append(mapping)
    return tuple(rows)


def _collect_config_rows(raw: Any) -> tuple[Any, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        for key in (
            "source_masks",
            "source_mask_rows",
            "target_references",
            "transform_chains",
            "handoff_rows",
            "existing_roi_build_preview_rows",
            "roi_source_rows",
            "completed_ffx_output_rows",
            "rows",
            "actions",
            "planned_outputs",
            "outputs",
        ):
            value = raw.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return tuple(value)
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return tuple(raw)
    return ()


def _validate_transform_row_container(
    raw: Any,
    label: str,
    errors: list[str],
    *,
    required: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    if raw is None:
        if required:
            errors.append(f"{label} must contain at least one mapping.")
        return ()
    if not isinstance(raw, Mapping) and not (
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray))
    ):
        errors.append(f"{label} must contain a mapping or sequence of mappings.")
        return ()
    rows = _collect_config_rows(raw)
    if not rows:
        if required:
            errors.append(f"{label} must contain at least one mapping.")
        return ()
    validated: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"{label}[{index}] must contain a mapping.")
            continue
        validated.append(row)
    return tuple(validated)


def _validate_declared_path(
    row: Mapping[str, Any],
    keys: Sequence[str],
    label: str,
    errors: list[str],
) -> None:
    for key in keys:
        if row.get(key) is not None:
            _validate_path_spec(row.get(key), label, errors)
            return
    _validate_path_spec(row, label, errors, path_keys=keys)


def _validate_path_spec(
    spec: Any,
    label: str,
    errors: list[str],
    *,
    path_keys: Sequence[str] = ("path", "relative_path", "pattern", "template", "path_template"),
) -> None:
    if isinstance(spec, (str, os.PathLike)):
        text = str(spec).strip()
        if not text:
            errors.append(f"{label} must be a non-empty path reference.")
        elif configured_path_is_unsafe(text):
            errors.append(f"{label} must be relative and remain beneath its configured root.")
        return
    if not isinstance(spec, Mapping):
        errors.append(f"{label} must be a string or path-reference mapping.")
        return
    if "root_ref" in spec and _optional_text(spec.get("root_ref")) is None:
        errors.append(f"{label}.root_ref must be a non-empty string when declared.")
    for key in path_keys:
        if key in spec:
            value = spec.get(key)
            if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
                errors.append(f"{label}.{key} must be a non-empty path string.")
            elif configured_path_is_unsafe(str(value)):
                errors.append(
                    f"{label}.{key} must be relative and remain beneath its configured root."
                )
            return
    errors.append(f"{label} must define one of: {', '.join(path_keys)}.")


def _match_row(source_row: Mapping[str, Any], rows: Sequence[Any], *, source_index: int) -> Mapping[str, Any]:
    if not rows:
        return {}
    if len(rows) == 1:
        return _row_mapping(rows[0])
    source_keys = _match_keys(source_row)
    for row in rows:
        mapping = _row_mapping(row)
        if _int_value(mapping.get("source_index"), default=-1) == source_index:
            return mapping
        row_keys = _match_keys(mapping)
        if row_keys and source_keys and all(source_keys.get(key) == value for key, value in row_keys.items()):
            return mapping
    return {}


def _raw_transform_specs(source_row: Mapping[str, Any], rows: Sequence[Any], *, source_index: int) -> tuple[Any, ...]:
    for key in ("transforms", "transform_chain"):
        raw = source_row.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return tuple(raw)
    if not rows:
        return ()
    matched_items = tuple(
        row
        for row in rows
        if _is_transform_path_row(_row_mapping(row)) and _row_matches_source(_row_mapping(row), source_row, source_index)
    )
    if matched_items:
        return matched_items
    if all(_is_transform_item(_row_mapping(row)) for row in rows):
        return tuple(rows)
    matched = _match_row(source_row, rows, source_index=source_index)
    if not matched and len(rows) == 1:
        matched = _row_mapping(rows[0])
    for key in ("transforms", "transform_chain"):
        raw = matched.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return tuple(raw)
    if matched:
        return (matched,)
    return ()


def _is_transform_item(row: Mapping[str, Any]) -> bool:
    has_match_key = bool(_match_keys(row)) or row.get("source_index") is not None
    return _is_transform_path_row(row) and not has_match_key


def _is_transform_path_row(row: Mapping[str, Any]) -> bool:
    has_path = any(row.get(key) is not None for key in ("transform_path", "path", "transform", "filename"))
    has_chain = any(row.get(key) is not None for key in ("transforms", "transform_chain"))
    return has_path and not has_chain


def _row_matches_source(row: Mapping[str, Any], source_row: Mapping[str, Any], source_index: int) -> bool:
    if row.get("source_index") is not None:
        return _int_value(row.get("source_index"), default=-1) == source_index
    row_keys = _match_keys(row)
    source_keys = _match_keys(source_row)
    return bool(row_keys and source_keys and all(source_keys.get(key) == value for key, value in row_keys.items()))


def _transform_path_and_invert(raw: str | os.PathLike[str] | Mapping[str, Any] | TransformChainSpecRow) -> tuple[str | None, bool]:
    if isinstance(raw, TransformChainSpecRow):
        return raw.transform_path, raw.invert
    if isinstance(raw, (str, os.PathLike)):
        return str(raw), False
    mapping = _row_mapping(raw)
    path = _optional_text(mapping.get("transform_path") or mapping.get("path") or mapping.get("transform") or mapping.get("filename"))
    return path, _bool_value(mapping.get("invert") or mapping.get("inverted"))


def _format_transform_arg(path: str, *, invert: bool) -> str:
    return f"[{path},1]" if invert else path


def _payload_mapping(raw: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    inner = raw.get("roi_transform_plan") or raw.get("roi_transforms") or raw.get("mni_to_t1w_roi_transforms")
    return inner if isinstance(inner, Mapping) else raw


def _mapping_or_none(raw: Any) -> Mapping[str, Any] | None:
    return raw if isinstance(raw, Mapping) else None


def _mapping_or_empty(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def _row_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    if is_dataclass(row) and not isinstance(row, type):
        return _json_safe_dataclass(row)
    fields_value = getattr(row, "fields", None)
    if isinstance(fields_value, Mapping):
        result = dict(fields_value)
        for key in _IDENTIFIER_KEYS:
            value = getattr(row, key, None)
            if value is not None:
                result.setdefault(key, value)
        return result
    return {}


def _identifiers(row: Mapping[str, Any]) -> dict[str, str | None]:
    subject = _optional_text(row.get("subject_id") or row.get("participant_id"))
    return {
        "subject_id": subject,
        "session_id": _optional_text(row.get("session_id") or row.get("session")),
        "task_id": _optional_text(row.get("task_id") or row.get("task")),
        "run_id": _optional_text(row.get("run_id") or row.get("run")),
        "model": _optional_text(row.get("model") or row.get("model_id")),
        "contrast_id": _optional_text(row.get("contrast_id") or row.get("contrast")),
        "roi_label": _optional_text(row.get("roi_label") or row.get("label")),
    }


def _row_identifiers(row: RoiTransformSourceMaskRow) -> dict[str, str | None]:
    return {
        "subject_id": row.subject_id,
        "session_id": row.session_id,
        "task_id": row.task_id,
        "run_id": row.run_id,
        "model": row.model,
        "contrast_id": row.contrast_id,
        "roi_label": row.roi_label,
    }


def _match_keys(row: Mapping[str, Any]) -> dict[str, str]:
    keys = {}
    for key, value in _identifiers(row).items():
        if value is not None:
            keys[key] = value
    return keys


def _root_ref_rows(roots: Mapping[str, str | Path] | None) -> list[RoiTransformProvenanceRow]:
    if not roots:
        return []
    rows: list[RoiTransformProvenanceRow] = []
    for root_ref, path in sorted(roots.items()):
        text = Path(path).as_posix()
        exists = Path(path).exists() if _path_is_checkable(text) else None
        rows.append(
            RoiTransformProvenanceRow(
                provenance_kind="root_ref",
                root_ref=str(root_ref),
                path=text,
                status="ok" if exists else "preview_only" if exists is None else "missing",
                message=None if exists else "Root is not checkable yet." if exists is None else "Root path does not exist.",
            )
        )
    return rows


def _render_template(template: str, context: Mapping[str, Any]) -> str:
    rendered = template
    for key, value in context.items():
        if value is None:
            continue
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered


def _exists_message(label: str, exists: bool | None) -> str:
    if exists is True:
        return f"{label} exists."
    if exists is False:
        return f"{label} is missing."
    return f"{label} existence is preview-only because the path is not fully checkable."


def _aggregate_status(*, errors: Sequence[str], warnings: Sequence[str], rows: Sequence[Any]) -> str:
    if errors or any(getattr(row, "status", None) == "error" for row in rows):
        return "error"
    if warnings or any(getattr(row, "status", None) == "warning" for row in rows):
        return "warning"
    if any(getattr(row, "status", None) in {"preview_only", "missing"} for row in rows):
        return "preview_only"
    return "ok"


def _normalize_missing_policy(value: str | None) -> str:
    text = (value or MISSING_POLICY_FAIL).strip().lower()
    if text in {"error", "raise"}:
        return MISSING_POLICY_FAIL
    if text not in MISSING_POLICIES:
        return MISSING_POLICY_FAIL
    return text


def _looks_like_path(value: str) -> bool:
    return os.sep in value or (os.altsep is not None and os.altsep in value) or value.startswith(".") or value.startswith("~")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[str] = []
        for item in value:
            text = _optional_text(item)
            if text is not None:
                result.append(text)
        return tuple(result)
    text = _optional_text(value)
    return (text,) if text is not None else ()


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return str(value)


__all__ = [
    "ANTS_APPLY_TRANSFORMS",
    "DEFAULT_DIMENSION",
    "DEFAULT_INTERPOLATION",
    "ExecutableMniToT1wRoiTransformJob",
    "MniToT1wRoiTransformPlan",
    "PlannedTransformedRoiOutputRow",
    "RoiTransformBinaryMaskQcRow",
    "RoiTransformCommandExecutionRecord",
    "RoiTransformCoverageOverlapQcRow",
    "RoiTransformExecutionJobRow",
    "RoiTransformExecutionResult",
    "RoiTransformExecutionValidationResult",
    "RoiTransformGeometryQcRow",
    "RoiTransformJsonWriteRow",
    "RoiQcPreviewRow",
    "RoiTransformPlan",
    "RoiTransformProvenanceRow",
    "RoiTransformSourceMaskRow",
    "RoiTransformVoxelCountQcRow",
    "TargetReferenceImageRow",
    "TransformChainSpecRow",
    "TransformCommandPlanRow",
    "TransformToolPreflightRow",
    "TransformToolSpecRow",
    "TransformedRoiOutputVerificationRow",
    "build_ants_apply_transforms_argv",
    "check_transform_tool_availability",
    "execute_mni_to_t1w_roi_transform_plan",
    "plan_mni_to_t1w_roi_transforms",
    "preflight_ants_apply_transforms",
    "select_executable_mni_to_t1w_roi_transform_jobs",
    "validate_mni_to_t1w_roi_transform_document",
    "validate_mni_to_t1w_roi_transform_execution_plan",
]
