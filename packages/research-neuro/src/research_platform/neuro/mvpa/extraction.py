"""Runtime-only local MVPA pattern extraction from discovery plans.

This module consumes integrated MVPA discovery plans and extracts PE image
values inside ROI masks. It does not compute distances, write outputs, run
FSL, invoke CLI commands, or import cross-package analysis layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np

from research_platform.neuro.nifti import GeometryMismatchError, load_nifti_image, validate_compatible_geometry
from research_platform.neuro.roi_masks import validate_binary_mask


VOXEL_ORDER_C_FLAT_INDEX = "c_flat_index"
_EXECUTABLE_STATUSES = frozenset({"", "ok", "warning"})
_SKIPPED_STATUSES = frozenset({"skipped", "excluded"})
_NON_EXECUTABLE_STATUSES = frozenset({"error", "invalid", "preview_only", "not_checked", "deferred"})
_NOISE_NONPOSITIVE_POLICY_STRICT = "strict"
_NOISE_NONPOSITIVE_POLICY_DROP_FEATURES = "drop_features"


@dataclass(frozen=True)
class MvpaPatternExtractionRow:
    """One extracted condition-by-ROI pattern vector."""

    pattern_id: str
    condition_id: str | None
    cv_unit: str | None
    subject_id: str | None
    session_id: str | None
    run_id: str | None
    task_id: str | None
    direction: str | None
    model: str | None
    pattern_source_name: str | None
    roi_source_name: str | None
    roi_label: str | None
    pe_image: str | None
    mask_path: str | None
    noise_image: str | None
    voxel_count: int
    valid_voxel_count: int
    feature_count: int
    voxel_order: str
    voxel_index_hash: str
    usable: bool
    feature_values: tuple[float, ...]
    event_count: int | None = None
    mean_centering_applied: bool = False
    mean_centering_scope: str = "none"
    grouping_values: Mapping[str, Any] = field(default_factory=dict)
    noise_loaded: bool = False
    noise_status: str = "not_requested"
    noise_usable: bool = False
    noise_feature_count: int = 0
    noise_voxel_order: str | None = None
    noise_voxel_index_hash: str | None = None
    noise_min: float | None = None
    noise_max: float | None = None
    noise_mean: float | None = None
    noise_nonfinite_count: int = 0
    noise_nonpositive_count: int = 0
    noise_values: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload.update(_safe_grouping_columns(self.grouping_values, existing=payload))
        return payload


@dataclass(frozen=True)
class MvpaPatternQcRow:
    """One QC row for an attempted or skipped condition-by-ROI candidate."""

    subject_id: str | None
    session_id: str | None
    run_id: str | None
    condition_id: str | None
    roi_label: str | None
    pattern_source_name: str | None
    roi_source_name: str | None
    pe_image: str | None
    mask_path: str | None
    noise_image: str | None
    status: str
    usable: bool
    reason: str | None
    excluded: bool
    exclusion_reason: str | None
    pe_exists: bool | None
    mask_exists: bool | None
    noise_exists: bool | None
    geometry_status: str
    mask_status: str
    voxel_count: int | None
    valid_voxel_count: int | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    event_threshold_status: str
    exclusion_id: str | None = None
    exclusion_source_field: str | None = None
    skipped_stage: str | None = None
    grouping_values: Mapping[str, Any] = field(default_factory=dict)
    noise_loaded: bool = False
    noise_status: str = "not_requested"
    noise_usable: bool = False
    noise_feature_count: int = 0
    noise_voxel_order: str | None = None
    noise_voxel_index_hash: str | None = None
    noise_min: float | None = None
    noise_max: float | None = None
    noise_mean: float | None = None
    noise_nonfinite_count: int = 0
    noise_nonpositive_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload.update(_safe_grouping_columns(self.grouping_values, existing=payload))
        return payload


@dataclass(frozen=True)
class MvpaPatternExtractionResult:
    """JSON-safe result for in-memory MVPA pattern extraction."""

    pattern_rows: tuple[MvpaPatternExtractionRow, ...]
    qc_rows: tuple[MvpaPatternQcRow, ...]
    provenance: Mapping[str, Any]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    executed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class _CandidateMetadata:
    subject_id: str | None
    session_id: str | None
    run_id: str | None
    condition_id: str | None
    cv_unit: str | None
    task_id: str | None
    direction: str | None
    model: str | None
    pattern_source_name: str | None
    roi_source_name: str | None
    roi_label: str | None
    pe_image: str | None
    mask_path: str | None
    noise_image: str | None
    event_count: int | None
    grouping_values: Mapping[str, Any]


@dataclass(frozen=True)
class _BlockingStatus:
    status: str
    reason: str
    excluded: bool = False
    exclusion_reason: str | None = None
    exclusion_id: str | None = None
    exclusion_source_field: str | None = None
    skipped_stage: str | None = None


@dataclass(frozen=True)
class _NoiseExtraction:
    loaded: bool
    status: str
    usable: bool
    feature_count: int = 0
    voxel_order: str | None = None
    voxel_index_hash: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    nonfinite_count: int = 0
    nonpositive_count: int = 0
    values: tuple[float, ...] = ()
    warning: str | None = None


@dataclass(frozen=True)
class _RunExclusionPolicy:
    id: str
    subject_id: str
    session_id: str
    run_id: str
    reason: str
    source_config_field: str


@dataclass(frozen=True)
class _MeanCenteringPolicy:
    enabled: bool = False
    scope: str = "none"


def extract_mvpa_patterns_from_discovery_plan(
    plan: Any,
    *,
    load_noise: bool = False,
    fail_fast: bool = False,
) -> MvpaPatternExtractionResult:
    """Extract ROI voxel vectors from an integrated MVPA discovery plan.

    ``plan`` may be an ``MvpaDiscoveryPlan`` instance or a JSON-safe mapping
    produced by ``plan.to_dict()``. Only joined condition/ROI candidates that
    are not skipped, excluded, or plan-blocked are loaded from disk.
    """

    payload = _coerce_plan_mapping(plan)
    condition_rows = _row_mappings(payload.get("condition_pe_rows"))
    roi_rows = _row_mappings(payload.get("roi_source_rows"))
    pattern_unit_rows = _row_mappings(payload.get("pattern_source_rows"))
    input_checks = _row_mappings(payload.get("input_checks"))
    provenance_rows = _row_mappings(payload.get("provenance_rows"))
    event_threshold_rows = _event_threshold_rows(payload)
    event_threshold_status = _event_threshold_status(event_threshold_rows)
    cv_unit = _cv_unit(payload)
    grouping_columns = _grouping_columns(payload)
    run_exclusions = _run_exclusion_policies(payload)
    mean_centering = _mean_centering_policy(payload)
    noise_nonpositive_policy = _noise_nonpositive_policy(payload)

    pattern_units = _index_pattern_units(pattern_unit_rows)
    roi_rows_by_unit = _index_rows_by_unit(roi_rows)
    matched_roi_indexes: set[int] = set()
    pattern_rows: list[MvpaPatternExtractionRow] = []
    qc_rows: list[MvpaPatternQcRow] = []
    warnings: list[str] = []
    errors: list[str] = []

    if not condition_rows:
        warnings.append("Discovery plan has no condition_pe_rows to extract.")
    if not roi_rows:
        warnings.append("Discovery plan has no roi_source_rows to extract.")

    for condition_row in condition_rows:
        unit_key = _unit_key(condition_row)
        matches = roi_rows_by_unit.get(unit_key, ())
        pattern_unit = _matching_pattern_unit(pattern_units, condition_row)
        if not matches:
            metadata = _candidate_metadata(
                condition_row,
                None,
                pattern_unit,
                cv_unit=cv_unit,
                grouping_columns=grouping_columns,
            )
            exclusion = _matching_run_exclusion(run_exclusions, metadata)
            if exclusion is not None:
                qc = _qc_row(
                    metadata,
                    status="excluded",
                    usable=False,
                    reason=exclusion.reason,
                    excluded=True,
                    exclusion_reason=exclusion.reason,
                    exclusion_id=exclusion.id,
                    exclusion_source_field=exclusion.source_config_field,
                    skipped_stage="before_extraction",
                    event_threshold_status=event_threshold_status,
                    noise_status=_not_checked_or_not_requested(load_noise),
                    warnings=_row_warnings(condition_row, pattern_unit),
                    errors=_row_errors(condition_row, pattern_unit),
                )
                qc_rows.append(qc)
                _raise_if_fail_fast(fail_fast, qc)
                continue
            qc = _qc_row(
                metadata,
                status="missing_roi_source_row",
                usable=False,
                reason="No ROI source row matched this condition row by subject/session/run.",
                event_threshold_status=event_threshold_status,
                noise_status=_not_checked_or_not_requested(load_noise),
                warnings=_row_warnings(condition_row, pattern_unit),
                errors=_row_errors(condition_row, pattern_unit),
            )
            qc_rows.append(qc)
            warnings.append(qc.reason or qc.status)
            _raise_if_fail_fast(fail_fast, qc)
            continue

        for roi_index, roi_row in matches:
            matched_roi_indexes.add(roi_index)
            pattern_unit = _matching_pattern_unit(pattern_units, condition_row)
            pattern_row, qc = _extract_joined_candidate(
                condition_row,
                roi_row,
                pattern_unit,
                cv_unit=cv_unit,
                grouping_columns=grouping_columns,
                run_exclusions=run_exclusions,
                mean_centering=mean_centering,
                event_threshold_status=event_threshold_status,
                load_noise=load_noise,
                noise_nonpositive_policy=noise_nonpositive_policy,
                fail_fast=fail_fast,
            )
            qc_rows.append(qc)
            if pattern_row is not None:
                pattern_rows.append(pattern_row)
            _extend_messages(warnings, qc.warnings)
            _extend_messages(errors, qc.errors)

    for roi_index, roi_row in enumerate(roi_rows):
        if roi_index in matched_roi_indexes:
            continue
        metadata = _candidate_metadata(
            None,
            roi_row,
            None,
            cv_unit=cv_unit,
            grouping_columns=grouping_columns,
        )
        exclusion = _matching_run_exclusion(run_exclusions, metadata)
        if exclusion is not None:
            qc = _qc_row(
                metadata,
                status="excluded",
                usable=False,
                reason=exclusion.reason,
                excluded=True,
                exclusion_reason=exclusion.reason,
                exclusion_id=exclusion.id,
                exclusion_source_field=exclusion.source_config_field,
                skipped_stage="before_extraction",
                event_threshold_status=event_threshold_status,
                noise_status=_not_checked_or_not_requested(load_noise),
                warnings=_row_warnings(roi_row),
                errors=_row_errors(roi_row),
            )
            qc_rows.append(qc)
            _raise_if_fail_fast(fail_fast, qc)
            continue
        qc = _qc_row(
            metadata,
            status="missing_condition_pe_row",
            usable=False,
            reason="No condition-to-PE row matched this ROI source row by subject/session/run.",
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=_row_warnings(roi_row),
            errors=_row_errors(roi_row),
        )
        qc_rows.append(qc)
        warnings.append(qc.reason or qc.status)
        _raise_if_fail_fast(fail_fast, qc)

    provenance = {
        "source": "research_platform.neuro.mvpa.extraction",
        "plan_status": _optional_text(payload.get("status")),
        "mvpa_set": _optional_text(payload.get("mvpa_set")) or _optional_text(payload.get("mvpa_set_name")),
        "load_noise": bool(load_noise),
        "noise_nonpositive_policy": noise_nonpositive_policy,
        "fail_fast": bool(fail_fast),
        "condition_pe_row_count": len(condition_rows),
        "roi_source_row_count": len(roi_rows),
        "input_check_count": len(input_checks),
        "provenance_row_count": len(provenance_rows),
        "event_threshold_status": event_threshold_status,
        "event_threshold_rows": tuple(event_threshold_rows),
        "grouping_columns": tuple(grouping_columns),
        "run_exclusion_policy_rows": tuple(_run_exclusion_policy_rows(run_exclusions)),
        "excluded_run_rows": tuple(_excluded_run_rows(qc_rows)),
        "mean_centering": {
            "enabled": mean_centering.enabled,
            "scope": mean_centering.scope,
        },
        "voxel_order": VOXEL_ORDER_C_FLAT_INDEX,
    }
    provenance.update(_noise_provenance(load_noise=load_noise, qc_rows=qc_rows))
    return MvpaPatternExtractionResult(
        pattern_rows=tuple(pattern_rows),
        qc_rows=tuple(qc_rows),
        provenance=provenance,
        warnings=tuple(_unique_text(warnings)),
        errors=tuple(_unique_text(errors)),
        executed=True,
    )


def _extract_joined_candidate(
    condition_row: Mapping[str, Any],
    roi_row: Mapping[str, Any],
    pattern_unit: Mapping[str, Any] | None,
    *,
    cv_unit: str | None,
    grouping_columns: Sequence[str],
    run_exclusions: Sequence["_RunExclusionPolicy"],
    mean_centering: "_MeanCenteringPolicy",
    event_threshold_status: str,
    load_noise: bool,
    noise_nonpositive_policy: str,
    fail_fast: bool,
) -> tuple[MvpaPatternExtractionRow | None, MvpaPatternQcRow]:
    metadata = _candidate_metadata(
        condition_row,
        roi_row,
        pattern_unit,
        cv_unit=cv_unit,
        grouping_columns=grouping_columns,
    )
    row_warnings = _row_warnings(condition_row, roi_row, pattern_unit)
    row_errors = _row_errors(condition_row, roi_row, pattern_unit)

    exclusion = _matching_run_exclusion(run_exclusions, metadata)
    blocking_status = (
        _BlockingStatus(
            status="excluded",
            reason=exclusion.reason,
            excluded=True,
            exclusion_reason=exclusion.reason,
            exclusion_id=exclusion.id,
            exclusion_source_field=exclusion.source_config_field,
            skipped_stage="before_extraction",
        )
        if exclusion is not None
        else _candidate_blocking_status(condition_row, roi_row, pattern_unit)
    )
    if blocking_status is not None:
        qc = _qc_row(
            metadata,
            status=blocking_status.status,
            usable=False,
            reason=blocking_status.reason,
            excluded=blocking_status.excluded,
            exclusion_reason=blocking_status.exclusion_reason,
            exclusion_id=blocking_status.exclusion_id,
            exclusion_source_field=blocking_status.exclusion_source_field,
            skipped_stage=blocking_status.skipped_stage,
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=row_errors,
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    pe_exists = _path_exists(metadata.pe_image)
    mask_exists = _path_exists(metadata.mask_path)
    noise_exists = _path_exists(metadata.noise_image)

    if pe_exists is not True:
        qc = _qc_row(
            metadata,
            status="missing_pe_image",
            usable=False,
            reason="PE image is missing.",
            pe_exists=pe_exists,
            mask_exists=mask_exists,
            noise_exists=noise_exists,
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=(*row_errors, "PE image is missing."),
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    if mask_exists is not True:
        qc = _qc_row(
            metadata,
            status="missing_roi_mask",
            usable=False,
            reason="ROI mask is missing.",
            pe_exists=pe_exists,
            mask_exists=mask_exists,
            noise_exists=noise_exists,
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=(*row_errors, "ROI mask is missing."),
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    try:
        pe_image = load_nifti_image(str(metadata.pe_image))
        mask_image = load_nifti_image(str(metadata.mask_path))
    except Exception as exc:  # pragma: no cover - exact nibabel errors vary.
        qc = _qc_row(
            metadata,
            status="image_load_error",
            usable=False,
            reason=str(exc),
            pe_exists=pe_exists,
            mask_exists=mask_exists,
            noise_exists=noise_exists,
            geometry_status="not_checked",
            mask_status="not_checked",
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=(*row_errors, str(exc)),
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    try:
        validate_compatible_geometry(pe_image, mask_image)
    except GeometryMismatchError as exc:
        qc = _qc_row(
            metadata,
            status="geometry_mismatch",
            usable=False,
            reason=str(exc),
            pe_exists=pe_exists,
            mask_exists=mask_exists,
            noise_exists=noise_exists,
            geometry_status="mismatch",
            mask_status="not_checked",
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=(*row_errors, str(exc)),
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    pe_values = np.asarray(pe_image.get_fdata(dtype=np.float64))
    if pe_values.ndim != 3:
        qc = _qc_row(
            metadata,
            status="invalid_pe_image",
            usable=False,
            reason="PE image must be 3D.",
            pe_exists=pe_exists,
            mask_exists=mask_exists,
            noise_exists=noise_exists,
            geometry_status="invalid",
            mask_status="not_checked",
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=(*row_errors, "PE image must be 3D."),
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    try:
        mask = validate_binary_mask(mask_image.get_fdata(), allow_empty=True, label="roi_mask")
    except ValueError as exc:
        qc = _qc_row(
            metadata,
            status="invalid_roi_mask",
            usable=False,
            reason=str(exc),
            pe_exists=pe_exists,
            mask_exists=mask_exists,
            noise_exists=noise_exists,
            geometry_status="ok",
            mask_status="invalid",
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=(*row_errors, str(exc)),
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    voxel_indices = np.flatnonzero(mask.ravel(order="C"))
    voxel_count = int(voxel_indices.size)
    if voxel_count == 0:
        qc = _qc_row(
            metadata,
            status="empty_roi_mask",
            usable=False,
            reason="ROI mask contains no selected voxels.",
            pe_exists=pe_exists,
            mask_exists=mask_exists,
            noise_exists=noise_exists,
            geometry_status="ok",
            mask_status="empty",
            voxel_count=0,
            valid_voxel_count=0,
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=(*row_errors, "ROI mask contains no selected voxels."),
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    selected_values = pe_values.ravel(order="C")[voxel_indices]
    finite = np.isfinite(selected_values)
    valid_voxel_count = int(np.count_nonzero(finite))
    if valid_voxel_count != voxel_count:
        qc = _qc_row(
            metadata,
            status="nonfinite_pe_values",
            usable=False,
            reason="PE values inside the ROI must all be finite.",
            pe_exists=pe_exists,
            mask_exists=mask_exists,
            noise_exists=noise_exists,
            geometry_status="ok",
            mask_status="ok",
            voxel_count=voxel_count,
            valid_voxel_count=valid_voxel_count,
            event_threshold_status=event_threshold_status,
            noise_status=_not_checked_or_not_requested(load_noise),
            warnings=row_warnings,
            errors=(*row_errors, "PE values inside the ROI must all be finite."),
        )
        _raise_if_fail_fast(fail_fast, qc)
        return None, qc

    voxel_index_hash = _voxel_index_hash(voxel_indices)
    centered_values, mean_centering_applied, mean_centering_scope = _mean_centered_feature_values(
        selected_values,
        mean_centering,
    )
    feature_values = tuple(float(value) for value in centered_values.tolist())
    noise = _extract_noise_values(
        metadata,
        pe_image,
        noise_exists=noise_exists,
        voxel_indices=voxel_indices,
        feature_count=len(feature_values),
        voxel_index_hash=voxel_index_hash,
        load_noise=load_noise,
        noise_nonpositive_policy=noise_nonpositive_policy,
    )
    pattern_row = MvpaPatternExtractionRow(
        pattern_id=_pattern_id(metadata, voxel_index_hash),
        condition_id=metadata.condition_id,
        cv_unit=metadata.cv_unit,
        subject_id=metadata.subject_id,
        session_id=metadata.session_id,
        run_id=metadata.run_id,
        task_id=metadata.task_id,
        direction=metadata.direction,
        model=metadata.model,
        pattern_source_name=metadata.pattern_source_name,
        roi_source_name=metadata.roi_source_name,
        roi_label=metadata.roi_label,
        pe_image=metadata.pe_image,
        mask_path=metadata.mask_path,
        noise_image=metadata.noise_image,
        voxel_count=voxel_count,
        valid_voxel_count=valid_voxel_count,
        feature_count=len(feature_values),
        voxel_order=VOXEL_ORDER_C_FLAT_INDEX,
        voxel_index_hash=voxel_index_hash,
        usable=True,
        feature_values=feature_values,
        mean_centering_applied=mean_centering_applied,
        mean_centering_scope=mean_centering_scope,
        grouping_values=metadata.grouping_values,
        event_count=metadata.event_count,
        noise_loaded=noise.loaded,
        noise_status=noise.status,
        noise_usable=noise.usable,
        noise_feature_count=noise.feature_count,
        noise_voxel_order=noise.voxel_order,
        noise_voxel_index_hash=noise.voxel_index_hash,
        noise_min=noise.minimum,
        noise_max=noise.maximum,
        noise_mean=noise.mean,
        noise_nonfinite_count=noise.nonfinite_count,
        noise_nonpositive_count=noise.nonpositive_count,
        noise_values=noise.values,
    )

    qc_warnings = row_warnings
    if noise.warning is not None:
        qc_warnings = (*qc_warnings, noise.warning)

    qc = _qc_row(
        metadata,
        status="ok",
        usable=True,
        reason=None,
        pe_exists=pe_exists,
        mask_exists=mask_exists,
        noise_exists=noise_exists,
        geometry_status="ok",
        mask_status="ok",
        voxel_count=voxel_count,
        valid_voxel_count=valid_voxel_count,
        event_threshold_status=event_threshold_status,
        noise_loaded=noise.loaded,
        noise_status=noise.status,
        noise_usable=noise.usable,
        noise_feature_count=noise.feature_count,
        noise_voxel_order=noise.voxel_order,
        noise_voxel_index_hash=noise.voxel_index_hash,
        noise_min=noise.minimum,
        noise_max=noise.maximum,
        noise_mean=noise.mean,
        noise_nonfinite_count=noise.nonfinite_count,
        noise_nonpositive_count=noise.nonpositive_count,
        warnings=qc_warnings,
        errors=row_errors,
    )
    return pattern_row, qc


def _extract_noise_values(
    metadata: _CandidateMetadata,
    pe_image: Any,
    *,
    noise_exists: bool | None,
    voxel_indices: np.ndarray,
    feature_count: int,
    voxel_index_hash: str,
    load_noise: bool,
    noise_nonpositive_policy: str,
) -> _NoiseExtraction:
    if not load_noise:
        return _NoiseExtraction(loaded=False, status="not_requested", usable=False)

    if metadata.noise_image is None or noise_exists is not True:
        return _NoiseExtraction(
            loaded=False,
            status="missing_noise_image",
            usable=False,
            warning="Noise image is missing; PE pattern values remain usable without noise normalization.",
        )

    try:
        noise_image = load_nifti_image(str(metadata.noise_image))
    except Exception as exc:  # pragma: no cover - exact nibabel errors vary.
        return _NoiseExtraction(
            loaded=False,
            status="noise_load_error",
            usable=False,
            warning=f"Noise image could not be loaded: {exc}",
        )

    try:
        validate_compatible_geometry(pe_image, noise_image)
    except GeometryMismatchError as exc:
        return _NoiseExtraction(
            loaded=True,
            status="noise_geometry_mismatch",
            usable=False,
            warning=f"Noise image geometry does not match the PE image: {exc}",
        )

    noise_values = np.asarray(noise_image.get_fdata(dtype=np.float64))
    if noise_values.ndim != 3:
        return _NoiseExtraction(
            loaded=True,
            status="invalid_noise_image",
            usable=False,
            warning="Noise image must be 3D.",
        )

    selected_values = noise_values.ravel(order="C")[voxel_indices]
    noise_feature_count = int(selected_values.size)
    if noise_feature_count != int(feature_count):
        return _NoiseExtraction(
            loaded=True,
            status="noise_feature_count_mismatch",
            usable=False,
            feature_count=noise_feature_count,
            warning=(
                "Noise image selected voxel count does not match the PE pattern "
                f"feature count: {noise_feature_count} != {int(feature_count)}."
            ),
        )

    finite = np.isfinite(selected_values)
    nonfinite_count = int(noise_feature_count - np.count_nonzero(finite))
    finite_values = selected_values[finite]
    nonpositive_count = int(np.count_nonzero(finite_values <= 0.0))
    minimum, maximum, mean = _finite_summary(finite_values)
    base = {
        "loaded": True,
        "feature_count": noise_feature_count,
        "voxel_order": VOXEL_ORDER_C_FLAT_INDEX,
        "voxel_index_hash": voxel_index_hash,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "nonfinite_count": nonfinite_count,
        "nonpositive_count": nonpositive_count,
    }

    if nonfinite_count:
        return _NoiseExtraction(
            status="nonfinite_noise_values",
            usable=False,
            warning=f"Noise values inside the ROI include {nonfinite_count} non-finite value(s).",
            **base,
        )
    if nonpositive_count:
        if noise_nonpositive_policy == _NOISE_NONPOSITIVE_POLICY_DROP_FEATURES:
            return _NoiseExtraction(
                status="nonpositive_noise_values",
                usable=False,
                values=tuple(float(value) for value in selected_values.tolist()),
                warning=f"Noise values inside the ROI include {nonpositive_count} zero or negative value(s).",
                **base,
            )
        return _NoiseExtraction(
            status="nonpositive_noise_values",
            usable=False,
            warning=f"Noise values inside the ROI include {nonpositive_count} zero or negative value(s).",
            **base,
        )

    return _NoiseExtraction(
        status="ok",
        usable=True,
        values=tuple(float(value) for value in selected_values.tolist()),
        **base,
    )


def _candidate_metadata(
    condition_row: Mapping[str, Any] | None,
    roi_row: Mapping[str, Any] | None,
    pattern_unit: Mapping[str, Any] | None,
    *,
    cv_unit: str | None,
    grouping_columns: Sequence[str],
) -> _CandidateMetadata:
    rows = tuple(row for row in (condition_row, roi_row, pattern_unit) if row is not None)
    return _CandidateMetadata(
        subject_id=_first_text(rows, "subject_id"),
        session_id=_first_text(rows, "session_id"),
        run_id=_first_text(rows, "run_id"),
        condition_id=_first_text((condition_row,) if condition_row is not None else (), "condition_id", "id"),
        cv_unit=cv_unit,
        task_id=_first_text(rows, "task_id"),
        direction=_first_text(rows, "direction"),
        model=_first_text(rows, "model"),
        pattern_source_name=_first_text(
            tuple(row for row in (condition_row, pattern_unit) if row is not None),
            "pattern_source_name",
            "source_name",
        ),
        roi_source_name=_first_text((roi_row,) if roi_row is not None else (), "roi_source_name", "source_name"),
        roi_label=_first_text((roi_row,) if roi_row is not None else (), "roi_label", "label"),
        pe_image=_first_text((condition_row,) if condition_row is not None else (), "pe_image"),
        mask_path=_first_text((roi_row,) if roi_row is not None else (), "mask_path"),
        noise_image=_first_text(tuple(row for row in (condition_row, pattern_unit) if row is not None), "noise_image"),
        event_count=_first_int((condition_row,) if condition_row is not None else (), "event_count", "n_events"),
        grouping_values=_grouping_values(rows, grouping_columns),
    )


def _candidate_blocking_status(
    condition_row: Mapping[str, Any],
    roi_row: Mapping[str, Any],
    pattern_unit: Mapping[str, Any] | None,
) -> _BlockingStatus | None:
    for label, row in (
        ("pattern source", pattern_unit),
        ("condition-to-PE", condition_row),
        ("ROI source", roi_row),
    ):
        if row is None:
            continue
        if _row_excluded(row):
            reason = _optional_text(row.get("exclusion_reason")) or _optional_text(row.get("reason"))
            return _BlockingStatus(
                status="excluded",
                reason=reason or f"{label} row is excluded.",
                excluded=True,
                exclusion_reason=reason,
                exclusion_id=_optional_text(row.get("exclusion_id")),
                exclusion_source_field=_optional_text(row.get("exclusion_source_field") or row.get("source_config_field")),
                skipped_stage="before_extraction",
            )

    for label, row in (
        ("pattern source", pattern_unit),
        ("condition-to-PE", condition_row),
        ("ROI source", roi_row),
    ):
        if row is None:
            continue
        status = _row_status(row)
        if status in _SKIPPED_STATUSES:
            reason = _optional_text(row.get("exclusion_reason")) or _optional_text(row.get("reason"))
            return _BlockingStatus(status="skipped", reason=reason or f"{label} row status is {status}.")
        if status in _NON_EXECUTABLE_STATUSES:
            output_status = "plan_error" if status in {"error", "invalid"} else "not_executable_plan_status"
            return _BlockingStatus(status=output_status, reason=f"{label} row status is {status}.")
        if status not in _EXECUTABLE_STATUSES:
            return _BlockingStatus(
                status="not_executable_plan_status",
                reason=f"{label} row status {status!r} is not executable by local extraction.",
            )
    return None


def _qc_row(
    metadata: _CandidateMetadata,
    *,
    status: str,
    usable: bool,
    reason: str | None,
    event_threshold_status: str,
    excluded: bool = False,
    exclusion_reason: str | None = None,
    exclusion_id: str | None = None,
    exclusion_source_field: str | None = None,
    skipped_stage: str | None = None,
    pe_exists: bool | None = None,
    mask_exists: bool | None = None,
    noise_exists: bool | None = None,
    geometry_status: str = "not_checked",
    mask_status: str = "not_checked",
    voxel_count: int | None = None,
    valid_voxel_count: int | None = None,
    noise_loaded: bool = False,
    noise_status: str = "not_requested",
    noise_usable: bool = False,
    noise_feature_count: int = 0,
    noise_voxel_order: str | None = None,
    noise_voxel_index_hash: str | None = None,
    noise_min: float | None = None,
    noise_max: float | None = None,
    noise_mean: float | None = None,
    noise_nonfinite_count: int = 0,
    noise_nonpositive_count: int = 0,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
) -> MvpaPatternQcRow:
    return MvpaPatternQcRow(
        subject_id=metadata.subject_id,
        session_id=metadata.session_id,
        run_id=metadata.run_id,
        condition_id=metadata.condition_id,
        roi_label=metadata.roi_label,
        pattern_source_name=metadata.pattern_source_name,
        roi_source_name=metadata.roi_source_name,
        pe_image=metadata.pe_image,
        mask_path=metadata.mask_path,
        noise_image=metadata.noise_image,
        status=status,
        usable=usable,
        reason=reason,
        excluded=excluded,
        exclusion_reason=exclusion_reason,
        exclusion_id=exclusion_id,
        exclusion_source_field=exclusion_source_field,
        skipped_stage=skipped_stage,
        pe_exists=pe_exists,
        mask_exists=mask_exists,
        noise_exists=noise_exists,
        geometry_status=geometry_status,
        mask_status=mask_status,
        voxel_count=voxel_count,
        valid_voxel_count=valid_voxel_count,
        warnings=tuple(_unique_text(warnings)),
        errors=tuple(_unique_text(errors)),
        event_threshold_status=event_threshold_status,
        grouping_values=metadata.grouping_values,
        noise_loaded=noise_loaded,
        noise_status=noise_status,
        noise_usable=noise_usable,
        noise_feature_count=noise_feature_count,
        noise_voxel_order=noise_voxel_order,
        noise_voxel_index_hash=noise_voxel_index_hash,
        noise_min=noise_min,
        noise_max=noise_max,
        noise_mean=noise_mean,
        noise_nonfinite_count=noise_nonfinite_count,
        noise_nonpositive_count=noise_nonpositive_count,
    )


def _coerce_plan_mapping(plan: Any) -> Mapping[str, Any]:
    if isinstance(plan, Mapping):
        return dict(plan)
    to_dict = getattr(plan, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if is_dataclass(plan) and not isinstance(plan, type):
        return _json_safe_dataclass(plan)
    raise TypeError("plan must be an MvpaDiscoveryPlan-like object or a mapping.")


def _row_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (_mapping_with_string_keys(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_mapping_with_string_keys(_row_to_mapping(row)) for row in value)
    return ()


def _row_to_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    if is_dataclass(row) and not isinstance(row, type):
        return _json_safe_dataclass(row)
    return {"value": row}


def _mapping_with_string_keys(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {str(key): value for key, value in row.items()}


def _index_rows_by_unit(rows: Sequence[Mapping[str, Any]]) -> Mapping[tuple[str | None, str | None, str | None], tuple[tuple[int, Mapping[str, Any]], ...]]:
    grouped: dict[tuple[str | None, str | None, str | None], list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(_unit_key(row), []).append((index, row))
    return {key: tuple(value) for key, value in grouped.items()}


def _index_pattern_units(rows: Sequence[Mapping[str, Any]]) -> Mapping[tuple[str | None, str | None, str | None, str | None], Mapping[str, Any]]:
    indexed: dict[tuple[str | None, str | None, str | None, str | None], Mapping[str, Any]] = {}
    for row in rows:
        indexed[
            (
                _optional_text(row.get("subject_id")),
                _optional_text(row.get("session_id")),
                _optional_text(row.get("run_id")),
                _first_text((row,), "pattern_source_name", "source_name"),
            )
        ] = row
    return indexed


def _matching_pattern_unit(
    pattern_units: Mapping[tuple[str | None, str | None, str | None, str | None], Mapping[str, Any]],
    condition_row: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    source_name = _first_text((condition_row,), "pattern_source_name", "source_name")
    key = (*_unit_key(condition_row), source_name)
    if key in pattern_units:
        return pattern_units[key]
    unit_key = _unit_key(condition_row)
    matches = [row for row_key, row in pattern_units.items() if row_key[:3] == unit_key]
    return matches[0] if len(matches) == 1 else None


def _unit_key(row: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        _optional_text(row.get("subject_id")),
        _optional_text(row.get("session_id")),
        _optional_text(row.get("run_id")),
    )


def _cv_unit(payload: Mapping[str, Any]) -> str | None:
    distances = _row_mappings(payload.get("distances"))
    if distances:
        value = _optional_text(distances[0].get("cv_unit"))
        if value is not None:
            return value
    distance_payload = payload.get("distance")
    if isinstance(distance_payload, Mapping):
        cross_validation = distance_payload.get("cross_validation")
        if isinstance(cross_validation, Mapping):
            return _optional_text(cross_validation.get("unit"))
    return None


def _noise_nonpositive_policy(payload: Mapping[str, Any]) -> str:
    distances = _row_mappings(payload.get("distances"))
    for row in distances:
        value = _optional_text(row.get("noise_nonpositive_policy") or row.get("nonpositive_noise_policy"))
        if value is not None:
            return _normalize_noise_nonpositive_policy(value)
    distance_payload = payload.get("distance")
    if isinstance(distance_payload, Mapping):
        noise_payload = distance_payload.get("noise_normalization")
        if isinstance(noise_payload, Mapping):
            value = _optional_text(noise_payload.get("nonpositive_policy") or noise_payload.get("nonpositive"))
            if value is not None:
                return _normalize_noise_nonpositive_policy(value)
    return _NOISE_NONPOSITIVE_POLICY_STRICT


def _normalize_noise_nonpositive_policy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", _NOISE_NONPOSITIVE_POLICY_STRICT, "fail", "drop_pattern"}:
        return _NOISE_NONPOSITIVE_POLICY_STRICT
    if normalized in {_NOISE_NONPOSITIVE_POLICY_DROP_FEATURES, "filter_nonpositive_features"}:
        return _NOISE_NONPOSITIVE_POLICY_DROP_FEATURES
    return _NOISE_NONPOSITIVE_POLICY_STRICT


def _grouping_columns(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = payload.get("grouping_columns")
    if raw is None:
        distance_payload = payload.get("distance")
        if isinstance(distance_payload, Mapping):
            raw = distance_payload.get("grouping_columns")
            cross_validation = distance_payload.get("cross_validation", distance_payload.get("cv"))
            if raw is None and isinstance(cross_validation, Mapping):
                raw = cross_validation.get("grouping_columns") or cross_validation.get("group_by")
    return tuple(_text_sequence(raw))


def _grouping_values(rows: Sequence[Mapping[str, Any]], grouping_columns: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in grouping_columns:
        for row in rows:
            if column not in row:
                continue
            value = _json_safe(row.get(column))
            if value is not None:
                values[column] = value
                break
    return values


def _safe_grouping_columns(grouping_values: Mapping[str, Any], *, existing: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(value)
        for key, value in grouping_values.items()
        if str(key) not in existing
    }


def _run_exclusion_policies(payload: Mapping[str, Any]) -> tuple[_RunExclusionPolicy, ...]:
    policies: list[_RunExclusionPolicy] = []
    for index, row in enumerate(_row_mappings(payload.get("exclusions")), start=1):
        subject_id = _optional_text(row.get("subject_id") or row.get("subject"))
        session_id = _optional_text(row.get("session_id") or row.get("session"))
        run_id = _optional_text(row.get("run_id") or row.get("run"))
        if subject_id is None or session_id is None or run_id is None:
            continue
        exclusion_id = _optional_text(row.get("id") or row.get("name")) or f"run-exclusion-{index}"
        reason = _optional_text(row.get("reason")) or exclusion_id
        source_config_field = _optional_text(row.get("source_config_field")) or "mvpa_set.exclusions.rules"
        policies.append(
            _RunExclusionPolicy(
                id=exclusion_id,
                subject_id=_strip_entity_prefix(subject_id, "sub"),
                session_id=_strip_entity_prefix(session_id, "ses"),
                run_id=_strip_entity_prefix(run_id, "run"),
                reason=reason,
                source_config_field=source_config_field,
            )
        )
    return tuple(policies)


def _matching_run_exclusion(
    policies: Sequence[_RunExclusionPolicy],
    metadata: _CandidateMetadata,
) -> _RunExclusionPolicy | None:
    subject_id = _strip_entity_prefix(metadata.subject_id, "sub")
    session_id = _strip_entity_prefix(metadata.session_id, "ses")
    run_id = _strip_entity_prefix(metadata.run_id, "run")
    for policy in policies:
        if (policy.subject_id, policy.session_id, policy.run_id) == (subject_id, session_id, run_id):
            return policy
    return None


def _run_exclusion_policy_rows(policies: Sequence[_RunExclusionPolicy]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "id": policy.id,
            "subject_id": policy.subject_id,
            "session_id": policy.session_id,
            "run_id": policy.run_id,
            "reason": policy.reason,
            "source_config_field": policy.source_config_field,
            "status": "configured",
        }
        for policy in policies
    )


def _excluded_run_rows(qc_rows: Sequence[MvpaPatternQcRow]) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for qc in qc_rows:
        if not qc.excluded:
            continue
        key = (
            qc.subject_id,
            qc.session_id,
            qc.run_id,
            qc.exclusion_id,
            qc.skipped_stage or "before_extraction",
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "subject_id": qc.subject_id,
                "session_id": qc.session_id,
                "run_id": qc.run_id,
                "reason": qc.exclusion_reason or qc.reason,
                "exclusion_id": qc.exclusion_id,
                "source_config_field": qc.exclusion_source_field,
                "skipped_stage": qc.skipped_stage or "before_extraction",
                "status": "excluded",
            }
        )
    return tuple(rows)


def _mean_centering_policy(payload: Mapping[str, Any]) -> _MeanCenteringPolicy:
    raw = payload.get("mean_centering")
    if not isinstance(raw, Mapping):
        return _MeanCenteringPolicy()
    enabled = _truthy(raw.get("enabled"))
    scope = _optional_text(raw.get("scope")) or ("roi" if enabled else "none")
    if scope == "within_roi":
        scope = "roi"
    if not enabled:
        scope = "none"
    return _MeanCenteringPolicy(enabled=enabled, scope=scope)


def _mean_centered_feature_values(
    values: np.ndarray,
    policy: _MeanCenteringPolicy,
) -> tuple[np.ndarray, bool, str]:
    if not policy.enabled or policy.scope != "roi":
        return values, False, "none"
    return values - float(np.mean(values)), True, "roi"


def _not_checked_or_not_requested(load_noise: bool) -> str:
    return "not_checked" if load_noise else "not_requested"


def _noise_provenance(*, load_noise: bool, qc_rows: Sequence[MvpaPatternQcRow]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in qc_rows:
        status_counts[row.noise_status] = status_counts.get(row.noise_status, 0) + 1
    return {
        "noise_requested_count": sum(
            1 for row in qc_rows if row.noise_status not in {"not_requested", "not_checked"}
        ),
        "noise_loaded_count": sum(1 for row in qc_rows if row.noise_loaded),
        "noise_usable_count": sum(1 for row in qc_rows if row.noise_usable),
        "noise_status_counts": status_counts or ({"not_requested": 0} if not load_noise else {"not_checked": 0}),
    }


def _event_threshold_rows(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = _row_mappings(payload.get("event_threshold_rows"))
    if rows:
        normalized: list[Mapping[str, Any]] = []
        for row in rows:
            row_payload = dict(row)
            row_payload["status"] = "not_evaluated"
            normalized.append(row_payload)
        return tuple(normalized)

    thresholds = payload.get("event_thresholds")
    if isinstance(thresholds, Mapping) and thresholds:
        return tuple(
            {"threshold": str(key), "value": _json_safe(value), "status": "not_evaluated"}
            for key, value in thresholds.items()
        )
    return ()


def _event_threshold_status(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "not_evaluated"
    if all(_row_status(row) == "not_evaluated" for row in rows):
        return "not_evaluated"
    return "not_evaluated"


def _row_status(row: Mapping[str, Any]) -> str:
    return (_optional_text(row.get("status")) or "ok").lower()


def _row_excluded(row: Mapping[str, Any]) -> bool:
    return _truthy(row.get("excluded")) or _row_status(row) == "excluded"


def _row_warnings(*rows: Mapping[str, Any] | None) -> tuple[str, ...]:
    warnings: list[str] = []
    for row in rows:
        if row is None:
            continue
        warnings.extend(_message_values(row.get("warnings")))
        message = _optional_text(row.get("warning"))
        if message is not None:
            warnings.append(message)
    return tuple(_unique_text(warnings))


def _row_errors(*rows: Mapping[str, Any] | None) -> tuple[str, ...]:
    errors: list[str] = []
    for row in rows:
        if row is None:
            continue
        errors.extend(_message_values(row.get("errors")))
        message = _optional_text(row.get("error"))
        if message is not None:
            errors.append(message)
    return tuple(_unique_text(errors))


def _message_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(text for item in value if (text := _optional_text(item)) is not None)
    text = _optional_text(value)
    return (text,) if text is not None else ()


def _text_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[str] = []
        for item in value:
            text = _optional_text(item)
            if text is not None:
                values.append(text)
        return tuple(values)
    text = _optional_text(value)
    return (text,) if text is not None else ()


def _first_text(rows: Sequence[Mapping[str, Any]], *keys: str) -> str | None:
    for row in rows:
        for key in keys:
            if key in row:
                text = _optional_text(row.get(key))
                if text is not None:
                    return text
    return None


def _first_int(rows: Sequence[Mapping[str, Any]], *keys: str) -> int | None:
    for row in rows:
        for key in keys:
            if key not in row or row.get(key) is None:
                continue
            try:
                return int(row.get(key))
            except (TypeError, ValueError):
                continue
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strip_entity_prefix(value: str | None, prefix: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    marker = f"{prefix}-"
    suffix = text[len(marker) :] if text.startswith(marker) else None
    return suffix if suffix and suffix[0].isdigit() else text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _path_exists(path: str | None) -> bool | None:
    if path is None:
        return None
    return Path(path).exists()


def _voxel_index_hash(voxel_indices: np.ndarray) -> str:
    payload = ",".join(str(int(index)) for index in voxel_indices.tolist())
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _finite_summary(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    return float(np.min(values)), float(np.max(values)), float(np.mean(values))


def _pattern_id(metadata: _CandidateMetadata, voxel_index_hash: str) -> str:
    parts = (
        metadata.subject_id,
        metadata.session_id,
        metadata.run_id,
        metadata.condition_id,
        metadata.pattern_source_name,
        metadata.roi_source_name,
        metadata.roi_label,
        voxel_index_hash[:16],
    )
    return "__".join(_safe_identifier(part) for part in parts if part is not None)


def _safe_identifier(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


def _raise_if_fail_fast(fail_fast: bool, qc: MvpaPatternQcRow) -> None:
    if not fail_fast:
        return
    if qc.usable or qc.status in {"excluded", "skipped", "not_executable_plan_status"}:
        return
    raise ValueError(qc.reason or qc.status)


def _extend_messages(target: list[str], messages: Sequence[str]) -> None:
    seen = set(target)
    for message in messages:
        if message not in seen:
            target.append(message)
            seen.add(message)


def _unique_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            output.append(text)
            seen.add(text)
    return output


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
    if isinstance(value, np.ndarray):
        return [_json_safe(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return str(value)


__all__ = [
    "MvpaPatternExtractionResult",
    "MvpaPatternExtractionRow",
    "MvpaPatternQcRow",
    "extract_mvpa_patterns_from_discovery_plan",
]
