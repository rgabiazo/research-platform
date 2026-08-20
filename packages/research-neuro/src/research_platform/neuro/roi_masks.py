"""Mask operations and sidecar writing for generic NIfTI ROI workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from research_platform.neuro.nifti import (
    image_geometry_metadata,
    make_nifti_like,
    save_nifti_image,
    validate_compatible_geometry,
)
from research_platform.neuro.roi import normalize_sidecar_provenance, validate_roi_sidecar_document


@dataclass(frozen=True)
class CoverageApplication:
    """Result of intersecting an ROI mask with optional coverage masks."""

    mask: np.ndarray
    applied_masks: tuple[str, ...]


@dataclass(frozen=True)
class VoxelCountQc:
    """Minimum voxel-count QC result."""

    voxel_count: int
    passed: bool
    warnings: tuple[str, ...]
    qc_flags: tuple[str, ...]


def validate_binary_mask(mask: Any, *, allow_empty: bool = False, label: str = "mask") -> np.ndarray:
    """Return a boolean mask after validating that values are binary."""

    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"{label} must be a 3D mask.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    unique_values = np.unique(array)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"{label} must contain only binary values.")
    boolean_mask = array.astype(bool)
    if not allow_empty and not np.any(boolean_mask):
        raise ValueError(f"{label} must contain at least one voxel.")
    return boolean_mask


def intersect_masks(*masks: Any, allow_empty: bool = True) -> np.ndarray:
    """Return the binary intersection of two or more masks."""

    if not masks:
        raise ValueError("At least one mask is required.")
    normalized = [
        validate_binary_mask(mask, allow_empty=allow_empty, label=f"masks[{index}]")
        for index, mask in enumerate(masks)
    ]
    first_shape = normalized[0].shape
    for index, mask in enumerate(normalized[1:], start=1):
        if mask.shape != first_shape:
            raise ValueError(f"masks[{index}] shape {mask.shape} does not match {first_shape}.")
    return np.logical_and.reduce(normalized)


def apply_coverage_masks(
    mask: Any,
    *,
    group_mask: Any | None = None,
    subject_mask: Any | None = None,
    run_mask: Any | None = None,
    coverage_masks: Mapping[str, Any] | None = None,
) -> CoverageApplication:
    """Apply supplied group, subject, run, and named coverage masks by intersection."""

    base = validate_binary_mask(mask, allow_empty=True, label="mask")
    named_masks: list[tuple[str, Any]] = [
        ("group", group_mask),
        ("subject", subject_mask),
        ("run", run_mask),
    ]
    if coverage_masks:
        named_masks.extend((str(name), value) for name, value in coverage_masks.items())

    applied: list[str] = []
    output = base.copy()
    for name, coverage_mask in named_masks:
        if coverage_mask is None:
            continue
        coverage = validate_binary_mask(coverage_mask, allow_empty=True, label=f"{name}_coverage_mask")
        if coverage.shape != output.shape:
            raise ValueError(f"{name}_coverage_mask shape {coverage.shape} does not match ROI mask shape {output.shape}.")
        output = np.logical_and(output, coverage)
        applied.append(name)
    return CoverageApplication(mask=output, applied_masks=tuple(applied))


def check_min_voxel_count(
    mask: Any,
    *,
    min_voxels_warn: int | None = None,
    min_voxels_fail: int | None = None,
) -> VoxelCountQc:
    """Evaluate minimum voxel-count warn/fail thresholds for a mask."""

    normalized = validate_binary_mask(mask, allow_empty=True)
    voxel_count = int(np.count_nonzero(normalized))
    warnings: list[str] = []
    qc_flags: list[str] = []
    passed = True

    if min_voxels_fail is not None and voxel_count < int(min_voxels_fail):
        passed = False
        qc_flags.append("fail_min_voxels")
        warnings.append(f"voxel_count {voxel_count} is below min_voxels_fail {int(min_voxels_fail)}")
    if min_voxels_warn is not None and voxel_count < int(min_voxels_warn):
        qc_flags.append("warn_min_voxels")
        warnings.append(f"voxel_count {voxel_count} is below min_voxels_warn {int(min_voxels_warn)}")
    if passed and not qc_flags:
        qc_flags.append("pass")

    return VoxelCountQc(
        voxel_count=voxel_count,
        passed=passed,
        warnings=tuple(warnings),
        qc_flags=tuple(qc_flags),
    )


def write_roi_nifti_mask(mask: Any, reference_image: Any, output_path: str | Path) -> Path:
    """Write a binary ROI mask as a uint8 NIfTI image."""

    normalized = validate_binary_mask(mask, allow_empty=True)
    image = make_nifti_like(reference_image, normalized.astype(np.uint8), dtype=np.uint8)
    return save_nifti_image(image, output_path)


def write_roi_sidecar(sidecar_path: str | Path, provenance: Mapping[str, Any]) -> Path:
    """Write and validate a Phase 1-compatible ROI JSON sidecar."""

    payload = normalize_sidecar_provenance(provenance)
    errors = validate_roi_sidecar_document(payload)
    if errors:
        raise ValueError("Invalid ROI sidecar provenance: " + "; ".join(errors))
    output_path = Path(sidecar_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def mask_sidecar_metadata(reference_image: Any, mask: Any) -> dict[str, Any]:
    """Return voxel-count and voxel-volume metadata for a mask sidecar."""

    normalized = validate_binary_mask(mask, allow_empty=True)
    return image_geometry_metadata(reference_image, mask=normalized)


def validate_mask_image_geometry(reference_image: Any, mask_image: Any) -> None:
    """Validate that a mask image has compatible shape and affine."""

    validate_compatible_geometry(reference_image, mask_image)
    validate_binary_mask(mask_image.get_fdata(), allow_empty=True)
