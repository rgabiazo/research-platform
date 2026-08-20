"""Generic NIfTI ROI builders for Phase 2 reusable ROI workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_platform.neuro.nifti import (
    create_sphere_mask,
    load_nifti_image,
    spatial_shape,
    validate_compatible_geometry,
)
from research_platform.neuro.roi_masks import (
    VoxelCountQc,
    apply_coverage_masks,
    check_min_voxel_count,
    mask_sidecar_metadata,
    validate_binary_mask,
    write_roi_nifti_mask,
    write_roi_sidecar,
)
from research_platform.neuro.roi_peaks import PeakSelection, select_peak_from_map


@dataclass(frozen=True)
class RoiBuildResult:
    """Result of building or copying one generic NIfTI ROI mask."""

    mask_path: Path
    sidecar_path: Path | None
    voxel_count: int
    qc: VoxelCountQc
    provenance: Mapping[str, Any]
    peak: PeakSelection | None = None


def build_coordinate_sphere_roi(
    *,
    reference_image: Any,
    center_xyz_mm: Sequence[float],
    radius_mm: float,
    roi_label: str,
    output_mask_path: str | Path,
    sidecar_path: str | Path | None = None,
    desc: str | None = None,
    coverage_masks: Mapping[str, Any] | None = None,
    min_voxels_warn: int | None = None,
    min_voxels_fail: int | None = None,
    fail_on_qc: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> RoiBuildResult:
    """Build a coordinate-sphere ROI mask from a reference image."""

    mask = create_sphere_mask(spatial_shape(reference_image), reference_image.affine, center_xyz_mm, radius_mm)
    payload = {
        "roi_label": roi_label,
        "roi_family": "coordinate_sphere",
        "backend": "generic_nifti",
        "desc": desc,
        "coordinate": [float(value) for value in center_xyz_mm],
        "radius_mm": float(radius_mm),
        "fallback_status": "not_applicable",
    }
    return _write_mask_with_sidecar(
        mask,
        reference_image=reference_image,
        output_mask_path=output_mask_path,
        sidecar_path=sidecar_path,
        coverage_masks=coverage_masks,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
        fail_on_qc=fail_on_qc,
        provenance=_merge_provenance(payload, provenance),
    )


def copy_manual_mask_roi(
    *,
    source_mask_path: str | Path,
    output_mask_path: str | Path,
    roi_label: str,
    reference_image: Any | None = None,
    sidecar_path: str | Path | None = None,
    desc: str | None = None,
    source_ref: str | None = None,
    coverage_masks: Mapping[str, Any] | None = None,
    min_voxels_warn: int | None = None,
    min_voxels_fail: int | None = None,
    fail_on_qc: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> RoiBuildResult:
    """Validate and copy a manual mask while writing Phase 1-compatible provenance."""

    source_image = load_nifti_image(source_mask_path)
    if reference_image is not None:
        validate_compatible_geometry(reference_image, source_image)
    else:
        reference_image = source_image
    mask = validate_binary_mask(source_image.get_fdata(), allow_empty=True, label="source_mask")
    payload: dict[str, Any] = {
        "roi_label": roi_label,
        "roi_family": "manual_mask",
        "backend": "manual",
        "desc": desc,
        "fallback_status": "not_applicable",
    }
    if source_ref:
        payload["source_ref"] = source_ref
    return _write_mask_with_sidecar(
        mask,
        reference_image=reference_image,
        output_mask_path=output_mask_path,
        sidecar_path=sidecar_path,
        coverage_masks=coverage_masks,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
        fail_on_qc=fail_on_qc,
        provenance=_merge_provenance(payload, provenance),
    )


def build_functional_threshold_map_roi(
    *,
    stat_image: Any,
    roi_label: str,
    output_mask_path: str | Path,
    sphere_radius_mm: float,
    threshold: float,
    sidecar_path: str | Path | None = None,
    desc: str | None = None,
    search_mask_image: Any | None = None,
    seed_xyz_mm: Sequence[float] | None = None,
    search_radius_mm: float | None = None,
    allow_below_threshold_fallback: bool = False,
    coverage_masks: Mapping[str, Any] | None = None,
    min_voxels_warn: int | None = None,
    min_voxels_fail: int | None = None,
    fail_on_qc: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> RoiBuildResult:
    """Build an ROI around a peak selected from an existing statistical map."""

    peak = select_peak_from_map(
        stat_image,
        threshold=threshold,
        search_mask_image=search_mask_image,
        seed_xyz_mm=seed_xyz_mm,
        search_radius_mm=search_radius_mm,
        allow_below_threshold_fallback=allow_below_threshold_fallback,
    )
    mask = create_sphere_mask(spatial_shape(stat_image), stat_image.affine, peak.coordinate_xyz_mm, sphere_radius_mm)
    payload: dict[str, Any] = {
        "roi_label": roi_label,
        "roi_family": "functional_threshold_map",
        "backend": "generic_nifti",
        "desc": desc,
        "sphere_radius_mm": float(sphere_radius_mm),
    }
    if seed_xyz_mm is not None:
        payload["seed_coordinate"] = [float(value) for value in seed_xyz_mm]
    if search_radius_mm is not None:
        payload["search_radius_mm"] = float(search_radius_mm)
    payload.update(peak.provenance_fields())

    result = _write_mask_with_sidecar(
        mask,
        reference_image=stat_image,
        output_mask_path=output_mask_path,
        sidecar_path=sidecar_path,
        coverage_masks=coverage_masks,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
        fail_on_qc=fail_on_qc,
        provenance=_merge_provenance(payload, provenance),
    )
    return RoiBuildResult(
        mask_path=result.mask_path,
        sidecar_path=result.sidecar_path,
        voxel_count=result.voxel_count,
        qc=result.qc,
        provenance=result.provenance,
        peak=peak,
    )


def build_loso_group_map_roi(
    *,
    zstat_image: Any,
    roi_label: str,
    output_mask_path: str | Path,
    sphere_radius_mm: float,
    z_threshold: float,
    seed_xyz_mm: Sequence[float],
    search_radius_mm: float,
    sidecar_path: str | Path | None = None,
    desc: str | None = None,
    allow_below_threshold_fallback: bool = False,
    exploratory_z_threshold: float | None = None,
    coverage_masks: Mapping[str, Any] | None = None,
    min_voxels_warn: int | None = None,
    min_voxels_fail: int | None = None,
    fail_on_qc: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> RoiBuildResult:
    """Build a LOSO ROI mask around a peak in a held-out-safe group zstat map."""

    peak = select_peak_from_map(
        zstat_image,
        threshold=z_threshold,
        seed_xyz_mm=seed_xyz_mm,
        search_radius_mm=search_radius_mm,
        allow_below_threshold_fallback=allow_below_threshold_fallback,
    )
    mask = create_sphere_mask(spatial_shape(zstat_image), zstat_image.affine, peak.coordinate_xyz_mm, sphere_radius_mm)
    payload: dict[str, Any] = {
        "roi_label": roi_label,
        "roi_family": "loso_group_map",
        "backend": "fsl_flame1",
        "desc": desc,
        "seed_coordinate": [float(value) for value in seed_xyz_mm],
        "search_radius_mm": float(search_radius_mm),
        "sphere_radius_mm": float(sphere_radius_mm),
        "z_threshold": float(z_threshold),
    }
    if exploratory_z_threshold is not None:
        payload["exploratory_z_threshold"] = float(exploratory_z_threshold)
    payload.update(peak.provenance_fields())
    payload["loso_peak_coordinate"] = payload["selected_peak_coordinate"]
    payload["selected_peak_z"] = payload["selected_peak_stat"]

    result = _write_mask_with_sidecar(
        mask,
        reference_image=zstat_image,
        output_mask_path=output_mask_path,
        sidecar_path=sidecar_path,
        coverage_masks=coverage_masks,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
        fail_on_qc=fail_on_qc,
        provenance=_merge_provenance(payload, provenance),
    )
    return RoiBuildResult(
        mask_path=result.mask_path,
        sidecar_path=result.sidecar_path,
        voxel_count=result.voxel_count,
        qc=result.qc,
        provenance=result.provenance,
        peak=peak,
    )


def _write_mask_with_sidecar(
    mask: Any,
    *,
    reference_image: Any,
    output_mask_path: str | Path,
    sidecar_path: str | Path | None,
    coverage_masks: Mapping[str, Any] | None,
    min_voxels_warn: int | None,
    min_voxels_fail: int | None,
    fail_on_qc: bool,
    provenance: Mapping[str, Any],
) -> RoiBuildResult:
    coverage = apply_coverage_masks(mask, coverage_masks=coverage_masks)
    qc = check_min_voxel_count(
        coverage.mask,
        min_voxels_warn=min_voxels_warn,
        min_voxels_fail=min_voxels_fail,
    )
    if fail_on_qc and not qc.passed:
        raise ValueError("; ".join(qc.warnings) or "ROI mask failed voxel-count QC.")

    mask_path = write_roi_nifti_mask(coverage.mask, reference_image, output_mask_path)
    metadata = mask_sidecar_metadata(reference_image, coverage.mask)
    payload = {
        key: value
        for key, value in {
            **dict(provenance),
            **metadata,
            "voxel_count": qc.voxel_count,
            "mask_intersection_policy": "intersection" if coverage.applied_masks else "none",
            "coverage_masks": list(coverage.applied_masks),
            "warnings": list(qc.warnings),
            "qc_flags": list(qc.qc_flags),
        }.items()
        if value is not None
    }

    written_sidecar_path: Path | None = None
    if sidecar_path is not None:
        written_sidecar_path = write_roi_sidecar(sidecar_path, payload)

    return RoiBuildResult(
        mask_path=mask_path,
        sidecar_path=written_sidecar_path,
        voxel_count=qc.voxel_count,
        qc=qc,
        provenance=payload,
    )


def _merge_provenance(base: Mapping[str, Any], extra: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = {key: value for key, value in dict(base).items() if value is not None}
    if extra:
        payload.update(dict(extra))
    return payload
