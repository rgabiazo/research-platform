"""Peak selection helpers for existing generic NIfTI statistical maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from research_platform.neuro.nifti import (
    create_sphere_mask,
    spatial_shape,
    validate_compatible_geometry,
    voxel_to_xyz,
)
from research_platform.neuro.roi_masks import validate_binary_mask


@dataclass(frozen=True)
class PeakSelection:
    """Selected peak from an existing statistical image."""

    voxel_index: tuple[int, int, int]
    coordinate_xyz_mm: tuple[float, float, float]
    value: float
    threshold: float | None
    fallback_status: str
    searched_voxel_count: int
    thresholded_voxel_count: int

    def provenance_fields(self) -> dict[str, Any]:
        """Return JSON-friendly provenance fields for ROI sidecars."""

        return {
            "selected_peak_voxel": list(self.voxel_index),
            "selected_peak_coordinate": [float(value) for value in self.coordinate_xyz_mm],
            "selected_peak_stat": float(self.value),
            "z_threshold": self.threshold,
            "fallback_status": self.fallback_status,
            "searched_voxel_count": self.searched_voxel_count,
            "thresholded_voxel_count": self.thresholded_voxel_count,
        }


def select_peak_from_map(
    stat_image: Any,
    *,
    threshold: float | None = None,
    search_mask_image: Any | None = None,
    search_mask: Any | None = None,
    seed_xyz_mm: Sequence[float] | None = None,
    search_radius_mm: float | None = None,
    allow_below_threshold_fallback: bool = False,
    local_maxima: bool = True,
) -> PeakSelection:
    """Select the strongest local maximum from an existing statistical map."""

    data = np.asarray(stat_image.get_fdata(dtype=np.float64))
    if data.ndim != 3:
        raise ValueError("stat_image must be a 3D statistical map.")

    domain_mask = np.isfinite(data)
    if search_mask_image is not None:
        validate_compatible_geometry(stat_image, search_mask_image)
        domain_mask &= validate_binary_mask(search_mask_image.get_fdata(), allow_empty=True, label="search_mask_image")
    if search_mask is not None:
        domain_mask &= validate_binary_mask(search_mask, allow_empty=True, label="search_mask")
    if seed_xyz_mm is not None:
        if search_radius_mm is None:
            raise ValueError("search_radius_mm is required when seed_xyz_mm is supplied.")
        domain_mask &= create_sphere_mask(spatial_shape(stat_image), stat_image.affine, seed_xyz_mm, search_radius_mm)
    elif search_radius_mm is not None:
        raise ValueError("seed_xyz_mm is required when search_radius_mm is supplied.")

    searched_voxel_count = int(np.count_nonzero(domain_mask))
    if searched_voxel_count == 0:
        raise ValueError("No voxels are available in the peak search domain.")

    candidate_mask = _local_maxima_mask(data, domain_mask) if local_maxima else domain_mask
    if not np.any(candidate_mask):
        candidate_mask = domain_mask

    if threshold is None:
        selected_mask = candidate_mask
        thresholded_count = int(np.count_nonzero(candidate_mask))
        fallback_status = "not_applicable"
    else:
        thresholded_mask = candidate_mask & (data >= float(threshold))
        thresholded_count = int(np.count_nonzero(thresholded_mask))
        if thresholded_count:
            selected_mask = thresholded_mask
            fallback_status = "thresholded"
        elif allow_below_threshold_fallback:
            selected_mask = candidate_mask
            fallback_status = "below_threshold_fallback"
        else:
            raise ValueError("No local maximum met the configured threshold.")

    voxel_index = _argmax_index(data, selected_mask)
    return PeakSelection(
        voxel_index=voxel_index,
        coordinate_xyz_mm=voxel_to_xyz(voxel_index, stat_image.affine),
        value=float(data[voxel_index]),
        threshold=float(threshold) if threshold is not None else None,
        fallback_status=fallback_status,
        searched_voxel_count=searched_voxel_count,
        thresholded_voxel_count=thresholded_count,
    )


def _local_maxima_mask(data: np.ndarray, domain_mask: np.ndarray) -> np.ndarray:
    maxima = np.zeros(domain_mask.shape, dtype=bool)
    for index in np.argwhere(domain_mask):
        i, j, k = (int(value) for value in index)
        slices = tuple(
            slice(max(axis - 1, 0), min(axis + 2, data.shape[dim]))
            for dim, axis in enumerate((i, j, k))
        )
        neighborhood_domain = domain_mask[slices]
        neighborhood_values = data[slices][neighborhood_domain]
        if neighborhood_values.size and data[i, j, k] >= np.max(neighborhood_values):
            maxima[i, j, k] = True
    return maxima


def _argmax_index(data: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("Cannot select a peak from an empty mask.")
    values = data[tuple(indices.T)]
    selected = indices[int(np.argmax(values))]
    return tuple(int(value) for value in selected)
