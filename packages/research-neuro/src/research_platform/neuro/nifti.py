
"""NIfTI IO and geometry helpers for generic ROI workflows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:  # Keep config-only imports independent from the image runtime.
    import nibabel as nib
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    nib = None  # type: ignore[assignment]


RoundMode = Literal["nearest", "floor", "ceil"]


class NiftiDependencyError(RuntimeError):
    """Raised when a NIfTI helper is used without nibabel installed."""


class GeometryMismatchError(ValueError):
    """Raised when NIfTI images do not share compatible geometry."""


def load_nifti_image(path: str | Path) -> Any:
    """Load a NIfTI image from disk using nibabel."""

    return _require_nibabel().load(str(path))


def save_nifti_image(image: Any, path: str | Path) -> Path:
    """Save a NIfTI image to disk, creating parent directories as needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_nibabel().save(image, str(output_path))
    return output_path


def spatial_shape(image: Any) -> tuple[int, int, int]:
    """Return the first three dimensions of a NIfTI-like image."""

    shape = tuple(int(value) for value in image.shape)
    if len(shape) < 3:
        raise ValueError("NIfTI image must have at least three spatial dimensions.")
    return shape[:3]


def affine_matrix(image_or_affine: Any) -> np.ndarray:
    """Return a 4x4 affine matrix from an image or matrix-like object."""

    affine = getattr(image_or_affine, "affine", image_or_affine)
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Affine must be a 4x4 matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Affine must contain only finite values.")
    return matrix


def validate_compatible_geometry(
    reference_image: Any,
    other_image: Any,
    *,
    check_shape: bool = True,
    check_affine: bool = True,
    atol: float = 1e-5,
) -> None:
    """Raise if two images do not share compatible spatial shape and affine."""

    if check_shape and spatial_shape(reference_image) != spatial_shape(other_image):
        raise GeometryMismatchError(
            f"Spatial shapes differ: {spatial_shape(reference_image)} != {spatial_shape(other_image)}."
        )
    if check_affine and not np.allclose(
        affine_matrix(reference_image),
        affine_matrix(other_image),
        rtol=0.0,
        atol=atol,
    ):
        raise GeometryMismatchError("Affine matrices differ beyond tolerance.")


def xyz_to_voxel(
    xyz_mm: Sequence[float],
    affine: Any,
    *,
    round_mode: RoundMode = "nearest",
    shape: Sequence[int] | None = None,
    clip: bool = False,
) -> tuple[int, int, int]:
    """Convert xyz millimeter coordinates into voxel indices."""

    xyz = _coordinate_array(xyz_mm, "xyz_mm")
    voxel_float = np.linalg.inv(affine_matrix(affine)) @ np.array([*xyz, 1.0])
    rounded = _round_voxel(voxel_float[:3], round_mode)
    if shape is not None:
        shape3 = _shape_tuple(shape)
        if clip:
            rounded = np.clip(rounded, [0, 0, 0], np.array(shape3) - 1)
        elif not _inside_shape(rounded, shape3):
            raise IndexError(f"Voxel index {tuple(int(value) for value in rounded)} is outside shape {shape3}.")
    return tuple(int(value) for value in rounded)


def voxel_to_xyz(voxel_index: Sequence[int], affine: Any) -> tuple[float, float, float]:
    """Convert voxel indices into xyz millimeter coordinates."""

    voxel = _coordinate_array(voxel_index, "voxel_index")
    xyz = affine_matrix(affine) @ np.array([*voxel, 1.0])
    return tuple(float(value) for value in xyz[:3])


def create_sphere_mask(
    shape: Sequence[int],
    affine: Any,
    center_xyz_mm: Sequence[float],
    radius_mm: float,
    *,
    include_boundary: bool = True,
) -> np.ndarray:
    """Create a 3D boolean mask containing voxel centers inside a sphere."""

    shape3 = _shape_tuple(shape)
    radius = float(radius_mm)
    if radius <= 0:
        raise ValueError("radius_mm must be greater than zero.")

    center = _coordinate_array(center_xyz_mm, "center_xyz_mm")
    grid = np.indices(shape3, dtype=float).reshape(3, -1).T
    homogeneous = np.column_stack([grid, np.ones(grid.shape[0], dtype=float)])
    xyz = (homogeneous @ affine_matrix(affine).T)[:, :3]
    distances = np.linalg.norm(xyz - center, axis=1)
    tolerance = 1e-9 if include_boundary else -1e-9
    return (distances <= radius + tolerance).reshape(shape3)


def count_voxels(mask: Any) -> int:
    """Count non-zero voxels in an array-like mask."""

    return int(np.count_nonzero(np.asarray(mask)))


def voxel_volume_mm3(image_or_affine: Any) -> float:
    """Return the voxel volume implied by an affine matrix, in cubic mm."""

    return float(abs(np.linalg.det(affine_matrix(image_or_affine)[:3, :3])))


def image_geometry_metadata(image: Any, *, mask: Any | None = None) -> dict[str, Any]:
    """Return shape and voxel-volume metadata for an image and optional mask."""

    metadata: dict[str, Any] = {
        "shape": list(spatial_shape(image)),
        "voxel_volume_mm3": voxel_volume_mm3(image),
    }
    if mask is not None:
        voxel_count = count_voxels(mask)
        metadata["voxel_count"] = voxel_count
        metadata["volume_mm3"] = voxel_count * metadata["voxel_volume_mm3"]
    return metadata


def make_nifti_like(reference_image: Any, data: Any, *, dtype: Any | None = None) -> Any:
    """Create a NIfTI image with data using a reference image's affine/header."""

    nib_module = _require_nibabel()
    array = np.asarray(data)
    if array.shape[:3] != spatial_shape(reference_image):
        raise GeometryMismatchError(
            f"Data spatial shape {array.shape[:3]} does not match reference {spatial_shape(reference_image)}."
        )
    header = reference_image.header.copy()
    if dtype is not None:
        array = array.astype(dtype)
        header.set_data_dtype(np.dtype(dtype))
    return nib_module.Nifti1Image(array, affine_matrix(reference_image), header=header)


def _require_nibabel() -> Any:
    if nib is None:
        raise NiftiDependencyError("nibabel is required for NIfTI image IO helpers.")
    return nib


def _shape_tuple(shape: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in shape[:3])
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("shape must contain three positive spatial dimensions.")
    return values


def _coordinate_array(values: Sequence[float], label: str) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values.")
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain exactly three finite numeric values.")
    return array


def _round_voxel(values: np.ndarray, round_mode: RoundMode) -> np.ndarray:
    if round_mode == "nearest":
        return np.rint(values).astype(int)
    if round_mode == "floor":
        return np.floor(values).astype(int)
    if round_mode == "ceil":
        return np.ceil(values).astype(int)
    raise ValueError("round_mode must be one of: nearest, floor, ceil.")


def _inside_shape(voxel_index: np.ndarray, shape: tuple[int, int, int]) -> bool:
    return bool(np.all(voxel_index >= 0) and np.all(voxel_index < np.asarray(shape)))
