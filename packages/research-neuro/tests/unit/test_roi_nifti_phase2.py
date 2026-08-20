from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import tempfile
import types
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - local minimal environments may skip.
    np = None  # type: ignore[assignment]
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - local minimal environments may skip.
    nib = None  # type: ignore[assignment]

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
BIDS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-bids"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(BIDS_PACKAGE_ROOT / "src"))

from research_platform.bids.roi import build_roi_extraction_table_path, build_roi_mask_path, build_roi_sidecar_path
from research_platform.neuro.nifti import (
    GeometryMismatchError,
    create_sphere_mask,
    validate_compatible_geometry,
    voxel_to_xyz,
    xyz_to_voxel,
)
from research_platform.neuro.roi_builders import (
    build_coordinate_sphere_roi,
    build_functional_threshold_map_roi,
    copy_manual_mask_roi,
)
from research_platform.neuro.roi_extraction import (
    build_extraction_result,
    extract_roi_metrics,
    extraction_result_to_row,
    write_extraction_table,
)
from research_platform.neuro.roi_masks import check_min_voxel_count, intersect_masks, write_roi_sidecar
from research_platform.neuro.roi_peaks import select_peak_from_map


@unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for Phase 2 NIfTI ROI tests")
class RoiNiftiPhase2Tests(unittest.TestCase):
    def test_xyz_mm_to_voxel_and_voxel_to_xyz_conversion(self) -> None:
        affine = np.array(
            [
                [2, 0, 0, 10],
                [0, 3, 0, 20],
                [0, 0, 4, 30],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        voxel = xyz_to_voxel([14, 26, 38], affine, shape=(5, 5, 5))
        xyz = voxel_to_xyz(voxel, affine)

        self.assertEqual(voxel, (2, 2, 2))
        self.assertEqual(xyz, (14.0, 26.0, 38.0))

    def test_affine_and_shape_compatibility_checks(self) -> None:
        reference = _image(np.zeros((3, 3, 3)))
        matching = _image(np.ones((3, 3, 3)))
        shape_mismatch = _image(np.ones((4, 3, 3)))
        affine_mismatch = _image(np.ones((3, 3, 3)), affine=np.diag([2, 2, 2, 1]))

        validate_compatible_geometry(reference, matching)
        with self.assertRaises(GeometryMismatchError):
            validate_compatible_geometry(reference, shape_mismatch)
        with self.assertRaises(GeometryMismatchError):
            validate_compatible_geometry(reference, affine_mismatch)

    def test_sphere_mask_creation_counts_center_and_face_neighbors(self) -> None:
        mask = create_sphere_mask((5, 5, 5), np.eye(4), center_xyz_mm=[2, 2, 2], radius_mm=1.01)

        self.assertEqual(int(np.count_nonzero(mask)), 7)
        self.assertTrue(mask[2, 2, 2])
        self.assertTrue(mask[1, 2, 2])
        self.assertFalse(mask[1, 1, 2])

    def test_mask_intersection(self) -> None:
        left = np.zeros((3, 3, 3), dtype=np.uint8)
        right = np.zeros((3, 3, 3), dtype=np.uint8)
        left[1, 1, 1] = 1
        left[2, 2, 2] = 1
        right[1, 1, 1] = 1

        intersection = intersect_masks(left, right)

        self.assertEqual(int(np.count_nonzero(intersection)), 1)
        self.assertTrue(intersection[1, 1, 1])

    def test_thresholded_peak_selection_with_search_mask(self) -> None:
        data = np.zeros((5, 5, 5), dtype=float)
        data[1, 1, 1] = 4.0
        data[3, 3, 3] = 6.0
        search_mask = np.zeros((5, 5, 5), dtype=np.uint8)
        search_mask[1, 1, 1] = 1
        search_mask[3, 3, 3] = 1
        stat_image = _image(data)

        peak = select_peak_from_map(stat_image, threshold=5.0, search_mask=search_mask)

        self.assertEqual(peak.voxel_index, (3, 3, 3))
        self.assertEqual(peak.fallback_status, "thresholded")
        self.assertEqual(peak.thresholded_voxel_count, 1)

    def test_below_threshold_fallback_peak_selection(self) -> None:
        data = np.zeros((5, 5, 5), dtype=float)
        data[2, 2, 2] = 4.0

        peak = select_peak_from_map(
            _image(data),
            threshold=10.0,
            allow_below_threshold_fallback=True,
        )

        self.assertEqual(peak.voxel_index, (2, 2, 2))
        self.assertEqual(peak.fallback_status, "below_threshold_fallback")
        self.assertEqual(peak.thresholded_voxel_count, 0)

    def test_minimum_voxel_count_qc(self) -> None:
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[1, 1, 1] = 1

        qc = check_min_voxel_count(mask, min_voxels_warn=4, min_voxels_fail=3)

        self.assertFalse(qc.passed)
        self.assertIn("fail_min_voxels", qc.qc_flags)
        self.assertIn("warn_min_voxels", qc.qc_flags)

    def test_manual_mask_validation_copy_and_sidecar_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "manual_mask.nii.gz"
            source_data = np.zeros((4, 4, 4), dtype=np.uint8)
            source_data[1, 1, 1] = 1
            nib.save(_image(source_data), source_path)
            mask_path = build_roi_mask_path(
                root / "derivatives",
                subject_id="001",
                task_id="exampletask",
                space="MNI152NLin2009cAsym",
                roi_label="ManualMask",
                method_desc="CuratedMask",
            )
            sidecar_path = build_roi_sidecar_path(mask_path)

            result = copy_manual_mask_roi(
                source_mask_path=source_path,
                output_mask_path=mask_path,
                sidecar_path=sidecar_path,
                roi_label="ManualMask",
                desc="CuratedMask",
                source_ref="manual_masks/sub-001_label-ManualMask_mask.nii.gz",
            )

            copied = nib.load(result.mask_path).get_fdata()
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(result.voxel_count, 1)
            self.assertEqual(int(np.count_nonzero(copied)), 1)
            self.assertEqual(sidecar["roi_family"], "manual_mask")
            self.assertEqual(sidecar["source_ref"], "manual_masks/sub-001_label-ManualMask_mask.nii.gz")

    def test_writing_sidecar_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = Path(tmpdir) / "sub-001_label-SeedSphere_desc-CoordinateSphere_mask.json"

            output = write_roi_sidecar(
                sidecar_path,
                {
                    "roi_label": "SeedSphere",
                    "roi_family": "coordinate_sphere",
                    "backend": "generic_nifti",
                    "coordinate": [0, 0, 0],
                    "radius_mm": 2,
                    "fallback_status": "not_applicable",
                    "voxel_count": 7,
                    "warnings": [],
                    "qc_flags": ["pass"],
                },
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["roi_label"], "SeedSphere")
            self.assertEqual(payload["voxel_count"], 7)

    def test_functional_threshold_map_builder_writes_peak_sphere_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = np.zeros((5, 5, 5), dtype=float)
            data[2, 2, 2] = 5.5
            data[1, 1, 1] = 4.0
            mask_path = build_roi_mask_path(
                root / "derivatives",
                subject_id="001",
                task_id="exampletask",
                space="MNI152NLin2009cAsym",
                roi_label="PeakSphere",
                method_desc="ExistingMap",
            )
            sidecar_path = build_roi_sidecar_path(mask_path)

            result = build_functional_threshold_map_roi(
                stat_image=_image(data),
                roi_label="PeakSphere",
                desc="ExistingMap",
                output_mask_path=mask_path,
                sidecar_path=sidecar_path,
                sphere_radius_mm=1.01,
                threshold=5.0,
                seed_xyz_mm=[2, 2, 2],
                search_radius_mm=3,
            )

            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(result.peak.voxel_index, (2, 2, 2))
            self.assertEqual(result.voxel_count, 7)
            self.assertEqual(sidecar["fallback_status"], "thresholded")
            self.assertEqual(sidecar["selected_peak_stat"], 5.5)
            self.assertEqual(sidecar["roi_family"], "functional_threshold_map")

    def test_generic_extraction_metrics(self) -> None:
        values = np.arange(27, dtype=float).reshape((3, 3, 3))
        values[0, 0, 2] = np.nan
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[0, 0, 1] = 1
        mask[0, 0, 2] = 1

        metrics = extract_roi_metrics(_image(values), _image(mask))

        self.assertEqual(metrics["voxel_count"], 3)
        self.assertEqual(metrics["valid_voxel_count"], 2)
        self.assertEqual(metrics["mean"], 0.5)
        self.assertEqual(metrics["median"], 0.5)
        self.assertEqual(metrics["sum"], 1.0)
        self.assertEqual(metrics["std"], 0.5)
        self.assertEqual(metrics["min"], 0.0)
        self.assertEqual(metrics["max"], 1.0)

    def test_generic_extraction_table_uses_bids_like_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            values = np.arange(8, dtype=float).reshape((2, 2, 2))
            mask = np.ones((2, 2, 2), dtype=np.uint8)
            result = build_extraction_result(
                _image(values),
                _image(mask),
                roi_label="SeedSphere",
                value_desc="ExampleValue",
            )
            table_path = build_roi_extraction_table_path(
                root / "derivatives",
                subject_id="001",
                task_id="exampletask",
                space="MNI152NLin2009cAsym",
                extraction_desc="ExampleValue",
            )

            output = write_extraction_table([extraction_result_to_row(result)], table_path)

            self.assertTrue(output.exists())
            self.assertIn("desc-ExampleValue_roiextract.tsv", output.name)
            self.assertIn("valid_voxel_count", output.read_text(encoding="utf-8").splitlines()[0])

    def test_bids_like_output_path_integration_for_coordinate_sphere(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mask_path = build_roi_mask_path(
                root / "derivatives",
                subject_id="001",
                session_id="01",
                task_id="exampletask",
                direction="AP",
                space="MNI152NLin2009cAsym",
                roi_label="SeedSphere",
                method_desc="CoordinateSphere",
            )
            sidecar_path = build_roi_sidecar_path(mask_path)

            result = build_coordinate_sphere_roi(
                reference_image=_image(np.zeros((5, 5, 5), dtype=float)),
                center_xyz_mm=[2, 2, 2],
                radius_mm=1.01,
                roi_label="SeedSphere",
                desc="CoordinateSphere",
                output_mask_path=mask_path,
                sidecar_path=sidecar_path,
            )

            self.assertEqual(result.voxel_count, 7)
            self.assertTrue(result.mask_path.exists())
            self.assertTrue(result.sidecar_path.exists())
            self.assertIn("label-SeedSphere_desc-CoordinateSphere_mask.nii.gz", result.mask_path.name)
            self.assertFalse(math.isnan(json.loads(sidecar_path.read_text(encoding="utf-8"))["voxel_volume_mm3"]))


def _image(data: np.ndarray, *, affine: np.ndarray | None = None) -> object:
    return nib.Nifti1Image(np.asarray(data), affine if affine is not None else np.eye(4))


if __name__ == "__main__":
    unittest.main()
