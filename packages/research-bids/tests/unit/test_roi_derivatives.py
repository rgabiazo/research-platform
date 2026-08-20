from __future__ import annotations

from pathlib import Path
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.bids.roi import (
    build_loso_flame1_mask_path,
    build_loso_flame1_roistats_path,
    build_loso_flame1_statmap_path,
    build_loso_group_map_path,
    build_roi_extraction_table_path,
    build_roi_group_extraction_table_path,
    build_roi_mask_path,
    build_roi_sidecar_path,
    build_roi_sidecar_path_from_entities,
)


class RoiDerivativePathTests(unittest.TestCase):
    def test_roi_mask_path_defaults_to_label_and_desc_entities(self) -> None:
        path = build_roi_mask_path(
            "derivatives",
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin2009cAsym",
            resolution="2",
            roi_label="Hippocampus",
            method_desc="ModelAContrast",
        )

        self.assertEqual(
            path,
            Path(
                "derivatives/roi/sub-001/ses-01/func/"
                "sub-001_ses-01_task-memory_dir-AP_space-MNI152NLin2009cAsym_res-2_"
                "label-Hippocampus_desc-ModelAContrast_mask.nii.gz"
            ),
        )

    def test_roi_json_sidecar_uses_mask_stem(self) -> None:
        mask_path = build_roi_mask_path(
            "derivatives",
            subject_id="sub-001",
            task_id="memory",
            space="MNI152NLin2009cAsym",
            roi_label="SeedSphere",
            method_desc="CoordinateSphere",
        )

        self.assertEqual(
            build_roi_sidecar_path(mask_path),
            Path(
                "derivatives/roi/sub-001/func/"
                "sub-001_task-memory_space-MNI152NLin2009cAsym_label-SeedSphere_"
                "desc-CoordinateSphere_mask.json"
            ),
        )
        self.assertEqual(
            build_roi_sidecar_path_from_entities(
                "derivatives",
                subject_id="001",
                task_id="memory",
                space="MNI152NLin2009cAsym",
                roi_label="SeedSphere",
                method_desc="CoordinateSphere",
            ),
            build_roi_sidecar_path(mask_path),
        )

    def test_loso_group_map_path_is_bids_like_and_records_heldout_entity(self) -> None:
        path = build_loso_group_map_path(
            "derivatives",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin2009cAsym",
            roi_label="SeedSphere",
            method_desc="ModelAContrast",
            heldout_subject="sub-001",
            statistic="z",
        )

        self.assertEqual(
            path,
            Path(
                "derivatives/roi/group/ses-01/func/"
                "ses-01_task-memory_dir-AP_space-MNI152NLin2009cAsym_label-SeedSphere_"
                "desc-ModelAContrast_stat-z_heldout-sub001_groupmap.nii.gz"
            ),
        )

    def test_loso_group_map_path_can_use_zstat_suffix_without_stat_entity(self) -> None:
        path = build_loso_group_map_path(
            "derivatives/loso_groupmaps/modelA",
            session_id="01",
            task_id="memory",
            space="MNI152NLin2009cAsym",
            method_desc="ModelAContrast",
            heldout_subject="001",
            statistic=None,
            suffix="zstat",
            pipeline_name=None,
        )

        self.assertEqual(
            path,
            Path(
                "derivatives/loso_groupmaps/modelA/group/ses-01/func/"
                "ses-01_task-memory_space-MNI152NLin2009cAsym_desc-ModelAContrast_"
                "heldout-sub001_zstat.nii.gz"
            ),
        )

    def test_extraction_table_path_uses_subject_directory_and_desc(self) -> None:
        path = build_roi_extraction_table_path(
            "derivatives",
            subject_id="001",
            session_id="01",
            task_id="memory",
            space="MNI152NLin2009cAsym",
            extraction_desc="ModelATimeseries",
        )

        self.assertEqual(
            path,
            Path(
                "derivatives/roi/sub-001/ses-01/func/"
                "sub-001_ses-01_task-memory_space-MNI152NLin2009cAsym_"
                "desc-ModelATimeseries_roiextract.tsv"
            ),
        )

    def test_group_extraction_table_path_is_bids_like(self) -> None:
        path = build_roi_group_extraction_table_path(
            "derivatives/roi_extract/modelA",
            session_id="01",
            task_id="memory",
            extraction_desc="ModelAFeatquery",
            pipeline_name=None,
        )

        self.assertEqual(
            path,
            Path("derivatives/roi_extract/modelA/group/ses-01/group_ses-01_task-memory_desc-ModelAFeatquery_values.tsv"),
        )

    def test_published_loso_flame1_statmap_path_matches_requested_layout(self) -> None:
        path = build_loso_flame1_statmap_path(
            "derivatives/roi-loso-flame1",
            session_id="ses-01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            contrast_alias="PairEncHitGtItemEncHit",
            heldout_subject="sub-002",
            map_desc="ModelALOSOFlame1",
        )

        self.assertEqual(
            path,
            Path(
                "derivatives/roi-loso-flame1/maps/group/ses-01/func/"
                "ses-01_task-memory_dir-AP_space-MNI152NLin6Asym_res-2_"
                "contrast-PairEncHitGtItemEncHit_stat-z_heldout-sub002_"
                "desc-ModelALOSOFlame1_statmap.nii.gz"
            ),
        )

    def test_published_loso_flame1_mask_path_matches_requested_layout(self) -> None:
        path = build_loso_flame1_mask_path(
            "derivatives/roi-loso-flame1",
            subject_id="sub-002",
            session_id="ses-01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            roi_label="EncodingPrecuneus",
            contrast_alias="PairEncHitGtItemEncHit",
            mask_desc="ModelALOSOFlame1Sphere6mm",
        )

        self.assertEqual(
            path,
            Path(
                "derivatives/roi-loso-flame1/masks/sub-002/ses-01/func/"
                "sub-002_ses-01_task-memory_dir-AP_space-MNI152NLin6Asym_res-2_"
                "label-EncodingPrecuneus_contrast-PairEncHitGtItemEncHit_"
                "desc-ModelALOSOFlame1Sphere6mm_mask.nii.gz"
            ),
        )

    def test_published_loso_flame1_roistats_path_matches_requested_layout(self) -> None:
        path = build_loso_flame1_roistats_path(
            "derivatives/roi-loso-flame1",
            session_id="ses-01",
            task_id="memory",
            direction="AP",
            table_desc="ModelALOSOFlame1FeatqueryQC",
        )

        self.assertEqual(
            path,
            Path(
                "derivatives/roi-loso-flame1/tables/group/ses-01/func/"
                "ses-01_task-memory_dir-AP_desc-ModelALOSOFlame1FeatqueryQC_roistats.tsv"
            ),
        )


if __name__ == "__main__":
    unittest.main()
