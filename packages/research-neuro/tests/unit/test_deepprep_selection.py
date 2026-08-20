from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CORE_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-core"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.neuro.deepprep.selection import discover_batch_rows, expected_remote_input_files


class DeepPrepSelectionTests(unittest.TestCase):
    def test_discover_batch_rows_reads_raw_bids_subject_task_session_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bids_root = Path(tmp_dir) / "bids"
            func_dir = bids_root / "sub-synthetic02" / "ses-01" / "func"
            func_dir.mkdir(parents=True)
            (bids_root / "participants.tsv").write_text("participant_id\nsub-synthetic02\n", encoding="utf-8")
            (func_dir / "sub-synthetic02_ses-01_task-exampletask_run-01_bold.nii.gz").write_text("", encoding="utf-8")

            rows = discover_batch_rows(
                bids_root,
                selectors={"subject_id": "synthetic02", "task_id": "exampletask"},
            )

        self.assertEqual(
            rows,
            [
                {
                    "subject_id": "sub-synthetic02",
                    "session_id": "ses-01",
                    "task_id": "task-exampletask",
                    "run_id": "run-01",
                }
            ],
        )

    def test_expected_remote_input_files_includes_raw_subject_bold_t1w_and_fmap_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bids_root = Path(tmp_dir) / "bids"
            func_dir = bids_root / "sub-synthetic02" / "ses-01" / "func"
            anat_dir = bids_root / "sub-synthetic02" / "ses-01" / "anat"
            fmap_dir = bids_root / "sub-synthetic02" / "ses-01" / "fmap"
            func_dir.mkdir(parents=True)
            anat_dir.mkdir(parents=True)
            fmap_dir.mkdir(parents=True)
            (bids_root / "dataset_description.json").write_text("{}", encoding="utf-8")
            (bids_root / "participants.tsv").write_text("participant_id\nsub-synthetic02\n", encoding="utf-8")
            bold_path = func_dir / "sub-synthetic02_ses-01_task-exampletask_run-01_bold.nii.gz"
            bold_path.write_text("", encoding="utf-8")
            (func_dir / "sub-synthetic02_ses-01_task-exampletask_run-01_bold.json").write_text("{}", encoding="utf-8")
            t1w_path = anat_dir / "sub-synthetic02_ses-01_T1w.nii.gz"
            t1w_path.write_text("", encoding="utf-8")
            (anat_dir / "sub-synthetic02_ses-01_T1w.json").write_text("{}", encoding="utf-8")
            fmap_path = fmap_dir / "sub-synthetic02_ses-01_dir-AP_epi.nii.gz"
            fmap_path.write_text("", encoding="utf-8")
            (fmap_dir / "sub-synthetic02_ses-01_dir-AP_epi.json").write_text("{}", encoding="utf-8")

            expected = expected_remote_input_files(
                bids_root,
                remote_bids_root="/scratch/project/example-bids",
                row={
                    "subject_id": "sub-synthetic02",
                    "session_id": "ses-01",
                    "task_id": "task-exampletask",
                    "run_id": "run-01",
                },
            )

        self.assertIn("/scratch/project/example-bids/dataset_description.json", expected)
        self.assertIn("/scratch/project/example-bids/participants.tsv", expected)
        self.assertIn("/scratch/project/example-bids/sub-synthetic02", expected)
        self.assertIn("/scratch/project/example-bids/sub-synthetic02/ses-01/func/sub-synthetic02_ses-01_task-exampletask_run-01_bold.nii.gz", expected)
        self.assertIn("/scratch/project/example-bids/sub-synthetic02/ses-01/func/sub-synthetic02_ses-01_task-exampletask_run-01_bold.json", expected)
        self.assertIn("/scratch/project/example-bids/sub-synthetic02/ses-01/anat/sub-synthetic02_ses-01_T1w.nii.gz", expected)
        self.assertIn("/scratch/project/example-bids/sub-synthetic02/ses-01/anat/sub-synthetic02_ses-01_T1w.json", expected)
        self.assertIn("/scratch/project/example-bids/sub-synthetic02/ses-01/fmap/sub-synthetic02_ses-01_dir-AP_epi.nii.gz", expected)
        self.assertIn("/scratch/project/example-bids/sub-synthetic02/ses-01/fmap/sub-synthetic02_ses-01_dir-AP_epi.json", expected)


if __name__ == "__main__":
    unittest.main()
