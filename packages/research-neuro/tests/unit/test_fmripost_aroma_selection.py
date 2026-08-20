from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.fmripost_aroma.selection import (
    DerivativeRun,
    build_flat_bids_filter,
    discover_batch_rows,
    discover_derivative_runs,
    group_flat_filter_compatible_runs,
    group_runtime_plan_runs,
)


class FmripostAromaSelectionTests(unittest.TestCase):
    def test_discover_derivative_runs_is_task_label_agnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            derivative_root = Path(tmp_dir) / "derivatives"
            exampletask_path = derivative_root / "sub-001" / "ses-01" / "func"
            exampletask_path.mkdir(parents=True, exist_ok=True)
            (exampletask_path / "sub-001_ses-01_task-memoryprobe_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")

            runs = discover_derivative_runs(
                derivative_root,
                subject_id="sub-001",
                session_id="ses-01",
                task_id="memoryprobe",
                run_id="run-01",
            )

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].task_id, "memoryprobe")
        self.assertEqual(runs[0].run_id, "01")

    def test_build_flat_filter_promotes_recognized_entities_without_nested_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            derivative_root = Path(tmp_dir) / "derivatives"
            func_path = derivative_root / "sub-002" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            (func_path / "sub-002_ses-01_task-nback_acq-fast_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")
            (func_path / "sub-002_ses-01_task-nback_acq-fast_dir-PA_run-02_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")

            runs = discover_derivative_runs(
                derivative_root,
                subject_id="sub-002",
                session_id="ses-01",
                task_id="nback",
            )
            filter_payload = build_flat_bids_filter(runs, session_id="ses-01", task_id="nback")

        self.assertEqual(filter_payload["session"], ["01"])
        self.assertEqual(filter_payload["task"], ["nback"])
        self.assertEqual(filter_payload["acquisition"], ["fast"])
        self.assertEqual(filter_payload["direction"], ["AP", "PA"])
        self.assertEqual(filter_payload["run"], ["01", "02"])
        self.assertNotIn("bold", filter_payload)
        self.assertNotIn("bold_raw", filter_payload)

    def test_build_flat_filter_prefers_explicit_run_selector_over_discovered_runs(self) -> None:
        filter_payload = build_flat_bids_filter(
            [
                DerivativeRun("002", "01", "nback", "01", None, None, "/tmp/run-01.nii.gz"),
                DerivativeRun("002", "01", "nback", "02", None, None, "/tmp/run-02.nii.gz"),
            ],
            task_id="nback",
            run_id="run-02",
        )

        self.assertEqual(filter_payload["run"], ["02"])

    def test_discover_batch_rows_uses_strict_preproc_filename_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            derivative_root = Path(tmp_dir) / "derivatives"
            func_path = derivative_root / "sub-003" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            (func_path / "sub-003_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")
            (func_path / "sub-003_ses-01_task-rest_run-02_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")

            rows = discover_batch_rows(derivative_root, selectors={"subject_id": "sub-003", "session_id": None, "task_id": None, "run_id": None})

        self.assertEqual(
            rows,
            [{"subject_id": "sub-003", "session_id": "ses-01", "task_id": "task-rest", "run_id": "run-01"}],
        )

    def test_group_flat_filter_compatible_runs_merges_same_direction_runs(self) -> None:
        runs = [
            DerivativeRun("002", "01", "memory", "01", None, "AP", "/tmp/run-01.nii.gz"),
            DerivativeRun("002", "01", "memory", "02", None, "AP", "/tmp/run-02.nii.gz"),
        ]

        groups = group_flat_filter_compatible_runs(runs)

        self.assertEqual(len(groups), 1)
        self.assertEqual([run.run_id for run in groups[0]], ["01", "02"])

    def test_group_flat_filter_compatible_runs_splits_mixed_directions(self) -> None:
        runs = [
            DerivativeRun("002", "01", "memory", "01", None, "AP", "/tmp/run-01.nii.gz"),
            DerivativeRun("002", "01", "memory", "02", None, "PA", "/tmp/run-02.nii.gz"),
        ]

        groups = group_flat_filter_compatible_runs(runs)

        self.assertEqual(len(groups), 2)
        self.assertEqual([[run.direction for run in group] for group in groups], [["AP"], ["PA"]])

    def test_group_runtime_plan_runs_compatible_preserves_current_merge_behavior(self) -> None:
        runs = [
            DerivativeRun("002", "01", "memory", "01", None, "AP", "/tmp/run-01.nii.gz"),
            DerivativeRun("002", "01", "memory", "02", None, "AP", "/tmp/run-02.nii.gz"),
        ]

        groups = group_runtime_plan_runs(runs, runtime_grouping="compatible")

        self.assertEqual(len(groups), 1)
        self.assertEqual([run.run_id for run in groups[0]], ["01", "02"])

    def test_group_runtime_plan_runs_row_returns_singleton_groups_in_deterministic_order(self) -> None:
        runs = [
            DerivativeRun("002", "01", "memory", "02", None, "AP", "/tmp/run-02.nii.gz"),
            DerivativeRun("002", "01", "memory", "01", None, "AP", "/tmp/run-01.nii.gz"),
        ]

        groups = group_runtime_plan_runs(runs, runtime_grouping="row")

        self.assertEqual(len(groups), 2)
        self.assertEqual([[run.run_id for run in group] for group in groups], [["01"], ["02"]])
