from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.fmripost_aroma.adapter import FmripostAromaAdapter


class FmripostAromaAdapterTests(unittest.TestCase):
    def _pipeline_defaults(self) -> dict[str, object]:
        return {
            "workflow": {"default_target": "fmripost_aroma", "rule_name": "fmripost_aroma"},
            "planner": {
                "outputs": {
                    "runtime_plan_filename": "plan.json",
                    "command_script_filename": "run.sh",
                    "completion_marker_filename": "done.txt",
                    "output_data_dirname": "fmripost_aroma",
                }
            },
        }

    def _bundle(self, *, tool_options: dict[str, object]) -> dict[str, object]:
        return {
            "preprocessing": {
                "preprocessing": {
                    "tool": "fmripost_aroma",
                    "input_derivative": "deepprep-bold",
                    "tool_options": tool_options,
                }
            },
            "dataset": {"dataset": {"input_derivative": "deepprep-bold"}},
        }

    def test_discover_batch_rows_returns_deterministic_selector_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            derivative_root = Path(tmp_dir) / "derivatives"
            func_path = derivative_root / "sub-002" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            (func_path / "sub-002_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")
            (func_path / "sub-002_ses-01_task-memory_run-02_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")

            rows = FmripostAromaAdapter().discover_batch_rows(
                derivative_root=str(derivative_root),
                selectors={"subject_id": "sub-002", "session_id": "ses-01", "task_id": None, "run_id": None},
            )

        self.assertEqual(
            rows,
            [
                {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-01"},
                {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-02"},
            ],
        )

    def test_runtime_metadata_reads_required_pipeline_defaults(self) -> None:
        metadata = FmripostAromaAdapter().runtime_metadata(
            pipeline_defaults={
                "workflow": {"default_target": "fmripost_aroma", "rule_name": "fmripost_aroma"},
                "planner": {
                    "outputs": {
                        "runtime_plan_filename": "plan.json",
                        "command_script_filename": "run.sh",
                        "completion_marker_filename": "done.txt",
                        "output_data_dirname": "fmripost_aroma",
                    }
                },
            },
            output_dir="/tmp/run/outputs",
        )

        self.assertEqual(metadata["workflow_target"], "fmripost_aroma")
        self.assertEqual(metadata["rule_name"], "fmripost_aroma")
        self.assertEqual(metadata["execution_rule_name"], "fmripost_aroma")
        self.assertEqual(metadata["runtime_plan_filename"], "plan.json")
        self.assertEqual(metadata["output_data_dirname"], "fmripost_aroma")

    def test_runtime_metadata_supports_distinct_execution_rule_name(self) -> None:
        metadata = FmripostAromaAdapter().runtime_metadata(
            pipeline_defaults={
                "workflow": {
                    "default_target": "fmripost_aroma",
                    "rule_name": "fmripost_aroma",
                    "execution_rule_name": "fmripost_aroma_unit",
                },
                "planner": {
                    "outputs": {
                        "runtime_plan_filename": "plan.json",
                        "command_script_filename": "run.sh",
                        "completion_marker_filename": "done.txt",
                        "output_data_dirname": "fmripost_aroma",
                    }
                },
            },
            output_dir="/tmp/run/outputs",
        )

        self.assertEqual(metadata["workflow_target"], "fmripost_aroma")
        self.assertEqual(metadata["rule_name"], "fmripost_aroma")
        self.assertEqual(metadata["execution_rule_name"], "fmripost_aroma_unit")

    def test_validate_project_accepts_row_runtime_grouping_and_rejects_unknown_values(self) -> None:
        adapter = FmripostAromaAdapter()

        accepted = adapter.validate_project(
            bundle=self._bundle(tool_options={"runtime_grouping": "row"}),
            pipeline_defaults=self._pipeline_defaults(),
            workspace_root="/tmp/workspace",
        )
        rejected = adapter.validate_project(
            bundle=self._bundle(tool_options={"runtime_grouping": "per_session"}),
            pipeline_defaults=self._pipeline_defaults(),
            workspace_root="/tmp/workspace",
        )

        self.assertEqual(accepted, [])
        self.assertEqual(
            rejected,
            ["preprocessing.tool_options.runtime_grouping must be one of: compatible, row."],
        )

    def test_expected_remote_input_files_reuses_discovered_strict_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            derivative_root = Path(tmp_dir) / "derivatives"
            func_path = derivative_root / "sub-002" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            (
                func_path
                / "sub-002_ses-01_task-memory_acq-fast_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")

            paths = FmripostAromaAdapter().expected_remote_input_files(
                derivative_root=str(derivative_root),
                remote_derivative_root="/remote/studies/demo/derivatives/deepprep-bold",
                row={
                    "subject_id": "sub-002",
                    "session_id": "ses-01",
                    "task_id": "task-memory",
                    "run_id": "run-01",
                },
            )

        self.assertEqual(
            paths,
            [
                "/remote/studies/demo/derivatives/deepprep-bold/sub-002/ses-01/func/"
                "sub-002_ses-01_task-memory_acq-fast_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ],
        )

    def test_scaffold_project_defaults_infers_supported_derivative(self) -> None:
        defaults = FmripostAromaAdapter().scaffold_project_defaults(
            project_name="demo-study",
            study_root="/tmp/study",
            derivative_root="/tmp/derivatives/fmriprep",
            task_id="memory",
        )

        self.assertEqual(defaults["pipeline"], "preprocess-bids")
        self.assertEqual(defaults["input_derivative"], "fmriprep")
        self.assertEqual(defaults["default_batch"], "default")
        self.assertEqual(defaults["task_id"], "memory")
        self.assertEqual(defaults["compute"]["policy"]["default_preset"], "neuro-bids")
        self.assertEqual(defaults["compute"]["policy"]["presets"]["neuro-bids"]["ram_gb"], 32)
        self.assertEqual(defaults["compute"]["slurm"]["mem"], "32G")
        self.assertEqual(defaults["compute"]["slurm"]["time"], "12:00:00")
        self.assertEqual(defaults["compute"]["slurm"]["modules"], ["apptainer/1.3"])


if __name__ == "__main__":
    unittest.main()
