from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.fmripost_aroma.command import (
    DEFAULT_APPTAINER_IMAGE_ROOT,
    DEFAULT_IMAGE_TAG,
    THREAD_ENVIRONMENT,
    build_thread_environment,
    build_batch_runtime_plan,
    resolve_fmripost_aroma_runtime_resources,
    write_command_script,
    write_runtime_plan,
)


class FmripostAromaCommandTests(unittest.TestCase):
    def test_build_plan_uses_derivative_first_filter_and_excludes_raw_only_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_func = root / "rawdata" / "sub-002" / "ses-01" / "func"
            derivative_func = root / "derivatives" / "sub-002" / "ses-01" / "func"
            raw_func.mkdir(parents=True, exist_ok=True)
            derivative_func.mkdir(parents=True, exist_ok=True)
            (raw_func / "sub-002_ses-01_task-exampletask_run-01_bold.nii.gz").write_text("", encoding="utf-8")
            (raw_func / "sub-002_ses-01_task-rest_run-02_bold.nii.gz").write_text("", encoding="utf-8")
            (derivative_func / "sub-002_ses-01_task-exampletask_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")

            output_root = root / "artifacts" / "runs" / "unit" / "outputs"
            work_root = root / "artifacts" / "runs" / "unit" / "work"
            plan_path = output_root / "fmripost-aroma-plan.json"
            command_script = output_root / "run-fmripost-aroma.sh"
            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=root / "derivatives",
                derivative_name="fmriprep",
                batch_rows=[{"subject_id": "sub-002", "session_id": "ses-01"}],
                output_root=output_root,
                work_root=work_root,
                plan_path=plan_path,
                command_script_path=command_script,
                selection={"task_id": "exampletask"},
                backend="docker",
            )
            write_runtime_plan(plan, plan_path)
            write_command_script(plan, command_script)
            persisted = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(len(plan["steps"]), 1)
            self.assertEqual(persisted["image"]["tag"], DEFAULT_IMAGE_TAG)
            self.assertIn("bids_filter_file", plan["steps"][0])
            self.assertTrue(Path(plan["steps"][0]["bids_filter_file"]).exists())
            self.assertEqual(plan["steps"][0]["bids_filter"]["task"], ["exampletask"])
            self.assertNotIn("rest", json.dumps(plan["steps"][0]))
            self.assertIn("--bids-filter-file", plan["steps"][0]["command"])
            docker_command = plan["steps"][0]["command"]
            image_index = docker_command.index(f"nipreps/fmripost-aroma:{DEFAULT_IMAGE_TAG}")
            self.assertEqual(docker_command[image_index + 1 : image_index + 4], ["/data", "/out", "participant"])
            self.assertNotIn("--mem", docker_command)
            self.assertTrue(command_script.exists())
            self.assertEqual(plan["steps"][0]["env"]["OMP_NUM_THREADS"], THREAD_ENVIRONMENT["OMP_NUM_THREADS"])

    def test_build_plan_supports_apptainer_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_func = root / "derivatives" / "sub-003" / "ses-02" / "func"
            derivative_func.mkdir(parents=True, exist_ok=True)
            (derivative_func / "sub-003_ses-02_task-nback_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")
            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=root / "derivatives",
                derivative_name="deepprep-bold",
                batch_rows=[{"subject_id": "sub-003", "session_id": "ses-02"}],
                output_root=root / "artifacts" / "runs" / "unit-hpc" / "outputs",
                work_root=root / "artifacts" / "runs" / "unit-hpc" / "work",
                plan_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "plan.json",
                command_script_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "run.sh",
                selection={"task_id": "nback"},
                backend="apptainer",
            )
            self.assertEqual(plan["backend"], "apptainer")
            self.assertIsNotNone(plan["container_prep"])
            prep_command = plan["container_prep"]["command"]
            self.assertEqual(prep_command[:2], ["bash", "-lc"])
            self.assertIn('export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SLURM_TMPDIR:-$SCRATCH/apptainer-tmp}}"', prep_command[2])
            self.assertIn('apptainer pull "$TMP_IMAGE" "$IMAGE_SOURCE"', prep_command[2])
            self.assertIn('rm -rf "${RUNTIME_IMAGE}.lock.d"', prep_command[2])
            command = plan["steps"][0]["command"]
            self.assertEqual(command[:2], ["bash", "-lc"])
            self.assertNotIn("apptainer pull", command[2])
            self.assertIn(DEFAULT_APPTAINER_IMAGE_ROOT, command[2])
            self.assertIn("nipreps-fmripost-aroma-0.0.12.sif", command[2])
            self.assertIn('exec apptainer run --cleanenv', command[2])
            self.assertIn("/data /out participant", command[2])
            self.assertEqual(plan["steps"][0]["success_markers"], ["fMRIPost-AROMA finished successfully!"])

    def test_build_plan_allows_prepulled_apptainer_image_without_pull_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_func = root / "derivatives" / "sub-003" / "ses-02" / "func"
            derivative_func.mkdir(parents=True, exist_ok=True)
            (derivative_func / "sub-003_ses-02_task-nback_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")
            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=root / "derivatives",
                derivative_name="deepprep-bold",
                batch_rows=[{"subject_id": "sub-003", "session_id": "ses-02"}],
                output_root=root / "artifacts" / "runs" / "unit-hpc" / "outputs",
                work_root=root / "artifacts" / "runs" / "unit-hpc" / "work",
                plan_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "plan.json",
                command_script_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "run.sh",
                selection={"task_id": "nback"},
                backend="apptainer",
                container_pull_mode="never",
                container_image_root="/shared/containers/fmripost_aroma",
                container_image_name="fmripost-aroma-0.0.12.sif",
            )

            command = plan["steps"][0]["command"]
            self.assertEqual(command[:2], ["bash", "-lc"])
            self.assertNotIn("apptainer pull", command[2])
            self.assertIsNone(plan["container_prep"])
            self.assertIn('RUNTIME_IMAGE="docker://nipreps/fmripost-aroma:0.0.12"', command[2])
            self.assertIn('exec apptainer run --cleanenv', command[2])

    def test_build_plan_groups_steps_into_subject_units_with_deterministic_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_root = root / "derivatives"
            for filename in (
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_run-02_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                "sub-007/ses-01/func/sub-007_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
            ):
                path = derivative_root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=derivative_root,
                derivative_name="deepprep-bold",
                batch_rows=[
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-01"},
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-02"},
                    {"subject_id": "sub-007", "session_id": "ses-01"},
                ],
                output_root=root / "artifacts" / "runs" / "unit-fanout" / "outputs",
                work_root=root / "artifacts" / "runs" / "unit-fanout" / "work",
                plan_path=root / "artifacts" / "runs" / "unit-fanout" / "outputs" / "plan.json",
                command_script_path=root / "artifacts" / "runs" / "unit-fanout" / "outputs" / "run.sh",
                backend="apptainer",
            )

        unit_ids = [unit["unit_id"] for unit in plan["units"]]
        self.assertEqual(unit_ids, ["sub-002", "sub-007"])
        self.assertEqual([unit["step_count"] for unit in plan["units"]], [1, 1])
        self.assertEqual(plan["steps"][0]["bids_filter"], {"session": ["01"], "task": ["memory"], "run": ["01", "02"]})
        self.assertTrue(plan["units"][0]["marker_path"].endswith("outputs/runtime-plan-markers/fmripost_aroma/sub-002.txt"))
        self.assertTrue(plan["units"][1]["marker_path"].endswith("outputs/runtime-plan-markers/fmripost_aroma/sub-007.txt"))

    def test_build_plan_merges_safe_same_direction_runs_into_one_step_and_one_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_root = root / "derivatives"
            output_root = root / "artifacts" / "runs" / "lossless-merge" / "outputs"
            command_script = output_root / "run.sh"
            for filename in (
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_dir-AP_run-02_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
            ):
                path = derivative_root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=derivative_root,
                derivative_name="deepprep-bold",
                batch_rows=[
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-01"},
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-02"},
                ],
                output_root=output_root,
                work_root=root / "artifacts" / "runs" / "lossless-merge" / "work",
                plan_path=output_root / "plan.json",
                command_script_path=command_script,
                backend="apptainer",
            )
            write_command_script(plan, command_script)
            command_lines = [line for line in command_script.read_text(encoding="utf-8").splitlines() if " participant " in line]

        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(len(plan["units"]), 1)
        self.assertEqual(plan["tool_options"]["runtime_grouping"], "compatible")
        self.assertEqual(plan["units"][0]["unit_id"], "sub-002")
        self.assertEqual(plan["units"][0]["step_count"], 1)
        self.assertEqual(
            plan["steps"][0]["bids_filter"],
            {"session": ["01"], "task": ["memory"], "run": ["01", "02"], "direction": ["AP"]},
        )
        self.assertEqual(len(command_lines), 1)
        self.assertIn("--task-id", plan["steps"][0]["command"][2])
        self.assertIn("memory", plan["steps"][0]["command"][2])

    def test_build_plan_splits_mixed_directions_to_avoid_overbroad_flat_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_root = root / "derivatives"
            output_root = root / "artifacts" / "runs" / "lossless-split" / "outputs"
            command_script = output_root / "run.sh"
            for filename in (
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_dir-PA_run-02_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
            ):
                path = derivative_root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=derivative_root,
                derivative_name="deepprep-bold",
                batch_rows=[
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-01"},
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-02"},
                ],
                output_root=output_root,
                work_root=root / "artifacts" / "runs" / "lossless-split" / "work",
                plan_path=output_root / "plan.json",
                command_script_path=command_script,
                backend="apptainer",
            )
            write_command_script(plan, command_script)
            command_lines = [line for line in command_script.read_text(encoding="utf-8").splitlines() if " participant " in line]

        filters = sorted(plan["steps"], key=lambda step: step.get("direction") or "")
        self.assertEqual(len(plan["steps"]), 2)
        self.assertEqual(len(plan["units"]), 1)
        self.assertEqual(plan["units"][0]["unit_id"], "sub-002")
        self.assertEqual(plan["units"][0]["step_count"], 2)
        self.assertEqual(filters[0]["bids_filter"], {"session": ["01"], "task": ["memory"], "run": ["01"], "direction": ["AP"]})
        self.assertEqual(filters[1]["bids_filter"], {"session": ["01"], "task": ["memory"], "run": ["02"], "direction": ["PA"]})
        self.assertFalse(
            any(
                step["bids_filter"].get("direction") == ["AP", "PA"] and step["bids_filter"].get("run") == ["01", "02"]
                for step in plan["steps"]
            )
        )
        self.assertEqual(len(command_lines), 2)

    def test_build_plan_runtime_grouping_row_splits_steps_but_preserves_subject_unit_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_root = root / "derivatives"
            output_root = root / "artifacts" / "runs" / "row-grouping" / "outputs"
            command_script = output_root / "run.sh"
            for filename in (
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_dir-AP_run-02_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
            ):
                path = derivative_root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=derivative_root,
                derivative_name="deepprep-bold",
                batch_rows=[
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-01"},
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-02"},
                    {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-01"},
                ],
                output_root=output_root,
                work_root=root / "artifacts" / "runs" / "row-grouping" / "work",
                plan_path=output_root / "plan.json",
                command_script_path=command_script,
                backend="apptainer",
                tool_options={"runtime_grouping": "row"},
            )
            write_command_script(plan, command_script)
            command_lines = [line for line in command_script.read_text(encoding="utf-8").splitlines() if " participant " in line]

        filters = sorted(plan["steps"], key=lambda step: step["bids_filter"]["run"][0])
        self.assertEqual(plan["tool_options"]["runtime_grouping"], "row")
        self.assertEqual(len(plan["steps"]), 2)
        self.assertEqual(len(plan["units"]), 1)
        self.assertEqual(plan["units"][0]["unit_id"], "sub-002")
        self.assertEqual(plan["units"][0]["step_count"], 2)
        self.assertEqual(filters[0]["bids_filter"], {"session": ["01"], "task": ["memory"], "run": ["01"], "direction": ["AP"]})
        self.assertEqual(filters[1]["bids_filter"], {"session": ["01"], "task": ["memory"], "run": ["02"], "direction": ["AP"]})
        self.assertEqual(len(command_lines), 2)

    def test_build_plan_does_not_infer_task_cli_arg_when_task_was_not_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_root = root / "derivatives"
            path = derivative_root / "sub-005" / "ses-01" / "func" / "sub-005_ses-01_task-rest_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=derivative_root,
                derivative_name="deepprep-bold",
                batch_rows=[{"subject_id": "sub-005", "session_id": "ses-01"}],
                output_root=root / "artifacts" / "runs" / "implicit-task" / "outputs",
                work_root=root / "artifacts" / "runs" / "implicit-task" / "work",
                plan_path=root / "artifacts" / "runs" / "implicit-task" / "outputs" / "plan.json",
                command_script_path=root / "artifacts" / "runs" / "implicit-task" / "outputs" / "run.sh",
                backend="apptainer",
            )

        self.assertEqual(plan["steps"][0]["bids_filter"], {"session": ["01"], "task": ["rest"], "run": ["01"], "direction": ["AP"]})
        self.assertNotIn("--task-id", plan["steps"][0]["command"][2])

    def test_build_plan_maps_normalized_resources_to_runtime_flags_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_func = root / "derivatives" / "sub-003" / "ses-02" / "func"
            derivative_func.mkdir(parents=True, exist_ok=True)
            (derivative_func / "sub-003_ses-02_task-nback_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")
            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=root / "derivatives",
                derivative_name="deepprep-bold",
                batch_rows=[{"subject_id": "sub-003", "session_id": "ses-02"}],
                output_root=root / "artifacts" / "runs" / "unit-hpc" / "outputs",
                work_root=root / "artifacts" / "runs" / "unit-hpc" / "work",
                plan_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "plan.json",
                command_script_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "run.sh",
                selection={"task_id": "nback"},
                backend="apptainer",
                resources={"cpus": 4, "threads": 2},
            )

            command = plan["steps"][0]["command"]
            shell = command[2]
            self.assertEqual(plan["resources"], {"nprocs": 4, "omp_nthreads": 2})
            self.assertIn("--nprocs 4", shell)
            self.assertIn("--omp-nthreads 2", shell)
            self.assertNotIn("--mem ", shell)
            self.assertEqual(plan["steps"][0]["env"]["OMP_NUM_THREADS"], "2")
            self.assertEqual(plan["steps"][0]["env"]["MKL_NUM_THREADS"], "2")

    def test_build_plan_maps_memory_budget_when_ram_gb_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_func = root / "derivatives" / "sub-003" / "ses-02" / "func"
            derivative_func.mkdir(parents=True, exist_ok=True)
            (derivative_func / "sub-003_ses-02_task-nback_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")
            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=root / "derivatives",
                derivative_name="deepprep-bold",
                batch_rows=[{"subject_id": "sub-003", "session_id": "ses-02"}],
                output_root=root / "artifacts" / "runs" / "unit-hpc" / "outputs",
                work_root=root / "artifacts" / "runs" / "unit-hpc" / "work",
                plan_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "plan.json",
                command_script_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "run.sh",
                selection={"task_id": "nback"},
                backend="apptainer",
                resources={"cpus": 4, "threads": 2, "ram_gb": 16},
            )

            command = plan["steps"][0]["command"]
            shell = command[2]
            self.assertEqual(plan["resources"], {"nprocs": 2, "omp_nthreads": 2, "mem_mb": 16384})
            self.assertIn("--nprocs 2", shell)
            self.assertIn("--mem 16384", shell)

    def test_build_plan_keeps_full_nprocs_when_memory_budget_is_large_enough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_func = root / "derivatives" / "sub-003" / "ses-02" / "func"
            derivative_func.mkdir(parents=True, exist_ok=True)
            (derivative_func / "sub-003_ses-02_task-nback_dir-AP_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text("", encoding="utf-8")
            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=root / "derivatives",
                derivative_name="deepprep-bold",
                batch_rows=[{"subject_id": "sub-003", "session_id": "ses-02"}],
                output_root=root / "artifacts" / "runs" / "unit-hpc" / "outputs",
                work_root=root / "artifacts" / "runs" / "unit-hpc" / "work",
                plan_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "plan.json",
                command_script_path=root / "artifacts" / "runs" / "unit-hpc" / "outputs" / "run.sh",
                selection={"task_id": "nback"},
                backend="apptainer",
                resources={"cpus": 4, "threads": 1, "ram_gb": 32},
            )

            command = plan["steps"][0]["command"]
            shell = command[2]
            self.assertEqual(plan["resources"], {"nprocs": 4, "omp_nthreads": 1, "mem_mb": 32768})
            self.assertIn("--nprocs 4", shell)
            self.assertIn("--mem 32768", shell)

    def test_runtime_resource_helpers_preserve_default_single_thread_behavior(self) -> None:
        self.assertEqual(resolve_fmripost_aroma_runtime_resources(None), {"nprocs": 1, "omp_nthreads": 1})
        self.assertEqual(build_thread_environment(1), THREAD_ENVIRONMENT)

    def test_build_plan_maps_tool_options_to_runtime_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            derivative_func = root / "derivatives" / "sub-004" / "ses-01" / "func"
            derivative_func.mkdir(parents=True, exist_ok=True)
            (
                derivative_func
                / "sub-004_ses-01_task-rest_run-03_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")
            plan = build_batch_runtime_plan(
                raw_bids_root=root / "rawdata",
                derivative_root=root / "derivatives",
                derivative_name="deepprep-bold",
                batch_rows=[{"subject_id": "sub-004", "session_id": "ses-01", "task_id": "task-rest", "run_id": "run-03"}],
                output_root=root / "artifacts" / "runs" / "unit-tool-options" / "outputs",
                work_root=root / "artifacts" / "runs" / "unit-tool-options" / "work",
                plan_path=root / "artifacts" / "runs" / "unit-tool-options" / "outputs" / "plan.json",
                command_script_path=root / "artifacts" / "runs" / "unit-tool-options" / "outputs" / "run.sh",
                backend="docker",
                tool_options={
                    "denoising_method": "nonaggr",
                    "melodic_dimensionality": 25,
                    "melodic_seed": 11,
                    "dummy_scans": 0,
                    "low_mem": True,
                },
            )

        command = plan["steps"][0]["command"]
        self.assertIn("--denoising-method", command)
        self.assertIn("nonaggr", command)
        self.assertIn("--melodic-dimensionality", command)
        self.assertIn("25", command)
        self.assertIn("--random-seed", command)
        self.assertIn("11", command)
        self.assertIn("--dummy-scans", command)
        self.assertIn("0", command)
        self.assertIn("--low-mem", command)
