from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.core.runtime_plan import execute_runtime_plan_unit, write_runtime_plan_marker


class _FakePopenProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[str] | None = None,
        wait_results: list[object] | None = None,
    ) -> None:
        self.stdout = io.StringIO("".join(stdout_lines or []))
        self._wait_results = list(wait_results or [0])
        self.returncode: int | None = None
        self.wait_timeouts: list[float | None] = []
        self.terminate_called = False
        self.kill_called = False

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self._wait_results:
            result = self._wait_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            self.returncode = int(result)
            return self.returncode
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9


class RuntimePlanTests(unittest.TestCase):
    def test_execute_runtime_plan_unit_runs_only_selected_unit_steps(self) -> None:
        plan = {
            "units": [
                {
                    "unit_id": "sub-001",
                    "steps": [
                        {"command": ["echo", "first"]},
                        {"command": ["echo", "second"]},
                    ],
                },
                {
                    "unit_id": "sub-002",
                    "steps": [
                        {"command": ["echo", "third"]},
                    ],
                },
            ]
        }

        with (
            mock.patch("research_platform.core.runtime_plan.subprocess.run") as run_mock,
            mock.patch("research_platform.core.runtime_plan.subprocess.Popen") as popen_mock,
        ):
            run_mock.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            exit_code = execute_runtime_plan_unit(plan, "sub-002", cwd="/tmp/workspace")

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_mock.call_count, 1)
        popen_mock.assert_not_called()
        self.assertEqual(run_mock.call_args.args[0], ["echo", "third"])
        self.assertTrue(run_mock.call_args.kwargs["capture_output"])
        self.assertTrue(run_mock.call_args.kwargs["text"])

    def test_execute_runtime_plan_unit_treats_opt_in_success_markers_as_success_even_with_nonzero_exit(self) -> None:
        plan = {
            "units": [
                {
                    "unit_id": "sub-007",
                    "steps": [
                        {
                            "command": ["synthetic-tool", "--run"],
                            "success_markers": ["finished successfully"],
                        },
                    ],
                }
            ]
        }

        with (
            mock.patch("research_platform.core.runtime_plan.subprocess.run") as run_mock,
            mock.patch("research_platform.core.runtime_plan.subprocess.Popen") as popen_mock,
        ):
            process = _FakePopenProcess(
                stdout_lines=["synthetic-tool finished successfully\n", "trailing wrapper noise\n"],
                wait_results=[1],
            )
            popen_mock.return_value = process
            exit_code = execute_runtime_plan_unit(plan, "sub-007", cwd="/tmp/workspace")

        self.assertEqual(exit_code, 0)
        run_mock.assert_not_called()
        self.assertEqual(popen_mock.call_args.args[0], ["synthetic-tool", "--run"])
        self.assertIs(popen_mock.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertIs(popen_mock.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(popen_mock.call_args.kwargs["text"])
        self.assertEqual(popen_mock.call_args.kwargs["bufsize"], 1)
        self.assertEqual(process.wait_timeouts, [10.0])

    def test_execute_runtime_plan_unit_keeps_non_opt_in_nonzero_failures_fatal(self) -> None:
        plan = {
            "units": [
                {
                    "unit_id": "sub-007",
                    "steps": [
                        {"command": ["synthetic-tool", "--run"]},
                    ],
                }
            ]
        }

        with mock.patch("research_platform.core.runtime_plan.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(
                returncode=1,
                stdout="some progress output\n",
                stderr="FATAL: container runtime failed\n",
            )
            exit_code = execute_runtime_plan_unit(plan, "sub-007", cwd="/tmp/workspace")

        self.assertEqual(exit_code, 1)
        self.assertIsNone(run_mock.call_args.kwargs["env"])

    def test_execute_runtime_plan_unit_passes_step_environment_through_to_subprocess(self) -> None:
        plan = {
            "units": [
                {
                    "unit_id": "sub-010",
                    "steps": [
                        {"command": ["synthetic-tool", "--run"], "env": {"FSLOUTPUTTYPE": "NIFTI_GZ"}},
                    ],
                }
            ]
        }

        with mock.patch("research_platform.core.runtime_plan.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            exit_code = execute_runtime_plan_unit(plan, "sub-010", cwd="/tmp/workspace")

        self.assertEqual(exit_code, 0)
        self.assertIn("env", run_mock.call_args.kwargs)
        self.assertEqual(run_mock.call_args.kwargs["env"]["FSLOUTPUTTYPE"], "NIFTI_GZ")

    def test_execute_runtime_plan_unit_terminates_lingering_process_after_success_marker(self) -> None:
        plan = {
            "units": [
                {
                    "unit_id": "sub-007",
                    "steps": [
                        {
                            "command": ["synthetic-tool", "--run"],
                            "success_markers": ["fMRIPost-AROMA finished successfully!"],
                        },
                    ],
                }
            ]
        }

        lingering_timeout = subprocess.TimeoutExpired(cmd="synthetic-tool", timeout=10)
        with mock.patch("research_platform.core.runtime_plan.subprocess.Popen") as popen_mock:
            process = _FakePopenProcess(
                stdout_lines=["fMRIPost-AROMA finished successfully!\n"],
                wait_results=[lingering_timeout, 0],
            )
            popen_mock.return_value = process
            exit_code = execute_runtime_plan_unit(plan, "sub-007", cwd="/tmp/workspace")

        self.assertEqual(exit_code, 0)
        self.assertTrue(process.terminate_called)
        self.assertFalse(process.kill_called)
        self.assertEqual(process.wait_timeouts, [10.0, 5.0])

    def test_write_runtime_plan_marker_writes_deterministic_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            marker_path = Path(tmp_dir) / "outputs" / "runtime-plan-markers" / "fmripost_aroma" / "sub-001.txt"
            write_runtime_plan_marker(marker_path, unit_id="sub-001", step_count=3)
            marker_text = marker_path.read_text(encoding="utf-8")

        self.assertEqual(marker_text, "unit_id=sub-001\nsteps=3\n")

    def test_run_fmripost_aroma_script_executes_exactly_one_unit_by_id(self) -> None:
        module_path = WORKSPACE_ROOT / "pipelines" / "preprocess-bids" / "scripts" / "run_fmripost_aroma.py"
        module_spec = importlib.util.spec_from_file_location("run_fmripost_aroma_test_module", module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        runner_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            batch_path = tmp_root / "batch.tsv"
            batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n", encoding="utf-8")
            plan_path = tmp_root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "steps": [{"command": ["echo", "sub-001"]}, {"command": ["echo", "sub-002"]}],
                        "units": [
                            {
                                "unit_id": "sub-001",
                                "step_count": 1,
                                "marker_path": str(tmp_root / "outputs" / "runtime-plan-markers" / "fmripost_aroma" / "sub-001.txt"),
                                "steps": [{"command": ["echo", "sub-001"]}],
                            },
                            {
                                "unit_id": "sub-002",
                                "step_count": 1,
                                "marker_path": str(tmp_root / "outputs" / "runtime-plan-markers" / "fmripost_aroma" / "sub-002.txt"),
                                "steps": [{"command": ["echo", "sub-002"]}],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = tmp_root / "run-manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  root: /tmp/dataset",
                        "  derivative_root: /tmp/derivatives",
                        "  derivative_name: deepprep-bold",
                        "batch:",
                        f"  path: {batch_path}",
                        "execution:",
                        "  mode: slurm",
                        "  dry_run: false",
                        f"  output_dir: {tmp_root / 'outputs'}",
                        f"  work_dir: {tmp_root / 'work'}",
                        "outputs:",
                        f"  runtime_plan: {plan_path}",
                        f"  command_script: {tmp_root / 'run.sh'}",
                        "tool:",
                        "  options: {}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            marker_path = tmp_root / "sub-002.marker"

            with mock.patch.object(runner_module, "execute_runtime_plan_unit", return_value=0) as execute_mock:
                exit_code = runner_module.main(
                    [
                        "--run-manifest",
                        str(manifest_path),
                        "--plan-path",
                        str(plan_path),
                        "--unit-id",
                        "sub-002",
                        "--marker",
                        str(marker_path),
                    ]
                )

            marker_text = marker_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        execute_mock.assert_called_once()
        self.assertEqual(execute_mock.call_args.args[1], "sub-002")
        self.assertEqual(marker_text, "unit_id=sub-002\nsteps=1\n")

    def test_run_bids_preprocess_script_executes_existing_plan_unit_by_id(self) -> None:
        module_path = WORKSPACE_ROOT / "pipelines" / "preprocess-bids" / "scripts" / "run_bids_preprocess.py"
        module_spec = importlib.util.spec_from_file_location("run_bids_preprocess_test_module", module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        runner_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            plan_path = tmp_root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "steps": [{"command": ["echo", "sub-synthetic02"]}],
                        "units": [
                            {
                                "unit_id": "sub-synthetic02-exampletask",
                                "step_count": 1,
                                "marker_path": str(tmp_root / "outputs" / "runtime-plan-markers" / "deepprep" / "sub-synthetic02-exampletask.txt"),
                                "steps": [{"command": ["echo", "sub-synthetic02"]}],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = tmp_root / "run-manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "execution:",
                        "  mode: slurm",
                        "  dry_run: false",
                        f"  output_dir: {tmp_root / 'outputs'}",
                        f"  work_dir: {tmp_root / 'work'}",
                        "outputs:",
                        f"  runtime_plan: {plan_path}",
                        f"  command_script: {tmp_root / 'run-deepprep.sh'}",
                        "tool:",
                        "  name: deepprep",
                        "  adapter: research_platform.neuro.deepprep.adapter:DeepPrepAdapter",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            marker_path = tmp_root / "sub-synthetic02.marker"

            with mock.patch.object(runner_module, "execute_runtime_plan_unit", return_value=0) as execute_mock:
                exit_code = runner_module.main(
                    [
                        "--run-manifest",
                        str(manifest_path),
                        "--plan-path",
                        str(plan_path),
                        "--unit-id",
                        "sub-synthetic02-exampletask",
                        "--marker",
                        str(marker_path),
                    ]
                )

            marker_text = marker_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        execute_mock.assert_called_once()
        self.assertEqual(execute_mock.call_args.args[1], "sub-synthetic02-exampletask")
        self.assertEqual(marker_text, "unit_id=sub-synthetic02-exampletask\nsteps=1\n")

    def test_run_fmripost_aroma_script_prepares_container_before_selected_unit(self) -> None:
        module_path = WORKSPACE_ROOT / "pipelines" / "preprocess-bids" / "scripts" / "run_fmripost_aroma.py"
        module_spec = importlib.util.spec_from_file_location("run_fmripost_aroma_test_module_container_prep", module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        runner_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            batch_path = tmp_root / "batch.tsv"
            batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-002\tses-01\ttask-rest\trun-01\n", encoding="utf-8")
            prep_marker_path = tmp_root / "outputs" / "runtime-plan-markers" / "fmripost_aroma" / "_container-ready.txt"
            plan_path = tmp_root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "container_prep": {
                            "marker_path": str(prep_marker_path),
                            "command": ["bash", "-lc", "echo prep"],
                        },
                        "steps": [{"command": ["echo", "sub-002"]}],
                        "units": [
                            {
                                "unit_id": "sub-002",
                                "step_count": 1,
                                "marker_path": str(tmp_root / "outputs" / "runtime-plan-markers" / "fmripost_aroma" / "sub-002.txt"),
                                "steps": [{"command": ["echo", "sub-002"]}],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = tmp_root / "run-manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  root: /tmp/dataset",
                        "  derivative_root: /tmp/derivatives",
                        "  derivative_name: deepprep-bold",
                        "batch:",
                        f"  path: {batch_path}",
                        "execution:",
                        "  mode: slurm",
                        "  dry_run: false",
                        f"  output_dir: {tmp_root / 'outputs'}",
                        f"  work_dir: {tmp_root / 'work'}",
                        "outputs:",
                        f"  runtime_plan: {plan_path}",
                        f"  command_script: {tmp_root / 'run.sh'}",
                        "tool:",
                        "  options: {}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            unit_marker_path = tmp_root / "sub-002.marker"

            with (
                mock.patch.object(runner_module, "execute_runtime_plan_unit", return_value=0) as execute_mock,
                mock.patch.object(runner_module.subprocess, "run", return_value=mock.Mock(returncode=0)) as run_mock,
            ):
                exit_code = runner_module.main(
                    [
                        "--run-manifest",
                        str(manifest_path),
                        "--plan-path",
                        str(plan_path),
                        "--unit-id",
                        "sub-002",
                        "--marker",
                        str(unit_marker_path),
                    ]
                )
            prep_marker_text = prep_marker_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with(["bash", "-lc", "echo prep"], cwd=runner_module.WORKSPACE_ROOT, check=False)
        execute_mock.assert_called_once()
        self.assertEqual(prep_marker_text, "steps=1\n")

    def test_run_fmripost_aroma_script_prepare_container_only_writes_marker(self) -> None:
        module_path = WORKSPACE_ROOT / "pipelines" / "preprocess-bids" / "scripts" / "run_fmripost_aroma.py"
        module_spec = importlib.util.spec_from_file_location("run_fmripost_aroma_test_module_prepare_only", module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        runner_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            batch_path = tmp_root / "batch.tsv"
            batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-006\tses-01\ttask-rest\trun-01\n", encoding="utf-8")
            marker_path = tmp_root / "container-ready.txt"
            plan_path = tmp_root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "container_prep": {"marker_path": str(marker_path), "command": ["bash", "-lc", "echo prep"]},
                        "steps": [{"command": ["echo", "sub-006"]}],
                        "units": [{"unit_id": "sub-006", "step_count": 1, "marker_path": str(tmp_root / "sub-006.txt"), "steps": []}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = tmp_root / "run-manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  root: /tmp/dataset",
                        "  derivative_root: /tmp/derivatives",
                        "  derivative_name: deepprep-bold",
                        "batch:",
                        f"  path: {batch_path}",
                        "execution:",
                        "  mode: slurm",
                        "  dry_run: false",
                        f"  output_dir: {tmp_root / 'outputs'}",
                        f"  work_dir: {tmp_root / 'work'}",
                        "outputs:",
                        f"  runtime_plan: {plan_path}",
                        f"  command_script: {tmp_root / 'run.sh'}",
                        "tool:",
                        "  options: {}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(runner_module.subprocess, "run", return_value=mock.Mock(returncode=0)) as run_mock:
                exit_code = runner_module.main(
                    [
                        "--run-manifest",
                        str(manifest_path),
                        "--plan-path",
                        str(plan_path),
                        "--prepare-container-only",
                        "--marker",
                        str(marker_path),
                    ]
                )
            prep_marker_text = marker_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once()
        self.assertEqual(prep_marker_text, "steps=1\n")

    def test_run_bids_analysis_script_prepares_container_before_selected_unit(self) -> None:
        module_path = WORKSPACE_ROOT / "pipelines" / "analysis-bids" / "scripts" / "run_bids_analysis.py"
        module_spec = importlib.util.spec_from_file_location("run_bids_analysis_test_module_container_prep", module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        runner_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            batch_path = tmp_root / "batch.tsv"
            batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-002\tses-01\ttask-rest\trun-01\n", encoding="utf-8")
            prep_marker_path = tmp_root / "outputs" / "runtime-plan-markers" / "feat" / "_container-ready.txt"
            plan_path = tmp_root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "container_prep": {
                            "marker_path": str(prep_marker_path),
                            "command": ["bash", "-lc", "echo prep"],
                        },
                        "steps": [{"command": ["echo", "sub-002"]}],
                        "units": [
                            {
                                "unit_id": "sub-002",
                                "step_count": 1,
                                "marker_path": str(tmp_root / "outputs" / "runtime-plan-markers" / "feat" / "sub-002.txt"),
                                "steps": [{"command": ["echo", "sub-002"]}],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = tmp_root / "run-manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  root: /tmp/dataset",
                        "  derivative_root: /tmp/derivatives",
                        "batch:",
                        f"  path: {batch_path}",
                        "execution:",
                        "  mode: slurm",
                        "  dry_run: false",
                        f"  output_dir: {tmp_root / 'outputs'}",
                        f"  work_dir: {tmp_root / 'work'}",
                        "outputs:",
                        f"  runtime_plan: {plan_path}",
                        f"  command_script: {tmp_root / 'run.sh'}",
                        "tool:",
                        "  options: {}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            unit_marker_path = tmp_root / "sub-002.marker"

            with (
                mock.patch.object(runner_module, "execute_runtime_plan_unit", return_value=0) as execute_mock,
                mock.patch.object(runner_module.subprocess, "run", return_value=mock.Mock(returncode=0)) as run_mock,
            ):
                exit_code = runner_module.main(
                    [
                        "--run-manifest",
                        str(manifest_path),
                        "--plan-path",
                        str(plan_path),
                        "--unit-id",
                        "sub-002",
                        "--marker",
                        str(unit_marker_path),
                    ]
                )
            prep_marker_text = prep_marker_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with(["bash", "-lc", "echo prep"], cwd=runner_module.WORKSPACE_ROOT, check=False)
        execute_mock.assert_called_once()
        self.assertEqual(prep_marker_text, "steps=1\n")

    def test_run_bids_analysis_script_prepare_container_only_writes_marker(self) -> None:
        module_path = WORKSPACE_ROOT / "pipelines" / "analysis-bids" / "scripts" / "run_bids_analysis.py"
        module_spec = importlib.util.spec_from_file_location("run_bids_analysis_test_module_prepare_only", module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        runner_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            batch_path = tmp_root / "batch.tsv"
            batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-006\tses-01\ttask-rest\trun-01\n", encoding="utf-8")
            marker_path = tmp_root / "container-ready.txt"
            plan_path = tmp_root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "container_prep": {"marker_path": str(marker_path), "command": ["bash", "-lc", "echo prep"]},
                        "steps": [{"command": ["echo", "sub-006"]}],
                        "units": [{"unit_id": "sub-006", "step_count": 1, "marker_path": str(tmp_root / "sub-006.txt"), "steps": []}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = tmp_root / "run-manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  root: /tmp/dataset",
                        "  derivative_root: /tmp/derivatives",
                        "batch:",
                        f"  path: {batch_path}",
                        "execution:",
                        "  mode: slurm",
                        "  dry_run: false",
                        f"  output_dir: {tmp_root / 'outputs'}",
                        f"  work_dir: {tmp_root / 'work'}",
                        "outputs:",
                        f"  runtime_plan: {plan_path}",
                        f"  command_script: {tmp_root / 'run.sh'}",
                        "tool:",
                        "  options: {}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(runner_module.subprocess, "run", return_value=mock.Mock(returncode=0)) as run_mock:
                exit_code = runner_module.main(
                    [
                        "--run-manifest",
                        str(manifest_path),
                        "--plan-path",
                        str(plan_path),
                        "--prepare-container-only",
                        "--marker",
                        str(marker_path),
                    ]
                )
            prep_marker_text = marker_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once()
        self.assertEqual(prep_marker_text, "steps=1\n")

    def test_run_fmripost_aroma_script_writes_marker_when_step_reports_opt_in_success_marker(self) -> None:
        module_path = WORKSPACE_ROOT / "pipelines" / "preprocess-bids" / "scripts" / "run_fmripost_aroma.py"
        module_spec = importlib.util.spec_from_file_location("run_fmripost_aroma_test_module_success_markers", module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        runner_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            batch_path = tmp_root / "batch.tsv"
            batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-007\tses-01\ttask-rest\trun-01\n", encoding="utf-8")
            plan_path = tmp_root / "plan.json"
            unit_marker_path = tmp_root / "outputs" / "runtime-plan-markers" / "fmripost_aroma" / "sub-007.txt"
            plan_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "command": [
                                    "apptainer",
                                    "run",
                                    "docker://nipreps/fmripost-aroma:0.0.12",
                                    "/data",
                                    "/out",
                                    "participant",
                                ],
                                "success_markers": ["fMRIPost-AROMA finished successfully!"],
                            }
                        ],
                        "units": [
                            {
                                "unit_id": "sub-007",
                                "step_count": 1,
                                "marker_path": str(unit_marker_path),
                                "steps": [
                                    {
                                        "command": [
                                            "apptainer",
                                            "run",
                                            "docker://nipreps/fmripost-aroma:0.0.12",
                                            "/data",
                                            "/out",
                                            "participant",
                                        ],
                                        "success_markers": ["fMRIPost-AROMA finished successfully!"],
                                    }
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = tmp_root / "run-manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  root: /tmp/dataset",
                        "  derivative_root: /tmp/derivatives",
                        "  derivative_name: deepprep-bold",
                        "batch:",
                        f"  path: {batch_path}",
                        "execution:",
                        "  mode: slurm",
                        "  dry_run: false",
                        f"  output_dir: {tmp_root / 'outputs'}",
                        f"  work_dir: {tmp_root / 'work'}",
                        "outputs:",
                        f"  runtime_plan: {plan_path}",
                        f"  command_script: {tmp_root / 'run.sh'}",
                        "tool:",
                        "  options: {}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch("research_platform.core.runtime_plan.subprocess.Popen") as popen_mock:
                popen_mock.return_value = _FakePopenProcess(
                    stdout_lines=[
                        "fMRIPost-AROMA finished successfully!\n",
                        "wrapper returned exit 1 after tool completion\n",
                    ],
                    wait_results=[1],
                )
                exit_code = runner_module.main(
                    [
                        "--run-manifest",
                        str(manifest_path),
                        "--plan-path",
                        str(plan_path),
                        "--unit-id",
                        "sub-007",
                        "--marker",
                        str(unit_marker_path),
                    ]
                )

            marker_text = unit_marker_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(marker_text, "unit_id=sub-007\nsteps=1\n")

    def test_run_fmripost_aroma_script_rejects_local_plan_only_for_remote_slurm_paths(self) -> None:
        module_path = WORKSPACE_ROOT / "pipelines" / "preprocess-bids" / "scripts" / "run_fmripost_aroma.py"
        module_spec = importlib.util.spec_from_file_location("run_fmripost_aroma_test_module_remote_plan_validation", module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        runner_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            batch_path = tmp_root / "batch.tsv"
            batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n", encoding="utf-8")
            manifest_path = tmp_root / "run-manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  root: /tmp/dataset",
                        "  derivative_root: /tmp/derivatives",
                        "  derivative_name: deepprep-bold",
                        "batch:",
                        f"  path: {batch_path}",
                        "execution:",
                        "  mode: slurm",
                        "  dry_run: true",
                        "  output_dir: /scratch/example/research-platform/artifacts/run-001/outputs",
                        "  work_dir: /scratch/example/research-platform/artifacts/run-001/work",
                        "outputs:",
                        "  runtime_plan: /scratch/example/research-platform/artifacts/run-001/outputs/fmripost-aroma-plan.json",
                        "  command_script: /scratch/example/research-platform/artifacts/run-001/outputs/run-fmripost-aroma.sh",
                        "tool:",
                        "  options: {}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(runner_module, "WORKSPACE_ROOT", tmp_root),
                mock.patch.object(runner_module, "_load_or_build_plan") as build_plan_mock,
            ):
                with self.assertRaises(SystemExit) as exc:
                    runner_module.main(["--run-manifest", str(manifest_path), "--plan-only"])

        self.assertIn("This manifest is planned for remote/SLURM execution", str(exc.exception))
        self.assertIn("rp run plan preprocess bids", str(exc.exception))
        build_plan_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
