from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.core.cli import main
from research_platform.core.config import load_yaml, write_yaml


class AnalysisModelCliTests(unittest.TestCase):
    def _write_workspace(self, workspace_root: Path) -> None:
        write_yaml(
            workspace_root / "WORKSPACE.yaml",
            {
                "paths": {
                    "artifacts_root": "./artifacts",
                    "datasets_root": "./datasets",
                    "ops_root": "./ops",
                },
                "repos": {
                    "project_root": "./project",
                    "pipelines_root": "./pipelines",
                },
                "projects": {"default": "project-default"},
            },
        )

    def _write_analysis_project(self, workspace_root: Path, *, project_name: str = "project-default") -> Path:
        project_root = workspace_root / "project" / project_name
        write_yaml(
            project_root / "project.yaml",
            {
                "name": project_name,
                "version": "0.1.0",
            },
        )
        write_yaml(
            project_root / "config" / "dataset.yaml",
            {
                "dataset": {
                    "primary": project_name,
                    "bids_root": str((workspace_root / "datasets" / project_name).resolve()),
                    "input_derivative": "fmriprep",
                    "input_derivative_root": str((workspace_root / "datasets" / project_name / "derivatives" / "fmriprep").resolve()),
                }
            },
        )
        write_yaml(project_root / "config" / "compute.yaml", {"compute": {"default_profile": "local", "local": {"jobs": 1}}})
        write_yaml(
            project_root / "config" / "analysis.yaml",
            {
                "analysis": {
                    "slice": "bids",
                    "pipeline": "analysis-bids",
                    "local_profile": "local",
                    "slurm_profile": "slurm",
                    "defaults": {
                        "tool": "feat",
                        "stage": "first_level",
                        "model_ref": "task_glm",
                    },
                    "tools": {
                        "feat": {
                            "adapter": "research_platform.neuro.fsl.feat.adapter:FeatAnalysisAdapter",
                            "runtime_profile": "fsl",
                        }
                    },
                    "inputs": {},
                    "stages": {
                        "first_level": {
                            "tool": "feat",
                            "default_batch": "feat_first_level",
                        }
                    },
                }
            },
        )
        return project_root

    def _write_model(self, project_root: Path, name: str = "task_glm") -> None:
        write_yaml(
            project_root / "config" / "analysis" / "models" / f"{name}.yaml",
            {
                "model": {
                    "name": name,
                    "ev_order": ["condition_a", "condition_b", "button_press"],
                    "derivative_on": ["condition_a", "condition_b"],
                    "nonconvolved": ["button_press"],
                    "contrasts": [{"name": "condition_a_gt_baseline", "weights": [1, 0, 0]}],
                }
            },
        )

    def _run_cli(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        extra_env: dict[str, str | None] | None = None,
        input_values: list[str] | None = None,
    ) -> tuple[int, str]:
        env = {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}
        unset_keys: list[str] = []
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                    unset_keys.append(key)
                else:
                    env[key] = value
        buffer = io.StringIO()
        input_iter = iter(input_values or [])
        with mock.patch.dict(os.environ, env, clear=False):
            for key in unset_keys:
                os.environ.pop(key, None)
            with redirect_stdout(buffer):
                with mock.patch("builtins.input", side_effect=lambda prompt="": next(input_iter)):
                    exit_code = main(args)
        return exit_code, buffer.getvalue()

    def _run_cli_system_exit(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        input_values: list[str] | None = None,
    ) -> str:
        buffer = io.StringIO()
        input_iter = iter(input_values or [])
        with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=False):
            with redirect_stdout(buffer):
                with mock.patch("builtins.input", side_effect=lambda prompt="": next(input_iter)):
                    with self.assertRaises(SystemExit) as exc_info:
                        main(args)
        return str(exc_info.exception)

    def test_init_writes_yaml_with_inferred_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            project_root = self._write_analysis_project(workspace_root)

            exit_code, output = self._run_cli(
                [
                    "analysis",
                    "model",
                    "init",
                    "example_glm",
                    "--project",
                    "project-default",
                    "--ev-order",
                    "face_enc_hit",
                    "place_enc_hit",
                    "instr_all",
                    "--derivative-on",
                    "face_enc_hit",
                    "place_enc_hit",
                    "--nonconvolved",
                    "instr_all",
                    "--contrast",
                    "encoding_gt_baseline:1,1,0",
                ],
                workspace_root=workspace_root,
            )
            document = load_yaml(project_root / "config" / "analysis" / "models" / "example_glm.yaml")
            payload = json.loads(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(document["model"]["name"], "example_glm")
        self.assertEqual(document["model"]["ev_order"], ["face_enc_hit", "place_enc_hit", "instr_all"])
        self.assertEqual(document["model"]["nonconvolved"], ["instr_all"])
        self.assertEqual(payload["tool"], "feat")

    def test_missing_project_reports_overlay_path_and_init_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)

            message = self._run_cli_system_exit(
                [
                    "analysis",
                    "model",
                    "show",
                    "example_glm",
                    "--project",
                    "project-missing-neutral",
                ],
                workspace_root=workspace_root,
            )

        self.assertIn("Project overlay 'project-missing-neutral' was not found", message)
        self.assertIn("project/project-missing-neutral", message)
        self.assertIn("rp project init project-missing-neutral", message)
        self.assertNotIn("config/analysis.yaml", message)
        self.assertNotIn("Traceback", message)

    def test_init_interactive_writes_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            project_root = self._write_analysis_project(workspace_root)

            exit_code, _ = self._run_cli(
                [
                    "analysis",
                    "model",
                    "init",
                    "wizard_glm",
                    "--project",
                    "project-default",
                    "--interactive",
                ],
                workspace_root=workspace_root,
                input_values=[
                    "condition_a condition_b button_press",
                    "condition_a condition_b",
                    "button_press",
                    "condition_a_gt_baseline:1,0,0",
                    "",
                ],
            )
            document = load_yaml(project_root / "config" / "analysis" / "models" / "wizard_glm.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(document["model"]["name"], "wizard_glm")
        self.assertEqual(document["model"]["contrasts"][0]["name"], "condition_a_gt_baseline")

    def test_init_requires_complete_non_interactive_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            self._write_analysis_project(workspace_root)

            error = self._run_cli_system_exit(
                [
                    "analysis",
                    "model",
                    "init",
                    "broken_glm",
                    "--project",
                    "project-default",
                    "--ev-order",
                    "condition_a",
                    "condition_b",
                ],
                workspace_root=workspace_root,
            )

        self.assertIn("Incomplete FEAT model input", error)
        self.assertIn("--interactive", error)
        self.assertIn("--template", error)

    def test_copy_show_and_validate_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            project_root = self._write_analysis_project(workspace_root)
            self._write_model(project_root, "task_glm")
            write_yaml(
                project_root / "config" / "analysis" / "models" / "broken_glm.yaml",
                {
                    "model": {
                        "name": "wrong_name",
                        "ev_order": ["condition_a", "condition_b"],
                        "derivative_on": ["missing_ev"],
                        "nonconvolved": [],
                        "contrasts": [{"name": "bad", "weights": [1]}],
                    }
                },
            )

            copy_exit, copy_output = self._run_cli(
                [
                    "analysis",
                    "model",
                    "copy",
                    "task_glm",
                    "task_glm_copy",
                    "--project",
                    "project-default",
                ],
                workspace_root=workspace_root,
            )
            show_summary_exit, show_summary_output = self._run_cli(
                [
                    "analysis",
                    "model",
                    "show",
                    "task_glm_copy",
                    "--project",
                    "project-default",
                    "--format",
                    "summary",
                ],
                workspace_root=workspace_root,
            )
            show_yaml_exit, show_yaml_output = self._run_cli(
                [
                    "analysis",
                    "model",
                    "show",
                    "task_glm_copy",
                    "--project",
                    "project-default",
                    "--format",
                    "yaml",
                ],
                workspace_root=workspace_root,
            )
            validate_single_exit, validate_single_output = self._run_cli(
                [
                    "analysis",
                    "model",
                    "validate",
                    "task_glm_copy",
                    "--project",
                    "project-default",
                ],
                workspace_root=workspace_root,
            )
            validate_all_exit, validate_all_output = self._run_cli(
                [
                    "analysis",
                    "model",
                    "validate",
                    "--all",
                    "--project",
                    "project-default",
                ],
                workspace_root=workspace_root,
            )
            copied_document = load_yaml(project_root / "config" / "analysis" / "models" / "task_glm_copy.yaml")

        self.assertEqual(copy_exit, 0)
        self.assertEqual(json.loads(copy_output)["dest"], "task_glm_copy")
        self.assertEqual(copied_document["model"]["name"], "task_glm_copy")
        self.assertEqual(show_summary_exit, 0)
        self.assertIn("FEAT first-level model: task_glm_copy", show_summary_output)
        self.assertIn("EV order (3)", show_summary_output)
        self.assertEqual(show_yaml_exit, 0)
        self.assertIn("model:", show_yaml_output)
        self.assertIn("task_glm_copy", show_yaml_output)
        self.assertEqual(validate_single_exit, 0)
        self.assertTrue(json.loads(validate_single_output)["valid"])
        self.assertEqual(validate_all_exit, 1)
        validate_all_payload = json.loads(validate_all_output)
        self.assertFalse(validate_all_payload["valid"])
        broken = next(item for item in validate_all_payload["models"] if item["model"] == "broken_glm")
        self.assertFalse(broken["valid"])

    def test_show_can_use_explicit_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            project_root = self._write_analysis_project(workspace_root)
            self._write_model(project_root, "task_glm")

            exit_code, output = self._run_cli(
                [
                    "analysis",
                    "model",
                    "show",
                    "task_glm",
                    "--project",
                    "project-default",
                    "--tool",
                    "feat",
                    "--format",
                    "summary",
                ],
                workspace_root=workspace_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("FEAT first-level model: task_glm", output)

    def test_clear_error_when_tool_lacks_authoring_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_workspace(workspace_root)
            project_root = self._write_analysis_project(workspace_root)
            self._write_model(project_root, "task_glm")

            with mock.patch(
                "research_platform.core.cli.require_bids_analysis_model_authoring_adapter",
                side_effect=ValueError("Analysis tool 'feat' does not expose model authoring support."),
            ):
                error = self._run_cli_system_exit(
                    [
                        "analysis",
                        "model",
                        "show",
                        "task_glm",
                        "--project",
                        "project-default",
                    ],
                    workspace_root=workspace_root,
                )

        self.assertIn("does not expose model authoring support", error)


if __name__ == "__main__":
    unittest.main()
