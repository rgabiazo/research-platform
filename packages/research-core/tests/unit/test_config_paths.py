from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from research_platform.core import cli as core_cli
from research_platform.core.cli import main
from research_platform.core.config import load_project_bundle, summarize_bundle


class ConfigPathsTests(unittest.TestCase):
    def test_help_identifies_named_analysis_input_paths(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["config", "paths", "--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("named analysis-input paths", " ".join(output.getvalue().split()))

    def _write_workspace(self, root: Path, *, analysis: object) -> Path:
        project_root = root / "project" / "project-private"
        config_root = project_root / "config"
        config_root.mkdir(parents=True)
        (root / "workspace-inputs").mkdir()
        (project_root / "project-inputs").mkdir()

        self._write_json(
            root / "WORKSPACE.yaml",
            {
                "paths": {
                    "artifacts_root": "./artifacts",
                    "datasets_root": "./datasets",
                    "ops_root": "./ops",
                },
                "repos": {
                    "pipelines_root": "./pipelines",
                    "project_root": "./project",
                },
                "projects": {"default": "project-private"},
            },
        )
        self._write_json(project_root / "project.yaml", {"name": "project-private"})
        self._write_json(
            config_root / "dataset.yaml",
            {
                "dataset": {
                    "primary": "ds-private",
                    "canonical_dataset": "ds-private",
                    "canonical_features_root": "features",
                }
            },
        )
        self._write_json(config_root / "compute.yaml", {"compute": {"default_profile": "local"}})
        self._write_json(config_root / "preprocessing.yaml", {"preprocessing": {"slice": "tabular"}})
        self._write_json(config_root / "analysis.yaml", {"analysis": analysis})
        return project_root

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _snapshot(root: Path) -> dict[str, tuple[str, int, str]]:
        return {
            path.relative_to(root).as_posix(): (
                "directory" if path.is_dir() else "file",
                path.stat().st_mode,
                "" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(root.rglob("*"))
        }

    def _run_paths(
        self,
        workspace_root: Path,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, object]]:
        environment = {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}
        if extra_env:
            environment.update(extra_env)
        output = io.StringIO()
        with mock.patch.dict(os.environ, environment, clear=False), redirect_stdout(output):
            exit_code = main(["config", "paths", "--project", "project-private"])
        payload = json.loads(output.getvalue())
        self.assertIsInstance(payload, dict)
        return exit_code, payload

    def test_named_analysis_roots_report_canonical_resolved_fields_without_mutation_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            external_root = root / "outside-workspace" / "private-inputs"
            external_root.mkdir(parents=True)
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            project_root = self._write_workspace(
                workspace_root,
                analysis={
                    "external_input_roots": {
                        "environment_backed": {
                            "label": "private-inputs",
                            "local_root": "${RP_PRIVATE_INPUT_ROOT}",
                            "sync_enabled": False,
                        },
                        "workspace_relative": {
                            "local_root": "workspace-inputs",
                            "remote_root": "remote-private-inputs",
                        },
                        "project_relative": {
                            "label": "project-private-inputs",
                            "local_root": "project-inputs",
                            "sync_enabled": False,
                        },
                        "missing_relative": {
                            "label": "not-materialized-yet",
                            "local_root": "missing-inputs",
                            "sync_enabled": False,
                        },
                    }
                },
            )
            before = self._snapshot(root)

            with mock.patch.dict(
                os.environ,
                {
                    "RESEARCH_PLATFORM_ROOT": str(workspace_root),
                    "RP_PRIVATE_INPUT_ROOT": str(external_root),
                },
                clear=False,
            ):
                bundle = load_project_bundle("project-private", workspace_root)
                expected_existing_paths = summarize_bundle(bundle, root=workspace_root)["resolved_paths"]

            with mock.patch.object(core_cli.subprocess, "run") as subprocess_run:
                with mock.patch.object(core_cli.socket, "create_connection") as create_connection:
                    exit_code, payload = self._run_paths(
                        workspace_root,
                        extra_env={"RP_PRIVATE_INPUT_ROOT": str(external_root)},
                    )

            self.assertEqual(exit_code, 0)
            roots = payload["analysis_external_input_roots"]
            self.assertEqual(
                roots,
                {
                    "environment_backed": {
                        "label": "private-inputs",
                        "local_root": str(external_root.resolve()),
                        "exists": True,
                        "sync_enabled": False,
                    },
                    "missing_relative": {
                        "label": "not-materialized-yet",
                        "local_root": str((project_root / "missing-inputs").resolve()),
                        "exists": False,
                        "sync_enabled": False,
                    },
                    "project_relative": {
                        "label": "project-private-inputs",
                        "local_root": str((project_root / "project-inputs").resolve()),
                        "exists": True,
                        "sync_enabled": False,
                    },
                    "workspace_relative": {
                        "label": "workspace-relative-root",
                        "local_root": str((workspace_root / "workspace-inputs").resolve()),
                        "exists": True,
                        "sync_enabled": True,
                        "remote_root": "remote-private-inputs",
                    },
                },
            )
            self.assertEqual(
                {key: value for key, value in payload.items() if key != "analysis_external_input_roots"},
                expected_existing_paths,
            )
            self.assertNotIn("remote_root", roots["environment_backed"])
            self.assertNotIn("remote_root", roots["project_relative"])
            self.assertEqual(self._snapshot(root), before)
            subprocess_run.assert_not_called()
            create_connection.assert_not_called()

    def test_project_without_named_analysis_roots_retains_existing_paths_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir) / "workspace"
            workspace_root.mkdir()
            self._write_workspace(workspace_root, analysis={})

            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=False):
                bundle = load_project_bundle("project-private", workspace_root)
                expected = summarize_bundle(bundle, root=workspace_root)["resolved_paths"]
                exit_code, payload = self._run_paths(workspace_root)

            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, expected)
            self.assertNotIn("analysis_external_input_roots", payload)

    def test_malformed_or_missing_named_root_declarations_return_controlled_errors(self) -> None:
        cases = (
            (
                ["not-a-mapping"],
                "config/analysis.yaml analysis.external_input_roots must contain a mapping when declared.",
            ),
            (
                {"private_inputs": {"label": "private-inputs", "sync_enabled": False}},
                "config/analysis.yaml analysis.external_input_roots.private_inputs must define local_root.",
            ),
            (
                {"private_inputs": {"local_root": "${RP_UNSET_PRIVATE_INPUT_ROOT}"}},
                "config/analysis.yaml analysis.external_input_roots.private_inputs must define local_root.",
            ),
        )
        for external_roots, expected_error in cases:
            with self.subTest(external_roots=external_roots):
                with tempfile.TemporaryDirectory() as temp_dir:
                    workspace_root = Path(temp_dir) / "workspace"
                    workspace_root.mkdir()
                    self._write_workspace(
                        workspace_root,
                        analysis={"external_input_roots": external_roots},
                    )
                    before = self._snapshot(workspace_root)
                    with mock.patch.dict(
                        os.environ,
                        {
                            "RESEARCH_PLATFORM_ROOT": str(workspace_root),
                            "RP_UNSET_PRIVATE_INPUT_ROOT": "",
                        },
                        clear=False,
                    ):
                        exit_code, payload = self._run_paths(workspace_root)

                    self.assertEqual(exit_code, 1)
                    self.assertEqual(payload, {"valid": False, "errors": [expected_error]})
                    self.assertEqual(self._snapshot(workspace_root), before)


if __name__ == "__main__":
    unittest.main()
