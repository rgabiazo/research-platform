from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HPC_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))

from research_platform.hpc._yaml import write_yaml
from research_platform.hpc.bootstrap import build_bootstrap_execution_plan, execute_bootstrap_plan


class BootstrapExecutionTests(unittest.TestCase):
    def test_build_bootstrap_execution_plan_uses_manifest_profile_connection_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            ssh_config = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            run_root.mkdir(parents=True, exist_ok=True)
            write_yaml(
                ssh_config,
                {
                    "profiles": {
                        "interactive-login": {
                            "host": "cluster.example",
                            "user": "alice",
                        }
                    }
                },
            )
            manifest = {
                "run_id": "unit",
                "hpc": {
                    "ssh_host": "legacy-hpc",
                    "connection": {
                        "kind": "ssh-profile",
                        "profile": "interactive-login",
                        "role": "login",
                        "config": "secrets/hpc/ssh-profiles.yaml",
                    },
                },
                "bootstrap": {
                    "enabled": True,
                    "remote_directories": [
                        {"label": "remote-workspace-root", "path": "remote/workspace"},
                        {"label": "remote-run-root", "path": "remote/workspace/artifacts/runs/unit"},
                    ],
                    "hook_scopes": [
                        {"name": "common", "hooks": [{"name": "base-env", "kind": "python-env", "command": "python3 -m venv ~/.venvs/rp"}]},
                        {"name": "tabular-ml", "hooks": [{"name": "wheel-cache", "kind": "cache-prefetch", "command": "python3 -m pip cache dir"}]},
                    ],
                },
            }

            plan = build_bootstrap_execution_plan(
                run_root=run_root,
                manifest=manifest,
                status={"state": "planned"},
                workspace_root=workspace_root,
            )["report"]
            local_files_exist = all(Path(path).is_file() for path in plan["local_files_written"])

        self.assertTrue(plan["enabled"])
        self.assertTrue(plan["executable"])
        self.assertEqual(plan["connection"]["kind"], "ssh-profile")
        self.assertEqual(plan["connection"]["profile"], "interactive-login")
        self.assertEqual(plan["connection"]["target"], "alice@cluster.example")
        self.assertEqual(plan["connection"]["config"], str(ssh_config.resolve()))
        self.assertEqual(len(plan["directory_commands"]), 2)
        self.assertEqual(len(plan["hook_commands"]), 2)
        self.assertEqual(plan["directory_commands"][0]["remote_command"], "mkdir -p remote/workspace")
        self.assertEqual(plan["directory_commands"][0]["command"][:2], ["ssh", "-o"])
        self.assertEqual(plan["hook_commands"][0]["scope"], "common")
        self.assertEqual(plan["hook_commands"][1]["scope"], "tabular-ml")
        self.assertEqual(plan["local_files_written"], [str(run_root / "hpc" / "bootstrap-plan.yaml")])
        self.assertTrue(local_files_exist)

    def test_build_bootstrap_execution_plan_falls_back_to_legacy_ssh_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "artifacts" / "runs" / "unit"
            run_root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": "unit",
                "hpc": {"ssh_host": "example-hpc"},
                "bootstrap": {
                    "enabled": True,
                    "remote_directories": [
                        {"label": "remote-workspace-root", "path": "remote/workspace"},
                    ],
                    "hook_scopes": [],
                },
            }

            plan = build_bootstrap_execution_plan(run_root=run_root, manifest=manifest, status={"state": "planned"})["report"]

        self.assertTrue(plan["enabled"])
        self.assertTrue(plan["executable"])
        self.assertEqual(plan["connection"]["kind"], "ssh-host")
        self.assertEqual(plan["connection"]["target"], "example-hpc")
        self.assertEqual(plan["directory_commands"][0]["command"], ["ssh", "example-hpc", "mkdir -p remote/workspace"])

    def test_build_bootstrap_execution_plan_reports_missing_connection_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "artifacts" / "runs" / "unit"
            run_root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": "unit",
                "bootstrap": {
                    "enabled": True,
                    "remote_directories": [
                        {"label": "remote-workspace-root", "path": "remote/workspace"},
                    ],
                    "hook_scopes": [],
                },
            }

            plan = build_bootstrap_execution_plan(run_root=run_root, manifest=manifest, status={"state": "planned"})["report"]

        self.assertTrue(plan["enabled"])
        self.assertFalse(plan["executable"])
        self.assertEqual(plan["connection"], {})
        self.assertIn("HPC connection is not configured.", plan["connection_error"])
        self.assertEqual(plan["directory_commands"][0]["command"], [])

    def test_execute_bootstrap_plan_runs_steps_in_order(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        plan = {
            "enabled": True,
            "executable": True,
            "directory_commands": [
                {
                    "name": "remote-workspace-root",
                    "scope": "common",
                    "kind": "directory-preparation",
                    "command": ["ssh", "example-hpc", "mkdir -p remote/workspace"],
                }
            ],
            "hook_commands": [
                {
                    "name": "base-env",
                    "scope": "common",
                    "kind": "python-env",
                    "command": ["ssh", "example-hpc", "python3 -m venv ~/.venvs/rp"],
                },
                {
                    "name": "template-cache",
                    "scope": "neuro-bids",
                    "kind": "cache-prefetch",
                    "command": ["ssh", "example-hpc", "python3 -m pip cache dir"],
                },
            ],
        }

        report = execute_bootstrap_plan(plan, runner=runner)

        self.assertTrue(report["ok"])
        self.assertEqual(report["returncode"], 0)
        self.assertEqual(
            calls,
            [
                ["ssh", "example-hpc", "mkdir -p remote/workspace"],
                ["ssh", "example-hpc", "python3 -m venv ~/.venvs/rp"],
                ["ssh", "example-hpc", "python3 -m pip cache dir"],
            ],
        )

    def test_execute_bootstrap_plan_fails_when_plan_is_not_executable(self) -> None:
        report = execute_bootstrap_plan({"enabled": True, "executable": False, "directory_commands": [], "hook_commands": []})

        self.assertFalse(report["ok"])
        self.assertEqual(report["returncode"], 1)


if __name__ == "__main__":
    unittest.main()
