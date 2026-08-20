from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

HPC_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CORE_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-core"
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))

from research_platform.hpc.remote import build_pull_plan, build_stage_plan


class ProvisionSyncTests(unittest.TestCase):
    def test_stage_plan_renders_push_commands_per_scope_with_scope_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            (workspace_root / "project" / "demo").mkdir(parents=True, exist_ok=True)
            (workspace_root / "packages" / "research-core").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.txt").write_text(".git/\n", encoding="utf-8")
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.common.txt").write_text("tests/\n", encoding="utf-8")
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "run-manifest.yaml").write_text("run_id: unit\n", encoding="utf-8")
            (run_root / "submit.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")

            manifest = {
                "run_id": "unit",
                "execution": {"command": ["bash", "artifacts/runs/unit/execute.sh"]},
                "slurm": {"script_path": "artifacts/runs/unit/submit.sbatch"},
                "hpc": {
                    "ssh_host": "example-hpc",
                    "remote_workspace_root": "remote/workspace",
                    "remote_run_root": "remote/workspace/artifacts/runs/unit",
                },
                "provision": {
                    "remote_workspace_root": "remote/workspace",
                    "scopes": [
                        {
                            "name": "common",
                            "entries": [
                                {
                                    "label": "project-overlay",
                                    "kind": "directory",
                                    "source": "project/demo",
                                    "destination": "project/demo",
                                    "exclude_files": ["ops/sync/rsync/exclude.txt", "ops/sync/rsync/exclude.common.txt"],
                                }
                            ],
                        }
                    ],
                },
            }
            (run_root / "execute.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            plan = build_stage_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status={"state": "planned"},
                exclude_file=workspace_root / "ops" / "sync" / "rsync" / "exclude.txt",
            )["report"]

        self.assertEqual(len(plan["push_commands"]), 2)
        scope_push = next(push for push in plan["push_commands"] if push["scope"] == "common")
        rendered = " ".join(scope_push["command"])
        self.assertIn("--exclude-from", rendered)
        self.assertIn("ops/sync/rsync/exclude.common.txt", rendered)

    def test_stage_plan_preserves_run_local_command_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            batch_path = run_root / "inputs" / "filtered.tsv"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "run-manifest.yaml").write_text("run_id: unit\n", encoding="utf-8")
            (run_root / "submit.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            batch_path.write_text("subject_id\nsub-001\n", encoding="utf-8")

            manifest = {
                "run_id": "unit",
                "execution": {
                    "command": [
                        "snakemake",
                        "--config",
                        "batch_manifest=artifacts/runs/unit/inputs/filtered.tsv",
                    ]
                },
                "slurm": {"script_path": "artifacts/runs/unit/submit.sbatch"},
                "hpc": {
                    "ssh_host": "example-hpc",
                    "remote_workspace_root": "remote/workspace",
                    "remote_run_root": "remote/workspace/artifacts/runs/unit",
                },
            }

            plan = build_stage_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status={"state": "planned"},
                exclude_file=None,
            )["report"]
            staged_exists = (run_root / "hpc" / "stage" / "inputs" / "filtered.tsv").exists()
            local_files_exist = all(Path(path).is_file() for path in plan["local_files_written"])

        staged_files = [Path(path).relative_to(run_root / "hpc" / "stage") for path in plan["staged_files"]]
        self.assertIn(Path("inputs/filtered.tsv"), staged_files)
        self.assertTrue(staged_exists)
        self.assertEqual(
            plan["local_files_written"],
            [*plan["staged_files"], str(run_root / "hpc" / "stage-plan.yaml")],
        )
        self.assertTrue(local_files_exist)

    def test_stage_plan_preserves_workspace_relative_destinations_for_file_and_directory_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            (workspace_root / "WORKSPACE.yaml").write_text("workspace:\n  name: demo\n", encoding="utf-8")
            (workspace_root / "packages" / "research-ml").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.txt").write_text(".git/\n", encoding="utf-8")
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "run-manifest.yaml").write_text("run_id: unit\n", encoding="utf-8")

            manifest = {
                "run_id": "unit",
                "execution": {"command": ["python3", "-m", "module"]},
                "hpc": {
                    "ssh_host": "example-hpc",
                    "remote_workspace_root": "remote/workspace",
                    "remote_run_root": "remote/workspace/artifacts/runs/unit",
                },
                "provision": {
                    "remote_workspace_root": "remote/workspace",
                    "scopes": [
                        {
                            "name": "common",
                            "entries": [
                                {
                                    "label": "workspace-config",
                                    "kind": "file",
                                    "source": "WORKSPACE.yaml",
                                    "destination": "WORKSPACE.yaml",
                                    "exclude_files": ["ops/sync/rsync/exclude.txt"],
                                },
                                {
                                    "label": "research-ml",
                                    "kind": "directory",
                                    "source": "packages/research-ml",
                                    "destination": "packages/research-ml",
                                    "exclude_files": ["ops/sync/rsync/exclude.txt"],
                                },
                            ],
                        }
                    ],
                },
            }

            plan = build_stage_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status={"state": "planned"},
                exclude_file=workspace_root / "ops" / "sync" / "rsync" / "exclude.txt",
            )["report"]

        file_push = next(push for push in plan["push_commands"] if push["label"] == "workspace-config")
        directory_push = next(push for push in plan["push_commands"] if push["label"] == "research-ml")
        self.assertEqual(file_push["destination"], "WORKSPACE.yaml")
        self.assertTrue(file_push["command"][-1].endswith("remote/workspace/WORKSPACE.yaml"))
        self.assertEqual(directory_push["destination"], "packages/research-ml")
        self.assertTrue(directory_push["command"][-1].endswith("remote/workspace/packages/research-ml/"))

    def test_stage_plan_renders_push_commands_for_ops_root_and_research_hpc_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            (workspace_root / "ops" / "envs" / "dev").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "scripts").mkdir(parents=True, exist_ok=True)
            (workspace_root / "packages" / "research-hpc").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.txt").write_text(".git/\n", encoding="utf-8")
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.common.txt").write_text("tests/\n", encoding="utf-8")
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "run-manifest.yaml").write_text("run_id: unit\n", encoding="utf-8")

            manifest = {
                "run_id": "unit",
                "execution": {"command": ["python3", "-m", "module"]},
                "hpc": {
                    "ssh_host": "example-hpc",
                    "remote_workspace_root": "remote/workspace",
                    "remote_run_root": "remote/workspace/artifacts/runs/unit",
                },
                "provision": {
                    "remote_workspace_root": "remote/workspace",
                    "scopes": [
                        {
                            "name": "common",
                            "entries": [
                                {
                                    "label": "ops-root",
                                    "kind": "directory",
                                    "source": "ops",
                                    "destination": "ops",
                                    "exclude_files": ["ops/sync/rsync/exclude.txt", "ops/sync/rsync/exclude.common.txt"],
                                },
                                {
                                    "label": "research-hpc",
                                    "kind": "directory",
                                    "source": "packages/research-hpc",
                                    "destination": "packages/research-hpc",
                                    "exclude_files": ["ops/sync/rsync/exclude.txt", "ops/sync/rsync/exclude.common.txt"],
                                },
                            ],
                        }
                    ],
                },
            }

            plan = build_stage_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status={"state": "planned"},
                exclude_file=workspace_root / "ops" / "sync" / "rsync" / "exclude.txt",
            )["report"]

        labels = {push["label"] for push in plan["push_commands"]}
        self.assertIn("ops-root", labels)
        self.assertIn("research-hpc", labels)
        ops_push = next(push for push in plan["push_commands"] if push["label"] == "ops-root")
        hpc_push = next(push for push in plan["push_commands"] if push["label"] == "research-hpc")
        self.assertTrue(ops_push["command"][-1].endswith("remote/workspace/ops/"))
        self.assertTrue(hpc_push["command"][-1].endswith("remote/workspace/packages/research-hpc/"))
        self.assertIn("ops/sync/rsync/exclude.common.txt", " ".join(ops_push["command"]))
        self.assertIn("ops/sync/rsync/exclude.common.txt", " ".join(hpc_push["command"]))

    def test_pull_plan_remains_artifact_only_for_phase_2a(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            (workspace_root / "ops" / "sync" / "rsync").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.txt").write_text(".git/\n", encoding="utf-8")
            run_root.mkdir(parents=True, exist_ok=True)

            manifest = {
                "run_id": "unit",
                "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/workspace/artifacts/runs/unit"},
            }
            plan = build_pull_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status={"state": "planned"},
                exclude_file=workspace_root / "ops" / "sync" / "rsync" / "exclude.txt",
            )["report"]
            local_files_exist = all(Path(path).is_file() for path in plan["local_files_written"])

        self.assertEqual(plan["pull_scope"], "artifacts")
        self.assertEqual(plan["subpath"], "")
        self.assertEqual(plan["remote_source"], "remote/workspace/artifacts/runs/unit")
        self.assertEqual(plan["destination"], str(run_root / "hpc" / "pulled"))
        self.assertTrue(plan["progress"])
        self.assertEqual(plan["pull_command"][0], "rsync")
        self.assertIn("--progress", plan["pull_command"])
        self.assertEqual(plan["local_files_written"], [str(run_root / "hpc" / "pull-plan.yaml")])
        self.assertTrue(local_files_exist)

    def test_pull_plan_with_subpath_defaults_destination_under_pulled_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            run_root.mkdir(parents=True, exist_ok=True)

            manifest = {
                "run_id": "unit",
                "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/workspace/artifacts/runs/unit"},
            }
            plan = build_pull_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status={"state": "planned"},
                exclude_file=None,
                subpath="outputs/fmripost_aroma",
            )["report"]

        expected_destination = run_root / "hpc" / "pulled" / "outputs" / "fmripost_aroma"
        self.assertEqual(plan["subpath"], "outputs/fmripost_aroma")
        self.assertEqual(plan["remote_source"], "remote/workspace/artifacts/runs/unit/outputs/fmripost_aroma")
        self.assertEqual(plan["destination"], str(expected_destination))
        self.assertTrue(plan["pull_command"][-2].endswith(":remote/workspace/artifacts/runs/unit/outputs/fmripost_aroma/"))
        self.assertEqual(plan["pull_command"][-1], f"{expected_destination}/")

    def test_pull_plan_with_subpath_and_explicit_destination_syncs_contents_into_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            run_root.mkdir(parents=True, exist_ok=True)
            destination = Path(tmp_dir) / "exports" / "fmripost_aroma"

            manifest = {
                "run_id": "unit",
                "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/workspace/artifacts/runs/unit"},
            }
            plan = build_pull_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status={"state": "planned"},
                exclude_file=None,
                subpath="outputs/fmripost_aroma",
                destination=destination,
            )["report"]

        self.assertEqual(plan["destination"], str(destination.resolve()))
        self.assertTrue(plan["pull_command"][-2].endswith(":remote/workspace/artifacts/runs/unit/outputs/fmripost_aroma/"))
        self.assertEqual(plan["pull_command"][-1], f"{destination.resolve()}/")

    def test_pull_plan_rejects_invalid_subpaths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            run_root = workspace_root / "artifacts" / "runs" / "unit"
            run_root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": "unit",
                "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/workspace/artifacts/runs/unit"},
            }

            with self.assertRaisesRegex(ValueError, "relative directory|relative to the remote run root|must stay within"):
                build_pull_plan(
                    workspace_root=workspace_root,
                    run_root=run_root,
                    manifest=manifest,
                    status={"state": "planned"},
                    exclude_file=None,
                    subpath="../outputs",
                )


if __name__ == "__main__":
    unittest.main()
