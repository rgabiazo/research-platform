from __future__ import annotations

from contextlib import redirect_stdout
import io
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
from research_platform.core.config import load_yaml
from research_platform.core.hpc_projects import build_project_hpc_context
from research_platform.core.provision import (
    build_project_data_sync_plan,
    build_project_sync_plan,
    build_provision_plan,
    build_workspace_sync_plan,
)


class ProvisionPlanTests(unittest.TestCase):
    def _run_cli(self, args: list[str], artifact_root: Path) -> int:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(WORKSPACE_ROOT),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_HPC_HOST": "example-hpc",
            "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
        }
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False):
            with redirect_stdout(buffer):
                return main(args)

    def test_bids_slurm_manifest_contains_workspace_common_and_neuro_bids_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code = self._run_cli(["run", "slurm", "preprocess", "bids", "--run-id", "unit-provision-bids"], artifact_root)
            manifest = load_yaml(artifact_root / "runs" / "unit-provision-bids" / "run-manifest.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["provision"]["selected_scopes"], ["common", "neuro-bids"])
        scope_entries = {scope["name"]: {entry["source"] for entry in scope["entries"]} for scope in manifest["provision"]["scopes"]}
        self.assertIn("WORKSPACE.yaml", scope_entries["common"])
        self.assertIn("project/project-pilot-bids", scope_entries["common"])
        self.assertIn("ops", scope_entries["common"])
        self.assertIn("packages/research-core", scope_entries["common"])
        self.assertIn("packages/research-hpc", scope_entries["common"])
        self.assertIn("pipelines/preprocess-bids", scope_entries["neuro-bids"])
        self.assertIn("packages/research-neuro", scope_entries["neuro-bids"])
        self.assertIn("datasets/ds-bids-example", scope_entries["neuro-bids"])
        self.assertIn("datasets/ds-bids-example/derivatives/deepprep-bold", scope_entries["neuro-bids"])
        pipeline_entry = next(
            entry
            for scope in manifest["provision"]["scopes"]
            if scope["name"] == "neuro-bids"
            for entry in scope["entries"]
            if entry["label"] == "pipeline-root"
        )
        self.assertEqual(pipeline_entry["sync_scope"], "project")
        raw_dataset_entry = next(
            entry
            for scope in manifest["provision"]["scopes"]
            if scope["name"] == "neuro-bids"
            for entry in scope["entries"]
            if entry["label"] == "raw-dataset-root"
        )
        self.assertEqual(raw_dataset_entry["sync_scope"], "data")
        self.assertIn("ops/sync/rsync/exclude.neuro-bids.txt", raw_dataset_entry["exclude_files"])
        ops_entry = next(
            entry
            for scope in manifest["provision"]["scopes"]
            if scope["name"] == "common"
            for entry in scope["entries"]
            if entry["label"] == "ops-root"
        )
        self.assertEqual(ops_entry["sync_scope"], "project")
        self.assertIn("ops/sync/rsync/exclude.common.txt", ops_entry["exclude_files"])

    def test_tabular_slurm_manifest_contains_workspace_common_and_tabular_ml_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code = self._run_cli(
                ["run", "slurm", "train", "model", "--project", "project-pilot-tabular", "--run-id", "unit-provision-tabular"],
                artifact_root,
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-provision-tabular" / "run-manifest.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["provision"]["selected_scopes"], ["common", "tabular-ml"])
        scope_entries = {scope["name"]: {entry["source"] for entry in scope["entries"]} for scope in manifest["provision"]["scopes"]}
        self.assertIn("WORKSPACE.yaml", scope_entries["common"])
        self.assertIn("ops", scope_entries["common"])
        self.assertIn("packages/research-hpc", scope_entries["common"])
        self.assertIn("packages/research-analysis", scope_entries["tabular-ml"])
        self.assertIn("packages/research-ml", scope_entries["tabular-ml"])
        self.assertIn(
            "datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv",
            scope_entries["tabular-ml"],
        )
        feature_table_entry = next(
            entry
            for scope in manifest["provision"]["scopes"]
            if scope["name"] == "tabular-ml"
            for entry in scope["entries"]
            if entry["label"] == "feature-table"
        )
        self.assertEqual(feature_table_entry["sync_scope"], "data")

    def test_tabular_slurm_manifest_omits_neuro_and_bids_payload_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code = self._run_cli(
                ["run", "slurm", "preprocess", "tabular", "--project", "project-pilot-tabular", "--run-id", "unit-provision-tabular-prep"],
                artifact_root,
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-provision-tabular-prep" / "run-manifest.yaml")

        self.assertEqual(exit_code, 0)
        all_sources = {entry["source"] for scope in manifest["provision"]["scopes"] for entry in scope["entries"]}
        self.assertNotIn("packages/research-neuro", all_sources)
        self.assertNotIn("packages/research-bids", all_sources)
        self.assertNotIn("pipelines/preprocess-bids", all_sources)
        self.assertNotIn("datasets/ds-bids-example", all_sources)

    def test_bids_provision_plan_omits_data_sync_entries_when_remote_roots_are_declared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            local_dataset_root = Path(tmp_dir) / "synthetic-bids"
            local_derivative_root = Path(tmp_dir) / "synthetic-derivatives" / "deepprep-bold"
            local_dataset_root.mkdir(parents=True)
            local_derivative_root.mkdir(parents=True)

            def write_fixture(relative_path: str, content: str) -> None:
                target = workspace_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            write_fixture(
                "WORKSPACE.yaml",
                "\n".join(
                    [
                        "workspace:",
                        "  name: synthetic-workspace",
                        "  version: 0.1.0",
                        "paths:",
                        "  datasets_root: ./datasets",
                        "  artifacts_root: ./artifacts",
                        "repos:",
                        "  packages_root: ./packages",
                        "  pipelines_root: ./pipelines",
                        "  project_root: ./project",
                        "projects:",
                        "  default: project-demo-bids",
                        "",
                    ]
                ),
            )
            write_fixture(
                "project/project-demo-bids/project.yaml",
                "\n".join(
                    [
                        "name: project-demo-bids",
                        "version: 0.1.0",
                        "datasets:",
                        "  - ds-synthetic-bids",
                        "pipelines:",
                        "  - preprocess-bids",
                        "compute_profile: local",
                        "",
                    ]
                ),
            )
            write_fixture(
                "project/project-demo-bids/config/dataset.yaml",
                "\n".join(
                    [
                        "dataset:",
                        "  primary: ds-synthetic-bids",
                        f"  bids_root: {local_dataset_root}",
                        "  input_derivative: deepprep-bold",
                        f"  input_derivative_root: {local_derivative_root}",
                        "  remote_bids_root: /remote/datasets/ds-synthetic-bids",
                        "  remote_input_derivative_root: /remote/derivatives/deepprep-bold",
                        "",
                    ]
                ),
            )
            write_fixture(
                "project/project-demo-bids/config/preprocessing.yaml",
                "\n".join(
                    [
                        "preprocessing:",
                        "  slice: bids",
                        "  pipeline: preprocess-bids",
                        "  tool: fmripost_aroma",
                        "  tool_adapter: research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter",
                        "  input_derivative: deepprep-bold",
                        "  default_batch: synthetic",
                        "  local_profile: local",
                        "  slurm_profile: slurm",
                        "",
                    ]
                ),
            )
            write_fixture(
                "project/project-demo-bids/config/compute.yaml",
                "\n".join(
                    [
                        "compute:",
                        "  default_profile: local",
                        "  local:",
                        "    jobs: 1",
                        "  slurm:",
                        "    cpus: 1",
                        "    mem: 1G",
                        "    time: 00:10:00",
                        "",
                    ]
                ),
            )
            write_fixture(
                "project/project-demo-bids/manifests/batches/synthetic.tsv",
                "subject_id\tsession_id\nsub-synthetic01\tses-01\n",
            )
            write_fixture(
                "pipelines/preprocess-bids/config/defaults.yaml",
                "\n".join(
                    [
                        "workflow:",
                        "  default_target: bids_preprocess",
                        "  rule_name: bids_preprocess",
                        "planner:",
                        "  outputs:",
                        "    runtime_plan_filename: plan.json",
                        "    command_script_filename: run.sh",
                        "    completion_marker_filename: complete.txt",
                        "    output_data_dirname: preprocess_bids",
                        "",
                    ]
                ),
            )
            for relative_path in (
                "ops/sync/rsync",
                "packages/research-core",
                "packages/research-hpc",
                "packages/research-neuro",
            ):
                (workspace_root / relative_path).mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(
                os.environ,
                {
                    "RESEARCH_PLATFORM_ROOT": str(workspace_root),
                    "RP_REMOTE_WORKSPACE_ROOT": "/remote/workspace",
                    "RP_REMOTE_ARTIFACTS_ROOT": "/remote/artifacts",
                },
                clear=False,
            ):
                context = build_project_hpc_context("project-demo-bids")
                provision = build_provision_plan(
                    context=context,
                    manifest={"slice": "bids", "hpc": {"remote_workspace_root": "/remote/workspace"}},
                )

        scope_entries = {
            scope["name"]: {entry["label"] for entry in scope["entries"]}
            for scope in provision["scopes"]
        }
        self.assertEqual(provision["selected_scopes"], ["common", "neuro-bids"])
        self.assertIn("pipeline-root", scope_entries["neuro-bids"])
        self.assertIn("research-neuro", scope_entries["neuro-bids"])
        self.assertNotIn("raw-dataset-root", scope_entries["neuro-bids"])
        self.assertNotIn("input-derivative-root", scope_entries["neuro-bids"])

    def test_project_data_sync_plan_preserves_explicit_nested_bids_root_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            external_root = Path(tmp_dir) / "external"
            dataset_root = external_root / "study"
            derivative_root = dataset_root / "derivatives" / "DeepPrep" / "BOLD"
            dataset_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.txt").write_text(".git/\n", encoding="utf-8")
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.neuro-bids.txt").write_text("derivatives/\n", encoding="utf-8")

            plan = build_project_data_sync_plan(
                context={
                    "workspace_root": workspace_root,
                    "data_roots": [
                        {
                            "label": "raw-dataset-root",
                            "path": dataset_root,
                            "remote_root": "/remote/studies/demo-study",
                        },
                        {
                            "label": "input-derivative-root",
                            "path": derivative_root,
                            "remote_root": "/remote/studies/demo-study/derivatives/DeepPrep/BOLD",
                            "preserve_nested_sync_target": True,
                        },
                    ],
                }
            )

        self.assertEqual(plan["kind"], "data-sync-plan")
        self.assertEqual([entry["label"] for entry in plan["entries"]], ["raw-dataset-root", "input-derivative-root"])
        raw_entry, derivative_entry = plan["entries"]
        self.assertEqual(raw_entry["destination"], "/remote/studies/demo-study")
        self.assertEqual(derivative_entry["destination"], "/remote/studies/demo-study/derivatives/DeepPrep/BOLD")
        self.assertIn("ops/sync/rsync/exclude.neuro-bids.txt", raw_entry["exclude_files"])
        self.assertEqual(derivative_entry["exclude_files"], ["ops/sync/rsync/exclude.txt"])
        self.assertNotIn("mkdir_command", raw_entry)
        self.assertNotIn("rsync_command", raw_entry)

    def test_project_data_sync_plan_preserves_generic_overlapping_roots_when_marked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            external_root = Path(tmp_dir) / "external"
            parent_root = external_root / "inputs"
            child_root = parent_root / "derived" / "subset"
            parent_root.mkdir(parents=True, exist_ok=True)
            child_root.mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "sync" / "rsync" / "exclude.txt").write_text(".git/\n", encoding="utf-8")

            plan = build_project_data_sync_plan(
                context={
                    "workspace_root": workspace_root,
                    "data_roots": [
                        {
                            "label": "source-root",
                            "path": parent_root,
                            "remote_root": "/remote/data/source",
                        },
                        {
                            "label": "subset-root",
                            "path": child_root,
                            "remote_root": "/remote/data/subset",
                            "preserve_nested_sync_target": True,
                        },
                    ],
                }
            )

        self.assertEqual(
            [(entry["label"], entry["destination"]) for entry in plan["entries"]],
            [
                ("source-root", "/remote/data/source"),
                ("subset-root", "/remote/data/subset"),
            ],
        )

    def test_workspace_sync_plan_uses_tracked_default_exclude_file(self) -> None:
        plan = build_workspace_sync_plan(
            workspace_root=WORKSPACE_ROOT,
            remote_workspace_root="remote/workspace",
        )

        self.assertEqual(plan["kind"], "workspace-sync-plan")
        self.assertEqual(plan["destination"], "remote/workspace")
        self.assertEqual(
            plan["exclude_files"],
            [WORKSPACE_ROOT / "ops" / "sync" / "rsync" / "exclude.workspace.txt"],
        )

    def test_project_sync_plan_uses_project_overlay_excludes_for_project_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = workspace_root / "project" / "project-demo-notebook"
            project_root.mkdir(parents=True)
            exclude_root = workspace_root / "ops" / "sync" / "rsync"
            exclude_root.mkdir(parents=True)
            for filename in ("exclude.txt", "exclude.common.txt", "exclude.project-overlay.txt"):
                (exclude_root / filename).write_text(".git/\n", encoding="utf-8")

            plan = build_project_sync_plan(
                context={
                    "workspace_root": workspace_root,
                    "project_root": project_root,
                    "slice": "overlay",
                }
            )

        project_entry = next(entry for entry in plan["entries"] if entry["label"] == "project-overlay")
        self.assertIn("ops/sync/rsync/exclude.project-overlay.txt", project_entry["exclude_files"])
        self.assertNotIn("ops/sync/rsync/exclude.common.txt", project_entry["exclude_files"])


if __name__ == "__main__":
    unittest.main()
