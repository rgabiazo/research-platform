from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HPC_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CORE_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-core"
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))

from research_platform.hpc.remote import build_pull_plan, build_stage_plan, build_submit_plan, resolve_remote_run_root
from research_platform.hpc.manifest import write_run_manifest, write_status
from research_platform.hpc.slurm import (
    build_slurm_command_script,
    build_slurm_jobspec,
    build_slurm_setup_commands,
    normalize_jobspec,
    normalize_slurm_batch_script,
    render_slurm_script,
    write_slurm_script,
)
from research_platform.hpc.sync import build_rsync_pull_command, build_rsync_push_command
from research_platform.core.config import write_yaml


class HpcSliceHelperTests(unittest.TestCase):
    def test_hpc_source_is_compatible_with_supported_python_grammar(self) -> None:
        for source_path in sorted((HPC_PACKAGE_ROOT / "src").rglob("*.py")):
            source = source_path.read_text(encoding="utf-8")
            with self.subTest(source_path=source_path.relative_to(HPC_PACKAGE_ROOT)):
                compile(source, str(source_path), "exec")
                self.assertNotRegex(source, r'''f["'][^\n]*\{[^}\n]*\\[^}\n]*\}''')

    def test_normalize_jobspec_and_render_script(self) -> None:
        jobspec = normalize_jobspec({"job_name": "pilot", "cpus": 2, "mem": "8G", "time": "03:00:00", "command": "echo hello"})
        self.assertEqual(jobspec["cpus"], 2)
        self.assertEqual(jobspec["mem"], "8G")

        rendered = render_slurm_script(
            template_path=WORKSPACE_ROOT / "ops" / "slurm" / "job_templates" / "sbatch.job.sh",
            jobspec=jobspec,
        )
        self.assertIn("#SBATCH --job-name=pilot", rendered)
        self.assertIn("#SBATCH --mem=8G", rendered)
        self.assertIn("echo hello", rendered)

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = write_slurm_script(Path(tmp_dir) / "submit.sbatch", rendered)
            self.assertTrue(script_path.exists())

    def test_write_slurm_script_normalizes_shebang_to_first_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = write_slurm_script(
                Path(tmp_dir) / "submit.sbatch",
                "\ufeff\r\n#!/usr/bin/env bash\r\n#SBATCH --job-name=pilot\r\n\r\necho hello\r\n",
            )
            script_bytes = script_path.read_bytes()
            script_text = script_path.read_text(encoding="utf-8")

        self.assertTrue(script_bytes.startswith(b"#!"))
        self.assertEqual(script_text.splitlines()[0], "#!/usr/bin/env bash")
        self.assertNotEqual(script_text[:1], "\n")
        self.assertNotIn("\r", script_text)

    def test_normalize_slurm_batch_script_preserves_existing_shebang(self) -> None:
        script_text = normalize_slurm_batch_script("\n\n#!/bin/bash\n\necho hello\n")

        self.assertEqual(script_text, "#!/bin/bash\n\necho hello\n")

    def test_normalize_slurm_batch_script_adds_default_shebang_when_missing(self) -> None:
        script_text = normalize_slurm_batch_script("\n#SBATCH --job-name=pilot\n\necho hello\n")

        self.assertEqual(
            script_text,
            "#!/usr/bin/env bash\n#SBATCH --job-name=pilot\n\necho hello\n",
        )
        self.assertEqual(script_text.count("#!/usr/bin/env bash"), 1)

    def test_remote_root_prefers_remote_artifacts_root(self) -> None:
        remote_root = resolve_remote_run_root(
            run_id="unit-slurm",
            remote_workspace_root="remote/workspace",
            remote_artifacts_root="remote/workspace/artifacts",
        )
        self.assertEqual(remote_root, "remote/workspace/artifacts/runs/unit-slurm")

    def test_sync_command_builders_include_expected_endpoints(self) -> None:
        push_command = build_rsync_push_command(source="artifacts/runs/unit", ssh_host="example-hpc", destination="remote/run")
        pull_command = build_rsync_pull_command(ssh_host="example-hpc", source="remote/run", destination="artifacts/pulled")
        self.assertEqual(push_command[-1], "example-hpc:remote/run/")
        self.assertEqual(pull_command[2], "example-hpc:remote/run/")

    def test_sync_pull_command_builder_supports_progress(self) -> None:
        pull_command = build_rsync_pull_command(
            ssh_host="example-hpc",
            source="remote/run",
            destination="artifacts/pulled",
            progress=True,
        )

        self.assertEqual(pull_command[:3], ["rsync", "-az", "--progress"])

    def test_sync_command_builders_support_file_transfers(self) -> None:
        push_command = build_rsync_push_command(
            source="WORKSPACE.yaml",
            ssh_host="example-hpc",
            destination="remote/workspace/WORKSPACE.yaml",
            source_is_directory=False,
        )
        self.assertEqual(push_command[-2], "WORKSPACE.yaml")
        self.assertEqual(push_command[-1], "example-hpc:remote/workspace/WORKSPACE.yaml")

    def test_build_slurm_jobspec_uses_normalized_resources(self) -> None:
        jobspec = build_slurm_jobspec(
            resources={"cpus": 3, "ram_gb": 12},
            job_name="planned-job",
            time="00:45:00",
            log_out="logs/slurm.out",
            log_err="logs/slurm.err",
            command="echo hello",
        )
        self.assertEqual(jobspec["cpus"], 3)
        self.assertEqual(jobspec["mem"], "12G")
        self.assertEqual(jobspec["time"], "00:45:00")

    def test_render_slurm_script_can_omit_memory_directive_for_site_policy(self) -> None:
        jobspec = build_slurm_jobspec(
            resources={"cpus": 4, "ram_gb": 32},
            job_name="cluster-b-job",
            time="24:00:00",
            log_out="logs/slurm.out",
            log_err="logs/slurm.err",
            command="echo hello",
            slurm_site={"omit_mem_directive": True},
        )

        rendered = render_slurm_script(
            template_path=WORKSPACE_ROOT / "ops" / "slurm" / "job_templates" / "sbatch.job.sh",
            jobspec=jobspec,
        )

        self.assertEqual(jobspec["mem"], "32G")
        self.assertEqual(jobspec["omit_mem_directive"], True)
        self.assertNotIn("#SBATCH --mem=", rendered)
        self.assertIn("#SBATCH --cpus-per-task=4", rendered)
        self.assertIn("#SBATCH --time=24:00:00", rendered)

    def test_render_slurm_script_includes_optional_site_directives_only_when_configured(self) -> None:
        base = build_slurm_jobspec(
            resources={"cpus": 1, "ram_gb": 4},
            job_name="base-job",
            time="00:30:00",
            log_out="logs/slurm.out",
            log_err="logs/slurm.err",
            command="echo base",
        )
        configured = build_slurm_jobspec(
            resources={"cpus": 1, "ram_gb": 4},
            job_name="configured-job",
            time="00:30:00",
            log_out="logs/slurm.out",
            log_err="logs/slurm.err",
            command="echo configured",
            slurm_site={
                "account": "example-account",
                "partition": "compute",
                "qos": "normal",
                "nodes": "2",
                "ntasks": "1",
                "mem_per_cpu": "8G",
            },
        )

        base_rendered = render_slurm_script(
            template_path=WORKSPACE_ROOT / "ops" / "slurm" / "job_templates" / "sbatch.job.sh",
            jobspec=base,
        )
        configured_rendered = render_slurm_script(
            template_path=WORKSPACE_ROOT / "ops" / "slurm" / "job_templates" / "sbatch.job.sh",
            jobspec=configured,
        )

        self.assertNotIn("#SBATCH --account", base_rendered)
        self.assertIn("#SBATCH --account=example-account", configured_rendered)
        self.assertIn("#SBATCH --partition=compute", configured_rendered)
        self.assertIn("#SBATCH --qos=normal", configured_rendered)
        self.assertIn("#SBATCH --nodes=2", configured_rendered)
        self.assertIn("#SBATCH --ntasks=1", configured_rendered)
        self.assertIn("#SBATCH --mem-per-cpu=8G", configured_rendered)

    def test_build_slurm_setup_commands_orders_modules_before_pre_activate_bootstrap_and_activation(self) -> None:
        commands = build_slurm_setup_commands(
            remote_workspace_root="remote/workspace",
            modules=["toolchain/1.0", "python/3.11"],
            pre_activate_commands=["module load dataframe/1.0", "export RP_CLUSTER=1"],
            bootstrap_command="bash ops/envs/dev/bootstrap.sh",
            activate_command="source .venv/bin/activate",
        )

        self.assertEqual(
            commands,
            [
                "cd remote/workspace",
                "module load toolchain/1.0 python/3.11",
                "module load dataframe/1.0",
                "export RP_CLUSTER=1",
                "bash ops/envs/dev/bootstrap.sh",
                "source .venv/bin/activate",
            ],
        )

    def test_build_slurm_setup_commands_renders_environment_exports_with_remote_shell_expansion(self) -> None:
        commands = build_slurm_setup_commands(
            remote_workspace_root="remote/workspace",
            modules=["apptainer/1.3"],
            environment={
                "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
            },
            pre_activate_commands=["module load arrow/23.0.1"],
            bootstrap_command="bash ops/envs/dev/bootstrap.sh",
            activate_command="source .venv/bin/activate",
        )

        self.assertEqual(
            commands,
            [
                "cd remote/workspace",
                "module load apptainer/1.3",
                'export APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache"',
                'export APPTAINER_TMPDIR="$SCRATCH/apptainer-tmp"',
                "module load arrow/23.0.1",
                "bash ops/envs/dev/bootstrap.sh",
                "source .venv/bin/activate",
            ],
        )

    def test_build_slurm_setup_commands_renders_prepare_directories_after_pre_activate_commands(self) -> None:
        commands = build_slurm_setup_commands(
            remote_workspace_root="remote/workspace",
            modules=["apptainer/1.3"],
            environment={
                "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
            },
            pre_activate_commands=["module load arrow/23.0.1"],
            prepare_directories=[
                "$APPTAINER_CACHEDIR",
                "$APPTAINER_TMPDIR",
            ],
            bootstrap_command="bash ops/envs/dev/bootstrap.sh",
            activate_command="source .venv/bin/activate",
        )

        self.assertEqual(
            commands,
            [
                "cd remote/workspace",
                "module load apptainer/1.3",
                'export APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache"',
                'export APPTAINER_TMPDIR="$SCRATCH/apptainer-tmp"',
                "module load arrow/23.0.1",
                'mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"',
                "bash ops/envs/dev/bootstrap.sh",
                "source .venv/bin/activate",
            ],
        )

    def test_build_slurm_command_script_appends_workflow_after_setup_commands(self) -> None:
        script = build_slurm_command_script(
            setup_commands=[
                "cd remote/workspace",
                "module load toolchain/1.0 python/3.11",
                "bash ops/envs/dev/bootstrap.sh",
                "source .venv/bin/activate",
            ],
            required_executables=["snakemake"],
            workflow_command="snakemake --snakefile pipelines/preprocess-bids/workflow/Snakefile",
        )

        self.assertIn("command -v snakemake >/dev/null 2>&1", script)
        self.assertIn("repo bootstrap dependencies", script)
        self.assertLess(script.index("source .venv/bin/activate"), script.index("command -v snakemake"))
        self.assertLess(
            script.index("command -v snakemake"),
            script.index("snakemake --snakefile pipelines/preprocess-bids/workflow/Snakefile"),
        )

    def test_build_submit_plan_prefers_manifest_submit_command(self) -> None:
        plan = build_submit_plan(
            manifest={
                "run_id": "unit-submit",
                "hpc": {
                    "ssh_host": "example-hpc",
                    "remote_run_root": "remote/run",
                    "submit_command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"],
                },
            },
            status={},
        )

        self.assertEqual(plan["submit_command"], ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"])

    def test_build_submit_plan_falls_back_to_remote_run_root(self) -> None:
        plan = build_submit_plan(
            manifest={"run_id": "unit-submit", "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/run"}},
            status={},
        )

        self.assertEqual(plan["submit_command"], ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"])

    def test_build_stage_plan_prefers_manifest_ssh_host_over_env_profile_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            run_root = workspace_root / "artifacts" / "runs" / "unit-stage-host"
            stage_source = run_root / "submit.sbatch"
            workspace_file = workspace_root / "WORKSPACE.yaml"
            stage_source.parent.mkdir(parents=True, exist_ok=True)
            workspace_file.write_text("projects:\n  default: demo\n", encoding="utf-8")
            stage_source.write_text("#!/bin/bash\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "RESEARCH_HPC_PROFILE": "interactive-login",
                    "RESEARCH_HPC_SSH_CONFIG": str(workspace_root / "missing-ssh-profiles.yaml"),
                },
                clear=False,
            ):
                report = build_stage_plan(
                    workspace_root=workspace_root,
                    run_root=run_root,
                    manifest={
                        "run_id": "unit-stage-host",
                        "execution": {"mode": "slurm", "command": [], "output_dir": "artifacts/runs/unit-stage-host/outputs"},
                        "slurm": {"script_path": "artifacts/runs/unit-stage-host/submit.sbatch"},
                        "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/run", "remote_workspace_root": "remote/workspace"},
                        "provision": {"scopes": []},
                    },
                    status={"run_id": "unit-stage-host", "state": "planned"},
                    exclude_file=None,
                )["report"]

        self.assertEqual(report["connection"]["kind"], "ssh-host")
        self.assertEqual(report["connection"]["target"], "example-hpc")
        self.assertEqual(report["prepare_commands"][0][:2], ["ssh", "example-hpc"])

    def test_build_stage_plan_prepares_remote_parent_directories_for_scope_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            run_root = workspace_root / "artifacts" / "runs" / "unit-stage"
            stage_source = run_root / "submit.sbatch"
            scope_source = workspace_root / "packages" / "research-core"
            workspace_file = workspace_root / "WORKSPACE.yaml"
            stage_source.parent.mkdir(parents=True, exist_ok=True)
            scope_source.mkdir(parents=True, exist_ok=True)
            workspace_file.write_text("projects:\n  default: demo\n", encoding="utf-8")
            stage_source.write_text("#!/bin/bash\n", encoding="utf-8")
            write_run_manifest(
                run_root,
                {
                    "run_id": "unit-stage",
                    "execution": {"mode": "slurm", "command": [], "output_dir": "artifacts/runs/unit-stage/outputs"},
                    "slurm": {"script_path": "artifacts/runs/unit-stage/submit.sbatch"},
                    "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/run", "remote_workspace_root": "remote/workspace"},
                    "provision": {
                        "scopes": [
                            {
                                "name": "common",
                                "entries": [
                                    {
                                        "label": "workspace-config",
                                        "kind": "file",
                                        "source": "WORKSPACE.yaml",
                                        "destination": "WORKSPACE.yaml",
                                        "exclude_files": [],
                                    },
                                    {
                                        "label": "research-core",
                                        "kind": "directory",
                                        "source": "packages/research-core",
                                        "destination": "packages/research-core",
                                        "exclude_files": [],
                                    },
                                ],
                            }
                        ]
                    },
                },
            )
            write_status(run_root, {"run_id": "unit-stage", "state": "planned"})

            report = build_stage_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={
                    "run_id": "unit-stage",
                    "execution": {"mode": "slurm", "command": [], "output_dir": "artifacts/runs/unit-stage/outputs"},
                    "slurm": {"script_path": "artifacts/runs/unit-stage/submit.sbatch"},
                    "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/run", "remote_workspace_root": "remote/workspace"},
                    "provision": {
                        "scopes": [
                            {
                                "name": "common",
                                "entries": [
                                    {
                                        "label": "workspace-config",
                                        "kind": "file",
                                        "source": "WORKSPACE.yaml",
                                        "destination": "WORKSPACE.yaml",
                                        "exclude_files": [],
                                    },
                                    {
                                        "label": "research-core",
                                        "kind": "directory",
                                        "source": "packages/research-core",
                                        "destination": "packages/research-core",
                                        "exclude_files": [],
                                    },
                                ],
                            }
                        ]
                    },
                },
                status={"run_id": "unit-stage", "state": "planned"},
                exclude_file=None,
            )["report"]

        self.assertEqual(report["prepare_commands"][0][:2], ["ssh", "example-hpc"])
        self.assertIn("remote/run", report["prepare_commands"][0][-1])
        self.assertIn("remote/workspace", report["prepare_commands"][0][-1])
        self.assertIn("remote/workspace/packages/research-core", report["prepare_commands"][0][-1])

    def test_build_stage_plan_prepares_remote_runtime_directories_from_shared_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            run_root = workspace_root / "artifacts" / "runs" / "unit-stage-runtime-dirs"
            stage_source = run_root / "submit.sbatch"
            stage_source.parent.mkdir(parents=True, exist_ok=True)
            (workspace_root / "WORKSPACE.yaml").write_text("projects:\n  default: demo\n", encoding="utf-8")
            stage_source.write_text("#!/bin/bash\n", encoding="utf-8")

            report = build_stage_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest={
                    "run_id": "unit-stage-runtime-dirs",
                    "execution": {
                        "mode": "slurm",
                        "command": [],
                        "work_dir": "remote/run/work",
                        "output_dir": "remote/run/outputs",
                        "log_dir": "remote/run/logs",
                    },
                    "slurm": {
                        "script_path": "artifacts/runs/unit-stage-runtime-dirs/submit.sbatch",
                        "jobspec": {
                            "log_out": "remote/run/logs/slurm.out",
                            "log_err": "remote/run/logs/slurm.err",
                        },
                    },
                    "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/run", "remote_workspace_root": "remote/workspace"},
                    "provision": {"scopes": []},
                },
                status={"run_id": "unit-stage-runtime-dirs", "state": "planned"},
                exclude_file=None,
            )["report"]

        prepare_command = report["prepare_commands"][0]
        self.assertEqual(prepare_command[:2], ["ssh", "example-hpc"])
        self.assertIn("remote/run", prepare_command[-1])
        self.assertIn("remote/run/logs", prepare_command[-1])
        self.assertIn("remote/run/work", prepare_command[-1])
        self.assertIn("remote/run/outputs", prepare_command[-1])

    def test_build_stage_plan_uses_profile_aware_connection_when_manifest_declares_profile(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            subdirectory = workspace_root / "project" / "demo"
            run_root = workspace_root / "artifacts" / "runs" / "unit-stage-profile"
            stage_source = run_root / "submit.sbatch"
            scope_source = workspace_root / "packages" / "research-core"
            ssh_config = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            stage_source.parent.mkdir(parents=True, exist_ok=True)
            scope_source.mkdir(parents=True, exist_ok=True)
            subdirectory.mkdir(parents=True, exist_ok=True)
            (workspace_root / "WORKSPACE.yaml").write_text("projects:\n  default: demo\n", encoding="utf-8")
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
            stage_source.write_text("#!/bin/bash\n", encoding="utf-8")
            os.chdir(subdirectory)
            try:
                report = build_stage_plan(
                    workspace_root=workspace_root,
                    run_root=run_root,
                    manifest={
                        "run_id": "unit-stage-profile",
                        "execution": {"mode": "slurm", "command": [], "output_dir": "artifacts/runs/unit-stage-profile/outputs"},
                        "slurm": {"script_path": "artifacts/runs/unit-stage-profile/submit.sbatch"},
                        "hpc": {
                            "remote_run_root": "remote/run",
                            "remote_workspace_root": "remote/workspace",
                            "connection": {
                                "kind": "ssh-profile",
                                "profile": "interactive-login",
                                "role": "login",
                                "config": "secrets/hpc/ssh-profiles.yaml",
                            },
                        },
                        "provision": {
                            "scopes": [
                                {
                                    "name": "common",
                                    "entries": [
                                        {
                                            "label": "research-core",
                                            "kind": "directory",
                                            "source": "packages/research-core",
                                            "destination": "packages/research-core",
                                            "exclude_files": [],
                                        }
                                    ],
                                }
                            ]
                        },
                    },
                    status={"run_id": "unit-stage-profile", "state": "planned"},
                    exclude_file=None,
                )["report"]
            finally:
                os.chdir(original_cwd)

        self.assertEqual(report["connection"]["profile"], "interactive-login")
        self.assertEqual(report["connection"]["config"], str(ssh_config.resolve()))
        self.assertEqual(report["prepare_commands"][0][0], "ssh")
        self.assertIn("alice@cluster.example", report["prepare_commands"][0])
        self.assertTrue(report["prepare_commands"])
        self.assertIn("-e", report["push_command"])

    def test_build_pull_plan_uses_profile_aware_connection_when_manifest_declares_profile(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            subdirectory = workspace_root / "project" / "demo"
            run_root = workspace_root / "artifacts" / "runs" / "unit-pull-profile"
            ssh_config = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            run_root.mkdir(parents=True, exist_ok=True)
            subdirectory.mkdir(parents=True, exist_ok=True)
            (workspace_root / "WORKSPACE.yaml").write_text("projects:\n  default: demo\n", encoding="utf-8")
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
            os.chdir(subdirectory)
            try:
                report = build_pull_plan(
                    workspace_root=workspace_root,
                    run_root=run_root,
                    manifest={
                        "run_id": "unit-pull-profile",
                        "hpc": {
                            "remote_run_root": "remote/workspace/artifacts/runs/unit-pull-profile",
                            "connection": {
                                "kind": "ssh-profile",
                                "profile": "interactive-login",
                                "role": "login",
                                "config": "secrets/hpc/ssh-profiles.yaml",
                            },
                        },
                    },
                    status={"run_id": "unit-pull-profile", "state": "planned"},
                    exclude_file=None,
                )["report"]
            finally:
                os.chdir(original_cwd)

        self.assertEqual(report["connection"]["profile"], "interactive-login")
        self.assertEqual(report["connection"]["config"], str(ssh_config.resolve()))
        self.assertEqual(report["pull_command"][0], "rsync")
        self.assertTrue(report["pull_command"])
        self.assertTrue(report["progress"])
        self.assertIn("alice@cluster.example:remote/workspace/artifacts/runs/unit-pull-profile/", report["pull_command"])


if __name__ == "__main__":
    unittest.main()
