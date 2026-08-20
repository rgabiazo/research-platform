from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
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

from research_platform.core.cli import (
    _build_hpc_data_verification_plan,
    _build_selected_data_sync_plan,
    _build_ssh_tunnel_command,
    _build_notebook_bootstrap_command,
    _build_notebook_allocation_proxy_script,
    _build_notebook_url_proxy_script,
    _build_notebook_remote_launch_command,
    _build_local_notebook_url,
    _compute_notebook_bootstrap_stamp,
    _build_hpc_container_prepare_remote_command,
    _build_parser,
    _launch_remote_notebook_start,
    _parse_machine_readable_markers,
    _parse_salloc_job_id,
    _resolve_project_container_prepare_spec,
    _render_command,
    main,
)
from research_platform.core.config import write_yaml
from research_platform.hpc.ssh_profiles import load_ssh_profile


class _FakePopenProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[str] | None = None,
        stderr_text: str = "",
        returncode: int = 0,
        finished: bool = False,
    ) -> None:
        self.stdout = iter(stdout_lines or [])
        self.stderr = io.StringIO(stderr_text)
        self.returncode = returncode
        self._finished = finished

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self._finished = True
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode if self._finished else None

    def terminate(self) -> None:
        self._finished = True

    def kill(self) -> None:
        self._finished = True


class ProjectAwareHpcCliTests(unittest.TestCase):
    @staticmethod
    def _normalize_sync_output(output: str) -> str:
        return re.sub(r"\S*exclude\.git-auto\.txt", "<git-auto-exclude-file>", output)

    def _run_cli(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        artifact_root: Path,
        extra_env: dict[str, str | None] | None = None,
    ) -> tuple[int, str]:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(workspace_root),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
        }
        unset_keys: list[str] = []
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                    unset_keys.append(key)
                    continue
                env[key] = value
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False):
            for key in unset_keys:
                os.environ.pop(key, None)
            with redirect_stdout(buffer):
                exit_code = main(args)
        return exit_code, buffer.getvalue()

    def _run_cli_system_exit(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        artifact_root: Path,
        extra_env: dict[str, str | None] | None = None,
    ) -> str:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(workspace_root),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
        }
        unset_keys: list[str] = []
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                    unset_keys.append(key)
                    continue
                env[key] = value
        with mock.patch.dict(os.environ, env, clear=False):
            for key in unset_keys:
                os.environ.pop(key, None)
            with self.assertRaises(SystemExit) as exc_info:
                main(args)
        return str(exc_info.exception)

    def _hpc_help(self, *command: str) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as exc_info:
                _build_parser().parse_args(["hpc", *command, "--help"])
        self.assertEqual(exc_info.exception.code, 0)
        normalized = re.sub(r"\s+", " ", buffer.getvalue()).strip()
        return normalized.replace("- ", "-")

    def _write_ssh_config(self, root: Path) -> Path:
        config_path = root / "ssh-profiles.yaml"
        write_yaml(
            config_path,
            {
                "profiles": {
                    "interactive-login": {
                        "host": "cluster.example",
                        "user": "alice",
                    }
                }
            },
        )
        return config_path

    def _write_external_ssh_config(self, workspace_root: Path) -> Path:
        return self._write_ssh_config(workspace_root.parent / f"{workspace_root.name}-ssh")

    def _write_fake_jupyter_executable(self, root: Path) -> Path:
        executable = root / "jupyter"
        executable.write_text(
            "\n".join(
                [
                    "#!/bin/sh",
                    "port=''",
                    "for arg in \"$@\"; do",
                    "  case \"$arg\" in",
                    "    --port=*) port=\"${arg#--port=}\" ;;",
                    "  esac",
                    "done",
                    "printf 'Notebook ready\\n'",
                    "printf 'http://127.0.0.1:%s/lab?token=abc123\\n' \"$port\"",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    def _write_fake_socket_module(self, root: Path, *, selected_port: int) -> Path:
        module_path = root / "socket.py"
        module_path.write_text(
            "\n".join(
                [
                    "AF_INET = object()",
                    "SOCK_STREAM = object()",
                    "",
                    "class socket:",
                    "    def __init__(self, family, socktype):",
                    "        self.family = family",
                    "        self.socktype = socktype",
                    "        self._address = ('127.0.0.1', 0)",
                    "",
                    "    def __enter__(self):",
                    "        return self",
                    "",
                    "    def __exit__(self, exc_type, exc, tb):",
                    "        return False",
                    "",
                    "    def bind(self, address):",
                    f"        self._address = (address[0], {selected_port})",
                    "",
                    "    def getsockname(self):",
                    "        return self._address",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return module_path

    def _write_role_aware_ssh_config(self, root: Path) -> Path:
        config_path = root / "ssh-profiles.yaml"
        write_yaml(
            config_path,
            {
                "profiles": {
                    "env-profile": {
                        "defaults": {"host": "env.cluster.example"},
                        "roles": {
                            "login": {"user": "env-login"},
                            "robot": {"user": "env-robot"},
                        },
                    },
                    "cli-profile": {
                        "defaults": {"host": "cli.cluster.example"},
                        "roles": {
                            "login": {"user": "cli-login"},
                            "robot": {"user": "cli-robot"},
                        },
                    },
                    "interactive-login": {
                        "defaults": {"host": "cluster.example"},
                        "roles": {
                            "login": {"user": "alice"},
                            "robot": {"user": "robot"},
                        },
                    },
                }
            },
        )
        return config_path

    def _write_local_hpc_env_file(
        self,
        workspace_root: Path,
        *,
        ssh_config: Path,
        remote_workspace_root: str,
        profile: str | None = None,
        role: str | None = None,
        profile_key: str = "RP_HPC_PROFILE",
        role_key: str = "RP_HPC_ROLE",
    ) -> Path:
        env_path = workspace_root / "secrets" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"RP_SSH_CONFIG={ssh_config}",
            f"RP_REMOTE_WORKSPACE_ROOT={remote_workspace_root}",
            "ALLIANCE_USER=alice",
        ]
        if profile is not None:
            lines.append(f"{profile_key}={profile}")
        if role is not None:
            lines.append(f"{role_key}={role}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return env_path

    def _write_overlay_workspace(
        self,
        root: Path,
        *,
        notebook_hpc: dict[str, object] | None = None,
        hpc_data_roots: list[dict[str, object]] | None = None,
        runtime_slurm: dict[str, object] | None = None,
        slurm_template_text: str | None = None,
        create_notebook: bool = True,
    ) -> None:
        workspace_config: dict[str, object] = {
            "workspace": {"name": "research-platform", "version": "0.1.0"},
            "paths": {
                "datasets_root": "./datasets",
                "artifacts_root": "./artifacts",
                "ops_root": "./ops",
            },
            "repos": {
                "project_root": "./project",
                "pipelines_root": "./pipelines",
            },
            "projects": {"default": "project-demo-notebook"},
        }
        if runtime_slurm is not None:
            workspace_config["hpc"] = {
                "runtime_defaults": {
                    "default": "site-default",
                    "catalog": {
                        "site-default": {
                            "slurm": runtime_slurm,
                        }
                    },
                }
            }
        write_yaml(root / "WORKSPACE.yaml", workspace_config)
        bootstrap_script = root / "ops" / "envs" / "dev" / "bootstrap.sh"
        bootstrap_script.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        slurm_template = root / "ops" / "slurm" / "job_templates" / "sbatch.job.sh"
        slurm_template.parent.mkdir(parents=True, exist_ok=True)
        slurm_template.write_text(
            slurm_template_text
            if slurm_template_text is not None
            else "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "#SBATCH --job-name={{ job_name }}",
                    "#SBATCH --cpus-per-task={{ cpus }}",
                    "#SBATCH --mem={{ mem }}",
                    "#SBATCH --time={{ time }}",
                    "#SBATCH --output={{ log_out }}",
                    "#SBATCH --error={{ log_err }}",
                    "",
                    "set -euo pipefail",
                    "",
                    "{{ command }}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        activate_script = root / ".venv" / "bin" / "activate"
        activate_script.parent.mkdir(parents=True, exist_ok=True)
        activate_script.write_text("# test activate\n", encoding="utf-8")
        (root / "packages" / "research-core" / "src").mkdir(parents=True, exist_ok=True)
        (root / "packages" / "research-hpc" / "src").mkdir(parents=True, exist_ok=True)
        project_root = root / "project" / "project-demo-notebook"
        notebook_path = project_root / "notebooks" / "notebook_analysis.ipynb"
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        if create_notebook:
            notebook_path.write_text("{}", encoding="utf-8")
        project_config: dict[str, object] = {
            "name": "project-demo-notebook",
            "datasets": ["ds-private-example"],
            "overlay": {
                "private_data_root": "datasets/ds-private-example",
                "raw_inputs": {
                    "source_a_root": "datasets/ds-private-example/raw/source_a",
                    "source_b_root": "datasets/ds-private-example/raw/source_b",
                    "source_c_root": "datasets/ds-private-example/raw/source_c",
                },
                "outputs": {
                    "figures": "artifacts/figures/project-demo-notebook",
                },
                "notebook": "project/project-demo-notebook/notebooks/notebook_analysis.ipynb",
            },
        }
        hpc_config: dict[str, object] = {}
        if notebook_hpc is not None:
            hpc_config["notebook"] = notebook_hpc
        if hpc_data_roots is not None:
            hpc_config["data_roots"] = hpc_data_roots
        if hpc_config:
            project_config["hpc"] = hpc_config
        write_yaml(project_root / "project.yaml", project_config)
        for relative_path in (
            "datasets/ds-private-example/raw/source_a",
            "datasets/ds-private-example/raw/source_b",
            "datasets/ds-private-example/raw/source_c",
        ):
            (root / relative_path).mkdir(parents=True, exist_ok=True)

    def _write_structured_bids_workspace(
        self,
        root: Path,
        *,
        project_name: str,
        dataset_root: Path,
        derivative_root: Path,
        remote_dataset_root: str | None = None,
        remote_derivative_root: str | None = None,
    ) -> None:
        pipeline_root = root / "pipelines" / "preprocess-bids"
        project_root = root / "project" / project_name
        for path in (
            pipeline_root / "config",
            pipeline_root / "profiles" / "local",
            pipeline_root / "profiles" / "slurm",
            project_root / "config",
            project_root / "manifests" / "batches",
            root / "ops" / "sync" / "rsync",
        ):
            path.mkdir(parents=True, exist_ok=True)

        write_yaml(
            root / "WORKSPACE.yaml",
            {
                "workspace": {"name": "research-platform", "version": "0.1.0"},
                "paths": {
                    "datasets_root": "./datasets",
                    "artifacts_root": "./artifacts",
                    "ops_root": "./ops",
                },
                "repos": {
                    "project_root": "./project",
                    "pipelines_root": "./pipelines",
                },
                "projects": {"default": project_name},
            },
        )
        write_yaml(project_root / "project.yaml", {"name": project_name, "version": "0.1.0"})
        dataset_config = {
            "dataset": {
                "primary": project_name,
                "bids_root": str(dataset_root),
                "input_derivative": "deepprep-bold",
                "input_derivative_root": str(derivative_root),
            }
        }
        if remote_dataset_root is not None:
            dataset_config["dataset"]["remote_bids_root"] = remote_dataset_root
        if remote_derivative_root is not None:
            dataset_config["dataset"]["remote_input_derivative_root"] = remote_derivative_root
        write_yaml(project_root / "config" / "dataset.yaml", dataset_config)
        write_yaml(
            project_root / "config" / "compute.yaml",
            {
                "compute": {
                    "default_profile": "local",
                    "slurm": {
                        "ssh_host": "${RP_HPC_HOST:-}",
                        "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                        "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                        "cpus": 4,
                        "mem": "16G",
                        "time": "02:00:00",
                    },
                }
            },
        )
        write_yaml(
            project_root / "config" / "preprocessing.yaml",
            {
                "preprocessing": {
                    "slice": "bids",
                    "pipeline": "preprocess-bids",
                    "tool": "fmripost_aroma",
                    "tool_adapter": "research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter",
                    "input_derivative": "deepprep-bold",
                    "default_batch": "default",
                    "local_profile": "local",
                    "slurm_profile": "slurm",
                    "tool_options": {"denoising_method": "nonaggr", "dummy_scans": 0, "low_mem": False},
                    "publish_back": {"default_policy": "never"},
                }
            },
        )
        (project_root / "manifests" / "batches" / "default.tsv").write_text(
            "subject_id\tsession_id\ttask_id\trun_id\n",
            encoding="utf-8",
        )
        write_yaml(
            pipeline_root / "config" / "defaults.yaml",
            {
                "workflow": {"default_target": "fmripost_aroma", "rule_name": "fmripost_aroma"},
                "planner": {
                    "outputs": {
                        "runtime_plan_filename": "fmripost-aroma-plan.json",
                        "command_script_filename": "run-fmripost-aroma.sh",
                        "completion_marker_filename": "fmripost-aroma-complete.txt",
                        "output_data_dirname": "fmripost_aroma",
                    }
                },
            },
        )
        write_yaml(pipeline_root / "profiles" / "local" / "config.yaml", {"profile": {"name": "local"}})
        write_yaml(pipeline_root / "profiles" / "slurm" / "config.yaml", {"profile": {"name": "slurm"}})
        (root / "ops" / "sync" / "rsync" / "exclude.txt").write_text("", encoding="utf-8")
        (root / "ops" / "sync" / "rsync" / "exclude.neuro-bids.txt").write_text("", encoding="utf-8")

    def _write_structured_tabular_workspace(
        self,
        root: Path,
        *,
        project_name: str,
        hpc_data_roots: list[dict[str, object]] | None = None,
        include_canonical_fields: bool = True,
    ) -> None:
        project_root = root / "project" / project_name
        data_root = root / "datasets" / "ds-tabular-example"
        data_root.mkdir(parents=True, exist_ok=True)
        (project_root / "config").mkdir(parents=True, exist_ok=True)
        (project_root / "manifests" / "batches").mkdir(parents=True, exist_ok=True)

        write_yaml(
            root / "WORKSPACE.yaml",
            {
                "workspace": {"name": "research-platform", "version": "0.1.0"},
                "paths": {
                    "datasets_root": "./datasets",
                    "artifacts_root": "./artifacts",
                    "ops_root": "./ops",
                },
                "repos": {
                    "project_root": "./project",
                    "pipelines_root": "./pipelines",
                },
                "projects": {"default": project_name},
            },
        )

        project_config: dict[str, object] = {"name": project_name, "version": "0.1.0"}
        if hpc_data_roots is not None:
            project_config["hpc"] = {"data_roots": hpc_data_roots}
        write_yaml(project_root / "project.yaml", project_config)

        dataset_config: dict[str, object] = {"primary": "ds-tabular-example"}
        if include_canonical_fields:
            dataset_config["canonical_dataset"] = "ds-tabular-example"
            dataset_config["canonical_features_root"] = "derivatives/features"
            (data_root / "derivatives" / "features").mkdir(parents=True, exist_ok=True)
        write_yaml(project_root / "config" / "dataset.yaml", {"dataset": dataset_config})

        write_yaml(
            project_root / "config" / "compute.yaml",
            {
                "compute": {
                    "default_profile": "local",
                    "local": {"jobs": 1},
                    "slurm": {
                        "ssh_host": "${RP_HPC_HOST:-}",
                        "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                        "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                        "cpus": 1,
                        "mem": "4G",
                        "time": "00:30:00",
                    },
                }
            },
        )
        write_yaml(
            project_root / "config" / "preprocessing.yaml",
            {
                "preprocessing": {
                    "slice": "tabular",
                    "default_batch": "default",
                }
            },
        )
        write_yaml(
            project_root / "config" / "models.yaml",
            {
                "models": {
                    "default": {
                        "kind": "logistic_regression",
                        "feature_columns": ["feature_a"],
                    }
                }
            },
        )
        (project_root / "manifests" / "batches" / "default.tsv").write_text("row_id\n", encoding="utf-8")

    def _write_sync_workspace_git_repo(self, root: Path) -> None:
        write_yaml(
            root / "WORKSPACE.yaml",
            {
                "workspace": {"name": "research-platform", "version": "0.1.0"},
                "projects": {"default": "project-test"},
            },
        )
        write_yaml(root / "project" / "project-test" / "project.yaml", {"name": "project-test"})
        exclude_target = root / "ops" / "sync" / "rsync" / "exclude.workspace.txt"
        exclude_target.parent.mkdir(parents=True, exist_ok=True)
        exclude_target.write_text(
            (WORKSPACE_ROOT / "ops" / "sync" / "rsync" / "exclude.workspace.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
        tracked_file = root / "tracked.txt"
        tracked_file.write_text("tracked\n", encoding="utf-8")
        untracked_dir = root / "local-notes"
        untracked_dir.mkdir(parents=True, exist_ok=True)
        (untracked_dir / "note.txt").write_text("note\n", encoding="utf-8")
        (root / "ignored.log").write_text("ignored\n", encoding="utf-8")

        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "add", "WORKSPACE.yaml", "project/project-test/project.yaml", "ops/sync/rsync/exclude.workspace.txt", ".gitignore", "tracked.txt"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True, text=True)
        tracked_file.write_text("tracked but modified\n", encoding="utf-8")

    def _write_bootstrap_stamp_fixture(self, workspace_root: Path) -> None:
        (workspace_root / "ops" / "envs" / "dev" / "requirements-notebook.txt").write_text(
            "jupyterlab\n",
            encoding="utf-8",
        )
        (workspace_root / "packages" / "research-core" / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=68"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                    "[project]",
                    'name = "research-core"',
                    'version = "0.1.0"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (workspace_root / "ops" / "envs" / "dev" / "bootstrap.sh").write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "mkdir -p .venv/bin",
                    "printf '# test activate\\n' > .venv/bin/activate",
                    "count_file='.bootstrap-count'",
                    "count=0",
                    "if [ -f \"$count_file\" ]; then",
                    "  count=\"$(cat \"$count_file\")\"",
                    "fi",
                    "count=$((count + 1))",
                    "printf '%s\\n' \"$count\" > \"$count_file\"",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _run_notebook_bootstrap_command(self, workspace_root: Path) -> subprocess.CompletedProcess[str]:
        command = _build_notebook_bootstrap_command({"workspace_root": workspace_root})
        self.assertIsNotNone(command)
        assert command is not None
        return subprocess.run(
            ["bash", "-lc", command],
            check=False,
            capture_output=True,
            text=True,
            cwd=workspace_root,
        )

    def _assert_sync_command_output_matches(
        self,
        *,
        legacy_args: list[str],
        alias_args: list[str],
        workspace_root: Path,
        artifact_root: Path,
    ) -> None:
        legacy_exit_code, legacy_output = self._run_cli(
            legacy_args,
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
        alias_exit_code, alias_output = self._run_cli(
            alias_args,
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

        self.assertEqual(legacy_exit_code, 0)
        self.assertEqual(alias_exit_code, 0)
        self.assertEqual(self._normalize_sync_output(alias_output), self._normalize_sync_output(legacy_output))

    def test_hpc_help_states_local_remote_and_evidence_boundaries(self) -> None:
        help_contracts = {
            ("setup",): (
                "Canonical beginner setup",
                "provider-neutral generic target",
                "Alliance integration explicitly",
                "makes no network call",
                "does not test credentials",
                "rp hpc validate",
            ),
            ("init",): (
                "Legacy/backward-compatible Alliance-oriented helper",
                "rp hpc setup",
                "no connectivity or provider-readiness claim",
            ),
            ("doctor",): (
                "immediately check connectivity to the configured host",
                "authentication, or MFA",
                "not a local-only validation",
            ),
            ("verify", "data"): (
                "immediately contact the selected host over SSH",
                "read-only remote check",
                "no live-cluster validation claim",
            ),
            ("status",): (
                "recorded local manifest/status state without a subprocess",
                "run one squeue query",
                "does not query sacct",
                "reported ambiguously",
            ),
            ("cancel",): (
                "Record local cancel-requested state",
                "invokes no SSH or scheduler subprocess",
                "does not confirm remote cancellation",
            ),
            ("pull",): (
                "merge-oriented rsync -az",
                "does not prove scheduler success",
                "atomically publish a complete result",
                "guarantee interrupted-transfer recovery",
            ),
        }
        for command, expected_phrases in help_contracts.items():
            with self.subTest(command=command):
                help_text = self._hpc_help(*command)
                for phrase in expected_phrases:
                    self.assertIn(phrase, help_text)

        cancel_args = _build_parser().parse_args(["hpc", "cancel", "--run-id", "unit-cancel"])
        self.assertFalse(hasattr(cancel_args, "execute"))

    def test_hpc_doctor_reports_project_paths_and_next_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = self._write_ssh_config(temp_root)
            with mock.patch(
                "research_platform.core.cli.run_ssh_connectivity_check",
                return_value={"ok": True, "mode_used": "batch", "host_key_fix_guidance": ""},
            ) as connectivity_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "doctor",
                        "--project",
                        "project-pilot-bids",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )

            connectivity_mock.assert_called_once()
            self.assertEqual(connectivity_mock.call_args.kwargs["mode"], "auto")

        self.assertEqual(exit_code, 0)
        self.assertIn("HPC doctor for project-pilot-bids", output)
        self.assertIn("project/project-pilot-bids", output)
        self.assertIn("pipelines/preprocess-bids", output)
        self.assertIn("datasets/ds-bids-example", output)
        self.assertIn("multiplexing: not configured", output)
        self.assertIn("MFA-backed clusters may prompt once per SSH/rsync connection", output)
        self.assertIn(
            "rp hpc sync-project --project project-pilot-bids --profile interactive-login --role login",
            output,
        )
        self.assertIn("Review both rendered plans before authorizing remote changes.", output)
        self.assertIn(
            "rp hpc sync-project --project project-pilot-bids --profile interactive-login --role login --execute",
            output,
        )

    def test_hpc_doctor_reports_enabled_ssh_multiplexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = temp_root / "ssh-profiles.yaml"
            write_yaml(
                ssh_config,
                {
                    "profiles": {
                        "interactive-login": {
                            "host": "cluster.example",
                            "user": "alice",
                            "options": {
                                "ControlMaster": "auto",
                                "ControlPath": "~/.ssh/cm-%C",
                                "ControlPersist": "2h",
                            },
                        }
                    }
                },
            )
            with mock.patch(
                "research_platform.core.cli.run_ssh_connectivity_check",
                return_value={"ok": True, "mode_used": "batch", "host_key_fix_guidance": ""},
            ):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "doctor",
                        "--project",
                        "project-pilot-bids",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("multiplexing: enabled", output)
        self.assertNotIn("MFA-backed clusters may prompt once per SSH/rsync connection", output)

    def test_hpc_connect_opens_interactive_multiplexed_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = temp_root / "ssh-profiles.yaml"
            write_yaml(
                ssh_config,
                {
                    "profiles": {
                        "interactive-login": {
                            "host": "cluster.example",
                            "user": "alice",
                            "options": {
                                "ControlMaster": "auto",
                                "ControlPath": "~/.ssh/cm-%C",
                                "ControlPersist": "2h",
                            },
                        }
                    }
                },
            )
            with mock.patch(
                "research_platform.core.cli.run_ssh_connectivity_check",
                return_value={"ok": True, "returncode": 0},
            ) as ssh_check:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "connect",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Opening reusable SSH connection", output)
        self.assertIn("Multiplexing: enabled", output)
        self.assertIn("Connection ready", output)
        ssh_check.assert_called_once()
        self.assertEqual(ssh_check.call_args.kwargs["mode"], "interactive")
        self.assertEqual(ssh_check.call_args.kwargs["remote_command"], "true")

    def test_hpc_connect_rejects_profile_without_multiplexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = self._write_ssh_config(temp_root)
            with mock.patch("research_platform.core.cli.run_ssh_connectivity_check") as ssh_check:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "connect",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("Multiplexing: not configured", output)
        self.assertIn("not configured for reusable SSH connections", output)
        ssh_check.assert_not_called()

    def test_hpc_connect_can_confirm_batch_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = temp_root / "ssh-profiles.yaml"
            write_yaml(
                ssh_config,
                {
                    "profiles": {
                        "interactive-login": {
                            "host": "cluster.example",
                            "user": "alice",
                            "options": {
                                "ControlMaster": "auto",
                                "ControlPath": "~/.ssh/cm-%C",
                                "ControlPersist": "2h",
                            },
                        }
                    }
                },
            )
            with mock.patch(
                "research_platform.core.cli.run_ssh_connectivity_check",
                side_effect=[
                    {"ok": True, "returncode": 0},
                    {"ok": True, "returncode": 0, "fallback_to_interactive": False, "stdout": "login01\n"},
                ],
            ) as ssh_check:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "connect",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--test-reuse",
                    ],
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Reuse check: ok", output)
        self.assertIn("Reuse host: login01", output)
        self.assertEqual(ssh_check.call_count, 2)
        self.assertEqual(ssh_check.call_args_list[0].kwargs["mode"], "interactive")
        self.assertEqual(ssh_check.call_args_list[1].kwargs["mode"], "batch")

    def test_hpc_sync_project_renders_code_only_sync_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = self._write_ssh_config(temp_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-project",
                    "--project",
                    "project-pilot-bids",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=WORKSPACE_ROOT,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("WORKSPACE.yaml -> remote/workspace/WORKSPACE.yaml", output)
        self.assertIn("project/project-pilot-bids -> remote/workspace/project/project-pilot-bids", output)
        self.assertIn("ops -> remote/workspace/ops", output)
        self.assertIn("packages/research-hpc -> remote/workspace/packages/research-hpc", output)
        self.assertIn("pipelines/preprocess-bids -> remote/workspace/pipelines/preprocess-bids", output)
        self.assertIn("ops/sync/rsync/exclude.project-overlay.txt", output)
        self.assertNotIn("datasets/ds-bids-example ->", output)
        self.assertIn("--dry-run", output)
        self.assertIn("BatchMode=no", output)

    def test_hpc_sync_workspace_alias_matches_legacy_command_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = self._write_ssh_config(temp_root)
            self._assert_sync_command_output_matches(
                legacy_args=[
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                alias_args=[
                    "hpc",
                    "sync",
                    "workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=WORKSPACE_ROOT,
                artifact_root=artifact_root,
            )

    def test_hpc_sync_project_alias_matches_legacy_command_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = self._write_ssh_config(temp_root)
            self._assert_sync_command_output_matches(
                legacy_args=[
                    "hpc",
                    "sync-project",
                    "--project",
                    "project-pilot-bids",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                alias_args=[
                    "hpc",
                    "sync",
                    "project",
                    "--project",
                    "project-pilot-bids",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=WORKSPACE_ROOT,
                artifact_root=artifact_root,
            )

    def test_hpc_sync_data_alias_matches_legacy_command_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            self._assert_sync_command_output_matches(
                legacy_args=[
                    "hpc",
                    "sync-data",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                alias_args=[
                    "hpc",
                    "sync",
                    "data",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

    def test_hpc_sync_project_and_data_default_or_dry_run_make_no_remote_calls(self) -> None:
        cases = (
            ["hpc", "sync-project", "--project", "project-demo-notebook"],
            ["hpc", "sync-project", "--project", "project-demo-notebook", "--dry-run"],
            ["hpc", "sync", "project", "--project", "project-demo-notebook"],
            ["hpc", "sync", "project", "--project", "project-demo-notebook", "--dry-run"],
            ["hpc", "sync-data", "--project", "project-demo-notebook"],
            ["hpc", "sync-data", "--project", "project-demo-notebook", "--dry-run"],
            ["hpc", "sync", "data", "--project", "project-demo-notebook"],
            ["hpc", "sync", "data", "--project", "project-demo-notebook", "--dry-run"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            for base_argv in cases:
                argv = [*base_argv, "--profile", "interactive-login", "--config", str(ssh_config)]
                with self.subTest(argv=argv):
                    with mock.patch(
                        "research_platform.core.cli.subprocess.run",
                        side_effect=AssertionError("sync planning must not invoke SSH or rsync"),
                    ) as run_mock:
                        exit_code, output = self._run_cli(
                            argv,
                            workspace_root=workspace_root,
                            artifact_root=artifact_root,
                        )

                    self.assertEqual(exit_code, 0)
                    run_mock.assert_not_called()
                    self.assertIn("sync plan", output.lower())

    def test_hpc_sync_workspace_default_or_dry_run_make_no_remote_calls(self) -> None:
        real_run = subprocess.run
        cases = (
            ["hpc", "sync-workspace"],
            ["hpc", "sync-workspace", "--dry-run"],
            ["hpc", "sync", "workspace"],
            ["hpc", "sync", "workspace", "--dry-run"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_sync_workspace_git_repo(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)

            def local_git_only(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
                self.assertEqual(command[0], "git", f"unexpected remote subprocess: {command}")
                return real_run(command, **kwargs)

            for base_argv in cases:
                argv = [*base_argv, "--profile", "interactive-login", "--config", str(ssh_config)]
                with self.subTest(argv=argv):
                    with mock.patch("research_platform.core.cli.subprocess.run", side_effect=local_git_only) as run_mock:
                        exit_code, output = self._run_cli(
                            argv,
                            workspace_root=workspace_root,
                            artifact_root=artifact_root,
                        )

                    self.assertEqual(exit_code, 0)
                    self.assertTrue(run_mock.call_args_list)
                    self.assertTrue(all(call.args[0][0] == "git" for call in run_mock.call_args_list))
                    match = re.search(r"Temporary local exclude file \(removed after planning\): (.+)", output)
                    self.assertIsNotNone(match)
                    assert match is not None
                    self.assertFalse(Path(match.group(1)).exists())

    def test_hpc_setup_writes_local_only_defaults_and_ssh_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)

            with (
                mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("setup must not invoke a subprocess"),
                ) as run_mock,
                mock.patch(
                    "research_platform.core.cli.run_ssh_connectivity_check",
                    side_effect=AssertionError("setup must not check SSH connectivity"),
                ) as connectivity_mock,
                mock.patch(
                    "research_platform.core.cli.subprocess.Popen",
                    side_effect=AssertionError("setup must not launch a subprocess"),
                ) as popen_mock,
                mock.patch(
                    "research_platform.core.cli.socket.create_connection",
                    side_effect=AssertionError("setup must not create a socket connection"),
                ) as socket_connect_mock,
                mock.patch(
                    "research_platform.core.cli.socket.getaddrinfo",
                    side_effect=AssertionError("setup must not resolve a host"),
                ) as dns_lookup_mock,
                mock.patch(
                    "research_platform.core.cli.socket.socket",
                    side_effect=AssertionError("setup must not construct a socket"),
                ) as socket_constructor_mock,
            ):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "setup",
                        "--target",
                        "synthetic-target",
                        "--user",
                        "alice",
                        "--host",
                        "login.synthetic.invalid",
                        "--remote-workspace-root",
                        "/scratch/alice/research-platform",
                        "--remote-artifacts-root",
                        "/scratch/alice/research-platform/artifacts",
                        "--role",
                        "login",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

            ssh_config = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            env_path = workspace_root / "secrets" / ".env"
            self.assertEqual(exit_code, 0)
            self.assertIn("HPC setup complete", output)
            self.assertTrue(ssh_config.exists())
            self.assertTrue(env_path.exists())
            self.assertEqual((workspace_root / "secrets").stat().st_mode & 0o777, 0o700)
            self.assertEqual((workspace_root / "secrets" / "hpc").stat().st_mode & 0o777, 0o700)
            self.assertEqual(ssh_config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(env_path.stat().st_mode & 0o777, 0o600)
            self.assertIn("RESEARCH_HPC_PROFILE=synthetic-target", env_path.read_text(encoding="utf-8"))
            self.assertNotIn("ALLIANCE_USER=", env_path.read_text(encoding="utf-8"))
            self.assertIn("rp hpc validate --target synthetic-target", output)
            run_mock.assert_not_called()
            connectivity_mock.assert_not_called()
            popen_mock.assert_not_called()
            socket_connect_mock.assert_not_called()
            dns_lookup_mock.assert_not_called()
            socket_constructor_mock.assert_not_called()

    def test_hpc_sync_data_selected_only_plans_batch_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            dataset_root = workspace_root / "external" / "BIDS"
            derivative_root = workspace_root / "external" / "BIDS" / "derivatives" / "DeepPrep" / "BOLD"
            bold_path = (
                derivative_root
                / "sub-synthetic16"
                / "ses-01"
                / "func"
                / "sub-synthetic16_ses-01_task-exampletask_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
            bold_path.parent.mkdir(parents=True, exist_ok=True)
            bold_path.write_text("placeholder", encoding="utf-8")
            dataset_root.mkdir(parents=True, exist_ok=True)
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/BIDS",
                remote_derivative_root="/remote/BIDS/derivatives/DeepPrep/BOLD",
            )
            (workspace_root / "project" / "project-bids" / "manifests" / "batches" / "smoke.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\n"
                "sub-synthetic16\tses-01\ttask-exampletask\trun-01\n",
                encoding="utf-8",
            )
            ssh_config = self._write_ssh_config(workspace_root)

            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync",
                    "data",
                    "--project",
                    "project-bids",
                    "--batch",
                    "smoke",
                    "--selected-only",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Data sync plan for project-bids", output)
        self.assertIn("selected-input", output)
        self.assertIn("sub-synthetic16_ses-01_task-exampletask_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz", output)
        self.assertIn("/remote/BIDS/derivatives/DeepPrep/BOLD/sub-synthetic16/ses-01/func", output)

    def test_hpc_sync_data_selected_only_fails_when_project_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_structured_tabular_workspace(workspace_root, project_name="project-tabular")
            ssh_config = self._write_ssh_config(workspace_root)

            message = self._run_cli_system_exit(
                [
                    "hpc",
                    "sync",
                    "data",
                    "--project",
                    "project-tabular",
                    "--batch",
                    "default",
                    "--selected-only",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertIn("selected-only data sync is currently supported only for adapter-backed BIDS projects", message)

    def test_hpc_status_live_queries_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            run_root = artifact_root / "runs" / "unit-live"
            run_root.mkdir(parents=True, exist_ok=True)
            write_yaml(
                run_root / "run-manifest.yaml",
                {
                    "run_id": "unit-live",
                    "execution": {"mode": "slurm"},
                    "slurm": {"job_id": "12345"},
                },
            )
            write_yaml(
                run_root / "status.yaml",
                {"run_id": "unit-live", "state": "submitted", "job_id": "12345", "last_updated": "2026-01-01T00:00:00+00:00"},
            )
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout="12345\tRUNNING\tjob\talice\t00:02\tcompute1\n",
                stderr="",
            )
            with mock.patch("research_platform.core.cli.subprocess.run", return_value=completed) as run_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "status",
                        "--run-id",
                        "unit-live",
                        "--live",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        payload = json.loads(output)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["scheduler"]["state"], "RUNNING")
        run_mock.assert_called_once()
        live_command = run_mock.call_args.args[0]
        self.assertEqual(live_command[0], "ssh")
        self.assertIn("squeue", live_command[-1])
        self.assertNotIn("sacct", live_command[-1])

    def test_hpc_status_live_preserves_ambiguous_empty_squeue_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            run_root = artifact_root / "runs" / "unit-live-empty"
            run_root.mkdir(parents=True, exist_ok=True)
            write_yaml(
                run_root / "run-manifest.yaml",
                {
                    "run_id": "unit-live-empty",
                    "execution": {"mode": "slurm"},
                    "slurm": {"job_id": "12345"},
                },
            )
            write_yaml(
                run_root / "status.yaml",
                {
                    "run_id": "unit-live-empty",
                    "state": "submitted",
                    "job_id": "12345",
                    "last_updated": "2026-01-01T00:00:00+00:00",
                },
            )
            completed = subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout="", stderr="")
            with mock.patch("research_platform.core.cli.subprocess.run", return_value=completed) as run_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "status",
                        "--run-id",
                        "unit-live-empty",
                        "--live",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        payload = json.loads(output)
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            payload["scheduler"],
            {
                "checked": True,
                "ok": True,
                "command": run_mock.call_args.args[0],
                "state": "not-found-or-completed",
            },
        )

    def test_hpc_local_status_and_cancel_invoke_no_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            run_root = artifact_root / "runs" / "unit-local-control"
            run_root.mkdir(parents=True, exist_ok=True)
            write_yaml(
                run_root / "run-manifest.yaml",
                {
                    "run_id": "unit-local-control",
                    "execution": {"mode": "slurm"},
                    "hpc": {"ssh_host": "synthetic.invalid"},
                    "slurm": {"job_id": "12345"},
                },
            )
            write_yaml(
                run_root / "status.yaml",
                {
                    "run_id": "unit-local-control",
                    "state": "submitted",
                    "job_id": "12345",
                    "last_updated": "2026-01-01T00:00:00+00:00",
                },
            )
            with (
                mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("local status/cancel must not invoke a subprocess"),
                ) as core_run_mock,
                mock.patch(
                    "research_platform.hpc.remote.subprocess.run",
                    side_effect=AssertionError("cancel must not invoke a remote subprocess"),
                ) as remote_run_mock,
            ):
                status_exit, status_output = self._run_cli(
                    ["hpc", "status", "--run-id", "unit-local-control"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
                cancel_exit, cancel_output = self._run_cli(
                    ["hpc", "cancel", "--run-id", "unit-local-control"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

            stored_status = (run_root / "status.yaml").read_text(encoding="utf-8")

        self.assertEqual(status_exit, 0)
        self.assertEqual(json.loads(status_output)["state"], "submitted")
        self.assertNotIn("scheduler", json.loads(status_output))
        self.assertEqual(cancel_exit, 0)
        cancel_payload = json.loads(cancel_output)
        self.assertEqual(cancel_payload["cancel_command"][-1], "scancel 12345")
        self.assertNotIn("cancelled", cancel_output.lower())
        self.assertIn("state: cancel-requested", stored_status)
        core_run_mock.assert_not_called()
        remote_run_mock.assert_not_called()

    def test_hpc_sync_workspace_dry_run_renders_remote_mkdir_and_default_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            ssh_config = self._write_ssh_config(temp_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=WORKSPACE_ROOT,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Workspace sync plan", output)
        self.assertIn("Remote mkdir command: ssh -o BatchMode=no", output)
        self.assertIn("mkdir -p remote/workspace", output)
        self.assertIn("Rsync command: rsync -az", output)
        self.assertIn("--dry-run", output)
        self.assertIn("Tracked exclude files: ops/sync/rsync/exclude.workspace.txt", output)
        self.assertIn("Git-untracked excludes:", output)
        self.assertIn("Git-ignored excludes:", output)
        self.assertIn("Auto-excluded Git paths:", output)

    def test_hpc_sync_workspace_reads_local_defaults_from_secrets_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_sync_workspace_git_repo(workspace_root)
            ssh_config = self._write_external_ssh_config(workspace_root)
            self._write_local_hpc_env_file(
                workspace_root,
                ssh_config=ssh_config,
                remote_workspace_root="/remote/from-env-file",
            )
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "RP_REMOTE_WORKSPACE_ROOT": None,
                    "RP_SSH_CONFIG": None,
                    "RESEARCH_HPC_SSH_CONFIG": None,
                },
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(f"SSH config: {ssh_config.resolve()}", output)
        self.assertIn("Remote workspace root: /remote/from-env-file", output)
        self.assertIn("mkdir -p /remote/from-env-file", output)

    def test_hpc_sync_workspace_resolves_saved_relative_ssh_config_from_repo_subdirectory(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_sync_workspace_git_repo(workspace_root)
            ssh_config = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            ssh_config.parent.mkdir(parents=True, exist_ok=True)
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
            self._write_local_hpc_env_file(
                workspace_root,
                ssh_config=Path("secrets/hpc/ssh-profiles.yaml"),
                remote_workspace_root="/remote/from-env-file",
            )
            os.chdir(workspace_root / "project" / "project-test")
            try:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "sync",
                        "workspace",
                        "--profile",
                        "interactive-login",
                        "--dry-run",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                    extra_env={
                        "RESEARCH_PLATFORM_ROOT": None,
                        "RP_REMOTE_WORKSPACE_ROOT": None,
                        "RP_SSH_CONFIG": None,
                        "RESEARCH_HPC_SSH_CONFIG": None,
                    },
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(exit_code, 0)
        self.assertIn(f"SSH config: {ssh_config.resolve()}", output)
        self.assertIn("mkdir -p /remote/from-env-file", output)

    def test_hpc_sync_workspace_real_environment_overrides_local_defaults_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_sync_workspace_git_repo(workspace_root)
            file_ssh_config = self._write_external_ssh_config(workspace_root)
            env_ssh_config = self._write_ssh_config(workspace_root)
            self._write_local_hpc_env_file(
                workspace_root,
                ssh_config=file_ssh_config,
                remote_workspace_root="/remote/from-env-file",
            )
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "RP_REMOTE_WORKSPACE_ROOT": "/remote/from-process-env",
                    "RP_SSH_CONFIG": str(env_ssh_config),
                    "RESEARCH_HPC_SSH_CONFIG": None,
                },
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(f"SSH config: {env_ssh_config.resolve()}", output)
        self.assertNotIn(f"SSH config: {file_ssh_config.resolve()}", output)
        self.assertIn("Remote workspace root: /remote/from-process-env", output)
        self.assertNotIn("Remote workspace root: /remote/from-env-file", output)

    def test_hpc_init_writes_supported_local_defaults_and_next_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            env_path = workspace_root / "secrets" / ".env"
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text(
                "\n".join(
                    (
                        "# local defaults",
                        "OTHER_KEY=keep-me",
                        "ALLIANCE_USER=stale-user",
                        "RP_REMOTE_WORKSPACE_ROOT=/old/workspace",
                        "RESEARCH_HPC_SSH_CONFIG=/old/ssh-profiles.yaml",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "init",
                    "--alliance-user",
                    "alice",
                    "--remote-workspace-root",
                    "/home/alice/research-platform",
                    "--remote-artifacts-root",
                    "/scratch/alice/research-platform/artifacts",
                    "--profile",
                    "interactive-login",
                    "--ssh-config",
                    "secrets/hpc/ssh-profiles.yaml",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

            env_text = env_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            env_text,
            "\n".join(
                (
                    "# local defaults",
                    "OTHER_KEY=keep-me",
                    "ALLIANCE_USER=alice",
                    "RP_REMOTE_WORKSPACE_ROOT=/home/alice/research-platform",
                    "RESEARCH_HPC_SSH_CONFIG=secrets/hpc/ssh-profiles.yaml",
                    "RP_REMOTE_ARTIFACTS_ROOT=/scratch/alice/research-platform/artifacts",
                    "RESEARCH_HPC_PROFILE=interactive-login",
                    "",
                )
            ),
        )
        self.assertIn("Wrote secrets/.env with local HPC defaults.", output)
        self.assertIn("rp hpc sync workspace --profile interactive-login\n", output)
        self.assertIn("Review both rendered plans before authorizing remote changes.", output)
        self.assertIn("rp hpc sync workspace --profile interactive-login --execute", output)
        self.assertIn("rp hpc sync project --project <project> --profile interactive-login --execute", output)

    def test_hpc_init_persists_remote_artifacts_root_from_environment_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"

            exit_code, _ = self._run_cli(
                [
                    "hpc",
                    "init",
                    "--alliance-user",
                    "alice",
                    "--remote-workspace-root",
                    "/home/alice/research-platform",
                    "--profile",
                    "interactive-login",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={"RP_REMOTE_ARTIFACTS_ROOT": "/scratch/alice/from-env"},
            )
            env_text = (workspace_root / "secrets" / ".env").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("RP_REMOTE_ARTIFACTS_ROOT=/scratch/alice/from-env", env_text)

    def test_hpc_init_leaves_remote_artifacts_root_unset_when_no_value_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"

            exit_code, _ = self._run_cli(
                [
                    "hpc",
                    "init",
                    "--alliance-user",
                    "alice",
                    "--remote-workspace-root",
                    "/home/alice/research-platform",
                    "--profile",
                    "interactive-login",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={"RP_REMOTE_ARTIFACTS_ROOT": None},
            )
            env_text = (workspace_root / "secrets" / ".env").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertNotIn("RP_REMOTE_ARTIFACTS_ROOT=", env_text)

    def test_hpc_init_rejects_non_absolute_or_shell_expanded_remote_workspace_root(self) -> None:
        for remote_workspace_root in ("~/research-platform", "$HOME/research-platform", "research-platform"):
            with self.subTest(remote_workspace_root=remote_workspace_root):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    workspace_root = Path(tmp_dir)
                    artifact_root = workspace_root / "artifacts"
                    message = self._run_cli_system_exit(
                        [
                            "hpc",
                            "init",
                            "--alliance-user",
                            "alice",
                            "--remote-workspace-root",
                            remote_workspace_root,
                            "--profile",
                            "interactive-login",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                        extra_env={"RP_SSH_CONFIG": None, "RESEARCH_HPC_SSH_CONFIG": None},
                    )

                self.assertIn("absolute path", message)
                self.assertIn("/home/<user>/research-platform", message)

    def test_hpc_sync_workspace_default_excludes_git_untracked_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            self._write_sync_workspace_git_repo(temp_root)
            ssh_config = self._write_external_ssh_config(temp_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=temp_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Git-untracked excludes: active (1 paths)", output)
        self.assertIn("Auto-excluded Git paths: 2", output)

    def test_hpc_sync_workspace_default_excludes_git_ignored_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            self._write_sync_workspace_git_repo(temp_root)
            ssh_config = self._write_external_ssh_config(temp_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=temp_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Git-ignored excludes: active (1 paths)", output)
        self.assertIn("Auto-excluded Git paths: 2", output)

    def test_hpc_sync_workspace_default_keeps_tracked_files_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            self._write_sync_workspace_git_repo(temp_root)
            ssh_config = self._write_external_ssh_config(temp_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=temp_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Auto-excluded Git paths: 2", output)
        self.assertNotIn("Auto-excluded Git paths: 3", output)

    def test_hpc_sync_workspace_rejects_shell_expanded_remote_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            self._write_sync_workspace_git_repo(temp_root)
            ssh_config = self._write_external_ssh_config(temp_root)
            message = self._run_cli_system_exit(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=temp_root,
                artifact_root=artifact_root,
                extra_env={"RP_REMOTE_WORKSPACE_ROOT": "~/research-platform"},
            )

        self.assertIn("must be an absolute path", message)
        self.assertIn("shell-expanded value like '~/...' or '$HOME/...'", message)
        self.assertIn("/home/<user>/research-platform", message)

    def test_hpc_sync_workspace_include_untracked_disables_only_untracked_safety_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            self._write_sync_workspace_git_repo(temp_root)
            ssh_config = self._write_external_ssh_config(temp_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                    "--include-untracked",
                ],
                workspace_root=temp_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Git-untracked excludes: inactive", output)
        self.assertIn("Git-ignored excludes: active (1 paths)", output)
        self.assertIn("Auto-excluded Git paths: 1", output)

    def test_hpc_sync_workspace_include_ignored_disables_only_ignored_safety_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            self._write_sync_workspace_git_repo(temp_root)
            ssh_config = self._write_external_ssh_config(temp_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-workspace",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                    "--include-ignored",
                ],
                workspace_root=temp_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Git-untracked excludes: active (1 paths)", output)
        self.assertIn("Git-ignored excludes: inactive", output)
        self.assertIn("Auto-excluded Git paths: 1", output)

    def test_hpc_sync_workspace_execute_runs_remote_mkdir_before_rsync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            artifact_root = temp_root / "artifacts"
            self._write_sync_workspace_git_repo(temp_root)
            ssh_config = self._write_external_ssh_config(temp_root)
            extra_exclude_file = temp_root / "extra-exclude.txt"
            extra_exclude_file.write_text("tmp/\n", encoding="utf-8")
            real_run = subprocess.run
            executed_commands: list[list[str]] = []

            def run_side_effect(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
                if command[0] == "git":
                    return real_run(command, **kwargs)
                executed_commands.append(command)
                return subprocess.CompletedProcess(args=command, returncode=0)

            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=run_side_effect,
            ) as run_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "sync-workspace",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--extra-exclude-file",
                        str(extra_exclude_file),
                        "--execute",
                    ],
                    workspace_root=temp_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn(str(extra_exclude_file), output)
        self.assertGreaterEqual(run_mock.call_count, 3)
        self.assertEqual(len(executed_commands), 2)
        mkdir_command, rsync_command = executed_commands
        self.assertEqual(mkdir_command[0], "ssh")
        self.assertEqual(mkdir_command[-1], "mkdir -p remote/workspace")
        self.assertEqual(rsync_command[0], "rsync")
        self.assertIn(
            str((temp_root / "ops" / "sync" / "rsync" / "exclude.workspace.txt").resolve()),
            rsync_command,
        )
        self.assertIn(str(extra_exclude_file.resolve()), rsync_command)
        self.assertNotIn("--dry-run", rsync_command)
        self.assertEqual(run_mock.call_args_list[-2].kwargs, {"check": False, "text": True})
        self.assertEqual(run_mock.call_args_list[-1].kwargs, {"check": False, "text": True})

    def test_hpc_sync_data_uses_overlay_data_roots_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-data",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("datasets/ds-private-example -> remote/workspace/datasets/ds-private-example", output)
        self.assertNotIn("raw/source_a ->", output)
        self.assertNotIn("raw/source_b ->", output)
        self.assertNotIn("raw/source_c ->", output)
        self.assertNotIn("source-a-root-root", output)
        self.assertNotIn("artifacts/figures", output)
        self.assertNotIn("project/project-demo-notebook ->", output)

    def test_hpc_sync_data_uses_declared_external_bids_roots_with_explicit_remote_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            dataset_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/deepprep-bold",
            )
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-data",
                    "--project",
                    "project-external-bids",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(f"{dataset_root} -> /remote/studies/project-external-bids", output)
        self.assertIn(
            f"{derivative_root} -> /remote/studies/project-external-bids/derivatives/deepprep-bold",
            output,
        )

    def test_hpc_sync_data_preserves_nested_external_bids_derivative_root_with_explicit_remote_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = dataset_root / "derivatives" / "DeepPrep" / "BOLD"
            dataset_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/DeepPrep/BOLD",
            )
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-data",
                    "--project",
                    "project-external-bids",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(f"{dataset_root} -> /remote/studies/project-external-bids", output)
        self.assertIn(f"{derivative_root} -> /remote/studies/project-external-bids/derivatives/DeepPrep/BOLD", output)
        self.assertIn("mkdir -p /remote/studies/project-external-bids", output)
        self.assertIn("mkdir -p /remote/studies/project-external-bids/derivatives/DeepPrep/BOLD", output)

    def test_hpc_sync_data_resolves_saved_relative_ssh_config_from_repo_subdirectory(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            dataset_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/deepprep-bold",
            )
            ssh_config = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            ssh_config.parent.mkdir(parents=True, exist_ok=True)
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
            self._write_local_hpc_env_file(
                workspace_root,
                ssh_config=Path("secrets/hpc/ssh-profiles.yaml"),
                remote_workspace_root="/remote/from-env-file",
            )
            os.chdir(workspace_root / "project" / "project-external-bids")
            try:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "sync",
                        "data",
                        "--project",
                        "project-external-bids",
                        "--profile",
                        "interactive-login",
                        "--dry-run",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                    extra_env={
                        "RESEARCH_PLATFORM_ROOT": None,
                        "RP_SSH_CONFIG": None,
                        "RESEARCH_HPC_SSH_CONFIG": None,
                    },
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(exit_code, 0)
        self.assertIn(f"SSH config: {ssh_config.resolve()}", output)
        self.assertIn(f"{dataset_root} -> /remote/studies/project-external-bids", output)
        self.assertIn(
            f"{derivative_root} -> /remote/studies/project-external-bids/derivatives/deepprep-bold",
            output,
        )

    def test_hpc_sync_data_skips_external_bids_roots_without_declared_remote_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            dataset_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
            )
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "sync-data",
                    "--project",
                    "project-external-bids",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--dry-run",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("No syncable entries were inferred for this project.", output)
        self.assertNotIn(str(dataset_root), output)
        self.assertNotIn(str(derivative_root), output)

    def test_hpc_notebook_plan_uses_repo_bootstrap_and_virtualenv_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Notebook plan for project-demo-notebook", output)
        self.assertIn("ssh -o BatchMode=no", output)
        self.assertIn("salloc --job-name rp-notebook-project-demo-notebook", output)
        self.assertIn("bash ops/envs/dev/bootstrap.sh", output)
        self.assertIn("source .venv/bin/activate", output)
        self.assertNotIn("conda activate research-platform-hpc", output)
        self.assertIn("jupyter lab --no-browser --ip=127.0.0.1 --port=8888", output)
        self.assertIn("project/project-demo-notebook/notebooks/notebook_analysis.ipynb", output)

    def test_hpc_notebook_plan_reports_compute_host_tunnel_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Print the compute hostname from the allocated shell: hostname -f", output)
        self.assertIn("replacing <compute-hostname>", output)
        self.assertIn("rp hpc tunnel --profile interactive-login --role login", output)
        self.assertIn("-J alice@cluster.example", output)
        self.assertIn("alice@<compute-hostname>", output)

    def test_hpc_notebook_plan_uses_overlay_notebook_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                notebook_hpc={
                    "cpus": 6,
                    "mem": "24G",
                    "time": "03:15:00",
                    "local_port": 9010,
                    "remote_port": 9011,
                },
            )
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("--cpus-per-task 6 --mem 24G --time 03:15:00", output)

    def test_hpc_notebook_plan_applies_workspace_runtime_defaults_for_overlay_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                runtime_slurm={
                    "environment": {
                        "JUPYTER_CONFIG_DIR": "$SCRATCH/jupyter-config",
                        "JUPYTER_DATA_DIR": "$SCRATCH/jupyter-data",
                        "JUPYTER_RUNTIME_DIR": "$SCRATCH/jupyter-runtime",
                        "IPYTHONDIR": "$SCRATCH/ipython",
                    },
                    "prepare_directories": [
                        "$JUPYTER_CONFIG_DIR",
                        "$JUPYTER_DATA_DIR",
                        "$JUPYTER_RUNTIME_DIR",
                        "$IPYTHONDIR",
                    ],
                },
            )
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            'Prepare remote runtime directories: mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"',
            output,
        )

    def test_hpc_notebook_plan_keeps_overlay_workspace_relative_notebook_path_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root, create_notebook=False)
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "project/project-demo-notebook/notebooks/notebook_analysis.ipynb",
            output,
        )
        self.assertNotIn(
            "project/project-demo-notebook/project/project-demo-notebook/notebooks/notebook_analysis.ipynb",
            output,
        )

    def test_hpc_notebook_plan_cli_overrides_take_precedence_over_overlay_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                notebook_hpc={
                    "cpus": 6,
                    "mem": "24G",
                    "time": "03:15:00",
                    "local_port": 9010,
                    "remote_port": 9011,
                },
            )
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--cpus",
                    "2",
                    "--mem",
                    "12G",
                    "--time",
                    "01:00:00",
                    "--local-port",
                    "9020",
                    "--remote-port",
                    "9021",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("--cpus-per-task 2 --mem 12G --time 01:00:00", output)
        self.assertIn("jupyter lab --no-browser --ip=127.0.0.1 --port=9021", output)
        self.assertIn("--local-port 9020 --remote-port 9021", output)
        self.assertNotIn("--cpus-per-task 6 --mem 24G --time 03:15:00", output)

    def test_hpc_notebook_plan_renders_module_load_before_virtualenv_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                notebook_hpc={"modules": ["StdEnv/2023", "python/3.11"]},
            )
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("module load StdEnv/2023 python/3.11", output)
        self.assertLess(output.index("module load StdEnv/2023 python/3.11"), output.index("source .venv/bin/activate"))

    def test_hpc_notebook_plan_renders_tunnel_recommendation_with_selected_ports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--local-port",
                    "9030",
                    "--remote-port",
                    "9031",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            f"rp hpc tunnel --profile interactive-login --role login --config {ssh_config.resolve()} --compute-host 'alice@<compute-hostname>' --local-port 9030 --remote-port 9031",
            output,
        )
        self.assertIn("jupyter lab --no-browser --ip=127.0.0.1 --port=9031", output)

    def test_hpc_notebook_plan_rejects_shell_expanded_remote_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            message = self._run_cli_system_exit(
                [
                    "hpc",
                    "notebook",
                    "plan",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={"RP_REMOTE_WORKSPACE_ROOT": "$HOME/research-platform"},
            )

        self.assertIn("must be an absolute path", message)
        self.assertIn("shell-expanded value like '~/...' or '$HOME/...'", message)
        self.assertIn("/home/<user>/research-platform", message)

    def test_hpc_notebook_start_plan_only_preserves_compact_project_aware_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_external_ssh_config(workspace_root)
            self._write_local_hpc_env_file(
                workspace_root,
                ssh_config=ssh_config,
                remote_workspace_root="/remote/from-env-file",
                profile="interactive-login",
            )
            with mock.patch("research_platform.core.cli._launch_remote_notebook_start", return_value=0) as launch_mock:
                with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "start",
                            "--plan-only",
                            "--project",
                            "project-demo-notebook",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                        extra_env={
                            "RP_REMOTE_WORKSPACE_ROOT": None,
                            "RP_SSH_CONFIG": None,
                            "RESEARCH_HPC_SSH_CONFIG": None,
                            "RP_HPC_PROFILE": None,
                            "RESEARCH_HPC_PROFILE": None,
                        },
                    )

        self.assertEqual(exit_code, 0)
        launch_mock.assert_not_called()
        self.assertIn("Notebook start for project-demo-notebook", output)
        self.assertIn("cd /remote/from-env-file", output)
        self.assertIn("salloc --job-name rp-notebook-project-demo-notebook", output)
        self.assertIn("bash ops/envs/dev/bootstrap.sh", output)
        self.assertIn("source .venv/bin/activate", output)
        self.assertIn("jupyter lab --no-browser --ip=0.0.0.0 --port=8888", output)
        self.assertIn(
            f"rp hpc tunnel --profile interactive-login --role login --config {ssh_config.resolve()} --compute-host 'alice@<compute-hostname>' --local-port 9042 --remote-port 8888 --tunnel-mode login-forward",
            output,
        )
        self.assertIn("Local URL: http://127.0.0.1:9042", output)

    def test_hpc_notebook_start_derives_url_path_from_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "notebook",
                        "start",
                        "--plan-only",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--token",
                        "abc123",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("--url-path '/lab?token=abc123'", output)
        self.assertIn("Local URL: http://127.0.0.1:9042/lab?token=abc123", output)

    def test_hpc_notebook_start_renders_concrete_tunnel_command_when_compute_host_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._launch_remote_notebook_start", return_value=0) as launch_mock:
                with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "start",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--compute-host",
                            "compute001",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        self.assertEqual(exit_code, 0)
        launch_mock.assert_not_called()
        self.assertIn(f"--compute-host alice@compute001 --local-port 9042 --remote-port 8888", output)
        self.assertIn("Local URL: http://127.0.0.1:9042", output)
        self.assertNotIn("<compute-hostname>", output)

    def test_hpc_notebook_start_execute_tunnel_requires_compute_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            message = self._run_cli_system_exit(
                [
                    "hpc",
                    "notebook",
                    "start",
                    "--project",
                    "project-demo-notebook",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--execute-tunnel",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(message, "--execute-tunnel requires --compute-host.")

    def test_hpc_notebook_start_is_plan_only_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._launch_remote_notebook_start", return_value=0) as launch_mock:
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("notebook start planning must not invoke SSH"),
                ) as run_mock:
                    with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                        exit_code, output = self._run_cli(
                            [
                                "hpc",
                                "notebook",
                                "start",
                                "--project",
                                "project-demo-notebook",
                                "--profile",
                                "interactive-login",
                                "--config",
                                str(ssh_config),
                            ],
                            workspace_root=workspace_root,
                            artifact_root=artifact_root,
                        )

        self.assertEqual(exit_code, 0)
        launch_mock.assert_not_called()
        run_mock.assert_not_called()
        self.assertIn("Notebook start for project-demo-notebook", output)
        self.assertIn("salloc --job-name rp-notebook-project-demo-notebook", output)

    def test_hpc_notebook_start_remote_launch_uses_dynamic_remote_port_when_remote_port_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                notebook_hpc={
                    "local_port": 9010,
                    "remote_port": 9011,
                },
            )
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                with mock.patch("research_platform.core.cli._launch_remote_notebook_start", return_value=0) as launch_mock:
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "start",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--execute",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "")
        launch_mock.assert_called_once()
        self.assertEqual(launch_mock.call_args.kwargs["notebook_settings"]["local_port"], 9042)
        self.assertEqual(launch_mock.call_args.kwargs["notebook_settings"]["remote_port"], 0)

    def test_hpc_notebook_start_remote_launch_explicit_remote_port_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root, notebook_hpc={"remote_port": 9011})
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                with mock.patch("research_platform.core.cli._launch_remote_notebook_start", return_value=0) as launch_mock:
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "start",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--remote-port",
                            "9031",
                            "--execute",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "")
        self.assertEqual(launch_mock.call_args.kwargs["notebook_settings"]["remote_port"], 9031)

    def test_hpc_tunnel_uses_local_profile_and_role_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            ssh_config = self._write_role_aware_ssh_config(workspace_root)
            self._write_local_hpc_env_file(
                workspace_root,
                ssh_config=ssh_config,
                remote_workspace_root="/remote/from-env-file",
                profile="env-profile",
                role="robot",
                profile_key="RESEARCH_HPC_PROFILE",
                role_key="RP_HPC_ROLE",
            )
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "tunnel",
                    "--compute-host",
                    "compute001",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "RP_SSH_CONFIG": None,
                    "RESEARCH_HPC_SSH_CONFIG": None,
                    "RP_HPC_PROFILE": None,
                    "RESEARCH_HPC_PROFILE": None,
                    "RP_HPC_ROLE": None,
                    "RESEARCH_HPC_ROLE": None,
                },
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Tunnel command: ssh -o BatchMode=no", output)
        self.assertIn("-J env-robot@env.cluster.example", output)
        self.assertIn("8890:127.0.0.1:8888", output)
        self.assertIn("env-robot@compute001", output)

    def test_hpc_tunnel_explicit_cli_profile_and_role_override_local_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            ssh_config = self._write_role_aware_ssh_config(workspace_root)
            self._write_local_hpc_env_file(
                workspace_root,
                ssh_config=ssh_config,
                remote_workspace_root="/remote/from-env-file",
                profile="env-profile",
                role="robot",
            )
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "tunnel",
                    "--profile",
                    "cli-profile",
                    "--role",
                    "login",
                    "--compute-host",
                    "compute001",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "RP_SSH_CONFIG": None,
                    "RESEARCH_HPC_SSH_CONFIG": None,
                    "RP_HPC_PROFILE": None,
                    "RESEARCH_HPC_PROFILE": None,
                    "RP_HPC_ROLE": None,
                    "RESEARCH_HPC_ROLE": None,
                },
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("-J cli-login@cli.cluster.example", output)
        self.assertIn("cli-login@compute001", output)
        self.assertNotIn("env-robot@env.cluster.example", output)
        self.assertNotIn("env-robot@compute001", output)

    def test_hpc_tunnel_prefixes_compute_host_with_profile_user_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "tunnel",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--compute-host",
                    "compute001",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("alice@compute001", output)

    def test_hpc_tunnel_renders_expected_command_for_each_tunnel_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            ssh_config = self._write_ssh_config(workspace_root)

            for tunnel_mode in ("direct", "login-forward"):
                with self.subTest(tunnel_mode=tunnel_mode):
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "tunnel",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--compute-host",
                            "compute001",
                            "--tunnel-mode",
                            tunnel_mode,
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

                    self.assertEqual(exit_code, 0)
                    if tunnel_mode == "direct":
                        self.assertIn("-J alice@cluster.example", output)
                        self.assertIn("8890:127.0.0.1:8888", output)
                        self.assertIn("alice@compute001", output)
                    else:
                        self.assertNotIn("-J alice@cluster.example", output)
                        self.assertIn("8890:compute001:8888", output)
                        self.assertIn("alice@cluster.example", output)
                        self.assertNotIn("alice@compute001:8888", output)

    def test_hpc_tunnel_renders_local_url_when_url_path_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            ssh_config = self._write_ssh_config(workspace_root)
            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "tunnel",
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--compute-host",
                    "compute001",
                    "--url-path",
                    "lab/tree/demo?token=abc",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Local URL: http://127.0.0.1:8890/lab/tree/demo?token=abc", output)

    def test_hpc_tunnel_execute_keeps_manual_command_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            ssh_config = self._write_ssh_config(workspace_root)
            profile = load_ssh_profile(ssh_config, "interactive-login", role="login")
            expected_command = _build_ssh_tunnel_command(
                profile,
                local_port=8890,
                remote_port=8888,
                remote_host="alice@compute001",
                jump_host=profile.target(),
                tunnel_mode="direct",
            )
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ) as run_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "tunnel",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--compute-host",
                        "compute001",
                        "--execute",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "")
        run_mock.assert_called_once_with(expected_command, check=False)

    def test_hpc_sync_project_execute_runs_remote_mkdir_before_rsync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=lambda command, **kwargs: subprocess.CompletedProcess(args=command, returncode=0),
            ) as run_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "sync-project",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--execute",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("mkdir:", output)
        self.assertIn("ops -> remote/workspace/ops", output)
        self.assertIn("packages/research-hpc -> remote/workspace/packages/research-hpc", output)
        executed_commands = [call.args[0] for call in run_mock.call_args_list]
        executed_programs = [command[0] for command in executed_commands]
        self.assertEqual(len(executed_commands) % 2, 0)
        self.assertEqual(len(executed_commands), 10)
        expected_targets = [
            "remote/workspace",
            "remote/workspace/project/project-demo-notebook",
            "remote/workspace/ops",
            "remote/workspace/packages/research-core",
            "remote/workspace/packages/research-hpc",
        ]
        self.assertEqual(executed_programs, ["ssh", "rsync"] * len(expected_targets))
        for index, target in enumerate(expected_targets):
            mkdir_call = run_mock.call_args_list[index * 2]
            rsync_call = run_mock.call_args_list[index * 2 + 1]
            self.assertEqual(mkdir_call.kwargs, {"check": False, "text": True})
            self.assertEqual(rsync_call.kwargs, {"check": False, "text": True})
            self.assertEqual(mkdir_call.args[0][0], "ssh")
            self.assertEqual(rsync_call.args[0][0], "rsync")
            self.assertIn(f"mkdir -p {target}", mkdir_call.args[0][-1])

    def test_hpc_sync_data_execute_runs_remote_mkdir_before_rsync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess(args=["ssh"], returncode=0),
                    subprocess.CompletedProcess(args=["rsync"], returncode=0),
                ],
            ) as run_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "sync-data",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--execute",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("mkdir:", output)
        self.assertEqual([call.args[0][0] for call in run_mock.call_args_list], ["ssh", "rsync"])
        self.assertIn("datasets/ds-private-example -> remote/workspace/datasets/ds-private-example", output)

    def test_hpc_sync_data_execute_runs_distinct_nested_bids_mkdir_and_rsync_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = dataset_root / "derivatives" / "DeepPrep" / "BOLD"
            dataset_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/DeepPrep/BOLD",
            )
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess(args=["ssh"], returncode=0),
                    subprocess.CompletedProcess(args=["rsync"], returncode=0),
                    subprocess.CompletedProcess(args=["ssh"], returncode=0),
                    subprocess.CompletedProcess(args=["rsync"], returncode=0),
                ],
            ) as run_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "sync-data",
                        "--project",
                        "project-external-bids",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--execute",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual([call.args[0][0] for call in run_mock.call_args_list], ["ssh", "rsync", "ssh", "rsync"])
        self.assertIn("mkdir -p /remote/studies/project-external-bids", run_mock.call_args_list[0].args[0][-1])
        self.assertEqual(run_mock.call_args_list[1].args[0][0], "rsync")
        self.assertIn(
            "mkdir -p /remote/studies/project-external-bids/derivatives/DeepPrep/BOLD",
            run_mock.call_args_list[2].args[0][-1],
        )
        self.assertEqual(run_mock.call_args_list[3].args[0][0], "rsync")
        self.assertIn(f"{derivative_root} -> /remote/studies/project-external-bids/derivatives/DeepPrep/BOLD", output)

    def test_hpc_sync_project_and_data_execute_reach_boundary_for_both_spellings(self) -> None:
        cases = (
            ["hpc", "sync-project", "--project", "project-demo-notebook", "--execute"],
            ["hpc", "sync", "project", "--project", "project-demo-notebook", "--execute"],
            ["hpc", "sync-data", "--project", "project-demo-notebook", "--execute"],
            ["hpc", "sync", "data", "--project", "project-demo-notebook", "--execute"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            for base_argv in cases:
                argv = [*base_argv, "--profile", "interactive-login", "--config", str(ssh_config)]
                with self.subTest(argv=argv):
                    with mock.patch("research_platform.core.cli._execute_sync_actions", return_value=0) as execute_mock:
                        exit_code, _ = self._run_cli(
                            argv,
                            workspace_root=workspace_root,
                            artifact_root=artifact_root,
                        )

                    self.assertEqual(exit_code, 0)
                    execute_mock.assert_called_once()

    def test_hpc_sync_workspace_execute_reaches_each_remote_boundary_for_both_spellings(self) -> None:
        cases = (
            ["hpc", "sync-workspace", "--execute"],
            ["hpc", "sync", "workspace", "--execute"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_sync_workspace_git_repo(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            for base_argv in cases:
                argv = [
                    *base_argv,
                    "--profile",
                    "interactive-login",
                    "--config",
                    str(ssh_config),
                    "--include-untracked",
                    "--include-ignored",
                ]
                completed = subprocess.CompletedProcess(args=[], returncode=0)
                with self.subTest(argv=argv):
                    with mock.patch("research_platform.core.cli.subprocess.run", return_value=completed) as run_mock:
                        exit_code, _ = self._run_cli(
                            argv,
                            workspace_root=workspace_root,
                            artifact_root=artifact_root,
                        )

                    self.assertEqual(exit_code, 0)
                    self.assertEqual([call.args[0][0] for call in run_mock.call_args_list], ["ssh", "rsync"])

    def test_hpc_sync_parser_accepts_execute_and_keeps_legacy_spellings(self) -> None:
        parser = _build_parser()
        cases = (
            (
                ["hpc", "sync-workspace", "--profile", "interactive-login"],
                "_handle_hpc_sync_workspace",
            ),
            (
                ["hpc", "sync", "workspace", "--profile", "interactive-login"],
                "_handle_hpc_sync_workspace",
            ),
            (
                ["hpc", "sync-project", "--project", "project-pilot-bids", "--profile", "interactive-login"],
                "_handle_hpc_sync_project",
            ),
            (
                ["hpc", "sync", "project", "--project", "project-pilot-bids", "--profile", "interactive-login"],
                "_handle_hpc_sync_project",
            ),
            (
                ["hpc", "sync-data", "--project", "project-pilot-bids", "--profile", "interactive-login"],
                "_handle_hpc_sync_data",
            ),
            (
                ["hpc", "sync", "data", "--project", "project-pilot-bids", "--profile", "interactive-login"],
                "_handle_hpc_sync_data",
            ),
            (
                ["hpc", "container", "prepare", "--project", "project-pilot-bids", "--profile", "interactive-login"],
                "_handle_hpc_container_prepare",
            ),
            (
                ["hpc", "prepare-container", "--project", "project-pilot-bids", "--profile", "interactive-login"],
                "_handle_hpc_container_prepare",
            ),
        )

        for argv, expected_handler in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertEqual(args.handler.__name__, expected_handler)
                self.assertIsNone(args.role)
                self.assertFalse(args.execute)

        execute_cases = (
            ["hpc", "sync-workspace", "--profile", "interactive-login", "--execute"],
            ["hpc", "sync-project", "--project", "project-pilot-bids", "--profile", "interactive-login", "--execute"],
            ["hpc", "sync-data", "--project", "project-pilot-bids", "--profile", "interactive-login", "--execute"],
            ["hpc", "sync", "workspace", "--profile", "interactive-login", "--execute"],
            ["hpc", "sync", "project", "--project", "project-pilot-bids", "--profile", "interactive-login", "--execute"],
            ["hpc", "sync", "data", "--project", "project-pilot-bids", "--profile", "interactive-login", "--execute"],
        )

        for argv in execute_cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertTrue(args.execute)

    def test_hpc_sync_and_notebook_submit_reject_dry_run_with_execute(self) -> None:
        parser = _build_parser()
        cases = (
            ["hpc", "sync-workspace", "--dry-run", "--execute"],
            ["hpc", "sync", "workspace", "--dry-run", "--execute"],
            ["hpc", "sync-project", "--project", "project-pilot-bids", "--dry-run", "--execute"],
            ["hpc", "sync", "project", "--project", "project-pilot-bids", "--dry-run", "--execute"],
            ["hpc", "sync-data", "--project", "project-pilot-bids", "--dry-run", "--execute"],
            ["hpc", "sync", "data", "--project", "project-pilot-bids", "--dry-run", "--execute"],
            ["hpc", "notebook", "submit", "--project", "project-pilot-bids", "--dry-run", "--execute"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    with self.assertRaises(SystemExit) as exc_info:
                        parser.parse_args(argv)

                self.assertEqual(exc_info.exception.code, 2)
                self.assertIn("not allowed with argument --dry-run", stderr.getvalue())

    def test_remote_mutation_parsers_reject_removed_apply_alias(self) -> None:
        parser = _build_parser()
        cases = (
            ["hpc", "sync-workspace", "--apply"],
            ["hpc", "sync-project", "--project", "project-pilot-bids", "--apply"],
            ["hpc", "sync-data", "--project", "project-pilot-bids", "--apply"],
            ["hpc", "sync", "workspace", "--apply"],
            ["hpc", "sync", "project", "--project", "project-pilot-bids", "--apply"],
            ["hpc", "sync", "data", "--project", "project-pilot-bids", "--apply"],
            ["hpc", "container", "prepare", "--project", "project-pilot-bids", "--apply"],
            ["hpc", "prepare-container", "--project", "project-pilot-bids", "--apply"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                with mock.patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as exc_info:
                        parser.parse_args(argv)

                self.assertEqual(exc_info.exception.code, 2)

    def test_hpc_parser_keeps_existing_run_id_commands(self) -> None:
        parser = _build_parser()
        expected_handlers = {
            ("hpc", "stage", "--run-id", "run-123"): "_handle_hpc_stage",
            ("hpc", "bootstrap", "--run-id", "run-123"): "_handle_hpc_bootstrap",
            ("hpc", "status", "--run-id", "run-123"): "_handle_hpc_status",
            ("hpc", "pull", "--run-id", "run-123"): "_handle_hpc_pull",
            ("hpc", "cancel", "--run-id", "run-123"): "_handle_hpc_cancel",
        }

        for argv, expected_handler in expected_handlers.items():
            args = parser.parse_args(list(argv))
            self.assertEqual(args.handler.__name__, expected_handler)

    def test_hpc_container_prepare_renders_login_node_prestage_command(self) -> None:
        context = {
            "slice": "bids",
            "runtime_profile_name": "deepprep",
            "runtime_profile": {
                "slurm": {
                    "execution_backend": "apptainer",
                    "nextflow": {
                        "enabled": True,
                        "version": "24.10.3",
                        "host_home": "$SCRATCH/deepprep/nextflow",
                    },
                    "container": {
                        "source_image": "docker://pbfslab/deepprep:25.1.0",
                        "image": "$SCRATCH/containers/deepprep/deepprep_25.1.0.sif",
                        "pull_mode": "never",
                    },
                }
            },
            "compute": {
                "slurm": {
                    "modules": ["StdEnv/2023", "apptainer/1.4.5"],
                    "environment": {
                        "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                        "APPTAINER_CONFIGDIR": "$SCRATCH/apptainer-config",
                    },
                    "prepare_directories": ["$APPTAINER_CACHEDIR", "$APPTAINER_CONFIGDIR"],
                }
            },
        }

        spec = _resolve_project_container_prepare_spec(context=context)
        command = _build_hpc_container_prepare_remote_command(context=context, spec=spec)

        self.assertEqual(spec["runtime_image"], "$SCRATCH/containers/deepprep/deepprep_25.1.0.sif")
        self.assertEqual(spec["source_image"], "docker://pbfslab/deepprep:25.1.0")
        self.assertIn("module load StdEnv/2023 apptainer/1.4.5", command)
        self.assertIn('export APPTAINER_CONFIGDIR="$SCRATCH/apptainer-config"', command)
        self.assertIn('mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_CONFIGDIR"', command)
        self.assertIn('RUNTIME_IMAGE="$SCRATCH/containers/deepprep/deepprep_25.1.0.sif"', command)
        self.assertIn('IMAGE_SOURCE="docker://pbfslab/deepprep:25.1.0"', command)
        self.assertIn('apptainer pull "$TMP_IMAGE" "$IMAGE_SOURCE"', command)
        self.assertIn('ls -lh "$RUNTIME_IMAGE"', command)
        self.assertEqual(spec["nextflow_home"], "$SCRATCH/deepprep/nextflow")
        self.assertIn('NEXTFLOW_JAR_URL="https://www.nextflow.io/releases/v24.10.3/nextflow-24.10.3-one.jar"', command)
        self.assertIn('curl -fL --retry 3 --connect-timeout 30 -o "$TMP_NEXTFLOW_JAR" "$NEXTFLOW_JAR_URL"', command)

    def test_hpc_parser_accepts_init_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "hpc",
                "init",
                "--alliance-user",
                "alice",
                "--remote-workspace-root",
                "/home/alice/research-platform",
                "--profile",
                "interactive-login",
            ]
        )

        self.assertEqual(args.handler.__name__, "_handle_hpc_init")
        self.assertEqual(args.alliance_user, "alice")
        self.assertEqual(args.remote_workspace_root, "/home/alice/research-platform")
        self.assertEqual(args.profile, "interactive-login")

    def test_hpc_parser_accepts_tunnel_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "hpc",
                "tunnel",
                "--compute-host",
                "compute001",
            ]
        )

        self.assertEqual(args.handler.__name__, "_handle_hpc_tunnel")
        self.assertIsNone(args.profile)
        self.assertIsNone(args.role)
        self.assertEqual(args.compute_host, "compute001")
        self.assertEqual(args.local_port, 8890)
        self.assertEqual(args.remote_port, 8888)
        self.assertEqual(args.tunnel_mode, "direct")
        self.assertFalse(args.execute)

    def test_hpc_notebook_start_parser_accepts_plan_only_and_execute_flags(self) -> None:
        parser = _build_parser()
        plan_args = parser.parse_args(
            [
                "hpc",
                "notebook",
                "start",
                "--project",
                "project-pilot-bids",
                "--profile",
                "interactive-login",
                "--plan-only",
                "--open-browser",
            ]
        )
        execute_args = parser.parse_args(
            [
                "hpc",
                "notebook",
                "start",
                "--project",
                "project-pilot-bids",
                "--profile",
                "interactive-login",
                "--execute",
            ]
        )

        self.assertEqual(plan_args.handler.__name__, "_handle_hpc_notebook_start")
        self.assertTrue(plan_args.plan_only)
        self.assertFalse(plan_args.execute)
        self.assertTrue(plan_args.open_browser)
        self.assertEqual(plan_args.tunnel_mode, "login-forward")
        self.assertEqual(execute_args.handler.__name__, "_handle_hpc_notebook_start")
        self.assertTrue(execute_args.execute)

    def test_hpc_notebook_submit_parser_accepts_new_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "hpc",
                "notebook",
                "submit",
                "--project",
                "project-pilot-bids",
                "--notebook",
                "project/project-pilot-bids/notebooks/demo.ipynb",
                "--job-name",
                "demo-job",
                "--cpus",
                "4",
                "--mem",
                "12G",
                "--time",
                "03:00:00",
                "--output-notebook",
                "demo.executed.ipynb",
                "--profile",
                "interactive-login",
                "--role",
                "robot",
                "--config",
                "ssh-profiles.yaml",
                "--dry-run",
            ]
        )

        self.assertEqual(args.handler.__name__, "_handle_hpc_notebook_submit")
        self.assertEqual(args.notebook, "project/project-pilot-bids/notebooks/demo.ipynb")
        self.assertEqual(args.job_name, "demo-job")
        self.assertEqual(args.cpus, 4)
        self.assertEqual(args.mem, "12G")
        self.assertEqual(args.time, "03:00:00")
        self.assertEqual(args.output_notebook, "demo.executed.ipynb")
        self.assertEqual(args.profile, "interactive-login")
        self.assertEqual(args.role, "robot")
        self.assertEqual(args.config, "ssh-profiles.yaml")
        self.assertTrue(args.dry_run)
        self.assertFalse(args.execute)

    def test_hpc_notebook_parser_does_not_add_pre_activate_command_flags(self) -> None:
        parser = _build_parser()

        for subcommand in ("start", "submit"):
            with self.subTest(subcommand=subcommand):
                with self.assertRaises(SystemExit):
                    parser.parse_args(
                        [
                            "hpc",
                            "notebook",
                            subcommand,
                            "--project",
                            "project-pilot-bids",
                            "--pre-activate-command",
                            "module load arrow/23.0.1",
                        ]
                    )

    def test_hpc_notebook_marker_parser_extracts_expected_values(self) -> None:
        host = _parse_machine_readable_markers(
            "RP_NOTEBOOK_HOST=compute001.cluster.example\n",
            marker_names=("RP_NOTEBOOK_HOST", "RP_NOTEBOOK_URL"),
        )
        url = _parse_machine_readable_markers(
            "RP_NOTEBOOK_URL=http://127.0.0.1:8888/lab?token=abc123\n",
            marker_names=("RP_NOTEBOOK_HOST", "RP_NOTEBOOK_URL"),
        )
        unrelated = _parse_machine_readable_markers(
            "Server ready\n",
            marker_names=("RP_NOTEBOOK_HOST", "RP_NOTEBOOK_URL"),
        )

        self.assertEqual(host, {"RP_NOTEBOOK_HOST": "compute001.cluster.example"})
        self.assertEqual(url, {"RP_NOTEBOOK_URL": "http://127.0.0.1:8888/lab?token=abc123"})
        self.assertEqual(unrelated, {})

    def test_hpc_notebook_start_remote_launch_command_uses_no_shell_allocation_without_parsable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_overlay_workspace(workspace_root)

            command = _build_notebook_remote_launch_command(
                project_name="project-demo-notebook",
                project_context={
                    "workspace_root": workspace_root,
                    "remote_workspace_root": "/remote/workspace",
                    "notebook_path": workspace_root
                    / "project"
                    / "project-demo-notebook"
                    / "notebooks"
                    / "notebook_analysis.ipynb",
                },
                notebook_settings={
                    "cpus": 1,
                    "mem": "8G",
                    "time": "01:00:00",
                    "local_port": 8890,
                    "remote_port": 8888,
                    "modules": [],
                    "pre_activate_commands": [],
                },
            )

        self.assertIn("--no-shell", command)
        self.assertNotIn("--parsable", command)
        self.assertIn("Granted job allocation", command)
        self.assertIn("srun --jobid {jobid} hostname -f", command)

    def test_hpc_notebook_start_remote_launch_command_runs_modules_and_pre_activate_commands_before_bootstrap_and_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_overlay_workspace(workspace_root)

            command = _build_notebook_remote_launch_command(
                project_name="project-demo-notebook",
                project_context={
                    "workspace_root": workspace_root,
                    "remote_workspace_root": "/remote/workspace",
                    "notebook_path": workspace_root
                    / "project"
                    / "project-demo-notebook"
                    / "notebooks"
                    / "notebook_analysis.ipynb",
                },
                notebook_settings={
                    "cpus": 1,
                    "mem": "8G",
                    "time": "01:00:00",
                    "local_port": 8890,
                    "remote_port": 8888,
                    "modules": ["StdEnv/2023", "python/3.11"],
                    "pre_activate_commands": [
                        "module load arrow/23.0.1",
                        "export RP_CLUSTER=1",
                    ],
                    "prepare_directories": [],
                },
            )

        self.assertIn("module load StdEnv/2023 python/3.11", command)
        self.assertIn("module load arrow/23.0.1", command)
        self.assertIn("export RP_CLUSTER=1", command)
        self.assertIn("bash ops/envs/dev/bootstrap.sh", command)
        self.assertIn("source .venv/bin/activate", command)
        self.assertLess(command.index("module load StdEnv/2023 python/3.11"), command.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(command.index("module load arrow/23.0.1"), command.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(command.index("export RP_CLUSTER=1"), command.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(command.index("bash ops/envs/dev/bootstrap.sh"), command.index("source .venv/bin/activate"))

    def test_hpc_notebook_start_remote_launch_command_runs_prepare_directories_before_bootstrap_and_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_overlay_workspace(workspace_root)

            command = _build_notebook_remote_launch_command(
                project_name="project-demo-notebook",
                project_context={
                    "workspace_root": workspace_root,
                    "remote_workspace_root": "/remote/workspace",
                    "compute": {
                        "slurm": {
                            "environment": {
                                "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                                "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
                            }
                        }
                    },
                    "notebook_path": workspace_root
                    / "project"
                    / "project-demo-notebook"
                    / "notebooks"
                    / "notebook_analysis.ipynb",
                },
                notebook_settings={
                    "cpus": 1,
                    "mem": "8G",
                    "time": "01:00:00",
                    "local_port": 8890,
                    "remote_port": 8888,
                    "modules": ["apptainer/1.3"],
                    "pre_activate_commands": ["module load arrow/23.0.1"],
                    "prepare_directories": [
                        "$APPTAINER_CACHEDIR",
                        "$APPTAINER_TMPDIR",
                    ],
                },
            )

        self.assertIn("export APPTAINER_CACHEDIR=", command)
        self.assertIn("$SCRATCH/apptainer-cache", command)
        self.assertIn("export APPTAINER_TMPDIR=", command)
        self.assertIn("$SCRATCH/apptainer-tmp", command)
        self.assertIn("mkdir -p", command)
        self.assertIn("$APPTAINER_CACHEDIR", command)
        self.assertIn("$APPTAINER_TMPDIR", command)
        self.assertLess(command.index("module load arrow/23.0.1"), command.index("mkdir -p"))
        self.assertLess(command.index("mkdir -p"), command.index("bash ops/envs/dev/bootstrap.sh"))

    def test_hpc_notebook_bootstrap_command_runs_when_environment_is_missing_or_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_overlay_workspace(workspace_root)
            self._write_bootstrap_stamp_fixture(workspace_root)

            with self.subTest(state="missing"):
                result = self._run_notebook_bootstrap_command(workspace_root)

                self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
                self.assertEqual((workspace_root / ".bootstrap-count").read_text(encoding="utf-8").strip(), "1")
                self.assertEqual(
                    (workspace_root / ".rp-notebook-bootstrap.sha256").read_text(encoding="utf-8").strip(),
                    _compute_notebook_bootstrap_stamp(workspace_root),
                )

            with self.subTest(state="stale"):
                stale_stamp = _compute_notebook_bootstrap_stamp(workspace_root)
                (workspace_root / "packages" / "research-core" / "pyproject.toml").write_text(
                    "\n".join(
                        [
                            "[build-system]",
                            'requires = ["setuptools>=68"]',
                            'build-backend = "setuptools.build_meta"',
                            "",
                            "[project]",
                            'name = "research-core"',
                            'version = "0.2.0"',
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (workspace_root / ".rp-notebook-bootstrap.sha256").write_text(f"{stale_stamp}\n", encoding="utf-8")

                result = self._run_notebook_bootstrap_command(workspace_root)

                self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
                self.assertEqual((workspace_root / ".bootstrap-count").read_text(encoding="utf-8").strip(), "2")
                self.assertEqual(
                    (workspace_root / ".rp-notebook-bootstrap.sha256").read_text(encoding="utf-8").strip(),
                    _compute_notebook_bootstrap_stamp(workspace_root),
                )

    def test_repo_bootstrap_requirements_include_snakemake(self) -> None:
        requirements_path = WORKSPACE_ROOT / "ops" / "envs" / "dev" / "requirements-notebook.txt"
        requirements = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertIn("snakemake", requirements)

    def test_hpc_runtime_bootstrap_requirements_include_snakemake(self) -> None:
        requirements_path = WORKSPACE_ROOT / "ops" / "envs" / "hpc" / "requirements-runtime.txt"
        requirements = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertIn("snakemake", requirements)
        self.assertIn("snakemake-executor-plugin-slurm", requirements)

    def test_hpc_notebook_bootstrap_command_skips_when_stamp_is_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_overlay_workspace(workspace_root)
            self._write_bootstrap_stamp_fixture(workspace_root)
            current_stamp = _compute_notebook_bootstrap_stamp(workspace_root)
            (workspace_root / ".rp-notebook-bootstrap.sha256").write_text(f"{current_stamp}\n", encoding="utf-8")

            result = self._run_notebook_bootstrap_command(workspace_root)
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("Bootstrap already current; skipping.", result.stdout)
            self.assertFalse((workspace_root / ".bootstrap-count").exists())

    def test_hpc_notebook_start_remote_launch_command_without_pre_activate_commands_keeps_existing_setup_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_overlay_workspace(workspace_root)

            command = _build_notebook_remote_launch_command(
                project_name="project-demo-notebook",
                project_context={
                    "workspace_root": workspace_root,
                    "remote_workspace_root": "/remote/workspace",
                    "notebook_path": workspace_root
                    / "project"
                    / "project-demo-notebook"
                    / "notebooks"
                    / "notebook_analysis.ipynb",
                },
                notebook_settings={
                    "cpus": 1,
                    "mem": "8G",
                    "time": "01:00:00",
                    "local_port": 8890,
                    "remote_port": 8888,
                    "modules": [],
                    "pre_activate_commands": [],
                    "prepare_directories": [],
                },
            )

        self.assertIn("bash ops/envs/dev/bootstrap.sh", command)
        self.assertIn("source .venv/bin/activate", command)
        self.assertLess(command.index("bash ops/envs/dev/bootstrap.sh"), command.index("source .venv/bin/activate"))
        self.assertNotIn("module load StdEnv/2023", command)

    def test_hpc_notebook_start_remote_launch_command_still_launches_notebook_after_bootstrap_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_overlay_workspace(workspace_root)

            command = _build_notebook_remote_launch_command(
                project_name="project-demo-notebook",
                project_context={
                    "workspace_root": workspace_root,
                    "remote_workspace_root": "/remote/workspace",
                    "notebook_path": workspace_root
                    / "project"
                    / "project-demo-notebook"
                    / "notebooks"
                    / "notebook_analysis.ipynb",
                },
                notebook_settings={
                    "cpus": 1,
                    "mem": "8G",
                    "time": "01:00:00",
                    "local_port": 8890,
                    "remote_port": 8888,
                    "modules": [],
                    "pre_activate_commands": [],
                    "prepare_directories": [],
                },
            )

        self.assertIn("Bootstrap already current; skipping.", command)
        self.assertIn("source .venv/bin/activate", command)
        self.assertIn("jupyter lab --no-browser --ip=127.0.0.1 --port=8888", command)
        self.assertLess(command.index("Bootstrap already current; skipping."), command.index("source .venv/bin/activate"))
        self.assertLess(command.index("source .venv/bin/activate"), command.index("jupyter lab --no-browser --ip=127.0.0.1 --port=8888"))

    def test_hpc_notebook_start_remote_launch_command_auto_remote_port_selects_on_compute_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_overlay_workspace(workspace_root)

            for tunnel_mode, expected_bind_ip in (("direct", "127.0.0.1"), ("login-forward", "0.0.0.0")):
                with self.subTest(tunnel_mode=tunnel_mode):
                    command = _build_notebook_remote_launch_command(
                        project_name="project-demo-notebook",
                        project_context={
                            "workspace_root": workspace_root,
                            "remote_workspace_root": "/remote/workspace",
                            "notebook_path": workspace_root
                            / "project"
                            / "project-demo-notebook"
                            / "notebooks"
                            / "notebook_analysis.ipynb",
                        },
                        notebook_settings={
                            "cpus": 1,
                            "mem": "8G",
                            "time": "01:00:00",
                            "local_port": 8890,
                            "remote_port": 0,
                            "modules": [],
                            "pre_activate_commands": [],
                            "prepare_directories": [],
                        },
                        tunnel_mode=tunnel_mode,
                    )

                    self.assertNotIn("--port=0", command)
                    self.assertIn("sock.bind((", command)
                    self.assertIn(expected_bind_ip, command)
                    self.assertIn("jupyter lab --no-browser --ip=", command)

    def test_hpc_notebook_start_parses_salloc_job_id_from_granted_allocation_line(self) -> None:
        self.assertEqual(_parse_salloc_job_id("Granted job allocation 12345\n"), "12345")
        self.assertEqual(_parse_salloc_job_id("salloc: Granted job allocation 67890\n"), "67890")
        self.assertIsNone(_parse_salloc_job_id("Pending job allocation 12345\n"))

    def test_hpc_notebook_start_builds_final_local_url_from_remote_url(self) -> None:
        local_url = _build_local_notebook_url(
            local_port=8890,
            remote_url="http://127.0.0.1:9031/lab/tree/demo.ipynb?token=abc123#cell-id",
        )

        self.assertEqual(local_url, "http://127.0.0.1:8890/lab/tree/demo.ipynb?token=abc123#cell-id")

    def test_hpc_notebook_url_proxy_script_auto_port_discovers_running_server_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            module_root = Path(tmp_dir)
            package_root = module_root / "jupyter_server"
            package_root.mkdir(parents=True, exist_ok=True)
            (package_root / "__init__.py").write_text("", encoding="utf-8")
            (package_root / "serverapp.py").write_text(
                "\n".join(
                    [
                        "from __future__ import annotations",
                        "",
                        "import os",
                        "",
                        "def list_running_servers():",
                        "    return [{'url': 'http://127.0.0.1:9031/', 'token': 'abc123', 'root_dir': os.getcwd()}]",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            launch_command = (
                f"{shlex.quote(sys.executable)} -c "
                + shlex.quote("import sys; sys.stdout.write('http://127.0.0.1:0/lab?token=abc123\\n')")
            )
            script = _build_notebook_url_proxy_script(launch_command, requested_port=0)
            env = dict(os.environ)
            pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(module_root) if not pythonpath else f"{module_root}{os.pathsep}{pythonpath}"
            )

            result = subprocess.run(
                [sys.executable, "-c", script],
                check=False,
                capture_output=True,
                text=True,
                cwd=tmp_dir,
                env=env,
            )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("http://127.0.0.1:0/lab?token=abc123", result.stdout)
        self.assertIn("RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123", result.stdout)
        self.assertNotIn("RP_NOTEBOOK_URL=http://127.0.0.1:0/lab?token=abc123", result.stdout)

    def test_hpc_notebook_url_proxy_script_launch_target_auto_port_emits_nonzero_url_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            self._write_fake_jupyter_executable(temp_root)
            self._write_fake_socket_module(temp_root, selected_port=9031)
            script = _build_notebook_url_proxy_script(
                "",
                requested_port=0,
                launch_target="demo.ipynb",
            )
            env = dict(os.environ)
            path = env.get("PATH")
            env["PATH"] = str(temp_root) if not path else f"{temp_root}{os.pathsep}{path}"
            pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(temp_root) if not pythonpath else f"{temp_root}{os.pathsep}{pythonpath}"
            )

            result = subprocess.run(
                [sys.executable, "-c", script],
                check=False,
                capture_output=True,
                text=True,
                cwd=tmp_dir,
                env=env,
            )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        match = re.search(r"RP_NOTEBOOK_URL=http://127\.0\.0\.1:(\d+)/lab\?token=abc123", result.stdout)
        self.assertIsNotNone(match, msg=result.stdout + result.stderr)
        self.assertNotEqual(match.group(1), "0")
        self.assertNotIn("RP_NOTEBOOK_URL=http://127.0.0.1:0/lab?token=abc123", result.stdout)

    def test_hpc_notebook_url_proxy_script_explicit_remote_port_still_uses_logged_url(self) -> None:
        script = _build_notebook_url_proxy_script(
            "printf 'http://127.0.0.1:9031/lab?token=abc123\\n'",
            requested_port=9031,
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123", result.stdout)

    def test_hpc_notebook_url_proxy_script_launch_target_explicit_remote_port_still_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            self._write_fake_jupyter_executable(temp_root)
            script = _build_notebook_url_proxy_script(
                "",
                requested_port=9031,
                launch_target="demo.ipynb",
            )
            env = dict(os.environ)
            path = env.get("PATH")
            env["PATH"] = str(temp_root) if not path else f"{temp_root}{os.pathsep}{path}"

            result = subprocess.run(
                [sys.executable, "-c", script],
                check=False,
                capture_output=True,
                text=True,
                cwd=tmp_dir,
                env=env,
            )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("http://127.0.0.1:9031/lab?token=abc123", result.stdout)
        self.assertIn("RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123", result.stdout)

    def test_hpc_notebook_url_proxy_script_can_be_embedded_in_format_template(self) -> None:
        script = _build_notebook_url_proxy_script("printf 'http://127.0.0.1:9031/lab?token=abc123\\n'")
        notebook_command = f"{shlex.quote(sys.executable)} -u -c {shlex.quote(script)}"

        formatted_command = notebook_command.format(jobid="12345")

        self.assertEqual(formatted_command, notebook_command)
        self.assertIn("RP_NOTEBOOK_URL=", formatted_command)

    def test_hpc_notebook_allocation_proxy_script_emits_host_and_url_markers(self) -> None:
        notebook_script = _build_notebook_url_proxy_script("printf 'Notebook ready\\nhttp://127.0.0.1:9031/lab?token=abc123\\n'")
        allocation_script = _build_notebook_allocation_proxy_script(
            allocation_command="printf 'salloc: Granted job allocation 12345\\n'",
            hostname_command="printf 'compute001.cluster.example\\n'",
            notebook_command=f"{shlex.quote(sys.executable)} -u -c {shlex.quote(notebook_script)}",
        )

        result = subprocess.run(
            [sys.executable, "-c", allocation_script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("RP_NOTEBOOK_HOST=compute001.cluster.example", result.stdout)
        self.assertIn("RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123", result.stdout)

    def test_hpc_notebook_start_browser_opening_requires_open_browser_flag(self) -> None:
        profile = mock.Mock()
        profile.user = "alice"
        profile.name = "interactive-login"
        profile.role = "login"
        profile.port = None
        profile.identity_file = None
        profile.known_hosts_file = None
        profile.options = {}
        profile.target.return_value = "alice@cluster.example"
        notebook_settings = {"local_port": 9042, "remote_port": 0}

        for open_browser, expected_calls in ((False, 0), (True, 1)):
            with self.subTest(open_browser=open_browser):
                remote_process = _FakePopenProcess(
                    stdout_lines=[
                        "RP_NOTEBOOK_HOST=compute001.cluster.example\n",
                        "RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123\n",
                    ]
                )
                tunnel_process = _FakePopenProcess()
        with mock.patch(
            "research_platform.core.cli._build_notebook_remote_launch_command",
            return_value="ssh-remote-launch",
        ):
            with mock.patch(
                "research_platform.core.cli.subprocess.Popen",
                side_effect=[remote_process, tunnel_process],
            ) as popen_mock:
                with mock.patch(
                    "research_platform.core.cli._observe_tunnel_process_startup",
                    return_value="running",
                ):
                    with mock.patch("research_platform.core.cli.socket.create_connection") as connect_mock:
                        with mock.patch("research_platform.core.cli.webbrowser.open") as browser_open:
                            if open_browser:
                                connect_mock.return_value = mock.Mock(close=mock.Mock())
                            with redirect_stdout(io.StringIO()):
                                exit_code = _launch_remote_notebook_start(
                                    project_name="project-demo-notebook",
                                    project_context={},
                                    profile=profile,
                                    notebook_settings=notebook_settings,
                                    fallback_url_path=None,
                                    open_browser=open_browser,
                                )

                self.assertEqual(exit_code, 0)
                self.assertEqual(popen_mock.call_count, 2)
                self.assertIn("9042:127.0.0.1:9031", popen_mock.call_args_list[1].args[0])
                self.assertEqual(browser_open.call_count, expected_calls)
                if open_browser:
                    browser_open.assert_called_once_with("http://127.0.0.1:9042/lab?token=abc123")
                    connect_mock.assert_called_once_with(("127.0.0.1", 9042), timeout=0.1)
                else:
                    connect_mock.assert_not_called()

    def test_hpc_notebook_start_remote_launch_first_uses_normal_rendered_tunnel_command(self) -> None:
        profile = mock.Mock()
        profile.user = "alice"
        profile.name = "interactive-login"
        profile.role = "login"
        profile.port = None
        profile.identity_file = None
        profile.known_hosts_file = None
        profile.options = {}
        profile.target.return_value = "alice@cluster.example"
        notebook_settings = {"local_port": 9042, "remote_port": 0}
        remote_process = _FakePopenProcess(
            stdout_lines=[
                "RP_NOTEBOOK_HOST=compute001.cluster.example\n",
                "RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123\n",
            ]
        )
        tunnel_process = _FakePopenProcess()

        with mock.patch(
            "research_platform.core.cli._build_notebook_remote_launch_command",
            return_value="ssh-remote-launch",
        ):
            with mock.patch(
                "research_platform.core.cli.subprocess.Popen",
                side_effect=[remote_process, tunnel_process],
            ) as popen_mock:
                with mock.patch(
                    "research_platform.core.cli._observe_tunnel_process_startup",
                    return_value="running",
                ):
                    with redirect_stdout(io.StringIO()):
                        exit_code = _launch_remote_notebook_start(
                            project_name="project-demo-notebook",
                            project_context={},
                            profile=profile,
                            notebook_settings=notebook_settings,
                            fallback_url_path=None,
                            open_browser=False,
                            tunnel_mode="login-forward",
                        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(popen_mock.call_count, 2)
        tunnel_command = popen_mock.call_args_list[1].args[0]
        self.assertEqual(tunnel_command[:1], ["ssh"])
        self.assertNotIn("ControlMaster=no", tunnel_command)
        self.assertNotIn("ControlPath=none", tunnel_command)
        self.assertIn("9042:compute001.cluster.example:9031", tunnel_command)

    def test_hpc_notebook_start_remote_launch_retries_once_after_mux_session_refused(self) -> None:
        profile = mock.Mock()
        profile.user = "alice"
        profile.name = "interactive-login"
        profile.role = "login"
        profile.port = None
        profile.identity_file = None
        profile.known_hosts_file = None
        profile.options = {}
        profile.target.return_value = "alice@cluster.example"
        notebook_settings = {"local_port": 9042, "remote_port": 0}
        remote_process = _FakePopenProcess(
            stdout_lines=[
                "RP_NOTEBOOK_HOST=compute001.cluster.example\n",
                "RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123\n",
            ]
        )
        primary_tunnel_process = _FakePopenProcess(
            stderr_text="mux_client_request_session: session request failed: Session open refused by peer\n",
            returncode=255,
            finished=True,
        )
        retry_tunnel_process = _FakePopenProcess()

        with mock.patch(
            "research_platform.core.cli._build_notebook_remote_launch_command",
            return_value="ssh-remote-launch",
        ):
            with mock.patch(
                "research_platform.core.cli.subprocess.Popen",
                side_effect=[remote_process, primary_tunnel_process, retry_tunnel_process],
            ) as popen_mock:
                buffer = io.StringIO()
                with mock.patch(
                    "research_platform.core.cli._observe_tunnel_process_startup",
                    return_value="exited",
                ):
                    with redirect_stdout(buffer):
                        exit_code = _launch_remote_notebook_start(
                            project_name="project-demo-notebook",
                            project_context={},
                            profile=profile,
                            notebook_settings=notebook_settings,
                            fallback_url_path=None,
                            open_browser=False,
                            tunnel_mode="login-forward",
                        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(popen_mock.call_count, 3)
        primary_command = popen_mock.call_args_list[1].args[0]
        retry_command = popen_mock.call_args_list[2].args[0]
        self.assertNotIn("ControlMaster=no", primary_command)
        self.assertEqual(retry_command[:5], ["ssh", "-o", "ControlMaster=no", "-o", "ControlPath=none"])
        self.assertIn("9042:compute001.cluster.example:9031", primary_command)
        self.assertEqual(retry_command[5:], primary_command[1:])
        self.assertEqual(
            [line for line in buffer.getvalue().splitlines() if line.startswith("Tunnel command: ")],
            [
                f"Tunnel command: {_render_command(primary_command)}",
                f"Tunnel command: {_render_command(retry_command)}",
            ],
        )

    def test_hpc_notebook_start_browser_opening_waits_for_local_port(self) -> None:
        profile = mock.Mock()
        profile.user = "alice"
        profile.name = "interactive-login"
        profile.role = "login"
        profile.port = None
        profile.identity_file = None
        profile.known_hosts_file = None
        profile.options = {}
        profile.target.return_value = "alice@cluster.example"
        notebook_settings = {"local_port": 9042, "remote_port": 0}
        remote_process = _FakePopenProcess(
            stdout_lines=[
                "RP_NOTEBOOK_HOST=compute001.cluster.example\n",
                "RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123\n",
            ]
        )
        tunnel_process = _FakePopenProcess()
        events: list[tuple[str, object]] = []

        def _create_connection(address: tuple[str, int], timeout: float) -> mock.Mock:
            events.append(("connect", address))
            self.assertEqual(timeout, 0.1)
            if len([event for event, _ in events if event == "connect"]) == 1:
                raise OSError("not ready")
            return mock.Mock(close=mock.Mock())

        with mock.patch(
            "research_platform.core.cli._build_notebook_remote_launch_command",
            return_value="ssh-remote-launch",
        ):
            with mock.patch(
                "research_platform.core.cli.subprocess.Popen",
                side_effect=[remote_process, tunnel_process],
            ):
                with mock.patch(
                    "research_platform.core.cli._observe_tunnel_process_startup",
                    return_value="running",
                ):
                    with mock.patch("research_platform.core.cli.socket.create_connection", side_effect=_create_connection):
                        with mock.patch("research_platform.core.cli.time.sleep") as sleep_mock:
                            with mock.patch(
                                "research_platform.core.cli.webbrowser.open",
                                side_effect=lambda url: events.append(("browser", url)),
                            ):
                                with redirect_stdout(io.StringIO()):
                                    exit_code = _launch_remote_notebook_start(
                                        project_name="project-demo-notebook",
                                        project_context={},
                                        profile=profile,
                                        notebook_settings=notebook_settings,
                                        fallback_url_path=None,
                                        open_browser=True,
                                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            events,
            [
                ("connect", ("127.0.0.1", 9042)),
                ("connect", ("127.0.0.1", 9042)),
                ("browser", "http://127.0.0.1:9042/lab?token=abc123"),
            ],
        )
        sleep_mock.assert_called_once_with(0.1)

    def test_hpc_notebook_start_browser_opening_waits_for_local_port_after_mux_retry(self) -> None:
        profile = mock.Mock()
        profile.user = "alice"
        profile.name = "interactive-login"
        profile.role = "login"
        profile.port = None
        profile.identity_file = None
        profile.known_hosts_file = None
        profile.options = {}
        profile.target.return_value = "alice@cluster.example"
        notebook_settings = {"local_port": 9042, "remote_port": 0}
        remote_process = _FakePopenProcess(
            stdout_lines=[
                "RP_NOTEBOOK_HOST=compute001.cluster.example\n",
                "RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123\n",
            ]
        )
        primary_tunnel_process = _FakePopenProcess(
            stderr_text="mux_client_request_session: session request failed: Session open refused by peer\n",
            returncode=255,
            finished=True,
        )
        retry_tunnel_process = _FakePopenProcess()

        with mock.patch(
            "research_platform.core.cli._build_notebook_remote_launch_command",
            return_value="ssh-remote-launch",
        ):
            with mock.patch(
                "research_platform.core.cli.subprocess.Popen",
                side_effect=[remote_process, primary_tunnel_process, retry_tunnel_process],
            ):
                with mock.patch(
                    "research_platform.core.cli._observe_tunnel_process_startup",
                    return_value="exited",
                ):
                    with mock.patch(
                        "research_platform.core.cli._wait_for_local_port",
                        side_effect=lambda *, local_port, process, **_: process is retry_tunnel_process and local_port == 9042,
                    ) as wait_mock:
                        with mock.patch("research_platform.core.cli.webbrowser.open") as browser_open:
                            with redirect_stdout(io.StringIO()):
                                exit_code = _launch_remote_notebook_start(
                                    project_name="project-demo-notebook",
                                    project_context={},
                                    profile=profile,
                                    notebook_settings=notebook_settings,
                                    fallback_url_path=None,
                                    open_browser=True,
                                    tunnel_mode="login-forward",
                                )

        self.assertEqual(exit_code, 0)
        wait_mock.assert_called_once()
        self.assertIs(wait_mock.call_args.kwargs["process"], retry_tunnel_process)
        browser_open.assert_called_once_with("http://127.0.0.1:9042/lab?token=abc123")

    def test_hpc_notebook_start_remote_launch_uses_actual_emitted_remote_port_when_dynamic(self) -> None:
        profile = mock.Mock()
        profile.user = "alice"
        profile.name = "interactive-login"
        profile.role = "login"
        profile.port = None
        profile.identity_file = None
        profile.known_hosts_file = None
        profile.options = {}
        profile.target.return_value = "alice@cluster.example"
        notebook_settings = {"local_port": 9042, "remote_port": 0}
        remote_process = _FakePopenProcess(
            stdout_lines=[
                "RP_NOTEBOOK_HOST=compute001.cluster.example\n",
                "RP_NOTEBOOK_URL=http://127.0.0.1:9031/lab?token=abc123\n",
            ]
        )
        tunnel_process = _FakePopenProcess()

        with mock.patch(
            "research_platform.core.cli._build_notebook_remote_launch_command",
            return_value="ssh-remote-launch",
        ):
            with mock.patch(
                "research_platform.core.cli.subprocess.Popen",
                side_effect=[remote_process, tunnel_process],
            ) as popen_mock:
                with mock.patch(
                    "research_platform.core.cli._observe_tunnel_process_startup",
                    return_value="running",
                ):
                    with redirect_stdout(io.StringIO()):
                        exit_code = _launch_remote_notebook_start(
                            project_name="project-demo-notebook",
                            project_context={},
                            profile=profile,
                            notebook_settings=notebook_settings,
                            fallback_url_path=None,
                            open_browser=False,
                        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(popen_mock.call_count, 2)
        self.assertIn("9042:127.0.0.1:9031", popen_mock.call_args_list[1].args[0])
        self.assertNotIn("9042:127.0.0.1:0", popen_mock.call_args_list[1].args[0])

    def test_hpc_notebook_start_manual_compute_host_mode_still_renders_without_remote_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._launch_remote_notebook_start", return_value=0) as launch_mock:
                with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "start",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--compute-host",
                            "compute001",
                            "--token",
                            "abc123",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        self.assertEqual(exit_code, 0)
        launch_mock.assert_not_called()
        self.assertIn("--compute-host alice@compute001 --local-port 9042 --remote-port 8888", output)
        self.assertIn("Local URL: http://127.0.0.1:9042/lab?token=abc123", output)

    def test_hpc_notebook_start_honors_explicit_direct_tunnel_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._launch_remote_notebook_start", return_value=0) as launch_mock:
                with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "start",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--compute-host",
                            "compute001",
                            "--token",
                            "abc123",
                            "--tunnel-mode",
                            "direct",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        self.assertEqual(exit_code, 0)
        launch_mock.assert_not_called()
        self.assertIn("jupyter lab --no-browser --ip=127.0.0.1 --port=8888", output)
        self.assertIn("--compute-host alice@compute001 --local-port 9042 --remote-port 8888", output)
        self.assertNotIn("--tunnel-mode login-forward", output)
        self.assertIn("Local URL: http://127.0.0.1:9042/lab?token=abc123", output)

    def test_hpc_notebook_start_login_forward_carries_mode_into_rendered_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._launch_remote_notebook_start", return_value=0) as launch_mock:
                with mock.patch("research_platform.core.cli._select_available_local_port", return_value=9042):
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "start",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--compute-host",
                            "compute001",
                            "--token",
                            "abc123",
                            "--tunnel-mode",
                            "login-forward",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        self.assertEqual(exit_code, 0)
        launch_mock.assert_not_called()
        self.assertIn("jupyter lab --no-browser --ip=0.0.0.0 --port=8888", output)
        self.assertIn("rp hpc tunnel --profile interactive-login --role login", output)
        self.assertIn(f"--config {ssh_config.resolve()}", output)
        self.assertIn("--compute-host alice@compute001 --local-port 9042 --remote-port 8888", output)
        self.assertIn("--tunnel-mode login-forward", output)
        self.assertIn("Local URL: http://127.0.0.1:9042/lab?token=abc123", output)

    def test_hpc_notebook_submit_dry_run_prints_planned_paths_and_submit_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                with mock.patch("research_platform.core.cli.subprocess.run") as run_mock:
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "submit",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--dry-run",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        local_run_dir = (artifact_root / "notebook-runs" / "project-demo-notebook" / "20260322T120000Z").resolve()
        self.assertEqual(exit_code, 0)
        run_mock.assert_not_called()
        self.assertIn(f"Local script path: {local_run_dir / 'submit.sbatch'}", output)
        self.assertIn(
            "Remote run directory: remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z",
            output,
        )
        self.assertIn(
            "Output notebook path: remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z/notebook_analysis.executed.ipynb",
            output,
        )
        self.assertIn("SSH submit command: ssh", output)
        self.assertIn(f"< {local_run_dir / 'submit.sbatch'}", output)

    def test_hpc_notebook_submit_defaults_to_plan_without_remote_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("notebook submission planning must not invoke SSH"),
                ) as run_mock:
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "submit",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        self.assertEqual(exit_code, 0)
        run_mock.assert_not_called()
        self.assertIn("Local run directory:", output)
        self.assertIn("Local script path:", output)
        self.assertIn("SSH submit command: ssh", output)

    def test_hpc_notebook_submit_dry_run_writes_valid_submit_script_at_byte_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                slurm_template_text=(
                    "\ufeff\r\n#!/usr/bin/env bash\r\n"
                    "#SBATCH --job-name={{ job_name }}\r\n"
                    "#SBATCH --cpus-per-task={{ cpus }}\r\n"
                    "#SBATCH --mem={{ mem }}\r\n"
                    "#SBATCH --time={{ time }}\r\n"
                    "#SBATCH --output={{ log_out }}\r\n"
                    "#SBATCH --error={{ log_err }}\r\n"
                    "\r\n"
                    "set -euo pipefail\r\n"
                    "\r\n"
                    "{{ command }}\r\n"
                ),
            )
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                exit_code, _ = self._run_cli(
                    [
                        "hpc",
                        "notebook",
                        "submit",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--dry-run",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

            submit_script = artifact_root / "notebook-runs" / "project-demo-notebook" / "20260322T120000Z" / "submit.sbatch"
            script_bytes = submit_script.read_bytes()
            script_text = submit_script.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertTrue(script_bytes.startswith(b"#!"))
        self.assertEqual(script_text.splitlines()[0], "#!/usr/bin/env bash")
        self.assertNotEqual(script_text[:1], "\n")
        self.assertNotIn("\r", script_text)
        self.assertIn("set -euo pipefail", script_text)

    def test_hpc_notebook_submit_writes_modules_and_pre_activate_commands_before_bootstrap_and_virtualenv_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                notebook_hpc={
                    "modules": ["StdEnv/2023", "python/3.11"],
                    "pre_activate_commands": [
                        "module load arrow/23.0.1",
                        "export RP_CLUSTER=1",
                    ],
                },
            )
            write_yaml(
                workspace_root / "project" / "project-demo-notebook" / "config" / "compute.yaml",
                {
                    "compute": {
                        "default_profile": "slurm",
                        "slurm": {
                            "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                            "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                            "environment": {
                                "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                                "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
                            },
                            "prepare_directories": [
                                "$APPTAINER_CACHEDIR",
                                "$APPTAINER_TMPDIR",
                            ],
                        },
                    }
                },
            )
            write_yaml(
                workspace_root / "project" / "project-demo-notebook" / "config" / "dataset.yaml",
                {"dataset": {"primary": "ds-private-example"}},
            )
            write_yaml(
                workspace_root / "project" / "project-demo-notebook" / "config" / "preprocessing.yaml",
                {"preprocessing": {"slice": "tabular"}},
            )
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                exit_code, _ = self._run_cli(
                    [
                        "hpc",
                        "notebook",
                        "submit",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--dry-run",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

            submit_script = artifact_root / "notebook-runs" / "project-demo-notebook" / "20260322T120000Z" / "submit.sbatch"
            script_text = submit_script.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("module load StdEnv/2023 python/3.11", script_text)
        self.assertIn("module load arrow/23.0.1", script_text)
        self.assertIn('export APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache"', script_text)
        self.assertIn('export APPTAINER_TMPDIR="$SCRATCH/apptainer-tmp"', script_text)
        self.assertIn("export RP_CLUSTER=1", script_text)
        self.assertIn('mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"', script_text)
        self.assertIn("bash ops/envs/dev/bootstrap.sh", script_text)
        self.assertIn("source .venv/bin/activate", script_text)
        self.assertLess(
            script_text.index("module load StdEnv/2023 python/3.11"),
            script_text.index("bash ops/envs/dev/bootstrap.sh"),
        )
        self.assertLess(script_text.index("module load arrow/23.0.1"), script_text.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(script_text.index("export RP_CLUSTER=1"), script_text.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(script_text.index("export RP_CLUSTER=1"), script_text.index('mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"'))
        self.assertLess(script_text.index('mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"'), script_text.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(script_text.index("bash ops/envs/dev/bootstrap.sh"), script_text.index("source .venv/bin/activate"))

    def test_hpc_notebook_submit_uses_default_project_notebook_when_notebook_flag_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "notebook",
                        "submit",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--dry-run",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

            submit_script = artifact_root / "notebook-runs" / "project-demo-notebook" / "20260322T120000Z" / "submit.sbatch"
            script_text = submit_script.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("notebook_analysis.executed.ipynb", output)
        self.assertIn(
            "jupyter nbconvert --to notebook --execute remote/workspace/project/project-demo-notebook/notebooks/notebook_analysis.ipynb",
            script_text,
        )
        self.assertIn(
            "--output notebook_analysis.executed.ipynb --output-dir remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z",
            script_text,
        )

    def test_hpc_notebook_submit_applies_workspace_runtime_defaults_for_overlay_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                runtime_slurm={
                    "environment": {
                        "JUPYTER_CONFIG_DIR": "$SCRATCH/jupyter-config",
                        "JUPYTER_DATA_DIR": "$SCRATCH/jupyter-data",
                        "JUPYTER_RUNTIME_DIR": "$SCRATCH/jupyter-runtime",
                        "IPYTHONDIR": "$SCRATCH/ipython",
                    },
                    "prepare_directories": [
                        "$JUPYTER_CONFIG_DIR",
                        "$JUPYTER_DATA_DIR",
                        "$JUPYTER_RUNTIME_DIR",
                        "$IPYTHONDIR",
                    ],
                },
            )
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                exit_code, _ = self._run_cli(
                    [
                        "hpc",
                        "notebook",
                        "submit",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--dry-run",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

            submit_script = artifact_root / "notebook-runs" / "project-demo-notebook" / "20260322T120000Z" / "submit.sbatch"
            script_text = submit_script.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn('export JUPYTER_CONFIG_DIR="$SCRATCH/jupyter-config"', script_text)
        self.assertIn('export JUPYTER_DATA_DIR="$SCRATCH/jupyter-data"', script_text)
        self.assertIn('export JUPYTER_RUNTIME_DIR="$SCRATCH/jupyter-runtime"', script_text)
        self.assertIn('export IPYTHONDIR="$SCRATCH/ipython"', script_text)
        self.assertIn(
            'mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"',
            script_text,
        )

    def test_hpc_notebook_submit_keeps_overlay_workspace_relative_notebook_path_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root, create_notebook=False)
            ssh_config = self._write_ssh_config(workspace_root)
            with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                exit_code, _ = self._run_cli(
                    [
                        "hpc",
                        "notebook",
                        "submit",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--dry-run",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

            submit_script = artifact_root / "notebook-runs" / "project-demo-notebook" / "20260322T120000Z" / "submit.sbatch"
            script_text = submit_script.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "jupyter nbconvert --to notebook --execute remote/workspace/project/project-demo-notebook/notebooks/notebook_analysis.ipynb",
            script_text,
        )
        self.assertNotIn(
            "remote/workspace/project/project-demo-notebook/project/project-demo-notebook/notebooks/notebook_analysis.ipynb",
            script_text,
        )

    def test_hpc_notebook_submit_dry_run_resolves_workspace_from_repo_subdirectory(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            os.chdir(workspace_root / "project" / "project-demo-notebook" / "notebooks")
            try:
                with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "submit",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--dry-run",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                        extra_env={"RESEARCH_PLATFORM_ROOT": None},
                    )
            finally:
                os.chdir(original_cwd)

            submit_script = artifact_root / "notebook-runs" / "project-demo-notebook" / "20260322T120000Z" / "submit.sbatch"
            script_text = submit_script.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn(f"Local script path: {submit_script.resolve()}", output)
        self.assertIn(
            "Remote run directory: remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z",
            output,
        )
        self.assertIn(
            "jupyter nbconvert --to notebook --execute remote/workspace/project/project-demo-notebook/notebooks/notebook_analysis.ipynb",
            script_text,
        )

    def test_hpc_notebook_submit_parser_keeps_existing_flags(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(
            [
                "hpc",
                "notebook",
                "submit",
                "--project",
                "project-demo-notebook",
                "--profile",
                "interactive-login",
                "--config",
                "ssh-profiles.yaml",
                "--notebook",
                "project/project-demo-notebook/notebooks/custom.ipynb",
                "--job-name",
                "custom-job",
                "--cpus",
                "4",
                "--mem",
                "16G",
                "--time",
                "04:00:00",
                "--output-notebook",
                "custom.executed.ipynb",
                "--dry-run",
            ]
        )

        self.assertEqual(args.command, "hpc")
        self.assertEqual(args.hpc_command, "notebook")
        self.assertEqual(args.hpc_notebook_command, "submit")
        self.assertEqual(args.project, "project-demo-notebook")
        self.assertEqual(args.profile, "interactive-login")
        self.assertEqual(args.config, "ssh-profiles.yaml")
        self.assertEqual(args.notebook, "project/project-demo-notebook/notebooks/custom.ipynb")
        self.assertEqual(args.job_name, "custom-job")
        self.assertEqual(args.cpus, 4)
        self.assertEqual(args.mem, "16G")
        self.assertEqual(args.time, "04:00:00")
        self.assertEqual(args.output_notebook, "custom.executed.ipynb")
        self.assertTrue(args.dry_run)

    def test_hpc_notebook_submit_streams_script_over_ssh_and_reports_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)
            ssh_config = self._write_ssh_config(workspace_root)
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout="Submitted batch job 424242\n",
                stderr="",
            )
            with mock.patch("research_platform.core.cli._notebook_submit_timestamp", return_value="20260322T120000Z"):
                with mock.patch("research_platform.core.cli.subprocess.run", return_value=completed) as run_mock:
                    exit_code, output = self._run_cli(
                        [
                            "hpc",
                            "notebook",
                            "submit",
                            "--project",
                            "project-demo-notebook",
                            "--profile",
                            "interactive-login",
                            "--config",
                            str(ssh_config),
                            "--execute",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once()
        submit_command = run_mock.call_args.args[0]
        self.assertEqual(submit_command[0], "ssh")
        self.assertIn("alice@cluster.example", submit_command)
        self.assertEqual(
            submit_command[-1],
            "bash -lc 'set -euo pipefail; mkdir -p "
            "remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z; "
            "cat > remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z/submit.sbatch; "
            "sbatch "
            "remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z/submit.sbatch'",
        )
        self.assertEqual(run_mock.call_args.kwargs["text"], True)
        self.assertEqual(run_mock.call_args.kwargs["capture_output"], True)
        self.assertIn("sbatch remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z/submit.sbatch", submit_command[-1])
        self.assertTrue(run_mock.call_args.kwargs["input"].startswith("#!"))
        self.assertIn("#SBATCH --job-name=rp-notebook-project-demo-notebook", run_mock.call_args.kwargs["input"])
        self.assertIn("source .venv/bin/activate", run_mock.call_args.kwargs["input"])
        self.assertIn("Job id: 424242", output)
        self.assertIn(
            f"Local run directory: {(artifact_root / 'notebook-runs' / 'project-demo-notebook' / '20260322T120000Z').resolve()}",
            output,
        )
        self.assertIn(
            "Remote run directory: remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z",
            output,
        )
        self.assertIn(
            "Expected executed notebook path: remote/workspace/artifacts/notebook-runs/project-demo-notebook/20260322T120000Z/notebook_analysis.executed.ipynb",
            output,
        )

    def test_hpc_verify_data_reports_present_remote_roots_and_row_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            func_path = derivative_root / "sub-002" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            dataset_root.mkdir(parents=True, exist_ok=True)
            local_file = func_path / "sub-002_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            local_file.write_text("", encoding="utf-8")
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/deepprep-bold",
            )
            (
                workspace_root / "project" / "project-external-bids" / "manifests" / "batches" / "default.tsv"
            ).write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-002\tses-01\ttask-memory\trun-01\n",
                encoding="utf-8",
            )
            ssh_config = self._write_role_aware_ssh_config(workspace_root)
            remote_file = (
                "/remote/studies/project-external-bids/derivatives/deepprep-bold/"
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout=(
                    "present\t/remote/studies/project-external-bids\n"
                    "present\t/remote/studies/project-external-bids/derivatives/deepprep-bold\n"
                    f"present\t{remote_file}\n"
                ),
                stderr="",
            )
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed) as remote_run_mock:
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "verify",
                        "data",
                        "--project",
                        "project-external-bids",
                        "--profile",
                        "cli-profile",
                        "--role",
                        "robot",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        remote_run_mock.assert_called_once()
        verify_command = remote_run_mock.call_args.args[0]
        self.assertEqual(verify_command[0], "ssh")
        self.assertNotIn("rsync", verify_command)
        self.assertIn("Remote dataset root: present (/remote/studies/project-external-bids)", output)
        self.assertIn(
            "Remote derivative root: present (/remote/studies/project-external-bids/derivatives/deepprep-bold)",
            output,
        )
        self.assertIn("Rows checked: 1", output)
        self.assertIn("Rows with all expected remote files present: 1", output)
        self.assertIn("Row source: batch manifest", output)
        self.assertIn("SSH profile: cli-profile (robot)", output)
        self.assertIn(f"SSH config: {ssh_config.resolve()}", output)

    def test_hpc_verify_data_runs_layer_a_for_adapterless_generic_remote_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(
                workspace_root,
                hpc_data_roots=[
                    {
                        "local_path": "datasets/ds-private-example",
                        "remote_root": "/remote/private-data",
                    }
                ],
            )
            ssh_config = self._write_ssh_config(workspace_root)
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout="present\t/remote/private-data\n",
                stderr="",
            )
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "verify",
                        "data",
                        "--project",
                        "project-demo-notebook",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Remote root (private-data-root remote root): present (/remote/private-data)", output)
        self.assertIn("SSH profile: interactive-login (login)", output)
        self.assertIn("Row-level verification: skipped because this project is not using a BIDS tool adapter", output)

    def test_hpc_verify_data_runs_layer_a_for_structured_tabular_project_without_canonical_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_structured_tabular_workspace(
                workspace_root,
                project_name="project-tabular-verify",
                hpc_data_roots=[
                    {
                        "label": "tabular-input-root",
                        "local_path": "datasets/ds-tabular-example",
                        "remote_root": "/remote/tabular-inputs",
                    }
                ],
                include_canonical_fields=False,
            )
            ssh_config = self._write_ssh_config(workspace_root)
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout="present\t/remote/tabular-inputs\n",
                stderr="",
            )
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "verify",
                        "data",
                        "--project",
                        "project-tabular-verify",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Remote root (dataset-root remote root): present (/remote/tabular-inputs)", output)
        self.assertIn(
            "Project validation issue not blocking Layer A: config/dataset.yaml must define dataset.canonical_dataset for the tabular slice.",
            output,
        )
        self.assertIn(
            "Project validation issue not blocking Layer A: config/dataset.yaml must define dataset.canonical_features_root for the tabular slice.",
            output,
        )
        self.assertIn("Row-level verification: skipped because this project is not using a BIDS tool adapter", output)

    def test_hpc_verify_data_reports_missing_row_files_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            func_path = derivative_root / "sub-002" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            dataset_root.mkdir(parents=True, exist_ok=True)
            (func_path / "sub-002_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text(
                "",
                encoding="utf-8",
            )
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/deepprep-bold",
            )
            (
                workspace_root / "project" / "project-external-bids" / "manifests" / "batches" / "default.tsv"
            ).write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-002\tses-01\ttask-memory\trun-01\n",
                encoding="utf-8",
            )
            ssh_config = self._write_ssh_config(workspace_root)
            remote_file = (
                "/remote/studies/project-external-bids/derivatives/deepprep-bold/"
                "sub-002/ses-01/func/sub-002_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout=(
                    "present\t/remote/studies/project-external-bids\n"
                    "present\t/remote/studies/project-external-bids/derivatives/deepprep-bold\n"
                    f"missing\t{remote_file}\n"
                ),
                stderr="",
            )
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "verify",
                        "data",
                        "--project",
                        "project-external-bids",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("Rows with all expected remote files present: 0", output)
        self.assertIn("Missing paths", output)
        self.assertIn(remote_file, output)

    def test_hpc_verify_data_reports_transport_failures_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            func_path = derivative_root / "sub-002" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            dataset_root.mkdir(parents=True, exist_ok=True)
            (func_path / "sub-002_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text(
                "",
                encoding="utf-8",
            )
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/deepprep-bold",
            )
            (
                workspace_root / "project" / "project-external-bids" / "manifests" / "batches" / "default.tsv"
            ).write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-002\tses-01\ttask-memory\trun-01\n",
                encoding="utf-8",
            )
            ssh_config = self._write_ssh_config(workspace_root)
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=255,
                stdout="",
                stderr="ssh: connect to host cluster.example port 22: Operation timed out\n",
            )
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "verify",
                        "data",
                        "--project",
                        "project-external-bids",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 255)
        self.assertIn("Remote dataset root: not checked (/remote/studies/project-external-bids)", output)
        self.assertIn(
            "Remote derivative root: not checked (/remote/studies/project-external-bids/derivatives/deepprep-bold)",
            output,
        )
        self.assertIn("Rows with all expected remote files present: not determined (transport failure)", output)
        self.assertIn("Transport failure", output)
        self.assertIn("remote path probe exited with code 255", output)
        self.assertIn("ssh: connect to host cluster.example port 22: Operation timed out", output)
        self.assertNotIn("Missing paths", output)
        self.assertNotIn("Remote dataset root: missing", output)

    def test_hpc_verify_data_missing_batch_manifest_does_not_block_layer_a(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            func_path = derivative_root / "sub-004" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            dataset_root.mkdir(parents=True, exist_ok=True)
            (
                func_path / "sub-004_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/deepprep-bold",
            )
            (workspace_root / "project" / "project-external-bids" / "manifests" / "batches" / "default.tsv").unlink()
            ssh_config = self._write_ssh_config(workspace_root)
            remote_file = (
                "/remote/studies/project-external-bids/derivatives/deepprep-bold/"
                "sub-004/ses-01/func/sub-004_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout=(
                    "present\t/remote/studies/project-external-bids\n"
                    "present\t/remote/studies/project-external-bids/derivatives/deepprep-bold\n"
                    f"present\t{remote_file}\n"
                ),
                stderr="",
            )
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "verify",
                        "data",
                        "--project",
                        "project-external-bids",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--subject-id",
                        "sub-004",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Rows checked: 1", output)
        self.assertIn("Row source: adapter discovery", output)
        self.assertIn("Project validation issue not blocking Layer A: Missing default batch manifest", output)

    def test_hpc_verify_data_discovers_rows_when_batch_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            func_path = derivative_root / "sub-004" / "ses-01" / "func"
            func_path.mkdir(parents=True, exist_ok=True)
            dataset_root.mkdir(parents=True, exist_ok=True)
            (func_path / "sub-004_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz").write_text(
                "",
                encoding="utf-8",
            )
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/deepprep-bold",
            )
            ssh_config = self._write_ssh_config(workspace_root)
            remote_file = (
                "/remote/studies/project-external-bids/derivatives/deepprep-bold/"
                "sub-004/ses-01/func/sub-004_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout=(
                    "present\t/remote/studies/project-external-bids\n"
                    "present\t/remote/studies/project-external-bids/derivatives/deepprep-bold\n"
                    f"present\t{remote_file}\n"
                ),
                stderr="",
            )
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "verify",
                        "data",
                        "--project",
                        "project-external-bids",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                        "--subject-id",
                        "sub-004",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Rows checked: 1", output)
        self.assertIn("Row source: adapter discovery", output)

    def test_hpc_verify_data_skips_layer_b_when_adapter_inputs_are_unresolvable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            external_root = workspace_root.parent / f"{workspace_root.name}-external"
            dataset_root = external_root / "study"
            derivative_root = external_root / "study-derivatives" / "deepprep-bold"
            dataset_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            self._write_structured_bids_workspace(
                workspace_root,
                project_name="project-external-bids",
                dataset_root=dataset_root,
                derivative_root=derivative_root,
                remote_dataset_root="/remote/studies/project-external-bids",
                remote_derivative_root="/remote/studies/project-external-bids/derivatives/deepprep-bold",
            )
            (
                workspace_root / "project" / "project-external-bids" / "manifests" / "batches" / "default.tsv"
            ).write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-002\tses-01\ttask-memory\trun-01\n",
                encoding="utf-8",
            )
            ssh_config = self._write_ssh_config(workspace_root)
            completed = subprocess.CompletedProcess(
                args=["ssh"],
                returncode=0,
                stdout=(
                    "present\t/remote/studies/project-external-bids\n"
                    "present\t/remote/studies/project-external-bids/derivatives/deepprep-bold\n"
                ),
                stderr="",
            )
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed):
                exit_code, output = self._run_cli(
                    [
                        "hpc",
                        "verify",
                        "data",
                        "--project",
                        "project-external-bids",
                        "--profile",
                        "interactive-login",
                        "--config",
                        str(ssh_config),
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Rows checked: 0", output)
        self.assertIn(
            "Row-level verification: skipped because adapter-specific inputs could not be resolved from the available rows",
            output,
        )
        self.assertNotIn("Missing paths", output)

    def test_hpc_verify_data_skips_row_level_checks_for_adapterless_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_overlay_workspace(workspace_root)

            exit_code, output = self._run_cli(
                [
                    "hpc",
                    "verify",
                    "data",
                    "--project",
                    "project-demo-notebook",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Remote dataset root: not configured", output)
        self.assertIn("Remote derivative root: not configured", output)
        self.assertIn("Rows checked: 0", output)
        self.assertIn("Row-level verification: skipped because this project is not using a BIDS tool adapter", output)

    def test_hpc_verify_data_plan_uses_adapter_supplied_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "project"
            batch_path = project_root / "manifests" / "batches" / "default.tsv"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            batch_path.write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            fake_adapter = mock.Mock()
            fake_adapter.expected_remote_input_files.return_value = ["/remote/custom/sentinel-input.nii.gz"]
            context = {
                "project_root": project_root,
                "slice": "bids",
                "tool_adapter": fake_adapter,
                "input_derivative_root": Path(tmp_dir) / "derivatives",
                "remote_input_derivative_root": "/remote/derivatives",
                "preprocessing": {"default_batch": "default"},
                "data_roots": [],
            }

            plan = _build_hpc_data_verification_plan(
                project_name="project-sentinel",
                project_context=context,
                batch_name=None,
                selectors={"subject_id": None, "session_id": None, "task_id": None, "run_id": None},
            )

        self.assertEqual(plan["rows"][0]["expected_paths"], ["/remote/custom/sentinel-input.nii.gz"])
        fake_adapter.expected_remote_input_files.assert_called_once()

    def test_selected_data_sync_plan_skips_directory_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "project"
            batch_path = project_root / "manifests" / "batches" / "default.tsv"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            batch_path.write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            bids_root = Path(tmp_dir) / "bids"
            subject_dir = bids_root / "sub-001"
            file_path = subject_dir / "ses-01" / "anat" / "sub-001_ses-01_T1w.nii.gz"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("", encoding="utf-8")
            fake_adapter = mock.Mock()
            fake_adapter.expected_remote_input_files.return_value = [
                "/remote/bids/sub-001",
                "/remote/bids/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz",
            ]
            context = {
                "project_root": project_root,
                "slice": "bids",
                "tool_adapter": fake_adapter,
                "input_derivative_root": bids_root,
                "remote_input_derivative_root": "/remote/bids",
                "preprocessing": {"default_batch": "default"},
                "data_roots": [],
            }

            plan = _build_selected_data_sync_plan(
                project_name="project-sentinel",
                project_context=context,
                batch_name="default",
            )

        self.assertEqual(len(plan["entries"]), 1)
        self.assertEqual(plan["entries"][0]["source"], str(file_path.resolve()))
        self.assertEqual(
            plan["entries"][0]["destination"],
            "/remote/bids/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz",
        )

    def test_hpc_verify_data_plan_includes_preprocessing_auxiliary_files(self) -> None:
        class FakeAdapter:
            def expected_remote_input_files(self, **_kwargs: object) -> list[str]:
                return ["/remote/custom/sentinel-input.nii.gz"]

            def expected_remote_auxiliary_files(self, **_kwargs: object) -> list[str]:
                return ["/remote/licenses/freesurfer/license.txt"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "project"
            batch_path = project_root / "manifests" / "batches" / "default.tsv"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            batch_path.write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            context = {
                "project_root": project_root,
                "slice": "bids",
                "tool_adapter": FakeAdapter(),
                "input_derivative_root": Path(tmp_dir) / "bids",
                "remote_input_derivative_root": "/remote/bids",
                "preprocessing": {"default_batch": "default"},
                "data_roots": [],
            }

            plan = _build_hpc_data_verification_plan(
                project_name="project-sentinel",
                project_context=context,
                batch_name=None,
                selectors={"subject_id": None, "session_id": None, "task_id": None, "run_id": None},
            )

        self.assertIn("/remote/licenses/freesurfer/license.txt", plan["paths"])

    def test_hpc_verify_data_plan_filters_batch_by_subject_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "project"
            batch_path = project_root / "manifests" / "batches" / "default.tsv"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            batch_path.write_text(
                "\n".join(
                    [
                        "subject_id\tsession_id\ttask_id\trun_id",
                        "sub-007\tses-01\ttask-rest\trun-01",
                        "sub-009\tses-01\ttask-rest\trun-01",
                        "sub-010\tses-01\ttask-rest\trun-01",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            fake_adapter = mock.Mock()
            fake_adapter.expected_remote_input_files.side_effect = lambda **kwargs: [
                f"/remote/custom/{kwargs['row']['subject_id']}.nii.gz"
            ]
            context = {
                "project_root": project_root,
                "slice": "bids",
                "tool_adapter": fake_adapter,
                "input_derivative_root": Path(tmp_dir) / "derivatives",
                "remote_input_derivative_root": "/remote/derivatives",
                "preprocessing": {"default_batch": "default"},
                "data_roots": [],
            }

            plan = _build_hpc_data_verification_plan(
                project_name="project-sentinel",
                project_context=context,
                batch_name=None,
                selectors={"subject_id": ("007", "sub-009"), "session_id": None, "task_id": None, "run_id": None},
            )

        self.assertEqual([row["row"]["subject_id"] for row in plan["rows"]], ["sub-007", "sub-009"])
        self.assertEqual(fake_adapter.expected_remote_input_files.call_count, 2)

    def test_hpc_verify_data_plan_lists_analysis_external_remote_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "project"
            batch_path = project_root / "manifests" / "batches" / "default.tsv"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            batch_path.write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            fake_adapter = mock.Mock()
            fake_adapter.expected_remote_input_files.return_value = [
                "/remote/analysis/events/sub-001/ses-01/func/sub-001_ses-01_task-rest_run-01_desc-condition_a_events.txt"
            ]
            context = {
                "project_root": project_root,
                "slice": "bids",
                "tool_adapter": fake_adapter,
                "input_derivative_root": Path(tmp_dir) / "derivatives",
                "remote_input_derivative_root": "/remote/derivatives",
                "analysis": {"defaults": {"stage": "first_level"}},
                "analysis_stage": {"default_batch": "default"},
                "data_roots": [
                    {
                        "label": "evs-root",
                        "path": Path(tmp_dir) / "events",
                        "remote_root": "/remote/analysis/events",
                    }
                ],
            }

            plan = _build_hpc_data_verification_plan(
                project_name="project-analysis-roots",
                project_context=context,
                batch_name=None,
                selectors={"subject_id": None, "session_id": None, "task_id": None, "run_id": None},
            )

        self.assertIn(
            {"label": "evs-root remote root", "path": "/remote/analysis/events"},
            plan["remote_roots"],
        )


if __name__ == "__main__":
    unittest.main()
