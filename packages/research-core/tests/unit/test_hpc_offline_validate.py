from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
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
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))

from research_platform.core.cli import _build_parser, main
from research_platform.core import config as core_config
from research_platform.core.config import (
    apply_hpc_target_defaults,
    load_local_hpc_env_defaults,
    load_yaml,
    write_yaml,
)
from research_platform.hpc.offline_validation import (
    _resolve_target_environment_like_production,
    resolve_local_hpc_environment_defaults,
)


class HpcOfflineValidateCliTests(unittest.TestCase):
    def _write_workspace(
        self,
        root: Path,
        *,
        placeholder_host: bool = False,
        include_project: bool = False,
    ) -> None:
        (root / "secrets" / "hpc").mkdir(parents=True)
        write_yaml(
            root / "WORKSPACE.yaml",
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
                "projects": {"default": "project-demo"},
            },
        )
        write_yaml(
            root / "secrets" / "hpc" / "targets.yaml",
            {
                "version": 1,
                "default": "other",
                "targets": {
                    "other": {
                        "ssh_profile": "other",
                        "role": "login",
                        "ssh_config": "secrets/hpc/ssh-profiles.yaml",
                        "env": {"RP_REMOTE_WORKSPACE_ROOT": "/remote/other"},
                        "promotion": {"mode": "atomic_no_replace"},
                    },
                    "synthetic": {
                        "ssh_profile": "synthetic",
                        "role": "login",
                        "ssh_config": "secrets/hpc/ssh-profiles.yaml",
                        "env": {
                            "RP_REMOTE_WORKSPACE_ROOT": "/remote/research-platform",
                            "RP_REMOTE_ARTIFACTS_ROOT": "/remote/research-platform/artifacts",
                        },
                        "promotion": {"mode": "atomic_no_replace"},
                        "projects": (
                            {
                                "project-demo": {
                                    "env": {
                                        "DEMO_REMOTE_DATA_ROOT": "/remote/private-data",
                                    }
                                }
                            }
                            if include_project
                            else {}
                        ),
                    },
                },
            },
        )
        write_yaml(
            root / "secrets" / "hpc" / "ssh-profiles.yaml",
            {
                "profiles": {
                    "other": {
                        "host": "login.other.invalid",
                        "user": "other-user",
                    },
                    "synthetic": {
                        "host": (
                            "login.cluster.example.org"
                            if placeholder_host
                            else "login.synthetic.invalid"
                        ),
                        "user": "synthetic-user",
                    },
                }
            },
        )
        (root / "secrets" / ".env").write_text(
            "\n".join(
                (
                    "RESEARCH_HPC_TARGET=synthetic",
                    "RESEARCH_HPC_TARGETS_CONFIG=secrets/hpc/targets.yaml",
                    "RESEARCH_HPC_PROFILE=synthetic",
                    "RESEARCH_HPC_ROLE=login",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        if include_project:
            project_root = root / "project" / "project-demo"
            (project_root / "config").mkdir(parents=True)
            write_yaml(project_root / "project.yaml", {"name": "project-demo", "version": "0.1.0"})
            write_yaml(
                project_root / "config" / "compute.yaml",
                {
                    "compute": {
                        "default_profile": "local",
                        "slurm": {
                            "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                            "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                        },
                    }
                },
            )
            write_yaml(
                project_root / "config" / "dataset.yaml",
                {"dataset": {"primary": "synthetic-dataset"}},
            )

    def _tree_snapshot(self, root: Path) -> dict[str, bytes | str]:
        snapshot: dict[str, bytes | str] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                snapshot[relative] = f"link:{os.readlink(path)}"
            elif path.is_file():
                snapshot[relative] = path.read_bytes()
            elif path.is_dir():
                snapshot[relative] = "directory"
        return snapshot

    def test_validate_uses_local_defaults_without_mutating_environment_or_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root)
            before_tree = self._tree_snapshot(root)
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {
                    "RESEARCH_PLATFORM_ROOT": str(root),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                },
                clear=True,
            ):
                before_environment = dict(os.environ)
                with (
                    mock.patch("research_platform.core.cli.run_ssh_connectivity_check") as connectivity,
                    mock.patch("research_platform.core.cli.subprocess.run") as subprocess_run,
                    mock.patch("research_platform.core.cli.subprocess.Popen") as subprocess_popen,
                    mock.patch("research_platform.core.cli.socket.create_connection") as socket_connect,
                    mock.patch("research_platform.core.cli.socket.getaddrinfo") as dns_lookup,
                    mock.patch("research_platform.core.cli.socket.socket") as socket_constructor,
                    mock.patch.object(Path, "write_text", side_effect=AssertionError("unexpected write")),
                    redirect_stdout(output),
                ):
                    exit_code = main(["hpc", "validate", "--json"])
                self.assertEqual(dict(os.environ), before_environment)

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["configuration_valid"], payload["errors"])
            self.assertEqual(payload["target"], "synthetic")
            self.assertEqual(payload["profile"], "synthetic")
            self.assertTrue(payload["offline"])
            self.assertFalse(payload["network_contacted"])
            self.assertNotIn("login.synthetic.invalid", output.getvalue())
            self.assertNotIn("synthetic-user", output.getvalue())
            connectivity.assert_not_called()
            subprocess_run.assert_not_called()
            subprocess_popen.assert_not_called()
            socket_connect.assert_not_called()
            dns_lookup.assert_not_called()
            socket_constructor.assert_not_called()
            self.assertEqual(self._tree_snapshot(root), before_tree)

    def test_validate_reports_environment_cycle_without_traceback_or_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root)
            ssh_path = root / "secrets" / "hpc" / "ssh-profiles.yaml"
            ssh_document = load_yaml(ssh_path, resolve_env=False)
            ssh_document["profiles"]["synthetic"]["host"] = "${CYCLE_A}"
            write_yaml(ssh_path, ssh_document)
            before_tree = self._tree_snapshot(root)
            output = io.StringIO()
            error = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {
                    "RESEARCH_PLATFORM_ROOT": str(root),
                    "CYCLE_A": "${CYCLE_B:-${CYCLE_A}}",
                    "CYCLE_B": "${CYCLE_A}",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                },
                clear=True,
            ):
                before_environment = dict(os.environ)
                with redirect_stdout(output), redirect_stderr(error):
                    exit_code = main(["hpc", "validate", "--json"])
                self.assertEqual(dict(os.environ), before_environment)

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["configuration_valid"])
            self.assertTrue(
                any(
                    "Cyclic environment placeholder expansion" in item
                    for item in payload["errors"]
                ),
                payload["errors"],
            )
            self.assertNotIn("Traceback", error.getvalue())
            self.assertEqual(self._tree_snapshot(root), before_tree)

    def test_invalid_placeholder_returns_one_without_connectivity_or_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root, placeholder_host=True)
            output = io.StringIO()
            error = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                clear=True,
            ):
                with (
                    mock.patch("research_platform.core.cli.run_ssh_connectivity_check") as connectivity,
                    redirect_stdout(output),
                    redirect_stderr(error),
                ):
                    exit_code = main(["hpc", "validate", "--target", "synthetic", "--json"])

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["configuration_valid"])
            self.assertTrue(any("starter placeholder" in item for item in payload["errors"]))
            self.assertNotIn("Traceback", error.getvalue())
            connectivity.assert_not_called()

    def test_validate_rejects_unsafe_unmanaged_process_root_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root)
            output = io.StringIO()
            before_tree = self._tree_snapshot(root)
            process_environment = {
                "RESEARCH_PLATFORM_ROOT": str(root),
                "RP_REMOTE_WORKSPACE_ROOT": "relative/unsafe",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
            with mock.patch.dict(os.environ, process_environment, clear=True):
                before_environment = dict(os.environ)
                with (
                    mock.patch("research_platform.core.cli.run_ssh_connectivity_check") as connectivity,
                    redirect_stdout(output),
                ):
                    exit_code = main(
                        ["hpc", "validate", "--target", "synthetic", "--json"]
                    )
                self.assertEqual(dict(os.environ), before_environment)

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["configuration_valid"])
            self.assertTrue(
                any(
                    "effective environment.RP_REMOTE_WORKSPACE_ROOT must be "
                    "an absolute POSIX path"
                    in error
                    for error in payload["errors"]
                ),
                payload["errors"],
            )
            connectivity.assert_not_called()
            self.assertEqual(self._tree_snapshot(root), before_tree)

    def test_pure_offline_resolution_matches_production_target_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root)
            targets_path = root / "secrets" / "hpc" / "targets.yaml"
            targets = load_yaml(targets_path, resolve_env=False)
            target = targets["targets"]["synthetic"]
            target["env"] = {
                "RP_REMOTE_WORKSPACE_ROOT": "/target/workspace",
                "RP_REMOTE_ARTIFACTS_ROOT": (
                    "${RP_REMOTE_WORKSPACE_ROOT}/artifacts"
                ),
                "RP_REMOTE_CACHE_ROOT": "/target/cache",
            }
            target["slurm"] = {
                "environment": {
                    "RP_REMOTE_TMP_ROOT": (
                        "${RP_REMOTE_WORKSPACE_ROOT}/scheduler-tmp"
                    )
                }
            }
            target["projects"] = {
                "project-demo": {
                    "env": {
                        "RP_REMOTE_PROJECT_DATA_ROOT": (
                            "${RP_REMOTE_WORKSPACE_ROOT}/project-data"
                        )
                    }
                }
            }
            write_yaml(targets_path, targets)
            (root / "secrets" / ".env").write_text(
                "\n".join(
                    (
                        "RESEARCH_HPC_TARGET=synthetic",
                        "RP_REMOTE_WORKSPACE_ROOT=/local/workspace",
                        "RP_REMOTE_ARTIFACTS_ROOT=/local/artifacts",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            process_environment = {
                "RESEARCH_PLATFORM_ROOT": str(root),
                "RP_REMOTE_CACHE_ROOT": "/process/cache",
            }

            offline_environment, locally_managed = (
                resolve_local_hpc_environment_defaults(
                    workspace_root=root,
                    environment=process_environment,
                )
            )
            (
                _,
                selected_name,
                offline_target,
                effective_environment,
            ) = _resolve_target_environment_like_production(
                targets,
                target_name="synthetic",
                project_name="project-demo",
                environment=offline_environment,
                managed_defaults=locally_managed,
            )

            with (
                mock.patch.dict(os.environ, process_environment, clear=True),
                mock.patch.dict(
                    core_config._LOCAL_HPC_ENV_DEFAULT_VALUES,
                    {},
                    clear=True,
                ),
                mock.patch.dict(
                    core_config._HPC_TARGET_ENV_DEFAULT_VALUES,
                    {},
                    clear=True,
                ),
            ):
                load_local_hpc_env_defaults(root)
                apply_hpc_target_defaults(
                    project_name=None,
                    target_name="synthetic",
                    root=root,
                )
                production_target = apply_hpc_target_defaults(
                    project_name="project-demo",
                    target_name="synthetic",
                    root=root,
                )
                production_environment = {
                    name: os.environ.get(name)
                    for name in (
                        "RP_REMOTE_WORKSPACE_ROOT",
                        "RP_REMOTE_ARTIFACTS_ROOT",
                        "RP_REMOTE_CACHE_ROOT",
                        "RP_REMOTE_PROJECT_DATA_ROOT",
                    )
                }

        self.assertEqual(selected_name, "synthetic")
        self.assertIsNotNone(offline_target)
        self.assertIsNotNone(production_target)
        assert offline_target is not None
        assert production_target is not None
        self.assertEqual(
            {
                name: effective_environment.get(name)
                for name in production_environment
            },
            production_environment,
        )
        self.assertEqual(offline_target["slurm"], production_target["slurm"])

    def test_project_validation_is_read_only_and_does_not_plan_or_discover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root, include_project=True)
            before_tree = self._tree_snapshot(root)
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                clear=True,
            ):
                with (
                    mock.patch("research_platform.core.cli._plan_run") as plan_run,
                    mock.patch("research_platform.core.cli.build_project_hpc_context") as hpc_context,
                    redirect_stdout(output),
                ):
                    exit_code = main(
                        [
                            "hpc",
                            "validate",
                            "--target",
                            "synthetic",
                            "--project",
                            "project-demo",
                            "--json",
                        ]
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0, payload["errors"])
            self.assertEqual(payload["project"], "project-demo")
            plan_run.assert_not_called()
            hpc_context.assert_not_called()
            self.assertEqual(self._tree_snapshot(root), before_tree)

    def test_malformed_workspace_project_selection_fails_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root)
            (root / "WORKSPACE.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
            output = io.StringIO()
            error = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                clear=True,
            ):
                with redirect_stdout(output), redirect_stderr(error):
                    exit_code = main(
                        [
                            "hpc",
                            "validate",
                            "--target",
                            "synthetic",
                            "--project",
                            "project-demo",
                            "--json",
                        ]
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["configuration_valid"])
            self.assertTrue(
                any("could not be resolved from WORKSPACE.yaml" in item for item in payload["errors"])
            )
            self.assertNotIn("Traceback", error.getvalue())

    def test_malformed_local_defaults_fail_without_traceback_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root)
            (root / "secrets" / ".env").write_bytes(b"\xff")
            before_tree = self._tree_snapshot(root)
            output = io.StringIO()
            error = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                clear=True,
            ):
                with redirect_stdout(output), redirect_stderr(error):
                    exit_code = main(
                        [
                            "hpc",
                            "validate",
                            "--targets-config",
                            "secrets/hpc/targets.yaml",
                            "--target",
                            "synthetic",
                            "--json",
                        ]
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["configuration_valid"])
            self.assertTrue(
                any("Local HPC defaults could not be read safely" in item for item in payload["errors"])
            )
            self.assertNotIn("Traceback", error.getvalue())
            self.assertEqual(self._tree_snapshot(root), before_tree)

    def test_symlinked_targets_config_fails_closed_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root)
            real_targets = root / "secrets" / "hpc" / "targets.yaml"
            symlink_targets = root / "secrets" / "hpc" / "selected-targets.yaml"
            symlink_targets.symlink_to(real_targets)
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                clear=True,
            ):
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "hpc",
                            "validate",
                            "--targets-config",
                            str(symlink_targets),
                            "--target",
                            "synthetic",
                            "--json",
                        ]
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["configuration_valid"])
            self.assertTrue(
                any("not a symlink" in item for item in payload["errors"]),
                payload["errors"],
            )

    def test_project_remote_root_placeholders_fail_without_planning_or_data_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root, include_project=True)
            project_root = root / "project" / "project-demo"
            write_yaml(
                project_root / "project.yaml",
                {
                    "name": "project-demo",
                    "hpc": {
                        "data_roots": [
                            {
                                "local_path": "datasets/private-inputs",
                                "remote_root": "${PRIVATE_REMOTE_ROOT:-}",
                                "sync_enabled": False,
                            }
                        ]
                    },
                },
            )
            write_yaml(
                project_root / "config" / "analysis.yaml",
                {
                    "analysis": {
                        "external_input_roots": {
                            "private_inputs": {
                                "local_root": "${PRIVATE_INPUT_ROOT:-}",
                                "remote_root": "relative/remote/input",
                                "sync_enabled": False,
                            }
                        }
                    }
                },
            )
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                clear=True,
            ):
                with (
                    mock.patch("research_platform.core.cli._plan_run") as plan_run,
                    mock.patch("research_platform.core.cli.build_project_hpc_context") as hpc_context,
                    redirect_stdout(output),
                ):
                    exit_code = main(
                        [
                            "hpc",
                            "validate",
                            "--target",
                            "synthetic",
                            "--project",
                            "project-demo",
                            "--json",
                        ]
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["configuration_valid"])
            errors = "\n".join(payload["errors"])
            self.assertIn("project.yaml hpc.data_roots[0].remote_root", errors)
            self.assertIn(
                "config/analysis.yaml analysis.external_input_roots.private_inputs.local_root "
                "must resolve to a nonblank path",
                errors,
            )
            self.assertIn(
                "config/analysis.yaml analysis.external_input_roots.private_inputs.remote_root "
                "must be an absolute POSIX path",
                errors,
            )
            plan_run.assert_not_called()
            hpc_context.assert_not_called()

    def test_project_local_root_placeholders_fail_even_with_safe_remote_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root, include_project=True)
            project_root = root / "project" / "project-demo"
            write_yaml(
                project_root / "project.yaml",
                {
                    "name": "project-demo",
                    "hpc": {
                        "data_roots": [
                            {
                                "local_path": "${PRIVATE_DATA_ROOT:-}",
                                "remote_root": "/remote/private-data",
                            }
                        ]
                    },
                },
            )
            write_yaml(
                project_root / "config" / "analysis.yaml",
                {
                    "analysis": {
                        "external_input_roots": {
                            "private_inputs": {
                                "local_root": "${PRIVATE_INPUT_ROOT:-}",
                                "remote_root": "/remote/private-analysis",
                                "sync_enabled": False,
                            }
                        }
                    }
                },
            )
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                clear=True,
            ):
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "hpc",
                            "validate",
                            "--target",
                            "synthetic",
                            "--project",
                            "project-demo",
                            "--json",
                        ]
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            errors = "\n".join(payload["errors"])
            self.assertIn(
                "project.yaml hpc.data_roots[0].local_path must resolve to a nonblank path",
                errors,
            )
            self.assertIn(
                "config/analysis.yaml analysis.external_input_roots.private_inputs.local_root "
                "must resolve to a nonblank path",
                errors,
            )

    def test_project_read_only_input_aliases_may_share_one_remote_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_workspace(root, include_project=True)
            project_root = root / "project" / "project-demo"
            write_yaml(
                project_root / "config" / "analysis.yaml",
                {
                    "analysis": {
                        "external_input_roots": {
                            "evs": {
                                "local_root": "external/events",
                                "remote_root": "/remote/study/derivatives/inputs",
                                "sync_enabled": False,
                            },
                            "feat_confounds": {
                                "local_root": "external/confounds",
                                "remote_root": "/remote/study/derivatives/inputs",
                                "sync_enabled": False,
                            },
                        }
                    }
                },
            )
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
                clear=True,
            ):
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "hpc",
                            "validate",
                            "--target",
                            "synthetic",
                            "--project",
                            "project-demo",
                            "--json",
                        ]
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0, payload["errors"])
            self.assertTrue(payload["configuration_valid"])

    def test_validate_parser_has_no_remote_or_mutating_authorization_options(self) -> None:
        parser = _build_parser()
        for option in ("--execute", "--live", "--fix", "--force"):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                    parser.parse_args(["hpc", "validate", option])

        args = parser.parse_args(
            [
                "hpc",
                "validate",
                "--target",
                "synthetic",
                "--targets-config",
                "targets.yaml",
                "--project",
                "project-demo",
                "--profile",
                "synthetic",
                "--role",
                "login",
                "--ssh-config",
                "ssh.yaml",
                "--json",
            ]
        )
        self.assertEqual(args.target, "synthetic")
        self.assertEqual(args.targets_config, "targets.yaml")
        self.assertEqual(args.project, "project-demo")
        self.assertEqual(args.profile, "synthetic")
        self.assertEqual(args.role, "login")
        self.assertEqual(args.ssh_config, "ssh.yaml")
        self.assertTrue(args.json)


if __name__ == "__main__":
    unittest.main()
