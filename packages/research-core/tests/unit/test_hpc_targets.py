from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
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

from research_platform.core.cli import main
from research_platform.core.config import (
    load_yaml,
    load_local_hpc_env_defaults,
    load_project_bundle,
    resolve_bids_remote_dataset_root,
    resolve_hpc_target,
    write_yaml,
)
from research_platform.hpc.ssh_profiles import load_ssh_profile


class HpcTargetTests(unittest.TestCase):
    def test_generic_setup_rejects_missing_noninteractive_target_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with self.assertRaises(SystemExit) as exc_info:
                    main(
                        [
                            "hpc",
                            "setup",
                            "--user",
                            "synthetic-user",
                            "--host",
                            "login.synthetic.invalid",
                            "--remote-workspace-root",
                            "/remote/synthetic/workspace",
                        ]
                    )

            self.assertIn("--target", str(exc_info.exception))
            self.assertFalse((workspace_root / "secrets").exists())

    def test_generic_setup_requires_each_noninteractive_value_before_writing(self) -> None:
        required_arguments = {
            "--target": "synthetic-target",
            "--user": "synthetic-user",
            "--host": "login.synthetic.invalid",
            "--remote-workspace-root": "/remote/synthetic/workspace",
        }
        for missing_flag in required_arguments:
            with self.subTest(missing_flag=missing_flag), tempfile.TemporaryDirectory() as tmp_dir:
                workspace_root = Path(tmp_dir) / "workspace"
                write_yaml(
                    workspace_root / "WORKSPACE.yaml",
                    {
                        "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                        "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                        "projects": {"default": "project-demo"},
                    },
                )
                args = ["hpc", "setup"]
                for flag, value in required_arguments.items():
                    if flag != missing_flag:
                        args.extend((flag, value))
                with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                    with self.assertRaises(SystemExit) as exc_info:
                        main(args)

                self.assertIn(missing_flag, str(exc_info.exception))
                self.assertFalse((workspace_root / "secrets").exists())

    def test_generic_setup_rejects_unsafe_destinations_and_invalid_values_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            common = [
                "hpc",
                "setup",
                "--target",
                "synthetic-target",
                "--user",
                "synthetic-user",
                "--host",
                "login.synthetic.invalid",
                "--remote-workspace-root",
                "/remote/synthetic/workspace",
            ]
            invalid_cases = (
                ([*common, "--ssh-config", "outside.yaml"], "--ssh-config"),
                ([*common, "--targets-config", "outside.yaml"], "--targets-config"),
                ([*common, "--host", "example.invalid.example.org"], "placeholder"),
            )
            for args, expected in invalid_cases:
                with self.subTest(args=args), mock.patch.dict(
                    os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True
                ):
                    with self.assertRaises(SystemExit) as exc_info:
                        main(args)
                self.assertIn(expected, str(exc_info.exception))
                self.assertFalse((workspace_root / "secrets").exists())

            ssh_destination = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            ssh_destination.parent.mkdir(parents=True)
            ssh_destination.symlink_to(workspace_root / "outside-ssh-profiles.yaml")
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with self.assertRaises(SystemExit) as exc_info:
                    main(common)
            self.assertIn("symlink", str(exc_info.exception))
            self.assertFalse((workspace_root / "secrets" / "hpc" / "targets.yaml").exists())
            self.assertFalse((workspace_root / "secrets" / ".env").exists())

    def test_generic_setup_rejects_every_public_secrets_scaffold_destination(self) -> None:
        common = [
            "hpc",
            "setup",
            "--target",
            "synthetic-target",
            "--user",
            "synthetic-user",
            "--host",
            "login.synthetic.invalid",
            "--remote-workspace-root",
            "/remote/synthetic/workspace",
        ]
        for flag in ("--ssh-config", "--targets-config"):
            for relative in (
                "secrets/local/README.md",
                "secrets/local/readme.md",
                "secrets/local/ReadMe.MD",
                "secrets/local/.gitkeep",
                "secrets/local/.GITKEEP",
                "secrets/local/.GitKeep",
                "secrets/local/config.example",
                "secrets/local/config.EXAMPLE",
                "secrets/local/CONFIG.Example",
            ):
                with (
                    self.subTest(flag=flag, relative=relative),
                    tempfile.TemporaryDirectory() as tmp_dir,
                ):
                    workspace_root = Path(tmp_dir) / "workspace"
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
                            "projects": {"default": "project-demo"},
                        },
                    )
                    destination = workspace_root / relative
                    destination.parent.mkdir(parents=True)
                    sentinel = b"public scaffold sentinel\n"
                    destination.write_bytes(sentinel)
                    original_mode = destination.stat().st_mode

                    with mock.patch.dict(
                        os.environ,
                        {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
                        clear=True,
                    ):
                        with self.assertRaises(SystemExit) as exc_info:
                            main([*common, flag, relative])

                    self.assertIn("public secrets scaffold", str(exc_info.exception))
                    self.assertEqual(destination.read_bytes(), sentinel)
                    self.assertEqual(destination.stat().st_mode, original_mode)
                    self.assertFalse(
                        (workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml").exists()
                    )
                    self.assertFalse(
                        (workspace_root / "secrets" / "hpc" / "targets.yaml").exists()
                    )
                    self.assertFalse((workspace_root / "secrets" / ".env").exists())
                    self.assertFalse((workspace_root / "secrets" / "hpc").exists())

    def test_public_scaffold_exceptions_are_not_ignored_private_destinations(self) -> None:
        private = subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "-q",
                "secrets/hpc/ssh-profiles.yaml",
            ],
            cwd=WORKSPACE_ROOT,
            check=False,
        )
        self.assertEqual(private.returncode, 0)
        for path in (
            "secrets/local/README.md",
            "secrets/local/.gitkeep",
            "secrets/local/config.example",
        ):
            with self.subTest(path=path):
                public = subprocess.run(
                    ["git", "check-ignore", "--no-index", "-q", path],
                    cwd=WORKSPACE_ROOT,
                    check=False,
                )
                self.assertEqual(public.returncode, 1)

    def test_generic_setup_rejects_external_hard_link_without_modifying_any_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            workspace_root = root / "workspace"
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
                    "projects": {"default": "project-demo"},
                },
            )
            outside = root / "outside-private-config.yaml"
            sentinel = b"outside hard-link sentinel\n"
            outside.write_bytes(sentinel)
            ssh_path = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            ssh_path.parent.mkdir(parents=True)
            os.link(outside, ssh_path)

            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
                clear=True,
            ):
                with self.assertRaises(SystemExit) as exc_info:
                    main(
                        [
                            "hpc",
                            "setup",
                            "--target",
                            "synthetic-target",
                            "--user",
                            "synthetic-user",
                            "--host",
                            "login.synthetic.invalid",
                            "--remote-workspace-root",
                            "/remote/synthetic/workspace",
                        ]
                    )

            self.assertIn("hard-linked", str(exc_info.exception))
            self.assertEqual(outside.read_bytes(), sentinel)
            self.assertEqual(ssh_path.read_bytes(), sentinel)
            self.assertFalse((workspace_root / "secrets" / "hpc" / "targets.yaml").exists())
            self.assertFalse((workspace_root / "secrets" / ".env").exists())

    def test_generic_setup_rejects_broken_secrets_symlink_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            secrets_path = workspace_root / "secrets"
            secrets_path.symlink_to(workspace_root / "missing-private-directory")
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with self.assertRaises(SystemExit) as exc_info:
                    main(
                        [
                            "hpc",
                            "setup",
                            "--target",
                            "synthetic-target",
                            "--user",
                            "synthetic-user",
                            "--host",
                            "login.synthetic.invalid",
                            "--remote-workspace-root",
                            "/remote/synthetic/workspace",
                        ]
                    )

            self.assertIn("symlinked private configuration directory", str(exc_info.exception))
            self.assertTrue(secrets_path.is_symlink())
            self.assertFalse((workspace_root / "missing-private-directory").exists())

    def test_generic_setup_rejects_nonregular_config_destination_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            targets_path = workspace_root / "secrets" / "hpc" / "targets.yaml"
            targets_path.mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with self.assertRaises(SystemExit) as exc_info:
                    main(
                        [
                            "hpc",
                            "setup",
                            "--target",
                            "synthetic-target",
                            "--user",
                            "synthetic-user",
                            "--host",
                            "login.synthetic.invalid",
                            "--remote-workspace-root",
                            "/remote/synthetic/workspace",
                        ]
                    )

            self.assertIn("real regular file", str(exc_info.exception))
            self.assertTrue(targets_path.is_dir())
            self.assertFalse((workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml").exists())
            self.assertFalse((workspace_root / "secrets" / ".env").exists())

    def test_generic_setup_rejects_colliding_destinations_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            common = [
                "hpc",
                "setup",
                "--target",
                "synthetic-target",
                "--user",
                "synthetic-user",
                "--host",
                "login.synthetic.invalid",
                "--remote-workspace-root",
                "/remote/synthetic/workspace",
            ]
            for arguments in (
                [
                    *common,
                    "--ssh-config",
                    "secrets/hpc/shared.yaml",
                    "--targets-config",
                    "secrets/hpc/shared.yaml",
                ],
                [
                    *common,
                    "--ssh-config",
                    "secrets/.env",
                ],
            ):
                with self.subTest(arguments=arguments), mock.patch.dict(
                    os.environ,
                    {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
                    clear=True,
                ):
                    with self.assertRaises(SystemExit) as exc_info:
                        main(arguments)
                self.assertIn("distinct destinations", str(exc_info.exception))
                self.assertFalse((workspace_root / "secrets").exists())

    def test_generic_setup_rejects_global_defaults_and_nonlogin_role_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            ssh_path = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            ssh_path.parent.mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            write_yaml(
                ssh_path,
                {
                    "defaults": {
                        "identity_file": "/synthetic/provider-key",
                        "options": {"ControlMaster": "auto"},
                    },
                    "profiles": {
                        "unrelated": {
                            "host": "other.synthetic.invalid",
                            "user": "other",
                        }
                    },
                },
            )
            common = [
                "hpc",
                "setup",
                "--target",
                "synthetic-target",
                "--user",
                "synthetic-user",
                "--host",
                "login.synthetic.invalid",
                "--remote-workspace-root",
                "/remote/synthetic/workspace",
            ]
            original = ssh_path.read_bytes()
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
                clear=True,
            ):
                with self.assertRaises(SystemExit) as exc_info:
                    main(common)
            self.assertIn("would be inherited implicitly", str(exc_info.exception))
            self.assertEqual(ssh_path.read_bytes(), original)
            self.assertFalse((workspace_root / "secrets" / "hpc" / "targets.yaml").exists())
            self.assertFalse((workspace_root / "secrets" / ".env").exists())

            write_yaml(ssh_path, {"profiles": {}})
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
                clear=True,
            ):
                with self.assertRaises(SystemExit) as exc_info:
                    main([*common, "--role", "robot"])
            self.assertIn("login role only", str(exc_info.exception))
            self.assertEqual(load_yaml(ssh_path, resolve_env=False), {"profiles": {}})
            self.assertFalse((workspace_root / "secrets" / "hpc" / "targets.yaml").exists())
            self.assertFalse((workspace_root / "secrets" / ".env").exists())

    def test_generic_setup_validates_existing_local_defaults_before_other_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            (workspace_root / "secrets").mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            env_path = workspace_root / "secrets" / ".env"
            env_path.write_bytes(b"\xff")
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with self.assertRaises(SystemExit) as exc_info:
                    main(
                        [
                            "hpc",
                            "setup",
                            "--target",
                            "synthetic-target",
                            "--user",
                            "synthetic-user",
                            "--host",
                            "login.synthetic.invalid",
                            "--remote-workspace-root",
                            "/remote/synthetic/workspace",
                        ]
                    )

            self.assertIn("could not be read safely", str(exc_info.exception))
            self.assertEqual(env_path.read_bytes(), b"\xff")
            self.assertFalse((workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml").exists())
            self.assertFalse((workspace_root / "secrets" / "hpc" / "targets.yaml").exists())

    def test_generic_setup_permission_failure_preserves_existing_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            hpc_root = workspace_root / "secrets" / "hpc"
            hpc_root.mkdir(parents=True)
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
                    "projects": {"default": "project-demo"},
                },
            )
            ssh_path = hpc_root / "ssh-profiles.yaml"
            targets_path = hpc_root / "targets.yaml"
            env_path = workspace_root / "secrets" / ".env"
            write_yaml(ssh_path, {"profiles": {}})
            write_yaml(targets_path, {"version": 1, "targets": {}})
            env_path.write_text("SENTINEL=value\n", encoding="utf-8")
            for path in (ssh_path, targets_path, env_path):
                path.chmod(0o644)
            original = {
                path: path.read_bytes()
                for path in (ssh_path, targets_path, env_path)
            }

            with (
                mock.patch.dict(
                    os.environ,
                    {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
                    clear=True,
                ),
                mock.patch(
                    "research_platform.core.cli.os.fchmod",
                    side_effect=PermissionError("synthetic permission failure"),
                ),
            ):
                with self.assertRaises(SystemExit) as exc_info:
                    main(
                        [
                            "hpc",
                            "setup",
                            "--target",
                            "synthetic-target",
                            "--user",
                            "synthetic-user",
                            "--host",
                            "login.synthetic.invalid",
                            "--remote-workspace-root",
                            "/remote/synthetic/workspace",
                        ]
                    )

            self.assertIn("private permissions", str(exc_info.exception))
            for path, expected in original.items():
                self.assertEqual(path.read_bytes(), expected)
                self.assertEqual(path.stat().st_mode & 0o777, 0o644)

    def test_generic_setup_creates_private_modes_and_persists_explicit_container_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
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
                    "projects": {"default": "project-demo"},
                },
            )

            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
                clear=True,
            ):
                self.assertEqual(
                    main(
                        [
                            "hpc",
                            "setup",
                            "--target",
                            "synthetic-target",
                            "--user",
                            "synthetic-user",
                            "--host",
                            "login.synthetic.invalid",
                            "--remote-workspace-root",
                            "/remote/synthetic/workspace",
                            "--remote-container-root",
                            "/remote/synthetic/containers",
                        ]
                    ),
                    0,
                )

            for directory in (
                workspace_root / "secrets",
                workspace_root / "secrets" / "hpc",
            ):
                self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            for path in (
                workspace_root / "secrets" / ".env",
                workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml",
                workspace_root / "secrets" / "hpc" / "targets.yaml",
            ):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            targets = load_yaml(
                workspace_root / "secrets" / "hpc" / "targets.yaml",
                resolve_env=False,
            )
            self.assertEqual(
                targets["targets"]["synthetic-target"]["env"][
                    "RP_REMOTE_CONTAINER_ROOT"
                ],
                "/remote/synthetic/containers",
            )

    def test_high_level_alliance_robot_setup_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
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
                    "projects": {"default": "project-demo"},
                },
            )
            with mock.patch.dict(
                os.environ,
                {"RESEARCH_PLATFORM_ROOT": str(workspace_root)},
                clear=True,
            ):
                with self.assertRaises(SystemExit) as exc_info:
                    main(
                        [
                            "hpc",
                            "setup",
                            "--template",
                            "alliance",
                            "--target",
                            "synthetic-target",
                            "--role",
                            "robot",
                            "--user",
                            "alice",
                            "--host",
                            "login.synthetic.invalid",
                            "--remote-workspace-root",
                            "/remote/synthetic/workspace",
                        ]
                    )

            self.assertIn("robot credentials are not yet modeled", str(exc_info.exception))
            self.assertFalse((workspace_root / "secrets").exists())

    def test_generic_setup_force_is_scoped_and_private_files_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            (workspace_root / "secrets" / "hpc").mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml",
                {"profiles": {"unrelated": {"host": "other.synthetic.invalid", "user": "other"}}},
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "targets.yaml",
                {"version": 1, "default": "unrelated", "targets": {"unrelated": {"ssh_profile": "unrelated"}}},
            )
            args = [
                "hpc",
                "setup",
                "--target",
                "synthetic-target",
                "--user",
                "synthetic-user",
                "--host",
                "login.synthetic.invalid",
                "--remote-workspace-root",
                "/remote/synthetic/workspace",
            ]
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                self.assertEqual(main(args), 0)
                original_ssh = (workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml").read_bytes()
                original_targets = (workspace_root / "secrets" / "hpc" / "targets.yaml").read_bytes()
                with self.assertRaises(SystemExit):
                    main([*args[:-3], "login.changed.invalid", *args[-2:]])
                self.assertEqual((workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml").read_bytes(), original_ssh)
                self.assertEqual((workspace_root / "secrets" / "hpc" / "targets.yaml").read_bytes(), original_targets)
                self.assertEqual(main([*args[:-3], "login.changed.invalid", *args[-2:], "--force"]), 0)

            ssh_document = load_yaml(workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml", resolve_env=False)
            target_document = load_yaml(workspace_root / "secrets" / "hpc" / "targets.yaml", resolve_env=False)
            self.assertIn("unrelated", ssh_document["profiles"])
            self.assertIn("unrelated", target_document["targets"])
            self.assertEqual(target_document["default"], "unrelated")
            self.assertEqual(ssh_document["profiles"]["synthetic-target"]["host"], "login.changed.invalid")
            for path in (
                workspace_root / "secrets" / ".env",
                workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml",
                workspace_root / "secrets" / "hpc" / "targets.yaml",
            ):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_target_env_overrides_local_defaults_and_injects_project_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            project_root = workspace_root / "project" / "project-demo"
            (project_root / "config").mkdir(parents=True)
            (workspace_root / "secrets" / "hpc").mkdir(parents=True)
            (workspace_root / "secrets" / ".env").write_text(
                "\n".join(
                    [
                        "RESEARCH_HPC_TARGET=cluster-b",
                        "RP_REMOTE_WORKSPACE_ROOT=/remote/cluster-a/research-platform",
                        "RP_REMOTE_ARTIFACTS_ROOT=/remote/cluster-a/research-platform/artifacts",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            write_yaml(project_root / "project.yaml", {"name": "project-demo"})
            write_yaml(
                project_root / "config" / "dataset.yaml",
                {
                    "dataset": {
                        "primary": "demo-bids",
                        "input_derivative": "demo-deriv",
                        "remote_bids_root": "${DEMO_REMOTE_BIDS_ROOT:-}",
                    }
                },
            )
            write_yaml(
                project_root / "config" / "compute.yaml",
                {
                    "compute": {
                        "default_profile": "slurm",
                        "slurm": {
                            "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                            "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
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
                        "default_batch": "default",
                    }
                },
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "targets.yaml",
                {
                    "version": 1,
                    "default": "cluster-a",
                    "targets": {
                        "cluster-b": {
                            "ssh_profile": "cluster-b",
                            "role": "login",
                            "env": {
                                "RP_REMOTE_WORKSPACE_ROOT": "/remote/cluster-b/research-platform",
                                "RP_REMOTE_ARTIFACTS_ROOT": "/remote/cluster-b/research-platform/artifacts",
                            },
                            "slurm": {"account": "example-account", "partition": "compute"},
                            "projects": {
                                "project-demo": {
                                    "env": {"DEMO_REMOTE_BIDS_ROOT": "/remote/cluster-b/demo-bids"}
                                }
                            },
                        }
                    },
                },
            )

            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                load_local_hpc_env_defaults(workspace_root)
                bundle = load_project_bundle("project-demo", workspace_root)

            slurm = bundle["compute"]["compute"]["slurm"]
            self.assertEqual(slurm["remote_workspace_root"], "/remote/cluster-b/research-platform")
            self.assertEqual(slurm["account"], "example-account")
            self.assertEqual(slurm["partition"], "compute")
            self.assertEqual(resolve_bids_remote_dataset_root(bundle), "/remote/cluster-b/demo-bids")

    def test_target_show_reports_missing_profile_and_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            (workspace_root / "secrets" / "hpc").mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "targets.yaml",
                {
                    "version": 1,
                    "targets": {
                        "cluster-b": {
                            "ssh_profile": "missing-profile",
                            "ssh_config": "secrets/hpc/ssh-profiles.yaml",
                            "env": {"RP_REMOTE_WORKSPACE_ROOT": "/remote/<group>/<user>/research-platform"},
                        }
                    },
                },
            )

            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                target = resolve_hpc_target(target_name="cluster-b", root=workspace_root)

            self.assertIsNotNone(target)
            assert target is not None
            self.assertTrue(any("SSH config is missing" in warning for warning in target["warnings"]))
            self.assertTrue(any("<group>" in warning for warning in target["warnings"]))

    def test_target_use_writes_active_target_to_local_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            (workspace_root / "secrets" / "hpc").mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "targets.yaml",
                {"version": 1, "targets": {"cluster-b": {"env": {"RP_REMOTE_WORKSPACE_ROOT": "/remote/demo"}}}},
            )

            buffer = io.StringIO()
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with redirect_stdout(buffer):
                    exit_code = main(["hpc", "target", "use", "cluster-b"])

            self.assertEqual(exit_code, 0)
            self.assertIn("Active HPC target set to cluster-b", buffer.getvalue())
            self.assertIn(
                "RESEARCH_HPC_TARGET=cluster-b",
                (workspace_root / "secrets" / ".env").read_text(encoding="utf-8"),
            )

    def test_generic_setup_writes_only_explicit_profile_and_target_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            (workspace_root / "secrets" / "hpc").mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                    "hpc": {
                        "runtime_defaults": {
                            "default": "generic-slurm",
                            "catalog": {
                                "generic-slurm": {
                                    "slurm": {
                                        "modules": ["python/3.11", "apptainer/1.4"],
                                        "environment": {"TMPDIR": "/remote/runtime/tmp"},
                                        "prepare_directories": ["$TMPDIR"],
                                    }
                                }
                            },
                        }
                    },
                },
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "targets.yaml",
                {"version": 1, "targets": {"cluster-a": {"ssh_profile": "cluster-a", "role": "login"}}},
            )

            identity_file = workspace_root / "secrets" / "hpc" / "synthetic-identity"
            identity_file.write_text("synthetic path reference only\n", encoding="utf-8")
            buffer = io.StringIO()
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with redirect_stdout(buffer):
                    exit_code = main(
                        [
                            "hpc",
                            "setup",
                            "--cluster",
                            "cluster-b",
                            "--user",
                            "alice",
                            "--host",
                            "login.synthetic.invalid",
                            "--identity-file",
                            str(identity_file),
                            "--remote-workspace-root",
                            "/remote/alice/research-platform",
                            "--account",
                            "example-account",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            ssh_config = load_yaml(workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml", resolve_env=False)
            self.assertEqual(ssh_config["profiles"]["cluster-b"]["host"], "login.synthetic.invalid")
            self.assertEqual(ssh_config["profiles"]["cluster-b"]["user"], "alice")
            self.assertEqual(ssh_config["profiles"]["cluster-b"]["identity_file"], str(identity_file))
            targets = load_yaml(workspace_root / "secrets" / "hpc" / "targets.yaml", resolve_env=False)
            self.assertIn("cluster-a", targets["targets"])
            cluster_b = targets["targets"]["cluster-b"]
            self.assertEqual(cluster_b["ssh_profile"], "cluster-b")
            self.assertEqual(cluster_b["env"]["RP_REMOTE_WORKSPACE_ROOT"], "/remote/alice/research-platform")
            self.assertEqual(cluster_b["env"]["RP_REMOTE_ARTIFACTS_ROOT"], "/remote/alice/research-platform/artifacts")
            self.assertEqual(cluster_b["slurm"]["account"], "example-account")
            self.assertEqual(cluster_b["promotion"], {"mode": "atomic_no_replace"})
            self.assertNotIn("RP_REMOTE_CONTAINER_ROOT", cluster_b["env"])
            self.assertNotIn("modules", cluster_b["slurm"])
            env_text = (workspace_root / "secrets" / ".env").read_text(encoding="utf-8")
            self.assertIn("RESEARCH_HPC_TARGET=cluster-b", env_text)
            self.assertIn("RESEARCH_HPC_PROFILE=cluster-b", env_text)
            self.assertIn("Target: cluster-b", buffer.getvalue())
            self.assertIn("rp hpc validate --target cluster-b", buffer.getvalue())
            setup_output = buffer.getvalue()
            self.assertLess(
                setup_output.index("rp hpc validate --target cluster-b"),
                setup_output.index("rp hpc doctor"),
            )
            self.assertLess(
                setup_output.index("rp hpc doctor"),
                setup_output.index("Later runtime-readiness"),
            )

    def test_cluster_setup_preserves_parser_safe_empty_role_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            (workspace_root / "secrets" / "hpc").mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml",
                {
                    "profiles": {
                        "alliance": {
                            "defaults": {"host": "${ALLIANCE_LOGIN_HOST:-login.cluster.example}"},
                            "roles": {"login": {}, "robot": {"user": "automation"}},
                        }
                    }
                },
            )
            identity_file = workspace_root / "secrets" / "hpc" / "synthetic-identity"
            identity_file.write_text("synthetic path reference only\n", encoding="utf-8")

            buffer = io.StringIO()
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with redirect_stdout(buffer):
                    exit_code = main(
                        [
                            "hpc",
                            "setup",
                            "--target",
                            "cluster-b",
                            "--user",
                            "alice",
                            "--host",
                            "login.synthetic.invalid",
                            "--identity-file",
                            str(identity_file),
                            "--remote-workspace-root",
                            "/remote/alice/research-platform",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            ssh_config_text = (workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml").read_text(encoding="utf-8")
            self.assertIn("login: {}", ssh_config_text)
            profile = load_ssh_profile(workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml", "cluster-b")
            self.assertEqual(profile.target(), "alice@login.synthetic.invalid")

    def test_cluster_shorthand_sets_active_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            (workspace_root / "secrets" / "hpc").mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "targets.yaml",
                {"version": 1, "targets": {"cluster-a": {"env": {"RP_REMOTE_WORKSPACE_ROOT": "/remote/cluster-a"}}}},
            )

            buffer = io.StringIO()
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with redirect_stdout(buffer):
                    exit_code = main(["hpc", "cluster", "cluster-a"])

            self.assertEqual(exit_code, 0)
            self.assertIn("Active HPC cluster set to cluster-a", buffer.getvalue())
            self.assertIn(
                "RESEARCH_HPC_TARGET=cluster-a",
                (workspace_root / "secrets" / ".env").read_text(encoding="utf-8"),
            )

    def test_target_list_and_show_cli_read_local_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            (workspace_root / "secrets" / "hpc").mkdir(parents=True)
            write_yaml(
                workspace_root / "WORKSPACE.yaml",
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                    "projects": {"default": "project-demo"},
                },
            )
            write_yaml(
                workspace_root / "secrets" / "hpc" / "targets.yaml",
                {
                    "version": 1,
                    "default": "cluster-a",
                    "targets": {
                        "cluster-a": {"ssh_profile": "cluster-a", "role": "login"},
                        "cluster-b": {
                            "ssh_profile": "cluster-b",
                            "role": "login",
                            "env": {"RP_REMOTE_WORKSPACE_ROOT": "/remote/demo/research-platform"},
                            "slurm": {"partition": "compute"},
                        },
                    },
                },
            )

            list_buffer = io.StringIO()
            show_buffer = io.StringIO()
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=True):
                with redirect_stdout(list_buffer):
                    list_code = main(["hpc", "target", "list"])
                with redirect_stdout(show_buffer):
                    show_code = main(["hpc", "target", "show", "cluster-b"])

            self.assertEqual(list_code, 0)
            self.assertEqual(show_code, 0)
            self.assertIn("- cluster-a (active, default)", list_buffer.getvalue())
            self.assertIn("- cluster-b", list_buffer.getvalue())
            self.assertIn("HPC target cluster-b", show_buffer.getvalue())
            self.assertIn("- partition: compute", show_buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
