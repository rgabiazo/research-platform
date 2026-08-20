from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc.cli import main
from research_platform.hpc._yaml import dump_yaml, load_yaml, write_yaml
from research_platform.hpc.manifest import write_run_manifest, write_status


class HpcCliTests(unittest.TestCase):
    def _write_ssh_config(self, root: Path) -> Path:
        config_path = root / "ssh.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "profiles:",
                    "  cluster-a:",
                    "    defaults:",
                    "      host: cluster.example",
                    "    roles:",
                    "      login:",
                    "        user: runner",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return config_path

    def test_ssh_list_and_show_support_profile_families_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "ssh.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "defaults:",
                        "  options:",
                        "    ServerAliveInterval: 30",
                        "profiles:",
                        "  cluster-a:",
                        "    defaults:",
                        "      host: cluster.example",
                        "    roles:",
                        "      login:",
                        "        user: analyst",
                        "      robot:",
                        "        user: robot-user",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            list_stdout = io.StringIO()
            with contextlib.redirect_stdout(list_stdout):
                exit_code = main(["ssh", "list", "--config", str(config_path)])

            show_stdout = io.StringIO()
            with contextlib.redirect_stdout(show_stdout):
                show_code = main(
                    ["ssh", "show", "--config", str(config_path), "--profile", "cluster-a", "--role", "robot"]
                )

        list_report = json.loads(list_stdout.getvalue())
        show_report = json.loads(show_stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(show_code, 0)
        self.assertEqual(
            list_report["profiles"],
            [{"name": "cluster-a", "kind": "family", "roles": ["login", "robot"]}],
        )
        self.assertEqual(show_report["profile"], "cluster-a")
        self.assertEqual(show_report["role"], "robot")
        self.assertEqual(show_report["target"], "robot-user@cluster.example")
        self.assertEqual(show_report["resolved"]["options"]["ServerAliveInterval"], "30")

    def test_ssh_init_config_writes_alliance_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["ssh", "init-config", "--template", "alliance", "--output", str(output_path)])

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertEqual(report["template"], "alliance")
            template_text = output_path.read_text(encoding="utf-8")
            self.assertIn("profiles:", template_text)
            self.assertIn("robot:", template_text)
            self.assertIn("ControlMaster: auto", template_text)
            self.assertIn('ControlPath: "~/.ssh/cm-%C"', template_text)
            self.assertIn("ControlPersist: 2h", template_text)

    def test_ssh_init_config_defaults_to_concrete_generic_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            stdout = io.StringIO()
            with (
                mock.patch("subprocess.run") as run_mock,
                mock.patch("subprocess.Popen") as popen_mock,
                mock.patch("socket.create_connection") as socket_connect_mock,
                mock.patch("socket.getaddrinfo") as dns_lookup_mock,
                mock.patch("socket.socket") as socket_constructor_mock,
            ):
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "ssh",
                            "init-config",
                            "--output",
                            str(output_path),
                            "--profile",
                            "synthetic",
                            "--host",
                            "login.synthetic.invalid",
                            "--user",
                            "analyst",
                            "--port",
                            "2222",
                        ]
                    )

            report = json.loads(stdout.getvalue())
            document = load_yaml(output_path)
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["template"], "generic")
            self.assertEqual(report["profile"], "synthetic")
            self.assertEqual(
                document,
                {
                    "profiles": {
                        "synthetic": {
                            "host": "login.synthetic.invalid",
                            "user": "analyst",
                            "port": 2222,
                        }
                    }
                },
            )
            rendered = output_path.read_text(encoding="utf-8").lower()
            self.assertNotIn("alliance", rendered)
            self.assertNotIn("robot", rendered)
            self.assertNotIn("controlmaster", rendered)
            self.assertNotIn("example.org", rendered)
            run_mock.assert_not_called()
            popen_mock.assert_not_called()
            socket_connect_mock.assert_not_called()
            dns_lookup_mock.assert_not_called()
            socket_constructor_mock.assert_not_called()

    def test_ssh_init_config_missing_generic_values_fails_before_write(self) -> None:
        for omitted, arguments in (
            ("host", ["--user", "analyst"]),
            ("user", ["--host", "login.synthetic.invalid"]),
        ):
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as tmp_dir:
                output_path = Path(tmp_dir) / "ssh-profiles.yaml"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["ssh", "init-config", "--output", str(output_path), *arguments])

                report = json.loads(stdout.getvalue())
                self.assertEqual(exit_code, 1)
                self.assertFalse(report["created"])
                self.assertIn("requires explicit host and user", report["error"])
                self.assertFalse(output_path.exists())

    def test_ssh_init_config_preserves_unrelated_profiles_and_scopes_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            write_yaml(
                output_path,
                {
                    "profiles": {
                        "unrelated": {
                            "host": "other.synthetic.invalid",
                            "user": "other",
                            "options": {"ServerAliveInterval": 30},
                        },
                        "selected": {"host": "old.synthetic.invalid", "user": "analyst"},
                    },
                },
            )
            before_conflict = output_path.read_bytes()
            conflict_stdout = io.StringIO()
            with contextlib.redirect_stdout(conflict_stdout):
                conflict_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--profile",
                        "selected",
                        "--host",
                        "new.synthetic.invalid",
                        "--user",
                        "analyst",
                    ]
                )
            self.assertEqual(conflict_code, 1)
            self.assertEqual(output_path.read_bytes(), before_conflict)

            force_stdout = io.StringIO()
            with contextlib.redirect_stdout(force_stdout):
                force_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--profile",
                        "selected",
                        "--host",
                        "new.synthetic.invalid",
                        "--user",
                        "analyst",
                        "--force",
                    ]
                )

            document = load_yaml(output_path)
            self.assertEqual(force_code, 0)
            self.assertEqual(document["profiles"]["selected"]["host"], "new.synthetic.invalid")
            self.assertEqual(document["profiles"]["unrelated"]["host"], "other.synthetic.invalid")
            self.assertEqual(
                document["profiles"]["unrelated"]["options"]["ServerAliveInterval"],
                30,
            )

    def test_generic_init_config_rejects_implicit_global_defaults_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            write_yaml(
                output_path,
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
            original = output_path.read_bytes()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--profile",
                        "synthetic",
                        "--host",
                        "login.synthetic.invalid",
                        "--user",
                        "analyst",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(report["created"])
            self.assertIn("would be inherited implicitly", report["error"])
            self.assertEqual(output_path.read_bytes(), original)

    def test_ssh_init_config_rejects_symlink_destination_without_mutating_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "target.yaml"
            target_path.write_text("sentinel\n", encoding="utf-8")
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            output_path.symlink_to(target_path)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--host",
                        "login.synthetic.invalid",
                        "--user",
                        "analyst",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertIn("not a symlink", report["error"])
            self.assertEqual(target_path.read_text(encoding="utf-8"), "sentinel\n")

    @unittest.skipUnless(os.name == "posix", "POSIX private-mode contract")
    def test_ssh_init_config_creates_private_file_and_new_parent_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            private_parent = Path(tmp_dir) / "new-private" / "hpc"
            output_path = private_parent / "ssh-profiles.yaml"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--profile",
                        "synthetic",
                        "--host",
                        "login.synthetic.invalid",
                        "--user",
                        "analyst",
                    ]
                )

            self.assertEqual(exit_code, 0, stdout.getvalue())
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(private_parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(private_parent.parent.stat().st_mode), 0o700)

    @unittest.skipUnless(hasattr(os, "link"), "hard links are unavailable")
    def test_ssh_init_config_rejects_external_hard_link_without_reading_or_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            outside = root / "outside-private-config.yaml"
            original = b"profiles:\n  selected:\n    host: old.synthetic.invalid\n    user: analyst\n"
            outside.write_bytes(original)
            output_path = root / "workspace" / "ssh-profiles.yaml"
            output_path.parent.mkdir()
            os.link(outside, output_path)
            stdout = io.StringIO()
            with (
                mock.patch("research_platform.hpc.cli.parse_yaml") as parse_mock,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--profile",
                        "selected",
                        "--host",
                        "new.synthetic.invalid",
                        "--user",
                        "analyst",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertIn("hard-linked", report["error"])
            self.assertEqual(outside.read_bytes(), original)
            self.assertEqual(output_path.read_bytes(), original)
            parse_mock.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX private-mode contract")
    def test_ssh_init_config_permission_failure_preserves_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            original = b"profiles:\n  selected:\n    host: old.synthetic.invalid\n    user: analyst\n"
            output_path.write_bytes(original)
            output_path.chmod(0o644)
            stdout = io.StringIO()
            with (
                mock.patch(
                    "research_platform.hpc.cli.os.fchmod",
                    side_effect=PermissionError("synthetic permission failure"),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--profile",
                        "selected",
                        "--host",
                        "new.synthetic.invalid",
                        "--user",
                        "analyst",
                        "--force",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertIn("synthetic permission failure", report["error"])
            self.assertEqual(output_path.read_bytes(), original)
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o644)

    @unittest.skipUnless(os.name == "posix", "POSIX private-mode contract")
    def test_ssh_init_config_secure_creation_failure_removes_owned_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            stdout = io.StringIO()
            with (
                mock.patch(
                    "research_platform.hpc.cli.os.fchmod",
                    side_effect=PermissionError("synthetic permission failure"),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--host",
                        "login.synthetic.invalid",
                        "--user",
                        "analyst",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertFalse(output_path.exists())

    def test_ssh_init_config_rejects_special_file_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            output_path.mkdir()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--host",
                        "login.synthetic.invalid",
                        "--user",
                        "analyst",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertIn("regular file", report["error"])

    def test_ssh_init_config_rejects_symlinked_parent_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            real_parent = root / "real-private"
            real_parent.mkdir()
            linked_parent = root / "linked-private"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            output_path = linked_parent / "ssh-profiles.yaml"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "ssh",
                        "init-config",
                        "--output",
                        str(output_path),
                        "--host",
                        "login.synthetic.invalid",
                        "--user",
                        "analyst",
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertIn("not a symlink", report["error"])
            self.assertFalse(output_path.exists())
            self.assertFalse((real_parent / "ssh-profiles.yaml").exists())

    def test_ssh_init_config_rejects_unsafe_identity_grammar_before_writing(self) -> None:
        cases = (
            ("profile", ["--profile", "unsafe:name", "--host", "login.synthetic.invalid", "--user", "analyst"]),
            ("host", ["--profile", "safe", "--host=-oProxyCommand=other", "--user", "analyst"]),
            ("user", ["--profile", "safe", "--host", "login.synthetic.invalid", "--user=--proxy"]),
            (
                "identity",
                [
                    "--profile",
                    "safe",
                    "--host",
                    "login.synthetic.invalid",
                    "--user",
                    "analyst",
                    "--identity-file",
                    "${SSH_KEY}",
                ],
            ),
            (
                "known-hosts",
                [
                    "--profile",
                    "safe",
                    "--host",
                    "login.synthetic.invalid",
                    "--user",
                    "analyst",
                    "--known-hosts-file=-oProxyCommand=other",
                ],
            ),
        )
        for label, arguments in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                output_path = Path(tmp_dir) / "ssh-profiles.yaml"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["ssh", "init-config", "--output", str(output_path), *arguments])
                self.assertEqual(exit_code, 1, stdout.getvalue())
                self.assertFalse(output_path.exists())

    def test_alliance_init_config_is_idempotent_and_force_is_selected_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ssh-profiles.yaml"
            base_arguments = [
                "ssh",
                "init-config",
                "--template",
                "alliance",
                "--output",
                str(output_path),
                "--profile",
                "reviewed-site",
                "--host",
                "login.synthetic.invalid",
                "--user",
                "analyst",
                "--identity-file",
                "/private/tmp/synthetic-id",
            ]
            first_stdout = io.StringIO()
            with contextlib.redirect_stdout(first_stdout):
                first_code = main(base_arguments)
            first_bytes = output_path.read_bytes()
            first_document = load_yaml(output_path)

            second_stdout = io.StringIO()
            with contextlib.redirect_stdout(second_stdout):
                second_code = main(base_arguments)
            self.assertEqual(first_code, 0, first_stdout.getvalue())
            self.assertEqual(second_code, 0, second_stdout.getvalue())
            self.assertEqual(output_path.read_bytes(), first_bytes)

            first_document["profiles"]["unrelated"] = {
                "host": "other.synthetic.invalid",
                "user": "other",
            }
            output_path.write_text(
                dump_yaml(first_document),
                encoding="utf-8",
            )
            output_path.chmod(0o600)
            before_conflict = output_path.read_bytes()
            conflict_arguments = [
                "ssh",
                "init-config",
                "--template",
                "alliance",
                "--output",
                str(output_path),
                "--profile",
                "reviewed-site",
                "--host",
                "changed.synthetic.invalid",
                "--user",
                "analyst",
                "--identity-file",
                "/private/tmp/synthetic-id",
            ]
            conflict_stdout = io.StringIO()
            with contextlib.redirect_stdout(conflict_stdout):
                conflict_code = main(conflict_arguments)
            self.assertEqual(conflict_code, 1)
            self.assertEqual(output_path.read_bytes(), before_conflict)

            force_stdout = io.StringIO()
            with contextlib.redirect_stdout(force_stdout):
                force_code = main([*conflict_arguments, "--force"])
            updated = load_yaml(output_path)
            self.assertEqual(force_code, 0, force_stdout.getvalue())
            self.assertEqual(updated["profiles"]["unrelated"], first_document["profiles"]["unrelated"])
            selected = updated["profiles"]["reviewed-site"]
            self.assertEqual(selected["defaults"]["host"], "changed.synthetic.invalid")

    def test_stage_default_plans_without_remote_calls_and_execute_runs_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            run_root = workspace_root / "artifacts" / "runs" / "unit-stage"
            stage_source = workspace_root / "artifacts" / "runs" / "unit-stage" / "submit.sbatch"
            scope_source = workspace_root / "packages" / "research-core"
            stage_source.parent.mkdir(parents=True, exist_ok=True)
            scope_source.mkdir(parents=True, exist_ok=True)
            stage_source.write_text("#!/bin/bash\n", encoding="utf-8")
            write_run_manifest(
                run_root,
                {
                    "run_id": "unit-stage",
                    "execution": {
                        "mode": "slurm",
                        "command": [],
                        "work_dir": "remote/run/work",
                        "output_dir": "remote/run/outputs",
                        "log_dir": "remote/run/logs",
                    },
                    "slurm": {
                        "script_path": "artifacts/runs/unit-stage/submit.sbatch",
                        "jobspec": {
                            "log_out": "remote/run/logs/slurm.out",
                            "log_err": "remote/run/logs/slurm.err",
                        },
                    },
                    "hpc": {"ssh_host": "example-hpc", "remote_run_root": "remote/run", "remote_workspace_root": "remote/workspace"},
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
            )
            write_status(run_root, {"run_id": "unit-stage", "state": "planned"})
            plan_stdout = io.StringIO()
            with mock.patch("research_platform.hpc.remote.subprocess.run") as plan_run_mock:
                with contextlib.redirect_stdout(plan_stdout):
                    plan_exit_code = main(
                        ["stage", "--workspace-root", str(workspace_root), "--run-root", str(run_root)]
                    )

            stdout = io.StringIO()
            completed = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed) as run_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        ["stage", "--workspace-root", str(workspace_root), "--run-root", str(run_root), "--execute"]
                    )

        plan_report = json.loads(plan_stdout.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertEqual(plan_exit_code, 0)
        self.assertNotIn("execution", plan_report)
        self.assertEqual(
            plan_report["local_files_written"],
            [*plan_report["staged_files"], str(run_root.resolve() / "hpc" / "stage-plan.yaml")],
        )
        plan_run_mock.assert_not_called()
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["execution"]["ok"])
        self.assertEqual(run_mock.call_count, 3)
        prepare_command = run_mock.call_args_list[0].args[0]
        self.assertEqual(prepare_command[:2], ["ssh", "example-hpc"])
        self.assertIn("remote/run", prepare_command[-1])
        self.assertIn("remote/run/logs", prepare_command[-1])
        self.assertIn("remote/run/work", prepare_command[-1])
        self.assertIn("remote/run/outputs", prepare_command[-1])
        self.assertIn("remote/workspace/packages/research-core", prepare_command[-1])

    def test_submit_default_only_renders_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "run"
            write_run_manifest(
                run_root,
                {
                    "run_id": "unit-submit",
                    "execution": {"mode": "slurm"},
                    "hpc": {"submit_command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"]},
                },
            )
            write_status(run_root, {"run_id": "unit-submit", "state": "staged"})
            stdout = io.StringIO()
            with mock.patch("research_platform.hpc.remote.subprocess.run") as run_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["submit", "--run-root", str(run_root)])

            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertFalse(report["executed"])
            self.assertNotIn("execution", report)
            run_mock.assert_not_called()

    def test_submit_execute_runs_manifest_command_once_and_extracts_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "run"
            write_run_manifest(
                run_root,
                {
                    "run_id": "unit-submit",
                    "execution": {"mode": "slurm"},
                    "hpc": {"submit_command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"]},
                },
            )
            write_status(run_root, {"run_id": "unit-submit", "state": "staged"})
            stdout = io.StringIO()
            completed = mock.Mock(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed) as run_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["submit", "--run-root", str(run_root), "--execute"])

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["executed"])
        self.assertEqual(report["execution"]["job_id"], "12345")
        run_mock.assert_called_once()

    def test_rsync_default_and_dry_run_never_invoke_rsync(self) -> None:
        for operation in ("push", "pull"):
            for extra_args in ([], ["--dry-run"]):
                with self.subTest(operation=operation, extra_args=extra_args), tempfile.TemporaryDirectory() as tmp_dir:
                    config_path = self._write_ssh_config(Path(tmp_dir))
                    stdout = io.StringIO()
                    argv = [
                        "rsync",
                        operation,
                        "--profile",
                        "cluster-a",
                        "--config",
                        str(config_path),
                        "--source",
                        "local/source",
                        "--destination",
                        "remote/destination",
                        *extra_args,
                    ]
                    with mock.patch("research_platform.hpc.transfers.subprocess.run") as run_mock:
                        with contextlib.redirect_stdout(stdout):
                            exit_code = main(argv)

                    report = json.loads(stdout.getvalue())
                    self.assertEqual(exit_code, 0)
                    self.assertFalse(report["executed"])
                    self.assertEqual(report["dry_run"], bool(extra_args))
                    self.assertEqual("--dry-run" in report["command"], bool(extra_args))
                    run_mock.assert_not_called()

    def test_rsync_execute_invokes_each_transfer_once(self) -> None:
        for operation in ("push", "pull"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp_dir:
                config_path = self._write_ssh_config(Path(tmp_dir))
                stdout = io.StringIO()
                completed = mock.Mock(returncode=0)
                argv = [
                    "rsync",
                    operation,
                    "--profile",
                    "cluster-a",
                    "--config",
                    str(config_path),
                    "--source",
                    "local/source",
                    "--destination",
                    "remote/destination",
                    "--execute",
                ]
                with mock.patch("research_platform.hpc.transfers.subprocess.run", return_value=completed) as run_mock:
                    with contextlib.redirect_stdout(stdout):
                        exit_code = main(argv)

                report = json.loads(stdout.getvalue())
                self.assertEqual(exit_code, 0)
                self.assertTrue(report["executed"])
                self.assertFalse(report["dry_run"])
                run_mock.assert_called_once_with(report["command"], check=False, text=True)

    def test_rsync_rejects_dry_run_with_execute(self) -> None:
        for operation in ("push", "pull"):
            with self.subTest(operation=operation):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "rsync",
                            operation,
                            "--profile",
                            "cluster-a",
                            "--source",
                            "local/source",
                            "--destination",
                            "remote/destination",
                            "--dry-run",
                            "--execute",
                        ]
                    )

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("not allowed with argument", stderr.getvalue())

    def test_stage_reports_clean_error_when_connection_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            run_root = workspace_root / "artifacts" / "runs" / "unit-stage-missing"
            stage_source = run_root / "submit.sbatch"
            stage_source.parent.mkdir(parents=True, exist_ok=True)
            stage_source.write_text("#!/bin/bash\n", encoding="utf-8")
            write_run_manifest(
                run_root,
                {
                    "run_id": "unit-stage-missing",
                    "execution": {"mode": "slurm", "command": [], "output_dir": "artifacts/runs/unit-stage-missing/outputs"},
                    "slurm": {"script_path": "artifacts/runs/unit-stage-missing/submit.sbatch"},
                    "hpc": {"remote_run_root": "remote/run", "remote_workspace_root": "remote/workspace"},
                    "provision": {"scopes": []},
                },
            )
            write_status(run_root, {"run_id": "unit-stage-missing", "state": "planned"})
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["stage", "--workspace-root", str(workspace_root), "--run-root", str(run_root)])

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertIn("HPC connection is not configured.", report["error"])

    def test_pull_default_plans_without_remote_calls_and_execute_runs_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            run_root = workspace_root / "artifacts" / "runs" / "unit-pull"
            destination = Path(tmp_dir) / "exports" / "fmripost_aroma"
            run_root.mkdir(parents=True, exist_ok=True)
            write_run_manifest(
                run_root,
                {
                    "run_id": "unit-pull",
                    "execution": {"mode": "slurm"},
                    "hpc": {
                        "ssh_host": "example-hpc",
                        "remote_run_root": "remote/workspace/artifacts/runs/unit-pull",
                    },
                },
            )
            write_status(run_root, {"run_id": "unit-pull", "state": "planned"})

            stdout = io.StringIO()
            argv = [
                "pull",
                "--workspace-root",
                str(workspace_root),
                "--run-root",
                str(run_root),
                "--subpath",
                "outputs/fmripost_aroma",
                "--destination",
                str(destination),
            ]
            with mock.patch("research_platform.hpc.remote.subprocess.run") as plan_run_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(argv)

            execute_stdout = io.StringIO()
            completed = mock.Mock(returncode=0)
            with mock.patch("research_platform.hpc.remote.subprocess.run", return_value=completed) as execute_run_mock:
                with contextlib.redirect_stdout(execute_stdout):
                    execute_exit_code = main([*argv, "--execute"])

        report = json.loads(stdout.getvalue())
        execute_report = json.loads(execute_stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertNotIn("execution", report)
        plan_run_mock.assert_not_called()
        self.assertEqual(report["subpath"], "outputs/fmripost_aroma")
        self.assertEqual(report["remote_source"], "remote/workspace/artifacts/runs/unit-pull/outputs/fmripost_aroma")
        self.assertEqual(report["destination"], str(destination.resolve()))
        self.assertTrue(report["progress"])
        self.assertIn("--progress", report["pull_command"])
        self.assertEqual(report["local_files_written"], [str(run_root.resolve() / "hpc" / "pull-plan.yaml")])
        self.assertEqual(execute_exit_code, 0)
        self.assertTrue(execute_report["execution"]["ok"])
        execute_run_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
