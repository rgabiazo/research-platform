from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc.offline_validation import (
    _resolve_target_environment_like_production,
    render_hpc_validation_report,
    resolve_local_hpc_environment_defaults,
    validate_hpc_configuration,
)


class OfflineHpcValidationTests(unittest.TestCase):
    def _documents(self) -> tuple[dict[str, object], dict[str, object]]:
        targets = {
            "version": 1,
            "default": "synthetic",
            "targets": {
                "synthetic": {
                    "ssh_profile": "synthetic",
                    "role": "login",
                    "ssh_config": "secrets/hpc/ssh-profiles.yaml",
                    "env": {
                        "RP_REMOTE_WORKSPACE_ROOT": "/remote/research-platform",
                        "RP_REMOTE_ARTIFACTS_ROOT": "/remote/research-platform/artifacts",
                    },
                    "promotion": {"mode": "atomic_no_replace"},
                }
            },
        }
        ssh = {
            "profiles": {
                "synthetic": {
                    "host": "login.synthetic.invalid",
                    "user": "synthetic-user",
                }
            }
        }
        return targets, ssh

    def _validate(
        self,
        *,
        targets: dict[str, object] | None = None,
        ssh: dict[str, object] | None = None,
        environment: dict[str, str] | None = None,
    ) -> dict[str, object]:
        default_targets, default_ssh = self._documents()
        return validate_hpc_configuration(
            workspace_root="/private/tmp/synthetic-workspace",
            targets_config_path="/private/tmp/synthetic-workspace/secrets/hpc/targets.yaml",
            targets_document=default_targets if targets is None else targets,
            ssh_document=default_ssh if ssh is None else ssh,
            environment={} if environment is None else environment,
        )

    def test_valid_generic_configuration_passes_without_mutating_environment(self) -> None:
        before = dict(os.environ)

        report = self._validate()

        self.assertTrue(report["configuration_valid"], report["errors"])
        self.assertEqual(dict(os.environ), before)
        self.assertTrue(report["offline"])
        self.assertFalse(report["network_contacted"])
        self.assertEqual(report["target"], "synthetic")
        self.assertEqual(report["profile"], "synthetic")
        self.assertEqual(report["role"], "login")
        self.assertEqual(report["promotion_policy"]["mode"], "atomic_no_replace")
        self.assertEqual(
            report["promotion_policy"]["verification"],
            "declared, not remotely verified",
        )
        rendered = render_hpc_validation_report(report)
        self.assertIn("Network contacted: no", rendered)
        self.assertIn("filesystem promotion capability", rendered)
        self.assertNotIn("login.synthetic.invalid", rendered)
        self.assertNotIn("synthetic-user", rendered)

    def test_unmanaged_process_root_retains_precedence_and_is_validated(self) -> None:
        targets, ssh = self._documents()

        report = self._validate(
            targets=targets,
            ssh=ssh,
            environment={"RP_REMOTE_WORKSPACE_ROOT": "relative/unsafe"},
        )

        self.assertFalse(report["configuration_valid"])
        self.assertTrue(
            any(
                "effective environment.RP_REMOTE_WORKSPACE_ROOT must be an absolute POSIX path"
                in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_managed_local_root_is_replaced_by_target_default(self) -> None:
        targets, ssh = self._documents()

        report = validate_hpc_configuration(
            workspace_root="/private/tmp/synthetic-workspace",
            targets_config_path="/private/tmp/synthetic-workspace/secrets/hpc/targets.yaml",
            targets_document=targets,
            ssh_document=ssh,
            environment={"RP_REMOTE_WORKSPACE_ROOT": "relative/local-default"},
            managed_environment_defaults={
                "RP_REMOTE_WORKSPACE_ROOT": "relative/local-default"
            },
        )

        self.assertTrue(report["configuration_valid"], report["errors"])

    def test_local_default_resolution_tracks_only_values_absent_from_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "secrets").mkdir()
            (root / "secrets" / ".env").write_text(
                "\n".join(
                    (
                        "RP_REMOTE_WORKSPACE_ROOT=/local/workspace",
                        "RP_REMOTE_ARTIFACTS_ROOT=/local/artifacts",
                        "UNRELATED_VALUE=ignored",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            effective, managed = resolve_local_hpc_environment_defaults(
                workspace_root=root,
                environment={"RP_REMOTE_WORKSPACE_ROOT": "/process/workspace"},
            )

        self.assertEqual(
            effective["RP_REMOTE_WORKSPACE_ROOT"],
            "/process/workspace",
        )
        self.assertEqual(
            effective["RP_REMOTE_ARTIFACTS_ROOT"],
            "/local/artifacts",
        )
        self.assertEqual(
            managed,
            {"RP_REMOTE_ARTIFACTS_ROOT": "/local/artifacts"},
        )
        self.assertNotIn("UNRELATED_VALUE", effective)

    def test_target_and_project_interpolation_follow_production_pass_order(self) -> None:
        targets, _ = self._documents()
        target = targets["targets"]["synthetic"]
        target["env"] = {
            "RP_REMOTE_WORKSPACE_ROOT": "/target/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "${RP_REMOTE_WORKSPACE_ROOT}/artifacts",
            "RP_REMOTE_CACHE_ROOT": "/target/cache",
        }
        target["slurm"] = {
            "environment": {
                "RP_REMOTE_TMP_ROOT": "${RP_REMOTE_WORKSPACE_ROOT}/scratch",
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

        (
            _,
            selected_name,
            resolved_target,
            effective,
        ) = _resolve_target_environment_like_production(
            targets,
            target_name="synthetic",
            project_name="project-demo",
            environment={
                "RP_REMOTE_WORKSPACE_ROOT": "/local/workspace",
                "RP_REMOTE_ARTIFACTS_ROOT": "/local/artifacts",
                "RP_REMOTE_CACHE_ROOT": "/process/cache",
            },
            managed_defaults={
                "RP_REMOTE_WORKSPACE_ROOT": "/local/workspace",
                "RP_REMOTE_ARTIFACTS_ROOT": "/local/artifacts",
            },
        )

        self.assertEqual(selected_name, "synthetic")
        self.assertIsNotNone(resolved_target)
        assert resolved_target is not None
        self.assertEqual(
            effective["RP_REMOTE_WORKSPACE_ROOT"],
            "/target/workspace",
        )
        self.assertEqual(
            effective["RP_REMOTE_ARTIFACTS_ROOT"],
            "/target/workspace/artifacts",
        )
        self.assertEqual(effective["RP_REMOTE_CACHE_ROOT"], "/process/cache")
        self.assertEqual(
            effective["RP_REMOTE_PROJECT_DATA_ROOT"],
            "/target/workspace/project-data",
        )
        self.assertEqual(
            resolved_target["slurm"]["environment"]["RP_REMOTE_TMP_ROOT"],
            "/target/workspace/scratch",
        )

    def test_absent_value_interpolation_matches_first_production_parse(self) -> None:
        targets, _ = self._documents()
        target = targets["targets"]["synthetic"]
        target["env"]["RP_REMOTE_ARTIFACTS_ROOT"] = (
            "${RP_REMOTE_WORKSPACE_ROOT}/artifacts"
        )
        target["slurm"] = {
            "environment": {
                "RP_REMOTE_TMP_ROOT": "${RP_REMOTE_WORKSPACE_ROOT}/scratch",
            }
        }

        _, _, resolved_target, effective = _resolve_target_environment_like_production(
            targets,
            target_name=None,
            project_name=None,
            environment={},
            managed_defaults={},
        )

        self.assertIsNotNone(resolved_target)
        assert resolved_target is not None
        self.assertEqual(
            effective["RP_REMOTE_WORKSPACE_ROOT"],
            "/remote/research-platform",
        )
        self.assertEqual(effective["RP_REMOTE_ARTIFACTS_ROOT"], "/artifacts")
        self.assertEqual(
            resolved_target["slurm"]["environment"]["RP_REMOTE_TMP_ROOT"],
            "/scratch",
        )

    def test_explicit_selection_precedes_environment_and_default(self) -> None:
        targets, ssh = self._documents()
        targets["targets"]["other"] = {
            "ssh_profile": "other",
            "role": "login",
            "ssh_config": "secrets/hpc/ssh-profiles.yaml",
            "env": {"RP_REMOTE_WORKSPACE_ROOT": "/remote/other"},
            "promotion": {"mode": "atomic_no_replace"},
        }
        ssh["profiles"]["other"] = {
            "host": "login.other.invalid",
            "user": "other-user",
        }

        report = validate_hpc_configuration(
            workspace_root="/private/tmp/synthetic-workspace",
            targets_config_path="/private/tmp/synthetic-workspace/secrets/hpc/targets.yaml",
            target_name="other",
            profile_name="other",
            role="login",
            ssh_config_path="secrets/hpc/ssh-profiles.yaml",
            targets_document=targets,
            ssh_document=ssh,
            environment={
                "RESEARCH_HPC_TARGET": "synthetic",
                "RESEARCH_HPC_PROFILE": "synthetic",
                "RESEARCH_HPC_ROLE": "robot",
                "RESEARCH_HPC_SSH_CONFIG": "secrets/hpc/other-ssh.yaml",
            },
        )

        self.assertTrue(report["configuration_valid"], report["errors"])
        self.assertEqual(report["target"], "other")
        self.assertEqual(report["profile"], "other")
        self.assertEqual(report["role"], "login")
        self.assertTrue(report["configuration_paths"]["ssh"].endswith("secrets/hpc/ssh-profiles.yaml"))

    def test_unmanaged_process_profile_precedes_target_profile(self) -> None:
        targets, ssh = self._documents()
        ssh["profiles"]["process-profile"] = {
            "host": "login.process.invalid",
            "user": "process-user",
        }

        report = self._validate(
            targets=targets,
            ssh=ssh,
            environment={
                "RESEARCH_HPC_PROFILE": "process-profile",
                "RESEARCH_HPC_ROLE": "login",
            },
        )

        self.assertTrue(report["configuration_valid"], report["errors"])
        self.assertEqual(report["profile"], "process-profile")
        self.assertEqual(report["role"], "login")

    def test_placeholder_and_incomplete_profiles_fail_closed(self) -> None:
        targets, ssh = self._documents()
        ssh["profiles"]["synthetic"] = {
            "host": "${SYNTHETIC_HOST}",
            "user": "your-username",
        }

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("host" in error for error in report["errors"]))
        self.assertTrue(any("starter placeholder" in error for error in report["errors"]))

        for missing_field in ("host", "user"):
            with self.subTest(missing_field=missing_field):
                targets, ssh = self._documents()
                del ssh["profiles"]["synthetic"][missing_field]
                report = self._validate(targets=targets, ssh=ssh)
                self.assertFalse(report["configuration_valid"])
                self.assertTrue(any(missing_field in error for error in report["errors"]))

        targets, ssh = self._documents()
        targets["targets"]["synthetic"]["ssh_profile"] = "missing-profile"
        report = self._validate(targets=targets, ssh=ssh)
        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("is not declared" in error for error in report["errors"]))

    def test_example_hostname_fails_even_when_structurally_valid(self) -> None:
        targets, ssh = self._documents()
        ssh["profiles"]["synthetic"]["host"] = "login.cluster.example.org"

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("starter placeholder" in error for error in report["errors"]))

    def test_placeholder_config_path_and_ssh_option_value_fail(self) -> None:
        targets, ssh = self._documents()
        targets["targets"]["synthetic"]["ssh_config"] = "secrets/<replace-me>/ssh.yaml"
        ssh["profiles"]["synthetic"]["options"] = {"ProxyJump": "jump.example"}

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        errors = "\n".join(report["errors"])
        self.assertIn("ssh_config contains an unresolved placeholder", errors)
        self.assertIn("option 'ProxyJump' is not permitted", errors)

    def test_invalid_port_and_command_owned_ssh_options_fail(self) -> None:
        targets, ssh = self._documents()
        ssh["profiles"]["synthetic"].update(
            {
                "port": 65536,
                "options": {
                    "BatchMode": "no",
                    "StrictHostKeyChecking": "no",
                    "UserKnownHostsFile": "/dev/null",
                },
            }
        )

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("1 through 65535" in error for error in report["errors"]))
        self.assertTrue(any("not permitted" in error for error in report["errors"]))

    def test_forbidden_ssh_options_fail_case_insensitively(self) -> None:
        forbidden = (
            "hostname",
            "USER",
            "Port",
            "identityfile",
            "USERKNOWNHOSTSFILE",
            "batchmode",
            "StrictHostKeyChecking",
            "proxycommand",
            "ProxyJump",
            "localcommand",
            "PermitLocalCommand",
            "remotecommand",
            "KnownHostsCommand",
            "localforward",
            "RemoteForward",
            "dynamicforward",
            "Include",
            "pkcs11provider",
            "SecurityKeyProvider",
        )
        for option in forbidden:
            with self.subTest(option=option):
                targets, ssh = self._documents()
                ssh["profiles"]["synthetic"]["options"] = {option: "unsafe"}

                report = self._validate(targets=targets, ssh=ssh)

                self.assertFalse(report["configuration_valid"])
                self.assertTrue(
                    any(
                        f"option {option!r} is not permitted" in error
                        for error in report["errors"]
                    ),
                    report["errors"],
                )

    def test_reviewed_nonrouting_ssh_options_pass_offline_validation(self) -> None:
        targets, ssh = self._documents()
        ssh["profiles"]["synthetic"]["options"] = {
            "ConnectionAttempts": 2,
            "ConnectTimeout": 15,
            "ControlMaster": "auto",
            "ControlPath": "~/.ssh/cm-%C",
            "ControlPersist": "2h",
            "PreferredAuthentications": "publickey,keyboard-interactive",
            "ServerAliveCountMax": 3,
            "ServerAliveInterval": 30,
            "TCPKeepAlive": "yes",
        }

        report = self._validate(targets=targets, ssh=ssh)

        self.assertTrue(report["configuration_valid"], report["errors"])

    def test_identity_path_is_validated_as_a_reference_without_reading_contents(self) -> None:
        targets, ssh = self._documents()
        ssh["profiles"]["synthetic"]["identity_file"] = (
            "/private/tmp/synthetic-workspace/identity-does-not-exist"
        )

        report = self._validate(targets=targets, ssh=ssh)

        self.assertTrue(report["configuration_valid"], report["errors"])

    def test_synthetic_identity_and_known_hosts_regular_files_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            identity = root / "id_synthetic"
            known_hosts = root / "known_hosts"
            identity.write_text("synthetic-not-a-private-key\n", encoding="utf-8")
            known_hosts.write_text("synthetic.invalid marker\n", encoding="utf-8")
            targets, ssh = self._documents()
            ssh["profiles"]["synthetic"].update(
                {
                    "identity_file": str(identity),
                    "known_hosts_file": str(known_hosts),
                }
            )

            report = self._validate(targets=targets, ssh=ssh)

        self.assertTrue(report["configuration_valid"], report["errors"])

    def test_remote_root_failures_are_independent_and_nesting_is_allowed(self) -> None:
        invalid_values = (
            "relative/root",
            "/",
            "/remote/../escape",
            "/remote/<group>/workspace",
            "/remote/workspace\nother",
            "/remote/workspace;touch",
            "/remote/work*",
            "/remote/(group)/workspace",
            "/remote//workspace",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                targets, ssh = self._documents()
                targets["targets"]["synthetic"]["env"]["RP_REMOTE_WORKSPACE_ROOT"] = value
                report = self._validate(targets=targets, ssh=ssh)
                self.assertFalse(report["configuration_valid"], value)

        report = self._validate()
        self.assertTrue(report["configuration_valid"], report["errors"])

    def test_conflicting_exact_remote_roots_fail(self) -> None:
        targets, ssh = self._documents()
        targets["targets"]["synthetic"]["env"]["RP_REMOTE_ARTIFACTS_ROOT"] = (
            "/remote/research-platform"
        )

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("semantically distinct purposes" in error for error in report["errors"]))

    def test_scheduler_type_range_control_and_directive_failures(self) -> None:
        targets, ssh = self._documents()
        targets["targets"]["synthetic"]["slurm"] = {
            "nodes": 0,
            "modules": "python",
            "environment": {"SAFE": ["not", "scalar"]},
            "pre_activate_commands": ["echo ok", "#SBATCH --account=unsafe"],
            "partition": "compute\nother",
        }

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        errors = "\n".join(report["errors"])
        self.assertIn("positive integer", errors)
        self.assertIn("must contain a list", errors)
        self.assertIn("must contain a nonblank string", errors)
        self.assertIn("#SBATCH", errors)
        self.assertIn("control character", errors)

    def test_scheduler_environment_remote_roots_use_the_same_safe_path_contract(self) -> None:
        targets, ssh = self._documents()
        targets["targets"]["synthetic"]["slurm"] = {
            "environment": {
                "TMPDIR": "relative/tmp",
                "APPTAINER_CACHEDIR": "/remote/cache*",
            }
        }

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        errors = "\n".join(report["errors"])
        self.assertIn("slurm.environment.TMPDIR must be an absolute POSIX path", errors)
        self.assertIn("slurm.environment.APPTAINER_CACHEDIR contains an unsafe", errors)

    def test_unknown_structure_and_unsupported_promotion_fail(self) -> None:
        targets, ssh = self._documents()
        targets["unexpected"] = True
        targets["targets"]["synthetic"]["unexpected"] = True
        targets["targets"]["synthetic"]["promotion"] = {"mode": "overwrite"}

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        errors = "\n".join(report["errors"])
        self.assertIn("unknown key", errors)
        self.assertIn("atomic_no_replace", errors)

    def test_unselected_target_slurm_unknown_keys_fail_structural_validation(self) -> None:
        targets, ssh = self._documents()
        targets["targets"]["unselected"] = {
            "ssh_profile": "unselected",
            "role": "login",
            "ssh_config": "secrets/hpc/ssh-profiles.yaml",
            "env": {"RP_REMOTE_WORKSPACE_ROOT": "/remote/unselected"},
            "promotion": {"mode": "atomic_no_replace"},
            "slurm": {"unsupported_directive": "value"},
        }
        ssh["profiles"]["unselected"] = {
            "host": "login.unselected.invalid",
            "user": "unselected-user",
        }

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        self.assertTrue(
            any(
                "HPC target 'unselected' slurm contains unknown key 'unsupported_directive'"
                in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_missing_default_unknown_target_and_malformed_project_override_fail(self) -> None:
        targets, ssh = self._documents()
        targets["default"] = "missing"
        targets["targets"]["synthetic"]["projects"] = {
            "project-demo": {"slurm": {"partition": "compute"}}
        }

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        errors = "\n".join(report["errors"])
        self.assertIn("does not refer to a declared target", errors)
        self.assertIn("unknown key 'slurm'", errors)

    def test_unsupported_schema_and_no_selected_target_fail_independently(self) -> None:
        targets, ssh = self._documents()
        targets["version"] = 2
        report = self._validate(targets=targets, ssh=ssh)
        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("schema version 1" in error for error in report["errors"]))

        targets, ssh = self._documents()
        targets.pop("default")
        report = self._validate(targets=targets, ssh=ssh)
        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("Select an HPC target" in error for error in report["errors"]))

    def test_unselected_target_environment_and_promotion_shapes_are_validated(self) -> None:
        targets, ssh = self._documents()
        targets["targets"]["malformed"] = {
            "ssh_profile": "malformed",
            "ssh_config": "secrets/hpc/ssh-profiles.yaml",
            "env": ["not", "a", "mapping"],
            "projects": {
                "project-demo": {
                    "env": ["also", "not", "a", "mapping"],
                }
            },
            "promotion": "overwrite",
        }
        targets["targets"]["unsupported-promotion"] = {
            "ssh_profile": "unsupported-promotion",
            "ssh_config": "secrets/hpc/ssh-profiles.yaml",
            "env": {"RP_REMOTE_WORKSPACE_ROOT": "/remote/other"},
            "promotion": {"mode": "overwrite"},
        }

        report = self._validate(targets=targets, ssh=ssh)

        self.assertFalse(report["configuration_valid"])
        errors = "\n".join(report["errors"])
        self.assertIn("HPC target 'malformed' env must contain a mapping", errors)
        self.assertIn(
            "HPC target 'malformed' project override 'project-demo' env "
            "must contain a mapping",
            errors,
        )
        self.assertIn("HPC target 'malformed' promotion must declare", errors)
        self.assertIn(
            "HPC target 'unsupported-promotion' promotion.mode must be "
            "atomic_no_replace",
            errors,
        )

    def test_unknown_explicit_target_fails(self) -> None:
        targets, ssh = self._documents()

        report = validate_hpc_configuration(
            workspace_root="/private/tmp/synthetic-workspace",
            targets_config_path="/private/tmp/synthetic-workspace/secrets/hpc/targets.yaml",
            target_name="missing",
            targets_document=targets,
            ssh_document=ssh,
            environment={},
        )

        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("not declared" in error for error in report["errors"]))

    def test_invalid_role_and_ambiguous_host_alias_fail(self) -> None:
        targets, ssh = self._documents()
        targets["targets"]["synthetic"]["role"] = "robot"
        ssh["profiles"]["synthetic"] = {
            "defaults": {
                "host": "login.synthetic.invalid",
                "user": "synthetic-user",
            },
            "roles": {"login": {}},
        }
        report = self._validate(targets=targets, ssh=ssh)
        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("does not declare role 'robot'" in error for error in report["errors"]))

        targets, ssh = self._documents()
        ssh["profiles"]["synthetic"]["ssh_config_host"] = "synthetic-login"
        report = self._validate(targets=targets, ssh=ssh)
        self.assertFalse(report["configuration_valid"])
        self.assertTrue(any("exactly one" in error for error in report["errors"]))

    def test_declared_target_role_must_be_a_nonblank_valid_name(self) -> None:
        for invalid_role in (123, "", " \t", "login/other"):
            with self.subTest(role=invalid_role):
                targets, ssh = self._documents()
                targets["targets"]["synthetic"]["role"] = invalid_role

                report = self._validate(targets=targets, ssh=ssh)

                self.assertFalse(report["configuration_valid"])
                self.assertTrue(
                    any(
                        "target 'synthetic' role" in error
                        and ("nonblank string" in error or "is invalid" in error)
                        for error in report["errors"]
                    ),
                    report["errors"],
                )

    def test_missing_and_malformed_file_backed_documents_fail_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            missing = root / "missing-targets.yaml"
            report = validate_hpc_configuration(
                workspace_root=root,
                targets_config_path=missing,
                environment={},
            )
            self.assertFalse(report["configuration_valid"])
            self.assertTrue(any("was not found" in error for error in report["errors"]))

            malformed = root / "targets.yaml"
            malformed.write_text("- not-a-mapping\n", encoding="utf-8")
            report = validate_hpc_configuration(
                workspace_root=root,
                targets_config_path=malformed,
                environment={},
            )
            self.assertFalse(report["configuration_valid"])
            self.assertTrue(any("top-level mapping" in error for error in report["errors"]))

            targets, _ = self._documents()
            targets["targets"]["synthetic"]["ssh_config"] = str(root / "missing-ssh.yaml")
            report = validate_hpc_configuration(
                workspace_root=root,
                targets_config_path=root / "unused-targets.yaml",
                targets_document=targets,
                environment={},
            )
            self.assertFalse(report["configuration_valid"])
            self.assertTrue(
                any("SSH profile config was not found" in error for error in report["errors"])
            )

            malformed_ssh = root / "malformed-ssh.yaml"
            malformed_ssh.write_text("- not-a-mapping\n", encoding="utf-8")
            targets["targets"]["synthetic"]["ssh_config"] = str(malformed_ssh)
            report = validate_hpc_configuration(
                workspace_root=root,
                targets_config_path=root / "unused-targets.yaml",
                targets_document=targets,
                environment={},
            )
            self.assertFalse(report["configuration_valid"])
            self.assertTrue(
                any("SSH profile config must contain a top-level mapping" in error for error in report["errors"])
            )

    def test_input_documents_are_not_mutated(self) -> None:
        targets, ssh = self._documents()
        expected_targets = deepcopy(targets)
        expected_ssh = deepcopy(ssh)

        self._validate(targets=targets, ssh=ssh)

        self.assertEqual(targets, expected_targets)
        self.assertEqual(ssh, expected_ssh)


if __name__ == "__main__":
    unittest.main()
