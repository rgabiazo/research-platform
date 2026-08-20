from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc._yaml import expand_env_placeholders
from research_platform.hpc.ssh import build_ssh_command, render_ssh_shell
from research_platform.hpc.ssh_profiles import (
    build_ssh_config_template,
    build_ssh_profile_entry,
    list_ssh_profiles,
    load_ssh_profile,
    materialize_ssh_profile_entry,
    require_generic_profile_isolation,
    resolve_ssh_profile_config_path,
    upsert_ssh_profile_document,
)


class SshProfileTests(unittest.TestCase):
    def test_load_profile_keeps_flat_profile_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "ssh.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  login:",
                        "    host: cluster.example",
                        "    user: alice",
                        "    port: 2222",
                        "    identity_file: ~/.ssh/id_test",
                        "    known_hosts_file: ~/.ssh/known_hosts_test",
                        "    options:",
                        "      ServerAliveInterval: 30",
                        "      PreferredAuthentications: publickey,keyboard-interactive",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            profile = load_ssh_profile(config_path, "login")

        self.assertEqual(profile.target(), "alice@cluster.example")
        self.assertEqual(profile.role, "login")
        self.assertEqual(profile.port, 2222)
        self.assertTrue(profile.identity_file.endswith(".ssh/id_test"))
        self.assertEqual(profile.options["ServerAliveInterval"], "30")

    def test_alliance_template_enables_ssh_multiplexing_for_mfa_clusters(self) -> None:
        template = build_ssh_config_template("alliance")

        options = template["defaults"]["options"]
        self.assertEqual(options["ControlMaster"], "auto")
        self.assertEqual(options["ControlPath"], "~/.ssh/cm-%C")
        self.assertEqual(options["ControlPersist"], "2h")

    def test_generic_template_requires_concrete_host_and_user_and_emits_only_explicit_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires explicit host and user"):
            build_ssh_config_template()

        template = build_ssh_config_template(
            host="login.research.invalid",
            user="analyst",
            port=2222,
            identity_file="/private/synthetic/id",
        )

        self.assertEqual(
            template,
            {
                "profiles": {
                    "generic": {
                        "host": "login.research.invalid",
                        "user": "analyst",
                        "port": 2222,
                        "identity_file": "/private/synthetic/id",
                    }
                }
            },
        )
        rendered = repr(template).lower()
        for forbidden in ("alliance", "nibi", "mfa", "robot", "scratch", "module", "apptainer", "container"):
            self.assertNotIn(forbidden, rendered)

    def test_generic_profile_builder_rejects_invalid_port_and_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 through 65535"):
            build_ssh_profile_entry(host="cluster.invalid", user="analyst", port=0)
        with self.assertRaisesRegex(ValueError, "control characters"):
            build_ssh_profile_entry(host="cluster.invalid\nProxyJump=other", user="analyst")
        for host, user in (
            ("cluster.example.org", "analyst"),
            ("${SSH_HOST}", "analyst"),
            ("cluster.invalid", "your-username"),
            ("<ssh-host>", "analyst"),
        ):
            with self.subTest(host=host, user=user), self.assertRaisesRegex(ValueError, "starter placeholder"):
                build_ssh_profile_entry(host=host, user=user)

    def test_lower_level_generic_builder_rejects_unsafe_identity_grammar(self) -> None:
        invalid_cases = (
            {"profile_name": "-oProxyCommand", "host": "login.synthetic.invalid", "user": "analyst"},
            {"profile_name": "unsafe:name", "host": "login.synthetic.invalid", "user": "analyst"},
            {"profile_name": "safe", "host": "-oProxyJump=other", "user": "analyst"},
            {"profile_name": "safe", "host": "login synthetic.invalid", "user": "analyst"},
            {"profile_name": "safe", "host": "login.synthetic.invalid", "user": "-oProxyCommand"},
            {"profile_name": "safe", "host": "login.synthetic.invalid", "user": "analyst:other"},
        )
        for case in invalid_cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                build_ssh_config_template(
                    profile_name=case["profile_name"],
                    host=case["host"],
                    user=case["user"],
                )

        for field, value in (
            ("identity_file", "-oProxyCommand=other"),
            ("identity_file", "${SSH_KEY}"),
            ("known_hosts_file", "change-me"),
            ("known_hosts_file", "known\nhosts"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build_ssh_config_template(
                    profile_name="safe",
                    host="login.synthetic.invalid",
                    user="analyst",
                    **{field: value},
                )

    def test_reviewed_ssh_option_allowlist_is_case_insensitive_and_non_routing(self) -> None:
        allowed = {
            "ConnectTimeout": 10,
            "ServerAliveInterval": 30,
            "serveralivecountmax": 3,
            "TCPKeepAlive": True,
            "PreferredAuthentications": "publickey,keyboard-interactive",
            "ControlMaster": "auto",
            "ControlPath": "~/.ssh/cm-%C",
            "ControlPersist": "2h",
        }
        entry = build_ssh_profile_entry(
            host="login.synthetic.invalid",
            user="analyst",
            options=allowed,
        )
        self.assertEqual(entry["options"], allowed)

        forbidden = (
            "HostName",
            "USER",
            "port",
            "IdentityFile",
            "userknownhostsfile",
            "BatchMode",
            "strictHOSTkeyCHECKING",
            "ProxyCommand",
            "proxyjump",
            "LocalCommand",
            "PermitLocalCommand",
            "RemoteCommand",
            "KnownHostsCommand",
            "LocalForward",
            "RemoteForward",
            "DynamicForward",
            "Include",
            "PKCS11Provider",
            "SecurityKeyProvider",
        )
        for option in forbidden:
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "not permitted"):
                build_ssh_profile_entry(
                    host="login.synthetic.invalid",
                    user="analyst",
                    options={option: "unsafe"},
                )

    def test_profile_upsert_preserves_unrelated_content_and_force_is_selected_only(self) -> None:
        original = {
            "defaults": {"options": {"ServerAliveInterval": 30}},
            "profiles": {
                "unrelated": {"host": "other.invalid", "user": "other"},
                "selected": {"host": "old.invalid", "user": "analyst"},
            },
        }
        replacement = {"host": "new.invalid", "user": "analyst"}

        with self.assertRaisesRegex(ValueError, "use --force"):
            upsert_ssh_profile_document(
                original,
                profile_name="selected",
                profile=replacement,
            )

        updated = upsert_ssh_profile_document(
            original,
            profile_name="selected",
            profile=replacement,
            force=True,
        )

        self.assertEqual(updated["profiles"]["selected"], replacement)
        self.assertEqual(updated["profiles"]["unrelated"], original["profiles"]["unrelated"])
        self.assertEqual(updated["defaults"], original["defaults"])
        self.assertEqual(original["profiles"]["selected"]["host"], "old.invalid")

    def test_generic_profile_isolation_rejects_inherited_global_defaults(self) -> None:
        with self.assertRaisesRegex(ValueError, "would be inherited implicitly"):
            require_generic_profile_isolation(
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
                }
            )

        require_generic_profile_isolation(
            {
                "defaults": {},
                "profiles": {
                    "unrelated": {
                        "host": "other.synthetic.invalid",
                        "user": "other",
                    }
                },
            }
        )

    def test_alliance_profile_materialization_is_self_contained(self) -> None:
        template = build_ssh_config_template(
            "alliance",
            profile_name="site-reviewed",
            host="login.synthetic.invalid",
            user="analyst",
            identity_file="/private/synthetic/id",
        )

        profile = materialize_ssh_profile_entry(template, profile_name="site-reviewed")

        self.assertEqual(profile["defaults"]["host"], "login.synthetic.invalid")
        self.assertEqual(profile["defaults"]["user"], "analyst")
        self.assertEqual(profile["defaults"]["identity_file"], "/private/synthetic/id")
        self.assertEqual(profile["defaults"]["options"]["ControlMaster"], "auto")
        self.assertIn("robot", profile["roles"])

    def test_load_profile_merges_defaults_and_role_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "ssh.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "defaults:",
                        "  user: shared-user",
                        "  identity_file: ~/.ssh/id_shared",
                        "  options:",
                        "    ServerAliveInterval: 30",
                        "profiles:",
                        "  cluster-a:",
                        "    defaults:",
                        "      host: cluster.example",
                        "      options:",
                        "        PreferredAuthentications: publickey",
                        "    roles:",
                        "      login:",
                        "        options:",
                        "          ControlMaster: no",
                        "      robot:",
                        "        user: robot-user",
                        "        identity_file: ~/.ssh/id_robot",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            login = load_ssh_profile(config_path, "cluster-a", role="login")
            robot = load_ssh_profile(config_path, "cluster-a", role="robot")

        self.assertEqual(login.target(), "shared-user@cluster.example")
        self.assertEqual(login.options["ServerAliveInterval"], "30")
        self.assertEqual(login.options["PreferredAuthentications"], "publickey")
        self.assertEqual(login.options["ControlMaster"], "no")
        self.assertEqual(robot.target(), "robot-user@cluster.example")
        self.assertTrue(robot.identity_file.endswith(".ssh/id_robot"))

    def test_expand_env_placeholders_recurses_through_nested_mappings_and_lists(self) -> None:
        payload = {
            "profiles": {
                "site": {
                    "defaults": {
                        "host": "${SSH_HOST}",
                        "identity_file": "${SSH_KEY:-~/.ssh/id_default}",
                        "options": {
                            "ProxyJump": "${SSH_JUMP:-jump.example}",
                        },
                    },
                    "roles": {
                        "robot": {
                            "user": "${SSH_ROBOT_USER:-${SSH_USER:-robot}}",
                            "metadata": ["${SSH_HOST}", "${SSH_MISSING:-fallback-value}"],
                        }
                    },
                }
            }
        }

        expanded = expand_env_placeholders(payload, env={"SSH_HOST": "cluster.example", "SSH_USER": "alice"})

        self.assertEqual(expanded["profiles"]["site"]["defaults"]["host"], "cluster.example")
        self.assertEqual(expanded["profiles"]["site"]["roles"]["robot"]["user"], "alice")
        self.assertEqual(expanded["profiles"]["site"]["roles"]["robot"]["metadata"], ["cluster.example", "fallback-value"])

    def test_environment_expansion_detects_mutual_and_default_cycles(self) -> None:
        for value, environment in (
            ("${FIRST}", {"FIRST": "${SECOND}", "SECOND": "${FIRST}"}),
            ("${FIRST:-${SECOND:-${FIRST}}}", {}),
            ("${FIRST}", {"FIRST": "${SECOND:-${FIRST}}"}),
        ):
            with self.subTest(value=value, environment=environment), self.assertRaisesRegex(
                ValueError,
                "Cyclic environment placeholder expansion",
            ):
                expand_env_placeholders(value, env=environment)

    def test_environment_expansion_handles_valid_nested_defaults_and_unresolved_values(self) -> None:
        self.assertEqual(
            expand_env_placeholders(
                "${FIRST:-${SECOND:-fallback}}/${THIRD}",
                env={"THIRD": "value"},
            ),
            "fallback/value",
        )
        self.assertEqual(expand_env_placeholders("prefix-${MISSING}-suffix", env={}), "prefix--suffix")
        self.assertEqual(
            expand_env_placeholders("${FIRST}", env={"FIRST": "${SECOND}", "SECOND": "resolved"}),
            "resolved",
        )

    def test_resolve_ssh_profile_config_path_prefers_explicit_and_env_before_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            explicit_path = Path(tmp_dir) / "explicit.yaml"
            env_path = Path(tmp_dir) / "env.yaml"
            explicit_path.write_text("profiles: {}\n", encoding="utf-8")
            env_path.write_text("profiles: {}\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"RESEARCH_HPC_SSH_CONFIG": str(env_path), "RP_SSH_CONFIG": str(Path(tmp_dir) / "legacy.yaml")},
                clear=False,
            ):
                self.assertEqual(resolve_ssh_profile_config_path(explicit_path), explicit_path.resolve())
                self.assertEqual(resolve_ssh_profile_config_path(None), env_path.resolve())

    def test_resolve_ssh_profile_config_path_anchors_relative_candidates_to_workspace_root(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            subdirectory = workspace_root / "project" / "demo"
            config_path = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            subdirectory.mkdir(parents=True, exist_ok=True)
            (workspace_root / "WORKSPACE.yaml").write_text("projects:\n  default: demo\n", encoding="utf-8")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("profiles: {}\n", encoding="utf-8")
            os.chdir(subdirectory)
            try:
                with patch.dict(os.environ, {"RESEARCH_HPC_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml"}, clear=False):
                    self.assertEqual(
                        resolve_ssh_profile_config_path(None, workspace_root=workspace_root),
                        config_path.resolve(),
                    )
                    self.assertEqual(
                        resolve_ssh_profile_config_path("secrets/hpc/ssh-profiles.yaml", workspace_root=workspace_root),
                        config_path.resolve(),
                    )
            finally:
                os.chdir(original_cwd)

    def test_resolve_ssh_profile_config_path_uses_repo_fallback_when_present(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fallback_path = Path(tmp_dir) / "secrets" / "hpc" / "ssh-profiles.yaml"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_path.write_text("profiles: {}\n", encoding="utf-8")
            os.chdir(tmp_dir)
            try:
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(resolve_ssh_profile_config_path(None).resolve(), fallback_path.resolve())
            finally:
                os.chdir(original_cwd)

    def test_render_ssh_shell_uses_batch_mode_and_alias_when_config_host_is_defined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "ssh.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  automation:",
                        "    ssh_config_host: automation-login",
                        "    user: automation-user",
                        "    options:",
                        "      ConnectTimeout: 10",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            profile = load_ssh_profile(config_path, "automation")

        shell = render_ssh_shell(profile, mode="batch")
        self.assertIn("ssh", shell)
        self.assertIn("BatchMode=yes", shell)
        self.assertIn("StrictHostKeyChecking=yes", shell)
        self.assertIn("ConnectTimeout=10", shell)
        self.assertEqual(profile.target(), "automation-user@automation-login")
        command = build_ssh_command(profile, mode="batch", remote_command="true")
        self.assertEqual(command[-2:], ["automation-user@automation-login", "true"])

    def test_command_rendering_rejects_forbidden_options_even_for_direct_profiles(self) -> None:
        from research_platform.hpc.ssh_profiles import SshProfile

        for option in ("proxyjump", "HOSTNAME", "LocalCommand", "BatchMode"):
            profile = SshProfile(
                name="unsafe",
                host="login.synthetic.invalid",
                user="analyst",
                options={option: "unsafe"},
            )
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "not permitted"):
                build_ssh_command(profile, mode="batch", remote_command="true")

        for profile in (
            SshProfile(name="unsafe-host", host="-oProxyCommand=other", user="analyst"),
            SshProfile(name="unsafe-alias", ssh_config_host="alias value", user="analyst"),
            SshProfile(name="unsafe-user", host="login.synthetic.invalid", user="--proxy"),
        ):
            with self.subTest(profile=profile.name), self.assertRaises(ValueError):
                build_ssh_command(profile, mode="batch", remote_command="true")

    def test_list_profiles_reports_flat_and_family_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "ssh.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  flat:",
                        "    host: flat.example",
                        "  site:",
                        "    defaults:",
                        "      host: site.example",
                        "    roles:",
                        "      login: {}",
                        "      robot: {}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            catalog = list_ssh_profiles(config_path)

        self.assertEqual(
            catalog,
            [
                {"name": "flat", "kind": "flat", "roles": ["login"]},
                {"name": "site", "kind": "family", "roles": ["login", "robot"]},
            ],
        )


if __name__ == "__main__":
    unittest.main()
