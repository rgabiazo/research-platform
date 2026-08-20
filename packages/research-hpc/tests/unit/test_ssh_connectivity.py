from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc.ssh import analyze_ssh_failure, run_ssh_connectivity_check
from research_platform.hpc.ssh_profiles import SshProfile


class SshConnectivityTests(unittest.TestCase):
    def test_alias_plus_user_is_preserved_in_connectivity_command_and_report(self) -> None:
        calls: list[list[str]] = []

        def batch_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        profile = SshProfile(
            name="alias",
            ssh_config_host="configured-alias",
            user="configured-user",
        )
        report = run_ssh_connectivity_check(
            profile,
            mode="batch",
            batch_runner=batch_runner,
        )

        self.assertEqual(report["target"], "configured-user@configured-alias")
        self.assertEqual(calls[0][-2:], ["configured-user@configured-alias", "true"])

    def test_auto_mode_runs_batch_probe_first_and_returns_success_without_fallback(self) -> None:
        calls: list[list[str]] = []

        def batch_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

        profile = SshProfile(name="login", host="cluster.example", user="alice")
        report = run_ssh_connectivity_check(profile, mode="auto", batch_runner=batch_runner, interactive_available=False)

        self.assertTrue(report["ok"])
        self.assertEqual(report["mode_used"], "batch")
        self.assertFalse(report["fallback_to_interactive"])
        self.assertEqual(len(calls), 1)
        self.assertIn("BatchMode=yes", " ".join(calls[0]))

    def test_auto_mode_can_fall_back_to_interactive_for_host_key_acceptance_or_mfa(self) -> None:
        batch_calls: list[list[str]] = []
        interactive_calls: list[list[str]] = []

        def batch_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            batch_calls.append(command)
            return subprocess.CompletedProcess(command, 255, stdout="", stderr="Host key verification failed\n")

        def interactive_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            interactive_calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        profile = SshProfile(name="login", host="cluster.example", user="alice")
        report = run_ssh_connectivity_check(
            profile,
            mode="auto",
            batch_runner=batch_runner,
            interactive_runner=interactive_runner,
            interactive_available=True,
        )

        self.assertTrue(report["fallback_to_interactive"])
        self.assertTrue(report["interactive_attempted"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["mode_used"], "interactive")
        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(len(interactive_calls), 1)
        self.assertIn("-tt", interactive_calls[0])

    def test_host_key_mismatch_emits_fix_guidance_without_interactive_retry(self) -> None:
        report = analyze_ssh_failure("@@@ REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\nOffending key in ~/.ssh/known_hosts:7\n")
        self.assertEqual(report["failure_type"], "host_key_mismatch")
        self.assertFalse(report["should_retry_interactive"])
        self.assertIn("known_hosts", report["host_key_fix_guidance"])


if __name__ == "__main__":
    unittest.main()
