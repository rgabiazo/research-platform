from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc.ssh_profiles import SshProfile
from research_platform.hpc.transfers import build_rsync_pull_command, build_rsync_push_command


class RsyncModeTests(unittest.TestCase):
    def test_alias_plus_user_is_preserved_in_push_and_pull_remote_endpoints(self) -> None:
        profile = SshProfile(
            name="alias",
            ssh_config_host="configured-alias",
            user="configured-user",
        )

        push = build_rsync_push_command(
            source="artifacts/runs/unit",
            profile=profile,
            destination="remote/run",
            mode="batch",
        )
        pull = build_rsync_pull_command(
            profile=profile,
            source="remote/run",
            destination="artifacts/pulled",
            mode="batch",
        )

        self.assertEqual(push[-1], "configured-user@configured-alias:remote/run/")
        self.assertEqual(pull[-2], "configured-user@configured-alias:remote/run/")

    def test_batch_push_command_uses_batch_safe_ssh_shell_and_excludes(self) -> None:
        profile = SshProfile(name="automation", host="cluster.example", user="runner", options={"ConnectTimeout": "10"})
        with tempfile.TemporaryDirectory() as tmp_dir:
            exclude_path = Path(tmp_dir) / "exclude.txt"
            exclude_path.write_text("scratch/\n", encoding="utf-8")

            command = build_rsync_push_command(
                source="artifacts/runs/unit",
                profile=profile,
                destination="remote/run",
                mode="batch",
                exclude_file=exclude_path,
            )

        rendered = " ".join(command)
        self.assertIn("rsync -az -e", rendered)
        self.assertIn("BatchMode=yes", rendered)
        self.assertIn("StrictHostKeyChecking=yes", rendered)
        self.assertIn("--exclude-from", command)
        self.assertEqual(command[-1], "runner@cluster.example:remote/run/")

    def test_interactive_pull_command_uses_mfa_friendly_ssh_shell(self) -> None:
        profile = SshProfile(name="interactive", host="cluster.example", user="alice")
        command = build_rsync_pull_command(
            profile=profile,
            source="remote/run",
            destination="artifacts/pulled",
            mode="interactive",
        )

        rendered = " ".join(command)
        self.assertIn("BatchMode=no", rendered)
        self.assertIn("StrictHostKeyChecking=ask", rendered)
        self.assertEqual(command[-2], "alice@cluster.example:remote/run/")
        self.assertEqual(command[-1], "artifacts/pulled/")

    def test_pull_command_renders_optional_progress_flag(self) -> None:
        profile = SshProfile(name="interactive", host="cluster.example", user="alice")
        command = build_rsync_pull_command(
            profile=profile,
            source="remote/run",
            destination="artifacts/pulled",
            mode="interactive",
            progress=True,
        )

        self.assertIn("--progress", command)

    def test_push_command_renders_optional_dry_run_progress_and_delete_flags(self) -> None:
        profile = SshProfile(name="automation", host="cluster.example", user="runner")
        command = build_rsync_push_command(
            source="artifacts/runs/unit",
            profile=profile,
            destination="remote/run",
            mode="batch",
            dry_run=True,
            progress=True,
            itemize_changes=True,
            delete=True,
        )

        self.assertIn("--dry-run", command)
        self.assertIn("--progress", command)
        self.assertIn("--itemize-changes", command)
        self.assertIn("--delete", command)


if __name__ == "__main__":
    unittest.main()
