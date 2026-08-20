from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.hpc.remote import verify_remote_paths


class RemoteVerifyTests(unittest.TestCase):
    def test_verify_remote_paths_resolves_profile_role_and_parses_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "ssh.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  site:",
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
            calls: list[list[str]] = []

            def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="present\t/remote/a\nmissing\t/remote/b\n",
                    stderr="",
                )

            report = verify_remote_paths(
                paths=["/remote/a", "/remote/b", "/remote/a"],
                profile_name="site",
                role="robot",
                config_path=config_path,
                runner=runner,
            )

        self.assertEqual(report["returncode"], 0)
        self.assertFalse(report["ok"])
        self.assertEqual(
            report["paths"],
            [
                {"path": "/remote/a", "exists": True},
                {"path": "/remote/b", "exists": False},
            ],
        )
        self.assertEqual(report["connection"]["profile"], "site")
        self.assertEqual(report["connection"]["role"], "robot")
        self.assertEqual(report["connection"]["target"], "robot-user@cluster.example")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "ssh")
        self.assertIn("BatchMode=yes", " ".join(calls[0]))


if __name__ == "__main__":
    unittest.main()
