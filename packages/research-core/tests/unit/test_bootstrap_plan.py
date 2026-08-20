from __future__ import annotations

from pathlib import Path
import sys
import unittest

CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))

from research_platform.core.bootstrap import build_bootstrap_manifest


class BootstrapPlanTests(unittest.TestCase):
    def test_build_bootstrap_manifest_returns_none_when_bootstrap_is_unset(self) -> None:
        manifest = {
            "run_id": "unit",
            "hpc": {
                "remote_workspace_root": "remote/workspace",
                "remote_artifacts_root": "remote/workspace/artifacts",
                "remote_run_root": "remote/workspace/artifacts/runs/unit",
            },
            "provision": {"selected_scopes": ["common", "tabular-ml"]},
        }
        context = {"compute": {"slurm": {}}}

        bootstrap = build_bootstrap_manifest(context=context, manifest=manifest)

        self.assertIsNone(bootstrap)

    def test_build_bootstrap_manifest_selects_only_common_and_tabular_hooks_for_tabular_runs(self) -> None:
        manifest = {
            "run_id": "unit",
            "hpc": {
                "remote_workspace_root": "remote/workspace",
                "remote_artifacts_root": "remote/workspace/artifacts",
                "remote_run_root": "remote/workspace/artifacts/runs/unit",
            },
            "provision": {"selected_scopes": ["common", "tabular-ml"]},
        }
        context = {
            "compute": {
                "slurm": {
                    "bootstrap": {
                        "enabled": True,
                        "hooks": {
                            "common": [{"name": "base-env", "kind": "python-env", "command": "python3 -m venv ~/.venvs/rp"}],
                            "neuro-bids": [{"name": "templateflow", "kind": "cache-prefetch", "command": "echo skip"}],
                            "tabular-ml": [{"name": "wheel-cache", "kind": "cache-prefetch", "command": "python3 -m pip cache dir"}],
                        },
                    }
                }
            }
        }

        bootstrap = build_bootstrap_manifest(context=context, manifest=manifest)

        self.assertIsNotNone(bootstrap)
        assert bootstrap is not None
        self.assertEqual(bootstrap["selected_scopes"], ["common", "tabular-ml"])
        self.assertEqual([entry["label"] for entry in bootstrap["remote_directories"]], ["remote-workspace-root", "remote-artifacts-root", "remote-run-root"])
        scopes = {scope["name"]: scope["hooks"] for scope in bootstrap["hook_scopes"]}
        self.assertEqual([hook["name"] for hook in scopes["common"]], ["base-env"])
        self.assertEqual([hook["name"] for hook in scopes["tabular-ml"]], ["wheel-cache"])
        self.assertNotIn("neuro-bids", bootstrap["selected_scopes"])

    def test_build_bootstrap_manifest_selects_common_and_neuro_hooks_for_bids_runs(self) -> None:
        manifest = {
            "run_id": "unit",
            "hpc": {
                "remote_workspace_root": "remote/workspace",
                "remote_artifacts_root": "remote/workspace/artifacts",
                "remote_run_root": "remote/workspace/artifacts/runs/unit",
            },
            "provision": {"selected_scopes": ["common", "neuro-bids"]},
        }
        context = {
            "compute": {
                "slurm": {
                    "bootstrap": {
                        "enabled": True,
                        "hooks": {
                            "common": [{"name": "base-env", "command": "python3 --version"}],
                            "neuro-bids": [{"name": "container-cache", "kind": "container", "command": "apptainer cache list"}],
                            "tabular-ml": [{"name": "ml-cache", "command": "echo unused"}],
                        },
                    }
                }
            }
        }

        bootstrap = build_bootstrap_manifest(context=context, manifest=manifest)

        self.assertIsNotNone(bootstrap)
        assert bootstrap is not None
        self.assertEqual(bootstrap["selected_scopes"], ["common", "neuro-bids"])
        scopes = {scope["name"]: scope["hooks"] for scope in bootstrap["hook_scopes"]}
        self.assertEqual(scopes["neuro-bids"][0]["kind"], "container")
        self.assertNotIn("tabular-ml", scopes)


if __name__ == "__main__":
    unittest.main()
