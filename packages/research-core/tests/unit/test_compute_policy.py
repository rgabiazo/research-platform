from __future__ import annotations

from pathlib import Path
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.core.compute import parse_ram_gb, resolve_resource_plan


class ComputePolicyTests(unittest.TestCase):
    def test_legacy_slurm_bids_workload_uses_allocated_cpus_as_threads(self) -> None:
        resources = resolve_resource_plan(
            compute_config={
                "local": {"jobs": 1},
                "slurm": {"cpus": 4, "mem": "16G"},
            },
            workload="bids_preprocess",
            mode="slurm",
        )
        self.assertEqual(
            resources,
            {
                "cpus": 4,
                "ram_gb": 16,
                "threads": 4,
                "n_jobs": 1,
                "workload": "bids_preprocess",
                "preset": "legacy",
            },
        )

    def test_resolve_resource_plan_uses_legacy_local_jobs_for_tabular_plan_mode(self) -> None:
        resources = resolve_resource_plan(
            compute_config={
                "local": {"jobs": 2},
                "slurm": {"cpus": 8, "mem": "24G"},
            },
            workload="tabular_train_model",
            mode="plan",
        )
        self.assertEqual(resources["cpus"], 2)
        self.assertEqual(resources["threads"], 1)
        self.assertEqual(resources["n_jobs"], 2)
        self.assertEqual(resources["ram_gb"], 24)
        self.assertEqual(resources["preset"], "legacy")

    def test_resolve_resource_plan_uses_policy_preset_and_workload_override(self) -> None:
        resources = resolve_resource_plan(
            compute_config={
                "local": {"jobs": 1},
                "slurm": {"cpus": 4, "mem": "16G"},
                "policy": {
                    "default_preset": "balanced",
                    "presets": {
                        "balanced": {"cpus": 4, "ram_gb": 16, "threads": 2, "n_jobs": 1},
                    },
                    "workloads": {
                        "tabular_train_model": {"n_jobs": 2},
                    },
                },
            },
            workload="tabular_train_model",
            mode="slurm",
        )
        self.assertEqual(
            resources,
            {
                "cpus": 4,
                "ram_gb": 16,
                "threads": 2,
                "n_jobs": 2,
                "workload": "tabular_train_model",
                "preset": "balanced",
            },
        )

    def test_resolve_resource_plan_rejects_threads_above_cpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "threads <= cpus"):
            resolve_resource_plan(
                compute_config={
                    "policy": {
                        "default_preset": "bad",
                        "presets": {
                            "bad": {"cpus": 2, "ram_gb": 4, "threads": 3, "n_jobs": 1},
                        },
                    },
                },
                workload="bids_preprocess",
                mode="plan",
            )

    def test_resolve_resource_plan_rejects_cpus_below_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "cpus >= 1"):
            resolve_resource_plan(
                compute_config={
                    "policy": {
                        "default_preset": "bad",
                        "presets": {
                            "bad": {"cpus": 0, "ram_gb": 4, "threads": 1, "n_jobs": 1},
                        },
                    },
                },
                workload="bids_preprocess",
                mode="plan",
            )

    def test_resolve_resource_plan_rejects_ram_below_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "ram_gb >= 1"):
            resolve_resource_plan(
                compute_config={
                    "policy": {
                        "default_preset": "bad",
                        "presets": {
                            "bad": {"cpus": 1, "ram_gb": 0, "threads": 1, "n_jobs": 1},
                        },
                    },
                },
                workload="bids_preprocess",
                mode="plan",
            )

    def test_resolve_resource_plan_allows_n_jobs_above_per_unit_cpus(self) -> None:
        resources = resolve_resource_plan(
            compute_config={
                "policy": {
                    "default_preset": "parallel",
                    "presets": {
                        "parallel": {"cpus": 2, "ram_gb": 4, "threads": 1, "n_jobs": 3},
                    },
                },
            },
            workload="tabular_preprocess",
            mode="plan",
        )

        self.assertEqual(resources["cpus"], 2)
        self.assertEqual(resources["threads"], 1)
        self.assertEqual(resources["n_jobs"], 3)

    def test_resolve_resource_plan_accepts_matching_policy_and_slurm_memory(self) -> None:
        resources = resolve_resource_plan(
            compute_config={
                "local": {"jobs": 1},
                "slurm": {"cpus": 4, "mem": "16384M"},
                "policy": {
                    "default_preset": "balanced",
                    "presets": {
                        "balanced": {"cpus": 4, "ram_gb": 16, "threads": 2, "n_jobs": 1},
                    },
                    "workloads": {
                        "bids_preprocess": {"preset": "balanced"},
                    },
                },
            },
            workload="bids_preprocess",
            mode="slurm",
        )
        self.assertEqual(resources["ram_gb"], 16)
        self.assertEqual(resources["preset"], "balanced")

    def test_resolve_resource_plan_rejects_policy_and_slurm_memory_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "compute\\.slurm\\.mem.*compute\\.policy\\.presets\\.balanced\\.ram_gb",
        ):
            resolve_resource_plan(
                compute_config={
                    "local": {"jobs": 1},
                    "slurm": {"cpus": 4, "mem": "32G"},
                    "policy": {
                        "default_preset": "balanced",
                        "presets": {
                            "balanced": {"cpus": 4, "ram_gb": 16, "threads": 2, "n_jobs": 1},
                        },
                        "workloads": {
                            "bids_preprocess": {"preset": "balanced"},
                        },
                    },
                },
                workload="bids_preprocess",
                mode="slurm",
            )

    def test_parse_ram_gb_accepts_legacy_slurm_memory_strings(self) -> None:
        self.assertEqual(parse_ram_gb("4G"), 4)
        self.assertEqual(parse_ram_gb("16GB"), 16)
        self.assertEqual(parse_ram_gb("4096M"), 4)


if __name__ == "__main__":
    unittest.main()
