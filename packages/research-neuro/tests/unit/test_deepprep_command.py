from __future__ import annotations

import shlex
from pathlib import Path
import sys
import tempfile
import unittest

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CORE_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-core"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.neuro.deepprep.command import build_runtime_plan


class DeepPrepCommandTests(unittest.TestCase):
    def test_build_runtime_plan_renders_apptainer_command_with_pinned_image_and_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            batch_path = workspace_root / "project" / "manifests" / "batches" / "deepprep.tsv"
            batch_path.parent.mkdir(parents=True)
            batch_path.write_text("subject_id\ttask_id\nsub-synthetic02\ttask-exampletask\n", encoding="utf-8")
            (workspace_root / "datasets" / "example-bids").mkdir(parents=True)
            output_root = workspace_root / "artifacts" / "runs" / "deepprep-synthetic02" / "outputs"
            work_root = workspace_root / "artifacts" / "runs" / "deepprep-synthetic02" / "work"
            manifest = {
                "batch": {"path": "project/manifests/batches/deepprep.tsv"},
                "dataset": {"root": "datasets/example-bids"},
                "execution": {
                    "mode": "slurm",
                    "output_dir": "artifacts/runs/deepprep-synthetic02/outputs",
                    "work_dir": "artifacts/runs/deepprep-synthetic02/work",
                },
                "resources": {"cpus": 4},
                "selection": {"subject_id": "synthetic02", "task_id": "exampletask"},
                "tool": {
                    "inputs": {"fs_license_file": "secrets/freesurfer/license.txt", "remote_fs_license_file": "/scratch/licenses/license.txt"},
                    "output": {"unit_dir_template": "{subject_id}-{task_id}"},
                    "options": {
                        "bold_task_type": "exampletask",
                        "bold_surface_spaces": "fsnative",
                        "bold_volume_space": "MNI152NLin6Asym",
                        "bold_volume_res": "02",
                        "bold_sdc": True,
                        "bold_confounds": True,
                        "bold_skip_frame": 0,
                        "device": "cpu",
                    },
                    "runtime_metadata": {"output_data_dirname": "deepprep_units"},
                    "runtime_profile": {
                        "name": "deepprep",
                        "config": {
                            "slurm": {
                                "execution_backend": "apptainer",
                                "container": {
                                    "image": "docker://pbfslab/deepprep:25.1.0",
                                    "pull_mode": "if_missing",
                                    "image_root": "/scratch/containers/deepprep",
                                    "image_name": "deepprep_25.1.0.sif",
                                },
                            }
                        },
                    },
                },
            }

            plan = build_runtime_plan(
                manifest=manifest,
                workspace_root=workspace_root,
                plan_path=output_root / "deepprep-plan.json",
                command_script_path=output_root / "run-deepprep.sh",
            )

        self.assertEqual(plan["image"]["reference"], "docker://pbfslab/deepprep:25.1.0")
        self.assertEqual(plan["image"]["runtime_image"], "/scratch/containers/deepprep/deepprep_25.1.0.sif")
        self.assertEqual(plan["resources"], {"cpus": 4, "memory_gb": 32})
        self.assertEqual(plan["steps"][0]["unit_id"], "sub-synthetic02-exampletask")
        self.assertEqual(plan["steps"][0]["output_dir"], str((output_root / "deepprep_units" / "sub-synthetic02-exampletask").resolve()))
        self.assertIsNotNone(plan["container_prep"])
        command_shell = plan["steps"][0]["command"][2]
        self.assertIn("apptainer run", command_shell)
        self.assertIn("/scratch/containers/deepprep/deepprep_25.1.0.sif", command_shell)
        self.assertIn("--home \"$DEEPPREP_HOME_HOST:$DEEPPREP_HOME_DEST\"", command_shell)
        self.assertIn("NEXTFLOW_JAR_SOURCE", command_shell)
        self.assertIn("nextflow-24.10.3-one.jar", command_shell)
        self.assertNotIn("APPTAINERENV_HOME", command_shell)
        self.assertIn("--fs_license_file /fs_license.txt", command_shell)
        quoted_label = shlex.quote("synthetic02")
        self.assertIn(f"--participant_label {quoted_label}", command_shell)
        self.assertIn("--bold_task_type exampletask", command_shell)
        self.assertIn("--bold_surface_spaces fsnative", command_shell)
        self.assertIn("--cpus 4", command_shell)
        self.assertIn("--memory 32", command_shell)

    def test_build_runtime_plan_uses_prestaged_apptainer_sif_without_container_pull(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            batch_path = workspace_root / "project" / "manifests" / "batches" / "deepprep.tsv"
            batch_path.parent.mkdir(parents=True)
            batch_path.write_text("subject_id\ttask_id\nsub-synthetic02\ttask-exampletask\n", encoding="utf-8")
            (workspace_root / "datasets" / "example-bids").mkdir(parents=True)
            output_root = workspace_root / "artifacts" / "runs" / "deepprep-synthetic02" / "outputs"
            manifest = {
                "batch": {"path": "project/manifests/batches/deepprep.tsv"},
                "dataset": {"root": "datasets/example-bids"},
                "execution": {
                    "mode": "slurm",
                    "output_dir": "artifacts/runs/deepprep-synthetic02/outputs",
                    "work_dir": "artifacts/runs/deepprep-synthetic02/work",
                },
                "resources": {"cpus": 4},
                "selection": {"subject_id": "synthetic02", "task_id": "exampletask"},
                "tool": {
                    "inputs": {"remote_fs_license_file": "/scratch/licenses/license.txt"},
                    "options": {"bold_task_type": "exampletask", "device": "cpu"},
                    "runtime_metadata": {"output_data_dirname": "deepprep_units"},
                    "runtime_profile": {
                        "name": "deepprep",
                        "config": {
                            "slurm": {
                                "execution_backend": "apptainer",
                                "container": {
                                    "image": "$SCRATCH/containers/deepprep/deepprep_25.1.0.sif",
                                    "pull_mode": "never",
                                },
                            }
                        },
                    },
                },
            }

            plan = build_runtime_plan(
                manifest=manifest,
                workspace_root=workspace_root,
                plan_path=output_root / "deepprep-plan.json",
                command_script_path=output_root / "run-deepprep.sh",
            )

        self.assertIsNone(plan["container_prep"])
        self.assertEqual(plan["image"]["runtime_image"], "$SCRATCH/containers/deepprep/deepprep_25.1.0.sif")
        self.assertEqual(plan["nextflow"]["version"], "24.10.3")
        self.assertIn("$SCRATCH/containers/deepprep/deepprep_25.1.0.sif", plan["steps"][0]["command"][2])
        self.assertIn("Missing pre-staged Nextflow jar", plan["steps"][0]["command"][2])


if __name__ == "__main__":
    unittest.main()
