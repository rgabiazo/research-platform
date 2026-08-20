from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CORE_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-core"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.neuro.deepprep.adapter import DeepPrepAdapter


class DeepPrepAdapterTests(unittest.TestCase):
    def test_scaffold_defaults_use_raw_bids_and_pinned_configurable_container(self) -> None:
        defaults = DeepPrepAdapter().scaffold_project_defaults(
            project_name="project-study-deepprep",
            study_root="datasets/study-bids",
            derivative_root=None,
            task_id="exampletask",
        )

        self.assertNotIn("input_derivative", defaults)
        self.assertEqual(defaults["runtime_profile"], "deepprep")
        self.assertEqual(defaults["slurm_profile"], "local")
        self.assertEqual(defaults["inputs"]["fs_license_file"], "${FS_LICENSE_FILE:-secrets/freesurfer/license.txt}")
        self.assertEqual(defaults["tool_options"]["bold_task_type"], "exampletask")
        self.assertEqual(defaults["tool_options"]["bold_surface_spaces"], "fsnative")
        self.assertEqual(defaults["compute"]["policy"]["presets"]["deepprep"]["ram_gb"], 32)
        self.assertEqual(defaults["compute"]["slurm"]["mem"], "32G")
        self.assertEqual(defaults["compute"]["slurm"]["environment"]["XDG_DATA_HOME"], "$SCRATCH/.local/share")
        self.assertEqual(defaults["compute"]["slurm"]["environment"]["APPTAINER_CONFIGDIR"], "$SCRATCH/apptainer-config")
        self.assertIn("$XDG_DATA_HOME", defaults["compute"]["slurm"]["prepare_directories"])
        self.assertIn("$APPTAINER_CONFIGDIR", defaults["compute"]["slurm"]["prepare_directories"])
        nextflow = defaults["compute"]["tool_profiles"]["deepprep"]["slurm"]["nextflow"]
        self.assertTrue(nextflow["enabled"])
        self.assertEqual(nextflow["version"], "24.10.3")
        self.assertEqual(nextflow["host_home"], "${RP_DEEPPREP_NEXTFLOW_HOME:-$SCRATCH/deepprep/nextflow}")
        container = defaults["compute"]["tool_profiles"]["deepprep"]["slurm"]["container"]
        self.assertEqual(container["source_image"], "${RP_DEEPPREP_CONTAINER_SOURCE_IMAGE:-docker://pbfslab/deepprep:25.1.0}")
        self.assertEqual(container["image"], "${RP_DEEPPREP_CONTAINER_IMAGE:-docker://pbfslab/deepprep:25.1.0}")
        self.assertEqual(container["image_name"], "${RP_DEEPPREP_CONTAINER_IMAGE_NAME:-deepprep_25.1.0.sif}")

    def test_validate_project_accepts_raw_bids_without_input_derivative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            bids_root = workspace_root / "datasets" / "study-bids"
            bids_root.mkdir(parents=True)
            bundle = {
                "project_root": str(workspace_root / "project" / "project-study-deepprep"),
                "dataset": {"dataset": {"primary": "study-bids", "bids_root": str(bids_root)}},
                "preprocessing": {
                    "preprocessing": {
                        "slice": "bids",
                        "pipeline": "preprocess-bids",
                        "tool": "deepprep",
                        "tool_adapter": "research_platform.neuro.deepprep.adapter:DeepPrepAdapter",
                        "default_batch": "deepprep_default",
                        "inputs": {"fs_license_file": "secrets/freesurfer/license.txt"},
                        "tool_options": {"bold_task_type": "exampletask"},
                    }
                },
            }
            pipeline_defaults = {
                "workflow": {"default_target": "bids_preprocess", "rule_name": "bids_preprocess", "execution_rule_name": "bids_preprocess_unit"},
                "planner": {"outputs": {}},
            }

            errors = DeepPrepAdapter().validate_project(
                bundle=bundle,
                pipeline_defaults=pipeline_defaults,
                workspace_root=str(workspace_root),
            )

        self.assertEqual(errors, [])

    def test_validate_project_rejects_unknown_option_and_mutually_exclusive_modes(self) -> None:
        bundle = {
            "dataset": {"dataset": {"primary": "study-bids", "bids_root": "datasets/study-bids"}},
            "preprocessing": {
                "preprocessing": {
                    "tool": "deepprep",
                    "inputs": {"fs_license_file": "secrets/freesurfer/license.txt"},
                    "tool_options": {"anat_only": True, "bold_only": True, "cpus": 8},
                }
            },
        }
        pipeline_defaults = {"workflow": {"default_target": "bids_preprocess"}, "planner": {"outputs": {}}}

        errors = DeepPrepAdapter().validate_project(
            bundle=bundle,
            pipeline_defaults=pipeline_defaults,
            workspace_root="workspace",
        )

        self.assertIn("Unsupported preprocessing.tool_options for deepprep: cpus.", errors)
        self.assertIn("preprocessing.tool_options.anat_only and bold_only cannot both be true.", errors)


if __name__ == "__main__":
    unittest.main()
