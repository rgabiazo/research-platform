from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
CORE_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-core"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))

from research_platform.neuro.fsl.feat.adapter import FeatAnalysisAdapter


class FeatAnalysisAdapterTests(unittest.TestCase):
    def test_scaffold_project_defaults_make_slurm_container_ready_without_tool_profile_modules(self) -> None:
        defaults = FeatAnalysisAdapter().scaffold_project_defaults(
            project_name="project-demo",
            study_root="/tmp/study",
            derivative_root="/tmp/derivatives/fmriprep",
            task_id="rest",
        )

        self.assertEqual(defaults["runtime_profile"], "fsl")
        self.assertEqual(defaults["compute"]["tool_profiles"]["fsl"]["local"]["execution_backend"], "native")
        self.assertEqual(defaults["compute"]["tool_profiles"]["fsl"]["slurm"]["execution_backend"], "apptainer")
        self.assertNotIn("modules", defaults["compute"]["tool_profiles"]["fsl"]["slurm"])
        self.assertNotIn("pre_activate_commands", defaults["compute"]["tool_profiles"]["fsl"]["slurm"])
        self.assertEqual(
            defaults["compute"]["tool_profiles"]["fsl"]["slurm"]["container"]["image"],
            "${RP_FSL_CONTAINER_IMAGE:-docker://vnmd/fsl_6.0.7.4:latest}",
        )
        self.assertEqual(
            defaults["compute"]["tool_profiles"]["fsl"]["slurm"]["container"]["image_name"],
            "${RP_FSL_CONTAINER_IMAGE_NAME:-fsl_6.0.7.4.sif}",
        )
        self.assertEqual(
            defaults["compute"]["tool_profiles"]["fsl"]["slurm"]["container"]["image_root"],
            "${RP_REMOTE_CONTAINER_ROOT:-$SCRATCH/containers/fsl}",
        )
        self.assertEqual(
            defaults["compute"]["slurm"]["pre_activate_commands"],
            ['[ -n "$SCRATCH" ] || { echo "ERROR: SCRATCH is not set on the remote node." >&2; exit 1; }'],
        )
        self.assertEqual(
            defaults["compute"]["slurm"]["prepare_directories"],
            ["$APPTAINER_CACHEDIR", "$APPTAINER_TMPDIR", "$TMPDIR"],
        )
        self.assertEqual(defaults["stage"]["validation"]["empty_ev_policy"], "as_zero")
        self.assertEqual(defaults["template"], "generic")

    def test_fmripost_aroma_template_sets_external_roots_and_patterns(self) -> None:
        defaults = FeatAnalysisAdapter().scaffold_project_defaults(
            project_name="project-demo",
            study_root="/tmp/study",
            derivative_root="/tmp/study/derivatives/fmripost_aroma",
            task_id="exampletask",
            template="fmripost-aroma-first-level",
            events_root="/tmp/study/derivatives/evs",
            remote_events_root="/remote/study/derivatives/evs",
        )

        self.assertEqual(defaults["template"], "fmripost-aroma-first-level")
        self.assertEqual(defaults["external_input_roots"]["evs"]["local_root"], "/tmp/study/derivatives/evs")
        self.assertEqual(defaults["external_input_roots"]["evs"]["remote_root"], "/remote/study/derivatives/evs")
        self.assertEqual(defaults["external_input_roots"]["feat_confounds"]["local_root"], "/tmp/study/derivatives/evs")
        self.assertEqual(defaults["external_input_roots"]["feat_confounds"]["remote_root"], "/remote/study/derivatives/evs")
        self.assertTrue(defaults["inputs"]["confounds"]["required"])
        self.assertEqual(defaults["inputs"]["confounds"]["root_ref"], "feat_confounds")
        self.assertEqual(defaults["inputs"]["evs"]["root_ref"], "evs")
        self.assertIn("desc-nonaggrDenoised_bold.nii.gz", "\n".join(defaults["inputs"]["bold"]["patterns"]))
        self.assertIn("desc-confounds_noGSR.txt", "\n".join(defaults["inputs"]["confounds"]["patterns"]))
        self.assertEqual(defaults["stage"]["settings"]["norm"], 1)

    def test_auto_template_selects_fmripost_aroma_when_events_root_is_supplied(self) -> None:
        defaults = FeatAnalysisAdapter().scaffold_project_defaults(
            project_name="project-demo",
            study_root="/tmp/study",
            derivative_root="/tmp/study/derivatives/fmripost_aroma",
            task_id="exampletask",
            template="auto",
            events_root="/tmp/study/derivatives/evs",
        )

        self.assertEqual(defaults["template"], "fmripost-aroma-first-level")

    def test_deepprep_t1w_template_sets_t1w_inputs_and_zero_smoothing(self) -> None:
        defaults = FeatAnalysisAdapter().scaffold_project_defaults(
            project_name="project-demo",
            study_root="/tmp/study",
            derivative_root="/tmp/study/derivatives/DeepPrep/BOLD",
            task_id="exampletask",
            template="deepprep-t1w-first-level",
            events_root="/tmp/study/derivatives/evs",
            remote_events_root="/remote/study/derivatives/evs",
        )

        self.assertEqual(defaults["template"], "deepprep-t1w-first-level")
        self.assertEqual(defaults["external_input_roots"]["evs"]["local_root"], "/tmp/study/derivatives/evs")
        self.assertEqual(defaults["external_input_roots"]["feat_confounds"]["local_root"], "/tmp/study/derivatives/evs")
        self.assertTrue(defaults["inputs"]["confounds"]["required"])
        self.assertEqual(defaults["inputs"]["confounds"]["root_ref"], "feat_confounds")
        self.assertEqual(defaults["inputs"]["evs"]["root_ref"], "evs")
        self.assertIn("space-T1w_desc-preproc_bold.nii.gz", "\n".join(defaults["inputs"]["bold"]["patterns"]))
        self.assertIn("desc-confounds_noGSR.txt", "\n".join(defaults["inputs"]["confounds"]["patterns"]))
        self.assertEqual(defaults["stage"]["settings"]["smooth_mm"], 0.0)
        self.assertEqual(defaults["stage"]["settings"]["norm"], 0)

    def test_auto_template_selects_deepprep_t1w_when_deepprep_bold_root_is_supplied(self) -> None:
        defaults = FeatAnalysisAdapter().scaffold_project_defaults(
            project_name="project-demo",
            study_root="/tmp/study",
            derivative_root="/tmp/study/derivatives/DeepPrep/BOLD",
            task_id="exampletask",
            template="auto",
            events_root="/tmp/study/derivatives/evs",
        )

        self.assertEqual(defaults["template"], "deepprep-t1w-first-level")

    def test_discover_batch_rows_returns_deterministic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            derivative_root = Path(tmp_dir) / "derivatives"
            func_root = derivative_root / "sub-002" / "ses-01" / "func"
            func_root.mkdir(parents=True, exist_ok=True)
            (
                func_root
                / "sub-002_ses-01_task-memory_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")
            (
                func_root
                / "sub-002_ses-01_task-memory_run-02_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")

            rows = FeatAnalysisAdapter().discover_batch_rows(
                derivative_root=str(derivative_root),
                selectors={"subject_id": "sub-002", "session_id": "ses-01", "task_id": None, "run_id": None},
                context={
                    "analysis_inputs": {
                        "bold": {
                            "patterns": [
                                "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
                            ]
                        }
                    }
                },
            )

        self.assertEqual(
            rows,
            [
                {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-01"},
                {"subject_id": "sub-002", "session_id": "ses-01", "task_id": "task-memory", "run_id": "run-02"},
            ],
        )

    def test_build_runtime_plan_writes_fsf_and_runtime_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            func_root = derivative_root / "sub-001" / "ses-01" / "func"
            func_root.mkdir(parents=True, exist_ok=True)
            bold_path = (
                func_root
                / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
            bold_path.write_text("", encoding="utf-8")

            ev_root = workspace_root / "datasets" / "project-demo" / "derivatives" / "events" / "sub-001" / "ses-01" / "func"
            ev_root.mkdir(parents=True, exist_ok=True)
            for ev_name in ("condition_a", "condition_b", "button_press"):
                (
                    ev_root / f"sub-001_ses-01_task-rest_run-01_desc-{ev_name}_events.txt"
                ).write_text("0 1 1\n", encoding="utf-8")

            batch_path = workspace_root / "project" / "project-demo" / "manifests" / "batches" / "feat_first_level.tsv"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n", encoding="utf-8")

            manifest = {
                "batch": {"path": str(batch_path.relative_to(workspace_root))},
                "dataset": {
                    "root": "datasets/project-demo",
                    "derivative_root": str(derivative_root.relative_to(workspace_root)),
                    "input_derivative": "fmriprep",
                },
                "execution": {
                    "mode": "local",
                    "output_dir": "artifacts/runs/feat-plan/outputs",
                    "work_dir": "artifacts/runs/feat-plan/work",
                },
                "analysis": {
                    "stage": "first_level",
                    "inputs": {
                        "bold": {
                            "patterns": [
                                "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
                            ]
                        },
                        "evs": {
                            "root": "datasets/project-demo/derivatives/events",
                            "patterns": [
                                "{ev_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt"
                            ],
                        },
                    },
                    "stage_config": {
                        "validation": {"require_confounds": False, "allow_missing_evs": False},
                        "overwrite": {"design": False, "results": False},
                        "settings": {"tr": None, "hpf": 100.0, "smooth_mm": 5.0},
                    },
                    "model_ref": "task_glm",
                    "model": {
                        "name": "task_glm",
                        "ev_order": ["condition_a", "condition_b", "button_press"],
                        "derivative_on": ["condition_a", "condition_b"],
                        "nonconvolved": ["button_press"],
                        "contrasts": [{"name": "condition_a_gt_baseline", "weights": [1, 0, 0]}],
                    },
                },
                "tool": {
                    "runtime_metadata": {"output_data_dirname": "fsl_feat"},
                    "runtime_profile": {
                        "name": "fsl",
                        "config": {"local": {"environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"}}},
                    },
                },
            }
            plan_path = workspace_root / "artifacts" / "runs" / "feat-plan" / "outputs" / "fsl-feat-plan.json"
            command_script_path = workspace_root / "artifacts" / "runs" / "feat-plan" / "outputs" / "run-fsl-feat.sh"

            with (
                mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)),
                mock.patch("research_platform.neuro.fsl.feat.runtime.preflight_feat_model", return_value=(True, "feat_model ok")),
            ):
                plan = FeatAnalysisAdapter().build_runtime_plan(
                    manifest=manifest,
                    workspace_root=str(workspace_root),
                    plan_path=str(plan_path),
                    command_script_path=str(command_script_path),
                )

            fsf_root = workspace_root / "artifacts" / "runs" / "feat-plan" / "outputs" / "fsf"
            fsf_files = list(fsf_root.rglob("*.fsf"))
            plan_exists = plan_path.exists()
            command_script_exists = command_script_path.exists()
            fsf_text = fsf_files[0].read_text(encoding="utf-8") if fsf_files else ""

        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(plan["steps"][0]["env"]["FSLOUTPUTTYPE"], "NIFTI_GZ")
        self.assertEqual(plan["steps"][0]["env"]["FSL_FEAT_WATCH"], "0")
        self.assertEqual(plan["steps"][0]["env"]["BROWSER"], "false")
        self.assertTrue(plan_exists)
        self.assertTrue(command_script_exists)
        self.assertEqual(len(fsf_files), 1)
        self.assertIn("set fmri(outputdir)", fsf_text)

    def test_expected_remote_input_files_uses_external_analysis_input_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            event_root = workspace_root / "external" / "events"
            func_root = derivative_root / "sub-001" / "ses-01" / "func"
            event_func_root = event_root / "sub-001" / "ses-01" / "func"
            func_root.mkdir(parents=True, exist_ok=True)
            event_func_root.mkdir(parents=True, exist_ok=True)
            (
                func_root
                / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")
            (
                event_func_root / "sub-001_ses-01_task-rest_run-01_desc-confounds_noGSR.txt"
            ).write_text("0 0\n", encoding="utf-8")
            for ev_name in ("condition_a", "condition_b", "button_press"):
                (
                    event_func_root / f"sub-001_ses-01_task-rest_run-01_desc-{ev_name}_events.txt"
                ).write_text("0 1 1\n", encoding="utf-8")

            paths = FeatAnalysisAdapter().expected_remote_input_files(
                derivative_root=str(derivative_root),
                remote_derivative_root="/remote/derivatives/fmriprep",
                row={
                    "subject_id": "sub-001",
                    "session_id": "ses-01",
                    "task_id": "task-rest",
                    "run_id": "run-01",
                },
                context={
                    "workspace_root": workspace_root,
                    "dataset_root": workspace_root / "external" / "study",
                    "remote_dataset_root": "/remote/study",
                    "compute": {"slurm": {"remote_workspace_root": "/remote/workspace"}},
                    "analysis_inputs": {
                        "bold": {
                            "patterns": [
                                "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
                            ]
                        },
                        "confounds": {
                            "root_ref": "feat_confounds",
                            "patterns": [
                                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-confounds_noGSR.txt"
                            ],
                        },
                        "evs": {
                            "root_ref": "evs",
                            "patterns": [
                                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt"
                            ],
                        },
                    },
                    "analysis_input_roots": {
                        "evs": {
                            "path": event_root,
                            "remote_root": "/remote/analysis/events",
                        },
                        "feat_confounds": {
                            "path": event_root,
                            "remote_root": "/remote/analysis/events",
                        },
                    },
                    "analysis_model": {
                        "ev_order": ["condition_a", "condition_b", "button_press"],
                    },
                    "data_roots": [
                        {
                            "label": "evs-root",
                            "path": event_root,
                            "remote_root": "/remote/analysis/events",
                        }
                    ],
                },
            )

        self.assertIn(
            "/remote/analysis/events/sub-001/ses-01/func/sub-001_ses-01_task-rest_run-01_desc-confounds_noGSR.txt",
            paths,
        )
        self.assertIn(
            "/remote/analysis/events/sub-001/ses-01/func/sub-001_ses-01_task-rest_run-01_desc-condition_a_events.txt",
            paths,
        )
        self.assertIn(
            "/remote/derivatives/fmriprep/sub-001/ses-01/func/sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
            paths,
        )


if __name__ == "__main__":
    unittest.main()
