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

from research_platform.neuro.fsl.feat.runtime import _feat_output_parent, _feat_output_stem, build_runtime_plan


class FeatRuntimeBackendTests(unittest.TestCase):
    def test_build_runtime_plan_keeps_native_feat_command_and_headless_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest, plan_path, command_script_path = _build_manifest(
                workspace_root=workspace_root,
                mode="local",
                runtime_profile={
                    "local": {
                        "execution_backend": "native",
                        "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                    }
                },
            )

            with (
                mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)),
                mock.patch("research_platform.neuro.fsl.feat.runtime.preflight_feat_model", return_value=(True, "feat_model ok")),
            ):
                plan = build_runtime_plan(
                    manifest=manifest,
                    workspace_root=str(workspace_root),
                    plan_path=str(plan_path),
                    command_script_path=str(command_script_path),
                )

        step = plan["steps"][0]
        self.assertEqual(step["command"], ["feat", step["fsf_path"]])
        self.assertEqual(step["backend"], "native")
        self.assertEqual(step["env"]["FSLOUTPUTTYPE"], "NIFTI_GZ")
        self.assertEqual(step["env"]["FSL_FEAT_WATCH"], "0")
        self.assertEqual(step["env"]["BROWSER"], "false")

    def test_build_runtime_plan_wraps_apptainer_backend_with_pull_and_external_binds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest, plan_path, command_script_path = _build_manifest(
                workspace_root=workspace_root,
                mode="slurm",
                runtime_profile={
                    "slurm": {
                        "execution_backend": "apptainer",
                        "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                        "container": {
                            "enabled": True,
                            "backend": "apptainer",
                            "image": "docker://ghcr.io/example/fsl:6.0.7",
                            "pull_mode": "if_missing",
                            "image_name": "fsl-feat",
                            "image_root": "$SCRATCH/containers/fsl",
                        },
                    }
                },
            )

            with (
                mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)),
                mock.patch("research_platform.neuro.fsl.feat.runtime.preflight_feat_model", return_value=(True, "feat_model ok")),
            ):
                plan = build_runtime_plan(
                    manifest=manifest,
                    workspace_root=str(workspace_root),
                    plan_path=str(plan_path),
                    command_script_path=str(command_script_path),
                )

        step = plan["steps"][0]
        self.assertEqual(step["command"][:2], ["bash", "-lc"])
        self.assertEqual(step["backend"], "apptainer")
        self.assertIsNotNone(plan["container_prep"])
        prep_command = plan["container_prep"]["command"]
        self.assertEqual(prep_command[:2], ["bash", "-lc"])
        self.assertIn('export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SLURM_TMPDIR:-$SCRATCH/apptainer-tmp}}"', prep_command[2])
        self.assertIn('apptainer pull "$TMP_IMAGE" "$IMAGE_SOURCE"', prep_command[2])
        self.assertIn('rm -rf "${RUNTIME_IMAGE}.lock.d"', prep_command[2])
        self.assertIn("exec apptainer exec --cleanenv", step["command"][2])
        self.assertNotIn("apptainer pull", step["command"][2])
        self.assertIn('$SCRATCH/containers/fsl/fsl-feat.sif', step["command"][2])
        self.assertIn(
            f"--bind {(workspace_root.parent / f'{workspace_root.name}-events').resolve()}:{(workspace_root.parent / f'{workspace_root.name}-events').resolve()}",
            step["command"][2],
        )
        self.assertIn(
            f"--bind {workspace_root.resolve()}:{workspace_root.resolve()}",
            step["command"][2],
        )
        self.assertEqual(step["env"]["BROWSER"], "false")
        self.assertEqual(step["env"]["FSL_FEAT_WATCH"], "0")

    def test_build_runtime_plan_supports_singularity_with_prebuilt_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest, plan_path, command_script_path = _build_manifest(
                workspace_root=workspace_root,
                mode="slurm",
                runtime_profile={
                    "slurm": {
                        "execution_backend": "singularity",
                        "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                        "container": {
                            "enabled": True,
                            "backend": "singularity",
                            "image": "/shared/containers/fsl-feat.sif",
                            "pull_mode": "never",
                        },
                    }
                },
            )

            with (
                mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)),
                mock.patch("research_platform.neuro.fsl.feat.runtime.preflight_feat_model", return_value=(True, "feat_model ok")),
            ):
                plan = build_runtime_plan(
                    manifest=manifest,
                    workspace_root=str(workspace_root),
                    plan_path=str(plan_path),
                    command_script_path=str(command_script_path),
                )

        step = plan["steps"][0]
        self.assertEqual(step["command"][:2], ["bash", "-lc"])
        self.assertIn("exec singularity exec --cleanenv", step["command"][2])
        self.assertNotIn("singularity pull", step["command"][2])
        self.assertIsNone(plan["container_prep"])
        self.assertIn("/shared/containers/fsl-feat.sif", step["command"][2])

    def test_build_runtime_plan_nests_feat_outputs_by_subject_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest, plan_path, command_script_path = _build_manifest(
                workspace_root=workspace_root,
                mode="local",
                runtime_profile={
                    "local": {
                        "execution_backend": "native",
                        "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                    }
                },
            )

            with (
                mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)),
                mock.patch("research_platform.neuro.fsl.feat.runtime.preflight_feat_model", return_value=(True, "feat_model ok")),
            ):
                plan = build_runtime_plan(
                    manifest=manifest,
                    workspace_root=str(workspace_root),
                    plan_path=str(plan_path),
                    command_script_path=str(command_script_path),
                )

        step = plan["steps"][0]
        self.assertIn("/outputs/fsl_feat/sub-001/ses-01/sub-001_ses-01_task-rest_run-01.feat", step["output_dir"])
        self.assertIn("/outputs/fsf/sub-001/ses-01/sub-001_ses-01_task-rest_run-01.fsf", step["fsf_path"])

    def test_build_runtime_plan_appends_output_desc_to_feat_and_fsf_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest, plan_path, command_script_path = _build_manifest(
                workspace_root=workspace_root,
                mode="local",
                runtime_profile={
                    "local": {
                        "execution_backend": "native",
                        "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                    }
                },
                outputs={"desc": "modelA"},
            )

            with (
                mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)),
                mock.patch("research_platform.neuro.fsl.feat.runtime.preflight_feat_model", return_value=(True, "feat_model ok")),
            ):
                plan = build_runtime_plan(
                    manifest=manifest,
                    workspace_root=str(workspace_root),
                    plan_path=str(plan_path),
                    command_script_path=str(command_script_path),
                )

        step = plan["steps"][0]
        self.assertIn("/outputs/fsl_feat/sub-001/ses-01/sub-001_ses-01_task-rest_run-01_desc-modelA.feat", step["output_dir"])
        self.assertIn("/outputs/fsf/sub-001/ses-01/sub-001_ses-01_task-rest_run-01_desc-modelA.fsf", step["fsf_path"])

    def test_build_runtime_plan_renders_zero_smoothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest, plan_path, command_script_path = _build_manifest(
                workspace_root=workspace_root,
                mode="local",
                runtime_profile={
                    "local": {
                        "execution_backend": "native",
                        "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                    }
                },
                settings={"tr": None, "hpf": 100.0, "smooth_mm": 0.0},
            )

            with (
                mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)),
                mock.patch("research_platform.neuro.fsl.feat.runtime.preflight_feat_model", return_value=(True, "feat_model ok")),
            ):
                plan = build_runtime_plan(
                    manifest=manifest,
                    workspace_root=str(workspace_root),
                    plan_path=str(plan_path),
                    command_script_path=str(command_script_path),
                )
            fsf_text = Path(plan["steps"][0]["fsf_path"]).read_text(encoding="utf-8")

        self.assertIn("set fmri(smooth) 0.0", fsf_text)

    def test_feat_output_parent_omits_session_segment_when_missing(self) -> None:
        parent = _feat_output_parent(
            row={"subject_id": "sub-001", "session_id": ""},
            entities={"subject_id": "sub-001"},
        )

        self.assertEqual(parent, Path("sub-001"))

    def test_feat_output_stem_appends_optional_desc(self) -> None:
        self.assertEqual(_feat_output_stem("sub-001_task-rest_run-01", output_desc=None), "sub-001_task-rest_run-01")
        self.assertEqual(
            _feat_output_stem("sub-001_task-rest_run-01", output_desc="modelA"),
            "sub-001_task-rest_run-01_desc-modelA",
        )

    def test_build_runtime_plan_renders_empty_ev_as_zero_and_disables_derivative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest, plan_path, command_script_path = _build_manifest(
                workspace_root=workspace_root,
                mode="local",
                runtime_profile={
                    "local": {
                        "execution_backend": "native",
                        "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                    }
                },
                empty_ev_names={"condition_a"},
                validation={"require_confounds": False, "allow_missing_evs": False, "empty_ev_policy": "as_zero"},
            )

            with (
                mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)),
                mock.patch("research_platform.neuro.fsl.feat.runtime.preflight_feat_model", return_value=(True, "feat_model ok")),
            ):
                plan = build_runtime_plan(
                    manifest=manifest,
                    workspace_root=str(workspace_root),
                    plan_path=str(plan_path),
                    command_script_path=str(command_script_path),
                )
            fsf_text = Path(plan["steps"][0]["fsf_path"]).read_text(encoding="utf-8")

        self.assertIn("set fmri(evs_real) 4", fsf_text)
        self.assertIn("set fmri(shape1) 10", fsf_text)
        self.assertIn("set fmri(convolve1) 0", fsf_text)
        self.assertIn("set fmri(deriv_yn1) 0", fsf_text)
        self.assertNotIn("set fmri(custom1)", fsf_text)
        self.assertIn("set fmri(shape2) 3", fsf_text)
        self.assertIn("set fmri(deriv_yn2) 1", fsf_text)
        self.assertIn("set fmri(con_real1.1) 1.0", fsf_text)
        self.assertIn("set fmri(con_real1.4) 0.0", fsf_text)
        self.assertNotIn("set fmri(con_real1.5)", fsf_text)

    def test_build_runtime_plan_rejects_empty_ev_when_policy_is_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest, plan_path, command_script_path = _build_manifest(
                workspace_root=workspace_root,
                mode="local",
                runtime_profile={
                    "local": {
                        "execution_backend": "native",
                        "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                    }
                },
                empty_ev_names={"condition_a"},
                validation={"require_confounds": False, "allow_missing_evs": False, "empty_ev_policy": "fail"},
            )

            with mock.patch("research_platform.neuro.fsl.feat.runtime.infer_nvols_and_tr", return_value=(120, 2.0)):
                with self.assertRaises(ValueError) as exc_info:
                    build_runtime_plan(
                        manifest=manifest,
                        workspace_root=str(workspace_root),
                        plan_path=str(plan_path),
                        command_script_path=str(command_script_path),
                    )

        self.assertIn("EV file is empty", str(exc_info.exception))


def _build_manifest(
    *,
    workspace_root: Path,
    mode: str,
    runtime_profile: dict[str, object],
    empty_ev_names: set[str] | None = None,
    validation: dict[str, object] | None = None,
    outputs: dict[str, object] | None = None,
    settings: dict[str, object] | None = None,
) -> tuple[dict[str, object], Path, Path]:
    dataset_root = workspace_root / "datasets" / "project-demo"
    derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
    events_root = workspace_root.parent / f"{workspace_root.name}-events"
    func_root = derivative_root / "sub-001" / "ses-01" / "func"
    event_func_root = events_root / "sub-001" / "ses-01" / "func"
    func_root.mkdir(parents=True, exist_ok=True)
    event_func_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    (
        func_root
        / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    ).write_text("", encoding="utf-8")
    empty_ev_names = empty_ev_names or set()
    for ev_name in ("condition_a", "condition_b", "button_press"):
        (
            event_func_root / f"sub-001_ses-01_task-rest_run-01_desc-{ev_name}_events.txt"
        ).write_text("" if ev_name in empty_ev_names else "0 1 1\n", encoding="utf-8")

    batch_path = workspace_root / "project" / "project-demo" / "manifests" / "batches" / "feat_first_level.tsv"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text("subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n", encoding="utf-8")

    output_dir = workspace_root / "artifacts" / "runs" / f"feat-{mode}" / "outputs"
    plan_path = output_dir / "fsl-feat-plan.json"
    command_script_path = output_dir / "run-fsl-feat.sh"

    manifest: dict[str, object] = {
        "batch": {"path": str(batch_path.relative_to(workspace_root))},
        "dataset": {
            "name": "project-demo",
            "root": str(dataset_root.relative_to(workspace_root)),
            "derivative_root": str(derivative_root.relative_to(workspace_root)),
        },
        "execution": {
            "mode": mode,
            "output_dir": str(output_dir.relative_to(workspace_root)) if mode == "local" else str(output_dir),
            "work_dir": str((workspace_root / "artifacts" / "runs" / f"feat-{mode}" / "work").relative_to(workspace_root))
            if mode == "local"
            else str(workspace_root / "artifacts" / "runs" / f"feat-{mode}" / "work"),
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
                    "root_ref": "evs",
                    "patterns": [
                        "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt"
                    ],
                },
            },
            "input_roots": {
                "evs": {
                    "path": str(events_root),
                    "remote_root": "/remote/analysis/events",
                    "sync_enabled": True,
                }
            },
            "stage_config": {
                "validation": validation or {"require_confounds": False, "allow_missing_evs": False},
                "overwrite": {"design": False, "results": False},
                "settings": settings or {"tr": None, "hpf": 100.0, "smooth_mm": 5.0},
                "outputs": outputs or {},
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
                "config": runtime_profile,
            },
        },
    }
    return manifest, plan_path, command_script_path


if __name__ == "__main__":
    unittest.main()
