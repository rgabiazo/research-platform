from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
    import nibabel as nib
except ImportError:  # pragma: no cover - local minimal environments may skip.
    np = None  # type: ignore[assignment]
    nib = None  # type: ignore[assignment]

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
BIDS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-bids"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(BIDS_PACKAGE_ROOT / "src"))

from research_platform.neuro.fsl.flame import (
    build_flame1_command_plan,
    execute_flame1_command_plan,
    write_one_sample_group_mean_design,
)
from research_platform.neuro.roi_execution import (
    RoiExecutionContext,
    plan_roi_build,
    preflight_roi_build,
    run_roi_build,
)
from research_platform.neuro.roi import (
    RoiDefinition,
    normalize_sidecar_provenance,
    validate_portable_provenance_paths,
    validate_roi_sidecar_document,
)
from research_platform.neuro.roi_loso import (
    build_loso_group_map_cache_key,
    discover_loso_group_mask_inputs,
    discover_subject_fixed_effects_inputs,
    execute_loso_build_action,
    load_loso_roi_set_config,
    materialize_generated_group_mask,
    plan_loso_group_map_jobs,
)


def fake_find_loso_fsl_tool(tool: str, **_: object) -> str | None:
    if tool in {"fslmerge", "flameo"}:
        return f"/mock/fsl/bin/{tool}"
    return None


class RoiLosoPhase4Tests(unittest.TestCase):
    def test_loso_config_validation_and_fixed_effects_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document()
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"))

            roi_set = load_loso_roi_set_config(document)
            inputs = discover_subject_fixed_effects_inputs(document, context=context)
            jobs = plan_loso_group_map_jobs(document, context=context)

        self.assertEqual(roi_set.name, "loso_demo")
        self.assertEqual(len(inputs), 3)
        self.assertTrue(all(item.complete for item in inputs))
        self.assertEqual(len(jobs), 1)
        self.assertEqual([item.subject_id for item in jobs[0].training_inputs], ["002", "003"])
        self.assertEqual(jobs[0].heldout_input.subject_id, "001")

    def test_direct_subject_ffx_feat_layout_resolves_fixed_effects_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document()
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["direction"] = "AP"  # type: ignore[index]
            roi_set["fixed_effects_inputs"] = {  # type: ignore[index]
                "root_ref": "derivative_root",
                "cope_dir": (
                    "{subject_dir}/{session_dir}/"
                    "{subject_dir}_{session_dir}_task-{task_id}_dir-{direction}_desc-{model}_contrast-{contrast_id}.feat"
                ),
                "cope_image": "cope1.nii.gz",
                "varcope_image": "varcope1.nii.gz",
                "mask_image": "mask.nii.gz",
            }
            for subject in ("001", "002", "003"):
                ffx_dir = (
                    root
                    / "derivatives"
                    / f"sub-{subject}"
                    / "ses-01"
                    / f"sub-{subject}_ses-01_task-memory_dir-AP_desc-ModelA_contrast-CondA.feat"
                )
                ffx_dir.mkdir(parents=True, exist_ok=True)
                (ffx_dir / "cope1.nii.gz").write_text("cope", encoding="utf-8")
                (ffx_dir / "varcope1.nii.gz").write_text("varcope", encoding="utf-8")
                (ffx_dir / "mask.nii.gz").write_text("mask", encoding="utf-8")

            inputs = discover_subject_fixed_effects_inputs(document, context=context)

        self.assertEqual(len(inputs), 3)
        self.assertTrue(all(item.complete for item in inputs))
        self.assertTrue(inputs[0].cope_path.as_posix().endswith("contrast-CondA.feat/cope1.nii.gz"))
        self.assertTrue(inputs[0].varcope_path.as_posix().endswith("contrast-CondA.feat/varcope1.nii.gz"))
        self.assertTrue(inputs[0].mask_path.as_posix().endswith("contrast-CondA.feat/mask.nii.gz"))

    def test_fsl_gfeat_nested_layout_resolves_cope_level_feat_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=2)
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["direction"] = "AP"  # type: ignore[index]
            roi_set["model"] = "AlltrialsLocalizerFFX"  # type: ignore[index]
            roi_set["fixed_effects_inputs"] = {  # type: ignore[index]
                "layout": "fsl_gfeat_nested",
                "root_ref": "derivative_root",
                "gfeat_dir": "{subject_dir}/{session_dir}/{subject_dir}_{session_dir}_task-{task_id}_dir-{direction}_desc-{model}.gfeat",
            }
            _write_nested_gfeat_tree(
                root,
                subjects=("001", "002", "003"),
                model="AlltrialsLocalizerFFX",
                cope_numbers=("1",),
            )

            inputs = discover_subject_fixed_effects_inputs(document, context=context)
            plan = plan_roi_build(document, context=context)

        self.assertEqual(len(inputs), 3)
        self.assertTrue(all(item.complete for item in inputs))
        self.assertTrue(
            inputs[0].cope_path.as_posix().endswith(
                "sub-001_ses-01_task-memory_dir-AP_desc-AlltrialsLocalizerFFX.gfeat/cope1.feat/stats/cope1.nii.gz"
            )
        )
        self.assertTrue(inputs[0].varcope_path.as_posix().endswith("cope1.feat/stats/varcope1.nii.gz"))
        self.assertTrue(inputs[0].mask_path.as_posix().endswith("cope1.feat/mask.nii.gz"))
        self.assertEqual(len(plan.actions), 1)
        training_input = plan.actions[0].metadata["loso_group_job"]["training_inputs"][0]
        self.assertIn("AlltrialsLocalizerFFX.gfeat/cope1.feat/stats/cope1.nii.gz", training_input["cope_path"])

    def test_generated_group_mask_strategy_plans_loso_training_subject_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=2)
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["group_mask"] = {  # type: ignore[index]
                "strategy": "fixed_effects_mask_intersection",
                "scope": "loso_training_subjects",
                "root_ref": "artifacts_root",
                "path": (
                    "generated-masks/{heldout_subject_dir}/"
                    "group_{session_dir}_task-{task_id}_desc-{contrast_desc}_heldout-sub{heldout_subject_id}_mask.nii.gz"
                ),
            }
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"))

            jobs = plan_loso_group_map_jobs(document, context=context)
            masks = discover_loso_group_mask_inputs(document, context=context)
            generated = jobs[0].generated_group_mask

        self.assertIsNotNone(generated)
        assert generated is not None
        self.assertEqual(len(masks), 1)
        self.assertTrue(masks[0].generated)
        self.assertEqual(masks[0].missing, ())
        self.assertEqual(masks[0].source_mask_paths, generated.source_mask_paths)
        self.assertEqual(generated.strategy, "fixed_effects_mask_intersection")
        self.assertEqual(generated.scope, "loso_training_subjects")
        self.assertEqual(generated.included_subjects, ("002", "003"))
        self.assertEqual(generated.excluded_subjects, ("001",))
        self.assertTrue(generated.mask_path.as_posix().endswith("heldout-sub001_mask.nii.gz"))
        self.assertEqual(len(generated.source_mask_paths), 2)
        self.assertEqual(jobs[0].group_mask_path, generated.mask_path)
        self.assertNotIn("group mask is missing", "\n".join(jobs[0].warnings))

    def test_generated_group_mask_collision_across_distinct_jobs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=1)
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["held_out_subjects"] = ["sub-001", "sub-002"]  # type: ignore[index]
            roi_set["group_mask"] = {  # type: ignore[index]
                "strategy": "fixed_effects_mask_intersection",
                "root_ref": "artifacts_root",
                "path": "generated-masks/shared_mask.nii.gz",
            }
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"))

            report = preflight_roi_build(document, context=context)

        self.assertFalse(report.ready_for_execution)
        self.assertTrue(any("duplicate destinations" in message for message in report.errors))

    def test_generated_group_mask_still_reports_missing_subject_fixed_effects_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=1)
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["group_mask"] = {  # type: ignore[index]
                "strategy": "fixed_effects_mask_intersection",
                "root_ref": "artifacts_root",
                "path": "generated-masks/{heldout_subject_dir}/{contrast_desc}_mask.nii.gz",
            }
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), missing={("002", "mask")})

            jobs = plan_loso_group_map_jobs(document, context=context)

        warnings = "\n".join(jobs[0].warnings)
        self.assertIn("sub-002 missing mask", warnings)
        self.assertEqual([item.subject_id for item in jobs[0].training_inputs], ["003"])
        self.assertEqual(jobs[0].generated_group_mask.included_subjects, ("003",))

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for generated mask execution tests")
    def test_generated_group_mask_execution_intersects_source_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=1)
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["group_mask"] = {  # type: ignore[index]
                "strategy": "fixed_effects_mask_intersection",
                "root_ref": "artifacts_root",
                "path": "generated-masks/{heldout_subject_dir}/{contrast_desc}_mask.nii.gz",
            }
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), nifti_masks=True)
            sub002_mask = np.zeros((5, 5, 5), dtype=np.uint8)
            sub002_mask[1, 1, 1] = 1
            sub002_mask[2, 2, 2] = 1
            sub003_mask = np.zeros((5, 5, 5), dtype=np.uint8)
            sub003_mask[2, 2, 2] = 1
            nib.save(_image(sub002_mask), root / "derivatives" / "sub-002" / "ses-01" / "func" / "task-memory_model-ModelA_contrast-CondA" / "mask.nii.gz")
            nib.save(_image(sub003_mask), root / "derivatives" / "sub-003" / "ses-01" / "func" / "task-memory_model-ModelA_contrast-CondA" / "mask.nii.gz")
            job = plan_loso_group_map_jobs(document, context=context)[0]

            output_path = materialize_generated_group_mask(job.to_dict(), root_refs=context.root_refs)
            output = nib.load(str(output_path)).get_fdata().astype(np.uint8)
            sidecar = json.loads(job.generated_group_mask.sidecar_path.read_text(encoding="utf-8"))

        self.assertEqual(int(output.sum()), 1)
        self.assertEqual(int(output[2, 2, 2]), 1)
        self.assertEqual(sidecar["generation_policy"], "fixed_effects_mask_intersection")
        self.assertEqual(sidecar["included_subjects"], ["sub-002", "sub-003"])
        self.assertEqual(sidecar["voxel_count"], 1)

    def test_missing_cope_varcope_and_mask_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=1)
            _write_fixed_effects_tree(
                root,
                subjects=("001", "002", "003"),
                missing={("002", "cope"), ("002", "varcope"), ("002", "mask")},
            )

            jobs = plan_loso_group_map_jobs(document, context=context)

        warnings = "\n".join(jobs[0].warnings)
        self.assertIn("sub-002 missing cope", warnings)
        self.assertIn("sub-002 missing varcope", warnings)
        self.assertIn("sub-002 missing mask", warnings)
        self.assertEqual([item.subject_id for item in jobs[0].training_inputs], ["003"])

    def test_min_group_n_failure_excludes_heldout_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=2)
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), missing={("002", "cope")})

            with self.assertRaisesRegex(ValueError, "fewer than min_group_n"):
                plan_loso_group_map_jobs(document, context=context)

    def test_one_sample_group_design_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            design = write_one_sample_group_mean_design(Path(tmpdir), n_subjects=3)

            mat = design.design_mat.read_text(encoding="utf-8")
            con = design.design_con.read_text(encoding="utf-8")
            grp = design.design_grp.read_text(encoding="utf-8")

        self.assertIn("/NumPoints 3", mat)
        self.assertEqual(mat.split("/Matrix\n", 1)[1].strip().splitlines(), ["1", "1", "1"])
        self.assertIn("/NumContrasts 1", con)
        self.assertIn("/NumPoints 3", grp)

    def test_fsl_command_construction_does_not_require_fsl(self) -> None:
        plan = build_flame1_command_plan(
            cope_inputs=["inputs/sub-001/cope.nii.gz", "inputs/sub-002/cope.nii.gz"],
            varcope_inputs=["inputs/sub-001/varcope.nii.gz", "inputs/sub-002/varcope.nii.gz"],
            mask_path="inputs/group/mask.nii.gz",
            work_dir="work/flame",
            output_zstat_path="outputs/zstat.nii.gz",
            environment={"FSLOUTPUTTYPE": "NIFTI_GZ"},
        )

        commands = [command[0] for command in plan.commands]
        self.assertEqual(commands, ["fslmerge", "fslmerge", "flameo"])
        self.assertIn("--runmode=flame1", plan.commands[2])
        self.assertEqual(plan.environment["FSLOUTPUTTYPE"], "NIFTI_GZ")

    def test_recursive_sidecar_provenance_normalization_rewrites_nested_paths(self) -> None:
        feat_root = Path("/home/alice/example/feat-root")
        deriv_root = Path("/home/alice/example/roi-derivatives")
        payload = {
            "command_plan": {
                "commands": [
                    [
                        "fslmerge",
                        "-t",
                        str(deriv_root / ".cache" / "loso" / "merged_cope.nii.gz"),
                        str(feat_root / "sub-001" / "cope1.nii.gz"),
                    ],
                    [
                        "flameo",
                        f"--cope={deriv_root / '.cache' / 'loso' / 'merged_cope.nii.gz'}",
                        f"--mask={feat_root / 'group' / 'mask.nii.gz'}",
                    ],
                ],
            },
            "paths": [deriv_root / "loso_groupmaps" / "modelA" / "zstat.nii.gz"],
        }

        normalized = normalize_sidecar_provenance(
            payload,
            env_roots={"ROI_FEAT_ROOT": feat_root, "ROI_DERIV_ROOT": deriv_root},
        )

        self.assertEqual(normalized["command_plan"]["commands"][0][2], "${ROI_DERIV_ROOT:-}/.cache/loso/merged_cope.nii.gz")
        self.assertEqual(normalized["command_plan"]["commands"][0][3], "${ROI_FEAT_ROOT:-}/sub-001/cope1.nii.gz")
        self.assertEqual(normalized["command_plan"]["commands"][1][1], "--cope=${ROI_DERIV_ROOT:-}/.cache/loso/merged_cope.nii.gz")
        self.assertEqual(normalized["command_plan"]["commands"][1][2], "--mask=${ROI_FEAT_ROOT:-}/group/mask.nii.gz")
        self.assertEqual(normalized["paths"][0], "${ROI_DERIV_ROOT:-}/loso_groupmaps/modelA/zstat.nii.gz")
        self.assertEqual(validate_portable_provenance_paths(normalized), [])

    def test_roi_sidecar_validation_rejects_nested_personal_command_paths(self) -> None:
        document = {
            "roi_label": "LosoMap",
            "roi_family": "loso_group_map",
            "command_plan": {
                "commands": [["flameo", "--cope=/home/alice/example/unmapped/cope1.nii.gz"]],
            },
        }

        errors = validate_roi_sidecar_document(document)

        self.assertTrue(any("command_plan.commands[0][1]" in error and "personal absolute path" in error for error in errors))

    def test_mocked_local_fsl_execution_materializes_real_flameo_zstat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan = build_flame1_command_plan(
                cope_inputs=[root / "cope1.nii.gz"],
                varcope_inputs=[root / "varcope1.nii.gz"],
                mask_path=root / "mask.nii.gz",
                work_dir=root / "work",
                output_zstat_path=root / "out" / "zstat.nii.gz",
            )
            calls: list[tuple[str, ...]] = []

            def runner(command: tuple[str, ...]) -> None:
                calls.append(tuple(command))
                if command[0] == "flameo":
                    zstat = plan.flame_output_dir / "zstat1.nii.gz"
                    zstat.parent.mkdir(parents=True, exist_ok=True)
                    zstat.write_text("synthetic zstat", encoding="utf-8")

            output = execute_flame1_command_plan(plan, runner=runner)
            output_text = output.read_text(encoding="utf-8")

        self.assertEqual([call[0] for call in calls], ["fslmerge", "fslmerge", "flameo"])
        self.assertEqual(output_text, "synthetic zstat")

    def test_mocked_local_fsl_execution_accepts_legacy_stats_zstat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan = build_flame1_command_plan(
                cope_inputs=[root / "cope1.nii.gz"],
                varcope_inputs=[root / "varcope1.nii.gz"],
                mask_path=root / "mask.nii.gz",
                work_dir=root / "work",
                output_zstat_path=root / "out" / "zstat.nii.gz",
            )

            def runner(command: tuple[str, ...]) -> None:
                if command[0] == "flameo":
                    zstat = plan.flame_output_dir / "stats" / "zstat1.nii.gz"
                    zstat.parent.mkdir(parents=True, exist_ok=True)
                    zstat.write_text("legacy wrapper zstat", encoding="utf-8")

            output = execute_flame1_command_plan(plan, runner=runner)
            output_text = output.read_text(encoding="utf-8")

        self.assertEqual(output_text, "legacy wrapper zstat")

    def test_loso_cache_key_changes_when_input_fingerprint_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cope.nii.gz"
            path.write_text("first", encoding="utf-8")
            first = build_loso_group_map_cache_key(
                input_paths=[path],
                contrast_id="CondA",
                heldout_subject="sub-001",
                design_config={"kind": "one_sample_group_mean"},
                backend_config={"runmode": "flame1"},
            )
            path.write_text("second-longer", encoding="utf-8")
            second = build_loso_group_map_cache_key(
                input_paths=[path],
                contrast_id="CondA",
                heldout_subject="sub-001",
                design_config={"kind": "one_sample_group_mean"},
                backend_config={"runmode": "flame1"},
            )

        self.assertNotEqual(first, second)

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for LOSO NIfTI mask tests")
    def test_loso_builder_thresholded_and_below_threshold_paths(self) -> None:
        from research_platform.neuro.roi_builders import build_loso_group_map_roi

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = np.zeros((5, 5, 5), dtype=float)
            data[2, 2, 2] = 4.5
            thresholded = build_loso_group_map_roi(
                zstat_image=_image(data),
                roi_label="SeedA",
                desc="CondA",
                output_mask_path=root / "thresholded.nii.gz",
                sidecar_path=root / "thresholded.json",
                seed_xyz_mm=[2, 2, 2],
                search_radius_mm=2,
                sphere_radius_mm=1.01,
                z_threshold=3.1,
            )
            fallback = build_loso_group_map_roi(
                zstat_image=_image(data),
                roi_label="SeedB",
                desc="CondA",
                output_mask_path=root / "fallback.nii.gz",
                sidecar_path=root / "fallback.json",
                seed_xyz_mm=[2, 2, 2],
                search_radius_mm=2,
                sphere_radius_mm=1.01,
                z_threshold=10,
                allow_below_threshold_fallback=True,
            )

            thresholded_sidecar = json.loads((root / "thresholded.json").read_text(encoding="utf-8"))
            fallback_sidecar = json.loads((root / "fallback.json").read_text(encoding="utf-8"))

        self.assertEqual(thresholded.peak.fallback_status, "thresholded")
        self.assertEqual(thresholded_sidecar["fallback_status"], "thresholded")
        self.assertEqual(fallback.peak.fallback_status, "below_threshold_fallback")
        self.assertEqual(fallback_sidecar["thresholded_voxel_count"], 0)

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for LOSO NIfTI execution tests")
    def test_execute_loso_roi_reuses_one_group_map_across_roi_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(two_rois=True, min_voxels_warn=7)
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), nifti_masks=True)

            def fake_execute(plan: object) -> Path:
                data = np.zeros((5, 5, 5), dtype=float)
                data[2, 2, 2] = 5.5
                nib.save(_image(data), plan.output_zstat_path)
                return plan.output_zstat_path

            with (
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_loso_fsl_tool,
                ) as find_loso_fsl_tool,
                mock.patch(
                    "research_platform.neuro.fsl.flame.execute_flame1_command_plan",
                    side_effect=fake_execute,
                ) as execute,
            ):
                result = run_roi_build(document, context=context)

            actions = result.actions
            sidecar = json.loads(actions[0].sidecar_path.read_text(encoding="utf-8"))
            group_sidecar = json.loads(Path(actions[0].metadata["loso_group_job"]["sidecar_path"]).read_text(encoding="utf-8"))
            command_plan = group_sidecar["command_plan"]
            self.assertEqual(len(actions), 2)
            self.assertEqual(
                find_loso_fsl_tool.call_args_list,
                [mock.call("fslmerge"), mock.call("flameo")],
            )
            self.assertEqual(execute.call_count, 1)
            self.assertTrue(actions[0].mask_path.exists())
            self.assertTrue(actions[0].sidecar_path.exists())
            self.assertTrue(actions[1].mask_path.exists())
            self.assertTrue(actions[1].sidecar_path.exists())
            self.assertEqual(sidecar["roi_family"], "loso_group_map")
            self.assertEqual(sidecar["fallback_status"], "thresholded")
            self.assertEqual(sidecar["voxel_count"], 6)
            self.assertTrue(sidecar["group_map_path"].startswith("${ROI_DERIV_ROOT:-}/loso_groupmaps/"))
            self.assertEqual(validate_roi_sidecar_document(sidecar), [])
            self.assertTrue(group_sidecar["zstat_path"].startswith("${ROI_DERIV_ROOT:-}/loso_groupmaps/"))
            self.assertTrue(group_sidecar["group_mask_path"].startswith("${ROI_FEAT_ROOT:-}/group/"))
            self.assertTrue(command_plan["commands"][0][2].startswith("${ROI_DERIV_ROOT:-}/.cache/loso_groupmaps/"))
            self.assertTrue(command_plan["commands"][0][3].startswith("${ROI_FEAT_ROOT:-}/sub-002/"))
            self.assertTrue(command_plan["commands"][1][2].startswith("${ROI_DERIV_ROOT:-}/.cache/loso_groupmaps/"))
            self.assertTrue(command_plan["commands"][1][3].startswith("${ROI_FEAT_ROOT:-}/sub-002/"))
            self.assertTrue(any(part.startswith("--cope=${ROI_DERIV_ROOT:-}/.cache/") for part in command_plan["commands"][2]))
            self.assertTrue(any(part.startswith("--vc=${ROI_DERIV_ROOT:-}/.cache/") for part in command_plan["commands"][2]))
            self.assertTrue(any(part.startswith("--mask=${ROI_FEAT_ROOT:-}/group/") for part in command_plan["commands"][2]))
            self.assertTrue(any(part.startswith("--ld=${ROI_DERIV_ROOT:-}/.cache/") for part in command_plan["commands"][2]))
            self.assertNotIn(str(root.resolve()), json.dumps(group_sidecar))
            self.assertIn("warn_min_voxels", sidecar["qc_flags"])
            self.assertIn("label-SeedA_desc-CondA_mask.nii.gz", actions[0].mask_path.name)
            self.assertIn("/rois/loso_demo/sub-001/ses-01/func/", str(actions[0].mask_path))
            self.assertFalse((root / "artifacts" / "roi-derivatives" / "roi-loso-flame1").exists())

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for LOSO NIfTI execution tests")
    def test_loso_tool_failure_restores_cache_and_work_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document()
            document["roi_set"]["runtime"] = {"existing_output": "replace"}  # type: ignore[index]
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), nifti_masks=True)
            planned = plan_roi_build(document, context=context)
            action = planned.actions[0]
            job = action.metadata["loso_group_job"]
            zstat_path = Path(str(job["zstat_path"]))
            group_sidecar = Path(str(job["sidecar_path"]))
            work_dir = Path(str(job["work_dir"]))
            zstat_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(_image(np.full((5, 5, 5), 2.0)), zstat_path)
            group_sidecar.write_text('{"cache_key": "stale", "sentinel": true}\n', encoding="utf-8")
            work_sentinel = work_dir / "sentinel.txt"
            work_sentinel.parent.mkdir(parents=True, exist_ok=True)
            work_sentinel.write_text("keep\n", encoding="utf-8")
            prior_zstat = zstat_path.read_bytes()

            def fail_after_mutation(command_plan: object) -> Path:
                self.assertNotEqual(command_plan.work_dir, work_dir)
                self.assertNotEqual(command_plan.output_zstat_path, zstat_path)
                self.assertEqual(zstat_path.read_bytes(), prior_zstat)
                self.assertEqual(work_sentinel.read_text(encoding="utf-8"), "keep\n")
                command_plan.work_dir.mkdir(parents=True, exist_ok=True)
                (command_plan.work_dir / "partial.txt").write_text("partial\n", encoding="utf-8")
                nib.save(_image(np.full((5, 5, 5), 9.0)), command_plan.output_zstat_path)
                raise RuntimeError("injected FLAME failure")

            with (
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_loso_fsl_tool,
                ) as find_loso_fsl_tool,
                mock.patch(
                    "research_platform.neuro.fsl.flame.execute_flame1_command_plan",
                    side_effect=fail_after_mutation,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected FLAME failure"):
                    run_roi_build(document, context=context)

            self.assertEqual(
                find_loso_fsl_tool.call_args_list,
                [mock.call("fslmerge"), mock.call("flameo")],
            )
            self.assertEqual(zstat_path.read_bytes(), prior_zstat)
            self.assertEqual(
                group_sidecar.read_text(encoding="utf-8"),
                '{"cache_key": "stale", "sentinel": true}\n',
            )
            self.assertEqual(work_sentinel.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse(action.mask_path.exists())
            self.assertFalse(action.sidecar_path.exists())
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for LOSO NIfTI execution tests")
    def test_loso_unreadable_input_fails_before_external_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document()
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), nifti_masks=True)
            unreadable = (
                root
                / "derivatives"
                / "sub-002"
                / "ses-01"
                / "func"
                / "task-memory_model-ModelA_contrast-CondA"
                / "cope1.nii.gz"
            )
            unreadable.write_text("not an image\n", encoding="utf-8")

            with mock.patch("research_platform.neuro.fsl.flame.execute_flame1_command_plan") as execute:
                with self.assertRaisesRegex(Exception, "not ready for execution"):
                    run_roi_build(document, context=context)

            execute.assert_not_called()
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for LOSO NIfTI execution tests")
    def test_generated_group_mask_failure_restores_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=1)
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["runtime"] = {"existing_output": "replace"}  # type: ignore[index]
            roi_set["group_mask"] = {  # type: ignore[index]
                "strategy": "fixed_effects_mask_intersection",
                "root_ref": "artifacts_root",
                "path": "generated-masks/{heldout_subject_dir}/{contrast_desc}_mask.nii.gz",
            }
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), nifti_masks=True)
            planned = plan_roi_build(document, context=context)
            action = planned.actions[0]
            job = action.metadata["loso_group_job"]
            generated = job["generated_group_mask"]
            mask_path = Path(str(generated["mask_path"]))
            sidecar_path = Path(str(generated["sidecar_path"]))
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(_image(np.zeros((5, 5, 5), dtype=np.uint8)), mask_path)
            sidecar_path.write_text('{"sentinel": true}\n', encoding="utf-8")
            prior_mask = mask_path.read_bytes()

            def fail_after_generated_mask(command_plan: object) -> Path:
                self.assertNotEqual(command_plan.mask_path, mask_path)
                self.assertEqual(mask_path.read_bytes(), prior_mask)
                self.assertEqual(sidecar_path.read_text(encoding="utf-8"), '{"sentinel": true}\n')
                nib.save(_image(np.full((5, 5, 5), 8.0)), command_plan.output_zstat_path)
                raise RuntimeError("injected post-mask failure")

            with (
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_loso_fsl_tool,
                ) as find_loso_fsl_tool,
                mock.patch(
                    "research_platform.neuro.fsl.flame.execute_flame1_command_plan",
                    side_effect=fail_after_generated_mask,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected post-mask failure"):
                    run_roi_build(document, context=context)

            self.assertEqual(
                find_loso_fsl_tool.call_args_list,
                [mock.call("fslmerge"), mock.call("flameo")],
            )
            self.assertEqual(mask_path.read_bytes(), prior_mask)
            self.assertEqual(sidecar_path.read_text(encoding="utf-8"), '{"sentinel": true}\n')
            self.assertFalse(Path(str(job["zstat_path"])).exists())
            self.assertFalse(action.mask_path.exists())
            self.assertFalse(action.sidecar_path.exists())
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for LOSO NIfTI execution tests")
    def test_generated_group_mask_success_promotes_only_portable_final_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(min_group_n=1)
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["group_mask"] = {  # type: ignore[index]
                "strategy": "fixed_effects_mask_intersection",
                "root_ref": "artifacts_root",
                "path": "generated-masks/{heldout_subject_dir}/{contrast_desc}_mask.nii.gz",
            }
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), nifti_masks=True)
            planned = plan_roi_build(document, context=context)
            job = planned.actions[0].metadata["loso_group_job"]
            generated = job["generated_group_mask"]
            generated_mask = Path(str(generated["mask_path"]))
            generated_sidecar = Path(str(generated["sidecar_path"]))

            def fake_execute(command_plan: object) -> Path:
                self.assertNotEqual(command_plan.mask_path, generated_mask)
                data = np.zeros((5, 5, 5), dtype=float)
                data[2, 2, 2] = 5.5
                nib.save(_image(data), command_plan.output_zstat_path)
                return command_plan.output_zstat_path

            with (
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_loso_fsl_tool,
                ) as find_loso_fsl_tool,
                mock.patch(
                    "research_platform.neuro.fsl.flame.execute_flame1_command_plan",
                    side_effect=fake_execute,
                ),
            ):
                result = run_roi_build(document, context=context)

            self.assertEqual(
                find_loso_fsl_tool.call_args_list,
                [mock.call("fslmerge"), mock.call("flameo")],
            )
            generated_payload = json.loads(generated_sidecar.read_text(encoding="utf-8"))
            group_payload = json.loads(Path(str(job["sidecar_path"])).read_text(encoding="utf-8"))

            self.assertTrue(generated_mask.is_file())
            self.assertTrue(result.actions[0].mask_path.is_file())
            self.assertEqual(validate_portable_provenance_paths(generated_payload), [])
            self.assertEqual(validate_portable_provenance_paths(group_payload), [])
            self.assertNotIn(".roi-runtime-", json.dumps(generated_payload))
            self.assertNotIn(".roi-runtime-", json.dumps(group_payload))
            self.assertNotIn(str(root.resolve()), json.dumps(generated_payload))
            self.assertNotIn(str(root.resolve()), json.dumps(group_payload))
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for LOSO NIfTI execution tests")
    def test_publication_failure_leaves_complete_committed_runtime_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document()
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), nifti_masks=True)
            planned = plan_roi_build(document, context=context)
            job = planned.actions[0].metadata["loso_group_job"]

            def fake_execute(command_plan: object) -> Path:
                data = np.zeros((5, 5, 5), dtype=float)
                data[2, 2, 2] = 5.5
                nib.save(_image(data), command_plan.output_zstat_path)
                return command_plan.output_zstat_path

            with (
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_loso_fsl_tool,
                ) as find_loso_fsl_tool,
                mock.patch(
                    "research_platform.neuro.fsl.flame.execute_flame1_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution._publish_loso_roi_build_if_enabled",
                    side_effect=RuntimeError("injected publication failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected publication failure"):
                    run_roi_build(document, context=context)

            self.assertEqual(
                find_loso_fsl_tool.call_args_list,
                [mock.call("fslmerge"), mock.call("flameo")],
            )
            self.assertTrue(planned.actions[0].mask_path.is_file())
            self.assertTrue(planned.actions[0].sidecar_path.is_file())
            self.assertTrue(Path(str(job["zstat_path"])).is_file())
            self.assertTrue(Path(str(job["sidecar_path"])).is_file())
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    @unittest.skipIf(nib is None or np is None, "numpy and nibabel are required for LOSO NIfTI publication tests")
    def test_loso_build_publishes_bidslike_maps_masks_and_sidecars_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document()
            roi_set = document["roi_set"]  # type: ignore[index]
            roi_set["direction"] = "AP"  # type: ignore[index]
            roi_set["resolution"] = "2"  # type: ignore[index]
            roi_set["publication"] = {  # type: ignore[index]
                "enabled": True,
                "layout": "loso_flame1_bidslike",
                "root": {"root_ref": "artifacts_root", "path": "roi-derivatives/roi-loso-flame1"},
                "map_desc": "{model}LOSOFlame1",
                "mask_desc": "{model}LOSOFlame1Sphere{sphere_radius_mm}mm",
                "existing_output": "replace",
            }
            roi_set["runtime"] = {"existing_output": "replace"}  # type: ignore[index]
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"), nifti_masks=True)

            def fake_execute(plan: object) -> Path:
                data = np.zeros((5, 5, 5), dtype=float)
                data[2, 2, 2] = 5.5
                nib.save(_image(data), plan.output_zstat_path)
                return plan.output_zstat_path

            with (
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_loso_fsl_tool,
                ) as find_loso_fsl_tool,
                mock.patch(
                    "research_platform.neuro.fsl.flame.execute_flame1_command_plan",
                    side_effect=fake_execute,
                ) as execute,
            ):
                run_roi_build(document, context=context)
                run_roi_build(document, context=context)

            published_root = root / "artifacts" / "roi-derivatives" / "roi-loso-flame1"
            map_path = (
                published_root
                / "maps"
                / "group"
                / "ses-01"
                / "func"
                / "ses-01_task-memory_dir-AP_space-MNI152NLin2009cAsym_res-2_contrast-CondA_stat-z_heldout-sub001_desc-ModelALOSOFlame1_statmap.nii.gz"
            )
            mask_path = (
                published_root
                / "masks"
                / "sub-001"
                / "ses-01"
                / "func"
                / "sub-001_ses-01_task-memory_dir-AP_space-MNI152NLin2009cAsym_res-2_label-SeedA_contrast-CondA_desc-ModelALOSOFlame1Sphere101mm_mask.nii.gz"
            )
            dataset = json.loads((published_root / "dataset_description.json").read_text(encoding="utf-8"))
            map_exists = map_path.exists()
            mask_exists = mask_path.exists()
            map_sidecar = json.loads(map_path.with_name(map_path.name[:-7] + ".json").read_text(encoding="utf-8"))
            mask_sidecar = json.loads(mask_path.with_name(mask_path.name[:-7] + ".json").read_text(encoding="utf-8"))

        self.assertTrue(map_exists)
        self.assertTrue(mask_exists)
        self.assertEqual(
            find_loso_fsl_tool.call_args_list,
            [
                mock.call("fslmerge"),
                mock.call("flameo"),
                mock.call("fslmerge"),
                mock.call("flameo"),
            ],
        )
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(dataset["DatasetType"], "derivative")
        self.assertEqual(dataset["ContrastAliases"], {"CondA": "CondA"})
        self.assertEqual(map_sidecar["Estimator"], "FLAME1")
        self.assertEqual(map_sidecar["HeldOutSubject"], "sub-001")
        self.assertEqual(map_sidecar["ContrastAlias"], "CondA")
        self.assertEqual(mask_sidecar["AnalysisLevel"], "participant")
        self.assertEqual(mask_sidecar["DefiningMap"], map_path.relative_to(published_root).as_posix())
        self.assertEqual(mask_sidecar["PeakZStatistic"], 5.5)

    def test_loso_roi_sidecar_provenance_uses_safe_root_reference_without_nibabel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = Path("/home/alice/example/artifacts/roi-derivatives")
            zstat_path = output_root / "loso_groupmaps" / "loso_demo" / "group" / "func" / "task-memory_desc-CondA_heldout-sub001_zstat.nii.gz"
            action = _mock_loso_action(root, output_root=output_root, zstat_path=zstat_path)
            context = RoiExecutionContext(
                workspace_root=root,
                project_root=root / "project" / "project-default",
                artifacts_root=root / "artifacts",
                project_name="project-default",
                root_refs={"derivative_root": output_root},
            )

            fake_nifti = types.ModuleType("research_platform.neuro.nifti")
            fake_nifti.load_nifti_image = lambda _path: object()
            fake_nifti.validate_compatible_geometry = lambda *_args, **_kwargs: None

            fake_builders = types.ModuleType("research_platform.neuro.roi_builders")

            def fake_build_loso_group_map_roi(**kwargs: object) -> object:
                payload = {
                    "roi_label": kwargs["roi_label"],
                    "roi_family": "loso_group_map",
                    "backend": "fsl_flame1",
                    "seed_coordinate": list(kwargs["seed_xyz_mm"]),
                    "search_radius_mm": kwargs["search_radius_mm"],
                    "sphere_radius_mm": kwargs["sphere_radius_mm"],
                    "z_threshold": kwargs["z_threshold"],
                    "fallback_status": "thresholded",
                    "selected_peak_coordinate": [2.0, 2.0, 2.0],
                    "loso_peak_coordinate": [2.0, 2.0, 2.0],
                    "selected_peak_stat": 5.5,
                    "selected_peak_z": 5.5,
                    "mask_intersection_policy": "none",
                    "voxel_count": 1,
                    "warnings": [],
                    "qc_flags": ["pass"],
                    **dict(kwargs["provenance"]),
                }
                errors = validate_roi_sidecar_document(payload)
                if errors:
                    raise AssertionError(errors)
                mask_path = Path(kwargs["output_mask_path"])
                sidecar_path = Path(kwargs["sidecar_path"])
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.write_text("mask", encoding="utf-8")
                sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
                return types.SimpleNamespace(
                    mask_path=mask_path,
                    sidecar_path=sidecar_path,
                    voxel_count=1,
                    qc=types.SimpleNamespace(qc_flags=("pass",), warnings=()),
                    provenance=payload,
                    peak=None,
                )

            fake_builders.build_loso_group_map_roi = fake_build_loso_group_map_roi
            with mock.patch.dict(
                sys.modules,
                {
                    "research_platform.neuro.nifti": fake_nifti,
                    "research_platform.neuro.roi_builders": fake_builders,
                },
            ):
                result = execute_loso_build_action(
                    action,
                    RoiDefinition(label="SeedA", family="loso_group_map", backend="fsl_flame1"),
                    context=context,
                    cache_state={"abc123": {"zstat_path": str(zstat_path), "status": "reused_in_memory"}},
                )

            sidecar = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))

        self.assertEqual(sidecar["group_map_path"], "${ROI_DERIV_ROOT:-}/loso_groupmaps/loso_demo/group/func/task-memory_desc-CondA_heldout-sub001_zstat.nii.gz")
        self.assertNotIn("/home/alice/example", json.dumps(sidecar))
        self.assertEqual(validate_roi_sidecar_document(sidecar), [])

    def test_plan_only_loso_build_reports_group_map_paths_without_writing_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = _context(root)
            document = _loso_document(two_rois=True)
            _write_fixed_effects_tree(root, subjects=("001", "002", "003"))

            plan = plan_roi_build(document, context=context)

        self.assertFalse(plan.executed)
        self.assertEqual(len(plan.actions), 2)
        self.assertEqual(
            plan.actions[0].metadata["loso_group_job"]["cache_key"],
            plan.actions[1].metadata["loso_group_job"]["cache_key"],
        )
        self.assertIn("loso_groupmaps/loso_demo/group/ses-01/func", plan.actions[0].metadata["loso_group_job"]["zstat_path"])
        self.assertFalse(plan.actions[0].mask_path.exists())


def _context(root: Path) -> RoiExecutionContext:
    project_root = root / "project" / "project-default"
    project_root.mkdir(parents=True, exist_ok=True)
    return RoiExecutionContext(
        workspace_root=root,
        project_root=project_root,
        artifacts_root=root / "artifacts",
        project_name="project-default",
        root_refs={
            "derivative_root": root / "derivatives",
            "artifacts_root": root / "artifacts",
        },
    )


def _loso_document(
    *,
    min_group_n: int = 2,
    two_rois: bool = False,
    min_voxels_warn: int | None = None,
) -> dict[str, object]:
    rois: list[dict[str, object]] = [
        {
            "label": "SeedA",
            "family": "loso_group_map",
            "backend": "fsl_flame1",
            "desc": "CondA",
            "contrast": "CondA",
            "seed_coordinate": [2, 2, 2],
            "search_radius_mm": 2,
            "sphere_radius_mm": 1.01,
            "z_threshold": 3.1,
            "allow_below_threshold_fallback": True,
        }
    ]
    if min_voxels_warn is not None:
        rois[0]["min_voxels_warn"] = min_voxels_warn
    if two_rois:
        rois.append({**rois[0], "label": "SeedB", "seed_coordinate": [2, 2, 2]})
    return {
        "roi_set": {
            "name": "loso_demo",
            "backend": "fsl_flame1",
            "subjects": ["sub-001", "sub-002", "sub-003"],
            "held_out_subjects": ["sub-001"],
            "session": "ses-01",
            "task": "memory",
            "model": "ModelA",
            "space": "MNI152NLin2009cAsym",
            "min_group_n": min_group_n,
            "outputs": {"root_ref": "artifacts_root", "path": "roi-derivatives"},
            "fixed_effects_inputs": {
                "root_ref": "derivative_root",
                "cope_dir": "{subject_dir}/{session_dir}/func/task-{task_id}_model-{model}_contrast-{contrast_id}",
                "cope_image": "cope{cope_number}.nii.gz",
                "varcope_image": "varcope{cope_number}.nii.gz",
                "mask_image": "mask.nii.gz",
            },
            "group_mask": {
                "root_ref": "derivative_root",
                "pattern": "group/{session_dir}/func/group_{session_dir}_task-{task_id}_space-{space}_mask.nii.gz",
            },
            "contrasts": [{"id": "CondA", "cope_number": 1, "desc": "CondA"}],
            "rois": rois,
        }
    }


def _write_fixed_effects_tree(
    root: Path,
    *,
    subjects: tuple[str, ...],
    missing: set[tuple[str, str]] | None = None,
    nifti_masks: bool = False,
) -> None:
    missing = missing or set()
    derivative = root / "derivatives"
    group_mask = derivative / "group" / "ses-01" / "func" / "group_ses-01_task-memory_space-MNI152NLin2009cAsym_mask.nii.gz"
    group_mask.parent.mkdir(parents=True, exist_ok=True)
    if nifti_masks and nib is not None:
        nib.save(_image(np.ones((5, 5, 5), dtype=np.uint8)), group_mask)
    else:
        group_mask.write_text("group mask", encoding="utf-8")
    for subject in subjects:
        cope_dir = derivative / f"sub-{subject}" / "ses-01" / "func" / "task-memory_model-ModelA_contrast-CondA"
        cope_dir.mkdir(parents=True, exist_ok=True)
        for kind, filename in (
            ("cope", "cope1.nii.gz"),
            ("varcope", "varcope1.nii.gz"),
            ("mask", "mask.nii.gz"),
        ):
            if (subject, kind) in missing:
                continue
            path = cope_dir / filename
            if nifti_masks and nib is not None:
                data = np.ones((5, 5, 5), dtype=np.uint8 if kind == "mask" else float)
                if kind == "mask" and subject == "001":
                    data[1, 2, 2] = 0
                nib.save(_image(data), path)
            else:
                path.write_text(kind, encoding="utf-8")


def _write_nested_gfeat_tree(
    root: Path,
    *,
    subjects: tuple[str, ...],
    model: str,
    cope_numbers: tuple[str, ...],
) -> None:
    derivative = root / "derivatives"
    group_mask = derivative / "group" / "ses-01" / "func" / "group_ses-01_task-memory_space-MNI152NLin2009cAsym_mask.nii.gz"
    group_mask.parent.mkdir(parents=True, exist_ok=True)
    group_mask.write_text("group mask", encoding="utf-8")
    for subject in subjects:
        gfeat_dir = (
            derivative
            / f"sub-{subject}"
            / "ses-01"
            / f"sub-{subject}_ses-01_task-memory_dir-AP_desc-{model}.gfeat"
        )
        for cope_number in cope_numbers:
            cope_dir = gfeat_dir / f"cope{cope_number}.feat"
            (cope_dir / "stats").mkdir(parents=True, exist_ok=True)
            (cope_dir / "stats" / "cope1.nii.gz").write_text("cope", encoding="utf-8")
            (cope_dir / "stats" / "varcope1.nii.gz").write_text("varcope", encoding="utf-8")
            (cope_dir / "mask.nii.gz").write_text("mask", encoding="utf-8")


def _image(data: np.ndarray) -> object:
    return nib.Nifti1Image(np.asarray(data), np.eye(4))


def _mock_loso_action(root: Path, *, output_root: Path, zstat_path: Path) -> object:
    from research_platform.neuro.roi_execution import RoiBuildAction

    return RoiBuildAction(
        roi_label="SeedA",
        family="loso_group_map",
        backend="fsl_flame1",
        mask_path=root / "out" / "sub-001_task-memory_space-MNI_label-SeedA_desc-CondA_mask.nii.gz",
        sidecar_path=root / "out" / "sub-001_task-memory_space-MNI_label-SeedA_desc-CondA_mask.json",
        input_paths={},
        metadata={
            "desc": "CondA",
            "roi_parameters": {
                "seed_coordinate": [2, 2, 2],
                "search_radius_mm": 2,
                "sphere_radius_mm": 1.01,
                "z_threshold": 3.1,
            },
            "mask_intersection_policy": "none",
            "loso_group_job": {
                "roi_set": "loso_demo",
                "heldout_subject": "001",
                "session_id": None,
                "task_id": "memory",
                "model": "ModelA",
                "contrast": {"contrast_id": "CondA", "cope_number": "1"},
                "training_inputs": [{"subject_id": "002"}, {"subject_id": "003"}],
                "group_mask_path": str(output_root / "group_mask.nii.gz"),
                "zstat_path": str(zstat_path),
                "sidecar_path": str(output_root / "zstat.json"),
                "work_dir": str(output_root / ".cache"),
                "output_root": str(output_root),
                "input_root": str(output_root),
                "cache_key": "abc123",
                "cache_reuse": True,
                "backend_config": {},
            },
        },
    )


if __name__ == "__main__":
    unittest.main()
