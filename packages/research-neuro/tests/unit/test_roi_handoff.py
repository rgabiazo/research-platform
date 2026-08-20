from __future__ import annotations

from pathlib import Path
import copy
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
BIDS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-bids"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(BIDS_PACKAGE_ROOT / "src"))

from research_platform.neuro.localizer_ffx import plan_localizer_fixed_effects
from research_platform.neuro.roi_execution import RoiExecutionContext
from research_platform.neuro.roi_handoff import (
    generate_loso_roi_set_plan_object,
    inventory_completed_subject_ffx_outputs,
    map_ffx_outputs_to_loso_fixed_effects_inputs,
    plan_loso_roi_handoff,
)


SUBJECTS = ("participant-a", "participant-b", "participant-c")
SESSION_ID = "sessiona"
TASK_ID = "taskalpha"
RUN_ID = "runa"
MODEL_ID = "modelalpha"
CONTRAST_ID = "contrastalpha"
ROI_SET_NAME = "roialpha"
ROI_LABEL = "RoiAlpha"


def _localizer_document() -> dict[str, object]:
    return {
        "localizer_fixed_effects": {
            "feat_sources": [
                {
                    "name": "localizer-source",
                    "root_ref": "feat-root",
                    "feat_dir_template": "{participant_id}/{session_id}/{task_id}/{run_id}.feat",
                }
            ],
            "contrast_aliases": [{"id": CONTRAST_ID, "aliases": ["contrast alpha"]}],
            "runs": [
                {
                    "participant_id": subject_id,
                    "session_id": SESSION_ID,
                    "task_id": TASK_ID,
                    "run_id": RUN_ID,
                }
                for subject_id in SUBJECTS
            ],
            "outputs": {
                "root_ref": "ffx-root",
                "output_dir_template": "fixed-effects/{participant_id}/{session_id}/{task_id}/{contrast_id}/ffx.feat",
                "work_dir_template": "work/{participant_id}/{session_id}/{task_id}/{contrast_id}",
            },
            "min_complete_runs": 1,
        }
    }


def _write_run_level_feat(feat_root: Path, subject_id: str) -> None:
    feat_dir = feat_root / subject_id / SESSION_ID / TASK_ID / f"{RUN_ID}.feat"
    stats_dir = feat_dir / "stats"
    stats_dir.mkdir(parents=True)
    (feat_dir / "design.fsf").write_text('set fmri(conname_real.1) "contrast alpha"\n', encoding="utf-8")
    (stats_dir / "cope1.nii.gz").write_text("run cope\n", encoding="utf-8")
    (stats_dir / "varcope1.nii.gz").write_text("run varcope\n", encoding="utf-8")
    (feat_dir / "mask.nii.gz").write_text("run mask\n", encoding="utf-8")


def _planned_path(plan: object, job_id: str, path_kind: str) -> Path:
    for row in plan.output_path_rows:
        if row.job_id == job_id and row.path_kind == path_kind:
            return Path(row.path)
    raise AssertionError(f"Missing planned path {job_id} {path_kind}")


def _materialize_subject_ffx_outputs(plan: object, *, missing: set[tuple[str, str]] | None = None) -> dict[str, object]:
    missing = missing or set()
    checks: list[dict[str, str]] = []
    completed_rows: list[dict[str, object]] = []
    for job in plan.ffx_job_rows:
        job_checks: list[dict[str, str]] = []
        for path_kind, text in (
            ("output_cope", "ffx cope\n"),
            ("output_varcope", "ffx varcope\n"),
            ("output_mask", "ffx mask\n"),
        ):
            path = _planned_path(plan, job.job_id, path_kind)
            if (job.job_id, path_kind) not in missing:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            row = {
                "job_id": job.job_id,
                "path_kind": path_kind,
                "path": str(path),
                "status": "present" if path.is_file() else "missing",
            }
            checks.append(row)
            job_checks.append(row)
        if all(row["status"] == "present" for row in job_checks):
            completed_rows.append({"job_id": job.job_id, "status": "completed", "expected_outputs": job_checks})
    return {
        "status": "ok" if completed_rows else "error",
        "executed": True,
        "completed_job_rows": completed_rows,
        "expected_output_check_rows": checks,
    }


def _plan(root: Path):
    feat_root = root / "feat"
    ffx_root = root / "ffx"
    for subject_id in SUBJECTS:
        _write_run_level_feat(feat_root, subject_id)
    return plan_localizer_fixed_effects(
        _localizer_document(),
        roots={"feat-root": feat_root, "ffx-root": ffx_root},
    )


def _roi_document() -> dict[str, object]:
    return {
        "roi_set": {
            "name": ROI_SET_NAME,
            "backend": "fsl_flame1",
            "subjects": list(SUBJECTS),
            "held_out_subjects": ["participant-a"],
            "session": SESSION_ID,
            "task": TASK_ID,
            "model": MODEL_ID,
            "space": "spaceAlpha",
            "min_group_n": 1,
            "outputs": {"root_ref": "roi-output-root", "path": "roi-runtime"},
            "fixed_effects_inputs": _fixed_effects_inputs(),
            "group_mask": {
                "root_ref": "ffx-root",
                "pattern": "group/{session_id}/func/group_{session_id}_task-{task_id}_space-{space}_mask.nii.gz",
            },
            "contrasts": [{"id": CONTRAST_ID, "cope_number": 1, "desc": "contrastAlpha"}],
            "rois": [
                {
                    "label": ROI_LABEL,
                    "family": "loso_group_map",
                    "backend": "fsl_flame1",
                    "desc": "contrastAlpha",
                    "contrast": CONTRAST_ID,
                    "group": "roi-group-alpha",
                    "tags": ["tag-alpha"],
                    "seed_coordinate": [1, 2, 3],
                    "search_radius_mm": 4,
                    "sphere_radius_mm": 2,
                    "z_threshold": 3,
                    "allow_below_threshold_fallback": True,
                    "mask_intersection_policy": "none",
                }
            ],
        }
    }


def _fixed_effects_inputs() -> dict[str, object]:
    return {
        "root_ref": "ffx-root",
        "cope_dir": "fixed-effects/{subject_id}/{session_id}/{task_id}/{contrast_id}/ffx.feat",
        "cope_image": "stats/cope{cope_number}.nii.gz",
        "varcope_image": "stats/varcope{cope_number}.nii.gz",
        "mask_image": "mask.nii.gz",
    }


def _write_group_mask(root: Path) -> None:
    group_mask = root / "ffx" / "group" / SESSION_ID / "func" / f"group_{SESSION_ID}_task-{TASK_ID}_space-spaceAlpha_mask.nii.gz"
    group_mask.parent.mkdir(parents=True)
    group_mask.write_text("group mask\n", encoding="utf-8")


def _context(root: Path) -> RoiExecutionContext:
    return RoiExecutionContext(
        workspace_root=root,
        project_root=root / "project",
        artifacts_root=root / "artifacts",
        root_refs={
            "ffx-root": root / "ffx",
            "roi-output-root": root / "roi-output",
        },
    )


def _snapshot(root: Path) -> tuple[tuple[str, str, int | None], ...]:
    rows: list[tuple[str, str, int | None]] = []
    for path in sorted(root.rglob("*")):
        kind = "file" if path.is_file() else "dir"
        size = path.stat().st_size if path.is_file() else None
        rows.append((path.relative_to(root).as_posix(), kind, size))
    return tuple(rows)


class RoiHandoffTests(unittest.TestCase):
    def test_completed_ffx_outputs_map_to_existing_loso_fixed_effects_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            execution = _materialize_subject_ffx_outputs(plan)
            _write_group_mask(root)
            document = _roi_document()
            before = _snapshot(root)

            handoff = plan_loso_roi_handoff(
                roi_set_reference="roi-alpha.yaml",
                roi_set_document=document,
                localizer_ffx_plan=plan,
                localizer_ffx_execution_result=execution,
                context=_context(root),
                preview_existing_roi_build=False,
            )
            after = _snapshot(root)

        self.assertEqual(handoff.status, "ok")
        self.assertFalse(handoff.executed)
        self.assertTrue(handoff.plan_only)
        self.assertEqual(before, after)
        self.assertEqual(len(handoff.completed_ffx_output_rows), 3)
        self.assertTrue(all(row.complete for row in handoff.completed_ffx_output_rows))
        self.assertEqual({row.status for row in handoff.fixed_effects_input_mapping_rows}, {"matched"})
        self.assertEqual(handoff.existing_roi_build_preview_rows, ())
        json.dumps(handoff.to_dict(), allow_nan=False, sort_keys=True)

    def test_missing_completed_ffx_output_produces_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            first_job = plan.ffx_job_rows[0].job_id
            execution = _materialize_subject_ffx_outputs(plan, missing={(first_job, "output_mask")})

            handoff = plan_loso_roi_handoff(
                roi_set_reference="roi-alpha.yaml",
                localizer_ffx_plan=plan,
                localizer_ffx_execution_result=execution,
                roots={"ffx-root": root / "ffx", "roi-output-root": root / "roi-output"},
                preview_existing_roi_build=False,
            )

        self.assertEqual(handoff.status, "error")
        self.assertTrue(any(row.path_kind == "output_mask" for row in handoff.missing_ffx_output_rows))
        self.assertTrue(any("not complete" in error for error in handoff.errors))

    def test_existing_roi_set_reference_and_document_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            execution = _materialize_subject_ffx_outputs(plan)
            _write_group_mask(root)
            document = _roi_document()
            original = copy.deepcopy(document)

            handoff = plan_loso_roi_handoff(
                roi_set_reference="preserved-roi-set.yaml",
                roi_set_document=document,
                localizer_ffx_plan=plan,
                localizer_ffx_execution_result=execution,
                context=_context(root),
                preview_existing_roi_build=False,
            )

        self.assertEqual(document, original)
        self.assertEqual(handoff.roi_set_rows[0].roi_set_reference, "preserved-roi-set.yaml")
        self.assertTrue(handoff.roi_set_rows[0].preserved)

    def test_generated_roi_set_plan_object_is_in_memory_and_json_safe(self) -> None:
        document = _roi_document()
        original = copy.deepcopy(document)
        del document["roi_set"]["fixed_effects_inputs"]  # type: ignore[index]

        generated = generate_loso_roi_set_plan_object(
            document,
            fixed_effects_inputs=_fixed_effects_inputs(),
            validate_personal_paths=False,
        )

        self.assertNotIn("fixed_effects_inputs", document["roi_set"])  # type: ignore[operator]
        self.assertIn("fixed_effects_inputs", generated["roi_set"])
        self.assertIn("fixed_effects_inputs", original["roi_set"])  # sanity check the helper input is complete elsewhere
        json.dumps(generated, allow_nan=False, sort_keys=True)

    def test_handoff_reuses_existing_roi_build_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            execution = _materialize_subject_ffx_outputs(plan)
            _write_group_mask(root)
            fake_plan = types.SimpleNamespace(
                roi_set_name=ROI_SET_NAME,
                actions=(
                    types.SimpleNamespace(
                        roi_label=ROI_LABEL,
                        family="loso_group_map",
                        backend="fsl_flame1",
                        mask_path=root / "planned-mask.nii.gz",
                        sidecar_path=root / "planned-mask.json",
                        metadata={"loso_group_job": {"warnings": []}},
                    ),
                ),
            )

            with mock.patch("research_platform.neuro.roi_execution.plan_roi_build", return_value=fake_plan) as planner:
                handoff = plan_loso_roi_handoff(
                    roi_set_document=_roi_document(),
                    localizer_ffx_plan=plan,
                    localizer_ffx_execution_result=execution,
                    context=_context(root),
                )

        self.assertEqual(planner.call_count, 1)
        self.assertEqual(handoff.existing_roi_build_preview_rows[0].mask_path, str(root / "planned-mask.nii.gz"))
        self.assertEqual(handoff.package_plan_preview_rows[0].package_api, "research_platform.neuro.roi_execution.plan_roi_build")

    def test_catalog_metadata_and_contrast_aliases_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            execution = _materialize_subject_ffx_outputs(plan)
            handoff = plan_loso_roi_handoff(
                roi_set_document=_roi_document(),
                localizer_ffx_plan=plan,
                localizer_ffx_execution_result=execution,
                context=_context(root),
                preview_existing_roi_build=False,
            )

        catalog = handoff.roi_catalog_rows[0].to_dict()
        self.assertEqual(catalog["metadata"]["group"], "roi-group-alpha")
        self.assertNotIn("seed_coordinate", catalog["metadata"])
        self.assertNotIn("z_threshold", catalog["metadata"])
        self.assertTrue(all(row.status == "metadata_only" for row in handoff.contrast_alias_rows))

    def test_module_keeps_bounded_import_and_call_surface(self) -> None:
        source = (PACKAGE_ROOT / "src" / "research_platform" / "neuro" / "roi_handoff.py").read_text(encoding="utf-8")
        forbidden = (
            "research_platform.analysis",
            "research_platform.hpc",
            "pipelines",
            "ops.",
            "nibabel",
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "nilearn",
            "rsatoolbox",
            "run_roi_build",
            "run_roi_extraction",
            "execute_loso_build_action",
            "run_mvpa",
            "extract_patterns",
            "compute_distances",
            "confidential-study-marker",
            "private-task-marker",
            "private-cohort-marker",
            "participant-alpha",
            "participant-beta",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_public_inventory_and_mapping_helpers_are_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            execution = _materialize_subject_ffx_outputs(plan)
            rows = inventory_completed_subject_ffx_outputs(plan, execution_result=execution)
            mapped = map_ffx_outputs_to_loso_fixed_effects_inputs(rows)

        self.assertTrue(all(row.complete for row in rows))
        self.assertEqual(len(mapped), len(rows))
        json.dumps([row.to_dict() for row in rows], allow_nan=False, sort_keys=True)
        json.dumps([row.to_dict() for row in mapped], allow_nan=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
