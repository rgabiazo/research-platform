from __future__ import annotations

from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.mvpa import plan_mvpa_roi_sources


def _base_config(source: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "mvpa_set": {
            "name": "memory_mvpa",
            "subjects": ["sub-001"],
            "sessions": ["ses-01"],
            "runs": ["01"],
            "entities": {
                "task": "memory",
                "direction": "AP",
                "model": "ModelA",
                "space": "MNI152NLin6Asym",
                "resolution": "2",
            },
            "conditions": [{"id": "faces", "fsl_ev_title": "Faces"}],
            "pattern_sources": [
                {
                    "name": "first_level_pe",
                    "backend": "fsl_feat_pe",
                    "root_ref": "feat_root",
                    "feat_dir_template": "{subject_dir}/{session_dir}/func/model.feat",
                }
            ],
            "roi_sources": [source or _publication_source()],
            "distance": {
                "metrics": ["crossnobis"],
                "engine": "native_reference",
                "cross_validation": {"unit": "run", "grouping_columns": ["subject_id", "session_id", "run_id"]},
                "noise_normalization": {"method": "identity"},
            },
            "outputs": {
                "runtime_root": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/{mvpa_set}"},
            },
            "missing_input_policy": "warn",
        }
    }


def _publication_source(**overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "name": "published_rois",
        "source": "roi_set_publication",
        "roi_set_ref": "loso_modelA",
        "roi_labels": ["SeedA"],
    }
    source.update(overrides)
    return source


def _runtime_source(source_value: str = "roi_set_runtime", **overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "name": "runtime_rois",
        "source": source_value,
        "roi_set_ref": "loso_modelA",
        "roi_labels": ["SeedA"],
    }
    source.update(overrides)
    return source


def _explicit_source(**overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "name": "explicit_rois",
        "source": "explicit_masks",
        "root_ref": "roi_root",
        "roi_labels": ["SeedA"],
        "mask_template": (
            "manual_masks/{subject_dir}/{session_dir}/"
            "sub-{subject_id}_{session_dir}_task-{task_id}_run-{run_id}_label-{roi_label}_mask.nii.gz"
        ),
    }
    source.update(overrides)
    return source


def _roi_set_document() -> dict[str, object]:
    return {
        "roi_set": {
            "name": "loso_modelA",
            "subjects": ["sub-001"],
            "session": "ses-01",
            "task": "memory",
            "model": "ModelA",
            "space": "MNI152NLin6Asym",
            "resolution": "2",
            "direction": "AP",
            "outputs": {
                "root_ref": "derivatives_root",
                "path": ".research-platform/roi-loso-flame1-runtime",
            },
            "publication": {
                "enabled": True,
                "root": {"root_ref": "derivatives_root", "path": "roi-loso-flame1"},
                "mask_desc": "{model}LOSOFlame1Sphere{sphere_radius_mm}mm",
            },
            "contrasts": [{"id": "CondA", "cope_number": 1, "desc": "CondA"}],
            "rois": [
                {
                    "label": "SeedA",
                    "family": "loso_group_map",
                    "backend": "fsl_flame1",
                    "desc": "CondA",
                    "contrast": "CondA",
                    "sphere_radius_mm": 6,
                }
            ],
        }
    }


def _plan(document: dict[str, object], *, root: Path | None = None):
    roots = {"derivatives_root": root, "roi_root": root} if root is not None else None
    return plan_mvpa_roi_sources(document, roots=roots, roi_sets={"loso_modelA": _roi_set_document()})


def _sidecar_path(mask_path: str | Path) -> Path:
    path = Path(mask_path)
    if path.name.endswith(".nii.gz"):
        return path.with_name(f"{path.name[:-7]}.json")
    return path.with_suffix(".json")


class MvpaRoiSourcePlanningTests(unittest.TestCase):
    def test_roi_set_publication_path_previews(self) -> None:
        plan = _plan(_base_config(_publication_source()))

        row = plan.rows[0]
        self.assertEqual(row.source, "roi_set_publication")
        self.assertIn("derivatives_root/roi-loso-flame1/masks/sub-001/ses-01/func", row.mask_path)
        self.assertIn("label-SeedA_contrast-CondA_desc-ModelALOSOFlame1Sphere6mm_mask.nii.gz", row.mask_path)
        self.assertEqual(row.status, "preview_only")

    def test_roi_set_runtime_path_previews(self) -> None:
        plan = _plan(_base_config(_runtime_source()))

        row = plan.rows[0]
        self.assertEqual(row.source, "roi_set_runtime")
        self.assertIn("derivatives_root/.research-platform/roi-loso-flame1-runtime/rois/loso_modelA", row.mask_path)
        self.assertIn("label-SeedA_desc-CondA_mask.nii.gz", row.mask_path)

    def test_roi_set_alias_behaves_like_runtime_source(self) -> None:
        runtime_plan = _plan(_base_config(_runtime_source()))
        alias_plan = _plan(_base_config(_runtime_source("roi_set")))

        self.assertEqual(alias_plan.rows[0].configured_source, "roi_set")
        self.assertEqual(alias_plan.rows[0].source, "roi_set_runtime")
        self.assertEqual(alias_plan.rows[0].mask_path, runtime_plan.rows[0].mask_path)

    def test_explicit_masks_path_previews(self) -> None:
        plan = _plan(_base_config(_explicit_source()))

        row = plan.rows[0]
        self.assertEqual(row.source, "explicit_masks")
        self.assertIn("roi_root/manual_masks/sub-001/ses-01", row.mask_path)
        self.assertIn("run-01_label-SeedA_mask.nii.gz", row.mask_path)
        self.assertEqual(row.status, "preview_only")

    def test_sidecar_template_and_same_stem_json_inference(self) -> None:
        explicit_plan = _plan(_base_config(_explicit_source()))
        templated_plan = _plan(
            _base_config(
                _explicit_source(
                    sidecar_template="manual_masks/{subject_dir}/{session_dir}/sidecars/{roi_label}_mask.json"
                )
            )
        )

        self.assertEqual(explicit_plan.rows[0].sidecar_path, str(_sidecar_path(explicit_plan.rows[0].mask_path)))
        self.assertIn("sidecars/SeedA_mask.json", templated_plan.rows[0].sidecar_path)

    def test_concrete_mask_and_sidecar_existence_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            document = _base_config(_explicit_source())
            preview = _plan(document, root=root)
            mask_path = Path(preview.rows[0].mask_path)
            sidecar_path = _sidecar_path(mask_path)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.write_text("mask placeholder", encoding="utf-8")
            sidecar_path.write_text("{}", encoding="utf-8")

            plan = _plan(document, root=root)

        self.assertEqual(plan.status, "ok")
        self.assertTrue(plan.rows[0].mask_exists)
        self.assertTrue(plan.rows[0].sidecar_exists)
        self.assertEqual(plan.rows[0].status, "ok")

    def test_missing_roi_mask_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plan = _plan(_base_config(_explicit_source(missing={"roi_mask": "warn"})), root=Path(tmp_dir))

        mask_checks = [row for row in plan.input_checks if row.input_kind == "roi_mask"]
        self.assertEqual(plan.status, "warning")
        self.assertEqual(mask_checks[0].status, "warning")
        self.assertIn("ROI mask is missing", mask_checks[0].message)

    def test_missing_roi_sidecar_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            document = _base_config(_explicit_source(missing={"roi_sidecar": "warn"}))
            preview = _plan(document, root=root)
            mask_path = Path(preview.rows[0].mask_path)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.write_text("mask placeholder", encoding="utf-8")

            plan = _plan(document, root=root)

        sidecar_checks = [row for row in plan.input_checks if row.input_kind == "roi_sidecar"]
        self.assertEqual(plan.status, "warning")
        self.assertEqual(plan.rows[0].mask_exists, True)
        self.assertEqual(plan.rows[0].sidecar_exists, False)
        self.assertEqual(sidecar_checks[0].status, "warning")
        self.assertEqual(plan.provenance_rows[0].status, "warning")

    def test_missing_roi_mask_skip_marks_row_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plan = _plan(_base_config(_explicit_source(missing={"roi_mask": "skip"})), root=Path(tmp_dir))

        mask_checks = [row for row in plan.input_checks if row.input_kind == "roi_mask"]
        sidecar_checks = [row for row in plan.input_checks if row.input_kind == "roi_sidecar"]
        self.assertEqual(plan.rows[0].status, "skipped")
        self.assertEqual(mask_checks[0].status, "skipped")
        self.assertEqual(sidecar_checks[0].status, "not_checked")

    def test_missing_roi_mask_fail_and_optional_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            document = _base_config(_explicit_source(missing={"roi_mask": "fail"}))
            plan = _plan(document, root=root)

            with self.assertRaises(ValueError):
                plan_mvpa_roi_sources(
                    document,
                    roots={"roi_root": root},
                    roi_sets={"loso_modelA": _roi_set_document()},
                    raise_on_fail_policy=True,
                )

        self.assertEqual(plan.status, "error")
        self.assertEqual(plan.rows[0].status, "error")
        self.assertTrue(plan.errors)

    def test_unresolved_placeholders_are_previewed_not_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plan = _plan(
                _base_config(_explicit_source(mask_template="manual_masks/{unknown}/label-{roi_label}_mask.nii.gz")),
                root=Path(tmp_dir),
            )

        mask_checks = [row for row in plan.input_checks if row.input_kind == "roi_mask"]
        self.assertEqual(mask_checks[0].status, "preview_only")
        self.assertIsNone(mask_checks[0].exists)
        self.assertIn("{unknown}", mask_checks[0].path)

    def test_glob_like_templates_are_previewed_not_globbed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            matching = root / "manual_masks" / "sub-001" / "SeedA_mask.nii.gz"
            matching.parent.mkdir(parents=True, exist_ok=True)
            matching.write_text("mask placeholder", encoding="utf-8")

            plan = _plan(
                _base_config(_explicit_source(mask_template="manual_masks/{subject_dir}/*_mask.nii.gz")),
                root=root,
            )

        mask_checks = [row for row in plan.input_checks if row.input_kind == "roi_mask"]
        self.assertEqual(mask_checks[0].status, "preview_only")
        self.assertIsNone(mask_checks[0].exists)
        self.assertIn("*_mask.nii.gz", mask_checks[0].path)

    def test_exclusions_are_recorded_and_exact_run_match_marks_unit_excluded(self) -> None:
        document = _base_config(_explicit_source(missing={"roi_mask": "fail"}))
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["subjects"] = ["sub-001", "sub-002"]  # type: ignore[index]
        mvpa_set["runs"] = ["01", "02"]  # type: ignore[index]
        mvpa_set["exclusions"] = {  # type: ignore[index]
            "rules": [
                {
                    "id": "motion_run",
                    "reason": "Configured motion exclusion",
                    "subject_id": "001",
                    "session_id": "01",
                    "run_id": "02",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            plan = _plan(document, root=Path(tmp_dir))

        excluded = [row for row in plan.rows if row.excluded]
        self.assertEqual(len(plan.rows), 4)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].subject_id, "001")
        self.assertEqual(excluded[0].run_id, "02")
        self.assertEqual(excluded[0].exclusion_reason, "Configured motion exclusion")

    def test_json_dumps_plan_to_dict_works(self) -> None:
        plan = _plan(_base_config(_publication_source()))

        encoded = json.dumps(plan.to_dict(), sort_keys=True)

        self.assertIn('"executed": false', encoded)
        self.assertIn('"provenance_rows"', encoded)
        self.assertIn('"rows"', encoded)

    def test_plan_only_does_not_build_masks_or_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(_base_config(_explicit_source()), root=root)

            self.assertEqual(list(root.iterdir()), [])

        self.assertFalse(plan.executed)
        self.assertFalse(plan.rows[0].mask_exists)

    def test_importing_roi_sources_and_tests_use_no_forbidden_dependencies(self) -> None:
        script = textwrap.dedent(
            """
            import builtins
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path("packages/research-neuro/src").resolve()))
            sys.path.insert(0, str(Path("packages/research-neuro/tests/unit").resolve()))
            forbidden = {
                "nibabel",
                "nilearn",
                "rsatoolbox",
                "numpy",
                "pandas",
                "polars",
                "scipy",
                "mvpa2",
                "sklearn",
                "research_platform.core",
                "research_platform.bids",
                "research_platform.analysis",
                "research_platform.viz",
                "research_platform.ml",
                "pipelines",
                "ops",
            }
            real_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name in forbidden or any(name.startswith(prefix + ".") for prefix in forbidden):
                    raise RuntimeError(f"forbidden import: {name}")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import
            import research_platform.neuro.mvpa.roi_sources  # noqa: F401
            import test_mvpa_roi_sources  # noqa: F401
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PACKAGE_ROOT.parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
