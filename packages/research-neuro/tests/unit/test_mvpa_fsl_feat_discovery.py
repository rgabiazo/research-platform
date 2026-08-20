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

from research_platform.neuro.mvpa import plan_fsl_feat_pattern_source, validate_mvpa_set_document


def _base_config(
    *,
    conditions: list[dict[str, object]] | None = None,
    missing: dict[str, str] | None = None,
    distance_noise: dict[str, object] | None = None,
    source_overrides: dict[str, object] | None = None,
    exclusions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    source: dict[str, object] = {
        "name": "first_level_pe",
        "backend": "fsl_feat_pe",
        "root_ref": "feat_root",
        "feat_dir_template": (
            "{subject_dir}/{session_dir}/func/"
            "sub-{subject_id}_{session_dir}_task-{task_id}_run-{run_id}_model-{model}.feat"
        ),
        "design_file": "design.fsf",
        "pe_image_template": "stats/pe{pe_number}.nii.gz",
        "noise_image_template": "stats/sigmasquareds.nii.gz",
    }
    if missing is not None:
        source["missing"] = missing
    if source_overrides is not None:
        source.update(source_overrides)

    return {
        "mvpa_set": {
            "name": "memory_mvpa",
            "subjects": ["001"],
            "sessions": ["01"],
            "runs": ["01"],
            "entities": {
                "task": "memory",
                "direction": "AP",
                "model": "modelA",
                "space": "MNI152NLin6Asym",
                "resolution": "2",
            },
            "conditions": conditions
            or [
                {"id": "faces", "fsl_ev_title": "Faces"},
                {"id": "places", "fsl_ev_title": "Places"},
            ],
            "pattern_sources": [source],
            "roi_sources": [{"name": "memory_rois", "source": "roi_set", "roi_set_ref": "memory_roi_set"}],
            "event_thresholds": {
                "min_events_per_condition_per_run": 1,
                "min_runs_per_condition": 1,
            },
            "exclusions": {"rules": exclusions or []},
            "distance": {
                "metrics": ["crossnobis"],
                "engine": "native_reference",
                "cross_validation": {"unit": "run", "grouping_columns": ["subject_id", "session_id", "run_id"]},
                "noise_normalization": distance_noise
                if distance_noise is not None
                else {"method": "diagonal", "variance_source": "sigmasquareds"},
            },
            "outputs": {
                "runtime_root": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/{mvpa_set}"},
            },
            "missing_input_policy": "warn",
        }
    }


def _feat_dir(root: Path) -> Path:
    return root / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_run-01_model-modelA.feat"


def _write_design(feat_dir: Path, text: str) -> None:
    feat_dir.mkdir(parents=True, exist_ok=True)
    (feat_dir / "design.fsf").write_text(textwrap.dedent(text), encoding="utf-8")


def _touch_stats(feat_dir: Path, *names: str) -> None:
    stats = feat_dir / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    for name in names:
        (stats / name).touch()


def _write_events(root: Path, condition_id: str, text: str) -> None:
    path = root / "sub-001" / "ses-01" / "func" / f"sub-001_ses-01_task-memory_run-01_desc-{condition_id}_events.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plan(document: dict[str, object], root: Path):
    return plan_fsl_feat_pattern_source(document, roots={"feat_root": root})


class FslFeatMvpaDiscoveryTests(unittest.TestCase):
    def test_valid_feat_tree_maps_conditions_to_pe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(
                feat_dir,
                """
                set fmri(evtitle1) "Faces"
                set fmri(evtitle2) "Places"
                """,
            )
            _touch_stats(feat_dir, "pe1.nii.gz", "pe2.nii.gz", "sigmasquareds.nii.gz")

            plan = _plan(_base_config(), root)

        rows = {row.condition_id: row for row in plan.condition_pe_rows}
        self.assertEqual(plan.status, "ok")
        self.assertFalse(plan.executed)
        self.assertEqual(rows["faces"].pe_number, 1)
        self.assertEqual(rows["places"].pe_number, 2)
        self.assertTrue(rows["faces"].pe_image.endswith("stats/pe1.nii.gz"))
        self.assertTrue(rows["places"].pe_image.endswith("stats/pe2.nii.gz"))
        self.assertTrue(rows["faces"].noise_image.endswith("stats/sigmasquareds.nii.gz"))

    def test_missing_feat_directory_uses_configured_fail_policy_without_duplicate_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plan = _plan(_base_config(missing={"feat_dir": "fail"}), Path(tmp_dir))

        feat_checks = [row for row in plan.input_checks if row.input_kind == "feat_dir"]
        design_checks = [row for row in plan.input_checks if row.input_kind == "design_fsf"]
        pe_checks = [row for row in plan.input_checks if row.input_kind == "pe_image"]
        self.assertEqual(plan.status, "error")
        self.assertEqual(feat_checks[0].status, "error")
        self.assertEqual(design_checks[0].status, "not_checked")
        self.assertFalse([row for row in pe_checks if row.status == "error"])

    def test_missing_design_fsf_produces_missing_input_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _feat_dir(root).mkdir(parents=True)

            plan = _plan(_base_config(missing={"design_fsf": "warn"}), root)

        checks = [row for row in plan.input_checks if row.input_kind == "design_fsf"]
        self.assertEqual(checks[0].exists, False)
        self.assertEqual(checks[0].status, "warning")
        self.assertIn("design.fsf", checks[0].message)

    def test_malformed_design_fsf_produces_design_parse_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(feat_dir, "set fmri(evtitle) Broken\n")
            _touch_stats(feat_dir, "sigmasquareds.nii.gz")

            plan = _plan(_base_config(missing={"design_parse": "warn"}), root)

        checks = [row for row in plan.input_checks if row.input_kind == "design_parse"]
        self.assertEqual(plan.status, "warning")
        self.assertEqual(checks[0].status, "warning")
        self.assertIn("could not be parsed", checks[0].message)

    def test_duplicate_ev_title_causes_ambiguous_condition_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(
                feat_dir,
                """
                set fmri(evtitle1) "Faces"
                set fmri(evtitle2) "Faces"
                """,
            )
            _touch_stats(feat_dir, "pe1.nii.gz", "pe2.nii.gz", "sigmasquareds.nii.gz")

            plan = _plan(_base_config(missing={"ambiguous_ev_title": "fail"}), root)

        checks = [row for row in plan.input_checks if row.input_kind == "ambiguous_ev_title"]
        self.assertEqual(plan.status, "error")
        self.assertEqual(checks[0].status, "error")
        self.assertIn("ambiguous", checks[0].message)

    def test_alias_mapping_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(feat_dir, 'set fmri(evtitle1) "Faces"\n')
            _touch_stats(feat_dir, "pe1.nii.gz", "sigmasquareds.nii.gz")
            document = _base_config(conditions=[{"id": "faces", "aliases": ["FaceTrials", "Faces"]}])

            plan = _plan(document, root)

        row = plan.condition_pe_rows[0]
        self.assertEqual(row.status, "ok")
        self.assertEqual(row.pe_number, 1)
        self.assertEqual(row.matched_alias, "Faces")

    def test_configured_event_file_counts_are_carried_to_condition_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(feat_dir, 'set fmri(evtitle1) "Faces"\n')
            _touch_stats(feat_dir, "pe1.nii.gz", "sigmasquareds.nii.gz")
            events_root = root / "events"
            _write_events(events_root, "faces", "0\t1\t1\n# comment\n2\t1\t1\n\n")
            document = _base_config(
                conditions=[{"id": "faces", "fsl_ev_title": "Faces", "event_id": "faces"}],
                source_overrides={
                    "events": {
                        "root_ref": "events_root",
                        "path": "{subject_dir}/{session_dir}/func/{subject_dir}_{session_dir}_task-{task_id}_run-{run_id}_desc-{event_id}_events.txt",
                    },
                },
            )

            plan = plan_fsl_feat_pattern_source(document, roots={"feat_root": root, "events_root": events_root})

        row = plan.condition_pe_rows[0]
        checks = [item for item in plan.input_checks if item.input_kind == "event_file_parse"]
        self.assertEqual(row.status, "ok")
        self.assertEqual(row.event_count, 2)
        self.assertTrue(row.event_file.endswith("_desc-faces_events.txt"))
        self.assertEqual(checks[0].event_count, 2)

    def test_missing_condition_ev_title_produces_condition_ev_title_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(feat_dir, 'set fmri(evtitle1) "Faces"\n')
            _touch_stats(feat_dir, "pe1.nii.gz", "sigmasquareds.nii.gz")
            document = _base_config(
                conditions=[{"id": "faces"}],
                missing={"condition_ev_title": "warn"},
            )

            plan = _plan(document, root)

        checks = [row for row in plan.input_checks if row.input_kind == "condition_ev_title"]
        self.assertEqual(plan.condition_pe_rows[0].status, "warning")
        self.assertEqual(checks[0].status, "warning")
        self.assertIn("No EV title", checks[0].message)

    def test_missing_pe_image_can_skip_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(feat_dir, 'set fmri(evtitle1) "Faces"\n')
            _touch_stats(feat_dir, "sigmasquareds.nii.gz")
            document = _base_config(
                conditions=[{"id": "faces", "fsl_ev_title": "Faces"}],
                missing={"pe_image": "skip"},
            )

            plan = _plan(document, root)

        checks = [row for row in plan.input_checks if row.input_kind == "pe_image"]
        self.assertEqual(plan.condition_pe_rows[0].status, "skipped")
        self.assertEqual(checks[0].status, "skipped")

    def test_missing_noise_warns_only_when_noise_normalization_requires_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(feat_dir, 'set fmri(evtitle1) "Faces"\n')
            _touch_stats(feat_dir, "pe1.nii.gz")
            document = _base_config(
                conditions=[{"id": "faces", "fsl_ev_title": "Faces"}],
                missing={"noise_image": "warn"},
            )

            diagonal_plan = _plan(document, root)
            identity_document = copy.deepcopy(document)
            identity_document["mvpa_set"]["distance"]["noise_normalization"] = {"method": "identity"}  # type: ignore[index]
            identity_plan = _plan(identity_document, root)

        diagonal_checks = [row for row in diagonal_plan.input_checks if row.input_kind == "noise_image"]
        identity_checks = [row for row in identity_plan.input_checks if row.input_kind == "noise_image"]
        self.assertEqual(diagonal_checks[0].status, "warning")
        self.assertEqual(identity_checks[0].status, "not_required")
        self.assertEqual(identity_plan.status, "ok")

    def test_exclusions_are_recorded_and_exact_run_match_marks_unit_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(feat_dir, 'set fmri(evtitle1) "Faces"\n')
            _touch_stats(feat_dir, "pe1.nii.gz", "sigmasquareds.nii.gz")
            document = _base_config(
                conditions=[{"id": "faces", "fsl_ev_title": "Faces"}],
                exclusions=[
                    {
                        "id": "motion_run",
                        "reason": "Configured motion exclusion",
                        "subject_id": "001",
                        "session_id": "01",
                        "run_id": "01",
                    }
                ],
            )

            plan = _plan(document, root)

        self.assertTrue(plan.units[0].excluded)
        self.assertEqual(plan.units[0].exclusion_reason, "Configured motion exclusion")
        self.assertEqual(plan.context["exclusions"][0]["status"], "configured")

    def test_to_dict_output_is_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_dir = _feat_dir(root)
            _write_design(feat_dir, 'set fmri(evtitle1) "Faces"\n')
            _touch_stats(feat_dir, "pe1.nii.gz", "sigmasquareds.nii.gz")
            document = _base_config(conditions=[{"id": "faces", "fsl_ev_title": "Faces"}])

            plan = _plan(document, root)

        encoded = json.dumps(plan.to_dict(), sort_keys=True)
        self.assertIn('"executed": false', encoded)
        self.assertIn('"condition_pe_rows"', encoded)

    def test_unresolved_placeholders_and_glob_templates_are_previewed_not_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            document = _base_config(
                source_overrides={
                    "feat_dir_template": "{subject_dir}/{session_dir}/func/*_{unknown}.feat",
                }
            )

            plan = _plan(document, root)

        feat_checks = [row for row in plan.input_checks if row.input_kind == "feat_dir"]
        self.assertEqual(feat_checks[0].status, "preview_only")
        self.assertIsNone(feat_checks[0].exists)
        self.assertIn("*_{unknown}.feat", feat_checks[0].path)

    def test_fsl_feat_optional_fields_get_tiny_shape_and_path_validation(self) -> None:
        document = _base_config(
            source_overrides={
                "feat_dir_template": "../model.feat",
                "case_sensitive": "yes",
                "missing": {"pe_image": "ignore"},
            }
        )

        errors = validate_mvpa_set_document(document)

        self.assertTrue(any("feat_dir_template" in error and "parent-directory" in error for error in errors))
        self.assertTrue(any("case_sensitive" in error and "boolean" in error for error in errors))
        self.assertTrue(any("missing.pe_image" in error and "warn" in error for error in errors))

    def test_importing_fsl_feat_discovery_uses_no_forbidden_dependencies(self) -> None:
        script = textwrap.dedent(
            """
            import builtins
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path("packages/research-neuro/src").resolve()))
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
            import research_platform.neuro.mvpa.fsl_feat  # noqa: F401
            import research_platform.neuro.mvpa  # noqa: F401
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
