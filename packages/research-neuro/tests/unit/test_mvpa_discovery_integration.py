from __future__ import annotations

from pathlib import Path
from unittest import mock
import copy
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.mvpa import plan_mvpa_discovery


def _base_config(
    *,
    pattern_source_overrides: dict[str, object] | None = None,
    roi_source_overrides: dict[str, object] | None = None,
    exclusions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    pattern_source: dict[str, object] = {
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
    if pattern_source_overrides is not None:
        pattern_source.update(pattern_source_overrides)

    roi_source: dict[str, object] = {
        "name": "explicit_rois",
        "source": "explicit_masks",
        "root_ref": "roi_root",
        "roi_labels": ["SeedA"],
        "mask_template": (
            "manual_masks/{subject_dir}/{session_dir}/"
            "sub-{subject_id}_{session_dir}_task-{task_id}_run-{run_id}_label-{roi_label}_mask.nii.gz"
        ),
    }
    if roi_source_overrides is not None:
        roi_source.update(roi_source_overrides)

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
            "conditions": [
                {"id": "faces", "fsl_ev_title": "Faces"},
                {"id": "places", "fsl_ev_title": "Places"},
            ],
            "pattern_sources": [pattern_source],
            "roi_sources": [roi_source],
            "event_thresholds": {
                "min_events_per_condition_per_run": 1,
                "min_runs_per_condition": 1,
            },
            "exclusions": {"rules": exclusions or []},
            "distance": {
                "metrics": ["crossnobis"],
                "engine": "native_reference",
                "cross_validation": {"unit": "run", "grouping_columns": ["subject_id", "session_id", "run_id"]},
                "noise_normalization": {"method": "diagonal", "variance_source": "sigmasquareds"},
            },
            "outputs": {
                "runtime_root": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/{mvpa_set}"},
                "published_root": {"root_ref": "dataset_derivatives_root", "path": "mvpa-crossnobis/{mvpa_set}"},
            },
            "missing_input_policy": "warn",
        }
    }


def _feat_dir(root: Path) -> Path:
    return root / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_run-01_model-modelA.feat"


def _write_valid_feat_tree(root: Path) -> None:
    feat_dir = _feat_dir(root)
    stats_dir = feat_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (feat_dir / "design.fsf").write_text(
        textwrap.dedent(
            """
            set fmri(evtitle1) "Faces"
            set fmri(evtitle2) "Places"
            """
        ),
        encoding="utf-8",
    )
    for name in ("pe1.nii.gz", "pe2.nii.gz", "sigmasquareds.nii.gz"):
        (stats_dir / name).touch()


def _roi_mask_path(root: Path) -> Path:
    return (
        root
        / "manual_masks"
        / "sub-001"
        / "ses-01"
        / "sub-001_ses-01_task-memory_run-01_label-SeedA_mask.nii.gz"
    )


def _roi_sidecar_path(mask_path: Path) -> Path:
    return mask_path.with_name(f"{mask_path.name[:-7]}.json")


def _write_valid_roi_tree(root: Path) -> None:
    mask_path = _roi_mask_path(root)
    sidecar_path = _roi_sidecar_path(mask_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.touch()
    sidecar_path.touch()


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _plan(document: dict[str, object], *, feat_root: Path, roi_root: Path):
    return plan_mvpa_discovery(document, roots={"feat_root": feat_root, "roi_root": roi_root})


class MvpaDiscoveryIntegrationTests(unittest.TestCase):
    def test_valid_integrated_plan_flattens_pattern_roi_checks_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            roi_root = root / "roi"
            _write_valid_feat_tree(feat_root)
            _write_valid_roi_tree(roi_root)

            plan = _plan(_base_config(), feat_root=feat_root, roi_root=roi_root)

        self.assertIn(plan.status, {"valid", "warning"})
        self.assertTrue(plan.pattern_source_rows)
        self.assertTrue(plan.condition_pe_rows)
        self.assertTrue(plan.roi_source_rows)
        self.assertTrue(plan.input_checks)
        self.assertTrue(plan.provenance_rows)
        json.dumps(plan.to_dict(), sort_keys=True)

    def test_old_deferred_behavior_still_works_without_roots_or_roi_sets(self) -> None:
        plan = plan_mvpa_discovery(_base_config())

        self.assertEqual(plan.status, "deferred")
        self.assertFalse(plan.backend_summary["integration_attempted"])
        self.assertEqual(plan.pattern_sources[0].status, "deferred")
        self.assertEqual(plan.roi_sources[0].status, "deferred")
        self.assertEqual(plan.pattern_source_rows, ())
        self.assertEqual(plan.roi_source_rows, ())
        self.assertEqual(plan.input_checks, ())

    def test_backend_discovery_flag_preserves_deferred_behavior_with_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_feat_tree(root / "feat")
            _write_valid_roi_tree(root / "roi")

            plan = plan_mvpa_discovery(
                _base_config(),
                roots={"feat_root": root / "feat", "roi_root": root / "roi"},
                enable_backend_discovery=False,
            )

        self.assertEqual(plan.status, "deferred")
        self.assertFalse(plan.backend_summary["integration_attempted"])
        self.assertEqual(plan.pattern_source_rows, ())
        self.assertEqual(plan.roi_source_rows, ())

    def test_invalid_config_returns_invalid_without_calling_subplanners(self) -> None:
        document = copy.deepcopy(_base_config())
        document["mvpa_set"]["distance"]["metrics"] = ["unsupported"]  # type: ignore[index]

        with mock.patch(
            "research_platform.neuro.mvpa.pattern_source_adapters.FslFeatPePatternSourceAdapter.plan_source"
        ) as fsl_plan:
            with mock.patch("research_platform.neuro.mvpa.plan.plan_mvpa_roi_sources") as roi_plan:
                result = plan_mvpa_discovery(document, roots={"feat_root": Path("feat"), "roi_root": Path("roi")})

        self.assertEqual(result.status, "invalid")
        self.assertTrue(result.errors)
        fsl_plan.assert_not_called()
        roi_plan.assert_not_called()

    def test_missing_feat_inputs_propagate_to_top_level_warnings_and_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "feat").mkdir()
            _write_valid_roi_tree(root / "roi")

            plan = _plan(_base_config(), feat_root=root / "feat", roi_root=root / "roi")

        feat_checks = [
            row
            for row in plan.input_checks
            if row.get("source_type") == "pattern_source" and row.get("input_kind") == "feat_dir"
        ]
        self.assertEqual(plan.status, "warning")
        self.assertTrue(plan.warnings)
        self.assertEqual(feat_checks[0]["status"], "warning")
        self.assertIn("pattern source 'first_level_pe'", plan.warnings[0])

    def test_missing_roi_inputs_propagate_to_top_level_warnings_checks_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_feat_tree(root / "feat")

            plan = _plan(_base_config(), feat_root=root / "feat", roi_root=root / "roi")

        roi_mask_checks = [
            row for row in plan.input_checks if row.get("source_type") == "roi_source" and row.get("input_kind") == "roi_mask"
        ]
        self.assertEqual(plan.status, "warning")
        self.assertTrue(plan.warnings)
        self.assertTrue(plan.provenance_rows)
        self.assertEqual(roi_mask_checks[0]["status"], "warning")
        self.assertIn("ROI source 'explicit_rois'", plan.warnings[0])

    def test_exclusions_are_preserved_in_base_and_detailed_rows(self) -> None:
        document = _base_config(
            exclusions=[
                {
                    "id": "motion_run",
                    "reason": "Configured motion exclusion",
                    "subject_id": "001",
                    "session_id": "01",
                    "run_id": "01",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_feat_tree(root / "feat")
            _write_valid_roi_tree(root / "roi")

            plan = _plan(document, feat_root=root / "feat", roi_root=root / "roi")

        self.assertEqual(plan.exclusions[0].id, "motion_run")
        self.assertTrue(any(row.get("excluded") for row in plan.pattern_source_rows))
        self.assertTrue(any(row.get("excluded") for row in plan.roi_source_rows))

    def test_event_threshold_rows_remain_not_evaluated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_feat_tree(root / "feat")
            _write_valid_roi_tree(root / "roi")

            plan = _plan(_base_config(), feat_root=root / "feat", roi_root=root / "roi")

        self.assertEqual(plan.event_thresholds["min_events_per_condition_per_run"], 1)
        self.assertTrue(plan.event_threshold_rows)
        self.assertTrue(all(row["status"] == "not_evaluated" for row in plan.event_threshold_rows))

    def test_non_fsl_pattern_backends_remain_deferred(self) -> None:
        document = _base_config(
            pattern_source_overrides={
                "backend": "bids_derivative_pattern_table",
                "path": "patterns/{subject_dir}/{session_dir}/patterns.tsv",
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_valid_roi_tree(root / "roi")

            plan = _plan(document, feat_root=root / "feat", roi_root=root / "roi")

        self.assertEqual(plan.status, "deferred")
        self.assertEqual(plan.pattern_sources[0].status, "deferred")
        self.assertEqual(plan.pattern_source_rows, ())
        self.assertEqual(plan.condition_pe_rows, ())

    def test_plan_only_integration_does_not_write_outputs_or_new_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            roi_root = root / "roi"
            _write_valid_feat_tree(feat_root)
            _write_valid_roi_tree(roi_root)
            before = _relative_files(root)

            plan = _plan(_base_config(), feat_root=feat_root, roi_root=roi_root)

            after = _relative_files(root)

        self.assertFalse(plan.executed)
        self.assertEqual(after, before)

    def test_integration_imports_use_no_forbidden_dependencies(self) -> None:
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
            import research_platform.neuro.mvpa.plan  # noqa: F401
            import research_platform.neuro.mvpa  # noqa: F401
            import test_mvpa_discovery_integration  # noqa: F401
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
