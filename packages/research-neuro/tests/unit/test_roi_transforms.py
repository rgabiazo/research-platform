from __future__ import annotations

from pathlib import Path
import ast
import json
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.roi_transforms import (
    DEFAULT_INTERPOLATION,
    build_ants_apply_transforms_argv,
    plan_mni_to_t1w_roi_transforms,
    preflight_ants_apply_transforms,
    validate_mni_to_t1w_roi_transform_document,
)


PARTICIPANT_ID = "participant-a"
SESSION_ID = "session-a"
TASK_ID = "task-alpha"
RUN_ID = "run-a"
MODEL_ID = "model-alpha"
CONTRAST_ID = "contrast-alpha"
ROI_LABEL = "roi-alpha"


def _write(path: Path, text: str = "placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _snapshot(root: Path) -> tuple[tuple[str, str, int | None], ...]:
    rows: list[tuple[str, str, int | None]] = []
    for path in sorted(root.rglob("*")):
        kind = "file" if path.is_file() else "dir"
        size = path.stat().st_size if path.is_file() else None
        rows.append((path.relative_to(root).as_posix(), kind, size))
    return tuple(rows)


def _chmod_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | 0o111)


class RoiTransformPlanningTests(unittest.TestCase):
    def _roots(self, root: Path) -> dict[str, Path]:
        return {
            "roi-root": root / "roi root with spaces",
            "target-root": root / "target-root",
            "xfm-root": root / "xfm root with spaces",
            "out-root": root / "planned outputs",
            "tool-root": root / "tool-root",
        }

    def _config(self, executable: Path | None = None) -> dict[str, object]:
        config: dict[str, object] = {
            "missing_policy": "fail",
            "tool_missing_policy": "fail",
            "qc": {
                "min_voxels_warn": 12,
                "min_voxels_fail": 1,
                "coverage_min_overlap_fraction_warn": 0.80,
            },
            "source_masks": [
                {
                    "subject_id": PARTICIPANT_ID,
                    "session_id": SESSION_ID,
                    "task_id": TASK_ID,
                    "run_id": RUN_ID,
                    "model": MODEL_ID,
                    "contrast_id": CONTRAST_ID,
                    "roi_label": ROI_LABEL,
                    "source_space": "MNI",
                    "source_mask_path": {"root_ref": "roi-root", "path": "mni masks/roi alpha mask.nii.gz"},
                    "target_reference_path": {"root_ref": "target-root", "path": "refs/t1 ref.nii.gz"},
                    "coverage_mask_path": {"root_ref": "target-root", "path": "refs/brain mask.nii.gz"},
                    "planned_output_mask_path": {
                        "root_ref": "out-root",
                        "path": "transformed/{subject_id}/{session_id}/{run_id}/{roi_label}_mask.nii.gz",
                    },
                    "transforms": [
                        {"path": {"root_ref": "xfm-root", "path": "warp one.nii.gz"}},
                        {"path": {"root_ref": "xfm-root", "path": "affine two.mat"}, "invert": True},
                    ],
                }
            ],
        }
        if executable is not None:
            config["tool"] = {"executable": str(executable)}
        return config

    def _materialize_inputs(self, root: Path, executable: Path | None = None) -> None:
        roots = self._roots(root)
        for path in roots.values():
            path.mkdir(parents=True, exist_ok=True)
        _write(roots["roi-root"] / "mni masks" / "roi alpha mask.nii.gz")
        _write(roots["target-root"] / "refs" / "t1 ref.nii.gz")
        _write(roots["target-root"] / "refs" / "brain mask.nii.gz")
        _write(roots["xfm-root"] / "warp one.nii.gz")
        _write(roots["xfm-root"] / "affine two.mat")
        if executable is not None:
            _write(executable, "#!/bin/sh\nexit 0\n")
            _chmod_executable(executable)

    def test_structural_validation_does_not_require_runtime_paths_to_exist(self) -> None:
        self.assertEqual(validate_mni_to_t1w_roi_transform_document(self._config()), [])

    def test_structural_validation_rejects_malformed_row_containers_and_paths(self) -> None:
        errors = validate_mni_to_t1w_roi_transform_document(
            {
                "roi_transform_plan": {
                    "source_masks": "not-a-row-container",
                    "target_references": [{"target_reference_path": {"root_ref": "target-root"}}],
                    "transform_chains": [{"path": 42}],
                    "outputs": {"root_ref": "out-root", "path": "output.nii.gz"},
                }
            }
        )

        self.assertTrue(any("source_masks must contain a mapping or sequence" in error for error in errors))
        self.assertTrue(any("target_reference_path must define one of" in error for error in errors))
        self.assertTrue(any("transform_chains[0].path must be a string" in error for error in errors))

    def test_complete_transform_plan_is_plan_only_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)
            before = _snapshot(root)

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

            after = _snapshot(root)

        self.assertEqual(before, after)
        self.assertFalse(plan.executed)
        self.assertTrue(plan.plan_only)
        self.assertEqual(plan.status, "ok")
        self.assertEqual(len(plan.command_plans), 1)
        self.assertEqual(plan.command_plans[0].status, "planned")
        self.assertFalse(plan.errors)

    def test_missing_ants_apply_transforms_produces_preflight_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._materialize_inputs(root)
            config = self._config()
            config["tool_missing_policy"] = "warn"
            with mock.patch("research_platform.neuro.roi_transforms.shutil.which", return_value=None):
                plan = plan_mni_to_t1w_roi_transforms(config, roots=self._roots(root))

        self.assertEqual(plan.status, "warning")
        self.assertEqual(plan.tool_preflight[0].status, "warning")
        self.assertIn("antsApplyTransforms", plan.tool_preflight[0].message)
        self.assertTrue(plan.warnings)

    def test_preflight_can_report_missing_tool_as_error(self) -> None:
        with mock.patch("research_platform.neuro.roi_transforms.shutil.which", return_value=None):
            row = preflight_ants_apply_transforms(missing_policy="fail")

        self.assertEqual(row.status, "error")
        self.assertFalse(row.available)
        self.assertIn("Install ANTs separately", row.message)

    def test_configured_executable_path_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

        self.assertEqual(plan.tool_preflight[0].requested_executable, str(executable))
        self.assertEqual(plan.command_plans[0].argv[0], str(executable))

    def test_shell_safe_argv_vectors_keep_paths_with_spaces_as_single_elements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "ants Apply Transforms"
            self._materialize_inputs(root, executable=executable)

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

        argv = plan.command_plans[0].argv
        self.assertIsInstance(argv, tuple)
        self.assertEqual(argv[0], str(executable))
        self.assertIn("roi alpha mask.nii.gz", argv[argv.index("-i") + 1])
        self.assertIn("t1 ref.nii.gz", argv[argv.index("-r") + 1])
        self.assertTrue(any(arg.endswith("warp one.nii.gz") for arg in argv))

    def test_nearest_neighbor_interpolation_for_masks_by_default(self) -> None:
        argv = build_ants_apply_transforms_argv(
            executable="antsApplyTransforms",
            input_mask="roi mask.nii.gz",
            reference_image="reference image.nii.gz",
            output_path="output mask.nii.gz",
            transforms=["transform one.nii.gz"],
        )

        self.assertEqual(argv[argv.index("-n") + 1], DEFAULT_INTERPOLATION)
        self.assertEqual(argv[argv.index("-i") + 1], "roi mask.nii.gz")
        self.assertEqual(argv[argv.index("-t") + 1], "transform one.nii.gz")

    def test_missing_source_mask_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)
            (self._roots(root)["roi-root"] / "mni masks" / "roi alpha mask.nii.gz").unlink()

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

        self.assertEqual(plan.source_masks[0].status, "error")
        self.assertTrue(any("source MNI ROI mask is missing" in message for message in plan.errors))

    def test_missing_target_reference_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)
            (self._roots(root)["target-root"] / "refs" / "t1 ref.nii.gz").unlink()

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

        self.assertEqual(plan.target_references[0].status, "error")
        self.assertTrue(any("target reference image is missing" in message for message in plan.errors))

    def test_missing_transform_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)
            (self._roots(root)["xfm-root"] / "warp one.nii.gz").unlink()

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

        self.assertEqual(plan.transform_chains[0].status, "error")
        self.assertTrue(any("transform chain item 0 is missing" in message for message in plan.errors))

    def test_missing_transform_file_can_warn_by_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)
            (self._roots(root)["xfm-root"] / "warp one.nii.gz").unlink()
            config = self._config(executable)
            config["missing_policy"] = "warn"

            plan = plan_mni_to_t1w_roi_transforms(config, roots=self._roots(root))

        self.assertEqual(plan.transform_chains[0].status, "warning")
        self.assertTrue(any("transform chain item 0 is missing" in message for message in plan.warnings))

    def test_ordered_transform_chain_and_inverted_flag_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

        self.assertEqual([row.order_index for row in plan.transform_chains], [0, 1])
        self.assertFalse(plan.transform_chains[0].invert)
        self.assertTrue(plan.transform_chains[1].invert)
        transform_args = [plan.command_plans[0].argv[index + 1] for index, arg in enumerate(plan.command_plans[0].argv) if arg == "-t"]
        self.assertTrue(transform_args[0].endswith("warp one.nii.gz"))
        self.assertTrue(transform_args[1].endswith("affine two.mat,1]"))
        self.assertTrue(transform_args[1].startswith("["))

    def test_planned_output_paths_are_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

        output = plan.planned_outputs[0]
        self.assertEqual(output.status, "planned")
        self.assertIn(f"transformed/{PARTICIPANT_ID}/{SESSION_ID}/{RUN_ID}/{ROI_LABEL}_mask.nii.gz", output.output_mask_path)
        self.assertIn(output.output_mask_path, plan.command_plans[0].argv)

    def test_qc_preview_rows_include_preflight_and_post_execution_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))

        kinds = {row.check_kind for row in plan.qc_preview}
        self.assertIn("source_mask_exists", kinds)
        self.assertIn("target_reference_exists", kinds)
        self.assertIn("transform_exists", kinds)
        self.assertIn("coverage_mask_exists", kinds)
        self.assertIn("planned_output_path", kinds)
        self.assertIn("post_execution_geometry_matches_reference", kinds)
        self.assertIn("post_execution_voxel_count", kinds)
        self.assertIn("post_execution_empty_mask", kinds)
        self.assertIn("post_execution_small_mask", kinds)
        self.assertIn("post_execution_overlap_coverage", kinds)
        self.assertIn("interpolation_policy", kinds)
        self.assertIn("transform_chain_provenance", kinds)

    def test_json_safe_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)

            plan = plan_mni_to_t1w_roi_transforms(self._config(executable), roots=self._roots(root))
            encoded = json.dumps(plan.to_dict(), allow_nan=False, sort_keys=True)

        self.assertIn('"executed": false', encoded)
        self.assertIn('"plan_only": true', encoded)

    def test_environment_root_placeholder_paths_are_preview_only(self) -> None:
        config = {
            "tool_missing_policy": "warn",
            "missing_policy": "warn",
            "source_masks": [
                {
                    "subject_id": PARTICIPANT_ID,
                    "session_id": SESSION_ID,
                    "run_id": RUN_ID,
                    "roi_label": ROI_LABEL,
                    "source_mask_path": "${ROI_ROOT:-roi}/roi-alpha.nii.gz",
                    "target_reference_path": "${TARGET_ROOT:-target}/t1w.nii.gz",
                    "planned_output_mask_path": "${OUTPUT_ROOT:-outputs}/roi-alpha_t1w.nii.gz",
                    "transforms": [{"path": "${TRANSFORM_ROOT:-xfm}/mni-to-t1w.mat"}],
                }
            ],
        }
        with mock.patch("research_platform.neuro.roi_transforms.shutil.which", return_value=None):
            plan = plan_mni_to_t1w_roi_transforms(config)

        self.assertEqual(plan.source_masks[0].exists, None)
        self.assertEqual(plan.target_references[0].exists, None)
        self.assertEqual(plan.transform_chains[0].exists, None)

    def test_roi_build_output_dictionary_can_supply_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)
            roi_build_outputs = {
                "actions": [
                    {
                        "subject_id": PARTICIPANT_ID,
                        "session_id": SESSION_ID,
                        "run_id": RUN_ID,
                        "roi_label": ROI_LABEL,
                        "mask_path": str(self._roots(root)["roi-root"] / "mni masks" / "roi alpha mask.nii.gz"),
                        "target_reference_path": {"root_ref": "target-root", "path": "refs/t1 ref.nii.gz"},
                        "planned_output_mask_path": {
                            "root_ref": "out-root",
                            "path": "transformed/{subject_id}/{session_id}/{run_id}/{roi_label}_mask.nii.gz",
                        },
                        "transforms": [{"path": {"root_ref": "xfm-root", "path": "warp one.nii.gz"}}],
                    }
                ]
            }

            plan = plan_mni_to_t1w_roi_transforms(
                {"tool": {"executable": str(executable)}, "missing_policy": "fail"},
                roots=self._roots(root),
                roi_build_outputs=roi_build_outputs,
            )

        self.assertEqual(len(plan.source_masks), 1)
        self.assertEqual(plan.source_masks[0].source_mask_path, roi_build_outputs["actions"][0]["mask_path"])
        self.assertEqual(plan.status, "ok")

    def test_separate_target_and_transform_config_rows_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable = self._roots(root)["tool-root"] / "antsApplyTransforms"
            self._materialize_inputs(root, executable=executable)
            config = {
                "tool": {"executable": str(executable)},
                "missing_policy": "fail",
                "source_masks": [
                    {
                        "subject_id": PARTICIPANT_ID,
                        "session_id": SESSION_ID,
                        "run_id": RUN_ID,
                        "roi_label": ROI_LABEL,
                        "source_mask_path": {"root_ref": "roi-root", "path": "mni masks/roi alpha mask.nii.gz"},
                    }
                ],
                "target_references": [
                    {"target_reference_path": {"root_ref": "target-root", "path": "refs/t1 ref.nii.gz"}}
                ],
                "transform_chains": [
                    {"path": {"root_ref": "xfm-root", "path": "warp one.nii.gz"}},
                    {"path": {"root_ref": "xfm-root", "path": "affine two.mat"}, "invert": True},
                ],
                "outputs": {
                    "root_ref": "out-root",
                    "path_template": "transformed/{subject_id}/{session_id}/{run_id}/{roi_label}_mask.nii.gz",
                },
            }

            plan = plan_mni_to_t1w_roi_transforms(config, roots=self._roots(root))

        self.assertEqual(plan.status, "ok")
        self.assertEqual(len(plan.transform_chains), 2)
        self.assertEqual(plan.command_plans[0].argv.count("-t"), 2)

    def test_selector_expansion_renders_subject_run_target_and_transform_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            roots = self._roots(root)
            executable = roots["tool-root"] / "antsApplyTransforms"
            for path in roots.values():
                path.mkdir(parents=True, exist_ok=True)
            _write(executable, "#!/bin/sh\nexit 0\n")
            _chmod_executable(executable)
            _write(roots["roi-root"] / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_label-SeedA_mask.nii.gz")
            _write(roots["target-root"] / "sub-001" / "ses-01" / "run-01" / "pe1.nii.gz")
            _write(roots["xfm-root"] / "sub-001" / "ses-01" / "anat" / "sub-001_ses-01_from-MNI_to-T1w.h5")
            config = {
                "selectors": {"subjects": ["sub-001"], "sessions": ["ses-01"], "runs": ["run-01"]},
                "missing_policy": "fail",
                "tool_missing_policy": "fail",
                "tool": {"executable": str(executable)},
                "source_masks": [
                    {
                        "roi_label": "SeedA",
                        "source_mask_path": {
                            "root_ref": "roi-root",
                            "path": "{subject_dir}/{session_dir}/func/{subject_dir}_{session_dir}_label-{roi_label}_mask.nii.gz",
                        },
                    }
                ],
                "target_references": [
                    {
                        "target_reference_path": {
                            "root_ref": "target-root",
                            "path": "{subject_dir}/{session_dir}/{run_entity}/pe1.nii.gz",
                        }
                    }
                ],
                "transform_chains": [
                    {
                        "path": {
                            "root_ref": "xfm-root",
                            "path": "{subject_dir}/{session_dir}/anat/{subject_dir}_{session_dir}_from-MNI_to-T1w.h5",
                        }
                    }
                ],
                "outputs": {
                    "root_ref": "out-root",
                    "path": "transformed/{subject_dir}/{session_dir}/{run_entity}/{roi_label}_mask.nii.gz",
                },
            }

            plan = plan_mni_to_t1w_roi_transforms(config, roots=roots)

        self.assertEqual(plan.status, "ok")
        self.assertEqual(plan.source_masks[0].subject_id, "001")
        self.assertEqual(plan.source_masks[0].run_id, "01")
        self.assertIn("sub-001/ses-01/run-01/pe1.nii.gz", plan.target_references[0].target_reference_path)
        self.assertIn("sub-001_ses-01_from-MNI_to-T1w.h5", plan.transform_chains[0].transform_path)
        self.assertIn("transformed/sub-001/ses-01/run-01/SeedA_mask.nii.gz", plan.planned_outputs[0].output_mask_path)

    def test_plan_module_has_no_forbidden_runtime_imports_or_study_constants(self) -> None:
        module_path = PACKAGE_ROOT / "src" / "research_platform" / "neuro" / "roi_transforms.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

        forbidden_imports = {
            "research_analysis",
            "pipelines",
            "ops",
            "research_hpc",
            "nibabel",
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "nilearn",
            "rsatoolbox",
            "ants",
            "fsl",
        }
        self.assertFalse(imported_roots & forbidden_imports)
        forbidden_text = [
            "confidential-study-marker",
            "private-task-marker",
            "private-cohort-marker",
            "participant-alpha",
            "participant-beta",
            "cope1",
            "cope2",
            "private-absolute-path-marker",
        ]
        for value in forbidden_text:
            self.assertNotIn(value, source)


if __name__ == "__main__":
    unittest.main()
