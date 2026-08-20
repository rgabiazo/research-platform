from __future__ import annotations

from pathlib import Path
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.roi import (
    runtime_existing_output_policy,
    validate_extraction_set_document,
    validate_roi_set_document,
    validate_roi_sidecar_document,
)
from research_platform.neuro.roi_scaffold import (
    apply_extraction_set_overrides,
    apply_roi_set_overrides,
    build_extraction_set_document,
    build_roi_set_document,
    extraction_set_template_help,
    render_yaml,
    roi_set_template_help,
    supported_extraction_set_templates,
    supported_path_profiles,
    supported_roi_set_templates,
    validate_extraction_set_scaffold,
    validate_roi_set_scaffold,
)


FORBIDDEN_SCAFFOLD_STRINGS = (
    "/home/alice/private-example",
    "/mnt/private-volume",
    "PrivateStudy",
    "ActiveStudy",
)


def _synthetic_personal_root() -> Path:
    return Path("/home/alice/private-example/rp-roi-test")


def _valid_roi_set() -> dict[str, object]:
    return {
        "roi_set": {
            "name": "modelA_rois",
            "desc": "ModelA",
            "rois": [
                {
                    "label": "ManualMask",
                    "family": "manual_mask",
                    "desc": "CuratedMask",
                    "source": "${PROJECT_ROI_ROOT}/manual/sub-001_mask.nii.gz",
                },
                {
                    "label": "SeedSphere",
                    "family": "coordinate_sphere",
                    "coordinate": [0, -52, 26],
                    "radius_mm": 6,
                },
                {
                    "label": "AtlasLabel",
                    "family": "atlas_label",
                    "atlas": "ExampleAtlas",
                    "labels": [17, 53],
                },
                {
                    "label": "ThresholdMap",
                    "family": "functional_threshold_map",
                    "backend": "generic_nifti",
                    "source_contrast": "condition_a_gt_baseline",
                    "z_threshold": 3.1,
                    "exploratory_z_threshold": 2.3,
                    "min_voxels_fail": 4,
                    "min_voxels_warn": 12,
                },
                {
                    "label": "LosoMap",
                    "family": "loso_group_map",
                    "backend": "fsl_flame1",
                    "search_radius_mm": 12,
                    "sphere_radius_mm": 6,
                },
                {
                    "label": "HookRoi",
                    "family": "data_driven_hook",
                    "backend": "custom_hook",
                    "hook": "package.module:build_roi",
                },
            ],
            "provenance": {"schema_version": "1"},
        }
    }


def _valid_extraction_set() -> dict[str, object]:
    return {
        "extraction_set": {
            "name": "modelA_extraction",
            "roi_set": "modelA_rois",
            "targets": [
                {
                    "name": "futureFeatquery",
                    "backend": "fsl_featquery",
                    "desc": "ModelAFeatquery",
                    "inputs": {
                        "feat_dir": "${ROI_FEAT_ROOT:-}/{subject_dir}/{session_dir}/func/{subject_dir}_{session_dir}_task-{task_id}_model-{model}.feat",
                    },
                },
            ],
            "provenance": {"schema_version": "1"},
        }
    }


class RoiConfigValidationTests(unittest.TestCase):
    def test_generic_roi_set_validates_without_runtime_dependencies(self) -> None:
        self.assertEqual(validate_roi_set_document(_valid_roi_set()), [])

    def test_duplicate_roi_labels_are_rejected(self) -> None:
        document = _valid_roi_set()
        rois = document["roi_set"]["rois"]  # type: ignore[index]
        rois[1]["label"] = "ManualMask"  # type: ignore[index]

        errors = validate_roi_set_document(document)

        self.assertTrue(any("duplicate ROI label" in error for error in errors))

    def test_invalid_family_backend_and_bids_label_are_rejected(self) -> None:
        document = _valid_roi_set()
        roi = document["roi_set"]["rois"][0]  # type: ignore[index]
        roi["label"] = "manual_mask"  # type: ignore[index]
        roi["family"] = "unknown_family"  # type: ignore[index]
        roi["backend"] = "unknown_backend"  # type: ignore[index]

        errors = validate_roi_set_document(document)

        self.assertTrue(any(".label must contain only letters and numbers" in error for error in errors))
        self.assertTrue(any(".family must be one of" in error for error in errors))
        self.assertTrue(any(".backend must be one of" in error for error in errors))

    def test_radius_threshold_and_min_voxel_rules_are_rejected(self) -> None:
        document = _valid_roi_set()
        roi = document["roi_set"]["rois"][3]  # type: ignore[index]
        roi["radius_mm"] = 0  # type: ignore[index]
        roi["z_threshold"] = 2.0  # type: ignore[index]
        roi["exploratory_z_threshold"] = 2.5  # type: ignore[index]
        roi["min_voxels_fail"] = 20  # type: ignore[index]
        roi["min_voxels_warn"] = 10  # type: ignore[index]

        errors = validate_roi_set_document(document)

        self.assertTrue(any("radius_mm must be greater than zero" in error for error in errors))
        self.assertTrue(any("z_threshold must be greater than or equal" in error for error in errors))
        self.assertTrue(any("min_voxels_fail must be less than or equal" in error for error in errors))

    def test_personal_absolute_paths_are_rejected(self) -> None:
        document = _valid_roi_set()
        roi = document["roi_set"]["rois"][0]  # type: ignore[index]
        personal_root = _synthetic_personal_root()
        roi["source"] = str(personal_root / "tmp" / "project" / "roi.nii.gz")  # type: ignore[index]

        errors = validate_roi_set_document(document)

        self.assertTrue(any("personal absolute path" in error for error in errors))

    def test_env_placeholder_paths_are_not_rejected_as_personal_paths(self) -> None:
        document = _valid_roi_set()
        roi_set = document["roi_set"]  # type: ignore[index]
        roi_set["outputs"] = {"root": "${ROI_DERIV_ROOT:-}"}  # type: ignore[index]
        roi_set["fixed_effects_inputs"] = {"root": "${ROI_FEAT_ROOT:-}"}  # type: ignore[index]
        roi_set["group_mask"] = {"path": "${ROI_FEAT_ROOT:-}/group/mask.nii.gz"}  # type: ignore[index]

        errors = validate_roi_set_document(document)

        self.assertFalse([error for error in errors if "personal absolute path" in error])

    def test_publication_existing_output_defaults_when_omitted(self) -> None:
        roi_document = _valid_roi_set()
        roi_document["roi_set"]["publication"] = {"enabled": True}  # type: ignore[index]
        extraction_document = _valid_extraction_set()
        extraction_document["extraction_set"]["publication"] = {"enabled": True}  # type: ignore[index]

        self.assertEqual(validate_roi_set_document(roi_document), [])
        self.assertEqual(validate_extraction_set_document(extraction_document), [])

    def test_publication_existing_output_replace_is_valid(self) -> None:
        roi_document = _valid_roi_set()
        roi_document["roi_set"]["publication"] = {"existing_output": "replace"}  # type: ignore[index]
        extraction_document = _valid_extraction_set()
        extraction_document["extraction_set"]["publication"] = {"existing_output": "replace"}  # type: ignore[index]

        self.assertEqual(validate_roi_set_document(roi_document), [])
        self.assertEqual(validate_extraction_set_document(extraction_document), [])

    def test_invalid_publication_existing_output_is_rejected(self) -> None:
        roi_document = _valid_roi_set()
        roi_document["roi_set"]["publication"] = {"existing_output": "overwrite"}  # type: ignore[index]
        extraction_document = _valid_extraction_set()
        extraction_document["extraction_set"]["publication"] = {"existing_output": "overwrite"}  # type: ignore[index]

        roi_errors = validate_roi_set_document(roi_document)
        extraction_errors = validate_extraction_set_document(extraction_document)

        self.assertIn("roi_set.publication.existing_output must be one of: fail, replace.", roi_errors)
        self.assertIn("extraction_set.publication.existing_output must be one of: fail, replace.", extraction_errors)

    def test_nonmapping_publication_is_rejected(self) -> None:
        roi_document = _valid_roi_set()
        roi_document["roi_set"]["publication"] = "replace"  # type: ignore[index]
        extraction_document = _valid_extraction_set()
        extraction_document["extraction_set"]["publication"] = "replace"  # type: ignore[index]

        roi_errors = validate_roi_set_document(roi_document)
        extraction_errors = validate_extraction_set_document(extraction_document)

        self.assertIn("roi_set.publication must contain a mapping when declared.", roi_errors)
        self.assertIn("extraction_set.publication must contain a mapping when declared.", extraction_errors)

    def test_publication_root_subpath_must_remain_beneath_named_root(self) -> None:
        roi_document = _valid_roi_set()
        roi_document["roi_set"]["publication"] = {  # type: ignore[index]
            "root": {"root_ref": "dataset_derivatives_root", "path": "../outside"}
        }
        extraction_document = _valid_extraction_set()
        extraction_document["extraction_set"]["publication"] = {  # type: ignore[index]
            "root": {"root_ref": "dataset_derivatives_root", "path": r"..\outside"}
        }

        roi_errors = validate_roi_set_document(roi_document)
        extraction_errors = validate_extraction_set_document(extraction_document)

        self.assertIn(
            "roi_set.publication.root.path must be relative and remain beneath its configured root.",
            roi_errors,
        )
        self.assertIn(
            "extraction_set.publication.root.path must be relative and remain beneath its configured root.",
            extraction_errors,
        )

    def test_valid_runtime_cleanup_values_pass_validation(self) -> None:
        roi_document = _valid_roi_set()
        roi_set = roi_document["roi_set"]  # type: ignore[index]
        roi_set["runtime"] = {"cleanup": {"after_roi_build": "roi_runtime", "after_extraction": "roi_runtime"}}  # type: ignore[index]
        extraction_document = _valid_extraction_set()
        extraction_set = extraction_document["extraction_set"]  # type: ignore[index]
        extraction_set["runtime"] = {"cleanup": {"after_extraction": "extraction_runtime"}}  # type: ignore[index]
        extraction_set["roi_mask_source"] = {"source": "roi_set_publication"}  # type: ignore[index]

        self.assertEqual(validate_roi_set_document(roi_document), [])
        self.assertEqual(validate_extraction_set_document(extraction_document), [])

    def test_runtime_existing_output_defaults_to_fail_and_accepts_replace(self) -> None:
        roi_document = _valid_roi_set()
        extraction_document = _valid_extraction_set()

        self.assertEqual(runtime_existing_output_policy(roi_document, payload_key="roi_set"), "fail")
        self.assertEqual(
            runtime_existing_output_policy(extraction_document, payload_key="extraction_set"),
            "fail",
        )

        roi_document["roi_set"]["runtime"] = {"existing_output": "replace"}  # type: ignore[index]
        extraction_document["extraction_set"]["runtime"] = {"existing_output": "replace"}  # type: ignore[index]

        self.assertEqual(validate_roi_set_document(roi_document), [])
        self.assertEqual(validate_extraction_set_document(extraction_document), [])
        self.assertEqual(runtime_existing_output_policy(roi_document, payload_key="roi_set"), "replace")
        self.assertEqual(
            runtime_existing_output_policy(extraction_document, payload_key="extraction_set"),
            "replace",
        )

    def test_invalid_runtime_existing_output_is_rejected(self) -> None:
        roi_document = _valid_roi_set()
        roi_document["roi_set"]["runtime"] = {"existing_output": "overwrite"}  # type: ignore[index]
        extraction_document = _valid_extraction_set()
        extraction_document["extraction_set"]["runtime"] = {"existing_output": "overwrite"}  # type: ignore[index]

        self.assertIn(
            "roi_set.runtime.existing_output must be one of: fail, replace.",
            validate_roi_set_document(roi_document),
        )
        self.assertIn(
            "extraction_set.runtime.existing_output must be one of: fail, replace.",
            validate_extraction_set_document(extraction_document),
        )
        with self.assertRaisesRegex(ValueError, "runtime.existing_output must be one of"):
            runtime_existing_output_policy(roi_document, payload_key="roi_set")

    def test_nonmapping_runtime_is_rejected_once(self) -> None:
        roi_document = _valid_roi_set()
        roi_document["roi_set"]["runtime"] = "replace"  # type: ignore[index]

        errors = validate_roi_set_document(roi_document)

        self.assertEqual(errors.count("roi_set.runtime must contain a mapping when declared."), 1)

    def test_invalid_runtime_cleanup_values_fail_validation(self) -> None:
        roi_document = _valid_roi_set()
        roi_set = roi_document["roi_set"]  # type: ignore[index]
        roi_set["runtime"] = {"cleanup": {"after_roi_build": "all_runtime", "after_extraction": "cache_only"}}  # type: ignore[index]
        extraction_document = _valid_extraction_set()
        extraction_set = extraction_document["extraction_set"]  # type: ignore[index]
        extraction_set["runtime"] = {"cleanup": {"after_extraction": "roi_runtime"}}  # type: ignore[index]
        extraction_set["roi_mask_source"] = {"source": "published"}  # type: ignore[index]

        roi_errors = validate_roi_set_document(roi_document)
        extraction_errors = validate_extraction_set_document(extraction_document)

        self.assertTrue(any("roi_set.runtime.cleanup.after_roi_build must be one of: cache_only, none, roi_runtime" in error for error in roi_errors))
        self.assertTrue(any("roi_set.runtime.cleanup.after_extraction must be one of: none, roi_runtime" in error for error in roi_errors))
        self.assertTrue(any("extraction_set.roi_mask_source.source must be one of: roi_set_publication, roi_set_runtime" in error for error in extraction_errors))
        self.assertTrue(
            any(
                "extraction_set.runtime.cleanup.after_extraction must be one of: extraction_runtime, none" in error
                for error in extraction_errors
            )
        )

    def test_env_placeholder_personal_runtime_paths_are_checked_against_raw_config(self) -> None:
        raw_document = _valid_roi_set()
        raw_roi_set = raw_document["roi_set"]  # type: ignore[index]
        raw_roi_set["outputs"] = {"root": "${ROI_DERIV_ROOT:-}"}  # type: ignore[index]
        raw_roi_set["fixed_effects_inputs"] = {"root": "${ROI_FEAT_ROOT:-}"}  # type: ignore[index]
        raw_roi_set["group_mask"] = {"path": "${ROI_FEAT_ROOT:-}/group/mask.nii.gz"}  # type: ignore[index]

        runtime_document = _valid_roi_set()
        runtime_roi_set = runtime_document["roi_set"]  # type: ignore[index]
        personal_root = _synthetic_personal_root()
        runtime_roi_set["outputs"] = {"root": str(personal_root / "roi-derivatives")}  # type: ignore[index]
        runtime_roi_set["fixed_effects_inputs"] = {"root": str(personal_root / "feat")}  # type: ignore[index]
        runtime_roi_set["group_mask"] = {"path": str(personal_root / "feat" / "group" / "mask.nii.gz")}  # type: ignore[index]

        errors = validate_roi_set_document(runtime_document, personal_path_document=raw_document)

        self.assertFalse([error for error in errors if "personal absolute path" in error])

    def test_extraction_set_validates_future_backend_names_without_installing_tools(self) -> None:
        document = _valid_extraction_set()

        self.assertEqual(validate_extraction_set_document(document), [])

    def test_extraction_env_placeholder_personal_runtime_paths_are_checked_against_raw_config(self) -> None:
        raw_document = _valid_extraction_set()
        raw_extraction_set = raw_document["extraction_set"]  # type: ignore[index]
        raw_extraction_set["outputs"] = {"root": "${ROI_DERIV_ROOT:-}"}  # type: ignore[index]

        runtime_document = _valid_extraction_set()
        runtime_extraction_set = runtime_document["extraction_set"]  # type: ignore[index]
        runtime_target = runtime_extraction_set["targets"][0]  # type: ignore[index]
        personal_root = _synthetic_personal_root()
        runtime_extraction_set["outputs"] = {"root": str(personal_root / "roi-derivatives")}  # type: ignore[index]
        runtime_target["inputs"] = {  # type: ignore[index]
            "feat_dir": str(personal_root / "feat" / "{subject_dir}" / "{session_dir}" / "model.feat"),
        }

        errors = validate_extraction_set_document(runtime_document, personal_path_document=raw_document)

        self.assertFalse([error for error in errors if "personal absolute path" in error])

    def test_extraction_literal_personal_absolute_path_is_rejected(self) -> None:
        document = _valid_extraction_set()
        extraction_set = document["extraction_set"]  # type: ignore[index]
        extraction_set["outputs"] = {"root": str(_synthetic_personal_root() / "roi-derivatives")}  # type: ignore[index]

        errors = validate_extraction_set_document(document)

        self.assertTrue(any("extraction_set.outputs.root contains a personal absolute path" in error for error in errors))

    def test_sidecar_provenance_validation_accepts_phase1_fields(self) -> None:
        document = {
            "roi_label": "LosoMap",
            "roi_family": "loso_group_map",
            "held_out_subject": "sub-001",
            "session": "ses-01",
            "task": "memory",
            "model": "ModelA",
            "source_contrast": "condition_a_gt_baseline",
            "full_sample_seed_coordinate": [0, -52, 26],
            "search_radius_mm": 12,
            "loso_peak_coordinate": [2, -50, 24],
            "selected_peak_z": 4.2,
            "sphere_radius_mm": 6,
            "z_threshold": 3.1,
            "exploratory_z_threshold": 2.3,
            "fallback_status": "thresholded",
            "mask_intersection_policy": "intersection",
            "voxel_count": 104,
            "warnings": ["review edge voxels"],
            "qc_flags": ["pass"],
        }

        self.assertEqual(validate_roi_sidecar_document(document), [])

    def test_sidecar_rejects_bad_coordinates_and_voxel_count(self) -> None:
        document = {
            "roi_label": "LosoMap",
            "roi_family": "loso_group_map",
            "loso_peak_coordinate": [1, 2],
            "voxel_count": -1,
        }

        errors = validate_roi_sidecar_document(document)

        self.assertTrue(any("loso_peak_coordinate must contain exactly three" in error for error in errors))
        self.assertTrue(any("voxel_count must be greater than or equal to zero" in error for error in errors))

    def test_scaffold_coordinate_sphere_roi_set_renders_valid_yaml(self) -> None:
        document = build_roi_set_document("example_rois", "coordinate_sphere")
        content = render_yaml(document)
        roi_set = document["roi_set"]

        self.assertEqual(validate_roi_set_scaffold(document), [])
        self.assertEqual(roi_set["subject"], "sub-001")
        self.assertNotIn("subjects", roi_set)
        self.assertIn("label: SeedA", content)
        self.assertIn("label: SeedB", content)
        self.assertIn("task: exampletask", content)
        self.assertIn("root_ref: artifacts_root", content)
        self.assertIn("path: roi-runtime/example_rois", content)
        self.assertIn("path: inputs/roi/example_reference.nii.gz", content)
        self.assertIn("existing_output: fail", content)
        self.assertIn('resolution: "2"', content)

    def test_single_entity_roi_scaffolds_use_exact_singular_subject_override(self) -> None:
        templates = (
            "coordinate_sphere",
            "manual_mask",
            "functional_threshold_map",
            "atlas_label",
            "data_driven_hook",
        )

        for template in templates:
            with self.subTest(template=template):
                document = build_roi_set_document(
                    f"example_{template}",
                    template,
                    overrides={"subjects": "sub-101"},
                )
                roi_set = document["roi_set"]
                self.assertEqual(roi_set["subject"], "sub-101")
                self.assertNotIn("subjects", roi_set)
                self.assertNotIn("subject_ids", roi_set)
                self.assertEqual(validate_roi_set_scaffold(document), [])

    def test_single_entity_roi_scaffolds_reject_multi_subject_overrides(self) -> None:
        templates = (
            "coordinate_sphere",
            "manual_mask",
            "functional_threshold_map",
            "atlas_label",
            "data_driven_hook",
        )
        subject_specs = ("sub-101,sub-102", "sub-101:sub-102")

        for template in templates:
            for subject_spec in subject_specs:
                with self.subTest(template=template, subject_spec=subject_spec):
                    with self.assertRaisesRegex(ValueError, "represents one configured subject"):
                        build_roi_set_document(
                            f"example_{template}",
                            template,
                            overrides={"subjects": subject_spec},
                        )

    def test_single_entity_roi_scaffold_accepts_one_value_range(self) -> None:
        document = build_roi_set_document(
            "example_rois",
            "coordinate_sphere",
            overrides={"subjects": "sub-101:sub-101"},
        )

        self.assertEqual(document["roi_set"]["subject"], "sub-101")
        self.assertNotIn("subjects", document["roi_set"])

    def test_subject_overrides_fail_closed_without_recognized_scaffold_identity(self) -> None:
        cases = (
            (
                build_roi_set_document("example_rois", "coordinate_sphere"),
                "roi_set",
                apply_roi_set_overrides,
            ),
            (
                build_extraction_set_document(
                    "example_values",
                    roi_set="example_rois",
                    template="generic_nifti",
                ),
                "extraction_set",
                apply_extraction_set_overrides,
            ),
        )

        for document, payload_key, apply_overrides in cases:
            for scaffold_identity in (None, "unclassified_future_template"):
                with self.subTest(payload_key=payload_key, scaffold_identity=scaffold_identity):
                    payload = document[payload_key]
                    if scaffold_identity is None:
                        payload["provenance"].pop("scaffold")
                    else:
                        payload["provenance"]["scaffold"] = scaffold_identity
                    with self.assertRaisesRegex(ValueError, "scaffold identity is missing or unsupported"):
                        apply_overrides(document, {"subjects": "sub-101"})

    def test_scaffold_loso_group_map_roi_set_renders_valid_yaml(self) -> None:
        document = build_roi_set_document("loso_modelA", "loso_group_map")
        content = render_yaml(document)
        roi_set = document["roi_set"]

        self.assertEqual(validate_roi_set_scaffold(document), [])
        self.assertEqual(roi_set["min_group_n"], 1)
        self.assertIn("subjects:", content)
        self.assertIn("held_out_subjects:", content)
        self.assertNotIn("${ROI_FEAT_ROOT:-}", content)
        self.assertNotIn("${ROI_DERIV_ROOT:-}", content)
        self.assertEqual(
            roi_set["outputs"],
            {
                "root_ref": "dataset_derivatives_root",
                "path": ".research-platform/roi-loso-flame1-runtime/loso_modelA",
            },
        )
        self.assertEqual(roi_set["runtime"]["existing_output"], "fail")
        self.assertEqual(
            roi_set["runtime"]["cleanup"],
            {"after_roi_build": "roi_runtime", "after_extraction": "none"},
        )
        self.assertIn("fixed_effects_inputs:\n    root_ref: project_root", content)
        self.assertIn("inputs/roi/fixed-effects/", content)
        self.assertIn("cope{cope_number}.gfeat/mask.nii.gz", content)
        self.assertIn("label: SeedA", content)
        self.assertIn("label: SeedB", content)

    def test_scaffold_loso_group_map_generic_path_profile_preserves_default_yaml(self) -> None:
        default_content = render_yaml(build_roi_set_document("loso_modelA", "loso_group_map"))
        generic_content = render_yaml(build_roi_set_document("loso_modelA", "loso_group_map", path_profile="generic"))

        self.assertEqual(generic_content, default_content)

    def test_scaffold_loso_group_map_research_platform_fsl_ffx_path_profile_validates(self) -> None:
        document = build_roi_set_document("loso_modelA", "loso_group_map", path_profile="research_platform_fsl_ffx")
        content = render_yaml(document)
        roi_set = document["roi_set"]

        self.assertEqual(validate_roi_set_scaffold(document), [])
        self.assertEqual(roi_set["missing_input_policy"], "warn")
        self.assertEqual(
            roi_set["runtime"]["cleanup"],
            {"after_roi_build": "roi_runtime", "after_extraction": "none"},
        )
        self.assertEqual(roi_set["runtime"]["existing_output"], "fail")
        self.assertEqual(
            roi_set["outputs"]["path"],
            ".research-platform/roi-loso-flame1-runtime/loso_modelA",
        )
        self.assertIn("FFX.gfeat/cope{cope_number}.feat", content)
        self.assertIn("cope_image: stats/cope1.nii.gz", content)
        self.assertIn("varcope_image: stats/varcope1.nii.gz", content)
        self.assertIn("cope{cope_number}.gfeat/mask.nii.gz", content)
        self.assertIn("higher_level/group_{session_dir}_task-{task_id}_dir-{direction}_desc-{model}FFX_FLAME1", content)

    def test_scaffold_loso_group_map_path_profile_rejects_unknown_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "generic, research_platform_fsl_ffx"):
            build_roi_set_document("loso_modelA", "loso_group_map", path_profile="site_local")

    def test_scaffold_loso_group_map_subject_range_override(self) -> None:
        document = build_roi_set_document("loso_modelA", "loso_group_map", overrides={"subjects": "sub-002:sub-004"})

        roi_set = document["roi_set"]
        self.assertEqual(roi_set["subjects"], ["sub-002", "sub-003", "sub-004"])
        self.assertEqual(roi_set["held_out_subjects"], ["sub-002", "sub-003", "sub-004"])
        self.assertEqual(roi_set["min_group_n"], 2)
        self.assertIn("min_group_n: 2", render_yaml(document))
        self.assertEqual(validate_roi_set_scaffold(document), [])

    def test_scaffold_loso_group_map_subject_range_infers_min_group_n(self) -> None:
        document = build_roi_set_document(
            "loso_modelA",
            "loso_group_map",
            overrides={"subjects": "sub-101:sub-103", "held_out_subjects": "same-as-subjects"},
        )

        roi_set = document["roi_set"]
        self.assertEqual(len(roi_set["subjects"]), 3)
        self.assertEqual(roi_set["held_out_subjects"], roi_set["subjects"])
        self.assertEqual(roi_set["min_group_n"], 2)
        self.assertIn("min_group_n: 2", render_yaml(document))
        self.assertEqual(validate_roi_set_scaffold(document), [])

    def test_scaffold_loso_group_map_subject_comma_and_mixed_overrides(self) -> None:
        document = build_roi_set_document(
            "loso_modelA",
            "loso_group_map",
            overrides={"subjects": "sub-002,sub-004:sub-006", "held_out_subjects": "sub-004,sub-006"},
        )

        roi_set = document["roi_set"]
        self.assertEqual(roi_set["subjects"], ["sub-002", "sub-004", "sub-005", "sub-006"])
        self.assertEqual(roi_set["held_out_subjects"], ["sub-004", "sub-006"])
        self.assertEqual(roi_set["min_group_n"], 3)

    def test_scaffold_loso_group_map_invalid_subject_ranges_fail(self) -> None:
        bad_specs = ("sub-004:sub-002", "sub-02:sub-004", "sub-002:ses-004", "sub-002,,sub-004")
        for spec in bad_specs:
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, "--subjects must be"):
                    build_roi_set_document("loso_modelA", "loso_group_map", overrides={"subjects": spec})

    def test_scaffold_loso_group_map_held_out_same_as_subjects_override(self) -> None:
        document = build_roi_set_document(
            "loso_modelA",
            "loso_group_map",
            overrides={"subjects": "sub-002:sub-004", "held_out_subjects": "same-as-subjects"},
        )

        roi_set = document["roi_set"]
        self.assertEqual(roi_set["subjects"], ["sub-002", "sub-003", "sub-004"])
        self.assertEqual(roi_set["held_out_subjects"], ["sub-002", "sub-003", "sub-004"])
        self.assertEqual(roi_set["min_group_n"], 2)
        self.assertEqual(validate_roi_set_scaffold(document), [])

    def test_scaffold_loso_group_map_without_subject_override_preserves_template_min_group_n(self) -> None:
        document = build_roi_set_document("loso_modelA", "loso_group_map")

        roi_set = document["roi_set"]
        self.assertEqual(roi_set["subjects"], ["sub-001", "sub-002"])
        self.assertEqual(roi_set["held_out_subjects"], ["sub-001", "sub-002"])
        self.assertEqual(roi_set["min_group_n"], 1)
        self.assertEqual(validate_roi_set_scaffold(document), [])

    def test_scaffold_loso_group_map_contrast_overrides_replace_defaults(self) -> None:
        document = build_roi_set_document(
            "loso_modelA",
            "loso_group_map",
            overrides={
                "contrasts": [
                    "pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit",
                    "item_rec_hit_gt_miss:2:ItemRecHitGtMiss",
                ]
            },
        )
        content = render_yaml(document)

        self.assertEqual(
            document["roi_set"]["contrasts"],
            [
                {"id": "pair_enc_hit_gt_item_enc_hit", "cope_number": 1, "desc": "PairEncHitGtItemEncHit"},
                {"id": "item_rec_hit_gt_miss", "cope_number": 2, "desc": "ItemRecHitGtMiss"},
            ],
        )
        self.assertIn("cope_number: 1", content)
        self.assertNotIn("id: ContrastA", content)
        self.assertEqual(document["roi_set"]["rois"][0]["contrast"], "pair_enc_hit_gt_item_enc_hit")
        self.assertEqual(document["roi_set"]["rois"][1]["contrast"], "item_rec_hit_gt_miss")

    def test_scaffold_loso_group_map_malformed_contrast_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "--contrast must use"):
            build_roi_set_document("loso_modelA", "loso_group_map", overrides={"contrasts": ["bad:cope"]})

    def test_scaffold_loso_group_map_roi_overrides_replace_defaults(self) -> None:
        document = build_roi_set_document(
            "loso_modelA",
            "loso_group_map",
            overrides={
                "contrasts": ["pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit"],
                "rois": ["EncodingPrecuneus:pair_enc_hit_gt_item_enc_hit:-2,-58,64:PairEncHitGtItemEncHit"],
            },
        )

        rois = document["roi_set"]["rois"]
        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0]["label"], "EncodingPrecuneus")
        self.assertEqual(rois[0]["contrast"], "pair_enc_hit_gt_item_enc_hit")
        self.assertEqual(rois[0]["seed_coordinate"], [-2, -58, 64])
        self.assertEqual(validate_roi_set_scaffold(document), [])

    def test_scaffold_loso_group_map_malformed_roi_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "--roi must use"):
            build_roi_set_document(
                "loso_modelA",
                "loso_group_map",
                overrides={
                    "contrasts": ["pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit"],
                    "rois": ["EncodingPrecuneus:pair_enc_hit_gt_item_enc_hit:-2,-58"],
                },
            )

    def test_scaffold_loso_group_map_duplicate_roi_labels_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "--roi values must use unique labels"):
            build_roi_set_document(
                "loso_modelA",
                "loso_group_map",
                overrides={
                    "contrasts": ["pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit"],
                    "rois": [
                        "EncodingPrecuneus:pair_enc_hit_gt_item_enc_hit:-2,-58,64",
                        "EncodingPrecuneus:pair_enc_hit_gt_item_enc_hit:0,-52,26",
                    ],
                },
            )

    def test_scaffold_loso_group_map_unknown_roi_contrast_reference_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown contrast_id"):
            build_roi_set_document(
                "loso_modelA",
                "loso_group_map",
                overrides={
                    "contrasts": ["pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit"],
                    "rois": ["EncodingPrecuneus:other_contrast:-2,-58,64"],
                },
            )

    def test_scaffold_loso_group_map_radius_threshold_and_min_voxels_render(self) -> None:
        document = build_roi_set_document(
            "loso_modelA",
            "loso_group_map",
            overrides={
                "search_radius_mm": "20",
                "sphere_radius_mm": "6",
                "z_threshold": "3.1",
                "min_voxels_warn": "10",
                "min_voxels_fail": "5",
            },
        )

        for roi in document["roi_set"]["rois"]:
            self.assertEqual(roi["search_radius_mm"], 20)
            self.assertEqual(roi["sphere_radius_mm"], 6)
            self.assertEqual(roi["z_threshold"], 3.1)
            self.assertEqual(roi["min_voxels_warn"], 10)
            self.assertEqual(roi["min_voxels_fail"], 5)
        self.assertIn("search_radius_mm: 20", render_yaml(document))

    def test_scaffold_manual_mask_roi_set_uses_project_relative_placeholder(self) -> None:
        document = build_roi_set_document("manual_rois", "manual_mask")
        content = render_yaml(document)

        self.assertEqual(validate_roi_set_scaffold(document), [])
        self.assertIn("root_ref: project_root", content)
        self.assertIn("path: inputs/roi/example_mask.nii.gz", content)

    def test_scaffold_generic_nifti_extraction_set_renders_valid_yaml(self) -> None:
        document = build_extraction_set_document("modelA_values", roi_set="loso_modelA", template="generic_nifti")
        content = render_yaml(document)
        extraction_set = document["extraction_set"]

        self.assertEqual(validate_extraction_set_scaffold(document), [])
        self.assertEqual(extraction_set["subject"], "sub-001")
        self.assertNotIn("subjects", extraction_set)
        self.assertIn("backend: generic_nifti", content)
        self.assertIn("metrics:", content)
        self.assertIn("- mean", content)
        self.assertIn("- median", content)
        self.assertIn("- voxel_count", content)

    def test_generic_nifti_scaffold_uses_exact_singular_subject_override(self) -> None:
        document = build_extraction_set_document(
            "modelA_values",
            roi_set="modelA_rois",
            template="generic_nifti",
            overrides={"subjects": "sub-101"},
        )

        extraction_set = document["extraction_set"]
        self.assertEqual(extraction_set["subject"], "sub-101")
        self.assertNotIn("subjects", extraction_set)
        self.assertNotIn("subject_ids", extraction_set)
        self.assertEqual(validate_extraction_set_scaffold(document), [])

    def test_generic_nifti_scaffold_rejects_multi_subject_overrides(self) -> None:
        for subject_spec in ("sub-101,sub-102", "sub-101:sub-102"):
            with self.subTest(subject_spec=subject_spec):
                with self.assertRaisesRegex(ValueError, "represents one configured subject"):
                    build_extraction_set_document(
                        "modelA_values",
                        roi_set="modelA_rois",
                        template="generic_nifti",
                        overrides={"subjects": subject_spec},
                    )

    def test_generic_nifti_scaffold_accepts_one_value_range(self) -> None:
        document = build_extraction_set_document(
            "modelA_values",
            roi_set="modelA_rois",
            template="generic_nifti",
            overrides={"subjects": "sub-101:sub-101"},
        )

        self.assertEqual(document["extraction_set"]["subject"], "sub-101")
        self.assertNotIn("subjects", document["extraction_set"])

    def test_scaffold_fsl_featquery_extraction_set_renders_valid_yaml(self) -> None:
        document = build_extraction_set_document("modelA_featquery", roi_set="loso_modelA", template="fsl_featquery")
        content = render_yaml(document)
        extraction_set = document["extraction_set"]

        self.assertEqual(validate_extraction_set_scaffold(document), [])
        self.assertEqual(extraction_set["roi_mask_source"], {"source": "roi_set_publication"})
        self.assertEqual(extraction_set["runtime"]["existing_output"], "fail")
        self.assertEqual(extraction_set["runtime"]["cleanup"], {"after_extraction": "extraction_runtime"})
        self.assertIn("roi_mask_source:", content)
        self.assertIn("source: roi_set_publication", content)
        self.assertIn("backend: fsl_featquery", content)
        self.assertIn("- mean_cope", content)
        self.assertIn("- roi_voxel_count", content)
        self.assertNotIn("percent_signal_change", content)
        self.assertNotIn("${ROI_FEAT_ROOT:-}", content)
        self.assertIn("root_ref: project_root", content)
        self.assertIn("inputs/roi/feat/", content)

    def test_scaffold_fsl_featquery_generic_path_profile_preserves_default_yaml(self) -> None:
        default_content = render_yaml(build_extraction_set_document("modelA_featquery", roi_set="loso_modelA", template="fsl_featquery"))
        generic_content = render_yaml(
            build_extraction_set_document(
                "modelA_featquery",
                roi_set="loso_modelA",
                template="fsl_featquery",
                path_profile="generic",
            )
        )

        self.assertEqual(generic_content, default_content)

    def test_scaffold_fsl_featquery_research_platform_fsl_ffx_path_profile_validates(self) -> None:
        document = build_extraction_set_document(
            "modelA_featquery",
            roi_set="loso_modelA",
            template="fsl_featquery",
            path_profile="research_platform_fsl_ffx",
        )
        content = render_yaml(document)
        target = document["extraction_set"]["targets"][0]

        self.assertEqual(validate_extraction_set_scaffold(document), [])
        self.assertEqual(document["extraction_set"]["roi_mask_source"], {"source": "roi_set_publication"})
        self.assertEqual(document["extraction_set"]["runtime"]["cleanup"], {"after_extraction": "extraction_runtime"})
        self.assertEqual(document["extraction_set"]["runtime"]["existing_output"], "fail")
        self.assertEqual(
            document["extraction_set"]["outputs"]["path"],
            ".research-platform/roi-loso-flame1-runtime/modelA_featquery",
        )
        self.assertEqual(target["metrics"], ["mean_cope", "roi_voxel_count"])
        self.assertNotIn("percent_signal_change", content)
        self.assertEqual(target["inputs"]["value_image"], "stats/cope1.nii.gz")
        self.assertIn("FFX.gfeat/cope{cope}.feat", content)
        self.assertIn("featquery_output_name: fq_loso_{roi_label}_{source_contrast}_cope{cope}", content)

    def test_scaffold_research_platform_fsl_ffx_keeps_explicit_overrides(self) -> None:
        roi_document = build_roi_set_document(
            "loso_modelA",
            "loso_group_map",
            path_profile="research_platform_fsl_ffx",
            overrides={
                "subjects": "sub-002:sub-004",
                "session": "ses-02",
                "task": "workingmemory",
                "direction": "PA",
                "model": "ModelB",
                "contrasts": ["cond_a_gt_b:3:CondAGtB"],
                "rois": ["DlpfcSeed:cond_a_gt_b:12,34,56:CondAGtB"],
            },
        )
        extraction_document = build_extraction_set_document(
            "modelB_featquery",
            roi_set="loso_modelA",
            template="fsl_featquery",
            path_profile="research_platform_fsl_ffx",
            overrides={
                "subjects": "sub-002:sub-004",
                "session": "ses-02",
                "task": "workingmemory",
                "direction": "PA",
                "model": "ModelB",
                "metrics": ["mean_cope"],
                "contrasts": ["cond_a_gt_b:3:CondAGtB"],
                "roi_labels": ["DlpfcSeed"],
            },
        )

        roi_set = roi_document["roi_set"]
        roi = roi_set["rois"][0]
        extraction_set = extraction_document["extraction_set"]
        target = extraction_set["targets"][0]
        self.assertEqual(roi_set["subjects"], ["sub-002", "sub-003", "sub-004"])
        self.assertNotIn("subject", roi_set)
        self.assertEqual(roi_set["session"], "ses-02")
        self.assertEqual(roi_set["task"], "workingmemory")
        self.assertEqual(roi_set["direction"], "PA")
        self.assertEqual(roi_set["model"], "ModelB")
        self.assertEqual(roi_set["contrasts"], [{"id": "cond_a_gt_b", "cope_number": 3, "desc": "CondAGtB"}])
        self.assertEqual(roi["label"], "DlpfcSeed")
        self.assertEqual(extraction_set["subjects"], ["sub-002", "sub-003", "sub-004"])
        self.assertNotIn("subject", extraction_set)
        self.assertEqual(target["metrics"], ["mean_cope"])
        self.assertEqual(target["contrasts"], [{"id": "cond_a_gt_b", "cope": 3, "desc": "CondAGtB"}])
        self.assertEqual(target["roi_labels"], ["DlpfcSeed"])
        self.assertEqual(validate_roi_set_scaffold(roi_document), [])
        self.assertEqual(validate_extraction_set_scaffold(extraction_document), [])

    def test_scaffold_extraction_metric_overrides_replace_defaults(self) -> None:
        document = build_extraction_set_document(
            "modelA_values",
            roi_set="loso_modelA",
            template="generic_nifti",
            overrides={"metrics": ["mean"]},
        )

        self.assertEqual(document["extraction_set"]["targets"][0]["metrics"], ["mean"])
        self.assertNotIn("- median", render_yaml(document))

    def test_scaffold_extraction_percent_signal_change_metric_preserves_psc_intent(self) -> None:
        document = build_extraction_set_document(
            "modelA_featquery",
            roi_set="loso_modelA",
            template="fsl_featquery",
            overrides={"metrics": ["percent_signal_change"]},
        )
        content = render_yaml(document)
        target = document["extraction_set"]["targets"][0]

        self.assertEqual(validate_extraction_set_scaffold(document), [])
        self.assertEqual(target["metrics"], ["percent_signal_change"])
        self.assertEqual(target["featquery"], {"include_percent_signal_change": True})
        self.assertIn("- percent_signal_change", content)
        self.assertIn("include_percent_signal_change: true", content)

    def test_scaffold_extraction_malformed_metric_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "--metric must be"):
            build_extraction_set_document(
                "modelA_featquery",
                roi_set="loso_modelA",
                template="fsl_featquery",
                overrides={"metrics": [""]},
            )

    def test_scaffold_extraction_contrast_overrides_use_active_cope_key(self) -> None:
        document = build_extraction_set_document(
            "modelA_featquery",
            roi_set="loso_modelA",
            template="fsl_featquery",
            overrides={"contrasts": ["pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit"]},
        )

        self.assertEqual(
            document["extraction_set"]["targets"][0]["contrasts"],
            [{"id": "pair_enc_hit_gt_item_enc_hit", "cope": 1, "desc": "PairEncHitGtItemEncHit"}],
        )
        self.assertNotIn("cope_number", render_yaml(document))

    def test_scaffold_extraction_roi_label_overrides_preserve_order(self) -> None:
        document = build_extraction_set_document(
            "modelA_featquery",
            roi_set="loso_modelA",
            template="fsl_featquery",
            overrides={"roi_labels": ["EncodingPrecuneus", "EncodingAngular"]},
        )

        self.assertEqual(document["extraction_set"]["targets"][0]["roi_labels"], ["EncodingPrecuneus", "EncodingAngular"])

    def test_scaffold_extraction_duplicate_roi_label_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "--roi-label values must be unique"):
            build_extraction_set_document(
                "modelA_featquery",
                roi_set="loso_modelA",
                template="fsl_featquery",
                overrides={"roi_labels": ["EncodingPrecuneus", "EncodingPrecuneus"]},
            )

    def test_scaffold_extraction_invalid_roi_label_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "--roi-label must be"):
            build_extraction_set_document(
                "modelA_featquery",
                roi_set="loso_modelA",
                template="fsl_featquery",
                overrides={"roi_labels": ["encoding_precuneus"]},
            )

    def test_scaffold_override_yaml_validates_and_avoids_forbidden_personal_strings(self) -> None:
        document = build_roi_set_document(
            "loso_modelA",
            "loso_group_map",
            overrides={
                "subjects": "sub-002:sub-003",
                "contrasts": ["pair_enc_hit_gt_item_enc_hit:1:PairEncHitGtItemEncHit"],
                "rois": ["EncodingPrecuneus:pair_enc_hit_gt_item_enc_hit:-2,-58,64:PairEncHitGtItemEncHit"],
            },
        )
        content = render_yaml(document)

        self.assertEqual(validate_roi_set_scaffold(document), [])
        for forbidden in FORBIDDEN_SCAFFOLD_STRINGS:
            self.assertNotIn(forbidden, content)

    def test_all_scaffold_yaml_avoids_forbidden_personal_strings(self) -> None:
        self.assertEqual(supported_path_profiles(), ("generic", "research_platform_fsl_ffx"))
        self.assertEqual(
            supported_roi_set_templates(),
            (
                "atlas_label",
                "coordinate_sphere",
                "data_driven_hook",
                "functional_threshold_map",
                "loso_group_map",
                "manual_mask",
            ),
        )
        self.assertEqual(supported_extraction_set_templates(), ("fsl_featquery", "generic_nifti"))
        self.assertIn("coordinate_sphere: Local NIfTI", roi_set_template_help())
        self.assertIn("loso_group_map: LOSO", roi_set_template_help())
        self.assertIn("atlas_label: Configuration scaffold only", roi_set_template_help())
        self.assertIn("generic_nifti: Local NIfTI", extraction_set_template_help())
        self.assertIn("fsl_featquery: FSL featquery", extraction_set_template_help())
        contents = [
            render_yaml(build_roi_set_document(f"roi_{template}", template))
            for template in (
                "coordinate_sphere",
                "manual_mask",
                "atlas_label",
                "functional_threshold_map",
                "loso_group_map",
                "data_driven_hook",
            )
        ]
        contents.extend(
            [
                render_yaml(build_extraction_set_document("generic_values", roi_set="loso_modelA", template="generic_nifti")),
                render_yaml(build_extraction_set_document("featquery_values", roi_set="loso_modelA", template="fsl_featquery")),
                render_yaml(
                    build_roi_set_document(
                        "loso_modelA",
                        "loso_group_map",
                        path_profile="research_platform_fsl_ffx",
                    )
                ),
                render_yaml(
                    build_extraction_set_document(
                        "modelA_featquery",
                        roi_set="loso_modelA",
                        template="fsl_featquery",
                        path_profile="research_platform_fsl_ffx",
                    )
                ),
            ]
        )
        combined = "\n".join(contents)

        for forbidden in FORBIDDEN_SCAFFOLD_STRINGS:
            self.assertNotIn(forbidden, combined)

        default_contents = [
            render_yaml(build_roi_set_document(f"neutral_{template}", template))
            for template in supported_roi_set_templates()
        ]
        default_contents.extend(
            render_yaml(build_extraction_set_document(f"neutral_{template}", roi_set="neutral_rois", template=template))
            for template in supported_extraction_set_templates()
        )
        default_combined = "\n".join(default_contents).lower()
        self.assertNotIn("memory", default_combined)
        for document in (
            *(build_roi_set_document(f"neutral_{template}", template) for template in supported_roi_set_templates()),
            *(
                build_extraction_set_document(f"neutral_{template}", roi_set="neutral_rois", template=template)
                for template in supported_extraction_set_templates()
            ),
        ):
            payload = document.get("roi_set", document.get("extraction_set"))
            self.assertEqual(payload["runtime"]["existing_output"], "fail")
            self.assertTrue(payload["outputs"]["path"].endswith(f"/{payload['name']}"))


if __name__ == "__main__":
    unittest.main()
