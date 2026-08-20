from __future__ import annotations

from pathlib import Path
import copy
import subprocess
import sys
import textwrap
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.mvpa import (
    MvpaSetConfig,
    parse_mvpa_set_document,
    validate_mvpa_set_document,
)


def _valid_config() -> dict[str, object]:
    return {
        "mvpa_set": {
            "name": "memory_mvpa",
            "subjects": ["sub-001", "sub-002"],
            "sessions": ["ses-01"],
            "runs": ["01", "02"],
            "entities": {
                "task": "memory",
                "direction": "AP",
                "model": "ModelA",
                "space": "MNI152NLin6Asym",
                "resolution": "2",
            },
            "conditions": [
                {"id": "encode_faces", "aliases": ["faces", "face_trials"], "selector": {"trial_type": "face"}},
                {"id": "encode_places", "aliases": ["places"], "selector": {"trial_type": "place"}},
            ],
            "pattern_sources": [
                {
                    "name": "first_level_pe",
                    "backend": "fsl_feat_pe",
                    "root_ref": "dataset_derivatives_root",
                    "pattern": "sub-{subject_id}/{session_id}/func/{run_id}/model.feat/stats/pe*.nii.gz",
                }
            ],
            "roi_sources": [
                {
                    "name": "memory_rois",
                    "source": "roi_set",
                    "roi_set_ref": "memory_roi_set",
                }
            ],
            "event_thresholds": {
                "min_events_per_condition_per_run": 1,
                "min_runs_per_condition": 2,
            },
            "exclusions": {
                "rules": [{"id": "motion_outliers", "reason": "Configured QC exclusion"}],
            },
            "distance": {
                "metrics": ["crossnobis"],
                "engine": {"preferred": "native_reference", "fallback": "rsatoolbox"},
                "cross_validation": {
                    "unit": "run",
                    "grouping_columns": ["subject_id", "session_id", "run_id"],
                },
                "noise_normalization": {"method": "diagonal", "variance_source": "sigmasquareds"},
            },
            "outputs": {
                "runtime_root": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/{mvpa_set}",
                },
                "published_root": {
                    "root_ref": "dataset_derivatives_root",
                    "path": "mvpa-crossnobis/{mvpa_set}",
                },
            },
            "publication": {
                "enabled": True,
                "derivative_name": "mvpa-crossnobis",
                "write_json_sidecars": True,
                "write_provenance": True,
            },
            "missing_input_policy": "warn",
            "provenance": {"schema_version": "2b"},
        }
    }


def _errors_for(mutator: object) -> list[str]:
    document = copy.deepcopy(_valid_config())
    mutator(document["mvpa_set"])  # type: ignore[index,operator]
    return validate_mvpa_set_document(document)


class MvpaConfigValidationTests(unittest.TestCase):
    def test_valid_minimal_config_validates_without_errors(self) -> None:
        self.assertEqual(validate_mvpa_set_document(_valid_config()), [])

    def test_entities_parse_into_dataclasses(self) -> None:
        config = parse_mvpa_set_document(_valid_config())

        self.assertIsInstance(config, MvpaSetConfig)
        self.assertEqual(config.selector.subjects, ("sub-001", "sub-002"))
        self.assertEqual(config.selector.sessions, ("ses-01",))
        self.assertEqual(config.selector.runs, ("01", "02"))
        self.assertEqual(config.entities.task, "memory")
        self.assertEqual(config.entities.direction, "AP")
        self.assertEqual(config.entities.model, "ModelA")
        self.assertEqual(config.entities.space, "MNI152NLin6Asym")
        self.assertEqual(config.entities.resolution, "2")

    def test_condition_aliases_parse_correctly(self) -> None:
        config = parse_mvpa_set_document(_valid_config())

        self.assertEqual(config.conditions[0].aliases, ("faces", "face_trials"))

    def test_duplicate_condition_ids_are_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["conditions"][1]["id"] = "encode_faces"  # type: ignore[index]

        self.assertTrue(any("duplicate condition id" in error for error in _errors_for(mutate)))

    def test_duplicate_condition_aliases_are_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["conditions"][1]["aliases"] = ["faces"]  # type: ignore[index]

        self.assertTrue(any("duplicate condition alias" in error for error in _errors_for(mutate)))

    def test_hard_coded_pe_cope_condition_fields_are_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["conditions"][0]["pe_number"] = 3  # type: ignore[index]
            mvpa_set["conditions"][1]["selector"] = {"cope": 2}  # type: ignore[index]

        errors = _errors_for(mutate)

        self.assertTrue(any("pe_number" in error and "PE/COPE" in error for error in errors))
        self.assertTrue(any(".cope" in error and "PE/COPE" in error for error in errors))

    def test_unsupported_backend_is_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["pattern_sources"][0]["backend"] = "unknown_backend"  # type: ignore[index]

        self.assertTrue(any(".backend must be one of" in error for error in _errors_for(mutate)))

    def test_unsupported_metric_is_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["distance"]["metrics"] = ["mahalanobis"]  # type: ignore[index]

        self.assertTrue(any("distance.metrics" in error for error in _errors_for(mutate)))

    def test_unsupported_distance_engine_is_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["distance"]["engine"] = "custom_engine"  # type: ignore[index]

        self.assertTrue(any("distance.engine" in error for error in _errors_for(mutate)))

    def test_unsupported_noise_normalization_method_is_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["distance"]["noise_normalization"]["method"] = "full_covariance"  # type: ignore[index]

        self.assertTrue(any("noise_normalization.method" in error for error in _errors_for(mutate)))

    def test_unsupported_cv_unit_is_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["distance"]["cross_validation"]["unit"] = "fold"  # type: ignore[index]

        self.assertTrue(any("cross_validation.unit" in error for error in _errors_for(mutate)))

    def test_unsupported_missing_input_policy_is_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["missing_input_policy"] = "ignore"  # type: ignore[index]

        self.assertTrue(any("missing_input_policy" in error for error in _errors_for(mutate)))

    def test_explicit_roi_mask_source_validates(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["roi_sources"] = [  # type: ignore[index]
            {
                "name": "explicit_rois",
                "source": "explicit_masks",
                "root_ref": "dataset_derivatives_root",
                "masks": [
                    {
                        "label": "V1",
                        "path": "roi-loso-flame1/masks/sub-{subject_id}/ses-01/func/{mask_name}.nii.gz",
                    }
                ],
            }
        ]

        self.assertEqual(validate_mvpa_set_document(document), [])

    def test_explicit_roi_mask_pattern_source_validates(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["roi_sources"] = [  # type: ignore[index]
            {
                "name": "explicit_roi_pattern",
                "source": "explicit_masks",
                "root_ref": "dataset_derivatives_root",
                "mask_pattern": "roi-loso-flame1/masks/sub-{subject_id}/ses-01/func/*_mask.nii.gz",
            }
        ]

        self.assertEqual(validate_mvpa_set_document(document), [])

    def test_roi_set_source_validates(self) -> None:
        self.assertEqual(validate_mvpa_set_document(_valid_config()), [])

    def test_roi_set_publication_and_runtime_sources_validate(self) -> None:
        for source_value in ("roi_set_publication", "roi_set_runtime"):
            document = copy.deepcopy(_valid_config())
            mvpa_set = document["mvpa_set"]  # type: ignore[index]
            mvpa_set["roi_sources"][0]["source"] = source_value  # type: ignore[index]

            self.assertEqual(validate_mvpa_set_document(document), [])

    def test_roi_set_source_allows_mask_template_override(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["roi_sources"][0].update(  # type: ignore[index]
            {
                "source": "roi_set_runtime",
                "root_ref": "dataset_derivatives_root",
                "roi_labels": ["SeedA"],
                "mask_template": "rois/{roi_set_ref}/{subject_dir}/{session_dir}/func/label-{roi_label}_mask.nii.gz",
            }
        )

        self.assertEqual(validate_mvpa_set_document(document), [])

    def test_invalid_mixed_roi_source_shape_is_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["roi_sources"][0]["masks"] = [{"label": "V1", "path": "masks/V1.nii.gz"}]  # type: ignore[index]

        self.assertTrue(any("must not mix" in error for error in _errors_for(mutate)))

    def test_personal_absolute_path_literals_are_rejected(self) -> None:
        personal_path = "/home/alice/private-example/model.feat"

        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["pattern_sources"][0]["pattern"] = personal_path  # type: ignore[index]

        self.assertTrue(any("personal absolute path" in error for error in _errors_for(mutate)))

    def test_parent_directory_traversal_is_rejected(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["outputs"]["runtime_root"]["path"] = "../mvpa-runtime"  # type: ignore[index]

        self.assertTrue(any("parent-directory traversal" in error for error in _errors_for(mutate)))

    def test_nonexistent_roots_and_masks_do_not_cause_filesystem_errors(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["roi_sources"] = [  # type: ignore[index]
            {
                "name": "explicit_rois",
                "source": "explicit_masks",
                "root_ref": "definitely_not_a_real_root",
                "masks": [{"label": "V1", "path": "missing/mask.nii.gz"}],
            }
        ]

        self.assertEqual(validate_mvpa_set_document(document), [])

    def test_materialized_feature_roi_identity_needs_no_mask_path(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["roi_sources"] = [  # type: ignore[index]
            {
                "name": "prepared-rois",
                "source": "materialized_features",
                "roi_labels": ["SeedA", "SeedB"],
                "feature_space_id": "example-feature-space",
                "roi_definition_id": "example-roi-definition",
            }
        ]

        self.assertEqual(validate_mvpa_set_document(document), [])
        config = parse_mvpa_set_document(document)
        self.assertEqual(config.roi_sources[0].source, "materialized_features")
        self.assertEqual(config.roi_sources[0].masks, ())

    def test_materialized_feature_roi_identity_rejects_mask_mixing(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["roi_sources"] = [  # type: ignore[index]
            {
                "name": "prepared-rois",
                "source": "materialized_features",
                "roi_labels": ["SeedA"],
                "feature_space_id": "example-feature-space",
                "roi_definition_id": "example-roi-definition",
                "path": "SeedA_mask.nii.gz",
            }
        ]

        errors = validate_mvpa_set_document(document)
        self.assertTrue(any("must not declare ROI-set references or mask paths" in error for error in errors))

    def test_runtime_existing_output_defaults_to_fail(self) -> None:
        config = parse_mvpa_set_document(_valid_config())

        self.assertEqual(config.runtime.existing_output, "fail")
        self.assertEqual(config.runtime.fields, {})

    def test_runtime_existing_output_accepts_only_fail_in_v1(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["runtime"] = {"existing_output": "replace"}  # type: ignore[index]

        errors = validate_mvpa_set_document(document)
        self.assertIn("mvpa_set.runtime.existing_output must be 'fail' in v1.", errors)

        mvpa_set["runtime"] = {"existing_output": "fail"}  # type: ignore[index]
        self.assertEqual(validate_mvpa_set_document(document), [])
        self.assertEqual(parse_mvpa_set_document(document).runtime.existing_output, "fail")

    def test_direct_payload_shape_is_accepted(self) -> None:
        payload = _valid_config()["mvpa_set"]

        self.assertEqual(validate_mvpa_set_document(payload), [])

    def test_step7_runtime_refinements_parse_from_neutral_config(self) -> None:
        document = {
            "mvpa_set": {
                "name": "runtime_refinements",
                "subjects": ["participant-a", "participant-b"],
                "sessions": ["session-a"],
                "runs": ["run-a", "run-b"],
                "conditions": [
                    {"id": "condition-a", "selector": {"trial_type": "alpha"}},
                    {"id": "condition-b", "selector": {"trial_type": "beta"}},
                ],
                "condition_pairs": [
                    {"id": "pair-alpha", "left": "condition-a", "right": "condition-b"}
                ],
                "threshold_sweeps": [
                    {"id": "threshold-alpha", "min_events": 2, "min_observations": 3}
                ],
                "within_roi_mean_centering": True,
                "grouping_columns": ["participant_group", "task_id"],
                "run_exclusions": [
                    {
                        "id": "exclude-run-a",
                        "subject_id": "participant-a",
                        "session_id": "session-a",
                        "run_id": "run-a",
                        "reason": "Synthetic configured exclusion",
                    }
                ],
                "pattern_sources": [
                    {"name": "patterns", "backend": "bids_derivative_pattern_table"}
                ],
                "roi_sources": [{"name": "rois", "source": "roi_set", "roi_set_ref": "roi_set_alpha"}],
                "distance": {
                    "metrics": ["crossnobis"],
                    "engine": {"preferred": "native_reference", "fallback": "rsatoolbox"},
                    "cross_validation": {"unit": "run"},
                },
                "outputs": {
                    "runtime_root": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/{mvpa_set}"}
                },
            }
        }

        self.assertEqual(validate_mvpa_set_document(document), [])

        config = parse_mvpa_set_document(document)

        self.assertEqual(config.condition_pairs[0].id, "pair-alpha")
        self.assertEqual(config.condition_pairs[0].condition_id_a, "condition-a")
        self.assertEqual(config.condition_pairs[0].condition_id_b, "condition-b")
        self.assertEqual(config.threshold_sweeps[0].id, "threshold-alpha")
        self.assertEqual(config.threshold_sweeps[0].min_events, 2)
        self.assertEqual(config.threshold_sweeps[0].min_observations, 3)
        self.assertTrue(config.mean_centering.enabled)
        self.assertEqual(config.mean_centering.scope, "roi")
        self.assertEqual(config.distance.grouping_columns, ("participant_group", "task_id"))
        self.assertEqual(config.exclusions[0].id, "exclude-run-a")
        self.assertEqual(config.exclusions[0].source_config_field, "mvpa_set.run_exclusions")

    def test_all_pairs_condition_pair_mode_generates_deterministic_pairs(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["conditions"] = [  # type: ignore[index]
            {"id": "condition_a"},
            {"id": "condition_b"},
            {"id": "condition_c"},
            {"id": "condition_d"},
        ]
        mvpa_set["condition_pairs"] = {  # type: ignore[index]
            "mode": "all_pairs",
            "conditions": ["condition_a", "condition_b", "condition_c", "condition_d"],
            "id_template": "{condition_a}_minus_{condition_b}",
        }

        self.assertEqual(validate_mvpa_set_document(document), [])
        config = parse_mvpa_set_document(document)

        self.assertEqual(len(config.condition_pairs), 6)
        self.assertEqual(
            [(pair.id, pair.condition_id_a, pair.condition_id_b) for pair in config.condition_pairs],
            [
                ("condition_a_minus_condition_b", "condition_a", "condition_b"),
                ("condition_a_minus_condition_c", "condition_a", "condition_c"),
                ("condition_a_minus_condition_d", "condition_a", "condition_d"),
                ("condition_b_minus_condition_c", "condition_b", "condition_c"),
                ("condition_b_minus_condition_d", "condition_b", "condition_d"),
                ("condition_c_minus_condition_d", "condition_c", "condition_d"),
            ],
        )

    def test_all_pairs_condition_pair_mode_supports_two_and_three_conditions(self) -> None:
        for condition_ids, expected_count in ((["a", "b"], 1), (["a", "b", "c"], 3)):
            document = copy.deepcopy(_valid_config())
            mvpa_set = document["mvpa_set"]  # type: ignore[index]
            mvpa_set["conditions"] = [{"id": condition_id} for condition_id in condition_ids]  # type: ignore[index]
            mvpa_set["condition_pairs"] = {"mode": "all_pairs"}  # type: ignore[index]

            self.assertEqual(validate_mvpa_set_document(document), [])
            config = parse_mvpa_set_document(document)
            self.assertEqual(len(config.condition_pairs), expected_count)
            self.assertTrue(all("_minus_" in pair.id for pair in config.condition_pairs))

    def test_all_pairs_condition_pair_mode_rejects_unknown_duplicate_and_unsafe_conditions(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["conditions"] = [{"id": "condition_a"}, {"id": "condition_b"}]
            mvpa_set["condition_pairs"] = {
                "mode": "all_pairs",
                "conditions": ["condition_a", "condition_a", "condition_c", "bad condition"],
            }

        errors = _errors_for(mutate)

        self.assertTrue(any("duplicate condition id" in error for error in errors))
        self.assertTrue(any("unsafe condition id" in error for error in errors))
        self.assertTrue(any("unknown condition id" in error for error in errors))

    def test_all_pairs_condition_pair_mode_rejects_unsupported_mode(self) -> None:
        def mutate(mvpa_set: dict[str, object]) -> None:
            mvpa_set["condition_pairs"] = {"mode": "cartesian"}

        self.assertTrue(any("condition_pairs.mode" in error for error in _errors_for(mutate)))

    def test_manual_crossnobis_phase_specific_roi_config_validates(self) -> None:
        document = {
            "mvpa_set": {
                "name": "encoding_manual_main",
                "subjects": ["sub-002", "sub-003"],
                "sessions": ["ses-01"],
                "runs": ["run-01", "run-02"],
                "entities": {"task": "mvpa", "model": "ModelA", "space": "T1w"},
                "conditions": [
                    {"id": "pair_enc_hit", "aliases": ["PairEncHit"], "event_id": "pair_enc_hit"},
                    {"id": "item_enc_hit", "aliases": ["ItemEncHit"], "event_id": "item_enc_hit"},
                ],
                "condition_pairs": [{"id": "encoding_pair_minus_item", "left": "pair_enc_hit", "right": "item_enc_hit"}],
                "threshold_sweeps": [{"id": "main", "min_events": 1}],
                "pattern_sources": [
                    {
                        "name": "success_t1w_feat",
                        "backend": "fsl_feat_pe",
                        "root_ref": "success_feat_root",
                        "feat_dir_template": "{subject_dir}/{session_dir}/{subject_dir}_{session_dir}_{run_entity}_desc-ModelA.feat",
                        "events": {
                            "root_ref": "events_root",
                            "path": "{subject_dir}/{session_dir}/func/{subject_dir}_{session_dir}_{run_entity}_desc-{event_id}_events.txt",
                        },
                    }
                ],
                "roi_sources": [
                    {
                        "name": "encoding_rois",
                        "source": "explicit_masks",
                        "root_ref": "derivatives_root",
                        "roi_labels": ["EncodingFrontalPole", "EncodingPrecuneus"],
                        "mask_template": "rois/{subject_dir}/{session_dir}/{run_entity}/{roi_label}_mask.nii.gz",
                    }
                ],
                "distance": {
                    "metrics": ["crossnobis"],
                    "engine": "manual_diagonal_crossnobis_v1",
                    "cross_validation": {"unit": "run", "grouping_columns": ["subject_id", "session_id", "roi_label"]},
                    "noise_normalization": {
                        "method": "diagonal",
                        "variance_source": "sigmasquareds",
                        "nonpositive_policy": "filter_nonpositive_features",
                        "min_retained_features": 5,
                        "warn_dropped_feature_fraction": 0.10,
                    },
                },
                "run_exclusions": [
                    {"id": "exclude_toy_run01", "subject_id": "sub-001", "session_id": "ses-01", "run_id": "run-01"}
                ],
                "outputs": {
                    "runtime_root": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/{mvpa_set}"}
                },
            }
        }

        self.assertEqual(validate_mvpa_set_document(document), [])
        config = parse_mvpa_set_document(document)
        self.assertEqual(config.distance.engine, "manual_diagonal_crossnobis_v1")
        self.assertEqual(config.distance.noise_normalization.method, "diagonal")
        self.assertEqual(config.distance.noise_normalization.nonpositive_policy, "drop_features")
        self.assertEqual(config.distance.noise_normalization.min_retained_features, 5)
        self.assertEqual(config.distance.noise_normalization.warn_dropped_feature_fraction, 0.10)
        self.assertEqual(config.roi_sources[0].fields["roi_labels"], ["EncodingFrontalPole", "EncodingPrecuneus"])

    def test_importing_mvpa_does_not_require_forbidden_dependencies(self) -> None:
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
