from __future__ import annotations

from pathlib import Path
import copy
import json
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.mvpa import MvpaDiscoveryPlan, plan_mvpa_discovery

from test_mvpa_config import _valid_config


class MvpaPlanTests(unittest.TestCase):
    def test_plan_returns_invalid_plan_with_errors_for_invalid_config(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["distance"]["metrics"] = ["unsupported"]  # type: ignore[index]

        plan = plan_mvpa_discovery(document)

        self.assertEqual(plan.status, "invalid")
        self.assertFalse(plan.valid)
        self.assertTrue(plan.errors)
        self.assertEqual(plan.mvpa_set_name, "memory_mvpa")

    def test_plan_returns_valid_deferred_plan_for_valid_config(self) -> None:
        plan = plan_mvpa_discovery(_valid_config(), context={"caller": "unit_test"})

        self.assertEqual(plan.status, "deferred")
        self.assertTrue(plan.valid)
        self.assertEqual(plan.errors, ())
        self.assertEqual(plan.mvpa_set_name, "memory_mvpa")
        self.assertEqual(plan.context["caller"], "unit_test")

    def test_deferred_backend_rows_are_generated(self) -> None:
        plan = plan_mvpa_discovery(_valid_config())

        self.assertEqual(len(plan.pattern_sources), 1)
        self.assertEqual(plan.pattern_sources[0].status, "deferred")
        self.assertEqual(plan.pattern_sources[0].reason, "adapter_planning_not_attempted")
        self.assertEqual(plan.pattern_sources[0].backend, "fsl_feat_pe")
        self.assertTrue(plan.distances)
        self.assertTrue(all(row.status == "planned" for row in plan.distances))
        self.assertTrue(
            all(
                row.reason == "configured_for_representation_aware_runtime"
                for row in plan.distances
            )
        )

    def test_json_dumps_plan_to_dict_works(self) -> None:
        plan = plan_mvpa_discovery(_valid_config())

        encoded = json.dumps(plan.to_dict(), sort_keys=True)

        self.assertIn('"status": "deferred"', encoded)
        self.assertIn('"relative_path_template"', encoded)

    def test_plan_valid_is_distinct_from_schema_valid(self) -> None:
        deferred = plan_mvpa_discovery(_valid_config())
        failed_plan = MvpaDiscoveryPlan(
            mvpa_set_name="neutral_mvpa",
            status="error",
            schema_valid=True,
            errors=("source planning failed",),
        )

        self.assertTrue(deferred.schema_valid)
        self.assertTrue(deferred.plan_valid)
        self.assertTrue(deferred.to_dict()["valid"])
        self.assertFalse(failed_plan.plan_valid)
        self.assertFalse(failed_plan.valid)
        self.assertEqual(
            failed_plan.to_dict()["schema_valid"],
            True,
        )
        self.assertEqual(failed_plan.to_dict()["plan_valid"], False)
        self.assertEqual(failed_plan.to_dict()["valid"], False)

    def test_plan_output_previews_keep_root_refs_and_relative_templates(self) -> None:
        plan = plan_mvpa_discovery(_valid_config())
        previews = {row.name: row for row in plan.outputs}

        self.assertEqual(previews["runtime_root"].root_ref, "artifact_root")
        self.assertEqual(previews["runtime_root"].relative_path_template, ".research-platform/mvpa/{mvpa_set}")
        self.assertEqual(previews["published_root"].root_ref, "dataset_derivatives_root")

    def test_missing_input_rows_are_not_checked(self) -> None:
        plan = plan_mvpa_discovery(_valid_config())

        self.assertEqual(plan.missing_inputs[0].policy, "warn")
        self.assertEqual(plan.missing_inputs[0].status, "not_checked")

    def test_nonexistent_roots_and_masks_do_not_cause_filesystem_errors_in_plan(self) -> None:
        document = copy.deepcopy(_valid_config())
        mvpa_set = document["mvpa_set"]  # type: ignore[index]
        mvpa_set["roi_sources"] = [  # type: ignore[index]
            {
                "name": "explicit_rois",
                "source": "explicit_masks",
                "root_ref": "root_that_does_not_exist",
                "masks": [{"label": "V1", "path": "no/such/mask.nii.gz"}],
            }
        ]

        plan = plan_mvpa_discovery(document)

        self.assertEqual(plan.status, "deferred")
        self.assertEqual(plan.roi_sources[0].status, "deferred")
        self.assertEqual(plan.roi_sources[0].mask_count, 1)


if __name__ == "__main__":
    unittest.main()
