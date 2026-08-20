from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import textwrap
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CORE_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-core"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))

from research_platform.core.config import parse_yaml
from research_platform.neuro.mvpa import (
    MvpaWorkflowExtensionConfig,
    parse_mvpa_workflow_extension_document,
    plan_mvpa_workflow_extension,
    validate_mvpa_workflow_extension_document,
)


def _workflow_yaml() -> str:
    return """
analysis_workflow:
  name: workflow_template
  extensions:
    mvpa:
      localizer_feat_sources:
        -
          name: localizer_feat
          root_ref: localizer_feat_root
          feat_dir_template: localizer/{subject}/{session}/{task}/{run}/model.feat
          design_file: design.fsf
          pe_image_template: stats/pe{pe_number}.nii.gz
      mvpa_feat_sources:
        -
          name: mvpa_feat
          root_ref: mvpa_feat_root
          feat_dir_template: mvpa/{subject}/{session}/{task}/{run}/model.feat
          design_file: design.fsf
          pe_image_template: stats/pe{pe_number}.nii.gz
          noise_image_template: stats/sigmasquareds.nii.gz
      roi_collections:
        -
          name: roi_collection_template
          catalog_ref: roi_catalog_template
          roi_set_refs:
            - roi_set_template
          metadata:
            family: reusable_roi_catalog
      loso:
        enabled: true
        heldout_unit: subject
        grouping:
          - session
          - task
        min_training_observations: 1
      transforms:
        -
          name: roi_to_mvpa_space
          from_space: "<roi_space>"
          to_space: "<mvpa_space>"
          transform_ref: transform_template
          apply_to:
            - roi_masks
      pe_mapping_aliases:
        <condition_a>:
          - localizer_condition_a
          - mvpa_condition_a
        <condition_b>:
          - localizer_condition_b
          - mvpa_condition_b
      condition_pairs:
        -
          name: pair_template
          left: "<condition_a>"
          right: "<condition_b>"
      threshold_sweeps:
        -
          name: event_sweep_template
          min_events: 1
          min_observations: 2
      run_exclusions:
        -
          id: run_exclusion_template
          reason: synthetic exclusion
          selectors:
            subject_id: "<subject>"
            session_id: "<session>"
            run_id: "<run>"
      cross_validation:
        unit: run
      noise_normalization:
        method: diagonal
      mean_centering:
        enabled: true
        scope: run
      publication:
        enabled: true
        derivative_name: mvpa_workflow_template
      reporting:
        enabled: true
        formats:
          - json
        sections:
          - quality_control
"""


def _document() -> dict[str, object]:
    document = parse_yaml(textwrap.dedent(_workflow_yaml()), resolve_env=False)
    assert isinstance(document, dict)
    return document


class MvpaWorkflowExtensionTests(unittest.TestCase):
    def test_synthetic_yaml_parses_to_mvpa_extension_config(self) -> None:
        config = parse_mvpa_workflow_extension_document(_document())

        self.assertIsInstance(config, MvpaWorkflowExtensionConfig)
        self.assertEqual(config.localizer_feat_sources[0].name, "localizer_feat")
        self.assertEqual(config.mvpa_feat_sources[0].noise_image_template, "stats/sigmasquareds.nii.gz")
        self.assertEqual(config.roi_collections[0].catalog_ref, "roi_catalog_template")
        self.assertEqual(config.loso.heldout_unit, "subject")
        self.assertEqual(config.crossvalidation_unit, "run")
        self.assertEqual(config.noise_normalization, "diagonal")
        self.assertTrue(config.mean_centering["enabled"])
        self.assertEqual(config.condition_pairs[0].left, "<condition_a>")

    def test_mvpa_extension_plan_is_json_serializable_and_nonexecuting(self) -> None:
        plan = plan_mvpa_workflow_extension(_document())
        payload = plan.to_dict()

        self.assertEqual(plan.status, "deferred")
        self.assertTrue(plan.valid)
        self.assertFalse(plan.executed)
        self.assertTrue(payload["plan_only"])
        self.assertEqual(payload["workflow_name"], "workflow_template")
        self.assertEqual(payload["localizer_feat_sources"][0]["status"], "deferred")
        self.assertEqual(payload["mvpa_feat_sources"][0]["reason"], "feat_source_discovery_not_implemented_schema_slice")
        self.assertEqual(payload["roi_collections"][0]["status"], "configured")
        self.assertEqual(payload["transforms"][0]["reason"], "mask_transform_execution_not_implemented_schema_slice")
        self.assertEqual(payload["threshold_sweeps"][0]["status"], "not_evaluated")
        self.assertEqual(payload["settings"]["noise_normalization"], "diagonal")
        json.dumps(payload, sort_keys=True)

    def test_hard_coded_pe_numbers_are_rejected(self) -> None:
        document = parse_yaml(
            textwrap.dedent(
                """
                mvpa:
                  localizer_feat_sources:
                    -
                      name: localizer_feat
                      root_ref: localizer_feat_root
                  mvpa_feat_sources:
                    -
                      name: mvpa_feat
                      root_ref: mvpa_feat_root
                  pe_mapping_aliases:
                    -
                      condition: "<condition_a>"
                      aliases:
                        - condition_a
                      pe_number: 1
                """
            ),
            resolve_env=False,
        )

        errors = validate_mvpa_workflow_extension_document(document)
        plan = plan_mvpa_workflow_extension(document)

        self.assertTrue(any("PE/COPE/contrast numbers" in error for error in errors))
        self.assertEqual(plan.status, "invalid")
        self.assertFalse(plan.valid)

    def test_hard_coded_pe_numbers_in_templates_are_rejected(self) -> None:
        document = parse_yaml(
            textwrap.dedent(
                """
                mvpa:
                  localizer_feat_sources:
                    -
                      name: localizer_feat
                      root_ref: localizer_feat_root
                      pe_image_template: stats/pe1.nii.gz
                  mvpa_feat_sources:
                    -
                      name: mvpa_feat
                      root_ref: mvpa_feat_root
                      pe_image_template: stats/pe{pe_number}.nii.gz
                """
            ),
            resolve_env=False,
        )

        errors = validate_mvpa_workflow_extension_document(document)

        self.assertTrue(any("hard-coded PE/COPE/contrast number" in error for error in errors))

    def test_absolute_feat_template_is_rejected_without_loading_inputs(self) -> None:
        document = parse_yaml(
            textwrap.dedent(
                """
                mvpa:
                  localizer_feat_sources:
                    -
                      name: localizer_feat
                      root_ref: localizer_feat_root
                      feat_dir_template: /absolute/localizer/model.feat
                  mvpa_feat_sources:
                    -
                      name: mvpa_feat
                      root_ref: mvpa_feat_root
                """
            ),
            resolve_env=False,
        )

        errors = validate_mvpa_workflow_extension_document(document)

        self.assertTrue(any("absolute path" in error for error in errors))

    def test_importing_mvpa_workflow_extension_does_not_import_core_or_heavy_packages(self) -> None:
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
            import research_platform.neuro.mvpa.workflow  # noqa: F401
            import research_platform.neuro.mvpa  # noqa: F401
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=WORKSPACE_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
