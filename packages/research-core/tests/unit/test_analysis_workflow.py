from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import textwrap
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.core.analysis_workflow import (
    AnalysisWorkflowConfig,
    parse_analysis_workflow_document,
    plan_analysis_workflow_recipe,
    validate_analysis_workflow_document,
)
from research_platform.core.config import parse_yaml


def _workflow_yaml() -> str:
    return """
analysis_workflow:
  name: workflow_template
  runtime_tag: runtime_template
  cohort:
    subjects:
      include:
        - "<subject>"
    sessions:
      - "<session>"
    tasks:
      - "<task>"
    runs:
      - "<run>"
  root_refs:
    localizer_feat_root:
      description: localizer FEAT source root
    mvpa_feat_root:
      description: MVPA FEAT source root
    roi_catalog_root:
      description: ROI catalog source root
  output_roots:
    runtime_root:
      root_ref: artifact_root
      path: .research-platform/analysis/{workflow}/{runtime_tag}
    published_root:
      root_ref: derivatives_root
      path: derivatives/analysis/{workflow}
  stages:
    -
      name: localizer_inventory
      kind: neuro.localizer_inventory
      extension: mvpa
    -
      name: mvpa_plan
      kind: neuro.mvpa
      extension: mvpa
  publication:
    enabled: false
  reporting:
    enabled: true
    formats:
      - json
  extensions:
    mvpa:
      localizer_feat_sources: []
"""


def _document() -> dict[str, object]:
    document = parse_yaml(textwrap.dedent(_workflow_yaml()), resolve_env=False)
    assert isinstance(document, dict)
    return document


class AnalysisWorkflowRecipeTests(unittest.TestCase):
    def test_synthetic_yaml_parses_to_generic_config(self) -> None:
        config = parse_analysis_workflow_document(_document())

        self.assertIsInstance(config, AnalysisWorkflowConfig)
        self.assertEqual(config.name, "workflow_template")
        self.assertEqual(config.runtime_tag, "runtime_template")
        self.assertEqual(config.selector.subjects, ("<subject>",))
        self.assertEqual(config.selector.sessions, ("<session>",))
        self.assertEqual(config.selector.tasks, ("<task>",))
        self.assertEqual(config.selector.runs, ("<run>",))
        self.assertEqual([stage.name for stage in config.stages], ["localizer_inventory", "mvpa_plan"])
        self.assertEqual(sorted(config.extensions), ["mvpa"])

    def test_plan_is_json_serializable_and_deferred(self) -> None:
        plan = plan_analysis_workflow_recipe(_document(), context={"caller": "unit_test"})
        payload = plan.to_dict()

        self.assertEqual(plan.status, "deferred")
        self.assertTrue(plan.valid)
        self.assertFalse(plan.executed)
        self.assertTrue(payload["plan_only"])
        self.assertEqual(payload["context"]["caller"], "unit_test")
        self.assertEqual(payload["extensions"][0]["reason"], "extension_validation_deferred_to_owner_package")
        self.assertEqual(payload["output_roots"][0]["relative_path_template"], ".research-platform/analysis/{workflow}/{runtime_tag}")
        json.dumps(payload, sort_keys=True)

    def test_invalid_absolute_output_path_is_rejected_without_filesystem_checks(self) -> None:
        document = parse_yaml(
            textwrap.dedent(
                """
                analysis_workflow:
                  name: workflow_template
                  cohort:
                    subjects:
                      - "<subject>"
                    sessions:
                      - "<session>"
                    tasks:
                      - "<task>"
                    runs:
                      - "<run>"
                  stages:
                    -
                      name: inventory
                      kind: inventory
                  output_roots:
                    runtime_root:
                      root_ref: artifact_root
                      path: /absolute/output/root
                """
            ),
            resolve_env=False,
        )

        errors = validate_analysis_workflow_document(document)
        plan = plan_analysis_workflow_recipe(document)

        self.assertTrue(any("absolute path" in error for error in errors))
        self.assertEqual(plan.status, "invalid")
        self.assertFalse(plan.valid)

    def test_execution_fields_are_rejected_in_schema_only_recipe(self) -> None:
        document = parse_yaml(
            textwrap.dedent(
                """
                analysis_workflow:
                  name: workflow_template
                  cohort:
                    subjects:
                      - "<subject>"
                    sessions:
                      - "<session>"
                    tasks:
                      - "<task>"
                    runs:
                      - "<run>"
                  stages:
                    -
                      name: inventory
                      kind: inventory
                      command: run-something
                  output_roots:
                    runtime_root:
                      root_ref: artifact_root
                      path: analysis/{workflow}
                """
            ),
            resolve_env=False,
        )

        errors = validate_analysis_workflow_document(document)

        self.assertTrue(any("plan-only workflow recipe" in error for error in errors))

    def test_importing_generic_workflow_contracts_does_not_import_domain_packages(self) -> None:
        script = textwrap.dedent(
            """
            import builtins
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path("packages/research-core/src").resolve()))
            forbidden = {
                "nibabel",
                "nilearn",
                "numpy",
                "pandas",
                "polars",
                "research_platform.neuro",
                "research_platform.analysis",
                "research_platform.bids",
                "research_platform.viz",
                "research_platform.ml",
            }
            real_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name in forbidden or any(name.startswith(prefix + ".") for prefix in forbidden):
                    raise RuntimeError(f"forbidden import: {name}")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import
            import research_platform.core.analysis_workflow  # noqa: F401
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
