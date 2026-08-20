from __future__ import annotations

from pathlib import Path
import sys
import unittest

CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.core.tool_adapters import (
    load_bids_analysis_tool_adapter,
    load_bids_tool_adapter,
    load_registered_bids_analysis_tool_adapter,
    load_registered_bids_tool_adapter,
    require_bids_analysis_model_authoring_adapter,
    registered_bids_analysis_tools,
    resolve_bids_analysis_tool_adapter_ref,
    registered_bids_tools,
    resolve_bids_tool_adapter_ref,
    validate_tool_options_shape,
)


class ToolAdapterTests(unittest.TestCase):
    def test_load_bids_tool_adapter_returns_fmripost_aroma_adapter(self) -> None:
        adapter = load_bids_tool_adapter(
            {"tool_adapter": "research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter"}
        )

        self.assertEqual(adapter.tool_name(), "fmripost_aroma")
        self.assertEqual(adapter.supported_input_derivatives(), ("deepprep-bold", "fmriprep"))
        self.assertTrue(callable(getattr(adapter, "expected_remote_input_files", None)))
        self.assertTrue(adapter.requires_input_derivative())

    def test_load_bids_tool_adapter_returns_deepprep_adapter(self) -> None:
        adapter = load_bids_tool_adapter(
            {"tool_adapter": "research_platform.neuro.deepprep.adapter:DeepPrepAdapter"}
        )

        self.assertEqual(adapter.tool_name(), "deepprep")
        self.assertEqual(adapter.supported_input_derivatives(), ())
        self.assertFalse(adapter.requires_input_derivative())
        self.assertTrue(callable(getattr(adapter, "build_runtime_plan", None)))

    def test_validate_tool_options_shape_rejects_compute_fields(self) -> None:
        errors = validate_tool_options_shape({"tool_options": {"cpus": 4, "denoising_method": "nonaggr"}})

        self.assertEqual(
            errors,
            ["preprocessing.tool_options must not define compute resources such as 'cpus'; keep CPU/memory under compute."],
        )

    def test_registered_bids_tool_registry_resolves_fmripost_aroma(self) -> None:
        self.assertEqual(registered_bids_tools(), ("deepprep", "fmripost_aroma"))
        self.assertEqual(
            resolve_bids_tool_adapter_ref("deepprep"),
            "research_platform.neuro.deepprep.adapter:DeepPrepAdapter",
        )
        self.assertEqual(
            resolve_bids_tool_adapter_ref("fmripost_aroma"),
            "research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter",
        )
        self.assertEqual(load_registered_bids_tool_adapter("deepprep").tool_name(), "deepprep")
        self.assertEqual(load_registered_bids_tool_adapter("fmripost_aroma").tool_name(), "fmripost_aroma")

    def test_registered_bids_analysis_tool_registry_resolves_feat(self) -> None:
        self.assertEqual(registered_bids_analysis_tools(), ("feat",))
        self.assertEqual(
            resolve_bids_analysis_tool_adapter_ref("feat"),
            "research_platform.neuro.fsl.feat.adapter:FeatAnalysisAdapter",
        )
        self.assertEqual(
            load_registered_bids_analysis_tool_adapter("feat").tool_name(),
            "feat",
        )

    def test_load_bids_analysis_tool_adapter_returns_feat_adapter(self) -> None:
        adapter = load_bids_analysis_tool_adapter(
            {"adapter": "research_platform.neuro.fsl.feat.adapter:FeatAnalysisAdapter"}
        )

        self.assertEqual(adapter.tool_name(), "feat")
        self.assertTrue(callable(getattr(adapter, "build_runtime_plan", None)))
        self.assertTrue(callable(getattr(require_bids_analysis_model_authoring_adapter(adapter, tool_name="feat"), "init_model_document", None)))


if __name__ == "__main__":
    unittest.main()
