from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.bids.cli import main
from research_platform.neuro.events import load_build_spec


class ToyMemoryEventsIntegrationTest(unittest.TestCase):
    def _build_toy_memory(self, *, spec_name: str, artifact_root: Path) -> None:
        spec_path = PACKAGE_ROOT.parents[1] / "project" / "project-example" / "config" / "events" / spec_name
        source_path = (
            PACKAGE_ROOT
            / "tests"
            / "fixtures"
            / "toy-memory"
            / "raw"
            / "toy01_visit01_toymemory_2099-01-01.csv"
        )
        exit_code = main(
            [
                "events",
                "build",
                "--spec",
                str(spec_path),
                "--source",
                str(source_path),
                "--artifact-root",
                str(artifact_root),
                "--backend",
                "pandas",
            ]
        )
        self.assertEqual(exit_code, 0)

    @unittest.skipUnless(
        importlib.util.find_spec("pandas") is not None,
        "pandas backend extra is not installed",
    )
    def test_toy_memory_build_matches_three_golden_tsvs_for_legacy_and_generic_specs(self) -> None:
        expected_dir = PACKAGE_ROOT / "tests" / "fixtures" / "toy-memory" / "expected"

        with tempfile.TemporaryDirectory(prefix="toy_memory_events_") as temp_root:
            legacy_artifact_root = Path(temp_root) / "legacy"
            generic_artifact_root = Path(temp_root) / "generic"
            self._build_toy_memory(spec_name="toy-memory.yaml", artifact_root=legacy_artifact_root)
            self._build_toy_memory(spec_name="toy-memory.v2.yaml", artifact_root=generic_artifact_root)

            for run in (1, 2, 3):
                output_name = (
                    f"sub-toy01_ses-01_task-toymemory_dir-AP_run-{run:02d}_events.tsv"
                )
                expected_path = expected_dir / output_name
                relative_output = Path("staged") / "sub-toy01" / "ses-01" / "func" / output_name
                legacy_path = legacy_artifact_root / relative_output
                generic_path = generic_artifact_root / relative_output
                expected_text = expected_path.read_text(encoding="utf-8")
                self.assertEqual(legacy_path.read_text(encoding="utf-8"), expected_text)
                self.assertEqual(generic_path.read_text(encoding="utf-8"), expected_text)
                self.assertEqual(generic_path.read_text(encoding="utf-8"), legacy_path.read_text(encoding="utf-8"))

    def test_toy_memory_required_source_columns_come_from_compiled_plan(self) -> None:
        config_root = PACKAGE_ROOT.parents[1] / "project" / "project-example" / "config" / "events"
        spec_path = config_root / "toy-memory.yaml"
        generic_spec_path = config_root / "toy-memory.v2.yaml"

        spec = load_build_spec(spec_path)
        generic_spec = load_build_spec(generic_spec_path)

        self.assertEqual(spec.required_source_columns, spec.compiled_plan.required_source_columns)
        self.assertEqual(generic_spec.required_source_columns, generic_spec.compiled_plan.required_source_columns)
        self.assertEqual(spec.required_source_columns, generic_spec.required_source_columns)
        self.assertIn("image_old_new", spec.required_source_columns)
        self.assertIn("response_shape_match", spec.required_source_columns)
        self.assertIn("response_color_tone", spec.required_source_columns)
        self.assertIn("response_word_kind", spec.required_source_columns)
        self.assertIn("response_seen_new", spec.required_source_columns)
        self.assertIn("stimulus_file", spec.required_source_columns)
        self.assertIn("toy_id", spec.required_source_columns)
        self.assertIn("visit", spec.required_source_columns)


if __name__ == "__main__":
    unittest.main()
