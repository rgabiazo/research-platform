from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.bids.cli import main
from research_platform.neuro.events import load_build_spec


class GenericNonRecognitionEventsIntegrationTest(unittest.TestCase):
    def _build_events(self, *, spec_name: str, source_name: str, artifact_root: Path) -> None:
        spec_path = PACKAGE_ROOT.parents[1] / "project" / "project-example" / "config" / "events" / spec_name
        source_path = PACKAGE_ROOT / "tests" / "fixtures" / "nonrecognition" / "raw" / source_name

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
    def test_simplecues_generic_build_matches_expected_tsv(self) -> None:
        expected_path = (
            PACKAGE_ROOT
            / "tests"
            / "fixtures"
            / "nonrecognition"
            / "expected"
            / "sub-007_task-simplecues_run-01_events.tsv"
        )

        with tempfile.TemporaryDirectory(prefix="simplecues_events_") as artifact_root:
            self._build_events(spec_name="simplecues.v2.yaml", source_name="simple_cues.csv", artifact_root=Path(artifact_root))
            staged_path = Path(artifact_root) / "staged" / "sub-007" / "func" / "sub-007_task-simplecues_run-01_events.tsv"
            self.assertEqual(staged_path.read_text(encoding="utf-8"), expected_path.read_text(encoding="utf-8"))

    @unittest.skipUnless(
        importlib.util.find_spec("pandas") is not None,
        "pandas backend extra is not installed",
    )
    def test_simplecues_plus_generic_build_matches_expected_tsvs(self) -> None:
        expected_dir = PACKAGE_ROOT / "tests" / "fixtures" / "nonrecognition" / "expected"

        with tempfile.TemporaryDirectory(prefix="simplecues_plus_events_") as artifact_root:
            self._build_events(
                spec_name="simplecues-plus.v2.yaml",
                source_name="simple_cues_plus.csv",
                artifact_root=Path(artifact_root),
            )

            for run in (1, 2):
                staged_path = (
                    Path(artifact_root)
                    / "staged"
                    / "sub-007"
                    / "func"
                    / f"sub-007_task-simplecuesplus_run-0{run}_events.tsv"
                )
                expected_path = expected_dir / f"sub-007_task-simplecuesplus_run-0{run}_events.tsv"
                self.assertEqual(staged_path.read_text(encoding="utf-8"), expected_path.read_text(encoding="utf-8"))

    def test_simplecues_required_source_columns_come_from_compiled_plan(self) -> None:
        spec_path = PACKAGE_ROOT.parents[1] / "project" / "project-example" / "config" / "events" / "simplecues.v2.yaml"
        spec = load_build_spec(spec_path)

        self.assertEqual(spec.required_source_columns, spec.compiled_plan.required_source_columns)
        self.assertEqual(
            spec.required_source_columns,
            {"participant", "run", "stim_path", "onset_s", "duration_s", "rt_s", "key_resp", "cue_type"},
        )

    def test_simplecues_plus_required_source_columns_come_from_compiled_plan(self) -> None:
        spec_path = PACKAGE_ROOT.parents[1] / "project" / "project-example" / "config" / "events" / "simplecues-plus.v2.yaml"
        spec = load_build_spec(spec_path)

        self.assertEqual(spec.required_source_columns, spec.compiled_plan.required_source_columns)
        self.assertEqual(
            spec.required_source_columns,
            {"participant", "run", "stim_path", "onset_s", "duration_s", "rt_s", "response_key", "expected_key"},
        )

    def test_v2_only_conditions_unknown_name_fails_loudly(self) -> None:
        spec_dir = PACKAGE_ROOT.parents[1] / "project" / "project-example" / "config" / "events"
        base_path = spec_dir / "simplecues-plus.v2.yaml"
        ops_path = spec_dir / "simplecues-plus.v2.ops.yaml"
        sidecar_path = spec_dir / "simplecues-plus.v2.events.json.yaml"

        with tempfile.TemporaryDirectory(prefix="bad_only_conditions_") as temp_root:
            temp_dir = Path(temp_root)
            temp_spec = temp_dir / "bad-simplecues-plus.v2.yaml"
            temp_ops = temp_dir / "bad-simplecues-plus.v2.ops.yaml"
            temp_sidecar = temp_dir / "bad-simplecues-plus.v2.events.json.yaml"

            temp_spec.write_text(base_path.read_text(encoding="utf-8"), encoding="utf-8")
            ops_payload = json.loads(ops_path.read_text(encoding="utf-8"))
            ops_payload["row_set_templates"][0]["only_conditions"] = ["go", "typo_condition"]
            temp_ops.write_text(json.dumps(ops_payload, indent=2), encoding="utf-8")
            temp_sidecar.write_text(sidecar_path.read_text(encoding="utf-8"), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"run_\{run\}_\{condition\}.*typo_condition.*Declared conditions: go, hold",
            ):
                load_build_spec(temp_spec)

if __name__ == "__main__":
    unittest.main()
