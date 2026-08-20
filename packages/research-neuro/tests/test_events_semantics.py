from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = PACKAGE_ROOT / "tests" / "fixtures"
SPECS_ROOT = FIXTURES_ROOT / "specs" / "events"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import research_platform.neuro.events as neuro_events
from research_platform.neuro.events import (
    EventsSemanticAPI,
    EventsSemanticResult,
    EventsSemanticRun,
    SemanticEventsBuildResult,
    SemanticRunRows,
    build_run_rows,
    build_semantic_events,
    events_semantic_api,
    load_build_spec,
    plan_semantic_events,
    read_source_rows,
    resolve_run_groups,
)


def _expected_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


class NeuroEventsSemanticsTest(unittest.TestCase):
    def _assert_toy_memory_rows_match_expected(
        self,
        actual_rows: list[dict[str, str]],
        expected_rows: list[dict[str, str]],
    ) -> None:
        self.assertEqual(len(actual_rows), len(expected_rows))
        for actual, expected in zip(actual_rows, expected_rows):
            self.assertEqual(set(actual), set(expected))
            for column, expected_value in expected.items():
                actual_value = actual[column]
                if column in {"onset", "duration", "response_time"} and expected_value != "n/a":
                    self.assertAlmostEqual(float(actual_value), float(expected_value), places=12)
                else:
                    self.assertEqual(actual_value, expected_value)

    def _assert_toy_memory_coverage(self, rows: list[dict[str, str]], *, run: int) -> None:
        self.assertEqual(len(rows), 78)
        self.assertEqual(Counter(row["condition"] for row in rows), Counter({
            "shape": 26,
            "color": 26,
            "word": 26,
        }))
        self.assertEqual(Counter(row["phase"] for row in rows), Counter({
            "encoding": 39,
            "recognition": 39,
        }))
        self.assertEqual(sum(row["is_instruction"] == "1" for row in rows), 6)

        encoding_rows = [
            row for row in rows if row["phase"] == "encoding" and row["is_instruction"] != "1"
        ]
        self.assertEqual(
            Counter(row["enc_later_outcome"] for row in encoding_rows),
            Counter({"hit": 9, "miss": 9, "not_tested": 18}),
        )

        recognition_rows = [
            row for row in rows if row["phase"] == "recognition" and row["is_instruction"] != "1"
        ]
        expected_recognition = Counter(
            {"hit": 9, "miss": 9, "false_alarm": 9, "correct_rejection": 9}
        )
        if run == 2:
            expected_recognition["correct_rejection"] = 8
            expected_recognition["n/a"] = 1
        self.assertEqual(Counter(row["acc_label"] for row in recognition_rows), expected_recognition)
        self.assertEqual(sum(row["is_error"] == "1" for row in rows), 27)
        self.assertEqual(
            sum(row["analysis_include"] == "1" for row in rows),
            17 if run == 2 else 18,
        )

    def test_toy_memory_spec_loading_matches_legacy_and_generic_required_columns(self) -> None:
        legacy_spec = load_build_spec(SPECS_ROOT / "toy-memory.yaml")
        generic_spec = load_build_spec(SPECS_ROOT / "toy-memory.v2.yaml")

        self.assertEqual(legacy_spec.required_source_columns, legacy_spec.compiled_plan.required_source_columns)
        self.assertEqual(generic_spec.required_source_columns, generic_spec.compiled_plan.required_source_columns)
        self.assertEqual(legacy_spec.required_source_columns, generic_spec.required_source_columns)

    def test_semantic_contract_exports_canonical_and_compatibility_types(self) -> None:
        expected_exports = {
            "EventsSemanticAPI",
            "EventsSemanticResult",
            "EventsSemanticRun",
            "SemanticEventsBuildResult",
            "SemanticRunRows",
            "build_run_rows",
            "build_semantic_events",
            "events_semantic_api",
            "load_build_spec",
            "plan_semantic_events",
            "read_source_rows",
            "resolve_run_groups",
            "resolve_subject_session",
            "validate_compiled_plan_rows",
        }

        self.assertIs(EventsSemanticResult, SemanticEventsBuildResult)
        self.assertIs(EventsSemanticRun, SemanticRunRows)
        self.assertIs(events_semantic_api.result_type, EventsSemanticResult)
        self.assertIs(events_semantic_api.run_type, EventsSemanticRun)
        self.assertTrue(expected_exports.issubset(set(neuro_events.__all__)))
        self.assertTrue(callable(load_build_spec))
        self.assertTrue(callable(read_source_rows))
        self.assertTrue(callable(resolve_run_groups))
        self.assertTrue(callable(build_run_rows))

    def test_events_semantic_api_is_canonical_root_surface(self) -> None:
        api: EventsSemanticAPI = events_semantic_api

        plan_result = api.plan(
            spec_path=SPECS_ROOT / "simplecues.v2.yaml",
            source_path=FIXTURES_ROOT / "nonrecognition" / "raw" / "simple_cues.csv",
        )
        build_result = api.build(
            spec_path=SPECS_ROOT / "simplecues.v2.yaml",
            source_path=FIXTURES_ROOT / "nonrecognition" / "raw" / "simple_cues.csv",
        )

        self.assertIsInstance(plan_result, EventsSemanticResult)
        self.assertIsInstance(build_result, EventsSemanticResult)
        self.assertEqual(plan_result.run_rows_by_run(), build_result.run_rows_by_run())
        self.assertEqual(plan_result.total_row_count(), build_result.total_row_count())

    def test_toy_memory_grouping_and_rows_match_expected_for_legacy_and_generic_specs(self) -> None:
        raw_path = (
            FIXTURES_ROOT
            / "toy-memory"
            / "raw"
            / "toy01_visit01_toymemory_2099-01-01.csv"
        )
        expected_dir = FIXTURES_ROOT / "toy-memory" / "expected"

        legacy_spec = load_build_spec(SPECS_ROOT / "toy-memory.yaml")
        generic_spec = load_build_spec(SPECS_ROOT / "toy-memory.v2.yaml")
        rows = read_source_rows(raw_path, encoding=legacy_spec.source_encoding)

        legacy_groups = resolve_run_groups(rows, legacy_spec)
        generic_groups = resolve_run_groups(rows, generic_spec)

        self.assertEqual(len(legacy_groups), 18)
        self.assertEqual(len(generic_groups), 18)

        legacy_run_rows = build_run_rows(legacy_groups, legacy_spec)
        generic_run_rows = build_run_rows(generic_groups, generic_spec)

        for run in (1, 2, 3):
            expected = _expected_rows(
                expected_dir
                / f"sub-toy01_ses-01_task-toymemory_dir-AP_run-{run:02d}_events.tsv"
            )
            self._assert_toy_memory_rows_match_expected(legacy_run_rows[run], expected)
            self._assert_toy_memory_rows_match_expected(generic_run_rows[run], expected)
            self._assert_toy_memory_coverage(legacy_run_rows[run], run=run)
            self._assert_toy_memory_coverage(generic_run_rows[run], run=run)
            self.assertEqual(generic_run_rows[run], legacy_run_rows[run])

    def test_plan_semantic_events_returns_canonical_handoff_object(self) -> None:
        result = plan_semantic_events(
            spec_path=SPECS_ROOT / "simplecues.v2.yaml",
            source_path=FIXTURES_ROOT / "nonrecognition" / "raw" / "simple_cues.csv",
        )

        self.assertIsInstance(result, EventsSemanticResult)
        self.assertEqual(result.subject, "007")
        self.assertIsNone(result.session)
        self.assertEqual(result.spec.name, "simplecues")
        self.assertEqual(result.total_row_count(), len(result.runs[0].rows))
        self.assertEqual(result.run_rows_by_run(), {1: result.runs[0].rows})
        self.assertEqual(result.warnings, [])
        self.assertEqual([run.run for run in result.runs], [1])
        self.assertTrue(all(isinstance(run, EventsSemanticRun) for run in result.runs))

    def test_simplecues_build_semantic_events_matches_expected_rows(self) -> None:
        result = build_semantic_events(
            spec_path=SPECS_ROOT / "simplecues.v2.yaml",
            source_path=FIXTURES_ROOT / "nonrecognition" / "raw" / "simple_cues.csv",
        )
        self.assertEqual(result.subject, "007")
        self.assertIsNone(result.session)
        self.assertEqual(len(result.runs), 1)
        self.assertEqual(result.total_row_count(), result.runs[0].row_count)
        self.assertEqual(
            result.runs[0].rows,
            _expected_rows(
                FIXTURES_ROOT / "nonrecognition" / "expected" / "sub-007_task-simplecues_run-01_events.tsv"
            ),
        )

    def test_simplecues_plus_build_semantic_events_matches_expected_rows(self) -> None:
        result = build_semantic_events(
            spec_path=SPECS_ROOT / "simplecues-plus.v2.yaml",
            source_path=FIXTURES_ROOT / "nonrecognition" / "raw" / "simple_cues_plus.csv",
        )

        expected_dir = FIXTURES_ROOT / "nonrecognition" / "expected"
        self.assertEqual(result.subject, "007")
        self.assertEqual([run.run for run in result.runs], [1, 2])
        self.assertEqual(result.run_rows_by_run(), {run.run: run.rows for run in result.runs})
        for run in result.runs:
            self.assertEqual(
                run.rows,
                _expected_rows(expected_dir / f"sub-007_task-simplecuesplus_run-0{run.run}_events.tsv"),
            )

    def test_simplecues_plus_grouping_tracks_expected_row_sets(self) -> None:
        spec_path = SPECS_ROOT / "simplecues-plus.v2.yaml"
        source_path = FIXTURES_ROOT / "nonrecognition" / "raw" / "simple_cues_plus.csv"
        spec = load_build_spec(spec_path)
        rows = read_source_rows(source_path, encoding=spec.source_encoding)
        groups = resolve_run_groups(rows, spec)

        self.assertEqual(
            [(group.run, group.phase.condition, group.phase.phase, group.phase.name) for group in groups],
            [
                (1, "go", "task", "run_1_go"),
                (1, "hold", "task", "run_1_hold"),
                (2, "go", "task", "run_2_go"),
                (2, "hold", "task", "run_2_hold"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
