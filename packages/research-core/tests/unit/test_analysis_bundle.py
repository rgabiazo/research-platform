from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from research_platform.core.analysis_workflow import (
    resolve_analysis_bundle,
    validate_analysis_bundle_document,
)
from research_platform.core.manifests import (
    read_manifest_table,
    write_manifest_table,
)


def _bundle(
    *,
    selection: dict[str, object] | None = None,
    key_columns: list[str] | None = None,
    occasion_column: str | None = None,
    occasion_order_column: str | None = None,
    required_occasions: list[str] | None = None,
    incomplete: str = "allow",
    components: dict[str, str] | None = None,
    stages: list[str] | None = None,
) -> dict[str, object]:
    units: dict[str, object] = {
        "key_columns": key_columns or ["subject_id"],
        "subject_column": "subject_id",
        "incomplete": incomplete,
    }
    if occasion_column is not None:
        units["occasion_column"] = occasion_column
    if occasion_order_column is not None:
        units["occasion_order_column"] = occasion_order_column
    if required_occasions is not None:
        units["required_occasions"] = required_occasions
    return {
        "analysis_bundle": {
            "name": "example-bundle",
            "selection": selection or {"batch": "units"},
            "units": units,
            "components": components if components is not None else {"roi_set": "example-rois"},
            "stages": stages if stages is not None else ["roi_build"],
        }
    }


def _cohorts(view: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "cohorts": {
            "example-cohort": view
            or {
                "batch": "units",
                "include": {},
                "exclude": [],
            }
        }
    }


def _write_table(
    root: Path,
    text: str,
    *,
    name: str = "units",
) -> dict[str, object]:
    path = root / f"{name}.tsv"
    path.write_bytes(text.encode("utf-8"))
    return {name: read_manifest_table(path)}


def _resolve(
    document: dict[str, object],
    tables: dict[str, object],
    *,
    cohorts: dict[str, object] | None = None,
    components: dict[str, tuple[str, ...]] | None = None,
):
    return resolve_analysis_bundle(
        document,
        cohorts_document=cohorts or {"cohorts": {}},
        batch_tables=tables,
        available_components=components or {"roi_set": ("example-rois",)},
        expected_name="example-bundle",
    )


class AnalysisBundleContractTests(unittest.TestCase):
    def test_empty_wrapped_bundle_is_invalid_without_resolver_traceback(self) -> None:
        document = {"analysis_bundle": {}}
        errors = validate_analysis_bundle_document(document)

        self.assertTrue(any("analysis_bundle.name" in error for error in errors))
        self.assertTrue(any("analysis_bundle.selection" in error for error in errors))
        result = resolve_analysis_bundle(
            document,
            cohorts_document={"cohorts": {}},
            batch_tables={},
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.errors, tuple(errors))

    def test_cross_sectional_batch_requires_no_fake_optional_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tables = _write_table(
                Path(tmp_dir),
                "subject_id\tcohort_id\nsub-001\tgroup-a\nsub-002\tgroup-b\n",
            )
            result = _resolve(_bundle(), tables)

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.source_batch["columns"], ["subject_id", "cohort_id"])
        self.assertEqual(
            [unit["values"]["subject_id"] for unit in result.included_units],
            ["sub-001", "sub-002"],
        )
        self.assertEqual(result.counts["occasions"], 0)
        self.assertEqual(result.counts["tasks"], 0)
        self.assertEqual(result.counts["runs"], 0)

    def test_irregular_longitudinal_rows_are_not_cartesian_expanded(self) -> None:
        text = (
            "subject_id\tsession_id\trun_id\tvisit_index\tadapter_note\n"
            "sub-001\tses-early\trun-01\t1\tone\n"
            "sub-001\tses-late\trun-02\t3\ttwo\n"
            "sub-002\tses-early\trun-01\t1\tthree\n"
            "sub-002\tses-early\trun-02\t1\tfour\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(
                    key_columns=["subject_id", "session_id", "run_id"],
                    occasion_column="session_id",
                    occasion_order_column="visit_index",
                ),
                _write_table(Path(tmp_dir), text),
            )

        self.assertTrue(result.valid, result.errors)
        resolved = [
            (row["values"]["subject_id"], row["values"]["session_id"], row["values"]["run_id"])
            for row in result.included_units
        ]
        self.assertEqual(
            resolved,
            [
                ("sub-001", "ses-early", "run-01"),
                ("sub-001", "ses-late", "run-02"),
                ("sub-002", "ses-early", "run-01"),
                ("sub-002", "ses-early", "run-02"),
            ],
        )
        self.assertEqual(result.counts["source_units"], 4)
        self.assertEqual(result.counts["included_units"], 4)
        self.assertEqual(result.included_units[1]["values"]["adapter_note"], "two")

    def test_include_filters_are_or_within_and_across_columns_then_exclusions_apply(self) -> None:
        text = (
            "subject_id\tcohort_id\teligible\tqc_status\texclusion_reason\n"
            "sub-001\tgroup-a\ttrue\tpass\t\n"
            "sub-002\tgroup-b\ttrue\tfail\tquality flag\n"
            "sub-003\tgroup-c\ttrue\tpass\t\n"
            "sub-004\tgroup-a\tfalse\tpass\t\n"
        )
        cohort = {
            "batch": "units",
            "include": {"cohort_id": ["group-a", "group-b"], "eligible": True},
            "exclude": [
                {
                    "id": "qc-failed",
                    "filters": {"qc_status": "fail"},
                    "reason_field": "exclusion_reason",
                },
                {
                    "id": "never-matched",
                    "filters": {"qc_status": "pending"},
                    "reason": "Pending review.",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(selection={"cohort": "example-cohort"}),
                _write_table(Path(tmp_dir), text),
                cohorts=_cohorts(cohort),
            )

        self.assertTrue(result.valid, result.errors)
        self.assertEqual([row["values"]["subject_id"] for row in result.included_units], ["sub-001"])
        self.assertEqual(result.excluded_units[0]["exclusion_ids"], ["qc-failed"])
        self.assertEqual(result.excluded_units[0]["reasons"], ["quality flag"])
        self.assertEqual(
            [row["values"]["subject_id"] for row in result.not_included_units],
            ["sub-003", "sub-004"],
        )
        self.assertEqual(result.unmatched_exclusion_rules, ("never-matched",))
        self.assertTrue(result.warnings)

    def test_unknown_include_and_exclusion_columns_are_errors(self) -> None:
        cohort = {
            "batch": "units",
            "include": {"unknown_include": "value"},
            "exclude": [
                {"id": "unknown-rule", "filters": {"unknown_exclude": "value"}}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(selection={"cohort": "example-cohort"}),
                _write_table(Path(tmp_dir), "subject_id\nsub-001\n"),
                cohorts=_cohorts(cohort),
            )

        self.assertFalse(result.valid)
        self.assertTrue(any("unknown_include" in error for error in result.errors))
        self.assertTrue(any("unknown_exclude" in error for error in result.errors))

    def test_empty_reason_field_falls_back_to_stable_exclusion_reason(self) -> None:
        cohort = {
            "batch": "units",
            "include": {},
            "exclude": [
                {
                    "id": "not-eligible",
                    "filters": {"eligible": "false"},
                    "reason_field": "exclusion_reason",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(selection={"cohort": "example-cohort"}),
                _write_table(
                    Path(tmp_dir),
                    "subject_id\teligible\texclusion_reason\nsub-001\tfalse\t\n",
                ),
                cohorts=_cohorts(cohort),
            )

        self.assertFalse(result.valid)
        self.assertTrue(any("resolved zero included units" in error for error in result.errors))
        self.assertEqual(result.excluded_units[0]["exclusion_ids"], ["not-eligible"])
        self.assertEqual(
            result.excluded_units[0]["reasons"],
            ["Excluded by cohort rule not-eligible; configured reason field exclusion_reason was empty."],
        )

    def test_duplicate_keys_are_rejected_across_full_source_before_filters(self) -> None:
        cohort = {
            "batch": "units",
            "include": {"eligible": "true"},
            "exclude": [],
        }
        text = (
            "subject_id\tsession_id\teligible\n"
            "sub-001\tses-01\ttrue\n"
            "sub-001\tses-01\tfalse\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(
                    selection={"cohort": "example-cohort"},
                    key_columns=["subject_id", "session_id"],
                ),
                _write_table(Path(tmp_dir), text),
                cohorts=_cohorts(cohort),
            )

        self.assertFalse(result.valid)
        self.assertTrue(any("Duplicate analysis-unit key" in error for error in result.errors))

    def test_whitespace_equivalent_unit_keys_are_duplicate_identities(self) -> None:
        text = (
            "subject_id\tsession_id\n"
            "sub-001 \tses-01\n"
            " sub-001\tses-01 \n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(key_columns=["subject_id", "session_id"]),
                _write_table(Path(tmp_dir), text),
            )

        self.assertFalse(result.valid)
        self.assertTrue(any("Duplicate analysis-unit key" in error for error in result.errors))

    def test_identity_comparisons_are_consistent_without_rewriting_rows(self) -> None:
        text = (
            "subject_id\tsession_id\ttask_id\trun_id\n"
            " sub-001 \t ses-01 \t exampletask \t run-01 \n"
            "sub-001\tses-02\texampletask\trun-02\n"
            "sub-002\tses-01\texampletask\trun-01\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(
                    key_columns=["subject_id", "session_id", "run_id"],
                    occasion_column="session_id",
                    required_occasions=["ses-01", "ses-02"],
                ),
                _write_table(Path(tmp_dir), text),
            )

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.counts["subjects"], 2)
        self.assertEqual(result.counts["occasions"], 2)
        self.assertEqual(result.counts["tasks"], 1)
        self.assertEqual(result.counts["runs"], 2)
        self.assertEqual(len(result.incomplete_subjects), 1)
        self.assertEqual(result.incomplete_subjects[0]["subject_id"], "sub-002")
        self.assertEqual(
            result.included_units[0]["values"],
            {
                "subject_id": " sub-001 ",
                "session_id": " ses-01 ",
                "task_id": " exampletask ",
                "run_id": " run-01 ",
            },
        )

    def test_missing_subject_and_configured_key_values_are_errors(self) -> None:
        text = "subject_id\tsession_id\n\tses-01\nsub-002\t\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(key_columns=["subject_id", "session_id"]),
                _write_table(Path(tmp_dir), text),
            )

        self.assertFalse(result.valid)
        self.assertTrue(any("missing required subject_id" in error for error in result.errors))
        self.assertTrue(any("session_id" in error and "missing required value" in error for error in result.errors))

    def test_required_occasion_fail_drop_and_allow_are_explicit(self) -> None:
        text = (
            "subject_id\tsession_id\n"
            "sub-001\tses-01\n"
            "sub-001\tses-02\n"
            "sub-002\tses-01\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tables = _write_table(Path(tmp_dir), text)
            common = {
                "key_columns": ["subject_id", "session_id"],
                "occasion_column": "session_id",
                "required_occasions": ["ses-01", "ses-02"],
            }
            failed = _resolve(_bundle(**common, incomplete="fail"), tables)
            dropped = _resolve(_bundle(**common, incomplete="drop"), tables)
            allowed = _resolve(_bundle(**common, incomplete="allow"), tables)

        self.assertFalse(failed.valid)
        self.assertTrue(any("Incomplete longitudinal units" in error for error in failed.errors))
        self.assertEqual(
            next(check for check in failed.checks if check.id == "longitudinal_completeness").status,
            "fail",
        )
        self.assertTrue(dropped.valid, dropped.errors)
        self.assertEqual([row["values"]["subject_id"] for row in dropped.included_units], ["sub-001", "sub-001"])
        self.assertEqual([row["values"]["subject_id"] for row in dropped.dropped_units], ["sub-002"])
        self.assertEqual(dropped.dropped_units[0]["reason_id"], "incomplete_required_occasions")
        drop_check = next(check for check in dropped.checks if check.id == "longitudinal_completeness")
        self.assertEqual(drop_check.status, "warning")
        self.assertTrue(any("removed 1 unit(s)" in message for message in drop_check.messages))
        self.assertTrue(allowed.valid, allowed.errors)
        self.assertEqual(len(allowed.included_units), 3)
        self.assertEqual(allowed.incomplete_subjects[0]["missing_occasions"], ["ses-02"])
        allow_check = next(check for check in allowed.checks if check.id == "longitudinal_completeness")
        self.assertEqual(allow_check.status, "warning")
        self.assertTrue(any("retains 1 unit(s)" in message for message in allow_check.messages))
        self.assertEqual(allowed.warnings, allow_check.messages)

    def test_explicit_visit_order_does_not_reorder_units_or_sort_session_labels(self) -> None:
        text = (
            "subject_id\tsession_id\tvisit_index\n"
            "sub-001\tses-a-late\t2\n"
            "sub-001\tses-z-early\t1\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(
                    key_columns=["subject_id", "session_id"],
                    occasion_column="session_id",
                    occasion_order_column="visit_index",
                ),
                _write_table(Path(tmp_dir), text),
            )

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(
            [row["values"]["session_id"] for row in result.included_units],
            ["ses-a-late", "ses-z-early"],
        )
        self.assertEqual(
            [row["occasion"] for row in result.occasion_order[0]["occasions"]],
            ["ses-z-early", "ses-a-late"],
        )

    def test_occasion_order_column_must_not_reuse_session_identity(self) -> None:
        document = _bundle(
            key_columns=["subject_id", "session_id"],
            occasion_column="session_id",
            occasion_order_column="session_id",
        )

        errors = validate_analysis_bundle_document(document)

        self.assertTrue(any("must be distinct from occasion_column" in error for error in errors))

    def test_nonfinite_visit_order_values_are_stable_text_not_sort_crashes(self) -> None:
        text = (
            "subject_id\tsession_id\tvisit_index\n"
            "sub-001\tses-nan\tNaN\n"
            "sub-001\tses-numeric\t1\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(
                    key_columns=["subject_id", "session_id"],
                    occasion_column="session_id",
                    occasion_order_column="visit_index",
                ),
                _write_table(Path(tmp_dir), text),
            )

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(
            [row["occasion"] for row in result.occasion_order[0]["occasions"]],
            ["ses-numeric", "ses-nan"],
        )

    def test_distinct_occasions_must_not_share_an_explicit_order_value(self) -> None:
        text = (
            "subject_id\tsession_id\tvisit_index\n"
            "sub-001\tses-first\t1\n"
            "sub-001\tses-second\t1.0\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(
                    key_columns=["subject_id", "session_id"],
                    occasion_column="session_id",
                    occasion_order_column="visit_index",
                ),
                _write_table(Path(tmp_dir), text),
            )

        self.assertFalse(result.valid)
        self.assertTrue(
            any("distinct occasions" in error and "same 'visit_index' value" in error for error in result.errors)
        )

    def test_stored_bids_identifiers_are_not_rewritten(self) -> None:
        text = "subject_id\tsession_id\trun_id\nsub-001\tses-02\trun-03\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(key_columns=["subject_id", "session_id", "run_id"]),
                _write_table(Path(tmp_dir), text),
            )

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.included_units[0]["values"], {
            "subject_id": "sub-001",
            "session_id": "ses-02",
            "run_id": "run-03",
        })

    def test_selection_is_xor_and_inline_selectors_are_rejected(self) -> None:
        mixed = _bundle(selection={"cohort": "example-cohort", "batch": "units"})
        inline = _bundle(selection={"batch": "units", "subjects": ["sub-001"]})
        sibling_inline = {**_bundle(), "subjects": ["sub-001"]}
        sibling_execution = {**_bundle(), "command": "run-analysis"}

        self.assertTrue(any("exactly one" in error for error in validate_analysis_bundle_document(mixed)))
        errors = validate_analysis_bundle_document(inline)
        self.assertTrue(any("inline subject" in error for error in errors))
        self.assertTrue(
            any("sibling inline subject" in error for error in validate_analysis_bundle_document(sibling_inline))
        )
        self.assertTrue(
            any("sibling execution" in error for error in validate_analysis_bundle_document(sibling_execution))
        )

    def test_bundle_owned_lists_reject_blank_and_non_string_items(self) -> None:
        blank_key = _bundle(key_columns=["subject_id", " "])
        null_occasion = _bundle(
            occasion_column="session_id",
            required_occasions=["ses-01", None],  # type: ignore[list-item]
        )
        blank_stage = _bundle(stages=["roi_build", " "])

        self.assertTrue(
            any(
                "key_columns[1] must be a non-empty string" in error
                for error in validate_analysis_bundle_document(blank_key)
            )
        )
        self.assertTrue(
            any(
                "required_occasions[1] must be a string" in error
                for error in validate_analysis_bundle_document(null_occasion)
            )
        )
        self.assertTrue(
            any(
                "stages[1] must be a non-empty string" in error
                for error in validate_analysis_bundle_document(blank_stage)
            )
        )

    def test_cohort_filter_alternatives_do_not_silently_drop_blanks(self) -> None:
        cohort = _cohorts(
            {
                "batch": "units",
                "include": {"qc_status": ["pass", " "]},
                "exclude": [],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(selection={"cohort": "example-cohort"}),
                _write_table(Path(tmp_dir), "subject_id\tqc_status\nsub-001\tpass\n"),
                cohorts=cohort,
            )

        self.assertFalse(result.valid)
        self.assertTrue(any("cohorts.example-cohort.include.qc_status[1]" in error for error in result.errors))

    def test_cohort_filters_reject_mappings_and_nested_collections(self) -> None:
        invalid_values = (
            {"not": "pass"},
            ["pass", ["review"]],
        )
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value), tempfile.TemporaryDirectory() as tmp_dir:
                result = _resolve(
                    _bundle(selection={"cohort": "example-cohort"}),
                    _write_table(Path(tmp_dir), "subject_id\tqc_status\nsub-001\tpass\n"),
                    cohorts=_cohorts(
                        {
                            "batch": "units",
                            "include": {"qc_status": invalid_value},
                            "exclude": [],
                        }
                    ),
                )

            self.assertFalse(result.valid)
            self.assertTrue(
                any("scalar string, boolean, or number" in error for error in result.errors)
            )

    def test_effective_empty_filter_or_exclusion_is_not_ready_but_remains_auditable(self) -> None:
        cases = (
            (
                {"batch": "units", "include": {"qc_status": "review"}, "exclude": []},
                "not_included_units",
            ),
            (
                {
                    "batch": "units",
                    "include": {},
                    "exclude": [{"id": "omit-pass", "filters": {"qc_status": "pass"}}],
                },
                "excluded_units",
            ),
        )
        for cohort, audit_field in cases:
            with self.subTest(audit_field=audit_field), tempfile.TemporaryDirectory() as tmp_dir:
                result = _resolve(
                    _bundle(selection={"cohort": "example-cohort"}),
                    _write_table(Path(tmp_dir), "subject_id\tqc_status\nsub-001\tpass\n"),
                    cohorts=_cohorts(cohort),
                )

            self.assertFalse(result.ready_for_planning)
            self.assertTrue(any("resolved zero included units" in error for error in result.errors))
            self.assertEqual(len(getattr(result, audit_field)), 1)

    def test_drop_policy_that_removes_every_unit_is_not_ready_and_reports_drops(self) -> None:
        text = "subject_id\tsession_id\nsub-001\tses-01\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                _bundle(
                    key_columns=["subject_id", "session_id"],
                    occasion_column="session_id",
                    required_occasions=["ses-01", "ses-02"],
                    incomplete="drop",
                ),
                _write_table(Path(tmp_dir), text),
            )

        self.assertFalse(result.ready_for_planning)
        self.assertTrue(any("resolved zero included units" in error for error in result.errors))
        self.assertEqual(len(result.dropped_units), 1)
        self.assertEqual(result.counts["included_units"], 0)
        self.assertTrue(any("removed 1 unit(s)" in warning for warning in result.warnings))

    def test_empty_scaffold_is_schema_valid_but_not_ready_for_planning(self) -> None:
        scaffold = _bundle(components={}, stages=[])
        self.assertEqual(validate_analysis_bundle_document(scaffold), [])
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(scaffold, _write_table(Path(tmp_dir), "subject_id\nsub-001\n"), components={})

        self.assertFalse(result.ready_for_planning)
        self.assertTrue(any("at least one stage" in error for error in result.errors))

    def test_stage_components_and_order_are_contextually_checked(self) -> None:
        document = _bundle(
            components={"roi_set": "example-rois", "extraction_set": "example-values"},
            stages=["roi_extraction", "roi_build"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _resolve(
                document,
                _write_table(Path(tmp_dir), "subject_id\nsub-001\n"),
                components={"roi_set": ("example-rois",), "extraction_set": ()},
            )

        self.assertFalse(result.valid)
        self.assertTrue(any("must precede" in error for error in result.errors))
        self.assertTrue(any("example-values" in error for error in result.errors))

    def test_digests_are_stable_host_independent_and_cohort_sensitive(self) -> None:
        document = _bundle(selection={"cohort": "example-cohort"})
        cohort = _cohorts()
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = _resolve(document, _write_table(Path(first_dir), "subject_id\nsub-001\n"), cohorts=cohort)
            second = _resolve(document, _write_table(Path(second_dir), "subject_id\nsub-001\n"), cohorts=cohort)
            changed = _resolve(
                document,
                _write_table(Path(second_dir), "subject_id\nsub-001\n", name="changed"),
                cohorts=_cohorts({"batch": "changed", "include": {}, "exclude": []}),
            )

        self.assertEqual(first.source_batch_sha256, second.source_batch_sha256)
        self.assertEqual(first.effective_config_sha256, second.effective_config_sha256)
        self.assertEqual(first.plan_digest, second.plan_digest)
        self.assertNotEqual(first.effective_config_sha256, changed.effective_config_sha256)
        self.assertNotEqual(first.plan_digest, changed.plan_digest)
        self.assertEqual(len(first.plan_digest), 64)

    def test_manifest_writer_preserves_bids_header_and_arbitrary_metadata(self) -> None:
        rows = [
            {"subject_id": "sub-001", "qc_status": "pass", "visit_index": "1"},
            {"subject_id": "sub-002", "direction": "AP", "qc_status": "review"},
        ]
        preferred = ("subject_id", "session_id", "task_id", "run_id")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "batch.tsv"
            write_manifest_table(path, rows, preferred_columns=preferred)
            raw = path.read_bytes()
            table = read_manifest_table(path)

        self.assertEqual(
            table.columns,
            ("subject_id", "session_id", "task_id", "run_id", "qc_status", "visit_index", "direction"),
        )
        self.assertEqual(table.rows[0]["qc_status"], "pass")
        self.assertEqual(table.rows[1]["direction"], "AP")
        self.assertEqual(table.source_sha256, sha256(raw).hexdigest())
        self.assertNotIn(b"\r\n", raw)

    def test_manifest_reader_and_writer_retain_tabular_contract_and_literal_values(self) -> None:
        rows = [{"feature_table": "${DATA_ROOT}/toy.tsv", "target_column": "binary_target"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tabular.tsv"
            write_manifest_table(path, rows, columns=("feature_table", "target_column"))
            table = read_manifest_table(path)

        self.assertEqual(table.columns, ("feature_table", "target_column"))
        self.assertEqual(table.rows, tuple(rows))


if __name__ == "__main__":
    unittest.main()
