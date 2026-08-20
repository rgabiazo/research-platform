from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import nibabel as nib
    import numpy as np
except ImportError:  # pragma: no cover - local minimal environments may skip.
    nib = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
BIDS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-bids"
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(BIDS_PACKAGE_ROOT / "src"))

from research_platform.neuro.fsl.featquery import (
    build_featquery_command_plan,
    parse_featquery_report_text,
)
from research_platform.neuro import _roi_runtime_outputs as runtime_outputs
from research_platform.neuro._roi_runtime_outputs import RoiRuntimeOutputError
from research_platform.bids.roi import build_loso_flame1_mask_path, build_roi_sidecar_path
from research_platform.neuro.roi import validate_extraction_set_document
from research_platform.neuro.roi_cleanup import cleanup_after_loso_featquery_extraction, cleanup_after_loso_roi_build
from research_platform.neuro.roi_execution import (
    RoiExecutionContext,
    _qc_summary_table_path,
    _write_extraction_summary_tables,
    plan_roi_build,
    plan_roi_extraction,
    run_roi_extraction,
)
from research_platform.neuro.roi_publication import (
    _column_description,
    contrast_alias_for,
    publish_loso_featquery_extraction,
    publish_loso_roi_build_result,
)


def fake_find_featquery(tool: str, **_: object) -> str | None:
    if tool == "featquery":
        return "/mock/fsl/bin/featquery"
    return None


class FslFeatqueryPhase5Tests(unittest.TestCase):
    def test_featquery_mean_cope_command_plan_is_shell_safe_and_does_not_require_fsl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan = build_featquery_command_plan(
                feat_dir=root / "sub-001.feat",
                roi_mask_path=root / "roi masks" / "seed.nii.gz",
                output_name="fq_SeedA_CondA",
                value_image="stats/cope1",
                metrics=["mean_cope"],
            )

        self.assertEqual(plan.command[0], "featquery")
        self.assertEqual(
            plan.command,
            (
                "featquery",
                "1",
                str((root / "sub-001.feat").resolve()),
                "1",
                "stats/cope1",
                "fq_SeedA_CondA",
                str((root / "roi masks" / "seed.nii.gz").resolve()),
            ),
        )
        self.assertNotIn("-p", plan.command)
        self.assertFalse(plan.include_percent_signal_change)

    def test_featquery_percent_signal_change_command_plan_adds_p_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan = build_featquery_command_plan(
                feat_dir=root / "sub-001.feat",
                roi_mask_path=root / "roi masks" / "seed.nii.gz",
                output_name="fq_SeedA_CondA",
                value_image="stats/cope1",
                metrics=["percent_signal_change", "roi_voxel_count"],
            )

        self.assertIn("-p", plan.command)
        self.assertEqual(plan.command[-2:], ("-p", str((root / "roi masks" / "seed.nii.gz").resolve())))
        self.assertTrue(plan.include_percent_signal_change)

    def test_featquery_command_plan_rejects_mixed_raw_cope_and_psc(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate extraction targets"):
            build_featquery_command_plan(
                feat_dir="example.feat",
                roi_mask_path="seed.nii.gz",
                output_name="fq_SeedA_CondA",
                metrics=["mean_cope", "percent_signal_change"],
            )

    def test_featquery_command_rejects_unsafe_output_name(self) -> None:
        with self.assertRaises(ValueError):
            build_featquery_command_plan(
                feat_dir="example.feat",
                roi_mask_path="seed.nii.gz",
                output_name="../bad",
            )

    def test_publication_contrast_aliases_are_generic_and_configurable(self) -> None:
        self.assertEqual(
            contrast_alias_for(
                "pair_enc_hit_gt_item_enc_hit",
                contrast_desc="PairEncHitGtItemEncHit",
                settings={},
            ),
            "PairEncHitGtItemEncHit",
        )
        self.assertEqual(
            contrast_alias_for(
                "pair_enc_hit_gt_item_enc_hit",
                contrast_desc="PairEncHitGtItemEncHit",
                settings={"contrast_aliases": {"pair_enc_hit_gt_item_enc_hit": "EncPairGtItem"}},
            ),
            "EncPairGtItem",
        )

    def test_report_parser_extracts_percent_signal_change_and_cope_mean(self) -> None:
        report = parse_featquery_report_text(
            "Mean % signal change:\t1.25\nMean cope: 2.5\nVoxels = 42\n",
            required_metrics=["percent_signal_change", "mean_cope", "roi_voxel_count"],
        )

        self.assertTrue(report.usable)
        self.assertEqual(report.mean_psc, 1.25)
        self.assertEqual(report.mean_cope, 2.5)
        self.assertEqual(report.roi_voxel_count, 42)

    def test_report_parser_extracts_non_percent_featquery_main_row(self) -> None:
        report = parse_featquery_report_text(
            "1 stats/cope1 117 -18.97 0 31.71 32.88 65.28 88.15 24.42 44 34 69 -2.0 -58.0 66.0\n",
            required_metrics=["mean_cope", "roi_voxel_count"],
        )

        self.assertTrue(report.usable)
        self.assertEqual(report.stats_image, "stats/cope1")
        self.assertEqual(report.roi_voxel_count, 117)
        self.assertEqual(report.mean_cope, 31.71)
        self.assertEqual(report.median_cope, 32.88)
        self.assertEqual(report.max_cope, 88.15)
        self.assertEqual(report.max_voxel_coordinate, (44, 34, 69))
        self.assertEqual(report.max_mm_coordinate, (-2.0, -58.0, 66.0))
        self.assertIsNone(report.mean_psc)

    def test_report_parser_does_not_synthesize_psc_for_non_percent_row(self) -> None:
        report = parse_featquery_report_text(
            "1 stats/cope1 117 -18.97 0 31.71 32.88 65.28 88.15 24.42 44 34 69 -2.0 -58.0 66.0\n",
            required_metrics=["percent_signal_change", "mean_cope", "roi_voxel_count"],
        )

        self.assertIsNone(report.mean_psc)
        self.assertEqual(report.mean_cope, 31.71)
        self.assertIn("missing_report_values", report.qc_flags)
        self.assertTrue(any("mean_psc" in warning for warning in report.warnings))

    def test_report_parser_maps_percent_signal_change_result_row_to_mean_psc(self) -> None:
        report = parse_featquery_report_text(
            "1 stats/cope1 117 -18.97 0 1.25 1.10 2.5 3.0 0.9 44 34 69 -2.0 -58.0 66.0\n",
            required_metrics=["percent_signal_change", "roi_voxel_count"],
        )

        self.assertTrue(report.usable)
        self.assertEqual(report.stats_image, "stats/cope1")
        self.assertEqual(report.mean_psc, 1.25)
        self.assertEqual(report.median_psc, 1.10)
        self.assertIsNone(report.mean_cope)

    def test_report_parser_tolerates_whitespace_variants(self) -> None:
        report = parse_featquery_report_text(
            "mean percent signal change    -0.125\nmedian cope\t1.75\nmean value    2.25\nnvoxels: 7\n",
            required_metrics=["percent_signal_change", "mean_cope"],
        )

        self.assertEqual(report.mean_psc, -0.125)
        self.assertEqual(report.mean_cope, 2.25)
        self.assertEqual(report.median_cope, 1.75)
        self.assertEqual(report.roi_voxel_count, 7)

    def test_malformed_report_marks_qc_without_inventing_values(self) -> None:
        report = parse_featquery_report_text(
            "Mean % signal change: 1.0 2.0\nunrelated text\n",
            required_metrics=["percent_signal_change"],
        )

        self.assertFalse(report.usable)
        self.assertIsNone(report.mean_psc)
        self.assertIn("ambiguous_report_values", report.qc_flags)
        self.assertIn("missing_report_values", report.qc_flags)

    def test_fsl_featquery_extraction_config_validation(self) -> None:
        document = self._explicit_extraction_document()

        self.assertEqual(validate_extraction_set_document(document), [])

    def test_summary_writer_splits_analysis_values_and_qc_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            values_path = (
                Path(tmpdir)
                / "group_ses-01_task-memory_desc-ModelAFeatquery_values.tsv"
            )
            _write_extraction_summary_tables([self._summary_writer_row(mean_psc=1.25)], values_path)

            values_header, values_rows = self._read_tsv(values_path)
            qc_header, qc_rows = self._read_tsv(
                values_path.with_name("group_ses-01_task-memory_desc-ModelAFeatquery_qc.tsv")
            )

        forbidden = {
            "feat_dir",
            "roi_mask_path",
            "featquery_output_dir",
            "report_path",
            "backend",
            "featquery_command",
            "usable",
            "qc_flags",
            "warnings",
        }
        self.assertEqual(len(values_rows), 1)
        self.assertEqual(len(qc_rows), 1)
        self.assertFalse(forbidden & set(values_header))
        self.assertTrue(forbidden <= set(qc_header))
        self.assertIn("included_in_values", qc_header)
        self.assertIn("exclude_reason", qc_header)

    def test_summary_writer_omits_all_empty_mean_psc_from_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            values_path = Path(tmpdir) / "group_ses-01_task-memory_desc-ModelAFeatquery_values.tsv"
            _write_extraction_summary_tables(
                [
                    self._summary_writer_row(subject_id="sub-002", mean_psc=None),
                    self._summary_writer_row(subject_id="sub-003", mean_psc=""),
                ],
                values_path,
            )

            values_header, _values_rows = self._read_tsv(values_path)

        self.assertNotIn("mean_psc", values_header)
        self.assertIn("mean_cope", values_header)

    def test_summary_writer_keeps_mean_psc_when_any_value_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            values_path = Path(tmpdir) / "group_ses-01_task-memory_desc-ModelAFeatquery_values.tsv"
            _write_extraction_summary_tables(
                [
                    self._summary_writer_row(subject_id="sub-002", mean_psc=None),
                    self._summary_writer_row(subject_id="sub-003", mean_psc=0.5),
                ],
                values_path,
            )

            values_header, values_rows = self._read_tsv(values_path)

        self.assertIn("mean_psc", values_header)
        self.assertEqual(values_rows[1]["mean_psc"], "0.5")

    def test_publication_column_metadata_labels_mean_psc_as_percent_signal_change(self) -> None:
        mean_psc = _column_description("mean_psc")
        mean_cope = _column_description("mean_cope")

        self.assertEqual(mean_psc["Units"], "percent signal change")
        self.assertIn("percent signal change", mean_psc["Description"])
        self.assertNotEqual(mean_psc["Units"], "millimeters")
        self.assertEqual(mean_cope["Units"], "arbitrary")
        self.assertIn("Raw mean COPE", mean_cope["Description"])

    def test_summary_writer_normalizes_subject_and_session_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            values_path = Path(tmpdir) / "group_ses-01_task-memory_desc-ModelAFeatquery_values.tsv"
            _write_extraction_summary_tables(
                [
                    self._summary_writer_row(subject_id="sub-002", session_id="ses-01"),
                    self._summary_writer_row(subject_id="002", session_id="01"),
                ],
                values_path,
            )

            _values_header, values_rows = self._read_tsv(values_path)
            _qc_header, qc_rows = self._read_tsv(
                values_path.with_name("group_ses-01_task-memory_desc-ModelAFeatquery_qc.tsv")
            )

        self.assertEqual([row["subject_id"] for row in values_rows], ["002", "002"])
        self.assertEqual([row["session_id"] for row in values_rows], ["01", "01"])
        self.assertEqual([row["subject_id"] for row in qc_rows], ["002", "002"])
        self.assertEqual([row["session_id"] for row in qc_rows], ["01", "01"])

    def test_summary_writer_uses_values_stem_for_qc_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            values_path = Path(tmpdir) / "group_ses-01_task-memory_desc-ModelAFeatquery_values.tsv"
            written = _write_extraction_summary_tables([self._summary_writer_row()], values_path)

        self.assertEqual(written[0].name, "group_ses-01_task-memory_desc-ModelAFeatquery_values.tsv")
        self.assertEqual(written[1].name, "group_ses-01_task-memory_desc-ModelAFeatquery_qc.tsv")

    def test_summary_writer_excludes_unusable_rows_from_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            values_path = Path(tmpdir) / "group_ses-01_task-memory_desc-ModelAFeatquery_values.tsv"
            _write_extraction_summary_tables(
                [
                    self._summary_writer_row(subject_id="sub-002", usable=True, qc_flags="pass"),
                    self._summary_writer_row(
                        subject_id="sub-003",
                        usable=False,
                        qc_flags="missing_report_values",
                        warnings="Missing requested featquery report value(s): mean_cope.",
                    ),
                ],
                values_path,
            )

            _values_header, values_rows = self._read_tsv(values_path)
            _qc_header, qc_rows = self._read_tsv(
                values_path.with_name("group_ses-01_task-memory_desc-ModelAFeatquery_qc.tsv")
            )

        self.assertEqual([row["subject_id"] for row in values_rows], ["002"])
        self.assertEqual([row["included_in_values"] for row in qc_rows], ["true", "false"])
        self.assertIn("missing_report_values", qc_rows[1]["exclude_reason"])

    def test_plan_only_featquery_extraction_from_explicit_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)

            plan = plan_roi_extraction(self._explicit_extraction_document(), context=context)

        action = plan.actions[0]
        self.assertFalse(plan.executed)
        self.assertEqual(action.backend, "fsl_featquery")
        self.assertIn("roi_extract/featquery_values/group/ses-01", str(action.table_path))
        self.assertEqual(action.metadata["missing_inputs"], [])
        self.assertEqual(action.metadata["command"][0], "featquery")
        self.assertFalse(action.table_path.exists())
        self.assertTrue(str(project_root) in str(action.mask_path))

    def test_plan_percent_signal_change_featquery_extraction_uses_p_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)

            plan = plan_roi_extraction(
                self._explicit_extraction_document(metrics=["percent_signal_change", "roi_voxel_count"], include_psc=True),
                context=context,
            )

        action = plan.actions[0]
        self.assertEqual(action.metrics, ("percent_signal_change", "roi_voxel_count"))
        self.assertIn("-p", action.metadata["command"])
        self.assertEqual(action.metadata["command"][-2], "-p")
        self.assertTrue(action.metadata["include_percent_signal_change"])

    def test_plan_mixed_raw_cope_and_percent_signal_change_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _root, _project_root, context = self._workspace_context(tmpdir)

            with self.assertRaisesRegex(ValueError, "separate extraction targets"):
                plan_roi_extraction(
                    self._explicit_extraction_document(metrics=["mean_cope", "percent_signal_change", "roi_voxel_count"]),
                    context=context,
                )

    def test_plan_reports_missing_feat_dir_and_roi_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _root, _project_root, context = self._workspace_context(tmpdir)

            plan = plan_roi_extraction(self._explicit_extraction_document(), context=context)

        action = plan.actions[0]
        self.assertEqual(action.metadata["missing_inputs"], ["feat_dir", "roi_mask"])
        self.assertTrue(any("FEAT directory is missing" in warning for warning in action.metadata["warnings"]))
        self.assertTrue(any("ROI mask is missing" in warning for warning in action.metadata["warnings"]))

    def test_roi_masks_can_be_discovered_from_roi_set_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            roi_document = self._coordinate_roi_document()
            build_plan = plan_roi_build(roi_document, context=context)
            mask_path = build_plan.actions[0].mask_path
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.write_text("mask", encoding="utf-8")

            plan = plan_roi_extraction(
                self._roi_set_ref_extraction_document(),
                roi_set_document=roi_document,
                context=context,
            )

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].mask_path, mask_path)
        self.assertEqual(plan.actions[0].metadata["missing_inputs"], [])

    def test_old_fsl_roi_set_config_without_mask_source_uses_runtime_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_loso_fixed_effects_inputs(root)
            roi_document = self._published_loso_roi_document()
            extraction_document = self._published_mask_extraction_document()
            extraction_document["extraction_set"].pop("roi_mask_source")  # type: ignore[index]

            plan = plan_roi_extraction(
                extraction_document,
                roi_set_document=roi_document,
                context=context,
            )
            runtime_mask = plan_roi_build(roi_document, context=context).actions[0].mask_path

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].mask_path, runtime_mask)
        self.assertIn(".research-platform/roi-loso-flame1-runtime/rois/loso_modelA", str(plan.actions[0].mask_path))

    def test_explicit_roi_masks_override_published_mask_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            explicit_mask = self._write_explicit_mask(root)
            document = self._explicit_extraction_document()
            document["extraction_set"]["roi_mask_source"] = {"source": "roi_set_publication"}  # type: ignore[index]

            plan = plan_roi_extraction(document, context=context)

        self.assertEqual(plan.actions[0].mask_path, explicit_mask.resolve())

    def test_published_loso_masks_are_resolved_from_roi_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_loso_fixed_effects_inputs(root)
            roi_document = self._published_loso_roi_document()
            expected_mask = self._write_published_loso_mask(root)

            plan = plan_roi_extraction(
                self._published_mask_extraction_document(),
                roi_set_document=roi_document,
                context=context,
            )

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].mask_path, expected_mask.resolve())
        self.assertIn("/roi-loso-flame1/masks/", str(plan.actions[0].mask_path))
        self.assertEqual(plan.actions[0].metadata["missing_inputs"], [])
        self.assertEqual(plan.actions[0].metadata["roi_sidecar"]["voxel_count"], 19)
        self.assertEqual(plan.actions[0].metadata["roi_sidecar"]["fallback_status"], "thresholded")

    def test_published_loso_mask_execution_uses_normalized_sidecar_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_loso_fixed_effects_inputs(root)
            roi_document = self._published_loso_roi_document()
            self._write_published_loso_mask(root)

            def fake_execute(command_plan: object) -> Path:
                command_plan.output_dir.mkdir(parents=True, exist_ok=True)
                command_plan.report_path.write_text("Mean cope: 4.5\n", encoding="utf-8")
                return command_plan.report_path

            with (
                mock.patch(
                    "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_featquery,
                ) as find_fsl_tool,
            ):
                plan = run_roi_extraction(
                    self._published_mask_extraction_document(),
                    roi_set_document=roi_document,
                    context=context,
                )
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

            _values_header, values_rows = self._read_tsv(plan.tables[0])

        self.assertEqual(values_rows[0]["mean_cope"], "4.5")
        self.assertEqual(values_rows[0]["roi_voxel_count"], "19")
        self.assertEqual(values_rows[0]["thresholded_peak"], "True")
        self.assertEqual(values_rows[0]["peak_x_mm"], "1")
        self.assertIn("/roi-loso-flame1/masks/", plan.actions[0].metadata["command"][-1])

    def test_duplicate_published_mask_destinations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_loso_fixed_effects_inputs(root, models=("ModelA", "ModelB"))
            roi_document = self._published_loso_roi_document(models=["ModelA", "ModelB"])
            roi_document["roi_set"]["publication"]["mask_desc"] = "SameMask"  # type: ignore[index]

            with self.assertRaisesRegex(ValueError, "duplicate destination path"):
                plan_roi_extraction(
                    self._published_mask_extraction_document(),
                    roi_set_document=roi_document,
                    context=context,
                )

    def test_mocked_execution_writes_summary_table_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)

            def fake_execute(command_plan: object) -> Path:
                command_plan.output_dir.mkdir(parents=True, exist_ok=True)
                command_plan.report_path.write_text(
                    "1 stats/cope1 117 -18.97 0 31.71 32.88 65.28 88.15 24.42 44 34 69 -2.0 -58.0 66.0\n",
                    encoding="utf-8",
                )
                return command_plan.report_path

            with (
                mock.patch(
                    "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_featquery,
                ) as find_fsl_tool,
            ):
                plan = run_roi_extraction(self._explicit_extraction_document(), context=context)
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

            values_path, qc_path = plan.tables
            values_header, values_rows = self._read_tsv(values_path)
            qc_header, qc_rows = self._read_tsv(qc_path)

            self.assertTrue(values_path.exists())
            self.assertTrue(qc_path.exists())
            self.assertEqual(qc_path.name, values_path.name.replace("_values.tsv", "_qc.tsv"))
            self.assertNotIn("backend", values_header)
            self.assertNotIn("featquery_command", values_header)
            self.assertNotIn("usable", values_header)
            self.assertNotIn("qc_flags", values_header)
            self.assertNotIn("warnings", values_header)
            self.assertNotIn("mean_psc", values_header)
            self.assertEqual(values_rows[0]["subject_id"], "001")
            self.assertEqual(values_rows[0]["session_id"], "01")
            self.assertEqual(values_rows[0]["mean_cope"], "31.71")
            self.assertEqual(values_rows[0]["roi_voxel_count"], "117")
            self.assertIn("backend", qc_header)
            self.assertEqual(qc_rows[0]["backend"], "fsl_featquery")
            self.assertEqual(qc_rows[0]["mean_psc"], "")
            self.assertEqual(qc_rows[0]["usable"], "True")
            self.assertEqual(qc_rows[0]["included_in_values"], "true")
            self.assertEqual(qc_rows[0]["qc_flags"], "pass")
            self.assertEqual(qc_rows[0]["warnings"], "")
            self.assertIn("report.txt", qc_rows[0]["report_path"])
            self.assertIn("featquery", qc_rows[0]["featquery_command"])

    def test_percent_signal_change_execution_writes_mean_psc_not_mean_cope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)
            document = self._explicit_extraction_document(metrics=["percent_signal_change", "roi_voxel_count"], include_psc=True)

            def fake_execute(command_plan: object) -> Path:
                command_plan.output_dir.mkdir(parents=True, exist_ok=True)
                command_plan.report_path.write_text(
                    "1 stats/cope1 117 -18.97 0 1.25 1.10 2.5 3.0 0.9 44 34 69 -2.0 -58.0 66.0\n",
                    encoding="utf-8",
                )
                return command_plan.report_path

            with (
                mock.patch(
                    "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_featquery,
                ) as find_fsl_tool,
            ):
                plan = run_roi_extraction(document, context=context)
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

            values_header, values_rows = self._read_tsv(plan.tables[0])
            _qc_header, qc_rows = self._read_tsv(plan.tables[1])

        self.assertIn("mean_psc", values_header)
        self.assertNotIn("mean_cope", values_header)
        self.assertEqual(values_rows[0]["mean_psc"], "1.25")
        self.assertEqual(qc_rows[0]["mean_psc"], "1.25")
        self.assertEqual(qc_rows[0]["mean_cope"], "")
        self.assertIn("-p", plan.actions[0].metadata["command"])

    def test_featquery_execution_publishes_bidslike_tables_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)
            document = self._explicit_extraction_document()
            extraction_set = document["extraction_set"]  # type: ignore[index]
            extraction_set["direction"] = "AP"  # type: ignore[index]
            extraction_set["resolution"] = "2"  # type: ignore[index]
            extraction_set["publication"] = {  # type: ignore[index]
                "enabled": True,
                "layout": "loso_flame1_bidslike",
                "root": {"root_ref": "artifacts_root", "path": "roi-loso-flame1"},
                "table_desc": "{model}LOSOFlame1Featquery",
                "existing_output": "replace",
            }
            extraction_set["runtime"] = {"existing_output": "replace"}  # type: ignore[index]

            def fake_execute(command_plan: object) -> Path:
                command_plan.output_dir.mkdir(parents=True, exist_ok=True)
                command_plan.report_path.write_text(
                    "1 stats/cope1 117 -18.97 0 31.71 32.88 65.28 88.15 24.42 44 34 69 -2.0 -58.0 66.0\n",
                    encoding="utf-8",
                )
                return command_plan.report_path

            with (
                mock.patch(
                    "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_featquery,
                ) as find_fsl_tool,
            ):
                run_roi_extraction(document, context=context)
                run_roi_extraction(document, context=context)
            self.assertEqual(
                [call.args[0] for call in find_fsl_tool.call_args_list],
                ["featquery", "featquery"],
            )

            published_root = root / "artifacts" / "roi-loso-flame1"
            values_path = (
                published_root
                / "tables"
                / "group"
                / "ses-01"
                / "func"
                / "ses-01_task-memory_dir-AP_desc-ModelALOSOFlame1Featquery_roistats.tsv"
            )
            qc_path = values_path.with_name("ses-01_task-memory_dir-AP_desc-ModelALOSOFlame1FeatqueryQC_roistats.tsv")
            values_exists = values_path.exists()
            qc_exists = qc_path.exists()
            values_json = json.loads(values_path.with_suffix(".json").read_text(encoding="utf-8"))
            qc_json = json.loads(qc_path.with_suffix(".json").read_text(encoding="utf-8"))
            dataset = json.loads((published_root / "dataset_description.json").read_text(encoding="utf-8"))

        self.assertTrue(values_exists)
        self.assertTrue(qc_exists)
        self.assertEqual(dataset["ContrastAliases"], {"CondA": "CondA"})
        self.assertIn("Columns", values_json)
        self.assertIn("mean_cope", values_json["Columns"])
        self.assertEqual(values_json["ROIExtraction"]["Backend"], "fsl_featquery")
        self.assertEqual(qc_json["Description"], "QC/audit ROI extraction table.")

    def test_featquery_execution_cleanup_removes_runtime_tables_after_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)
            document = self._explicit_extraction_document()
            extraction_set = document["extraction_set"]  # type: ignore[index]
            extraction_set["direction"] = "AP"  # type: ignore[index]
            extraction_set["outputs"] = {"root_ref": "artifacts_root", "path": "roi-runtime", "format": "tsv"}  # type: ignore[index]
            extraction_set["runtime"] = {"cleanup": {"after_extraction": "extraction_runtime"}}  # type: ignore[index]
            extraction_set["publication"] = {  # type: ignore[index]
                "enabled": True,
                "layout": "loso_flame1_bidslike",
                "root": {"root_ref": "artifacts_root", "path": "roi-loso-flame1"},
                "table_desc": "{model}LOSOFlame1Featquery",
            }

            def fake_execute(command_plan: object) -> Path:
                command_plan.output_dir.mkdir(parents=True, exist_ok=True)
                command_plan.report_path.write_text("Mean cope: 4.5\nVoxels = 19\n", encoding="utf-8")
                return command_plan.report_path

            with (
                mock.patch(
                    "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_featquery,
                ) as find_fsl_tool,
            ):
                plan = run_roi_extraction(document, context=context)
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

            runtime_table_root = root / "artifacts" / "roi-runtime" / "roi_extract" / "featquery_values"
            published_root = root / "artifacts" / "roi-loso-flame1"
            published_tables = list((published_root / "tables").rglob("*_roistats.tsv"))

        self.assertFalse(runtime_table_root.exists())
        self.assertTrue(published_tables)
        self.assertEqual(plan.cleanup[0]["scope"], "extraction_set")
        self.assertEqual(plan.cleanup[0]["policy"], "extraction_runtime")
        self.assertEqual(plan.cleanup[0]["targets"][0]["status"], "removed")

    def test_cleanup_cache_only_removes_only_current_loso_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            runtime_root = root / "artifacts" / "roi-runtime"
            current_cache = runtime_root / ".cache" / "loso_groupmaps" / "loso_modelA"
            current_groupmaps = runtime_root / "loso_groupmaps" / "loso_modelA"
            current_rois = runtime_root / "rois" / "loso_modelA"
            sibling_cache = runtime_root / ".cache" / "loso_groupmaps" / "other_rois"
            for path in (current_cache, current_groupmaps, current_rois, sibling_cache):
                path.mkdir(parents=True)
                (path / "keep.txt").write_text("x", encoding="utf-8")
            document = self._cleanup_roi_document(
                "loso_modelA",
                outputs={"root_ref": "artifacts_root", "path": "roi-runtime"},
                cleanup={"after_roi_build": "cache_only"},
            )

            cleanup = cleanup_after_loso_roi_build(
                document,
                context=context,
                publication_complete=True,
                publication_root=root / "artifacts" / "roi-loso-flame1",
            )
            current_cache_exists = current_cache.exists()
            current_groupmaps_exists = current_groupmaps.exists()
            current_rois_exists = current_rois.exists()
            sibling_cache_exists = sibling_cache.exists()

        self.assertFalse(current_cache_exists)
        self.assertTrue(current_groupmaps_exists)
        self.assertTrue(current_rois_exists)
        self.assertTrue(sibling_cache_exists)
        self.assertEqual(cleanup[0]["targets"][0]["status"], "removed")

    def test_cleanup_roi_runtime_after_build_preserves_sibling_roi_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            runtime_root = root / "artifacts" / "roi-runtime"
            current_paths = (
                runtime_root / ".cache" / "loso_groupmaps" / "loso_modelA",
                runtime_root / "loso_groupmaps" / "loso_modelA",
                runtime_root / "rois" / "loso_modelA",
            )
            sibling_paths = (
                runtime_root / ".cache" / "loso_groupmaps" / "other_rois",
                runtime_root / "loso_groupmaps" / "other_rois",
                runtime_root / "rois" / "other_rois",
            )
            for path in (*current_paths, *sibling_paths):
                path.mkdir(parents=True)
                (path / "keep.txt").write_text("x", encoding="utf-8")
            document = self._cleanup_roi_document(
                "loso_modelA",
                outputs={"root_ref": "artifacts_root", "path": "roi-runtime"},
                cleanup={"after_roi_build": "roi_runtime"},
            )

            cleanup = cleanup_after_loso_roi_build(
                document,
                context=context,
                publication_complete=True,
                publication_root=root / "artifacts" / "roi-loso-flame1",
            )
            current_exists = tuple(path.exists() for path in current_paths)
            sibling_exists = tuple(path.exists() for path in sibling_paths)

        self.assertTrue(all(not exists for exists in current_exists))
        self.assertTrue(all(sibling_exists))
        self.assertEqual(cleanup[0]["policy"], "roi_runtime")
        self.assertEqual({target["status"] for target in cleanup[0]["targets"]}, {"removed"})

    def test_cleanup_reports_missing_targets_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            document = self._cleanup_roi_document(
                "loso_modelA",
                outputs={"root_ref": "artifacts_root", "path": "roi-runtime"},
                cleanup={"after_roi_build": "roi_runtime"},
            )

            cleanup = cleanup_after_loso_roi_build(
                document,
                context=context,
                publication_complete=True,
                publication_root=root / "artifacts" / "roi-loso-flame1",
            )

        self.assertEqual(cleanup[0]["status"], "skipped")
        self.assertEqual({target["status"] for target in cleanup[0]["targets"]}, {"missing"})

    def test_cleanup_extraction_runtime_removes_only_current_extraction_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            runtime_root = root / "artifacts" / "roi-runtime"
            current = runtime_root / "roi_extract" / "featquery_values"
            sibling = runtime_root / "roi_extract" / "other_values"
            for path in (current, sibling):
                path.mkdir(parents=True)
                (path / "table.tsv").write_text("x", encoding="utf-8")
            document = self._cleanup_extraction_document(
                "featquery_values",
                outputs={"root_ref": "artifacts_root", "path": "roi-runtime"},
                cleanup={"after_extraction": "extraction_runtime"},
            )

            cleanup = cleanup_after_loso_featquery_extraction(
                document,
                roi_set_document=None,
                context=context,
                publication_complete=True,
                publication_root=root / "artifacts" / "roi-loso-flame1",
            )
            current_exists = current.exists()
            sibling_exists = sibling.exists()

        self.assertFalse(current_exists)
        self.assertTrue(sibling_exists)
        self.assertEqual(cleanup[0]["targets"][0]["status"], "removed")

    def test_cleanup_roi_runtime_after_extraction_preserves_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            runtime_root = root / "artifacts" / "roi-runtime"
            current_paths = (
                runtime_root / ".cache" / "loso_groupmaps" / "loso_modelA",
                runtime_root / "loso_groupmaps" / "loso_modelA",
                runtime_root / "rois" / "loso_modelA",
            )
            sibling_paths = (
                runtime_root / ".cache" / "loso_groupmaps" / "other_rois",
                runtime_root / "loso_groupmaps" / "other_rois",
                runtime_root / "rois" / "other_rois",
            )
            for path in (*current_paths, *sibling_paths):
                path.mkdir(parents=True)
                (path / "keep.txt").write_text("x", encoding="utf-8")
            extraction_document = self._cleanup_extraction_document(
                "featquery_values",
                outputs={"root_ref": "artifacts_root", "path": "roi-runtime"},
            )
            roi_document = self._cleanup_roi_document(
                "loso_modelA",
                outputs={"root_ref": "artifacts_root", "path": "roi-runtime"},
                cleanup={"after_extraction": "roi_runtime"},
            )

            cleanup = cleanup_after_loso_featquery_extraction(
                extraction_document,
                roi_set_document=roi_document,
                context=context,
                publication_complete=True,
                publication_root=root / "artifacts" / "roi-loso-flame1",
            )
            current_exists = tuple(path.exists() for path in current_paths)
            sibling_exists = tuple(path.exists() for path in sibling_paths)

        self.assertTrue(all(not exists for exists in current_exists))
        self.assertTrue(all(sibling_exists))
        self.assertEqual(cleanup[0]["scope"], "roi_set")
        self.assertEqual({target["status"] for target in cleanup[0]["targets"]}, {"removed"})

    def test_cleanup_skips_when_publication_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            runtime_root = root / "artifacts" / "roi-runtime"
            current_cache = runtime_root / ".cache" / "loso_groupmaps" / "loso_modelA"
            current_cache.mkdir(parents=True)
            document = self._cleanup_roi_document(
                "loso_modelA",
                outputs={"root_ref": "artifacts_root", "path": "roi-runtime"},
                cleanup={"after_roi_build": "cache_only"},
            )

            cleanup = cleanup_after_loso_roi_build(
                document,
                context=context,
                publication_complete=False,
                publication_root=root / "artifacts" / "roi-loso-flame1",
            )
            current_cache_exists = current_cache.exists()

        self.assertTrue(current_cache_exists)
        self.assertEqual(cleanup[0]["status"], "skipped")
        self.assertEqual(cleanup[0]["reason"], "publication_incomplete")

    def test_cleanup_skips_when_publication_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            runtime_root = root / "artifacts" / "roi-runtime"
            current_rois = runtime_root / "rois" / "loso_modelA"
            current_rois.mkdir(parents=True)
            document = self._cleanup_roi_document(
                "loso_modelA",
                outputs={"root_ref": "artifacts_root", "path": "roi-runtime"},
                cleanup={"after_roi_build": "roi_runtime"},
            )

            cleanup = cleanup_after_loso_roi_build(
                document,
                context=context,
                publication_complete=False,
                publication_root=None,
            )
            current_rois_exists = current_rois.exists()

        self.assertTrue(current_rois_exists)
        self.assertEqual(cleanup[0]["status"], "skipped")
        self.assertEqual(cleanup[0]["reason"], "publication_incomplete")

    def test_cleanup_skips_when_roi_publication_result_is_missing_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            runtime_root = root / "dataset" / "derivatives" / ".research-platform" / "roi-loso-flame1-runtime"
            current_rois = runtime_root / "rois" / "loso_modelA"
            current_rois.mkdir(parents=True)
            action = SimpleNamespace(
                family="loso_group_map",
                roi_label="SeedA",
                mask_path=runtime_root / "rois" / "loso_modelA" / "missing_mask.nii.gz",
                sidecar_path=runtime_root / "rois" / "loso_modelA" / "missing_mask.json",
                metadata={
                    "entities": {
                        "subject_id": "001",
                        "session_id": "01",
                        "task_id": "memory",
                        "direction": "AP",
                        "space": "MNI152NLin6Asym",
                        "resolution": "2",
                        "model": "ModelA",
                    },
                    "roi_parameters": {"sphere_radius_mm": 6},
                    "loso_group_job": {
                        "contrast": {"contrast_id": "CondA", "cope_number": 1, "desc": "CondA"},
                        "zstat_path": runtime_root / "loso_groupmaps" / "missing_zstat.nii.gz",
                        "heldout_subject": "001",
                        "session_id": "01",
                        "task_id": "memory",
                        "model": "ModelA",
                        "group_mask_path": root / "missing_group_mask.nii.gz",
                        "training_inputs": [],
                    },
                },
            )
            document = self._published_loso_roi_document()
            document["roi_set"]["runtime"] = {"cleanup": {"after_roi_build": "roi_runtime"}}  # type: ignore[index]

            publication = publish_loso_roi_build_result(document, actions=[action], context=context)
            cleanup = cleanup_after_loso_roi_build(
                document,
                context=context,
                publication_complete=publication.complete,
                publication_root=publication.root,
            )
            current_rois_exists = current_rois.exists()

        self.assertFalse(publication.complete)
        self.assertTrue(publication.missing_sources)
        self.assertTrue(current_rois_exists)
        self.assertEqual(cleanup[0]["status"], "skipped")
        self.assertEqual(cleanup[0]["reason"], "publication_incomplete")

    def test_old_configs_without_cleanup_do_not_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            runtime_root = root / "artifacts" / "roi-runtime"
            current_cache = runtime_root / ".cache" / "loso_groupmaps" / "loso_modelA"
            current_cache.mkdir(parents=True)
            document = self._cleanup_roi_document("loso_modelA", outputs={"root_ref": "artifacts_root", "path": "roi-runtime"})

            cleanup = cleanup_after_loso_roi_build(
                document,
                context=context,
                publication_complete=True,
                publication_root=root / "artifacts" / "roi-loso-flame1",
            )
            current_cache_exists = current_cache.exists()

        self.assertEqual(cleanup, ())
        self.assertTrue(current_cache_exists)

    def test_featquery_publication_accepts_env_resolved_personal_paths_after_raw_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_derivatives_root = root / "dataset" / "derivatives"
            context = RoiExecutionContext(
                workspace_root=root,
                project_root=root / "project" / "project-demo",
                artifacts_root=root / "artifacts",
                project_name="project-demo",
                root_refs={"dataset_derivatives_root": dataset_derivatives_root},
            )
            table_path = root / "artifacts" / "roi-runtime" / "group_ses-01_task-memory_desc-ModelA_values.tsv"
            table_path.parent.mkdir(parents=True, exist_ok=True)
            table_path.write_text("subject_id\tmean_cope\n001\t4.5\n", encoding="utf-8")
            raw_document = self._explicit_extraction_document()
            resolved_document = self._explicit_extraction_document()
            raw_extraction = raw_document["extraction_set"]  # type: ignore[index]
            resolved_extraction = resolved_document["extraction_set"]  # type: ignore[index]
            raw_extraction["outputs"] = {"root": "${ROI_DERIV_ROOT:-}", "format": "tsv"}  # type: ignore[index]
            raw_extraction["publication"] = {  # type: ignore[index]
                "enabled": True,
                "layout": "loso_flame1_bidslike",
                "root": {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"},
                "table_desc": "{model}LOSOFlame1Featquery",
            }
            resolved_extraction["outputs"] = {"root": "/mnt/ExampleDataset/derivatives/roi-runtime", "format": "tsv"}  # type: ignore[index]
            resolved_extraction["publication"] = dict(raw_extraction["publication"])  # type: ignore[index]
            raw_target = raw_extraction["targets"][0]  # type: ignore[index]
            resolved_target = resolved_extraction["targets"][0]  # type: ignore[index]
            raw_target["inputs"] = {  # type: ignore[index]
                "feat_dir": "${ROI_FEAT_ROOT:-}/{subject_dir}/{session_dir}/func/example.feat",
                "value_image": "stats/cope{cope}",
            }
            resolved_target["inputs"] = {  # type: ignore[index]
                "feat_dir": "/mnt/ExampleDataset/derivatives/fsl/feat/{subject_dir}/{session_dir}/func/example.feat",
                "value_image": "stats/cope{cope}",
            }
            action = SimpleNamespace(
                backend="fsl_featquery",
                roi_label="SeedA",
                table_path=table_path,
                metrics=("mean_cope",),
                metadata={
                    "session_id": "01",
                    "task_id": "memory",
                    "direction": "AP",
                    "model": "ModelA",
                    "source_contrast": "CondA",
                    "cope": "1",
                },
            )

            validation_errors = validate_extraction_set_document(resolved_document, personal_path_document=raw_document)
            published = publish_loso_featquery_extraction(
                resolved_document,
                roi_set_document=None,
                actions=[action],
                tables=[table_path],
                context=context,
            )
            published_root = dataset_derivatives_root / "roi-loso-flame1"
            dataset_exists = (published_root / "dataset_description.json").exists()

        self.assertEqual(validation_errors, [])
        self.assertTrue(published)
        self.assertTrue(dataset_exists)
        self.assertTrue(any(path.name.endswith("_roistats.tsv") for path in published))

    def test_sidecar_roi_voxel_count_satisfies_requested_count_when_report_omits_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            mask = self._write_explicit_mask(root)
            build_roi_sidecar_path(mask).write_text(
                json.dumps({"roi_label": "SeedA", "roi_family": "manual_mask", "voxel_count": 19, "qc_flags": ["pass"]}),
                encoding="utf-8",
            )

            def fake_execute(command_plan: object) -> Path:
                command_plan.output_dir.mkdir(parents=True, exist_ok=True)
                command_plan.report_path.write_text("Mean cope: 4.5\n", encoding="utf-8")
                return command_plan.report_path

            with (
                mock.patch(
                    "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_featquery,
                ) as find_fsl_tool,
            ):
                plan = run_roi_extraction(self._explicit_extraction_document(), context=context)
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

            _values_header, values_rows = self._read_tsv(plan.tables[0])
            _qc_header, qc_rows = self._read_tsv(plan.tables[1])

        self.assertEqual(values_rows[0]["mean_cope"], "4.5")
        self.assertEqual(values_rows[0]["roi_voxel_count"], "19")
        self.assertEqual(qc_rows[0]["usable"], "True")
        self.assertNotIn("roi_voxel_count", qc_rows[0]["warnings"])

    def test_missing_psc_still_fails_when_report_values_policy_is_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)
            document = self._explicit_extraction_document()
            target = document["extraction_set"]["targets"][0]  # type: ignore[index]
            target["metrics"] = ["percent_signal_change", "roi_voxel_count"]  # type: ignore[index]
            target["featquery"] = {"include_percent_signal_change": True}  # type: ignore[index]
            target["missing"] = {"report_values": "fail"}  # type: ignore[index]

            def fake_execute(command_plan: object) -> Path:
                command_plan.output_dir.mkdir(parents=True, exist_ok=True)
                command_plan.report_path.write_text("Mean cope: 31.71\nVoxels = 117\n", encoding="utf-8")
                return command_plan.report_path

            with (
                mock.patch(
                    "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_featquery,
                ) as find_fsl_tool,
            ):
                with self.assertRaisesRegex(ValueError, "mean_psc"):
                    run_roi_extraction(document, context=context)
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

    def test_featquery_failure_preserves_existing_output_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)
            document = self._explicit_extraction_document()
            document["extraction_set"]["runtime"] = {"existing_output": "replace"}  # type: ignore[index]
            planned = plan_roi_extraction(document, context=context)
            output_dir = Path(str(planned.actions[0].metadata["featquery_output_dir"]))
            sentinel = output_dir / "sentinel.txt"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("keep\n", encoding="utf-8")

            def fail_after_partial_output(command_plan: object) -> Path:
                self.assertTrue(command_plan.output_name.startswith(".roi-runtime-"))
                command_plan.output_dir.mkdir(parents=True)
                command_plan.report_path.write_text("partial\n", encoding="utf-8")
                raise RuntimeError("injected featquery failure")

            with mock.patch(
                "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                side_effect=fail_after_partial_output,
            ), mock.patch(
                "research_platform.neuro.roi_execution.shutil.which",
                side_effect=fake_find_featquery,
            ) as find_fsl_tool:
                with self.assertRaisesRegex(RuntimeError, "injected featquery failure"):
                    run_roi_extraction(document, context=context)
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse(planned.tables[0].exists())
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    def test_featquery_geometry_failure_occurs_before_external_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            mask = self._write_explicit_mask(root)
            if nib is None or np is None:
                self.skipTest("numpy and nibabel are required for ROI execution tests")
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), mask)

            with mock.patch("research_platform.neuro.fsl.featquery.execute_featquery_command_plan") as execute:
                with self.assertRaisesRegex(Exception, "not ready for execution"):
                    run_roi_extraction(self._explicit_extraction_document(), context=context)

            execute.assert_not_called()
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    def test_featquery_success_promotes_complete_directory_without_staging_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)
            document = self._explicit_extraction_document()
            document["extraction_set"]["runtime"] = {"existing_output": "replace"}  # type: ignore[index]
            planned = plan_roi_extraction(document, context=context)
            output_dir = Path(str(planned.actions[0].metadata["featquery_output_dir"]))
            stale = output_dir / "stale.txt"
            stale.parent.mkdir(parents=True)
            stale.write_text("replace me\n", encoding="utf-8")

            def fake_execute(command_plan: object) -> Path:
                self.assertTrue(command_plan.output_name.startswith(".roi-runtime-"))
                command_plan.output_dir.mkdir(parents=True)
                command_plan.report_path.write_text("Mean cope: 4.5\nVoxels = 19\n", encoding="utf-8")
                return command_plan.report_path

            with mock.patch(
                "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                side_effect=fake_execute,
            ), mock.patch(
                "research_platform.neuro.roi_execution.shutil.which",
                side_effect=fake_find_featquery,
            ) as find_fsl_tool:
                result = run_roi_extraction(document, context=context)
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

            self.assertFalse(stale.exists())
            self.assertTrue((output_dir / "report.txt").is_file())
            self.assertEqual(result.actions[0].result["report_path"], str(output_dir / "report.txt"))
            self.assertNotIn(".roi-runtime-", result.tables[0].read_text(encoding="utf-8"))
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    def test_featquery_table_promotion_failure_restores_directory_and_table_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, _project_root, context = self._workspace_context(tmpdir)
            self._write_feat_dir(root)
            self._write_explicit_mask(root)
            document = self._explicit_extraction_document()
            document["extraction_set"]["runtime"] = {"existing_output": "replace"}  # type: ignore[index]
            planned = plan_roi_extraction(document, context=context)
            output_dir = Path(str(planned.actions[0].metadata["featquery_output_dir"]))
            directory_sentinel = output_dir / "sentinel.txt"
            directory_sentinel.parent.mkdir(parents=True)
            directory_sentinel.write_text("keep directory\n", encoding="utf-8")
            table_sentinels: dict[Path, str] = {}
            for table_path in (planned.tables[0], _qc_summary_table_path(planned.tables[0])):
                table_path.parent.mkdir(parents=True, exist_ok=True)
                table_path.write_text(f"keep {table_path.name}\n", encoding="utf-8")
                table_sentinels[table_path] = table_path.read_text(encoding="utf-8")

            def fake_execute(command_plan: object) -> Path:
                command_plan.output_dir.mkdir(parents=True)
                command_plan.report_path.write_text("Mean cope: 4.5\nVoxels = 19\n", encoding="utf-8")
                return command_plan.report_path

            real_replace = runtime_outputs.os.replace

            def fail_table_promotion(source: object, destination: object) -> None:
                source_path = Path(source)
                destination_path = Path(destination)
                if "candidate" in source_path.parts and destination_path.suffix == ".tsv":
                    raise OSError("injected table promotion failure")
                real_replace(source, destination)

            with (
                mock.patch(
                    "research_platform.neuro.fsl.featquery.execute_featquery_command_plan",
                    side_effect=fake_execute,
                ),
                mock.patch(
                    "research_platform.neuro.roi_execution.shutil.which",
                    side_effect=fake_find_featquery,
                ) as find_fsl_tool,
                mock.patch.object(runtime_outputs.os, "replace", side_effect=fail_table_promotion),
            ):
                with self.assertRaisesRegex(RoiRuntimeOutputError, "prior destination set was restored"):
                    run_roi_extraction(document, context=context)
            self.assertEqual([call.args[0] for call in find_fsl_tool.call_args_list], ["featquery"])

            self.assertEqual(directory_sentinel.read_text(encoding="utf-8"), "keep directory\n")
            self.assertEqual(
                {path: path.read_text(encoding="utf-8") for path in table_sentinels},
                table_sentinels,
            )
            self.assertFalse(any(root.rglob(".roi-runtime-*")))

    def _read_tsv(self, path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
        return list(reader.fieldnames or []), rows

    def _summary_writer_row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "subject_id": "sub-002",
            "session_id": "ses-01",
            "task_id": "memory",
            "model": "ModelA",
            "roi_set": "loso_modelA",
            "roi_label": "SeedA",
            "roi_desc": "SeedADesc",
            "roi_family": "loso_group_map",
            "source_contrast": "CondA",
            "cope": "1",
            "feat_dir": "datasets/first-level/sub-002/ses-01/func/example.feat",
            "roi_mask_path": "artifacts/roi/sub-002/ses-01/func/mask.nii.gz",
            "featquery_output_dir": "artifacts/roi_extract/fq_SeedA_CondA_cope1",
            "report_path": "artifacts/roi_extract/fq_SeedA_CondA_cope1/report.txt",
            "mean_psc": "",
            "mean_cope": 31.71,
            "roi_voxel_count": 117,
            "usable": True,
            "thresholded_peak": True,
            "below_threshold_fallback": False,
            "peak_x_mm": -2.0,
            "peak_y_mm": -58.0,
            "peak_z_mm": 66.0,
            "z_at_peak": 4.2,
            "backend": "fsl_featquery",
            "featquery_command": '["featquery", "1"]',
            "qc_flags": "pass",
            "warnings": "",
        }
        row.update(overrides)
        return row

    def _workspace_context(self, tmpdir: str) -> tuple[Path, Path, RoiExecutionContext]:
        root = Path(tmpdir)
        project_root = root / "project" / "project-demo"
        artifacts_root = root / "artifacts"
        project_root.mkdir(parents=True, exist_ok=True)
        return (
            root,
            project_root,
            RoiExecutionContext(
                workspace_root=root,
                project_root=project_root,
                artifacts_root=artifacts_root,
                project_name="project-demo",
                root_refs={
                    "feat_root": root / "datasets" / "first-level",
                    "mask_root": project_root / "config" / "analysis" / "masks",
                    "dataset_derivatives_root": root / "dataset" / "derivatives",
                },
            ),
        )

    def _write_feat_dir(self, root: Path) -> Path:
        feat_dir = root / "datasets" / "first-level" / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_model-ModelA.feat"
        stats = feat_dir / "stats"
        stats.mkdir(parents=True, exist_ok=True)
        _write_test_nifti(stats / "cope1.nii.gz", binary=False)
        return feat_dir

    def _write_loso_fixed_effects_inputs(self, root: Path, *, models: tuple[str, ...] = ("ModelA",)) -> None:
        feat_root = root / "datasets" / "first-level"
        for model in models:
            for subject in ("sub-001", "sub-002"):
                cope_dir = feat_root / subject / "ses-01" / "func" / f"task-memory_model-{model}_contrast-CondA"
                cope_dir.mkdir(parents=True, exist_ok=True)
                _write_test_nifti(cope_dir / "cope1.nii.gz", binary=False)
                _write_test_nifti(cope_dir / "varcope1.nii.gz", binary=False)
                _write_test_nifti(cope_dir / "mask.nii.gz", binary=True)
            group_mask = feat_root / "group" / "ses-01" / "func" / f"task-memory_model-{model}_contrast-CondA_mask.nii.gz"
            group_mask.parent.mkdir(parents=True, exist_ok=True)
            _write_test_nifti(group_mask, binary=True)

    def _write_explicit_mask(self, root: Path) -> Path:
        mask = (
            root
            / "project"
            / "project-demo"
            / "config"
            / "analysis"
            / "masks"
            / "sub-001"
            / "ses-01"
            / "func"
            / "sub-001_ses-01_task-memory_label-SeedA_mask.nii.gz"
        )
        mask.parent.mkdir(parents=True, exist_ok=True)
        _write_test_nifti(mask, binary=True)
        return mask

    def _explicit_extraction_document(
        self,
        *,
        metrics: list[str] | None = None,
        include_psc: bool | None = None,
    ) -> dict[str, object]:
        target: dict[str, object] = {
            "name": "FeatqueryValues",
            "backend": "fsl_featquery",
            "desc": "ModelAFeatquery",
            "metrics": metrics or ["mean_cope", "roi_voxel_count"],
            "inputs": {
                "feat_root_ref": "feat_root",
                "feat_dir": "{subject_dir}/{session_dir}/func/sub-{subject_id}_{session_dir}_task-{task_id}_model-{model}.feat",
                "value_image": "stats/cope{cope}",
            },
            "contrasts": [{"id": "CondA", "cope": 1, "desc": "CondA"}],
            "featquery_output_name": "fq_{roi_label}_{source_contrast}_cope{cope}",
            "roi_masks": [
                {
                    "label": "SeedA",
                    "family": "manual_mask",
                    "root_ref": "mask_root",
                    "pattern": "{subject_dir}/{session_dir}/func/sub-{subject_id}_{session_dir}_task-{task_id}_label-SeedA_mask.nii.gz",
                }
            ],
        }
        if include_psc is not None:
            target["featquery"] = {"include_percent_signal_change": include_psc}
        return {
            "extraction_set": {
                "name": "featquery_values",
                "subjects": ["sub-001"],
                "session": "ses-01",
                "task": "memory",
                "model": "ModelA",
                "outputs": {"root_ref": "artifacts_root", "path": "roi-derivatives", "format": "tsv"},
                "targets": [target],
            }
        }

    def _coordinate_roi_document(self) -> dict[str, object]:
        return {
            "roi_set": {
                "name": "coordinate",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "memory",
                "space": "MNI152NLin2009cAsym",
                "outputs": {"root_ref": "artifacts_root", "path": "roi-derivatives"},
                "rois": [
                    {
                        "label": "SeedSphere",
                        "family": "coordinate_sphere",
                        "backend": "generic_nifti",
                        "desc": "CoordinateSphere",
                        "reference_image": {"root_ref": "mask_root", "pattern": "reference.nii.gz"},
                        "coordinate": [0, 0, 0],
                        "radius_mm": 2,
                    }
                ],
            }
        }

    def _published_loso_roi_document(self, *, models: object = "ModelA") -> dict[str, object]:
        return {
            "roi_set": {
                "name": "loso_modelA",
                "backend": "fsl_flame1",
                "subjects": ["sub-001", "sub-002"],
                "held_out_subjects": ["sub-001"],
                "session": "ses-01",
                "task": "memory",
                "direction": "AP",
                "model": models,
                "space": "MNI152NLin6Asym",
                "resolution": "2",
                "min_group_n": 1,
                "outputs": {
                    "root_ref": "dataset_derivatives_root",
                    "path": ".research-platform/roi-loso-flame1-runtime",
                },
                "publication": {
                    "enabled": True,
                    "layout": "loso_flame1_bidslike",
                    "root": {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"},
                    "map_desc": "{model}LOSOFlame1",
                    "mask_desc": "{model}LOSOFlame1Sphere{sphere_radius_mm}mm",
                },
                "fixed_effects_inputs": {
                    "root_ref": "feat_root",
                    "cope_dir": "{subject_dir}/{session_dir}/func/task-{task_id}_model-{model}_contrast-{contrast_id}",
                    "cope_image": "cope{cope_number}.nii.gz",
                    "varcope_image": "varcope{cope_number}.nii.gz",
                    "mask_image": "mask.nii.gz",
                },
                "group_mask": {
                    "root_ref": "feat_root",
                    "pattern": "group/{session_dir}/func/task-{task_id}_model-{model}_contrast-{contrast_id}_mask.nii.gz",
                },
                "contrasts": [{"id": "CondA", "cope_number": 1, "desc": "CondA"}],
                "rois": [
                    {
                        "label": "SeedA",
                        "family": "loso_group_map",
                        "backend": "fsl_flame1",
                        "desc": "CondA",
                        "contrast": "CondA",
                        "seed_coordinate": [0, 0, 0],
                        "search_radius_mm": 12,
                        "sphere_radius_mm": 6,
                        "z_threshold": 3.1,
                        "allow_below_threshold_fallback": True,
                    }
                ],
            }
        }

    def _published_mask_extraction_document(self) -> dict[str, object]:
        document = self._explicit_extraction_document()
        extraction = document["extraction_set"]  # type: ignore[index]
        extraction["roi_set_ref"] = "loso_modelA"  # type: ignore[index]
        extraction["direction"] = "AP"  # type: ignore[index]
        extraction["space"] = "MNI152NLin6Asym"  # type: ignore[index]
        extraction["resolution"] = "2"  # type: ignore[index]
        extraction["roi_mask_source"] = {"source": "roi_set_publication"}  # type: ignore[index]
        target = extraction["targets"][0]  # type: ignore[index]
        target.pop("roi_masks")  # type: ignore[attr-defined]
        target["roi_labels"] = ["SeedA"]  # type: ignore[index]
        return document

    def _write_published_loso_mask(self, root: Path) -> Path:
        mask = build_loso_flame1_mask_path(
            root / "dataset" / "derivatives" / "roi-loso-flame1",
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            roi_label="SeedA",
            contrast_alias="CondA",
            mask_desc="ModelALOSOFlame1Sphere6mm",
        )
        mask.parent.mkdir(parents=True, exist_ok=True)
        _write_test_nifti(mask, binary=True)
        build_roi_sidecar_path(mask).write_text(
            json.dumps(
                {
                    "ROILabel": "SeedA",
                    "ContrastName": "CondA",
                    "FSLCOPE": 1,
                    "VoxelCount": 19,
                    "QCFlags": ["pass"],
                    "FallbackStatus": "thresholded",
                    "PeakCoordinate": [1, 2, 3],
                    "PeakZStatistic": 4.2,
                }
            ),
            encoding="utf-8",
        )
        return mask
    def _roi_set_ref_extraction_document(self) -> dict[str, object]:
        document = self._explicit_extraction_document()
        extraction = document["extraction_set"]  # type: ignore[index]
        extraction["roi_set_ref"] = "coordinate"  # type: ignore[index]
        target = extraction["targets"][0]  # type: ignore[index]
        target.pop("roi_masks")  # type: ignore[attr-defined]
        target["roi_labels"] = ["SeedSphere"]  # type: ignore[index]
        return document

    def _cleanup_roi_document(
        self,
        name: str,
        *,
        outputs: dict[str, object],
        cleanup: dict[str, object] | None = None,
    ) -> dict[str, object]:
        roi_set: dict[str, object] = {
            "name": name,
            "outputs": outputs,
            "rois": [{"label": "SeedA", "family": "loso_group_map"}],
        }
        if cleanup is not None:
            roi_set["runtime"] = {"cleanup": cleanup}
        return {"roi_set": roi_set}

    def _cleanup_extraction_document(
        self,
        name: str,
        *,
        outputs: dict[str, object],
        cleanup: dict[str, object] | None = None,
    ) -> dict[str, object]:
        extraction_set: dict[str, object] = {
            "name": name,
            "roi_set": "loso_modelA",
            "outputs": outputs,
            "targets": [{"name": "FeatqueryValues", "backend": "fsl_featquery"}],
        }
        if cleanup is not None:
            extraction_set["runtime"] = {"cleanup": cleanup}
        return {"extraction_set": extraction_set}


def _write_test_nifti(path: Path, *, binary: bool) -> None:
    if nib is None or np is None:
        raise unittest.SkipTest("numpy and nibabel are required for ROI execution tests")
    data = np.ones((5, 5, 5), dtype=np.uint8 if binary else float)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)


if __name__ == "__main__":
    unittest.main()
