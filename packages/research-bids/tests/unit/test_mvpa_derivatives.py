from __future__ import annotations

from pathlib import Path
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.bids.mvpa import (
    build_mvpa_distance_table_filename,
    build_mvpa_distance_table_path,
    build_mvpa_figure_path,
    build_mvpa_group_summary_table_filename,
    build_mvpa_group_summary_table_path,
    build_mvpa_pattern_table_filename,
    build_mvpa_pattern_table_path,
    build_mvpa_qc_table_filename,
    build_mvpa_qc_table_path,
    build_mvpa_rdm_table_filename,
    build_mvpa_rdm_table_path,
    build_mvpa_report_path,
    build_mvpa_sidecar_path,
)


EXPECTED_SUBJECT_STEM = (
    "sub-001_ses-01_task-memory_dir-AP_space-MNI152NLin6Asym_res-2_"
    "model-ModelA_label-ExampleROI_desc-Crossnobis"
)
EXPECTED_GROUP_STEM = (
    "group_ses-01_task-memory_dir-AP_space-MNI152NLin6Asym_res-2_"
    "model-ModelA_label-ExampleROI_desc-Crossnobis"
)


class MvpaDerivativePathTests(unittest.TestCase):
    def test_pattern_table_filename_uses_stable_bids_like_order(self) -> None:
        filename = build_mvpa_pattern_table_filename(
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(filename, f"{EXPECTED_SUBJECT_STEM}_patterns.tsv")

    def test_pattern_table_path_uses_published_derivative_layout(self) -> None:
        path = build_mvpa_pattern_table_path(
            publication_root="derivatives",
            derivative_name="mvpa-crossnobis",
            mvpa_set="example",
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(
            path,
            Path(
                "derivatives/mvpa-crossnobis/example/patterns/sub-001/ses-01/"
                f"{EXPECTED_SUBJECT_STEM}_patterns.tsv"
            ),
        )

    def test_distance_table_filename_uses_expected_suffix(self) -> None:
        filename = build_mvpa_distance_table_filename(
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(filename, f"{EXPECTED_SUBJECT_STEM}_distances.tsv")

    def test_distance_table_path_uses_distances_section(self) -> None:
        path = build_mvpa_distance_table_path(
            publication_root="derivatives",
            mvpa_set="example",
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(path.parts[-4:], ("distances", "sub-001", "ses-01", f"{EXPECTED_SUBJECT_STEM}_distances.tsv"))

    def test_rdm_table_filename_uses_expected_suffix(self) -> None:
        filename = build_mvpa_rdm_table_filename(
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(filename, f"{EXPECTED_SUBJECT_STEM}_rdm.tsv")

    def test_rdm_table_path_uses_rdms_section(self) -> None:
        path = build_mvpa_rdm_table_path(
            publication_root="derivatives",
            mvpa_set="example",
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(path.parts[-4:], ("rdms", "sub-001", "ses-01", f"{EXPECTED_SUBJECT_STEM}_rdm.tsv"))

    def test_group_summary_table_filename_is_group_level(self) -> None:
        filename = build_mvpa_group_summary_table_filename(
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(filename, f"{EXPECTED_GROUP_STEM}_mvpasummary.tsv")

    def test_group_summary_table_path_uses_tables_group_layout(self) -> None:
        path = build_mvpa_group_summary_table_path(
            publication_root="derivatives",
            mvpa_set="example",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(path.parts[-4:], ("tables", "group", "ses-01", f"{EXPECTED_GROUP_STEM}_mvpasummary.tsv"))

    def test_qc_table_filename_uses_expected_suffix(self) -> None:
        filename = build_mvpa_qc_table_filename(
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(filename, f"{EXPECTED_SUBJECT_STEM}_mvpaqc.tsv")

    def test_qc_table_path_uses_qc_section(self) -> None:
        path = build_mvpa_qc_table_path(
            publication_root="derivatives",
            mvpa_set="example",
            subject_id="001",
            session_id="01",
            task_id="memory",
            direction="AP",
            space="MNI152NLin6Asym",
            resolution="2",
            model="ModelA",
            label="ExampleROI",
            desc="Crossnobis",
        )

        self.assertEqual(path.parts[-4:], ("qc", "sub-001", "ses-01", f"{EXPECTED_SUBJECT_STEM}_mvpaqc.tsv"))

    def test_runtime_and_publication_roots_use_distinct_layouts(self) -> None:
        runtime_path = build_mvpa_pattern_table_path(
            runtime_root="artifacts",
            mvpa_set="example",
            subject_id="001",
            desc="Crossnobis",
        )
        publication_path = build_mvpa_pattern_table_path(
            publication_root="derivatives",
            derivative_name="mvpa-crossnobis",
            mvpa_set="example",
            subject_id="001",
            desc="Crossnobis",
        )

        self.assertIn(".research-platform", runtime_path.parts)
        self.assertIn("mvpa", runtime_path.parts)
        self.assertEqual(runtime_path.parts[:4], ("artifacts", ".research-platform", "mvpa", "example"))
        self.assertNotIn(".research-platform", publication_path.parts)
        self.assertEqual(publication_path.parts[:3], ("derivatives", "mvpa-crossnobis", "example"))

    def test_sidecar_helper_uses_same_stem_for_supported_outputs(self) -> None:
        for extension in (".tsv", ".png", ".pdf", ".svg", ".html"):
            with self.subTest(extension=extension):
                path = Path(f"derivatives/mvpa/example/sub-001_desc-Crossnobis_patterns{extension}")

                self.assertEqual(
                    build_mvpa_sidecar_path(path),
                    Path("derivatives/mvpa/example/sub-001_desc-Crossnobis_patterns.json"),
                )

    def test_figure_and_report_helpers_support_configurable_extensions(self) -> None:
        figure_path = build_mvpa_figure_path(
            publication_root="derivatives",
            mvpa_set="example",
            subject_id="001",
            task_id="memory",
            model="ModelA",
            desc="RdmHeatmap",
            extension="svg",
        )
        report_path = build_mvpa_report_path(
            publication_root="derivatives",
            mvpa_set="example",
            task_id="memory",
            model="ModelA",
            desc="Summary",
            extension=".pdf",
        )

        self.assertEqual(figure_path.parts[-2:], ("sub-001", "sub-001_task-memory_model-ModelA_desc-RdmHeatmap_figure.svg"))
        self.assertEqual(report_path.name, "task-memory_model-ModelA_desc-Summary_report.pdf")
        self.assertIn("figures", figure_path.parts)
        self.assertIn("reports", report_path.parts)

    def test_mapping_entities_are_ordered_stably(self) -> None:
        filename = build_mvpa_pattern_table_filename(
            entities={
                "desc": "Crossnobis",
                "label": "ExampleROI",
                "model": "ModelA",
                "res": "2",
                "space": "MNI152NLin6Asym",
                "dir": "AP",
                "task": "memory",
                "ses": "01",
                "sub": "001",
            }
        )

        self.assertEqual(filename, f"{EXPECTED_SUBJECT_STEM}_patterns.tsv")

    def test_validation_rejects_unsafe_entity_values(self) -> None:
        unsafe_values = ["", "   ", "bad/value", r"bad\value", "../value", "/tmp/value", r"C:\tmp\value", "bad-value", "bad_value"]

        for value in unsafe_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_mvpa_pattern_table_filename(subject_id="001", task_id=value)

    def test_validation_rejects_unsafe_path_segments_and_root_combinations(self) -> None:
        unsafe_segments = ["", "   ", "..", "bad/value", r"bad\value", "/tmp/value"]

        for value in unsafe_segments:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_mvpa_pattern_table_path(
                        publication_root="derivatives",
                        mvpa_set=value,
                        subject_id="001",
                        desc="Crossnobis",
                    )

        with self.assertRaises(ValueError):
            build_mvpa_pattern_table_path(
                publication_root="derivatives",
                derivative_name="..",
                mvpa_set="example",
                subject_id="001",
                desc="Crossnobis",
            )
        with self.assertRaises(ValueError):
            build_mvpa_pattern_table_path(
                runtime_root="artifacts",
                publication_root="derivatives",
                mvpa_set="example",
                subject_id="001",
                desc="Crossnobis",
            )
        with self.assertRaises(ValueError):
            build_mvpa_pattern_table_path(mvpa_set="example", subject_id="001", desc="Crossnobis")

    def test_generated_stem_does_not_leak_local_absolute_root_pieces(self) -> None:
        path = build_mvpa_distance_table_path(
            publication_root="/tmp/research-platform/derivatives",
            mvpa_set="example",
            subject_id="001",
            task_id="memory",
            desc="Crossnobis",
        )

        self.assertEqual(path.name, "sub-001_task-memory_desc-Crossnobis_distances.tsv")
        self.assertNotIn("tmp", path.stem)
        self.assertNotIn("research-platform", path.stem)
        self.assertNotIn("derivatives", path.stem)


if __name__ == "__main__":
    unittest.main()
