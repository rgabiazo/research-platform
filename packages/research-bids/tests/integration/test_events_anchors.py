from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.bids.cli import main


class EventsAnchorIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_root = PACKAGE_ROOT / "tests" / "data" / "events_anchors"
        self.raw_path = self.data_root / "raw" / "basic_memory.csv"

    def _artifact_root(self, suffix: str) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(prefix=f"{suffix}_")

    def test_no_session_layout_inherits_acq_and_dir(self) -> None:
        with self._artifact_root("no_session") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "basic_memory.yaml"),
                    "--source",
                    str(self.raw_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "no_session_dataset"),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "func" / "sub-001_task-memory_acq-fast_dir-AP_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_single_session_layout_resolves_unique_session(self) -> None:
        with self._artifact_root("single_session") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "basic_memory.yaml"),
                    "--source",
                    str(self.raw_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "single_session_dataset"),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_acq-fast_dir-PA_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_label_session_layout_supports_non_numeric_session_labels(self) -> None:
        with self._artifact_root("label_session") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "label_memory.yaml"),
                    "--source",
                    str(self.data_root / "raw" / "label_memory.csv"),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "label_session_dataset"),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "ses-baseline" / "func" / "sub-001_ses-baseline_task-memory_acq-fast_dir-AP_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_widthless_numeric_session_from_source_preserves_leading_zero(self) -> None:
        with self._artifact_root("widthless_source") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "widthless_memory.yaml"),
                    "--source",
                    str(self.data_root / "raw" / "widthless_memory.csv"),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "widthless_session_dataset"),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_acq-fast_dir-AP_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_widthless_numeric_session_override_preserves_leading_zero(self) -> None:
        with self._artifact_root("widthless_override") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "widthless_memory.yaml"),
                    "--source",
                    str(self.raw_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "widthless_session_dataset"),
                    "--session",
                    "01",
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_acq-fast_dir-AP_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_mixed_width_anchor_sessions_remain_distinguishable_from_source(self) -> None:
        with self._artifact_root("mixed_width_source") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "widthless_memory.yaml"),
                    "--source",
                    str(self.data_root / "raw" / "widthless_memory.csv"),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "mixed_width_session_dataset"),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_acq-fast_dir-AP_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_mixed_width_anchor_session_override_resolves_exact_label(self) -> None:
        with self._artifact_root("mixed_width_override") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "widthless_memory.yaml"),
                    "--source",
                    str(self.raw_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "mixed_width_session_dataset"),
                    "--session",
                    "01",
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_acq-fast_dir-AP_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_multi_session_layout_fails_without_explicit_session(self) -> None:
        with self._artifact_root("multi_session_ambiguous") as artifact_root:
            with self.assertRaisesRegex(ValueError, "Ambiguous BOLD anchor"):
                main(
                    [
                        "events",
                        "build",
                        "--spec",
                        str(self.data_root / "basic_memory.yaml"),
                        "--source",
                        str(self.raw_path),
                        "--artifact-root",
                        str(artifact_root),
                        "--dataset-root",
                        str(self.data_root / "multi_session_dataset"),
                    ]
                )

    def test_multi_session_layout_honors_explicit_session(self) -> None:
        with self._artifact_root("multi_session_explicit") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "basic_memory.yaml"),
                    "--source",
                    str(self.raw_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "multi_session_dataset"),
                    "--session",
                    "02",
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "ses-02" / "func" / "sub-001_ses-02_task-memory_acq-fast_dir-PA_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_ambiguous_bold_anchor_fails_loudly(self) -> None:
        with self._artifact_root("anchor_ambiguous") as artifact_root:
            with self.assertRaisesRegex(ValueError, "Ambiguous BOLD anchor"):
                main(
                    [
                        "events",
                        "build",
                        "--spec",
                        str(self.data_root / "basic_memory.yaml"),
                        "--source",
                        str(self.raw_path),
                        "--artifact-root",
                        str(artifact_root),
                        "--dataset-root",
                        str(self.data_root / "ambiguous_anchor_dataset"),
                        "--session",
                        "01",
                    ]
                )

    def test_ped_fallback_applies_only_when_enabled(self) -> None:
        with self._artifact_root("ped_disabled") as disabled_root, self._artifact_root("ped_enabled") as enabled_root:
            disabled_exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "basic_memory.yaml"),
                    "--source",
                    str(self.raw_path),
                    "--artifact-root",
                    str(disabled_root),
                    "--dataset-root",
                    str(self.data_root / "ped_dataset"),
                    "--session",
                    "01",
                ]
            )
            enabled_exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "ped_memory.yaml"),
                    "--source",
                    str(self.raw_path),
                    "--artifact-root",
                    str(enabled_root),
                    "--dataset-root",
                    str(self.data_root / "ped_dataset"),
                    "--session",
                    "01",
                ]
            )

            self.assertEqual(disabled_exit_code, 0)
            self.assertEqual(enabled_exit_code, 0)
            self.assertTrue(
                (Path(disabled_root) / "staged" / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_acq-fast_run-01_events.tsv").exists()
            )
            self.assertTrue(
                (Path(enabled_root) / "staged" / "sub-001" / "ses-01" / "func" / "sub-001_ses-01_task-memory_acq-fast_dir-PA_run-01_events.tsv").exists()
            )

    def test_uncompressed_nifti_anchor_is_discovered(self) -> None:
        with self._artifact_root("uncompressed_anchor") as artifact_root:
            exit_code = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(self.data_root / "basic_memory.yaml"),
                    "--source",
                    str(self.raw_path),
                    "--artifact-root",
                    str(artifact_root),
                    "--dataset-root",
                    str(self.data_root / "uncompressed_dataset"),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected = Path(artifact_root) / "staged" / "sub-001" / "func" / "sub-001_task-memory_acq-fast_dir-AP_run-01_events.tsv"
            self.assertTrue(expected.exists())

    def test_mixed_sessions_in_source_rows_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mixed_sessions_source_rows_") as temp_root, self._artifact_root(
            "mixed_sessions_source"
        ) as artifact_root:
            rows_path = Path(temp_root) / "mixed_sessions_memory.csv"
            rows_path.write_text(
                "participant,session,run,image_file,encode_onset,encode_rt,encode_response,recog_onset,recog_rt,recog_response,image_old_new\n"
                "001,01,1,block1_Run1_Pair_Encoding/stim_a.bmp,20.5,1.1111111111111111,fits,,,,\n"
                "001,02,1,block1_Run1_Pair_Recog/stim_a.bmp,,,,60.25,1.2222222222222223,OLD,OLD\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Mixed sessions found in source rows"):
                main(
                    [
                        "events",
                        "build",
                        "--spec",
                        str(self.data_root / "widthless_memory.yaml"),
                        "--source",
                        str(rows_path),
                        "--artifact-root",
                        str(artifact_root),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
