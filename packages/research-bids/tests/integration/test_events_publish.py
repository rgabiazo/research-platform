from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.bids.cli import main


class EventsPublishIntegrationTest(unittest.TestCase):
    def test_stimuli_enabled_without_copy_mode_preserves_non_copy_stim_file_behavior(self) -> None:
        data_root = PACKAGE_ROOT / "tests" / "data" / "events_publish"
        with tempfile.TemporaryDirectory(prefix="events_publish_no_copy_") as artifact_root:
            build_exit = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(data_root / "publish_memory.yaml"),
                    "--source",
                    str(data_root / "raw" / "publish_memory.csv"),
                    "--artifact-root",
                    str(artifact_root),
                ]
            )
            self.assertEqual(build_exit, 0)
            staged_tsv = Path(artifact_root) / "staged" / "sub-001" / "func" / "sub-001_task-memory_run-01_events.tsv"
            tsv_text = staged_tsv.read_text(encoding="utf-8")
            self.assertIn("\tstim_a.bmp\t", tsv_text)
            self.assertIn("\tstim_b.bmp\t", tsv_text)
            self.assertNotIn("stimuli/stim_a.bmp", tsv_text)
            self.assertNotIn("block1_Run1_Pair_Encoding/stimuli/stim_a.bmp", tsv_text)

    def test_build_and_publish_stage_sidecar_and_stimuli(self) -> None:
        data_root = PACKAGE_ROOT / "tests" / "data" / "events_publish"
        with tempfile.TemporaryDirectory(prefix="events_publish_artifacts_") as artifact_root, tempfile.TemporaryDirectory(
            prefix="events_publish_dataset_"
        ) as dataset_root:
            build_exit = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(data_root / "publish_memory.yaml"),
                    "--source",
                    str(data_root / "raw" / "publish_memory.csv"),
                    "--artifact-root",
                    str(artifact_root),
                    "--write-sidecars",
                    "--copy-stimuli",
                ]
            )
            self.assertEqual(build_exit, 0)

            staged_tsv = Path(artifact_root) / "staged" / "sub-001" / "func" / "sub-001_task-memory_run-01_events.tsv"
            staged_json = staged_tsv.with_suffix(".json")
            staged_stimulus = Path(artifact_root) / "staged" / "stimuli" / "stim_a.bmp"
            staged_stimulus_b = Path(artifact_root) / "staged" / "stimuli" / "stim_b.bmp"
            manifest_path = Path(artifact_root) / "manifests" / "build-manifest.json"

            self.assertTrue(staged_tsv.exists())
            self.assertTrue(staged_json.exists())
            self.assertTrue(staged_stimulus.exists())
            self.assertTrue(staged_stimulus_b.exists())
            tsv_text = staged_tsv.read_text(encoding="utf-8")
            self.assertIn("stimuli/stim_a.bmp", tsv_text)
            self.assertIn("stimuli/stim_b.bmp", tsv_text)
            sidecar_payload = json.loads(staged_json.read_text(encoding="utf-8"))
            self.assertEqual(list(sidecar_payload.keys()), sorted(sidecar_payload.keys()))
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["stimuli"][0]["publish_relpath"], "stimuli/stim_a.bmp")
            self.assertEqual(manifest_payload["outputs"][0]["publish_path"], "sub-001/func/sub-001_task-memory_run-01_events.tsv")
            self.assertIn("input_path", manifest_payload)
            self.assertIn("spec_files", manifest_payload)
            self.assertIn("spec_hash", manifest_payload)
            self.assertIn("counts", manifest_payload)
            self.assertIn("warnings", manifest_payload)

            publish_exit = main(
                [
                    "events",
                    "publish",
                    "--dataset-root",
                    str(dataset_root),
                    "--manifest",
                    str(manifest_path),
                ]
            )
            self.assertEqual(publish_exit, 0)

            published_tsv = Path(dataset_root) / "sub-001" / "func" / "sub-001_task-memory_run-01_events.tsv"
            published_json = Path(dataset_root) / "sub-001" / "func" / "sub-001_task-memory_run-01_events.json"
            published_stimulus = Path(dataset_root) / "stimuli" / "stim_a.bmp"
            published_stimulus_b = Path(dataset_root) / "stimuli" / "stim_b.bmp"

            self.assertEqual(published_tsv.read_text(encoding="utf-8"), staged_tsv.read_text(encoding="utf-8"))
            self.assertEqual(published_json.read_text(encoding="utf-8"), staged_json.read_text(encoding="utf-8"))
            self.assertEqual(published_stimulus.read_text(encoding="utf-8"), staged_stimulus.read_text(encoding="utf-8"))
            self.assertEqual(published_stimulus_b.read_text(encoding="utf-8"), staged_stimulus_b.read_text(encoding="utf-8"))

    def test_copy_stimuli_fails_loudly_on_duplicate_basename_collision(self) -> None:
        data_root = PACKAGE_ROOT / "tests" / "data" / "events_publish"
        with tempfile.TemporaryDirectory(prefix="events_publish_collision_") as artifact_root:
            with self.assertRaisesRegex(ValueError, "Duplicate stimulus basename collision"):
                main(
                    [
                        "events",
                        "build",
                        "--spec",
                        str(data_root / "publish_memory_duplicate.yaml"),
                        "--source",
                        str(data_root / "raw" / "publish_memory.csv"),
                        "--artifact-root",
                        str(artifact_root),
                        "--copy-stimuli",
                    ]
                )

    def test_copy_stimuli_fails_loudly_on_nested_paths_with_same_basename(self) -> None:
        data_root = PACKAGE_ROOT / "tests" / "data" / "events_publish"
        with tempfile.TemporaryDirectory(prefix="events_publish_nested_collision_") as artifact_root:
            with self.assertRaisesRegex(ValueError, "collapse to the same emitted stim_id"):
                main(
                    [
                        "events",
                        "build",
                        "--spec",
                        str(data_root / "publish_memory.yaml"),
                        "--source",
                        str(data_root / "raw" / "publish_memory_nested_collision.csv"),
                        "--artifact-root",
                        str(artifact_root),
                        "--copy-stimuli",
                    ]
                )

    def test_build_fails_loudly_on_same_run_condition_duplicate_basename_without_copy_mode(self) -> None:
        data_root = PACKAGE_ROOT / "tests" / "data" / "events_publish"
        with tempfile.TemporaryDirectory(prefix="events_publish_row_identity_collision_") as artifact_root:
            with self.assertRaisesRegex(ValueError, "collapse to the same emitted stim_id"):
                main(
                    [
                        "events",
                        "build",
                        "--spec",
                        str(data_root / "publish_memory.yaml"),
                        "--source",
                        str(data_root / "raw" / "publish_memory_nested_collision.csv"),
                        "--artifact-root",
                        str(artifact_root),
                    ]
                )

    def test_build_fails_loudly_on_non_stimuli_nested_paths_with_same_emitted_basename(self) -> None:
        data_root = PACKAGE_ROOT / "tests" / "data" / "events_publish"
        with tempfile.TemporaryDirectory(prefix="events_publish_non_stimuli_row_identity_collision_") as artifact_root:
            with self.assertRaisesRegex(ValueError, "collapse to the same emitted stim_id"):
                main(
                    [
                        "events",
                        "build",
                        "--spec",
                        str(data_root / "publish_memory.yaml"),
                        "--source",
                        str(data_root / "raw" / "publish_memory_nonstim_nested_collision.csv"),
                        "--artifact-root",
                        str(artifact_root),
                    ]
                )

    def test_copy_stimuli_fails_when_only_shorter_suffix_exists(self) -> None:
        data_root = PACKAGE_ROOT / "tests" / "data" / "events_publish"
        with tempfile.TemporaryDirectory(prefix="events_publish_missing_exact_") as artifact_root:
            with self.assertRaisesRegex(ValueError, "Unable to resolve stimulus source"):
                main(
                    [
                        "events",
                        "build",
                        "--spec",
                        str(data_root / "publish_memory.yaml"),
                        "--source",
                        str(data_root / "raw" / "publish_memory_missing_exact_path.csv"),
                        "--artifact-root",
                        str(artifact_root),
                        "--copy-stimuli",
                    ]
                )

    def test_publish_fails_by_default_and_overwrites_only_with_opt_in(self) -> None:
        data_root = PACKAGE_ROOT / "tests" / "data" / "events_publish"
        with tempfile.TemporaryDirectory(prefix="events_publish_artifacts_") as artifact_root, tempfile.TemporaryDirectory(
            prefix="events_publish_dataset_"
        ) as dataset_root:
            build_exit = main(
                [
                    "events",
                    "build",
                    "--spec",
                    str(data_root / "publish_memory.yaml"),
                    "--source",
                    str(data_root / "raw" / "publish_memory.csv"),
                    "--artifact-root",
                    str(artifact_root),
                    "--write-sidecars",
                    "--copy-stimuli",
                ]
            )
            self.assertEqual(build_exit, 0)
            manifest_path = Path(artifact_root) / "manifests" / "build-manifest.json"
            published_tsv = Path(dataset_root) / "sub-001" / "func" / "sub-001_task-memory_run-01_events.tsv"
            published_json = Path(dataset_root) / "sub-001" / "func" / "sub-001_task-memory_run-01_events.json"
            published_stimulus = Path(dataset_root) / "stimuli" / "stim_a.bmp"
            published_tsv.parent.mkdir(parents=True, exist_ok=True)
            published_stimulus.parent.mkdir(parents=True, exist_ok=True)
            published_tsv.write_text("existing-tsv", encoding="utf-8")
            published_json.write_text("existing-json", encoding="utf-8")
            published_stimulus.write_text("existing-stimulus", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Publish target conflicts detected"):
                main(
                    [
                        "events",
                        "publish",
                        "--dataset-root",
                        str(dataset_root),
                        "--manifest",
                        str(manifest_path),
                    ]
                )

            overwrite_exit = main(
                [
                    "events",
                    "publish",
                    "--dataset-root",
                    str(dataset_root),
                    "--manifest",
                    str(manifest_path),
                    "--overwrite",
                ]
            )
            self.assertEqual(overwrite_exit, 0)
            self.assertNotEqual(published_tsv.read_text(encoding="utf-8"), "existing-tsv")
            self.assertNotEqual(published_json.read_text(encoding="utf-8"), "existing-json")
            self.assertNotEqual(published_stimulus.read_text(encoding="utf-8"), "existing-stimulus")


if __name__ == "__main__":
    unittest.main()
