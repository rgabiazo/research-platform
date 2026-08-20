from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np


CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
BIDS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-bids"
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(BIDS_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.core.cli import main
from research_platform.neuro._roi_path_safety import (
    published_text_contains_local_path_reference,
    published_value_local_path_fields,
)


DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "ds-roi-example"
REFERENCE_IMAGE = DATASET_ROOT / "images" / "toy_reference.nii"
VALUES_IMAGE = DATASET_ROOT / "images" / "toy_values.nii"
EXPECTED_INPUT_HASHES = {
    "toy_reference.nii": "20a340f4f2c5cd833e102dc35b26d096f46a46e9dfd65659079b9590837affce",
    "toy_values.nii": "c39302a0c937d405af0c60fef066453b8753427b60636b0d229c59e36dc58ee9",
}
EXPECTED_SHAPE = (9, 9, 9)
EXPECTED_AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, -8.0],
        [0.0, 2.0, 0.0, -8.0],
        [0.0, 0.0, 2.0, -8.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
EXPECTED_VALUES_COLUMNS = (
    "subject_id",
    "session_id",
    "task_id",
    "roi_label",
    "value_desc",
    "space",
    "mean",
    "median",
    "voxel_count",
)
EXPECTED_METRICS = {
    "SeedA": {"mean": 34.285714285714285, "median": 34.0, "voxel_count": 7},
    "SeedB": {"mean": 66.28571428571429, "median": 66.0, "voxel_count": 7},
}


class ToyRoiProjectCliIntegrationTests(unittest.TestCase):
    def test_checked_in_images_match_the_deterministic_specification(self) -> None:
        expected_grid = np.indices(EXPECTED_SHAPE, dtype=np.float32)
        expected_values = (
            10.0
            + expected_grid[0] * expected_grid[0]
            + 2.0 * expected_grid[1]
            + 3.0 * expected_grid[2]
        ).astype(np.dtype("<f4"))

        for path in (REFERENCE_IMAGE, VALUES_IMAGE):
            with self.subTest(path=path.name):
                image = nib.load(path)
                self.assertEqual(tuple(image.shape), EXPECTED_SHAPE)
                np.testing.assert_array_equal(image.affine, EXPECTED_AFFINE)
                self.assertEqual(image.get_data_dtype(), np.dtype("<f4"))
                self.assertEqual(image.header.endianness, "<")
                self.assertEqual(sha256(path.read_bytes()).hexdigest(), EXPECTED_INPUT_HASHES[path.name])

        np.testing.assert_array_equal(
            nib.load(REFERENCE_IMAGE).get_fdata(dtype=np.float32),
            np.zeros(EXPECTED_SHAPE, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            nib.load(VALUES_IMAGE).get_fdata(dtype=np.float32),
            expected_values,
        )

    def test_real_rp_handlers_run_the_complete_toy_roi_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rp-toy-roi-integration-") as temp_dir:
            temp_root = Path(temp_dir)
            first_root = temp_root / "first"
            second_root = temp_root / "second"
            first_root.mkdir()
            second_root.mkdir()

            first = self._run_lifecycle(first_root)
            second = self._run_lifecycle(second_root)

            self._assert_output_inventory(first_root, first)
            self._assert_output_inventory(second_root, second)
            self._assert_clean_runs_match(first_root, second_root)

            first_snapshot = self._tree_snapshot(first_root)
            for command in (
                ["analysis", "roi", "build", "toy-spheres", "--project", "project-example", "--execute"],
                [
                    "analysis",
                    "roi",
                    "extraction",
                    "run",
                    "toy-values",
                    "--project",
                    "project-example",
                    "--execute",
                ],
            ):
                code, output = self._invoke(command, artifacts_root=first_root)
                self.assertNotEqual(code, 0)
                self.assertIn("already exists", output)
                self.assertEqual(self._tree_snapshot(first_root), first_snapshot)
                self.assertFalse(self._transaction_remnants(temp_root))

    def _run_lifecycle(self, artifacts_root: Path) -> dict[str, object]:
        commands = {
            "roi_validate": [
                "analysis",
                "roi",
                "validate",
                "toy-spheres",
                "--project",
                "project-example",
            ],
            "roi_doctor": [
                "analysis",
                "roi",
                "doctor",
                "toy-spheres",
                "--project",
                "project-example",
            ],
            "roi_plan": [
                "analysis",
                "roi",
                "build",
                "toy-spheres",
                "--project",
                "project-example",
            ],
            "roi_execute": [
                "analysis",
                "roi",
                "build",
                "toy-spheres",
                "--project",
                "project-example",
                "--execute",
            ],
            "extraction_validate": [
                "analysis",
                "roi",
                "extraction",
                "validate",
                "toy-values",
                "--project",
                "project-example",
            ],
            "extraction_doctor": [
                "analysis",
                "roi",
                "extraction",
                "doctor",
                "toy-values",
                "--project",
                "project-example",
            ],
            "extraction_plan": [
                "analysis",
                "roi",
                "extraction",
                "run",
                "toy-values",
                "--project",
                "project-example",
            ],
            "extraction_execute": [
                "analysis",
                "roi",
                "extraction",
                "run",
                "toy-values",
                "--project",
                "project-example",
                "--execute",
            ],
        }
        payloads: dict[str, object] = {}
        for name, command in commands.items():
            before = self._tree_snapshot(artifacts_root)
            code, output = self._invoke(command, artifacts_root=artifacts_root)
            self.assertEqual(code, 0, output)
            payload = json.loads(output)
            payloads[name] = payload
            if name.endswith(("validate", "doctor", "plan")):
                self.assertEqual(self._tree_snapshot(artifacts_root), before, name)
            if name.endswith("validate"):
                self.assertTrue(payload["valid"])
            if name.endswith("doctor"):
                self.assertTrue(payload["schema_valid"])
                self.assertTrue(payload["ready_for_execution"])
            if name.endswith("plan"):
                self.assertEqual(payload["mode"], "plan")
                self.assertFalse(payload["executed"])
            if name.endswith("execute"):
                self.assertEqual(payload["mode"], "execute")
                self.assertTrue(payload["executed"])
            self.assertFalse(self._transaction_remnants(artifacts_root.parent))
        return payloads

    def _assert_output_inventory(self, artifacts_root: Path, payloads: dict[str, object]) -> None:
        build_payload = payloads["roi_execute"]
        extraction_payload = payloads["extraction_execute"]
        self.assertIsInstance(build_payload, dict)
        self.assertIsInstance(extraction_payload, dict)
        actions = build_payload["actions"]
        extraction_actions = extraction_payload["actions"]
        self.assertEqual(len(actions), 2)
        self.assertEqual(len(extraction_actions), 2)
        self.assertEqual({action["roi_label"] for action in actions}, set(EXPECTED_METRICS))

        masks = sorted(artifacts_root.rglob("*_mask.nii.gz"))
        sidecars = sorted(artifacts_root.rglob("*_mask.json"))
        qc_tables = sorted(artifacts_root.rglob("*_roiextract_qc.tsv"))
        values_tables = sorted(
            path
            for path in artifacts_root.rglob("*_roiextract.tsv")
            if path not in qc_tables
        )
        self.assertEqual(len(masks), 2)
        self.assertEqual(len(sidecars), 2)
        self.assertEqual(len(values_tables), 1)
        self.assertEqual(len(qc_tables), 1)
        self.assertEqual(len([path for path in artifacts_root.rglob("*") if path.is_file()]), 6)

        for mask_path in masks:
            image = nib.load(mask_path)
            mask = image.get_fdata(dtype=np.float32)
            self.assertEqual(tuple(image.shape), EXPECTED_SHAPE)
            np.testing.assert_array_equal(image.affine, EXPECTED_AFFINE)
            self.assertEqual(image.get_data_dtype(), np.dtype("uint8"))
            self.assertEqual(int(np.count_nonzero(mask)), 7)
            self.assertEqual(set(np.unique(mask)), {0.0, 1.0})

        sidecars_by_label: dict[str, dict[str, object]] = {}
        for sidecar_path in sidecars:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            label = str(payload["roi_label"])
            sidecars_by_label[label] = payload
            self.assertEqual(payload["roi_family"], "coordinate_sphere")
            self.assertEqual(payload["backend"], "generic_nifti")
            self.assertEqual(payload["voxel_count"], 7)
            self.assertEqual(payload["project"], "project-example")
            self.assertEqual(payload["roi_set"], "toy-spheres")
            self.assertFalse(published_value_local_path_fields(payload, label="sidecar"))
        self.assertEqual(set(sidecars_by_label), set(EXPECTED_METRICS))
        self.assertEqual(sidecars_by_label["SeedA"]["coordinate"], [-4.0, 0.0, 0.0])
        self.assertEqual(sidecars_by_label["SeedB"]["coordinate"], [4.0, 0.0, 0.0])

        with values_tables[0].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            self.assertEqual(tuple(reader.fieldnames or ()), EXPECTED_VALUES_COLUMNS)
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["roi_label"] for row in rows], ["SeedA", "SeedB"])
        for row in rows:
            expected = EXPECTED_METRICS[row["roi_label"]]
            self.assertEqual(row["subject_id"], "toy01")
            self.assertEqual(row["session_id"], "01")
            self.assertEqual(row["task_id"], "exampletask")
            self.assertEqual(row["value_desc"], "ToyValues")
            self.assertEqual(row["space"], "ToyGrid")
            self.assertAlmostEqual(float(row["mean"]), expected["mean"], places=12)
            self.assertAlmostEqual(float(row["median"]), expected["median"], places=12)
            self.assertEqual(int(row["voxel_count"]), expected["voxel_count"])
        self.assertFalse(published_text_contains_local_path_reference(values_tables[0].read_text(encoding="utf-8")))

        with qc_tables[0].open(newline="", encoding="utf-8") as handle:
            qc_rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(qc_rows), 2)
        self.assertEqual({row["included_in_values"] for row in qc_rows}, {"true"})
        self.assertFalse(self._transaction_remnants(artifacts_root.parent))

    def _assert_clean_runs_match(self, first_root: Path, second_root: Path) -> None:
        first_files = self._relative_files(first_root)
        second_files = self._relative_files(second_root)
        self.assertEqual(set(first_files), set(second_files))

        for relative in sorted(first_files):
            first_path = first_files[relative]
            second_path = second_files[relative]
            if relative.endswith("_qc.tsv"):
                # The QC table is the private run audit and intentionally records
                # runtime paths. Public-facing values and sidecars remain portable.
                continue
            if first_path.suffix in {".json", ".tsv"}:
                self.assertEqual(first_path.read_bytes(), second_path.read_bytes(), relative)
                continue
            if first_path.name.endswith((".nii", ".nii.gz")):
                first_image = nib.load(first_path)
                second_image = nib.load(second_path)
                np.testing.assert_array_equal(first_image.affine, second_image.affine)
                np.testing.assert_array_equal(first_image.get_fdata(), second_image.get_fdata())

    def _invoke(self, args: list[str], *, artifacts_root: Path) -> tuple[int, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        env = {
            "RESEARCH_PLATFORM_ROOT": str(WORKSPACE_ROOT),
            "ARTIFACTS_ROOT": str(artifacts_root),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    code = main(args)
                except SystemExit as exc:
                    code = int(exc.code) if isinstance(exc.code, int) else 1
                    if exc.code not in (None, 0):
                        stderr.write(str(exc.code))
        return int(code or 0), stdout.getvalue() + stderr.getvalue()

    def _relative_files(self, root: Path) -> dict[str, Path]:
        return {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*")
            if path.is_file()
        }

    def _tree_snapshot(self, root: Path) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = []
        for path in sorted(root.rglob("*"), key=str):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append((relative, f"symlink:{os.readlink(path)}"))
            elif path.is_file():
                rows.append((relative, sha256(path.read_bytes()).hexdigest()))
            else:
                rows.append((relative, "directory"))
        return tuple(rows)

    def _transaction_remnants(self, root: Path) -> list[Path]:
        return [
            path
            for path in root.rglob("*")
            if ".roi-runtime-" in path.name or ".roi-publication-" in path.name
        ]


if __name__ == "__main__":
    unittest.main()
