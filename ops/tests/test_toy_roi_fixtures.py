from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "ops" / "scripts" / "generate_toy_roi_fixtures.py"
DATASET_ROOT = REPO_ROOT / "datasets" / "ds-roi-example"
EXPECTED_HASHES = {
    "dataset_description.json": (
        "f9511f17b3e4e32a9a012b32609e099ffd3217a31261d61fc094b776b664c69f"
    ),
    "images/toy_reference.nii": (
        "20a340f4f2c5cd833e102dc35b26d096f46a46e9dfd65659079b9590837affce"
    ),
    "images/toy_values.nii": (
        "c39302a0c937d405af0c60fef066453b8753427b60636b0d229c59e36dc58ee9"
    ),
}
EXPECTED_AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, -8.0],
        [0.0, 2.0, 0.0, -8.0],
        [0.0, 0.0, 2.0, -8.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_toy_roi_files_have_exact_bytes_geometry_and_values() -> None:
    for relative_path, expected_hash in EXPECTED_HASHES.items():
        assert _sha256(DATASET_ROOT / relative_path) == expected_hash

    reference = nib.load(DATASET_ROOT / "images" / "toy_reference.nii")
    values = nib.load(DATASET_ROOT / "images" / "toy_values.nii")

    for image in (reference, values):
        assert image.shape == (9, 9, 9)
        assert image.get_data_dtype() == np.dtype("<f4")
        np.testing.assert_array_equal(image.affine, EXPECTED_AFFINE)
        assert image.header.get_xyzt_units() == ("mm", "unknown")
        assert int(image.header["qform_code"]) == 2
        assert int(image.header["sform_code"]) == 2

    np.testing.assert_array_equal(np.asanyarray(reference.dataobj), np.zeros((9, 9, 9)))
    x, y, z = np.indices((9, 9, 9), dtype=np.float32)
    expected_values = np.asarray(10.0 + x * x + 2.0 * y + 3.0 * z, dtype="<f4")
    np.testing.assert_array_equal(np.asanyarray(values.dataobj), expected_values)

    metadata = json.loads((DATASET_ROOT / "dataset_description.json").read_text("utf-8"))
    assert metadata["SyntheticData"] is True
    assert metadata["DatasetType"] == "synthetic-fixture"


def test_generator_check_is_non_mutating() -> None:
    paths = [DATASET_ROOT / relative_path for relative_path in EXPECTED_HASHES]
    before = {path: (path.stat().st_mtime_ns, _sha256(path)) for path in paths}
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )

    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "verified 3 deterministic toy ROI fixtures" in completed.stdout
    assert {path: (path.stat().st_mtime_ns, _sha256(path)) for path in paths} == before
