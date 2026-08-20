#!/usr/bin/env python3
"""Generate deterministic, human-data-free toy ROI fixtures.

The fixture grid is 9 by 9 by 9 voxels.  Its affine has 2 mm diagonal
elements and a -8 mm translation on each spatial axis.  The reference image
is zero everywhere.  For zero-based voxel indices ``x``, ``y``, and ``z``,
the value image is defined by ``10 + x*x + 2*y + 3*z``.  All arrays are
written as little-endian float32 in uncompressed NIfTI-1 files.

Every identifier, voxel, affine element, header field, and metadata value is
an algorithmic invention.  No participant, patient, health, demographic, or
other human data and no external dataset are used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "datasets" / "ds-roi-example"
SHAPE = (9, 9, 9)
DTYPE = np.dtype("<f4")
AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, -8.0],
        [0.0, 2.0, 0.0, -8.0],
        [0.0, 0.0, 2.0, -8.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _synthetic_arrays() -> dict[str, np.ndarray]:
    """Return all image arrays from the single deterministic specification."""

    x, y, z = np.indices(SHAPE, dtype=np.float32)
    return {
        "toy_reference.nii": np.zeros(SHAPE, dtype=DTYPE),
        "toy_values.nii": np.asarray(10.0 + x * x + 2.0 * y + 3.0 * z, dtype=DTYPE),
    }


def _nifti_bytes(data: np.ndarray, *, description: str) -> bytes:
    """Render one stable, uncompressed, little-endian NIfTI-1 image."""

    header = nib.Nifti1Header(endianness="<")
    header.set_data_shape(SHAPE)
    header.set_data_dtype(DTYPE)
    header.set_zooms((2.0, 2.0, 2.0))
    header.set_xyzt_units("mm")
    header["descrip"] = description.encode("ascii")
    header["cal_min"] = np.float32(np.min(data))
    header["cal_max"] = np.float32(np.max(data))

    image = nib.Nifti1Image(data, AFFINE, header=header)
    image.set_qform(AFFINE, code=2)
    image.set_sform(AFFINE, code=2)
    image.header.set_data_dtype(DTYPE)
    image.header.set_xyzt_units("mm")
    image.header["descrip"] = description.encode("ascii")
    image.header["cal_min"] = np.float32(np.min(data))
    image.header["cal_max"] = np.float32(np.max(data))

    buffer = io.BytesIO()
    image.to_file_map({"image": nib.FileHolder(fileobj=buffer)})
    return buffer.getvalue()


def _dataset_description_bytes() -> bytes:
    metadata = {
        "Name": "Deterministic Toy ROI Mechanics",
        "Description": (
            "Algorithmically invented generic-NIfTI inputs for local ROI mechanics."
        ),
        "DatasetType": "synthetic-fixture",
        "SyntheticData": True,
        "GeneratedBy": [
            {
                "Name": "generate_toy_roi_fixtures.py",
                "Description": "Deterministic in-repository fixture generator",
            }
        ],
    }
    return (json.dumps(metadata, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _generated_files() -> dict[Path, bytes]:
    arrays = _synthetic_arrays()
    return {
        DATASET_ROOT / "dataset_description.json": _dataset_description_bytes(),
        DATASET_ROOT / "images" / "toy_reference.nii": _nifti_bytes(
            arrays["toy_reference.nii"],
            description="Deterministic toy zero reference",
        ),
        DATASET_ROOT / "images" / "toy_values.nii": _nifti_bytes(
            arrays["toy_values.nii"],
            description="Toy values: 10 + x*x + 2*y + 3*z",
        ),
    }


def _short_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in fixtures byte-for-byte without writing",
    )
    args = parser.parse_args()

    mismatches: list[Path] = []
    generated_files = _generated_files()
    for path, expected_bytes in generated_files.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != expected_bytes:
                mismatches.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected_bytes)

    if mismatches:
        for path in mismatches:
            print(f"mismatch: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    action = "verified" if args.check else "wrote"
    summaries = ", ".join(
        f"{path.relative_to(REPO_ROOT)}={_short_hash(payload)}"
        for path, payload in generated_files.items()
    )
    print(f"{action} {len(generated_files)} deterministic toy ROI fixtures")
    print(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
