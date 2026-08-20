#!/usr/bin/env python3
"""Generate deterministic, human-data-free materialized MVPA fixtures.

The single synthetic specification crosses two invented subjects, two actual
runs, two conditions, and two ROIs.  Each row contains five ROI-final prepared
features.  For one-based subject ``s``, run ``r``, ROI ``q``, and feature ``f``,
with condition indicator ``c`` equal to zero for ``condition_a`` and one for
``condition_b``, values are defined by::

    feature = 10*s + 3*r + 2*q + f/4 + c*(((s + q)*f + r)/2)
    variance = 1 + s/4 + q/2 + r/4 + f/8

Variances are positive, nonuniform, and identical across conditions for the
same exact unit and ROI.  Every identifier, feature, variance, and metadata
value is an algorithmic invention.  No participant, patient, clinical,
demographic, imaging, or other human data and no external dataset are used.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "datasets" / "ds-mvpa-example"
SCHEMA_VERSION = "research_platform.neuro.mvpa.materialized_pattern_table.v1"
SUBJECTS = ("sub-toy01", "sub-toy02")
RUNS = ("run-01", "run-02")
CONDITIONS = ("condition_a", "condition_b")
ROIS = ("SeedA", "SeedB")
FEATURE_COUNT = 5
TABLE_COLUMNS = (
    "schema_version",
    "pattern_id",
    "subject_id",
    "task_id",
    "run_id",
    "cross_validation_label",
    "condition_id",
    "pattern_source_name",
    "roi_source_name",
    "roi_label",
    "feature_count",
    "voxel_order",
    "voxel_index_hash",
    "feature_space_id",
    "roi_definition_id",
    "feature_values",
    "usable",
    "status",
    "mean_centering_applied",
    "mean_centering_scope",
    "event_count",
    "qc_status",
    "qc_reason",
    "grouping_values",
    "warnings",
    "errors",
    "roi_reference",
    "generator_version",
    "software_version",
    "derivation_id",
    "holdout_id",
    "noise_status",
    "noise_usable",
    "noise_values",
    "noise_feature_count",
    "noise_voxel_order",
    "noise_voxel_index_hash",
    "noise_feature_space_id",
    "noise_roi_definition_id",
    "noise_value_kind",
    "noise_estimation_scope",
    "noise_source",
)


def _json_number(value: Fraction) -> int | float:
    """Return a stable JSON number for an exactly representable fraction."""

    if value.denominator == 1:
        return value.numerator
    return float(value)


def _json_vector(values: tuple[Fraction, ...]) -> str:
    return json.dumps([_json_number(value) for value in values], separators=(",", ":"))


def _feature_ids(roi_label: str) -> tuple[str, ...]:
    return tuple(f"{roi_label}:feature-{index:02d}" for index in range(1, FEATURE_COUNT + 1))


def _feature_index_hash(roi_label: str) -> str:
    payload = ("\n".join(_feature_ids(roi_label)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _feature_values(
    subject_index: int,
    run_index: int,
    condition_index: int,
    roi_index: int,
) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(10 * subject_index + 3 * run_index + 2 * roi_index)
        + Fraction(feature_index, 4)
        + condition_index
        * Fraction((subject_index + roi_index) * feature_index + run_index, 2)
        for feature_index in range(1, FEATURE_COUNT + 1)
    )


def _noise_variances(
    subject_index: int,
    run_index: int,
    roi_index: int,
) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(1)
        + Fraction(subject_index, 4)
        + Fraction(roi_index, 2)
        + Fraction(run_index, 4)
        + Fraction(feature_index, 8)
        for feature_index in range(1, FEATURE_COUNT + 1)
    )


def _synthetic_rows() -> list[dict[str, str]]:
    """Return all 16 rows from the one deterministic formula specification."""

    rows: list[dict[str, str]] = []
    for subject_index, subject_id in enumerate(SUBJECTS, start=1):
        for run_index, run_id in enumerate(RUNS, start=1):
            for condition_index, condition_id in enumerate(CONDITIONS):
                for roi_index, roi_label in enumerate(ROIS, start=1):
                    index_hash = _feature_index_hash(roi_label)
                    feature_space_id = f"toy-feature-space:{roi_label}:v1"
                    roi_definition_id = f"toy-roi-definition:{roi_label}:v1"
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "pattern_id": (
                                f"pattern-{subject_id}_{run_id}_{condition_id}_{roi_label}"
                            ),
                            "subject_id": subject_id,
                            "task_id": "exampletask",
                            "run_id": run_id,
                            "cross_validation_label": run_id,
                            "condition_id": condition_id,
                            "pattern_source_name": "toy-prepared-patterns",
                            "roi_source_name": "toy-rois",
                            "roi_label": roi_label,
                            "feature_count": str(FEATURE_COUNT),
                            "voxel_order": "feature_index_ascending",
                            "voxel_index_hash": index_hash,
                            "feature_space_id": feature_space_id,
                            "roi_definition_id": roi_definition_id,
                            "feature_values": _json_vector(
                                _feature_values(
                                    subject_index,
                                    run_index,
                                    condition_index,
                                    roi_index,
                                )
                            ),
                            "usable": "true",
                            "status": "ok",
                            "mean_centering_applied": "false",
                            "mean_centering_scope": "none",
                            "event_count": "8",
                            "qc_status": "pass",
                            "qc_reason": "",
                            "grouping_values": '{"design":"toy-crossnobis"}',
                            "warnings": "[]",
                            "errors": "[]",
                            "roi_reference": (
                                f"root_ref:mvpa_example/README.md#{roi_label.casefold()}"
                            ),
                            "generator_version": "toy-mvpa-generator-v1",
                            "software_version": "",
                            "derivation_id": "algorithmic-formula-v1",
                            "holdout_id": "",
                            "noise_status": "ok",
                            "noise_usable": "true",
                            "noise_values": _json_vector(
                                _noise_variances(subject_index, run_index, roi_index)
                            ),
                            "noise_feature_count": str(FEATURE_COUNT),
                            "noise_voxel_order": "feature_index_ascending",
                            "noise_voxel_index_hash": index_hash,
                            "noise_feature_space_id": feature_space_id,
                            "noise_roi_definition_id": roi_definition_id,
                            "noise_value_kind": "variance",
                            "noise_estimation_scope": "exact_unit_roi",
                            "noise_source": "algorithmic-variance-formula-v1",
                        }
                    )
    return rows


def _render_table() -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=TABLE_COLUMNS,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(_synthetic_rows())
    return buffer.getvalue().encode("utf-8")


def _dataset_description_bytes() -> bytes:
    metadata = {
        "Name": "Deterministic Toy Materialized MVPA Patterns",
        "Description": (
            "Algorithmically invented ROI-final prepared vectors for local "
            "materialized-pattern crossnobis verification."
        ),
        "DatasetType": "synthetic-fixture",
        "SyntheticData": True,
        "SchemaVersion": SCHEMA_VERSION,
        "GeneratedBy": [
            {
                "Name": "generate_toy_mvpa_fixtures.py",
                "Description": "Deterministic standard-library-only fixture generator",
            }
        ],
    }
    return (json.dumps(metadata, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _generated_files() -> dict[Path, bytes]:
    return {
        DATASET_ROOT / "dataset_description.json": _dataset_description_bytes(),
        DATASET_ROOT / "patterns" / "toy_crossnobis_patterns.tsv": _render_table(),
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in fixtures byte-for-byte without writing",
    )
    args = parser.parse_args()

    generated_files = _generated_files()
    mismatches: list[Path] = []
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
    print(f"{action} {len(generated_files)} deterministic toy MVPA fixtures")
    for path, payload in generated_files.items():
        print(f"{path.relative_to(REPO_ROOT)} sha256={_sha256(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
