from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "ops" / "scripts" / "generate_toy_mvpa_fixtures.py"
DATASET_ROOT = REPO_ROOT / "datasets" / "ds-mvpa-example"
TABLE = DATASET_ROOT / "patterns" / "toy_crossnobis_patterns.tsv"
SCHEMA_VERSION = "research_platform.neuro.mvpa.materialized_pattern_table.v1"
EXPECTED_HASHES = {
    "dataset_description.json": (
        "45faea8f5b1274a4d5d108da4f48445d45651e79f60735dfe0cd280f7fc005f9"
    ),
    "patterns/toy_crossnobis_patterns.tsv": (
        "1a97946ebb25f8a327e6162fc4137c0ea4c65143607845c21a81cfbde42cddc9"
    ),
}
EXPECTED_DISTANCES = {
    ("sub-toy01", "SeedA"): 61172 / 1995,
    ("sub-toy01", "SeedB"): 25327351 / 493350,
    ("sub-toy02", "SeedA"): 1284 / 23,
    ("sub-toy02", "SeedB"): 2381 / 30,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> list[dict[str, str]]:
    with TABLE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _crossnobis(rows: list[dict[str, str]]) -> float:
    by_run: dict[str, dict[str, list[float]]] = {}
    noise_rows: list[list[float]] = []
    for row in rows:
        by_run.setdefault(row["run_id"], {})[row["condition_id"]] = json.loads(
            row["feature_values"]
        )
        noise_rows.append(json.loads(row["noise_values"]))
    pooled_variances = [
        sum(values) / len(values) for values in zip(*noise_rows, strict=True)
    ]
    run_ids = sorted(by_run)
    deltas = [
        [
            value_a - value_b
            for value_a, value_b in zip(
                by_run[run_id]["condition_a"],
                by_run[run_id]["condition_b"],
                strict=True,
            )
        ]
        for run_id in run_ids
    ]
    return sum(
        value_a * value_b / variance
        for value_a, value_b, variance in zip(
            deltas[0], deltas[1], pooled_variances, strict=True
        )
    )


def test_toy_mvpa_table_has_exact_bytes_design_formulas_and_distances() -> None:
    for relative_path, expected_hash in EXPECTED_HASHES.items():
        assert _sha256(DATASET_ROOT / relative_path) == expected_hash

    raw = TABLE.read_bytes()
    assert b"\r" not in raw
    rows = _rows()
    assert len(rows) == 16
    assert "session_id" not in rows[0]
    assert {row["schema_version"] for row in rows} == {SCHEMA_VERSION}
    assert {row["subject_id"] for row in rows} == {"sub-toy01", "sub-toy02"}
    assert {row["task_id"] for row in rows} == {"exampletask"}
    assert {row["run_id"] for row in rows} == {"run-01", "run-02"}
    assert {row["condition_id"] for row in rows} == {"condition_a", "condition_b"}
    assert {row["roi_label"] for row in rows} == {"SeedA", "SeedB"}
    assert len({row["pattern_id"] for row in rows}) == 16

    noise_by_unit_roi: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        subject_index = int(row["subject_id"][-2:])
        run_index = int(row["run_id"][-2:])
        condition_index = 0 if row["condition_id"] == "condition_a" else 1
        roi_index = 1 if row["roi_label"] == "SeedA" else 2
        expected_features = [
            10 * subject_index
            + 3 * run_index
            + 2 * roi_index
            + feature_index / 4
            + condition_index
            * (((subject_index + roi_index) * feature_index + run_index) / 2)
            for feature_index in range(1, 6)
        ]
        expected_noise = [
            1
            + subject_index / 4
            + roi_index / 2
            + run_index / 4
            + feature_index / 8
            for feature_index in range(1, 6)
        ]
        assert json.loads(row["feature_values"]) == expected_features
        assert json.loads(row["noise_values"]) == expected_noise
        assert all(math.isfinite(value) for value in expected_features)
        assert all(math.isfinite(value) and value > 0 for value in expected_noise)
        assert row["feature_count"] == row["noise_feature_count"] == "5"
        assert row["cross_validation_label"] == row["run_id"]
        assert row["noise_value_kind"] == "variance"
        assert row["noise_status"] == "ok"
        assert row["noise_usable"] == "true"
        key = (row["subject_id"], row["run_id"], row["roi_label"])
        noise_by_unit_roi.setdefault(key, set()).add(row["noise_values"])
    assert all(len(values) == 1 for values in noise_by_unit_roi.values())

    grouped = {
        key: [
            row
            for row in rows
            if (row["subject_id"], row["roi_label"]) == key
        ]
        for key in EXPECTED_DISTANCES
    }
    observed = {key: _crossnobis(group_rows) for key, group_rows in grouped.items()}
    for key, expected in EXPECTED_DISTANCES.items():
        assert math.isclose(observed[key], expected, rel_tol=0.0, abs_tol=1e-12)
        assert math.isfinite(observed[key]) and observed[key] > 0
    assert len({round(value, 12) for value in observed.values()}) == 4

    metadata = json.loads((DATASET_ROOT / "dataset_description.json").read_text("utf-8"))
    assert metadata["SyntheticData"] is True
    assert metadata["DatasetType"] == "synthetic-fixture"
    assert metadata["SchemaVersion"] == SCHEMA_VERSION


def test_toy_mvpa_generator_check_is_non_mutating() -> None:
    paths = [DATASET_ROOT / relative_path for relative_path in EXPECTED_HASHES]
    before = {path: (path.stat().st_mtime_ns, _sha256(path)) for path in paths}
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})

    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "verified 2 deterministic toy MVPA fixtures" in completed.stdout
    for relative_path, expected_hash in EXPECTED_HASHES.items():
        assert f"{relative_path} sha256={expected_hash}" in completed.stdout
    assert {path: (path.stat().st_mtime_ns, _sha256(path)) for path in paths} == before
