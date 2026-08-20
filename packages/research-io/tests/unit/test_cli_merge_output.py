from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


IO_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
FEATURES_ROOT = WORKSPACE_ROOT / "datasets" / "ds-derivatives-example" / "derivatives" / "features" / "project-pilot-tabular"
SOURCE_TABLE = WORKSPACE_ROOT / "datasets" / "ds-tabular-example" / "toy_observations.csv"


def _io_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(IO_PACKAGE_ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    return env


def test_merge_cli_supports_output_materialization(tmp_path: Path) -> None:
    output_path = tmp_path / "toy_features.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_platform.io.cli",
            "merge",
            str(FEATURES_ROOT / "sources" / "toy_core.tsv"),
            str(FEATURES_ROOT / "sources" / "toy_measurements.tsv"),
            "--on",
            "record_id",
            "--format",
            "tsv",
            "--backend",
            "polars",
            "--output",
            str(output_path),
        ],
        cwd=WORKSPACE_ROOT,
        env=_io_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == [
        "record_id",
        "feature_a",
        "feature_b",
        "feature_c",
        "binary_target",
        "measure_x",
        "measure_y",
        "feature_d",
        "continuous_target",
    ]
    assert len(rows) == 25
    assert output_path.read_bytes() == (FEATURES_ROOT / "toy_features.tsv").read_bytes()


def test_toy_observations_support_csv_preview_and_inspection() -> None:
    preview = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_platform.io.cli",
            "preview",
            str(SOURCE_TABLE),
            "--format",
            "csv",
            "--backend",
            "polars",
            "--head",
            "4",
        ],
        cwd=WORKSPACE_ROOT,
        env=_io_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert preview.returncode == 0, preview.stderr
    assert "record-001" in preview.stdout
    assert "binary_target" in preview.stdout

    inspection = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_platform.io.cli",
            "inspect",
            "dtypes",
            str(SOURCE_TABLE),
            "--format",
            "csv",
            "--backend",
            "polars",
        ],
        cwd=WORKSPACE_ROOT,
        env=_io_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert inspection.returncode == 0, inspection.stderr
    dtypes = json.loads(inspection.stdout)
    assert list(dtypes) == [
        "binary_target",
        "continuous_target",
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
        "measure_x",
        "measure_y",
        "record_id",
    ]
