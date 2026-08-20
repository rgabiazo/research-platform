from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from research_platform.io import read_tabular
from research_platform.io.dataframe.adapters import supports_backend


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "research_platform.io.cli", *args],
        cwd=WORKSPACE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_identifier_csv(path: Path) -> None:
    path.write_text(
        "CODE,VALUE\n"
        "10001,1\n"
        "10002,2\n"
        "0010001.00,3\n",
        encoding="utf-8",
    )


def _write_numeric_csv(path: Path) -> None:
    path.write_text(
        "CODE,VALUE\n"
        "10001,1\n"
        "10002,2\n"
        "10003,3\n",
        encoding="utf-8",
    )


def _column_values(table, column: str) -> list[object]:
    if hasattr(table, "get_column"):
        return table.get_column(column).to_list()
    return list(table[column].tolist())


def test_preview_cli_polars_supports_string_columns(tmp_path: Path) -> None:
    if not supports_backend("polars"):
        pytest.skip("polars backend is not installed")

    path = tmp_path / "mixed_code_polars.csv"
    _write_identifier_csv(path)

    result = _run_cli(
        [
            "preview",
            str(path),
            "--format",
            "csv",
            "--backend",
            "polars",
            "--string-columns",
            "CODE",
            "--head",
            "5",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "preview (polars)" in result.stdout
    assert "0010001.00" in result.stdout


def test_preview_cli_pandas_supports_string_columns(tmp_path: Path) -> None:
    if not supports_backend("pandas"):
        pytest.skip("pandas backend is not installed")

    path = tmp_path / "mixed_code_pandas.csv"
    _write_identifier_csv(path)

    result = _run_cli(
        [
            "preview",
            str(path),
            "--format",
            "csv",
            "--backend",
            "pandas",
            "--string-columns",
            "CODE",
            "--head",
            "5",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "preview (pandas)" in result.stdout
    assert "0010001.00" in result.stdout


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_default_numeric_inference_is_unchanged_when_string_columns_omitted(
    backend: str,
    tmp_path: Path,
) -> None:
    if not supports_backend(backend):
        pytest.skip(f"{backend} backend is not installed")

    path = tmp_path / f"numeric_codes_{backend}.csv"
    _write_numeric_csv(path)

    table = read_tabular(path, backend=backend, format="csv")

    assert _column_values(table, "CODE") == [10001, 10002, 10003]
    assert all(not isinstance(value, str) for value in _column_values(table, "CODE"))
