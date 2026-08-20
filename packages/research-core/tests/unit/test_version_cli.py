from __future__ import annotations

import json
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from research_platform.core import entrypoint
from research_platform.core.version import (
    CORE_DISTRIBUTION,
    _source_checkout_version,
    research_core_version,
    version_report,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "src"


def test_version_report_uses_research_core_distribution_metadata() -> None:
    with mock.patch(
        "research_platform.core.version.metadata.version", return_value="0.1.0a1"
    ) as lookup:
        assert research_core_version() == "0.1.0a1"
        assert version_report() == "research-platform 0.1.0a1"

    lookup.assert_called_with(CORE_DISTRIBUTION)


def test_version_report_falls_back_to_source_checkout_metadata() -> None:
    with mock.patch(
        "research_platform.core.version.metadata.version",
        side_effect=metadata.PackageNotFoundError,
    ):
        assert research_core_version() == "0.1.0a1"


def test_source_checkout_fallback_fails_clearly_without_project_metadata(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="installed or source-checkout metadata"):
        _source_checkout_version(tmp_path / "missing.toml")


def test_entrypoint_version_is_lazy_and_imports_no_optional_runtime() -> None:
    script = """
import json
import sys
from importlib import metadata

metadata.version = lambda name: "0.1.0a1"
from research_platform.core.entrypoint import main

code = main(["--version"])
forbidden = (
    "research_platform.analysis",
    "research_platform.hpc",
    "research_platform.neuro",
    "research_platform.viz",
    "matplotlib",
    "nibabel",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
)
loaded = sorted(name for name in sys.modules if name.startswith(forbidden))
print(json.dumps({"code": code, "loaded": loaded}, sort_keys=True))
"""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(SOURCE_ROOT),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "research-platform 0.1.0a1"
    assert json.loads(lines[-1]) == {"code": 0, "loaded": []}


def test_entrypoint_delegates_non_version_arguments_unchanged() -> None:
    command_args = ["setup"]
    with mock.patch("research_platform.core.cli.main", return_value=17) as cli_main:
        assert entrypoint.main(command_args) == 17

    cli_main.assert_called_once_with(command_args)


def test_full_cli_version_fast_path_skips_parser(capsys: pytest.CaptureFixture[str]) -> None:
    from research_platform.core import cli

    with mock.patch(
        "research_platform.core.cli._build_parser",
        side_effect=AssertionError("parser loaded"),
    ):
        with mock.patch(
            "research_platform.core.cli.version_report",
            return_value="research-platform 0.1.0a1",
        ):
            assert cli.main(["--version"]) == 0

    assert capsys.readouterr().out == "research-platform 0.1.0a1\n"
