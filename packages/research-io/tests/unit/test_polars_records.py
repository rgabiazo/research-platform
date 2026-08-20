from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from research_platform.io.dataframe.polars_ops import (
    POLARS_RECORDS_ADAPTER_KIND,
    POLARS_RECORDS_ADAPTER_VERSION,
    PolarsRecordsAdapter,
    detect_polars_dataframe_protocol,
    inspect_polars_dataframe_source,
    iter_polars_dataframe_records,
    plan_polars_records_adapter,
    polars_dataframe_to_records,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
MODULE_PATH = PACKAGE_ROOT / "src" / "research_platform" / "io" / "dataframe" / "polars_ops.py"

BLOCKED_IMPORTS = [
    "pandas",
    "polars",
    "numpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "research_platform.analysis",
    "research_platform.viz",
    "research_platform.core",
    "research_platform.neuro",
    "research_platform.bids",
]


def _polars():
    return pytest.importorskip("polars")


def _run_import_check(module_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import json, sys\n"
        f"__import__({module_name!r}, fromlist=['*'])\n"
        f"blocked = {BLOCKED_IMPORTS!r}\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_importing_io_modules_does_not_import_heavy_analysis_or_domain_modules() -> None:
    for module_name in (
        "research_platform.io",
        "research_platform.io.dataframe",
        "research_platform.io.dataframe.polars_ops",
    ):
        completed = _run_import_check(module_name)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []


def test_public_exports_are_available_without_importing_polars_or_peer_heavy_modules() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import json, sys\n"
        "from research_platform.io import POLARS_RECORDS_ADAPTER_KIND, POLARS_RECORDS_ADAPTER_VERSION, polars_dataframe_to_records\n"
        "from research_platform.io.dataframe import PolarsRecordsAdapter, plan_polars_records_adapter\n"
        "assert POLARS_RECORDS_ADAPTER_VERSION == "
        "'research_platform.io.dataframe.polars_records.v1'\n"
        "assert POLARS_RECORDS_ADAPTER_KIND == 'polars_records_adapter'\n"
        "assert callable(polars_dataframe_to_records)\n"
        "assert callable(plan_polars_records_adapter)\n"
        "assert PolarsRecordsAdapter.__name__ == 'PolarsRecordsAdapter'\n"
        f"blocked = {BLOCKED_IMPORTS!r}\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_supports_backend_polars_is_false_when_polars_import_is_unavailable() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import builtins, json, sys\n"
        "real_import = builtins.__import__\n"
        "def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name == 'polars' or name.startswith('polars.'):\n"
        "        raise ModuleNotFoundError(\"No module named 'polars'\")\n"
        "    return real_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = guarded_import\n"
        "from research_platform.io import supports_backend\n"
        "result = supports_backend('polars')\n"
        f"blocked = {BLOCKED_IMPORTS!r}\n"
        "print(json.dumps({'supports': result, 'loaded': [name for name in blocked if name in sys.modules]}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"supports": False, "loaded": []}


def test_polars_dataframe_conversion_preserves_order_columns_and_excludes_row_index_by_default() -> None:
    pl = _polars()
    df = pl.DataFrame(
        {
            "column-alpha": ["row-alpha", "row-beta", "row-gamma"],
            "column-beta": [1, 2, 3],
        }
    )

    result = polars_dataframe_to_records(df)

    assert result.valid is True
    assert result.status == "ok"
    assert result.adapter_spec.runtime_backend == "records"
    assert result.adapter_spec.requested_backend == "polars"
    assert result.row_source_kind == "polars_dataframe"
    assert result.observed_columns == ("column-alpha", "column-beta")
    assert list(result.records[0]) == ["column-alpha", "column-beta"]
    assert [record["column-alpha"] for record in result.records] == ["row-alpha", "row-beta", "row-gamma"]
    assert all("input_row_index" not in record for record in result.records)


def test_include_input_row_index_adds_deterministic_positional_indexes() -> None:
    pl = _polars()
    df = pl.DataFrame({"column-alpha": ["row-alpha", "row-beta"]})

    result = polars_dataframe_to_records(df, include_input_row_index=True)

    assert result.valid is True
    assert result.records == (
        {"column-alpha": "row-alpha", "input_row_index": 0},
        {"column-alpha": "row-beta", "input_row_index": 1},
    )
    assert result.observed_columns == ("column-alpha", "input_row_index")
    assert result.provenance_rows[0].include_input_row_index is True


def test_input_row_index_collision_returns_safe_error() -> None:
    pl = _polars()
    df = pl.DataFrame({"input_row_index": ["value-alpha"]})

    result = polars_dataframe_to_records(df, include_input_row_index=True)

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.qc_rows[0].code == "input_row_index_collision"
    assert result.output_paths_written == ()
    assert result.no_output_written is True


def test_polars_null_and_non_finite_values_are_json_safe() -> None:
    pl = _polars()
    df = pl.DataFrame(
        {
            "column-alpha": [None, float("nan"), float("inf"), float("-inf")],
            "column-beta": ["value-alpha", None, "value-gamma", "value-delta"],
        }
    )

    result = polars_dataframe_to_records(df)
    payload = result.to_dict()

    assert result.valid is True
    assert [record["column-alpha"] for record in result.records] == [None, "nan", "inf", "-inf"]
    assert payload["records"][1]["column-alpha"] == "nan"
    json.dumps(payload, allow_nan=False)


def test_dataframe_is_not_mutated_and_returned_payloads_are_copied() -> None:
    pl = _polars()
    df = pl.DataFrame({"column-alpha": ["row-alpha"], "column-beta": [1]})
    before = df.to_dict(as_series=False)

    result = polars_dataframe_to_records(df, include_input_row_index=True)
    payload = result.to_dict()
    payload["records"][0]["column-alpha"] = "changed-value"
    result.records[0]["column-beta"] = 99

    assert df.to_dict(as_series=False) == before
    assert payload["records"][0]["column-alpha"] == "changed-value"
    assert result.to_dict()["records"][0]["column-alpha"] == "row-alpha"
    assert df.to_dict(as_series=False)["column-beta"] == [1]


def test_polars_series_is_rejected_safely() -> None:
    pl = _polars()
    series = pl.Series("column-alpha", ["value-alpha"])

    result = polars_dataframe_to_records(series)

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.row_source_kind == "polars_series"
    assert result.qc_rows[0].code == "polars_series_source"
    assert result.provenance_rows[0].row_count == 0


def test_polars_lazyframe_is_rejected_safely_without_collecting() -> None:
    pl = _polars()
    lazy_frame = pl.DataFrame({"column-alpha": ["value-alpha"]}).lazy()

    result = polars_dataframe_to_records(lazy_frame)

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.row_source_kind == "polars_lazyframe"
    assert result.qc_rows[0].code == "polars_lazyframe_source"
    assert result.provenance_rows[0].row_count == 0


def test_unsupported_polars_source_is_rejected_safely() -> None:
    _polars()

    result = polars_dataframe_to_records(object())

    assert result.valid is False
    assert result.status == "error"
    assert result.records == ()
    assert result.row_source_kind == "unsupported"
    assert result.qc_rows[0].code == "unsupported_polars_source"


def test_qc_and_provenance_rows_are_tsv_safe_and_no_write() -> None:
    pl = _polars()
    df = pl.DataFrame({"column-alpha": ["value\talpha\nline"]})

    result = polars_dataframe_to_records(df, metadata={"source": "source-alpha"})

    assert result.adapter_spec.will_write is False
    assert result.adapter_spec.output_written is False
    assert result.adapter_spec.output_paths_written == ()
    assert result.adapter_spec.no_output_written is True
    assert result.output_paths_written == ()
    assert result.no_output_written is True
    assert result.to_dict()["output_paths_written"] == []
    for row in (*result.qc_rows, *result.provenance_rows):
        tsv_row = row.to_tsv_row()
        assert tsv_row
        assert all(isinstance(value, str) for value in tsv_row.values())
        assert all("\t" not in value and "\n" not in value and "\r" not in value for value in tsv_row.values())


def test_planning_detection_inspection_adapter_and_iterator_helpers() -> None:
    pl = _polars()
    df = pl.DataFrame({"column-alpha": ["row-alpha", "row-beta"]})
    spec = plan_polars_records_adapter(df, include_input_row_index=True)
    adapter = PolarsRecordsAdapter(spec=spec)

    inspected = inspect_polars_dataframe_source(df)
    iterated = tuple(iter_polars_dataframe_records(df))
    adapted = adapter.to_records(df)

    assert POLARS_RECORDS_ADAPTER_VERSION == "research_platform.io.dataframe.polars_records.v1"
    assert POLARS_RECORDS_ADAPTER_KIND == "polars_records_adapter"
    assert spec.runtime_backend == "records"
    assert spec.requested_backend == "polars"
    assert spec.row_source_kind == "polars_dataframe"
    assert detect_polars_dataframe_protocol(df) == "polars_dataframe"
    assert inspected.valid is True
    assert iterated == ({"column-alpha": "row-alpha"}, {"column-alpha": "row-beta"})
    assert adapted.records[0]["input_row_index"] == 0
    assert adapter.to_dict()["adapter_kind"] == "polars_records_adapter"


def test_no_hard_coded_study_specific_constants_or_subject_style_ids() -> None:
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            MODULE_PATH,
            PACKAGE_ROOT / "README.md",
            REPO_ROOT / "docs" / "decisions" / "ADR-0014-generic-tabular-association-workflow-roadmap.md",
        )
    )
    forbidden = [
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
        "participant-alpha",
        "participant-beta",
        "private-feature-marker",
        "private-contrast-marker",
    ]

    assert all(token not in production_text for token in forbidden)
    assert re.search(r"\bsub-\d{3}\b", production_text) is None
