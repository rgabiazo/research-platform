from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from research_platform.analysis._tabular import infer_numeric_feature_columns


ANALYSIS_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
ML_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-ml"
FEATURE_TABLE = (
    WORKSPACE_ROOT
    / "datasets"
    / "ds-derivatives-example"
    / "derivatives"
    / "features"
    / "project-pilot-tabular"
    / "toy_features.tsv"
)
PREDICTOR_COLUMNS = [
    "feature_a",
    "feature_b",
    "feature_c",
    "measure_x",
    "measure_y",
    "feature_d",
]


def _feature_contract_rows() -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = ["record_id", "feature_a", "feature_b", "target", "alternate_outcome", "split_set"]
    rows = [
        {
            "record_id": "row-001",
            "feature_a": "1.0",
            "feature_b": "2.0",
            "target": "0",
            "alternate_outcome": "10.0",
            "split_set": "train",
        },
        {
            "record_id": "row-002",
            "feature_a": "3.0",
            "feature_b": "4.0",
            "target": "1",
            "alternate_outcome": "20.0",
            "split_set": "test",
        },
    ]
    return fieldnames, rows


def _write_feature_contract_inputs(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    table_path = input_dir / "features.tsv"
    table_path.write_text(
        "record_id\tfeature_a\tfeature_b\ttarget\talternate_outcome\tsplit_set\n"
        "row-001\t1.0\t2.0\t0\t10.0\ttrain\n"
        "row-002\t3.0\t4.0\t1\t20.0\ttest\n",
        encoding="utf-8",
    )
    split_path = input_dir / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "kind": "tabular_split",
                "table_path": str(table_path),
                "target_column": "target",
                "row_count": 2,
                "seed": 23,
                "test_fraction": 0.5,
                "split_strategy": "stratified_binary",
                "train_rows": [0],
                "test_rows": [1],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return table_path, split_path


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ANALYSIS_PACKAGE_ROOT / "src"),
            str(ML_PACKAGE_ROOT / "src"),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "research_platform.analysis.cli", *args],
        cwd=WORKSPACE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_explicit_feature_columns_preserve_order() -> None:
    fieldnames, rows = _feature_contract_rows()

    assert infer_numeric_feature_columns(
        rows=rows,
        fieldnames=fieldnames,
        target_column="target",
        feature_columns=["feature_b", "feature_a"],
    ) == ["feature_b", "feature_a"]


@pytest.mark.parametrize(
    ("feature_columns", "expected_error"),
    [
        ([], "must contain at least one column"),
        ([""], "must contain only non-blank strings"),
        (["feature_a", 7], "must contain only non-blank strings"),
        (["feature_a", "feature_a"], "Duplicate feature columns are not allowed: feature_a"),
        (["target"], "Target column 'target' cannot also be a feature column"),
        (["split_set"], "Reserved generated columns cannot be feature columns: split_set"),
        (["missing_column"], "Unknown feature columns: missing_column"),
        (["record_id"], "Column 'record_id' must be numeric"),
        (["record_id", "feature_a"], "Column 'record_id' must be numeric"),
    ],
)
def test_explicit_feature_columns_are_strict(feature_columns: list[object], expected_error: str) -> None:
    fieldnames, rows = _feature_contract_rows()

    with pytest.raises(ValueError) as error:
        infer_numeric_feature_columns(
            rows=rows,
            fieldnames=fieldnames,
            target_column="target",
            feature_columns=feature_columns,  # type: ignore[arg-type]
        )

    assert expected_error in str(error.value)


def test_explicit_feature_columns_reject_missing_table_values() -> None:
    fieldnames, rows = _feature_contract_rows()
    rows[0]["feature_a"] = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Column 'feature_a' must be numeric"):
        infer_numeric_feature_columns(
            rows=rows,
            fieldnames=fieldnames,
            target_column="target",
            feature_columns=["feature_a"],
        )


def test_omitted_feature_columns_retain_legacy_numeric_inference() -> None:
    fieldnames, rows = _feature_contract_rows()

    assert infer_numeric_feature_columns(
        rows=rows,
        fieldnames=fieldnames,
        target_column="target",
        feature_columns=None,
    ) == ["feature_a", "feature_b", "alternate_outcome"]


@pytest.mark.parametrize("command", ["prep", "classification", "regression"])
@pytest.mark.parametrize(
    ("feature_columns", "expected_error"),
    [
        ([""], "must contain only non-blank strings"),
        (["feature_a", "feature_a"], "Duplicate feature columns are not allowed"),
        (["target"], "cannot also be a feature column"),
        (["split_set"], "Reserved generated columns cannot be feature columns"),
        (["missing_column"], "Unknown feature columns"),
        (["record_id"], "must be numeric"),
    ],
)
def test_explicit_feature_contract_failures_write_no_output(
    tmp_path: Path,
    command: str,
    feature_columns: list[str],
    expected_error: str,
) -> None:
    table_path, split_path = _write_feature_contract_inputs(tmp_path)
    output_dir = tmp_path / "outputs"
    output_path = output_dir / f"{command}.json"
    shared = [
        "--table",
        str(table_path),
        "--split",
        str(split_path),
        "--target-column",
        "target",
        "--feature-columns",
        *feature_columns,
    ]
    if command == "prep":
        args = ["prep", "fit", *shared, "--output", str(output_path)]
    elif command == "classification":
        args = ["model", "train", *shared, "--output", str(output_path)]
    else:
        args = [
            "regression",
            "train",
            *shared,
            "--kind",
            "elastic_net_regression",
            "--output",
            str(output_path),
        ]

    result = _run_cli(args)

    assert result.returncode == 1
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_dir.exists()


def test_tabular_cli_full_path(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    repeated_split_path = tmp_path / "split-repeat.json"
    prep_path = tmp_path / "prep.json"
    features_path = tmp_path / "features.tsv"
    model_path = tmp_path / "model.json"
    evaluation_path = tmp_path / "evaluation.json"

    commands = [
        ["split", "create", "--table", str(FEATURE_TABLE), "--target-column", "binary_target", "--output", str(split_path)],
        [
            "split",
            "create",
            "--table",
            str(FEATURE_TABLE),
            "--target-column",
            "binary_target",
            "--output",
            str(repeated_split_path),
        ],
        [
            "prep",
            "fit",
            "--table",
            str(FEATURE_TABLE),
            "--split",
            str(split_path),
            "--target-column",
            "binary_target",
            "--feature-columns",
            *PREDICTOR_COLUMNS,
            "--output",
            str(prep_path),
        ],
        ["prep", "apply", "--table", str(FEATURE_TABLE), "--plan", str(prep_path), "--split", str(split_path), "--output", str(features_path)],
        [
            "model",
            "train",
            "--table",
            str(features_path),
            "--split",
            str(split_path),
            "--target-column",
            "binary_target",
            "--feature-columns",
            *PREDICTOR_COLUMNS,
            "--output",
            str(model_path),
        ],
        ["model", "evaluate", "--table", str(features_path), "--split", str(split_path), "--target-column", "binary_target", "--model", str(model_path), "--output", str(evaluation_path)],
    ]

    for command in commands:
        result = _run_cli(command)
        assert result.returncode == 0, result.stderr

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    prep_payload = json.loads(prep_path.read_text(encoding="utf-8"))
    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))

    assert split_payload == json.loads(repeated_split_path.read_text(encoding="utf-8"))
    assert split_payload["row_count"] == 24
    assert len(split_payload["train_rows"]) == 18
    assert len(split_payload["test_rows"]) == 6
    assert prep_payload["kind"] == "standardize_numeric"
    with FEATURE_TABLE.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["binary_target"] for row in source_rows].count("0") == 12
    assert [row["binary_target"] for row in source_rows].count("1") == 12
    for label_zero, label_one in zip(source_rows[::2], source_rows[1::2], strict=True):
        assert (label_zero["binary_target"], label_one["binary_target"]) == ("0", "1")
        assert label_zero["continuous_target"] == label_one["continuous_target"]
    train_feature_a = [float(source_rows[index]["feature_a"]) for index in split_payload["train_rows"]]
    all_feature_a = [float(row["feature_a"]) for row in source_rows]
    assert math.isclose(prep_payload["statistics"]["feature_a"]["mean"], sum(train_feature_a) / len(train_feature_a))
    assert not math.isclose(prep_payload["statistics"]["feature_a"]["mean"], sum(all_feature_a) / len(all_feature_a))
    assert prep_payload["feature_columns"] == PREDICTOR_COLUMNS
    assert len(prep_payload["feature_columns"]) == 6
    assert model_payload["kind"] == "logistic_regression"
    assert model_payload["target_column"] == "binary_target"
    assert model_payload["feature_columns"] == PREDICTOR_COLUMNS
    assert len(model_payload["feature_columns"]) == 6
    assert evaluation_payload["kind"] == "logistic_regression_evaluation"
    assert evaluation_payload["target_column"] == "binary_target"
    assert evaluation_payload["feature_columns"] == PREDICTOR_COLUMNS
    assert {"record_id", "binary_target", "continuous_target"}.isdisjoint(PREDICTOR_COLUMNS)
    assert "accuracy" in evaluation_payload["metrics"]


def test_tabular_stats_correlation_cli(tmp_path: Path) -> None:
    table_path = tmp_path / "correlation.tsv"
    output_path = tmp_path / "correlation.json"
    table_path.write_text("x\ty\n1\t2\n2\t4\n3\t6\n", encoding="utf-8")

    result = _run_cli(
        [
            "stats",
            "correlation",
            "--table",
            str(table_path),
            "--x",
            "x",
            "--y",
            "y",
            "--method",
            "pearson",
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "correlation"
    assert math.isclose(payload["r"], 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_regression_cli_full_path(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    prep_path = tmp_path / "prep.json"
    features_path = tmp_path / "features.tsv"
    model_path = tmp_path / "model.json"
    evaluation_path = tmp_path / "evaluation.json"
    commands = [
        [
            "split",
            "create",
            "--table",
            str(FEATURE_TABLE),
            "--target-column",
            "continuous_target",
            "--strategy",
            "stratified_binned",
            "--stratify-bin-count",
            "4",
            "--output",
            str(split_path),
        ],
        [
            "prep",
            "fit",
            "--table",
            str(FEATURE_TABLE),
            "--split",
            str(split_path),
            "--target-column",
            "continuous_target",
            "--feature-columns",
            *PREDICTOR_COLUMNS,
            "--output",
            str(prep_path),
        ],
        [
            "prep",
            "apply",
            "--table",
            str(FEATURE_TABLE),
            "--plan",
            str(prep_path),
            "--split",
            str(split_path),
            "--output",
            str(features_path),
        ],
        [
            "regression",
            "train",
            "--table",
            str(features_path),
            "--split",
            str(split_path),
            "--target-column",
            "continuous_target",
            "--kind",
            "elastic_net_regression",
            "--alpha",
            "0.001",
            "--l1-ratio",
            "0.2",
            "--feature-columns",
            *PREDICTOR_COLUMNS,
            "--output",
            str(model_path),
        ],
        [
            "regression",
            "evaluate",
            "--table",
            str(features_path),
            "--split",
            str(split_path),
            "--target-column",
            "continuous_target",
            "--model",
            str(model_path),
            "--bootstrap-iterations",
            "50",
            "--output",
            str(evaluation_path),
        ],
    ]

    for command in commands:
        result = _run_cli(command)
        assert result.returncode == 0, result.stderr

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    prep_payload = json.loads(prep_path.read_text(encoding="utf-8"))
    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))

    assert split_payload["split_strategy"] == "stratified_binned"
    assert split_payload["stratify_bin_count"] == 4
    assert split_payload["row_count"] == 24
    assert prep_payload["target_column"] == "continuous_target"
    assert prep_payload["feature_columns"] == PREDICTOR_COLUMNS
    assert len(prep_payload["feature_columns"]) == 6
    assert model_payload["kind"] == "elastic_net_regression"
    assert model_payload["target_column"] == "continuous_target"
    assert model_payload["feature_columns"] == PREDICTOR_COLUMNS
    assert len(model_payload["feature_columns"]) == 6
    assert evaluation_payload["kind"] == "elastic_net_regression_evaluation"
    assert evaluation_payload["target_column"] == "continuous_target"
    assert evaluation_payload["feature_columns"] == PREDICTOR_COLUMNS
    assert evaluation_payload["metrics"]["r2"] > 0.99
