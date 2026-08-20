from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from research_platform.analysis import _tabular
from research_platform.analysis.cli import main


BAD_DIGEST = "0" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    table = tmp_path / "features.tsv"
    table.write_text("x\ttarget\n-2\t0\n-1\t0\n1\t1\n2\t1\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "kind": "tabular_split",
                "table_path": "inputs/features.tsv",
                "target_column": "target",
                "row_count": 4,
                "seed": 23,
                "test_fraction": 0.5,
                "split_strategy": "stratified_binary",
                "train_rows": [0, 3],
                "test_rows": [1, 2],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plan = tmp_path / "prep.json"
    plan.write_text(
        json.dumps(
            {
                "kind": "standardize_numeric",
                "table_path": "inputs/features.tsv",
                "target_column": "target",
                "feature_columns": ["x"],
                "statistics": {"x": {"mean": 0.0, "std": 1.0}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "kind": "logistic_regression",
                "table_path": "outputs/features.tsv",
                "target_column": "target",
                "feature_columns": ["x"],
                "intercept": 0.0,
                "weights": {"x": 1.0},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"table": table, "split": split, "plan": plan, "model": model}


def test_verified_table_is_parsed_from_the_exact_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = tmp_path / "table.tsv"
    table.write_text("value\noriginal\n", encoding="utf-8")
    expected_sha256 = _sha256(table)
    original_parser = _tabular._rows_from_bytes

    def mutate_after_read(payload: bytes, *, path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
        table.write_text("value\nchanged\n", encoding="utf-8")
        return original_parser(payload, path=path)

    monkeypatch.setattr(_tabular, "_rows_from_bytes", mutate_after_read)

    fieldnames, rows = _tabular.read_rows(table, expected_sha256=expected_sha256)

    assert fieldnames == ["value"]
    assert rows == [{"value": "original"}]
    assert table.read_text(encoding="utf-8") == "value\nchanged\n"


def test_verified_json_is_parsed_from_the_exact_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "document.json"
    document.write_text('{"value": "original"}\n', encoding="utf-8")
    expected_sha256 = _sha256(document)
    original_reader = _tabular.read_input_bytes

    def mutate_after_read(path: str | Path, *, expected_sha256: str | None = None) -> bytes:
        payload = original_reader(path, expected_sha256=expected_sha256)
        document.write_text('{"value": "changed"}\n', encoding="utf-8")
        return payload

    monkeypatch.setattr(_tabular, "read_input_bytes", mutate_after_read)

    payload = _tabular.read_json(document, expected_sha256=expected_sha256)

    assert payload == {"value": "original"}
    assert json.loads(document.read_text(encoding="utf-8")) == {"value": "changed"}


def test_digest_verified_reads_reject_symlinks_without_changing_legacy_default(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.tsv"
    target.write_text("value\n1\n", encoding="utf-8")
    link = tmp_path / "link.tsv"
    link.symlink_to(target)

    assert _tabular.read_input_bytes(link) == target.read_bytes()
    with pytest.raises(ValueError, match="regular nonsymlink"):
        _tabular.read_input_bytes(link, expected_sha256=_sha256(target))


def test_digest_verified_read_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "input.tsv"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="not a regular file"):
        _tabular.read_input_bytes(fifo, expected_sha256="0" * 64)


@pytest.mark.parametrize("expected_sha256", ["", "not-a-digest", "f" * 63, "g" * 64])
def test_verified_input_rejects_malformed_digest(tmp_path: Path, expected_sha256: str) -> None:
    table = tmp_path / "table.tsv"
    table.write_text("value\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
        _tabular.read_rows(table, expected_sha256=expected_sha256)


@pytest.mark.parametrize(
    "arguments",
    [
        ["split", "create", "--target-column", "target"],
        ["prep", "fit", "--split", "{split}", "--target-column", "target", "--feature-columns", "x"],
        ["prep", "apply", "--plan", "{plan}", "--split", "{split}"],
        ["model", "train", "--split", "{split}", "--target-column", "target", "--feature-columns", "x"],
        ["model", "evaluate", "--split", "{split}", "--target-column", "target", "--model", "{model}"],
        [
            "regression",
            "train",
            "--split",
            "{split}",
            "--target-column",
            "target",
            "--feature-columns",
            "x",
        ],
        ["regression", "evaluate", "--split", "{split}", "--target-column", "target", "--model", "{model}"],
        ["stats", "correlation", "--x", "x", "--y", "target"],
        ["stats", "summary_table", "--column", "x"],
        ["stats", "linear_model", "--outcome", "target", "--predictor", "x"],
        ["stats", "anova", "--outcome", "x", "--group", "target"],
        ["stats", "mixed_effects", "--outcome", "target", "--predictor", "x"],
    ],
)
def test_every_scoped_table_reader_rejects_digest_mismatch_before_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "outputs" / "result.json"
    expanded = [str(inputs[item[1:-1]]) if item.startswith("{") else item for item in arguments]

    result = main(
        [
            *expanded[:2],
            "--table",
            str(inputs["table"]),
            "--expected-table-sha256",
            BAD_DIGEST,
            *expanded[2:],
            "--output",
            str(output),
        ]
    )

    assert result == 1
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert not output.parent.exists()


@pytest.mark.parametrize(
    ("digest_option", "input_name"),
    [
        ("--expected-table-sha256", "table"),
        ("--expected-split-sha256", "split"),
        ("--expected-model-sha256", "model"),
    ],
)
def test_evaluation_rejects_each_input_digest_mismatch_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    digest_option: str,
    input_name: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "outputs" / "evaluation.json"
    digest_arguments = [
        "--expected-table-sha256",
        _sha256(inputs["table"]),
        "--expected-split-sha256",
        _sha256(inputs["split"]),
        "--expected-model-sha256",
        _sha256(inputs["model"]),
    ]
    digest_arguments[digest_arguments.index(digest_option) + 1] = BAD_DIGEST

    result = main(
        [
            "model",
            "evaluate",
            "--table",
            str(inputs["table"]),
            "--split",
            str(inputs["split"]),
            "--target-column",
            "target",
            "--model",
            str(inputs["model"]),
            *digest_arguments,
            "--output",
            str(output),
        ]
    )

    assert result == 1
    error = capsys.readouterr().err
    assert "SHA-256 mismatch" in error
    assert str(inputs[input_name]) in error
    assert not output.parent.exists()


def test_evaluation_accepts_matching_digests_and_preserves_optional_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _write_inputs(tmp_path)
    verified_output = tmp_path / "verified.json"
    default_output = tmp_path / "default.json"
    base_arguments = [
        "model",
        "evaluate",
        "--table",
        str(inputs["table"]),
        "--split",
        str(inputs["split"]),
        "--target-column",
        "target",
        "--model",
        str(inputs["model"]),
    ]

    assert (
        main(
            [
                *base_arguments,
                "--expected-table-sha256",
                _sha256(inputs["table"]),
                "--expected-split-sha256",
                _sha256(inputs["split"]),
                "--expected-model-sha256",
                _sha256(inputs["model"]),
                "--output",
                str(verified_output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main([*base_arguments, "--output", str(default_output)]) == 0

    assert json.loads(verified_output.read_text(encoding="utf-8")) == json.loads(
        default_output.read_text(encoding="utf-8")
    )


def test_evaluation_detects_source_mutation_before_same_byte_read(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _write_inputs(tmp_path)
    planned_table_sha256 = _sha256(inputs["table"])
    inputs["table"].write_text("x\ttarget\n-3\t0\n-1\t0\n1\t1\n2\t1\n", encoding="utf-8")
    output = tmp_path / "evaluation.json"

    result = main(
        [
            "model",
            "evaluate",
            "--table",
            str(inputs["table"]),
            "--expected-table-sha256",
            planned_table_sha256,
            "--split",
            str(inputs["split"]),
            "--target-column",
            "target",
            "--model",
            str(inputs["model"]),
            "--output",
            str(output),
        ]
    )

    assert result == 1
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert not output.exists()


def test_training_reads_staged_table_but_records_portable_table_reference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "model.json"

    result = main(
        [
            "model",
            "train",
            "--table",
            str(inputs["table"]),
            "--table-reference",
            "outputs/features.tsv",
            "--split",
            str(inputs["split"]),
            "--target-column",
            "target",
            "--feature-columns",
            "x",
            "--iterations",
            "2",
            "--output",
            str(output),
        ]
    )

    assert result == 0, capsys.readouterr().err
    assert json.loads(output.read_text(encoding="utf-8"))["table_path"] == "outputs/features.tsv"


def test_training_without_table_reference_preserves_existing_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "model.json"

    result = main(
        [
            "model",
            "train",
            "--table",
            str(inputs["table"]),
            "--split",
            str(inputs["split"]),
            "--target-column",
            "target",
            "--feature-columns",
            "x",
            "--iterations",
            "2",
            "--output",
            str(output),
        ]
    )

    assert result == 0, capsys.readouterr().err
    assert json.loads(output.read_text(encoding="utf-8"))["table_path"] == str(inputs["table"])


def test_regression_training_table_reference_is_metadata_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "regression-model.json"

    def fake_fit_regression_model(**values: object) -> dict[str, object]:
        return {
            "kind": values["kind"],
            "table_path": values["table_path"],
            "target_column": values["target_column"],
            "feature_columns": values["feature_columns"],
        }

    monkeypatch.setattr("research_platform.analysis.cli.fit_regression_model", fake_fit_regression_model)

    result = main(
        [
            "regression",
            "train",
            "--table",
            str(inputs["table"]),
            "--table-reference",
            "outputs/features.tsv",
            "--split",
            str(inputs["split"]),
            "--target-column",
            "target",
            "--feature-columns",
            "x",
            "--output",
            str(output),
        ]
    )

    assert result == 0, capsys.readouterr().err
    assert json.loads(output.read_text(encoding="utf-8"))["table_path"] == "outputs/features.tsv"
