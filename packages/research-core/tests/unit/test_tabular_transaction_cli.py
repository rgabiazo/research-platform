from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from unittest import mock


CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))

from research_platform.core.cli import main
from research_platform.core.config import load_yaml


def _run_cli(args: list[str], artifact_root: Path) -> tuple[int, str]:
    buffer = io.StringIO()
    environment = {
        "RESEARCH_PLATFORM_ROOT": str(WORKSPACE_ROOT),
        "ARTIFACTS_ROOT": str(artifact_root),
        "RP_HPC_HOST": "example-hpc",
        "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
        "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
    }
    with mock.patch.dict(os.environ, environment, clear=False), redirect_stdout(buffer):
        try:
            result = main(args)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                result = exc.code
            else:
                result = 1
                if exc.code:
                    print(exc.code)
    return int(result), buffer.getvalue()


def _regular_file_bytes(root: Path) -> dict[str, bytes]:
    observed: dict[str, bytes] = {}
    if not root.exists():
        return observed
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        identity = path.lstat()
        if stat.S_ISREG(identity.st_mode):
            observed[path.relative_to(root).as_posix()] = path.read_bytes()
    return observed


def _transaction_records(output_root: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    manifest_path = output_root / "transaction-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record["relative_path"]): record
        for record in manifest["outputs"]
    }
    return manifest, records


def test_real_preprocess_plan_and_same_id_execution_commit_exact_transaction() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_root = Path(tmp_dir) / "artifacts"
        run_id = "transaction-preprocess"
        plan_args = [
            "run",
            "plan",
            "preprocess",
            "tabular",
            "--project",
            "project-pilot-tabular",
            "--run-id",
            run_id,
        ]
        execute_args = [
            "run",
            "local",
            "preprocess",
            "tabular",
            "--project",
            "project-pilot-tabular",
            "--run-id",
            run_id,
            "--execute",
        ]

        plan_exit, _ = _run_cli(plan_args, artifact_root)
        run_root = artifact_root / "runs" / run_id
        planned_manifest = load_yaml(run_root / "run-manifest.yaml")
        planned_identity = planned_manifest["plan_identity"]
        transaction_plan = planned_manifest["output_transaction"]

        assert plan_exit == 0
        assert not (run_root / "outputs").exists()
        assert not tuple(run_root.glob(".outputs.*.staging"))
        assert transaction_plan == {
            "schema_version": "research_platform.core.tabular_output_transaction.v1",
            "run_id": run_id,
            "workflow": {"action": "preprocess", "target": "tabular"},
            "final_output_directory": "outputs",
            "outputs": [
                {
                    "logical_name": "split_manifest",
                    "relative_path": "split.json",
                    "content_type": "json",
                },
                {
                    "logical_name": "prep_plan",
                    "relative_path": "prep.json",
                    "content_type": "json",
                },
                {
                    "logical_name": "features_table",
                    "relative_path": "features.tsv",
                    "content_type": "tsv",
                },
            ],
            "transaction_manifest": "outputs/transaction-manifest.json",
            "existing_output": "fail",
        }
        assert "${RP_TABULAR_OUTPUT_ROOT}/split.json" in (
            run_root / "execute.sh"
        ).read_text(encoding="utf-8")

        execute_exit, _ = _run_cli(execute_args, artifact_root)

        assert execute_exit == 0
        output_root = run_root / "outputs"
        assert {path.name for path in output_root.iterdir()} == {
            "split.json",
            "prep.json",
            "features.tsv",
            "transaction-manifest.json",
        }
        assert not tuple(run_root.glob(".outputs.*.staging"))
        assert load_yaml(run_root / "status.yaml")["state"] == "succeeded"
        assert load_yaml(run_root / "run-manifest.yaml")["plan_identity"] == planned_identity

        transaction_manifest, records = _transaction_records(output_root)
        assert transaction_manifest["schema_version"] == (
            "research_platform.core.tabular_output_transaction.v1"
        )
        assert transaction_manifest["run_id"] == run_id
        assert transaction_manifest["workflow"] == {
            "action": "preprocess",
            "target": "tabular",
        }
        assert transaction_manifest["plan_identity"] == {
            "schema_version": planned_identity["schema_version"],
            "sha256": planned_identity["sha256"],
        }
        assert set(records) == {"split.json", "prep.json", "features.tsv"}
        for relative_path, record in records.items():
            data = (output_root / relative_path).read_bytes()
            assert record["byte_size"] == len(data)
            assert record["sha256"] == hashlib.sha256(data).hexdigest()
        assert records["features.tsv"]["row_count"] > 0
        assert records["features.tsv"]["columns"]
        assert b".outputs." not in b"".join(_regular_file_bytes(output_root).values())

        committed_bytes = _regular_file_bytes(output_root)
        repeat_exit, repeat_output = _run_cli(execute_args, artifact_root)

        assert repeat_exit != 0
        assert "choose a new run id" in repeat_output.lower()
        assert _regular_file_bytes(output_root) == committed_bytes
        assert not tuple(run_root.glob(".outputs.*.staging"))


def test_evaluation_requires_successful_unchanged_attested_train_transaction() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_root = Path(tmp_dir) / "artifacts"
        train_id = "transaction-train"
        evaluation_id = "transaction-evaluate"
        train_plan_args = [
            "run",
            "plan",
            "train",
            "model",
            "--project",
            "project-pilot-tabular",
            "--run-id",
            train_id,
        ]
        train_execute_args = [
            "run",
            "local",
            "train",
            "model",
            "--project",
            "project-pilot-tabular",
            "--run-id",
            train_id,
            "--execute",
        ]
        evaluation_plan_args = [
            "run",
            "plan",
            "evaluate",
            "model",
            "--project",
            "project-pilot-tabular",
            "--input-run",
            train_id,
            "--run-id",
            evaluation_id,
        ]
        evaluation_execute_args = [
            "run",
            "local",
            "evaluate",
            "model",
            "--project",
            "project-pilot-tabular",
            "--input-run",
            train_id,
            "--run-id",
            evaluation_id,
            "--execute",
        ]

        assert _run_cli(train_plan_args, artifact_root)[0] == 0
        rejected_exit, rejected_output = _run_cli(evaluation_plan_args, artifact_root)
        assert rejected_exit != 0
        assert "succeeded" in rejected_output.lower()
        assert not (artifact_root / "runs" / evaluation_id).exists()

        assert _run_cli(train_execute_args, artifact_root)[0] == 0
        assert _run_cli(evaluation_plan_args, artifact_root)[0] == 0
        evaluation_root = artifact_root / "runs" / evaluation_id
        source_output_root = artifact_root / "runs" / train_id / "outputs"
        source_transaction_bytes = (
            source_output_root / "transaction-manifest.json"
        ).read_bytes()
        evaluation_manifest = load_yaml(evaluation_root / "run-manifest.yaml")
        input_run = evaluation_manifest["input_run"]
        source_records = {
            record["logical_name"]: record
            for record in json.loads(source_transaction_bytes)["outputs"]
        }

        assert input_run["transaction_manifest_sha256"] == hashlib.sha256(
            source_transaction_bytes
        ).hexdigest()
        assert input_run["plan_identity"] == {
            "schema_version": load_yaml(
                artifact_root / "runs" / train_id / "run-manifest.yaml"
            )["plan_identity"]["schema_version"],
            "sha256": load_yaml(
                artifact_root / "runs" / train_id / "run-manifest.yaml"
            )["plan_identity"]["sha256"],
        }
        assert input_run["consumed"] == {
            "split_manifest": {
                "relative_path": "split.json",
                "sha256": source_records["split_manifest"]["sha256"],
            },
            "features_table": {
                "relative_path": "features.tsv",
                "sha256": source_records["features_table"]["sha256"],
            },
            "model": {
                "relative_path": "model.json",
                "sha256": source_records["model"]["sha256"],
            },
        }
        evaluation_before = _regular_file_bytes(evaluation_root)
        source_model = source_output_root / "model.json"
        source_model.write_bytes(source_model.read_bytes() + b"\n")

        with mock.patch(
            "research_platform.core.cli.subprocess.run",
            side_effect=AssertionError("mutated upstream transaction must fail before launch"),
        ) as process_mock:
            rejected_exit, rejected_output = _run_cli(evaluation_execute_args, artifact_root)

        assert rejected_exit != 0
        assert "input training transaction is invalid" in rejected_output.lower()
        assert "transaction manifest" in rejected_output.lower()
        process_mock.assert_not_called()
        assert _regular_file_bytes(evaluation_root) == evaluation_before
        assert not (evaluation_root / "outputs").exists()
        assert not tuple(evaluation_root.glob(".outputs.*.staging"))
        assert not (artifact_root / "runs" / f".{evaluation_id}.claim").exists()
