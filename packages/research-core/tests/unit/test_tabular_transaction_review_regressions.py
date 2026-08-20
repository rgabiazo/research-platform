from __future__ import annotations

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Callable
from unittest import mock

import pytest


CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))

from research_platform.core import cli as core_cli
from research_platform.core.cli import main
from research_platform.core.config import load_yaml, write_yaml
from research_platform.core.run_lifecycle import build_plan_identity
import research_platform.core.tabular_output_transaction as transaction


def _write_workspace(root: Path) -> tuple[Path, Path, Path]:
    project_root = root / "project" / "test-tabular"
    feature_root = (
        root
        / "datasets"
        / "ds-tabular"
        / "derivatives"
        / "features"
        / "test-tabular"
    )
    batch_path = project_root / "manifests" / "batches" / "default.tsv"
    for path in (
        project_root / "config",
        project_root / "config" / "analysis",
        batch_path.parent,
        root / "ops" / "sync" / "rsync",
        root / "ops" / "slurm" / "job_templates",
        root / "packages" / "research-core",
        root / "packages" / "research-hpc",
        root / "packages" / "research-analysis",
        root / "packages" / "research-ml",
        feature_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    write_yaml(
        root / "WORKSPACE.yaml",
        {
            "paths": {
                "artifacts_root": "./artifacts",
                "datasets_root": "./datasets",
                "ops_root": "./ops",
            },
            "repos": {
                "project_root": "./project",
                "pipelines_root": "./pipelines",
            },
            "projects": {"default": "test-tabular"},
        },
    )
    write_yaml(project_root / "project.yaml", {"name": "test-tabular", "version": "0.1.0"})
    write_yaml(
        project_root / "config" / "dataset.yaml",
        {
            "dataset": {
                "primary": "ds-tabular",
                "canonical_dataset": "ds-tabular",
                "canonical_features_root": "derivatives/features",
            }
        },
    )
    write_yaml(
        project_root / "config" / "compute.yaml",
        {
            "compute": {
                "default_profile": "local",
                "local": {"jobs": 1},
                "slurm": {
                    "ssh_host": "${RP_HPC_HOST:-}",
                    "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                    "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                    "cpus": 1,
                    "mem": "4G",
                    "time": "00:30:00",
                },
            }
        },
    )
    write_yaml(
        project_root / "config" / "preprocessing.yaml",
        {"preprocessing": {"slice": "tabular", "default_batch": "default"}},
    )
    write_yaml(
        project_root / "config" / "models.yaml",
        {
            "models": {
                "default": {
                    "kind": "logistic_regression",
                    "feature_columns": ["feature_a"],
                    "learning_rate": 0.2,
                    "iterations": 25,
                }
            }
        },
    )
    write_yaml(
        project_root / "config" / "analysis" / "feature-correlation.yaml",
        {
            "analysis": {
                "kind": "correlation",
                "input_table": (
                    "datasets/ds-tabular/derivatives/features/"
                    "test-tabular/toy_features.tsv"
                ),
                "x": "feature_a",
                "y": "binary_target",
                "method": "pearson",
            }
        },
    )
    batch_path.write_text(
        "feature_table\ttarget_column\n"
        "test-tabular/toy_features.tsv\tbinary_target\n",
        encoding="utf-8",
        newline="\n",
    )
    table_path = feature_root / "toy_features.tsv"
    table_path.write_text(
        "record_id\tfeature_a\tbinary_target\n"
        "record-001\t1\t0\n"
        "record-002\t2\t1\n"
        "record-003\t3\t0\n"
        "record-004\t4\t1\n",
        encoding="utf-8",
        newline="\n",
    )
    for name in ("exclude.txt", "exclude.common.txt", "exclude.tabular-ml.txt"):
        (root / "ops" / "sync" / "rsync" / name).write_text("", encoding="utf-8")
    (root / "ops" / "slurm" / "job_templates" / "sbatch.job.sh").write_text(
        "\n".join(
            (
                "#!/usr/bin/env bash",
                "#SBATCH --job-name={{ job_name }}",
                "#SBATCH --cpus-per-task={{ cpus }}",
                "#SBATCH --mem={{ mem }}",
                "#SBATCH --time={{ time }}",
                "#SBATCH --output={{ log_out }}",
                "#SBATCH --error={{ log_err }}",
                "",
                "{{ command }}",
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for package_name in (
        "research-analysis",
        "research-core",
        "research-hpc",
        "research-ml",
    ):
        (root / "packages" / package_name / "src").symlink_to(
            WORKSPACE_ROOT / "packages" / package_name / "src",
            target_is_directory=True,
        )
    return project_root, batch_path, table_path


def _run_cli(args: list[str], *, workspace_root: Path, artifact_root: Path) -> tuple[int, str]:
    output = io.StringIO()
    environment = {
        "RESEARCH_PLATFORM_ROOT": str(workspace_root),
        "ARTIFACTS_ROOT": str(artifact_root),
        "RP_HPC_HOST": "example-hpc",
        "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
        "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
    }
    with mock.patch.dict(os.environ, environment, clear=False), redirect_stdout(output):
        try:
            result = main(args)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                result = exc.code
            else:
                result = 1
                if exc.code:
                    print(exc.code)
    return int(result), output.getvalue()


def _preprocess_args(run_id: str, *, execute: bool) -> list[str]:
    args = [
        "run",
        "local" if execute else "plan",
        "preprocess",
        "tabular",
        "--project",
        "test-tabular",
        "--run-id",
        run_id,
    ]
    if execute:
        args.append("--execute")
    return args


def _write_mock_outputs(kwargs: dict[str, object]) -> Path:
    environment = kwargs.get("env")
    assert isinstance(environment, dict)
    output_root = Path(str(environment["RP_TABULAR_OUTPUT_ROOT"]))
    run_root = output_root.parent
    manifest = load_yaml(run_root / "run-manifest.yaml")
    workflow = manifest["workflow"]

    def write_json(name: str, value: dict[str, object]) -> None:
        (output_root / name).write_text(
            json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    if workflow == {"action": "evaluate", "target": "model"}:
        input_run = manifest["input_run"]
        source_root = run_root.parent / input_run["run_id"] / "outputs"
        split = json.loads((source_root / "split.json").read_text(encoding="utf-8"))
        test_rows = list(split["test_rows"])
        predictor = manifest["predictor_contract"]
        write_json(
            "evaluation.json",
            {
                "kind": "logistic_regression_evaluation",
                "target_column": predictor["target_column"],
                "feature_columns": predictor["feature_columns"],
                "table_path": str(source_root / "features.tsv"),
                "metrics": {
                    "accuracy": 1.0,
                    "log_loss": 0.2876820724517809,
                    "precision": 1.0,
                    "recall": 1.0,
                    "test_count": len(test_rows),
                },
                "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                "predictions": [
                    {
                        "actual": 1,
                        "predicted": 1,
                        "probability": 0.75,
                        "row_number": index,
                    }
                    for index, _ in enumerate(test_rows)
                ],
            },
        )
        return output_root

    source_path = Path(str(kwargs["cwd"])) / manifest["dataset"]["feature_table"]
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames is not None
        source_columns = list(reader.fieldnames)
        source_rows = [dict(row) for row in reader]
    predictor = manifest["predictor_contract"]
    features = list(predictor["feature_columns"])
    target = str(predictor["target_column"])
    train_rows = [0, 1, 2]
    test_rows = [3]
    write_json(
        "split.json",
        {
            "kind": "tabular_split",
            "table_path": manifest["dataset"]["feature_table"],
            "target_column": target,
            "row_count": len(source_rows),
            "seed": manifest["settings"]["preprocessing"]["split_seed"],
            "test_fraction": manifest["settings"]["preprocessing"]["test_fraction"],
            "split_strategy": "stratified_binary",
            "train_rows": train_rows,
            "test_rows": test_rows,
        },
    )
    write_json(
        "prep.json",
        {
            "kind": "standardize_numeric",
            "table_path": manifest["dataset"]["feature_table"],
            "target_column": target,
            "feature_columns": features,
            "statistics": {feature: {"mean": 2.0, "std": 1.0} for feature in features},
        },
    )
    membership = {index: "train" for index in train_rows} | {index: "test" for index in test_rows}
    columns = [*source_columns, "split_set"]
    with (output_root / "features.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for index, row in enumerate(source_rows):
            writer.writerow([*(row[column] for column in source_columns), membership[index]])
    if workflow == {"action": "train", "target": "model"}:
        write_json(
            "model.json",
            {
                "kind": manifest["tool"]["model"],
                "target_column": target,
                "feature_columns": features,
                "intercept": 0.0,
                "weights": {feature: 0.5 for feature in features},
                "learning_rate": manifest["settings"]["model"]["learning_rate"],
                "iterations": manifest["settings"]["model"]["iterations"],
                "training_metrics": {"accuracy": 1.0},
                "table_path": "outputs/features.tsv",
            },
        )
    return output_root


def _mocked_success(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
    _write_mock_outputs(kwargs)
    return subprocess.CompletedProcess(command, 0)


def _plan_and_execute_train(*, workspace_root: Path, artifact_root: Path, run_id: str) -> Path:
    plan = [
        "run",
        "plan",
        "train",
        "model",
        "--project",
        "test-tabular",
        "--run-id",
        run_id,
    ]
    execute = [
        "run",
        "local",
        "train",
        "model",
        "--project",
        "test-tabular",
        "--run-id",
        run_id,
        "--execute",
    ]
    assert _run_cli(plan, workspace_root=workspace_root, artifact_root=artifact_root)[0] == 0
    with mock.patch("research_platform.core.cli.shutil.which", return_value="/bin/bash"), mock.patch(
        "research_platform.core.cli.subprocess.run", side_effect=_mocked_success
    ):
        assert _run_cli(execute, workspace_root=workspace_root, artifact_root=artifact_root)[0] == 0
    return artifact_root / "runs" / run_id


def _reattest_source_run(run_root: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    manifest = load_yaml(run_root / "run-manifest.yaml")
    mutate(manifest)
    manifest["plan_identity"] = build_plan_identity(
        manifest,
        execute_script=(run_root / "execute.sh").read_bytes(),
    )
    write_yaml(run_root / "run-manifest.yaml", manifest)
    receipt_path = run_root / "outputs" / "transaction-manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["plan_identity"] = {
        "schema_version": manifest["plan_identity"]["schema_version"],
        "sha256": manifest["plan_identity"]["sha256"],
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_staged_scientific_output_hard_link_is_rejected_without_touching_source(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.json"
    outside_bytes = b'{"value":1}\n'
    outside.write_bytes(outside_bytes)
    run_root = tmp_path / "run"
    run_root.mkdir()
    staging = transaction.create_owned_staging(run_root)
    os.link(outside, staging.path / "result.json")

    with pytest.raises(
        transaction.TabularOutputTransactionError,
        match="hard-linked files",
    ):
        transaction.validate_staged_outputs(
            staging,
            (transaction.OutputSpec("result", "result.json", "json"),),
        )

    assert outside.read_bytes() == outside_bytes
    assert outside.stat().st_nlink == 2


def test_running_status_parent_fsync_failure_retains_claim_and_reports_uncertainty(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = "running-status-uncertain"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0
    original_writer = core_cli._write_tabular_status_atomic
    original_fsync = os.fsync

    def status_writer(*, run_root_path: Path, status: dict[str, object]) -> None:
        if status.get("state") != "running":
            original_writer(run_root_path=run_root_path, status=status)
            return

        def fail_directory_sync(descriptor: int) -> None:
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError("injected parent fsync failure")
            original_fsync(descriptor)

        with mock.patch.object(core_cli.os, "fsync", side_effect=fail_directory_sync):
            original_writer(run_root_path=run_root_path, status=status)

    with mock.patch.object(core_cli, "_write_tabular_status_atomic", side_effect=status_writer), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=AssertionError("uncertain running status must prevent launch"),
    ) as process_mock:
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    claim = artifact_root / "runs" / f".{run_id}.claim"
    assert code != 0
    assert "uncertain" in output.lower()
    assert claim.is_dir()
    assert not (artifact_root / "runs" / run_id / "outputs").exists()
    process_mock.assert_not_called()


def test_status_replace_and_owned_temporary_cleanup_failure_reports_exact_recovery_path(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = "status-temporary-recovery"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0
    run_root = artifact_root / "runs" / run_id
    original_unlink = Path.unlink

    def reject_owned_status_temporary(path: Path, *args: object, **kwargs: object) -> None:
        if (
            path.parent == run_root
            and path.name.startswith(".status.")
            and path.name.endswith(".tmp")
        ):
            raise OSError("injected owned status-temporary cleanup failure")
        original_unlink(path, *args, **kwargs)

    with mock.patch.object(
        core_cli.os,
        "replace",
        side_effect=OSError("injected status replacement failure"),
    ), mock.patch.object(
        Path,
        "unlink",
        new=reject_owned_status_temporary,
    ), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=AssertionError("uncertain status temporary must prevent launch"),
    ) as process_mock:
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    temporary_paths = tuple(run_root.glob(".status.*.tmp"))
    assert code != 0
    assert len(temporary_paths) == 1
    assert str(temporary_paths[0]) in output
    assert "recovery" in output.lower()
    assert load_yaml(run_root / "status.yaml")["state"] == "planned"
    assert (artifact_root / "runs" / f".{run_id}.claim").is_dir()
    assert not (run_root / "outputs").exists()
    assert not tuple(run_root.glob(".outputs.*.staging"))
    process_mock.assert_not_called()


def test_success_status_parent_fsync_failure_retains_claim_and_reports_uncertainty(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = "success-status-uncertain"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0
    original_writer = core_cli._write_tabular_status_atomic
    original_fsync = os.fsync

    def status_writer(*, run_root_path: Path, status: dict[str, object]) -> None:
        if status.get("state") != "succeeded":
            original_writer(run_root_path=run_root_path, status=status)
            return

        def fail_directory_sync(descriptor: int) -> None:
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError("injected parent fsync failure")
            original_fsync(descriptor)

        with mock.patch.object(core_cli.os, "fsync", side_effect=fail_directory_sync):
            original_writer(run_root_path=run_root_path, status=status)

    with mock.patch.object(core_cli, "_write_tabular_status_atomic", side_effect=status_writer), mock.patch(
        "research_platform.core.cli.shutil.which", return_value="/bin/bash"
    ), mock.patch("research_platform.core.cli.subprocess.run", side_effect=_mocked_success):
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    run_root = artifact_root / "runs" / run_id
    assert code != 0
    assert "uncertain" in output.lower()
    assert (artifact_root / "runs" / f".{run_id}.claim").is_dir()
    assert (run_root / "outputs" / "transaction-manifest.json").is_file()


def test_post_rename_parent_close_failure_preserves_committed_outputs_and_claim(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = "promotion-parent-close-uncertain"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0
    run_root = artifact_root / "runs" / run_id
    final_root = run_root / "outputs"
    original_rename = transaction._atomic_no_replace_directory
    original_close = os.close
    rename_completed = False
    close_failure_injected = False

    def rename_then_mark(source: Path, destination: Path) -> None:
        nonlocal rename_completed
        original_rename(source, destination)
        rename_completed = True

    def fail_first_directory_close_after_rename(descriptor: int) -> None:
        nonlocal close_failure_injected
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if rename_completed and is_directory and not close_failure_injected:
            close_failure_injected = True
            original_close(descriptor)
            raise OSError("injected post-rename parent descriptor close failure")
        original_close(descriptor)

    with mock.patch.object(
        transaction,
        "_atomic_no_replace_directory",
        side_effect=rename_then_mark,
    ), mock.patch.object(
        transaction.os,
        "close",
        side_effect=fail_first_directory_close_after_rename,
    ), mock.patch(
        "research_platform.core.cli.shutil.which",
        return_value="/bin/bash",
    ), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=_mocked_success,
    ) as process_mock:
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    assert close_failure_injected
    assert code != 0
    assert "committed or uncertain" in output.lower()
    assert str(final_root) in output
    assert load_yaml(run_root / "status.yaml")["state"] == "running"
    assert (artifact_root / "runs" / f".{run_id}.claim").is_dir()
    assert {path.name for path in final_root.iterdir()} == {
        "split.json",
        "prep.json",
        "features.tsv",
        "transaction-manifest.json",
    }
    assert not tuple(run_root.glob(".outputs.*.staging"))
    process_mock.assert_called_once()


def test_source_is_revalidated_immediately_before_subprocess(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _, _, table_path = _write_workspace(workspace_root)
    run_id = "source-before-launch"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0

    def mutate_during_lookup(_: str) -> str:
        table_path.write_bytes(table_path.read_bytes() + b"record-005\t5\t0\n")
        return "/bin/bash"

    with mock.patch("research_platform.core.cli.shutil.which", side_effect=mutate_during_lookup), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=AssertionError("source drift immediately before launch must prevent execution"),
    ) as process_mock:
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    run_root = artifact_root / "runs" / run_id
    assert code != 0
    assert "changed after planning" in output.lower()
    assert load_yaml(run_root / "status.yaml")["state"] == "failed"
    assert not (run_root / "outputs").exists()
    assert not tuple(run_root.glob(".outputs.*.staging"))
    process_mock.assert_not_called()


def test_replaced_reviewed_script_cannot_supply_unreviewed_execution_bytes(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = "reviewed-script-replaced"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0
    original_inspector = core_cli._inspect_tabular_reviewed_plan
    replaced = False

    def inspect_then_replace(**kwargs: object):
        nonlocal replaced
        result = original_inspector(**kwargs)
        if kwargs.get("allow_claim") is True and not replaced:
            replaced = True
            run_root = result[2]
            replacement = run_root / ".unreviewed-execute.sh"
            replacement.write_bytes(b"#!/usr/bin/env bash\nexit 77\n")
            os.replace(replacement, run_root / "execute.sh")
        return result

    with mock.patch.object(core_cli, "_inspect_tabular_reviewed_plan", side_effect=inspect_then_replace), mock.patch(
        "research_platform.core.cli.shutil.which", return_value="/bin/bash"
    ), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=AssertionError("unreviewed script bytes must never be launched"),
    ) as process_mock:
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    assert replaced
    assert code != 0
    assert "reviewed" in output.lower() or "script" in output.lower()
    assert not (artifact_root / "runs" / run_id / "outputs").exists()
    process_mock.assert_not_called()


def test_unreviewed_split_strategy_is_rejected_without_output_or_claim_residue(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = "wrong-split-strategy"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0

    def write_unreviewed_split(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[list[str]]:
        output_root = _write_mock_outputs(kwargs)
        split_path = output_root / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["split_strategy"] = "random"
        split_path.write_text(
            json.dumps(split, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return subprocess.CompletedProcess(command, 0)

    with mock.patch(
        "research_platform.core.cli.shutil.which",
        return_value="/bin/bash",
    ), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=write_unreviewed_split,
    ) as process_mock:
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    run_root = artifact_root / "runs" / run_id
    assert code != 0
    assert "split output conflicts" in output.lower()
    assert load_yaml(run_root / "status.yaml")["state"] == "failed"
    assert not (run_root / "outputs").exists()
    assert not tuple(run_root.glob(".outputs.*.staging"))
    assert not (artifact_root / "runs" / f".{run_id}.claim").exists()
    process_mock.assert_called_once()


@pytest.mark.parametrize("mutation_point", ("after_contract", "after_seal"))
def test_source_mutation_around_sealing_prevents_promotion(
    tmp_path: Path,
    mutation_point: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _, _, table_path = _write_workspace(workspace_root)
    run_id = f"source-{mutation_point}"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0

    patch_target: str
    original: Callable[..., object]
    if mutation_point == "after_contract":
        patch_target = "_validate_tabular_scientific_contract"
        original = core_cli._validate_tabular_scientific_contract
    else:
        patch_target = "seal_tabular_staged_transaction"
        original = core_cli.seal_tabular_staged_transaction

    def mutate_after(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        table_path.write_bytes(table_path.read_bytes() + b"record-005\t5\t0\n")
        return result

    with mock.patch.object(core_cli, patch_target, side_effect=mutate_after), mock.patch(
        "research_platform.core.cli.shutil.which", return_value="/bin/bash"
    ), mock.patch("research_platform.core.cli.subprocess.run", side_effect=_mocked_success):
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    run_root = artifact_root / "runs" / run_id
    assert code != 0
    assert "changed after planning" in output.lower()
    assert load_yaml(run_root / "status.yaml")["state"] == "failed"
    assert not (run_root / "outputs").exists()
    assert not tuple(run_root.glob(".outputs.*.staging"))


def test_source_mutation_after_sealed_validation_is_rechecked_before_promotion(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _, _, table_path = _write_workspace(workspace_root)
    run_id = "source-after-sealed-validation"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0
    original_validator = core_cli.validate_sealed_tabular_transaction

    def validate_then_mutate(*args: object, **kwargs: object) -> None:
        original_validator(*args, **kwargs)
        table_path.write_bytes(table_path.read_bytes() + b"record-005\t5\t0\n")

    with mock.patch.object(
        core_cli,
        "validate_sealed_tabular_transaction",
        side_effect=validate_then_mutate,
    ), mock.patch(
        "research_platform.core.cli.shutil.which",
        return_value="/bin/bash",
    ), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=_mocked_success,
    ) as process_mock:
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    run_root = artifact_root / "runs" / run_id
    assert code != 0
    assert "changed after planning" in output.lower()
    assert load_yaml(run_root / "status.yaml")["state"] == "failed"
    assert not (run_root / "outputs").exists()
    assert not tuple(run_root.glob(".outputs.*.staging"))
    assert not (artifact_root / "runs" / f".{run_id}.claim").exists()
    process_mock.assert_called_once()


def test_staging_creation_cleanup_failure_reports_recovery_path_and_retains_claim(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = "staging-create-recovery"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0

    with mock.patch.object(
        transaction,
        "_fsync_directory",
        side_effect=OSError("injected staging parent sync failure"),
    ), mock.patch.object(
        transaction.shutil,
        "rmtree",
        side_effect=OSError("injected staging cleanup failure"),
    ), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=AssertionError("failed staging creation must prevent launch"),
    ) as process_mock:
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    run_root = artifact_root / "runs" / run_id
    residue = tuple(run_root.glob(".outputs.*.staging"))
    assert code != 0
    assert len(residue) == 1
    assert str(residue[0]) in output
    assert "recovery" in output.lower()
    assert (artifact_root / "runs" / f".{run_id}.claim").is_dir()
    assert not (run_root / "outputs").exists()
    process_mock.assert_not_called()


@pytest.mark.parametrize(
    "interrupt_target",
    ("_validate_tabular_scientific_contract", "seal_tabular_staged_transaction"),
)
def test_keyboard_interrupt_outside_subprocess_retains_claim_and_staging(
    tmp_path: Path,
    interrupt_target: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = f"interrupt-{interrupt_target.removeprefix('_')[:24]}"
    assert _run_cli(
        _preprocess_args(run_id, execute=False),
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )[0] == 0

    with mock.patch.object(core_cli, interrupt_target, side_effect=KeyboardInterrupt), mock.patch(
        "research_platform.core.cli.shutil.which", return_value="/bin/bash"
    ), mock.patch("research_platform.core.cli.subprocess.run", side_effect=_mocked_success):
        code, output = _run_cli(
            _preprocess_args(run_id, execute=True),
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    run_root = artifact_root / "runs" / run_id
    residue = tuple(run_root.glob(".outputs.*.staging"))
    assert code == 130
    assert "recovery" in output.lower()
    assert len(residue) == 1
    assert str(residue[0]) in output
    assert (artifact_root / "runs" / f".{run_id}.claim").is_dir()
    assert not (run_root / "outputs").exists()


@pytest.mark.parametrize(
    "case",
    ("project_root", "batch_path", "batch_digest", "selected_row"),
)
def test_evaluation_requires_exact_project_and_batch_source_identity(
    tmp_path: Path,
    case: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    project_root, batch_path, _ = _write_workspace(workspace_root)
    source_root = _plan_and_execute_train(
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        run_id=f"source-{case}",
    )

    if case == "batch_path":
        alternate = project_root / "manifests" / "alternate" / "default.tsv"
        alternate.parent.mkdir(parents=True)
        alternate.write_bytes(batch_path.read_bytes())

    def mutate(manifest: dict[str, object]) -> None:
        if case == "project_root":
            manifest["project"]["root"] = "project/other-tabular"  # type: ignore[index]
        elif case == "batch_path":
            manifest["batch"]["path"] = "project/test-tabular/manifests/alternate/default.tsv"  # type: ignore[index]
        elif case == "batch_digest":
            manifest["source_digests"]["batch_sha256"] = "0" * 64  # type: ignore[index]
        else:
            manifest["batch"]["selected_row"] = {  # type: ignore[index]
                "feature_table": "test-tabular/other.tsv",
                "target_column": "binary_target",
            }

    _reattest_source_run(source_root, mutate)
    evaluation_id = f"evaluate-{case}"
    args = [
        "run",
        "plan",
        "evaluate",
        "model",
        "--project",
        "test-tabular",
        "--input-run",
        f"source-{case}",
        "--run-id",
        evaluation_id,
    ]
    with mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=AssertionError("invalid upstream identity must not launch"),
    ) as process_mock:
        code, output = _run_cli(args, workspace_root=workspace_root, artifact_root=artifact_root)

    assert code != 0
    assert "input" in output.lower() or "training" in output.lower()
    assert not (artifact_root / "runs" / evaluation_id).exists()
    process_mock.assert_not_called()


@pytest.mark.parametrize("race_entry", ("claim", "staging"))
def test_evaluation_planning_rejects_source_recovery_race_after_transaction_validation(
    tmp_path: Path,
    race_entry: str,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    source_id = f"source-race-{race_entry}"
    source_root = _plan_and_execute_train(
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        run_id=source_id,
    )
    evaluation_id = f"evaluate-race-{race_entry}"
    original_validator = core_cli.validate_committed_tabular_transaction
    injected_path: Path | None = None

    def validate_then_inject(*args: object, **kwargs: object):
        nonlocal injected_path
        result = original_validator(*args, **kwargs)
        if injected_path is None:
            if race_entry == "claim":
                injected_path = artifact_root / "runs" / f".{source_id}.claim"
            else:
                injected_path = source_root / ".outputs.race.staging"
            injected_path.mkdir()
        return result

    with mock.patch.object(
        core_cli,
        "validate_committed_tabular_transaction",
        side_effect=validate_then_inject,
    ), mock.patch(
        "research_platform.core.cli.subprocess.run",
        side_effect=AssertionError("a raced source recovery entry must prevent evaluation planning"),
    ) as process_mock:
        code, output = _run_cli(
            [
                "run",
                "plan",
                "evaluate",
                "model",
                "--project",
                "test-tabular",
                "--input-run",
                source_id,
                "--run-id",
                evaluation_id,
            ],
            workspace_root=workspace_root,
            artifact_root=artifact_root,
        )

    assert code != 0
    assert injected_path is not None and injected_path.is_dir()
    assert race_entry in output.lower() or "recovery" in output.lower()
    assert not (artifact_root / "runs" / evaluation_id).exists()
    process_mock.assert_not_called()


def test_slurm_evaluation_keeps_legacy_plan_only_contract(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    source_id = "source-for-slurm"
    _plan_and_execute_train(
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        run_id=source_id,
    )
    run_id = "slurm-evaluation-plan"
    code, _ = _run_cli(
        [
            "run",
            "slurm",
            "evaluate",
            "model",
            "--project",
            "test-tabular",
            "--input-run",
            source_id,
            "--run-id",
            run_id,
        ],
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )

    run_root = artifact_root / "runs" / run_id
    manifest = load_yaml(run_root / "run-manifest.yaml")
    reviewed_script = (run_root / "execute.sh").read_text(encoding="utf-8")
    assert code == 0
    assert manifest["execution"]["mode"] == "slurm"
    assert "output_transaction" not in manifest
    assert (run_root / "outputs").is_dir()
    assert not any((run_root / "outputs").iterdir())
    assert "model evaluate" in reviewed_script
    assert "--expected-table-sha256" not in reviewed_script
    assert "--expected-split-sha256" not in reviewed_script
    assert "--expected-model-sha256" not in reviewed_script


def test_real_local_configured_analysis_commits_one_attested_report(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = workspace_root / "artifacts"
    _write_workspace(workspace_root)
    run_id = "real-feature-correlation"

    code, output = _run_cli(
        [
            "run",
            "local",
            "analysis",
            "tabular",
            "--project",
            "test-tabular",
            "--analysis",
            "feature-correlation",
            "--run-id",
            run_id,
            "--execute",
        ],
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )

    run_root = artifact_root / "runs" / run_id
    output_root = run_root / "outputs"
    assert code == 0, output
    assert {path.name for path in output_root.iterdir()} == {
        "feature-correlation.json",
        "transaction-manifest.json",
    }
    report = json.loads((output_root / "feature-correlation.json").read_text(encoding="utf-8"))
    receipt_bytes = (output_root / "transaction-manifest.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    record = receipt["outputs"][0]
    report_bytes = (output_root / record["relative_path"]).read_bytes()
    assert report == {
        "kind": "correlation",
        "method": "pearson",
        "n": 4,
        "r": pytest.approx(0.4472135954999579),
        "table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
        "x": "feature_a",
        "y": "binary_target",
    }
    assert record["byte_size"] == len(report_bytes)
    assert record["sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert load_yaml(run_root / "status.yaml")["state"] == "succeeded"
    assert not tuple(run_root.glob(".outputs.*.staging"))


def test_stable_control_file_read_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "status.yaml"
    os.mkfifo(fifo)

    with pytest.raises(core_cli.RunLifecycleError, match="not a regular file"):
        core_cli._stable_regular_file_bytes(fifo, label="Run status")
