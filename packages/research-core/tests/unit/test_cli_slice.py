from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
PUBLIC_PROJECT_OVERLAYS = (
    "project-example",
    "project-pilot-bids",
    "project-pilot-tabular",
    "project-template",
)
TABULAR_PREDICTOR_COLUMNS = [
    "feature_a",
    "feature_b",
    "feature_c",
    "measure_x",
    "measure_y",
    "feature_d",
]
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.core import cli as core_cli
from research_platform.core.cli import (
    _compute_notebook_bootstrap_stamp,
    _merge_scaffold_runtime_compute_defaults,
    _snakemake_core_args,
    _snakemake_resource_args,
    main,
)
from research_platform.core.config import (
    load_project_bundle,
    load_yaml,
    parse_yaml,
    resolve_workspace_hpc_runtime_default,
    validate_tabular_feature_columns,
    validate_project_bundle,
    write_yaml,
)


def _analysis_cli_commands(script_text: str) -> list[list[str]]:
    return [
        shlex.split(line)
        for line in script_text.splitlines()
        if "research_platform.analysis.cli" in line
    ]


def _command_option_values(command: list[str], option: str) -> list[str]:
    if option not in command:
        return []
    start = command.index(option) + 1
    values: list[str] = []
    for value in command[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def _filesystem_snapshot(root: Path) -> dict[str, tuple[object, ...]]:
    """Return a deterministic, symlink-safe snapshot without mutating *root*."""

    if not os.path.lexists(root):
        return {}
    snapshot: dict[str, tuple[object, ...]] = {}

    def visit(path: Path, relative: str) -> None:
        metadata = path.lstat()
        permissions = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            snapshot[relative] = ("symlink", permissions, os.readlink(path))
            return
        if stat.S_ISDIR(metadata.st_mode):
            snapshot[relative] = ("directory", permissions)
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                child_relative = child.name if relative == "." else f"{relative}/{child.name}"
                visit(child, child_relative)
            return
        if stat.S_ISREG(metadata.st_mode):
            snapshot[relative] = (
                "file",
                permissions,
                metadata.st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            return
        snapshot[relative] = ("special", permissions, stat.S_IFMT(metadata.st_mode))

    visit(root, ".")
    return snapshot


def _write_mock_tabular_scientific_outputs(execution_kwargs: object) -> Path:
    """Materialize a deterministic, structurally valid mocked tabular run."""

    if not isinstance(execution_kwargs, dict):
        raise AssertionError("transactional tabular execution must pass keyword arguments")
    environment = execution_kwargs.get("env")
    if not isinstance(environment, dict):
        raise AssertionError("transactional tabular execution must pass an environment")
    raw_output_root = environment.get("RP_TABULAR_OUTPUT_ROOT")
    if not isinstance(raw_output_root, str) or not raw_output_root:
        raise AssertionError("transactional tabular execution must set RP_TABULAR_OUTPUT_ROOT")
    output_root = Path(raw_output_root)
    run_root = output_root.parent
    manifest = load_yaml(run_root / "run-manifest.yaml")
    workflow = manifest["workflow"]

    def write_json(relative_path: str, value: dict[str, object]) -> None:
        (output_root / relative_path).write_text(
            json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    if workflow == {"action": "analysis", "target": "tabular"}:
        analysis = manifest["analysis"]
        config = analysis["config"]
        kind = analysis["kind"]
        report: dict[str, object] = {"kind": kind, "table": config["input_table"]}
        if kind == "correlation":
            report.update({"method": config.get("method") or "pearson", "x": config["x"], "y": config["y"], "n": 4, "r": 0.5})
        elif kind == "summary_table":
            columns = list(config["columns"])
            report.update({"columns": columns, "summaries": {column: {"n": 4, "mean": 1.0, "std": 0.5, "min": 0.0, "max": 2.0} for column in columns}})
        elif kind == "linear_model":
            predictors = list(config["predictors"])
            report.update({"outcome": config["outcome"], "predictors": predictors, "n": len(predictors) + 2, "coefficients": {"intercept": 0.0, **{name: 0.5 for name in predictors}}})
        elif kind == "anova":
            predictors = list(config.get("predictors", []))
            group = config.get("group") or predictors[0]
            summary = {"n": 2, "mean": 1.0, "std": 0.5, "min": 0.5, "max": 1.5}
            report.update({"outcome": config["outcome"], "group": group, "n": 4, "groups": {"a": summary, "b": summary}, "f": 1.0, "df_between": 1, "df_within": 2})
        elif kind == "mixed_effects":
            predictors = list(config["predictors"])
            report.update({"outcome": config["outcome"], "predictors": predictors, "engine": "summary-only", "n": 4})
            if config.get("group"):
                report.update({"group": config["group"], "groups": {"a": {"n": 4, "mean": 1.0, "std": 0.5, "min": 0.0, "max": 2.0}}})
        write_json(
            f"{analysis['name']}.json",
            report,
        )
        return run_root

    predictor = manifest["predictor_contract"]
    target = predictor["target_column"]
    features = list(predictor["feature_columns"])
    if workflow == {"action": "evaluate", "target": "model"}:
        input_run = manifest["input_run"]
        source_output_root = run_root.parent / input_run["run_id"] / "outputs"
        source_split = json.loads((source_output_root / "split.json").read_text(encoding="utf-8"))
        test_rows = list(source_split["test_rows"])
        with (source_output_root / "features.tsv").open("r", encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle, delimiter="\t"))
        predictions = []
        for index, row_index in enumerate(test_rows):
            actual = int(float(source_rows[row_index][target]))
            predictions.append({"actual": actual, "predicted": actual, "probability": 0.9 if actual else 0.1, "row_number": index})
        true_positive = sum(item["actual"] == 1 for item in predictions)
        true_negative = len(predictions) - true_positive
        write_json(
            "evaluation.json",
            {
                "kind": "logistic_regression_evaluation",
                "target_column": target,
                "feature_columns": features,
                "table_path": str(source_output_root / "features.tsv"),
                "metrics": {"test_count": len(test_rows), "accuracy": 1.0, "log_loss": 0.1, "precision": 1.0, "recall": 1.0},
                "confusion_matrix": {"tn": true_negative, "fp": 0, "fn": 0, "tp": true_positive},
                "predictions": predictions,
            },
        )
        return run_root

    workspace_root = Path(str(execution_kwargs.get("cwd")))
    source_reference = manifest["dataset"]["feature_table"]
    source_path = Path(source_reference)
    if not source_path.is_absolute():
        source_path = workspace_root / source_path
    delimiter = "," if source_path.suffix.casefold() == ".csv" else "\t"
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        source_reader = csv.DictReader(handle, delimiter=delimiter)
        if source_reader.fieldnames is None:
            raise AssertionError("mocked tabular source must contain a header")
        source_columns = list(source_reader.fieldnames)
        source_rows = [dict(row) for row in source_reader]
    row_count = len(source_rows)
    train_rows = list(range(max(1, row_count - 1)))
    test_rows = list(range(max(1, row_count - 1), row_count))

    write_json(
        "split.json",
        {
            "kind": "tabular_split",
            "table_path": source_reference,
            "target_column": target,
            "row_count": row_count,
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
            "table_path": source_reference,
            "target_column": target,
            "feature_columns": features,
            "statistics": {
                feature: {"mean": float(index + 1), "std": 1.0}
                for index, feature in enumerate(features)
            },
        },
    )
    columns = [*source_columns, "split_set"]
    membership = {index: "train" for index in train_rows} | {index: "test" for index in test_rows}
    feature_rows = [
        [*(row[column] for column in source_columns), membership[index]]
        for index, row in enumerate(source_rows)
    ]
    (output_root / "features.tsv").write_text(
        "\t".join(columns)
        + "\n"
        + "".join("\t".join(row) + "\n" for row in feature_rows),
        encoding="utf-8",
        newline="\n",
    )
    if workflow == {"action": "train", "target": "model"}:
        write_json(
            "model.json",
            {
                "kind": manifest["tool"]["model"],
                "target_column": target,
                "feature_columns": features,
                "table_path": "outputs/features.tsv",
                "weights": {feature: 0.0 for feature in features},
                "intercept": 0.0,
                "learning_rate": manifest["settings"]["model"].get("learning_rate", 0.2),
                "iterations": manifest["settings"]["model"].get("iterations", 350),
                "training_metrics": {"accuracy": 1.0},
            },
        )
    return run_root


class CoreCliSliceTests(unittest.TestCase):
    def _run_cli(self, args: list[str], artifact_root: Path, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(WORKSPACE_ROOT),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_HPC_HOST": "example-hpc",
            "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
        }
        unset_keys: list[str] = []
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                    unset_keys.append(key)
                    continue
                env[key] = value
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False):
            for key in unset_keys:
                os.environ.pop(key, None)
            with redirect_stdout(buffer):
                exit_code = main(args)
        return exit_code, buffer.getvalue()

    def _run_cli_for_workspace(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        artifact_root: Path,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(workspace_root),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_HPC_HOST": "example-hpc",
            "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
        }
        unset_keys: list[str] = []
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                    unset_keys.append(key)
                    continue
                env[key] = value
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False):
            for key in unset_keys:
                os.environ.pop(key, None)
            with redirect_stdout(buffer):
                exit_code = main(args)
        return exit_code, buffer.getvalue()

    def _run_cli_system_exit_for_workspace(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        artifact_root: Path,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(workspace_root),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_HPC_HOST": "example-hpc",
            "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
        }
        unset_keys: list[str] = []
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                    unset_keys.append(key)
                    continue
                env[key] = value
        with mock.patch.dict(os.environ, env, clear=False):
            for key in unset_keys:
                os.environ.pop(key, None)
            with self.assertRaises(SystemExit) as exc_info:
                main(args)
        return str(exc_info.exception)

    def _run_cli_failure_for_workspace(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        artifact_root: Path,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[object, str]:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(workspace_root),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_HPC_HOST": "example-hpc",
            "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
        }
        if extra_env:
            env.update(extra_env)
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as exc_info:
                main(args)
        code = exc_info.exception.code
        if isinstance(code, str):
            return 1, code + buffer.getvalue()
        return code, buffer.getvalue()

    def _execute_mocked_tabular_success(
        self,
        args: list[str],
        *,
        artifact_root: Path,
        workspace_root: Path = WORKSPACE_ROOT,
    ) -> tuple[int, str]:
        def fake_run(
            command: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[list[str]]:
            _write_mock_tabular_scientific_outputs(kwargs)
            return subprocess.CompletedProcess(command, 0)

        with mock.patch("research_platform.core.cli.shutil.which", return_value="/bin/bash"):
            with mock.patch("research_platform.core.cli.subprocess.run", side_effect=fake_run):
                return self._run_cli_for_workspace(
                    args,
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

    def _write_ssh_profile_config(self, root: Path) -> Path:
        config_path = root / "ssh-profiles.yaml"
        write_yaml(
            config_path,
            {
                "profiles": {
                    "interactive-login": {
                        "host": "cluster.example",
                        "user": "alice",
                    },
                    "robot-login": {
                        "host": "cluster.example",
                        "user": "robot",
                    },
                }
            },
        )
        return config_path

    def _write_bids_init_workspace(self, workspace_root: Path, *, workspace_hpc: dict[str, object] | None = None) -> None:
        pipeline_root = workspace_root / "pipelines" / "preprocess-bids"
        for path in (
            pipeline_root / "config",
            pipeline_root / "profiles" / "local",
            pipeline_root / "profiles" / "slurm",
            workspace_root / "ops" / "sync" / "rsync",
        ):
            path.mkdir(parents=True, exist_ok=True)

        workspace_config: dict[str, object] = {
            "paths": {
                "artifacts_root": "./artifacts",
                "datasets_root": "./datasets",
                "ops_root": "./ops",
            },
            "repos": {
                "project_root": "./project",
                "pipelines_root": "./pipelines",
            },
            "projects": {"default": "project-default"},
        }
        if workspace_hpc is not None:
            workspace_config["hpc"] = workspace_hpc
        write_yaml(workspace_root / "WORKSPACE.yaml", workspace_config)
        write_yaml(
            pipeline_root / "config" / "defaults.yaml",
            {
                "workflow": {
                    "default_target": "fmripost_aroma",
                    "rule_name": "fmripost_aroma",
                    "execution_rule_name": "fmripost_aroma_unit",
                },
                "planner": {
                    "outputs": {
                        "runtime_plan_filename": "fmripost-aroma-plan.json",
                        "command_script_filename": "run-fmripost-aroma.sh",
                        "completion_marker_filename": "fmripost-aroma-complete.txt",
                        "output_data_dirname": "fmripost_aroma",
                    }
                },
            },
        )
        write_yaml(pipeline_root / "profiles" / "local" / "config.yaml", {"profile": {"name": "local"}})
        write_yaml(pipeline_root / "profiles" / "slurm" / "config.yaml", {"profile": {"name": "slurm"}})

    def _write_tabular_init_workspace(self, workspace_root: Path, *, workspace_hpc: dict[str, object] | None = None) -> None:
        workspace_config: dict[str, object] = {
            "paths": {
                "artifacts_root": "./artifacts",
                "datasets_root": "./datasets",
                "ops_root": "./ops",
            },
            "repos": {
                "project_root": "./project",
                "pipelines_root": "./pipelines",
            },
            "projects": {"default": "project-default"},
        }
        if workspace_hpc is not None:
            workspace_config["hpc"] = workspace_hpc
        write_yaml(workspace_root / "WORKSPACE.yaml", workspace_config)

    def _write_bids_validation_fixture(self, workspace_root: Path, *, derivative_name: str) -> dict[str, object]:
        project_root = workspace_root / "project" / "test-bids"
        dataset_root = workspace_root / "datasets" / "ds-bids-fixture"
        pipeline_root = workspace_root / "pipelines" / "preprocess-bids"

        (project_root / "manifests" / "batches").mkdir(parents=True, exist_ok=True)
        (dataset_root / "derivatives" / derivative_name).mkdir(parents=True, exist_ok=True)
        (pipeline_root / "profiles" / "local").mkdir(parents=True, exist_ok=True)
        (pipeline_root / "profiles" / "slurm").mkdir(parents=True, exist_ok=True)

        write_yaml(
            pipeline_root / "config" / "defaults.yaml",
            {
                "workflow": {
                    "default_target": "fmripost_aroma",
                    "rule_name": "fmripost_aroma",
                    "execution_rule_name": "fmripost_aroma_unit",
                },
                "planner": {
                    "outputs": {
                        "runtime_plan_filename": "fmripost-aroma-plan.json",
                        "command_script_filename": "run-fmripost-aroma.sh",
                        "completion_marker_filename": "fmripost-aroma-complete.txt",
                        "output_data_dirname": "fmripost_aroma",
                    }
                },
            },
        )
        write_yaml(pipeline_root / "profiles" / "local" / "config.yaml", {"profile": {"name": "local"}})
        write_yaml(pipeline_root / "profiles" / "slurm" / "config.yaml", {"profile": {"name": "slurm"}})
        (project_root / "manifests" / "batches" / "fmripost_aroma_subject_sessions.tsv").write_text(
            "subject_id\tsession_id\nsub-001\tses-01\n",
            encoding="utf-8",
        )

        return {
            "workspace": {
                "paths": {
                    "artifacts_root": "./artifacts",
                    "datasets_root": "./datasets",
                    "ops_root": "./ops",
                },
                "repos": {
                    "project_root": "./project",
                    "pipelines_root": "./pipelines",
                },
            },
            "project_root": project_root,
            "project": {"name": "test-bids"},
            "dataset": {
                "dataset": {
                    "primary": "ds-bids-fixture",
                    "input_derivative": derivative_name,
                }
            },
            "compute": {"compute": {"default_profile": "local"}},
            "preprocessing": {
                "preprocessing": {
                    "slice": "bids",
                    "pipeline": "preprocess-bids",
                    "tool": "fmripost_aroma",
                    "tool_adapter": "research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter",
                    "input_derivative": derivative_name,
                    "default_batch": "fmripost_aroma_subject_sessions",
                    "local_profile": "local",
                    "slurm_profile": "slurm",
                    "tool_options": {"denoising_method": "nonaggr", "dummy_scans": 0, "low_mem": False},
                    "publish_back": {"default_policy": "never"},
                }
            },
            "models": {},
        }

    def _write_bids_discovery_workspace(self, workspace_root: Path) -> Path:
        project_root = workspace_root / "project" / "test-bids"
        dataset_root = workspace_root / "datasets" / "ds-bids-fixture"
        derivative_root = dataset_root / "derivatives" / "deepprep-bold" / "sub-001" / "ses-01" / "func"
        pipeline_root = workspace_root / "pipelines" / "preprocess-bids"
        for path in (
            project_root / "config",
            project_root / "manifests" / "batches",
            pipeline_root / "config",
            pipeline_root / "profiles" / "local",
            pipeline_root / "profiles" / "slurm",
            derivative_root,
            workspace_root / "ops" / "sync" / "rsync",
            workspace_root / "ops" / "slurm" / "job_templates",
        ):
            path.mkdir(parents=True, exist_ok=True)

        write_yaml(
            workspace_root / "WORKSPACE.yaml",
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
                "projects": {"default": "test-bids"},
            },
        )
        write_yaml(project_root / "project.yaml", {"name": "test-bids", "version": "0.1.0"})
        write_yaml(
            project_root / "config" / "dataset.yaml",
            {"dataset": {"primary": "ds-bids-fixture", "input_derivative": "deepprep-bold"}},
        )
        write_yaml(
            project_root / "config" / "compute.yaml",
            {
                "compute": {
                    "default_profile": "local",
                    "slurm": {
                        "cpus": 4,
                        "mem": "16G",
                        "time": "02:00:00",
                        "ssh_host": "${RP_HPC_HOST:-}",
                        "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                        "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                    },
                }
            },
        )
        write_yaml(
            project_root / "config" / "preprocessing.yaml",
            {
                "preprocessing": {
                    "slice": "bids",
                    "pipeline": "preprocess-bids",
                    "tool": "fmripost_aroma",
                    "tool_adapter": "research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter",
                    "input_derivative": "deepprep-bold",
                    "default_batch": "discovered_batch",
                    "local_profile": "local",
                    "slurm_profile": "slurm",
                    "tool_options": {"denoising_method": "nonaggr", "dummy_scans": 0, "low_mem": False},
                    "publish_back": {"default_policy": "never"},
                }
            },
        )
        write_yaml(
            pipeline_root / "config" / "defaults.yaml",
            {
                "workflow": {
                    "default_target": "fmripost_aroma",
                    "rule_name": "fmripost_aroma",
                    "execution_rule_name": "fmripost_aroma_unit",
                },
                "planner": {
                    "outputs": {
                        "runtime_plan_filename": "fmripost-aroma-plan.json",
                        "command_script_filename": "run-fmripost-aroma.sh",
                        "completion_marker_filename": "fmripost-aroma-complete.txt",
                        "output_data_dirname": "fmripost_aroma",
                    }
                },
            },
        )
        write_yaml(pipeline_root / "profiles" / "local" / "config.yaml", {"profile": {"name": "local"}})
        write_yaml(pipeline_root / "profiles" / "slurm" / "config.yaml", {"profile": {"name": "slurm"}})
        (workspace_root / "ops" / "sync" / "rsync" / "exclude.txt").write_text("", encoding="utf-8")
        (workspace_root / "ops" / "slurm" / "job_templates" / "sbatch.job.sh").write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "#SBATCH --job-name={{ job_name }}",
                    "#SBATCH --cpus-per-task={{ cpus }}",
                    "#SBATCH --mem={{ mem }}",
                    "#SBATCH --time={{ time }}",
                    "#SBATCH --output={{ log_out }}",
                    "#SBATCH --error={{ log_err }}",
                    "",
                    "{{ command }}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (
            derivative_root / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
        ).write_text("", encoding="utf-8")
        return project_root

    def _write_tabular_slurm_workspace(self, workspace_root: Path) -> Path:
        project_root = workspace_root / "project" / "test-tabular"
        feature_root = workspace_root / "datasets" / "ds-tabular" / "derivatives" / "features" / "test-tabular"
        for path in (
            project_root / "config",
            project_root / "manifests" / "batches",
            workspace_root / "ops" / "sync" / "rsync",
            workspace_root / "ops" / "slurm" / "job_templates",
            workspace_root / "packages" / "research-core",
            workspace_root / "packages" / "research-hpc",
            workspace_root / "packages" / "research-analysis",
            workspace_root / "packages" / "research-ml",
            feature_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

        write_yaml(
            workspace_root / "WORKSPACE.yaml",
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
            {
                "preprocessing": {
                    "slice": "tabular",
                    "default_batch": "default",
                }
            },
        )
        write_yaml(
            project_root / "config" / "models.yaml",
            {
                "models": {
                    "default": {
                        "kind": "logistic_regression",
                        "feature_columns": ["feature_a"],
                    }
                }
            },
        )
        (project_root / "manifests" / "batches" / "default.tsv").write_text(
            "feature_table\ttarget_column\n"
            "test-tabular/toy_features.tsv\tbinary_target\n",
            encoding="utf-8",
        )
        (feature_root / "toy_features.tsv").write_text(
            "record_id\tfeature_a\tbinary_target\n"
            "record-001\t1\t0\n",
            encoding="utf-8",
        )
        (workspace_root / "ops" / "sync" / "rsync" / "exclude.txt").write_text("", encoding="utf-8")
        (workspace_root / "ops" / "sync" / "rsync" / "exclude.common.txt").write_text("", encoding="utf-8")
        (workspace_root / "ops" / "sync" / "rsync" / "exclude.tabular-ml.txt").write_text("", encoding="utf-8")
        (workspace_root / "ops" / "slurm" / "job_templates" / "sbatch.job.sh").write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "#SBATCH --job-name={{ job_name }}",
                    "#SBATCH --cpus-per-task={{ cpus }}",
                    "#SBATCH --mem={{ mem }}",
                    "#SBATCH --time={{ time }}",
                    "#SBATCH --output={{ log_out }}",
                    "#SBATCH --error={{ log_err }}",
                    "",
                    "{{ command }}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return project_root

    def test_public_tabular_runs_reject_ambiguous_multirow_batches_before_side_effects(self) -> None:
        command_cases = (
            ("preprocess-plan", ["run", "plan", "preprocess", "tabular"]),
            ("preprocess-local-dry", ["run", "local", "preprocess", "tabular", "--dry-run"]),
            ("preprocess-local-execute", ["run", "local", "preprocess", "tabular", "--execute"]),
            ("preprocess-slurm", ["run", "slurm", "preprocess", "tabular"]),
            ("train-plan", ["run", "plan", "train", "model"]),
            ("train-local-dry", ["run", "local", "train", "model", "--dry-run"]),
            ("train-local-execute", ["run", "local", "train", "model", "--execute"]),
            ("train-slurm", ["run", "slurm", "train", "model"]),
            (
                "evaluate-plan",
                ["run", "plan", "evaluate", "model", "--input-run", "must-not-be-loaded"],
            ),
            (
                "evaluate-local-dry",
                ["run", "local", "evaluate", "model", "--input-run", "must-not-be-loaded", "--dry-run"],
            ),
            (
                "evaluate-local-execute",
                ["run", "local", "evaluate", "model", "--input-run", "must-not-be-loaded", "--execute"],
            ),
            (
                "evaluate-slurm",
                ["run", "slurm", "evaluate", "model", "--input-run", "must-not-be-loaded"],
            ),
        )

        for case_name, command in command_cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmp_dir:
                workspace_root = Path(tmp_dir)
                artifact_root = workspace_root / "artifacts"
                project_root = self._write_tabular_slurm_workspace(workspace_root)
                batch_path = project_root / "manifests" / "batches" / "ambiguous.tsv"
                batch_path.write_text(
                    "feature_table\ttarget_column\n"
                    "test-tabular/toy_features.tsv\tbinary_target\n"
                    "test-tabular/toy_features.tsv\tbinary_target\n",
                    encoding="utf-8",
                    newline="\n",
                )
                run_id = f"ambiguous-{case_name}"
                full_command = [
                    *command,
                    "--project",
                    "test-tabular",
                    "--batch",
                    "ambiguous",
                    "--run-id",
                    run_id,
                ]

                with (
                    mock.patch.object(core_cli, "_run_paths") as run_paths_mock,
                    mock.patch.object(core_cli, "acquire_execution_claim") as claim_mock,
                    mock.patch.object(core_cli, "_load_tabular_input_run") as local_input_mock,
                    mock.patch.object(core_cli, "_load_legacy_tabular_input_run") as legacy_input_mock,
                    mock.patch.object(core_cli.subprocess, "run") as subprocess_mock,
                    mock.patch.object(core_cli, "execute_stage_plan") as stage_mock,
                    mock.patch.object(core_cli, "execute_submit_plan") as submit_mock,
                ):
                    error = self._run_cli_system_exit_for_workspace(
                        full_command,
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

                payload = json.loads(error)
                message = payload["error"]
                self.assertIn("ambiguous", message)
                self.assertIn("2 data rows", message)
                self.assertIn("exactly one data row", message)
                self.assertIn("implicit row selection", message)
                self.assertIn("Cartesian expansion", message)
                self.assertIn("named one-row batch", message)
                self.assertIn("--batch", message)
                self.assertNotIn("Traceback", error)
                run_paths_mock.assert_not_called()
                claim_mock.assert_not_called()
                local_input_mock.assert_not_called()
                legacy_input_mock.assert_not_called()
                subprocess_mock.assert_not_called()
                stage_mock.assert_not_called()
                submit_mock.assert_not_called()
                self.assertFalse(os.path.lexists(artifact_root / "runs" / run_id))
                self.assertFalse(os.path.lexists(artifact_root / "runs" / f".{run_id}.claim"))

    def test_public_tabular_empty_batch_retains_existing_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            batch_path = project_root / "manifests" / "batches" / "empty.tsv"
            batch_path.write_text(
                "feature_table\ttarget_column\n",
                encoding="utf-8",
                newline="\n",
            )

            with (
                mock.patch.object(core_cli, "_run_paths") as run_paths_mock,
                mock.patch.object(core_cli.subprocess, "run") as subprocess_mock,
            ):
                error = self._run_cli_system_exit_for_workspace(
                    [
                        "run",
                        "plan",
                        "preprocess",
                        "tabular",
                        "--project",
                        "test-tabular",
                        "--batch",
                        "empty",
                        "--run-id",
                        "empty-batch",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )

            self.assertIn("Batch manifest is empty:", error)
            self.assertNotIn("exactly one data row", error)
            run_paths_mock.assert_not_called()
            subprocess_mock.assert_not_called()
            self.assertFalse(os.path.lexists(artifact_root / "runs" / "empty-batch"))

    def test_config_validate_succeeds_for_pilot_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exit_code, output = self._run_cli(["config", "validate"], Path(tmp_dir) / "artifacts")

        self.assertEqual(exit_code, 0)
        self.assertTrue(json.loads(output)["valid"])

    def test_tabular_feature_column_config_is_ordered_and_fail_closed(self) -> None:
        self.assertEqual(
            validate_tabular_feature_columns(["feature_b", "feature_a"]),
            ["feature_b", "feature_a"],
        )

        invalid_values = (
            None,
            "feature_a",
            [],
            [""],
            ["   "],
            ["feature_a", 1],
            ["feature_a", "feature_a"],
            ["split_set"],
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_tabular_feature_columns(value)

    def test_tabular_project_validation_rejects_missing_feature_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_tabular_slurm_workspace(workspace_root)
            models_path = workspace_root / "project" / "test-tabular" / "config" / "models.yaml"
            models_document = load_yaml(models_path)
            del models_document["models"]["default"]["feature_columns"]
            write_yaml(models_path, models_document)

            bundle = load_project_bundle("test-tabular", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)

        self.assertIn(
            "config/models.yaml models.default.feature_columns must be an ordered, nonempty YAML sequence of strings.",
            errors,
        )

    def test_all_indexed_public_project_overlays_validate(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "project"],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        indexed_overlays = sorted(
            {
                parts[1]
                for path in result.stdout.split("\0")
                if path and len(parts := Path(path).parts) > 1
            }
        )

        self.assertEqual(indexed_overlays, list(PUBLIC_PROJECT_OVERLAYS))
        for project_name in PUBLIC_PROJECT_OVERLAYS:
            with self.subTest(project=project_name):
                bundle = load_project_bundle(project_name, WORKSPACE_ROOT)
                self.assertEqual(validate_project_bundle(bundle, root=WORKSPACE_ROOT), [])
                if project_name != "project-pilot-bids":
                    self.assertEqual(
                        bundle["models"]["models"]["default"]["feature_columns"],
                        TABULAR_PREDICTOR_COLUMNS,
                    )

    def test_validate_project_bundle_accepts_deepprep_bold_for_fmripost_aroma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            bundle = self._write_bids_validation_fixture(workspace_root, derivative_name="deepprep-bold")

            errors = validate_project_bundle(bundle, root=workspace_root)

        self.assertEqual(errors, [])

    def test_validate_project_bundle_accepts_fmriprep_for_fmripost_aroma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            bundle = self._write_bids_validation_fixture(workspace_root, derivative_name="fmriprep")

            errors = validate_project_bundle(bundle, root=workspace_root)

        self.assertEqual(errors, [])

    def test_batch_show_reads_tsv_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exit_code, output = self._run_cli(["batch", "show"], Path(tmp_dir) / "artifacts")

        payload = json.loads(output)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["name"], "fmripost_aroma_subject_sessions")
        self.assertEqual(payload["row_count"], 2)
        self.assertEqual(payload["columns"], ["subject_id", "session_id"])
        self.assertEqual(
            payload["rows"],
            [
                {"subject_id": "sub-001", "session_id": "ses-01"},
                {"subject_id": "sub-002", "session_id": "ses-01"},
            ],
        )

    def test_batch_discover_bids_bootstraps_missing_default_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = self._write_bids_discovery_workspace(workspace_root)
            buffer = io.StringIO()
            with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=False):
                with redirect_stdout(buffer):
                    exit_code = main(["batch", "discover", "bids"])
            batch_path = project_root / "manifests" / "batches" / "discovered_batch.tsv"
            batch_text = batch_path.read_text(encoding="utf-8")

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["batch"], "discovered_batch")
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(
            batch_text,
            "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
        )

    def test_project_init_bids_preprocess_scaffolds_valid_project_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "deepprep-bold"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            exit_code, output = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-preprocess",
                    "--project",
                    "project-new-bids",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "fmripost_aroma",
                    "--task-id",
                    "rest",
                    "--remote-study-root",
                    "/remote/studies/demo",
                    "--remote-derivative-root",
                    "/remote/studies/demo/derivatives/deepprep-bold",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            bundle = load_project_bundle("project-new-bids", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)
            batch_text = (
                workspace_root / "project" / "project-new-bids" / "manifests" / "batches" / "default.tsv"
            ).read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])
        self.assertIn("Initialized BIDS preprocessing project: project-new-bids", output)
        self.assertIn("Created files", output)
        self.assertIn("project/project-new-bids/config/preprocessing.yaml", output)
        self.assertIn("rp run submit preprocess bids --project project-new-bids --discover --run-id project-new-bids-submit", output)
        self.assertEqual(batch_text, "subject_id\tsession_id\ttask_id\trun_id\n")
        self.assertEqual(bundle["dataset"]["dataset"]["remote_bids_root"], "/remote/studies/demo")
        self.assertEqual(
            bundle["dataset"]["dataset"]["remote_input_derivative_root"],
            "/remote/studies/demo/derivatives/deepprep-bold",
        )
        self.assertEqual(bundle["preprocessing"]["preprocessing"]["task_id"], "rest")
        self.assertEqual(bundle["compute"]["compute"]["policy"]["default_preset"], "neuro-bids")
        self.assertEqual(bundle["compute"]["compute"]["slurm"]["modules"], ["apptainer/1.3"])
        self.assertNotIn("pre_activate_commands", bundle["compute"]["compute"]["slurm"])

    def test_project_init_bids_preprocess_scaffolds_deepprep_without_derivative_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            study_root.mkdir(parents=True, exist_ok=True)

            exit_code, output = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-preprocess",
                    "--project",
                    "project-new-deepprep",
                    "--study-root",
                    str(study_root),
                    "--tool",
                    "deepprep",
                    "--task-id",
                    "exampletask",
                    "--remote-study-root",
                    "/remote/studies/deepprep-demo",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            bundle = load_project_bundle("project-new-deepprep", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)
            preprocessing_text = (
                workspace_root / "project" / "project-new-deepprep" / "config" / "preprocessing.yaml"
            ).read_text(encoding="utf-8")
            compute_text = (
                workspace_root / "project" / "project-new-deepprep" / "config" / "compute.yaml"
            ).read_text(encoding="utf-8")
            batch_text = (
                workspace_root / "project" / "project-new-deepprep" / "manifests" / "batches" / "deepprep_default.tsv"
            ).read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])
        self.assertIn("Initialized BIDS preprocessing project: project-new-deepprep", output)
        self.assertEqual(batch_text, "subject_id\tsession_id\ttask_id\trun_id\n")
        dataset_config = bundle["dataset"]["dataset"]
        preprocessing_config = bundle["preprocessing"]["preprocessing"]
        self.assertNotIn("input_derivative", dataset_config)
        self.assertNotIn("input_derivative_root", dataset_config)
        self.assertNotIn("input_derivative", preprocessing_config)
        self.assertEqual(preprocessing_config["tool"], "deepprep")
        self.assertEqual(preprocessing_config["runtime_profile"], "deepprep")
        self.assertEqual(preprocessing_config["slurm_profile"], "local")
        self.assertIn("${FS_LICENSE_FILE:-secrets/freesurfer/license.txt}", preprocessing_text)
        self.assertEqual(preprocessing_config["inputs"]["fs_license_file"], "secrets/freesurfer/license.txt")
        self.assertEqual(preprocessing_config["tool_options"]["bold_task_type"], "exampletask")
        self.assertEqual(preprocessing_config["tool_options"]["bold_surface_spaces"], "fsnative")
        self.assertIn("${RP_DEEPPREP_CONTAINER_IMAGE:-docker://pbfslab/deepprep:25.1.0}", compute_text)
        self.assertIn("XDG_DATA_HOME", compute_text)
        self.assertEqual(
            bundle["compute"]["compute"]["slurm"]["environment"]["XDG_DATA_HOME"],
            "$SCRATCH/.local/share",
        )
        self.assertEqual(
            bundle["compute"]["compute"]["tool_profiles"]["deepprep"]["slurm"]["container"]["image"],
            "docker://pbfslab/deepprep:25.1.0",
        )

    def test_project_init_tabular_model_scaffolds_valid_project_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_init_workspace(workspace_root)

            exit_code, output = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "tabular-model",
                    "--project",
                    "project-new-tabular",
                    "--dataset",
                    "ds-tabular-inputs",
                    "--canonical-dataset",
                    "ds-tabular-derived",
                    "--canonical-features-root",
                    "derivatives/features/project-new-tabular",
                    "--batch",
                    "starter",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            validate_exit, validate_output = self._run_cli_for_workspace(
                ["config", "validate", "--project", "project-new-tabular"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            bundle = load_project_bundle("project-new-tabular", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)
            batch_text = (
                workspace_root / "project" / "project-new-tabular" / "manifests" / "batches" / "starter.tsv"
            ).read_text(encoding="utf-8")
            dataset_root_exists = (workspace_root / "datasets" / "ds-tabular-inputs").is_dir()
            canonical_features_root_exists = (
                workspace_root / "datasets" / "ds-tabular-derived" / "derivatives" / "features" / "project-new-tabular"
            ).is_dir()

        self.assertEqual(exit_code, 0)
        self.assertEqual(validate_exit, 0)
        self.assertTrue(json.loads(validate_output)["valid"])
        self.assertEqual(errors, [])
        self.assertIn("Initialized tabular model project: project-new-tabular", output)
        self.assertIn("project/project-new-tabular/config/models.yaml", output)
        self.assertIn("datasets/ds-tabular-inputs", output)
        self.assertIn("datasets/ds-tabular-derived/derivatives/features/project-new-tabular", output)
        self.assertIn("rp run plan preprocess tabular --project project-new-tabular --run-id project-new-tabular-preprocess", output)
        self.assertEqual(batch_text, "feature_table\ttarget_column\n")
        self.assertEqual(bundle["dataset"]["dataset"]["primary"], "ds-tabular-inputs")
        self.assertEqual(bundle["dataset"]["dataset"]["canonical_dataset"], "ds-tabular-derived")
        self.assertEqual(
            bundle["dataset"]["dataset"]["canonical_features_root"],
            "derivatives/features/project-new-tabular",
        )
        self.assertEqual(bundle["preprocessing"]["preprocessing"]["default_batch"], "starter")
        self.assertEqual(bundle["models"]["models"]["default"]["kind"], "logistic_regression")
        self.assertEqual(bundle["models"]["models"]["default"]["feature_columns"], ["feature_1", "feature_2"])
        self.assertTrue(dataset_root_exists)
        self.assertTrue(canonical_features_root_exists)

    def test_project_init_name_scaffolds_current_schema_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_init_workspace(workspace_root)

            exit_code, output = self._run_cli_for_workspace(
                ["project", "init", "project-demo-neutral"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            validate_exit, validate_output = self._run_cli_for_workspace(
                ["config", "validate", "--project", "project-demo-neutral"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            project_root = workspace_root / "project" / "project-demo-neutral"
            created_files = sorted(
                path.relative_to(project_root).as_posix()
                for path in project_root.rglob("*")
                if path.is_file()
            )
            bundle = load_project_bundle("project-demo-neutral", workspace_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(validate_exit, 0)
        self.assertTrue(json.loads(validate_output)["valid"])
        self.assertEqual(
            created_files,
            [
                "config/compute.yaml",
                "config/dataset.yaml",
                "config/models.yaml",
                "config/preprocessing.yaml",
                "manifests/batches/default.tsv",
                "project.yaml",
            ],
        )
        self.assertEqual(bundle["dataset"]["dataset"]["primary"], "project-demo-neutral")
        self.assertEqual(bundle["dataset"]["dataset"]["canonical_dataset"], "project-demo-neutral")
        self.assertEqual(bundle["preprocessing"]["preprocessing"]["default_batch"], "default")
        self.assertEqual(bundle["preprocessing"]["preprocessing"]["slice"], "tabular")
        self.assertEqual(bundle["models"]["models"]["default"]["kind"], "logistic_regression")
        self.assertEqual(bundle["models"]["models"]["default"]["feature_columns"], ["feature_1", "feature_2"])
        self.assertIn("Initialized project overlay: project-demo-neutral", output)
        self.assertIn("rp config validate --project project-demo-neutral", output)

    def test_onboard_preprocess_reuses_bids_preprocess_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "deepprep-bold"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            answers = [
                "",
                "fmripost_aroma",
                "project-onboard-bids",
                str(study_root),
                str(derivative_root),
                "rest",
                "/remote/studies/demo",
                "/remote/studies/demo/derivatives/deepprep-bold",
            ]
            with mock.patch("builtins.input", side_effect=answers):
                exit_code, output = self._run_cli_for_workspace(
                    ["onboard", "preprocess"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
            bundle = load_project_bundle("project-onboard-bids", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])
        self.assertIn("Initialized BIDS preprocessing project: project-onboard-bids", output)
        self.assertEqual(bundle["preprocessing"]["preprocessing"]["tool"], "fmripost_aroma")
        self.assertEqual(bundle["preprocessing"]["preprocessing"]["task_id"], "rest")
        self.assertEqual(bundle["dataset"]["dataset"]["remote_bids_root"], "/remote/studies/demo")

    def test_onboard_tabular_reuses_tabular_model_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_init_workspace(workspace_root)

            answers = [
                "project-onboard-tabular",
                "ds-tabular-inputs",
                "ds-tabular-derived",
                "derivatives/features/project-onboard-tabular",
                "starter",
            ]
            with mock.patch("builtins.input", side_effect=answers):
                exit_code, output = self._run_cli_for_workspace(
                    ["onboard", "tabular"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
            bundle = load_project_bundle("project-onboard-tabular", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])
        self.assertIn("Initialized tabular model project: project-onboard-tabular", output)
        self.assertEqual(bundle["dataset"]["dataset"]["primary"], "ds-tabular-inputs")
        self.assertEqual(bundle["preprocessing"]["preprocessing"]["default_batch"], "starter")
        self.assertEqual(bundle["models"]["models"]["default"]["kind"], "logistic_regression")
        self.assertEqual(bundle["models"]["models"]["default"]["feature_columns"], ["feature_1", "feature_2"])

    def test_setup_reports_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code, output = self._run_cli(["setup"], artifact_root)

        self.assertEqual(exit_code, 0)
        self.assertIn("research-platform setup", output)
        self.assertIn("Available workflows:", output)
        self.assertIn("preprocess", output)
        self.assertIn("analysis", output)
        self.assertIn("tabular", output)
        self.assertIn("Next command: rp onboard", output)

    def test_onboard_menu_lists_beginner_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code, output = self._run_cli(["onboard"], artifact_root)

        self.assertEqual(exit_code, 0)
        self.assertIn("Available workflows:", output)
        self.assertIn("notebook", output)
        self.assertIn("custom", output)

    def test_run_plan_analysis_tabular_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "default.tsv").write_text(
                "feature_table\ttarget_column\n"
                "test-tabular/toy_features.tsv\tbinary_target\n"
                "test-tabular/toy_features.tsv\tbinary_target\n",
                encoding="utf-8",
                newline="\n",
            )
            analysis_dir = project_root / "config" / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            write_yaml(
                analysis_dir / "feature_correlation.yaml",
                {
                    "analysis": {
                        "kind": "correlation",
                        "input_table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
                        "method": "pearson",
                        "x": "feature_a",
                        "y": "feature_b",
                    }
                },
            )

            exit_code, _ = self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "analysis",
                    "tabular",
                    "--project",
                    "test-tabular",
                    "--analysis",
                    "feature_correlation",
                    "--run-id",
                    "unit-tabular-analysis",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

            manifest = load_yaml(artifact_root / "runs" / "unit-tabular-analysis" / "run-manifest.yaml")
        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["workflow"], {"action": "analysis", "target": "tabular"})
        self.assertEqual(manifest["analysis"]["name"], "feature_correlation")
        self.assertEqual(manifest["tool"]["analysis"], "correlation")
        self.assertEqual(manifest["batch"], {"name": "", "path": "", "row_count": 0, "selected_row": {}})

    def test_resolve_workspace_hpc_runtime_default_returns_normalized_preset(self) -> None:
        preset = resolve_workspace_hpc_runtime_default(
            {
                "hpc": {
                    "runtime_defaults": {
                        "default": "site-default",
                        "catalog": {
                            "site-default": {
                                "slurm": {
                                    "modules": ["StdEnv/2023", "python/3.11", "StdEnv/2023", ""],
                                    "pre_activate_commands": "module load arrow/23.0.1",
                                    "prepare_directories": [
                                        "$APPTAINER_CACHEDIR",
                                        "$APPTAINER_TMPDIR",
                                    ],
                                    "remote_workspace_root": "/remote/workspace",
                                    "remote_artifacts_root": "/remote/scratch/artifacts",
                                }
                            }
                        },
                    }
                }
            }
        )

        self.assertEqual(
            preset,
            {
                "name": "site-default",
                "slurm": {
                    "modules": ["StdEnv/2023", "python/3.11", "StdEnv/2023"],
                    "pre_activate_commands": ["module load arrow/23.0.1"],
                    "prepare_directories": [
                        "$APPTAINER_CACHEDIR",
                        "$APPTAINER_TMPDIR",
                    ],
                    "remote_workspace_root": "/remote/workspace",
                    "remote_artifacts_root": "/remote/scratch/artifacts",
                },
            },
        )

    def test_resolve_workspace_hpc_runtime_default_returns_none_when_default_preset_is_missing(self) -> None:
        preset = resolve_workspace_hpc_runtime_default(
            {
                "hpc": {
                    "runtime_defaults": {
                        "default": "missing-site",
                        "catalog": {
                            "site-default": {
                                "slurm": {
                                    "modules": ["StdEnv/2023"],
                                }
                            }
                        },
                    }
                }
            }
        )

        self.assertIsNone(preset)

    def test_parse_yaml_treats_inline_empty_runtime_collections_as_structured_values(self) -> None:
        workspace_config = parse_yaml(
            "\n".join(
                [
                    "hpc:",
                    "  runtime_defaults:",
                    "    default: site-default",
                    "    catalog:",
                    "      site-default:",
                    "        slurm:",
                    "          modules:",
                    "            - StdEnv/2023",
                    "          pre_activate_commands: []",
                    "          prepare_directories: []",
                    "          metadata: {}",
                    "",
                ]
            )
        )

        slurm = workspace_config["hpc"]["runtime_defaults"]["catalog"]["site-default"]["slurm"]
        self.assertEqual(slurm["modules"], ["StdEnv/2023"])
        self.assertEqual(slurm["pre_activate_commands"], [])
        self.assertEqual(slurm["prepare_directories"], [])
        self.assertEqual(slurm["metadata"], {})

    def test_parse_yaml_accepts_json_style_yaml_documents(self) -> None:
        document = parse_yaml(
            '{"mvpa_set":{"name":"memory_mvpa","outputs":{"runtime_root":{"path":"${RP_MVPA_ROOT:-runtime}"}}}}'
        )

        self.assertEqual(document["mvpa_set"]["name"], "memory_mvpa")
        self.assertEqual(document["mvpa_set"]["outputs"]["runtime_root"]["path"], "runtime")

    def test_project_init_bids_preprocess_preserves_empty_pre_activate_commands_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(
                workspace_root,
                workspace_hpc={
                    "runtime_defaults": {
                        "default": "site-default",
                        "catalog": {
                            "site-default": {
                                "slurm": {
                                    "modules": ["StdEnv/2023", "python/3.11"],
                                    "pre_activate_commands": [],
                                    "prepare_directories": [],
                                }
                            }
                        },
                    }
                },
            )
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "deepprep-bold"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            exit_code, _ = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-preprocess",
                    "--project",
                    "project-new-bids",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "fmripost_aroma",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            compute_path = workspace_root / "project" / "project-new-bids" / "config" / "compute.yaml"
            compute_text = compute_path.read_text(encoding="utf-8")
            compute = load_yaml(compute_path)["compute"]["slurm"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(compute["modules"], ["StdEnv/2023", "python/3.11", "apptainer/1.3"])
        self.assertEqual(compute["pre_activate_commands"], [])
        self.assertEqual(compute["prepare_directories"], [])
        self.assertNotIn('- "[]"', compute_text)

    def test_project_init_bids_preprocess_materializes_workspace_runtime_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(
                workspace_root,
                workspace_hpc={
                    "runtime_defaults": {
                        "default": "site-default",
                        "catalog": {
                            "site-default": {
                                "slurm": {
                                    "modules": ["StdEnv/2023", "python/3.11"],
                                    "pre_activate_commands": [
                                        "module load arrow/23.0.1",
                                        "export RP_CLUSTER=1",
                                    ],
                                    "prepare_directories": [
                                        "$APPTAINER_CACHEDIR",
                                        "$APPTAINER_TMPDIR",
                                    ],
                                    "remote_workspace_root": "/remote/workspace-default",
                                    "remote_artifacts_root": "/remote/scratch/shared-artifacts",
                                }
                            }
                        },
                    }
                },
            )
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "deepprep-bold"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            exit_code, _ = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-preprocess",
                    "--project",
                    "project-new-bids",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "fmripost_aroma",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            compute = load_project_bundle("project-new-bids", workspace_root)["compute"]["compute"]["slurm"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(compute["modules"], ["StdEnv/2023", "python/3.11", "apptainer/1.3"])
        self.assertEqual(
            compute["pre_activate_commands"],
            ["module load arrow/23.0.1", "export RP_CLUSTER=1"],
        )
        self.assertEqual(
            compute["prepare_directories"],
            ["$APPTAINER_CACHEDIR", "$APPTAINER_TMPDIR"],
        )
        self.assertEqual(compute["remote_workspace_root"], "/remote/workspace-default")
        self.assertEqual(compute["remote_artifacts_root"], "/remote/scratch/shared-artifacts")
        self.assertEqual(compute["cpus"], 4)
        self.assertEqual(compute["mem"], "32G")
        self.assertEqual(compute["time"], "12:00:00")

    def test_project_init_tabular_model_materializes_workspace_runtime_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_init_workspace(
                workspace_root,
                workspace_hpc={
                    "runtime_defaults": {
                        "default": "site-default",
                        "catalog": {
                            "site-default": {
                                "slurm": {
                                    "modules": ["StdEnv/2023", "python/3.11"],
                                    "pre_activate_commands": [
                                        "module load arrow/23.0.1",
                                        "export RP_CLUSTER=1",
                                    ],
                                    "prepare_directories": [
                                        "$APPTAINER_CACHEDIR",
                                        "$APPTAINER_TMPDIR",
                                    ],
                                    "remote_workspace_root": "/remote/workspace-default",
                                    "remote_artifacts_root": "/remote/scratch/shared-artifacts",
                                }
                            }
                        },
                    }
                },
            )

            exit_code, _ = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "tabular-model",
                    "--project",
                    "project-new-tabular",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            compute = load_project_bundle("project-new-tabular", workspace_root)["compute"]["compute"]["slurm"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(compute["modules"], ["StdEnv/2023", "python/3.11"])
        self.assertEqual(
            compute["pre_activate_commands"],
            ["module load arrow/23.0.1", "export RP_CLUSTER=1"],
        )
        self.assertEqual(
            compute["prepare_directories"],
            ["$APPTAINER_CACHEDIR", "$APPTAINER_TMPDIR"],
        )
        self.assertEqual(compute["remote_workspace_root"], "/remote/workspace-default")
        self.assertEqual(compute["remote_artifacts_root"], "/remote/scratch/shared-artifacts")
        self.assertEqual(compute["cpus"], 2)
        self.assertEqual(compute["mem"], "4G")
        self.assertEqual(compute["time"], "00:30:00")

    def test_merge_scaffold_runtime_compute_defaults_keeps_adapter_scalars_and_dedupes_lists(self) -> None:
        merged = _merge_scaffold_runtime_compute_defaults(
            {
                "slurm": {
                    "cpus": 4,
                    "mem": "16G",
                    "time": "02:00:00",
                    "ssh_host": "${RP_HPC_HOST:-}",
                    "modules": ["python/3.11", "apptainer/1.3"],
                    "environment": {
                        "APPTAINER_CACHEDIR": "$PROJECT/cache",
                    },
                    "pre_activate_commands": [
                        "module load arrow/23.0.1",
                        "export RP_TOOL=1",
                    ],
                    "prepare_directories": [
                        "$APPTAINER_CACHEDIR",
                        "$PROJECT/tool-cache",
                    ],
                }
            },
            {
                "name": "site-default",
                "slurm": {
                    "modules": ["StdEnv/2023", "python/3.11"],
                    "environment": {
                        "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                        "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
                    },
                    "pre_activate_commands": [
                        "module load arrow/23.0.1",
                        "export RP_CLUSTER=1",
                    ],
                    "prepare_directories": [
                        "$APPTAINER_CACHEDIR",
                        "$APPTAINER_TMPDIR",
                    ],
                    "remote_workspace_root": "/remote/workspace-default",
                    "remote_artifacts_root": "/remote/scratch/site-artifacts",
                },
            },
        )

        self.assertEqual(merged["slurm"]["cpus"], 4)
        self.assertEqual(merged["slurm"]["mem"], "16G")
        self.assertEqual(merged["slurm"]["time"], "02:00:00")
        self.assertEqual(merged["slurm"]["ssh_host"], "${RP_HPC_HOST:-}")
        self.assertEqual(
            merged["slurm"]["modules"],
            ["StdEnv/2023", "python/3.11", "apptainer/1.3"],
        )
        self.assertEqual(
            merged["slurm"]["environment"],
            {
                "APPTAINER_CACHEDIR": "$PROJECT/cache",
                "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
            },
        )
        self.assertEqual(
            merged["slurm"]["pre_activate_commands"],
            [
                "module load arrow/23.0.1",
                "export RP_CLUSTER=1",
                "export RP_TOOL=1",
            ],
        )
        self.assertEqual(
            merged["slurm"]["prepare_directories"],
            [
                "$APPTAINER_CACHEDIR",
                "$APPTAINER_TMPDIR",
                "$PROJECT/tool-cache",
            ],
        )
        self.assertEqual(merged["slurm"]["remote_workspace_root"], "/remote/workspace-default")
        self.assertEqual(merged["slurm"]["remote_artifacts_root"], "/remote/scratch/site-artifacts")

    def test_load_project_bundle_applies_workspace_runtime_defaults_to_existing_structured_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_tabular_slurm_workspace(workspace_root)

            workspace_config = load_yaml(workspace_root / "WORKSPACE.yaml")
            workspace_config["hpc"] = {
                "runtime_defaults": {
                    "default": "site-default",
                    "catalog": {
                        "site-default": {
                            "slurm": {
                                "modules": ["StdEnv/2023", "python/3.11"],
                                "pre_activate_commands": ["module load arrow/23.0.1"],
                                "prepare_directories": ["$APPTAINER_CACHEDIR"],
                                "environment": {"APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp"},
                                "remote_workspace_root": "/remote/workspace-default",
                                "remote_artifacts_root": "/remote/scratch/shared-artifacts",
                            }
                        }
                    },
                }
            }
            write_yaml(workspace_root / "WORKSPACE.yaml", workspace_config)
            write_yaml(
                workspace_root / "project" / "test-tabular" / "config" / "compute.yaml",
                {
                    "compute": {
                        "default_profile": "local",
                        "local": {"jobs": 1},
                        "slurm": {
                            "cpus": 1,
                            "mem": "4G",
                            "time": "00:30:00",
                        },
                    }
                },
            )

            slurm = load_project_bundle("test-tabular", workspace_root)["compute"]["compute"]["slurm"]

        self.assertEqual(slurm["modules"], ["StdEnv/2023", "python/3.11"])
        self.assertEqual(slurm["pre_activate_commands"], ["module load arrow/23.0.1"])
        self.assertEqual(slurm["prepare_directories"], ["$APPTAINER_CACHEDIR"])
        self.assertEqual(slurm["environment"], {"APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp"})
        self.assertEqual(slurm["remote_workspace_root"], "/remote/workspace-default")
        self.assertEqual(slurm["remote_artifacts_root"], "/remote/scratch/shared-artifacts")
        self.assertEqual(slurm["cpus"], 1)
        self.assertEqual(slurm["mem"], "4G")
        self.assertEqual(slurm["time"], "00:30:00")

    def test_load_project_bundle_keeps_project_remote_root_overrides_over_workspace_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_tabular_slurm_workspace(workspace_root)

            workspace_config = load_yaml(workspace_root / "WORKSPACE.yaml")
            workspace_config["hpc"] = {
                "runtime_defaults": {
                    "default": "site-default",
                    "catalog": {
                        "site-default": {
                            "slurm": {
                                "remote_workspace_root": "/remote/workspace-default",
                                "remote_artifacts_root": "/remote/scratch/shared-artifacts",
                            }
                        }
                    },
                }
            }
            write_yaml(workspace_root / "WORKSPACE.yaml", workspace_config)
            write_yaml(
                workspace_root / "project" / "test-tabular" / "config" / "compute.yaml",
                {
                    "compute": {
                        "default_profile": "local",
                        "local": {"jobs": 1},
                        "slurm": {
                            "remote_workspace_root": "/remote/project-specific-workspace",
                            "remote_artifacts_root": "/remote/project-specific-artifacts",
                            "cpus": 1,
                            "mem": "4G",
                            "time": "00:30:00",
                        },
                    }
                },
            )

            slurm = load_project_bundle("test-tabular", workspace_root)["compute"]["compute"]["slurm"]

        self.assertEqual(slurm["remote_workspace_root"], "/remote/project-specific-workspace")
        self.assertEqual(slurm["remote_artifacts_root"], "/remote/project-specific-artifacts")

    def test_load_project_bundle_preserves_compute_when_workspace_runtime_defaults_are_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_tabular_slurm_workspace(workspace_root)
            compute_path = workspace_root / "project" / "test-tabular" / "config" / "compute.yaml"
            expected_compute = {
                "compute": {
                    "default_profile": "local",
                    "local": {"jobs": 1},
                    "slurm": {
                        "cpus": 2,
                        "mem": "8G",
                        "time": "01:00:00",
                    },
                }
            }
            write_yaml(compute_path, expected_compute)

            compute = load_project_bundle("test-tabular", workspace_root)["compute"]

        self.assertEqual(compute, expected_compute)

    def test_validate_project_bundle_rejects_empty_slurm_prepare_directories_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            bundle = self._write_bids_validation_fixture(workspace_root, derivative_name="deepprep-bold")
            bundle["compute"]["compute"] = {
                "default_profile": "slurm",
                "slurm": {
                    "prepare_directories": ["$APPTAINER_CACHEDIR", ""],
                },
            }

            errors = validate_project_bundle(bundle, root=workspace_root)

        self.assertIn(
            "WORKSPACE.yaml config/compute.yaml compute.slurm.prepare_directories[1] must contain a non-empty string.",
            errors,
        )

    def test_project_init_bids_preprocess_rejects_project_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "deepprep-bold"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            error = self._run_cli_system_exit_for_workspace(
                [
                    "project",
                    "init",
                    "bids-preprocess",
                    "--project",
                    "../escape",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "fmripost_aroma",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(error, "--project must be a simple project name, not a path.")

    def test_project_init_tabular_model_rejects_path_like_dataset_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_init_workspace(workspace_root)

            dataset_error = self._run_cli_system_exit_for_workspace(
                [
                    "project",
                    "init",
                    "tabular-model",
                    "--project",
                    "project-new-tabular",
                    "--dataset",
                    "../escape",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            canonical_dataset_error = self._run_cli_system_exit_for_workspace(
                [
                    "project",
                    "init",
                    "tabular-model",
                    "--project",
                    "project-new-tabular",
                    "--canonical-dataset",
                    "nested/name",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(dataset_error, "--dataset must be a simple name, not a path.")
        self.assertEqual(canonical_dataset_error, "--canonical-dataset must be a simple name, not a path.")

    def test_project_init_tabular_model_rejects_path_like_batch_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_init_workspace(workspace_root)

            error = self._run_cli_system_exit_for_workspace(
                [
                    "project",
                    "init",
                    "tabular-model",
                    "--project",
                    "project-new-tabular",
                    "--batch",
                    "../bad",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(error, "--batch must be a simple name, not a path.")

    def test_project_init_tabular_model_rejects_invalid_canonical_features_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_init_workspace(workspace_root)

            absolute_error = self._run_cli_system_exit_for_workspace(
                [
                    "project",
                    "init",
                    "tabular-model",
                    "--project",
                    "project-new-tabular",
                    "--canonical-features-root",
                    "/absolute/path",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            escape_error = self._run_cli_system_exit_for_workspace(
                [
                    "project",
                    "init",
                    "tabular-model",
                    "--project",
                    "project-new-tabular",
                    "--canonical-features-root",
                    "../escape",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(
            absolute_error,
            "--canonical-features-root must be a relative path under the canonical dataset root.",
        )
        self.assertEqual(
            escape_error,
            "--canonical-features-root must be a relative path under the canonical dataset root.",
        )

    def test_project_init_bids_preprocess_rejects_missing_local_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(workspace_root)

            error = self._run_cli_system_exit_for_workspace(
                [
                    "project",
                    "init",
                    "bids-preprocess",
                    "--project",
                    "project-new-bids",
                    "--study-root",
                    str(workspace_root / "missing-study"),
                    "--derivative-root",
                    str(workspace_root / "missing-derivative"),
                    "--tool",
                    "fmripost_aroma",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertIn("--study-root was not found:", error)

    def test_project_init_bids_preprocess_rejects_nonabsolute_remote_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "deepprep-bold"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            error = self._run_cli_system_exit_for_workspace(
                [
                    "project",
                    "init",
                    "bids-preprocess",
                    "--project",
                    "project-new-bids",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "fmripost_aroma",
                    "--remote-study-root",
                    "relative/remote/root",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertEqual(error, "--remote-study-root must be an absolute remote POSIX path.")

    def test_run_plan_writes_manifest_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code, _ = self._run_cli(
                ["run", "plan", "preprocess", "bids", "--run-id", "unit-plan"],
                artifact_root,
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-plan" / "run-manifest.yaml")
            status = load_yaml(artifact_root / "runs" / "unit-plan" / "status.yaml")
            pipeline_defaults = load_yaml(WORKSPACE_ROOT / "pipelines" / "preprocess-bids" / "config" / "defaults.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["run_id"], "unit-plan")
        self.assertEqual(manifest["tool"]["name"], "fmripost_aroma")
        self.assertEqual(
            manifest["tool"]["adapter"],
            "research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter",
        )
        self.assertEqual(manifest["dataset"]["derivative_name"], "deepprep-bold")
        self.assertEqual(manifest["execution"]["mode"], "plan")
        self.assertEqual(manifest["batch"]["row_count"], 2)
        self.assertEqual(manifest["resources"]["workload"], "bids_preprocess")
        self.assertNotEqual(manifest["resources"]["preset"], "legacy")
        self.assertIn("--cores", manifest["execution"]["command"])
        self.assertEqual(
            str(manifest["execution"]["command"][manifest["execution"]["command"].index("--cores") + 1]),
            "4",
        )
        self.assertIn("--set-threads", manifest["execution"]["command"])
        self.assertIn("bids_preprocess_unit=4", manifest["execution"]["command"])
        self.assertNotIn("bids_preprocess=4", manifest["execution"]["command"])
        self.assertIn("--set-resources", manifest["execution"]["command"])
        self.assertIn("bids_preprocess_unit:mem_mb=16384", manifest["execution"]["command"])
        self.assertIn("bids_preprocess_unit:cpus_per_task=4", manifest["execution"]["command"])
        self.assertIn("bids_preprocess_unit:runtime=2h", manifest["execution"]["command"])
        self.assertNotIn("bids_preprocess:mem_mb=16384", manifest["execution"]["command"])
        self.assertNotIn("bids_preprocess:cpus_per_task=4", manifest["execution"]["command"])
        self.assertNotIn("bids_preprocess:runtime=02:00:00", manifest["execution"]["command"])
        self.assertEqual(
            manifest["execution"]["command"][-2:],
            ["--", pipeline_defaults["workflow"]["default_target"]],
        )
        self.assertLess(
            manifest["execution"]["command"].index("--set-resources"),
            manifest["execution"]["command"].index("--"),
        )
        self.assertEqual(manifest["execution"]["command"][-1], pipeline_defaults["workflow"]["default_target"])
        self.assertTrue(
            manifest["outputs"]["runtime_plan"].endswith(pipeline_defaults["planner"]["outputs"]["runtime_plan_filename"])
        )
        self.assertTrue(
            manifest["outputs"]["command_script"].endswith(
                pipeline_defaults["planner"]["outputs"]["command_script_filename"]
            )
        )
        self.assertTrue(
            manifest["outputs"]["completion_marker"].endswith(
                pipeline_defaults["planner"]["outputs"]["completion_marker_filename"]
            )
        )
        self.assertEqual(status["state"], "planned")
        self.assertEqual(
            manifest["publish_back"],
            {
                "default_policy": "never",
                "plan_path": str((artifact_root / "runs" / "unit-plan" / "publish-back-plan.yaml").resolve()),
                "items": [
                    {
                        "source": str((artifact_root / "runs" / "unit-plan" / "outputs" / "fmripost_aroma").resolve()),
                        "destination": str((WORKSPACE_ROOT / "datasets" / "ds-bids-example" / "derivatives" / "fmripost_aroma").resolve()),
                    }
                ],
            },
        )

    def test_preprocess_bids_snakefile_exposes_unit_fanout_and_aggregate_marker_contract(self) -> None:
        snakefile_path = WORKSPACE_ROOT / "pipelines" / "preprocess-bids" / "workflow" / "Snakefile"
        snakefile_text = snakefile_path.read_text(encoding="utf-8")

        self.assertIn("checkpoint bids_preprocess_plan:", snakefile_text)
        self.assertIn("rule bids_preprocess_container_prep:", snakefile_text)
        self.assertIn("rule bids_preprocess_unit:", snakefile_text)
        self.assertIn('UNIT_MARKER_PATTERN = str(RUN_OUTPUT_ROOT / "runtime-plan-markers" / TOOL / "{unit_id}.txt")', snakefile_text)
        self.assertIn('CONTAINER_PREP_MARKER = str(RUN_OUTPUT_ROOT / "runtime-plan-markers" / TOOL / "_container-ready.txt")', snakefile_text)
        self.assertIn('CONTAINER_PREP_RUNTIME = "60m"', snakefile_text)
        self.assertIn("CONTAINER_PREP_CPUS_PER_TASK = 2", snakefile_text)
        self.assertIn("mem_mb=CONTAINER_PREP_MEM_MB", snakefile_text)
        self.assertIn("cpus_per_task=CONTAINER_PREP_CPUS_PER_TASK", snakefile_text)
        self.assertIn("container_prep=CONTAINER_PREP_MARKER", snakefile_text)
        self.assertIn("unit_markers=_runtime_unit_markers", snakefile_text)
        self.assertIn("--aggregate-only --marker {output}", snakefile_text)

    def test_analysis_bids_snakefile_exposes_container_prep_and_unit_fanout_contract(self) -> None:
        snakefile_path = WORKSPACE_ROOT / "pipelines" / "analysis-bids" / "workflow" / "Snakefile"
        snakefile_text = snakefile_path.read_text(encoding="utf-8")

        self.assertIn("checkpoint bids_analysis_plan:", snakefile_text)
        self.assertIn("rule bids_analysis_container_prep:", snakefile_text)
        self.assertIn("rule bids_analysis_unit:", snakefile_text)
        self.assertIn('UNIT_MARKER_PATTERN = str(RUN_OUTPUT_ROOT / "runtime-plan-markers" / TOOL / "{unit_id}.txt")', snakefile_text)
        self.assertIn('CONTAINER_PREP_MARKER = str(RUN_OUTPUT_ROOT / "runtime-plan-markers" / TOOL / "_container-ready.txt")', snakefile_text)
        self.assertIn('CONTAINER_PREP_RUNTIME = "60m"', snakefile_text)
        self.assertIn("CONTAINER_PREP_CPUS_PER_TASK = 2", snakefile_text)
        self.assertIn("mem_mb=CONTAINER_PREP_MEM_MB", snakefile_text)
        self.assertIn("cpus_per_task=CONTAINER_PREP_CPUS_PER_TASK", snakefile_text)
        self.assertIn("container_prep=CONTAINER_PREP_MARKER", snakefile_text)
        self.assertIn("unit_markers=_runtime_unit_markers", snakefile_text)
        self.assertIn("--aggregate-only --marker {output}", snakefile_text)

    def test_run_plan_targets_execution_rule_resources_but_keeps_final_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code, _ = self._run_cli(
                ["run", "plan", "preprocess", "bids", "--run-id", "unit-plan-fanout-resources"],
                artifact_root,
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-plan-fanout-resources" / "run-manifest.yaml")

        self.assertEqual(exit_code, 0)
        command = manifest["execution"]["command"]
        self.assertIn("bids_preprocess_unit=4", command)
        self.assertIn("bids_preprocess_unit:mem_mb=16384", command)
        self.assertIn("bids_preprocess_unit:cpus_per_task=4", command)
        self.assertIn("bids_preprocess_unit:runtime=2h", command)
        self.assertNotIn("bids_preprocess=4", command)
        self.assertNotIn("bids_preprocess:mem_mb=16384", command)
        self.assertEqual(command[-2:], ["--", "bids_preprocess"])

    def test_bids_task_scoped_plan_records_selection_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code, _ = self._run_cli(
                ["run", "plan", "preprocess", "bids", "--run-id", "unit-plan-task", "--task-id", "memory"],
                artifact_root,
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-plan-task" / "run-manifest.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["selection"]["task_id"], "memory")

    def test_run_submit_preprocess_bids_executes_generic_stage_and_submit_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            with mock.patch(
                "research_platform.core.cli.execute_stage_plan",
                return_value={"ok": True, "returncode": 0, "commands": [{"command": ["rsync"]}]},
            ) as stage_mock:
                with mock.patch(
                    "research_platform.core.cli.execute_submit_plan",
                    return_value={
                        "ok": True,
                        "returncode": 0,
                        "command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"],
                        "stdout": "Submitted batch job 12345\n",
                        "stderr": "",
                        "job_id": "12345",
                    },
                ) as submit_mock:
                    exit_code, output = self._run_cli(
                        ["run", "submit", "preprocess", "bids", "--run-id", "unit-submit-bids", "--execute"],
                        artifact_root,
                        extra_env={
                            "RP_HPC_PROFILE": "",
                            "RESEARCH_HPC_PROFILE": "",
                            "RP_SSH_CONFIG": "",
                            "RESEARCH_HPC_SSH_CONFIG": "",
                        },
                    )
            status = load_yaml(artifact_root / "runs" / "unit-submit-bids" / "status.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(stage_mock.call_count, 1)
        self.assertEqual(submit_mock.call_count, 1)
        self.assertIn("BIDS submit summary", output)
        self.assertIn("Tool: fmripost_aroma", output)
        self.assertIn(
            "Adapter: research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter",
            output,
        )
        self.assertIn("Run id: unit-submit-bids", output)
        self.assertIn("Stage result: staged", output)
        self.assertIn("Submit result: submitted", output)
        self.assertIn("Submit job id: 12345", output)
        self.assertEqual(status["state"], "submitted")
        self.assertEqual(str(status["job_id"]), "12345")

    def test_run_submit_bids_variants_plan_without_remote_execution(self) -> None:
        cases = (
            (
                ["run", "submit", "preprocess", "bids", "--run-id", "unit-submit-preprocess-plan"],
                "unit-submit-preprocess-plan",
                False,
            ),
            (
                ["run", "submit", "preprocess", "bids", "--run-id", "unit-submit-preprocess-dry", "--dry-run"],
                "unit-submit-preprocess-dry",
                True,
            ),
            (
                ["run", "submit", "analysis", "bids", "--run-id", "unit-submit-analysis-default"],
                "unit-submit-analysis-default",
                False,
            ),
            (
                ["run", "submit", "analysis", "bids", "--run-id", "unit-submit-analysis-plan", "--dry-run"],
                "unit-submit-analysis-plan",
                True,
            ),
        )
        for argv, run_id, expected_dry_run in cases:
            with self.subTest(run_id=run_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    artifact_root = Path(tmp_dir) / "artifacts"
                    with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                        with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                            with mock.patch(
                                "research_platform.core.cli.subprocess.run",
                                side_effect=AssertionError("submission planning must not invoke a subprocess"),
                            ):
                                exit_code, output = self._run_cli(argv, artifact_root)
                    payload = json.loads(output)
                    manifest = load_yaml(artifact_root / "runs" / run_id / "run-manifest.yaml")

                self.assertEqual(exit_code, 0)
                stage_mock.assert_not_called()
                submit_mock.assert_not_called()
                self.assertEqual(payload["mode"], "plan")
                self.assertEqual(payload["dry_run"], expected_dry_run)
                self.assertTrue(payload["stage"]["prepare_commands"])
                self.assertTrue(payload["submission"]["submit_command"])
                self.assertTrue(any(path.endswith("run-manifest.yaml") for path in payload["local_files"]))
                self.assertTrue(any(path.endswith("submit.sbatch") for path in payload["local_files"]))
                self.assertEqual(manifest["execution"]["dry_run"], expected_dry_run)

    def test_run_submit_tabular_analysis_defaults_to_plan_without_remote_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            analysis_dir = project_root / "config" / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            write_yaml(
                analysis_dir / "feature_correlation.yaml",
                {
                    "analysis": {
                        "kind": "correlation",
                        "input_table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
                        "method": "pearson",
                        "x": "feature_a",
                        "y": "binary_target",
                    }
                },
            )
            cases = (
                ("unit-submit-tabular-default", [], False),
                ("unit-submit-tabular-dry", ["--dry-run"], True),
            )
            for run_id, mode_args, expected_dry_run in cases:
                with self.subTest(run_id=run_id):
                    with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                        with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                            with mock.patch(
                                "research_platform.core.cli.subprocess.run",
                                side_effect=AssertionError("submission planning must not invoke a subprocess"),
                            ):
                                exit_code, output = self._run_cli_for_workspace(
                                    [
                                        "run",
                                        "submit",
                                        "analysis",
                                        "tabular",
                                        "--project",
                                        "test-tabular",
                                        "--analysis",
                                        "feature_correlation",
                                        "--run-id",
                                        run_id,
                                        *mode_args,
                                    ],
                                    workspace_root=workspace_root,
                                    artifact_root=artifact_root,
                                )
                    payload = json.loads(output)

                    self.assertEqual(exit_code, 0)
                    stage_mock.assert_not_called()
                    submit_mock.assert_not_called()
                    self.assertEqual(payload["mode"], "plan")
                    self.assertEqual(payload["dry_run"], expected_dry_run)
                    self.assertTrue(payload["stage"]["push_commands"])
                    self.assertTrue(payload["submission"]["submit_command"])

    def test_run_submit_analysis_variants_execute_each_remote_boundary_once(self) -> None:
        stage_result = {"ok": True, "returncode": 0, "commands": [{"command": ["rsync"]}]}
        submit_result = {
            "ok": True,
            "returncode": 0,
            "command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"],
            "stdout": "Submitted batch job 24680\n",
            "stderr": "",
            "job_id": "24680",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            with mock.patch("research_platform.core.cli.execute_stage_plan", return_value=stage_result) as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan", return_value=submit_result) as submit_mock:
                    exit_code, _ = self._run_cli(
                        ["run", "submit", "analysis", "bids", "--run-id", "unit-submit-analysis-execute", "--execute"],
                        artifact_root,
                    )

            self.assertEqual(exit_code, 0)
            stage_mock.assert_called_once()
            submit_mock.assert_called_once()

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            analysis_dir = project_root / "config" / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            write_yaml(
                analysis_dir / "feature_correlation.yaml",
                {
                    "analysis": {
                        "kind": "correlation",
                        "input_table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
                        "method": "pearson",
                        "x": "feature_a",
                        "y": "binary_target",
                    }
                },
            )
            with mock.patch("research_platform.core.cli.execute_stage_plan", return_value=stage_result) as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan", return_value=submit_result) as submit_mock:
                    exit_code, _ = self._run_cli_for_workspace(
                        [
                            "run",
                            "submit",
                            "analysis",
                            "tabular",
                            "--project",
                            "test-tabular",
                            "--analysis",
                            "feature_correlation",
                            "--run-id",
                            "unit-submit-tabular-execute",
                            "--execute",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

            self.assertEqual(exit_code, 0)
            stage_mock.assert_called_once()
            submit_mock.assert_called_once()

    def test_run_submit_dry_run_and_execute_are_mutually_exclusive(self) -> None:
        cases = (
            ["run", "submit", "preprocess", "bids", "--dry-run", "--execute"],
            ["run", "local", "preprocess", "bids", "--dry-run", "--execute"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    with self.assertRaises(SystemExit) as exc_info:
                        main(argv)

                self.assertEqual(exc_info.exception.code, 2)
                self.assertIn("not allowed with argument --dry-run", stderr.getvalue())

    def test_run_slurm_preprocess_bids_rejects_policy_slurm_memory_mismatch_before_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "discovered_batch.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            compute_path = project_root / "config" / "compute.yaml"
            compute_config = load_yaml(compute_path, resolve_env=False)
            compute_config["compute"]["policy"] = {
                "default_preset": "neuro-bids",
                "presets": {
                    "neuro-bids": {
                        "cpus": 4,
                        "ram_gb": 16,
                        "threads": 1,
                        "n_jobs": 1,
                    }
                },
                "workloads": {
                    "bids_preprocess": {"preset": "neuro-bids"},
                },
            }
            compute_config["compute"]["slurm"]["mem"] = "32G"
            write_yaml(compute_path, compute_config)

            error = self._run_cli_system_exit_for_workspace(
                ["run", "slurm", "preprocess", "bids", "--run-id", "unit-slurm-mem-mismatch"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            run_root = artifact_root / "runs" / "unit-slurm-mem-mismatch"

        self.assertIn("compute.slurm.mem", error)
        self.assertIn("compute.policy.presets.neuro-bids.ram_gb", error)
        self.assertIn("32G", error)
        self.assertIn("ram_gb=16", error)
        self.assertFalse((run_root / "submit.sbatch").exists())
        self.assertFalse((run_root / "run-manifest.yaml").exists())

    def test_run_submit_preprocess_bids_rejects_policy_slurm_memory_mismatch_before_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "discovered_batch.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            compute_path = project_root / "config" / "compute.yaml"
            compute_config = load_yaml(compute_path, resolve_env=False)
            compute_config["compute"]["policy"] = {
                "default_preset": "neuro-bids",
                "presets": {
                    "neuro-bids": {
                        "cpus": 4,
                        "ram_gb": 16,
                        "threads": 1,
                        "n_jobs": 1,
                    }
                },
                "workloads": {
                    "bids_preprocess": {"preset": "neuro-bids"},
                },
            }
            compute_config["compute"]["slurm"]["mem"] = "32G"
            write_yaml(compute_path, compute_config)

            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    error = self._run_cli_system_exit_for_workspace(
                        ["run", "submit", "preprocess", "bids", "--run-id", "unit-submit-mem-mismatch", "--execute"],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
            run_root = artifact_root / "runs" / "unit-submit-mem-mismatch"

        self.assertIn("compute.slurm.mem", error)
        self.assertIn("compute.policy.presets.neuro-bids.ram_gb", error)
        self.assertEqual(stage_mock.call_count, 0)
        self.assertEqual(submit_mock.call_count, 0)
        self.assertFalse((run_root / "submit.sbatch").exists())
        self.assertFalse((run_root / "run-manifest.yaml").exists())

    def test_run_submit_preprocess_bids_discover_bootstraps_batch_and_reports_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            with mock.patch(
                "research_platform.core.cli.execute_stage_plan",
                return_value={"ok": True, "returncode": 0, "commands": [{"command": ["rsync"]}]},
            ) as stage_mock:
                with mock.patch(
                    "research_platform.core.cli.execute_submit_plan",
                    return_value={
                        "ok": True,
                        "returncode": 0,
                        "command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"],
                        "stdout": "Submitted batch job 98765\n",
                        "stderr": "",
                        "job_id": "98765",
                    },
                ) as submit_mock:
                    exit_code, output = self._run_cli_for_workspace(
                        [
                            "run",
                            "submit",
                            "preprocess",
                            "bids",
                            "--discover",
                            "--run-id",
                            "unit-submit-discover",
                            "--execute",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
            batch_text = (project_root / "manifests" / "batches" / "discovered_batch.tsv").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(stage_mock.call_count, 1)
        self.assertEqual(submit_mock.call_count, 1)
        self.assertEqual(
            batch_text,
            "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
        )
        self.assertIn("Batch: project/test-bids/manifests/batches/discovered_batch.tsv", output)
        self.assertIn("Discovered rows: 1", output)
        self.assertIn("Run id: unit-submit-discover", output)
        self.assertIn("Submit job id: 98765", output)

    def test_run_submit_preprocess_bids_discover_plan_reports_external_batch_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            batch_path = project_root / "manifests" / "batches" / "discovered_batch.tsv"
            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    exit_code, output = self._run_cli_for_workspace(
                        [
                            "run",
                            "submit",
                            "preprocess",
                            "bids",
                            "--discover",
                            "--run-id",
                            "unit-submit-discover-plan",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
            payload = json.loads(output)

        self.assertEqual(exit_code, 0)
        stage_mock.assert_not_called()
        submit_mock.assert_not_called()
        self.assertIn(str(batch_path.resolve()), payload["local_files"])

    def test_run_submit_preprocess_bids_stages_submit_script_with_shebang_at_byte_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "discovered_batch.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            (workspace_root / "ops" / "slurm" / "job_templates" / "sbatch.job.sh").write_text(
                (
                    "\ufeff\r\n#!/usr/bin/env bash\r\n"
                    "#SBATCH --job-name={{ job_name }}\r\n"
                    "#SBATCH --cpus-per-task={{ cpus }}\r\n"
                    "#SBATCH --mem={{ mem }}\r\n"
                    "#SBATCH --time={{ time }}\r\n"
                    "#SBATCH --output={{ log_out }}\r\n"
                    "#SBATCH --error={{ log_err }}\r\n"
                    "\r\n"
                    "{{ command }}\r\n"
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "research_platform.core.cli.execute_stage_plan",
                return_value={"ok": True, "returncode": 0, "commands": [{"command": ["rsync"]}]},
            ):
                with mock.patch(
                    "research_platform.core.cli.execute_submit_plan",
                    return_value={
                        "ok": True,
                        "returncode": 0,
                        "command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"],
                        "stdout": "Submitted batch job 24680\n",
                        "stderr": "",
                        "job_id": "24680",
                    },
                ):
                    exit_code, _ = self._run_cli_for_workspace(
                        ["run", "submit", "preprocess", "bids", "--run-id", "unit-submit-shebang", "--execute"],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

            local_submit_script = artifact_root / "runs" / "unit-submit-shebang" / "submit.sbatch"
            staged_submit_script = artifact_root / "runs" / "unit-submit-shebang" / "hpc" / "stage" / "submit.sbatch"
            local_script_bytes = local_submit_script.read_bytes()
            local_script_text = local_submit_script.read_text(encoding="utf-8")
            staged_script_bytes = staged_submit_script.read_bytes()
            staged_script_text = staged_submit_script.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertTrue(local_script_bytes.startswith(b"#!"))
        self.assertEqual(local_script_text.splitlines()[0], "#!/usr/bin/env bash")
        self.assertNotEqual(local_script_text[:1], "\n")
        self.assertNotIn("\r", local_script_text)
        self.assertTrue(staged_script_bytes.startswith(b"#!"))
        self.assertEqual(staged_script_text.splitlines()[0], "#!/usr/bin/env bash")
        self.assertNotEqual(staged_script_text[:1], "\n")
        self.assertNotIn("\r", staged_script_text)

    def test_run_submit_preprocess_bids_writes_shared_slurm_setup_and_guard_before_snakemake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "discovered_batch.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            bootstrap_script = workspace_root / "ops" / "envs" / "dev" / "bootstrap.sh"
            bootstrap_script.parent.mkdir(parents=True, exist_ok=True)
            bootstrap_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            compute_path = project_root / "config" / "compute.yaml"
            compute_config = load_yaml(compute_path, resolve_env=False)
            compute_config["compute"]["slurm"]["modules"] = ["StdEnv/2023", "python/3.11"]
            compute_config["compute"]["slurm"]["pre_activate_commands"] = [
                "module load arrow/23.0.1",
                "export RP_CLUSTER=1",
            ]
            compute_config["compute"]["slurm"]["environment"] = {
                "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
            }
            compute_config["compute"]["slurm"]["prepare_directories"] = [
                "$APPTAINER_CACHEDIR",
                "$APPTAINER_TMPDIR",
            ]
            write_yaml(compute_path, compute_config)
            with mock.patch(
                "research_platform.core.cli.execute_stage_plan",
                return_value={"ok": True, "returncode": 0, "commands": [{"command": ["rsync"]}]},
            ):
                with mock.patch(
                    "research_platform.core.cli.execute_submit_plan",
                    return_value={
                        "ok": True,
                        "returncode": 0,
                        "command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"],
                        "stdout": "Submitted batch job 24680\n",
                        "stderr": "",
                        "job_id": "24680",
                    },
                ):
                    exit_code, _ = self._run_cli_for_workspace(
                        ["run", "submit", "preprocess", "bids", "--run-id", "unit-submit-setup", "--execute"],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

            submit_script = artifact_root / "runs" / "unit-submit-setup" / "submit.sbatch"
            script_text = submit_script.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("cd remote/workspace", script_text)
        self.assertIn("module load StdEnv/2023 python/3.11", script_text)
        self.assertIn("module load arrow/23.0.1", script_text)
        self.assertIn('export APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache"', script_text)
        self.assertIn('export APPTAINER_TMPDIR="$SCRATCH/apptainer-tmp"', script_text)
        self.assertIn("export RP_CLUSTER=1", script_text)
        self.assertIn('mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"', script_text)
        self.assertIn("bash ops/envs/dev/bootstrap.sh", script_text)
        self.assertIn("source .venv/bin/activate", script_text)
        self.assertIn("command -v snakemake >/dev/null 2>&1", script_text)
        self.assertIn("snakemake", script_text)
        self.assertIn(
            "--cores 4 --set-threads fmripost_aroma_unit=4 --set-resources fmripost_aroma_unit:mem_mb=16384 fmripost_aroma_unit:cpus_per_task=4 fmripost_aroma_unit:runtime=2h -- fmripost_aroma",
            script_text,
        )
        self.assertNotIn("--set-threads fmripost_aroma=4", script_text)
        self.assertNotIn("fmripost_aroma:mem_mb=16384", script_text)
        self.assertLess(script_text.index("cd remote/workspace"), script_text.index("module load StdEnv/2023 python/3.11"))
        self.assertLess(
            script_text.index("module load StdEnv/2023 python/3.11"),
            script_text.index("module load arrow/23.0.1"),
        )
        self.assertLess(script_text.index('export APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache"'), script_text.index("module load arrow/23.0.1"))
        self.assertLess(script_text.index("module load arrow/23.0.1"), script_text.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(script_text.index("export RP_CLUSTER=1"), script_text.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(script_text.index("export RP_CLUSTER=1"), script_text.index('mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"'))
        self.assertLess(script_text.index('mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"'), script_text.index("bash ops/envs/dev/bootstrap.sh"))
        self.assertLess(script_text.index("bash ops/envs/dev/bootstrap.sh"), script_text.index("source .venv/bin/activate"))
        self.assertLess(script_text.index("source .venv/bin/activate"), script_text.index("command -v snakemake"))
        self.assertLess(script_text.index("command -v snakemake"), script_text.index("snakemake"))
        self.assertLess(script_text.index("source .venv/bin/activate"), script_text.index("snakemake"))

    def test_bootstrap_requirements_include_snakemake_slurm_executor_plugin(self) -> None:
        requirements_path = WORKSPACE_ROOT / "ops" / "envs" / "dev" / "requirements-notebook.txt"
        requirement_lines = {
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("snakemake-executor-plugin-slurm", requirement_lines)

    def test_hpc_bootstrap_requirements_include_snakemake_slurm_executor_plugin(self) -> None:
        requirements_path = WORKSPACE_ROOT / "ops" / "envs" / "hpc" / "requirements-runtime.txt"
        requirement_lines = {
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("snakemake", requirement_lines)
        self.assertIn("snakemake-executor-plugin-slurm", requirement_lines)

    def test_compute_notebook_bootstrap_stamp_changes_when_bootstrap_dependency_inputs_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            (workspace_root / "ops" / "envs" / "dev").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "envs" / "hpc").mkdir(parents=True, exist_ok=True)
            (workspace_root / "ops" / "scripts").mkdir(parents=True, exist_ok=True)
            (workspace_root / "packages").mkdir(parents=True, exist_ok=True)

            bootstrap_path = workspace_root / "ops" / "envs" / "dev" / "bootstrap.sh"
            requirements_path = workspace_root / "ops" / "envs" / "dev" / "requirements-notebook.txt"
            hpc_requirements_path = workspace_root / "ops" / "envs" / "hpc" / "requirements-runtime.txt"
            detect_profile_path = workspace_root / "ops" / "scripts" / "detect_execution_profile.sh"

            bootstrap_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            requirements_path.write_text("snakemake\n", encoding="utf-8")
            hpc_requirements_path.write_text("snakemake\n", encoding="utf-8")
            detect_profile_path.write_text("#!/usr/bin/env bash\necho local\n", encoding="utf-8")

            baseline_stamp = _compute_notebook_bootstrap_stamp(workspace_root)

            requirements_path.write_text("snakemake\nsnakemake-executor-plugin-slurm\n", encoding="utf-8")
            requirements_stamp = _compute_notebook_bootstrap_stamp(workspace_root)

            hpc_requirements_path.write_text("snakemake\nsnakemake-executor-plugin-slurm\n", encoding="utf-8")
            hpc_requirements_stamp = _compute_notebook_bootstrap_stamp(workspace_root)

            detect_profile_path.write_text("#!/usr/bin/env bash\necho cluster-a\n", encoding="utf-8")
            detect_profile_stamp = _compute_notebook_bootstrap_stamp(workspace_root)

        self.assertNotEqual(baseline_stamp, requirements_stamp)
        self.assertNotEqual(requirements_stamp, hpc_requirements_stamp)
        self.assertNotEqual(hpc_requirements_stamp, detect_profile_stamp)

    def test_snakemake_resource_args_leave_runtime_unset_when_no_slurm_time_is_available(self) -> None:
        resource_args = _snakemake_resource_args(
            rule_name="fmripost_aroma",
            resources={"cpus": 4, "ram_gb": 16},
            slurm_time=None,
        )

        self.assertEqual(
            resource_args,
            [
                "--set-threads",
                "fmripost_aroma=4",
                "--set-resources",
                "fmripost_aroma:mem_mb=16384",
                "fmripost_aroma:cpus_per_task=4",
            ],
        )

    def test_snakemake_core_args_use_resolved_resource_cpus(self) -> None:
        self.assertEqual(_snakemake_core_args(resources={"cpus": 4}), ["--cores", "4"])

    def test_snakemake_core_args_use_threads_times_n_jobs_when_declared(self) -> None:
        self.assertEqual(
            _snakemake_core_args(resources={"cpus": 4, "threads": 4, "n_jobs": 10}),
            ["--cores", "40"],
        )

    def test_snakemake_resource_args_use_threads_for_rule_and_cpus_for_slurm_task(self) -> None:
        resource_args = _snakemake_resource_args(
            rule_name="deepprep_unit",
            resources={"cpus": 4, "threads": 2, "n_jobs": 10, "ram_gb": 32},
            slurm_time="24:00:00",
        )

        self.assertEqual(
            resource_args,
            [
                "--set-threads",
                "deepprep_unit=2",
                "--set-resources",
                "deepprep_unit:mem_mb=32768",
                "deepprep_unit:cpus_per_task=4",
                "deepprep_unit:runtime=24h",
            ],
        )

    def test_snakemake_resource_args_convert_slurm_walltime_to_snakemake_duration(self) -> None:
        resource_args = _snakemake_resource_args(
            rule_name="fmripost_aroma",
            resources={"cpus": 4, "ram_gb": 16},
            slurm_time="08:00:00",
        )

        self.assertEqual(
            resource_args,
            [
                "--set-threads",
                "fmripost_aroma=4",
                "--set-resources",
                "fmripost_aroma:mem_mb=16384",
                "fmripost_aroma:cpus_per_task=4",
                "fmripost_aroma:runtime=8h",
            ],
        )

    def test_run_slurm_preprocess_bids_records_profile_connection_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            ssh_config = self._write_ssh_profile_config(Path(tmp_dir))
            exit_code, _ = self._run_cli(
                ["run", "slurm", "preprocess", "bids", "--run-id", "unit-slurm-profile"],
                artifact_root,
                extra_env={
                    "RP_HPC_HOST": None,
                    "RP_HPC_PROFILE": "interactive-login",
                    "RP_SSH_CONFIG": str(ssh_config),
                    "RESEARCH_HPC_PROFILE": "",
                    "RESEARCH_HPC_ROLE": "",
                    "RESEARCH_HPC_SSH_CONFIG": str(ssh_config),
                },
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-slurm-profile" / "run-manifest.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["hpc"]["connection"]["profile"], "interactive-login")
        self.assertEqual(manifest["hpc"]["connection"]["role"], "login")
        self.assertEqual(manifest["hpc"]["connection"]["config"], str(ssh_config))

    def test_hpc_stage_and_pull_render_non_empty_commands_from_profile_defaults(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "discovered_batch.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            ssh_config = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            ssh_config.parent.mkdir(parents=True, exist_ok=True)
            write_yaml(
                ssh_config,
                {
                    "profiles": {
                        "interactive-login": {
                            "host": "cluster.example",
                            "user": "alice",
                        }
                    }
                },
            )
            os.chdir(project_root / "config")
            try:
                run_exit, _ = self._run_cli_for_workspace(
                    ["run", "slurm", "preprocess", "bids", "--run-id", "unit-stage-profile"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                    extra_env={
                        "RESEARCH_PLATFORM_ROOT": None,
                        "RP_HPC_HOST": None,
                        "RP_HPC_PROFILE": "interactive-login",
                        "RP_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml",
                        "RESEARCH_HPC_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml",
                    },
                )
                stage_exit, stage_output = self._run_cli_for_workspace(
                    ["hpc", "stage", "--run-id", "unit-stage-profile"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                    extra_env={
                        "RESEARCH_PLATFORM_ROOT": None,
                        "RP_HPC_HOST": None,
                        "RP_HPC_PROFILE": "interactive-login",
                        "RP_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml",
                        "RESEARCH_HPC_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml",
                    },
                )
                pull_exit, pull_output = self._run_cli_for_workspace(
                    ["hpc", "pull", "--run-id", "unit-stage-profile"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                    extra_env={
                        "RESEARCH_PLATFORM_ROOT": None,
                        "RP_HPC_HOST": None,
                        "RP_HPC_PROFILE": "interactive-login",
                        "RP_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml",
                        "RESEARCH_HPC_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml",
                    },
                )
            finally:
                os.chdir(original_cwd)

        stage_payload = json.loads(stage_output)
        pull_payload = json.loads(pull_output)
        self.assertEqual(run_exit, 0)
        self.assertEqual(stage_exit, 0)
        self.assertEqual(pull_exit, 0)
        self.assertEqual(stage_payload["connection"]["profile"], "interactive-login")
        self.assertEqual(stage_payload["connection"]["config"], str(ssh_config.resolve()))
        self.assertTrue(stage_payload["prepare_commands"])
        self.assertTrue(stage_payload["push_command"])
        self.assertTrue(stage_payload["submit_command"])
        self.assertEqual(pull_payload["connection"]["profile"], "interactive-login")
        self.assertEqual(pull_payload["connection"]["config"], str(ssh_config.resolve()))
        self.assertTrue(pull_payload["pull_command"])
        self.assertTrue(pull_payload["progress"])
        self.assertIn("--progress", pull_payload["pull_command"])

    def test_hpc_pull_supports_subpath_and_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            destination = Path(tmp_dir) / "exports" / "fmripost_aroma"
            self._write_bids_init_workspace(workspace_root)
            run_root = artifact_root / "runs" / "unit-pull-subpath"
            run_root.mkdir(parents=True, exist_ok=True)
            write_yaml(
                run_root / "run-manifest.yaml",
                {
                    "run_id": "unit-pull-subpath",
                    "hpc": {
                        "ssh_host": "example-hpc",
                        "remote_run_root": "remote/workspace/artifacts/runs/unit-pull-subpath",
                    },
                },
            )
            write_yaml(run_root / "status.yaml", {"run_id": "unit-pull-subpath", "state": "planned"})

            exit_code, output = self._run_cli_for_workspace(
                [
                    "hpc",
                    "pull",
                    "--run-id",
                    "unit-pull-subpath",
                    "--subpath",
                    "outputs/fmripost_aroma",
                    "--destination",
                    str(destination),
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        payload = json.loads(output)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["subpath"], "outputs/fmripost_aroma")
        self.assertEqual(payload["remote_source"], "remote/workspace/artifacts/runs/unit-pull-subpath/outputs/fmripost_aroma")
        self.assertEqual(payload["destination"], str(destination.resolve()))
        self.assertTrue(payload["progress"])
        self.assertIn("--progress", payload["pull_command"])

    def test_hpc_pull_rejects_invalid_subpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            self._write_bids_init_workspace(workspace_root)
            run_root = artifact_root / "runs" / "unit-invalid-subpath"
            run_root.mkdir(parents=True, exist_ok=True)
            write_yaml(
                run_root / "run-manifest.yaml",
                {
                    "run_id": "unit-invalid-subpath",
                    "hpc": {
                        "ssh_host": "example-hpc",
                        "remote_run_root": "remote/workspace/artifacts/runs/unit-invalid-subpath",
                    },
                },
            )
            write_yaml(run_root / "status.yaml", {"run_id": "unit-invalid-subpath", "state": "planned"})

            message = self._run_cli_system_exit_for_workspace(
                ["hpc", "pull", "--run-id", "unit-invalid-subpath", "--subpath", "../outputs"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertIn("Pull subpath", message)

    def test_run_submit_preprocess_bids_uses_profile_path_without_rp_hpc_host(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "discovered_batch.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            ssh_config = workspace_root / "secrets" / "hpc" / "ssh-profiles.yaml"
            ssh_config.parent.mkdir(parents=True, exist_ok=True)
            write_yaml(
                ssh_config,
                {
                    "profiles": {
                        "interactive-login": {
                            "host": "cluster.example",
                            "user": "alice",
                        }
                    }
                },
            )
            with mock.patch(
                "research_platform.core.cli.execute_stage_plan",
                return_value={"ok": True, "returncode": 0, "commands": [{"command": ["rsync"]}]},
            ) as stage_mock:
                with mock.patch(
                    "research_platform.core.cli.execute_submit_plan",
                    return_value={
                        "ok": True,
                        "returncode": 0,
                        "command": ["ssh", "alice@cluster.example", "cd remote/run && sbatch submit.sbatch"],
                        "stdout": "Submitted batch job 54321\n",
                        "stderr": "",
                        "job_id": "54321",
                    },
                ) as submit_mock:
                    os.chdir(project_root / "config")
                    try:
                        exit_code, output = self._run_cli_for_workspace(
                            ["run", "submit", "preprocess", "bids", "--run-id", "unit-submit-profile", "--execute"],
                            workspace_root=workspace_root,
                            artifact_root=artifact_root,
                            extra_env={
                                "RESEARCH_PLATFORM_ROOT": None,
                                "RP_HPC_HOST": None,
                                "RP_HPC_PROFILE": "interactive-login",
                                "RP_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml",
                                "RESEARCH_HPC_SSH_CONFIG": "secrets/hpc/ssh-profiles.yaml",
                            },
                        )
                    finally:
                        os.chdir(original_cwd)
            stage_plan = load_yaml(artifact_root / "runs" / "unit-submit-profile" / "hpc" / "stage-plan.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(stage_mock.call_count, 1)
        self.assertEqual(submit_mock.call_count, 1)
        self.assertEqual(stage_plan["connection"]["profile"], "interactive-login")
        self.assertEqual(stage_plan["connection"]["config"], str(ssh_config.resolve()))
        prepare_command = stage_plan["prepare_commands"][0][-1]
        self.assertIn("remote/workspace/artifacts/runs/unit-submit-profile/logs", prepare_command)
        self.assertIn("remote/workspace/artifacts/runs/unit-submit-profile/work", prepare_command)
        self.assertIn("remote/workspace/artifacts/runs/unit-submit-profile/outputs", prepare_command)
        self.assertIn("Submit job id: 54321", output)

    def test_hpc_stage_reports_clean_error_when_saved_profile_config_is_missing(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "discovered_batch.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )
            os.chdir(project_root / "config")
            try:
                run_exit, _ = self._run_cli_for_workspace(
                    ["run", "slurm", "preprocess", "bids", "--run-id", "unit-stage-missing-config"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                    extra_env={
                        "RESEARCH_PLATFORM_ROOT": None,
                        "RP_HPC_HOST": None,
                        "RP_HPC_PROFILE": "interactive-login",
                        "RP_SSH_CONFIG": "secrets/hpc/missing.yaml",
                        "RESEARCH_HPC_SSH_CONFIG": "secrets/hpc/missing.yaml",
                    },
                )
                message = self._run_cli_system_exit_for_workspace(
                    ["hpc", "stage", "--run-id", "unit-stage-missing-config"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                    extra_env={
                        "RESEARCH_PLATFORM_ROOT": None,
                        "RP_HPC_HOST": None,
                        "RP_HPC_PROFILE": "interactive-login",
                        "RP_SSH_CONFIG": "secrets/hpc/missing.yaml",
                        "RESEARCH_HPC_SSH_CONFIG": "secrets/hpc/missing.yaml",
                    },
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(run_exit, 0)
        self.assertIn("SSH profile config was not found:", message)
        self.assertIn(str((workspace_root / "secrets" / "hpc" / "missing.yaml").resolve()), message)

    def test_cli_reports_clean_error_when_workspace_root_cannot_be_discovered(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            outside_root = Path(tmp_dir) / "outside"
            outside_root.mkdir(parents=True, exist_ok=True)
            os.chdir(outside_root)
            try:
                message = self._run_cli_system_exit_for_workspace(
                    ["config", "show"],
                    workspace_root=outside_root,
                    artifact_root=outside_root / "artifacts",
                    extra_env={"RESEARCH_PLATFORM_ROOT": None},
                )
            finally:
                os.chdir(original_cwd)

        self.assertIn("Could not find WORKSPACE.yaml", message)
        self.assertNotIn("Traceback", message)

    def test_config_command_reports_missing_project_overlay_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_tabular_init_workspace(workspace_root)
            message = self._run_cli_system_exit_for_workspace(
                ["config", "show", "--project", "project-missing-neutral"],
                workspace_root=workspace_root,
                artifact_root=workspace_root / "artifacts",
            )

        self.assertIn("Project overlay 'project-missing-neutral' was not found", message)
        self.assertIn("project/project-missing-neutral", message)
        self.assertIn("Expected a project overlay under project/", message)
        self.assertIn("rp project init project-missing-neutral", message)
        self.assertNotIn("Traceback", message)

    def test_hpc_command_reports_missing_project_overlay_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_tabular_init_workspace(workspace_root)
            message = self._run_cli_system_exit_for_workspace(
                ["hpc", "doctor", "--project", "project-missing-neutral"],
                workspace_root=workspace_root,
                artifact_root=workspace_root / "artifacts",
            )

        self.assertIn("Project overlay 'project-missing-neutral' was not found", message)
        self.assertIn("project/project-missing-neutral", message)
        self.assertIn("rp project init project-missing-neutral", message)
        self.assertNotIn("Traceback", message)

    def test_analysis_roi_command_reports_missing_project_overlay_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            self._write_tabular_init_workspace(workspace_root)
            message = self._run_cli_system_exit_for_workspace(
                ["analysis", "roi", "list", "--project", "project-missing-neutral"],
                workspace_root=workspace_root,
                artifact_root=workspace_root / "artifacts",
            )

        self.assertIn("Project overlay 'project-missing-neutral' was not found", message)
        self.assertIn("project/project-missing-neutral", message)
        self.assertIn("rp project init project-missing-neutral", message)
        self.assertNotIn("Traceback", message)

    def test_hpc_stage_reports_clean_error_when_profile_and_host_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_exit, _ = self._run_cli(
                ["run", "slurm", "preprocess", "bids", "--run-id", "unit-stage-missing-connection"],
                artifact_root,
                extra_env={
                    "RP_HPC_HOST": None,
                    "RP_HPC_PROFILE": "",
                    "RESEARCH_HPC_PROFILE": "",
                    "RP_SSH_CONFIG": "",
                    "RESEARCH_HPC_SSH_CONFIG": "",
                },
            )
            message = self._run_cli_system_exit_for_workspace(
                ["hpc", "stage", "--run-id", "unit-stage-missing-connection"],
                workspace_root=WORKSPACE_ROOT,
                artifact_root=artifact_root,
                extra_env={
                    "RP_HPC_HOST": None,
                    "RP_HPC_PROFILE": "",
                    "RESEARCH_HPC_PROFILE": "",
                    "RP_SSH_CONFIG": "",
                    "RESEARCH_HPC_SSH_CONFIG": "",
                },
            )

        self.assertEqual(run_exit, 0)
        self.assertIn("HPC connection is not configured.", message)

    def test_publish_back_plan_command_writes_run_local_plan_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_exit, _ = self._run_cli(
                ["run", "plan", "preprocess", "bids", "--run-id", "unit-publish-back-empty"],
                artifact_root,
            )
            plan_exit, plan_output = self._run_cli(
                ["publish-back", "plan", "--run-id", "unit-publish-back-empty"],
                artifact_root,
            )
            plan_payload = json.loads(plan_output)
            plan = load_yaml(artifact_root / "runs" / "unit-publish-back-empty" / "publish-back-plan.yaml")

        self.assertEqual(run_exit, 0)
        self.assertEqual(plan_exit, 0)
        self.assertEqual(
            plan_payload["plan"],
            str((artifact_root / "runs" / "unit-publish-back-empty" / "publish-back-plan.yaml").resolve()),
        )
        self.assertEqual(plan["run_id"], "unit-publish-back-empty")
        self.assertEqual(plan["default_policy"], "never")
        self.assertEqual(plan["summary"], {"total_items": 1, "actionable_items": 0})
        self.assertEqual(plan["items"][0]["action"], "skip")
        self.assertEqual(plan["items"][0]["reason"], "source missing")

    def test_publish_back_plan_command_plans_if_absent_and_overwrite_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_exit, _ = self._run_cli(
                ["run", "plan", "preprocess", "bids", "--run-id", "unit-publish-back-actions"],
                artifact_root,
            )
            run_root = artifact_root / "runs" / "unit-publish-back-actions"
            source_a = run_root / "outputs" / "report-a.txt"
            source_b = run_root / "outputs" / "report-b.txt"
            destination_a = Path(tmp_dir) / "canonical" / "report-a.txt"
            destination_b = Path(tmp_dir) / "canonical" / "report-b.txt"
            source_a.write_text("artifact-a", encoding="utf-8")
            source_b.write_text("artifact-b", encoding="utf-8")
            destination_b.parent.mkdir(parents=True, exist_ok=True)
            destination_b.write_text("canonical-b", encoding="utf-8")

            manifest_path = run_root / "run-manifest.yaml"
            manifest = load_yaml(manifest_path)
            manifest["publish_back"]["default_policy"] = "if_absent"
            manifest["publish_back"]["items"] = [
                {
                    "source": str(source_a.resolve()),
                    "destination": str(destination_a),
                },
                {
                    "source": str(source_b.resolve()),
                    "destination": str(destination_b),
                    "policy": "overwrite",
                },
            ]
            write_yaml(manifest_path, manifest)

            plan_exit, _ = self._run_cli(
                ["publish-back", "plan", "--run-id", "unit-publish-back-actions"],
                artifact_root,
            )
            plan = load_yaml(run_root / "publish-back-plan.yaml")

        self.assertEqual(run_exit, 0)
        self.assertEqual(plan_exit, 0)
        self.assertEqual([item["policy"] for item in plan["items"]], ["if_absent", "overwrite"])
        self.assertEqual([item["action"] for item in plan["items"]], ["copy", "overwrite"])
        self.assertEqual(plan["summary"], {"total_items": 2, "actionable_items": 2})

    def test_slurm_run_supports_stage_pull_and_cancel_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code, _ = self._run_cli(
                ["run", "slurm", "preprocess", "bids", "--run-id", "unit-slurm"],
                artifact_root,
                extra_env={
                    "RP_HPC_PROFILE": "",
                    "RESEARCH_HPC_PROFILE": "",
                    "RP_SSH_CONFIG": "",
                    "RESEARCH_HPC_SSH_CONFIG": "",
                },
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-slurm" / "run-manifest.yaml")
            self.assertEqual(exit_code, 0)
            submit_script = artifact_root / "runs" / "unit-slurm" / "submit.sbatch"
            self.assertTrue(submit_script.exists())
            submit_text = submit_script.read_text(encoding="utf-8")

            stage_exit, stage_output = self._run_cli(["hpc", "stage", "--run-id", "unit-slurm"], artifact_root)
            bootstrap_exit, bootstrap_output = self._run_cli(["hpc", "bootstrap", "--run-id", "unit-slurm"], artifact_root)
            pull_exit, pull_output = self._run_cli(["hpc", "pull", "--run-id", "unit-slurm"], artifact_root)
            cancel_exit, cancel_output = self._run_cli(["hpc", "cancel", "--run-id", "unit-slurm"], artifact_root)
            status_payload = load_yaml(artifact_root / "runs" / "unit-slurm" / "status.yaml")
            stage_payload = json.loads(stage_output)
            bootstrap_payload = json.loads(bootstrap_output)
            pull_payload = json.loads(pull_output)
            cancel_payload = json.loads(cancel_output)
            status_path = str((artifact_root / "runs" / "unit-slurm" / "status.yaml").resolve())

        self.assertEqual(stage_exit, 0)
        self.assertNotIn(str(artifact_root), " ".join(manifest["execution"]["command"]))
        self.assertNotIn(str(artifact_root), submit_text)
        self.assertIn("remote/workspace/artifacts/runs/unit-slurm/work", " ".join(manifest["execution"]["command"]))
        self.assertIn("remote/workspace/artifacts/runs/unit-slurm/outputs", " ".join(manifest["execution"]["command"]))
        self.assertEqual(manifest["resources"]["workload"], "bids_preprocess")
        self.assertNotEqual(manifest["resources"]["preset"], "legacy")
        self.assertEqual(manifest["slurm"]["jobspec"]["cpus"], 4)
        self.assertEqual(manifest["slurm"]["jobspec"]["mem"], "16G")
        self.assertIn("rsync", " ".join(json.loads(stage_output)["push_command"]))
        self.assertGreaterEqual(len(json.loads(stage_output)["push_commands"]), 3)
        self.assertEqual(bootstrap_exit, 0)
        self.assertEqual(pull_exit, 0)
        self.assertIn("rsync", " ".join(json.loads(pull_output)["pull_command"]))
        self.assertEqual(cancel_exit, 0)
        self.assertEqual(json.loads(cancel_output)["cancel_command"], [])
        self.assertIn(status_path, stage_payload["local_files_written"])
        self.assertIn(status_path, bootstrap_payload["local_files_written"])
        self.assertIn(status_path, pull_payload["local_files_written"])
        self.assertEqual(cancel_payload["local_files_written"], [status_path])
        self.assertEqual(status_payload["state"], "cancel-requested")

    def test_bids_slurm_uses_remote_artifacts_root_for_remote_run_paths_and_stage_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_bids_discovery_workspace(workspace_root)
            (project_root / "manifests" / "batches" / "discovered_batch.tsv").write_text(
                "subject_id\tsession_id\ttask_id\trun_id\nsub-001\tses-01\ttask-rest\trun-01\n",
                encoding="utf-8",
            )

            plan_exit, _ = self._run_cli_for_workspace(
                ["run", "slurm", "preprocess", "bids", "--run-id", "unit-bids-remote-artifacts"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "RP_REMOTE_WORKSPACE_ROOT": "/remote/workspace",
                    "RP_REMOTE_ARTIFACTS_ROOT": "/remote/scratch/shared-artifacts",
                    "RP_HPC_PROFILE": "",
                    "RESEARCH_HPC_PROFILE": "",
                    "RP_SSH_CONFIG": "",
                    "RESEARCH_HPC_SSH_CONFIG": "",
                },
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-bids-remote-artifacts" / "run-manifest.yaml")

            stage_exit, _ = self._run_cli_for_workspace(
                ["hpc", "stage", "--run-id", "unit-bids-remote-artifacts"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "RP_REMOTE_WORKSPACE_ROOT": "/remote/workspace",
                    "RP_REMOTE_ARTIFACTS_ROOT": "/remote/scratch/shared-artifacts",
                    "RP_HPC_PROFILE": "",
                    "RESEARCH_HPC_PROFILE": "",
                    "RP_SSH_CONFIG": "",
                    "RESEARCH_HPC_SSH_CONFIG": "",
                },
            )
            stage_plan = load_yaml(artifact_root / "runs" / "unit-bids-remote-artifacts" / "hpc" / "stage-plan.yaml")

        self.assertEqual(plan_exit, 0)
        self.assertEqual(stage_exit, 0)
        self.assertEqual(manifest["hpc"]["remote_run_root"], "/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts")
        self.assertEqual(manifest["execution"]["work_dir"], "/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/work")
        self.assertEqual(manifest["execution"]["output_dir"], "/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/outputs")
        self.assertEqual(manifest["execution"]["log_dir"], "/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/logs")
        self.assertEqual(manifest["execution"]["status_path"], "/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/status.yaml")
        self.assertEqual(manifest["slurm"]["jobspec"]["log_out"], "/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/logs/slurm.out")
        self.assertEqual(manifest["slurm"]["jobspec"]["log_err"], "/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/logs/slurm.err")
        prepare_command = " ".join(stage_plan["prepare_commands"][0])
        self.assertIn("/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts", prepare_command)
        self.assertIn("/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/work", prepare_command)
        self.assertIn("/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/outputs", prepare_command)
        self.assertIn("/remote/scratch/shared-artifacts/runs/unit-bids-remote-artifacts/logs", prepare_command)

    def test_config_validate_succeeds_for_tabular_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exit_code, output = self._run_cli(
                ["config", "validate", "--project", "project-pilot-tabular"],
                Path(tmp_dir) / "artifacts",
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(json.loads(output)["valid"])

    def test_tabular_run_plan_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            exit_code, _ = self._run_cli(
                ["run", "plan", "preprocess", "tabular", "--project", "project-pilot-tabular", "--run-id", "unit-tabular-plan"],
                artifact_root,
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-tabular-plan" / "run-manifest.yaml")

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["slice"], "tabular")
        self.assertEqual(manifest["workflow"]["action"], "preprocess")
        self.assertEqual(manifest["workflow"]["target"], "tabular")
        self.assertEqual(
            manifest["dataset"]["feature_table"],
            "datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv",
        )
        self.assertEqual(manifest["resources"]["workload"], "tabular_preprocess")
        self.assertNotEqual(manifest["resources"]["preset"], "legacy")
        self.assertEqual(
            manifest["predictor_contract"],
            {
                "target_column": "binary_target",
                "feature_columns": TABULAR_PREDICTOR_COLUMNS,
                "feature_count": 6,
            },
        )
        self.assertEqual(sorted(manifest["outputs"]), ["features_table", "prep_plan", "split_manifest"])

    def test_tabular_predictor_contract_failures_create_no_run_tree(self) -> None:
        invalid_contracts = (
            ("missing", None, "ordered, nonempty YAML sequence"),
            ("non-list", "feature_a", "ordered, nonempty YAML sequence"),
            ("empty", [], "at least one predictor"),
            ("blank", [""], "nonblank string"),
            ("non-string", [1], "nonblank string"),
            ("duplicate", ["feature_a", "feature_a"], "duplicate predictor"),
            ("target", ["binary_target"], "selected target_column"),
            ("reserved", ["split_set"], "reserved generated column"),
            ("unknown", ["missing_feature"], "Unknown feature columns"),
            ("nonnumeric", ["record_id"], "must be numeric"),
        )
        for case_name, feature_columns, expected_message in invalid_contracts:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmp_dir:
                workspace_root = Path(tmp_dir)
                artifact_root = workspace_root / "artifacts"
                self._write_tabular_slurm_workspace(workspace_root)
                models_path = workspace_root / "project" / "test-tabular" / "config" / "models.yaml"
                models_document = load_yaml(models_path)
                if feature_columns is None:
                    del models_document["models"]["default"]["feature_columns"]
                else:
                    models_document["models"]["default"]["feature_columns"] = feature_columns
                write_yaml(models_path, models_document)
                for action, target in (("preprocess", "tabular"), ("train", "model")):
                    with self.subTest(case=case_name, action=action):
                        run_id = f"invalid-predictors-{case_name}-{action}"
                        message = self._run_cli_system_exit_for_workspace(
                            [
                                "run",
                                "plan",
                                action,
                                target,
                                "--project",
                                "test-tabular",
                                "--run-id",
                                run_id,
                            ],
                            workspace_root=workspace_root,
                            artifact_root=artifact_root,
                            extra_env={
                                "RP_REMOTE_WORKSPACE_ROOT": "/remote/workspace",
                                "RP_REMOTE_ARTIFACTS_ROOT": "/remote/artifacts",
                            },
                        )

                        self.assertIn(expected_message, message)
                        self.assertNotIn("Traceback", message)
                        self.assertFalse((artifact_root / "runs" / run_id).exists())

    def test_tabular_stage_scripts_enforce_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            preprocess_exit, _ = self._run_cli(
                [
                    "run",
                    "local",
                    "preprocess",
                    "tabular",
                    "--project",
                    "project-pilot-tabular",
                    "--run-id",
                    "unit-tabular-preprocess",
                    "--dry-run",
                ],
                artifact_root,
            )
            train_exit, _ = self._run_cli(
                [
                    "run",
                    "local",
                    "train",
                    "model",
                    "--project",
                    "project-pilot-tabular",
                    "--run-id",
                    "unit-tabular-train",
                    "--dry-run",
                ],
                artifact_root,
            )
            train_execute_exit, _ = self._execute_mocked_tabular_success(
                [
                    "run",
                    "local",
                    "train",
                    "model",
                    "--project",
                    "project-pilot-tabular",
                    "--run-id",
                    "unit-tabular-train",
                    "--execute",
                ],
                artifact_root=artifact_root,
            )
            evaluate_exit, _ = self._run_cli(
                [
                    "run",
                    "local",
                    "evaluate",
                    "model",
                    "--project",
                    "project-pilot-tabular",
                    "--run-id",
                    "unit-tabular-evaluate",
                    "--input-run",
                    "unit-tabular-train",
                    "--dry-run",
                ],
                artifact_root,
            )

            preprocess_script = (artifact_root / "runs" / "unit-tabular-preprocess" / "execute.sh").read_text(encoding="utf-8")
            train_script = (artifact_root / "runs" / "unit-tabular-train" / "execute.sh").read_text(encoding="utf-8")
            evaluate_script = (artifact_root / "runs" / "unit-tabular-evaluate" / "execute.sh").read_text(encoding="utf-8")
            preprocess_manifest = load_yaml(artifact_root / "runs" / "unit-tabular-preprocess" / "run-manifest.yaml")
            train_manifest = load_yaml(artifact_root / "runs" / "unit-tabular-train" / "run-manifest.yaml")
            evaluate_manifest = load_yaml(artifact_root / "runs" / "unit-tabular-evaluate" / "run-manifest.yaml")

        self.assertEqual(preprocess_exit, 0)
        self.assertEqual(train_exit, 0)
        self.assertEqual(train_execute_exit, 0)
        self.assertEqual(evaluate_exit, 0)
        self.assertIn("split create", preprocess_script)
        self.assertIn("prep fit", preprocess_script)
        self.assertIn("prep apply", preprocess_script)
        self.assertNotIn("model train", preprocess_script)
        self.assertNotIn("model evaluate", preprocess_script)
        self.assertIn("model train", train_script)
        self.assertNotIn("model evaluate", train_script)
        self.assertIn("model evaluate", evaluate_script)
        self.assertNotIn("split create", evaluate_script)
        self.assertNotIn("prep fit", evaluate_script)
        self.assertNotIn("prep apply", evaluate_script)
        self.assertNotIn("model train", evaluate_script)
        preprocess_commands = _analysis_cli_commands(preprocess_script)
        train_commands = _analysis_cli_commands(train_script)
        evaluate_commands = _analysis_cli_commands(evaluate_script)
        preprocess_by_action = {(command[3], command[4]): command for command in preprocess_commands}
        train_by_action = {(command[3], command[4]): command for command in train_commands}
        self.assertEqual(
            _command_option_values(preprocess_by_action[("prep", "fit")], "--feature-columns"),
            TABULAR_PREDICTOR_COLUMNS,
        )
        self.assertEqual(
            _command_option_values(train_by_action[("prep", "fit")], "--feature-columns"),
            TABULAR_PREDICTOR_COLUMNS,
        )
        self.assertEqual(
            _command_option_values(train_by_action[("model", "train")], "--feature-columns"),
            TABULAR_PREDICTOR_COLUMNS,
        )
        for command in (
            preprocess_by_action[("split", "create")],
            preprocess_by_action[("prep", "apply")],
            train_by_action[("split", "create")],
            train_by_action[("prep", "apply")],
            *evaluate_commands,
        ):
            self.assertNotIn("--feature-columns", command)
        expected_contract = {
            "target_column": "binary_target",
            "feature_columns": TABULAR_PREDICTOR_COLUMNS,
            "feature_count": 6,
        }
        self.assertEqual(preprocess_manifest["predictor_contract"], expected_contract)
        self.assertEqual(train_manifest["predictor_contract"], expected_contract)
        self.assertEqual(evaluate_manifest["predictor_contract"], expected_contract)
        self.assertEqual(evaluate_manifest["input_run"]["run_id"], "unit-tabular-train")
        self.assertEqual(evaluate_manifest["input_run"]["predictor_contract"], expected_contract)
        self.assertEqual(sorted(evaluate_manifest["outputs"]), ["evaluation"])

    def test_tabular_evaluation_uses_source_run_predictor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_slurm_workspace(workspace_root)
            train_exit, _ = self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "train",
                    "model",
                    "--project",
                    "test-tabular",
                    "--run-id",
                    "source-train",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            source_manifest = load_yaml(artifact_root / "runs" / "source-train" / "run-manifest.yaml")
            train_execute_exit, _ = self._execute_mocked_tabular_success(
                [
                    "run",
                    "local",
                    "train",
                    "model",
                    "--project",
                    "test-tabular",
                    "--run-id",
                    "source-train",
                    "--execute",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

            models_path = workspace_root / "project" / "test-tabular" / "config" / "models.yaml"
            models_document = load_yaml(models_path)
            models_document["models"]["default"]["feature_columns"] = ["record_id"]
            write_yaml(models_path, models_document)

            evaluate_exit, _ = self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "evaluate",
                    "model",
                    "--project",
                    "test-tabular",
                    "--run-id",
                    "source-evaluate",
                    "--input-run",
                    "source-train",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            evaluation_manifest = load_yaml(
                artifact_root / "runs" / "source-evaluate" / "run-manifest.yaml"
            )

        self.assertEqual(train_exit, 0)
        self.assertEqual(train_execute_exit, 0)
        self.assertEqual(evaluate_exit, 0)
        self.assertEqual(
            source_manifest["predictor_contract"],
            {
                "target_column": "binary_target",
                "feature_columns": ["feature_a"],
                "feature_count": 1,
            },
        )
        self.assertEqual(evaluation_manifest["predictor_contract"], source_manifest["predictor_contract"])
        self.assertEqual(
            evaluation_manifest["input_run"]["predictor_contract"],
            source_manifest["predictor_contract"],
        )

    def test_tabular_local_evaluate_executes_from_existing_train_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            train_exit, _ = self._run_cli(
                ["run", "local", "train", "model", "--project", "project-pilot-tabular", "--run-id", "unit-tabular-train", "--execute"],
                artifact_root,
            )
            evaluate_exit, _ = self._run_cli(
                [
                    "run",
                    "local",
                    "evaluate",
                    "model",
                    "--project",
                    "project-pilot-tabular",
                    "--run-id",
                    "unit-tabular-local",
                    "--input-run",
                    "unit-tabular-train",
                    "--execute",
                ],
                artifact_root,
            )
            train_run_root = artifact_root / "runs" / "unit-tabular-train"
            train_manifest = load_yaml(train_run_root / "run-manifest.yaml")
            prep_payload = json.loads((train_run_root / "outputs" / "prep.json").read_text(encoding="utf-8"))
            model_payload = json.loads((train_run_root / "outputs" / "model.json").read_text(encoding="utf-8"))
            evaluation_manifest = load_yaml(artifact_root / "runs" / "unit-tabular-local" / "run-manifest.yaml")
            evaluation_payload = json.loads(
                (artifact_root / "runs" / "unit-tabular-local" / "outputs" / "evaluation.json").read_text(
                    encoding="utf-8"
                )
            )
            status = load_yaml(artifact_root / "runs" / "unit-tabular-local" / "status.yaml")

        self.assertEqual(train_exit, 0)
        self.assertEqual(evaluate_exit, 0)
        self.assertEqual(sorted(evaluation_manifest["outputs"]), ["evaluation"])
        self.assertEqual(evaluation_manifest["input_run"]["run_id"], "unit-tabular-train")
        self.assertEqual(prep_payload["feature_columns"], TABULAR_PREDICTOR_COLUMNS)
        self.assertEqual(model_payload["feature_columns"], TABULAR_PREDICTOR_COLUMNS)
        self.assertEqual(evaluation_payload["feature_columns"], TABULAR_PREDICTOR_COLUMNS)
        self.assertEqual(train_manifest["predictor_contract"]["feature_count"], 6)
        self.assertEqual(
            evaluation_manifest["input_run"]["predictor_contract"],
            train_manifest["predictor_contract"],
        )
        self.assertEqual(status["state"], "succeeded")

    def test_tabular_slurm_run_writes_submit_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            ssh_config = self._write_ssh_profile_config(Path(tmp_dir))
            exit_code, _ = self._run_cli(
                ["run", "slurm", "train", "model", "--project", "project-pilot-tabular", "--run-id", "unit-tabular-slurm"],
                artifact_root,
                extra_env={
                    "RP_HPC_PROFILE": "interactive-login",
                    "RP_SSH_CONFIG": str(ssh_config),
                    "RESEARCH_HPC_PROFILE": "",
                    "RESEARCH_HPC_ROLE": "",
                    "RESEARCH_HPC_SSH_CONFIG": str(ssh_config),
                },
            )
            run_root = artifact_root / "runs" / "unit-tabular-slurm"
            manifest = load_yaml(run_root / "run-manifest.yaml")
            execute_script = (run_root / "execute.sh").read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertTrue((run_root / "submit.sbatch").exists())
            self.assertTrue((run_root / "execute.sh").exists())
            self.assertEqual(manifest["hpc"]["connection"]["profile"], "interactive-login")
            commands = _analysis_cli_commands(execute_script)
            commands_by_action = {(command[3], command[4]): command for command in commands}
            self.assertEqual(
                _command_option_values(commands_by_action[("prep", "fit")], "--feature-columns"),
                TABULAR_PREDICTOR_COLUMNS,
            )
            self.assertEqual(
                _command_option_values(commands_by_action[("model", "train")], "--feature-columns"),
                TABULAR_PREDICTOR_COLUMNS,
            )
            self.assertEqual(manifest["predictor_contract"]["feature_columns"], TABULAR_PREDICTOR_COLUMNS)

    def test_tabular_slurm_uses_remote_artifacts_root_and_stages_execute_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_slurm_workspace(workspace_root)

            plan_exit, _ = self._run_cli_for_workspace(
                ["run", "slurm", "train", "model", "--project", "test-tabular", "--run-id", "unit-tabular-remote-artifacts"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "RP_REMOTE_WORKSPACE_ROOT": "/remote/workspace",
                    "RP_REMOTE_ARTIFACTS_ROOT": "/remote/project/shared-artifacts",
                    "RP_HPC_PROFILE": "",
                    "RESEARCH_HPC_PROFILE": "",
                    "RP_SSH_CONFIG": "",
                    "RESEARCH_HPC_SSH_CONFIG": "",
                },
            )
            manifest = load_yaml(artifact_root / "runs" / "unit-tabular-remote-artifacts" / "run-manifest.yaml")
            submit_script = (artifact_root / "runs" / "unit-tabular-remote-artifacts" / "submit.sbatch").read_text(encoding="utf-8")
            execute_script = (artifact_root / "runs" / "unit-tabular-remote-artifacts" / "execute.sh").read_text(encoding="utf-8")

            stage_exit, _ = self._run_cli_for_workspace(
                ["hpc", "stage", "--run-id", "unit-tabular-remote-artifacts"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "RP_REMOTE_WORKSPACE_ROOT": "/remote/workspace",
                    "RP_REMOTE_ARTIFACTS_ROOT": "/remote/project/shared-artifacts",
                    "RP_HPC_PROFILE": "",
                    "RESEARCH_HPC_PROFILE": "",
                    "RP_SSH_CONFIG": "",
                    "RESEARCH_HPC_SSH_CONFIG": "",
                },
            )
            stage_plan = load_yaml(artifact_root / "runs" / "unit-tabular-remote-artifacts" / "hpc" / "stage-plan.yaml")

        self.assertEqual(plan_exit, 0)
        self.assertEqual(stage_exit, 0)
        self.assertEqual(manifest["hpc"]["remote_run_root"], "/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts")
        self.assertEqual(manifest["execution"]["work_dir"], "/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts/work")
        self.assertEqual(manifest["execution"]["output_dir"], "/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts/outputs")
        self.assertEqual(manifest["execution"]["log_dir"], "/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts/logs")
        self.assertEqual(manifest["execution"]["status_path"], "/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts/status.yaml")
        self.assertEqual(
            manifest["execution"]["command"],
            ["bash", "/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts/execute.sh"],
        )
        self.assertIn("bash /remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts/execute.sh", submit_script)
        self.assertIn("cd /remote/workspace", execute_script)
        self.assertIn("packages/research-analysis/src:packages/research-ml/src", execute_script)
        self.assertIn("/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts/outputs/model.json", execute_script)
        self.assertIn("datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv", execute_script)
        staged_files = stage_plan["staged_files"]
        self.assertTrue(any(path.endswith("/execute.sh") for path in staged_files))
        self.assertEqual(stage_plan["push_commands"][0]["destination"], "/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts")
        prepare_command = " ".join(stage_plan["prepare_commands"][0])
        self.assertIn("/remote/project/shared-artifacts/runs/unit-tabular-remote-artifacts", prepare_command)

    def test_tabular_plan_is_one_shot_and_exact_same_id_execution_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_id = "reviewed-preprocess"
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
            plan_exit, _ = self._run_cli(plan_args, artifact_root)
            run_root = artifact_root / "runs" / run_id
            claim = artifact_root / "runs" / f".{run_id}.claim"
            manifest = load_yaml(run_root / "run-manifest.yaml")
            status = load_yaml(run_root / "status.yaml")
            original_created_at = manifest["created_at"]
            original_identity = manifest["plan_identity"]

            self.assertEqual(plan_exit, 0)
            self.assertEqual(status["state"], "planned")
            self.assertFalse(claim.exists())
            self.assertEqual(
                original_identity["files"]["execute.sh"]["sha256"],
                hashlib.sha256((run_root / "execute.sh").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                original_identity["schema_version"],
                "research_platform.core.run_plan_identity.v1",
            )
            self.assertEqual(len(original_identity["sha256"]), 64)

            planned_snapshot = _filesystem_snapshot(run_root)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=AssertionError("a repeated plan must not execute"),
            ) as process_mock:
                code, output = self._run_cli_failure_for_workspace(
                    plan_args,
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), planned_snapshot)
            process_mock.assert_not_called()

            observed_states: list[str] = []

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
                observed_states.append(load_yaml(run_root / "status.yaml")["state"])
                _write_mock_tabular_scientific_outputs(kwargs)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("research_platform.core.cli.shutil.which", return_value="/bin/bash"):
                with mock.patch("research_platform.core.cli.subprocess.run", side_effect=fake_run) as process_mock:
                    execute_exit, _ = self._run_cli(execute_args, artifact_root)

            completed_manifest = load_yaml(run_root / "run-manifest.yaml")
            completed_status = load_yaml(run_root / "status.yaml")
            self.assertEqual(execute_exit, 0)
            self.assertEqual(observed_states, ["running"])
            process_mock.assert_called_once()
            self.assertEqual(completed_status["state"], "succeeded")
            self.assertEqual(completed_manifest["created_at"], original_created_at)
            self.assertEqual(completed_manifest["plan_identity"], original_identity)
            self.assertFalse(claim.exists())

            completed_snapshot = _filesystem_snapshot(run_root)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=AssertionError("a completed run must not execute twice"),
            ) as process_mock:
                code, output = self._run_cli_failure_for_workspace(
                    execute_args,
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), completed_snapshot)
            process_mock.assert_not_called()

            status_snapshot = _filesystem_snapshot(run_root)
            status_exit, status_output = self._run_cli(["hpc", "status", "--run-id", run_id], artifact_root)
            self.assertEqual(status_exit, 0)
            self.assertEqual(json.loads(status_output)["state"], "succeeded")
            self.assertEqual(json.loads(status_output)["run_id"], run_id)
            self.assertEqual(_filesystem_snapshot(run_root), status_snapshot)

    def test_tabular_local_dry_run_can_transition_once_to_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_id = "reviewed-local-dry-run"
            base_args = [
                "run",
                "local",
                "preprocess",
                "tabular",
                "--project",
                "project-pilot-tabular",
                "--run-id",
                run_id,
            ]
            dry_exit, _ = self._run_cli([*base_args, "--dry-run"], artifact_root)
            run_root = artifact_root / "runs" / run_id
            original_manifest = load_yaml(run_root / "run-manifest.yaml")

            def fake_run(
                command: list[str],
                **kwargs: object,
            ) -> subprocess.CompletedProcess[list[str]]:
                _write_mock_tabular_scientific_outputs(kwargs)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("research_platform.core.cli.shutil.which", return_value="/bin/bash"):
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=fake_run,
                ) as process_mock:
                    execute_exit, _ = self._run_cli([*base_args, "--execute"], artifact_root)

            self.assertEqual(dry_exit, 0)
            self.assertEqual(execute_exit, 0)
            process_mock.assert_called_once()
            completed_manifest = load_yaml(run_root / "run-manifest.yaml")
            self.assertEqual(completed_manifest["created_at"], original_manifest["created_at"])
            self.assertEqual(completed_manifest["plan_identity"], original_manifest["plan_identity"])
            self.assertEqual(load_yaml(run_root / "status.yaml")["state"], "succeeded")

    def test_tabular_fresh_execution_records_truthful_terminal_states(self) -> None:
        cases = (
            ("success", subprocess.CompletedProcess(["bash"], 0), 0, "succeeded", False),
            ("nonzero", subprocess.CompletedProcess(["bash"], 7), 1, "failed", False),
            ("interrupt", KeyboardInterrupt(), 130, "failed", True),
            ("launch-error", OSError("synthetic launch failure"), 1, "failed", False),
        )
        for label, result, expected_exit, expected_state, claim_retained in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp_dir:
                artifact_root = Path(tmp_dir) / "artifacts"
                run_id = f"fresh-{label}"
                args = [
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
                observed_states: list[str] = []

                def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
                    observed_states.append(
                        load_yaml(artifact_root / "runs" / run_id / "status.yaml")["state"]
                    )
                    if isinstance(result, BaseException):
                        raise result
                    if result.returncode == 0:
                        _write_mock_tabular_scientific_outputs(kwargs)
                    return result

                with mock.patch("research_platform.core.cli.shutil.which", return_value="/bin/bash"):
                    with mock.patch("research_platform.core.cli.subprocess.run", side_effect=fake_run) as process_mock:
                        exit_code, _ = self._run_cli(args, artifact_root)

                self.assertEqual(exit_code, expected_exit)
                self.assertEqual(observed_states, ["running"])
                process_mock.assert_called_once()
                self.assertEqual(
                    load_yaml(artifact_root / "runs" / run_id / "status.yaml")["state"],
                    expected_state,
                )
                self.assertEqual(
                    (artifact_root / "runs" / f".{run_id}.claim").exists(),
                    claim_retained,
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_id = "fresh-missing-shell"
            args = [
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
            with mock.patch("research_platform.core.cli.shutil.which", return_value=None):
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("a missing executable must fail before launch"),
                ) as process_mock:
                    exit_code, _ = self._run_cli(args, artifact_root)
            self.assertEqual(exit_code, 1)
            process_mock.assert_not_called()
            self.assertEqual(load_yaml(artifact_root / "runs" / run_id / "status.yaml")["state"], "failed")
            self.assertFalse((artifact_root / "runs" / f".{run_id}.claim").exists())

    def test_tabular_terminal_status_write_failure_preserves_recovery_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_id = "terminal-status-write-failure"
            args = [
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
            original_write_status = core_cli._write_tabular_terminal_status

            def fail_terminal_status(**kwargs: object) -> None:
                if kwargs["state"] == "running":
                    original_write_status(**kwargs)  # type: ignore[arg-type]
                    return
                raise OSError("synthetic terminal status write failure")

            def fake_run(
                command: list[str],
                **kwargs: object,
            ) -> subprocess.CompletedProcess[list[str]]:
                _write_mock_tabular_scientific_outputs(kwargs)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("research_platform.core.cli.shutil.which", return_value="/bin/bash"):
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=fake_run,
                ) as process_mock:
                    with mock.patch(
                        "research_platform.core.cli._write_tabular_terminal_status",
                        side_effect=fail_terminal_status,
                    ):
                        exit_code, output = self._run_cli(args, artifact_root)

            self.assertEqual(exit_code, 1)
            self.assertIn("success-status persistence failed", output)
            process_mock.assert_called_once()
            run_root = artifact_root / "runs" / run_id
            self.assertEqual(load_yaml(run_root / "status.yaml")["state"], "running")
            self.assertTrue((artifact_root / "runs" / f".{run_id}.claim").is_dir())
            self.assertTrue((run_root / "outputs" / "transaction-manifest.json").is_file())

    def test_tabular_unsafe_run_ids_are_rejected_before_filesystem_writes(self) -> None:
        unsafe_ids = (
            "",
            "   ",
            ".",
            "..",
            "/absolute",
            "nested/path",
            "nested\\path",
            "../escape",
            "escape/../run",
            "bad\x00id",
            "bad\nid",
            ".hidden",
        )
        for run_id in unsafe_ids:
            with self.subTest(run_id=repr(run_id)), tempfile.TemporaryDirectory() as tmp_dir:
                artifact_root = Path(tmp_dir) / "artifacts"
                code, output = self._run_cli_failure_for_workspace(
                    [
                        "run",
                        "plan",
                        "preprocess",
                        "tabular",
                        "--project",
                        "project-pilot-tabular",
                        "--run-id",
                        run_id,
                    ],
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )
                self.assertEqual(code, 1)
                self.assertIn("Run id", output)
                self.assertNotIn("Traceback", output)
                self.assertFalse((artifact_root / "runs").exists())
                self.assertFalse((Path(tmp_dir) / "escape").exists())

    def test_tabular_root_and_claim_admission_fail_closed_without_mutation(self) -> None:
        for entry_kind in ("file", "symlink", "directory"):
            with self.subTest(entry_kind=entry_kind), tempfile.TemporaryDirectory() as tmp_dir:
                artifact_root = Path(tmp_dir) / "artifacts"
                runs_root = artifact_root / "runs"
                runs_root.mkdir(parents=True)
                run_id = f"foreign-{entry_kind}"
                run_root = runs_root / run_id
                if entry_kind == "file":
                    run_root.write_bytes(b"foreign run sentinel\n")
                elif entry_kind == "symlink":
                    foreign = Path(tmp_dir) / "foreign-target"
                    foreign.mkdir()
                    (foreign / "sentinel").write_bytes(b"do not touch\n")
                    run_root.symlink_to(foreign, target_is_directory=True)
                else:
                    run_root.mkdir()
                    (run_root / "sentinel").write_bytes(b"foreign run sentinel\n")
                before = _filesystem_snapshot(run_root)
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("unsafe roots must fail before launch"),
                ) as process_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [
                            "run",
                            "plan",
                            "preprocess",
                            "tabular",
                            "--project",
                            "project-pilot-tabular",
                            "--run-id",
                            run_id,
                        ],
                        workspace_root=WORKSPACE_ROOT,
                        artifact_root=artifact_root,
                    )
                self.assertEqual(code, 1)
                self.assertIn("choose a new run id", output)
                self.assertEqual(_filesystem_snapshot(run_root), before)
                process_mock.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            runs_root = artifact_root / "runs"
            runs_root.mkdir(parents=True)
            run_id = "foreign-claim"
            claim = runs_root / f".{run_id}.claim"
            claim.mkdir()
            (claim / "owner-sentinel").write_bytes(b"foreign owner\n")
            before = _filesystem_snapshot(claim)
            code, output = self._run_cli_failure_for_workspace(
                [
                    "run",
                    "plan",
                    "preprocess",
                    "tabular",
                    "--project",
                    "project-pilot-tabular",
                    "--run-id",
                    run_id,
                ],
                workspace_root=WORKSPACE_ROOT,
                artifact_root=artifact_root,
            )
            self.assertEqual(code, 1)
            self.assertIn("execution claim", output)
            self.assertEqual(_filesystem_snapshot(claim), before)
            self.assertFalse((runs_root / run_id).exists())

    def test_tabular_state_and_control_file_rejections_preserve_existing_tree(self) -> None:
        rejected_states = (
            "running",
            "failed",
            "succeeded",
            "completed",
            "staged",
            "submitted",
            "stage-failed",
            "submit-failed",
            "cancel-requested",
            "cancelled",
            "unknown-state",
        )
        for state in rejected_states:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp_dir:
                artifact_root = Path(tmp_dir) / "artifacts"
                run_id = f"state-{state}"
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
                self._run_cli(plan_args, artifact_root)
                run_root = artifact_root / "runs" / run_id
                status = load_yaml(run_root / "status.yaml")
                status["state"] = state
                write_yaml(run_root / "status.yaml", status)
                before = _filesystem_snapshot(run_root)
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("a rejected state must not execute"),
                ) as process_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [
                            "run",
                            "local",
                            "preprocess",
                            "tabular",
                            "--project",
                            "project-pilot-tabular",
                            "--run-id",
                            run_id,
                            "--execute",
                        ],
                        workspace_root=WORKSPACE_ROOT,
                        artifact_root=artifact_root,
                    )
                self.assertEqual(code, 1)
                self.assertIn("choose a new run id", output)
                self.assertEqual(_filesystem_snapshot(run_root), before)
                process_mock.assert_not_called()

        def change_control_run_id(root: Path, name: str) -> None:
            path = root / name
            document = load_yaml(path)
            document["run_id"] = "different-run"
            write_yaml(path, document)

        control_mutations = (
            ("missing-manifest", lambda root: (root / "run-manifest.yaml").unlink()),
            ("missing-status", lambda root: (root / "status.yaml").unlink()),
            ("malformed-manifest", lambda root: (root / "run-manifest.yaml").write_text("[\n", encoding="utf-8")),
            ("malformed-status", lambda root: (root / "status.yaml").write_text("[\n", encoding="utf-8")),
            ("mismatched-manifest", lambda root: change_control_run_id(root, "run-manifest.yaml")),
            ("mismatched-status", lambda root: change_control_run_id(root, "status.yaml")),
        )
        for label, mutate in control_mutations:
            with self.subTest(control=label), tempfile.TemporaryDirectory() as tmp_dir:
                artifact_root = Path(tmp_dir) / "artifacts"
                run_id = f"controls-{label}"
                self._run_cli(
                    [
                        "run",
                        "plan",
                        "preprocess",
                        "tabular",
                        "--project",
                        "project-pilot-tabular",
                        "--run-id",
                        run_id,
                    ],
                    artifact_root,
                )
                run_root = artifact_root / "runs" / run_id
                mutate(run_root)
                before = _filesystem_snapshot(run_root)
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("malformed controls must fail before launch"),
                ) as process_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [
                            "run",
                            "local",
                            "preprocess",
                            "tabular",
                            "--project",
                            "project-pilot-tabular",
                            "--run-id",
                            run_id,
                            "--execute",
                        ],
                        workspace_root=WORKSPACE_ROOT,
                        artifact_root=artifact_root,
                    )
                self.assertEqual(code, 1)
                self.assertIn("choose a new run id", output)
                self.assertEqual(_filesystem_snapshot(run_root), before)
                process_mock.assert_not_called()

    def test_tabular_unexpected_payload_and_modified_script_are_rejected_without_mutation(self) -> None:
        def add_output_residue(root: Path) -> None:
            (root / "outputs").mkdir()
            (root / "outputs" / "partial.tsv").write_bytes(b"partial\n")

        mutations = (
            ("unexpected", lambda root: (root / "unexpected.txt").write_bytes(b"foreign payload\n")),
            ("output-residue", add_output_residue),
            ("missing-script", lambda root: (root / "execute.sh").unlink()),
            ("script", lambda root: (root / "execute.sh").write_bytes(b"#!/bin/sh\nexit 99\n")),
        )
        for label, mutate in mutations:
            with self.subTest(mutation=label), tempfile.TemporaryDirectory() as tmp_dir:
                artifact_root = Path(tmp_dir) / "artifacts"
                run_id = f"modified-{label}"
                self._run_cli(
                    [
                        "run",
                        "plan",
                        "preprocess",
                        "tabular",
                        "--project",
                        "project-pilot-tabular",
                        "--run-id",
                        run_id,
                    ],
                    artifact_root,
                )
                run_root = artifact_root / "runs" / run_id
                mutate(run_root)
                before = _filesystem_snapshot(run_root)
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("mutated plans must fail before launch"),
                ) as process_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [
                            "run",
                            "local",
                            "preprocess",
                            "tabular",
                            "--project",
                            "project-pilot-tabular",
                            "--run-id",
                            run_id,
                            "--execute",
                        ],
                        workspace_root=WORKSPACE_ROOT,
                        artifact_root=artifact_root,
                    )
                self.assertEqual(code, 1)
                self.assertIn("choose a new run id", output)
                self.assertEqual(_filesystem_snapshot(run_root), before)
                process_mock.assert_not_called()

    def test_tabular_reviewed_identity_rejects_configuration_and_workflow_drift(self) -> None:
        def configure_workspace(workspace_root: Path) -> tuple[Path, Path, Path, Path]:
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            table_path = (
                workspace_root
                / "datasets"
                / "ds-tabular"
                / "derivatives"
                / "features"
                / "test-tabular"
                / "toy_features.tsv"
            )
            table_path.write_text(
                "record_id\tfeature_a\tfeature_b\tbinary_target\talternate_target\n"
                "record-001\t1\t2\t0\t1\n",
                encoding="utf-8",
            )
            return (
                project_root / "config" / "models.yaml",
                project_root / "config" / "preprocessing.yaml",
                project_root / "manifests" / "batches" / "default.tsv",
                table_path,
            )

        def change_predictors(paths: tuple[Path, Path, Path, Path]) -> None:
            document = load_yaml(paths[0])
            document["models"]["default"]["feature_columns"] = ["feature_b"]
            write_yaml(paths[0], document)

        def change_split(paths: tuple[Path, Path, Path, Path]) -> None:
            document = load_yaml(paths[1])
            document["preprocessing"]["split_seed"] = 99
            write_yaml(paths[1], document)

        def change_model(paths: tuple[Path, Path, Path, Path]) -> None:
            document = load_yaml(paths[0])
            document["models"]["default"]["learning_rate"] = 0.125
            write_yaml(paths[0], document)

        def change_batch(paths: tuple[Path, Path, Path, Path]) -> None:
            paths[2].write_text(
                "feature_table\ttarget_column\n"
                "test-tabular/toy_features.tsv\talternate_target\n",
                encoding="utf-8",
            )

        def change_table(paths: tuple[Path, Path, Path, Path]) -> None:
            paths[3].write_text(
                "record_id\tfeature_a\tfeature_b\tbinary_target\talternate_target\n"
                "record-001\t9\t2\t0\t1\n",
                encoding="utf-8",
            )

        drift_cases = (
            ("predictors", "preprocess", "preprocess", change_predictors),
            ("split", "preprocess", "preprocess", change_split),
            ("model", "train", "train", change_model),
            ("batch", "preprocess", "preprocess", change_batch),
            ("input-table", "preprocess", "preprocess", change_table),
            ("workflow", "preprocess", "train", lambda _: None),
        )
        for label, planned_action, requested_action, mutate in drift_cases:
            with self.subTest(drift=label), tempfile.TemporaryDirectory() as tmp_dir:
                workspace_root = Path(tmp_dir) / "workspace"
                artifact_root = workspace_root / "artifacts"
                paths = configure_workspace(workspace_root)
                run_id = f"drift-{label}"
                plan_args = [
                    "run",
                    "plan",
                    planned_action,
                    "tabular" if planned_action == "preprocess" else "model",
                    "--project",
                    "test-tabular",
                    "--run-id",
                    run_id,
                ]
                self._run_cli_for_workspace(
                    plan_args,
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
                run_root = artifact_root / "runs" / run_id
                mutate(paths)
                before = _filesystem_snapshot(run_root)
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("identity drift must fail before launch"),
                ) as process_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [
                            "run",
                            "local",
                            requested_action,
                            "tabular" if requested_action == "preprocess" else "model",
                            "--project",
                            "test-tabular",
                            "--run-id",
                            run_id,
                            "--execute",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
                self.assertEqual(code, 1)
                self.assertIn("choose a new run id", output)
                self.assertEqual(_filesystem_snapshot(run_root), before)
                process_mock.assert_not_called()

    def test_tabular_analysis_and_evaluation_input_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            analysis_dir = project_root / "config" / "analysis"
            analysis_dir.mkdir(parents=True)
            analysis_path = analysis_dir / "feature_correlation.yaml"
            write_yaml(
                analysis_path,
                {
                    "analysis": {
                        "kind": "correlation",
                        "input_table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
                        "method": "pearson",
                        "x": "feature_a",
                        "y": "binary_target",
                    }
                },
            )
            run_id = "analysis-drift"
            self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "analysis",
                    "tabular",
                    "--project",
                    "test-tabular",
                    "--analysis",
                    "feature_correlation",
                    "--run-id",
                    run_id,
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            analysis_document = load_yaml(analysis_path)
            analysis_document["analysis"]["method"] = "spearman"
            write_yaml(analysis_path, analysis_document)
            run_root = artifact_root / "runs" / run_id
            before = _filesystem_snapshot(run_root)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=AssertionError("analysis drift must fail before launch"),
            ) as process_mock:
                code, output = self._run_cli_failure_for_workspace(
                    [
                        "run",
                        "local",
                        "analysis",
                        "tabular",
                        "--project",
                        "test-tabular",
                        "--analysis",
                        "feature_correlation",
                        "--run-id",
                        run_id,
                        "--execute",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), before)
            process_mock.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_slurm_workspace(workspace_root)
            for source_run in ("source-a", "source-b"):
                self._run_cli_for_workspace(
                    [
                        "run",
                        "plan",
                        "train",
                        "model",
                        "--project",
                        "test-tabular",
                        "--run-id",
                        source_run,
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
                source_execute_exit, _ = self._execute_mocked_tabular_success(
                    [
                        "run",
                        "local",
                        "train",
                        "model",
                        "--project",
                        "test-tabular",
                        "--run-id",
                        source_run,
                        "--execute",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
                self.assertEqual(source_execute_exit, 0)
            run_id = "evaluation-drift"
            self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "evaluate",
                    "model",
                    "--project",
                    "test-tabular",
                    "--input-run",
                    "source-a",
                    "--run-id",
                    run_id,
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            run_root = artifact_root / "runs" / run_id
            before = _filesystem_snapshot(run_root)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=AssertionError("input-run drift must fail before launch"),
            ) as process_mock:
                code, output = self._run_cli_failure_for_workspace(
                    [
                        "run",
                        "local",
                        "evaluate",
                        "model",
                        "--project",
                        "test-tabular",
                        "--input-run",
                        "source-b",
                        "--run-id",
                        run_id,
                        "--execute",
                    ],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), before)
            process_mock.assert_not_called()

    def test_tabular_train_evaluate_and_analysis_reviewed_plans_execute_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            analysis_dir = project_root / "config" / "analysis"
            analysis_dir.mkdir(parents=True)
            write_yaml(
                analysis_dir / "feature_correlation.yaml",
                {
                    "analysis": {
                        "kind": "correlation",
                        "input_table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
                        "method": "pearson",
                        "x": "feature_a",
                        "y": "binary_target",
                    }
                },
            )
            self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "train",
                    "model",
                    "--project",
                    "test-tabular",
                    "--run-id",
                    "source-train",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            source_execute_exit, _ = self._execute_mocked_tabular_success(
                [
                    "run",
                    "local",
                    "train",
                    "model",
                    "--project",
                    "test-tabular",
                    "--run-id",
                    "source-train",
                    "--execute",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            self.assertEqual(source_execute_exit, 0)
            cases = (
                (
                    "train-reviewed",
                    ["train", "model"],
                    [],
                ),
                (
                    "evaluate-reviewed",
                    ["evaluate", "model"],
                    ["--input-run", "source-train"],
                ),
                (
                    "analysis-reviewed",
                    ["analysis", "tabular"],
                    ["--analysis", "feature_correlation"],
                ),
            )
            for run_id, action_target, extra_args in cases:
                with self.subTest(run_id=run_id):
                    common = [
                        *action_target,
                        "--project",
                        "test-tabular",
                        *extra_args,
                        "--run-id",
                        run_id,
                    ]
                    plan_exit, _ = self._run_cli_for_workspace(
                        ["run", "plan", *common],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
                    run_root = artifact_root / "runs" / run_id
                    original_manifest = load_yaml(run_root / "run-manifest.yaml")

                    def fake_run(
                        command: list[str],
                        **kwargs: object,
                    ) -> subprocess.CompletedProcess[list[str]]:
                        _write_mock_tabular_scientific_outputs(kwargs)
                        return subprocess.CompletedProcess(command, 0)

                    with mock.patch("research_platform.core.cli.shutil.which", return_value="/bin/bash"):
                        with mock.patch(
                            "research_platform.core.cli.subprocess.run",
                            side_effect=fake_run,
                        ) as process_mock:
                            execute_exit, _ = self._run_cli_for_workspace(
                                ["run", "local", *common, "--execute"],
                                workspace_root=workspace_root,
                                artifact_root=artifact_root,
                            )
                    self.assertEqual(plan_exit, 0)
                    self.assertEqual(execute_exit, 0)
                    process_mock.assert_called_once()
                    self.assertEqual(load_yaml(run_root / "status.yaml")["state"], "succeeded")
                    completed_manifest = load_yaml(run_root / "run-manifest.yaml")
                    self.assertEqual(completed_manifest["plan_identity"], original_manifest["plan_identity"])
                    self.assertEqual(completed_manifest["created_at"], original_manifest["created_at"])

    def test_tabular_execution_claim_allows_exactly_one_concurrent_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_id = "concurrent-reviewed"
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
            self._run_cli(plan_args, artifact_root)
            run_root = artifact_root / "runs" / run_id
            entered = threading.Event()
            release = threading.Event()
            outcomes: list[object] = []
            outcome_lock = threading.Lock()

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
                self.assertEqual(load_yaml(run_root / "status.yaml")["state"], "running")
                entered.set()
                self.assertTrue(release.wait(timeout=10))
                _write_mock_tabular_scientific_outputs(kwargs)
                return subprocess.CompletedProcess(command, 0)

            def worker() -> None:
                try:
                    outcome: object = main(execute_args)
                except SystemExit as exc:
                    outcome = exc.code
                with outcome_lock:
                    outcomes.append(outcome)

            env = {
                "RESEARCH_PLATFORM_ROOT": str(WORKSPACE_ROOT),
                "ARTIFACTS_ROOT": str(artifact_root),
                "RP_HPC_HOST": "example-hpc",
                "RP_REMOTE_WORKSPACE_ROOT": "remote/workspace",
                "RP_REMOTE_ARTIFACTS_ROOT": "remote/workspace/artifacts",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("research_platform.core.cli.shutil.which", return_value="/bin/bash"):
                    with mock.patch("research_platform.core.cli.subprocess.run", side_effect=fake_run) as process_mock:
                        with mock.patch("builtins.print"):
                            owner = threading.Thread(target=worker)
                            contender = threading.Thread(target=worker)
                            owner.start()
                            self.assertTrue(entered.wait(timeout=10))
                            contender.start()
                            contender.join(timeout=10)
                            self.assertFalse(contender.is_alive())
                            release.set()
                            owner.join(timeout=10)
                            self.assertFalse(owner.is_alive())

            self.assertEqual(process_mock.call_count, 1)
            self.assertEqual(sum(outcome == 0 for outcome in outcomes), 1)
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(any(isinstance(outcome, str) and "execution claim" in outcome for outcome in outcomes))
            self.assertEqual(load_yaml(run_root / "status.yaml")["state"], "succeeded")
            self.assertFalse((artifact_root / "runs" / f".{run_id}.claim").exists())

    def test_tabular_project_drift_and_stale_claim_are_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            run_id = "project-drift"
            common = [
                "preprocess",
                "tabular",
                "--project",
                "test-tabular",
                "--run-id",
                run_id,
            ]
            self._run_cli_for_workspace(
                ["run", "plan", *common],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            write_yaml(project_root / "project.yaml", {"name": "changed-project", "version": "0.1.0"})
            run_root = artifact_root / "runs" / run_id
            before = _filesystem_snapshot(run_root)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=AssertionError("project drift must fail before launch"),
            ) as process_mock:
                code, output = self._run_cli_failure_for_workspace(
                    ["run", "local", *common, "--execute"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), before)
            process_mock.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            run_id = "stale-claim"
            common = [
                "preprocess",
                "tabular",
                "--project",
                "project-pilot-tabular",
                "--run-id",
                run_id,
            ]
            self._run_cli(["run", "plan", *common], artifact_root)
            run_root = artifact_root / "runs" / run_id
            claim = artifact_root / "runs" / f".{run_id}.claim"
            claim.mkdir()
            (claim / "recovery-evidence").write_bytes(b"foreign stale claim\n")
            root_before = _filesystem_snapshot(run_root)
            claim_before = _filesystem_snapshot(claim)
            with mock.patch(
                "research_platform.core.cli.subprocess.run",
                side_effect=AssertionError("a stale claim must fail before launch"),
            ) as process_mock:
                code, output = self._run_cli_failure_for_workspace(
                    ["run", "local", *common, "--execute"],
                    workspace_root=WORKSPACE_ROOT,
                    artifact_root=artifact_root,
                )
            self.assertEqual(code, 1)
            self.assertIn("execution claim", output)
            self.assertEqual(_filesystem_snapshot(run_root), root_before)
            self.assertEqual(_filesystem_snapshot(claim), claim_before)
            process_mock.assert_not_called()

    def test_tabular_and_bids_workflows_cannot_overwrite_each_other(self) -> None:
        cases = (
            (
                ["run", "plan", "preprocess", "tabular", "--project", "project-pilot-tabular"],
                ["run", "plan", "preprocess", "bids", "--project", "project-pilot-bids"],
            ),
            (
                ["run", "plan", "preprocess", "bids", "--project", "project-pilot-bids"],
                ["run", "plan", "preprocess", "tabular", "--project", "project-pilot-tabular"],
            ),
        )
        for index, (owner_args, contender_args) in enumerate(cases):
            with self.subTest(direction=index), tempfile.TemporaryDirectory() as tmp_dir:
                artifact_root = Path(tmp_dir) / "artifacts"
                run_id = f"shared-owner-{index}"
                owner_exit, _ = self._run_cli([*owner_args, "--run-id", run_id], artifact_root)
                run_root = artifact_root / "runs" / run_id
                before = _filesystem_snapshot(run_root)
                with mock.patch(
                    "research_platform.core.cli.subprocess.run",
                    side_effect=AssertionError("cross-owner reuse must not launch"),
                ) as process_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [*contender_args, "--run-id", run_id],
                        workspace_root=WORKSPACE_ROOT,
                        artifact_root=artifact_root,
                    )
                self.assertEqual(owner_exit, 0)
                self.assertEqual(code, 1)
                self.assertIn("choose a new run id", output)
                self.assertEqual(_filesystem_snapshot(run_root), before)
                process_mock.assert_not_called()

    def test_tabular_slurm_plans_are_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            self._write_tabular_slurm_workspace(workspace_root)
            args = [
                "run",
                "slurm",
                "train",
                "model",
                "--project",
                "test-tabular",
                "--run-id",
                "one-shot-slurm",
            ]
            plan_exit, _ = self._run_cli_for_workspace(
                args,
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            run_root = artifact_root / "runs" / "one-shot-slurm"
            before = _filesystem_snapshot(run_root)
            code, output = self._run_cli_failure_for_workspace(
                args,
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            self.assertEqual(plan_exit, 0)
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), before)

    def test_remote_tabular_analysis_plan_transitions_once_to_mocked_submit(self) -> None:
        stage_result = {"ok": True, "returncode": 0, "commands": [{"command": ["rsync"]}]}
        submit_result = {
            "ok": True,
            "returncode": 0,
            "command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"],
            "stdout": "Submitted batch job 24680\n",
            "stderr": "",
            "job_id": "24680",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            analysis_dir = project_root / "config" / "analysis"
            analysis_dir.mkdir(parents=True)
            write_yaml(
                analysis_dir / "feature_correlation.yaml",
                {
                    "analysis": {
                        "kind": "correlation",
                        "input_table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
                        "method": "pearson",
                        "x": "feature_a",
                        "y": "binary_target",
                    }
                },
            )
            args = [
                "run",
                "submit",
                "analysis",
                "tabular",
                "--project",
                "test-tabular",
                "--analysis",
                "feature_correlation",
                "--run-id",
                "remote-reviewed",
            ]
            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    plan_exit, _ = self._run_cli_for_workspace(
                        args,
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
            stage_mock.assert_not_called()
            submit_mock.assert_not_called()
            run_root = artifact_root / "runs" / "remote-reviewed"
            original_manifest = load_yaml(run_root / "run-manifest.yaml")

            planned_snapshot = _filesystem_snapshot(run_root)
            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        args,
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), planned_snapshot)
            stage_mock.assert_not_called()
            submit_mock.assert_not_called()

            with mock.patch(
                "research_platform.core.cli.execute_stage_plan", return_value=stage_result
            ) as stage_mock:
                with mock.patch(
                    "research_platform.core.cli.execute_submit_plan", return_value=submit_result
                ) as submit_mock:
                    execute_exit, _ = self._run_cli_for_workspace(
                        [*args, "--execute"],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

            self.assertEqual(plan_exit, 0)
            self.assertEqual(execute_exit, 0)
            stage_mock.assert_called_once()
            submit_mock.assert_called_once()
            completed_manifest = load_yaml(run_root / "run-manifest.yaml")
            self.assertEqual(completed_manifest["created_at"], original_manifest["created_at"])
            self.assertEqual(completed_manifest["plan_identity"], original_manifest["plan_identity"])
            status = load_yaml(run_root / "status.yaml")
            self.assertEqual(status["state"], "submitted")
            self.assertEqual(str(status["job_id"]), "24680")
            self.assertFalse((artifact_root / "runs" / ".remote-reviewed.claim").exists())

            before = _filesystem_snapshot(run_root)
            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [*args, "--execute"],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), before)
            stage_mock.assert_not_called()
            submit_mock.assert_not_called()

    def test_remote_tabular_analysis_rejects_modified_reviewed_stage_material(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            analysis_dir = project_root / "config" / "analysis"
            analysis_dir.mkdir(parents=True)
            write_yaml(
                analysis_dir / "feature_correlation.yaml",
                {
                    "analysis": {
                        "kind": "correlation",
                        "input_table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
                        "method": "pearson",
                        "x": "feature_a",
                        "y": "binary_target",
                    }
                },
            )
            args = [
                "run",
                "submit",
                "analysis",
                "tabular",
                "--project",
                "test-tabular",
                "--analysis",
                "feature_correlation",
                "--run-id",
                "remote-stage-drift",
            ]
            self._run_cli_for_workspace(
                args,
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            run_root = artifact_root / "runs" / "remote-stage-drift"
            staged_file = next(
                path for path in sorted((run_root / "hpc" / "stage").rglob("*")) if path.is_file()
            )
            staged_file.write_bytes(staged_file.read_bytes() + b"\n# unexpected drift\n")
            before = _filesystem_snapshot(run_root)
            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [*args, "--execute"],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
            self.assertEqual(code, 1)
            self.assertIn("choose a new run id", output)
            self.assertEqual(_filesystem_snapshot(run_root), before)
            stage_mock.assert_not_called()
            submit_mock.assert_not_called()

    def test_remote_tabular_analysis_requires_reviewed_submission_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            artifact_root = workspace_root / "artifacts"
            project_root = self._write_tabular_slurm_workspace(workspace_root)
            analysis_dir = project_root / "config" / "analysis"
            analysis_dir.mkdir(parents=True)
            write_yaml(
                analysis_dir / "feature_correlation.yaml",
                {
                    "analysis": {
                        "kind": "correlation",
                        "input_table": "datasets/ds-tabular/derivatives/features/test-tabular/toy_features.tsv",
                        "method": "pearson",
                        "x": "feature_a",
                        "y": "binary_target",
                    }
                },
            )
            args = [
                "run",
                "submit",
                "analysis",
                "tabular",
                "--project",
                "test-tabular",
                "--analysis",
                "feature_correlation",
                "--run-id",
                "remote-missing-submission-identity",
            ]
            self._run_cli_for_workspace(
                args,
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            run_root = artifact_root / "runs" / "remote-missing-submission-identity"
            manifest = load_yaml(run_root / "run-manifest.yaml")
            manifest.pop("submission_identity")
            write_yaml(run_root / "run-manifest.yaml", manifest)
            before = _filesystem_snapshot(run_root)

            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    code, output = self._run_cli_failure_for_workspace(
                        [*args, "--execute"],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )

            self.assertEqual(code, 1)
            self.assertIn("reviewed remote submission identity is incomplete", output)
            self.assertEqual(_filesystem_snapshot(run_root), before)
            stage_mock.assert_not_called()
            submit_mock.assert_not_called()

    def test_bids_submit_preserves_existing_plan_to_execute_transition(self) -> None:
        stage_result = {"ok": True, "returncode": 0, "commands": [{"command": ["rsync"]}]}
        submit_result = {
            "ok": True,
            "returncode": 0,
            "command": ["ssh", "example-hpc", "cd remote/run && sbatch submit.sbatch"],
            "stdout": "Submitted batch job 13579\n",
            "stderr": "",
            "job_id": "13579",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            args = [
                "run",
                "submit",
                "preprocess",
                "bids",
                "--project",
                "project-pilot-bids",
                "--run-id",
                "bids-reviewed-submit",
            ]
            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    plan_exit, _ = self._run_cli(args, artifact_root)
            stage_mock.assert_not_called()
            submit_mock.assert_not_called()
            with mock.patch(
                "research_platform.core.cli.execute_stage_plan", return_value=stage_result
            ) as stage_mock:
                with mock.patch(
                    "research_platform.core.cli.execute_submit_plan", return_value=submit_result
                ) as submit_mock:
                    execute_exit, _ = self._run_cli([*args, "--execute"], artifact_root)
            self.assertEqual(plan_exit, 0)
            self.assertEqual(execute_exit, 0)
            stage_mock.assert_called_once()
            submit_mock.assert_called_once()
            status = load_yaml(artifact_root / "runs" / "bids-reviewed-submit" / "status.yaml")
            self.assertEqual(status["state"], "submitted")
            self.assertEqual(str(status["job_id"]), "13579")


if __name__ == "__main__":
    unittest.main()
