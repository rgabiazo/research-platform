"""fMRIPost-AROMA BIDS tool adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .command import (
    DEFAULT_APPTAINER_IMAGE_ROOT,
    DEFAULT_APPTAINER_PULL_MODE,
    DEFAULT_HPC_BACKEND,
    DEFAULT_IMAGE_REPOSITORY,
    DEFAULT_IMAGE_TAG,
    DEFAULT_LOCAL_BACKEND,
    build_batch_runtime_plan,
    write_command_script,
    write_runtime_plan,
)
from .selection import discover_batch_rows, expected_remote_input_files

_ALLOWED_TOOL_OPTIONS = frozenset(
    {
        "denoising_method",
        "dummy_scans",
        "low_mem",
        "melodic_dimensionality",
        "melodic_seed",
        "runtime_grouping",
    }
)
_ALLOWED_DENOISING_METHODS = frozenset({"aggr", "nonaggr", "orthaggr"})
_ALLOWED_RUNTIME_GROUPINGS = frozenset({"compatible", "row"})


class FmripostAromaAdapter:
    def tool_name(self) -> str:
        return "fmripost_aroma"

    def requires_input_derivative(self) -> bool:
        return True

    def supported_input_derivatives(self) -> tuple[str, ...]:
        return ("deepprep-bold", "fmriprep")

    def validate_project(
        self,
        *,
        bundle: dict[str, Any],
        pipeline_defaults: dict[str, Any],
        workspace_root: str,
    ) -> list[str]:
        preprocessing = bundle["preprocessing"]["preprocessing"]
        dataset = bundle["dataset"]["dataset"]
        errors: list[str] = []

        if preprocessing.get("tool") != self.tool_name():
            errors.append(f"This adapter supports only preprocessing.tool={self.tool_name()}.")

        preprocessing_input = _optional_text(preprocessing.get("input_derivative"))
        dataset_input = _optional_text(dataset.get("input_derivative"))
        if preprocessing_input is None:
            errors.append("config/preprocessing.yaml must define preprocessing.input_derivative.")
        elif preprocessing_input not in self.supported_input_derivatives():
            allowed = ", ".join(self.supported_input_derivatives())
            errors.append(f"preprocessing.input_derivative must be one of: {allowed}.")

        if dataset_input is None:
            errors.append("config/dataset.yaml must define dataset.input_derivative.")
        elif preprocessing_input and dataset_input != preprocessing_input:
            errors.append("dataset.input_derivative and preprocessing.input_derivative must match.")

        tool_options = preprocessing.get("tool_options")
        if tool_options is not None and not isinstance(tool_options, dict):
            errors.append("config/preprocessing.yaml preprocessing.tool_options must be a mapping.")
        elif isinstance(tool_options, dict):
            unknown = sorted(set(tool_options) - _ALLOWED_TOOL_OPTIONS)
            if unknown:
                errors.append(
                    f"Unsupported preprocessing.tool_options for {self.tool_name()}: {', '.join(str(name) for name in unknown)}."
                )
            denoising_method = _optional_text(tool_options.get("denoising_method"))
            if denoising_method and denoising_method not in _ALLOWED_DENOISING_METHODS:
                allowed_methods = ", ".join(sorted(_ALLOWED_DENOISING_METHODS))
                errors.append(
                    f"preprocessing.tool_options.denoising_method must be one of: {allowed_methods}."
                )
            runtime_grouping = _optional_text(tool_options.get("runtime_grouping"))
            if runtime_grouping and runtime_grouping not in _ALLOWED_RUNTIME_GROUPINGS:
                allowed_groupings = ", ".join(sorted(_ALLOWED_RUNTIME_GROUPINGS))
                errors.append(
                    f"preprocessing.tool_options.runtime_grouping must be one of: {allowed_groupings}."
                )
            for name in ("melodic_dimensionality", "melodic_seed", "dummy_scans"):
                value = _optional_text(tool_options.get(name))
                if value is None:
                    continue
                try:
                    int(value)
                except ValueError:
                    errors.append(f"preprocessing.tool_options.{name} must be an integer-like value.")
            low_mem = tool_options.get("low_mem")
            if low_mem not in (None, "", True, False, "true", "false", "True", "False", "1", "0"):
                errors.append("preprocessing.tool_options.low_mem must be a boolean-like value.")

        try:
            runtime = self.runtime_metadata(
                pipeline_defaults=pipeline_defaults,
                output_dir=str(Path(workspace_root) / "artifacts" / "runs" / "validation"),
            )
        except ValueError as exc:
            errors.append(str(exc))
        else:
            required = (
                "workflow_target",
                "rule_name",
                "runtime_plan_filename",
                "command_script_filename",
                "completion_marker_filename",
                "output_data_dirname",
            )
            for name in required:
                if not runtime.get(name):
                    errors.append(f"Pipeline defaults are missing adapter runtime metadata: {name}.")

        return errors

    def discover_batch_rows(
        self,
        *,
        derivative_root: str,
        selectors: dict[str, str | None],
    ) -> list[dict[str, str]]:
        return discover_batch_rows(derivative_root, selectors=selectors)

    def expected_remote_input_files(
        self,
        *,
        derivative_root: str,
        remote_derivative_root: str,
        row: dict[str, str],
    ) -> list[str]:
        return expected_remote_input_files(
            derivative_root,
            remote_derivative_root=remote_derivative_root,
            row=row,
        )

    def expected_remote_auxiliary_files(
        self,
        *,
        context: dict[str, Any],
    ) -> list[str]:
        _ = context
        return []

    def runtime_metadata(
        self,
        *,
        pipeline_defaults: dict[str, Any],
        output_dir: str,
    ) -> dict[str, str]:
        workflow = pipeline_defaults.get("workflow", {})
        planner_outputs = pipeline_defaults.get("planner", {}).get("outputs", {})
        workflow_target = _required_text(workflow.get("default_target"), "workflow.default_target")
        rule_name = _optional_text(workflow.get("rule_name")) or workflow_target
        execution_rule_name = _optional_text(workflow.get("execution_rule_name")) or rule_name
        runtime_plan_filename = _required_text(
            planner_outputs.get("runtime_plan_filename"),
            "planner.outputs.runtime_plan_filename",
        )
        command_script_filename = _required_text(
            planner_outputs.get("command_script_filename"),
            "planner.outputs.command_script_filename",
        )
        completion_marker_filename = _required_text(
            planner_outputs.get("completion_marker_filename"),
            "planner.outputs.completion_marker_filename",
        )
        output_data_dirname = _optional_text(planner_outputs.get("output_data_dirname"))
        if output_data_dirname in (None, "preprocess_bids"):
            output_data_dirname = "fmripost_aroma"
        return {
            "workflow_target": workflow_target,
            "rule_name": rule_name,
            "execution_rule_name": execution_rule_name,
            "runtime_plan_filename": runtime_plan_filename,
            "command_script_filename": command_script_filename,
            "completion_marker_filename": completion_marker_filename,
            "output_data_dirname": output_data_dirname,
            "output_dir": str(Path(output_dir)),
        }

    def build_runtime_plan(
        self,
        *,
        manifest: dict[str, Any],
        workspace_root: str,
        plan_path: str,
        command_script_path: str,
    ) -> dict[str, Any]:
        import csv
        import os

        from research_platform.core.config import resolve_env_value

        workspace_root_path = Path(workspace_root).resolve()
        batch_path = _resolve_manifest_path(workspace_root_path, manifest["batch"]["path"])
        with batch_path.open("r", encoding="utf-8", newline="") as handle:
            batch_rows = [
                {key: resolve_env_value(value) or "" for key, value in row.items()}
                for row in csv.DictReader(handle, delimiter="\t")
            ]

        mode = str(manifest["execution"]["mode"])
        local_backend = os.environ.get("RP_FMRIPOST_AROMA_LOCAL_BACKEND", DEFAULT_LOCAL_BACKEND)
        slurm_backend = os.environ.get("RP_FMRIPOST_AROMA_SLURM_BACKEND", DEFAULT_HPC_BACKEND)
        backend = slurm_backend if mode == "slurm" else local_backend
        image_repository = os.environ.get("RP_FMRIPOST_AROMA_IMAGE_REPOSITORY", DEFAULT_IMAGE_REPOSITORY)
        image_tag = os.environ.get("RP_FMRIPOST_AROMA_IMAGE_TAG", DEFAULT_IMAGE_TAG)
        container_pull_mode = os.environ.get("RP_FMRIPOST_AROMA_CONTAINER_PULL_MODE")
        if container_pull_mode is None and mode == "slurm" and backend in {"apptainer", "singularity"}:
            container_pull_mode = DEFAULT_APPTAINER_PULL_MODE
        container_image_root = os.environ.get("RP_FMRIPOST_AROMA_IMAGE_ROOT")
        if container_image_root is None and mode == "slurm" and backend in {"apptainer", "singularity"}:
            container_image_root = DEFAULT_APPTAINER_IMAGE_ROOT
        templateflow_home = _optional_text(
            os.environ.get("RP_TEMPLATEFLOW_HOME")
            or os.environ.get("TEMPLATEFLOW_HOME")
        )

        plan = build_batch_runtime_plan(
            raw_bids_root=_resolve_manifest_path(workspace_root_path, manifest["dataset"]["root"]),
            derivative_root=_resolve_manifest_path(workspace_root_path, manifest["dataset"]["derivative_root"]),
            derivative_name=str(manifest["dataset"]["derivative_name"]),
            batch_rows=batch_rows,
            output_root=_resolve_manifest_path(workspace_root_path, manifest["execution"]["output_dir"]),
            work_root=_resolve_manifest_path(workspace_root_path, manifest["execution"]["work_dir"]),
            plan_path=plan_path,
            command_script_path=command_script_path,
            selection=manifest.get("selection", {}),
            backend=backend,
            image_repository=image_repository,
            image_tag=image_tag,
            tool_options=manifest.get("tool", {}).get("options"),
            templateflow_home=templateflow_home,
            resources=manifest.get("resources"),
            container_pull_mode=container_pull_mode,
            container_image_root=container_image_root,
            container_image_name=os.environ.get("RP_FMRIPOST_AROMA_IMAGE_NAME"),
        )
        write_runtime_plan(plan, plan_path)
        write_command_script(plan, command_script_path)
        return plan

    def sync_entries(
        self,
        *,
        workspace_root: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        _ = workspace_root
        return [
            {
                "label": "pipeline-root",
                "source": context["pipeline_root"],
                "sync_scope": "project",
                "exclude_files": [
                    "ops/sync/rsync/exclude.txt",
                    "ops/sync/rsync/exclude.common.txt",
                ],
            },
            {
                "label": "research-neuro",
                "source": Path(context["workspace_root"]) / "packages" / "research-neuro",
                "sync_scope": "project",
                "exclude_files": [
                    "ops/sync/rsync/exclude.txt",
                    "ops/sync/rsync/exclude.common.txt",
                ],
            },
        ]

    def build_publish_back_scaffold(
        self,
        *,
        manifest: dict[str, Any],
        run_root: str,
        workspace_root: str,
    ) -> dict[str, Any]:
        dataset = manifest.get("dataset", {})
        local_dataset_root = _optional_text(dataset.get("local_root"))
        if not local_dataset_root:
            return {}

        output_data_dirname = (
            manifest.get("tool", {}).get("runtime_metadata", {}).get("output_data_dirname") or "fmripost_aroma"
        )
        source_root = Path(run_root).resolve() / "outputs" / str(output_data_dirname)
        destination_root = (Path(workspace_root).resolve() / local_dataset_root / "derivatives" / "fmripost_aroma").resolve()
        return {
            "items": [
                {
                    "source": str(source_root),
                    "destination": str(destination_root),
                }
            ]
        }

    def scaffold_project_defaults(
        self,
        *,
        project_name: str,
        study_root: str,
        derivative_root: str | None,
        task_id: str | None,
    ) -> dict[str, Any]:
        _ = study_root
        if derivative_root is None:
            raise ValueError("fMRIPost-AROMA requires an input derivative root.")
        return {
            "project_name": project_name,
            "pipeline": "preprocess-bids",
            "input_derivative": _infer_input_derivative(derivative_root),
            "default_batch": "default",
            "compute": {
                "default_profile": "local",
                "local": {"jobs": 1},
                "policy": {
                    "default_preset": "neuro-bids",
                    "presets": {
                        "neuro-bids": {
                            "cpus": 4,
                            "ram_gb": 32,
                            "threads": 1,
                            "n_jobs": 1,
                        }
                    },
                    "workloads": {
                        "bids_preprocess": {"preset": "neuro-bids"},
                    },
                },
                "slurm": {
                    "cpus": 4,
                    "mem": "32G",
                    "time": "12:00:00",
                    "modules": ["apptainer/1.3"],
                    "ssh_host": "${RP_HPC_HOST:-}",
                    "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                    "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                },
            },
            "local_profile": "local",
            "slurm_profile": "slurm",
            "tool_options": {
                "denoising_method": "nonaggr",
                "melodic_dimensionality": "",
                "melodic_seed": "",
                "dummy_scans": 0,
                "low_mem": False,
            },
            "publish_back": {"default_policy": "never"},
            "task_id": _optional_text(task_id),
        }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(value: Any, label: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"Pipeline defaults must define {label}.")
    return text


def _resolve_manifest_path(workspace_root: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _infer_input_derivative(derivative_root: str) -> str:
    normalized = str(Path(derivative_root)).strip().lower()
    if "fmriprep" in normalized:
        return "fmriprep"
    return "deepprep-bold"
