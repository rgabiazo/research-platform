"""DeepPrep BIDS preprocessing adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .command import build_runtime_plan
from .selection import discover_batch_rows, expected_remote_input_files

_ALLOWED_TOOL_OPTIONS = frozenset(
    {
        "anat_only",
        "bold_only",
        "bold_sdc",
        "bold_confounds",
        "bold_skip_frame",
        "bold_skip_frames",
        "bold_cifti",
        "bold_surface_spaces",
        "bold_task_type",
        "bold_volume_res",
        "bold_volume_space",
        "device",
        "ignore_error",
        "resume",
        "skip_bids_validation",
    }
)
_ALLOWED_SURFACE_SPACES = frozenset({"None", "fsnative", "fsaverage", "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6"})
_ALLOWED_VOLUME_SPACES = frozenset({"None", "MNI152NLin6Asym", "MNI152NLin2009cAsym"})


class DeepPrepAdapter:
    def tool_name(self) -> str:
        return "deepprep"

    def requires_input_derivative(self) -> bool:
        return False

    def supported_input_derivatives(self) -> tuple[str, ...]:
        return ()

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
        if not _optional_text(dataset.get("bids_root")) and not _optional_text(dataset.get("primary")):
            errors.append("config/dataset.yaml must define dataset.bids_root or dataset.primary for DeepPrep.")

        inputs = preprocessing.get("inputs", {})
        if not isinstance(inputs, dict):
            errors.append("config/preprocessing.yaml preprocessing.inputs must be a mapping when declared.")
        elif _optional_text(inputs.get("fs_license_file")) is None:
            errors.append("config/preprocessing.yaml preprocessing.inputs.fs_license_file is required for DeepPrep.")

        tool_options = preprocessing.get("tool_options")
        if tool_options is not None and not isinstance(tool_options, dict):
            errors.append("config/preprocessing.yaml preprocessing.tool_options must be a mapping.")
        elif isinstance(tool_options, dict):
            errors.extend(_validate_tool_options(tool_options))

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
                "execution_rule_name",
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
        row: Mapping[str, str],
    ) -> list[str]:
        return expected_remote_input_files(
            derivative_root,
            remote_bids_root=remote_derivative_root,
            row=row,
        )

    def expected_remote_auxiliary_files(
        self,
        *,
        context: dict[str, Any],
    ) -> list[str]:
        inputs = context.get("preprocessing", {}).get("inputs", {})
        if not isinstance(inputs, dict):
            return []
        remote_license = _optional_text(_resolve_env_text(inputs.get("remote_fs_license_file")))
        return [remote_license] if remote_license else []

    def runtime_metadata(
        self,
        *,
        pipeline_defaults: dict[str, Any],
        output_dir: str,
    ) -> dict[str, str]:
        _ = pipeline_defaults
        workflow_target = "bids_preprocess"
        rule_name = "bids_preprocess"
        execution_rule_name = "bids_preprocess_unit"
        return {
            "workflow_target": workflow_target,
            "rule_name": rule_name,
            "execution_rule_name": execution_rule_name,
            "runtime_plan_filename": "deepprep-plan.json",
            "command_script_filename": "run-deepprep.sh",
            "completion_marker_filename": "deepprep-complete.txt",
            "output_data_dirname": "deepprep_units",
            "output_dir": str(Path(output_dir)),
        }

    def build_runtime_plan(
        self,
        *,
        manifest: Mapping[str, Any],
        workspace_root: str,
        plan_path: str,
        command_script_path: str,
    ) -> dict[str, Any]:
        return build_runtime_plan(
            manifest=manifest,
            workspace_root=workspace_root,
            plan_path=plan_path,
            command_script_path=command_script_path,
        )

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
        output_data_dirname = manifest.get("tool", {}).get("runtime_metadata", {}).get("output_data_dirname") or "deepprep_units"
        source_root = Path(run_root).resolve() / "outputs" / str(output_data_dirname)
        destination_root = (Path(workspace_root).resolve() / local_dataset_root / "derivatives" / "deepprep").resolve()
        return {"items": [{"source": str(source_root), "destination": str(destination_root)}]}

    def scaffold_project_defaults(
        self,
        *,
        project_name: str,
        study_root: str,
        derivative_root: str | None,
        task_id: str | None,
    ) -> dict[str, Any]:
        _ = derivative_root
        return {
            "project_name": project_name,
            "pipeline": "preprocess-bids",
            "default_batch": "deepprep_default",
            "compute": {
                "default_profile": "local",
                "local": {"jobs": 1},
                "policy": {
                    "default_preset": "deepprep",
                    "presets": {
                        "deepprep": {
                            "cpus": 4,
                            "ram_gb": 32,
                            "threads": 1,
                            "n_jobs": 1,
                        }
                    },
                    "workloads": {
                        "bids_preprocess": {"preset": "deepprep"},
                    },
                },
                "slurm": {
                    "cpus": 4,
                    "mem": "32G",
                    "time": "24:00:00",
                    "ssh_host": "${RP_HPC_HOST:-}",
                    "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                    "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                    "modules": ["apptainer/1.4.5"],
                    "environment": {
                        "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                        "APPTAINER_CONFIGDIR": "$SCRATCH/apptainer-config",
                        "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
                        "XDG_DATA_HOME": "$SCRATCH/.local/share",
                        "TMPDIR": "$SCRATCH/tmp",
                        "TEMP": "$SCRATCH/tmp",
                        "TMP": "$SCRATCH/tmp",
                    },
                    "prepare_directories": [
                        "$APPTAINER_CACHEDIR",
                        "$APPTAINER_CONFIGDIR",
                        "$APPTAINER_TMPDIR",
                        "$XDG_DATA_HOME",
                        "$TMPDIR",
                    ],
                },
                "tool_profiles": {
                    "deepprep": {
                        "local": {
                            "execution_backend": "docker",
                        },
                        "slurm": {
                            "execution_backend": "apptainer",
                            "nextflow": {
                                "enabled": True,
                                "version": "24.10.3",
                                "host_home": "${RP_DEEPPREP_NEXTFLOW_HOME:-$SCRATCH/deepprep/nextflow}",
                                "container_home": "/output/WorkDir/nextflow",
                            },
                            "container": {
                                "enabled": True,
                                "backend": "apptainer",
                                "source_image": "${RP_DEEPPREP_CONTAINER_SOURCE_IMAGE:-docker://pbfslab/deepprep:25.1.0}",
                                "image": "${RP_DEEPPREP_CONTAINER_IMAGE:-docker://pbfslab/deepprep:25.1.0}",
                                "pull_mode": "if_missing",
                                "image_name": "${RP_DEEPPREP_CONTAINER_IMAGE_NAME:-deepprep_25.1.0.sif}",
                                "image_root": "${RP_REMOTE_CONTAINER_ROOT:-$SCRATCH/containers/deepprep}",
                            },
                        },
                    }
                },
            },
            "local_profile": "local",
            "slurm_profile": "local",
            "runtime_profile": "deepprep",
            "inputs": {
                "fs_license_file": "${FS_LICENSE_FILE:-secrets/freesurfer/license.txt}",
                "remote_fs_license_file": "${FS_LICENSE_REMOTE:-}",
            },
            "output": {
                "derivative_name": "deepprep",
                "unit_dir_template": "{subject_id}-{task_id}",
            },
            "tool_options": {
                "bold_task_type": _optional_text(task_id) or "",
                "bold_surface_spaces": "fsnative",
                "bold_volume_space": "MNI152NLin6Asym",
                "bold_volume_res": "02",
                "bold_sdc": True,
                "bold_confounds": True,
                "bold_skip_frame": 0,
                "bold_cifti": False,
                "skip_bids_validation": True,
                "device": "cpu",
                "resume": True,
            },
            "publish_back": {"default_policy": "never"},
            "task_id": _optional_text(task_id),
            "study_root": study_root,
        }


def _validate_tool_options(tool_options: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    unknown = sorted(set(tool_options) - _ALLOWED_TOOL_OPTIONS)
    if unknown:
        errors.append(f"Unsupported preprocessing.tool_options for deepprep: {', '.join(str(name) for name in unknown)}.")
    surface_spaces = _optional_text(tool_options.get("bold_surface_spaces"))
    if surface_spaces and surface_spaces not in _ALLOWED_SURFACE_SPACES:
        allowed = ", ".join(sorted(_ALLOWED_SURFACE_SPACES))
        errors.append(f"preprocessing.tool_options.bold_surface_spaces must be one of: {allowed}.")
    volume_space = _optional_text(tool_options.get("bold_volume_space"))
    if volume_space and volume_space not in _ALLOWED_VOLUME_SPACES:
        allowed = ", ".join(sorted(_ALLOWED_VOLUME_SPACES))
        errors.append(f"preprocessing.tool_options.bold_volume_space must be one of: {allowed}.")
    for name in ("bold_skip_frame", "bold_skip_frames"):
        value = _optional_text(tool_options.get(name))
        if value is None:
            continue
        try:
            int(value)
        except ValueError:
            errors.append(f"preprocessing.tool_options.{name} must be an integer-like value.")
    for name in ("anat_only", "bold_only", "bold_sdc", "bold_confounds", "bold_cifti", "ignore_error", "resume", "skip_bids_validation"):
        value = tool_options.get(name)
        if value not in (None, "", True, False, "true", "false", "True", "False", "1", "0", "yes", "no", "on", "off"):
            errors.append(f"preprocessing.tool_options.{name} must be a boolean-like value.")
    if _coerce_bool(tool_options.get("anat_only")) and _coerce_bool(tool_options.get("bold_only")):
        errors.append("preprocessing.tool_options.anat_only and bold_only cannot both be true.")
    return errors


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_env_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        from research_platform.core.config import resolve_env_value
    except Exception:  # pragma: no cover - defensive
        return _optional_text(value)
    return resolve_env_value(value)


def _coerce_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
