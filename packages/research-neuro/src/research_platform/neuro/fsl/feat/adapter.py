"""First-level FEAT analysis adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .authoring import (
    init_model_document,
    interactive_init_model_document,
    rename_model_document,
    summarize_model_document,
    validate_model_document,
)
from .runtime import build_runtime_plan
from .selection import discover_batch_rows, expected_remote_input_files
from ..common import resolve_reference_path

_AUTO_TEMPLATE = "auto"
_GENERIC_TEMPLATE = "generic"
_FMRIPOST_AROMA_TEMPLATE = "fmripost-aroma-first-level"
_DEEPPREP_T1W_TEMPLATE = "deepprep-t1w-first-level"
_SUPPORTED_SCAFFOLD_TEMPLATES = frozenset(
    {
        _AUTO_TEMPLATE,
        _GENERIC_TEMPLATE,
        _FMRIPOST_AROMA_TEMPLATE,
        _DEEPPREP_T1W_TEMPLATE,
    }
)


class FeatAnalysisAdapter:
    def tool_name(self) -> str:
        return "feat"

    def validate_project(
        self,
        *,
        bundle: dict[str, Any],
        pipeline_defaults: dict[str, Any],
        workspace_root: str,
    ) -> list[str]:
        _ = workspace_root
        analysis = bundle.get("analysis", {}).get("analysis", {})
        dataset = bundle.get("dataset", {}).get("dataset", {})
        errors: list[str] = []

        defaults = analysis.get("defaults", {}) if isinstance(analysis, dict) else {}
        if str(defaults.get("tool", "")).strip() != self.tool_name():
            errors.append(f"This adapter supports only analysis.defaults.tool={self.tool_name()}.")
        if str(defaults.get("stage", "")).strip() != "first_level":
            errors.append("Phase 1 FEAT analysis supports only analysis.defaults.stage=first_level.")
        if dataset.get("input_derivative") is None:
            errors.append("config/dataset.yaml must define dataset.input_derivative for FEAT analysis.")

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
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        return discover_batch_rows(derivative_root, selectors=selectors, context=context)

    def expected_remote_input_files(
        self,
        *,
        derivative_root: str,
        remote_derivative_root: str,
        row: dict[str, str],
        context: dict[str, Any],
    ) -> list[str]:
        return expected_remote_input_files(
            derivative_root,
            remote_derivative_root=remote_derivative_root,
            row=row,
            context=context,
        )

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
        output_data_dirname = _optional_text(planner_outputs.get("output_data_dirname")) or "fsl_feat"
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

    def sync_entries(
        self,
        *,
        workspace_root: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        entries = [
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
        entries.extend(_legacy_external_root_sync_entries(workspace_root=workspace_root, context=context))
        return entries

    def build_runtime_plan(
        self,
        *,
        manifest: dict[str, Any],
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

    def build_publish_back_scaffold(
        self,
        *,
        manifest: dict[str, Any],
        run_root: str,
        workspace_root: str,
    ) -> dict[str, Any]:
        _ = manifest, run_root, workspace_root
        return {}

    def scaffold_project_defaults(
        self,
        *,
        project_name: str,
        study_root: str,
        derivative_root: str,
        task_id: str | None,
        template: str | None = None,
        events_root: str | None = None,
        confounds_root: str | None = None,
        remote_events_root: str | None = None,
        remote_confounds_root: str | None = None,
    ) -> dict[str, Any]:
        _ = study_root
        selected_template, template_reason = _resolve_scaffold_template(
            template=template,
            derivative_root=derivative_root,
            events_root=events_root,
            confounds_root=confounds_root,
        )
        defaults: dict[str, Any] = {
            "project_name": project_name,
            "pipeline": "analysis-bids",
            "input_derivative": _infer_input_derivative(derivative_root),
            "default_batch": "feat_first_level",
            "local_profile": "local",
            "slurm_profile": "slurm",
            "runtime_profile": "fsl",
            "model_ref": "task_glm",
            "compute": {
                "default_profile": "local",
                "local": {"jobs": 1},
                "policy": {
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
                        "bids_analysis_first_level": {"preset": "neuro-bids"},
                    },
                },
                "slurm": {
                    "cpus": 4,
                    "mem": "16G",
                    "time": "02:00:00",
                    "ssh_host": "${RP_HPC_HOST:-}",
                    "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
                    "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
                    "pre_activate_commands": [
                        '[ -n "$SCRATCH" ] || { echo "ERROR: SCRATCH is not set on the remote node." >&2; exit 1; }',
                    ],
                    "prepare_directories": [
                        "$APPTAINER_CACHEDIR",
                        "$APPTAINER_TMPDIR",
                        "$TMPDIR",
                    ],
                    "environment": {
                        "APPTAINER_CACHEDIR": "$SCRATCH/apptainer-cache",
                        "APPTAINER_TMPDIR": "$SCRATCH/apptainer-tmp",
                        "TMPDIR": "$SCRATCH/tmp",
                        "TEMP": "$SCRATCH/tmp",
                        "TMP": "$SCRATCH/tmp",
                    },
                },
                "tool_profiles": {
                    "fsl": {
                        "local": {
                            "execution_backend": "native",
                            "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                        },
                        "slurm": {
                            "execution_backend": "apptainer",
                            "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                            "container": {
                                "enabled": True,
                                "backend": "apptainer",
                                "image": "${RP_FSL_CONTAINER_IMAGE:-docker://vnmd/fsl_6.0.7.4:latest}",
                                "pull_mode": "if_missing",
                                "image_name": "${RP_FSL_CONTAINER_IMAGE_NAME:-fsl_6.0.7.4.sif}",
                                "image_root": "${RP_REMOTE_CONTAINER_ROOT:-$SCRATCH/containers/fsl}",
                            },
                            "scratch": {"root": "${RP_REMOTE_SCRATCH_ROOT:-$SCRATCH}"},
                        },
                    }
                },
            },
            "inputs": {
                "bold": {
                    "derivative_name": _infer_input_derivative(derivative_root),
                    "patterns": [
                        "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                        "{derivative_root}/{subject_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                    ],
                },
                "confounds": {
                    "required": False,
                    "patterns": [
                        "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-confounds_timeseries.tsv",
                        "{derivative_root}/{subject_dir}/func/{bids_base}_desc-confounds_timeseries.tsv",
                    ],
                },
                "evs": {
                    "root": f"datasets/{project_name}/derivatives/events",
                    "patterns": [
                        "{ev_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
                        "{ev_root}/{subject_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
                    ],
                },
            },
            "stage": {
                "validation": {
                    "require_confounds": False,
                    "allow_missing_evs": False,
                    "empty_ev_policy": "as_zero",
                },
                "overwrite": {
                    "design": False,
                    "results": False,
                },
                "settings": {
                    "tr": None,
                    "hpf": 100.0,
                    "smooth_mm": 5.0,
                    "delete_vols": 0,
                    "norm": 0,
                    "prewhiten": 1,
                    "slice_timing": 0,
                    "bet": 0,
                    "mc": 0,
                },
            },
            "model": {
                "name": "task_glm",
                "ev_order": ["condition_a", "condition_b", "button_press"],
                "derivative_on": ["condition_a", "condition_b"],
                "nonconvolved": ["button_press"],
                "contrasts": [
                    {"name": "condition_a_gt_baseline", "weights": [1, 0, 0]},
                    {"name": "condition_a_gt_condition_b", "weights": [1, -1, 0]},
                ],
            },
            "task_id": _optional_text(task_id),
        }
        defaults["template"] = selected_template
        defaults["template_reason"] = template_reason
        if selected_template == _FMRIPOST_AROMA_TEMPLATE:
            _apply_fmripost_aroma_first_level_template(
                defaults,
                events_root=events_root,
                confounds_root=confounds_root,
                remote_events_root=remote_events_root,
                remote_confounds_root=remote_confounds_root,
            )
        if selected_template == _DEEPPREP_T1W_TEMPLATE:
            _apply_deepprep_t1w_first_level_template(
                defaults,
                events_root=events_root,
                confounds_root=confounds_root,
                remote_events_root=remote_events_root,
                remote_confounds_root=remote_confounds_root,
            )
        return defaults

    def init_model_document(
        self,
        *,
        name: str,
        options: Mapping[str, Any],
        template: str | None = None,
    ) -> dict[str, Any]:
        return init_model_document(name=name, options=options, template=template)

    def interactive_init_model_document(
        self,
        *,
        name: str,
        template: str | None = None,
    ) -> dict[str, Any]:
        return interactive_init_model_document(name=name, template=template)

    def validate_model_document(
        self,
        *,
        model_name: str,
        document: Mapping[str, Any],
    ) -> list[str]:
        return validate_model_document(model_name=model_name, document=document)

    def summarize_model_document(
        self,
        *,
        model_name: str,
        document: Mapping[str, Any],
    ) -> str:
        return summarize_model_document(model_name=model_name, document=document)

    def rename_model_document(
        self,
        *,
        new_name: str,
        document: Mapping[str, Any],
    ) -> dict[str, Any]:
        return rename_model_document(new_name=new_name, document=document)


def _resolve_scaffold_template(
    *,
    template: str | None,
    derivative_root: str,
    events_root: str | None,
    confounds_root: str | None,
) -> tuple[str, str]:
    normalized_template = _optional_text(template) or _AUTO_TEMPLATE
    if normalized_template not in _SUPPORTED_SCAFFOLD_TEMPLATES:
        supported = ", ".join(sorted(_SUPPORTED_SCAFFOLD_TEMPLATES))
        raise ValueError(f"Unsupported FEAT analysis scaffold template {normalized_template!r}. Supported templates: {supported}.")

    if normalized_template != _AUTO_TEMPLATE:
        return normalized_template, "selected explicitly"

    if _looks_like_fmripost_aroma_root(derivative_root) and (
        _optional_text(events_root) is not None or _optional_text(confounds_root) is not None
    ):
        return (
            _FMRIPOST_AROMA_TEMPLATE,
            "selected automatically because the derivative root looks like fMRIPost-AROMA and an events/confounds root was provided",
        )

    if _looks_like_deepprep_bold_root(derivative_root) and (
        _optional_text(events_root) is not None or _optional_text(confounds_root) is not None
    ):
        return (
            _DEEPPREP_T1W_TEMPLATE,
            "selected automatically because the derivative root looks like DeepPrep BOLD and an events/confounds root was provided",
        )

    return _GENERIC_TEMPLATE, "selected automatically from generic FEAT defaults"


def _apply_fmripost_aroma_first_level_template(
    defaults: dict[str, Any],
    *,
    events_root: str | None,
    confounds_root: str | None,
    remote_events_root: str | None,
    remote_confounds_root: str | None,
) -> None:
    resolved_events_root = _optional_text(events_root)
    resolved_confounds_root = _optional_text(confounds_root) or resolved_events_root
    if resolved_events_root is None:
        raise ValueError("--events-root is required for --template fmripost-aroma-first-level.")
    if resolved_confounds_root is None:
        raise ValueError("--confounds-root is required for --template fmripost-aroma-first-level.")

    resolved_remote_events_root = _optional_text(remote_events_root)
    resolved_remote_confounds_root = _optional_text(remote_confounds_root) or resolved_remote_events_root

    defaults["external_input_roots"] = {
        "evs": {
            "local_root": resolved_events_root,
            "remote_root": resolved_remote_events_root or "",
            "sync_enabled": True,
        },
        "feat_confounds": {
            "local_root": resolved_confounds_root,
            "remote_root": resolved_remote_confounds_root or "",
            "sync_enabled": True,
        },
    }
    defaults["inputs"] = {
        "bold": {
            "derivative_name": defaults["input_derivative"],
            "patterns": [
                "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-nonaggrDenoised_bold.nii.gz",
                "{derivative_root}/{subject_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-nonaggrDenoised_bold.nii.gz",
                "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
                "{derivative_root}/{subject_dir}/func/{bids_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
            ],
        },
        "confounds": {
            "required": True,
            "root_ref": "feat_confounds",
            "patterns": [
                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-confounds_noGSR.txt",
                "{input_root}/{subject_dir}/func/{bids_base}_desc-confounds_noGSR.txt",
            ],
        },
        "evs": {
            "required": True,
            "root_ref": "evs",
            "patterns": [
                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
                "{input_root}/{subject_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
            ],
        },
    }
    defaults["stage"]["validation"]["require_confounds"] = True
    defaults["stage"]["validation"]["allow_missing_evs"] = False
    defaults["stage"]["validation"]["empty_ev_policy"] = "as_zero"
    defaults["stage"]["settings"]["norm"] = 1


def _apply_deepprep_t1w_first_level_template(
    defaults: dict[str, Any],
    *,
    events_root: str | None,
    confounds_root: str | None,
    remote_events_root: str | None,
    remote_confounds_root: str | None,
) -> None:
    resolved_events_root = _optional_text(events_root)
    resolved_confounds_root = _optional_text(confounds_root) or resolved_events_root
    if resolved_events_root is None:
        raise ValueError("--events-root is required for --template deepprep-t1w-first-level.")
    if resolved_confounds_root is None:
        raise ValueError("--confounds-root is required for --template deepprep-t1w-first-level.")

    resolved_remote_events_root = _optional_text(remote_events_root)
    resolved_remote_confounds_root = _optional_text(remote_confounds_root) or resolved_remote_events_root

    defaults["external_input_roots"] = {
        "evs": {
            "local_root": resolved_events_root,
            "remote_root": resolved_remote_events_root or "",
            "sync_enabled": True,
        },
        "feat_confounds": {
            "local_root": resolved_confounds_root,
            "remote_root": resolved_remote_confounds_root or "",
            "sync_enabled": True,
        },
    }
    defaults["inputs"] = {
        "bold": {
            "derivative_name": defaults["input_derivative"],
            "patterns": [
                "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}_space-T1w_desc-preproc_bold.nii.gz",
                "{derivative_root}/{subject_dir}/func/{bids_base}_space-T1w_desc-preproc_bold.nii.gz",
            ],
        },
        "confounds": {
            "required": True,
            "root_ref": "feat_confounds",
            "patterns": [
                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-confounds_noGSR.txt",
                "{input_root}/{subject_dir}/func/{bids_base}_desc-confounds_noGSR.txt",
            ],
        },
        "evs": {
            "required": True,
            "root_ref": "evs",
            "patterns": [
                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
                "{input_root}/{subject_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
            ],
        },
    }
    defaults["stage"]["validation"]["require_confounds"] = True
    defaults["stage"]["validation"]["allow_missing_evs"] = False
    defaults["stage"]["validation"]["empty_ev_policy"] = "as_zero"
    defaults["stage"]["settings"]["hpf"] = 100.0
    defaults["stage"]["settings"]["smooth_mm"] = 0.0
    defaults["stage"]["settings"]["delete_vols"] = 0
    defaults["stage"]["settings"]["norm"] = 0
    defaults["stage"]["settings"]["prewhiten"] = 1
    defaults["stage"]["settings"]["slice_timing"] = 0
    defaults["stage"]["settings"]["bet"] = 0
    defaults["stage"]["settings"]["mc"] = 0


def _looks_like_fmripost_aroma_root(value: str) -> bool:
    normalized = str(Path(value).name).lower().replace("_", "-")
    return "fmripost-aroma" in normalized or "aroma" in normalized


def _looks_like_deepprep_bold_root(value: str) -> bool:
    parts = [part.lower().replace("_", "-") for part in Path(value).parts]
    return "deepprep" in parts and ("bold" in parts or str(Path(value).name).lower() == "bold")


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


def _infer_input_derivative(derivative_root: str) -> str:
    normalized = Path(derivative_root).name.strip()
    if not normalized:
        return "fmriprep"
    if "fmriprep" in normalized.lower():
        return "fmriprep"
    return normalized


def _legacy_external_root_sync_entries(*, workspace_root: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    workspace_root_path = Path(workspace_root).resolve()
    inputs = context.get("analysis_inputs", {})
    if not isinstance(inputs, dict):
        return []

    entries: list[dict[str, Any]] = []
    for input_name, input_config in inputs.items():
        if not isinstance(input_config, dict):
            continue
        if str(input_config.get("root_ref", "")).strip():
            continue
        root_value = str(input_config.get("root", "")).strip()
        if not root_value:
            continue
        root_path = resolve_reference_path(workspace_root, root_value)
        try:
            root_path.relative_to(workspace_root_path)
        except ValueError:
            destination = str(input_config.get("remote_root", "")).strip() or None
        else:
            destination = None
        entries.append(
            {
                "label": f"feat-{input_name.replace('_', '-')}-root",
                "source": root_path,
                "destination": destination,
                "sync_scope": "data",
                "exclude_files": [
                    "ops/sync/rsync/exclude.txt",
                    "ops/sync/rsync/exclude.common.txt",
                ],
            }
        )
    return entries
