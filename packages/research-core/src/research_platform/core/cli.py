"""Workspace orchestration CLI for validated local workflows and plan-first integrations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import socket
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlsplit

from research_platform.hpc.bootstrap import build_bootstrap_execution_plan, execute_bootstrap_plan
from research_platform.hpc.connection import build_manifest_hpc_connection_hint
from research_platform.hpc.manifest import read_status, read_run_manifest, write_run_manifest, write_status
from research_platform.hpc.offline_validation import (
    render_hpc_validation_report,
    resolve_local_hpc_environment_defaults,
    validate_hpc_configuration,
)
from research_platform.hpc.remote import (
    build_cancel_plan,
    build_pull_plan,
    build_stage_plan,
    build_submit_plan,
    execute_pull_plan,
    execute_stage_plan,
    execute_submit_plan,
    resolve_remote_run_root,
    verify_remote_paths,
)
from research_platform.hpc.ssh import build_ssh_command, render_ssh_shell, run_ssh_connectivity_check
from research_platform.hpc.ssh_profiles import (
    build_ssh_config_template,
    load_ssh_profile,
    materialize_ssh_profile_entry,
    require_generic_profile_isolation,
    resolve_ssh_profile_config_path,
    upsert_ssh_profile_document,
)
from research_platform.hpc.sync import build_rsync_push_command as build_basic_rsync_push_command
from research_platform.hpc.slurm import (
    build_slurm_command_script,
    build_slurm_jobspec,
    build_slurm_setup_commands,
    normalize_slurm_batch_script,
    normalize_modules as normalize_slurm_modules,
    normalize_prepare_directories as normalize_slurm_prepare_directories,
    normalize_shell_commands as normalize_slurm_shell_commands,
    render_slurm_script,
    write_slurm_script,
)

from .bootstrap import build_bootstrap_manifest
from .compute import resolve_resource_plan
from .config import (
    ProjectOverlayNotFoundError,
    WorkspaceRootNotFoundError,
    apply_hpc_target_defaults,
    build_data_root_spec,
    dump_yaml,
    list_hpc_targets,
    merge_hpc_target_compute_defaults,
    merge_workspace_hpc_runtime_compute_defaults,
    merge_declared_data_roots,
    default_project_name,
    load_local_hpc_env_defaults,
    load_yaml,
    load_project_bundle,
    load_project_record,
    load_hpc_targets_config,
    load_workspace_config,
    parse_yaml,
    project_slice,
    resolve_hpc_slurm_site_settings,
    resolve_hpc_target,
    resolve_hpc_target_name,
    resolve_hpc_targets_config_path,
    resolve_bids_dataset_root,
    resolve_bids_input_derivative_root,
    resolve_bids_remote_dataset_root,
    resolve_bids_remote_input_derivative_root,
    resolve_project_hpc_data_root_declarations,
    resolve_analysis_external_input_root_declarations,
    resolve_env_value,
    resolve_workspace_hpc_runtime_default,
    summarize_bundle,
    validate_tabular_feature_columns,
    validate_project_bundle,
    write_local_hpc_env_defaults,
    write_yaml,
    workspace_root,
)
from .hpc_projects import (
    adapter_data_root_declarations,
    build_project_hpc_context,
    build_workspace_hpc_context,
    default_notebook_launch_path,
    notebook_launch_target,
)
from .manifests import filter_manifest_rows, normalize_filter_values
from .paths import dataset_path, ensure_dir, pipeline_path, project_path, run_path, to_workspace_relative, workspace_paths
from .publish_back import build_publish_back_plan, build_publish_back_scaffold
from .run_lifecycle import (
    RunLifecycleError,
    acquire_execution_claim,
    build_plan_identity,
    claim_path,
    path_entry_exists,
    validate_run_id,
    verify_plan_identity,
)
from .tabular_output_transaction import (
    FINAL_OUTPUT_DIRECTORY as TABULAR_FINAL_OUTPUT_DIRECTORY,
    OutputRecord as TabularOutputRecord,
    OutputSpec as TabularOutputSpec,
    OwnedStaging as TabularOwnedStaging,
    TabularOutputTransactionError,
    atomic_no_replace_support_error as tabular_atomic_no_replace_support_error,
    build_transaction_plan as build_tabular_transaction_plan,
    cleanup_owned_staging as cleanup_tabular_staging,
    create_owned_staging as create_tabular_staging,
    output_specs_from_plan as tabular_output_specs_from_plan,
    preflight_transaction_root as preflight_tabular_transaction_root,
    promote_staging_no_replace as promote_tabular_staging_no_replace,
    read_owned_regular_file as read_tabular_transaction_file,
    seal_staged_transaction as seal_tabular_staged_transaction,
    transaction_staging_entries as tabular_staging_entries,
    validate_committed_transaction as validate_committed_tabular_transaction,
    validate_sealed_transaction as validate_sealed_tabular_transaction,
    validate_staged_outputs as validate_tabular_staged_outputs,
)
from .provision import build_project_data_sync_plan, build_project_sync_plan, build_provision_plan, build_workspace_sync_plan
from .tool_adapters import (
    load_bids_analysis_tool_adapter,
    load_bids_tool_adapter,
    load_registered_bids_analysis_tool_adapter,
    load_registered_bids_tool_adapter,
    require_bids_analysis_model_authoring_adapter,
    registered_bids_analysis_tools,
    registered_bids_tools,
    resolve_bids_analysis_tool_adapter_ref,
    resolve_bids_tool_adapter_ref,
)
from .workflow_registry import WORKFLOWS, get_workflow, render_workflow_menu
from .version import version_report


@dataclass(frozen=True)
class _GitWorkspaceExcludeLayer:
    active: bool
    paths: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.paths)


@dataclass
class _WorkspaceGitSafetyExcludes:
    untracked: _GitWorkspaceExcludeLayer
    ignored: _GitWorkspaceExcludeLayer
    exclude_file: Path | None = None
    temporary_root: Path | None = None

    @property
    def total_count(self) -> int:
        return self.untracked.count + self.ignored.count

    def cleanup(self) -> None:
        if self.temporary_root is not None:
            shutil.rmtree(self.temporary_root, ignore_errors=True)


_SALLOC_GRANTED_JOB_ALLOCATION_PATTERN = re.compile(r"Granted job allocation (\d+)")
_SBATCH_SUBMITTED_JOB_PATTERN = re.compile(r"Submitted batch job (\d+)")
_HPC_TUNNEL_MODES = ("direct", "login-forward")
_SSH_MUX_SESSION_REFUSED_PATTERNS = (
    "mux_client_request_session: session request failed: Session open refused by peer",
    "session request failed: session open refused by peer",
    "session open refused by peer",
)
_NOTEBOOK_BOOTSTRAP_STAMP_FILENAME = ".rp-notebook-bootstrap.sha256"
_NOTEBOOK_BOOTSTRAP_METADATA_FILENAMES = ("pyproject.toml", "setup.cfg", "setup.py")
_TABULAR_OUTPUT_ROOT_VARIABLE = "RP_TABULAR_OUTPUT_ROOT"
_TABULAR_OUTPUT_ROOT_TOKEN = "${RP_TABULAR_OUTPUT_ROOT}"


_ROI_SET_SCAFFOLD_TEMPLATES = (
    "atlas_label",
    "coordinate_sphere",
    "data_driven_hook",
    "functional_threshold_map",
    "loso_group_map",
    "manual_mask",
)
_ROI_EXTRACTION_SCAFFOLD_TEMPLATES = ("fsl_featquery", "generic_nifti")
_ROI_SCAFFOLD_PATH_PROFILES = ("generic", "research_platform_fsl_ffx")


def main(argv: list[str] | None = None) -> int:
    command_args = list(sys.argv[1:] if argv is None else argv)
    if command_args == ["--version"]:
        print(version_report())
        return 0
    parser = _build_parser()
    generic_project_name = _literal_project_init_name(command_args)
    args = (
        argparse.Namespace(project=generic_project_name, handler=_handle_project_init_generic)
        if generic_project_name is not None
        else parser.parse_args(command_args)
    )
    if command_args[:2] not in (["hpc", "setup"], ["hpc", "validate"]):
        try:
            load_local_hpc_env_defaults()
            apply_hpc_target_defaults(project_name=None)
        except WorkspaceRootNotFoundError:
            pass
    try:
        return getattr(args, "handler")(args)
    except WorkspaceRootNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except ProjectOverlayNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except (RunLifecycleError, TabularOutputTransactionError, _TabularStatusPersistenceError) as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc


def _literal_project_init_name(argv: list[str]) -> str | None:
    if len(argv) != 3 or argv[:2] != ["project", "init"]:
        return None
    candidate = argv[2]
    if candidate.startswith("-") or candidate in {"bids-preprocess", "bids-analysis", "tabular-model"}:
        return None
    return candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Workspace orchestration CLI for validated local workflows and plan-first integrations."
    )
    parser.add_argument("--version", action="version", version=version_report())
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Check workspace readiness and show guided next steps.")
    setup.set_defaults(handler=_handle_setup)

    onboard = subparsers.add_parser("onboard", help="Guided beginner project onboarding.")
    onboard_subparsers = onboard.add_subparsers(dest="onboard_workflow")
    onboard.set_defaults(handler=_handle_onboard_menu)
    for workflow_name in ("preprocess", "analysis", "tabular", "notebook", "custom"):
        workflow_parser = onboard_subparsers.add_parser(workflow_name, help=f"Guided {workflow_name} onboarding.")
        workflow_parser.set_defaults(handler=_handle_onboard_workflow)

    project_parser = subparsers.add_parser("project", help="Initialize project overlays.")
    project_subparsers = project_parser.add_subparsers(dest="project_command", required=True)
    project_init = project_subparsers.add_parser(
        "init",
        help="Scaffold a beginner-friendly project overlay.",
        description=(
            "Use `rp project init <name>` for a validated tabular starter, or select a specialized "
            "BIDS or tabular scaffold below."
        ),
    )
    project_init_targets = project_init.add_subparsers(dest="project_init_target", required=True)
    project_init_bids = project_init_targets.add_parser(
        "bids-preprocess",
        help="Scaffold an adapter-driven BIDS preprocessing project.",
    )
    project_init_bids.add_argument("--project", required=True, help="Project overlay name to create.")
    project_init_bids.add_argument("--study-root", required=True, help="Local raw BIDS dataset root.")
    project_init_bids.add_argument(
        "--derivative-root",
        default=None,
        help="Local input derivative root. Required for derivative-backed tools.",
    )
    project_init_bids.add_argument(
        "--tool",
        required=True,
        choices=registered_bids_tools(),
        help="Configured BIDS preprocessing tool.",
    )
    project_init_bids.add_argument("--task-id", default=None, help="Optional default task selector.")
    project_init_bids.add_argument("--remote-study-root", default=None, help="Optional remote raw BIDS dataset root.")
    project_init_bids.add_argument(
        "--remote-derivative-root",
        default=None,
        help="Optional remote input derivative root for derivative-backed tools.",
    )
    project_init_bids.set_defaults(handler=_handle_project_init_bids_preprocess)
    project_init_analysis = project_init_targets.add_parser(
        "bids-analysis",
        help="Scaffold an adapter-driven BIDS analysis project.",
    )
    project_init_analysis.add_argument("--project", required=True, help="Project overlay name to create.")
    project_init_analysis.add_argument("--study-root", required=True, help="Local raw BIDS dataset root.")
    project_init_analysis.add_argument("--derivative-root", required=True, help="Local input derivative root.")
    project_init_analysis.add_argument(
        "--tool",
        required=True,
        choices=registered_bids_analysis_tools(),
        help="Configured BIDS analysis tool.",
    )
    project_init_analysis.add_argument("--task-id", default=None, help="Optional default task selector.")
    project_init_analysis.add_argument("--remote-study-root", default=None, help="Optional remote raw BIDS dataset root.")
    project_init_analysis.add_argument(
        "--remote-derivative-root",
        default=None,
        help="Optional remote input derivative root.",
    )
    project_init_analysis.add_argument(
        "--template",
        default="auto",
        help=(
            "Analysis scaffold template. Use auto, generic, or a tool-specific template "
            "such as fmripost-aroma-first-level or deepprep-t1w-first-level."
        ),
    )
    project_init_analysis.add_argument("--events-root", default=None, help="Optional local first-level event/EV root.")
    project_init_analysis.add_argument("--confounds-root", default=None, help="Optional local first-level confounds root.")
    project_init_analysis.add_argument("--remote-events-root", default=None, help="Optional remote first-level event/EV root.")
    project_init_analysis.add_argument("--remote-confounds-root", default=None, help="Optional remote first-level confounds root.")
    project_init_analysis.add_argument("--hpc-target", default=None, help="Optional HPC target whose defaults should be applied while scaffolding.")
    project_init_analysis.set_defaults(handler=_handle_project_init_bids_analysis)
    project_init_tabular = project_init_targets.add_parser(
        "tabular-model",
        help="Scaffold a beginner-friendly tabular model project.",
    )
    project_init_tabular.add_argument("--project", required=True, help="Project overlay name to create.")
    project_init_tabular.add_argument("--dataset", default=None, help="Primary dataset name. Defaults to the project name.")
    project_init_tabular.add_argument(
        "--canonical-dataset",
        default=None,
        help="Canonical dataset name used for feature tables. Defaults to --dataset.",
    )
    project_init_tabular.add_argument(
        "--canonical-features-root",
        default=None,
        help="Relative feature-table root under the canonical dataset. Defaults to derivatives/features/<project>.",
    )
    project_init_tabular.add_argument("--batch", default=None, help="Default batch name. Defaults to default.")
    project_init_tabular.set_defaults(handler=_handle_project_init_tabular_model)

    config_parser = subparsers.add_parser("config", help="Inspect and validate project configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    for name, handler, help_text in (
        ("validate", _handle_config_validate, "Validate project configuration contracts."),
        ("show", _handle_config_show, "Show the merged project configuration."),
        (
            "paths",
            _handle_config_paths,
            "Show resolved workspace, project, dataset, and named analysis-input paths.",
        ),
    ):
        command = config_subparsers.add_parser(name, help=help_text, description=help_text)
        command.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
        command.set_defaults(handler=handler)

    batch_parser = subparsers.add_parser("batch", help="Inspect project batch manifests.")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command", required=True)
    batch_list = batch_subparsers.add_parser("list", help="List batch manifests.")
    batch_list.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    batch_list.set_defaults(handler=_handle_batch_list)

    batch_show = batch_subparsers.add_parser("show", help="Show a batch manifest.")
    batch_show.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    batch_show.add_argument("--batch", default=None, help="Batch name without .tsv.")
    batch_show.set_defaults(handler=_handle_batch_show)

    batch_discover = batch_subparsers.add_parser("discover", help="Discover a deterministic BIDS batch manifest.")
    batch_discover_targets = batch_discover.add_subparsers(dest="batch_target", required=True)
    batch_discover_bids = batch_discover_targets.add_parser("bids", help="Discover BIDS preprocessing rows from the configured adapter.")
    batch_discover_bids.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    batch_discover_bids.add_argument("--batch", default=None, help="Batch name without .tsv. Defaults to preprocessing.default_batch.")
    _add_bids_selector_args(batch_discover_bids)
    batch_discover_bids.set_defaults(handler=_handle_batch_discover_bids)
    batch_discover_analysis = batch_discover_targets.add_parser("analysis", help="Discover project batch rows.")
    batch_discover_analysis_targets = batch_discover_analysis.add_subparsers(dest="batch_analysis_target", required=True)
    batch_discover_analysis_bids = batch_discover_analysis_targets.add_parser(
        "bids",
        help="Discover BIDS analysis rows from the configured adapter.",
    )
    batch_discover_analysis_bids.add_argument(
        "--project",
        default=None,
        help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.",
    )
    batch_discover_analysis_bids.add_argument(
        "--batch",
        default=None,
        help="Batch name without .tsv. Defaults to analysis.stages.<stage>.default_batch.",
    )
    batch_discover_analysis_bids.add_argument("--stage", default=None, help="Analysis stage. Defaults to analysis.defaults.stage.")
    _add_bids_selector_args(batch_discover_analysis_bids)
    batch_discover_analysis_bids.set_defaults(handler=_handle_batch_discover_analysis_bids)

    analysis_parser = subparsers.add_parser("analysis", help="Manage analysis configuration and lifecycle commands.")
    analysis_subparsers = analysis_parser.add_subparsers(dest="analysis_command", required=True)

    analysis_bundle = analysis_subparsers.add_parser(
        "bundle",
        help="Initialize, inspect, validate, and resolve plan-only analysis bundles.",
    )
    analysis_bundle_subparsers = analysis_bundle.add_subparsers(dest="analysis_bundle_command", required=True)

    analysis_bundle_init = analysis_bundle_subparsers.add_parser(
        "init",
        help="Create a small configuration-owned analysis-bundle scaffold.",
    )
    analysis_bundle_init.add_argument("name", help="Bundle name without .yaml.")
    analysis_bundle_init.add_argument("--project", required=True, help="Project overlay name.")
    analysis_bundle_init.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned path and YAML without writing it.",
    )
    analysis_bundle_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the bundle scaffold YAML if it already exists.",
    )
    analysis_bundle_init.set_defaults(handler=_handle_analysis_bundle_init)

    analysis_bundle_list = analysis_bundle_subparsers.add_parser("list", help="List analysis-bundle YAML files.")
    analysis_bundle_list.add_argument(
        "--project",
        default=None,
        help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.",
    )
    analysis_bundle_list.set_defaults(handler=_handle_analysis_bundle_list)

    for command_name, handler, help_text in (
        ("show", _handle_analysis_bundle_show, "Show one analysis-bundle document."),
        ("validate", _handle_analysis_bundle_validate, "Validate one analysis-bundle document."),
        ("doctor", _handle_analysis_bundle_doctor, "Inspect bundle selection and component readiness without execution."),
        ("plan", _handle_analysis_bundle_plan, "Resolve exact analysis units into a deterministic no-write plan."),
    ):
        command = analysis_bundle_subparsers.add_parser(command_name, help=help_text)
        command.add_argument("name", help="Bundle name without .yaml.")
        command.add_argument(
            "--project",
            default=None,
            help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.",
        )
        command.set_defaults(handler=handler)

    analysis_model = analysis_subparsers.add_parser("model", help="Create and inspect analysis model YAML files.")
    analysis_model_subparsers = analysis_model.add_subparsers(dest="analysis_model_command", required=True)

    analysis_model_init = analysis_model_subparsers.add_parser("init", help="Create a new analysis model YAML file.")
    _add_analysis_model_common_args(analysis_model_init)
    analysis_model_init.add_argument("name", help="Model file name without .yaml.")
    analysis_model_init.add_argument("--interactive", action="store_true", help="Launch the tool-specific interactive model wizard.")
    analysis_model_init.add_argument("--template", choices=("blank", "basic"), default=None, help="Write a tool-specific starter template.")
    analysis_model_init.add_argument("--force", action="store_true", help="Overwrite the destination file if it already exists.")
    analysis_model_init.add_argument("--ev-order", nargs="+", default=None, help="Ordered EV names.")
    analysis_model_init.add_argument("--derivative-on", nargs="+", default=None, help="EVs with temporal derivatives enabled.")
    analysis_model_init.add_argument("--nonconvolved", nargs="+", default=None, help="EVs that should stay nonconvolved.")
    analysis_model_init.add_argument(
        "--contrast",
        action="append",
        default=None,
        help="Contrast spec in the form name:w1,w2,... Repeat for multiple contrasts.",
    )
    analysis_model_init.set_defaults(handler=_handle_analysis_model_init)

    analysis_model_copy = analysis_model_subparsers.add_parser("copy", help="Copy an existing analysis model YAML file.")
    _add_analysis_model_common_args(analysis_model_copy)
    analysis_model_copy.add_argument("source", help="Existing model name without .yaml.")
    analysis_model_copy.add_argument("dest", help="New model name without .yaml.")
    analysis_model_copy.add_argument("--force", action="store_true", help="Overwrite the destination file if it already exists.")
    analysis_model_copy.set_defaults(handler=_handle_analysis_model_copy)

    analysis_model_show = analysis_model_subparsers.add_parser("show", help="Show an analysis model.")
    _add_analysis_model_common_args(analysis_model_show)
    analysis_model_show.add_argument("name", help="Model name without .yaml.")
    analysis_model_show.add_argument("--format", choices=("summary", "yaml"), default="summary", help="Render a human summary or the raw YAML file.")
    analysis_model_show.set_defaults(handler=_handle_analysis_model_show)

    analysis_model_validate = analysis_model_subparsers.add_parser("validate", help="Validate one or more analysis models.")
    _add_analysis_model_common_args(analysis_model_validate)
    analysis_model_validate.add_argument("name", nargs="?", help="Model name without .yaml.")
    analysis_model_validate.add_argument("--all", action="store_true", help="Validate every model file under config/analysis/models.")
    analysis_model_validate.set_defaults(handler=_handle_analysis_model_validate)

    analysis_design = analysis_subparsers.add_parser("design", help="Configure analysis design settings.")
    analysis_design_subparsers = analysis_design.add_subparsers(dest="analysis_design_command", required=True)
    analysis_design_configure = analysis_design_subparsers.add_parser("configure", help="Update generated analysis design config.")
    analysis_design_configure_targets = analysis_design_configure.add_subparsers(dest="analysis_design_target", required=True)
    analysis_design_configure_first = analysis_design_configure_targets.add_parser(
        "first-level",
        help="Configure first-level BIDS analysis inputs and settings.",
    )
    analysis_design_configure_first.add_argument("--project", required=True, help="Project overlay name.")
    analysis_design_configure_first.add_argument("--bold-space", default=None, help="BIDS space label for BOLD inputs, e.g. T1w.")
    analysis_design_configure_first.add_argument("--bold-desc", default=None, help="BIDS desc label for BOLD inputs, e.g. preproc.")
    analysis_design_configure_first.add_argument("--events-root", default=None, help="Optional local first-level EV root.")
    analysis_design_configure_first.add_argument("--remote-events-root", default=None, help="Optional remote first-level EV root.")
    analysis_design_configure_first.add_argument("--confounds-root", default=None, help="Optional local first-level confounds root.")
    analysis_design_configure_first.add_argument("--remote-confounds-root", default=None, help="Optional remote first-level confounds root.")
    analysis_design_configure_first.add_argument(
        "--confounds-pattern",
        default=None,
        help="Confound filename or pattern, e.g. desc-confounds_noGSR.txt.",
    )
    analysis_design_configure_first.add_argument("--tr", type=float, default=None, help="TR in seconds.")
    analysis_design_configure_first.add_argument("--hpf", type=float, default=None, help="High-pass filter cutoff in seconds.")
    analysis_design_configure_first.add_argument("--smooth-mm", type=float, default=None, help="Spatial smoothing FWHM in mm.")
    analysis_design_configure_first.add_argument("--norm", choices=("on", "off"), default=None, help="Intensity normalization.")
    analysis_design_configure_first.add_argument(
        "--motion-correction",
        choices=("on", "off"),
        default=None,
        help="Motion correction.",
    )
    analysis_design_configure_first.add_argument(
        "--slice-timing",
        choices=("on", "off"),
        default=None,
        help="Slice-timing correction.",
    )
    analysis_design_configure_first.add_argument("--bet", choices=("on", "off"), default=None, help="BET brain extraction.")
    analysis_design_configure_first.add_argument("--prewhiten", choices=("on", "off"), default=None, help="FILM prewhitening.")
    analysis_design_configure_first.add_argument(
        "--empty-ev-policy",
        choices=("as_zero", "fail"),
        default=None,
        help="How first-level FEAT should handle existing empty EV files.",
    )
    analysis_design_configure_first.add_argument(
        "--output-desc",
        default=None,
        help="Optional BIDS-style desc label to append to generated analysis output names.",
    )
    analysis_design_configure_first.add_argument("--dry-run", action="store_true", help="Print the updated YAML without writing it.")
    analysis_design_configure_first.set_defaults(handler=_handle_analysis_design_configure_first_level)

    analysis_roi = analysis_subparsers.add_parser(
        "roi",
        help="Initialize, validate, preflight, plan, or explicitly execute local ROI workflows.",
    )
    analysis_roi_subparsers = analysis_roi.add_subparsers(dest="analysis_roi_command", required=True)
    analysis_roi_list = analysis_roi_subparsers.add_parser("list", help="List ROI, transform, and extraction set YAML files.")
    analysis_roi_list.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_list.set_defaults(handler=_handle_analysis_roi_list)

    analysis_roi_init = analysis_roi_subparsers.add_parser("init", help="Scaffold a new ROI set YAML file.")
    analysis_roi_init.add_argument("name", help="ROI set name without .yaml.")
    analysis_roi_init.add_argument("--project", required=True, help="Project overlay name.")
    analysis_roi_init.add_argument(
        "--template",
        required=True,
        choices=_ROI_SET_SCAFFOLD_TEMPLATES,
        help=(
            "Scaffold template. coordinate_sphere, manual_mask, and functional_threshold_map use the local "
            "NIfTI runtime with user-provided inputs; loso_group_map requires FSL; atlas_label and "
            "data_driven_hook are deferred scaffolds."
        ),
    )
    analysis_roi_init.add_argument("--force", action="store_true", help="Overwrite the destination file if it already exists.")
    analysis_roi_init.add_argument("--dry-run", action="store_true", help="Print the planned path and YAML without writing.")
    _add_roi_set_scaffold_override_args(analysis_roi_init)
    analysis_roi_init.set_defaults(handler=_handle_analysis_roi_init)

    analysis_roi_show = analysis_roi_subparsers.add_parser("show", help="Show an ROI set.")
    analysis_roi_show.add_argument("name", help="ROI set name without .yaml.")
    analysis_roi_show.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_show.add_argument("--format", choices=("summary", "yaml"), default="summary", help="Render a summary or raw YAML.")
    analysis_roi_show.set_defaults(handler=_handle_analysis_roi_show)

    analysis_roi_validate = analysis_roi_subparsers.add_parser("validate", help="Validate an ROI set.")
    analysis_roi_validate.add_argument("name", help="ROI set name without .yaml.")
    analysis_roi_validate.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_validate.set_defaults(handler=_handle_analysis_roi_validate)

    analysis_roi_doctor = analysis_roi_subparsers.add_parser("doctor", help="Preflight an ROI set without executing expensive tools.")
    analysis_roi_doctor.add_argument("name", help="ROI set name without .yaml.")
    analysis_roi_doctor.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_doctor.set_defaults(handler=_handle_analysis_roi_doctor)

    analysis_roi_build = analysis_roi_subparsers.add_parser("build", help="Plan or execute a local ROI mask build.")
    analysis_roi_build.add_argument("name", help="ROI set name without .yaml.")
    analysis_roi_build.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_build.add_argument("--execute", action="store_true", help="Write ROI masks and sidecars after planning.")
    analysis_roi_build.set_defaults(handler=_handle_analysis_roi_build)

    analysis_roi_transform = analysis_roi_subparsers.add_parser(
        "transform",
        help="Advanced: validate, preflight, plan, or execute externally backed ROI mask transforms.",
    )
    analysis_roi_transform_subparsers = analysis_roi_transform.add_subparsers(dest="analysis_roi_transform_command", required=True)
    analysis_roi_transform_validate = analysis_roi_transform_subparsers.add_parser("validate", help="Validate an ROI transform set.")
    analysis_roi_transform_validate.add_argument("name", help="ROI transform set name without .yaml.")
    analysis_roi_transform_validate.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_transform_validate.set_defaults(handler=_handle_analysis_roi_transform_validate)
    analysis_roi_transform_doctor = analysis_roi_transform_subparsers.add_parser(
        "doctor",
        help="Preflight an ROI transform set without running transform tools.",
    )
    analysis_roi_transform_doctor.add_argument("name", help="ROI transform set name without .yaml.")
    analysis_roi_transform_doctor.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_transform_doctor.set_defaults(handler=_handle_analysis_roi_transform_doctor)
    analysis_roi_transform_plan = analysis_roi_transform_subparsers.add_parser("plan", help="Render a no-write ROI transform plan.")
    analysis_roi_transform_plan.add_argument("name", help="ROI transform set name without .yaml.")
    analysis_roi_transform_plan.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_transform_plan.set_defaults(handler=_handle_analysis_roi_transform_plan)
    analysis_roi_transform_run = analysis_roi_transform_subparsers.add_parser("run", help="Preview or execute ROI transforms.")
    analysis_roi_transform_run.add_argument("name", help="ROI transform set name without .yaml.")
    analysis_roi_transform_run.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_transform_run.add_argument("--execute", action="store_true", help="Run planned transforms and write planned masks/QC.")
    analysis_roi_transform_run.set_defaults(handler=_handle_analysis_roi_transform_run)

    analysis_roi_extraction = analysis_roi_subparsers.add_parser(
        "extraction",
        help="Initialize, validate, preflight, plan, or explicitly execute local ROI extraction sets.",
    )
    analysis_roi_extraction_subparsers = analysis_roi_extraction.add_subparsers(dest="analysis_roi_extraction_command", required=True)
    analysis_roi_extraction_init = analysis_roi_extraction_subparsers.add_parser("init", help="Scaffold a new ROI extraction set YAML file.")
    analysis_roi_extraction_init.add_argument("name", help="Extraction set name without .yaml.")
    analysis_roi_extraction_init.add_argument("--project", required=True, help="Project overlay name.")
    analysis_roi_extraction_init.add_argument("--roi-set", required=True, help="Referenced ROI set name without .yaml.")
    analysis_roi_extraction_init.add_argument(
        "--template",
        required=True,
        choices=_ROI_EXTRACTION_SCAFFOLD_TEMPLATES,
        help=(
            "Scaffold template. generic_nifti uses the local NIfTI runtime with user-provided inputs; "
            "fsl_featquery requires FSL and FEAT inputs."
        ),
    )
    analysis_roi_extraction_init.add_argument("--force", action="store_true", help="Overwrite the destination file if it already exists.")
    analysis_roi_extraction_init.add_argument("--dry-run", action="store_true", help="Print the planned path and YAML without writing.")
    _add_roi_extraction_scaffold_override_args(analysis_roi_extraction_init)
    analysis_roi_extraction_init.set_defaults(handler=_handle_analysis_roi_extraction_init)

    analysis_roi_extraction_validate = analysis_roi_extraction_subparsers.add_parser("validate", help="Validate an ROI extraction set.")
    analysis_roi_extraction_validate.add_argument("name", help="Extraction set name without .yaml.")
    analysis_roi_extraction_validate.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_extraction_validate.set_defaults(handler=_handle_analysis_roi_extraction_validate)

    analysis_roi_extraction_doctor = analysis_roi_extraction_subparsers.add_parser(
        "doctor",
        help="Preflight an ROI extraction set without running extraction tools.",
    )
    analysis_roi_extraction_doctor.add_argument("name", help="Extraction set name without .yaml.")
    analysis_roi_extraction_doctor.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_extraction_doctor.set_defaults(handler=_handle_analysis_roi_extraction_doctor)

    analysis_roi_extraction_run = analysis_roi_extraction_subparsers.add_parser(
        "run",
        help="Plan or execute local ROI extraction.",
    )
    analysis_roi_extraction_run.add_argument("name", help="Extraction set name without .yaml.")
    analysis_roi_extraction_run.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_roi_extraction_run.add_argument("--execute", action="store_true", help="Write extraction tables after planning.")
    analysis_roi_extraction_run.set_defaults(handler=_handle_analysis_roi_extraction_run)

    analysis_mvpa = analysis_subparsers.add_parser(
        "mvpa",
        help="Initialize, validate, plan, or run local MVPA configuration.",
    )
    analysis_mvpa_subparsers = analysis_mvpa.add_subparsers(dest="analysis_mvpa_command", required=True)

    analysis_mvpa_init = analysis_mvpa_subparsers.add_parser(
        "init",
        help="Create one config-first MVPA set scaffold (materialized prepared vectors by default).",
    )
    analysis_mvpa_init.add_argument("analysis_id", help="MVPA set name used for the generated YAML filename and config name.")
    analysis_mvpa_init.add_argument("--project", required=True, help="Project overlay name.")
    analysis_mvpa_init.add_argument(
        "--template",
        default="materialized-crossnobis",
        choices=("materialized-crossnobis", "fsl-feat-crossnobis", "distance-rdm"),
        help=(
            "Scaffold template: materialized-crossnobis is the dependency-light prepared-vector path; "
            "fsl-feat-crossnobis requires external FEAT/image inputs; distance-rdm is compatibility-only."
        ),
    )
    analysis_mvpa_init.add_argument("--dry-run", action="store_true", help="Preview the scaffold without writing files.")
    analysis_mvpa_init.add_argument("--force", action="store_true", help="Replace an existing scaffold YAML when writing.")
    analysis_mvpa_init_advanced = analysis_mvpa_init.add_argument_group("advanced compatibility overrides")
    analysis_mvpa_init_advanced.add_argument("--analysis-label", default=None, help="BIDS desc label for generated filenames.")
    analysis_mvpa_init_advanced.add_argument("--task", default=None, help="Optional task entity for compatibility scaffolds.")
    analysis_mvpa_init_advanced.add_argument("--session", default=None, help="Optional session entity for compatibility scaffolds.")
    analysis_mvpa_init_advanced.add_argument("--direction", default=None, help="Optional direction entity for compatibility scaffolds.")
    analysis_mvpa_init_advanced.add_argument("--metric", default="crossnobis", help="Distance metric override.")
    analysis_mvpa_init_advanced.add_argument(
        "--comparison-mode",
        default="explicit",
        choices=("explicit", "complete"),
        help="Condition-comparison scaffold mode.",
    )
    analysis_mvpa_init_advanced.add_argument(
        "--components",
        default=None,
        help="Comma-separated components: specs,runtime,tables,figures,rdms,derivatives.",
    )
    analysis_mvpa_init_advanced.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Condition spec as id[:label[:description[:source_selector]]]. Repeat for multiple conditions.",
    )
    analysis_mvpa_init_advanced.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="Comparison spec as id:condition_a:condition_b[:label[:description]]. Repeat for explicit comparisons.",
    )
    analysis_mvpa_init_advanced.add_argument(
        "--roi",
        action="append",
        default=[],
        help="ROI spec as id[:label[:source[:space[:mask_selector]]]]. Repeat for multiple ROIs.",
    )
    analysis_mvpa_init_advanced.add_argument("--analysis-variant", default="main", help="Analysis variant label for export scaffolds.")
    analysis_mvpa_init_advanced.add_argument("--phase-id", default="analysis", help="Phase/category label for export scaffolds.")
    analysis_mvpa_init.set_defaults(handler=_handle_analysis_mvpa_init)

    analysis_mvpa_list = analysis_mvpa_subparsers.add_parser("list", help="List MVPA set YAML files.")
    analysis_mvpa_list.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_list.set_defaults(handler=_handle_analysis_mvpa_list)

    analysis_mvpa_show = analysis_mvpa_subparsers.add_parser("show", help="Show an MVPA set as JSON.")
    analysis_mvpa_show.add_argument("name", help="MVPA set name without .yaml.")
    analysis_mvpa_show.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_show.set_defaults(handler=_handle_analysis_mvpa_show)

    analysis_mvpa_validate_help = "Validate an MVPA configuration schema without checking runtime readiness."
    analysis_mvpa_validate = analysis_mvpa_subparsers.add_parser(
        "validate",
        help=analysis_mvpa_validate_help,
        description=analysis_mvpa_validate_help,
    )
    analysis_mvpa_validate.add_argument("name", help="MVPA set name without .yaml.")
    analysis_mvpa_validate.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_validate.set_defaults(handler=_handle_analysis_mvpa_validate)

    analysis_mvpa_doctor_help = "Check MVPA adapter and input readiness without executing analysis."
    analysis_mvpa_doctor = analysis_mvpa_subparsers.add_parser(
        "doctor",
        help=analysis_mvpa_doctor_help,
        description=analysis_mvpa_doctor_help,
    )
    analysis_mvpa_doctor.add_argument("name", help="MVPA set name without .yaml.")
    analysis_mvpa_doctor.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_doctor.add_argument("--bundle", default=None, help="Analysis bundle supplying exact ordered units.")
    analysis_mvpa_doctor.set_defaults(handler=_handle_analysis_mvpa_doctor)

    analysis_mvpa_plan_help = "Resolve MVPA pattern sources and render a no-write planning preview."
    analysis_mvpa_plan = analysis_mvpa_subparsers.add_parser(
        "plan",
        help=analysis_mvpa_plan_help,
        description=analysis_mvpa_plan_help,
    )
    analysis_mvpa_plan.add_argument("name", help="MVPA set name without .yaml.")
    analysis_mvpa_plan.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_plan.add_argument("--bundle", default=None, help="Analysis bundle supplying exact ordered units.")
    analysis_mvpa_plan.set_defaults(handler=_handle_analysis_mvpa_plan)

    analysis_mvpa_smoke = analysis_mvpa_subparsers.add_parser(
        "smoke-manual-crossnobis",
        help="Compute one read-only manual diagonal crossnobis smoke/equivalence estimate.",
    )
    analysis_mvpa_smoke.add_argument("--subject", required=True)
    analysis_mvpa_smoke.add_argument("--roi-mask", required=True)
    analysis_mvpa_smoke.add_argument("--roi-label", default=None)
    analysis_mvpa_smoke.add_argument("--phase", default=None)
    analysis_mvpa_smoke.add_argument("--condition-a", required=True)
    analysis_mvpa_smoke.add_argument("--condition-b", required=True)
    analysis_mvpa_smoke.add_argument("--condition-a-alias", action="append", default=[])
    analysis_mvpa_smoke.add_argument("--condition-b-alias", action="append", default=[])
    analysis_mvpa_smoke.add_argument("--feat-run", action="append", required=True, help="Run FEAT directory as RUN=PATH.")
    analysis_mvpa_smoke.add_argument("--design-file", action="append", default=[], help="Optional design override as RUN=PATH.")
    analysis_mvpa_smoke.add_argument("--event-file", action="append", default=[], help="Event file as RUN:CONDITION:PATH.")
    analysis_mvpa_smoke.add_argument("--event-pattern", default=None)
    analysis_mvpa_smoke.add_argument("--min-events", type=int, default=1)
    analysis_mvpa_smoke.add_argument("--min-valid-voxels", type=int, default=1)
    analysis_mvpa_smoke.add_argument("--excluded-run", action="append", default=[])
    analysis_mvpa_smoke.add_argument("--reference-tsv", default=None)
    analysis_mvpa_smoke.add_argument("--reference-column", default="crossnobis")
    analysis_mvpa_smoke.add_argument("--reference-subject-column", default="subject_id")
    analysis_mvpa_smoke.add_argument("--reference-roi-column", default="roi_label")
    analysis_mvpa_smoke.add_argument("--reference-phase-column", default="phase")
    analysis_mvpa_smoke.add_argument("--tolerance", type=float, default=1e-8)
    analysis_mvpa_smoke.add_argument("--output", default=None)
    analysis_mvpa_smoke.set_defaults(handler=_handle_analysis_mvpa_smoke_manual_crossnobis)

    analysis_mvpa_run = analysis_mvpa_subparsers.add_parser("run", help="Preview or execute an MVPA runtime run.")
    analysis_mvpa_run.add_argument("name", help="MVPA set name without .yaml.")
    analysis_mvpa_run.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_run.add_argument("--bundle", default=None, help="Analysis bundle supplying exact ordered units.")
    analysis_mvpa_run.add_argument("--execute", action="store_true", help="Authorize local MVPA output creation after readiness checks.")
    analysis_mvpa_run.set_defaults(handler=_handle_analysis_mvpa_run)

    analysis_mvpa_export_tables = analysis_mvpa_subparsers.add_parser(
        "export-tables",
        help="Plan or execute MVPA prepared-distance table exports.",
    )
    analysis_mvpa_export_tables.add_argument("name", help="MVPA table export name without .yaml.")
    analysis_mvpa_export_tables.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_export_tables.add_argument("--execute", action="store_true", help="Write MVPA table export artifacts.")
    analysis_mvpa_export_tables.set_defaults(handler=_handle_analysis_mvpa_export_tables)

    analysis_mvpa_export_figures = analysis_mvpa_subparsers.add_parser(
        "export-figures",
        help="Plan or execute MVPA publication figure exports from subject-level tables.",
    )
    analysis_mvpa_export_figures.add_argument("name", help="MVPA figure export name without .yaml.")
    analysis_mvpa_export_figures.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_export_figures.add_argument("--execute", action="store_true", help="Write MVPA figure export artifacts.")
    analysis_mvpa_export_figures.set_defaults(handler=_handle_analysis_mvpa_export_figures)

    analysis_mvpa_export_rdms = analysis_mvpa_subparsers.add_parser(
        "export-rdms",
        help="Plan or execute MVPA RDM exports from subject-level tables.",
    )
    analysis_mvpa_export_rdms.add_argument("name", help="MVPA RDM export name without .yaml.")
    analysis_mvpa_export_rdms.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_export_rdms.add_argument("--execute", action="store_true", help="Write MVPA RDM export artifacts.")
    analysis_mvpa_export_rdms.set_defaults(handler=_handle_analysis_mvpa_export_rdms)

    analysis_mvpa_export_publication = analysis_mvpa_subparsers.add_parser(
        "export-publication",
        help="Plan or execute MVPA publication tables and figures from verified MVPA artifacts.",
    )
    analysis_mvpa_export_publication.add_argument("name", help="MVPA publication export name without .yaml.")
    analysis_mvpa_export_publication.add_argument(
        "--project",
        default=None,
        help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.",
    )
    analysis_mvpa_export_publication.add_argument(
        "--execute",
        action="store_true",
        help="Write MVPA publication table and figure artifacts.",
    )
    analysis_mvpa_export_publication.set_defaults(handler=_handle_analysis_mvpa_export_publication)

    analysis_mvpa_publish_derivatives = analysis_mvpa_subparsers.add_parser(
        "publish-derivatives",
        help="Plan or execute MVPA derivative publishing from verified artifacts.",
    )
    analysis_mvpa_publish_derivatives.add_argument("name", help="MVPA derivative publish config name without .yaml.")
    analysis_mvpa_publish_derivatives.add_argument(
        "--project",
        default=None,
        help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.",
    )
    analysis_mvpa_publish_derivatives.add_argument(
        "--target",
        default=None,
        help="Publish target from config. Defaults to local_artifact.",
    )
    analysis_mvpa_publish_derivatives.add_argument(
        "--execute",
        action="store_true",
        help="Write MVPA derivative files to the selected publish target.",
    )
    analysis_mvpa_publish_derivatives.set_defaults(handler=_handle_analysis_mvpa_publish_derivatives)

    analysis_mvpa_publish = analysis_mvpa_subparsers.add_parser("publish", help="Preview MVPA publication outputs without writing files.")
    analysis_mvpa_publish.add_argument("name", help="MVPA set name without .yaml.")
    analysis_mvpa_publish.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    analysis_mvpa_publish.set_defaults(handler=_handle_analysis_mvpa_publish)

    run_parser = subparsers.add_parser("run", help="Plan or execute the supported BIDS/tabular slices.")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)
    for mode, handler, help_text in (
        ("plan", _handle_run_plan, "Write a run manifest without execution."),
        ("local", _handle_run_local, "Write a run manifest and optionally execute locally."),
        ("slurm", _handle_run_slurm, "Write a run manifest and render a SLURM submission script."),
        ("submit", _handle_run_submit, "Plan a SLURM run and, with --execute, stage and submit it."),
    ):
        command = run_subparsers.add_parser(mode, help=help_text)
        action_subparsers = command.add_subparsers(dest="run_action", required=True)

        preprocess_parser = action_subparsers.add_parser("preprocess", help="Preprocess workflows.")
        preprocess_targets = preprocess_parser.add_subparsers(dest="run_target", required=True)
        bids_parser = preprocess_targets.add_parser("bids", help="Minimal BIDS preprocessing slice.")
        _add_run_common_args(bids_parser, mode=mode)
        _add_bids_run_filter_args(bids_parser)
        bids_parser.set_defaults(handler=handler)
        tabular_parser = preprocess_targets.add_parser("tabular", help="Minimal tabular preprocessing slice.")
        _add_run_common_args(tabular_parser, mode=mode)
        tabular_parser.set_defaults(handler=handler)

        train_parser = action_subparsers.add_parser("train", help="Train workflows.")
        train_targets = train_parser.add_subparsers(dest="run_target", required=True)
        train_model = train_targets.add_parser("model", help="Minimal logistic-regression model slice.")
        _add_run_common_args(train_model, mode=mode)
        train_model.set_defaults(handler=handler)

        analysis_parser = action_subparsers.add_parser("analysis", help="Analysis workflows.")
        analysis_targets = analysis_parser.add_subparsers(dest="run_target", required=True)
        analysis_bids = analysis_targets.add_parser("bids", help="Minimal BIDS analysis slice.")
        _add_run_common_args(analysis_bids, mode=mode)
        _add_bids_run_filter_args(analysis_bids)
        analysis_bids.add_argument("--stage", default=None, help="Analysis stage. Defaults to analysis.defaults.stage.")
        analysis_bids.add_argument("--model", default=None, help="Analysis model ref override.")
        analysis_bids.add_argument("--input-run", default=None, help="Upstream run id for future higher-level analysis stages.")
        analysis_bids.add_argument(
            "--output-desc",
            default=None,
            help="Optional BIDS-style desc label to append to generated analysis output names.",
        )
        analysis_bids.add_argument(
            "--empty-ev-policy",
            choices=("as_zero", "fail"),
            default=None,
            help="Override first-level FEAT empty EV handling for this run.",
        )
        analysis_bids.set_defaults(handler=handler)
        analysis_tabular = analysis_targets.add_parser("tabular", help="Generic tabular statistical analysis slice.")
        _add_run_common_args(analysis_tabular, mode=mode)
        analysis_tabular.add_argument("--analysis", required=True, help="Analysis spec name under config/analysis.")
        analysis_tabular.set_defaults(handler=handler)

        evaluate_parser = action_subparsers.add_parser("evaluate", help="Evaluation workflows.")
        evaluate_targets = evaluate_parser.add_subparsers(dest="run_target", required=True)
        evaluate_model = evaluate_targets.add_parser("model", help="Minimal logistic-regression evaluation slice.")
        _add_run_common_args(evaluate_model, mode=mode)
        evaluate_model.add_argument("--input-run", required=True, help="Existing train run id to evaluate.")
        evaluate_model.set_defaults(handler=handler)

    hpc_parser = subparsers.add_parser(
        "hpc",
        help="Inspect local HPC configuration and plans, or explicitly invoke experimental remote operations.",
    )
    hpc_subparsers = hpc_parser.add_subparsers(dest="hpc_command", required=True)
    for name, handler, help_text, description in (
        (
            "stage",
            _handle_hpc_stage,
            "Prepare a stage/sync plan for a planned run.",
            "Prepare a local stage/sync plan; remote changes require explicit execution.",
        ),
        (
            "bootstrap",
            _handle_hpc_bootstrap,
            "Prepare or explicitly execute an opt-in remote bootstrap plan.",
            "Prepare a local remote-bootstrap plan; remote changes require explicit execution.",
        ),
        (
            "status",
            _handle_hpc_status,
            "Read recorded local state, or use --live for an immediate SSH/squeue query.",
            "Read recorded local manifest/status state without a subprocess. With --live, immediately contact the "
            "configured host over SSH and run one squeue query. Live status does not query sacct, reconcile terminal "
            "accounting, or prove output completeness; an empty squeue result is reported ambiguously as "
            "not-found-or-completed.",
        ),
        (
            "pull",
            _handle_hpc_pull,
            "Plan retrieval, or use --execute for merge-oriented remote rsync.",
            "Prepare a local retrieval plan. With --execute, run merge-oriented rsync -az against the configured "
            "remote host; this does not prove scheduler success, atomically publish a complete result, attest "
            "digests, or guarantee interrupted-transfer recovery.",
        ),
        (
            "cancel",
            _handle_hpc_cancel,
            "Record a local cancellation request and render scancel; do not execute it.",
            "Record local cancel-requested state and render a possible scancel command when metadata permits. This "
            "command invokes no SSH or scheduler subprocess and does not confirm remote cancellation.",
        ),
    ):
        command = hpc_subparsers.add_parser(name, help=help_text, description=description)
        command.add_argument("--run-id", required=True, help="Run id created by rp run ...")
        if name == "status":
            command.add_argument(
                "--live",
                action="store_true",
                help=(
                    "Immediately contact the configured host over SSH and run one squeue query; does not query "
                    "sacct or prove a terminal outcome."
                ),
            )
            _add_hpc_profile_args(command)
        if name == "bootstrap":
            command.add_argument("--execute", action="store_true", help="Explicitly execute the bootstrap plan after writing it.")
        if name in {"stage", "pull"}:
            execute_help = (
                "Run merge-oriented remote rsync retrieval without scheduler-success, atomicity, digest, or "
                "recovery proof."
                if name == "pull"
                else "Explicitly execute the rendered sync command(s)."
            )
            command.add_argument("--execute", action="store_true", help=execute_help)
            if name == "pull":
                command.add_argument("--subpath", default=None, help="Optional relative directory under the remote run root.")
                command.add_argument("--destination", default=None, help="Optional local destination directory.")
            _add_hpc_profile_args(command)
        command.set_defaults(handler=handler)

    doctor = hpc_subparsers.add_parser(
        "doctor",
        help="Immediately check SSH connectivity and report project-aware HPC readiness.",
        description=(
            "Load the selected SSH profile and immediately check connectivity to the configured host. This may "
            "require host-key acceptance, authentication, or MFA; it is not a local-only validation and does not "
            "prove scheduler, runtime, storage, or data readiness."
        ),
    )
    _add_project_hpc_args(doctor)
    doctor.set_defaults(handler=_handle_hpc_doctor)

    connect = hpc_subparsers.add_parser("connect", help="Open a reusable SSH connection for HPC workflows.")
    _add_hpc_profile_args(connect)
    connect.add_argument(
        "--remote-command",
        default="true",
        help="Lightweight remote command used to warm the SSH connection. Defaults to true.",
    )
    connect.add_argument(
        "--test-reuse",
        action="store_true",
        help="After warming the connection, run a batch-mode hostname check to confirm reuse.",
    )
    connect.set_defaults(handler=_handle_hpc_connect)

    init = hpc_subparsers.add_parser(
        "init",
        help="Legacy Alliance-oriented helper that writes local defaults to secrets/.env.",
        description=(
            "Legacy/backward-compatible Alliance-oriented helper for writing local HPC defaults. New users should "
            "start with `rp hpc setup`; this command makes no connectivity or provider-readiness claim."
        ),
    )
    _add_hpc_init_args(init)
    init.set_defaults(handler=_handle_hpc_init)

    hpc_setup = hpc_subparsers.add_parser(
        "setup",
        help="Write a provider-neutral local HPC target starter under secrets/ without contacting a host.",
        description=(
            "Canonical beginner setup: write a provider-neutral generic target and SSH-profile starter under "
            "secrets/. This command makes no network call and does not test credentials, host reachability, "
            "scheduler behavior, runtimes, storage, or data readiness. Run `rp hpc validate` next; select the "
            "Alliance integration explicitly with --template alliance and review it for your site."
        ),
    )
    _add_hpc_setup_args(hpc_setup)
    hpc_setup.set_defaults(handler=_handle_hpc_setup)

    hpc_validate = hpc_subparsers.add_parser(
        "validate",
        help="Validate local HPC target and SSH configuration offline without subprocesses or writes.",
        description=(
            "Validate one local HPC target/profile configuration without subprocesses or writes, network "
            "contact, or remote-readiness claims. The promotion policy is checked as declared; remote promotion "
            "capability remains unverified."
        ),
    )
    _add_hpc_validate_args(hpc_validate)
    hpc_validate.set_defaults(handler=_handle_hpc_validate)

    cluster = hpc_subparsers.add_parser("cluster", help="Manage local-only HPC cluster defaults.")
    cluster.add_argument("cluster_args", nargs="*", help="Cluster command, for example list, show <name>, use <name>, or <name>.")
    cluster.add_argument("--project", default=None, help="Optional project overlay name for show output.")
    _add_hpc_target_config_arg(cluster)
    cluster.set_defaults(handler=_handle_hpc_cluster)

    hpc_use = hpc_subparsers.add_parser("use", help="Set the active local HPC cluster.")
    hpc_use.add_argument("name", help="Cluster name.")
    _add_hpc_target_config_arg(hpc_use)
    hpc_use.set_defaults(handler=_handle_hpc_target_use, hpc_target_label="cluster")

    target = hpc_subparsers.add_parser("target", help="Manage local-only HPC target defaults.")
    target_subparsers = target.add_subparsers(dest="hpc_target_command", required=True)
    target_list = target_subparsers.add_parser("list", help="List configured HPC targets.")
    _add_hpc_target_config_arg(target_list)
    target_list.set_defaults(handler=_handle_hpc_target_list)
    target_show = target_subparsers.add_parser("show", help="Show one configured HPC target.")
    target_show.add_argument("name", help="Target name.")
    target_show.add_argument("--project", default=None, help="Optional project overlay name for project-specific env.")
    _add_hpc_target_config_arg(target_show)
    target_show.set_defaults(handler=_handle_hpc_target_show)
    target_use = target_subparsers.add_parser("use", help="Set the active local HPC target in secrets/.env.")
    target_use.add_argument("name", help="Target name.")
    _add_hpc_target_config_arg(target_use)
    target_use.set_defaults(handler=_handle_hpc_target_use)

    sync_workspace = hpc_subparsers.add_parser("sync-workspace", help="Render or execute a first-time workspace bootstrap sync.")
    _add_hpc_sync_workspace_args(sync_workspace)
    sync_workspace.set_defaults(handler=_handle_hpc_sync_workspace)

    sync_project = hpc_subparsers.add_parser("sync-project", help="Render project/code sync commands for a selected project.")
    _add_hpc_sync_project_args(sync_project)
    sync_project.set_defaults(handler=_handle_hpc_sync_project)

    sync_data = hpc_subparsers.add_parser("sync-data", help="Render project data sync commands inferred from project config.")
    _add_hpc_sync_data_args(sync_data)
    sync_data.set_defaults(handler=_handle_hpc_sync_data)

    verify = hpc_subparsers.add_parser(
        "verify",
        help="Project-aware remote verification helpers that may contact the configured host.",
    )
    verify_subparsers = verify.add_subparsers(dest="hpc_verify_command", required=True)
    verify_data = verify_subparsers.add_parser(
        "data",
        help="Immediately inspect configured remote paths over SSH when verification paths exist.",
        description=(
            "When configured verification paths exist, immediately contact the selected host over SSH to inspect "
            "remote roots and expected files. This read-only remote check requires credentials and connectivity; "
            "it makes no remote data changes and has no live-cluster validation claim."
        ),
    )
    _add_hpc_verify_data_args(verify_data)
    verify_data.set_defaults(handler=_handle_hpc_verify_data)

    container = hpc_subparsers.add_parser("container", help="Project-aware container preparation helpers.")
    container_subparsers = container.add_subparsers(dest="hpc_container_command", required=True)
    container_prepare = container_subparsers.add_parser(
        "prepare",
        help="Render or execute a login-node pull for a project's configured Apptainer/Singularity image.",
    )
    _add_hpc_container_prepare_args(container_prepare)
    container_prepare.set_defaults(handler=_handle_hpc_container_prepare)

    prepare_container = hpc_subparsers.add_parser("prepare-container", help="Alias for `rp hpc container prepare`.")
    _add_hpc_container_prepare_args(prepare_container)
    prepare_container.set_defaults(handler=_handle_hpc_container_prepare)

    sync = hpc_subparsers.add_parser("sync", help="Beginner-friendly aliases for project-aware sync commands.")
    sync_subparsers = sync.add_subparsers(dest="hpc_sync_command", required=True)

    sync_workspace_alias = sync_subparsers.add_parser("workspace", help="Alias for sync-workspace.")
    _add_hpc_sync_workspace_args(sync_workspace_alias)
    sync_workspace_alias.set_defaults(handler=_handle_hpc_sync_workspace)

    sync_project_alias = sync_subparsers.add_parser("project", help="Alias for sync-project.")
    _add_hpc_sync_project_args(sync_project_alias)
    sync_project_alias.set_defaults(handler=_handle_hpc_sync_project)

    sync_data_alias = sync_subparsers.add_parser("data", help="Alias for sync-data.")
    _add_hpc_sync_data_args(sync_data_alias)
    sync_data_alias.set_defaults(handler=_handle_hpc_sync_data)

    tunnel = hpc_subparsers.add_parser("tunnel", help="Render or execute a generic SSH tunnel through the selected HPC login host.")
    _add_hpc_tunnel_args(tunnel)
    tunnel.set_defaults(handler=_handle_hpc_tunnel)

    notebook = hpc_subparsers.add_parser("notebook", help="Notebook-oriented HPC helpers.")
    notebook_subparsers = notebook.add_subparsers(dest="hpc_notebook_command", required=True)
    notebook_plan = notebook_subparsers.add_parser("plan", help="Print a first-time notebook access plan.")
    _add_hpc_notebook_plan_args(notebook_plan)
    notebook_plan.set_defaults(handler=_handle_hpc_notebook_plan)
    notebook_start = notebook_subparsers.add_parser("start", help="Print a compact notebook start flow and optional tunnel command.")
    _add_hpc_notebook_start_args(notebook_start)
    notebook_start.set_defaults(handler=_handle_hpc_notebook_start)
    notebook_submit = notebook_subparsers.add_parser(
        "submit",
        help="Plan or explicitly submit an unattended notebook execution job to SLURM.",
    )
    _add_hpc_notebook_submit_args(notebook_submit)
    notebook_submit.set_defaults(handler=_handle_hpc_notebook_submit)

    publish_back_parser = subparsers.add_parser("publish-back", help="Plan optional publish-back from run artifacts.")
    publish_back_subparsers = publish_back_parser.add_subparsers(dest="publish_back_command", required=True)
    publish_back_plan = publish_back_subparsers.add_parser("plan", help="Write a planning-only publish-back plan for a run.")
    publish_back_plan.add_argument("--run-id", required=True, help="Run id created by rp run ...")
    publish_back_plan.set_defaults(handler=_handle_publish_back_plan)

    return parser


def _add_run_common_args(parser: argparse.ArgumentParser, *, mode: str) -> None:
    parser.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    parser.add_argument("--batch", default=None, help="Batch name without .tsv.")
    parser.add_argument("--run-id", default=None, help="Explicit run id. Defaults to a timestamped id.")
    parser.add_argument("--task-id", default=None, help="Optional task label for task-scoped preprocessing runs.")
    parser.add_argument("--row-run-id", dest="row_run_id", default=None, help="Optional run label selector for BIDS preprocessing rows.")
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--dry-run", action="store_true", help="Mark the run manifest as dry-run and keep the command plan-only.")
    if mode == "local":
        authorization.add_argument("--execute", action="store_true", help="Execute the generated local workflow command.")
    elif mode == "submit":
        authorization.add_argument("--execute", action="store_true", help="Stage the planned run remotely and submit it to SLURM.")
    if mode in {"slurm", "submit"}:
        _add_hpc_profile_args(parser)
    if mode == "submit":
        parser.add_argument(
            "--discover",
            action="store_true",
            help="Discover adapter-selected batch rows before planning and submission.",
        )


def _add_bids_run_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subject-id",
        nargs="+",
        default=None,
        help="Optional BIDS subject selector(s), e.g. sub-001 or 001.",
    )


def _add_analysis_model_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=None, help="Project overlay name. Defaults to WORKSPACE.yaml projects.default.")
    parser.add_argument("--tool", default=None, help="Analysis tool name. Defaults to analysis.defaults.tool when configured.")


def _add_bids_selector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subject-id",
        nargs="+",
        default=None,
        help="Optional subject selector(s), e.g. sub-001 or 001.",
    )
    parser.add_argument("--session-id", default=None, help="Optional session selector.")
    parser.add_argument("--task-id", default=None, help="Optional task selector.")
    parser.add_argument("--run-id", dest="selector_run_id", default=None, help="Optional run selector.")


def _handle_setup(args: argparse.Namespace) -> int:
    _ = args
    root = workspace_root()
    lines = [
        "research-platform setup",
        f"Workspace: {root}",
        f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "",
        "Workspace checks",
    ]
    checks = [
        ("WORKSPACE.yaml", root / "WORKSPACE.yaml"),
        ("secrets/", root / "secrets"),
        ("artifacts/", root / "artifacts"),
        ("artifacts/runs/", root / "artifacts" / "runs"),
    ]
    for label, path in checks:
        lines.append(f"- {label}: {'present' if path.exists() else 'missing'}")

    package_names = sorted({package for workflow in WORKFLOWS for package in workflow.required_packages})
    lines.extend(["", "Editable package checks"])
    for package in package_names:
        module_name = _package_import_name(package)
        lines.append(f"- {package}: {'importable' if importlib.util.find_spec(module_name) is not None else 'not importable'}")

    lines.extend(["", *render_workflow_menu(), "", "Next command: rp onboard"])
    print("\n".join(lines))
    return 0


def _package_import_name(package: str) -> str:
    mapping = {
        "research-analysis": "research_platform.analysis",
        "research-bids": "research_platform.bids",
        "research-core": "research_platform.core",
        "research-hpc": "research_platform.hpc",
        "research-io": "research_platform.io",
        "research-ml": "research_platform.ml",
        "research-neuro": "research_platform.neuro",
    }
    return mapping.get(package, package.replace("-", "_"))


def _handle_onboard_menu(args: argparse.Namespace) -> int:
    _ = args
    print("\n".join([*render_workflow_menu(), "", "Run rp onboard <workflow> to start guided onboarding."]))
    return 0


def _handle_onboard_workflow(args: argparse.Namespace) -> int:
    workflow = get_workflow(args.onboard_workflow)
    if workflow.name == "preprocess":
        subtype = _prompt_choice("Preprocess subtype", workflow.subtypes, default="bids")
        if subtype != "bids":
            raise SystemExit(f"Unsupported preprocess subtype: {subtype}")
        tool = _prompt_choice("Preprocessing tool", registered_bids_tools(), default=registered_bids_tools()[0])
        adapter = load_registered_bids_tool_adapter(tool)
        return _handle_project_init_bids_preprocess(
            argparse.Namespace(
                project=_prompt_required("Project name"),
                study_root=_prompt_required("Local BIDS root"),
                derivative_root=_prompt_required("Local derivative root")
                if adapter.requires_input_derivative()
                else _prompt_optional("Local derivative root"),
                tool=tool,
                task_id=_prompt_optional("Default task id"),
                remote_study_root=_prompt_optional("Remote BIDS root"),
                remote_derivative_root=_prompt_optional("Remote derivative root"),
            )
        )
    if workflow.name == "analysis":
        subtype = _prompt_choice("Analysis subtype", workflow.subtypes, default="bids")
        if subtype == "bids":
            return _handle_project_init_bids_analysis(
                argparse.Namespace(
                    project=_prompt_required("Project name"),
                    study_root=_prompt_required("Local BIDS root"),
                    derivative_root=_prompt_required("Local derivative root"),
                    tool=_prompt_choice("Analysis tool", registered_bids_analysis_tools(), default=registered_bids_analysis_tools()[0]),
                    task_id=_prompt_optional("Default task id"),
                    remote_study_root=_prompt_optional("Remote BIDS root"),
                    remote_derivative_root=_prompt_optional("Remote derivative root"),
                )
            )
        if subtype == "tabular":
            return _handle_onboard_tabular_analysis()
        return _handle_onboard_custom(project_note=f"Guided analysis overlay ({subtype}).")
    if workflow.name == "tabular":
        return _handle_project_init_tabular_model(
            argparse.Namespace(
                project=_prompt_required("Project name"),
                dataset=_prompt_optional("Primary dataset name"),
                canonical_dataset=_prompt_optional("Canonical dataset name"),
                canonical_features_root=_prompt_optional("Canonical features root"),
                batch=_prompt_optional("Default batch name"),
            )
        )
    if workflow.name == "notebook":
        return _handle_onboard_custom(project_note="Guided notebook-first overlay.")
    if workflow.name == "custom":
        return _handle_onboard_custom(project_note="Guided custom overlay.")
    raise SystemExit(f"Unsupported workflow: {workflow.name}")


def _handle_onboard_tabular_analysis() -> int:
    project_name = _prompt_required("Project name")
    input_table = _prompt_required("Input table path")
    analysis_name = _prompt_required("Analysis name")
    kind = _prompt_choice(
        "Analysis kind",
        ("correlation", "linear_model", "anova", "mixed_effects", "summary_table"),
        default="correlation",
    )
    project_root_path = _create_minimal_project_overlay(project_name, notes=f"Scaffolded tabular analysis overlay for {analysis_name}.")
    analysis_dir = ensure_dir(project_root_path / "config" / "analysis")
    spec: dict[str, Any] = {"analysis": {"kind": kind, "input_table": input_table}}
    if kind == "correlation":
        spec["analysis"]["method"] = _prompt_choice("Correlation method", ("pearson", "spearman"), default="pearson")
        spec["analysis"]["x"] = _prompt_required("X column")
        spec["analysis"]["y"] = _prompt_required("Y column")
    elif kind == "summary_table":
        spec["analysis"]["columns"] = _prompt_required("Columns, comma-separated")
    else:
        spec["analysis"]["outcome"] = _prompt_required("Outcome column")
        spec["analysis"]["predictors"] = _prompt_required("Predictors/groups, comma-separated")
    analysis_path = write_yaml(analysis_dir / f"{_validate_simple_scaffold_name(analysis_name, label='analysis name')}.yaml", spec)
    print(
        "\n".join(
            [
                f"Initialized tabular analysis project: {project_name}",
                f"- {to_workspace_relative(project_root_path / 'project.yaml', workspace_root())}",
                f"- {to_workspace_relative(analysis_path, workspace_root())}",
                "",
                "Next commands",
                f"- rp hpc doctor --project {project_name}",
                f"- rp run plan analysis tabular --project {project_name} --analysis {analysis_name} --run-id {project_name}-{analysis_name}",
            ]
        )
    )
    return 0


def _handle_onboard_custom(*, project_note: str) -> int:
    project_name = _prompt_required("Project name")
    project_root_path = _create_minimal_project_overlay(project_name, notes=project_note)
    print(
        "\n".join(
            [
                f"Initialized custom project overlay: {project_name}",
                f"- {to_workspace_relative(project_root_path / 'project.yaml', workspace_root())}",
                "",
                "Next commands",
                f"- rp hpc doctor --project {project_name}",
            ]
        )
    )
    return 0


def _create_minimal_project_overlay(project_name: str, *, notes: str) -> Path:
    resolved_root = workspace_root()
    workspace = load_workspace_config(resolved_root)
    normalized = _validate_project_init_name(project_name)
    projects_root = project_path(resolved_root, workspace, "__project_root__").parent.resolve()
    project_root_path = (projects_root / normalized).resolve()
    try:
        project_root_path.relative_to(projects_root)
    except ValueError as exc:
        raise SystemExit(f"Project name must resolve within {projects_root}.") from exc
    if project_root_path.exists() and any(project_root_path.iterdir()):
        raise SystemExit(f"Project path already exists and is not empty: {project_root_path}")
    write_yaml(
        project_root_path / "project.yaml",
        {"name": normalized, "version": "0.1.0", "datasets": [], "notes": notes},
    )
    ensure_dir(project_root_path / "config")
    ensure_dir(project_root_path / "manifests")
    ensure_dir(project_root_path / "notebooks")
    ensure_dir(project_root_path / "reports")
    return project_root_path


def _prompt_required(label: str) -> str:
    value = input(f"{label}: ").strip()
    if not value:
        raise SystemExit(f"{label} is required.")
    return value


def _prompt_optional(label: str) -> str | None:
    value = input(f"{label} (optional): ").strip()
    return value or None


def _prompt_choice(label: str, choices: tuple[str, ...] | list[str], *, default: str) -> str:
    rendered = ", ".join(choices)
    value = input(f"{label} [{default}] ({rendered}): ").strip() or default
    if value not in choices:
        raise SystemExit(f"{label} must be one of: {rendered}")
    return value


def _handle_project_init_bids_preprocess(args: argparse.Namespace) -> int:
    payload = _scaffold_bids_preprocess_project(args)
    print("\n".join(_render_project_init_summary(payload)))
    return 0


def _handle_project_init_bids_analysis(args: argparse.Namespace) -> int:
    payload = _scaffold_bids_analysis_project(args)
    print("\n".join(_render_analysis_project_init_summary(payload)))
    return 0


def _handle_project_init_tabular_model(args: argparse.Namespace) -> int:
    payload = _scaffold_tabular_model_project(args)
    print("\n".join(_render_tabular_model_project_init_summary(payload)))
    return 0


def _handle_project_init_generic(args: argparse.Namespace) -> int:
    payload = _scaffold_tabular_model_project(
        argparse.Namespace(
            project=args.project,
            dataset=None,
            canonical_dataset=None,
            canonical_features_root=None,
            batch=None,
        )
    )
    lines = _render_tabular_model_project_init_summary(payload)
    lines[0] = f"Initialized project overlay: {payload['project_name']}"
    lines.insert(1, "Starter slice: tabular model")
    print("\n".join(lines))
    return 0


def _handle_config_validate(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    bundle, resolved_root = _load_bundle(project_name)
    errors = validate_project_bundle(bundle, root=resolved_root)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"valid": True, "project": project_name}, indent=2))
    return 0


def _handle_config_show(args: argparse.Namespace) -> int:
    bundle, resolved_root = _load_bundle(_resolve_project_name(args.project))
    print(json.dumps(summarize_bundle(bundle, root=resolved_root), indent=2))
    return 0


def _handle_config_paths(args: argparse.Namespace) -> int:
    bundle, resolved_root = _load_bundle(_resolve_project_name(args.project))
    resolved_paths = dict(summarize_bundle(bundle, root=resolved_root)["resolved_paths"])
    analysis_document = bundle.get("analysis") or {}
    analysis_config = (
        analysis_document.get("analysis", {})
        if isinstance(analysis_document, dict)
        else {}
    )
    external_roots, errors = resolve_analysis_external_input_root_declarations(
        analysis_config if isinstance(analysis_config, dict) else {},
        workspace_root=resolved_root,
        project_root=bundle["project_root"],
    )
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    if external_roots:
        resolved_paths["analysis_external_input_roots"] = {
            str(root["name"]): {
                "label": str(root["label"]),
                "local_root": str(root["path"]),
                "exists": bool(root["exists"]),
                "sync_enabled": bool(root["sync_enabled"]),
                **(
                    {"remote_root": str(root["remote_root"])}
                    if root.get("remote_root") is not None
                    else {}
                ),
            }
            for root in external_roots
        }
    print(json.dumps(resolved_paths, indent=2))
    return 0


def _handle_batch_list(args: argparse.Namespace) -> int:
    bundle, _ = _load_bundle(_resolve_project_name(args.project))
    project_root_path = Path(bundle["project_root"])
    batches_dir = project_root_path / "manifests" / "batches"
    names = sorted(path.stem for path in batches_dir.glob("*.tsv")) if batches_dir.exists() else []
    print(json.dumps({"project": bundle["project"]["name"], "batches": names}, indent=2))
    return 0


def _handle_batch_show(args: argparse.Namespace) -> int:
    bundle, _ = _load_bundle(_resolve_project_name(args.project))
    project_root_path = Path(bundle["project_root"])
    batch_name = args.batch or _default_project_batch_name(bundle)
    if batch_name is None:
        raise SystemExit(json.dumps({"error": "No default batch is configured for this project."}, indent=2))
    batch_path = project_root_path / "manifests" / "batches" / f"{batch_name}.tsv"
    rows = _read_tsv(batch_path) if batch_path.exists() else []
    print(
        json.dumps(
            {
                "name": batch_name,
                "path": str(batch_path),
                "row_count": len(rows),
                "columns": list(rows[0].keys()) if rows else [],
                "rows": rows,
                "selected_row": rows[0] if rows else {},
            },
            indent=2,
        )
    )
    return 0


def _handle_batch_discover_bids(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    context = _build_context(
        project_name,
        batch_name=args.batch,
        allow_missing_batch=True,
        require_default_batch=False,
    )
    if context["slice"] != "bids":
        raise SystemExit(json.dumps({"error": f"Project {context['project']['name']} does not use the BIDS slice."}, indent=2))

    adapter = load_bids_tool_adapter(context["preprocessing"])
    selectors = _bids_selector_values(args)
    rows = adapter.discover_batch_rows(
        derivative_root=str(context["input_derivative_root"]),
        selectors=_single_value_bids_selectors(selectors),
    )
    rows = _filter_bids_rows(rows, selectors)
    if not rows:
        raise SystemExit(json.dumps({"error": "No matching BIDS rows were discovered."}, indent=2))

    batch_name = args.batch or context["preprocessing"]["default_batch"]
    batch_path = context["project_root"] / "manifests" / "batches" / f"{batch_name}.tsv"
    _write_batch_manifest(batch_path, rows)
    print(
        json.dumps(
            {
                "project": context["project"]["name"],
                "batch": batch_name,
                "path": to_workspace_relative(batch_path, context["workspace_root"]),
                "row_count": len(rows),
            },
            indent=2,
        )
    )
    return 0


def _handle_batch_discover_analysis_bids(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    context = _build_analysis_context(
        project_name,
        batch_name=args.batch,
        stage_name=args.stage,
        allow_missing_batch=True,
        require_default_batch=False,
    )
    if context["slice"] != "bids":
        raise SystemExit(json.dumps({"error": f"Project {context['project']['name']} does not use the BIDS slice."}, indent=2))

    selectors = _bids_selector_values(args)
    rows = context["tool_adapter"].discover_batch_rows(
        derivative_root=str(context["input_derivative_root"]),
        selectors=_single_value_bids_selectors(selectors),
        context=context,
    )
    rows = _filter_bids_rows(rows, selectors)
    if not rows:
        raise SystemExit(json.dumps({"error": "No matching derivative-backed rows were discovered."}, indent=2))

    batch_name = args.batch or context["analysis_stage"]["default_batch"]
    batch_path = context["project_root"] / "manifests" / "batches" / f"{batch_name}.tsv"
    _write_batch_manifest(batch_path, rows)
    print(
        json.dumps(
            {
                "project": context["project"]["name"],
                "stage": context["analysis_stage_name"],
                "batch": batch_name,
                "path": to_workspace_relative(batch_path, context["workspace_root"]),
                "row_count": len(rows),
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_bundle_init(args: argparse.Namespace) -> int:
    context = _build_analysis_bundle_project_context(_resolve_project_name(args.project))
    bundle_name = _normalize_analysis_bundle_name(args.name)
    bundle_path = _analysis_bundle_config_path(context=context, name=bundle_name)
    document = _analysis_bundle_scaffold_document(bundle_name)
    yaml_content = dump_yaml(document)
    dry_run = bool(getattr(args, "dry_run", False))
    force = bool(getattr(args, "force", False))

    if bundle_path.exists() and not force and not dry_run:
        raise SystemExit(
            json.dumps(
                {
                    "error": (
                        f"Analysis bundle file already exists: {bundle_path.name}. "
                        "Use --force to overwrite the scaffold YAML."
                    )
                },
                indent=2,
            )
        )

    written = False
    if not dry_run:
        write_yaml(bundle_path, document)
        written = True

    payload = {
        "project": context["project_name"],
        "bundle": bundle_name,
        "path": to_workspace_relative(bundle_path, context["workspace_root"]),
        "written": written,
        "dry_run": dry_run,
        "next_steps": [
            "Review the generated YAML and select one batch or named cohort.",
            f"rp analysis bundle validate {bundle_name} --project {context['project_name']}",
            f"rp analysis bundle doctor {bundle_name} --project {context['project_name']}",
            f"rp analysis bundle plan {bundle_name} --project {context['project_name']}",
        ],
    }
    if dry_run:
        payload["yaml"] = yaml_content
    print(json.dumps(payload, indent=2))
    return 0


def _handle_analysis_bundle_list(args: argparse.Namespace) -> int:
    context = _build_analysis_bundle_project_context(_resolve_project_name(args.project))
    names = (
        sorted(path.stem for path in context["bundles_dir"].glob("*.yaml") if path.is_file())
        if context["bundles_dir"].exists()
        else []
    )
    print(
        json.dumps(
            {
                "project": context["project_name"],
                "config_dir": to_workspace_relative(context["bundles_dir"], context["workspace_root"]),
                "bundles": names,
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_bundle_show(args: argparse.Namespace) -> int:
    context, bundle_name, bundle_path, document = _analysis_bundle_cli_document(args)
    print(
        json.dumps(
            {
                "project": context["project_name"],
                "bundle": bundle_name,
                "path": to_workspace_relative(bundle_path, context["workspace_root"]),
                "document": document,
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_bundle_validate(args: argparse.Namespace) -> int:
    context, bundle_name, bundle_path, document = _analysis_bundle_cli_document(args)
    validate_document, _resolve_bundle = _analysis_bundle_functions()
    errors = list(validate_document(document))
    configured_name = _analysis_bundle_document_name(document)
    if configured_name is not None and configured_name != bundle_name:
        errors.append(
            f"analysis_bundle.name {configured_name!r} must match its configuration filename {bundle_name!r}."
        )
    payload = {
        "valid": not errors,
        "project": context["project_name"],
        "bundle": bundle_name,
        "path": to_workspace_relative(bundle_path, context["workspace_root"]),
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["valid"] else 1


def _handle_analysis_bundle_doctor(args: argparse.Namespace) -> int:
    context, bundle_name, bundle_path, document = _analysis_bundle_cli_document(args)
    resolution = _resolve_analysis_bundle_for_cli(
        context=context,
        bundle_name=bundle_name,
        document=document,
    )
    payload = {
        "project": context["project_name"],
        "bundle": bundle_name,
        "path": to_workspace_relative(bundle_path, context["workspace_root"]),
        "valid": resolution.valid,
        "ready_for_planning": resolution.ready_for_planning,
        "checks": [check.to_dict() for check in resolution.checks],
        "warnings": list(resolution.warnings),
        "errors": list(resolution.errors),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ready_for_planning"] else 1


def _handle_analysis_bundle_plan(args: argparse.Namespace) -> int:
    context, bundle_name, bundle_path, document = _analysis_bundle_cli_document(args)
    resolution = _resolve_analysis_bundle_for_cli(
        context=context,
        bundle_name=bundle_name,
        document=document,
    )
    payload = {
        "project": context["project_name"],
        "bundle": bundle_name,
        "path": to_workspace_relative(bundle_path, context["workspace_root"]),
        **resolution.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0 if resolution.ready_for_planning else 1


def _handle_analysis_model_init(args: argparse.Namespace) -> int:
    context = _build_analysis_model_project_context(_resolve_project_name(args.project))
    tool_name, _, adapter = _resolve_analysis_model_authoring_adapter(context=context, requested_tool=args.tool)
    model_name = _normalize_analysis_model_name(args.name)
    model_path = _analysis_model_path(context=context, model_name=model_name)
    _ensure_analysis_model_destination(path=model_path, force=bool(args.force))

    if args.interactive and args.template is not None:
        raise SystemExit(json.dumps({"error": "--interactive and --template cannot be combined."}, indent=2))

    options = _analysis_model_init_options(args)
    if args.interactive:
        if any(options.values()):
            raise SystemExit(
                json.dumps(
                    {
                        "error": (
                            "--interactive cannot be combined with explicit FEAT model flags. "
                            "Either use the wizard or pass the non-interactive flags."
                        )
                    },
                    indent=2,
                )
            )
        document = adapter.interactive_init_model_document(name=model_name, template=None)
    elif args.template is not None:
        if any(options.values()):
            raise SystemExit(
                json.dumps(
                    {
                        "error": (
                            "--template cannot be combined with explicit FEAT model flags. "
                            "Either write a template or provide the non-interactive flags."
                        )
                    },
                    indent=2,
                )
            )
        document = adapter.init_model_document(name=model_name, options={}, template=args.template)
    else:
        if not options["ev_order"] or not options["contrasts"]:
            raise SystemExit(
                json.dumps(
                    {
                        "error": (
                            "Incomplete FEAT model input. Provide --ev-order and at least one --contrast, "
                            "or re-run with --interactive or --template."
                        )
                    },
                    indent=2,
                )
            )
        document = adapter.init_model_document(name=model_name, options=options, template=None)
        errors = adapter.validate_model_document(model_name=model_name, document=document)
        if errors:
            raise SystemExit(json.dumps({"valid": False, "tool": tool_name, "model": model_name, "errors": errors}, indent=2))

    write_yaml(model_path, document)
    print(
        json.dumps(
            {
                "created": True,
                "project": context["project_name"],
                "tool": tool_name,
                "model": model_name,
                "path": to_workspace_relative(model_path, context["workspace_root"]),
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_model_copy(args: argparse.Namespace) -> int:
    context = _build_analysis_model_project_context(_resolve_project_name(args.project))
    tool_name, _, adapter = _resolve_analysis_model_authoring_adapter(context=context, requested_tool=args.tool)
    source_name = _normalize_analysis_model_name(args.source)
    dest_name = _normalize_analysis_model_name(args.dest)
    source_path = _analysis_model_path(context=context, model_name=source_name)
    dest_path = _analysis_model_path(context=context, model_name=dest_name)
    document = _load_analysis_model_document(source_path)
    _ensure_analysis_model_destination(path=dest_path, force=bool(args.force))
    updated = adapter.rename_model_document(new_name=dest_name, document=document)
    write_yaml(dest_path, updated)
    print(
        json.dumps(
            {
                "copied": True,
                "project": context["project_name"],
                "tool": tool_name,
                "source": source_name,
                "dest": dest_name,
                "path": to_workspace_relative(dest_path, context["workspace_root"]),
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_model_show(args: argparse.Namespace) -> int:
    context = _build_analysis_model_project_context(_resolve_project_name(args.project))
    _, _, adapter = _resolve_analysis_model_authoring_adapter(context=context, requested_tool=args.tool)
    model_name = _normalize_analysis_model_name(args.name)
    model_path = _analysis_model_path(context=context, model_name=model_name)
    if args.format == "yaml":
        print(model_path.read_text(encoding="utf-8"))
        return 0

    document = _load_analysis_model_document(model_path)
    print(adapter.summarize_model_document(model_name=model_name, document=document))
    return 0


def _handle_analysis_model_validate(args: argparse.Namespace) -> int:
    context = _build_analysis_model_project_context(_resolve_project_name(args.project))
    tool_name, _, adapter = _resolve_analysis_model_authoring_adapter(context=context, requested_tool=args.tool)
    if args.all and args.name:
        raise SystemExit(json.dumps({"error": "Use either a model name or --all, not both."}, indent=2))
    if not args.all and not args.name:
        raise SystemExit(json.dumps({"error": "Provide a model name or use --all."}, indent=2))

    if args.all:
        model_paths = sorted(context["models_dir"].glob("*.yaml"))
        if not model_paths:
            raise SystemExit(json.dumps({"error": "No model files were found under config/analysis/models."}, indent=2))
    else:
        model_name = _normalize_analysis_model_name(args.name)
        model_paths = [_analysis_model_path(context=context, model_name=model_name)]

    results: list[dict[str, Any]] = []
    for model_path in model_paths:
        document = _load_analysis_model_document(model_path)
        model_name = model_path.stem
        errors = adapter.validate_model_document(model_name=model_name, document=document)
        results.append(
            {
                "model": model_name,
                "path": to_workspace_relative(model_path, context["workspace_root"]),
                "valid": not errors,
                "errors": errors,
            }
        )

    valid = all(item["valid"] for item in results)
    print(json.dumps({"valid": valid, "project": context["project_name"], "tool": tool_name, "models": results}, indent=2))
    return 0 if valid else 1


def _handle_analysis_design_configure_first_level(args: argparse.Namespace) -> int:
    context = _build_analysis_model_project_context(_resolve_project_name(args.project))
    document = load_yaml(context["analysis_path"], resolve_env=False)
    if not isinstance(document, dict):
        raise SystemExit(json.dumps({"error": "config/analysis.yaml must contain a mapping."}, indent=2))
    analysis = document.get("analysis")
    if not isinstance(analysis, dict):
        raise SystemExit(json.dumps({"error": "config/analysis.yaml must contain a top-level analysis mapping."}, indent=2))

    stages = analysis.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise SystemExit(json.dumps({"error": "config/analysis.yaml analysis.stages must contain a mapping."}, indent=2))
    stage = stages.setdefault("first_level", {})
    if not isinstance(stage, dict):
        raise SystemExit(json.dumps({"error": "config/analysis.yaml analysis.stages.first_level must contain a mapping."}, indent=2))

    inputs = analysis.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise SystemExit(json.dumps({"error": "config/analysis.yaml analysis.inputs must contain a mapping."}, indent=2))

    changed: list[str] = []
    _configure_first_level_bold_inputs(args=args, inputs=inputs, changed=changed)
    _configure_first_level_external_root(
        args=args,
        analysis=analysis,
        local_arg="events_root",
        remote_arg="remote_events_root",
        root_name="evs",
        changed=changed,
    )
    _configure_first_level_external_root(
        args=args,
        analysis=analysis,
        local_arg="confounds_root",
        remote_arg="remote_confounds_root",
        root_name="feat_confounds",
        changed=changed,
    )
    _configure_first_level_evs_input(args=args, inputs=inputs, changed=changed)
    _configure_first_level_confounds_input(args=args, inputs=inputs, changed=changed)
    _configure_first_level_stage(args=args, stage=stage, changed=changed)

    if args.dry_run:
        print(dump_yaml(document))
        return 0

    write_yaml(context["analysis_path"], document)
    print(
        json.dumps(
            {
                "updated": True,
                "project": context["project_name"],
                "stage": "first_level",
                "path": to_workspace_relative(context["analysis_path"], context["workspace_root"]),
                "changed": changed,
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_roi_list(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    roi_sets = sorted(path.stem for path in context["roi_sets_dir"].glob("*.yaml")) if context["roi_sets_dir"].exists() else []
    extraction_sets = (
        sorted(path.stem for path in context["extraction_sets_dir"].glob("*.yaml"))
        if context["extraction_sets_dir"].exists()
        else []
    )
    transform_sets = (
        sorted(path.stem for path in context["transform_sets_dir"].glob("*.yaml"))
        if context["transform_sets_dir"].exists()
        else []
    )
    print(
        json.dumps(
            {
                "project": context["project_name"],
                "roi_sets": roi_sets,
                "transform_sets": transform_sets,
                "extraction_sets": extraction_sets,
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_roi_init(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    roi_name = _normalize_roi_config_name(args.name, label="ROI set")
    template = _normalize_roi_template_name(args.template, label="ROI scaffold template")
    roi_path = _analysis_roi_set_config_path(context=context, name=roi_name)
    scaffold = _roi_scaffold()
    try:
        document = scaffold.build_roi_set_document(
            roi_name,
            template,
            path_profile=getattr(args, "path_profile", "generic"),
            overrides=_analysis_roi_set_scaffold_overrides(args),
        )
        yaml_content = scaffold.render_yaml(document)
        validation_document = _parse_rendered_analysis_roi_scaffold(yaml_content, config_label="ROI set")
    except ValueError as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc

    errors = scaffold.validate_roi_set_scaffold(validation_document)
    valid = not errors
    if not valid:
        payload = {
            "valid": valid,
            "project": context["project_name"],
            "roi_set": roi_name,
            "path": to_workspace_relative(roi_path, context["workspace_root"]),
            "template": template,
            "written": False,
            "errors": errors,
            "next_steps": _analysis_roi_init_next_steps(context["project_name"], roi_name),
        }
        if bool(getattr(args, "dry_run", False)):
            payload["yaml"] = yaml_content
        print(json.dumps(payload, indent=2))
        return 1

    written = _write_analysis_roi_scaffold(
        roi_path,
        yaml_content,
        force=bool(getattr(args, "force", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        label="ROI set",
    )
    payload = {
        "valid": valid,
        "project": context["project_name"],
        "roi_set": roi_name,
        "path": to_workspace_relative(roi_path, context["workspace_root"]),
        "template": template,
        "written": written,
        "errors": errors,
        "next_steps": _analysis_roi_init_next_steps(context["project_name"], roi_name),
    }
    if bool(getattr(args, "dry_run", False)):
        payload["yaml"] = yaml_content
    print(json.dumps(payload, indent=2))
    return 0 if valid else 1


def _handle_analysis_roi_show(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    roi_name = _normalize_roi_config_name(args.name, label="ROI set")
    roi_path = _analysis_roi_set_path(context=context, name=roi_name)
    if args.format == "yaml":
        print(roi_path.read_text(encoding="utf-8"))
        return 0

    document = _load_analysis_roi_document(roi_path, config_label="ROI set")
    schema = _roi_schema()
    print(schema.summarize_roi_set_document(document))
    return 0


def _handle_analysis_roi_validate(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    roi_name = _normalize_roi_config_name(args.name, label="ROI set")
    roi_path = _analysis_roi_set_path(context=context, name=roi_name)
    raw_document = _load_analysis_roi_document(roi_path, config_label="ROI set", resolve_env=False)
    document = _load_analysis_roi_document(roi_path, config_label="ROI set")
    schema = _roi_schema()
    errors = schema.validate_roi_set_document(document, personal_path_document=raw_document)
    valid = not errors
    print(
        json.dumps(
            {
                "valid": valid,
                "project": context["project_name"],
                "roi_set": roi_name,
                "path": to_workspace_relative(roi_path, context["workspace_root"]),
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if valid else 1


def _handle_analysis_roi_doctor(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    roi_name = _normalize_roi_config_name(args.name, label="ROI set")
    roi_path = _analysis_roi_set_path(context=context, name=roi_name)
    raw_document = _load_analysis_roi_document(roi_path, config_label="ROI set", resolve_env=False)
    document = _load_analysis_roi_document(roi_path, config_label="ROI set")
    execution_context = _build_analysis_roi_execution_context(context)
    doctor = _roi_doctor()
    payload = _analysis_roi_doctor_status(
        doctor.doctor_roi_set(raw_document, context=execution_context, runtime_document=document)
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["ready_for_execution"] else 1


def _handle_analysis_roi_build(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    roi_name = _normalize_roi_config_name(args.name, label="ROI set")
    roi_path = _analysis_roi_set_path(context=context, name=roi_name)
    raw_document = _load_analysis_roi_document(roi_path, config_label="ROI set", resolve_env=False)
    document = _load_analysis_roi_document(roi_path, config_label="ROI set")
    schema = _roi_schema()
    errors = schema.validate_roi_set_document(document, personal_path_document=raw_document)
    if errors:
        raise SystemExit(json.dumps({"error": "; ".join(errors)}, indent=2))
    executor = _roi_executor()
    execution_context = _build_analysis_roi_execution_context(context)
    try:
        result = (
            executor.run_roi_build(document, context=execution_context, validate_personal_paths=False)
            if bool(getattr(args, "execute", False))
            else executor.plan_roi_build(document, context=execution_context, validate_personal_paths=False)
        )
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc

    payload = {
        "valid": True,
        "project": context["project_name"],
        "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "path": to_workspace_relative(roi_path, context["workspace_root"]),
        **result.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _handle_analysis_roi_transform_validate(args: argparse.Namespace) -> int:
    context, transform_name, transform_path, _document, raw_document = _analysis_roi_transform_inputs(args)
    errors = _unique_messages(_validate_analysis_roi_transform_document(raw_document))
    valid = not errors
    payload = {
        "valid": valid,
        "schema_valid": valid,
        "project": context["project_name"],
        "transform_set": transform_name,
        "path": to_workspace_relative(transform_path, context["workspace_root"]),
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    return 0 if valid else 1


def _handle_analysis_roi_transform_doctor(args: argparse.Namespace) -> int:
    context, transform_name, transform_path, document, raw_document = _analysis_roi_transform_inputs(args)
    schema_errors = _unique_messages(_validate_analysis_roi_transform_document(raw_document))
    schema_valid = not schema_errors
    roots = _analysis_roi_root_refs(context)
    if not schema_valid:
        payload = {
            "valid": False,
            "schema_valid": False,
            "ready_for_execution": False,
            "project": context["project_name"],
            "transform_set": transform_name,
            "mode": "doctor",
            "path": to_workspace_relative(transform_path, context["workspace_root"]),
            "root_refs": _json_path_mapping(roots),
            "checks": _analysis_roi_transform_doctor_checks(
                raw_document,
                roots=roots,
                schema_errors=schema_errors,
                ready_for_execution=False,
            ),
            "warnings": [],
            "errors": schema_errors,
        }
        print(json.dumps(payload, indent=2))
        return 1

    plan_transform, validate_execution, _execute_transform = _roi_transform_functions()
    try:
        plan = plan_transform(
            document,
            roots=roots,
            context=_analysis_roi_transform_plan_context(context, transform_path=transform_path),
        )
        plan_payload = plan.to_dict()
        validation_payload = validate_execution(plan).to_dict()
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        plan_payload = {"status": "error", "warnings": [], "errors": [str(exc)]}
        validation_payload = {"valid": False, "status": "error", "warnings": [], "errors": [str(exc)]}
    plan_errors = list(plan_payload.get("errors", []))
    readiness_errors = _unique_messages([*plan_errors, *list(validation_payload.get("errors", []))])
    ready_for_execution = schema_valid and bool(validation_payload.get("valid", False)) and not readiness_errors
    checks = _analysis_roi_transform_doctor_checks(
        raw_document,
        roots=roots,
        schema_errors=schema_errors,
        plan_payload=plan_payload,
        validation_payload=validation_payload,
        ready_for_execution=ready_for_execution,
    )
    payload = {
        "project": context["project_name"],
        "transform_set": transform_name,
        "mode": "doctor",
        "path": to_workspace_relative(transform_path, context["workspace_root"]),
        "root_refs": _json_path_mapping(roots),
        **plan_payload,
        "valid": schema_valid,
        "schema_valid": schema_valid,
        "ready_for_execution": ready_for_execution,
        "checks": checks,
        "validation": validation_payload,
        "errors": _unique_messages([*schema_errors, *readiness_errors]),
    }
    print(json.dumps(payload, indent=2))
    return 0 if ready_for_execution else 1


def _handle_analysis_roi_transform_plan(args: argparse.Namespace, *, mode: str = "plan") -> int:
    context, transform_name, transform_path, document, raw_document = _analysis_roi_transform_inputs(args)
    validation_errors = _unique_messages(_validate_analysis_roi_transform_document(raw_document))
    if validation_errors:
        print(
            json.dumps(
                {
                    "valid": False,
                    "project": context["project_name"],
                    "transform_set": transform_name,
                    "mode": mode,
                    "path": to_workspace_relative(transform_path, context["workspace_root"]),
                    "errors": validation_errors,
                },
                indent=2,
            )
        )
        return 1
    plan_transform, _validate_execution, _execute_transform = _roi_transform_functions()
    roots = _analysis_roi_root_refs(context)
    plan = plan_transform(
        document,
        roots=roots,
        context=_analysis_roi_transform_plan_context(context, transform_path=transform_path),
    )
    plan_payload = plan.to_dict()
    errors = _unique_messages([*validation_errors, *list(plan_payload.get("errors", []))])
    payload = {
        "valid": not errors and bool(plan_payload.get("valid", False)),
        "project": context["project_name"],
        "transform_set": transform_name,
        "mode": mode,
        "path": to_workspace_relative(transform_path, context["workspace_root"]),
        "root_refs": _json_path_mapping(roots),
        **plan_payload,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["valid"] else 1


def _handle_analysis_roi_transform_run(args: argparse.Namespace) -> int:
    context, transform_name, transform_path, document, raw_document = _analysis_roi_transform_inputs(args)
    validation_errors = _unique_messages(_validate_analysis_roi_transform_document(raw_document))
    if validation_errors:
        print(
            json.dumps(
                {
                    "valid": False,
                    "project": context["project_name"],
                    "transform_set": transform_name,
                    "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
                    "path": to_workspace_relative(transform_path, context["workspace_root"]),
                    "errors": validation_errors,
                },
                indent=2,
            )
        )
        return 1
    plan_transform, validate_execution, execute_transform = _roi_transform_functions()
    roots = _analysis_roi_root_refs(context)
    plan = plan_transform(
        document,
        roots=roots,
        context=_analysis_roi_transform_plan_context(context, transform_path=transform_path),
    )
    plan_payload = plan.to_dict()
    errors = _unique_messages([*validation_errors, *list(plan_payload.get("errors", []))])
    if errors or not bool(plan_payload.get("valid", False)):
        payload = {
            "valid": False,
            "project": context["project_name"],
            "transform_set": transform_name,
            "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
            "path": to_workspace_relative(transform_path, context["workspace_root"]),
            "root_refs": _json_path_mapping(roots),
            **plan_payload,
            "errors": errors,
        }
        print(json.dumps(payload, indent=2))
        return 1

    if not bool(getattr(args, "execute", False)):
        payload = {
            "valid": True,
            "project": context["project_name"],
            "transform_set": transform_name,
            "mode": "plan",
            "path": to_workspace_relative(transform_path, context["workspace_root"]),
            "root_refs": _json_path_mapping(roots),
            **plan_payload,
        }
        print(json.dumps(payload, indent=2))
        return 0

    validation = validate_execution(plan)
    if not validation.valid:
        payload = {
            "valid": False,
            "project": context["project_name"],
            "transform_set": transform_name,
            "mode": "execute",
            "path": to_workspace_relative(transform_path, context["workspace_root"]),
            "validation": validation.to_dict(),
            "errors": validation.errors,
        }
        print(json.dumps(payload, indent=2))
        return 1
    result = execute_transform(plan)
    payload = {
        "valid": result.valid,
        "project": context["project_name"],
        "transform_set": transform_name,
        "mode": "execute",
        "path": to_workspace_relative(transform_path, context["workspace_root"]),
        **result.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.valid else 1


def _handle_analysis_roi_extraction_validate(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    extraction_name = _normalize_roi_config_name(args.name, label="extraction set")
    extraction_path = _analysis_roi_extraction_set_path(context=context, name=extraction_name)
    raw_document = _load_analysis_roi_document(extraction_path, config_label="ROI extraction set", resolve_env=False)
    document = _load_analysis_roi_document(extraction_path, config_label="ROI extraction set")
    schema = _roi_schema()
    errors = schema.validate_extraction_set_document(document, personal_path_document=raw_document)
    valid = not errors
    print(
        json.dumps(
            {
                "valid": valid,
                "project": context["project_name"],
                "extraction_set": extraction_name,
                "path": to_workspace_relative(extraction_path, context["workspace_root"]),
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if valid else 1


def _handle_analysis_roi_extraction_doctor(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    extraction_name = _normalize_roi_config_name(args.name, label="extraction set")
    extraction_path = _analysis_roi_extraction_set_path(context=context, name=extraction_name)
    raw_extraction_document = _load_analysis_roi_document(
        extraction_path,
        config_label="ROI extraction set",
        resolve_env=False,
    )
    extraction_document = _load_analysis_roi_document(extraction_path, config_label="ROI extraction set")
    roi_document = None
    raw_roi_document = None
    schema = _roi_schema()
    extraction_errors = schema.validate_extraction_set_document(
        extraction_document,
        personal_path_document=raw_extraction_document,
    )
    if not extraction_errors:
        extraction_payload = raw_extraction_document.get("extraction_set", raw_extraction_document)
        raw_roi_set_ref = extraction_payload.get("roi_set") or extraction_payload.get("roi_set_ref")
        if raw_roi_set_ref is not None:
            roi_set_ref = _normalize_roi_config_name(str(raw_roi_set_ref), label="ROI set")
            roi_path = _analysis_roi_set_path(context=context, name=roi_set_ref)
            raw_roi_document = _load_analysis_roi_document(roi_path, config_label="ROI set", resolve_env=False)
            roi_document = _load_analysis_roi_document(roi_path, config_label="ROI set")
    execution_context = _build_analysis_roi_execution_context(context)
    doctor = _roi_doctor()
    payload = _analysis_roi_doctor_status(
        doctor.doctor_extraction_set(
            raw_extraction_document,
            roi_set_document=raw_roi_document,
            context=execution_context,
            runtime_document=extraction_document,
            runtime_roi_set_document=roi_document,
        )
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["ready_for_execution"] else 1


def _handle_analysis_roi_extraction_init(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    extraction_name = _normalize_roi_config_name(args.name, label="extraction set")
    roi_name = _normalize_roi_config_name(args.roi_set, label="ROI set")
    template = _normalize_roi_template_name(args.template, label="ROI extraction scaffold template")
    extraction_path = _analysis_roi_extraction_set_config_path(context=context, name=extraction_name)
    scaffold = _roi_scaffold()
    try:
        document = scaffold.build_extraction_set_document(
            extraction_name,
            roi_set=roi_name,
            template=template,
            path_profile=getattr(args, "path_profile", "generic"),
            overrides=_analysis_roi_extraction_scaffold_overrides(args),
        )
        document = _inherit_analysis_roi_publication_defaults(
            document,
            context=context,
            roi_name=roi_name,
            template=template,
        )
        yaml_content = scaffold.render_yaml(document)
        validation_document = _parse_rendered_analysis_roi_scaffold(yaml_content, config_label="ROI extraction set")
    except ValueError as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc

    errors = scaffold.validate_extraction_set_scaffold(validation_document)
    valid = not errors
    if not valid:
        payload = {
            "valid": valid,
            "project": context["project_name"],
            "extraction_set": extraction_name,
            "roi_set": roi_name,
            "path": to_workspace_relative(extraction_path, context["workspace_root"]),
            "template": template,
            "written": False,
            "errors": errors,
            "next_steps": _analysis_roi_extraction_init_next_steps(context["project_name"], extraction_name),
        }
        if bool(getattr(args, "dry_run", False)):
            payload["yaml"] = yaml_content
        print(json.dumps(payload, indent=2))
        return 1

    written = _write_analysis_roi_scaffold(
        extraction_path,
        yaml_content,
        force=bool(getattr(args, "force", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        label="ROI extraction set",
    )
    payload = {
        "valid": valid,
        "project": context["project_name"],
        "extraction_set": extraction_name,
        "roi_set": roi_name,
        "path": to_workspace_relative(extraction_path, context["workspace_root"]),
        "template": template,
        "written": written,
        "errors": errors,
        "next_steps": _analysis_roi_extraction_init_next_steps(context["project_name"], extraction_name),
    }
    if bool(getattr(args, "dry_run", False)):
        payload["yaml"] = yaml_content
    print(json.dumps(payload, indent=2))
    return 0 if valid else 1


def _handle_analysis_roi_extraction_run(args: argparse.Namespace) -> int:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    extraction_name = _normalize_roi_config_name(args.name, label="extraction set")
    extraction_path = _analysis_roi_extraction_set_path(context=context, name=extraction_name)
    raw_extraction_document = _load_analysis_roi_document(
        extraction_path,
        config_label="ROI extraction set",
        resolve_env=False,
    )
    extraction_document = _load_analysis_roi_document(extraction_path, config_label="ROI extraction set")
    schema = _roi_schema()
    errors = schema.validate_extraction_set_document(
        extraction_document,
        personal_path_document=raw_extraction_document,
    )
    if errors:
        print(
            json.dumps(
                {
                    "valid": False,
                    "project": context["project_name"],
                    "extraction_set": extraction_name,
                    "path": to_workspace_relative(extraction_path, context["workspace_root"]),
                    "errors": errors,
                },
                indent=2,
            )
        )
        return 1

    extraction_payload = raw_extraction_document.get("extraction_set", raw_extraction_document)
    raw_roi_set_ref = extraction_payload.get("roi_set") or extraction_payload.get("roi_set_ref")
    roi_document = None
    raw_roi_document = None
    if raw_roi_set_ref is not None:
        roi_set_ref = _normalize_roi_config_name(str(raw_roi_set_ref), label="ROI set")
        roi_path = _analysis_roi_set_path(context=context, name=roi_set_ref)
        raw_roi_document = _load_analysis_roi_document(roi_path, config_label="ROI set", resolve_env=False)
        roi_document = _load_analysis_roi_document(roi_path, config_label="ROI set")
        roi_errors = schema.validate_roi_set_document(roi_document, personal_path_document=raw_roi_document)
        if roi_errors:
            print(
                json.dumps(
                    {
                        "valid": False,
                        "project": context["project_name"],
                        "extraction_set": extraction_name,
                        "path": to_workspace_relative(extraction_path, context["workspace_root"]),
                        "errors": [f"referenced ROI set: {error}" for error in roi_errors],
                    },
                    indent=2,
                )
            )
            return 1
    executor = _roi_executor()
    execution_context = _build_analysis_roi_execution_context(context)
    try:
        result = (
            executor.run_roi_extraction(
                extraction_document,
                roi_set_document=roi_document,
                context=execution_context,
                validate_personal_paths=False,
            )
            if bool(getattr(args, "execute", False))
            else executor.plan_roi_extraction(
                extraction_document,
                roi_set_document=roi_document,
                context=execution_context,
                validate_personal_paths=False,
            )
        )
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc

    payload = {
        "valid": True,
        "project": context["project_name"],
        "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "path": to_workspace_relative(extraction_path, context["workspace_root"]),
        **result.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _handle_analysis_mvpa_list(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    mvpa_sets = (
        sorted(path.stem for path in context["mvpa_sets_dir"].glob("*.yaml") if path.is_file())
        if context["mvpa_sets_dir"].exists()
        else []
    )
    print(
        json.dumps(
            {
                "project": context["project_name"],
                "config_dir": to_workspace_relative(context["mvpa_sets_dir"], context["workspace_root"]),
                "mvpa_sets": mvpa_sets,
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_mvpa_init(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    analysis_id = _normalize_roi_config_name(args.analysis_id, label="MVPA analysis")
    scaffold = _mvpa_scaffold()
    config_root = Path(context["analysis_config_dir"])
    errors: list[str] = []
    warnings: list[str] = []
    if not config_root.exists():
        errors.append(
            f"Project analysis config root does not exist: {to_workspace_relative(config_root, context['workspace_root'])}."
        )
    try:
        plan = scaffold.build_mvpa_config_scaffold(
            analysis_id=analysis_id,
            analysis_label=getattr(args, "analysis_label", None),
            task=getattr(args, "task", None),
            session=getattr(args, "session", None),
            direction=getattr(args, "direction", None),
            template=getattr(args, "template", "materialized-crossnobis"),
            metric=getattr(args, "metric", "crossnobis"),
            comparison_mode=getattr(args, "comparison_mode", "explicit"),
            components=getattr(args, "components", None),
            condition_specs=getattr(args, "condition", None),
            comparison_specs=getattr(args, "comparison", None),
            roi_specs=getattr(args, "roi", None),
            analysis_variant=getattr(args, "analysis_variant", "main"),
            phase_id=getattr(args, "phase_id", "analysis"),
        )
    except ValueError as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc
    warnings.extend(plan.get("warnings", []))
    errors.extend(plan.get("errors", []))
    planned_files = _analysis_mvpa_scaffold_file_records(context, plan)
    validations = _analysis_mvpa_scaffold_validations(plan)
    for validation in validations:
        if validation["errors"]:
            errors.extend(f"{validation['relative_path']}: {error}" for error in validation["errors"])
    collisions = [record for record in planned_files if record["exists"]]
    dry_run = bool(getattr(args, "dry_run", False))
    execute = not dry_run
    force = bool(getattr(args, "force", False))
    if execute and collisions and not force:
        errors.append(
            "MVPA scaffold refuses to overwrite existing file(s): "
            + ", ".join(record["relative_path"] for record in collisions)
            + ". Use --force only when intentionally replacing the scaffold configuration."
        )

    written_files: list[dict[str, Any]] = []
    skipped_files: list[dict[str, Any]] = []
    if execute and not errors:
        for file_record in planned_files:
            _write_analysis_mvpa_scaffold_file(file_record)
            written_files.append({key: file_record[key] for key in ("component", "kind", "relative_path")})
    elif dry_run:
        skipped_files = [{key: record[key] for key in ("component", "kind", "relative_path")} for record in planned_files]

    runtime_relative_path = next(
        (
            record["relative_path"]
            for record in planned_files
            if record["component"] == "runtime"
        ),
        f"config/analysis/mvpa/{analysis_id}.yaml",
    )
    next_steps = [
        f"Review {runtime_relative_path}.",
        f"rp analysis mvpa validate {analysis_id} --project {context['project_name']}",
        f"rp analysis bundle list --project {context['project_name']}",
        (
            f"rp analysis mvpa doctor {analysis_id} --project {context['project_name']} "
            "--bundle <bundle>"
        ),
    ]

    payload = {
        "valid": not errors,
        "mode": "dry-run" if dry_run else "write",
        "executed": execute and not errors,
        "project": context["project_name"],
        "analysis_id": analysis_id,
        "analysis_label": plan.get("analysis_label"),
        "template": plan.get("template"),
        "metric": plan.get("metric"),
        "comparison_mode": plan.get("comparison_mode"),
        "components": plan.get("components", []),
        "planned_output_files": [
            {
                "component": record["component"],
                "kind": record["kind"],
                "relative_path": record["relative_path"],
                "exists": record["exists"],
                "collision": record["exists"],
            }
            for record in planned_files
        ],
        "existing_file_collisions": [record["relative_path"] for record in collisions],
        "written_files": written_files,
        "skipped_files": skipped_files,
        "dependencies": plan.get("dependencies", []),
        "validations": validations,
        "warnings": warnings,
        "errors": errors,
        "next_steps": next_steps,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["valid"] else 1


def _handle_analysis_mvpa_show(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    mvpa_name = _normalize_roi_config_name(args.name, label="MVPA set")
    mvpa_path = _analysis_mvpa_set_path(context=context, name=mvpa_name)
    document = _load_analysis_mvpa_document(mvpa_path, config_label="MVPA set", resolve_env=False)
    print(
        json.dumps(
            {
                "project": context["project_name"],
                "mvpa_set": mvpa_name,
                "path": to_workspace_relative(mvpa_path, context["workspace_root"]),
                "document": document,
            },
            indent=2,
        )
    )
    return 0


def _handle_analysis_mvpa_validate(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    mvpa_name = _normalize_roi_config_name(args.name, label="MVPA set")
    mvpa_path = _analysis_mvpa_set_path(context=context, name=mvpa_name)
    document = _load_analysis_mvpa_document(mvpa_path, config_label="MVPA set", resolve_env=False)
    validate_mvpa_set_document, _plan_mvpa_discovery = _mvpa_lifecycle_functions()
    errors = list(validate_mvpa_set_document(document))
    schema_valid = not errors
    print(
        json.dumps(
            {
                "valid": schema_valid,
                "schema_valid": schema_valid,
                "project": context["project_name"],
                "mvpa_set": mvpa_name,
                "path": to_workspace_relative(mvpa_path, context["workspace_root"]),
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if schema_valid else 1


def _handle_analysis_mvpa_doctor(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    mvpa_name = _normalize_roi_config_name(args.name, label="MVPA set")
    mvpa_path = _analysis_mvpa_set_path(context=context, name=mvpa_name)
    document, mvpa_config_sha256 = _load_analysis_mvpa_document_with_digest(
        mvpa_path,
        config_label="MVPA set",
    )
    state = _analysis_mvpa_lifecycle_state(
        context=context,
        mvpa_name=mvpa_name,
        mvpa_path=mvpa_path,
        document=document,
        bundle_name=getattr(args, "bundle", None),
        mvpa_config_sha256=mvpa_config_sha256,
    )
    payload = {
        "valid": state["ready_for_execution"],
        "schema_valid": state["schema_valid"],
        "bundle_valid": state["bundle_valid"],
        "plan_valid": state["plan_valid"],
        "ready_for_materialization": state["ready_for_materialization"],
        "ready_for_execution": state["ready_for_execution"],
        "executed": False,
        "project": context["project_name"],
        "mvpa_set": mvpa_name,
        "path": to_workspace_relative(mvpa_path, context["workspace_root"]),
        "plan_status": (
            state["plan_payload"].get("status") if state["plan_valid"] else "error"
        ),
        "discovery_status": state["plan_payload"].get("status"),
        "representation_kind": state["representation_kind"],
        "checks": state["checks"],
        "bundle": state["bundle_payload"],
        "runtime": state["runtime_payload"],
        "errors": state["errors"],
        "warnings": state["warnings"],
        "referenced_roi_sets": sorted(state["roi_sets"]),
        "missing_roi_sets": state["missing_roi_sets"],
        "root_refs": _json_path_mapping(state["roots"]),
        "adapter_availability": state["plan_payload"].get("adapter_availability", []),
        "backend_summary": state["plan_payload"].get("backend_summary", {}),
    }
    print(json.dumps(payload, indent=2))
    return 0 if state["ready_for_execution"] else 1


def _handle_analysis_mvpa_plan(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    mvpa_name = _normalize_roi_config_name(args.name, label="MVPA set")
    mvpa_path = _analysis_mvpa_set_path(context=context, name=mvpa_name)
    document, mvpa_config_sha256 = _load_analysis_mvpa_document_with_digest(
        mvpa_path,
        config_label="MVPA set",
    )
    state = _analysis_mvpa_lifecycle_state(
        context=context,
        mvpa_name=mvpa_name,
        mvpa_path=mvpa_path,
        document=document,
        bundle_name=getattr(args, "bundle", None),
        mvpa_config_sha256=mvpa_config_sha256,
    )
    plan_payload = state["plan_payload"]
    payload = {
        "project": context["project_name"],
        "path": to_workspace_relative(mvpa_path, context["workspace_root"]),
        "referenced_roi_sets": sorted(state["roi_sets"]),
        "missing_roi_sets": state["missing_roi_sets"],
        "root_refs": _json_path_mapping(state["roots"]),
        **plan_payload,
    }
    payload["valid"] = state["plan_valid"]
    payload["discovery_status"] = plan_payload.get("status")
    payload["status"] = plan_payload.get("status") if state["plan_valid"] else "error"
    payload["plan_status"] = payload["status"]
    payload["schema_valid"] = state["schema_valid"]
    payload["bundle_valid"] = state["bundle_valid"]
    payload["plan_valid"] = state["plan_valid"]
    payload["ready_for_materialization"] = state["ready_for_materialization"]
    payload["ready_for_execution"] = state["ready_for_execution"]
    payload["representation_kind"] = state["representation_kind"]
    payload["bundle"] = state["bundle_payload"]
    payload["runtime"] = state["runtime_payload"]
    payload["mvpa_config_sha256"] = state["mvpa_config_sha256"]
    payload["counts"] = state["plan_counts"]
    payload["cv_contract"] = state["cv_contract"]
    payload["checks"] = state["checks"]
    payload["adapter_availability"] = plan_payload.get("adapter_availability", [])
    payload["backend_summary"] = plan_payload.get("backend_summary", {})
    payload["warnings"] = state["warnings"]
    payload["errors"] = state["errors"]
    payload["executed"] = False
    print(json.dumps(payload, indent=2))
    return 0 if state["plan_valid"] else 1


def _handle_analysis_mvpa_smoke_manual_crossnobis(args: argparse.Namespace) -> int:
    try:
        from research_platform.neuro.mvpa.manual_smoke import compute_manual_crossnobis_smoke

        payload = compute_manual_crossnobis_smoke(
            subject=args.subject,
            roi_mask_path=args.roi_mask,
            roi_label=args.roi_label,
            phase=args.phase,
            feat_runs=_parse_key_path_specs(args.feat_run, label="--feat-run", separator="="),
            design_files=_parse_key_path_specs(args.design_file, label="--design-file", separator="="),
            event_files=_parse_event_file_specs(args.event_file),
            event_pattern=args.event_pattern,
            condition_a=args.condition_a,
            condition_b=args.condition_b,
            condition_a_aliases=tuple(args.condition_a_alias or ()),
            condition_b_aliases=tuple(args.condition_b_alias or ()),
            min_events=args.min_events,
            min_valid_voxels=args.min_valid_voxels,
            excluded_runs=tuple(args.excluded_run or ()),
            reference_tsv=args.reference_tsv,
            reference_column=args.reference_column,
            reference_subject_column=args.reference_subject_column,
            reference_roi_column=args.reference_roi_column,
            reference_phase_column=args.reference_phase_column,
            tolerance=args.tolerance,
            output_path=args.output,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "ok" else 1


def _parse_key_path_specs(values: list[str], *, label: str, separator: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        key, sep, path = str(value).partition(separator)
        if not sep or not key.strip() or not path.strip():
            raise ValueError(f"{label} values must use RUN{separator}PATH.")
        parsed[_normalize_run_selector(key)] = path.strip()
    return parsed


def _parse_event_file_specs(values: list[str]) -> dict[tuple[str, str], str]:
    parsed: dict[tuple[str, str], str] = {}
    for value in values or []:
        run_id, sep, remainder = str(value).partition(":")
        condition_id, sep2, path = remainder.partition(":")
        if not sep or not sep2 or not run_id.strip() or not condition_id.strip() or not path.strip():
            raise ValueError("--event-file values must use RUN:CONDITION:PATH.")
        parsed[(_normalize_run_selector(run_id), condition_id.strip())] = path.strip()
    return parsed


def _normalize_run_selector(value: object) -> str:
    text = str(value).strip()
    return text.removeprefix("run-")


def _handle_analysis_mvpa_run(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    mvpa_name = _normalize_roi_config_name(args.name, label="MVPA set")
    mvpa_path = _analysis_mvpa_set_path(context=context, name=mvpa_name)
    document, mvpa_config_sha256 = _load_analysis_mvpa_document_with_digest(
        mvpa_path,
        config_label="MVPA set",
    )
    state = _analysis_mvpa_lifecycle_state(
        context=context,
        mvpa_name=mvpa_name,
        mvpa_path=mvpa_path,
        document=document,
        bundle_name=getattr(args, "bundle", None),
        mvpa_config_sha256=mvpa_config_sha256,
    )
    if bool(getattr(args, "execute", False)):
        return _execute_analysis_mvpa_run(
            context=context,
            mvpa_name=mvpa_name,
            mvpa_path=mvpa_path,
            document=document,
            state=state,
        )
    payload = {
        "mode": "plan",
        "command": "analysis mvpa run",
        "executed": False,
        "project": context["project_name"],
        "mvpa_set": mvpa_name,
        "config_path": to_workspace_relative(mvpa_path, context["workspace_root"]),
        "valid": state["ready_for_execution"],
        "schema_valid": state["schema_valid"],
        "bundle_valid": state["bundle_valid"],
        "plan_valid": state["plan_valid"],
        "ready_for_materialization": state["ready_for_materialization"],
        "ready_for_execution": state["ready_for_execution"],
        "plan_status": (
            state["plan_payload"].get("status") if state["plan_valid"] else "error"
        ),
        "discovery_status": state["plan_payload"].get("status"),
        "representation_kind": state["representation_kind"],
        "bundle": state["bundle_payload"],
        "mvpa_config_sha256": state["mvpa_config_sha256"],
        "counts": state["plan_counts"],
        "cv_contract": state["cv_contract"],
        "runtime_root": state["runtime_root"],
        "runtime": state["runtime_payload"],
        "planned_outputs": _analysis_mvpa_runtime_output_previews(
            state["runtime_root"],
            representation_kind=state["representation_kind"],
        ),
        "planned_steps": _analysis_mvpa_run_planned_steps(),
        "checks": state["checks"],
        "referenced_roi_sets": sorted(state["roi_sets"]),
        "missing_roi_sets": state["missing_roi_sets"],
        "root_refs": _json_path_mapping(state["roots"]),
        "warnings": state["warnings"],
        "errors": state["errors"],
        "backend_summary": state["plan_payload"].get("backend_summary", {}),
    }
    print(json.dumps(payload, indent=2))
    return 0 if state["ready_for_execution"] else 1


def _handle_analysis_mvpa_export_tables(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    table_name = _normalize_roi_config_name(args.name, label="MVPA table export")
    table_path = _analysis_mvpa_table_export_path(context=context, name=table_name)
    document = _load_analysis_mvpa_document(table_path, config_label="MVPA table export", resolve_env=False)
    validate_export, plan_or_execute_export = _mvpa_table_export_functions()
    validation_errors = validate_export(document)
    roots = _analysis_mvpa_root_refs(context)
    if validation_errors:
        result = {
            "valid": False,
            "executed": False,
            "table_set": table_name,
            "outputs": {},
            "source_mvpa_sets": [],
            "sources": [],
            "row_counts": {},
            "table_a_columns": [],
            "audit_table_columns": [],
            "manifest": {},
            "warnings": [],
            "errors": validation_errors,
        }
    else:
        result = plan_or_execute_export(
            document,
            workspace_root=context["workspace_root"],
            root_refs=roots,
            execute=bool(getattr(args, "execute", False)),
        )
    payload = {
        "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "command": "analysis mvpa export-tables",
        "project": context["project_name"],
        "config_path": to_workspace_relative(table_path, context["workspace_root"]),
        "root_refs": _json_path_mapping(roots),
        **result,
    }
    print(json.dumps(payload, indent=2))
    return 0 if bool(payload.get("valid")) else 1


def _handle_analysis_mvpa_export_figures(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    figure_name = _normalize_roi_config_name(args.name, label="MVPA figure export")
    figure_path = _analysis_mvpa_figure_export_path(context=context, name=figure_name)
    document = _load_analysis_mvpa_document(figure_path, config_label="MVPA figure export", resolve_env=False)
    validate_export, plan_or_execute_export = _mvpa_figure_export_functions()
    validation_errors = validate_export(document)
    roots = _analysis_mvpa_root_refs(context)
    if validation_errors:
        result = {
            "valid": False,
            "executed": False,
            "figure_set": figure_name,
            "input_table": {},
            "output_root": {},
            "figures": [],
            "figure_count": 0,
            "supported_figure_kinds": [],
            "warnings": [],
            "errors": validation_errors,
        }
    else:
        result = plan_or_execute_export(
            document,
            workspace_root=context["workspace_root"],
            root_refs=roots,
            execute=bool(getattr(args, "execute", False)),
        )
    payload = {
        "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "command": "analysis mvpa export-figures",
        "project": context["project_name"],
        "config_path": to_workspace_relative(figure_path, context["workspace_root"]),
        "root_refs": _json_path_mapping(roots),
        **result,
    }
    print(json.dumps(payload, indent=2))
    return 0 if bool(payload.get("valid")) else 1


def _handle_analysis_mvpa_export_rdms(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    rdm_name = _normalize_roi_config_name(args.name, label="MVPA RDM export")
    rdm_path = _analysis_mvpa_rdm_export_path(context=context, name=rdm_name)
    document = _load_analysis_mvpa_document(rdm_path, config_label="MVPA RDM export", resolve_env=False)
    validate_export, plan_or_execute_export = _mvpa_rdm_export_functions()
    validation_errors = validate_export(document)
    roots = _analysis_mvpa_root_refs(context)
    if validation_errors:
        result = {
            "valid": False,
            "executed": False,
            "rdm_set": rdm_name,
            "input_table": {},
            "output_root": {},
            "rdms": [],
            "rdm_count": 0,
            "enabled_rdm_count": 0,
            "supported_rdm_kinds": [],
            "warnings": [],
            "errors": validation_errors,
        }
    else:
        result = plan_or_execute_export(
            document,
            workspace_root=context["workspace_root"],
            root_refs=roots,
            execute=bool(getattr(args, "execute", False)),
        )
    payload = {
        "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "command": "analysis mvpa export-rdms",
        "project": context["project_name"],
        "config_path": to_workspace_relative(rdm_path, context["workspace_root"]),
        "root_refs": _json_path_mapping(roots),
        **result,
    }
    print(json.dumps(payload, indent=2))
    return 0 if bool(payload.get("valid")) else 1


def _handle_analysis_mvpa_export_publication(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    publication_name = _normalize_roi_config_name(args.name, label="MVPA publication export")
    publication_path = _analysis_mvpa_publication_export_path(context=context, name=publication_name)
    document = _load_analysis_mvpa_document(publication_path, config_label="MVPA publication export", resolve_env=False)
    validate_export, plan_or_execute_export = _mvpa_publication_export_functions()
    validation_errors = validate_export(document)
    roots = _analysis_mvpa_root_refs(context)
    if validation_errors:
        result = {
            "valid": False,
            "executed": False,
            "publication_set": publication_name,
            "output_root": {},
            "input_tables": {},
            "table_families": {},
            "figures": [],
            "outputs": {},
            "warnings": [],
            "errors": validation_errors,
            "recomputed_mvpa_distances": False,
            "absolute_source_paths_excluded": True,
        }
    else:
        result = plan_or_execute_export(
            document,
            workspace_root=context["workspace_root"],
            root_refs=roots,
            execute=bool(getattr(args, "execute", False)),
        )
    payload = {
        "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "command": "analysis mvpa export-publication",
        "project": context["project_name"],
        "config_path": to_workspace_relative(publication_path, context["workspace_root"]),
        "root_refs": _json_path_mapping(roots),
        **result,
    }
    print(json.dumps(payload, indent=2))
    return 0 if bool(payload.get("valid")) else 1


def _handle_analysis_mvpa_publish_derivatives(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    publish_name = _normalize_roi_config_name(args.name, label="MVPA derivative publish")
    publish_path = _analysis_mvpa_derivative_publish_path(context=context, name=publish_name)
    document = _load_analysis_mvpa_document(publish_path, config_label="MVPA derivative publish", resolve_env=False)
    validate_publish, plan_or_execute_publish = _mvpa_derivative_publish_functions()
    validation_errors = validate_publish(document)
    roots = _analysis_mvpa_root_refs(context)
    if validation_errors:
        result = {
            "valid": False,
            "executed": False,
            "publish_set": publish_name,
            "analysis_label": publish_name,
            "derivative_name": "mvpa-crossnobis",
            "target": getattr(args, "target", None) or "local_artifact",
            "default_target": "local_artifact",
            "target_root": None,
            "outputs": {},
            "source_inputs": [],
            "table_sets": [],
            "rdm_set": None,
            "rdms": [],
            "row_counts": {},
            "warnings": [],
            "errors": validation_errors,
            "manifest": {},
        }
    else:
        result = plan_or_execute_publish(
            document,
            workspace_root=context["workspace_root"],
            root_refs=roots,
            target=getattr(args, "target", None),
            execute=bool(getattr(args, "execute", False)),
        )
    payload = {
        "mode": "execute" if bool(getattr(args, "execute", False)) else "plan",
        "command": "analysis mvpa publish-derivatives",
        "project": context["project_name"],
        "config_path": to_workspace_relative(publish_path, context["workspace_root"]),
        "root_refs": _json_path_mapping(roots),
        **result,
    }
    print(json.dumps(payload, indent=2))
    return 0 if bool(payload.get("valid")) else 1


def _handle_analysis_mvpa_publish(args: argparse.Namespace) -> int:
    context = _build_analysis_mvpa_project_context(_resolve_project_name(args.project))
    mvpa_name = _normalize_roi_config_name(args.name, label="MVPA set")
    mvpa_path = _analysis_mvpa_set_path(context=context, name=mvpa_name)
    document = _load_analysis_mvpa_document(mvpa_path, config_label="MVPA set", resolve_env=False)
    validate_mvpa_set_document, _plan_mvpa_discovery = _mvpa_lifecycle_functions()
    validation_errors = validate_mvpa_set_document(document)
    roots = _analysis_mvpa_root_refs(context)
    runtime_root, runtime_root_errors = _analysis_mvpa_runtime_root_preview(
        document,
        context=context,
        roots=roots,
        mvpa_name=mvpa_name,
    )
    publication_root, publication_root_errors = _analysis_mvpa_publication_root_preview(
        document,
        context=context,
        roots=roots,
        mvpa_name=mvpa_name,
    )
    table_desc, table_desc_warnings = _analysis_mvpa_publication_table_desc(document)
    required_runtime_inputs = _analysis_mvpa_runtime_input_previews(
        runtime_root,
        _MVPA_PUBLISH_REQUIRED_RUNTIME_INPUT_RELATIVE_PATHS,
        required=True,
    )
    optional_runtime_inputs = _analysis_mvpa_runtime_input_previews(
        runtime_root,
        _MVPA_PUBLISH_OPTIONAL_RUNTIME_INPUT_RELATIVE_PATHS,
        required=False,
    )
    required_input_errors = [
        f"Required Phase 4B.4 runtime input is missing: {record['relative_path']}"
        for record in required_runtime_inputs.values()
        if runtime_root is not None and not record["exists"]
    ]
    errors = _unique_messages([*validation_errors, *runtime_root_errors, *publication_root_errors, *required_input_errors])
    warnings = _unique_messages(table_desc_warnings)
    valid = not errors
    payload = {
        "mode": "plan",
        "command": "analysis mvpa publish",
        "executed": False,
        "project": context["project_name"],
        "mvpa_set": mvpa_name,
        "config_path": to_workspace_relative(mvpa_path, context["workspace_root"]),
        "valid": valid,
        "status": "ready" if valid else "error",
        "runtime_root": runtime_root,
        "publication_root": publication_root,
        "required_runtime_inputs": required_runtime_inputs,
        "optional_runtime_inputs": optional_runtime_inputs,
        "planned_outputs": _analysis_mvpa_publication_output_previews(publication_root, table_desc=table_desc),
        "planned_steps": _analysis_mvpa_publish_planned_steps(),
        "root_refs": _json_path_mapping(roots),
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    return 0 if valid else 1


def _execute_analysis_mvpa_run(
    *,
    context: dict[str, Any],
    mvpa_name: str,
    mvpa_path: Path,
    document: dict[str, Any],
    state: dict[str, Any],
) -> int:
    """Compute a complete MVPA run in memory, then publish it as one transaction."""

    plan = state["plan"]
    plan_payload = state["plan_payload"]
    runtime_root = state["runtime_root"]
    representation_kind = state["representation_kind"]
    transaction_plan = state["transaction_plan"]
    load_noise = _analysis_mvpa_load_noise_from_distance_rows(plan_payload)
    warnings = list(state["warnings"])

    def fail(
        stage: str,
        messages: list[str],
        *,
        extraction: dict[str, Any] | None = None,
        prepared: dict[str, Any] | None = None,
        distances: dict[str, Any] | None = None,
        summaries: dict[str, Any] | None = None,
        recovery_path: Path | None = None,
    ) -> int:
        payload = _analysis_mvpa_execute_payload(
            context=context,
            mvpa_name=mvpa_name,
            mvpa_path=mvpa_path,
            state=state,
            executed=False,
            valid=False,
            load_noise=load_noise,
            warnings=warnings,
            errors=_unique_messages(messages),
            failure_stage=stage,
            extraction=extraction,
            prepared=prepared,
            distances=distances,
            summaries=summaries,
            recovery_path=recovery_path,
        )
        print(json.dumps(payload, indent=2))
        return 1

    if (
        not state["ready_for_execution"]
        or runtime_root is None
        or representation_kind not in {"image", "prepared_features"}
        or transaction_plan is None
        or not transaction_plan.valid
    ):
        return fail(
            "readiness",
            [
                *state["errors"],
                "MVPA execution refused because the complete lifecycle is not ready for execution.",
            ],
        )

    distance_requests = _analysis_mvpa_distance_requests(plan_payload)
    if not distance_requests:
        return fail(
            "readiness",
            [*state["errors"], "MVPA discovery produced no distance computation rows."],
        )

    materialize_patterns, write_image_outputs = _mvpa_pattern_extraction_runtime_functions()
    try:
        extraction = materialize_patterns(plan, load_noise=load_noise)
    except BaseException as exc:  # transaction boundary also covers interruption
        return fail(
            "pattern_materialization",
            [*state["errors"], _analysis_mvpa_runtime_exception_message("MVPA pattern materialization", exc)],
        )

    extraction_summary = _analysis_mvpa_extraction_summary(extraction)
    if (
        _normalize_optional_cli_value(_analysis_mvpa_object_value(extraction, "representation_kind"))
        != representation_kind
    ):
        return fail(
            "pattern_materialization",
            ["MVPA pattern materialization returned a representation that differs from the exact plan."],
            extraction=extraction_summary,
        )
    extraction_errors = [
        *extraction_summary["errors"],
        *(
            ["MVPA pattern materialization produced no usable pattern rows."]
            if extraction_summary["usable_pattern_rows"] < 1
            else []
        ),
    ]
    if extraction_errors:
        return fail(
            "pattern_materialization",
            extraction_errors,
            extraction=extraction_summary,
        )

    (
        prepare_pattern_rows,
        compute_distances,
        summarize_distances,
        write_prepared_patterns,
        write_prepared_distances,
        write_prepared_summaries,
    ) = _mvpa_analysis_runtime_functions()
    try:
        preparation_options: dict[str, Any] = {
            "group_by": _analysis_mvpa_distance_group_by(document, plan_payload=plan_payload),
            "cv_unit": str(distance_requests[0]["cv_unit"]),
            "threshold_sweeps": _analysis_mvpa_threshold_sweeps(plan_payload),
        }
        if representation_kind == "prepared_features":
            preparation_options["cv_label_column"] = "cross_validation_label"
        prepared = prepare_pattern_rows(
            _analysis_mvpa_pattern_rows_for_preparation(extraction),
            **preparation_options,
        )
    except BaseException as exc:
        return fail(
            "row_preparation",
            [_analysis_mvpa_runtime_exception_message("MVPA row preparation", exc)],
            extraction=extraction_summary,
        )

    prepared_counts = _analysis_mvpa_prepared_counts(prepared)
    preparation_errors = list(prepared_counts["errors"])
    if (
        prepared_counts["group_count"] < 1
        or prepared_counts["prepared_row_count"] < 1
        or prepared_counts["usable_prepared_rows"] < 1
    ):
        preparation_errors.append("MVPA row preparation produced no usable prepared groups or rows.")
    if preparation_errors:
        return fail(
            "row_preparation",
            preparation_errors,
            extraction=extraction_summary,
            prepared=prepared_counts,
        )

    try:
        distance_results = [
            compute_distances(
                _analysis_mvpa_object_value(prepared, "groups", ()),
                metric=str(request["metric"]),
                engine_name=str(request["engine_name"]),
                noise_normalization_method=str(request["noise_normalization_method"]),
                noise_nonpositive_policy=str(request["noise_nonpositive_policy"]),
                min_retained_features=int(request["min_retained_features"]),
                warn_dropped_feature_fraction=float(request["warn_dropped_feature_fraction"]),
                condition_pairs=_analysis_mvpa_condition_pairs(plan_payload),
                threshold_sweeps=_analysis_mvpa_threshold_sweeps(plan_payload),
                preparation_qc_rows=_analysis_mvpa_object_value(prepared, "qc_rows", ()),
            )
            for request in distance_requests
        ]
        distances = _analysis_mvpa_combined_distance_result(distance_results, distance_requests=distance_requests)
    except BaseException as exc:
        return fail(
            "distance_computation",
            [_analysis_mvpa_runtime_exception_message("MVPA distance computation", exc)],
            extraction=extraction_summary,
            prepared=prepared_counts,
        )

    distance_counts = _analysis_mvpa_distance_counts(distances)
    distance_errors = list(distance_counts["errors"])
    if distance_counts["usable_distance_rows"] < 1:
        distance_errors.append("MVPA distance computation produced zero usable distance rows.")
    if distance_errors:
        return fail(
            "distance_computation",
            distance_errors,
            extraction=extraction_summary,
            prepared=prepared_counts,
            distances=distance_counts,
        )

    try:
        summaries = summarize_distances(
            distances,
            group_by=_analysis_mvpa_summary_group_by(document, plan_payload=plan_payload),
        )
    except BaseException as exc:
        return fail(
            "summary_computation",
            [_analysis_mvpa_runtime_exception_message("MVPA distance summary", exc)],
            extraction=extraction_summary,
            prepared=prepared_counts,
            distances=distance_counts,
        )

    summary_counts = _analysis_mvpa_summary_counts(summaries)
    summary_errors = list(summary_counts["errors"])
    if summary_counts["summary_row_count"] < 1:
        summary_errors.append("MVPA distance summary produced zero summary rows.")
    if summary_errors:
        return fail(
            "summary_computation",
            summary_errors,
            extraction=extraction_summary,
            prepared=prepared_counts,
            distances=distance_counts,
            summaries=summary_counts,
        )

    warnings = _unique_messages(
        [
            *warnings,
            *extraction_summary["warnings"],
            *prepared_counts["warnings"],
            *distance_counts["warnings"],
            *summary_counts["warnings"],
        ]
    )
    manifest_payload = _analysis_mvpa_success_manifest_payload(
        context=context,
        mvpa_name=mvpa_name,
        mvpa_path=mvpa_path,
        state=state,
        extraction=extraction,
        extraction_summary=extraction_summary,
        prepared_counts=prepared_counts,
        distance_counts=distance_counts,
        summary_counts=summary_counts,
        distance_requests=distance_requests,
        warnings=warnings,
    )

    def write_complete_runtime(staging_root: Path) -> dict[str, Any]:
        if representation_kind == "image":
            source_writer = write_image_outputs
            source_record = source_writer(
                extraction,
                output_root=staging_root,
                patterns_path="neuro/pattern-extraction/patterns.tsv",
                qc_path="neuro/pattern-extraction/qc.tsv",
                provenance_path="neuro/pattern-extraction/provenance.json",
                vector_metadata_path="neuro/pattern-extraction/vector_metadata.json",
                overwrite=False,
            )
            source_key = "neuro_pattern_extraction"
        else:
            source_writer = _mvpa_pattern_materialization_writer()
            source_record = source_writer(
                extraction,
                output_root=staging_root,
                patterns_path="neuro/pattern-materialization/patterns.tsv",
                qc_path="neuro/pattern-materialization/qc.tsv",
                provenance_path="neuro/pattern-materialization/provenance.json",
                vector_metadata_path="neuro/pattern-materialization/vector_metadata.json",
                overwrite=False,
            )
            source_key = "neuro_pattern_materialization"
        return {
            source_key: source_record,
            "analysis_prepared_patterns": write_prepared_patterns(
                prepared,
                output_root=staging_root,
                rows_path="analysis/prepared-patterns/rows.tsv",
                qc_path="analysis/prepared-patterns/qc.tsv",
                provenance_path="analysis/prepared-patterns/provenance.json",
                overwrite=False,
            ),
            "analysis_prepared_distances": write_prepared_distances(
                distances,
                output_root=staging_root,
                distances_path="analysis/prepared-distances/distances.tsv",
                qc_path="analysis/prepared-distances/qc.tsv",
                provenance_path="analysis/prepared-distances/provenance.json",
                overwrite=False,
            ),
            "analysis_prepared_summaries": write_prepared_summaries(
                summaries,
                output_root=staging_root,
                summaries_path="analysis/prepared-summaries/summaries.tsv",
                qc_path="analysis/prepared-summaries/qc.tsv",
                provenance_path="analysis/prepared-summaries/provenance.json",
                overwrite=False,
            ),
        }

    _plan_transaction, execute_transaction, _runtime_specs = _mvpa_runtime_transaction_functions()
    try:
        transaction_result = execute_transaction(
            transaction_plan,
            write_outputs=write_complete_runtime,
            manifest_payload=manifest_payload,
        )
    except BaseException as exc:
        return fail(
            "runtime_transaction",
            [_analysis_mvpa_runtime_exception_message("MVPA runtime transaction", exc)],
            extraction=extraction_summary,
            prepared=prepared_counts,
            distances=distance_counts,
            summaries=summary_counts,
            recovery_path=getattr(exc, "recovery_path", None),
        )

    payload = _analysis_mvpa_execute_payload(
        context=context,
        mvpa_name=mvpa_name,
        mvpa_path=mvpa_path,
        state=state,
        executed=True,
        valid=True,
        load_noise=load_noise,
        warnings=_unique_messages([*warnings, *transaction_result.warnings]),
        errors=[],
        extraction=extraction_summary,
        prepared=prepared_counts,
        distances=distance_counts,
        summaries=summary_counts,
        manifest=dict(transaction_result.manifest),
        output_sha256=dict(transaction_result.output_sha256),
        recovery_path=transaction_result.recovery_path,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _analysis_mvpa_execute_payload(
    *,
    context: dict[str, Any],
    mvpa_name: str,
    mvpa_path: Path,
    state: dict[str, Any],
    executed: bool,
    valid: bool,
    load_noise: bool,
    warnings: list[str],
    errors: list[str],
    failure_stage: str | None = None,
    extraction: dict[str, Any] | None = None,
    prepared: dict[str, Any] | None = None,
    distances: dict[str, Any] | None = None,
    summaries: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    output_sha256: dict[str, str] | None = None,
    recovery_path: Path | None = None,
) -> dict[str, Any]:
    plan_payload = state["plan_payload"]
    runtime_root = state["runtime_root"]
    representation_kind = state["representation_kind"]
    payload: dict[str, Any] = {
        "mode": "execute",
        "command": "analysis mvpa run",
        "executed": executed,
        "project": context["project_name"],
        "mvpa_set": mvpa_name,
        "config_path": to_workspace_relative(mvpa_path, context["workspace_root"]),
        "valid": valid,
        "schema_valid": state["schema_valid"],
        "bundle_valid": state["bundle_valid"],
        "plan_valid": state["plan_valid"],
        "ready_for_materialization": state["ready_for_materialization"],
        "ready_for_execution": state["ready_for_execution"],
        "plan_status": plan_payload.get("status") if valid else "error",
        "representation_kind": representation_kind,
        "bundle": state["bundle_payload"],
        "runtime_root": runtime_root,
        "runtime": state["runtime_payload"],
        "load_noise": load_noise,
        "planned_outputs": _analysis_mvpa_runtime_output_previews(
            runtime_root,
            representation_kind=representation_kind,
            executed=executed,
        ),
        "output_paths": _analysis_mvpa_runtime_output_previews(
            runtime_root,
            representation_kind=representation_kind,
            executed=executed,
        ),
        "referenced_roi_sets": sorted(state["roi_sets"]),
        "missing_roi_sets": state["missing_roi_sets"],
        "root_refs": _json_path_mapping(state["roots"]),
        "warnings": warnings,
        "errors": errors,
        "backend_summary": plan_payload.get("backend_summary", {}),
    }
    if failure_stage is not None:
        payload["failure_stage"] = failure_stage
    if extraction is not None:
        payload["extraction"] = extraction
    if prepared is not None:
        payload["prepared"] = prepared
    if distances is not None:
        payload["distances"] = distances
    if summaries is not None:
        payload["summaries"] = summaries
    payload["row_counts"] = _analysis_mvpa_runtime_row_counts(
        extraction=extraction,
        prepared=prepared,
        distances=distances,
        summaries=summaries,
    )
    if manifest is not None:
        payload["manifest"] = manifest
        payload["outputs"] = manifest.get("outputs", [])
    if output_sha256 is not None:
        payload["output_sha256"] = output_sha256
    if recovery_path is not None:
        payload["recovery_path"] = str(recovery_path)
    return payload


def _analysis_mvpa_runtime_exception_message(label: str, exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return f"{label} was interrupted; no final runtime output was published."
    message = str(exc).strip()
    return f"{label} failed: {message or type(exc).__name__}."


def _analysis_mvpa_runtime_row_counts(
    *,
    extraction: dict[str, Any] | None,
    prepared: dict[str, Any] | None,
    distances: dict[str, Any] | None,
    summaries: dict[str, Any] | None,
) -> dict[str, int]:
    return {
        "source_pattern_rows": int((extraction or {}).get("pattern_rows", 0)),
        "source_qc_rows": int((extraction or {}).get("qc_rows", 0)),
        "prepared_groups": int((prepared or {}).get("group_count", 0)),
        "prepared_pattern_rows": int((prepared or {}).get("prepared_row_count", 0)),
        "prepared_qc_rows": int((prepared or {}).get("qc_row_count", 0)),
        "distance_rows": int((distances or {}).get("distance_row_count", 0)),
        "distance_qc_rows": int((distances or {}).get("qc_row_count", 0)),
        "summary_rows": int((summaries or {}).get("summary_row_count", 0)),
        "summary_qc_rows": int((summaries or {}).get("qc_row_count", 0)),
    }


def _analysis_mvpa_success_manifest_payload(
    *,
    context: dict[str, Any],
    mvpa_name: str,
    mvpa_path: Path,
    state: dict[str, Any],
    extraction: Any,
    extraction_summary: dict[str, Any],
    prepared_counts: dict[str, Any],
    distance_counts: dict[str, Any],
    summary_counts: dict[str, Any],
    distance_requests: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    plan_payload = state["plan_payload"]
    bundle = state["bundle_payload"] or {}
    pattern_rows = tuple(
        _analysis_mvpa_mapping_like(row)
        for row in _analysis_mvpa_result_sequence(extraction, "pattern_rows")
    )
    selected_units = _analysis_mvpa_stable_unit_audits(
        plan_payload.get("analysis_units", []),
        key_columns=bundle.get("unit_key_columns", []),
    )
    excluded_units = _analysis_mvpa_stable_unit_audits(
        bundle.get("excluded_units", []),
        key_columns=bundle.get("unit_key_columns", []),
        include_reasons=True,
    )
    return {
        "project": context["project_name"],
        "mvpa_set": mvpa_name,
        # Bind successful provenance to the exact configuration bytes used by
        # lifecycle planning.  Re-reading here could describe a concurrent
        # edit that was never validated or executed.
        "mvpa_config_sha256": state["mvpa_config_sha256"],
        "bundle": {
            "name": bundle.get("name"),
            "digests": dict(bundle.get("digests", {})),
        },
        "source_tables": _analysis_mvpa_source_digest_records(
            plan_payload=plan_payload,
            extraction=extraction,
        ),
        "adapter": {
            "names": _unique_messages(
                [
                    str(row.get("backend_name"))
                    for row in plan_payload.get("pattern_rows", [])
                    if isinstance(row, dict) and row.get("backend_name")
                ]
            ),
            "representation_kind": state["representation_kind"],
        },
        "selection": {
            "selected_unit_count": len(selected_units),
            "selected_units": selected_units,
            "excluded_unit_count": len(excluded_units),
            "excluded_units": excluded_units,
        },
        "selected_unit_count": len(selected_units),
        "excluded_unit_count": len(excluded_units),
        "cross_validation": {
            "unit": distance_requests[0]["cv_unit"],
            "label_column": (
                "cross_validation_label"
                if state["representation_kind"] == "prepared_features"
                else _analysis_mvpa_default_cv_label_column(distance_requests[0]["cv_unit"])
            ),
            "labels": sorted(
                {
                    str(row.get("cross_validation_label"))
                    for row in pattern_rows
                    if isinstance(row, dict) and row.get("cross_validation_label") is not None
                }
            ),
        },
        "conditions": [
            str(row.get("id"))
            for row in plan_payload.get("conditions", [])
            if isinstance(row, dict) and row.get("id")
        ],
        "condition_pairs": _analysis_mvpa_condition_pairs(plan_payload) or (),
        "feature_identities": _analysis_mvpa_runtime_feature_identities(pattern_rows),
        "centering": _json_safe_cli_value(plan_payload.get("mean_centering", {})),
        "noise": _analysis_mvpa_runtime_noise_contract(
            pattern_rows=pattern_rows,
            distance_requests=distance_requests,
        ),
        "thresholds": {
            "event_thresholds": _json_safe_cli_value(plan_payload.get("event_thresholds")),
            "sweeps": _json_safe_cli_value(plan_payload.get("threshold_sweeps", [])),
        },
        "exclusions": _json_safe_cli_value(plan_payload.get("exclusions", [])),
        "distance_engine": _json_safe_cli_value(distance_requests),
        "row_counts": _analysis_mvpa_runtime_row_counts(
            extraction=extraction_summary,
            prepared=prepared_counts,
            distances=distance_counts,
            summaries=summary_counts,
        ),
        "warnings": warnings,
    }


def _analysis_mvpa_source_digest_records(
    *,
    plan_payload: dict[str, Any],
    extraction: Any,
) -> list[dict[str, Any]]:
    loaded_sources = _analysis_mvpa_object_value(extraction, "provenance", {})
    loaded_rows = loaded_sources.get("sources", []) if isinstance(loaded_sources, dict) else []
    loaded_by_name = {
        str(row.get("source_name")): row
        for row in loaded_rows
        if isinstance(row, dict) and row.get("source_name")
    }
    records: list[dict[str, Any]] = []
    for summary in plan_payload.get("pattern_source_summaries", []):
        if not isinstance(summary, dict):
            continue
        source_name = str(summary.get("source_name") or "")
        loaded = loaded_by_name.get(source_name, {})
        records.append(
            {
                "source_name": source_name,
                "backend": summary.get("backend"),
                "schema_version": summary.get("schema_version"),
                "portable_reference": summary.get("portable_reference"),
                "planned_sha256": summary.get("source_sha256"),
                "loaded_sha256": loaded.get("source_sha256"),
                "selected_rows": (summary.get("counts") or {}).get("selected_rows"),
                "unselected_rows": (summary.get("counts") or {}).get("unselected_rows"),
            }
        )
    return records


def _analysis_mvpa_stable_unit_audits(
    rows: Any,
    *,
    key_columns: Any,
    include_reasons: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(rows, (list, tuple)):
        return []
    keys = [str(value) for value in key_columns if str(value)] if isinstance(key_columns, (list, tuple)) else []
    audits: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        values = raw.get("values") if isinstance(raw.get("values"), dict) else raw
        record: dict[str, Any] = {}
        if raw.get("unit_id") is not None:
            record["unit_id"] = raw.get("unit_id")
        if raw.get("source_row") is not None:
            record["source_row"] = raw.get("source_row")
        for key in keys:
            if values.get(key) not in {None, ""}:
                record[key] = values.get(key)
        if include_reasons:
            for key in ("exclusion_ids", "exclusion_reasons", "reason_ids", "reasons"):
                raw_value = raw.get(key)
                values_value = values.get(key) if isinstance(values, dict) else None
                if raw_value not in (None, "", (), []):
                    record[key] = raw_value
                elif values_value not in (None, "", (), []):
                    record[key] = values_value
        audits.append(record)
    return audits


def _analysis_mvpa_runtime_feature_identities(pattern_rows: tuple[Any, ...]) -> list[dict[str, Any]]:
    fields = (
        "roi_source_name",
        "roi_label",
        "feature_space_id",
        "voxel_order",
        "voxel_index_hash",
        "roi_definition_id",
        "feature_count",
    )
    records: dict[str, dict[str, Any]] = {}
    for raw in pattern_rows:
        if not isinstance(raw, dict):
            continue
        record = {key: raw.get(key) for key in fields if raw.get(key) not in {None, ""}}
        if not record:
            continue
        records[json.dumps(record, sort_keys=True, separators=(",", ":"))] = record
    return [records[key] for key in sorted(records)]


def _analysis_mvpa_runtime_noise_contract(
    *,
    pattern_rows: tuple[Any, ...],
    distance_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    methods = _unique_messages([str(row["noise_normalization_method"]) for row in distance_requests])
    diagonal = "diagonal" in methods
    return {
        "methods": methods,
        "used": diagonal,
        "value_semantics": "variance" if diagonal else "unused",
        "sources": sorted(
            {
                str(row.get("noise_source"))
                for row in pattern_rows
                if isinstance(row, dict) and row.get("noise_source") is not None
            }
        ),
        "scopes": sorted(
            {
                str(row.get("noise_estimation_scope"))
                for row in pattern_rows
                if isinstance(row, dict) and row.get("noise_estimation_scope") is not None
            }
        ),
        "policies": _unique_messages([str(row["noise_nonpositive_policy"]) for row in distance_requests]),
    }


def _analysis_mvpa_default_cv_label_column(cv_unit: str) -> str:
    return {
        "run": "run_id",
        "session": "session_id",
        "subject": "subject_id",
    }.get(str(cv_unit), str(cv_unit))


def _mvpa_pattern_extraction_runtime_functions() -> tuple[Any, Any]:
    try:
        from research_platform.neuro.mvpa.runtime_execution import materialize_mvpa_patterns_from_plan
        from research_platform.neuro.mvpa.runtime_outputs import write_mvpa_pattern_extraction_outputs
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA execution requires research-neuro pattern runtime support: {exc}"}, indent=2)) from exc
    return materialize_mvpa_patterns_from_plan, write_mvpa_pattern_extraction_outputs


def _mvpa_pattern_materialization_writer() -> Any:
    try:
        from research_platform.neuro.mvpa.runtime_outputs import write_mvpa_pattern_materialization_outputs
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA execution requires research-neuro materialization outputs: {exc}"}, indent=2)) from exc
    return write_mvpa_pattern_materialization_outputs


def _mvpa_runtime_transaction_functions() -> tuple[Any, Any, Any]:
    try:
        from research_platform.neuro.mvpa.runtime_transaction import (
            execute_mvpa_runtime_transaction,
            plan_mvpa_runtime_transaction,
            runtime_output_specs,
        )
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA execution requires research-neuro runtime transactions: {exc}"}, indent=2)) from exc
    return plan_mvpa_runtime_transaction, execute_mvpa_runtime_transaction, runtime_output_specs


def _mvpa_analysis_runtime_functions() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from research_platform.analysis.mvpa.prepared_distances import compute_mvpa_distances_from_prepared_groups
        from research_platform.analysis.mvpa.prepared_summary import summarize_prepared_mvpa_distances
        from research_platform.analysis.mvpa.row_preparation import prepare_mvpa_pattern_row_groups
        from research_platform.analysis.mvpa.runtime_outputs import (
            write_prepared_mvpa_distance_outputs,
            write_prepared_mvpa_pattern_outputs,
            write_prepared_mvpa_summary_outputs,
        )
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA execution requires research-analysis MVPA APIs: {exc}"}, indent=2)) from exc
    return (
        prepare_mvpa_pattern_row_groups,
        compute_mvpa_distances_from_prepared_groups,
        summarize_prepared_mvpa_distances,
        write_prepared_mvpa_pattern_outputs,
        write_prepared_mvpa_distance_outputs,
        write_prepared_mvpa_summary_outputs,
    )


def _analysis_mvpa_load_noise_from_distance_rows(plan_payload: dict[str, Any]) -> bool:
    distances = plan_payload.get("distances")
    if not isinstance(distances, list):
        return False
    for row in distances:
        if not isinstance(row, dict):
            continue
        method = _normalize_optional_cli_value(row.get("noise_normalization_method"))
        if method == "diagonal":
            return True
    return False


def _analysis_mvpa_extraction_summary(extraction: Any) -> dict[str, Any]:
    pattern_rows = _analysis_mvpa_result_sequence(extraction, "pattern_rows")
    qc_rows = _analysis_mvpa_result_sequence(extraction, "qc_rows")
    warnings = _analysis_mvpa_result_messages(extraction, "warnings")
    errors = _analysis_mvpa_result_messages(extraction, "errors")
    return {
        "pattern_rows": len(pattern_rows),
        "usable_pattern_rows": sum(1 for row in pattern_rows if _analysis_mvpa_object_bool(row, "usable")),
        "qc_rows": len(qc_rows),
        "warnings": warnings,
        "errors": errors,
    }


def _analysis_mvpa_prepared_counts(prepared: Any) -> dict[str, Any]:
    groups = _analysis_mvpa_result_sequence(prepared, "groups")
    prepared_rows = tuple(row for group in groups for row in _analysis_mvpa_result_sequence(group, "rows"))
    qc_rows = _analysis_mvpa_result_sequence(prepared, "qc_rows")
    warnings = _analysis_mvpa_result_messages(prepared, "warnings")
    errors = _analysis_mvpa_result_messages(prepared, "errors")
    return {
        "group_count": len(groups),
        "prepared_row_count": len(prepared_rows),
        "usable_prepared_rows": sum(1 for row in prepared_rows if _analysis_mvpa_object_bool(row, "usable", default=True)),
        "qc_row_count": len(qc_rows),
        "warnings": warnings,
        "errors": errors,
    }


def _analysis_mvpa_distance_counts(distances: Any) -> dict[str, Any]:
    distance_rows = _analysis_mvpa_result_sequence(distances, "distances")
    qc_rows = _analysis_mvpa_result_sequence(distances, "qc_rows")
    warnings = _analysis_mvpa_result_messages(distances, "warnings")
    errors = _analysis_mvpa_result_messages(distances, "errors")
    return {
        "distance_row_count": len(distance_rows),
        "usable_distance_rows": sum(1 for row in distance_rows if _analysis_mvpa_object_bool(row, "usable", default=True)),
        "qc_row_count": len(qc_rows),
        "warnings": warnings,
        "errors": errors,
    }


def _analysis_mvpa_summary_counts(summary: Any) -> dict[str, Any]:
    summary_rows = _analysis_mvpa_result_sequence(summary, "summary_rows")
    qc_rows = _analysis_mvpa_result_sequence(summary, "qc_rows")
    warnings = _analysis_mvpa_result_messages(summary, "warnings")
    errors = _analysis_mvpa_result_messages(summary, "errors")
    return {
        "summary_row_count": len(summary_rows),
        "qc_row_count": len(qc_rows),
        "warnings": warnings,
        "errors": errors,
    }


def _analysis_mvpa_result_sequence(value: Any, key: str) -> tuple[Any, ...]:
    raw = _analysis_mvpa_object_value(value, key, ())
    if raw is None or isinstance(raw, (str, bytes)):
        return ()
    try:
        return tuple(raw)
    except TypeError:
        return ()


def _analysis_mvpa_result_messages(value: Any, key: str) -> list[str]:
    return _unique_messages(list(_analysis_mvpa_result_sequence(value, key)))


def _analysis_mvpa_object_bool(value: Any, key: str, *, default: bool = False) -> bool:
    return bool(_analysis_mvpa_object_value(value, key, default))


def _analysis_mvpa_object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    if hasattr(value, key):
        return getattr(value, key)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, dict):
            return mapped.get(key, default)
    return default


def _analysis_mvpa_pattern_rows_for_preparation(extraction: Any) -> tuple[Any, ...]:
    return tuple(
        _analysis_mvpa_flatten_grouping_values(_analysis_mvpa_mapping_like(row))
        for row in _analysis_mvpa_result_sequence(extraction, "pattern_rows")
    )


def _analysis_mvpa_mapping_like(value: Any) -> Any:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, dict):
            return mapped
    return value


def _analysis_mvpa_flatten_grouping_values(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    grouping_values = value.get("grouping_values")
    if isinstance(grouping_values, dict):
        for key, item in grouping_values.items():
            value.setdefault(str(key), item)
    return value


def _analysis_mvpa_distance_group_by(
    document: dict[str, Any],
    *,
    plan_payload: dict[str, Any] | None = None,
) -> tuple[str, ...] | None:
    if plan_payload is not None:
        plan_grouping = _analysis_mvpa_sequence_of_text(plan_payload.get("grouping_columns"))
        if plan_grouping:
            return plan_grouping
    payload = _analysis_mvpa_payload(document)
    distance = payload.get("distance") if isinstance(payload, dict) else None
    if not isinstance(distance, dict):
        return _analysis_mvpa_sequence_of_text(payload.get("grouping_columns")) if isinstance(payload, dict) else None
    raw_group_by = distance.get("grouping_columns")
    cross_validation = distance.get("cross_validation", distance.get("cv"))
    if raw_group_by is None and isinstance(cross_validation, dict):
        raw_group_by = cross_validation.get("grouping_columns") or cross_validation.get("group_by")
    if raw_group_by is None and isinstance(payload, dict):
        raw_group_by = payload.get("grouping_columns") or payload.get("group_by")
    return _analysis_mvpa_sequence_of_text(raw_group_by)


def _analysis_mvpa_condition_pairs(plan_payload: dict[str, Any]) -> tuple[dict[str, str], ...] | None:
    pairs = plan_payload.get("condition_pairs")
    if not isinstance(pairs, list) or not pairs:
        return None
    normalized: list[dict[str, str]] = []
    for row in pairs:
        if not isinstance(row, dict):
            continue
        pair_id = _normalize_optional_cli_value(row.get("id") or row.get("name"))
        left = _normalize_optional_cli_value(row.get("condition_id_a") or row.get("left") or row.get("condition_a"))
        right = _normalize_optional_cli_value(row.get("condition_id_b") or row.get("right") or row.get("condition_b"))
        if left is None or right is None:
            continue
        normalized.append({"id": pair_id or f"{left}__{right}", "left": left, "right": right})
    return tuple(normalized) or None


def _analysis_mvpa_threshold_sweeps(plan_payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    sweeps = plan_payload.get("threshold_sweeps")
    if not isinstance(sweeps, list):
        return ()
    normalized: list[dict[str, Any]] = []
    for row in sweeps:
        if not isinstance(row, dict):
            continue
        threshold_id = _normalize_optional_cli_value(row.get("id") or row.get("name"))
        normalized_row: dict[str, Any] = {"id": threshold_id or f"threshold-{len(normalized) + 1}"}
        if row.get("min_events") is not None:
            normalized_row["min_events"] = row.get("min_events")
        if row.get("min_observations") is not None:
            normalized_row["min_observations"] = row.get("min_observations")
        normalized.append(normalized_row)
    return tuple(normalized)


def _analysis_mvpa_summary_group_by(
    document: dict[str, Any],
    *,
    plan_payload: dict[str, Any],
) -> tuple[str, ...] | None:
    grouping_columns = _analysis_mvpa_distance_group_by(document, plan_payload=plan_payload) or ()
    condition_pairs = _analysis_mvpa_condition_pairs(plan_payload)
    if not grouping_columns and not condition_pairs:
        return None
    fields = [
        "group_id",
        *grouping_columns,
        "condition_id_a",
        "condition_id_b",
    ]
    if condition_pairs:
        fields.append("condition_pair_id")
    fields.extend(["metric", "engine_name", "normalization_method"])
    return tuple(_unique_messages(fields))


def _analysis_mvpa_sequence_of_text(value: Any) -> tuple[str, ...] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        values = tuple(str(column) for column in value)
    except TypeError:
        return None
    return tuple(column for column in values if column)


def _analysis_mvpa_distance_requests(plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    distances = plan_payload.get("distances")
    if not isinstance(distances, list):
        return []
    requests: list[dict[str, str]] = []
    for row in distances:
        if not isinstance(row, dict):
            continue
        requests.append(
            {
                "metric": _normalize_optional_cli_value(row.get("metric")) or "crossnobis",
                "engine_name": _normalize_optional_cli_value(row.get("engine") or row.get("engine_name")) or "native_reference",
                "cv_unit": _normalize_optional_cli_value(row.get("cv_unit")) or "run",
                "noise_normalization_method": _normalize_optional_cli_value(row.get("noise_normalization_method")) or "identity",
                "noise_nonpositive_policy": _normalize_optional_cli_value(
                    row.get("noise_nonpositive_policy") or row.get("nonpositive_noise_policy")
                )
                or "strict",
                "min_retained_features": _analysis_mvpa_positive_int(
                    row.get("min_retained_features"),
                    default=5,
                ),
                "warn_dropped_feature_fraction": _analysis_mvpa_fraction(
                    row.get("warn_dropped_feature_fraction"),
                    default=0.10,
                ),
            }
        )
    return requests


def _analysis_mvpa_positive_int(value: Any, *, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    if resolved < 1:
        return default
    return resolved


def _analysis_mvpa_fraction(value: Any, *, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    if resolved != resolved or resolved < 0.0 or resolved > 1.0:
        return default
    return resolved


def _analysis_mvpa_combined_distance_result(
    results: list[Any],
    *,
    distance_requests: list[dict[str, Any]],
) -> Any:
    if len(results) == 1:
        return results[0]
    return {
        "distances": [row for result in results for row in _analysis_mvpa_result_sequence(result, "distances")],
        "qc_rows": [row for result in results for row in _analysis_mvpa_result_sequence(result, "qc_rows")],
        "provenance": {
            "source": "research-core-cli",
            "distance_request_count": len(distance_requests),
            "distance_requests": distance_requests,
            "distance_result_provenance": [
                _analysis_mvpa_object_value(result, "provenance", ()) for result in results
            ],
        },
        "warnings": _unique_messages(
            [message for result in results for message in _analysis_mvpa_result_messages(result, "warnings")]
        ),
        "errors": _unique_messages(
            [message for result in results for message in _analysis_mvpa_result_messages(result, "errors")]
        ),
        "executed": all(_analysis_mvpa_object_bool(result, "executed", default=True) for result in results),
    }


def _analysis_mvpa_writer_record_summary(writer_records: dict[str, Any]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    overwrite_values: list[bool] = []
    for writer_name, record in writer_records.items():
        if isinstance(record, dict):
            overwrite_values.append(bool(record.get("overwrite", False)))
            for artifact in record.get("artifacts", []):
                if isinstance(artifact, dict):
                    artifacts.append({"writer": writer_name, **artifact})
    return {
        "overwrite": any(overwrite_values),
        "writer_records": writer_records,
        "artifacts": artifacts,
    }


def _handle_run_plan(args: argparse.Namespace) -> int:
    return _plan_run(args, mode="plan", execute=False)


def _handle_run_local(args: argparse.Namespace) -> int:
    return _plan_run(args, mode="local", execute=bool(getattr(args, "execute", False)))


def _handle_run_slurm(args: argparse.Namespace) -> int:
    return _plan_run(args, mode="slurm", execute=False)


def _handle_run_submit(args: argparse.Namespace) -> int:
    if args.run_action == "preprocess" and args.run_target == "bids":
        return _submit_bids_run(args)
    if args.run_action == "analysis" and args.run_target == "bids":
        return _submit_analysis_bids_run(args)
    if args.run_action == "analysis" and args.run_target == "tabular":
        return _submit_tabular_analysis_run(args)
    raise SystemExit(json.dumps({"error": "run submit currently supports preprocess bids, analysis bids, and analysis tabular."}, indent=2))


def _handle_hpc_stage(args: argparse.Namespace) -> int:
    context = _workspace_context()
    manifest, status, run_root_path = _load_run(context, args.run_id)
    try:
        plan = build_stage_plan(
            workspace_root=context["workspace_root"],
            run_root=run_root_path,
            manifest=manifest,
            status=status,
            exclude_file=context["workspace_root"] / "ops" / "sync" / "rsync" / "exclude.txt",
            profile_name=getattr(args, "profile", None),
            role=getattr(args, "role", None),
            config_path=getattr(args, "config", None),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    next_status = dict(plan["status"])
    report = dict(plan["report"])
    if getattr(args, "execute", False):
        execution = execute_stage_plan(report)
        report["execution"] = execution
        next_status["last_updated"] = _timestamp()
        next_status["state"] = "staged" if execution.get("ok") else "stage-failed"
    write_status(run_root_path, next_status)
    _record_local_files_written(
        report,
        run_root_path / "hpc" / "stage-plan.yaml",
        *(Path(path) for path in report.get("staged_files", [])),
        run_root_path / "status.yaml",
    )
    print(json.dumps(report, indent=2))
    return int(report.get("execution", {}).get("returncode", 0))


def _handle_hpc_status(args: argparse.Namespace) -> int:
    context = _workspace_context()
    manifest, status, _ = _load_run(context, args.run_id)
    job_id = status.get("job_id") or manifest.get("slurm", {}).get("job_id")
    payload = {
        "run_id": manifest["run_id"],
        "state": status.get("state"),
        "last_updated": status.get("last_updated"),
        "mode": status.get("mode") or manifest["execution"]["mode"],
        "job_id": job_id,
    }
    if getattr(args, "live", False):
        payload["scheduler"] = _query_live_scheduler_status(args, job_id=job_id)
    print(json.dumps(payload, indent=2))
    return 0


def _query_live_scheduler_status(args: argparse.Namespace, *, job_id: str | None) -> dict[str, Any]:
    if not job_id:
        return {"checked": False, "reason": "no job id recorded"}
    try:
        profile, _, role = _load_hpc_profile(args)
        mode = _hpc_profile_mode(role)
        remote_command = f"squeue -h -j {shlex.quote(str(job_id))} -o '%i\t%T\t%j\t%u\t%M\t%R'"
        command = build_ssh_command(profile, mode=mode, remote_command=remote_command, allocate_tty=False)
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - defensive
        return {"checked": False, "ok": False, "error": str(exc)}
    stdout = (result.stdout or "").strip()
    if result.returncode != 0:
        return {
            "checked": True,
            "ok": False,
            "command": command,
            "returncode": result.returncode,
            "stderr": (result.stderr or "").strip(),
        }
    if not stdout:
        return {"checked": True, "ok": True, "command": command, "state": "not-found-or-completed"}
    fields = stdout.splitlines()[0].split("\t")
    return {
        "checked": True,
        "ok": True,
        "command": command,
        "job_id": fields[0] if len(fields) > 0 else str(job_id),
        "state": fields[1] if len(fields) > 1 else "",
        "name": fields[2] if len(fields) > 2 else "",
        "user": fields[3] if len(fields) > 3 else "",
        "elapsed": fields[4] if len(fields) > 4 else "",
        "reason_or_node": fields[5] if len(fields) > 5 else "",
    }


def _handle_hpc_bootstrap(args: argparse.Namespace) -> int:
    context = _workspace_context()
    manifest, status, run_root_path = _load_run(context, args.run_id)
    plan = build_bootstrap_execution_plan(
        run_root=run_root_path,
        manifest=manifest,
        status=status,
        workspace_root=context["workspace_root"],
    )
    next_status = plan["status"]
    report = dict(plan["report"])
    if getattr(args, "execute", False):
        execution = execute_bootstrap_plan(report)
        report["execution"] = execution
        next_status = dict(next_status)
        next_status["last_updated"] = _timestamp()
        next_status["state"] = "bootstrap-complete" if execution.get("ok") else "bootstrap-failed"
    write_status(run_root_path, next_status)
    _record_local_files_written(
        report,
        run_root_path / "hpc" / "bootstrap-plan.yaml",
        run_root_path / "status.yaml",
    )
    print(json.dumps(report, indent=2))
    return int(report.get("execution", {}).get("returncode", 0))


def _handle_hpc_pull(args: argparse.Namespace) -> int:
    context = _workspace_context()
    manifest, status, run_root_path = _load_run(context, args.run_id)
    try:
        plan = build_pull_plan(
            workspace_root=context["workspace_root"],
            run_root=run_root_path,
            manifest=manifest,
            status=status,
            exclude_file=context["workspace_root"] / "ops" / "sync" / "rsync" / "exclude.txt",
            subpath=getattr(args, "subpath", None),
            destination=getattr(args, "destination", None),
            profile_name=getattr(args, "profile", None),
            role=getattr(args, "role", None),
            config_path=getattr(args, "config", None),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    next_status = dict(plan["status"])
    report = dict(plan["report"])
    if getattr(args, "execute", False):
        execution = execute_pull_plan(report)
        report["execution"] = execution
        next_status["last_updated"] = _timestamp()
        next_status["state"] = "pulled" if execution.get("ok") else "pull-failed"
    write_status(run_root_path, next_status)
    _record_local_files_written(
        report,
        run_root_path / "hpc" / "pull-plan.yaml",
        run_root_path / "status.yaml",
    )
    print(json.dumps(report, indent=2))
    return int(report.get("execution", {}).get("returncode", 0))


def _handle_hpc_cancel(args: argparse.Namespace) -> int:
    context = _workspace_context()
    manifest, status, run_root_path = _load_run(context, args.run_id)
    plan = build_cancel_plan(manifest=manifest, status=status)
    write_status(run_root_path, plan["status"])
    report = dict(plan["report"])
    _record_local_files_written(report, run_root_path / "status.yaml")
    print(json.dumps(report, indent=2))
    return 0


def _record_local_files_written(report: dict[str, Any], *paths: Path) -> None:
    recorded = {str(Path(path).resolve()) for path in report.get("local_files_written", [])}
    recorded.update(str(path.resolve()) for path in paths if path.is_file())
    report["local_files_written"] = sorted(recorded)


def _handle_hpc_init(args: argparse.Namespace) -> int:
    alliance_user = _resolve_hpc_init_value(
        value=args.alliance_user,
        label="Alliance username",
        flag="--alliance-user",
        default=os.environ.get("ALLIANCE_USER"),
    )
    remote_workspace_root = _validate_remote_workspace_root_for_init(
        _resolve_hpc_init_value(
            value=args.remote_workspace_root,
            label="Remote workspace root",
            flag="--remote-workspace-root",
            default=os.environ.get("RP_REMOTE_WORKSPACE_ROOT"),
            example="/home/<user>/research-platform",
        )
    )
    remote_artifacts_root_value = _resolve_hpc_init_value(
        value=args.remote_artifacts_root,
        label="Remote artifacts root",
        flag="--remote-artifacts-root",
        default=os.environ.get("RP_REMOTE_ARTIFACTS_ROOT"),
        optional=True,
        prompt_when_missing=False,
        example="/scratch/<user>/research-platform/artifacts",
    )
    remote_artifacts_root = (
        _validate_remote_artifacts_root_for_init(remote_artifacts_root_value)
        if remote_artifacts_root_value is not None
        else None
    )
    profile = _resolve_hpc_init_value(
        value=args.profile,
        label="SSH profile name",
        flag="--profile",
        default=None,
        example="interactive-login",
    )
    role = _resolve_hpc_init_value(
        value=args.role,
        label="SSH profile role",
        flag="--role",
        default=os.environ.get("RESEARCH_HPC_ROLE") or os.environ.get("RP_HPC_ROLE"),
        optional=True,
        prompt_when_missing=False,
        example="login",
    )
    ssh_config = _resolve_hpc_init_value(
        value=args.ssh_config,
        label="SSH profile config path",
        flag="--ssh-config",
        default=os.environ.get("RESEARCH_HPC_SSH_CONFIG") or os.environ.get("RP_SSH_CONFIG"),
        optional=True,
        prompt_when_missing=False,
        example="secrets/hpc/ssh-profiles.yaml",
    )

    env_path = write_local_hpc_env_defaults(
        {
            "ALLIANCE_USER": alliance_user,
            "RP_REMOTE_WORKSPACE_ROOT": remote_workspace_root,
            "RP_REMOTE_ARTIFACTS_ROOT": remote_artifacts_root,
            "RESEARCH_HPC_SSH_CONFIG": ssh_config,
            "RESEARCH_HPC_PROFILE": profile,
            "RESEARCH_HPC_ROLE": role,
        }
    )
    print(f"Wrote {to_workspace_relative(env_path, env_path.parent.parent)} with local HPC defaults.")
    print("")
    print("Next steps")
    print(f"- rp hpc sync workspace --profile {profile}")
    print(f"- rp hpc sync project --project <project> --profile {profile}")
    print("- Review both rendered plans before authorizing remote changes.")
    print(f"- rp hpc sync workspace --profile {profile} --execute")
    print(f"- rp hpc sync project --project <project> --profile {profile} --execute")
    return 0


def _handle_hpc_setup(args: argparse.Namespace) -> int:
    root = workspace_root()
    template = args.template or "generic"
    target = _normalize_optional_cli_value(getattr(args, "target", None)) or _normalize_optional_cli_value(
        getattr(args, "cluster", None)
    )
    target = target or _prompt_hpc_setup_required("HPC target name", flag="--target")
    profile = _normalize_optional_cli_value(args.profile) or target
    role = _normalize_optional_cli_value(args.role) or "login"
    user = (
        _normalize_optional_cli_value(getattr(args, "cluster_user", None))
        or _normalize_optional_cli_value(getattr(args, "alliance_user", None))
        or _prompt_hpc_setup_required("SSH username", flag="--user")
    )
    host = _normalize_optional_cli_value(getattr(args, "host", None)) or _prompt_hpc_setup_required(
        "SSH login host", flag="--host"
    )
    remote_workspace_root = _validate_remote_workspace_root_for_init(
        _normalize_optional_cli_value(args.remote_workspace_root)
        or _prompt_hpc_setup_required("Remote workspace root", flag="--remote-workspace-root")
    )
    remote_artifacts_root = _validate_remote_artifacts_root_for_init(
        _normalize_optional_cli_value(args.remote_artifacts_root)
        or str(PurePosixPath(remote_workspace_root) / "artifacts")
    )
    remote_container_root = _normalize_optional_cli_value(getattr(args, "remote_container_root", None))
    identity_file = _normalize_optional_cli_value(getattr(args, "identity_file", None))
    known_hosts_file = _normalize_optional_cli_value(getattr(args, "known_hosts_file", None))
    account = _normalize_optional_cli_value(getattr(args, "account", None))
    partition = _normalize_optional_cli_value(getattr(args, "partition", None))

    ssh_config_path = _resolve_hpc_setup_private_path(
        root=root,
        raw_path=getattr(args, "ssh_config", None) or "secrets/hpc/ssh-profiles.yaml",
        label="--ssh-config",
    )
    targets_path = _resolve_hpc_setup_private_path(
        root=root,
        raw_path=getattr(args, "targets_config", None) or "secrets/hpc/targets.yaml",
        label="--targets-config",
    )
    env_path = _resolve_hpc_setup_private_path(root=root, raw_path="secrets/.env", label="local defaults")
    for path in (ssh_config_path, targets_path, env_path):
        _ensure_hpc_setup_destination_safe(path=path, root=root)
    _reject_hpc_setup_destination_collisions(
        ssh_config_path=ssh_config_path,
        targets_path=targets_path,
        env_path=env_path,
    )

    existing_ssh_document = _load_hpc_setup_document(ssh_config_path, label="SSH profile config")
    existing_targets_document = _load_hpc_setup_document(targets_path, label="HPC targets config")
    try:
        if template == "generic":
            if role != "login":
                raise ValueError(
                    "The generic HPC setup supports the login role only; "
                    "multi-role automation remains provider-specific."
                )
            require_generic_profile_isolation(existing_ssh_document)
        elif template == "alliance" and role == "robot":
            raise ValueError(
                "High-level Alliance robot credentials are not yet modeled; "
                "use the login role or an explicitly authored private profile."
            )
        template_document = build_ssh_config_template(
            template,
            profile_name=profile,
            host=host,
            user=user,
            port=getattr(args, "port", None),
            identity_file=identity_file,
            known_hosts_file=known_hosts_file,
        )
        desired_profile = materialize_ssh_profile_entry(template_document, profile_name=profile)
        if template == "alliance" and identity_file is None:
            profile_defaults = desired_profile.get("defaults")
            if isinstance(profile_defaults, dict):
                profile_defaults.pop("identity_file", None)
        ssh_document = upsert_ssh_profile_document(
            existing_ssh_document,
            profile_name=profile,
            profile=desired_profile,
            force=bool(args.force),
        )
        targets_document = _upsert_hpc_setup_target_document(
            existing_targets_document,
            root=root,
            target=target,
            profile=profile,
            role=role,
            ssh_config_path=ssh_config_path,
            remote_workspace_root=remote_workspace_root,
            remote_artifacts_root=remote_artifacts_root,
            remote_container_root=remote_container_root,
            account=account,
            partition=partition,
            force=bool(args.force),
            set_default=bool(getattr(args, "set_default", False)),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    validation_report = validate_hpc_configuration(
        workspace_root=root,
        targets_config_path=targets_path,
        target_name=target,
        ssh_config_path=ssh_config_path,
        profile_name=profile,
        role=role,
        targets_document=targets_document,
        ssh_document=ssh_document,
        environment={},
    )
    if not validation_report["configuration_valid"]:
        errors = validation_report.get("errors", [])
        detail = errors[0] if errors else "proposed local configuration is invalid."
        raise SystemExit(f"HPC setup rejected before writing local configuration: {detail}")

    env_content = _build_hpc_setup_env_defaults(
        env_path=env_path,
        values={
            "RP_REMOTE_WORKSPACE_ROOT": remote_workspace_root,
            "RP_REMOTE_ARTIFACTS_ROOT": remote_artifacts_root,
            "RESEARCH_HPC_SSH_CONFIG": str(to_workspace_relative(ssh_config_path, root)),
            "RESEARCH_HPC_PROFILE": profile,
            "RESEARCH_HPC_ROLE": role,
            "RESEARCH_HPC_TARGET": target,
            "RESEARCH_HPC_TARGETS_CONFIG": str(to_workspace_relative(targets_path, root)),
        },
    )
    for path in (ssh_config_path, targets_path, env_path):
        _secure_hpc_setup_existing_destination(path)
    _write_hpc_setup_document(ssh_config_path, ssh_document)
    _write_hpc_setup_document(targets_path, targets_document)
    _ensure_hpc_setup_private_parent(env_path.parent)
    _write_private_text(env_path, env_content)
    lines = [
        "HPC setup complete",
        f"- SSH config: {to_workspace_relative(ssh_config_path, root)}",
        f"- Target config: {to_workspace_relative(targets_path, root)}",
        f"- Local defaults: {to_workspace_relative(env_path, root)}",
        f"- Target: {target}",
        f"- Profile: {profile}",
        f"- Role: {role}",
        f"- Template: {template}" + (" (optional provider integration; review for your site)" if template == "alliance" else ""),
    ]
    lines.extend(
        [
            "",
            "Next commands",
            f"- rp hpc validate --target {target}",
            "- rp hpc doctor --project <project>  # SSH-active connectivity check",
            (
                f"- research-hpc ssh check --profile {profile} --role {role} "
                f"--config {to_workspace_relative(ssh_config_path, root)} --mode auto  # SSH-active"
            ),
            "- Later runtime-readiness and remote-operation gates remain separate.",
        ]
    )
    print("\n".join(lines))
    return 0


def _handle_hpc_validate(args: argparse.Namespace) -> int:
    root = workspace_root()
    local_defaults_error: str | None = None
    managed_local_defaults: dict[str, str] = {}
    try:
        (
            offline_environment,
            managed_local_defaults,
        ) = resolve_local_hpc_environment_defaults(
            workspace_root=root,
            environment=dict(os.environ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        offline_environment = dict(os.environ)
        local_defaults_error = (
            f"Local HPC defaults could not be read safely from {root / 'secrets' / '.env'}: {exc}"
        )
    targets_config_value = (
        _normalize_optional_cli_value(getattr(args, "targets_config", None))
        or _normalize_optional_cli_value(offline_environment.get("RESEARCH_HPC_TARGETS_CONFIG"))
        or _normalize_optional_cli_value(offline_environment.get("RP_HPC_TARGETS_CONFIG"))
    )
    targets_path = Path(targets_config_value or "secrets/hpc/targets.yaml").expanduser()
    if not targets_path.is_absolute():
        targets_path = root / targets_path
    project_name = _normalize_optional_cli_value(getattr(args, "project", None))
    project_root_path: Path | None = None
    project_resolution_error: str | None = None
    if project_name is not None:
        try:
            workspace_document = load_workspace_config(root)
            project_root_path = project_path(root, workspace_document, project_name)
        except (OSError, TypeError, ValueError) as exc:
            project_resolution_error = (
                f"Project overlay {project_name!r} could not be resolved from WORKSPACE.yaml: {exc}"
            )

    report = validate_hpc_configuration(
        workspace_root=root,
        targets_config_path=targets_path,
        target_name=getattr(args, "target", None),
        ssh_config_path=getattr(args, "ssh_config", None),
        profile_name=getattr(args, "profile", None),
        role=getattr(args, "role", None),
        project_name=project_name,
        project_root=project_root_path,
        environment=offline_environment,
        managed_environment_defaults=managed_local_defaults,
    )
    if project_resolution_error is not None:
        report["configuration_valid"] = False
        report["errors"].append(project_resolution_error)
    if local_defaults_error is not None:
        report["configuration_valid"] = False
        report["errors"].append(local_defaults_error)
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
    else:
        print(render_hpc_validation_report(report))
    return 0 if report["configuration_valid"] else 1


def _upsert_hpc_setup_target_document(
    document: dict[str, Any],
    *,
    root: Path,
    target: str,
    profile: str,
    role: str,
    ssh_config_path: Path,
    remote_workspace_root: str,
    remote_artifacts_root: str,
    remote_container_root: str | None,
    account: str | None,
    partition: str | None,
    force: bool,
    set_default: bool,
) -> dict[str, Any]:
    updated = deepcopy(document)
    if not updated:
        updated = {"version": 1, "targets": {}}
    elif "version" not in updated:
        updated["version"] = 1
    if not isinstance(updated, dict):
        raise ValueError("HPC targets config must contain a top-level mapping.")
    targets = updated.get("targets")
    if targets is None:
        targets = {}
        updated["targets"] = targets
    if not isinstance(targets, dict):
        raise ValueError("HPC targets config targets must contain a mapping.")
    entry: dict[str, Any] = {
        "ssh_profile": profile,
        "role": role,
        "ssh_config": str(to_workspace_relative(ssh_config_path, root)),
        "env": {
            "RP_REMOTE_WORKSPACE_ROOT": remote_workspace_root,
            "RP_REMOTE_ARTIFACTS_ROOT": remote_artifacts_root,
        },
        "promotion": {"mode": "atomic_no_replace"},
    }
    if remote_container_root is not None:
        entry["env"]["RP_REMOTE_CONTAINER_ROOT"] = remote_container_root
    slurm: dict[str, str] = {}
    if account is not None:
        slurm["account"] = account
    if partition is not None:
        slurm["partition"] = partition
    if slurm:
        entry["slurm"] = slurm
    existing = targets.get(target)
    if existing is not None and existing != entry and not force:
        raise ValueError(
            f"HPC target {target!r} already exists with different settings; "
            "use --force to replace only that target."
        )
    targets[target] = entry
    if set_default or _normalize_optional_cli_value(updated.get("default")) is None:
        updated["default"] = target
    return updated


def _load_hpc_setup_document(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return {}
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} destination must be a real regular file, not a symlink or special file: {path}")
    try:
        document = load_yaml(path, resolve_env=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{label} could not be read: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit(f"{label} must contain a top-level mapping.")
    return document


def _resolve_hpc_setup_private_path(*, root: Path, raw_path: str, label: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if any(ord(character) < 32 or ord(character) == 127 for character in str(candidate)):
        raise SystemExit(f"rp hpc setup requires {label} to contain no control characters.")
    lexical = candidate if candidate.is_absolute() else root / candidate
    secrets_root = root / "secrets"
    try:
        relative = lexical.relative_to(secrets_root)
    except ValueError as exc:
        raise SystemExit(f"rp hpc setup writes local-only config and requires {label} under secrets/.") from exc
    if any(component in {"", ".", ".."} for component in relative.parts):
        raise SystemExit(f"rp hpc setup requires {label} to resolve directly beneath secrets/.")
    if _is_public_secrets_scaffold_path(relative):
        raise SystemExit(
            f"rp hpc setup refuses the tracked public secrets scaffold destination for {label}: "
            f"{lexical}"
        )
    return lexical


def _is_public_secrets_scaffold_path(relative: Path) -> bool:
    name = relative.name.casefold()
    return name in {"readme.md", ".gitkeep"} or name.endswith(".example")


def _ensure_hpc_setup_destination_safe(*, path: Path, root: Path) -> None:
    secrets_root = root / "secrets"
    if secrets_root.is_symlink():
        raise SystemExit(f"HPC setup refuses a symlinked private configuration directory: {secrets_root}")
    try:
        relative = path.relative_to(secrets_root)
    except ValueError as exc:
        raise SystemExit(f"HPC setup configuration destination must remain under {secrets_root}.") from exc
    current = secrets_root
    for component in relative.parts[:-1]:
        current = current / component
        if current.is_symlink():
            raise SystemExit(f"HPC setup refuses a symlinked private configuration directory: {current}")
        if current.exists() and not current.is_dir():
            raise SystemExit(f"HPC setup configuration parent must be a directory: {current}")
    try:
        destination_stat = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(destination_stat.st_mode):
        raise SystemExit(f"HPC setup refuses a symlinked private configuration file: {path}")
    if not stat.S_ISREG(destination_stat.st_mode):
        raise SystemExit(f"HPC setup configuration destination must be a real regular file: {path}")
    if destination_stat.st_nlink != 1:
        raise SystemExit(
            f"HPC setup refuses a hard-linked private configuration file: {path}"
        )


def _reject_hpc_setup_destination_collisions(
    *,
    ssh_config_path: Path,
    targets_path: Path,
    env_path: Path,
) -> None:
    destinations = {
        "SSH profile config": ssh_config_path,
        "HPC targets config": targets_path,
        "local defaults": env_path,
    }
    items = list(destinations.items())
    for index, (left_label, left_path) in enumerate(items):
        for right_label, right_path in items[index + 1 :]:
            same_path = left_path == right_path
            if not same_path and left_path.exists() and right_path.exists():
                try:
                    same_path = os.path.samefile(left_path, right_path)
                except OSError:
                    same_path = False
            if same_path:
                raise SystemExit(
                    f"HPC setup requires distinct destinations for {left_label} and {right_label}: "
                    f"{left_path}"
                )


def _write_hpc_setup_document(path: Path, document: dict[str, Any]) -> None:
    _ensure_hpc_setup_private_parent(path.parent)
    _write_private_text(path, dump_yaml(document))


def _build_hpc_setup_env_defaults(*, env_path: Path, values: dict[str, str | None]) -> str:
    try:
        existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    except (OSError, UnicodeError) as exc:
        raise SystemExit(f"Local HPC defaults could not be read safely: {env_path}") from exc
    rendered_lines: list[str] = []
    handled: set[str] = set()
    for raw_line in existing_lines:
        stripped = raw_line.strip()
        key, separator, _ = raw_line.partition("=")
        normalized_key = key.strip()
        if not stripped or stripped.startswith("#") or not separator or normalized_key not in values:
            rendered_lines.append(raw_line)
            continue
        if normalized_key in handled:
            continue
        handled.add(normalized_key)
        value = values[normalized_key]
        if value is not None:
            rendered_lines.append(f"{normalized_key}={value}")
    for key, value in values.items():
        if key not in handled and value is not None:
            rendered_lines.append(f"{key}={value}")
    return "\n".join(rendered_lines) + ("\n" if rendered_lines else "")


def _ensure_hpc_setup_private_parent(directory: Path) -> None:
    pending: list[Path] = []
    current = directory
    while not current.exists() and not current.is_symlink():
        pending.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise SystemExit(f"HPC setup private configuration parent is unsafe: {current}")
    for created in reversed(pending):
        try:
            created.mkdir(mode=0o700)
        except OSError as exc:
            raise SystemExit(
                f"HPC setup could not create private configuration directory: {created}"
            ) from exc
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(created, flags)
            try:
                descriptor_stat = os.fstat(descriptor)
                if not stat.S_ISDIR(descriptor_stat.st_mode):
                    raise SystemExit(
                        f"HPC setup private configuration parent is not a directory: {created}"
                    )
                os.fchmod(descriptor, 0o700)
                final_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
                if final_mode != 0o700:
                    raise SystemExit(
                        f"HPC setup could not enforce mode 0700 on private directory: {created}"
                    )
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise SystemExit(
                f"HPC setup could not secure private configuration directory: {created}"
            ) from exc


def _write_private_text(path: Path, content: str) -> None:
    existed = path.exists()
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= 0 if existed else os.O_CREAT | os.O_EXCL
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = not existed
    except OSError as exc:
        raise SystemExit(f"HPC setup could not open private configuration file: {path}") from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode) or descriptor_stat.st_nlink != 1:
            raise SystemExit(
                f"HPC setup private configuration destination is not an exclusive regular file: {path}"
            )
        try:
            os.fchmod(descriptor, 0o600)
        except OSError as exc:
            raise SystemExit(
                f"HPC setup could not enforce private permissions on: {path}"
            ) from exc
        final_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        if final_mode != 0o600:
            raise SystemExit(
                f"HPC setup could not verify mode 0600 on private configuration file: {path}"
            )
        os.ftruncate(descriptor, 0)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
        final_path_stat = path.lstat()
        if (
            not stat.S_ISREG(final_path_stat.st_mode)
            or final_path_stat.st_nlink != 1
            or (final_path_stat.st_dev, final_path_stat.st_ino)
            != (descriptor_stat.st_dev, descriptor_stat.st_ino)
        ):
            raise SystemExit(
                f"HPC setup private configuration file changed while writing: {path}"
            )
        if stat.S_IMODE(final_path_stat.st_mode) != 0o600:
            raise SystemExit(
                f"HPC setup private configuration file did not retain mode 0600: {path}"
            )
    except (OSError, UnicodeError) as exc:
        raise SystemExit(f"HPC setup could not write private configuration file: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created and sys.exc_info()[0] is not None:
            try:
                current = path.lstat()
                if (
                    stat.S_ISREG(current.st_mode)
                    and current.st_dev == descriptor_stat.st_dev
                    and current.st_ino == descriptor_stat.st_ino
                ):
                    path.unlink()
            except (FileNotFoundError, OSError, UnboundLocalError):
                pass


def _secure_hpc_setup_existing_destination(path: Path) -> None:
    try:
        initial_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(initial_stat.st_mode) or initial_stat.st_nlink != 1:
        raise SystemExit(
            f"HPC setup private configuration destination is not an exclusive regular file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SystemExit(f"HPC setup could not open private configuration file: {path}") from exc
    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (initial_stat.st_dev, initial_stat.st_ino)
        ):
            raise SystemExit(
                f"HPC setup private configuration destination changed during validation: {path}"
            )
        try:
            os.fchmod(descriptor, 0o600)
        except OSError as exc:
            raise SystemExit(
                f"HPC setup could not enforce private permissions on: {path}"
            ) from exc
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise SystemExit(
                f"HPC setup could not verify mode 0600 on private configuration file: {path}"
            )
    finally:
        os.close(descriptor)


def _prompt_hpc_setup_required(label: str, *, flag: str) -> str:
    if not sys.stdin.isatty():
        raise SystemExit(f"{label} is required; provide {flag} for noninteractive rp hpc setup.")
    try:
        return _prompt_required(label)
    except (EOFError, OSError) as exc:
        raise SystemExit(f"{label} is required; provide {flag} for noninteractive rp hpc setup.") from exc


def _handle_hpc_target_list(args: argparse.Namespace) -> int:
    try:
        config_path = resolve_hpc_targets_config_path(getattr(args, "config", None))
        targets = list_hpc_targets(config_path=getattr(args, "config", None))
        document = load_hpc_targets_config(getattr(args, "config", None), require=True)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    active_name = resolve_hpc_target_name(document, target_name=None)
    print("HPC targets")
    print(f"Config: {config_path}")
    print(f"Active: {active_name or 'not set'}")
    for target in targets:
        markers = []
        if target.get("active"):
            markers.append("active")
        if target.get("default"):
            markers.append("default")
        suffix = f" ({', '.join(markers)})" if markers else ""
        print(f"- {target['name']}{suffix}")
    return 0


def _handle_hpc_target_show(args: argparse.Namespace) -> int:
    project_name = _normalize_optional_cli_value(getattr(args, "project", None))
    try:
        target = resolve_hpc_target(
            target_name=args.name,
            project_name=project_name,
            config_path=getattr(args, "config", None),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if target is None:
        raise SystemExit("No HPC targets config found.")

    lines = _render_hpc_target_report(target, project_name=project_name)
    print("\n".join(lines))
    return 0


def _handle_hpc_target_use(args: argparse.Namespace) -> int:
    try:
        target = resolve_hpc_target(target_name=args.name, config_path=getattr(args, "config", None))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if target is None:
        raise SystemExit("No HPC targets config found.")

    updates: dict[str, str | None] = {"RESEARCH_HPC_TARGET": str(target["name"])}
    config_value = _normalize_optional_cli_value(getattr(args, "config", None))
    if config_value is not None:
        updates["RESEARCH_HPC_TARGETS_CONFIG"] = config_value
    env_path = write_local_hpc_env_defaults(updates)
    label = getattr(args, "hpc_target_label", "target")
    print(f"Active HPC {label} set to {target['name']} in {to_workspace_relative(env_path, env_path.parent.parent)}.")
    if target.get("warnings"):
        print("")
        print("Warnings")
        for warning in target["warnings"]:
            print(f"- {warning}")
    return 0


def _handle_hpc_cluster(args: argparse.Namespace) -> int:
    raw_parts = list(getattr(args, "cluster_args", []) or [])
    command = raw_parts[0] if raw_parts else "current"
    if command in {"list", "ls"}:
        return _handle_hpc_target_list(args)
    if command in {"current", "status"}:
        return _handle_hpc_cluster_current(args)
    if command == "show":
        if len(raw_parts) != 2:
            raise SystemExit("Usage: rp hpc cluster show <cluster>")
        args.name = raw_parts[1]
        return _handle_hpc_target_show(args)
    if command == "use":
        if len(raw_parts) != 2:
            raise SystemExit("Usage: rp hpc cluster use <cluster>")
        args.name = raw_parts[1]
        args.hpc_target_label = "cluster"
        return _handle_hpc_target_use(args)
    if len(raw_parts) == 1:
        args.name = command
        args.hpc_target_label = "cluster"
        return _handle_hpc_target_use(args)
    raise SystemExit("Usage: rp hpc cluster [list|current|show <cluster>|use <cluster>|<cluster>]")


def _handle_hpc_cluster_current(args: argparse.Namespace) -> int:
    try:
        config_path = resolve_hpc_targets_config_path(getattr(args, "config", None))
        document = load_hpc_targets_config(getattr(args, "config", None), require=True)
        targets = list_hpc_targets(config_path=getattr(args, "config", None))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    active_name = resolve_hpc_target_name(document, target_name=None)
    print(f"Active HPC cluster: {active_name or 'not set'}")
    print(f"Config: {config_path}")
    if targets:
        print("")
        print("Available clusters")
        for target in targets:
            markers = []
            if target.get("active"):
                markers.append("active")
            if target.get("default"):
                markers.append("default")
            suffix = f" ({', '.join(markers)})" if markers else ""
            print(f"- {target['name']}{suffix}")
    return 0


def _handle_hpc_doctor(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    project_context = build_project_hpc_context(project_name)
    profile, config_path, role = _load_hpc_profile(args)
    ssh_report = run_ssh_connectivity_check(profile, mode="auto")
    lines = _render_hpc_doctor_report(
        project_name=project_name,
        project_context=project_context,
        config_path=config_path,
        profile=profile,
        ssh_report=ssh_report,
        role=role,
    )
    print("\n".join(lines))
    has_errors = bool(project_context.get("validation_errors")) or not ssh_report.get("ok", False)
    if not project_context.get("remote_workspace_root"):
        has_errors = True
    return 1 if has_errors else 0


def _handle_hpc_connect(args: argparse.Namespace) -> int:
    profile, config_path, role = _load_hpc_profile(args)
    multiplexing_status, multiplexing_warnings = _ssh_multiplexing_status(profile)
    lines = [
        "Opening reusable SSH connection",
        f"Profile: {profile.name} ({role})",
        f"Target: {profile.target()}",
        f"SSH config: {config_path}",
        f"Multiplexing: {multiplexing_status}",
        "",
    ]
    if multiplexing_warnings:
        lines.append("This profile is not configured for reusable SSH connections.")
        lines.extend(f"- {warning}" for warning in multiplexing_warnings)
        print("\n".join(lines))
        return 1

    report = run_ssh_connectivity_check(
        profile,
        mode="interactive",
        remote_command=getattr(args, "remote_command", None) or "true",
    )
    if not report.get("ok"):
        lines.append("Connection failed.")
        guidance = report.get("host_key_fix_guidance")
        if guidance:
            lines.append(str(guidance))
        print("\n".join(lines))
        return int(report.get("returncode", 1) or 1)

    lines.append("Connection ready. Future rp hpc and rp run commands can reuse this SSH session.")
    if getattr(args, "test_reuse", False):
        reuse_report = run_ssh_connectivity_check(profile, mode="batch", remote_command="hostname")
        reuse_ok = bool(reuse_report.get("ok")) and not bool(reuse_report.get("fallback_to_interactive"))
        lines.append(f"Reuse check: {'ok' if reuse_ok else 'needs attention'}")
        if reuse_report.get("stdout"):
            lines.append(f"Reuse host: {str(reuse_report['stdout']).strip()}")
        if not reuse_ok:
            guidance = reuse_report.get("host_key_fix_guidance")
            if guidance:
                lines.append(str(guidance))
            print("\n".join(lines))
            return int(reuse_report.get("returncode", 1) or 1)

    print("\n".join(lines))
    return 0


def _handle_hpc_sync_workspace(args: argparse.Namespace) -> int:
    workspace_context = build_workspace_hpc_context()
    profile, config_path, role = _load_hpc_profile(args)
    workspace_root = Path(workspace_context["workspace_root"]).resolve()
    remote_workspace_root = _require_remote_workspace_root(workspace_context)
    git_safety = _WorkspaceGitSafetyExcludes(
        untracked=_GitWorkspaceExcludeLayer(active=False, paths=()),
        ignored=_GitWorkspaceExcludeLayer(active=False, paths=()),
    )
    try:
        plan = build_workspace_sync_plan(
            workspace_root=workspace_root,
            remote_workspace_root=remote_workspace_root,
            extra_exclude_file=getattr(args, "extra_exclude_file", None),
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    git_safety = _build_git_workspace_safety_excludes(
        workspace_root,
        include_untracked=bool(getattr(args, "include_untracked", False)),
        include_ignored=bool(getattr(args, "include_ignored", False)),
    )
    tracked_exclude_files = [Path(path) for path in plan["exclude_files"]]
    rsync_exclude_files = list(tracked_exclude_files)
    if git_safety.exclude_file is not None:
        rsync_exclude_files.append(git_safety.exclude_file)
    try:
        mode = _hpc_profile_mode(role)
        mkdir_command = build_ssh_command(
            profile,
            mode=mode,
            remote_command=f"mkdir -p {shlex.quote(remote_workspace_root)}",
            allocate_tty=False,
        )
        rsync_command = _build_profile_push_command(
            source=Path(plan["source"]),
            destination=plan["destination"],
            profile=profile,
            mode=mode,
            exclude_files=rsync_exclude_files,
            dry_run=bool(args.dry_run),
            source_is_directory=True,
        )
        lines = _render_workspace_sync_plan_report(
            workspace_root=workspace_root,
            remote_workspace_root=remote_workspace_root,
            profile=profile,
            config_path=config_path,
            tracked_exclude_files=tracked_exclude_files,
            git_safety=git_safety,
            mkdir_command=mkdir_command,
            rsync_command=rsync_command,
        )
        print("\n".join(lines))
        if not getattr(args, "execute", False):
            return 0

        mkdir_result = subprocess.run(mkdir_command, check=False, text=True)
        if mkdir_result.returncode != 0:
            return mkdir_result.returncode
        rsync_result = subprocess.run(rsync_command, check=False, text=True)
        return rsync_result.returncode
    finally:
        git_safety.cleanup()


def _handle_hpc_sync_project(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    project_context = build_project_hpc_context(project_name)
    profile, config_path, role = _load_hpc_profile(args)
    remote_workspace_root = _require_remote_workspace_root(project_context)
    mode = _hpc_profile_mode(role)
    plan = build_project_sync_plan(context=project_context)
    actions = _build_sync_plan_actions(
        plan=plan,
        workspace_root=Path(project_context["workspace_root"]),
        remote_workspace_root=remote_workspace_root,
        profile=profile,
        mode=mode,
        dry_run=bool(args.dry_run),
    )
    lines = _render_sync_plan_report(
        title=f"Project sync plan for {project_name}",
        actions=actions,
        profile=profile,
        config_path=config_path,
    )
    print("\n".join(lines))
    if not getattr(args, "execute", False):
        return 0
    return _execute_sync_actions(actions)


def _handle_hpc_sync_data(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    project_context = build_project_hpc_context(project_name)
    profile, config_path, role = _load_hpc_profile(args)
    remote_workspace_root = _require_remote_workspace_root(project_context)
    mode = _hpc_profile_mode(role)
    if getattr(args, "selected_only", False):
        plan = _build_selected_data_sync_plan(project_name=project_name, project_context=project_context, batch_name=args.batch)
    else:
        plan = build_project_data_sync_plan(context=project_context)
    actions = _build_sync_plan_actions(
        plan=plan,
        workspace_root=Path(project_context["workspace_root"]),
        remote_workspace_root=remote_workspace_root,
        profile=profile,
        mode=mode,
        dry_run=bool(args.dry_run),
    )
    lines = _render_sync_plan_report(
        title=f"Data sync plan for {project_name}",
        actions=actions,
        profile=profile,
        config_path=config_path,
    )
    print("\n".join(lines))
    if not getattr(args, "execute", False):
        return 0
    return _execute_sync_actions(actions)


def _build_selected_data_sync_plan(
    *,
    project_name: str,
    project_context: dict[str, Any],
    batch_name: str | None,
) -> dict[str, Any]:
    if not batch_name:
        raise SystemExit("--selected-only requires --batch.")
    if project_context.get("slice") != "bids" or project_context.get("tool_adapter") is None:
        raise SystemExit("--selected-only data sync is currently supported only for adapter-backed BIDS projects.")
    local_derivative_root = project_context.get("input_derivative_root")
    remote_derivative_root = _normalize_optional_cli_value(project_context.get("remote_input_derivative_root"))
    if local_derivative_root is None or remote_derivative_root is None:
        raise SystemExit("--selected-only data sync requires local and remote input derivative roots.")

    verification_plan = _build_hpc_data_verification_plan(
        project_name=project_name,
        project_context=project_context,
        batch_name=batch_name,
        selectors={},
    )
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    local_root = Path(local_derivative_root).resolve()
    remote_root = PurePosixPath(remote_derivative_root)
    for row in verification_plan.get("rows", []):
        for remote_path_text in row.get("expected_paths", []):
            remote_path = PurePosixPath(str(remote_path_text))
            try:
                relative = remote_path.relative_to(remote_root)
            except ValueError:
                continue
            source_path = local_root / Path(*relative.parts)
            if source_path.exists() and not source_path.is_file():
                continue
            key = (str(source_path), remote_path.as_posix())
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "label": "selected-input",
                    "kind": "file",
                    "source": str(source_path),
                    "destination": remote_path.as_posix(),
                    "exclude_files": [],
                    "sync_scope": "data",
                }
            )
    if not entries:
        note = verification_plan.get("layer_b_note") or "adapter did not resolve selected input files"
        raise SystemExit(f"--selected-only data sync could not resolve concrete input files: {note}.")
    return {
        "kind": "selected-data-sync-plan",
        "entries": entries,
        "batch": batch_name,
    }


def _handle_hpc_verify_data(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    project_context = build_project_hpc_context(project_name)

    verification_plan = _build_hpc_data_verification_plan(
        project_name=project_name,
        project_context=project_context,
        batch_name=getattr(args, "batch", None),
        selectors=_bids_selector_values(args),
    )

    remote_report: dict[str, Any] = {}
    if verification_plan["paths"]:
        try:
            remote_report = verify_remote_paths(
                paths=verification_plan["paths"],
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
                workspace_root=project_context["workspace_root"],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    summary = _summarize_hpc_data_verification(
        verification_plan=verification_plan,
        remote_report=remote_report,
    )
    print("\n".join(_render_hpc_data_verification_report(summary)))
    if remote_report and int(remote_report.get("returncode", 0)) != 0:
        return int(remote_report["returncode"])
    return 1 if summary["has_missing"] else 0


def _handle_hpc_container_prepare(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    context = _build_context(project_name, allow_missing_batch=True)
    profile, config_path, role = _load_hpc_profile(args)
    mode = _hpc_profile_mode(role)
    spec = _resolve_project_container_prepare_spec(
        context=context,
        backend_override=getattr(args, "backend", None),
        image_override=getattr(args, "image", None),
        source_image_override=getattr(args, "source_image", None),
    )
    remote_command = _build_hpc_container_prepare_remote_command(context=context, spec=spec)
    ssh_command = build_ssh_command(profile, mode=mode, remote_command=remote_command, allocate_tty=False)
    print(
        "\n".join(
            _render_hpc_container_prepare_report(
                project_name=project_name,
                profile=profile,
                config_path=config_path,
                spec=spec,
                ssh_command=ssh_command,
            )
        )
    )
    if not getattr(args, "execute", False):
        return 0
    completed = subprocess.run(ssh_command, check=False, text=True)
    return int(completed.returncode)


def _handle_hpc_notebook_plan(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    project_context = build_project_hpc_context(project_name)
    remote_workspace_root = project_context.get("remote_workspace_root")
    if remote_workspace_root:
        project_context = dict(project_context)
        project_context["remote_workspace_root"] = _validate_remote_workspace_root(str(remote_workspace_root))
    profile, config_path, _ = _load_hpc_profile(args)
    notebook_settings = _resolve_notebook_plan_settings(args=args, project_context=project_context)
    lines = _render_notebook_plan_report(
        project_name=project_name,
        project_context=project_context,
        config_path=config_path,
        profile=profile,
        notebook_settings=notebook_settings,
    )
    print("\n".join(lines))
    return 0


def _handle_hpc_notebook_submit(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    project_context = build_project_hpc_context(project_name)
    remote_workspace_root = project_context.get("remote_workspace_root")
    if remote_workspace_root:
        project_context = dict(project_context)
        project_context["remote_workspace_root"] = _validate_remote_workspace_root(str(remote_workspace_root))
    profile, _, role = _load_hpc_profile(args)
    plan = _plan_hpc_notebook_submit(
        args=args,
        project_name=project_name,
        project_context=project_context,
        profile=profile,
        role=role,
    )
    if not getattr(args, "execute", False):
        print(f"Local run directory: {plan['local_run_dir']}")
        print(f"Local script path: {plan['local_script_path']}")
        print(f"Remote run directory: {plan['remote_run_dir']}")
        print(f"Output notebook path: {plan['remote_output_notebook_path']}")
        print(f"SSH submit command: {plan['submit_command_text']}")
        return 0

    result = subprocess.run(
        plan["submit_command"],
        input=plan["script_content"],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        if stdout.strip():
            print(stdout.strip())
        if stderr.strip():
            print(stderr.strip(), file=sys.stderr)
        return result.returncode

    job_id = _parse_sbatch_submission_job_id(stdout)
    if job_id is None:
        if stdout.strip():
            print(stdout.strip())
        if stderr.strip():
            print(stderr.strip(), file=sys.stderr)
        return 1

    print(f"Job id: {job_id}")
    print(f"Local run directory: {plan['local_run_dir']}")
    print(f"Remote run directory: {plan['remote_run_dir']}")
    print(f"Expected executed notebook path: {plan['remote_output_notebook_path']}")
    return 0


def _handle_hpc_notebook_start(args: argparse.Namespace) -> int:
    launch_remote = bool(getattr(args, "execute", False))
    if not launch_remote and getattr(args, "execute_tunnel", False) and not getattr(args, "compute_host", None):
        raise SystemExit("--execute-tunnel requires --compute-host.")
    project_name = _resolve_project_name(args.project)
    project_context = build_project_hpc_context(project_name)
    remote_workspace_root = project_context.get("remote_workspace_root")
    if remote_workspace_root:
        project_context = dict(project_context)
        project_context["remote_workspace_root"] = _validate_remote_workspace_root(str(remote_workspace_root))
    profile, config_path, _ = _load_hpc_profile(args)
    notebook_settings = _resolve_notebook_start_settings(
        args=args,
        project_context=project_context,
        launch_remote=launch_remote,
    )
    url_path = _resolve_notebook_url_path(token=getattr(args, "token", None), url_path=getattr(args, "url_path", None))
    if launch_remote:
        return _launch_remote_notebook_start(
            project_name=project_name,
            project_context=project_context,
            profile=profile,
            notebook_settings=notebook_settings,
            fallback_url_path=url_path,
            open_browser=getattr(args, "open_browser", False),
            tunnel_mode=getattr(args, "tunnel_mode", "direct"),
        )
    lines, tunnel_command = _build_notebook_start_report(
        project_name=project_name,
        project_context=project_context,
        config_path=config_path,
        profile=profile,
        notebook_settings=notebook_settings,
        compute_host=getattr(args, "compute_host", None),
        url_path=url_path,
        tunnel_mode=getattr(args, "tunnel_mode", "direct"),
    )
    print("\n".join(lines))
    if getattr(args, "execute_tunnel", False):
        result = subprocess.run(tunnel_command, check=False)
        return result.returncode
    return 0


def _launch_remote_notebook_start(
    *,
    project_name: str,
    project_context: dict[str, Any],
    profile: Any,
    notebook_settings: dict[str, Any],
    fallback_url_path: str | None,
    open_browser: bool,
    tunnel_mode: str = "direct",
) -> int:
    remote_command = _build_notebook_remote_launch_command(
        project_name=project_name,
        project_context=project_context,
        notebook_settings=notebook_settings,
        tunnel_mode=tunnel_mode,
    )
    launcher_command = build_ssh_command(profile, mode="interactive", remote_command=remote_command, allocate_tty=False)
    remote_process = subprocess.Popen(
        launcher_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tunnel_process: subprocess.Popen[str] | None = None
    markers: dict[str, str] = {}
    try:
        for line, parsed_markers in _iter_process_output(
            remote_process,
            marker_names=("RP_NOTEBOOK_HOST", "RP_NOTEBOOK_URL"),
        ):
            print(line, end="")
            markers.update(parsed_markers)
            if tunnel_process is not None:
                continue
            host = markers.get("RP_NOTEBOOK_HOST")
            remote_url = markers.get("RP_NOTEBOOK_URL")
            if not host or not remote_url:
                continue
            compute_host = _resolve_tunnel_compute_host(host, profile=profile)
            actual_remote_port = _resolve_remote_notebook_port(
                remote_url=remote_url,
                fallback_port=int(notebook_settings["remote_port"]),
            )
            tunnel_command = _build_ssh_tunnel_command(
                profile,
                local_port=int(notebook_settings["local_port"]),
                remote_port=actual_remote_port,
                remote_host=_resolve_tunnel_forward_host(host) if tunnel_mode == "login-forward" else compute_host,
                jump_host=profile.target(),
                tunnel_mode=tunnel_mode,
            )
            local_url = _build_local_notebook_url(
                local_port=int(notebook_settings["local_port"]),
                remote_url=remote_url,
                fallback_url_path=fallback_url_path,
            )
            print(f"Tunnel command: {_render_command(tunnel_command)}")
            print(f"Local URL: {local_url}")
            tunnel_process, tunnel_stderr, tunnel_reader = _start_logged_tunnel_process(tunnel_command)
            startup_state = _observe_tunnel_process_startup(
                local_port=int(notebook_settings["local_port"]),
                process=tunnel_process,
            )
            if startup_state == "exited":
                tunnel_reader.join(timeout=0.1)
            if startup_state == "exited" and _is_ssh_mux_session_refused_output("".join(tunnel_stderr)):
                retry_tunnel_command = _disable_ssh_multiplexing(tunnel_command)
                print(f"Tunnel command: {_render_command(retry_tunnel_command)}")
                tunnel_process, _, _ = _start_logged_tunnel_process(retry_tunnel_command)
            if open_browser and _wait_for_local_port(
                local_port=int(notebook_settings["local_port"]),
                process=tunnel_process,
            ):
                webbrowser.open(local_url)
        returncode = remote_process.wait()
    except KeyboardInterrupt:
        _terminate_process(tunnel_process)
        _terminate_process(remote_process)
        return 130

    if returncode != 0:
        _terminate_process(tunnel_process)
        return returncode
    if tunnel_process is None:
        return 1
    _terminate_process(tunnel_process)
    return 0


def _iter_process_output(
    process: subprocess.Popen[str],
    *,
    marker_names: tuple[str, ...] = (),
) -> Iterator[tuple[str, dict[str, str]]]:
    if process.stdout is None:
        return
    for line in process.stdout:
        yield line, _parse_machine_readable_markers(line, marker_names=marker_names)


def _parse_machine_readable_markers(line: str, *, marker_names: tuple[str, ...]) -> dict[str, str]:
    normalized = line.strip()
    parsed: dict[str, str] = {}
    for marker_name in marker_names:
        prefix = f"{marker_name}="
        if normalized.startswith(prefix):
            parsed[marker_name] = normalized[len(prefix) :]
    return parsed


def _build_notebook_remote_launch_command(
    *,
    project_name: str,
    project_context: dict[str, Any],
    notebook_settings: dict[str, Any],
    tunnel_mode: str = "direct",
) -> str:
    remote_workspace_root = _require_remote_workspace_root(project_context)
    launch_target = notebook_launch_target(project_context)
    setup_commands = _build_repo_slurm_setup_commands(
        remote_workspace_root=str(remote_workspace_root),
        workspace_root=Path(project_context["workspace_root"]),
        modules=notebook_settings["modules"],
        environment=_resolve_project_slurm_environment(project_context),
        pre_activate_commands=notebook_settings["pre_activate_commands"],
        prepare_directories=notebook_settings.get("prepare_directories", []),
    )
    requested_remote_port = int(notebook_settings["remote_port"])
    bind_ip = _notebook_bind_ip_for_tunnel_mode(tunnel_mode)
    launch_command = ""
    if requested_remote_port != 0:
        launch_command = _notebook_launch_command(
            remote_port=requested_remote_port,
            launch_target=launch_target,
            bind_ip=bind_ip,
        )
    compute_shell_command = "; ".join(
        [
            "set -euo pipefail",
            *setup_commands,
            (
                "python -u -c "
                + shlex.quote(
                    _build_notebook_url_proxy_script(
                        launch_command,
                        requested_port=requested_remote_port,
                        launch_target=launch_target,
                        bind_ip=bind_ip,
                    )
                )
            ),
        ]
    )
    return "python -u -c " + shlex.quote(
        _build_notebook_allocation_proxy_script(
            allocation_command=_notebook_allocation_command(
                project_name=project_name,
                notebook_settings=notebook_settings,
                no_shell=True,
            ),
            hostname_command="srun --jobid {jobid} hostname -f",
            notebook_command=f"srun --jobid {{jobid}} bash -lc {shlex.quote(compute_shell_command)}",
        )
    )


def _build_repo_slurm_setup_commands(
    *,
    remote_workspace_root: str,
    workspace_root: Path,
    modules: list[str],
    environment: dict[str, str] | None,
    pre_activate_commands: list[str],
    prepare_directories: list[str],
) -> list[str]:
    return build_slurm_setup_commands(
        remote_workspace_root=remote_workspace_root,
        modules=modules,
        environment=environment,
        pre_activate_commands=pre_activate_commands,
        prepare_directories=prepare_directories,
        bootstrap_command=_build_repo_bootstrap_command(workspace_root),
        activate_command=_build_repo_activate_command(workspace_root),
    )


def _resolve_project_slurm_environment(project_context: dict[str, Any]) -> dict[str, str]:
    slurm_config = project_context.get("compute", {}).get("slurm", {})
    if not isinstance(slurm_config, dict):
        return {}
    return _resolve_slurm_environment(slurm_config)


def _build_repo_bootstrap_command(workspace_root: Path) -> str | None:
    bootstrap_script = workspace_root / "ops" / "envs" / "dev" / "bootstrap.sh"
    if not bootstrap_script.exists():
        return None
    activate_path = ".venv/bin/activate"
    stamp_path = _NOTEBOOK_BOOTSTRAP_STAMP_FILENAME
    current_stamp = _compute_notebook_bootstrap_stamp(workspace_root)
    return "; ".join(
        [
            (
                f"if [ -f {shlex.quote(activate_path)} ] && [ -f {shlex.quote(stamp_path)} ]"
                f" && [ \"$(tr -d '\\r\\n' < {shlex.quote(stamp_path)})\" = {shlex.quote(current_stamp)} ]"
            ),
            f"then printf '%s\\n' {shlex.quote('Bootstrap already current; skipping.')}",
            (
                "else "
                f"bash {shlex.quote(to_workspace_relative(bootstrap_script, workspace_root))}"
                f" && printf '%s\\n' {shlex.quote(current_stamp)} > {shlex.quote(stamp_path)}"
            ),
            "fi",
        ]
    )


def _build_repo_activate_command(workspace_root: Path) -> str | None:
    bootstrap_script = workspace_root / "ops" / "envs" / "dev" / "bootstrap.sh"
    activate_script = workspace_root / ".venv" / "bin" / "activate"
    if not activate_script.exists() and not bootstrap_script.exists():
        return None
    return f"source {shlex.quote(to_workspace_relative(activate_script, workspace_root))}"


def _build_notebook_bootstrap_command(project_context: dict[str, Any]) -> str | None:
    return _build_repo_bootstrap_command(Path(project_context["workspace_root"]))


def _compute_notebook_bootstrap_stamp(workspace_root: Path) -> str:
    digest = hashlib.sha256()

    def update_text(value: str) -> None:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")

    def update_file(path: Path) -> None:
        update_text(str(path.relative_to(workspace_root)))
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")

    update_text("notebook-bootstrap-stamp:v1")
    update_file(workspace_root / "ops" / "envs" / "dev" / "bootstrap.sh")
    update_file(workspace_root / "ops" / "envs" / "dev" / "requirements-notebook.txt")
    update_file(workspace_root / "ops" / "envs" / "hpc" / "requirements-runtime.txt")
    update_file(workspace_root / "ops" / "scripts" / "detect_execution_profile.sh")

    packages_root = workspace_root / "packages"
    for package_dir in sorted(packages_root.glob("research-*")):
        if not package_dir.is_dir():
            continue
        update_text(f"package:{package_dir.relative_to(workspace_root)}")
        metadata_found = False
        for filename in _NOTEBOOK_BOOTSTRAP_METADATA_FILENAMES:
            metadata_path = package_dir / filename
            if metadata_path.exists():
                update_file(metadata_path)
                metadata_found = True
        if not metadata_found:
            update_text("metadata:<missing>")

    return digest.hexdigest()


def _build_notebook_allocation_proxy_script(
    *,
    allocation_command: str,
    hostname_command: str,
    notebook_command: str,
) -> str:
    return "\n".join(
        [
            "import re",
            "import subprocess",
            "import sys",
            f"allocation_command = {json.dumps(allocation_command)}",
            f"hostname_command = {json.dumps(hostname_command)}",
            f"notebook_command = {json.dumps(notebook_command)}",
            f"jobid_pattern = re.compile({json.dumps(_SALLOC_GRANTED_JOB_ALLOCATION_PATTERN.pattern)})",
            "allocation_proc = subprocess.Popen(",
            "    allocation_command,",
            "    shell=True,",
            "    stdout=subprocess.PIPE,",
            "    stderr=subprocess.STDOUT,",
            "    text=True,",
            "    bufsize=1,",
            ")",
            "assert allocation_proc.stdout is not None",
            "jobid = None",
            "for line in allocation_proc.stdout:",
            "    sys.stdout.write(line)",
            "    sys.stdout.flush()",
            "    match = jobid_pattern.search(line)",
            "    if match:",
            "        jobid = match.group(1)",
            "        break",
            "if jobid is None:",
            "    raise SystemExit(allocation_proc.wait())",
            "hostname_result = subprocess.run(",
            "    hostname_command.format(jobid=jobid),",
            "    shell=True,",
            "    check=False,",
            "    capture_output=True,",
            "    text=True,",
            ")",
            "if hostname_result.returncode != 0:",
            "    sys.stdout.write(hostname_result.stdout)",
            "    sys.stdout.write(hostname_result.stderr)",
            "    sys.stdout.flush()",
            "    allocation_proc.terminate()",
            "    raise SystemExit(hostname_result.returncode)",
            "hostname = hostname_result.stdout.strip()",
            "if not hostname:",
            "    allocation_proc.terminate()",
            "    raise SystemExit(1)",
            "sys.stdout.write(f'RP_NOTEBOOK_HOST={hostname}\\n')",
            "sys.stdout.flush()",
            "notebook_proc = subprocess.Popen(",
            "    notebook_command.format(jobid=jobid),",
            "    shell=True,",
            "    stdout=subprocess.PIPE,",
            "    stderr=subprocess.STDOUT,",
            "    text=True,",
            "    bufsize=1,",
            ")",
            "assert notebook_proc.stdout is not None",
            "for line in notebook_proc.stdout:",
            "    sys.stdout.write(line)",
            "    sys.stdout.flush()",
            "notebook_returncode = notebook_proc.wait()",
            "allocation_proc.terminate()",
            "try:",
            "    allocation_proc.wait(timeout=5)",
            "except subprocess.TimeoutExpired:",
            "    allocation_proc.kill()",
            "    allocation_proc.wait()",
            "raise SystemExit(notebook_returncode)",
        ]
    )


def _parse_salloc_job_id(line: str) -> str | None:
    match = _SALLOC_GRANTED_JOB_ALLOCATION_PATTERN.search(line.strip())
    if match is None:
        return None
    return match.group(1)


def _build_notebook_url_proxy_script(
    launch_command: str,
    *,
    requested_port: int | None = None,
    launch_target: str | None = None,
    bind_ip: str = "127.0.0.1",
) -> str:
    return "\n".join(
        [
            "import os",
            "import re",
            "import shlex",
            "import socket",
            "import subprocess",
            "import sys",
            "import time",
            "from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit",
            f"launch_command = {launch_command!r}",
            f"launch_target = {launch_target!r}",
            f"requested_port = {requested_port!r}",
            f"bind_ip = {bind_ip!r}",
            "pattern = re.compile(r'https?://[^\\s]+')",
            "cwd = os.path.realpath(os.getcwd())",
            "resolved_port = requested_port",
            "if launch_target is not None:",
            "    if resolved_port in (None, 0):",
            "        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:",
            "            sock.bind((bind_ip, 0))",
            "            resolved_port = int(sock.getsockname()[1])",
            "    launch_command = (",
            "        'jupyter lab --no-browser --ip=' + bind_ip + ' --port='",
            "        + str(resolved_port)",
            "        + ' '",
            "        + shlex.quote(launch_target)",
            "    )",
            "",
            "def _iter_running_servers():",
            "    loaders = []",
            "    try:",
            "        from jupyter_server.serverapp import list_running_servers as list_jupyter_servers",
            "    except Exception:",
            "        list_jupyter_servers = None",
            "    if list_jupyter_servers is not None:",
            "        loaders.append(list_jupyter_servers)",
            "    try:",
            "        from notebook.notebookapp import list_running_servers as list_notebook_servers",
            "    except Exception:",
            "        list_notebook_servers = None",
            "    if list_notebook_servers is not None:",
            "        loaders.append(list_notebook_servers)",
            "    for loader in loaders:",
            "        try:",
            "            yield from loader()",
            "        except Exception:",
            "            continue",
            "",
            "def _normalize_server_url(server_info):",
            "    url = str(server_info.get('url') or '').strip()",
            "    if not url:",
            "        return None",
            "    try:",
            "        parsed = urlsplit(url)",
            "    except ValueError:",
            "        return None",
            "    path = parsed.path.rstrip('/')",
            "    if not path.endswith('/lab'):",
            "        path = path + '/lab' if path else '/lab'",
            "    query = parse_qsl(parsed.query, keep_blank_values=True)",
            "    token = str(server_info.get('token') or '').strip()",
            "    if token and not any(key == 'token' for key, _ in query):",
            "        query.append(('token', token))",
            "    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), ''))",
            "",
            "def _server_root(server_info):",
            "    root = server_info.get('root_dir') or server_info.get('notebook_dir')",
            "    if not root:",
            "        return None",
            "    return os.path.realpath(str(root))",
            "",
            "def _server_port(server_info):",
            "    url = str(server_info.get('url') or '').strip()",
            "    if not url:",
            "        return None",
            "    try:",
            "        return urlsplit(url).port",
            "    except ValueError:",
            "        return None",
            "",
            "def _discover_running_server_url(proc_pid):",
            "    candidates = []",
            "    for server_info in _iter_running_servers():",
            "        normalized_url = _normalize_server_url(server_info)",
            "        if normalized_url is None:",
            "            continue",
            "        score = 0",
            "        server_pid = server_info.get('pid')",
            "        if server_pid is not None and str(server_pid) == str(proc_pid):",
            "            score += 100",
            "        if resolved_port not in (None, 0) and _server_port(server_info) == resolved_port:",
            "            score += 50",
            "        if _server_root(server_info) == cwd:",
            "            score += 10",
            "        candidates.append((score, normalized_url))",
            "    if not candidates:",
            "        return None",
            "    candidates.sort(key=lambda item: item[0], reverse=True)",
            "    if candidates[0][0] > 0:",
            "        return candidates[0][1]",
            "    return None",
            "",
            "def _wait_for_running_server_url(proc_pid, *, timeout_seconds):",
            "    deadline = time.monotonic() + timeout_seconds",
            "    while time.monotonic() <= deadline:",
            "        discovered = _discover_running_server_url(proc_pid)",
            "        if discovered is not None:",
            "            return discovered",
            "        time.sleep(0.1)",
            "    return None",
            "proc = subprocess.Popen(",
            "    'exec ' + launch_command,",
            "    shell=True,",
            "    stdout=subprocess.PIPE,",
            "    stderr=subprocess.STDOUT,",
            "    text=True,",
            "    bufsize=1,",
            ")",
            "assert proc.stdout is not None",
            "emitted = False",
            "for line in proc.stdout:",
            "    sys.stdout.write(line)",
            "    sys.stdout.flush()",
            "    if not emitted:",
            "        match = pattern.search(line)",
            "        if match and resolved_port != 0:",
            "            sys.stdout.write('RP_NOTEBOOK_URL=' + match.group(0) + '\\n')",
            "            sys.stdout.flush()",
            "            emitted = True",
            "            continue",
            "        discovered_url = _wait_for_running_server_url(proc.pid, timeout_seconds=0.5)",
            "        if discovered_url is not None:",
            "            sys.stdout.write('RP_NOTEBOOK_URL=' + discovered_url + '\\n')",
            "            sys.stdout.flush()",
            "            emitted = True",
            "            continue",
            "if not emitted:",
            "    discovered_url = _wait_for_running_server_url(proc.pid, timeout_seconds=0.5)",
            "    if discovered_url is not None:",
            "        sys.stdout.write('RP_NOTEBOOK_URL=' + discovered_url + '\\n')",
            "        sys.stdout.flush()",
            "raise SystemExit(proc.wait())",
        ]
    )


def _build_local_notebook_url(*, local_port: int, remote_url: str | None, fallback_url_path: str | None = None) -> str:
    resolved_url_path = fallback_url_path or ""
    normalized_remote_url = _normalize_optional_cli_value(remote_url)
    if normalized_remote_url:
        parsed = urlsplit(normalized_remote_url)
        suffix = parsed.path or ""
        if parsed.query:
            suffix = f"{suffix}?{parsed.query}" if suffix else f"?{parsed.query}"
        if parsed.fragment:
            suffix = f"{suffix}#{parsed.fragment}" if suffix else f"#{parsed.fragment}"
        if suffix:
            resolved_url_path = suffix
    return _render_local_tunnel_url(local_port, resolved_url_path)


def _terminate_process(process: subprocess.Popen[str] | None, *, timeout: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def _handle_hpc_tunnel(args: argparse.Namespace) -> int:
    profile, _, _ = _load_hpc_profile(args)
    tunnel_mode = getattr(args, "tunnel_mode", "direct")
    compute_host = _resolve_tunnel_compute_host(args.compute_host, profile=profile)
    tunnel_command = _build_ssh_tunnel_command(
        profile,
        local_port=args.local_port,
        remote_port=args.remote_port,
        remote_host=_resolve_tunnel_forward_host(args.compute_host) if tunnel_mode == "login-forward" else compute_host,
        jump_host=profile.target(),
        tunnel_mode=tunnel_mode,
    )
    if getattr(args, "execute", False):
        result = subprocess.run(tunnel_command, check=False)
        return result.returncode

    print(f"Tunnel command: {_render_command(tunnel_command)}")
    if args.url_path:
        print(f"Local URL: {_render_local_tunnel_url(args.local_port, args.url_path)}")
    return 0


def _handle_publish_back_plan(args: argparse.Namespace) -> int:
    context = _workspace_context()
    manifest, _, run_root_path = _load_run(context, args.run_id)
    plan = build_publish_back_plan(
        workspace_root=context["workspace_root"],
        run_root=run_root_path,
        manifest=manifest,
    )
    scaffold = build_publish_back_scaffold(
        workspace_root=context["workspace_root"],
        run_root=run_root_path,
        publish_back=manifest.get("publish_back"),
    )
    plan_path = _resolve_reference_path(context["workspace_root"], scaffold["plan_path"])
    write_yaml(plan_path, plan)
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "plan": to_workspace_relative(plan_path, context["workspace_root"]),
                "summary": plan["summary"],
            },
            indent=2,
        )
    )
    return 0


def _workspace_context() -> dict[str, Any]:
    resolved_root = workspace_root()
    apply_hpc_target_defaults(project_name=None, root=resolved_root)
    workspace_config = load_workspace_config(resolved_root)
    return {"workspace_root": resolved_root, "workspace": workspace_config, "paths": workspace_paths(resolved_root, workspace_config)}


def _load_bundle(project_name: str) -> tuple[dict[str, Any], Path]:
    resolved_root = workspace_root()
    apply_hpc_target_defaults(project_name=project_name, root=resolved_root)
    return load_project_bundle(project_name, resolved_root), resolved_root


def _build_analysis_bundle_project_context(project_name: str) -> dict[str, Any]:
    resolved_root = workspace_root()
    record = load_project_record(project_name, resolved_root)
    project_root_path = Path(record["project_root"])
    analysis_config_dir = project_root_path / "config" / "analysis"
    return {
        "workspace_root": resolved_root,
        "workspace": record["workspace"],
        "project_name": project_name,
        "project_root": project_root_path,
        "bundles_dir": analysis_config_dir / "bundles",
        "cohorts_path": project_root_path / "config" / "cohorts.yaml",
        "batches_dir": project_root_path / "manifests" / "batches",
        "roi_sets_dir": analysis_config_dir / "roi_sets",
        "extraction_sets_dir": analysis_config_dir / "extraction_sets",
        "mvpa_sets_dir": analysis_config_dir / "mvpa",
    }


def _build_analysis_model_project_context(project_name: str) -> dict[str, Any]:
    resolved_root = workspace_root()
    apply_hpc_target_defaults(project_name=project_name, root=resolved_root)
    record = load_project_record(project_name, resolved_root)
    workspace = record["workspace"]
    project_root_path = Path(record["project_root"])
    analysis_path = project_root_path / "config" / "analysis.yaml"
    if not analysis_path.exists():
        raise SystemExit(
            json.dumps(
                {"error": f"Project {project_name!r} does not define config/analysis.yaml."},
                indent=2,
            )
        )
    analysis_document = load_yaml(analysis_path)
    analysis = analysis_document.get("analysis")
    if not isinstance(analysis, dict):
        raise SystemExit(json.dumps({"error": "config/analysis.yaml must contain a top-level analysis mapping."}, indent=2))
    return {
        "workspace_root": resolved_root,
        "workspace": workspace,
        "project_name": project_name,
        "project_root": project_root_path,
        "analysis": analysis,
        "analysis_path": analysis_path,
        "models_dir": project_root_path / "config" / "analysis" / "models",
    }


def _build_analysis_roi_project_context(project_name: str) -> dict[str, Any]:
    resolved_root = workspace_root()
    apply_hpc_target_defaults(project_name=project_name, root=resolved_root)
    record = load_project_record(project_name, resolved_root)
    workspace = record["workspace"]
    project_root_path = Path(record["project_root"])
    analysis_config_dir = project_root_path / "config" / "analysis"
    return {
        "workspace_root": resolved_root,
        "workspace": workspace,
        "project_name": project_name,
        "project_root": project_root_path,
        "analysis_config_dir": analysis_config_dir,
        "roi_sets_dir": analysis_config_dir / "roi_sets",
        "transform_sets_dir": analysis_config_dir / "roi_transforms",
        "extraction_sets_dir": analysis_config_dir / "extraction_sets",
    }


def _build_analysis_mvpa_project_context(project_name: str) -> dict[str, Any]:
    context = _build_analysis_roi_project_context(project_name)
    project_root = Path(context["project_root"])
    context["mvpa_sets_dir"] = Path(context["analysis_config_dir"]) / "mvpa"
    context["mvpa_tables_dir"] = Path(context["analysis_config_dir"]) / "mvpa_tables"
    context["mvpa_figures_dir"] = Path(context["analysis_config_dir"]) / "mvpa_figures"
    context["mvpa_rdms_dir"] = Path(context["analysis_config_dir"]) / "mvpa_rdms"
    context["mvpa_publication_dir"] = Path(context["analysis_config_dir"]) / "mvpa_publication"
    context["mvpa_derivatives_dir"] = Path(context["analysis_config_dir"]) / "mvpa_derivatives"
    context["bundles_dir"] = Path(context["analysis_config_dir"]) / "bundles"
    context["cohorts_path"] = project_root / "config" / "cohorts.yaml"
    context["batches_dir"] = project_root / "manifests" / "batches"
    return context


def _mvpa_lifecycle_functions() -> tuple[Any, Any]:
    try:
        from research_platform.neuro.mvpa import plan_mvpa_discovery, validate_mvpa_set_document
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA validation and planning require research-neuro: {exc}"}, indent=2)) from exc
    return validate_mvpa_set_document, plan_mvpa_discovery


def _mvpa_scaffold() -> Any:
    try:
        from research_platform.neuro.mvpa import scaffold
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA config scaffolding requires research-neuro: {exc}"}, indent=2)) from exc
    return scaffold


def _mvpa_table_export_functions() -> tuple[Any, Any]:
    try:
        from research_platform.analysis.mvpa.table_export import (
            plan_or_execute_mvpa_table_export,
            validate_mvpa_table_export_document,
        )
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA table exports require research-analysis: {exc}"}, indent=2)) from exc
    return validate_mvpa_table_export_document, plan_or_execute_mvpa_table_export


def _mvpa_figure_export_functions() -> tuple[Any, Any]:
    try:
        from research_platform.analysis.mvpa.figure_export import (
            plan_or_execute_mvpa_figure_export,
            validate_mvpa_figure_export_document,
        )
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA figure exports require research-analysis: {exc}"}, indent=2)) from exc
    return validate_mvpa_figure_export_document, plan_or_execute_mvpa_figure_export


def _mvpa_rdm_export_functions() -> tuple[Any, Any]:
    try:
        from research_platform.analysis.mvpa.rdm_export import (
            plan_or_execute_mvpa_rdm_export,
            validate_mvpa_rdm_export_document,
        )
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA RDM exports require research-analysis: {exc}"}, indent=2)) from exc
    return validate_mvpa_rdm_export_document, plan_or_execute_mvpa_rdm_export


def _mvpa_publication_export_functions() -> tuple[Any, Any]:
    try:
        from research_platform.analysis.mvpa.publication_export import (
            plan_or_execute_mvpa_publication_export,
            validate_mvpa_publication_export_document,
        )
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA publication exports require research-analysis: {exc}"}, indent=2)) from exc
    return validate_mvpa_publication_export_document, plan_or_execute_mvpa_publication_export


def _mvpa_derivative_publish_functions() -> tuple[Any, Any]:
    try:
        from research_platform.analysis.mvpa.derivative_publish import (
            plan_or_execute_mvpa_derivative_publish,
            validate_mvpa_derivative_publish_document,
        )
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"MVPA derivative publishing requires research-analysis: {exc}"}, indent=2)) from exc
    return validate_mvpa_derivative_publish_document, plan_or_execute_mvpa_derivative_publish


def _roi_schema() -> Any:
    try:
        from research_platform.neuro import roi as schema
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"ROI validation requires research-neuro: {exc}"}, indent=2)) from exc
    return schema


def _roi_scaffold() -> Any:
    try:
        from research_platform.neuro import roi_scaffold
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"ROI config scaffolding requires research-neuro: {exc}"}, indent=2)) from exc
    return roi_scaffold


def _roi_executor() -> Any:
    try:
        from research_platform.neuro import roi_execution
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"ROI execution requires research-neuro: {exc}"}, indent=2)) from exc
    return roi_execution


def _roi_doctor() -> Any:
    try:
        from research_platform.neuro import roi_doctor
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"ROI doctor requires research-neuro: {exc}"}, indent=2)) from exc
    return roi_doctor


def _roi_transform_functions() -> tuple[Any, Any, Any]:
    try:
        from research_platform.neuro.roi_transforms import (
            execute_mni_to_t1w_roi_transform_plan,
            plan_mni_to_t1w_roi_transforms,
            validate_mni_to_t1w_roi_transform_execution_plan,
        )
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"ROI transform planning requires research-neuro: {exc}"}, indent=2)) from exc
    return (
        plan_mni_to_t1w_roi_transforms,
        validate_mni_to_t1w_roi_transform_execution_plan,
        execute_mni_to_t1w_roi_transform_plan,
    )


def _build_analysis_roi_execution_context(context: dict[str, Any]) -> Any:
    executor = _roi_executor()
    root_refs = _analysis_roi_root_refs(context)
    return executor.RoiExecutionContext(
        workspace_root=context["workspace_root"],
        project_root=context["project_root"],
        artifacts_root=root_refs["artifacts_root"],
        project_name=context["project_name"],
        root_refs=root_refs,
    )


def _analysis_roi_root_refs(context: dict[str, Any]) -> dict[str, Path]:
    workspace_root_path = Path(context["workspace_root"])
    project_root_path = Path(context["project_root"])
    workspace_config = context["workspace"]
    configured_paths = _analysis_roi_workspace_paths(workspace_root_path, workspace_config)
    root_refs: dict[str, Path] = {
        "workspace_root": workspace_root_path,
        "project_root": project_root_path,
        "project_config_root": project_root_path / "config",
        "analysis_config_root": project_root_path / "config" / "analysis",
        "roi_config_root": project_root_path / "config" / "analysis",
        "project_roi_root": project_root_path / "config" / "analysis",
        "artifacts_root": configured_paths.get("artifacts_root", workspace_root_path / "artifacts"),
        "artifact_root": configured_paths.get("artifacts_root", workspace_root_path / "artifacts"),
        "datasets_root": configured_paths.get("datasets_root", workspace_root_path / "datasets"),
        "project_derivatives_root": project_root_path / "derivatives",
        "dataset_derivatives_root": project_root_path / "derivatives",
        "bids_derivatives_root": project_root_path / "derivatives",
    }
    for name, path in configured_paths.items():
        root_refs[str(name)] = Path(path)

    dataset_config_path = project_root_path / "config" / "dataset.yaml"
    if dataset_config_path.exists():
        dataset_document = load_yaml(dataset_config_path)
        dataset_config = dataset_document.get("dataset", dataset_document) if isinstance(dataset_document, dict) else {}
        if isinstance(dataset_config, dict):
            _add_analysis_roi_dataset_roots(root_refs, dataset_config, workspace_root=workspace_root_path, workspace=workspace_config)

    analysis_path = project_root_path / "config" / "analysis.yaml"
    if analysis_path.exists():
        analysis_document = load_yaml(analysis_path)
        analysis_config = analysis_document.get("analysis", analysis_document) if isinstance(analysis_document, dict) else {}
        if isinstance(analysis_config, dict):
            _add_analysis_roi_external_roots(root_refs, analysis_config, workspace_root=workspace_root_path, project_root=project_root_path)

    return {name: _analysis_roi_lexical_absolute_path(path) for name, path in root_refs.items()}


def _analysis_roi_workspace_paths(workspace_root: Path, workspace: dict[str, Any]) -> dict[str, Path]:
    """Resolve ROI named roots lexically so output preflight can see symlinks."""

    raw_paths = workspace.get("paths", {})
    if not isinstance(raw_paths, dict):
        return {}
    paths: dict[str, Path] = {}
    for name, raw_value in raw_paths.items():
        value = resolve_env_value(raw_value)
        if value is not None:
            paths[str(name)] = _resolve_analysis_roi_declared_path(
                value,
                workspace_root=workspace_root,
                project_root=workspace_root,
            )
    return paths


def _add_analysis_roi_dataset_roots(
    root_refs: dict[str, Path],
    dataset_config: dict[str, Any],
    *,
    workspace_root: Path,
    workspace: dict[str, Any],
) -> None:
    primary = _normalize_optional_cli_value(dataset_config.get("primary"))
    dataset_root_path: Path | None = None
    if primary is not None:
        configured_dataset = workspace.get("datasets", {}).get(primary)
        if configured_dataset is not None:
            configured_value = resolve_env_value(configured_dataset)
            dataset_root_path = (
                _resolve_analysis_roi_declared_path(
                    configured_value,
                    workspace_root=workspace_root,
                    project_root=workspace_root,
                )
                if configured_value is not None
                else None
            )
        if dataset_root_path is None:
            dataset_root_path = root_refs.get("datasets_root", workspace_root / "datasets") / primary
        root_refs["dataset_root"] = dataset_root_path
    bids_root = resolve_env_value(dataset_config.get("bids_root"))
    if bids_root:
        root_refs["bids_root"] = _resolve_analysis_roi_declared_path(bids_root, workspace_root=workspace_root, project_root=workspace_root)
    elif dataset_root_path is not None:
        root_refs["bids_root"] = dataset_root_path
    bids_root_path = root_refs.get("bids_root")

    derivatives_root = resolve_env_value(
        dataset_config.get("derivatives_root")
        or dataset_config.get("bids_derivatives_root")
        or dataset_config.get("dataset_derivatives_root")
    )
    if derivatives_root:
        resolved_derivatives_root = _resolve_analysis_roi_declared_path(
            derivatives_root,
            workspace_root=workspace_root,
            project_root=workspace_root,
        )
    elif bids_root_path is not None:
        resolved_derivatives_root = Path(bids_root_path) / "derivatives"
    else:
        resolved_derivatives_root = None
    if resolved_derivatives_root is not None:
        root_refs["dataset_derivatives_root"] = resolved_derivatives_root
        root_refs["bids_derivatives_root"] = resolved_derivatives_root
        root_refs["project_derivatives_root"] = resolved_derivatives_root

    derivative_root = resolve_env_value(dataset_config.get("input_derivative_root"))
    if derivative_root:
        resolved_derivative_root = _resolve_analysis_roi_declared_path(
            derivative_root,
            workspace_root=workspace_root,
            project_root=workspace_root,
        )
    elif dataset_root_path is not None and _normalize_optional_cli_value(dataset_config.get("input_derivative")) is not None:
        resolved_derivative_root = dataset_root_path / "derivatives" / str(dataset_config["input_derivative"])
    else:
        resolved_derivative_root = None
    if resolved_derivative_root is not None:
        root_refs["derivative_root"] = resolved_derivative_root
        root_refs["input_derivative_root"] = resolved_derivative_root


def _add_analysis_roi_external_roots(
    root_refs: dict[str, Path],
    analysis_config: dict[str, Any],
    *,
    workspace_root: Path,
    project_root: Path,
) -> None:
    external_roots = analysis_config.get("external_input_roots")
    if isinstance(external_roots, dict):
        for name, declaration in external_roots.items():
            if not isinstance(declaration, dict):
                continue
            local_root = resolve_env_value(declaration.get("local_root"))
            if local_root:
                root_refs[str(name)] = _resolve_analysis_roi_declared_path(
                    local_root,
                    workspace_root=workspace_root,
                    project_root=project_root,
                )


def _resolve_analysis_roi_declared_path(value: str, *, workspace_root: Path, project_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return _analysis_roi_lexical_absolute_path(candidate)
    workspace_candidate = _analysis_roi_lexical_absolute_path(workspace_root / candidate)
    if workspace_candidate.exists():
        return workspace_candidate
    return _analysis_roi_lexical_absolute_path(project_root / candidate)


def _analysis_roi_lexical_absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(value).expanduser())))


def _analysis_bundle_functions() -> tuple[Any, Any]:
    from .analysis_workflow import resolve_analysis_bundle, validate_analysis_bundle_document

    return validate_analysis_bundle_document, resolve_analysis_bundle


def _normalize_analysis_bundle_name(name: str) -> str:
    normalized = _normalize_roi_config_name(name, label="analysis bundle")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", normalized):
        raise SystemExit(
            json.dumps(
                {
                    "error": (
                        f"Invalid analysis bundle name {name!r}. Start with a letter or number and use only "
                        "letters, numbers, dots, underscores, and hyphens."
                    )
                },
                indent=2,
            )
        )
    return normalized


def _analysis_bundle_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["bundles_dir"]) / f"{name}.yaml"


def _analysis_bundle_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_bundle_config_path(context=context, name=name)
    if path.is_file():
        return path
    expected_path = to_workspace_relative(path, context["workspace_root"])
    next_step = f"rp analysis bundle init {name} --project {context['project_name']}"
    raise SystemExit(
        json.dumps(
            {
                "error": (
                    f"Analysis bundle {name!r} was not found at {expected_path}. "
                    f"Initialize it with: {next_step}"
                ),
                "name": name,
                "expected_path": expected_path,
                "next_step": next_step,
            },
            indent=2,
        )
    )


def _analysis_bundle_scaffold_document(name: str) -> dict[str, Any]:
    return {
        "analysis_bundle": {
            "name": name,
            "selection": {"batch": "default"},
            "units": {
                "key_columns": ["subject_id"],
                "subject_column": "subject_id",
                "incomplete": "allow",
            },
            "components": {},
            "stages": [],
        }
    }


def _analysis_bundle_cli_document(args: argparse.Namespace) -> tuple[dict[str, Any], str, Path, dict[str, Any]]:
    context = _build_analysis_bundle_project_context(_resolve_project_name(args.project))
    bundle_name = _normalize_analysis_bundle_name(args.name)
    bundle_path = _analysis_bundle_path(context=context, name=bundle_name)
    document = load_yaml(bundle_path, resolve_env=False)
    if not isinstance(document, dict):
        raise SystemExit(
            json.dumps(
                {"error": f"Analysis bundle file {bundle_path.name} must contain a mapping."},
                indent=2,
            )
        )
    return context, bundle_name, bundle_path, document


def _analysis_bundle_document_name(document: dict[str, Any]) -> str | None:
    payload = document.get("analysis_bundle")
    if not isinstance(payload, dict):
        payload = document
    return _normalize_optional_cli_value(payload.get("name")) if isinstance(payload, dict) else None


def _analysis_bundle_selected_batch_name(
    document: dict[str, Any],
    cohorts_document: dict[str, Any],
) -> str | None:
    payload = document.get("analysis_bundle")
    if not isinstance(payload, dict):
        payload = document
    selection = payload.get("selection") if isinstance(payload, dict) else None
    if not isinstance(selection, dict):
        return None
    batch_name = _normalize_optional_cli_value(selection.get("batch"))
    if batch_name is not None:
        return batch_name
    cohort_name = _normalize_optional_cli_value(selection.get("cohort"))
    cohorts = cohorts_document.get("cohorts")
    cohort = cohorts.get(cohort_name) if isinstance(cohorts, dict) and cohort_name is not None else None
    return _normalize_optional_cli_value(cohort.get("batch")) if isinstance(cohort, dict) else None


def _analysis_bundle_selected_cohort_name(document: dict[str, Any]) -> str | None:
    payload = document.get("analysis_bundle")
    if not isinstance(payload, dict):
        payload = document
    selection = payload.get("selection") if isinstance(payload, dict) else None
    return _normalize_optional_cli_value(selection.get("cohort")) if isinstance(selection, dict) else None


def _resolve_analysis_bundle_for_cli(
    *,
    context: dict[str, Any],
    bundle_name: str,
    document: dict[str, Any],
) -> Any:
    from .manifests import read_manifest_table

    cohorts_document: dict[str, Any] = {"cohorts": {}}
    cohorts_path = Path(context["cohorts_path"])
    if _analysis_bundle_selected_cohort_name(document) is not None and cohorts_path.exists():
        cohorts_document = load_yaml(cohorts_path, resolve_env=False)
    if not isinstance(cohorts_document, dict):
        raise SystemExit(json.dumps({"error": "config/cohorts.yaml must contain a mapping."}, indent=2))

    batch_tables: dict[str, Any] = {}
    selected_batch_name = _analysis_bundle_selected_batch_name(document, cohorts_document)
    if selected_batch_name is not None and re.fullmatch(r"[A-Za-z0-9_.-]+", selected_batch_name):
        batch_path = Path(context["batches_dir"]) / f"{selected_batch_name}.tsv"
        if batch_path.exists():
            try:
                batch_tables[selected_batch_name] = read_manifest_table(batch_path)
            except (OSError, ValueError):
                expected_path = to_workspace_relative(batch_path, context["workspace_root"])
                raise SystemExit(
                    json.dumps(
                        {
                            "error": (
                                f"Selected batch {selected_batch_name!r} is not a valid TSV manifest "
                                f"at {expected_path}."
                            ),
                            "batch": selected_batch_name,
                            "expected_path": expected_path,
                        },
                        indent=2,
                    )
                ) from None
    available_components = {
        "roi_set": tuple(
            sorted(
                path.stem
                for path in Path(context["roi_sets_dir"]).glob("*.yaml")
                if path.is_file()
            )
        ),
        "extraction_set": tuple(
            sorted(
                path.stem
                for path in Path(context["extraction_sets_dir"]).glob("*.yaml")
                if path.is_file()
            )
        ),
        "mvpa_set": tuple(
            sorted(
                path.stem
                for path in Path(context["mvpa_sets_dir"]).glob("*.yaml")
                if path.is_file()
            )
        ),
    }
    _validate_document, resolve_bundle = _analysis_bundle_functions()
    return resolve_bundle(
        document,
        cohorts_document=cohorts_document,
        batch_tables=batch_tables,
        available_components=available_components,
        expected_name=bundle_name,
    )


def _analysis_mvpa_bundle_handoff(
    *,
    context: dict[str, Any],
    mvpa_name: str,
    bundle_name: str | None,
) -> dict[str, Any] | None:
    """Resolve one requested bundle into the exact units accepted by MVPA planning."""

    if bundle_name is None:
        return None
    normalized_bundle = _normalize_analysis_bundle_name(bundle_name)
    bundle_path = _analysis_bundle_path(context=context, name=normalized_bundle)
    bundle_bytes = bundle_path.read_bytes()
    document = parse_yaml(bundle_bytes.decode("utf-8"), resolve_env=False)
    if not isinstance(document, dict):
        raise SystemExit(
            json.dumps(
                {
                    "error": f"Analysis bundle file {bundle_path.name} must contain a mapping.",
                    "bundle": normalized_bundle,
                    "expected_path": to_workspace_relative(bundle_path, context["workspace_root"]),
                    "next_step": f"rp analysis bundle show {normalized_bundle} --project {context['project_name']}",
                },
                indent=2,
            )
        )
    resolution = _resolve_analysis_bundle_for_cli(
        context=context,
        bundle_name=normalized_bundle,
        document=document,
    )
    observed_mvpa_set = resolution.components.get("mvpa_set")
    if observed_mvpa_set != mvpa_name:
        next_step = f"rp analysis bundle show {normalized_bundle} --project {context['project_name']}"
        raise SystemExit(
            json.dumps(
                {
                    "error": (
                        f"Analysis bundle {normalized_bundle!r} references components.mvpa_set="
                        f"{observed_mvpa_set!r}; expected {mvpa_name!r}. Review it with: {next_step}"
                    ),
                    "bundle": normalized_bundle,
                    "expected_mvpa_set": mvpa_name,
                    "observed_mvpa_set": observed_mvpa_set,
                    "expected_path": to_workspace_relative(bundle_path, context["workspace_root"]),
                    "next_step": next_step,
                },
                indent=2,
            )
        )
    return {
        "name": normalized_bundle,
        "path": bundle_path,
        "bundle_config_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "document": document,
        "resolution": resolution,
        "exact_units": resolution.included_units,
        "unit_key_columns": resolution.key_columns,
    }


def _normalize_roi_config_name(name: str, *, label: str) -> str:
    normalized = _normalize_optional_cli_value(name)
    if normalized is None:
        raise SystemExit(json.dumps({"error": f"{label} name must be a non-empty string."}, indent=2))
    if normalized != Path(normalized).name or normalized.endswith(".yaml") or "/" in normalized or "\\" in normalized:
        raise SystemExit(json.dumps({"error": f"Invalid {label} name {name!r}."}, indent=2))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise SystemExit(
            json.dumps(
                {
                    "error": (
                        f"Invalid {label} name {name!r}. Use only letters, numbers, dots, underscores, and hyphens."
                    )
                },
                indent=2,
            )
        )
    return normalized


def _normalize_roi_template_name(name: str, *, label: str) -> str:
    normalized = _normalize_optional_cli_value(name)
    if normalized is None:
        raise SystemExit(json.dumps({"error": f"{label} must be a non-empty string."}, indent=2))
    if not re.fullmatch(r"[A-Za-z0-9_]+", normalized):
        raise SystemExit(json.dumps({"error": f"Invalid {label} {name!r}."}, indent=2))
    return normalized


def _analysis_roi_set_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["roi_sets_dir"]) / f"{name}.yaml"


def _analysis_roi_set_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_roi_set_config_path(context=context, name=name)
    if not path.exists():
        _raise_missing_analysis_roi_config(
            context=context,
            name=name,
            path=path,
            label="ROI set",
            next_step=(
                f"rp analysis roi init {name} --project {context['project_name']} "
                "--template coordinate_sphere"
            ),
        )
    return path


def _analysis_roi_extraction_set_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["extraction_sets_dir"]) / f"{name}.yaml"


def _analysis_roi_extraction_set_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_roi_extraction_set_config_path(context=context, name=name)
    if not path.exists():
        _raise_missing_analysis_roi_config(
            context=context,
            name=name,
            path=path,
            label="ROI extraction set",
            next_step=(
                f"rp analysis roi extraction init {name} --project {context['project_name']} "
                "--roi-set <roi-set> --template generic_nifti"
            ),
        )
    return path


def _analysis_roi_transform_set_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["transform_sets_dir"]) / f"{name}.yaml"


def _analysis_roi_transform_set_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_roi_transform_set_config_path(context=context, name=name)
    if not path.exists():
        _raise_missing_analysis_roi_config(
            context=context,
            name=name,
            path=path,
            label="ROI transform set",
        )
    return path


def _raise_missing_analysis_roi_config(
    *,
    context: dict[str, Any],
    name: str,
    path: Path,
    label: str,
    next_step: str | None = None,
) -> None:
    expected_path = to_workspace_relative(path, context["workspace_root"])
    message = f"{label} {name!r} was not found at {expected_path}."
    if next_step is not None:
        message = f"{message} Initialize it with: {next_step}"
    payload = {
        "error": message,
        "name": name,
        "expected_path": expected_path,
    }
    if next_step is not None:
        payload["next_step"] = next_step
    raise SystemExit(json.dumps(payload, indent=2))


def _analysis_mvpa_set_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["mvpa_sets_dir"]) / f"{name}.yaml"


def _analysis_mvpa_set_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_mvpa_set_config_path(context=context, name=name)
    if not path.exists():
        _raise_missing_analysis_roi_config(
            context=context,
            name=name,
            path=path,
            label="MVPA set",
            next_step=(
                f"rp analysis mvpa init {name} --project {context['project_name']} "
                "--template materialized-crossnobis"
            ),
        )
    return path


def _analysis_mvpa_document_name(document: dict[str, Any]) -> str | None:
    payload = document.get("mvpa_set")
    if not isinstance(payload, dict):
        payload = document
    return _normalize_optional_cli_value(payload.get("name")) if isinstance(payload, dict) else None


def _analysis_mvpa_filename_errors(document: dict[str, Any], *, expected_name: str) -> list[str]:
    configured_name = _analysis_mvpa_document_name(document)
    if configured_name is None or configured_name == expected_name:
        return []
    return [
        (
            f"MVPA filename/name mismatch: requested {expected_name!r}, but mvpa_set.name is "
            f"{configured_name!r}. Rename the file or update mvpa_set.name so they match."
        )
    ]


def _analysis_mvpa_table_export_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["mvpa_tables_dir"]) / f"{name}.yaml"


def _analysis_mvpa_table_export_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_mvpa_table_export_config_path(context=context, name=name)
    if not path.exists():
        raise SystemExit(json.dumps({"error": f"MVPA table export file does not exist: {path.name}."}, indent=2))
    return path


def _analysis_mvpa_figure_export_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["mvpa_figures_dir"]) / f"{name}.yaml"


def _analysis_mvpa_figure_export_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_mvpa_figure_export_config_path(context=context, name=name)
    if not path.exists():
        raise SystemExit(json.dumps({"error": f"MVPA figure export file does not exist: {path.name}."}, indent=2))
    return path


def _analysis_mvpa_rdm_export_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["mvpa_rdms_dir"]) / f"{name}.yaml"


def _analysis_mvpa_rdm_export_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_mvpa_rdm_export_config_path(context=context, name=name)
    if not path.exists():
        raise SystemExit(json.dumps({"error": f"MVPA RDM export file does not exist: {path.name}."}, indent=2))
    return path


def _analysis_mvpa_publication_export_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["mvpa_publication_dir"]) / f"{name}.yaml"


def _analysis_mvpa_publication_export_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_mvpa_publication_export_config_path(context=context, name=name)
    if not path.exists():
        raise SystemExit(json.dumps({"error": f"MVPA publication export file does not exist: {path.name}."}, indent=2))
    return path


def _analysis_mvpa_derivative_publish_config_path(*, context: dict[str, Any], name: str) -> Path:
    return Path(context["mvpa_derivatives_dir"]) / f"{name}.yaml"


def _analysis_mvpa_derivative_publish_path(*, context: dict[str, Any], name: str) -> Path:
    path = _analysis_mvpa_derivative_publish_config_path(context=context, name=name)
    if not path.exists():
        raise SystemExit(json.dumps({"error": f"MVPA derivative publish file does not exist: {path.name}."}, indent=2))
    return path


def _analysis_mvpa_plan_context(context: dict[str, Any], *, mvpa_path: Path) -> dict[str, Any]:
    return {
        "caller": "research-core-cli",
        "project": context["project_name"],
        "workspace_root": context["workspace_root"],
        "project_root": context["project_root"],
        "config_path": to_workspace_relative(mvpa_path, context["workspace_root"]),
    }


def _analysis_mvpa_lifecycle_state(
    *,
    context: dict[str, Any],
    mvpa_name: str,
    mvpa_path: Path,
    document: dict[str, Any],
    bundle_name: str | None,
    mvpa_config_sha256: str,
) -> dict[str, Any]:
    """Resolve one read-only, exact MVPA lifecycle state for doctor/plan/run."""

    validate_mvpa_set_document, plan_mvpa_discovery = _mvpa_lifecycle_functions()
    schema_errors = _unique_messages(list(validate_mvpa_set_document(document)))
    filename_errors = _analysis_mvpa_filename_errors(document, expected_name=mvpa_name)
    payload = _analysis_mvpa_payload(document)
    unit_selection = payload.get("unit_selection") if isinstance(payload, dict) else None
    exact_mode = (
        isinstance(unit_selection, dict)
        and _normalize_optional_cli_value(unit_selection.get("mode")) == "exact_units"
    )
    handoff = _analysis_mvpa_bundle_handoff(
        context=context,
        mvpa_name=mvpa_name,
        bundle_name=_normalize_optional_cli_value(bundle_name),
    )
    bundle_errors: list[str] = []
    if exact_mode and handoff is None:
        bundle_errors.append(
            "Exact-unit MVPA execution requires --bundle <bundle>. "
            f"List configured bundles with: rp analysis bundle list --project {context['project_name']}"
        )
    resolution = handoff.get("resolution") if handoff is not None else None
    if resolution is not None:
        bundle_errors.extend(str(value) for value in getattr(resolution, "errors", ()))
    bundle_valid = (not exact_mode and handoff is None) or (
        handoff is not None and resolution is not None and bool(getattr(resolution, "valid", False))
    )

    roots = _analysis_mvpa_root_refs(context)
    roi_sets, missing_roi_sets = _analysis_mvpa_referenced_roi_sets(document, context=context)
    plan = plan_mvpa_discovery(
        document,
        context=_analysis_mvpa_plan_context(context, mvpa_path=mvpa_path),
        roots=roots,
        roi_sets=roi_sets,
        enable_backend_discovery=True,
        exact_units=handoff.get("exact_units") if handoff is not None else None,
        unit_key_columns=handoff.get("unit_key_columns") if handoff is not None else None,
    )
    plan_payload = plan.to_dict()
    plan_errors = _unique_messages(list(plan_payload.get("errors", [])))
    schema_valid = not schema_errors and bool(plan_payload.get("schema_valid", False))
    filename_valid = not filename_errors
    source_plan_valid = (
        schema_valid
        and filename_valid
        and bundle_valid
        and not plan_errors
        and str(plan_payload.get("status") or "") not in {"invalid", "error", "skipped-all"}
    )
    representation_kind, representation_errors = _analysis_mvpa_plan_representation(plan_payload)
    runtime_root, runtime_root_errors = _analysis_mvpa_runtime_root_preview(
        document,
        context=context,
        roots=roots,
        mvpa_name=mvpa_name,
    )
    transaction_plan = None
    transaction_errors: list[str] = []
    if runtime_root is not None and representation_kind is not None:
        plan_transaction, _execute_transaction, _runtime_specs = _mvpa_runtime_transaction_functions()
        named_root = roots.get(str(runtime_root["root_ref"]))
        if named_root is None:
            transaction_errors.append("The configured MVPA runtime named root is unavailable.")
        else:
            transaction_plan = plan_transaction(
                named_root=named_root,
                final_root=Path(str(runtime_root["path"])),
                representation_kind=representation_kind,
                existing_output=_analysis_mvpa_runtime_existing_output(document),
            )
            transaction_errors.extend(str(value) for value in transaction_plan.errors)
    else:
        transaction_errors.extend(runtime_root_errors)
    runtime_payload = transaction_plan.to_dict() if transaction_plan is not None else {
        "valid": False,
        "representation_kind": representation_kind,
        "existing_output": _analysis_mvpa_runtime_existing_output(document),
        "outputs": [],
        "collision_paths": [],
        "errors": _unique_messages([*runtime_root_errors, *transaction_errors]),
        "executed": False,
    }
    errors = _unique_messages(
        [
            *schema_errors,
            *filename_errors,
            *bundle_errors,
            *plan_errors,
            *representation_errors,
            *runtime_root_errors,
            *transaction_errors,
            *[f"Referenced ROI set is missing: {name}." for name in missing_roi_sets],
        ]
    )
    plan_valid = source_plan_valid and not representation_errors and not runtime_root_errors and not transaction_errors
    ready_for_materialization = source_plan_valid and bool(plan_payload.get("ready_for_materialization", False))
    ready_for_execution = (
        plan_valid
        and bool(plan_payload.get("ready_for_execution", False))
        and transaction_plan is not None
        and bool(transaction_plan.valid)
        and not missing_roi_sets
    )
    bundle_payload = _analysis_mvpa_bundle_payload(
        context=context,
        handoff=handoff,
        exact_mode=exact_mode,
    )
    checks = _analysis_mvpa_doctor_checks(
        schema_valid=schema_valid,
        schema_errors=schema_errors,
        filename_valid=filename_valid,
        filename_errors=filename_errors,
        exact_mode=exact_mode,
        bundle_valid=bundle_valid,
        bundle_errors=bundle_errors,
        plan_errors=plan_errors,
        plan_payload=plan_payload,
        representation_kind=representation_kind,
        runtime_root_valid=runtime_root is not None and not runtime_root_errors,
        runtime_root_errors=runtime_root_errors,
        transaction_valid=transaction_plan is not None and bool(transaction_plan.valid),
        transaction_errors=transaction_errors,
        missing_roi_sets=missing_roi_sets,
        ready_for_execution=ready_for_execution,
    )
    evidence_ready = all(
        bool(check.get("ok"))
        for check in checks
        if check.get("id") != "execution_readiness"
    )
    ready_for_execution = ready_for_execution and evidence_ready
    if not ready_for_execution:
        checks = _analysis_mvpa_doctor_checks(
            schema_valid=schema_valid,
            schema_errors=schema_errors,
            filename_valid=filename_valid,
            filename_errors=filename_errors,
            exact_mode=exact_mode,
            bundle_valid=bundle_valid,
            bundle_errors=bundle_errors,
            plan_errors=plan_errors,
            plan_payload=plan_payload,
            representation_kind=representation_kind,
            runtime_root_valid=runtime_root is not None and not runtime_root_errors,
            runtime_root_errors=runtime_root_errors,
            transaction_valid=transaction_plan is not None and bool(transaction_plan.valid),
            transaction_errors=transaction_errors,
            missing_roi_sets=missing_roi_sets,
            ready_for_execution=False,
        )
    return {
        "plan": plan,
        "plan_payload": plan_payload,
        "schema_valid": schema_valid,
        "bundle_valid": bundle_valid,
        "plan_valid": plan_valid,
        "ready_for_materialization": ready_for_materialization,
        "ready_for_execution": ready_for_execution,
        "representation_kind": representation_kind,
        "handoff": handoff,
        "bundle_payload": bundle_payload,
        "roots": roots,
        "roi_sets": roi_sets,
        "missing_roi_sets": missing_roi_sets,
        "runtime_root": runtime_root,
        "transaction_plan": transaction_plan,
        "runtime_payload": runtime_payload,
        "mvpa_config_sha256": mvpa_config_sha256,
        "plan_counts": _analysis_mvpa_plan_counts(
            plan_payload=plan_payload,
            bundle_payload=bundle_payload,
        ),
        "cv_contract": _analysis_mvpa_plan_cv_contract(plan_payload),
        "checks": checks,
        "warnings": _unique_messages(list(plan_payload.get("warnings", []))),
        "errors": errors,
    }


def _analysis_mvpa_plan_representation(plan_payload: dict[str, Any]) -> tuple[str | None, list[str]]:
    rows = plan_payload.get("pattern_rows")
    kinds = _unique_messages(
        [
            str(row.get("representation_kind"))
            for row in rows
            if isinstance(row, dict) and row.get("representation_kind")
        ]
        if isinstance(rows, list)
        else []
    )
    if len(kinds) == 1 and kinds[0] in {"image", "prepared_features"}:
        return kinds[0], []
    if len(kinds) > 1:
        return None, ["MVPA execution does not support mixed image and prepared_features sources in v1."]
    return None, ["MVPA planning did not produce one implemented image or prepared_features representation."]


def _analysis_mvpa_plan_counts(
    *,
    plan_payload: dict[str, Any],
    bundle_payload: dict[str, Any] | None,
) -> dict[str, int]:
    bundle = bundle_payload or {}
    roi_identities = {
        (
            str((row.get("backend_metadata") or {}).get("roi_source_name") or ""),
            str((row.get("backend_metadata") or {}).get("roi_label") or ""),
        )
        for row in plan_payload.get("pattern_rows", [])
        if isinstance(row, dict) and isinstance(row.get("backend_metadata"), dict)
    }
    return {
        "included_units": len(bundle.get("included_units", [])),
        "excluded_units": len(bundle.get("excluded_units", [])),
        "pattern_sources": len(plan_payload.get("pattern_sources", [])),
        "pattern_rows": len(plan_payload.get("pattern_rows", [])),
        "conditions": len(plan_payload.get("conditions", [])),
        "condition_pairs": len(plan_payload.get("condition_pairs", [])),
        "roi_sources": len(plan_payload.get("roi_sources", [])),
        "roi_identities": len({value for value in roi_identities if any(value)}),
    }


def _analysis_mvpa_plan_cv_contract(plan_payload: dict[str, Any]) -> dict[str, Any]:
    requests = _analysis_mvpa_distance_requests(plan_payload)
    units = _unique_messages([str(request["cv_unit"]) for request in requests])
    representations, _errors = _analysis_mvpa_plan_representation(plan_payload)
    return {
        "units": units,
        "label_column": (
            "cross_validation_label"
            if representations == "prepared_features"
            else (_analysis_mvpa_default_cv_label_column(units[0]) if len(units) == 1 else None)
        ),
    }


def _analysis_mvpa_runtime_existing_output(document: dict[str, Any]) -> str:
    payload = _analysis_mvpa_payload(document)
    runtime = payload.get("runtime") if isinstance(payload, dict) else None
    value = _normalize_optional_cli_value(runtime.get("existing_output")) if isinstance(runtime, dict) else None
    return value or "fail"


def _analysis_mvpa_bundle_payload(
    *,
    context: dict[str, Any],
    handoff: dict[str, Any] | None,
    exact_mode: bool,
) -> dict[str, Any] | None:
    if handoff is None:
        return {
            "required": exact_mode,
            "mode": "exact_units" if exact_mode else "legacy_cartesian",
            "valid": not exact_mode,
            "included_units": [],
            "excluded_units": [],
            "unit_key_columns": [],
        }
    resolution = handoff["resolution"]
    return {
        "required": exact_mode,
        "mode": "exact_units",
        "name": handoff["name"],
        "path": to_workspace_relative(handoff["path"], context["workspace_root"]),
        "valid": bool(resolution.valid),
        "unit_key_columns": list(resolution.key_columns),
        "included_units": _json_safe_cli_value(resolution.included_units),
        "excluded_units": _json_safe_cli_value(resolution.excluded_units),
        "not_included_units": _json_safe_cli_value(resolution.not_included_units),
        "dropped_units": _json_safe_cli_value(resolution.dropped_units),
        "incomplete_subjects": _json_safe_cli_value(resolution.incomplete_subjects),
        "components": _json_safe_cli_value(resolution.components),
        "stages": list(resolution.stages),
        "counts": _json_safe_cli_value(resolution.counts),
        "digests": {
            "bundle_config_sha256": handoff["bundle_config_sha256"],
            "source_batch_sha256": resolution.source_batch_sha256,
            "effective_selection_sha256": resolution.effective_config_sha256,
            "bundle_plan_sha256": resolution.plan_digest,
        },
    }


def _analysis_mvpa_doctor_checks(
    *,
    schema_valid: bool,
    schema_errors: list[str],
    filename_valid: bool,
    filename_errors: list[str],
    exact_mode: bool,
    bundle_valid: bool,
    bundle_errors: list[str],
    plan_errors: list[str],
    plan_payload: dict[str, Any],
    representation_kind: str | None,
    runtime_root_valid: bool,
    runtime_root_errors: list[str],
    transaction_valid: bool,
    transaction_errors: list[str],
    missing_roi_sets: list[str],
    ready_for_execution: bool,
) -> list[dict[str, Any]]:
    pattern_rows = tuple(
        row for row in plan_payload.get("pattern_rows", []) if isinstance(row, dict)
    )
    source_summaries = tuple(
        row for row in plan_payload.get("pattern_source_summaries", []) if isinstance(row, dict)
    )
    input_checks = tuple(
        row for row in plan_payload.get("input_checks", []) if isinstance(row, dict)
    )
    adapter_rows = tuple(
        row for row in plan_payload.get("adapter_availability", []) if isinstance(row, dict)
    )
    threshold_rows = tuple(
        row for row in plan_payload.get("event_threshold_rows", []) if isinstance(row, dict)
    )
    coverage = bool(plan_payload.get("analysis_units")) and bool(pattern_rows)
    materialized_summaries = tuple(
        row
        for row in source_summaries
        if row.get("backend") == "materialized_pattern_table"
    )
    usable_row_counts = tuple(
        (row.get("counts") or {}).get("usable_selected_rows")
        for row in materialized_summaries
    )
    usable_coverage_ready = (
        bool(materialized_summaries)
        and all(bool(row.get("usable_coverage_complete")) for row in materialized_summaries)
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in usable_row_counts
        )
    )
    selected_coverage_ready = coverage and (
        usable_coverage_ready
        if representation_kind == "prepared_features"
        else True
    )
    has_digest = any(bool(row.get("source_sha256")) for row in source_summaries)
    adapter_ready = bool(plan_payload.get("adapter_availability")) and all(
        bool(row.get("registered")) and bool(row.get("ready_for_execution"))
        for row in adapter_rows
    )
    adapter_findings = [
        str(row.get("reason") or f"Adapter {row.get('backend')!r} is not execution-ready.")
        for row in adapter_rows
        if not bool(row.get("ready_for_execution"))
    ]
    root_checks = tuple(row for row in input_checks if row.get("input_kind") == "root")
    materialized_roots_ready = bool(source_summaries) and all(
        bool(row.get("root_ref"))
        and bool(row.get("portable_reference"))
        and bool(row.get("source_sha256"))
        for row in source_summaries
        if row.get("backend") == "materialized_pattern_table"
    )
    image_roots_ready = bool(root_checks) and all(
        str(row.get("status")) == "ok" and row.get("exists") is not False
        for row in root_checks
    )
    source_root_ready = (
        materialized_roots_ready
        if representation_kind == "prepared_features"
        else image_roots_ready
    )
    bad_input_checks = tuple(
        row
        for row in input_checks
        if str(row.get("status")) in {"error", "missing", "not_checked", "preview_only", "skipped"}
        or row.get("exists") is False
    )
    required_inputs_ready = selected_coverage_ready and not bad_input_checks and (
        has_digest if representation_kind == "prepared_features" else bool(input_checks)
    )
    prepared_metadata = tuple(
        row.get("backend_metadata")
        for row in pattern_rows
        if isinstance(row.get("backend_metadata"), dict)
    )
    condition_roi_ready = selected_coverage_ready and all(
        bool(row.get("condition_id")) for row in pattern_rows
    )
    if representation_kind == "prepared_features":
        condition_roi_ready = condition_roi_ready and len(prepared_metadata) == len(pattern_rows) and all(
            bool(metadata.get("roi_source_name")) and bool(metadata.get("roi_label"))
            for metadata in prepared_metadata
        )
    else:
        condition_roi_ready = condition_roi_ready and bool(plan_payload.get("roi_source_rows"))
    feature_space_ready = representation_kind == "prepared_features" and len(prepared_metadata) == len(pattern_rows) and all(
        bool(metadata.get("feature_count"))
        and bool(metadata.get("voxel_index_hash"))
        and bool(metadata.get("feature_space_id"))
        and bool(metadata.get("roi_definition_id"))
        for metadata in prepared_metadata
    )
    cv_ready = selected_coverage_ready and all(
        bool(row.get("cross_validation_label")) for row in pattern_rows
    )
    centering = plan_payload.get("mean_centering")
    centering_ready = representation_kind == "prepared_features" and isinstance(centering, dict) and all(
        metadata.get("mean_centering_applied") == bool(centering.get("enabled"))
        and str(metadata.get("mean_centering_scope")) == str(centering.get("scope"))
        for metadata in prepared_metadata
    )
    thresholds_ready = all(str(row.get("status")) == "passed" for row in threshold_rows)
    threshold_findings = [
        (
            f"{row.get('threshold')}={row.get('value')} is {row.get('status')}: "
            f"{row.get('reason') or 'no evaluation reason was reported'}."
        )
        for row in threshold_rows
        if str(row.get("status")) != "passed"
    ]
    noise_methods = {
        str(row.get("noise_normalization_method"))
        for row in plan_payload.get("distances", [])
        if isinstance(row, dict)
    }
    noise_ready = representation_kind == "prepared_features" and len(prepared_metadata) == len(pattern_rows)
    if noise_methods == {"identity"}:
        noise_ready = noise_ready and all(
            metadata.get("noise_status") == "unused" and metadata.get("noise_usable") is False
            for metadata in prepared_metadata
        )
    elif noise_methods == {"diagonal"}:
        noise_ready = noise_ready and all(
            metadata.get("noise_status") in {"ok", "warning", "usable"}
            and metadata.get("noise_usable") is True
            for metadata in prepared_metadata
        )
    else:
        noise_ready = False
    external_inputs_ready = required_inputs_ready and (
        representation_kind == "prepared_features" or adapter_ready
    )
    transaction_supported = representation_kind == "prepared_features" and transaction_valid
    values = (
        ("configuration_schema", schema_valid, "MVPA configuration schema is valid.", "Repair the MVPA YAML schema.", schema_errors),
        ("configuration_name", filename_valid, "Filename and mvpa_set.name agree.", "Make the filename and mvpa_set.name agree.", filename_errors),
        ("bundle_reference", bundle_valid, "Bundle selection is valid or not required for compatibility mode.", "Supply a valid --bundle whose components.mvpa_set matches this set.", bundle_errors),
        ("exact_unit_resolution", (not exact_mode) or coverage, "Exact units resolved in configured order.", "Resolve at least one exact included bundle unit.", bundle_errors or plan_errors),
        ("adapter_availability", adapter_ready, "All selected adapters have implemented local execution boundaries.", "Choose an implemented adapter or complete its runtime integration.", adapter_findings),
        ("source_root", source_root_ready, "Configured source roots are available and portable.", "Configure every required named source root.", plan_errors),
        ("required_inputs", required_inputs_ready, "Required source inputs passed planning checks.", "Provide the required table, image, ROI, or external input.", [str(row.get("message") or row.get("input_kind")) for row in bad_input_checks]),
        ("source_digest", representation_kind == "image" or has_digest, "The source digest contract is available.", "Replan a readable materialized source table to obtain its digest.", plan_errors),
        ("selected_coverage", coverage, "Selected unit rows have planned pattern coverage.", "Provide complete selected unit pattern coverage.", plan_errors),
        ("condition_roi_coverage", condition_roi_ready, "Condition and ROI coverage is complete and usable.", "Provide complete usable, non-duplicate condition/ROI rows.", plan_errors),
        ("feature_space_consistency", feature_space_ready, "Feature-space and ROI identities are coherent at planning time.", "Provide portable feature-space, feature-index, and ROI-definition identities.", plan_errors),
        ("canonical_cv_labels", cv_ready, "Canonical cross-validation labels are coherent.", "Repair missing or mismatched CV labels.", plan_errors),
        ("centering_contract", centering_ready, "Centering state and scope match configuration.", "Repair the centering declaration or source rows.", plan_errors),
        ("event_thresholds", thresholds_ready, "Configured event thresholds passed source planning.", "Provide event counts and run coverage that satisfy configured thresholds.", threshold_findings),
        ("noise_requirements", noise_ready, "Noise metadata matches the configured normalization method.", "Repair variance identity, scope, positivity, or width.", plan_errors),
        ("external_runtime_inputs", external_inputs_ready, "Required local/external inputs are ready for the selected representation.", "Provide the missing external runtime or input files.", adapter_findings),
        ("runtime_root", runtime_root_valid, "Runtime root is contained beneath its named root.", "Choose a safe relative runtime root beneath a configured named root.", runtime_root_errors),
        ("output_collisions", transaction_valid, "The complete runtime output set has no collision.", "Choose a new runtime root; existing_output=fail never overwrites.", transaction_errors),
        ("transaction_support", transaction_supported, "Whole-runtime-root transactional execution is implemented.", "Prepared-feature runtime transactions are supported; image execution remains deferred.", transaction_errors),
        ("execution_readiness", ready_for_execution, "All execution-readiness checks passed.", "Resolve the failing checks before using --execute.", [*adapter_findings, *threshold_findings, *missing_roi_sets]),
    )
    return [
        {
            "id": check_id,
            "status": "pass" if ok else "fail",
            "ok": bool(ok),
            "messages": [success if ok else failure],
            "findings": [] if ok else _unique_messages([str(value) for value in findings if str(value)]),
        }
        for check_id, ok, success, failure, findings in values
    ]


def _json_safe_cli_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_cli_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_cli_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe_cli_value(to_dict())
    return str(value)


def _analysis_roi_transform_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], str, Path, dict[str, Any], dict[str, Any]]:
    context = _build_analysis_roi_project_context(_resolve_project_name(args.project))
    transform_name = _normalize_roi_config_name(args.name, label="ROI transform set")
    transform_path = _analysis_roi_transform_set_path(context=context, name=transform_name)
    raw_document = _load_analysis_roi_document(transform_path, config_label="ROI transform set", resolve_env=False)
    document = _load_analysis_roi_document(transform_path, config_label="ROI transform set")
    return context, transform_name, transform_path, document, raw_document


def _analysis_roi_transform_plan_context(context: dict[str, Any], *, transform_path: Path) -> dict[str, Any]:
    return {
        "caller": "research-core-cli",
        "project": context["project_name"],
        "workspace_root": context["workspace_root"],
        "project_root": context["project_root"],
        "config_path": to_workspace_relative(transform_path, context["workspace_root"]),
    }


def _validate_analysis_roi_transform_document(document: dict[str, Any]) -> list[str]:
    try:
        from research_platform.neuro.roi_transforms import validate_mni_to_t1w_roi_transform_document
    except ImportError as exc:
        raise SystemExit(json.dumps({"error": f"ROI transform validation requires research-neuro: {exc}"}, indent=2)) from exc

    errors = list(validate_mni_to_t1w_roi_transform_document(document))
    _collect_absolute_path_literal_errors(document, "roi_transform_set", errors)
    return _unique_messages(errors)


def _collect_absolute_path_literal_errors(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_absolute_path_literal_errors(child, f"{label}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_absolute_path_literal_errors(child, f"{label}[{index}]", errors)
    elif isinstance(value, str) and _looks_like_absolute_path_literal(value):
        errors.append(f"{label} contains an absolute path literal.")


def _looks_like_absolute_path_literal(value: str) -> bool:
    text = value.strip()
    if text.startswith("${"):
        return False
    return bool(
        re.search(r"(^|[=:\s])/(?!/)[^\s]*", text)
        or re.search(r"(^|[=:\s])~(?:[^/\s]+)?(?:/|$)", text)
        or re.search(r"(^|[=:\s])[A-Za-z]:[\\/][^\s]*", text)
        or re.search(r"(^|[=:\s])\\\\[^\\\s]+\\[^\s]+", text)
    )


def _analysis_mvpa_root_refs(context: dict[str, Any]) -> dict[str, Path]:
    root_refs = _analysis_roi_root_refs(context)
    mvpa_config_root = Path(context["mvpa_sets_dir"])
    root_refs["mvpa_config_root"] = mvpa_config_root
    root_refs["project_mvpa_root"] = mvpa_config_root
    return {name: _analysis_roi_lexical_absolute_path(path) for name, path in root_refs.items()}


def _analysis_mvpa_referenced_roi_sets(
    document: dict[str, Any],
    *,
    context: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    roi_sets: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for roi_set_ref in _analysis_mvpa_referenced_roi_set_names(document):
        roi_path = _analysis_roi_set_config_path(context=context, name=roi_set_ref)
        if not roi_path.exists():
            missing.append(roi_set_ref)
            continue
        roi_sets[roi_set_ref] = _load_analysis_roi_document(roi_path, config_label="ROI set", resolve_env=False)
    return roi_sets, missing


def _analysis_mvpa_referenced_roi_set_names(document: dict[str, Any]) -> list[str]:
    payload = _analysis_mvpa_payload(document)
    sources = payload.get("roi_sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list):
        return []
    names: list[str] = []
    seen: set[str] = set()
    roi_set_source_types = {"roi_set", "roi_set_runtime", "roi_set_publication"}
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_type = _normalize_optional_cli_value(source.get("source"))
        roi_set_ref = _normalize_optional_cli_value(source.get("roi_set_ref") or source.get("roi_set"))
        if roi_set_ref is None or (source_type is not None and source_type not in roi_set_source_types):
            continue
        if roi_set_ref != Path(roi_set_ref).name or not re.fullmatch(r"[A-Za-z0-9_.-]+", roi_set_ref):
            continue
        if roi_set_ref not in seen:
            names.append(roi_set_ref)
            seen.add(roi_set_ref)
    return names


def _analysis_mvpa_payload(document: dict[str, Any]) -> dict[str, Any]:
    payload = document.get("mvpa_set")
    return payload if isinstance(payload, dict) else document


_MVPA_RUNTIME_OUTPUT_RELATIVE_PATHS: tuple[tuple[str, str], ...] = (
    ("neuro_patterns_tsv", "neuro/pattern-extraction/patterns.tsv"),
    ("neuro_pattern_qc_tsv", "neuro/pattern-extraction/qc.tsv"),
    ("neuro_pattern_provenance_json", "neuro/pattern-extraction/provenance.json"),
    ("neuro_pattern_vector_metadata_json", "neuro/pattern-extraction/vector_metadata.json"),
    ("analysis_prepared_pattern_rows_tsv", "analysis/prepared-patterns/rows.tsv"),
    ("analysis_prepared_pattern_qc_tsv", "analysis/prepared-patterns/qc.tsv"),
    ("analysis_prepared_pattern_provenance_json", "analysis/prepared-patterns/provenance.json"),
    ("analysis_prepared_distance_rows_tsv", "analysis/prepared-distances/distances.tsv"),
    ("analysis_prepared_distance_qc_tsv", "analysis/prepared-distances/qc.tsv"),
    ("analysis_prepared_distance_provenance_json", "analysis/prepared-distances/provenance.json"),
    ("analysis_prepared_summary_rows_tsv", "analysis/prepared-summaries/summaries.tsv"),
    ("analysis_prepared_summary_qc_tsv", "analysis/prepared-summaries/qc.tsv"),
    ("analysis_prepared_summary_provenance_json", "analysis/prepared-summaries/provenance.json"),
)


_MVPA_PUBLISH_REQUIRED_RUNTIME_INPUT_RELATIVE_PATHS: tuple[tuple[str, str], ...] = (
    ("analysis_prepared_summary_rows_tsv", "analysis/prepared-summaries/summaries.tsv"),
    ("analysis_prepared_summary_qc_tsv", "analysis/prepared-summaries/qc.tsv"),
    ("analysis_prepared_summary_provenance_json", "analysis/prepared-summaries/provenance.json"),
    ("analysis_prepared_distance_rows_tsv", "analysis/prepared-distances/distances.tsv"),
    ("analysis_prepared_distance_qc_tsv", "analysis/prepared-distances/qc.tsv"),
    ("analysis_prepared_distance_provenance_json", "analysis/prepared-distances/provenance.json"),
)


_MVPA_PUBLISH_OPTIONAL_RUNTIME_INPUT_RELATIVE_PATHS: tuple[tuple[str, str], ...] = (
    ("neuro_pattern_provenance_json", "neuro/pattern-extraction/provenance.json"),
    ("analysis_prepared_pattern_provenance_json", "analysis/prepared-patterns/provenance.json"),
)


_MVPA_RUN_PLANNED_STEPS: tuple[str, ...] = (
    "validate_config",
    "resolve_analysis_bundle",
    "plan_discovery",
    "preflight_complete_output_set",
    "materialize_or_extract_patterns",
    "prepare_pattern_rows",
    "compute_distances",
    "summarize_distances",
    "stage_complete_runtime_tree",
    "validate_staged_runtime_tree",
    "promote_runtime_tree",
)


_MVPA_PUBLISH_PLANNED_STEPS: tuple[str, ...] = (
    "validate_config",
    "resolve_runtime_root",
    "resolve_publication_root",
    "inspect_runtime_inputs",
    "preview_publication_outputs",
    "publish_tables",
    "publish_metadata",
)


def _analysis_mvpa_runtime_root_preview(
    document: dict[str, Any],
    *,
    context: dict[str, Any],
    roots: dict[str, Path],
    mvpa_name: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = _analysis_mvpa_payload(document)
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if not isinstance(outputs, dict):
        return None, ["mvpa_set.outputs.runtime_root must be defined for MVPA runs."]

    runtime_root = outputs.get("runtime_root")
    if not isinstance(runtime_root, dict):
        return None, ["mvpa_set.outputs.runtime_root must be defined for MVPA runs."]

    return _analysis_mvpa_declared_root_preview(
        runtime_root,
        context=context,
        roots=roots,
        mvpa_name=mvpa_name,
        label="mvpa_set.outputs.runtime_root",
    )


def _analysis_mvpa_publication_root_preview(
    document: dict[str, Any],
    *,
    context: dict[str, Any],
    roots: dict[str, Path],
    mvpa_name: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = _analysis_mvpa_payload(document)
    publication = payload.get("publication") if isinstance(payload, dict) else None
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    errors: list[str] = []
    if not isinstance(publication, dict) or not _normalize_config_bool(publication.get("enabled"), default=False):
        errors.append("mvpa_set.publication.enabled must be true for MVPA publication previews.")

    publication_root = publication.get("root") if isinstance(publication, dict) else None
    root_label = "mvpa_set.publication.root"
    if publication_root is None:
        publication_root = outputs.get("published_root") if isinstance(outputs, dict) else None
        root_label = "mvpa_set.outputs.published_root"
    elif not isinstance(publication_root, dict):
        errors.append("mvpa_set.publication.root must contain a mapping when declared.")

    if publication_root is None:
        errors.append("mvpa_set.publication.root or mvpa_set.outputs.published_root must be defined for MVPA publication previews.")
        return None, _unique_messages(errors)
    if not isinstance(publication_root, dict):
        return None, _unique_messages(errors)

    root_preview, root_errors = _analysis_mvpa_declared_root_preview(
        publication_root,
        context=context,
        roots=roots,
        mvpa_name=mvpa_name,
        label=root_label,
    )
    return root_preview, _unique_messages([*errors, *root_errors])


def _analysis_mvpa_declared_root_preview(
    declaration: dict[str, Any],
    *,
    context: dict[str, Any],
    roots: dict[str, Path],
    mvpa_name: str,
    label: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    root_ref = _normalize_optional_cli_value(declaration.get("root_ref"))
    relative_path_template = _normalize_optional_cli_value(declaration.get("path") or declaration.get("relative_path"))
    if root_ref is None:
        errors.append(f"{label}.root_ref must be defined.")
    elif root_ref not in roots:
        errors.append(f"{label}.root_ref {root_ref!r} is not a known root_ref.")
    if relative_path_template is None:
        errors.append(f"{label}.path must be defined.")
    elif _is_absolute_or_parent_relative_path(relative_path_template):
        errors.append(f"{label}.path must be a relative path under root_ref.")

    if errors:
        return None, errors

    assert root_ref is not None
    assert relative_path_template is not None
    relative_path = _render_analysis_mvpa_runtime_path_template(
        relative_path_template,
        mvpa_name=mvpa_name,
        project_name=str(context["project_name"]),
    )
    root_path = _analysis_roi_lexical_absolute_path(roots[root_ref] / relative_path)
    return (
        {
            "root_ref": root_ref,
            "relative_path_template": relative_path_template,
            "relative_path": relative_path,
            "path": str(root_path),
        },
        [],
    )


def _analysis_mvpa_runtime_output_previews(
    runtime_root: dict[str, Any] | None,
    *,
    representation_kind: str | None = None,
    executed: bool = False,
) -> dict[str, dict[str, Any]]:
    root_path = Path(str(runtime_root["path"])) if runtime_root is not None else None
    if representation_kind in {"image", "prepared_features"}:
        _plan_transaction, _execute_transaction, runtime_specs = _mvpa_runtime_transaction_functions()
        relative_paths = tuple(
            (spec.name, spec.relative_path) for spec in runtime_specs(representation_kind)
        )
    else:
        relative_paths = _MVPA_RUNTIME_OUTPUT_RELATIVE_PATHS
    previews: dict[str, dict[str, Any]] = {}
    for name, relative_path in relative_paths:
        previews[name] = {
            "relative_path": relative_path,
            "path": str(root_path / relative_path) if root_path is not None else None,
            "status": "written" if executed else "planned",
            "executed": executed,
        }
    return previews


def _analysis_mvpa_runtime_input_previews(
    runtime_root: dict[str, Any] | None,
    relative_paths: tuple[tuple[str, str], ...],
    *,
    required: bool,
) -> dict[str, dict[str, Any]]:
    root_path = Path(str(runtime_root["path"])) if runtime_root is not None else None
    previews: dict[str, dict[str, Any]] = {}
    for name, relative_path in relative_paths:
        path = root_path / relative_path if root_path is not None else None
        exists = bool(path is not None and path.is_file())
        previews[name] = {
            "relative_path": relative_path,
            "path": str(path) if path is not None else None,
            "required": required,
            "exists": exists,
            "status": "present" if exists else "missing",
        }
    return previews


def _analysis_mvpa_publication_output_previews(
    publication_root: dict[str, Any] | None,
    *,
    table_desc: str,
) -> dict[str, dict[str, Any]]:
    root_path = Path(str(publication_root["path"])) if publication_root is not None else None
    relative_paths = (
        ("dataset_description_json", "dataset_description.json"),
        ("readme_md", "README.md"),
        ("group_summary_tsv", f"tables/group/group_desc-{table_desc}_mvpasummary.tsv"),
        ("group_distances_tsv", f"distances/group/desc-{table_desc}Distances_distances.tsv"),
        ("summary_qc_tsv", f"qc/group/desc-{table_desc}SummaryQC_mvpaqc.tsv"),
        ("distance_qc_tsv", f"qc/group/desc-{table_desc}DistanceQC_mvpaqc.tsv"),
        ("provenance_json", f"provenance/desc-{table_desc}_provenance.json"),
        ("manifest_json", "manifest.json"),
    )
    previews: dict[str, dict[str, Any]] = {}
    for name, relative_path in relative_paths:
        previews[name] = {
            "relative_path": relative_path,
            "path": str(root_path / relative_path) if root_path is not None else None,
            "status": "planned",
            "executed": False,
        }
    return previews


def _analysis_mvpa_run_planned_steps() -> list[dict[str, Any]]:
    return [{"name": name, "status": "planned", "executed": False} for name in _MVPA_RUN_PLANNED_STEPS]


def _analysis_mvpa_publish_planned_steps() -> list[dict[str, Any]]:
    return [{"name": name, "status": "planned", "executed": False} for name in _MVPA_PUBLISH_PLANNED_STEPS]


def _analysis_mvpa_publication_table_desc(document: dict[str, Any]) -> tuple[str, list[str]]:
    payload = _analysis_mvpa_payload(document)
    publication = payload.get("publication") if isinstance(payload, dict) else None
    value = _normalize_optional_cli_value(publication.get("table_desc")) if isinstance(publication, dict) else None
    if value is None:
        return "Crossnobis", []
    if re.fullmatch(r"[A-Za-z0-9]+", value):
        return value, []
    return (
        "Crossnobis",
        ["mvpa_set.publication.table_desc must be alphanumeric for planned filenames; using Crossnobis."],
    )


def _render_analysis_mvpa_runtime_path_template(template: str, *, mvpa_name: str, project_name: str) -> str:
    values = {
        "mvpa_set": mvpa_name,
        "mvpa_set_name": mvpa_name,
        "name": mvpa_name,
        "project": project_name,
        "project_name": project_name,
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered.strip("/")


def _is_absolute_or_parent_relative_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        Path(value).expanduser().is_absolute()
        or windows.is_absolute()
        or any(part == ".." for part in (*posix.parts, *windows.parts))
    )


def _unique_messages(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def _analysis_roi_doctor_status(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the doctor validity/readiness split for the CLI contract."""

    normalized = dict(payload)
    schema_valid = bool(normalized.get("schema_valid", normalized.get("valid", False)))
    if "ready_for_execution" in normalized:
        ready_for_execution = bool(normalized["ready_for_execution"])
    else:
        ready_for_execution = _legacy_analysis_roi_doctor_readiness(normalized, schema_valid=schema_valid)
    normalized["valid"] = schema_valid
    normalized["schema_valid"] = schema_valid
    normalized["ready_for_execution"] = ready_for_execution
    return normalized


def _legacy_analysis_roi_doctor_readiness(payload: dict[str, Any], *, schema_valid: bool) -> bool:
    if not schema_valid or payload.get("errors") or payload.get("missing_inputs"):
        return False
    fsl_tools = payload.get("fsl_tools")
    if isinstance(fsl_tools, dict) and bool(fsl_tools.get("required")):
        tools = fsl_tools.get("tools")
        if isinstance(tools, dict) and any(
            not bool(status.get("available"))
            for status in tools.values()
            if isinstance(status, dict)
        ):
            return False
    for check in payload.get("command_checks", ()):
        if isinstance(check, dict) and check.get("errors"):
            return False
    return True


_ANALYSIS_ROI_TRANSFORM_CHECK_ACTIONS = {
    "configuration_valid": "Fix the reported configuration issue, then rerun validate and doctor.",
    "configured_root_available": "Configure or create the required named root before execution.",
    "input_exists": "Populate or correct the required transform input before execution.",
    "external_tool_available": "Install, load, or configure the required transform tool before execution.",
    "output_collision": "Choose unused output paths or deliberately enable replacement in configuration.",
    "execution_readiness": "Correct the reported readiness issue, then rerun doctor before execution.",
}


def _analysis_roi_transform_doctor_checks(
    document: dict[str, Any],
    *,
    roots: dict[str, Path],
    schema_errors: list[str],
    plan_payload: dict[str, Any] | None = None,
    validation_payload: dict[str, Any] | None = None,
    ready_for_execution: bool,
) -> list[dict[str, Any]]:
    """Summarize planner and execution-validator evidence without re-planning."""

    checks = [
        _analysis_roi_transform_check(
            "configuration_valid",
            status="error" if schema_errors else "ok",
            message=(
                f"ROI transform configuration has {len(schema_errors)} structural issue(s)."
                if schema_errors
                else "ROI transform configuration is structurally valid."
            ),
            category="configuration",
        )
    ]
    if schema_errors:
        blocked = (
            ("configured_root_available", "Named-root availability", "root"),
            ("input_exists", "Transform input checks", "input"),
            ("external_tool_available", "Transform-tool checks", "external_tool"),
            ("output_collision", "Output collision checks", "output"),
            ("execution_readiness", "Execution readiness", "execution"),
        )
        checks.extend(
            _analysis_roi_transform_check(
                check_id,
                status="blocked",
                message=f"{label} cannot be evaluated until the configuration is valid.",
                category=category,
            )
            for check_id, label, category in blocked
        )
        return checks

    plan = plan_payload or {}
    validation = validation_payload or {}
    root_refs = sorted(_analysis_roi_transform_root_refs_in_document(document))
    missing_roots = [root_ref for root_ref in root_refs if root_ref not in roots]
    checks.append(
        _analysis_roi_transform_check(
            "configured_root_available",
            status="error" if missing_roots else "ok",
            message=(
                "Missing named root(s): " + ", ".join(missing_roots) + "."
                if missing_roots
                else f"All {len(root_refs)} referenced named root(s) are configured."
            ),
            category="root",
        )
    )

    input_rows: list[dict[str, Any]] = []
    for collection, path_key, category in (
        ("source_masks", "source_mask_path", "source_mask"),
        ("target_references", "target_reference_path", "target_reference"),
        ("transform_chains", "transform_path", "transform"),
    ):
        rows = plan.get(collection, [])
        if isinstance(rows, list):
            input_rows.extend(
                {
                    "category": category,
                    "path": row.get(path_key),
                    "available": row.get("exists") is True,
                }
                for row in rows
                if isinstance(row, dict)
            )
    qc_rows = plan.get("qc_preview", [])
    if isinstance(qc_rows, list):
        input_rows.extend(
            {
                "category": "coverage_mask",
                "path": row.get("path"),
                "available": row.get("status") == "ok",
            }
            for row in qc_rows
            if isinstance(row, dict) and row.get("check_kind") == "coverage_mask_exists"
        )
    missing_inputs = [row for row in input_rows if not row["available"]]
    checks.append(
        _analysis_roi_transform_check(
            "input_exists",
            status="error" if missing_inputs or not input_rows else "ok",
            message=(
                f"{len(missing_inputs)} of {len(input_rows)} required transform input(s) are missing or unresolved."
                if missing_inputs
                else (
                    f"All {len(input_rows)} required transform input(s) are available."
                    if input_rows
                    else "No complete transform input set could be planned."
                )
            ),
            category="input",
            details=missing_inputs,
        )
    )

    tool_rows = [row for row in plan.get("tool_preflight", []) if isinstance(row, dict)]
    missing_tools = [row for row in tool_rows if row.get("available") is not True]
    checks.append(
        _analysis_roi_transform_check(
            "external_tool_available",
            status="error" if missing_tools or not tool_rows else "ok",
            message=(
                str(missing_tools[0].get("message"))
                if missing_tools
                else (
                    "The configured transform tool is available."
                    if tool_rows
                    else "No transform-tool availability check was produced."
                )
            ),
            category="external_tool",
        )
    )

    errors = [str(error) for error in (*plan.get("errors", []), *validation.get("errors", []))]
    collision_markers = (
        "already exists",
        "duplicate planned ROI transform destination",
        "symbolic-link",
        "parent is not a directory",
        "destination is not a regular file",
        "escapes its configured output root",
    )
    collision_errors = [error for error in errors if any(marker in error for marker in collision_markers)]
    planned_outputs = plan.get("planned_outputs", [])
    has_output = isinstance(planned_outputs, list) and bool(planned_outputs)
    missing_output = any(
        marker in error
        for error in errors
        for marker in (
            "output path was not supplied",
            "output path is not concrete enough",
        )
    )
    if collision_errors:
        status = "error"
        message = f"Output preflight found {len(_unique_messages(collision_errors))} collision or destination hazard(s)."
    elif not has_output or missing_output:
        status = "error"
        message = "A complete concrete transform output set could not be planned."
    else:
        status = "ok"
        message = "The planned transform output set has no detected collision or destination hazard."
    checks.append(
        _analysis_roi_transform_check(
            "output_collision",
            status=status,
            message=message,
            category="output",
        )
    )
    checks.append(
        _analysis_roi_transform_check(
            "execution_readiness",
            status="ok" if ready_for_execution else "error",
            message=(
                "The ROI transform plan is ready for explicit execution."
                if ready_for_execution
                else "The ROI transform plan is not ready for execution."
            ),
            category="execution",
        )
    )
    return checks


def _analysis_roi_transform_root_refs_in_document(value: Any) -> set[str]:
    root_refs: set[str] = set()
    if isinstance(value, dict):
        root_ref = value.get("root_ref")
        if isinstance(root_ref, str) and root_ref.strip():
            root_refs.add(root_ref.strip())
        for child in value.values():
            root_refs.update(_analysis_roi_transform_root_refs_in_document(child))
    elif isinstance(value, list):
        for child in value:
            root_refs.update(_analysis_roi_transform_root_refs_in_document(child))
    return root_refs


def _analysis_roi_transform_check(
    check_id: str,
    *,
    status: str,
    message: str,
    category: str,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    check = {
        "id": check_id,
        "check_id": check_id,
        "status": status,
        "message": message,
        "category": category,
    }
    if details:
        check["details"] = details
    if status != "ok":
        check["action"] = _ANALYSIS_ROI_TRANSFORM_CHECK_ACTIONS[check_id]
    return check


def _json_path_mapping(paths: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in sorted(paths.items())}


def _write_analysis_roi_scaffold(path: Path, content: str, *, force: bool, dry_run: bool, label: str) -> bool:
    if path.exists() and not force and not dry_run:
        raise SystemExit(
            json.dumps(
                {
                    "error": f"{label} file already exists: {path.name}. Use --force to overwrite it.",
                },
                indent=2,
            )
        )
    if dry_run:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _analysis_roi_set_scaffold_overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "subjects",
        "held_out_subjects",
        "session",
        "task",
        "direction",
        "model",
        "space",
        "resolution",
        "search_radius_mm",
        "sphere_radius_mm",
        "z_threshold",
        "min_voxels_warn",
        "min_voxels_fail",
        "contrasts",
        "rois",
    )
    return _analysis_roi_scaffold_overrides(args, keys=keys)


def _analysis_roi_extraction_scaffold_overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "subjects",
        "session",
        "task",
        "direction",
        "model",
        "space",
        "resolution",
        "metrics",
        "contrasts",
        "roi_labels",
    )
    return _analysis_roi_scaffold_overrides(args, keys=keys)


def _inherit_analysis_roi_publication_defaults(
    document: dict[str, Any],
    *,
    context: dict[str, Any],
    roi_name: str,
    template: str,
) -> dict[str, Any]:
    if template != "fsl_featquery":
        return document
    extraction_set = document.get("extraction_set") if isinstance(document.get("extraction_set"), dict) else document
    if not isinstance(extraction_set, dict) or isinstance(extraction_set.get("publication"), dict):
        return document
    roi_path = _analysis_roi_set_config_path(context=context, name=roi_name)
    if not roi_path.exists():
        return document
    roi_document = _load_analysis_roi_document(roi_path, config_label="ROI set", resolve_env=False)
    roi_set = roi_document.get("roi_set") if isinstance(roi_document.get("roi_set"), dict) else roi_document
    if not isinstance(roi_set, dict):
        return document
    publication = roi_set.get("publication")
    if not isinstance(publication, dict) or not bool(publication.get("enabled")):
        return document
    if publication.get("layout") not in (None, "loso_flame1_bidslike"):
        return document
    inherited = {
        "enabled": True,
        "layout": "loso_flame1_bidslike",
        "root": publication.get("root", {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"}),
        "dataset_description": publication.get(
            "dataset_description",
            {
                "name": "ROI LOSO FLAME1 outputs",
                "generated_by_name": "roi-loso-flame1",
            },
        ),
        "table_desc": publication.get("table_desc", "{model}LOSOFlame1Featquery"),
    }
    if isinstance(publication.get("contrast_aliases"), dict):
        inherited["contrast_aliases"] = dict(publication["contrast_aliases"])
    extraction_set["publication"] = inherited
    return document


def _analysis_roi_scaffold_overrides(args: argparse.Namespace, *, keys: tuple[str, ...]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


def _parse_rendered_analysis_roi_scaffold(content: str, *, config_label: str) -> dict[str, Any]:
    try:
        document = parse_yaml(content, resolve_env=False)
    except ValueError as exc:
        raise ValueError(f"Rendered {config_label} YAML is invalid: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Rendered {config_label} YAML must contain a mapping.")
    return document


def _analysis_mvpa_scaffold_file_records(context: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    project_root = Path(context["project_root"])
    for file_record in plan.get("files", []):
        relative_path = str(file_record["relative_path"])
        candidate = Path(relative_path)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise SystemExit(json.dumps({"error": f"Unsafe MVPA scaffold path: {relative_path}."}, indent=2))
        path = (project_root / candidate).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise SystemExit(json.dumps({"error": f"Unsafe MVPA scaffold path: {relative_path}."}, indent=2)) from exc
        records.append(
            {
                **file_record,
                "path": path,
                "exists": path.exists(),
                "workspace_relative_path": to_workspace_relative(path, context["workspace_root"]),
            }
        )
    return records


def _analysis_mvpa_scaffold_validations(plan: dict[str, Any]) -> list[dict[str, Any]]:
    validate_mvpa_set_document, _plan_mvpa_discovery = _mvpa_lifecycle_functions()
    validate_table, _plan_table = _mvpa_table_export_functions()
    validate_figure, _plan_figure = _mvpa_figure_export_functions()
    validate_rdm, _plan_rdm = _mvpa_rdm_export_functions()
    validate_derivative, _plan_derivative = _mvpa_derivative_publish_functions()
    validators = {
        "runtime": validate_mvpa_set_document,
        "tables": validate_table,
        "figures": validate_figure,
        "rdms": validate_rdm,
        "derivatives": validate_derivative,
    }
    validations: list[dict[str, Any]] = []
    for file_record in plan.get("files", []):
        component = str(file_record.get("component", ""))
        document = file_record.get("document")
        validator = validators.get(component)
        errors = validator(document) if validator is not None and isinstance(document, dict) else []
        validations.append(
            {
                "component": component,
                "relative_path": file_record.get("relative_path"),
                "valid": not errors,
                "errors": errors,
            }
        )
    return validations


def _write_analysis_mvpa_scaffold_file(file_record: dict[str, Any]) -> None:
    path = Path(file_record["path"])
    kind = file_record.get("kind")
    if kind == "yaml":
        document = file_record.get("document")
        if not isinstance(document, dict):
            raise SystemExit(json.dumps({"error": f"Generated YAML payload is invalid for {path.name}."}, indent=2))
        write_yaml(path, document)
        return
    content = file_record.get("content")
    if not isinstance(content, str):
        raise SystemExit(json.dumps({"error": f"Generated scaffold content is invalid for {path.name}."}, indent=2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _analysis_roi_init_next_steps(project_name: str, roi_name: str) -> list[str]:
    return [
        "Edit the generated YAML with the required roots, inputs, labels, and analysis choices.",
        f"Validate with: rp analysis roi validate {roi_name} --project {project_name}",
        f"Check execution readiness: rp analysis roi doctor {roi_name} --project {project_name}",
        f"Review the build plan: rp analysis roi build {roi_name} --project {project_name}",
        f"Execute after reviewing the plan: rp analysis roi build {roi_name} --project {project_name} --execute",
    ]


def _analysis_roi_extraction_init_next_steps(project_name: str, extraction_name: str) -> list[str]:
    return [
        "Edit the generated YAML with the required roots, masks, inputs, contrasts, and metrics.",
        f"Validate with: rp analysis roi extraction validate {extraction_name} --project {project_name}",
        f"Check execution readiness: rp analysis roi extraction doctor {extraction_name} --project {project_name}",
        f"Review the extraction plan: rp analysis roi extraction run {extraction_name} --project {project_name}",
        (
            "Execute after reviewing the plan: rp analysis roi extraction run "
            f"{extraction_name} --project {project_name} --execute"
        ),
    ]


def _load_analysis_roi_document(path: Path, *, config_label: str, resolve_env: bool = True) -> dict[str, Any]:
    document = load_yaml(path, resolve_env=resolve_env)
    if not isinstance(document, dict):
        raise SystemExit(json.dumps({"error": f"{config_label} file {path.name} must contain a mapping."}, indent=2))
    return document


def _load_analysis_mvpa_document(path: Path, *, config_label: str, resolve_env: bool = False) -> dict[str, Any]:
    document = load_yaml(path, resolve_env=resolve_env)
    if not isinstance(document, dict):
        raise SystemExit(json.dumps({"error": f"{config_label} file {path.name} must contain a mapping."}, indent=2))
    return document


def _load_analysis_mvpa_document_with_digest(
    path: Path,
    *,
    config_label: str,
) -> tuple[dict[str, Any], str]:
    """Parse and fingerprint the same immutable MVPA configuration bytes."""

    raw = path.read_bytes()
    document = parse_yaml(raw.decode("utf-8"), resolve_env=False)
    if not isinstance(document, dict):
        raise SystemExit(
            json.dumps(
                {"error": f"{config_label} file {path.name} must contain a mapping."},
                indent=2,
            )
        )
    return document, hashlib.sha256(raw).hexdigest()


def _resolve_analysis_model_authoring_adapter(
    *,
    context: dict[str, Any],
    requested_tool: str | None,
) -> tuple[str, dict[str, Any], Any]:
    requested = _normalize_optional_cli_value(requested_tool)
    defaults = context["analysis"].get("defaults", {})
    default_tool = _normalize_optional_cli_value(defaults.get("tool")) if isinstance(defaults, dict) else None
    tool_name = requested or default_tool
    if tool_name is None:
        raise SystemExit(
            json.dumps(
                {
                    "error": (
                        "Unable to infer an analysis tool for model authoring. "
                        "Provide --tool or configure analysis.defaults.tool."
                    )
                },
                indent=2,
            )
        )
    tools = context["analysis"].get("tools", {})
    tool_entry = tools.get(tool_name) if isinstance(tools, dict) else None
    if not isinstance(tool_entry, dict):
        raise SystemExit(
            json.dumps(
                {"error": f"config/analysis.yaml does not define analysis.tools.{tool_name}."},
                indent=2,
            )
        )
    try:
        adapter = load_bids_analysis_tool_adapter(tool_entry)
        authoring_adapter = require_bids_analysis_model_authoring_adapter(adapter, tool_name=tool_name)
    except ValueError as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc
    return tool_name, tool_entry, authoring_adapter


def _normalize_analysis_model_name(name: str) -> str:
    normalized = _normalize_optional_cli_value(name)
    if normalized is None:
        raise SystemExit(json.dumps({"error": "Model name must be a non-empty string."}, indent=2))
    if normalized != Path(normalized).name or normalized.endswith(".yaml") or "/" in normalized or "\\" in normalized:
        raise SystemExit(json.dumps({"error": f"Invalid model name {name!r}."}, indent=2))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise SystemExit(
            json.dumps(
                {
                    "error": (
                        f"Invalid model name {name!r}. Use only letters, numbers, dots, underscores, and hyphens."
                    )
                },
                indent=2,
            )
        )
    return normalized


def _analysis_model_path(*, context: dict[str, Any], model_name: str) -> Path:
    return Path(context["models_dir"]) / f"{model_name}.yaml"


def _ensure_analysis_model_destination(*, path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            json.dumps(
                {
                    "error": f"Model file already exists: {path.name}. Re-run with --force to overwrite it."
                },
                indent=2,
            )
        )


def _load_analysis_model_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(json.dumps({"error": f"Model file does not exist: {path.name}."}, indent=2))
    document = load_yaml(path)
    if not isinstance(document, dict):
        raise SystemExit(json.dumps({"error": f"Model file {path.name} must contain a mapping."}, indent=2))
    return document


def _analysis_model_init_options(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "ev_order": _normalize_cli_multi_value(args.ev_order),
        "derivative_on": _normalize_cli_multi_value(args.derivative_on),
        "nonconvolved": _normalize_cli_multi_value(args.nonconvolved),
        "contrasts": [value for value in (args.contrast or []) if _normalize_optional_cli_value(value) is not None],
    }


def _configure_first_level_bold_inputs(
    *,
    args: argparse.Namespace,
    inputs: dict[str, Any],
    changed: list[str],
) -> None:
    raw_space = _normalize_optional_cli_value(args.bold_space)
    raw_desc = _normalize_optional_cli_value(args.bold_desc)
    if raw_space is None and raw_desc is None:
        return
    if raw_space is None or raw_desc is None:
        raise SystemExit(
            json.dumps(
                {"error": "Use --bold-space and --bold-desc together when configuring first-level BOLD patterns."},
                indent=2,
            )
        )
    space = _validate_bids_label(raw_space, label="--bold-space")
    desc = _validate_bids_label(raw_desc, label="--bold-desc")
    bold_config = inputs.get("bold")
    if not isinstance(bold_config, dict):
        bold_config = {}
    bold_config["patterns"] = _first_level_bold_patterns(space=space, desc=desc)
    inputs["bold"] = bold_config
    changed.append("analysis.inputs.bold.patterns")


def _configure_first_level_external_root(
    *,
    args: argparse.Namespace,
    analysis: dict[str, Any],
    local_arg: str,
    remote_arg: str,
    root_name: str,
    changed: list[str],
) -> None:
    raw_local = _normalize_optional_cli_value(getattr(args, local_arg, None))
    raw_remote = _normalize_optional_cli_value(getattr(args, remote_arg, None))
    if raw_local is None and raw_remote is None:
        return
    external_roots = analysis.setdefault("external_input_roots", {})
    if not isinstance(external_roots, dict):
        raise SystemExit(
            json.dumps(
                {"error": "config/analysis.yaml analysis.external_input_roots must contain a mapping."},
                indent=2,
            )
        )
    root_config = external_roots.get(root_name)
    if not isinstance(root_config, dict):
        root_config = {}
    if raw_local is not None:
        root_config["local_root"] = str(_validate_existing_local_directory(raw_local, label=f"--{local_arg.replace('_', '-')}"))
    if raw_remote is not None:
        root_config["remote_root"] = _validate_remote_posix_root(raw_remote, label=f"--{remote_arg.replace('_', '-')}")
    elif "remote_root" not in root_config:
        root_config["remote_root"] = ""
    if "sync_enabled" not in root_config:
        root_config["sync_enabled"] = True
    external_roots[root_name] = root_config
    changed.append(f"analysis.external_input_roots.{root_name}")


def _configure_first_level_evs_input(
    *,
    args: argparse.Namespace,
    inputs: dict[str, Any],
    changed: list[str],
) -> None:
    if _normalize_optional_cli_value(args.events_root) is None and _normalize_optional_cli_value(args.remote_events_root) is None:
        return
    evs_config = inputs.get("evs")
    if not isinstance(evs_config, dict):
        evs_config = {}
    evs_config["required"] = True
    evs_config["root_ref"] = "evs"
    if not evs_config.get("patterns"):
        evs_config["patterns"] = _first_level_ev_patterns()
    inputs["evs"] = evs_config
    changed.append("analysis.inputs.evs")


def _configure_first_level_confounds_input(
    *,
    args: argparse.Namespace,
    inputs: dict[str, Any],
    changed: list[str],
) -> None:
    raw_pattern = _normalize_optional_cli_value(args.confounds_pattern)
    has_confounds_root = (
        _normalize_optional_cli_value(args.confounds_root) is not None
        or _normalize_optional_cli_value(args.remote_confounds_root) is not None
    )
    if raw_pattern is None and not has_confounds_root:
        return
    confounds_config = inputs.get("confounds")
    if not isinstance(confounds_config, dict):
        confounds_config = {}
    confounds_config["required"] = True
    confounds_config["root_ref"] = "feat_confounds"
    if raw_pattern is not None:
        confounds_config["patterns"] = _first_level_confounds_patterns(raw_pattern)
    elif not confounds_config.get("patterns"):
        confounds_config["patterns"] = _first_level_confounds_patterns("desc-confounds_noGSR.txt")
    inputs["confounds"] = confounds_config
    changed.append("analysis.inputs.confounds")


def _configure_first_level_stage(
    *,
    args: argparse.Namespace,
    stage: dict[str, Any],
    changed: list[str],
) -> None:
    validation = stage.setdefault("validation", {})
    if not isinstance(validation, dict):
        raise SystemExit(json.dumps({"error": "analysis.stages.first_level.validation must contain a mapping."}, indent=2))
    settings = stage.setdefault("settings", {})
    if not isinstance(settings, dict):
        raise SystemExit(json.dumps({"error": "analysis.stages.first_level.settings must contain a mapping."}, indent=2))

    if _normalize_optional_cli_value(args.confounds_root) is not None or _normalize_optional_cli_value(args.confounds_pattern) is not None:
        validation["require_confounds"] = True
        changed.append("analysis.stages.first_level.validation.require_confounds")
    if args.empty_ev_policy is not None:
        validation["empty_ev_policy"] = args.empty_ev_policy
        changed.append("analysis.stages.first_level.validation.empty_ev_policy")

    for arg_name, setting_name in (
        ("tr", "tr"),
        ("hpf", "hpf"),
        ("smooth_mm", "smooth_mm"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            settings[setting_name] = float(value)
            changed.append(f"analysis.stages.first_level.settings.{setting_name}")

    for arg_name, setting_name in (
        ("norm", "norm"),
        ("motion_correction", "mc"),
        ("slice_timing", "slice_timing"),
        ("bet", "bet"),
        ("prewhiten", "prewhiten"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            settings[setting_name] = 1 if value == "on" else 0
            changed.append(f"analysis.stages.first_level.settings.{setting_name}")

    if args.output_desc is not None:
        outputs = stage.setdefault("outputs", {})
        if not isinstance(outputs, dict):
            raise SystemExit(json.dumps({"error": "analysis.stages.first_level.outputs must contain a mapping."}, indent=2))
        outputs["desc"] = _validate_bids_label(args.output_desc, label="--output-desc")
        changed.append("analysis.stages.first_level.outputs.desc")


def _first_level_bold_patterns(*, space: str, desc: str) -> list[str]:
    suffix = f"_space-{space}_desc-{desc}_bold.nii.gz"
    return [
        "{derivative_root}/{subject_dir}/{session_dir}/func/{bids_base}" + suffix,
        "{derivative_root}/{subject_dir}/func/{bids_base}" + suffix,
    ]


def _first_level_ev_patterns() -> list[str]:
    return [
        "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
        "{input_root}/{subject_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
    ]


def _first_level_confounds_patterns(pattern: str) -> list[str]:
    normalized = _normalize_optional_cli_value(pattern)
    if normalized is None:
        raise SystemExit(json.dumps({"error": "--confounds-pattern must not be empty."}, indent=2))
    if "{" in normalized or "/" in normalized:
        return [normalized]
    filename = normalized.lstrip("_")
    if "{bids_base}" not in filename:
        filename = f"{{bids_base}}_{filename}"
    return [
        "{input_root}/{subject_dir}/{session_dir}/func/" + filename,
        "{input_root}/{subject_dir}/func/" + filename,
    ]


def _add_roi_path_profile_arg(parser: Any) -> None:
    parser.add_argument(
        "--path-profile",
        default="generic",
        choices=_ROI_SCAFFOLD_PATH_PROFILES,
        help=(
            "Scaffold-time path profile. generic is site-neutral; research_platform_fsl_ffx is an advanced "
            "FSL fixed-effects layout example."
        ),
    )


def _add_roi_set_scaffold_override_args(parser: argparse.ArgumentParser) -> None:
    advanced = parser.add_argument_group(
        "advanced scaffold overrides",
        "Optional compatibility overrides. Prefer editing analysis choices in the generated YAML.",
    )
    _add_roi_path_profile_arg(advanced)
    advanced.add_argument(
        "--subjects",
        default=None,
        help=(
            "Subject override. Single-entity ROI templates require exactly one subject; "
            "loso_group_map accepts comma-separated lists and inclusive ranges."
        ),
    )
    advanced.add_argument(
        "--held-out-subjects",
        default=None,
        help="Held-out subject list, inclusive range, or same-as-subjects.",
    )
    _add_roi_common_scaffold_override_args(advanced)
    advanced.add_argument("--search-radius-mm", default=None, help="Search radius in millimeters for generated ROIs.")
    advanced.add_argument("--sphere-radius-mm", default=None, help="Sphere radius in millimeters for generated ROIs.")
    advanced.add_argument("--z-threshold", default=None, help="Z threshold for generated ROIs.")
    advanced.add_argument("--min-voxels-warn", default=None, help="Warning voxel-count threshold for generated ROIs.")
    advanced.add_argument("--min-voxels-fail", default=None, help="Failure voxel-count threshold for generated ROIs.")
    advanced.add_argument(
        "--contrast",
        dest="contrasts",
        action="append",
        default=None,
        help="Contrast id:cope_number[:desc]. Repeat for multiple contrasts.",
    )
    advanced.add_argument(
        "--roi",
        dest="rois",
        action="append",
        default=None,
        help="ROI label:contrast_id:x,y,z[:desc]. Repeat for multiple ROIs.",
    )


def _add_roi_extraction_scaffold_override_args(parser: argparse.ArgumentParser) -> None:
    advanced = parser.add_argument_group(
        "advanced scaffold overrides",
        "Optional compatibility overrides. Prefer editing analysis choices in the generated YAML.",
    )
    _add_roi_path_profile_arg(advanced)
    advanced.add_argument(
        "--subjects",
        default=None,
        help=(
            "Subject override. generic_nifti requires exactly one subject; fsl_featquery accepts "
            "comma-separated lists and inclusive ranges."
        ),
    )
    _add_roi_common_scaffold_override_args(advanced)
    advanced.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Extraction metric. Repeat for multiple metrics.",
    )
    advanced.add_argument(
        "--contrast",
        dest="contrasts",
        action="append",
        default=None,
        help="Contrast id:cope_number[:desc]. Repeat for multiple contrasts.",
    )
    advanced.add_argument(
        "--roi-label",
        dest="roi_labels",
        action="append",
        default=None,
        help="ROI label to extract. Repeat for multiple labels.",
    )


def _add_roi_common_scaffold_override_args(parser: Any) -> None:
    parser.add_argument("--session", default=None, help="Session entity for scaffolded YAML.")
    parser.add_argument("--task", default=None, help="Task entity for scaffolded YAML.")
    parser.add_argument("--direction", default=None, help="Direction entity for scaffolded YAML.")
    parser.add_argument("--model", default=None, help="Model entity for scaffolded YAML.")
    parser.add_argument("--space", default=None, help="Space entity for scaffolded YAML.")
    parser.add_argument("--resolution", default=None, help="Resolution entity for scaffolded YAML.")


def _resolve_project_name(project_name: str | None) -> str:
    if project_name:
        return project_name
    resolved_root = workspace_root()
    return default_project_name(load_workspace_config(resolved_root))


def _add_project_hpc_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True, help="Project overlay name.")
    _add_hpc_profile_args(parser)


def _add_hpc_notebook_plan_args(parser: argparse.ArgumentParser) -> None:
    _add_project_hpc_args(parser)
    parser.add_argument("--cpus", type=int, default=None, help="Notebook allocation CPU count override.")
    parser.add_argument("--mem", default=None, help="Notebook allocation memory override, for example 8G.")
    parser.add_argument("--time", default=None, help="Notebook allocation time override, for example 02:00:00.")
    parser.add_argument("--local-port", type=int, default=None, help="Notebook tunnel local port override.")
    parser.add_argument("--remote-port", type=int, default=None, help="Notebook Jupyter and tunnel remote port override.")


def _add_hpc_notebook_start_args(parser: argparse.ArgumentParser) -> None:
    _add_hpc_notebook_plan_args(parser)
    parser.add_argument("--compute-host", default=None, help="Allocated compute hostname, optionally including user@.")
    parser.add_argument("--token", default=None, help="Optional Jupyter token used to derive a local URL.")
    parser.add_argument("--url-path", default=None, help="Optional local URL path suffix to print with the tunnel guidance.")
    _add_hpc_tunnel_mode_arg(parser)
    parser.set_defaults(tunnel_mode="login-forward")
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--execute", action="store_true", help="Launch the remote notebook allocation and start its tunnel.")
    authorization.add_argument("--execute-tunnel", action="store_true", help="Run only the rendered SSH tunnel command when --compute-host is set.")
    authorization.add_argument("--plan-only", action="store_true", help="Explicitly print the compact start guidance without launching remotely.")
    parser.add_argument("--open-browser", action="store_true", help="Open the final local notebook URL after the tunnel is ready.")


def _add_hpc_notebook_submit_args(parser: argparse.ArgumentParser) -> None:
    _add_project_hpc_args(parser)
    parser.add_argument("--notebook", default=None, help="Notebook path override, resolved relative to the workspace or project root.")
    parser.add_argument("--job-name", default=None, help="SLURM job name override.")
    parser.add_argument("--cpus", type=int, default=None, help="Notebook execution CPU count override.")
    parser.add_argument("--mem", default=None, help="Notebook execution memory override, for example 8G.")
    parser.add_argument("--time", default=None, help="Notebook execution time override, for example 02:00:00.")
    parser.add_argument("--output-notebook", default=None, help="Output notebook file name written inside the run directory.")
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--dry-run", action="store_true", help="Write the local submit script and print the planned remote submission command.")
    authorization.add_argument("--execute", action="store_true", help="Stream the local script to the remote host and submit it to SLURM.")


def _add_hpc_init_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alliance-user", default=None, help="Alliance username stored in secrets/.env.")
    parser.add_argument(
        "--remote-workspace-root",
        default=None,
        help="Absolute remote repo root, for example /home/<user>/research-platform.",
    )
    parser.add_argument(
        "--remote-artifacts-root",
        default=None,
        help="Optional absolute remote artifacts root, for example /scratch/<user>/research-platform/artifacts.",
    )
    parser.add_argument("--profile", default=None, help="SSH profile name used in the printed next steps.")
    parser.add_argument("--role", choices=("login", "robot"), default=None, help="Optional default SSH profile role stored in secrets/.env.")
    parser.add_argument("--ssh-config", default=None, help="Optional SSH profile config path stored in secrets/.env.")


def _add_hpc_setup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--template",
        choices=("generic", "alliance"),
        default=None,
        help="Provider-neutral generic starter (default), or explicit Alliance/MFA integration requiring site review.",
    )
    parser.add_argument("--target", "--cluster", dest="target", default=None, help="Named HPC target to create or update; --cluster is a backward-compatible alias.")
    parser.add_argument("--profile", default=None, help="SSH profile name. Defaults to the selected target.")
    parser.add_argument("--role", choices=("login", "robot"), default=None, help="SSH profile role. Defaults to login.")
    parser.add_argument("--user", dest="cluster_user", default=None, help="Explicit SSH username stored only in the selected private profile.")
    parser.add_argument("--alliance-user", default=None, help="Backward-compatible Alliance username alias; generic setup does not write ALLIANCE_USER.")
    parser.add_argument("--host", default=None, help="Explicit SSH login host. Generic setup never invents a hostname.")
    parser.add_argument("--port", type=int, default=None, help="Optional SSH port from 1 through 65535.")
    parser.add_argument("--identity-file", default=None, help="Optional SSH private-key path reference; key contents are never read.")
    parser.add_argument("--known-hosts-file", default=None, help="Optional known-hosts file path reference; contents are never read.")
    parser.add_argument("--remote-workspace-root", default=None, help="Absolute remote workspace root.")
    parser.add_argument("--remote-artifacts-root", default=None, help="Optional absolute remote artifacts root.")
    parser.add_argument("--remote-container-root", default=None, help="Optional remote container image root.")
    parser.add_argument("--account", default=None, help="Optional explicitly supplied SLURM account for this target.")
    parser.add_argument("--partition", default=None, help="Optional explicitly supplied SLURM partition for this target.")
    parser.add_argument("--ssh-config", default=None, help="SSH profile config path. Defaults to secrets/hpc/ssh-profiles.yaml.")
    parser.add_argument("--targets-config", default=None, help="HPC target config path. Defaults to secrets/hpc/targets.yaml.")
    parser.add_argument("--set-default", action="store_true", help="Also make this target the default when writing targets.yaml.")
    parser.add_argument("--force", action="store_true", help="Replace only the selected profile and target when their existing settings conflict.")


def _add_hpc_validate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", default=None, help="HPC target name. Defaults to local environment selection, then targets default.")
    parser.add_argument(
        "--targets-config",
        default=None,
        help="HPC targets config path. Defaults to the local environment selection or secrets/hpc/targets.yaml.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional project overlay whose local structure and selected target override are also validated.",
    )
    parser.add_argument("--profile", default=None, help="SSH profile override for offline validation.")
    parser.add_argument("--role", default=None, help="SSH role override for offline validation.")
    parser.add_argument("--ssh-config", default=None, help="SSH profile config path override.")
    parser.add_argument("--json", action="store_true", help="Emit the offline validation report as JSON.")


def _add_hpc_target_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="HPC targets config path. Defaults to secrets/hpc/targets.yaml.")


def _add_hpc_sync_workspace_args(parser: argparse.ArgumentParser) -> None:
    _add_hpc_profile_args(parser)
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--dry-run", action="store_true", help="Render rsync with --dry-run without invoking it.")
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="Include Git-untracked paths instead of auto-excluding them from the workspace sync.",
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Include Git-ignored paths instead of auto-excluding them from the workspace sync.",
    )
    parser.add_argument(
        "--extra-exclude-file",
        default=None,
        help="Optional extra local rsync exclude file, resolved relative to the workspace root when not absolute.",
    )
    authorization.add_argument("--execute", action="store_true", help="Create the remote root and run rsync.")


def _add_hpc_sync_project_args(parser: argparse.ArgumentParser) -> None:
    _add_project_hpc_args(parser)
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--dry-run", action="store_true", help="Render rsync with --dry-run without invoking it.")
    authorization.add_argument("--execute", action="store_true", help="Create remote targets and run rsync.")


def _add_hpc_sync_data_args(parser: argparse.ArgumentParser) -> None:
    _add_project_hpc_args(parser)
    parser.add_argument("--batch", default=None, help="Optional batch name for selected-only sync planning.")
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Sync only adapter-resolved files for the selected batch.",
    )
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--dry-run", action="store_true", help="Render rsync with --dry-run without invoking it.")
    authorization.add_argument("--execute", action="store_true", help="Create remote targets and run rsync.")


def _add_hpc_verify_data_args(parser: argparse.ArgumentParser) -> None:
    _add_project_hpc_args(parser)
    parser.add_argument("--batch", default=None, help="Batch name without .tsv.")
    _add_bids_selector_args(parser)


def _add_hpc_container_prepare_args(parser: argparse.ArgumentParser) -> None:
    _add_project_hpc_args(parser)
    parser.add_argument("--backend", choices=("apptainer", "singularity"), default=None, help="Container backend override.")
    parser.add_argument("--image", default=None, help="Remote runtime .sif path override.")
    parser.add_argument("--source-image", default=None, help="Container source image override, for example docker://...")
    parser.add_argument("--execute", action="store_true", help="Run the rendered SSH command.")


def _add_hpc_tunnel_args(parser: argparse.ArgumentParser) -> None:
    _add_hpc_profile_args(parser)
    parser.add_argument("--compute-host", required=True, help="Allocated compute hostname, optionally including user@.")
    parser.add_argument("--local-port", type=int, default=8890, help="Local port to bind for the tunnel.")
    parser.add_argument("--remote-port", type=int, default=8888, help="Remote port exposed on the compute host.")
    parser.add_argument("--url-path", default=None, help="Optional local URL path suffix to print alongside the tunnel command.")
    _add_hpc_tunnel_mode_arg(parser)
    parser.add_argument("--execute", action="store_true", help="Run the rendered SSH tunnel command.")


def _add_hpc_tunnel_mode_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tunnel-mode",
        choices=_HPC_TUNNEL_MODES,
        default="direct",
        help="Tunnel mode: direct SSH to the compute host or login-host TCP forwarding.",
    )


def _add_hpc_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, help="Named SSH profile.")
    parser.add_argument("--role", choices=("login", "robot"), default=None, help="Named role within a profile family.")
    parser.add_argument("--config", default=None, help="SSH profile config path.")


def _load_hpc_profile(args: argparse.Namespace) -> tuple[Any, Path, str]:
    profile_name = _resolve_hpc_profile_name(getattr(args, "profile", None))
    role = _resolve_hpc_role(getattr(args, "role", None))
    raw_config_path = getattr(args, "config", None)
    workspace_hint = None
    if raw_config_path is None or not Path(str(raw_config_path)).expanduser().is_absolute():
        workspace_hint = _workspace_root_if_available()
    config_path = resolve_ssh_profile_config_path(raw_config_path, workspace_root=workspace_hint)
    return load_ssh_profile(config_path, profile_name, role=role), config_path, role


def _workspace_root_if_available() -> Path | None:
    try:
        return workspace_root()
    except WorkspaceRootNotFoundError:
        return None


def _resolve_hpc_profile_name(profile_name: str | None) -> str:
    value = _normalize_optional_cli_value(profile_name)
    if value:
        return value
    env_value = os.environ.get("RESEARCH_HPC_PROFILE") or os.environ.get("RP_HPC_PROFILE")
    resolved = _normalize_optional_cli_value(env_value)
    if resolved:
        return resolved
    raise SystemExit(
        "SSH profile is required. Provide --profile, set RESEARCH_HPC_PROFILE or RP_HPC_PROFILE, "
        "or add one of those keys to secrets/.env."
    )


def _resolve_hpc_role(role: str | None) -> str:
    value = _normalize_optional_cli_value(role)
    if value:
        return value
    env_value = os.environ.get("RESEARCH_HPC_ROLE") or os.environ.get("RP_HPC_ROLE")
    resolved = _normalize_optional_cli_value(env_value)
    return resolved or "login"


def _normalize_optional_cli_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_config_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _validate_bids_label(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise SystemExit(f"{label} must not be empty.")
    if not re.fullmatch(r"[A-Za-z0-9]+", normalized):
        raise SystemExit(f"{label} must contain only letters and numbers.")
    return normalized


def _normalize_cli_multi_value(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for raw_value in values or []:
        for chunk in str(raw_value).split(","):
            token = chunk.strip()
            if token:
                normalized.append(token)
    return normalized


def _validate_remote_workspace_root(remote_workspace_root: str) -> str:
    value = str(remote_workspace_root).strip()
    if value.startswith(("~", "$")):
        raise SystemExit(
            "Remote workspace root must be an absolute path, not a shell-expanded value like '~/...' or '$HOME/...'. "
            "Use an absolute remote path such as /home/<user>/research-platform."
        )
    return value


def _validate_remote_workspace_root_for_init(remote_workspace_root: str) -> str:
    return _validate_remote_root_for_init(
        remote_workspace_root,
        label="Remote workspace root",
        example="/home/<user>/research-platform",
    )


def _validate_remote_artifacts_root_for_init(remote_artifacts_root: str) -> str:
    value = str(remote_artifacts_root).strip()
    if value.startswith(("~", "$")):
        raise SystemExit(
            "Remote artifacts root must not be a shell-expanded value like '~/...' or '$HOME/...'. "
            "Use the final remote path you want persisted in secrets/.env."
        )
    return value


def _validate_remote_root_for_init(remote_root: str, *, label: str, example: str) -> str:
    value = _validate_remote_workspace_root(remote_root)
    if not value.startswith("/"):
        raise SystemExit(
            f"{label} must be an absolute path such as {example}."
        )
    return value


def _resolve_hpc_init_value(
    *,
    value: str | None,
    label: str,
    flag: str,
    default: str | None,
    optional: bool = False,
    prompt_when_missing: bool = True,
    example: str | None = None,
) -> str | None:
    if value is not None:
        normalized = value.strip()
        if normalized:
            return normalized
        if optional:
            return None
        raise SystemExit(f"{label} is required. Provide {flag} or enter it interactively.")

    if not prompt_when_missing:
        return default

    prompt = label
    if example:
        prompt = f"{prompt} ({example})"
    if default:
        prompt = f"{prompt} [{default}]"
    try:
        response = input(f"{prompt}: ").strip()
    except EOFError as exc:
        raise SystemExit(f"{label} is required. Re-run with {flag} or provide a value interactively.") from exc
    if response:
        return response
    if default:
        return default
    if optional:
        return None
    raise SystemExit(f"{label} is required. Provide {flag} or enter it interactively.")


def _require_remote_workspace_root(project_context: dict[str, Any]) -> str:
    remote_workspace_root = project_context.get("remote_workspace_root")
    if remote_workspace_root:
        return _validate_remote_workspace_root(str(remote_workspace_root))
    raise SystemExit(
        "Remote workspace root is not configured. Set RP_REMOTE_WORKSPACE_ROOT or declare compute.slurm.remote_workspace_root."
    )


def _hpc_profile_mode(role: str) -> str:
    return "batch" if role == "robot" else "interactive"


def _render_hpc_target_report(target: dict[str, Any], *, project_name: str | None = None) -> list[str]:
    lines = [f"HPC target {target['name']}", ""]
    lines.extend(
        [
            "Connection defaults",
            f"- ssh profile: {target.get('ssh_profile') or 'not set'}",
            f"- role: {target.get('role') or 'not set'}",
            f"- ssh config: {target.get('ssh_config') or 'not set'}",
        ]
    )
    lines.append("")
    lines.append("Environment")
    env = target.get("env", {})
    if env:
        for key in sorted(env):
            lines.append(f"- {key}: {env[key]}")
    else:
        lines.append("- no target env defaults")
    if project_name is not None:
        lines.append("")
        lines.append(f"Project environment ({project_name})")
        project_env = target.get("project_env", {})
        if project_env:
            for key in sorted(project_env):
                lines.append(f"- {key}: {project_env[key]}")
        else:
            lines.append("- no project-specific env defaults")
    if target.get("projects"):
        lines.append("")
        lines.append("Project overrides")
        for name in target["projects"]:
            lines.append(f"- {name}")
    lines.append("")
    lines.append("SLURM site defaults")
    slurm = target.get("slurm", {})
    if slurm:
        for key in sorted(slurm):
            lines.append(f"- {key}: {slurm[key]}")
    else:
        lines.append("- no SLURM site defaults")
    if target.get("warnings"):
        lines.append("")
        lines.append("Warnings")
        for warning in target["warnings"]:
            lines.append(f"- {warning}")
    return lines


def _render_hpc_doctor_report(
    *,
    project_name: str,
    project_context: dict[str, Any],
    config_path: Path,
    profile: Any,
    ssh_report: dict[str, Any],
    role: str,
) -> list[str]:
    lines = [f"HPC doctor for {project_name}", ""]
    target = project_context.get("hpc_target") or {}
    if target:
        lines.extend(
            [
                "HPC target",
                f"- active target: {target.get('name')}",
                f"- config: {target.get('config_path')}",
                f"- ssh profile: {target.get('ssh_profile') or 'not set'}",
                f"- role default: {target.get('role') or 'not set'}",
            ]
        )
        site_settings = resolve_hpc_slurm_site_settings(project_context.get("compute", {}).get("slurm", {}))
        if site_settings:
            lines.append("- slurm site settings:")
            for key in sorted(site_settings):
                lines.append(f"  {key}: {site_settings[key]}")
        if target.get("warnings"):
            lines.append("- target warnings:")
            for warning in target["warnings"]:
                lines.append(f"  {warning}")
        lines.append("")
    lines.extend(
        [
            "Project",
            f"- kind: {project_context['project_kind']}",
            f"- slice: {project_context['slice']}",
            f"- project root: {to_workspace_relative(project_context['project_root'], project_context['workspace_root'])}",
        ]
    )
    if project_context.get("validation_errors"):
        lines.append("- config validation: issues found")
        for error in project_context["validation_errors"]:
            lines.append(f"  {error}")
    else:
        lines.append("- config validation: ok")
    notebook_path = project_context.get("notebook_path")
    if notebook_path is not None:
        lines.append(f"- notebook path: {to_workspace_relative(notebook_path, project_context['workspace_root'])}")
    else:
        lines.append("- notebook path: not declared")
    lines.append("")
    lines.extend(
        [
            "SSH profile",
            f"- config: {config_path}",
            f"- profile: {profile.name}",
            f"- role: {role}",
            f"- target: {profile.target()}",
            f"- mode used: {ssh_report.get('mode_used', 'batch')}",
            f"- connectivity: {'ok' if ssh_report.get('ok') else 'needs attention'}",
        ]
    )
    multiplexing_status, multiplexing_warnings = _ssh_multiplexing_status(profile)
    lines.append(f"- multiplexing: {multiplexing_status}")
    for warning in multiplexing_warnings:
        lines.append(f"- warning: {warning}")
    guidance = ssh_report.get("host_key_fix_guidance")
    if guidance:
        lines.append(f"- guidance: {guidance}")
    remote_workspace_root = project_context.get("remote_workspace_root") or "not resolved"
    lines.append(f"- remote workspace root: {remote_workspace_root}")
    remote_artifacts_root = project_context.get("remote_artifacts_root") or "not resolved"
    lines.append(f"- remote artifacts root: {remote_artifacts_root}")
    lines.append("")
    lines.append("Project roots")
    for root_spec in project_context.get("project_roots", []):
        lines.append(
            f"- {root_spec['label']}: {_render_root_line(root_spec['path'], workspace_root=project_context['workspace_root'])}"
        )
    lines.append("")
    lines.append("Data roots")
    for root_spec in project_context.get("data_roots", []):
        suffixes: list[str] = []
        if root_spec.get("remote_root"):
            suffixes.append(f"remote root declared: {root_spec['remote_root']}")
        elif root_spec.get("sync_enabled", True):
            suffixes.append("remote root not configured")
        if not root_spec.get("exists", False):
            suffixes.append("missing locally")
        suffix = f" ({'; '.join(suffixes)})" if suffixes else ""
        lines.append(
            f"- {root_spec['label']}: {_render_root_line(root_spec['path'], workspace_root=project_context['workspace_root'])}{suffix}"
        )
    lines.append("")
    lines.extend(
        [
            "Next commands",
            f"- rp hpc sync-project --project {project_name} --profile {profile.name} --role {role}",
            f"- rp hpc sync-data --project {project_name} --profile {profile.name} --role {role}",
            "- Review both rendered plans before authorizing remote changes.",
            f"- rp hpc sync-project --project {project_name} --profile {profile.name} --role {role} --execute",
            f"- rp hpc sync-data --project {project_name} --profile {profile.name} --role {role} --execute",
            f"- rp hpc notebook plan --project {project_name} --profile {profile.name} --role {role}",
        ]
    )
    return lines


def _ssh_multiplexing_status(profile: Any) -> tuple[str, list[str]]:
    options = getattr(profile, "options", {}) or {}
    control_master = str(options.get("ControlMaster", "")).strip().lower()
    control_path = str(options.get("ControlPath", "")).strip()
    control_persist = str(options.get("ControlPersist", "")).strip()
    enabled = control_master in {"auto", "autoask", "yes"} and control_path and control_path.lower() != "none"
    if enabled:
        persist_suffix = f", ControlPersist={control_persist}" if control_persist else ""
        return f"enabled ({options.get('ControlMaster')}, ControlPath={control_path}{persist_suffix})", []
    return (
        "not configured",
        [
            "SSH multiplexing is not configured; MFA-backed clusters may prompt once per SSH/rsync connection. "
            "Set ControlMaster=auto, ControlPath=~/.ssh/cm-%C, and ControlPersist=2h, then warm the connection with "
            "research-hpc ssh check --mode interactive --remote-command true."
        ],
    )


def _render_sync_plan_report(
    *,
    title: str,
    actions: list[dict[str, Any]],
    profile: Any,
    config_path: Path,
) -> list[str]:
    lines = [
        title,
        "",
        f"SSH profile: {profile.name} ({profile.role})",
        f"SSH config: {config_path}",
        "",
    ]
    if not actions:
        lines.append("No syncable entries were inferred for this project.")
        return lines
    lines.append("Entries")
    for action in actions:
        lines.append(
            f"- {action['label']}: {action['source']} -> {action['destination_path']}"
        )
        if action.get("exclude_files"):
            lines.append(f"  excludes: {', '.join(action['exclude_files'])}")
        lines.append(f"  mkdir: {_render_command(action['mkdir_command'])}")
        lines.append(f"  rsync: {_render_command(action['rsync_command'])}")
    return lines


def _render_hpc_data_verification_report(summary: dict[str, Any]) -> list[str]:
    transport_failure = summary.get("transport_failure")
    rows_all_present = (
        "not determined (transport failure)"
        if transport_failure
        else str(summary["rows_all_present"])
    )
    lines = [
        f"Remote data verification for {summary['project']}",
        "",
        f"Project: {summary['project']}",
        f"Remote dataset root: {_render_hpc_verify_root_status(summary['dataset_root'])}",
        f"Remote derivative root: {_render_hpc_verify_root_status(summary['derivative_root'])}",
        f"Rows checked: {summary['row_count']}",
        f"Rows with all expected remote files present: {rows_all_present}",
    ]
    for root in summary.get("additional_roots", []):
        lines.append(f"Remote root ({root['label']}): {_render_hpc_verify_root_status(root)}")
    connection = summary.get("connection", {})
    if connection:
        profile_name = connection.get("profile")
        role = connection.get("role")
        config_path = connection.get("config")
        target = connection.get("target")
        if profile_name:
            lines.append(f"SSH profile: {profile_name} ({role or 'login'})")
        elif target:
            lines.append(f"SSH target: {target}")
        if config_path:
            lines.append(f"SSH config: {config_path}")

    if summary.get("row_source"):
        lines.append(f"Row source: {summary['row_source']}")
    if summary.get("batch_name"):
        lines.append(f"Batch: {summary['batch_name']}")

    if summary.get("layer_b_note"):
        lines.append(f"Row-level verification: {summary['layer_b_note']}")

    if transport_failure:
        lines.extend(
            [
                "",
                "Transport failure",
                f"- remote path probe exited with code {transport_failure['returncode']}",
                f"- {transport_failure['message']}",
            ]
        )

    if summary["missing_paths"]:
        lines.extend(["", "Missing paths"])
        lines.extend(f"- {path}" for path in summary["missing_paths"])

    if summary["issues"]:
        lines.extend(["", "Notes"])
        lines.extend(f"- {issue}" for issue in summary["issues"])
    return lines


def _render_hpc_verify_root_status(root_status: dict[str, str]) -> str:
    if root_status["status"] == "not configured":
        return "not configured"
    if root_status["status"] == "not checked":
        return f"not checked ({root_status['path']})"
    return f"{root_status['status']} ({root_status['path']})"


def _render_workspace_sync_plan_report(
    *,
    workspace_root: Path,
    remote_workspace_root: str,
    profile: Any,
    config_path: Path,
    tracked_exclude_files: list[Path],
    git_safety: _WorkspaceGitSafetyExcludes,
    mkdir_command: list[str],
    rsync_command: list[str],
) -> list[str]:
    lines = [
        "Workspace sync plan",
        "",
        f"SSH profile: {profile.name} ({profile.role})",
        f"SSH config: {config_path}",
        f"Workspace root: {workspace_root}",
        f"Remote workspace root: {remote_workspace_root}",
    ]
    if tracked_exclude_files:
        rendered_excludes = ", ".join(
            _render_path_for_report(path, workspace_root=workspace_root) for path in tracked_exclude_files
        )
        lines.append(f"Tracked exclude files: {rendered_excludes}")
    lines.append(f"Git-untracked excludes: {_render_git_workspace_exclude_layer(git_safety.untracked)}")
    lines.append(f"Git-ignored excludes: {_render_git_workspace_exclude_layer(git_safety.ignored)}")
    lines.append(f"Auto-excluded Git paths: {git_safety.total_count}")
    if git_safety.exclude_file is not None:
        lines.append(f"Temporary local exclude file (removed after planning): {git_safety.exclude_file}")
    lines.extend(
        [
            "",
            f"Remote mkdir command: {_render_command(mkdir_command)}",
            f"Rsync command: {_render_command(rsync_command)}",
        ]
    )
    return lines


def _render_notebook_plan_report(
    *,
    project_name: str,
    project_context: dict[str, Any],
    config_path: Path,
    profile: Any,
    notebook_settings: dict[str, Any],
) -> list[str]:
    remote_workspace_root = project_context.get("remote_workspace_root") or "<remote-workspace-root>"
    launch_target = notebook_launch_target(project_context)
    allocation = _notebook_allocation_command(project_name=project_name, notebook_settings=notebook_settings)
    setup_steps = _notebook_setup_steps(
        project_context,
        modules=notebook_settings["modules"],
        pre_activate_commands=notebook_settings["pre_activate_commands"],
        prepare_directories=notebook_settings.get("prepare_directories", []),
    )
    login_command = build_ssh_command(profile, mode="interactive", remote_command=None, allocate_tty=False)
    compute_host_placeholder = "<compute-hostname>"
    compute_host_target = _resolve_tunnel_compute_host(compute_host_placeholder, profile=profile)
    tunnel_cli_command = _build_notebook_tunnel_cli_command(
        profile=profile,
        config_path=config_path,
        compute_host=compute_host_target,
        local_port=int(notebook_settings["local_port"]),
        remote_port=int(notebook_settings["remote_port"]),
    )
    tunnel_command = _build_ssh_tunnel_command(
        profile,
        local_port=int(notebook_settings["local_port"]),
        remote_port=int(notebook_settings["remote_port"]),
        remote_host=compute_host_target,
        jump_host=profile.target(),
    )
    lines = [
        f"Notebook plan for {project_name}",
        "",
        "Checklist",
        f"- Login command: {_render_command(login_command)}",
        f"- Change to repo root on the cluster: cd {shlex.quote(str(remote_workspace_root))}",
        f"- Allocate interactive resources: {allocation}",
    ]
    for label, command in setup_steps:
        lines.append(f"- {label}: {command}")
    lines.extend(
        [
            f"- Print the compute hostname from the allocated shell: {_notebook_compute_host_command()}",
            (
                "- Launch Jupyter on the compute node: "
                f"{_notebook_launch_command(remote_port=int(notebook_settings['remote_port']), launch_target=launch_target)}"
            ),
            (
                "- From your laptop, open the tunnel after replacing "
                f"{compute_host_placeholder}: {_render_command(tunnel_cli_command)}"
            ),
            f"- Equivalent SSH tunnel command: {_render_command(tunnel_command)}",
            f"- Notebook path to open: {launch_target}",
            "",
            "Resolved",
            f"- SSH config: {config_path}",
            f"- SSH target: {profile.target()}",
        ]
    )
    notebook_path = project_context.get("notebook_path")
    if notebook_path is not None:
        lines.append(
            f"- Local notebook path: {to_workspace_relative(notebook_path, project_context['workspace_root'])}"
        )
    else:
        lines.append(
            f"- Local notebook path: {to_workspace_relative(default_notebook_launch_path(project_context), project_context['workspace_root'])}"
        )
    return lines


def _build_notebook_start_report(
    *,
    project_name: str,
    project_context: dict[str, Any],
    config_path: Path,
    profile: Any,
    notebook_settings: dict[str, Any],
    compute_host: str | None,
    url_path: str | None,
    tunnel_mode: str = "direct",
) -> tuple[list[str], list[str]]:
    remote_workspace_root = project_context.get("remote_workspace_root") or "<remote-workspace-root>"
    launch_target = notebook_launch_target(project_context)
    allocation = _notebook_allocation_command(project_name=project_name, notebook_settings=notebook_settings)
    setup_steps = _notebook_setup_steps(
        project_context,
        modules=notebook_settings["modules"],
        pre_activate_commands=notebook_settings["pre_activate_commands"],
        prepare_directories=notebook_settings.get("prepare_directories", []),
    )
    resolved_compute_host = None
    if compute_host:
        resolved_compute_host = _resolve_tunnel_compute_host(compute_host, profile=profile)
    else:
        resolved_compute_host = _resolve_tunnel_compute_host("<compute-hostname>", profile=profile)
    tunnel_cli_command = _build_notebook_tunnel_cli_command(
        profile=profile,
        config_path=config_path,
        compute_host=resolved_compute_host,
        local_port=int(notebook_settings["local_port"]),
        remote_port=int(notebook_settings["remote_port"]),
        url_path=url_path,
        tunnel_mode=tunnel_mode,
    )
    tunnel_command = _build_ssh_tunnel_command(
        profile,
        local_port=int(notebook_settings["local_port"]),
        remote_port=int(notebook_settings["remote_port"]),
        remote_host=_resolve_tunnel_forward_host(resolved_compute_host) if tunnel_mode == "login-forward" else (resolved_compute_host if compute_host else None),
        jump_host=profile.target() if compute_host else None,
        tunnel_mode=tunnel_mode,
    )
    local_url = _render_local_tunnel_url(int(notebook_settings["local_port"]), url_path or "")
    lines = [
        f"Notebook start for {project_name}",
        f"- Change to repo root on the cluster: cd {shlex.quote(str(remote_workspace_root))}",
        f"- Allocate interactive resources: {allocation}",
    ]
    for label, command in setup_steps:
        lines.append(f"- {label}: {command}")
    lines.extend(
        [
            f"- Launch Jupyter on the compute node: {_notebook_launch_command(remote_port=int(notebook_settings['remote_port']), launch_target=launch_target, bind_ip=_notebook_bind_ip_for_tunnel_mode(tunnel_mode))}",
            f"- Recommended tunnel command: {_render_command(tunnel_cli_command)}",
            f"- Local URL: {local_url}",
        ]
    )
    return lines, tunnel_command


def _build_profile_push_command(
    *,
    source: Path,
    destination: str,
    profile: Any,
    mode: str,
    exclude_files: list[Path],
    dry_run: bool,
    source_is_directory: bool,
) -> list[str]:
    command = build_basic_rsync_push_command(
        source=source,
        ssh_host=profile.target(),
        destination=destination,
        exclude_files=exclude_files,
        source_is_directory=source_is_directory,
    )
    rendered_ssh = render_ssh_shell(profile, mode=mode)
    command[2:2] = ["-e", rendered_ssh]
    insertion_index = len(command) - 2
    optional_flags: list[str] = []
    if dry_run:
        optional_flags.append("--dry-run")
    optional_flags.append("--itemize-changes")
    command[insertion_index:insertion_index] = optional_flags
    return command


def _build_sync_plan_actions(
    *,
    plan: dict[str, Any],
    workspace_root: Path,
    remote_workspace_root: str,
    profile: Any,
    mode: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for entry in plan.get("entries", []):
        source_path = (workspace_root / entry["source"]).resolve()
        destination_path = str(PurePosixPath(remote_workspace_root) / entry["destination"])
        source_is_directory = entry.get("kind", "directory") == "directory"
        mkdir_target = destination_path if source_is_directory else str(PurePosixPath(destination_path).parent)
        mkdir_command = build_ssh_command(
            profile,
            mode=mode,
            remote_command=f"mkdir -p {shlex.quote(mkdir_target)}",
            allocate_tty=False,
        )
        rsync_command = _build_profile_push_command(
            source=source_path,
            destination=destination_path,
            profile=profile,
            mode=mode,
            exclude_files=[workspace_root / path for path in entry.get("exclude_files", [])],
            dry_run=dry_run,
            source_is_directory=source_is_directory,
        )
        actions.append(
            {
                "label": entry["label"],
                "source": entry["source"],
                "destination_path": destination_path,
                "exclude_files": list(entry.get("exclude_files", [])),
                "mkdir_command": mkdir_command,
                "rsync_command": rsync_command,
            }
        )
    return actions


def _execute_sync_actions(actions: list[dict[str, Any]]) -> int:
    created_targets: set[tuple[str, ...]] = set()
    for action in actions:
        mkdir_command = action["mkdir_command"]
        mkdir_key = tuple(mkdir_command)
        if mkdir_key not in created_targets:
            mkdir_result = subprocess.run(mkdir_command, check=False, text=True)
            if mkdir_result.returncode != 0:
                return mkdir_result.returncode
            created_targets.add(mkdir_key)
        rsync_result = subprocess.run(action["rsync_command"], check=False, text=True)
        if rsync_result.returncode != 0:
            return rsync_result.returncode
    return 0


def _build_ssh_tunnel_command(
    profile: Any,
    *,
    local_port: int,
    remote_port: int,
    remote_host: str | None = None,
    jump_host: str | None = None,
    tunnel_mode: str = "direct",
) -> list[str]:
    base = build_ssh_command(profile, mode="interactive", remote_command=None, allocate_tty=False)
    command = [*base[:-1]]
    if tunnel_mode == "login-forward":
        command.extend(["-N", "-L", f"{local_port}:{remote_host}:{remote_port}", base[-1]])
        return command
    target = remote_host or base[-1]
    if jump_host:
        command.extend(["-J", jump_host])
    command.extend(["-N", "-L", f"{local_port}:127.0.0.1:{remote_port}", target])
    return command


def _resolve_tunnel_compute_host(compute_host: str, *, profile: Any) -> str:
    normalized = compute_host.strip()
    if "@" in normalized or not getattr(profile, "user", None):
        return normalized
    return f"{profile.user}@{normalized}"


def _resolve_tunnel_forward_host(compute_host: str) -> str:
    normalized = compute_host.strip()
    if "@" in normalized:
        return normalized.rsplit("@", 1)[1]
    return normalized


def _render_local_tunnel_url(local_port: int, url_path: str) -> str:
    normalized = url_path.strip()
    if not normalized:
        return f"http://127.0.0.1:{local_port}"
    if normalized.startswith(("/", "?", "#")):
        suffix = normalized
    else:
        suffix = f"/{normalized}"
    return f"http://127.0.0.1:{local_port}{suffix}"


def _disable_ssh_multiplexing(command: list[str]) -> list[str]:
    if not command or command[0] != "ssh":
        return list(command)
    return [command[0], "-o", "ControlMaster=no", "-o", "ControlPath=none", *command[1:]]


def _is_ssh_mux_session_refused_output(output: str) -> bool:
    normalized = output.lower()
    return any(pattern in normalized for pattern in _SSH_MUX_SESSION_REFUSED_PATTERNS)


def _start_logged_tunnel_process(command: list[str]) -> tuple[subprocess.Popen[str], list[str], threading.Thread]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    captured_stderr: list[str] = []

    def _forward_stderr() -> None:
        if process.stderr is None:
            return
        while True:
            chunk = process.stderr.read(1)
            if not chunk:
                return
            captured_stderr.append(chunk)
            sys.stderr.write(chunk)
            sys.stderr.flush()

    reader = threading.Thread(target=_forward_stderr, daemon=True)
    reader.start()
    return process, captured_stderr, reader


def _observe_tunnel_process_startup(
    *,
    local_port: int,
    process: subprocess.Popen[str],
    host: str = "127.0.0.1",
    timeout_seconds: float = 1.0,
    interval_seconds: float = 0.1,
) -> str:
    attempts = max(1, int(timeout_seconds / interval_seconds))
    for attempt in range(attempts):
        if process.poll() is not None:
            return "exited"
        try:
            connection = socket.create_connection((host, local_port), timeout=interval_seconds)
        except OSError:
            if attempt + 1 < attempts:
                time.sleep(interval_seconds)
            continue
        connection.close()
        return "ready"
    if process.poll() is not None:
        return "exited"
    return "running"


def _wait_for_local_port(
    *,
    local_port: int,
    process: subprocess.Popen[str] | None = None,
    host: str = "127.0.0.1",
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.1,
) -> bool:
    attempts = max(1, int(timeout_seconds / interval_seconds))
    for attempt in range(attempts):
        if process is not None and process.poll() is not None:
            return False
        try:
            connection = socket.create_connection((host, local_port), timeout=interval_seconds)
        except OSError:
            if attempt + 1 < attempts:
                time.sleep(interval_seconds)
            continue
        connection.close()
        return True
    return False


def _select_available_local_port(*, host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _build_notebook_tunnel_cli_command(
    *,
    profile: Any,
    config_path: Path,
    compute_host: str,
    local_port: int,
    remote_port: int,
    url_path: str | None = None,
    tunnel_mode: str = "direct",
) -> list[str]:
    command = [
        "rp",
        "hpc",
        "tunnel",
        "--profile",
        str(profile.name),
    ]
    role = getattr(profile, "role", None)
    if role:
        command.extend(["--role", str(role)])
    command.extend(
        [
            "--config",
            str(config_path),
            "--compute-host",
            compute_host,
            "--local-port",
            str(local_port),
            "--remote-port",
            str(remote_port),
        ]
    )
    if tunnel_mode != "direct":
        command.extend(["--tunnel-mode", tunnel_mode])
    if url_path is not None:
        command.extend(["--url-path", url_path])
    return command


def _notebook_allocation_command(*, project_name: str, notebook_settings: dict[str, Any], no_shell: bool = False) -> str:
    cpus = notebook_settings["cpus"]
    mem = notebook_settings["mem"]
    time = notebook_settings["time"]
    command = (
        f"salloc --job-name rp-notebook-{project_name} --cpus-per-task {cpus} --mem {mem} "
        f"--time {time}"
    )
    if no_shell:
        command = f"{command} --no-shell"
    return command


def _notebook_setup_steps(
    project_context: dict[str, Any],
    *,
    modules: list[str],
    pre_activate_commands: list[str],
    prepare_directories: list[str],
) -> list[tuple[str, str]]:
    workspace_root = Path(project_context["workspace_root"])
    steps: list[tuple[str, str]] = []
    if modules:
        steps.append(("Load cluster modules", f"module load {' '.join(shlex.quote(module) for module in modules)}"))
    for index, command in enumerate(pre_activate_commands, start=1):
        steps.append((f"Run remote notebook pre-activate command {index}", command))
    if prepare_directories:
        steps.append(
            (
                "Prepare remote runtime directories",
                "mkdir -p " + " ".join(_quote_remote_shell_path(directory) for directory in prepare_directories),
            )
        )
    bootstrap_command = _build_repo_bootstrap_command(workspace_root)
    if bootstrap_command is not None:
        steps.append(
            (
                "Bootstrap the repo environment if needed",
                bootstrap_command,
            )
        )
    activate_command = _build_repo_activate_command(workspace_root)
    if activate_command is not None:
        steps.append(
            (
                "Activate the repo virtualenv",
                activate_command,
            )
        )
    return steps


def _notebook_compute_host_command() -> str:
    return "hostname -f"


def _quote_remote_shell_path(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _notebook_bind_ip_for_tunnel_mode(tunnel_mode: str) -> str:
    if tunnel_mode == "login-forward":
        return "0.0.0.0"
    return "127.0.0.1"


def _notebook_launch_command(*, remote_port: int, launch_target: str, bind_ip: str = "127.0.0.1") -> str:
    return f"jupyter lab --no-browser --ip={bind_ip} --port={remote_port} {shlex.quote(launch_target)}"


def _resolve_notebook_url_path(*, token: str | None, url_path: str | None) -> str | None:
    normalized_url_path = _normalize_optional_cli_value(url_path)
    if normalized_url_path is not None:
        return normalized_url_path
    normalized_token = _normalize_optional_cli_value(token)
    if normalized_token is not None:
        return f"/lab?token={normalized_token}"
    return None


def _resolve_remote_notebook_port(*, remote_url: str, fallback_port: int) -> int:
    normalized_remote_url = _normalize_optional_cli_value(remote_url)
    if normalized_remote_url:
        try:
            parsed = urlsplit(normalized_remote_url)
        except ValueError:
            return fallback_port
        if parsed.port is not None:
            return int(parsed.port)
    return fallback_port


def _resolve_notebook_start_settings(
    *,
    args: argparse.Namespace,
    project_context: dict[str, Any],
    launch_remote: bool,
) -> dict[str, Any]:
    notebook_settings = dict(_resolve_notebook_plan_settings(args=args, project_context=project_context))
    if getattr(args, "local_port", None) is None:
        notebook_settings["local_port"] = _select_available_local_port()
    if launch_remote and getattr(args, "remote_port", None) is None:
        notebook_settings["remote_port"] = 0
    return notebook_settings


def _plan_hpc_notebook_submit(
    *,
    args: argparse.Namespace,
    project_name: str,
    project_context: dict[str, Any],
    profile: Any,
    role: str,
) -> dict[str, Any]:
    notebook_settings = _resolve_notebook_plan_settings(args=args, project_context=project_context)
    local_notebook_path = _resolve_notebook_submit_local_path(args=args, project_context=project_context)
    remote_notebook_path = _resolve_notebook_submit_remote_path(
        args=args,
        project_context=project_context,
        local_notebook_path=local_notebook_path,
    )
    output_notebook_name = _resolve_output_notebook_name(
        cli_value=getattr(args, "output_notebook", None),
        source_notebook_path=local_notebook_path,
    )
    timestamp = _notebook_submit_timestamp()
    local_run_dir = ensure_dir(Path(project_context["paths"]["artifacts_root"]) / "notebook-runs" / project_name / timestamp)
    remote_artifacts_root = _resolve_notebook_remote_artifacts_root(project_context)
    remote_run_dir = str(PurePosixPath(remote_artifacts_root) / "notebook-runs" / project_name / timestamp)
    local_script_path = local_run_dir / "submit.sbatch"
    remote_script_path = str(PurePosixPath(remote_run_dir) / "submit.sbatch")
    remote_output_notebook_path = str(PurePosixPath(remote_run_dir) / output_notebook_name)
    job_name = _normalize_optional_cli_value(getattr(args, "job_name", None)) or f"rp-notebook-{project_name}"
    setup_commands = _build_repo_slurm_setup_commands(
        remote_workspace_root=_require_remote_workspace_root(project_context),
        workspace_root=Path(project_context["workspace_root"]),
        modules=notebook_settings["modules"],
        environment=_resolve_project_slurm_environment(project_context),
        pre_activate_commands=notebook_settings["pre_activate_commands"],
        prepare_directories=notebook_settings["prepare_directories"],
    )
    script_content = normalize_slurm_batch_script(
        render_slurm_script(
            template_path=project_context["paths"]["ops_root"] / "slurm" / "job_templates" / "sbatch.job.sh",
            jobspec={
                "job_name": job_name,
                "cpus": int(notebook_settings["cpus"]),
                "mem": str(notebook_settings["mem"]),
                "time": str(notebook_settings["time"]),
                "log_out": str(PurePosixPath(remote_run_dir) / "slurm-%j.out"),
                "log_err": str(PurePosixPath(remote_run_dir) / "slurm-%j.err"),
                "command": build_slurm_command_script(
                    setup_commands=setup_commands
                    + [
                        f"mkdir -p {shlex.quote(remote_run_dir)}",
                    ],
                    workflow_command=(
                        "jupyter nbconvert --to notebook --execute "
                        f"{shlex.quote(remote_notebook_path)} --output {shlex.quote(output_notebook_name)} "
                        f"--output-dir {shlex.quote(remote_run_dir)} --ExecutePreprocessor.timeout=-1"
                    ),
                ),
            },
        )
    )
    write_slurm_script(local_script_path, script_content)
    submit_command = _build_notebook_submit_command(
        profile=profile,
        role=role,
        remote_run_dir=remote_run_dir,
        remote_script_path=remote_script_path,
    )
    return {
        "local_run_dir": local_run_dir,
        "remote_run_dir": remote_run_dir,
        "local_script_path": local_script_path,
        "local_output_notebook_path": local_run_dir / output_notebook_name,
        "remote_output_notebook_path": remote_output_notebook_path,
        "submit_command": submit_command,
        "submit_command_text": _render_stdin_submit_command(submit_command, local_script_path),
        "script_content": script_content,
    }


def _resolve_notebook_plan_settings(*, args: argparse.Namespace, project_context: dict[str, Any]) -> dict[str, Any]:
    overlay_notebook = _notebook_overlay_config(project_context)
    slurm_config = _notebook_slurm_config(project_context)
    slurm_notebook = slurm_config.get("notebook", {}) if isinstance(slurm_config.get("notebook"), dict) else {}
    return {
        "cpus": _resolve_notebook_setting(
            cli_value=getattr(args, "cpus", None),
            overlay_notebook=overlay_notebook,
            slurm_notebook=slurm_notebook,
            slurm_config=slurm_config,
            key="cpus",
            fallback=2,
        ),
        "mem": _resolve_notebook_setting(
            cli_value=getattr(args, "mem", None),
            overlay_notebook=overlay_notebook,
            slurm_notebook=slurm_notebook,
            slurm_config=slurm_config,
            key="mem",
            fallback="8G",
        ),
        "time": _resolve_notebook_setting(
            cli_value=getattr(args, "time", None),
            overlay_notebook=overlay_notebook,
            slurm_notebook=slurm_notebook,
            slurm_config=slurm_config,
            key="time",
            fallback="02:00:00",
        ),
        "local_port": _resolve_notebook_setting(
            cli_value=getattr(args, "local_port", None),
            overlay_notebook=overlay_notebook,
            slurm_notebook=slurm_notebook,
            slurm_config=slurm_config,
            key="local_port",
            fallback=8890,
        ),
        "remote_port": _resolve_notebook_setting(
            cli_value=getattr(args, "remote_port", None),
            overlay_notebook=overlay_notebook,
            slurm_notebook=slurm_notebook,
            slurm_config=slurm_config,
            key="remote_port",
            fallback=8888,
        ),
        "modules": _resolve_notebook_modules(
            overlay_notebook=overlay_notebook,
            slurm_notebook=slurm_notebook,
            slurm_config=slurm_config,
        ),
        "pre_activate_commands": _resolve_notebook_pre_activate_commands(
            overlay_notebook=overlay_notebook,
            slurm_notebook=slurm_notebook,
            slurm_config=slurm_config,
        ),
        "prepare_directories": _resolve_notebook_prepare_directories(
            overlay_notebook=overlay_notebook,
            slurm_notebook=slurm_notebook,
            slurm_config=slurm_config,
        ),
    }


def _notebook_overlay_config(project_context: dict[str, Any]) -> dict[str, Any]:
    project_config = project_context.get("project")
    if not isinstance(project_config, dict):
        return {}
    hpc_config = project_config.get("hpc")
    if not isinstance(hpc_config, dict):
        return {}
    notebook_config = hpc_config.get("notebook")
    if not isinstance(notebook_config, dict):
        return {}
    return notebook_config


def _notebook_slurm_config(project_context: dict[str, Any]) -> dict[str, Any]:
    compute_config = project_context.get("compute")
    if not isinstance(compute_config, dict):
        return {}
    slurm_config = compute_config.get("slurm")
    if not isinstance(slurm_config, dict):
        return {}
    return slurm_config


def _resolve_notebook_setting(
    *,
    cli_value: Any,
    overlay_notebook: dict[str, Any],
    slurm_notebook: dict[str, Any],
    slurm_config: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    for value in (cli_value, overlay_notebook.get(key), slurm_notebook.get(key), slurm_config.get(key)):
        if value is not None:
            return value
    return fallback


def _resolve_notebook_modules(
    *,
    overlay_notebook: dict[str, Any],
    slurm_notebook: dict[str, Any],
    slurm_config: dict[str, Any],
) -> list[str]:
    if "modules" in overlay_notebook:
        return _normalize_notebook_modules(overlay_notebook.get("modules"))
    if "modules" in slurm_notebook:
        return _normalize_notebook_modules(slurm_notebook.get("modules"))
    if "modules" in slurm_config:
        return _normalize_notebook_modules(slurm_config.get("modules"))
    return []


def _resolve_notebook_pre_activate_commands(
    *,
    overlay_notebook: dict[str, Any],
    slurm_notebook: dict[str, Any],
    slurm_config: dict[str, Any],
) -> list[str]:
    if "pre_activate_commands" in overlay_notebook:
        return _normalize_notebook_shell_commands(overlay_notebook.get("pre_activate_commands"))
    if "pre_activate_commands" in slurm_notebook:
        return _normalize_notebook_shell_commands(slurm_notebook.get("pre_activate_commands"))
    if "pre_activate_commands" in slurm_config:
        return _normalize_notebook_shell_commands(slurm_config.get("pre_activate_commands"))
    return []


def _resolve_notebook_prepare_directories(
    *,
    overlay_notebook: dict[str, Any],
    slurm_notebook: dict[str, Any],
    slurm_config: dict[str, Any],
) -> list[str]:
    if "prepare_directories" in overlay_notebook:
        return _normalize_notebook_prepare_directories(overlay_notebook.get("prepare_directories"))
    if "prepare_directories" in slurm_notebook:
        return _normalize_notebook_prepare_directories(slurm_notebook.get("prepare_directories"))
    if "prepare_directories" in slurm_config:
        return _normalize_notebook_prepare_directories(slurm_config.get("prepare_directories"))
    return []


def _normalize_notebook_modules(value: Any) -> list[str]:
    return normalize_slurm_modules(value)


def _normalize_notebook_shell_commands(value: Any) -> list[str]:
    return normalize_slurm_shell_commands(value)


def _normalize_notebook_prepare_directories(value: Any) -> list[str]:
    return normalize_slurm_prepare_directories(value)


def _resolve_notebook_submit_local_path(*, args: argparse.Namespace, project_context: dict[str, Any]) -> Path:
    notebook_value = _normalize_optional_cli_value(getattr(args, "notebook", None))
    if notebook_value is None:
        resolved = default_notebook_launch_path(project_context)
    else:
        resolved = _resolve_project_relative_path(
            notebook_value,
            workspace_root=Path(project_context["workspace_root"]),
            project_root=Path(project_context["project_root"]),
        )
    if resolved.suffix != ".ipynb":
        raise SystemExit("Notebook submit requires a .ipynb notebook path. Provide --notebook or declare a project notebook.")
    return resolved


def _resolve_notebook_submit_remote_path(
    *,
    args: argparse.Namespace,
    project_context: dict[str, Any],
    local_notebook_path: Path,
) -> str:
    if _normalize_optional_cli_value(getattr(args, "notebook", None)) is None:
        return notebook_launch_target(project_context)
    return _resolve_remote_workspace_path(
        local_notebook_path,
        workspace_root=Path(project_context["workspace_root"]),
        remote_workspace_root=_require_remote_workspace_root(project_context),
    )


def _resolve_output_notebook_name(*, cli_value: str | None, source_notebook_path: Path) -> str:
    output_name = _normalize_optional_cli_value(cli_value) or f"{source_notebook_path.stem}.executed.ipynb"
    if Path(output_name).name != output_name:
        raise SystemExit("--output-notebook must be a file name, not a path.")
    return output_name


def _resolve_notebook_remote_artifacts_root(project_context: dict[str, Any]) -> str:
    remote_artifacts_root = _normalize_optional_cli_value(str(project_context.get("remote_artifacts_root") or ""))
    if remote_artifacts_root:
        return remote_artifacts_root
    return str(PurePosixPath(_require_remote_workspace_root(project_context)) / "artifacts")


def _build_notebook_submit_command(*, profile: Any, role: str, remote_run_dir: str, remote_script_path: str) -> list[str]:
    remote_command = "bash -lc " + shlex.quote(
        "; ".join(
            [
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(remote_run_dir)}",
                f"cat > {shlex.quote(remote_script_path)}",
                f"sbatch {shlex.quote(remote_script_path)}",
            ]
        )
    )
    return build_ssh_command(profile, mode=_hpc_profile_mode(role), remote_command=remote_command, allocate_tty=False)


def _render_stdin_submit_command(command: list[str], local_script_path: Path) -> str:
    return f"{_render_command(command)} < {shlex.quote(str(local_script_path))}"


def _resolve_project_relative_path(value: str, *, workspace_root: Path, project_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    workspace_candidate = (workspace_root / candidate).resolve()
    if workspace_candidate.exists():
        return workspace_candidate
    return (project_root / candidate).resolve()


def _resolve_remote_workspace_path(path: Path, *, workspace_root: Path, remote_workspace_root: str) -> str:
    try:
        relative = path.resolve().relative_to(workspace_root)
    except ValueError as exc:
        raise SystemExit("Notebook submit requires a notebook path inside the workspace so it can be resolved remotely.") from exc
    return str(PurePosixPath(remote_workspace_root) / relative.as_posix())


def _parse_sbatch_submission_job_id(output: str) -> str | None:
    match = _SBATCH_SUBMITTED_JOB_PATTERN.search(output.strip())
    if match is None:
        return None
    return match.group(1)


def _render_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _render_root_line(path: Path, *, workspace_root: Path) -> str:
    return to_workspace_relative(path, workspace_root)


def _render_path_for_report(path: Path, *, workspace_root: Path) -> str:
    try:
        return to_workspace_relative(path, workspace_root)
    except ValueError:
        return str(path)


def _render_git_workspace_exclude_layer(layer: _GitWorkspaceExcludeLayer) -> str:
    if layer.active:
        return f"active ({layer.count} paths)"
    return "inactive"


def _build_git_workspace_safety_excludes(
    workspace_root: Path,
    *,
    include_untracked: bool,
    include_ignored: bool,
) -> _WorkspaceGitSafetyExcludes:
    if include_untracked and include_ignored:
        return _WorkspaceGitSafetyExcludes(
            untracked=_GitWorkspaceExcludeLayer(active=False, paths=()),
            ignored=_GitWorkspaceExcludeLayer(active=False, paths=()),
        )
    status_entries = _read_git_workspace_status_entries(workspace_root)
    if status_entries is None:
        return _WorkspaceGitSafetyExcludes(
            untracked=_GitWorkspaceExcludeLayer(active=False, paths=()),
            ignored=_GitWorkspaceExcludeLayer(active=False, paths=()),
        )
    untracked_paths = () if include_untracked else _select_git_workspace_status_paths(status_entries, "?? ")
    ignored_paths = () if include_ignored else _select_git_workspace_status_paths(status_entries, "!! ")
    combined_paths = [*untracked_paths, *ignored_paths]
    exclude_file: Path | None = None
    temporary_root: Path | None = None
    if combined_paths:
        temporary_root = Path(tempfile.mkdtemp(prefix="rp-workspace-sync-"))
        exclude_file = temporary_root / "exclude.git-auto.txt"
        exclude_file.write_text("".join(f"{path}\n" for path in combined_paths), encoding="utf-8")
    return _WorkspaceGitSafetyExcludes(
        untracked=_GitWorkspaceExcludeLayer(active=not include_untracked, paths=untracked_paths),
        ignored=_GitWorkspaceExcludeLayer(active=not include_ignored, paths=ignored_paths),
        exclude_file=exclude_file,
        temporary_root=temporary_root,
    )


def _read_git_workspace_status_entries(workspace_root: Path) -> list[str] | None:
    command = [
        "git",
        "-C",
        str(workspace_root),
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=normal",
        "--ignored=matching",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return [entry.decode("utf-8", errors="surrogateescape") for entry in result.stdout.split(b"\0") if entry]


def _select_git_workspace_status_paths(entries: list[str], prefix: str) -> tuple[str, ...]:
    selected: list[str] = []
    for entry in entries:
        if not entry.startswith(prefix):
            continue
        candidate = entry[len(prefix) :]
        if candidate and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


def _build_hpc_data_verification_plan(
    *,
    project_name: str,
    project_context: dict[str, Any],
    batch_name: str | None,
    selectors: dict[str, object],
) -> dict[str, Any]:
    remote_roots = _hpc_data_verification_roots(project_context)
    verification_rows: list[dict[str, Any]] = []
    auxiliary_paths: list[str] = []
    issues = [
        f"Project validation issue not blocking Layer A: {error}"
        for error in project_context.get("validation_errors", [])
    ]
    layer_b_note = ""
    row_source = ""
    resolved_batch_name = _normalize_optional_cli_value(batch_name) or _default_context_batch_name(project_context)
    batch_path = (
        project_context["project_root"] / "manifests" / "batches" / f"{resolved_batch_name}.tsv"
        if resolved_batch_name
        else None
    )

    if project_context.get("slice") != "bids":
        layer_b_note = "skipped because this project is not using a BIDS tool adapter"
    elif project_context.get("tool_adapter") is None:
        layer_b_note = "skipped because no BIDS tool adapter is configured"
    else:
        remote_derivative_root = _normalize_optional_cli_value(project_context.get("remote_input_derivative_root"))
        local_derivative_root = project_context.get("input_derivative_root")
        if remote_derivative_root is None:
            layer_b_note = "skipped because no remote input derivative root is configured"
        elif local_derivative_root is None:
            layer_b_note = "skipped because adapter-specific local derivative inputs could not be resolved"
        else:
            selected_rows, row_source, layer_b_note = _resolve_hpc_data_verification_rows(
                project_context=project_context,
                batch_path=batch_path,
                selectors=selectors,
            )
            unresolved_rows: list[str] = []
            for row in selected_rows:
                if project_context.get("analysis"):
                    expected_paths = project_context["tool_adapter"].expected_remote_input_files(
                        derivative_root=str(local_derivative_root),
                        remote_derivative_root=remote_derivative_root,
                        row=row,
                        context=project_context,
                    )
                else:
                    expected_paths = project_context["tool_adapter"].expected_remote_input_files(
                        derivative_root=str(local_derivative_root),
                        remote_derivative_root=remote_derivative_root,
                        row=row,
                    )
                if not expected_paths:
                    unresolved_rows.append(_render_bids_row_label(row))
                    continue
                verification_rows.append(
                    {
                        "row": {name: row.get(name, "") for name in ("subject_id", "session_id", "task_id", "run_id")},
                        "expected_paths": expected_paths,
                    }
                )
            if unresolved_rows:
                issues.append(
                    "Skipped Layer B rows because adapter-specific inputs could not be resolved: "
                    + ", ".join(unresolved_rows)
                )
            if selected_rows and not verification_rows:
                layer_b_note = "skipped because adapter-specific inputs could not be resolved from the available rows"

            expected_auxiliary = getattr(project_context["tool_adapter"], "expected_remote_auxiliary_files", None)
            raw_auxiliary_paths = expected_auxiliary(context=project_context) if callable(expected_auxiliary) else []
            auxiliary_paths = (
                [str(path) for path in raw_auxiliary_paths]
                if isinstance(raw_auxiliary_paths, (list, tuple, set))
                else []
            )

    paths = _dedupe_text_values(
        [root["path"] for root in remote_roots]
        + [path for row in verification_rows for path in row["expected_paths"]]
        + auxiliary_paths
    )
    return {
        "project": project_name,
        "batch_name": resolved_batch_name or "",
        "batch_path": str(batch_path) if batch_path is not None else "",
        "remote_roots": remote_roots,
        "rows": verification_rows,
        "auxiliary_paths": auxiliary_paths,
        "row_source": row_source,
        "layer_b_note": layer_b_note,
        "issues": issues,
        "paths": paths,
    }


def _hpc_data_verification_roots(project_context: dict[str, Any]) -> list[dict[str, str]]:
    roots: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, path in (
        ("remote dataset root", _normalize_optional_cli_value(project_context.get("remote_dataset_root"))),
        ("remote derivative root", _normalize_optional_cli_value(project_context.get("remote_input_derivative_root"))),
    ):
        if path is None or path in seen:
            continue
        seen.add(path)
        roots.append({"label": label, "path": path})
    for root_spec in project_context.get("data_roots", []):
        remote_root = _normalize_optional_cli_value(root_spec.get("remote_root"))
        if remote_root is None or remote_root in seen:
            continue
        seen.add(remote_root)
        roots.append({"label": f"{root_spec['label']} remote root", "path": remote_root})
    return roots


def _resolve_hpc_data_verification_rows(
    *,
    project_context: dict[str, Any],
    batch_path: Path | None,
    selectors: dict[str, object],
) -> tuple[list[dict[str, str]], str, str]:
    batch_rows = _read_tsv(batch_path) if batch_path is not None and batch_path.exists() else []
    if batch_rows:
        selected_rows = _filter_bids_rows(batch_rows, selectors)
        if selected_rows:
            return selected_rows, "batch manifest", ""
        return [], "batch manifest", "no batch rows matched the requested selectors"

    discovered_rows = project_context["tool_adapter"].discover_batch_rows(
        derivative_root=str(project_context["input_derivative_root"]),
        selectors=_single_value_bids_selectors(selectors),
        **({"context": project_context} if project_context.get("analysis") else {}),
    )
    selected_rows = _filter_bids_rows(discovered_rows, selectors)
    if selected_rows:
        return selected_rows, "adapter discovery", ""
    return [], "adapter discovery", "no matching rows were discovered from project config and selectors"


def _single_value_bids_selectors(selectors: dict[str, object]) -> dict[str, str | None]:
    single_value_selectors: dict[str, str | None] = {}
    for key in ("subject_id", "session_id", "task_id", "run_id"):
        values = normalize_filter_values(selectors.get(key))
        single_value_selectors[key] = values[0] if len(values) == 1 else None
    return single_value_selectors


def _filter_bids_rows(rows: list[dict[str, str]], selectors: dict[str, object]) -> list[dict[str, str]]:
    normalizers = {
        "subject_id": lambda value: _normalize_bids_selector_label(value, prefix="sub") or "",
        "session_id": lambda value: _normalize_bids_selector_label(value, prefix="ses") or "",
        "task_id": lambda value: _normalize_bids_selector_label(value, prefix="task") or "",
        "run_id": lambda value: _normalize_bids_selector_label(value, prefix="run") or "",
    }
    available_selectors = {
        key: value
        for key, value in selectors.items()
        if any(str(row.get(key, "")).strip() for row in rows)
    }
    return filter_manifest_rows(rows, available_selectors, normalizers=normalizers)


def _render_bids_row_label(row: dict[str, str]) -> str:
    parts = [row.get(name, "") for name in ("subject_id", "session_id", "task_id", "run_id")]
    selected = [part for part in parts if part]
    return ", ".join(selected) if selected else "<unlabeled row>"


def _dedupe_text_values(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_bids_selector_label(value: str | None, *, prefix: str) -> str | None:
    normalized = _normalize_optional_cli_value(value)
    if normalized is None:
        return None
    if normalized.startswith(f"{prefix}-"):
        return normalized.split("-", 1)[1]
    return normalized


def _normalize_bids_filter_value(value: str, *, prefix: str) -> str:
    normalized = str(value).strip()
    if normalized.startswith(f"{prefix}-"):
        return normalized
    return f"{prefix}-{normalized}"


def _summarize_hpc_data_verification(
    *,
    verification_plan: dict[str, Any],
    remote_report: dict[str, Any],
) -> dict[str, Any]:
    transport_failure = _summarize_hpc_verify_transport_failure(remote_report)
    probe_failed = transport_failure is not None
    path_status = {
        item["path"]: bool(item["exists"])
        for item in remote_report.get("paths", [])
    } if not probe_failed else {}
    missing_paths = (
        []
        if probe_failed
        else _dedupe_text_values(
            [
                path
                for path in verification_plan["paths"]
                if not path_status.get(path, False)
            ]
        )
    )
    rows_all_present = 0
    if not probe_failed:
        for row in verification_plan["rows"]:
            expected_paths = row["expected_paths"]
            if expected_paths and all(path_status.get(path, False) for path in expected_paths):
                rows_all_present += 1

    dataset_root = _summarize_hpc_verify_root(
        _normalize_optional_cli_value(
            next(
                (root["path"] for root in verification_plan["remote_roots"] if root["label"] == "remote dataset root"),
                None,
            )
        ),
        path_status=path_status,
        probe_failed=probe_failed,
    )
    derivative_root = _summarize_hpc_verify_root(
        _normalize_optional_cli_value(
            next(
                (root["path"] for root in verification_plan["remote_roots"] if root["label"] == "remote derivative root"),
                None,
            )
        ),
        path_status=path_status,
        probe_failed=probe_failed,
    )
    additional_roots = [
        {
            "label": root["label"],
            "path": root["path"],
            "status": "not checked" if probe_failed else "present" if path_status.get(root["path"], False) else "missing",
        }
        for root in verification_plan["remote_roots"]
        if root["label"] not in {"remote dataset root", "remote derivative root"}
    ]
    return {
        "project": verification_plan["project"],
        "batch_name": verification_plan["batch_name"],
        "row_source": verification_plan["row_source"],
        "layer_b_note": verification_plan["layer_b_note"],
        "dataset_root": dataset_root,
        "derivative_root": derivative_root,
        "additional_roots": additional_roots,
        "row_count": len(verification_plan["rows"]),
        "rows_all_present": rows_all_present,
        "missing_paths": missing_paths,
        "issues": list(verification_plan["issues"]),
        "connection": remote_report.get("connection", {}),
        "has_missing": bool(missing_paths),
        "transport_failure": transport_failure,
    }


def _summarize_hpc_verify_root(
    path: str | None,
    *,
    path_status: dict[str, bool],
    probe_failed: bool = False,
) -> dict[str, str]:
    if path is None:
        return {"path": "", "status": "not configured"}
    if probe_failed:
        return {"path": path, "status": "not checked"}
    return {"path": path, "status": "present" if path_status.get(path, False) else "missing"}


def _summarize_hpc_verify_transport_failure(remote_report: dict[str, Any]) -> dict[str, Any] | None:
    if not remote_report:
        return None
    returncode = int(remote_report.get("returncode", 0))
    if returncode == 0:
        return None
    detail = _normalize_optional_cli_value(remote_report.get("stderr")) or _normalize_optional_cli_value(
        remote_report.get("stdout")
    )
    if detail is None:
        detail = "Remote path probe failed before any path status could be confirmed."
    else:
        detail = " ".join(line.strip() for line in str(detail).splitlines() if line.strip())
    return {"returncode": returncode, "message": detail}


def _resolve_project_container_prepare_spec(
    *,
    context: dict[str, Any],
    backend_override: str | None = None,
    image_override: str | None = None,
    source_image_override: str | None = None,
) -> dict[str, str]:
    if context.get("slice") != "bids":
        raise SystemExit("HPC container preparation currently supports BIDS projects.")
    runtime_profile = context.get("runtime_profile")
    if not isinstance(runtime_profile, dict):
        runtime_profile = {}
    slurm_profile = runtime_profile.get("slurm", {})
    if not isinstance(slurm_profile, dict):
        slurm_profile = {}
    container = slurm_profile.get("container", {})
    if not isinstance(container, dict):
        container = {}

    backend = _normalize_optional_cli_value(backend_override)
    if backend is None:
        backend = _normalize_optional_cli_value(container.get("backend")) or _normalize_optional_cli_value(
            slurm_profile.get("execution_backend")
        )
    if backend not in {"apptainer", "singularity"}:
        raise SystemExit(
            "Project container preparation requires an Apptainer/Singularity runtime profile. "
            "Set compute.tool_profiles.<tool>.slurm.execution_backend to apptainer or singularity."
        )

    raw_image = _normalize_optional_cli_value(resolve_env_value(container.get("image")))
    image_name = _normalize_sif_name(
        _normalize_optional_cli_value(resolve_env_value(container.get("image_name")))
        or f"{context.get('runtime_profile_name') or 'container'}.sif"
    )
    image_root = _normalize_optional_cli_value(resolve_env_value(container.get("image_root"))) or "$SCRATCH/containers"
    runtime_image = _normalize_optional_cli_value(resolve_env_value(image_override))
    if runtime_image is None:
        if raw_image and raw_image.startswith("docker://"):
            runtime_image = _join_remote_path(image_root, image_name)
        else:
            runtime_image = raw_image
    source_image = _normalize_optional_cli_value(resolve_env_value(source_image_override))
    if source_image is None:
        source_image = _normalize_optional_cli_value(resolve_env_value(container.get("source_image")))
    if source_image is None and raw_image and raw_image.startswith("docker://"):
        source_image = raw_image

    nextflow = slurm_profile.get("nextflow", {})
    if not isinstance(nextflow, dict):
        nextflow = {}
    nextflow_enabled = _normalize_config_bool(
        nextflow.get("enabled"),
        default=bool(context.get("runtime_profile_name") == "deepprep"),
    )
    nextflow_version = _normalize_optional_cli_value(resolve_env_value(nextflow.get("version"))) or "24.10.3"
    nextflow_jar_name = (
        _normalize_optional_cli_value(resolve_env_value(nextflow.get("jar_name")))
        or f"nextflow-{nextflow_version}-one.jar"
    )
    nextflow_home = (
        _normalize_optional_cli_value(resolve_env_value(nextflow.get("host_home") or nextflow.get("home")))
        or "$SCRATCH/deepprep/nextflow"
    )
    nextflow_jar_url = (
        _normalize_optional_cli_value(resolve_env_value(nextflow.get("jar_url")))
        or f"https://www.nextflow.io/releases/v{nextflow_version}/{nextflow_jar_name}"
    )

    if runtime_image is None:
        raise SystemExit("No runtime container image is configured. Set container.image or pass --image.")
    if source_image is None:
        raise SystemExit(
            "No container source image is configured. Set container.source_image or pass --source-image, "
            "for example docker://pbfslab/deepprep:25.1.0."
        )
    if "://" not in source_image:
        raise SystemExit(f"Container source image must be a registry URI such as docker://...; got {source_image!r}.")

    return {
        "backend": backend,
        "runtime_image": runtime_image,
        "source_image": source_image,
        "pull_mode": _normalize_optional_cli_value(container.get("pull_mode")) or "",
        "runtime_profile": str(context.get("runtime_profile_name") or ""),
        "nextflow_enabled": "true" if nextflow_enabled else "false",
        "nextflow_home": nextflow_home,
        "nextflow_version": nextflow_version,
        "nextflow_jar_name": nextflow_jar_name,
        "nextflow_jar_url": nextflow_jar_url,
    }


def _build_hpc_container_prepare_remote_command(*, context: dict[str, Any], spec: dict[str, str]) -> str:
    slurm_config = context.get("compute", {}).get("slurm", {})
    if not isinstance(slurm_config, dict):
        slurm_config = {}
    setup_commands = build_slurm_setup_commands(
        modules=_normalize_notebook_modules(slurm_config.get("modules")),
        environment=_resolve_slurm_environment(slurm_config),
        pre_activate_commands=_normalize_notebook_shell_commands(slurm_config.get("pre_activate_commands")),
        prepare_directories=_resolve_slurm_prepare_directories(slurm_config),
    )
    backend = spec["backend"]
    return "\n".join(
        [
            "set -euo pipefail",
            *setup_commands,
            f"RUNTIME_IMAGE={_double_quoted_remote_shell_value(spec['runtime_image'])}",
            f"IMAGE_SOURCE={_double_quoted_remote_shell_value(spec['source_image'])}",
            'IMAGE_ROOT="$(dirname "$RUNTIME_IMAGE")"',
            'TMP_IMAGE="${RUNTIME_IMAGE}.tmp.$$"',
            'cleanup(){ rm -f "$TMP_IMAGE" 2>/dev/null || true; }',
            "trap cleanup EXIT INT TERM",
            'mkdir -p "$IMAGE_ROOT"',
            'rm -rf "${RUNTIME_IMAGE}.lock.d"',
            'if [ ! -s "$RUNTIME_IMAGE" ]; then',
            f"  {shlex.quote(backend)} pull \"$TMP_IMAGE\" \"$IMAGE_SOURCE\"",
            '  mv "$TMP_IMAGE" "$RUNTIME_IMAGE"',
            "fi",
            'ls -lh "$RUNTIME_IMAGE"',
            *_build_nextflow_prepare_remote_lines(spec),
        ]
    )


def _build_nextflow_prepare_remote_lines(spec: dict[str, str]) -> list[str]:
    if spec.get("nextflow_enabled") != "true":
        return []
    return [
        f"NEXTFLOW_HOME={_double_quoted_remote_shell_value(spec['nextflow_home'])}",
        f"NEXTFLOW_VERSION={_double_quoted_remote_shell_value(spec['nextflow_version'])}",
        f"NEXTFLOW_JAR_NAME={_double_quoted_remote_shell_value(spec['nextflow_jar_name'])}",
        f"NEXTFLOW_JAR_URL={_double_quoted_remote_shell_value(spec['nextflow_jar_url'])}",
        'NEXTFLOW_JAR="$NEXTFLOW_HOME/framework/$NEXTFLOW_VERSION/$NEXTFLOW_JAR_NAME"',
        'TMP_NEXTFLOW_JAR="${NEXTFLOW_JAR}.tmp.$$"',
        'mkdir -p "$(dirname "$NEXTFLOW_JAR")"',
        'if [ ! -s "$NEXTFLOW_JAR" ]; then',
        '  if command -v curl >/dev/null 2>&1; then',
        '    curl -fL --retry 3 --connect-timeout 30 -o "$TMP_NEXTFLOW_JAR" "$NEXTFLOW_JAR_URL"',
        '  elif command -v wget >/dev/null 2>&1; then',
        '    wget -O "$TMP_NEXTFLOW_JAR" "$NEXTFLOW_JAR_URL"',
        "  else",
        '    echo "ERROR: curl or wget is required to fetch $NEXTFLOW_JAR_URL" >&2',
        "    exit 127",
        "  fi",
        '  mv "$TMP_NEXTFLOW_JAR" "$NEXTFLOW_JAR"',
        "fi",
        'ls -lh "$NEXTFLOW_JAR"',
    ]


def _render_hpc_container_prepare_report(
    *,
    project_name: str,
    profile: Any,
    config_path: Path,
    spec: dict[str, str],
    ssh_command: list[str],
) -> list[str]:
    lines = [
        f"HPC container prepare plan for {project_name}",
        "",
        f"SSH profile: {profile.name} ({profile.role})",
        f"SSH config: {config_path}",
        f"Runtime profile: {spec.get('runtime_profile') or 'not declared'}",
        f"Backend: {spec['backend']}",
        f"Source image: {spec['source_image']}",
        f"Remote runtime image: {spec['runtime_image']}",
    ]
    if spec.get("pull_mode"):
        lines.append(f"Project pull mode: {spec['pull_mode']}")
    if spec.get("nextflow_enabled") == "true":
        lines.extend(
            [
                f"Nextflow cache: {spec['nextflow_home']}",
                f"Nextflow jar URL: {spec['nextflow_jar_url']}",
            ]
        )
    lines.extend(
        [
            "",
            f"SSH command: {_render_command(ssh_command)}",
            "",
            "Next commands",
            "- Add --execute to run this plan.",
            f"- rp hpc sync project --project {project_name}",
            "- Review the rendered sync plan before authorizing remote changes.",
            f"- rp hpc sync project --project {project_name} --execute",
        ]
    )
    return lines


def _normalize_sif_name(value: str) -> str:
    return value if value.endswith(".sif") else f"{value}.sif"


def _join_remote_path(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf}" if root.rstrip("/") else leaf


def _double_quoted_remote_shell_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _build_context(
    project_name: str,
    batch_name: str | None = None,
    *,
    allow_missing_batch: bool = False,
    require_default_batch: bool = True,
) -> dict[str, Any]:
    bundle, resolved_root = _load_bundle(project_name)
    errors = validate_project_bundle(bundle, root=resolved_root, require_default_batch=require_default_batch)
    if errors:
        raise SystemExit(json.dumps({"valid": False, "errors": errors}, indent=2))

    workspace = bundle["workspace"]
    dataset_config = bundle["dataset"]["dataset"]
    preprocessing = bundle["preprocessing"]["preprocessing"]
    compute = bundle["compute"]["compute"]
    models = bundle.get("models", {}).get("models", {})
    selected_batch = batch_name or preprocessing["default_batch"]
    batch_path = Path(bundle["project_root"]) / "manifests" / "batches" / f"{selected_batch}.tsv"
    rows = _read_tsv(batch_path) if batch_path.exists() else []
    if not rows and not allow_missing_batch:
        raise SystemExit(json.dumps({"valid": False, "errors": [f"Batch manifest is empty: {batch_path}"]}, indent=2))

    context: dict[str, Any] = {
        "workspace_root": resolved_root,
        "workspace": workspace,
        "bundle": bundle,
        "paths": workspace_paths(resolved_root, workspace),
        "slice": project_slice(bundle),
        "project_root": Path(bundle["project_root"]),
        "project": bundle["project"],
        "dataset": dataset_config,
        "compute": compute,
        "preprocessing": preprocessing,
        "models": models,
        "hpc_target": bundle.get("hpc_target"),
        "batch": {
            "name": selected_batch,
            "path": str(batch_path),
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
            "selected_row": rows[0] if rows else {},
        },
    }

    if context["slice"] == "bids":
        context["tool_adapter"] = load_bids_tool_adapter(preprocessing)
        requires_input_derivative = context["tool_adapter"].requires_input_derivative()
        runtime_profile_name = str(preprocessing.get("runtime_profile") or preprocessing.get("tool") or "").strip()
        runtime_profile = _resolve_analysis_runtime_profile(compute, runtime_profile_name)
        context["runtime_profile_name"] = runtime_profile_name
        context["runtime_profile"] = runtime_profile
        context["requires_input_derivative"] = requires_input_derivative
        context["compute"] = _merge_analysis_runtime_profile_compute(compute, runtime_profile)
        context["dataset_root"] = resolve_bids_dataset_root(bundle, root=resolved_root)
        context["remote_dataset_root"] = resolve_bids_remote_dataset_root(bundle)
        if requires_input_derivative:
            context["input_derivative_root"] = resolve_bids_input_derivative_root(bundle, root=resolved_root)
            context["remote_input_derivative_root"] = resolve_bids_remote_input_derivative_root(bundle)
        else:
            context["input_derivative_root"] = context["dataset_root"]
            context["remote_input_derivative_root"] = context["remote_dataset_root"]
        context["pipeline_root"] = pipeline_path(resolved_root, workspace, preprocessing["pipeline"])
        context["pipeline_defaults"] = load_yaml(context["pipeline_root"] / "config" / "defaults.yaml")
    else:
        context["dataset_root"] = dataset_path(resolved_root, workspace, dataset_config["primary"])
        context["canonical_dataset_root"] = dataset_path(resolved_root, workspace, dataset_config["canonical_dataset"])
        context["canonical_features_root"] = context["canonical_dataset_root"] / dataset_config["canonical_features_root"]
        context["feature_table_path"] = context["canonical_features_root"] / context["batch"]["selected_row"]["feature_table"]
    return context


def _build_analysis_context(
    project_name: str,
    batch_name: str | None = None,
    *,
    stage_name: str | None = None,
    model_ref: str | None = None,
    allow_missing_batch: bool = False,
    require_default_batch: bool = True,
) -> dict[str, Any]:
    bundle, resolved_root = _load_bundle(project_name)
    errors = validate_project_bundle(bundle, root=resolved_root, require_default_batch=require_default_batch)
    if errors:
        raise SystemExit(json.dumps({"valid": False, "errors": errors}, indent=2))

    workspace = bundle["workspace"]
    dataset_config = bundle["dataset"]["dataset"]
    analysis = bundle["analysis"]["analysis"]
    analysis_models = bundle.get("analysis_models", {}).get("models", {})
    analysis_groupings = bundle.get("analysis_groupings", {}).get("groupings", {})
    defaults = analysis.get("defaults", {})
    compute = bundle["compute"]["compute"]

    selected_stage_name = _normalize_optional_cli_value(stage_name) or str(defaults["stage"])
    stage_config = analysis.get("stages", {}).get(selected_stage_name)
    if not isinstance(stage_config, dict):
        raise SystemExit(json.dumps({"error": f"Unknown analysis stage: {selected_stage_name}"}, indent=2))
    if selected_stage_name != "first_level":
        raise SystemExit(json.dumps({"error": "Phase 1 supports only analysis stage first_level."}, indent=2))

    selected_tool_name = str(stage_config.get("tool") or defaults["tool"])
    tool_entry = analysis.get("tools", {}).get(selected_tool_name)
    if not isinstance(tool_entry, dict):
        raise SystemExit(json.dumps({"error": f"Unknown analysis tool: {selected_tool_name}"}, indent=2))

    resolved_model_ref = _normalize_optional_cli_value(model_ref) or str(stage_config.get("model_ref") or defaults["model_ref"])
    selected_model = analysis_models.get(resolved_model_ref)
    if not isinstance(selected_model, dict):
        raise SystemExit(json.dumps({"error": f"Unknown analysis model ref: {resolved_model_ref}"}, indent=2))

    runtime_profile_name = str(tool_entry.get("runtime_profile", "")).strip()
    runtime_profile = _resolve_analysis_runtime_profile(compute, runtime_profile_name)
    merged_compute = _merge_analysis_runtime_profile_compute(compute, runtime_profile)

    selected_batch = batch_name or str(stage_config["default_batch"])
    batch_path = Path(bundle["project_root"]) / "manifests" / "batches" / f"{selected_batch}.tsv"
    rows = _read_tsv(batch_path) if batch_path.exists() else []
    if not rows and not allow_missing_batch:
        raise SystemExit(json.dumps({"valid": False, "errors": [f"Batch manifest is empty: {batch_path}"]}, indent=2))

    context: dict[str, Any] = {
        "workspace_root": resolved_root,
        "workspace": workspace,
        "bundle": bundle,
        "paths": workspace_paths(resolved_root, workspace),
        "slice": project_slice(bundle),
        "project_root": Path(bundle["project_root"]),
        "project": bundle["project"],
        "dataset": dataset_config,
        "compute": merged_compute,
        "analysis": analysis,
        "analysis_inputs": analysis.get("inputs", {}),
        "analysis_models": analysis_models,
        "analysis_groupings": analysis_groupings,
        "analysis_stage_name": selected_stage_name,
        "analysis_stage": stage_config,
        "analysis_tool_name": selected_tool_name,
        "analysis_tool": tool_entry,
        "analysis_model_ref": resolved_model_ref,
        "analysis_model": selected_model,
        "runtime_profile_name": runtime_profile_name,
        "runtime_profile": runtime_profile,
        "hpc_target": bundle.get("hpc_target"),
        "batch": {
            "name": selected_batch,
            "path": str(batch_path),
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
            "selected_row": rows[0] if rows else {},
        },
    }

    context["dataset_root"] = resolve_bids_dataset_root(bundle, root=resolved_root)
    context["input_derivative_root"] = resolve_bids_input_derivative_root(bundle, root=resolved_root)
    context["remote_dataset_root"] = resolve_bids_remote_dataset_root(bundle)
    context["remote_input_derivative_root"] = resolve_bids_remote_input_derivative_root(bundle)
    context["pipeline_root"] = pipeline_path(resolved_root, workspace, analysis["pipeline"])
    context["pipeline_defaults"] = load_yaml(context["pipeline_root"] / "config" / "defaults.yaml")
    context["tool_adapter"] = load_bids_analysis_tool_adapter(tool_entry)
    context["analysis_input_roots"] = _resolve_analysis_input_roots_for_context(
        analysis=analysis,
        workspace_root=resolved_root,
        project_root=Path(bundle["project_root"]),
    )
    data_roots: list[dict[str, Any]] = [
        build_data_root_spec(
            "raw-dataset-root",
            context["dataset_root"],
            remote_root=context["remote_dataset_root"],
        ),
        build_data_root_spec(
            "input-derivative-root",
            context["input_derivative_root"],
            remote_root=context["remote_input_derivative_root"],
        ),
    ]
    project_data_roots, project_data_root_errors = resolve_project_hpc_data_root_declarations(
        bundle["project"],
        workspace_root=resolved_root,
        project_root=Path(bundle["project_root"]),
        require_remote_root=False,
    )
    data_roots, project_merge_errors = merge_declared_data_roots(
        data_roots,
        [*project_data_roots, *context["analysis_input_roots"].values()],
        conflict_label="Project data roots",
    )
    adapter_data_roots = adapter_data_root_declarations(context=context)
    data_roots, adapter_merge_errors = merge_declared_data_roots(
        data_roots,
        adapter_data_roots,
        conflict_label="Adapter data roots",
    )
    context["data_roots"] = data_roots
    context["analysis_adapter_data_roots"] = adapter_data_roots
    context["validation_errors"] = list(
        dict.fromkeys([*project_data_root_errors, *project_merge_errors, *adapter_merge_errors])
    )
    if context["validation_errors"]:
        raise SystemExit(json.dumps({"valid": False, "errors": context["validation_errors"]}, indent=2))
    return context


def _resolve_analysis_input_roots_for_context(
    *,
    analysis: dict[str, Any],
    workspace_root: Path,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    resolved_roots, _ = resolve_analysis_external_input_root_declarations(
        analysis,
        workspace_root=workspace_root,
        project_root=project_root,
        require_remote_root=False,
    )
    return {
        str(root_spec["name"]): dict(root_spec)
        for root_spec in resolved_roots
        if root_spec.get("name")
    }


def _analysis_input_root_refs(context: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for input_config in context.get("analysis_inputs", {}).values():
        if not isinstance(input_config, dict):
            continue
        root_ref = _normalize_optional_cli_value(input_config.get("root_ref"))
        if root_ref is not None and root_ref not in refs:
            refs.append(root_ref)
    return refs


def _selected_analysis_external_roots(context: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    analysis_input_roots = context.get("analysis_input_roots", {})
    if isinstance(analysis_input_roots, dict):
        for root_ref in _analysis_input_root_refs(context):
            root_spec = analysis_input_roots.get(root_ref)
            if not isinstance(root_spec, dict):
                continue
            resolved_path = Path(root_spec["path"]).resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            selected.append(root_spec)
    for root_spec in context.get("analysis_adapter_data_roots", []):
        resolved_path = Path(root_spec["path"]).resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        selected.append(root_spec)
    return selected


def _ensure_analysis_slurm_external_roots(context: dict[str, Any]) -> None:
    workspace_root = Path(context["workspace_root"]).resolve()
    missing: list[dict[str, str]] = []
    for root_spec in _selected_analysis_external_roots(context):
        source_path = Path(root_spec["path"]).resolve()
        try:
            source_path.relative_to(workspace_root)
        except ValueError:
            remote_root = _normalize_optional_cli_value(root_spec.get("remote_root"))
            if remote_root is None:
                missing.append(
                    {
                        "label": str(root_spec.get("label", "analysis-input-root")),
                        "path": str(source_path),
                        "source": str(root_spec.get("source", "analysis")),
                    }
                )
        else:
            continue
    if not missing:
        return
    raise SystemExit(
        json.dumps(
            {
                "error": "Analysis SLURM planning requires remote destinations for selected external analysis input roots.",
                "missing_roots": missing,
            },
            indent=2,
        )
    )


def _analysis_root_execution_value(*, context: dict[str, Any], root_spec: dict[str, Any], mode: str) -> str:
    source_path = Path(root_spec["path"]).resolve()
    workspace_root = Path(context["workspace_root"]).resolve()
    try:
        relative = source_path.relative_to(workspace_root)
    except ValueError:
        relative = None
    if mode != "slurm":
        if relative is not None:
            return relative.as_posix()
        return str(source_path)
    if relative is not None:
        return relative.as_posix()
    remote_root = _normalize_optional_cli_value(root_spec.get("remote_root"))
    if remote_root is None:
        return str(source_path)
    return remote_root


def _analysis_manifest_input_roots(*, context: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    rendered: dict[str, dict[str, Any]] = {}
    for name, root_spec in context.get("analysis_input_roots", {}).items():
        if not isinstance(root_spec, dict):
            continue
        rendered[str(name)] = {
            "path": _analysis_root_execution_value(context=context, root_spec=root_spec, mode=mode),
            "remote_root": _normalize_optional_cli_value(root_spec.get("remote_root")),
            "sync_enabled": bool(root_spec.get("sync_enabled", True)),
        }
    return rendered


def _analysis_manifest_inputs(*, context: dict[str, Any], mode: str) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for input_name, input_config in context.get("analysis_inputs", {}).items():
        if not isinstance(input_config, dict):
            rendered[input_name] = input_config
            continue
        rendered_config = dict(input_config)
        if _normalize_optional_cli_value(rendered_config.get("root_ref")) is None:
            direct_root = _normalize_optional_cli_value(rendered_config.get("root"))
            if direct_root is not None:
                source_path = Path(direct_root).expanduser()
                if not source_path.is_absolute():
                    source_path = (Path(context["workspace_root"]).resolve() / source_path).resolve()
                rendered_config["root"] = _analysis_root_execution_value(
                    context=context,
                    root_spec={
                        "path": source_path,
                        "remote_root": rendered_config.get("remote_root"),
                    },
                    mode=mode,
                )
        rendered[input_name] = rendered_config
    return rendered


def _default_project_batch_name(bundle: dict[str, Any]) -> str | None:
    preprocessing = bundle.get("preprocessing", {}).get("preprocessing", {})
    if isinstance(preprocessing, dict) and preprocessing.get("default_batch"):
        return str(preprocessing["default_batch"])
    analysis = bundle.get("analysis", {}).get("analysis", {})
    if not isinstance(analysis, dict):
        return None
    defaults = analysis.get("defaults", {})
    stage_name = str(defaults.get("stage", "")).strip()
    stage = analysis.get("stages", {}).get(stage_name)
    if isinstance(stage, dict) and stage.get("default_batch"):
        return str(stage["default_batch"])
    return None


def _default_context_batch_name(context: dict[str, Any]) -> str | None:
    preprocessing = context.get("preprocessing", {})
    if isinstance(preprocessing, dict) and preprocessing.get("default_batch"):
        return str(preprocessing["default_batch"])
    analysis_stage = context.get("analysis_stage")
    if isinstance(analysis_stage, dict) and analysis_stage.get("default_batch"):
        return str(analysis_stage["default_batch"])
    analysis = context.get("analysis", {})
    if isinstance(analysis, dict):
        defaults = analysis.get("defaults", {})
        stage_name = str(defaults.get("stage", "")).strip()
        stage = analysis.get("stages", {}).get(stage_name)
        if isinstance(stage, dict) and stage.get("default_batch"):
            return str(stage["default_batch"])
    return None


def _resolve_analysis_runtime_profile(compute: dict[str, Any], runtime_profile_name: str) -> dict[str, Any]:
    tool_profiles = compute.get("tool_profiles", {})
    if not isinstance(tool_profiles, dict):
        return {}
    profile = tool_profiles.get(runtime_profile_name, {})
    return dict(profile) if isinstance(profile, dict) else {}


def _merge_analysis_runtime_profile_compute(compute: dict[str, Any], runtime_profile: dict[str, Any]) -> dict[str, Any]:
    merged_compute = dict(compute)
    merged_slurm = dict(compute.get("slurm", {})) if isinstance(compute.get("slurm"), dict) else {}
    profile_slurm = runtime_profile.get("slurm", {}) if isinstance(runtime_profile.get("slurm"), dict) else {}

    if profile_slurm.get("modules") is not None:
        merged_slurm["modules"] = list(profile_slurm.get("modules", []))
    if profile_slurm.get("pre_activate_commands") is not None:
        merged_slurm["pre_activate_commands"] = list(profile_slurm.get("pre_activate_commands", []))
    if isinstance(profile_slurm.get("environment"), dict):
        environment = dict(merged_slurm.get("environment", {})) if isinstance(merged_slurm.get("environment"), dict) else {}
        environment.update(profile_slurm["environment"])
        merged_slurm["environment"] = environment
    if profile_slurm.get("prepare_directories") is not None:
        merged_slurm["prepare_directories"] = list(profile_slurm.get("prepare_directories", []))
    merged_compute["slurm"] = merged_slurm
    return merged_compute


def _scaffold_bids_preprocess_project(args: argparse.Namespace) -> dict[str, Any]:
    resolved_root = workspace_root()
    workspace = load_workspace_config(resolved_root)
    project_name = _validate_project_init_name(args.project)
    projects_root = project_path(resolved_root, workspace, "__project_root__").parent.resolve()
    project_root_path = (projects_root / project_name).resolve()
    try:
        project_root_path.relative_to(projects_root)
    except ValueError as exc:
        raise SystemExit(f"Project name must resolve within {projects_root}.") from exc
    if project_root_path.exists() and any(project_root_path.iterdir()):
        raise SystemExit(f"Project path already exists and is not empty: {project_root_path}")

    adapter = load_registered_bids_tool_adapter(args.tool)
    adapter_ref = resolve_bids_tool_adapter_ref(args.tool)
    study_root = _validate_existing_local_directory(args.study_root, label="--study-root")
    derivative_root: Path | None = None
    if adapter.requires_input_derivative():
        if _normalize_optional_cli_value(args.derivative_root) is None:
            raise SystemExit("--derivative-root is required for derivative-backed BIDS preprocessing tools.")
        derivative_root = _validate_existing_local_directory(args.derivative_root, label="--derivative-root")
    elif _normalize_optional_cli_value(args.derivative_root) is not None:
        derivative_root = _validate_existing_local_directory(args.derivative_root, label="--derivative-root")
    defaults = adapter.scaffold_project_defaults(
        project_name=project_name,
        study_root=str(study_root),
        derivative_root=str(derivative_root) if derivative_root is not None else None,
        task_id=_normalize_optional_cli_value(args.task_id),
    )
    runtime_default = resolve_workspace_hpc_runtime_default(workspace)
    if runtime_default is not None:
        defaults["compute"] = _merge_scaffold_runtime_compute_defaults(
            defaults.get("compute"),
            runtime_default,
        )

    project_config = {
        "name": project_name,
        "version": "0.1.0",
        "datasets": [project_name],
        "pipelines": [str(defaults["pipeline"])],
        "compute_profile": "local",
        "notes": f"Scaffolded BIDS preprocessing overlay for {args.tool}.",
    }
    dataset_config = {
        "dataset": {
            "primary": project_name,
            "bids_root": str(study_root),
        }
    }
    if defaults.get("input_derivative") is not None:
        dataset_config["dataset"]["input_derivative"] = str(defaults["input_derivative"])
    if derivative_root is not None:
        dataset_config["dataset"]["input_derivative_root"] = str(derivative_root)
    remote_study_root = _validate_remote_posix_root(args.remote_study_root, label="--remote-study-root")
    remote_derivative_root = _validate_remote_posix_root(args.remote_derivative_root, label="--remote-derivative-root")
    if remote_study_root is not None:
        dataset_config["dataset"]["remote_bids_root"] = remote_study_root
    if remote_derivative_root is not None:
        dataset_config["dataset"]["remote_input_derivative_root"] = remote_derivative_root

    compute_config = {"compute": dict(defaults.get("compute", {}))}
    preprocessing_config = {
        "preprocessing": {
            "slice": "bids",
            "pipeline": str(defaults["pipeline"]),
            "tool": str(args.tool),
            "tool_adapter": adapter_ref,
            "default_batch": str(defaults["default_batch"]),
            "local_profile": str(defaults["local_profile"]),
            "slurm_profile": str(defaults["slurm_profile"]),
            "tool_options": dict(defaults.get("tool_options", {})),
            "publish_back": dict(defaults.get("publish_back", {})),
        }
    }
    if defaults.get("input_derivative") is not None:
        preprocessing_config["preprocessing"]["input_derivative"] = str(defaults["input_derivative"])
    if defaults.get("runtime_profile") is not None:
        preprocessing_config["preprocessing"]["runtime_profile"] = str(defaults["runtime_profile"])
    if isinstance(defaults.get("inputs"), dict):
        preprocessing_config["preprocessing"]["inputs"] = dict(defaults["inputs"])
    if isinstance(defaults.get("output"), dict):
        preprocessing_config["preprocessing"]["output"] = dict(defaults["output"])
    if defaults.get("task_id"):
        preprocessing_config["preprocessing"]["task_id"] = str(defaults["task_id"])

    created_files = [
        write_yaml(project_root_path / "project.yaml", project_config),
        write_yaml(project_root_path / "config" / "dataset.yaml", dataset_config),
        write_yaml(project_root_path / "config" / "compute.yaml", compute_config),
        write_yaml(project_root_path / "config" / "preprocessing.yaml", preprocessing_config),
    ]
    batch_path = project_root_path / "manifests" / "batches" / f"{defaults['default_batch']}.tsv"
    _write_batch_manifest(batch_path, [])
    created_files.append(batch_path)
    return {
        "project_name": project_name,
        "tool": str(args.tool),
        "adapter": adapter_ref,
        "project_root": project_root_path,
        "created_files": created_files,
        "default_batch": str(defaults["default_batch"]),
    }


def _scaffold_bids_analysis_project(args: argparse.Namespace) -> dict[str, Any]:
    resolved_root = workspace_root()
    workspace = load_workspace_config(resolved_root)
    project_name = _validate_project_init_name(args.project)
    target_default = None
    hpc_target = _normalize_optional_cli_value(getattr(args, "hpc_target", None))
    if hpc_target is not None:
        target_default = apply_hpc_target_defaults(project_name=project_name, target_name=hpc_target, root=resolved_root)
    projects_root = project_path(resolved_root, workspace, "__project_root__").parent.resolve()
    project_root_path = (projects_root / project_name).resolve()
    try:
        project_root_path.relative_to(projects_root)
    except ValueError as exc:
        raise SystemExit(f"Project name must resolve within {projects_root}.") from exc
    if project_root_path.exists() and any(project_root_path.iterdir()):
        raise SystemExit(f"Project path already exists and is not empty: {project_root_path}")

    adapter = load_registered_bids_analysis_tool_adapter(args.tool)
    adapter_ref = resolve_bids_analysis_tool_adapter_ref(args.tool)
    study_root = _validate_existing_local_directory(args.study_root, label="--study-root")
    derivative_root = _validate_existing_local_directory(args.derivative_root, label="--derivative-root")
    events_root = (
        _validate_existing_local_directory(args.events_root, label="--events-root")
        if _normalize_optional_cli_value(getattr(args, "events_root", None)) is not None
        else None
    )
    confounds_root = (
        _validate_existing_local_directory(args.confounds_root, label="--confounds-root")
        if _normalize_optional_cli_value(getattr(args, "confounds_root", None)) is not None
        else None
    )
    remote_events_root = _validate_remote_posix_root(getattr(args, "remote_events_root", None), label="--remote-events-root")
    remote_confounds_root = _validate_remote_posix_root(
        getattr(args, "remote_confounds_root", None),
        label="--remote-confounds-root",
    )
    try:
        defaults = adapter.scaffold_project_defaults(
            project_name=project_name,
            study_root=str(study_root),
            derivative_root=str(derivative_root),
            task_id=_normalize_optional_cli_value(args.task_id),
            template=_normalize_optional_cli_value(getattr(args, "template", None)),
            events_root=str(events_root) if events_root is not None else None,
            confounds_root=str(confounds_root) if confounds_root is not None else None,
            remote_events_root=remote_events_root,
            remote_confounds_root=remote_confounds_root,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    runtime_default = resolve_workspace_hpc_runtime_default(workspace)
    if runtime_default is not None:
        defaults["compute"] = _merge_scaffold_runtime_compute_defaults(defaults.get("compute"), runtime_default)
    if target_default is not None:
        defaults["compute"] = merge_hpc_target_compute_defaults(defaults.get("compute"), target_default)

    project_config = {
        "name": project_name,
        "version": "0.1.0",
        "datasets": [project_name],
        "pipelines": [str(defaults["pipeline"])],
        "compute_profile": "local",
        "notes": f"Scaffolded BIDS analysis overlay for {args.tool}.",
    }
    dataset_config = {
        "dataset": {
            "primary": project_name,
            "bids_root": str(study_root),
            "input_derivative": str(defaults["input_derivative"]),
            "input_derivative_root": str(derivative_root),
        }
    }
    remote_study_root = _validate_remote_posix_root(args.remote_study_root, label="--remote-study-root")
    remote_derivative_root = _validate_remote_posix_root(args.remote_derivative_root, label="--remote-derivative-root")
    if remote_study_root is not None:
        dataset_config["dataset"]["remote_bids_root"] = remote_study_root
    if remote_derivative_root is not None:
        dataset_config["dataset"]["remote_input_derivative_root"] = remote_derivative_root

    compute_config = {"compute": dict(defaults.get("compute", {}))}
    analysis_config = {
        "analysis": {
            "slice": "bids",
            "pipeline": str(defaults["pipeline"]),
            "local_profile": str(defaults["local_profile"]),
            "slurm_profile": str(defaults["slurm_profile"]),
            "defaults": {
                "tool": str(args.tool),
                "stage": "first_level",
                "model_ref": str(defaults["model_ref"]),
            },
            "tools": {
                str(args.tool): {
                    "adapter": adapter_ref,
                    "runtime_profile": str(defaults["runtime_profile"]),
                }
            },
            "inputs": dict(defaults["inputs"]),
            "stages": {
                "first_level": {
                    "tool": str(args.tool),
                    "default_batch": str(defaults["default_batch"]),
                    "model_ref": str(defaults["model_ref"]),
                    "validation": dict(defaults["stage"]["validation"]),
                    "overwrite": dict(defaults["stage"]["overwrite"]),
                    "settings": dict(defaults["stage"]["settings"]),
                }
            },
        }
    }
    if defaults.get("template") is not None:
        analysis_config["analysis"]["scaffold"] = {"template": str(defaults["template"])}
        if defaults.get("template_reason") is not None:
            analysis_config["analysis"]["scaffold"]["reason"] = str(defaults["template_reason"])
    if isinstance(defaults.get("external_input_roots"), dict):
        analysis_config["analysis"]["external_input_roots"] = dict(defaults["external_input_roots"])
    if defaults.get("task_id") is not None:
        analysis_config["analysis"]["stages"]["first_level"]["task_id"] = str(defaults["task_id"])

    created_files = [
        write_yaml(project_root_path / "project.yaml", project_config),
        write_yaml(project_root_path / "config" / "dataset.yaml", dataset_config),
        write_yaml(project_root_path / "config" / "compute.yaml", compute_config),
        write_yaml(project_root_path / "config" / "analysis.yaml", analysis_config),
        write_yaml(project_root_path / "config" / "analysis" / "models" / f"{defaults['model_ref']}.yaml", {"model": defaults["model"]}),
    ]
    grouping_dir = ensure_dir(project_root_path / "config" / "analysis" / "groupings")
    batch_path = project_root_path / "manifests" / "batches" / f"{defaults['default_batch']}.tsv"
    _write_batch_manifest(batch_path, [])
    created_files.extend([grouping_dir / ".gitkeep", batch_path])
    (grouping_dir / ".gitkeep").write_text("", encoding="utf-8")
    return {
        "project_name": project_name,
        "tool": str(args.tool),
        "adapter": adapter_ref,
        "project_root": project_root_path,
        "created_files": created_files,
        "default_batch": str(defaults["default_batch"]),
        "template": defaults.get("template"),
        "template_reason": defaults.get("template_reason"),
        "hpc_target": hpc_target,
    }


def _scaffold_tabular_model_project(args: argparse.Namespace) -> dict[str, Any]:
    resolved_root = workspace_root()
    workspace = load_workspace_config(resolved_root)
    project_name = _validate_project_init_name(args.project)
    dataset_name = _validate_simple_scaffold_name(args.dataset or project_name, label="--dataset")
    canonical_dataset_name = _validate_simple_scaffold_name(
        args.canonical_dataset or dataset_name,
        label="--canonical-dataset",
    )
    batch_name = _validate_simple_scaffold_name(args.batch or "default", label="--batch")
    canonical_features_root = _validate_relative_scaffold_subpath(
        args.canonical_features_root or f"derivatives/features/{project_name}",
        label="--canonical-features-root",
    )

    projects_root = project_path(resolved_root, workspace, "__project_root__").parent.resolve()
    project_root_path = (projects_root / project_name).resolve()
    try:
        project_root_path.relative_to(projects_root)
    except ValueError as exc:
        raise SystemExit(f"Project name must resolve within {projects_root}.") from exc
    if project_root_path.exists() and any(project_root_path.iterdir()):
        raise SystemExit(f"Project path already exists and is not empty: {project_root_path}")

    dataset_root_path = dataset_path(resolved_root, workspace, dataset_name)
    canonical_dataset_root_path = dataset_path(resolved_root, workspace, canonical_dataset_name)
    canonical_features_root_path = canonical_dataset_root_path / canonical_features_root

    compute_defaults: dict[str, Any] = {
        "default_profile": "local",
        "local": {"jobs": 1},
        "policy": {
            "presets": {
                "tabular-preprocess": {
                    "cpus": 1,
                    "ram_gb": 4,
                    "threads": 1,
                    "n_jobs": 1,
                },
                "tabular-model": {
                    "cpus": 2,
                    "ram_gb": 4,
                    "threads": 1,
                    "n_jobs": 1,
                },
            },
            "workloads": {
                "tabular_preprocess": {"preset": "tabular-preprocess"},
                "tabular_train_model": {"preset": "tabular-model"},
                "tabular_evaluate_model": {"preset": "tabular-model"},
            },
        },
        "slurm": {
            "cpus": 2,
            "mem": "4G",
            "time": "00:30:00",
            "ssh_host": "${RP_HPC_HOST:-}",
            "remote_workspace_root": "${RP_REMOTE_WORKSPACE_ROOT:-}",
            "remote_artifacts_root": "${RP_REMOTE_ARTIFACTS_ROOT:-}",
        },
    }
    runtime_default = resolve_workspace_hpc_runtime_default(workspace)
    if runtime_default is not None:
        compute_defaults = _merge_scaffold_runtime_compute_defaults(compute_defaults, runtime_default)

    project_datasets = [dataset_name]
    if canonical_dataset_name != dataset_name:
        project_datasets.append(canonical_dataset_name)
    project_config = {
        "name": project_name,
        "version": "0.1.0",
        "datasets": project_datasets,
        "notes": f"Scaffolded tabular model overlay for {project_name}.",
    }
    dataset_config = {
        "dataset": {
            "primary": dataset_name,
            "canonical_dataset": canonical_dataset_name,
            "canonical_features_root": canonical_features_root.as_posix(),
        }
    }
    compute_config = {"compute": compute_defaults}
    preprocessing_config = {
        "preprocessing": {
            "slice": "tabular",
            "default_batch": batch_name,
            "local_profile": "local",
            "slurm_profile": "slurm",
            "split_seed": 23,
            "test_fraction": 0.25,
        }
    }
    models_config = {
        "models": {
            "default": {
                "kind": "logistic_regression",
                "feature_columns": ["feature_1", "feature_2"],
                "learning_rate": 0.2,
                "iterations": 350,
            }
        }
    }

    created_files = [
        write_yaml(project_root_path / "project.yaml", project_config),
        write_yaml(project_root_path / "config" / "dataset.yaml", dataset_config),
        write_yaml(project_root_path / "config" / "compute.yaml", compute_config),
        write_yaml(project_root_path / "config" / "preprocessing.yaml", preprocessing_config),
        write_yaml(project_root_path / "config" / "models.yaml", models_config),
    ]
    batch_path = project_root_path / "manifests" / "batches" / f"{batch_name}.tsv"
    _write_tabular_batch_manifest_template(batch_path)
    created_files.append(batch_path)
    created_directories = [
        ensure_dir(dataset_root_path),
        ensure_dir(canonical_dataset_root_path),
        ensure_dir(canonical_features_root_path),
    ]
    return {
        "project_name": project_name,
        "project_root": project_root_path,
        "created_files": created_files,
        "created_directories": created_directories,
        "dataset_name": dataset_name,
        "canonical_dataset_name": canonical_dataset_name,
        "canonical_features_root": canonical_features_root.as_posix(),
        "default_batch": batch_name,
    }


def _merge_scaffold_runtime_compute_defaults(
    compute_defaults: Any,
    runtime_default: dict[str, Any],
) -> dict[str, Any]:
    return merge_workspace_hpc_runtime_compute_defaults(compute_defaults, runtime_default)


def _plan_run(args: argparse.Namespace, *, mode: str, execute: bool) -> int:
    if args.run_action == "preprocess" and args.run_target == "bids":
        context = _build_context(_resolve_project_name(args.project), batch_name=args.batch)
        return _plan_bids_run(args, context=context, mode=mode, execute=execute)
    if args.run_action == "analysis" and args.run_target == "bids":
        _validate_analysis_run_cli_overrides(args)
        context = _build_analysis_context(
            _resolve_project_name(args.project),
            batch_name=args.batch,
            stage_name=getattr(args, "stage", None),
            model_ref=getattr(args, "model", None),
        )
        return _plan_bids_analysis_run(args, context=context, mode=mode, execute=execute)
    if args.run_action == "analysis" and args.run_target == "tabular":
        context = _build_project_hpc_context_for_tabular_analysis(_resolve_project_name(args.project))
        return _plan_tabular_analysis_run(args, context=context, mode=mode, execute=execute)
    context = _build_context(_resolve_project_name(args.project), batch_name=args.batch)
    if context["slice"] != "tabular":
        raise SystemExit(json.dumps({"error": f"Project {context['project']['name']} does not support {args.run_action} {args.run_target}."}, indent=2))
    if args.run_action == "preprocess" and args.run_target == "tabular":
        return _plan_tabular_run(args, context=context, mode=mode, execute=execute)
    if args.run_target == "model" and args.run_action in {"train", "evaluate"}:
        return _plan_tabular_run(args, context=context, mode=mode, execute=execute)
    raise SystemExit(json.dumps({"error": f"Unsupported run target: {args.run_action} {args.run_target}"}, indent=2))


def _print_remote_submission_plan(
    *,
    run_id: str,
    run_root_path: Path,
    dry_run: bool,
    stage_report: dict[str, Any],
    submit_plan: dict[str, Any],
    additional_local_files: list[Path] | None = None,
) -> None:
    local_files = {path.resolve() for path in run_root_path.rglob("*") if path.is_file()}
    local_files.update(path.resolve() for path in additional_local_files or [] if path.is_file())
    print(
        json.dumps(
            {
                "run_id": run_id,
                "mode": "plan",
                "dry_run": dry_run,
                "local_files": sorted(str(path) for path in local_files),
                "stage": stage_report,
                "submission": submit_plan,
            },
            indent=2,
        )
    )


def _inspect_shared_planned_run(
    *,
    context: dict[str, Any],
    run_id: str,
    expected_slice: str,
    expected_workflow: dict[str, str],
    expected_project: str,
    allow_claim: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_root_path = run_path(context["paths"]["artifacts_root"], run_id)
    _require_real_directory(run_root_path, run_id=run_id, label="run root")
    if not allow_claim and path_entry_exists(claim_path(context["paths"]["artifacts_root"], run_id)):
        raise RunLifecycleError.for_reuse(run_id, "an execution claim already exists")
    manifest = _read_run_control_mapping(run_root_path / "run-manifest.yaml", run_id=run_id, label="run manifest")
    status = _read_run_control_mapping(run_root_path / "status.yaml", run_id=run_id, label="run status")
    if manifest.get("run_id") != run_id or status.get("run_id") != run_id:
        raise RunLifecycleError.for_reuse(run_id, "run control files identify a different run")
    required_status_keys = {"run_id", "state", "last_updated", "job_id", "mode"}
    if not required_status_keys.issubset(status):
        raise RunLifecycleError.for_reuse(run_id, "the run status is incomplete")
    if not all(isinstance(status.get(key), str) for key in ("state", "last_updated", "job_id", "mode")):
        raise RunLifecycleError.for_reuse(run_id, "the run status contains invalid metadata")
    if manifest.get("slice") != expected_slice or manifest.get("workflow") != expected_workflow:
        raise RunLifecycleError.for_reuse(run_id, "the run is owned by a different workflow")
    project = manifest.get("project")
    if not isinstance(project, dict) or project.get("name") != expected_project:
        raise RunLifecycleError.for_reuse(run_id, "the run belongs to a different project")
    if status.get("state") != "planned":
        raise RunLifecycleError.for_reuse(run_id, f"status is {status.get('state')!r}, not 'planned'")
    return manifest, status, run_root_path


def _submit_bids_run(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    discovery_report: dict[str, Any] | None = None
    if getattr(args, "discover", False):
        discovery_context = _build_context(
            project_name,
            batch_name=args.batch,
            allow_missing_batch=True,
            require_default_batch=False,
        )
        discovery_report = _discover_bids_batch_for_submit(args, context=discovery_context)
    context = _build_context(project_name, batch_name=args.batch)
    run_id = args.run_id if args.run_id is not None else _default_run_id("submit", "preprocess", "bids")
    submit_args = argparse.Namespace(**vars(args))
    submit_args.run_id = run_id
    submit_args.dry_run = bool(getattr(args, "dry_run", False))
    workspace_context = _workspace_context()
    execute = bool(getattr(args, "execute", False))
    existing_root = run_path(workspace_context["paths"]["artifacts_root"], run_id)
    if execute and path_entry_exists(existing_root):
        _inspect_shared_planned_run(
            context=workspace_context,
            run_id=run_id,
            expected_slice="bids",
            expected_workflow={"action": "preprocess", "target": "bids"},
            expected_project=context["project"]["name"],
            allow_claim=False,
        )
    else:
        _plan_bids_run(submit_args, context=context, mode="slurm", execute=False, quiet=True)

    manifest, status, run_root_path = _load_run(workspace_context, run_id)
    claim = None
    if execute:
        claim = acquire_execution_claim(workspace_context["paths"]["artifacts_root"], run_id)
        try:
            manifest, status, run_root_path = _inspect_shared_planned_run(
                context=workspace_context,
                run_id=run_id,
                expected_slice="bids",
                expected_workflow={"action": "preprocess", "target": "bids"},
                expected_project=context["project"]["name"],
                allow_claim=True,
            )
        except BaseException:
            claim.release()
            raise
    try:
        stage = build_stage_plan(
            workspace_root=workspace_context["workspace_root"],
            run_root=run_root_path,
            manifest=manifest,
            status=status,
            exclude_file=workspace_context["workspace_root"] / "ops" / "sync" / "rsync" / "exclude.txt",
            profile_name=getattr(args, "profile", None),
            role=getattr(args, "role", None),
            config_path=getattr(args, "config", None),
        )
    except ValueError as exc:
        if claim is not None:
            claim.release()
        raise SystemExit(str(exc)) from exc
    if not execute:
        try:
            submit_plan = build_submit_plan(
                manifest=manifest,
                status=stage["status"],
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
                workspace_root=workspace_context["workspace_root"],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        _print_remote_submission_plan(
            run_id=run_id,
            run_root_path=run_root_path,
            dry_run=submit_args.dry_run,
            stage_report=stage["report"],
            submit_plan=submit_plan,
            additional_local_files=(
                [_resolve_reference_path(workspace_context["workspace_root"], discovery_report["batch_path"])]
                if discovery_report is not None
                else None
            ),
        )
        return 0
    stage_execution = execute_stage_plan(stage["report"])
    if not stage_execution.get("ok"):
        next_status = dict(stage["status"])
        next_status["last_updated"] = _timestamp()
        next_status["state"] = "stage-failed"
        write_status(run_root_path, next_status)
        print(json.dumps({"run_id": run_id, "stage": stage["report"], "stage_execution": stage_execution}, indent=2))
        assert claim is not None
        claim.release()
        return int(stage_execution.get("returncode", 1))

    stage_status = dict(stage["status"])
    stage_status["last_updated"] = _timestamp()
    stage_status["state"] = "staged"
    write_status(run_root_path, stage_status)

    try:
        submit_plan = build_submit_plan(
            manifest=manifest,
            status=stage_status,
            profile_name=getattr(args, "profile", None),
            role=getattr(args, "role", None),
            config_path=getattr(args, "config", None),
            workspace_root=workspace_context["workspace_root"],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    submit_execution = execute_submit_plan(submit_plan)
    final_status = dict(stage_status)
    final_status["last_updated"] = _timestamp()
    final_status["state"] = "submitted" if submit_execution.get("ok") else "submit-failed"
    if submit_execution.get("job_id"):
        final_status["job_id"] = submit_execution["job_id"]
    write_status(run_root_path, final_status)
    assert claim is not None
    claim.release()
    print(
        "\n".join(
            _render_bids_submit_summary(
                context=context,
                run_id=run_id,
                stage_execution=stage_execution,
                submit_execution=submit_execution,
                discovery_report=discovery_report,
            )
        )
    )
    return int(submit_execution.get("returncode", 0))


def _submit_analysis_bids_run(args: argparse.Namespace) -> int:
    _validate_analysis_run_cli_overrides(args)
    project_name = _resolve_project_name(args.project)
    discovery_report: dict[str, Any] | None = None
    if getattr(args, "discover", False):
        discovery_context = _build_analysis_context(
            project_name,
            batch_name=args.batch,
            stage_name=getattr(args, "stage", None),
            model_ref=getattr(args, "model", None),
            allow_missing_batch=True,
            require_default_batch=False,
        )
        discovery_report = _discover_analysis_bids_batch_for_submit(args, context=discovery_context)
    context = _build_analysis_context(
        project_name,
        batch_name=args.batch,
        stage_name=getattr(args, "stage", None),
        model_ref=getattr(args, "model", None),
    )
    run_id = args.run_id if args.run_id is not None else _default_run_id("submit", "analysis", "bids")
    submit_args = argparse.Namespace(**vars(args))
    submit_args.run_id = run_id
    submit_args.dry_run = bool(getattr(args, "dry_run", False))
    workspace_context = _workspace_context()
    execute = bool(getattr(args, "execute", False))
    existing_root = run_path(workspace_context["paths"]["artifacts_root"], run_id)
    if execute and path_entry_exists(existing_root):
        _inspect_shared_planned_run(
            context=workspace_context,
            run_id=run_id,
            expected_slice="bids",
            expected_workflow={"action": "analysis", "target": "bids"},
            expected_project=context["project"]["name"],
            allow_claim=False,
        )
    else:
        _plan_bids_analysis_run(submit_args, context=context, mode="slurm", execute=False, quiet=True)

    manifest, status, run_root_path = _load_run(workspace_context, run_id)
    claim = None
    if execute:
        claim = acquire_execution_claim(workspace_context["paths"]["artifacts_root"], run_id)
        try:
            manifest, status, run_root_path = _inspect_shared_planned_run(
                context=workspace_context,
                run_id=run_id,
                expected_slice="bids",
                expected_workflow={"action": "analysis", "target": "bids"},
                expected_project=context["project"]["name"],
                allow_claim=True,
            )
        except BaseException:
            claim.release()
            raise
    try:
        stage = build_stage_plan(
            workspace_root=workspace_context["workspace_root"],
            run_root=run_root_path,
            manifest=manifest,
            status=status,
            exclude_file=workspace_context["workspace_root"] / "ops" / "sync" / "rsync" / "exclude.txt",
            profile_name=getattr(args, "profile", None),
            role=getattr(args, "role", None),
            config_path=getattr(args, "config", None),
        )
    except ValueError as exc:
        if claim is not None:
            claim.release()
        raise SystemExit(str(exc)) from exc
    if not execute:
        try:
            submit_plan = build_submit_plan(
                manifest=manifest,
                status=stage["status"],
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
                workspace_root=workspace_context["workspace_root"],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        _print_remote_submission_plan(
            run_id=run_id,
            run_root_path=run_root_path,
            dry_run=submit_args.dry_run,
            stage_report=stage["report"],
            submit_plan=submit_plan,
            additional_local_files=(
                [_resolve_reference_path(workspace_context["workspace_root"], discovery_report["batch_path"])]
                if discovery_report is not None
                else None
            ),
        )
        return 0
    stage_execution = execute_stage_plan(stage["report"])
    if not stage_execution.get("ok"):
        next_status = dict(stage["status"])
        next_status["last_updated"] = _timestamp()
        next_status["state"] = "stage-failed"
        write_status(run_root_path, next_status)
        print(json.dumps({"run_id": run_id, "stage": stage["report"], "stage_execution": stage_execution}, indent=2))
        assert claim is not None
        claim.release()
        return int(stage_execution.get("returncode", 1))

    stage_status = dict(stage["status"])
    stage_status["last_updated"] = _timestamp()
    stage_status["state"] = "staged"
    write_status(run_root_path, stage_status)

    try:
        submit_plan = build_submit_plan(
            manifest=manifest,
            status=stage_status,
            profile_name=getattr(args, "profile", None),
            role=getattr(args, "role", None),
            config_path=getattr(args, "config", None),
            workspace_root=workspace_context["workspace_root"],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    submit_execution = execute_submit_plan(submit_plan)
    final_status = dict(stage_status)
    final_status["last_updated"] = _timestamp()
    final_status["state"] = "submitted" if submit_execution.get("ok") else "submit-failed"
    if submit_execution.get("job_id"):
        final_status["job_id"] = submit_execution["job_id"]
    write_status(run_root_path, final_status)
    assert claim is not None
    claim.release()
    print(
        "\n".join(
            _render_analysis_bids_submit_summary(
                context=context,
                run_id=run_id,
                stage_execution=stage_execution,
                submit_execution=submit_execution,
                discovery_report=discovery_report,
            )
        )
    )
    return int(submit_execution.get("returncode", 0))


def _submit_tabular_analysis_run(args: argparse.Namespace) -> int:
    project_name = _resolve_project_name(args.project)
    context = _build_project_hpc_context_for_tabular_analysis(project_name)
    run_id = args.run_id if args.run_id is not None else _default_run_id("submit", "analysis", "tabular")
    submit_args = argparse.Namespace(**vars(args))
    submit_args.run_id = run_id
    submit_args.dry_run = bool(getattr(args, "dry_run", False))
    workspace_context = _workspace_context()
    execute = bool(getattr(args, "execute", False))
    existing_root = run_path(workspace_context["paths"]["artifacts_root"], run_id)
    reuse_reviewed_submission = execute and path_entry_exists(existing_root)
    review_result: dict[str, Any] = {}
    _plan_tabular_analysis_run(
        submit_args,
        context=context,
        mode="slurm",
        execute=False,
        quiet=True,
        allow_existing_review=reuse_reviewed_submission,
        review_result=review_result,
    )

    manifest, status, run_root_path = _load_run(workspace_context, run_id)
    reviewed_stage: dict[str, Any] | None = None
    reviewed_submit_plan: dict[str, Any] | None = None
    if reuse_reviewed_submission:
        reviewed_stage, reviewed_submit_plan = _load_reviewed_submission_plans(
            run_root_path=run_root_path,
            run_id=run_id,
            status=status,
        )
        try:
            requested_submit_plan = build_submit_plan(
                manifest=manifest,
                status=reviewed_stage["status"],
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
                workspace_root=workspace_context["workspace_root"],
            )
        except ValueError as exc:
            raise RunLifecycleError.for_reuse(run_id, "the requested remote connection is no longer valid") from exc
        if requested_submit_plan != reviewed_submit_plan:
            raise RunLifecycleError.for_reuse(run_id, "the requested remote target or submission command has changed")
    claim = None
    if execute:
        claim = acquire_execution_claim(workspace_context["paths"]["artifacts_root"], run_id)
        try:
            manifest, status, run_root_path = _inspect_tabular_reviewed_plan(
                context=workspace_context,
                run_id=run_id,
                requested_identity=review_result["plan_identity"],
                expected_workflow=review_result["workflow"],
                allow_claim=True,
                allow_remote_material=reuse_reviewed_submission,
            )
        except BaseException:
            claim.release()
            raise
    if reuse_reviewed_submission:
        stage, submit_plan = _load_reviewed_submission_plans(
            run_root_path=run_root_path,
            run_id=run_id,
            status=status,
        )
        try:
            requested_submit_plan = build_submit_plan(
                manifest=manifest,
                status=stage["status"],
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
                workspace_root=workspace_context["workspace_root"],
            )
        except ValueError as exc:
            assert claim is not None
            claim.release()
            raise RunLifecycleError.for_reuse(run_id, "the requested remote connection is no longer valid") from exc
        if requested_submit_plan != submit_plan:
            assert claim is not None
            claim.release()
            raise RunLifecycleError.for_reuse(run_id, "the requested remote target or submission command has changed")
    else:
        try:
            stage = build_stage_plan(
                workspace_root=workspace_context["workspace_root"],
                run_root=run_root_path,
                manifest=manifest,
                status=status,
                exclude_file=workspace_context["workspace_root"] / "ops" / "sync" / "rsync" / "exclude.txt",
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
            )
            submit_plan = build_submit_plan(
                manifest=manifest,
                status=stage["status"],
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
                workspace_root=workspace_context["workspace_root"],
            )
        except ValueError as exc:
            if claim is not None:
                claim.release()
            raise SystemExit(str(exc)) from exc
        _persist_reviewed_submission_identity(
            run_root_path=run_root_path,
            manifest=manifest,
            stage_report=stage["report"],
            submit_plan=submit_plan,
        )
    if not execute:
        _print_remote_submission_plan(
            run_id=run_id,
            run_root_path=run_root_path,
            dry_run=submit_args.dry_run,
            stage_report=stage["report"],
            submit_plan=submit_plan,
        )
        return 0
    stage_execution = execute_stage_plan(stage["report"])
    if not stage_execution.get("ok"):
        next_status = dict(stage["status"])
        next_status["last_updated"] = _timestamp()
        next_status["state"] = "stage-failed"
        write_status(run_root_path, next_status)
        print(json.dumps({"run_id": run_id, "stage": stage["report"], "stage_execution": stage_execution}, indent=2))
        assert claim is not None
        claim.release()
        return int(stage_execution.get("returncode", 1))

    stage_status = dict(stage["status"])
    stage_status["last_updated"] = _timestamp()
    stage_status["state"] = "staged"
    write_status(run_root_path, stage_status)
    submit_execution = execute_submit_plan(submit_plan)
    final_status = dict(stage_status)
    final_status["last_updated"] = _timestamp()
    final_status["state"] = "submitted" if submit_execution.get("ok") else "submit-failed"
    if submit_execution.get("job_id"):
        final_status["job_id"] = submit_execution["job_id"]
    write_status(run_root_path, final_status)
    assert claim is not None
    claim.release()
    print(
        "\n".join(
            [
                "Tabular analysis submit summary",
                f"Project: {context['project']['name']}",
                f"Analysis: {args.analysis}",
                f"Run id: {run_id}",
                f"Stage return code: {stage_execution.get('returncode')}",
                f"Submit return code: {submit_execution.get('returncode')}",
            ]
        )
    )
    return int(submit_execution.get("returncode", 0))


def _validate_analysis_run_cli_overrides(args: argparse.Namespace) -> None:
    output_desc = _normalize_optional_cli_value(getattr(args, "output_desc", None))
    if output_desc is not None:
        _validate_bids_label(output_desc, label="--output-desc")


def _plan_bids_run(args: argparse.Namespace, *, context: dict[str, Any], mode: str, execute: bool, quiet: bool = False) -> int:
    run_id = args.run_id if args.run_id is not None else _default_run_id(mode, "preprocess", "bids")
    run_root_path, work_dir, output_dir, log_dir, status_path, manifest_path = _prepare_run_dirs(context, run_id)
    context = _context_with_filtered_bids_batch(context=context, args=args, run_root_path=run_root_path)
    profile_name = context["preprocessing"]["local_profile"] if mode in {"plan", "local"} else context["preprocessing"]["slurm_profile"]
    profile_path = context["pipeline_root"] / "profiles" / profile_name
    snakefile_path = context["pipeline_root"] / "workflow" / "Snakefile"
    pipeline_defaults_path = context["pipeline_root"] / "config" / "defaults.yaml"
    runtime_metadata = _resolve_bids_runtime_metadata(context=context, output_dir=output_dir)
    selection = _resolve_bids_selection(args, context)
    bids_filter_root = output_dir / "bids-filters"
    execution_paths = _resolve_bids_execution_paths(context=context, mode=mode)
    slurm_run_paths = _resolve_slurm_run_paths(context=context, run_id=run_id) if mode == "slurm" else None
    work_dir_value = slurm_run_paths["work_dir"] if slurm_run_paths else to_workspace_relative(work_dir, context["workspace_root"])
    output_dir_value = slurm_run_paths["output_dir"] if slurm_run_paths else to_workspace_relative(output_dir, context["workspace_root"])
    log_dir_value = slurm_run_paths["log_dir"] if slurm_run_paths else to_workspace_relative(log_dir, context["workspace_root"])
    status_path_value = slurm_run_paths["status_path"] if slurm_run_paths else to_workspace_relative(status_path, context["workspace_root"])
    manifest_path_value = slurm_run_paths["manifest_path"] if slurm_run_paths else to_workspace_relative(manifest_path, context["workspace_root"])
    runtime_plan_value = str(PurePosixPath(output_dir_value) / runtime_metadata["runtime_plan_filename"])
    command_script_value = str(PurePosixPath(output_dir_value) / runtime_metadata["command_script_filename"])
    bids_filter_root_value = str(PurePosixPath(output_dir_value) / "bids-filters")
    completion_marker_value = str(PurePosixPath(output_dir_value) / runtime_metadata["completion_marker_filename"])
    resources = _resolve_run_resources(context=context, workload=_run_resource_workload_key("preprocess", "bids"), mode=mode)

    command = [
        "snakemake",
        "--snakefile",
        to_workspace_relative(snakefile_path, context["workspace_root"]),
        "--directory",
        work_dir_value,
        "--configfile",
        to_workspace_relative(pipeline_defaults_path, context["workspace_root"]),
        "--profile",
        to_workspace_relative(profile_path, context["workspace_root"]),
        "--config",
        f"run_manifest={manifest_path_value}",
        f"batch_manifest={to_workspace_relative(Path(context['batch']['path']), context['workspace_root'])}",
        f"dataset_root={execution_paths['dataset_root']}",
        f"input_derivative_root={execution_paths['input_derivative_root']}",
        f"run_output_root={output_dir_value}",
        f"tool={context['preprocessing']['tool']}",
    ]
    command.extend(_snakemake_core_args(resources=resources))
    command.extend(
        _snakemake_resource_args(
            rule_name=runtime_metadata.get("execution_rule_name", runtime_metadata["rule_name"]),
            resources=resources,
            slurm_time=context.get("compute", {}).get("slurm", {}).get("time"),
        )
    )
    if args.dry_run or mode == "plan":
        command.append("--dry-run")
    command.extend(["--", runtime_metadata["workflow_target"]])

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _timestamp(),
        "slice": "bids",
        "workflow": {"action": "preprocess", "target": "bids"},
        "project": {"name": context["project"]["name"], "root": to_workspace_relative(context["project_root"], context["workspace_root"])},
        "pipeline": {
            "name": context["preprocessing"]["pipeline"],
            "root": to_workspace_relative(context["pipeline_root"], context["workspace_root"]),
            "profile": profile_name,
        },
        "batch": _manifest_batch_block(context),
        "dataset": {
            "name": context["dataset"]["primary"],
            "root": execution_paths["dataset_root"],
            "local_root": to_workspace_relative(context["dataset_root"], context["workspace_root"]),
            "derivative_name": context["dataset"].get("input_derivative", ""),
            "derivative_root": execution_paths["input_derivative_root"],
            "local_derivative_root": to_workspace_relative(context["input_derivative_root"], context["workspace_root"]),
        },
        "tool": {
            "name": context["preprocessing"]["tool"],
            "adapter": context["preprocessing"]["tool_adapter"],
            "input_derivative": context["preprocessing"].get("input_derivative", ""),
            "inputs": dict(context["preprocessing"].get("inputs", {})),
            "output": dict(context["preprocessing"].get("output", {})),
            "options": dict(context["preprocessing"].get("tool_options", {})),
            "runtime_profile": {
                "name": context.get("runtime_profile_name", ""),
                "config": context.get("runtime_profile", {}),
            },
            "runtime_metadata": runtime_metadata,
        },
        "resources": resources,
        "outputs": {
            "runtime_plan": runtime_plan_value,
            "command_script": command_script_value,
            "bids_filter_root": bids_filter_root_value,
            "completion_marker": completion_marker_value,
        },
        "execution": _execution_block(
            mode=mode,
            command=command,
            work_dir=work_dir_value,
            output_dir=output_dir_value,
            log_dir=log_dir_value,
            status_path=status_path_value,
            workspace_root=context["workspace_root"],
            dry_run=bool(args.dry_run or mode == "plan"),
        ),
    }
    if any(value is not None for value in selection.values()):
        manifest["selection"] = {key: value for key, value in selection.items() if value is not None}
    return _finalize_run(
        context=context,
        run_root_path=run_root_path,
        manifest=manifest,
        command=command,
        mode=mode,
        execute=execute,
        log_dir=log_dir,
        manifest_path=manifest_path,
        job_name=f"{context['preprocessing']['tool']}-{run_id}",
        hpc_connection=_manifest_hpc_connection_hint(args),
        quiet=quiet,
    )


def _plan_bids_analysis_run(
    args: argparse.Namespace,
    *,
    context: dict[str, Any],
    mode: str,
    execute: bool,
    quiet: bool = False,
) -> int:
    context = _analysis_context_with_run_overrides(
        context=context,
        empty_ev_policy=getattr(args, "empty_ev_policy", None),
        output_desc=getattr(args, "output_desc", None),
    )
    if mode == "slurm":
        _ensure_analysis_slurm_external_roots(context)
    run_id = args.run_id if args.run_id is not None else _default_run_id(mode, "analysis", "bids")
    run_root_path, work_dir, output_dir, log_dir, status_path, manifest_path = _prepare_run_dirs(context, run_id)
    context = _context_with_filtered_bids_batch(context=context, args=args, run_root_path=run_root_path)
    profile_name = context["analysis"]["local_profile"] if mode in {"plan", "local"} else context["analysis"]["slurm_profile"]
    profile_path = context["pipeline_root"] / "profiles" / profile_name
    snakefile_path = context["pipeline_root"] / "workflow" / "Snakefile"
    pipeline_defaults_path = context["pipeline_root"] / "config" / "defaults.yaml"
    runtime_metadata = _resolve_bids_runtime_metadata(context=context, output_dir=output_dir)
    selection = _resolve_analysis_selection(args, context)
    execution_paths = _resolve_bids_execution_paths(context=context, mode=mode)
    slurm_run_paths = _resolve_slurm_run_paths(context=context, run_id=run_id) if mode == "slurm" else None
    work_dir_value = slurm_run_paths["work_dir"] if slurm_run_paths else to_workspace_relative(work_dir, context["workspace_root"])
    output_dir_value = slurm_run_paths["output_dir"] if slurm_run_paths else to_workspace_relative(output_dir, context["workspace_root"])
    log_dir_value = slurm_run_paths["log_dir"] if slurm_run_paths else to_workspace_relative(log_dir, context["workspace_root"])
    status_path_value = slurm_run_paths["status_path"] if slurm_run_paths else to_workspace_relative(status_path, context["workspace_root"])
    manifest_path_value = slurm_run_paths["manifest_path"] if slurm_run_paths else to_workspace_relative(manifest_path, context["workspace_root"])
    runtime_plan_value = str(PurePosixPath(output_dir_value) / runtime_metadata["runtime_plan_filename"])
    command_script_value = str(PurePosixPath(output_dir_value) / runtime_metadata["command_script_filename"])
    completion_marker_value = str(PurePosixPath(output_dir_value) / runtime_metadata["completion_marker_filename"])
    resources = _resolve_run_resources(context=context, workload=_run_resource_workload_key("analysis", "bids"), mode=mode)

    command = [
        "snakemake",
        "--snakefile",
        to_workspace_relative(snakefile_path, context["workspace_root"]),
        "--directory",
        work_dir_value,
        "--configfile",
        to_workspace_relative(pipeline_defaults_path, context["workspace_root"]),
        "--profile",
        to_workspace_relative(profile_path, context["workspace_root"]),
        "--config",
        f"run_manifest={manifest_path_value}",
        f"batch_manifest={to_workspace_relative(Path(context['batch']['path']), context['workspace_root'])}",
        f"dataset_root={execution_paths['dataset_root']}",
        f"input_derivative_root={execution_paths['input_derivative_root']}",
        f"run_output_root={output_dir_value}",
        f"tool={context['analysis_tool_name']}",
    ]
    command.extend(_snakemake_core_args(resources=resources))
    command.extend(
        _snakemake_resource_args(
            rule_name=runtime_metadata.get("execution_rule_name", runtime_metadata["rule_name"]),
            resources=resources,
            slurm_time=context.get("compute", {}).get("slurm", {}).get("time"),
        )
    )
    if args.dry_run or mode == "plan":
        command.append("--dry-run")
    command.extend(["--", runtime_metadata["workflow_target"]])

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _timestamp(),
        "slice": "bids",
        "workflow": {"action": "analysis", "target": "bids"},
        "project": {"name": context["project"]["name"], "root": to_workspace_relative(context["project_root"], context["workspace_root"])},
        "pipeline": {
            "name": context["analysis"]["pipeline"],
            "root": to_workspace_relative(context["pipeline_root"], context["workspace_root"]),
            "profile": profile_name,
        },
        "batch": _manifest_batch_block(context),
        "dataset": {
            "name": context["dataset"]["primary"],
            "root": execution_paths["dataset_root"],
            "local_root": to_workspace_relative(context["dataset_root"], context["workspace_root"]),
            "derivative_name": context["dataset"]["input_derivative"],
            "derivative_root": execution_paths["input_derivative_root"],
            "local_derivative_root": to_workspace_relative(context["input_derivative_root"], context["workspace_root"]),
        },
        "analysis": {
            "stage": context["analysis_stage_name"],
            "tool": context["analysis_tool_name"],
            "model_ref": context["analysis_model_ref"],
            "runtime_profile": context["runtime_profile_name"],
            "inputs": _analysis_manifest_inputs(context=context, mode=mode),
            "input_roots": _analysis_manifest_input_roots(context=context, mode=mode),
            "stage_config": context["analysis_stage"],
            "model": context["analysis_model"],
        },
        "resources": resources,
        "selection": selection,
        "tool": {
            "name": context["analysis_tool_name"],
            "adapter": context["analysis_tool"]["adapter"],
            "runtime_profile": {
                "name": context["runtime_profile_name"],
                "config": context["runtime_profile"],
            },
            "runtime_metadata": runtime_metadata,
        },
        "outputs": {
            "runtime_plan": runtime_plan_value,
            "command_script": command_script_value,
            "completion_marker": completion_marker_value,
        },
        "execution": _execution_block(
            mode=mode,
            command=command,
            work_dir=work_dir_value,
            output_dir=output_dir_value,
            log_dir=log_dir_value,
            status_path=status_path_value,
            workspace_root=context["workspace_root"],
            dry_run=bool(args.dry_run or mode == "plan"),
        ),
    }
    return _finalize_run(
        context=context,
        run_root_path=run_root_path,
        manifest=manifest,
        command=command,
        mode=mode,
        execute=execute,
        log_dir=log_dir,
        manifest_path=manifest_path,
        job_name=f"{context['analysis_tool_name']}-{run_id}",
        hpc_connection=_manifest_hpc_connection_hint(args),
        quiet=quiet,
    )


def _analysis_context_with_run_overrides(
    *,
    context: dict[str, Any],
    empty_ev_policy: str | None,
    output_desc: str | None,
) -> dict[str, Any]:
    policy = _normalize_optional_cli_value(empty_ev_policy)
    desc = _normalize_optional_cli_value(output_desc)
    if policy is None and desc is None:
        return context

    updated_context = dict(context)
    stage_config = dict(context["analysis_stage"])
    if policy is not None:
        validation_config = dict(stage_config.get("validation", {}))
        validation_config["empty_ev_policy"] = policy
        stage_config["validation"] = validation_config
    if desc is not None:
        stage_outputs = dict(stage_config.get("outputs", {}))
        stage_outputs["desc"] = _validate_bids_label(desc, label="--output-desc")
        stage_config["outputs"] = stage_outputs
    updated_context["analysis_stage"] = stage_config
    return updated_context


def _context_with_filtered_bids_batch(
    *,
    context: dict[str, Any],
    args: argparse.Namespace,
    run_root_path: Path,
) -> dict[str, Any]:
    filters = _bids_run_batch_filters(args)
    if not any(filters.values()):
        return context

    selected_rows = _filter_bids_rows(context["batch"].get("rows", []), filters)
    if not selected_rows:
        raise SystemExit(
            json.dumps(
                {
                    "error": "No batch rows matched the requested selectors.",
                    "batch": context["batch"]["name"],
                    "filters": _normalized_bids_filter_manifest(filters),
                },
                indent=2,
            )
        )

    filtered_batch_path = ensure_dir(run_root_path / "inputs") / f"{context['batch']['name']}-filtered.tsv"
    _write_batch_manifest(
        filtered_batch_path,
        selected_rows,
        columns=context["batch"].get("columns", ()),
    )

    updated_context = dict(context)
    original_batch = context["batch"]
    updated_batch = dict(original_batch)
    updated_batch.update(
        {
            "source_path": original_batch["path"],
            "source_row_count": original_batch["row_count"],
            "path": str(filtered_batch_path),
            "row_count": len(selected_rows),
            "rows": selected_rows,
            "selected_row": selected_rows[0] if selected_rows else {},
            "filters": _normalized_bids_filter_manifest(filters),
        }
    )
    updated_context["batch"] = updated_batch
    return updated_context


def _bids_run_batch_filters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "subject_id": getattr(args, "subject_id", None),
        "task_id": getattr(args, "task_id", None),
        "run_id": getattr(args, "row_run_id", None),
    }


def _manifest_batch_block(context: dict[str, Any]) -> dict[str, Any]:
    batch = context["batch"]
    block: dict[str, Any] = {
        "name": batch["name"],
        "path": to_workspace_relative(Path(batch["path"]), context["workspace_root"]),
        "row_count": batch["row_count"],
    }
    if batch.get("source_path"):
        block["source_path"] = to_workspace_relative(Path(batch["source_path"]), context["workspace_root"])
        block["source_row_count"] = batch.get("source_row_count")
    if batch.get("filters"):
        block["filters"] = batch["filters"]
    return block


def _normalized_bids_filter_manifest(filters: dict[str, object]) -> dict[str, list[str]]:
    prefixes = {
        "subject_id": "sub",
        "session_id": "ses",
        "task_id": "task",
        "run_id": "run",
    }
    manifest: dict[str, list[str]] = {}
    for key, values in filters.items():
        normalized = normalize_filter_values(values)
        if not normalized:
            continue
        prefix = prefixes.get(key)
        if prefix is None:
            manifest[key] = list(normalized)
        else:
            manifest[key] = [_normalize_bids_filter_value(value, prefix=prefix) for value in normalized]
    return manifest


def _resolve_bids_runtime_metadata(*, context: dict[str, Any], output_dir: Path) -> dict[str, str]:
    metadata = context["tool_adapter"].runtime_metadata(
        pipeline_defaults=context["pipeline_defaults"],
        output_dir=str(output_dir),
    )
    required = (
        "workflow_target",
        "rule_name",
        "runtime_plan_filename",
        "command_script_filename",
        "completion_marker_filename",
    )
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise ValueError(f"BIDS tool adapter runtime metadata is missing: {', '.join(missing)}")
    return metadata


def _resolve_tabular_predictor_contract(context: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the configuration-owned predictor contract."""

    model_config = context["models"].get("default", {})
    try:
        feature_columns = validate_tabular_feature_columns(model_config.get("feature_columns"))
    except ValueError as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc

    target_value = context["batch"]["selected_row"].get("target_column")
    target_column = target_value if isinstance(target_value, str) and target_value.strip() else ""
    if not target_column:
        raise SystemExit(json.dumps({"error": "The selected batch row must define a nonblank target_column."}, indent=2))
    if target_column in feature_columns:
        raise SystemExit(
            json.dumps(
                {"error": f"models.default.feature_columns must not include the selected target_column {target_column!r}."},
                indent=2,
            )
        )

    table_path = Path(context["feature_table_path"])
    suffix = table_path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".txt"}:
        raise SystemExit(
            json.dumps(
                {"error": f"Unsupported tabular format for predictor validation: {table_path}. Use .csv, .tsv, or .txt."},
                indent=2,
            )
        )
    delimiter = "," if suffix == ".csv" else "\t"
    try:
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"Feature table is missing a header row: {table_path}")
            fieldnames = list(reader.fieldnames)
            if target_column not in fieldnames:
                raise ValueError(f"Selected target_column {target_column!r} is not present in the feature table.")
            unknown = [column for column in feature_columns if column not in fieldnames]
            if unknown:
                raise ValueError(f"Unknown feature columns in the feature table: {', '.join(unknown)}")
            for row_number, row in enumerate(reader, start=1):
                for column in feature_columns:
                    try:
                        float(row.get(column, ""))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Configured feature column {column!r} must be numeric; invalid value at data row {row_number}."
                        ) from exc
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc

    return {
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
    }


def _tabular_input_predictor_contract(*, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    raw_contract = manifest.get("predictor_contract")
    if not isinstance(raw_contract, dict):
        raise SystemExit(
            json.dumps({"error": f"Input run {run_id} is missing its recorded predictor_contract."}, indent=2)
        )
    try:
        feature_columns = validate_tabular_feature_columns(
            raw_contract.get("feature_columns"),
            label="predictor_contract.feature_columns",
        )
    except ValueError as exc:
        raise SystemExit(
            json.dumps({"error": f"Input run {run_id} has an invalid predictor contract: {exc}"}, indent=2)
        ) from exc
    target_value = raw_contract.get("target_column")
    target_column = target_value if isinstance(target_value, str) and target_value.strip() else ""
    if not target_column:
        raise SystemExit(
            json.dumps({"error": f"Input run {run_id} has an invalid predictor contract: target_column is required."}, indent=2)
        )
    if target_column in feature_columns:
        raise SystemExit(
            json.dumps(
                {"error": f"Input run {run_id} has an invalid predictor contract: target_column is also a feature."},
                indent=2,
            )
        )
    feature_count = raw_contract.get("feature_count")
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count != len(feature_columns):
        raise SystemExit(
            json.dumps(
                {"error": f"Input run {run_id} has an invalid predictor contract: feature_count does not match feature_columns."},
                indent=2,
            )
        )
    selected_row = manifest.get("batch", {}).get("selected_row", {})
    if selected_row.get("target_column") != target_column:
        raise SystemExit(
            json.dumps(
                {"error": f"Input run {run_id} has an invalid predictor contract: target_column conflicts with its batch row."},
                indent=2,
            )
        )
    return {
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "feature_count": feature_count,
    }


def _tabular_transaction_plan(
    *,
    run_id: str,
    workflow: dict[str, str],
    outputs: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Return the stable, reviewed local output-transaction contract."""

    return build_tabular_transaction_plan(
        run_id=run_id,
        workflow_action=workflow["action"],
        workflow_target=workflow["target"],
        outputs=tuple(
            TabularOutputSpec(
                logical_name=logical_name,
                relative_path=relative_path,
                content_type=content_type,
            )
            for logical_name, relative_path, content_type in outputs
        ),
    )


def _tabular_planned_output_path(*, mode: str, output_dir: Path | str, filename: str) -> str:
    if mode == "slurm":
        return str(PurePosixPath(output_dir) / filename)
    return f"{_TABULAR_OUTPUT_ROOT_TOKEN}/{filename}"


def _tabular_evaluation_input_block(
    *,
    input_run: dict[str, Any],
    predictor_contract: dict[str, Any],
) -> dict[str, Any]:
    source_split = input_run["consumed"]["split_manifest"]
    source_features = input_run["consumed"]["features_table"]
    source_model = input_run["consumed"]["model"]
    return {
        "run_id": input_run["run_id"],
        "plan_identity": dict(input_run["plan_identity"]),
        "transaction_manifest_sha256": input_run["transaction_manifest_sha256"],
        "outputs": [dict(record) for record in input_run["output_records"]],
        "consumed": {
            "split_manifest": {
                "relative_path": source_split["relative_path"],
                "sha256": source_split["sha256"],
            },
            "features_table": {
                "relative_path": source_features["relative_path"],
                "sha256": source_features["sha256"],
            },
            "model": {
                "relative_path": source_model["relative_path"],
                "sha256": source_model["sha256"],
            },
        },
        "predictor_contract": {
            "target_column": predictor_contract["target_column"],
            "feature_columns": list(predictor_contract["feature_columns"]),
            "feature_count": predictor_contract["feature_count"],
        },
    }


def _plan_tabular_run(args: argparse.Namespace, *, context: dict[str, Any], mode: str, execute: bool, quiet: bool = False) -> int:
    batch_name = str(context["batch"]["name"])
    batch_row_count = int(context["batch"]["row_count"])
    if batch_row_count > 1:
        raise RunLifecycleError(
            f"Selected tabular batch {batch_name!r} contains {batch_row_count} data rows; "
            "public tabular preprocess, train, and evaluate workflows require exactly one "
            "data row and do not perform implicit row selection or Cartesian expansion. "
            "Create or select a named one-row batch with --batch."
        )

    run_id = args.run_id if args.run_id is not None else _default_run_id(mode, args.run_action, args.run_target)
    input_run: dict[str, Any] | None = None
    if args.run_action == "evaluate":
        input_run = (
            _load_tabular_input_run(context, getattr(args, "input_run"))
            if mode in {"plan", "local"}
            else _load_legacy_tabular_input_run(context, getattr(args, "input_run"))
        )
        predictor_contract = _tabular_input_predictor_contract(run_id=input_run["run_id"], manifest=input_run["manifest"])
    else:
        predictor_contract = _resolve_tabular_predictor_contract(context)

    run_root_path, work_dir, output_dir, log_dir, status_path, manifest_path = _run_paths(context, run_id)
    slurm_run_paths = _resolve_slurm_run_paths(context=context, run_id=run_id) if mode == "slurm" else None
    work_dir_value: Path | str = slurm_run_paths["work_dir"] if slurm_run_paths else work_dir
    output_dir_value: Path | str = slurm_run_paths["output_dir"] if slurm_run_paths else output_dir
    log_dir_value: Path | str = slurm_run_paths["log_dir"] if slurm_run_paths else log_dir
    status_path_value: Path | str = slurm_run_paths["status_path"] if slurm_run_paths else status_path
    split_path_value = _tabular_planned_output_path(mode=mode, output_dir=output_dir_value, filename="split.json")
    prep_path_value = _tabular_planned_output_path(mode=mode, output_dir=output_dir_value, filename="prep.json")
    features_path_value = _tabular_planned_output_path(mode=mode, output_dir=output_dir_value, filename="features.tsv")
    model_path_value = _tabular_planned_output_path(mode=mode, output_dir=output_dir_value, filename="model.json")
    evaluation_path_value = _tabular_planned_output_path(mode=mode, output_dir=output_dir_value, filename="evaluation.json")
    final_split_path = output_dir / "split.json"
    final_prep_path = output_dir / "prep.json"
    final_features_path = output_dir / "features.tsv"
    final_model_path = output_dir / "model.json"
    final_evaluation_path = output_dir / "evaluation.json"
    model_config = context["models"].get("default", {})
    resources = _resolve_run_resources(context=context, workload=_run_resource_workload_key(args.run_action, args.run_target), mode=mode)
    commands: list[list[str]]
    outputs: dict[str, str]
    tool: dict[str, str]
    input_run_block: dict[str, Any] | None = None

    if args.run_action == "evaluate":
        assert input_run is not None
        source_target_column = predictor_contract["target_column"]
        if slurm_run_paths:
            source_outputs = input_run["manifest"]["outputs"]
            source_split_value = str(source_outputs["split_manifest"])
            source_features_value = str(source_outputs["features_table"])
            source_model_value = str(source_outputs["model"])
            source_split_path = _resolve_reference_path(context["workspace_root"], source_split_value)
            source_features_path = _resolve_reference_path(context["workspace_root"], source_features_value)
            source_model_path = _resolve_reference_path(context["workspace_root"], source_model_value)
            evaluation_integrity_args: list[str] = []
        else:
            source_split = input_run["consumed"]["split_manifest"]
            source_features = input_run["consumed"]["features_table"]
            source_model = input_run["consumed"]["model"]
            source_split_path = Path(source_split["path"])
            source_features_path = Path(source_features["path"])
            source_model_path = Path(source_model["path"])
            evaluation_integrity_args = [
                "--expected-table-sha256",
                str(source_features["sha256"]),
                "--expected-split-sha256",
                str(source_split["sha256"]),
                "--expected-model-sha256",
                str(source_model["sha256"]),
            ]
        commands = [
            [
                "python3",
                "-m",
                "research_platform.analysis.cli",
                "model",
                "evaluate",
                "--table",
                str(source_features_path),
                "--split",
                str(source_split_path),
                "--target-column",
                source_target_column,
                "--model",
                str(source_model_path),
                *evaluation_integrity_args,
                "--output",
                evaluation_path_value,
            ]
        ]
        outputs = {
            "evaluation": evaluation_path_value if slurm_run_paths else to_workspace_relative(final_evaluation_path, context["workspace_root"]),
        }
        tool = {"model": str(input_run["manifest"]["tool"]["model"])}
        if slurm_run_paths:
            input_run_block = {
                "run_id": input_run["run_id"],
                "model": source_model_value,
                "features_table": source_features_value,
                "split_manifest": source_split_value,
                "predictor_contract": {
                    "target_column": predictor_contract["target_column"],
                    "feature_columns": list(predictor_contract["feature_columns"]),
                    "feature_count": predictor_contract["feature_count"],
                },
            }
            source_plan_identity = input_run["manifest"].get("plan_identity")
            if isinstance(source_plan_identity, dict):
                input_run_block["plan_identity"] = dict(source_plan_identity)
        else:
            input_run_block = _tabular_evaluation_input_block(
                input_run=input_run,
                predictor_contract=predictor_contract,
            )
    else:
        feature_table_value = to_workspace_relative(context["feature_table_path"], context["workspace_root"])
        source_table_sha256 = _file_sha256(Path(context["feature_table_path"]))
        table_integrity_args = [] if slurm_run_paths else ["--expected-table-sha256", source_table_sha256]
        commands = [
            [
                "python3",
                "-m",
                "research_platform.analysis.cli",
                "split",
                "create",
                "--table",
                feature_table_value,
                *table_integrity_args,
                "--target-column",
                context["batch"]["selected_row"]["target_column"],
                "--seed",
                str(context["preprocessing"].get("split_seed", 23)),
                "--test-fraction",
                str(context["preprocessing"].get("test_fraction", 0.25)),
                "--output",
                split_path_value,
            ],
            [
                "python3",
                "-m",
                "research_platform.analysis.cli",
                "prep",
                "fit",
                "--table",
                feature_table_value,
                *table_integrity_args,
                "--split",
                split_path_value,
                "--target-column",
                context["batch"]["selected_row"]["target_column"],
                "--feature-columns",
                *predictor_contract["feature_columns"],
                "--output",
                prep_path_value,
            ],
            [
                "python3",
                "-m",
                "research_platform.analysis.cli",
                "prep",
                "apply",
                "--table",
                feature_table_value,
                *table_integrity_args,
                "--plan",
                prep_path_value,
                "--split",
                split_path_value,
                "--output",
                features_path_value,
            ],
        ]
        outputs = {
            "split_manifest": split_path_value if slurm_run_paths else to_workspace_relative(final_split_path, context["workspace_root"]),
            "prep_plan": prep_path_value if slurm_run_paths else to_workspace_relative(final_prep_path, context["workspace_root"]),
            "features_table": features_path_value if slurm_run_paths else to_workspace_relative(final_features_path, context["workspace_root"]),
        }
        tool = {"preprocessing": "standardize_numeric"}
        if args.run_action == "train":
            commands.append(
                [
                    "python3",
                    "-m",
                    "research_platform.analysis.cli",
                    "model",
                    "train",
                    "--table",
                    features_path_value,
                    "--split",
                    split_path_value,
                    "--target-column",
                    context["batch"]["selected_row"]["target_column"],
                    "--feature-columns",
                    *predictor_contract["feature_columns"],
                    "--kind",
                    model_config.get("kind", "logistic_regression"),
                    "--learning-rate",
                    str(model_config.get("learning_rate", 0.2)),
                    "--iterations",
                    str(model_config.get("iterations", 350)),
                    *([] if slurm_run_paths else ["--table-reference", "outputs/features.tsv"]),
                    "--output",
                    model_path_value,
                ]
            )
            outputs["model"] = model_path_value if slurm_run_paths else to_workspace_relative(final_model_path, context["workspace_root"])
            tool["model"] = model_config.get("kind", "logistic_regression")

    execute_script = _render_run_shell_script(
        commands=commands,
        workspace_root_ref=_resolve_execution_workspace_root(context=context, mode=mode),
    )
    shell_script_path = run_root_path / "execute.sh"
    command = (
        ["bash", str(PurePosixPath(slurm_run_paths["run_root"]) / "execute.sh")]
        if slurm_run_paths
        else ["bash", to_workspace_relative(shell_script_path, context["workspace_root"])]
    )

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _timestamp(),
        "slice": "tabular",
        "workflow": {"action": args.run_action, "target": args.run_target},
        "project": {"name": context["project"]["name"], "root": to_workspace_relative(context["project_root"], context["workspace_root"])},
        "batch": {
            "name": context["batch"]["name"],
            "path": to_workspace_relative(Path(context["batch"]["path"]), context["workspace_root"]),
            "row_count": context["batch"]["row_count"],
            "selected_row": context["batch"]["selected_row"],
        },
        "dataset": {
            "name": context["dataset"]["primary"],
            "root": to_workspace_relative(context["dataset_root"], context["workspace_root"]),
            "canonical_dataset": context["dataset"]["canonical_dataset"],
            "canonical_features_root": to_workspace_relative(context["canonical_features_root"], context["workspace_root"]),
            "feature_table": to_workspace_relative(context["feature_table_path"], context["workspace_root"]),
        },
        "resources": resources,
        "tool": tool,
        "source_digests": {
            "batch_sha256": _file_sha256(Path(context["batch"]["path"])),
            "input_table_sha256": _file_sha256(Path(context["feature_table_path"])),
        },
        "settings": {
            "preprocessing": {
                "split_seed": context["preprocessing"].get("split_seed", 23),
                "test_fraction": context["preprocessing"].get("test_fraction", 0.25),
            },
            "model": dict(model_config) if args.run_action == "train" else {},
        },
        "plan": {"commands": [list(item) for item in commands]},
        "predictor_contract": {
            "target_column": predictor_contract["target_column"],
            "feature_columns": list(predictor_contract["feature_columns"]),
            "feature_count": predictor_contract["feature_count"],
        },
        "outputs": outputs,
        "execution": _execution_block(
            mode=mode,
            command=command,
            work_dir=work_dir_value,
            output_dir=output_dir_value,
            log_dir=log_dir_value,
            status_path=status_path_value,
            workspace_root=context["workspace_root"],
            dry_run=bool(args.dry_run or mode == "plan"),
        ),
    }
    if mode in {"plan", "local"}:
        if args.run_action == "evaluate":
            transaction_outputs = [("evaluation", "evaluation.json", "json")]
        else:
            transaction_outputs = [
                ("split_manifest", "split.json", "json"),
                ("prep_plan", "prep.json", "json"),
                ("features_table", "features.tsv", "tsv"),
            ]
            if args.run_action == "train":
                transaction_outputs.append(("model", "model.json", "json"))
        manifest["output_transaction"] = _tabular_transaction_plan(
            run_id=run_id,
            workflow=dict(manifest["workflow"]),
            outputs=transaction_outputs,
        )
    if input_run_block is not None:
        manifest["input_run"] = input_run_block
    manifest, slurm_script = _complete_run_manifest(
        context=context,
        run_root_path=run_root_path,
        manifest=manifest,
        command=command,
        mode=mode,
        job_name=f"{args.run_action}-{args.run_target}-{run_id}",
        hpc_connection=_manifest_hpc_connection_hint(args),
    )
    return _finalize_tabular_run(
        context=context,
        run_root_path=run_root_path,
        manifest=manifest,
        execute_script=execute_script,
        slurm_script=slurm_script,
        command=command,
        mode=mode,
        execute=execute,
        quiet=quiet,
    )


def _build_project_hpc_context_for_tabular_analysis(project_name: str) -> dict[str, Any]:
    context = build_project_hpc_context(project_name)
    context.setdefault("compute", {"local": {"jobs": 1}, "slurm": {}})
    return context


def _plan_tabular_analysis_run(
    args: argparse.Namespace,
    *,
    context: dict[str, Any],
    mode: str,
    execute: bool,
    quiet: bool = False,
    allow_existing_review: bool = False,
    review_result: dict[str, Any] | None = None,
) -> int:
    analysis_name = _validate_simple_scaffold_name(args.analysis, label="--analysis")
    analysis_spec = _load_tabular_analysis_spec(context=context, analysis_name=analysis_name)
    context = dict(context)
    context["slice"] = "tabular"
    context["feature_table_path"] = analysis_spec["input_table_path"]
    run_id = args.run_id if args.run_id is not None else _default_run_id(mode, "analysis", "tabular")
    run_root_path, work_dir, output_dir, log_dir, status_path, manifest_path = _run_paths(context, run_id)
    slurm_run_paths = _resolve_slurm_run_paths(context=context, run_id=run_id) if mode == "slurm" else None
    work_dir_value: Path | str = slurm_run_paths["work_dir"] if slurm_run_paths else work_dir
    output_dir_value: Path | str = slurm_run_paths["output_dir"] if slurm_run_paths else output_dir
    log_dir_value: Path | str = slurm_run_paths["log_dir"] if slurm_run_paths else log_dir
    status_path_value: Path | str = slurm_run_paths["status_path"] if slurm_run_paths else status_path
    report_filename = f"{analysis_name}.json"
    report_path_value = _tabular_planned_output_path(
        mode=mode,
        output_dir=output_dir_value,
        filename=report_filename,
    )
    source_table_sha256 = _file_sha256(analysis_spec["input_table_path"])
    stats_command = _tabular_analysis_command(analysis_spec=analysis_spec, output_path=report_path_value)
    if not slurm_run_paths:
        stats_command.extend(["--expected-table-sha256", source_table_sha256])
    execute_script = _render_run_shell_script(
        commands=[stats_command],
        workspace_root_ref=_resolve_execution_workspace_root(context=context, mode=mode),
    )
    shell_script_path = run_root_path / "execute.sh"
    command = (
        ["bash", str(PurePosixPath(slurm_run_paths["run_root"]) / "execute.sh")]
        if slurm_run_paths
        else ["bash", to_workspace_relative(shell_script_path, context["workspace_root"])]
    )
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _timestamp(),
        "slice": "tabular",
        "workflow": {"action": "analysis", "target": "tabular"},
        "project": {"name": context["project"]["name"], "root": to_workspace_relative(context["project_root"], context["workspace_root"])},
        "analysis": {
            "name": analysis_name,
            "path": to_workspace_relative(analysis_spec["path"], context["workspace_root"]),
            "kind": analysis_spec["analysis"]["kind"],
        },
        "batch": {"name": args.batch or "", "path": "", "row_count": 0, "selected_row": {}},
        "resources": _resolve_run_resources(context=context, workload=_run_resource_workload_key("analysis", "tabular"), mode=mode),
        "tool": {"analysis": analysis_spec["analysis"]["kind"]},
        "source_digests": {"input_table_sha256": source_table_sha256},
        "plan": {"commands": [list(stats_command)]},
        "outputs": {
            "analysis_report": report_path_value if slurm_run_paths else to_workspace_relative(output_dir / report_filename, context["workspace_root"]),
        },
        "execution": _execution_block(
            mode=mode,
            command=command,
            work_dir=work_dir_value,
            output_dir=output_dir_value,
            log_dir=log_dir_value,
            status_path=status_path_value,
            workspace_root=context["workspace_root"],
            dry_run=bool(args.dry_run or mode == "plan"),
        ),
    }
    manifest["analysis"]["config"] = dict(analysis_spec["analysis"])
    if mode in {"plan", "local"}:
        manifest["output_transaction"] = _tabular_transaction_plan(
            run_id=run_id,
            workflow=dict(manifest["workflow"]),
            outputs=[("analysis_report", report_filename, "json")],
        )
    manifest, slurm_script = _complete_run_manifest(
        context=context,
        run_root_path=run_root_path,
        manifest=manifest,
        command=command,
        mode=mode,
        job_name=f"analysis-tabular-{run_id}",
        hpc_connection=_manifest_hpc_connection_hint(args),
    )
    return _finalize_tabular_run(
        context=context,
        run_root_path=run_root_path,
        manifest=manifest,
        execute_script=execute_script,
        slurm_script=slurm_script,
        command=command,
        mode=mode,
        execute=execute,
        quiet=quiet,
        allow_existing_review=allow_existing_review,
        review_result=review_result,
    )


def _load_tabular_analysis_spec(*, context: dict[str, Any], analysis_name: str) -> dict[str, Any]:
    spec_path = context["project_root"] / "config" / "analysis" / f"{analysis_name}.yaml"
    if not spec_path.exists():
        raise SystemExit(json.dumps({"error": f"Analysis spec was not found: {spec_path}"}, indent=2))
    document = load_yaml(spec_path)
    analysis = document.get("analysis") if isinstance(document, dict) else None
    if not isinstance(analysis, dict):
        raise SystemExit(json.dumps({"error": f"Analysis spec must define analysis mapping: {spec_path}"}, indent=2))
    kind = _normalize_optional_cli_value(str(analysis.get("kind", "")))
    allowed = {"anova", "correlation", "linear_model", "mixed_effects", "summary_table"}
    if kind not in allowed:
        raise SystemExit(json.dumps({"error": f"analysis.kind must be one of: {', '.join(sorted(allowed))}"}, indent=2))
    input_table = _normalize_optional_cli_value(str(analysis.get("input_table", "")))
    if input_table is None:
        raise SystemExit(json.dumps({"error": "analysis.input_table is required."}, indent=2))
    table_path = _resolve_reference_path(context["workspace_root"], input_table)
    if not table_path.exists():
        raise SystemExit(json.dumps({"error": f"analysis.input_table was not found: {table_path}"}, indent=2))
    analysis = dict(analysis)
    analysis["kind"] = kind
    analysis["input_table"] = to_workspace_relative(table_path, context["workspace_root"])
    return {"path": spec_path, "analysis": analysis, "input_table_path": table_path}


def _tabular_analysis_command(*, analysis_spec: dict[str, Any], output_path: str) -> list[str]:
    analysis = analysis_spec["analysis"]
    kind = analysis["kind"]
    command = ["python3", "-m", "research_platform.analysis.cli", "stats", kind, "--table", str(analysis["input_table"]), "--output", output_path]
    if kind == "correlation":
        command.extend(["--x", _required_analysis_value(analysis, "x"), "--y", _required_analysis_value(analysis, "y")])
        command.extend(["--method", str(analysis.get("method") or "pearson")])
    elif kind == "summary_table":
        for column in _analysis_list_value(analysis.get("columns")):
            command.extend(["--column", column])
    elif kind == "linear_model":
        command.extend(["--outcome", _required_analysis_value(analysis, "outcome")])
        for predictor in _analysis_list_value(analysis.get("predictors")):
            command.extend(["--predictor", predictor])
    elif kind == "anova":
        command.extend(["--outcome", _required_analysis_value(analysis, "outcome")])
        command.extend(["--group", _required_analysis_value(analysis, "group", fallback_key="predictors")])
    elif kind == "mixed_effects":
        command.extend(["--outcome", _required_analysis_value(analysis, "outcome")])
        for predictor in _analysis_list_value(analysis.get("predictors")):
            command.extend(["--predictor", predictor])
        group = _normalize_optional_cli_value(str(analysis.get("group", "")))
        if group:
            command.extend(["--group", group])
    return command


def _required_analysis_value(analysis: dict[str, Any], key: str, *, fallback_key: str | None = None) -> str:
    value = analysis.get(key)
    if value is None and fallback_key is not None:
        values = _analysis_list_value(analysis.get(fallback_key))
        value = values[0] if values else None
    text = _normalize_optional_cli_value(str(value or ""))
    if text is None:
        raise SystemExit(json.dumps({"error": f"analysis.{key} is required."}, indent=2))
    return text


def _analysis_list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _normalize_optional_cli_value(str(value or ""))
    if text is None:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _run_paths(context: dict[str, Any], run_id: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    validate_run_id(run_id)
    run_root_path = run_path(context["paths"]["artifacts_root"], run_id)
    work_dir = run_root_path / "work"
    output_dir = run_root_path / "outputs"
    log_dir = run_root_path / "logs"
    return run_root_path, work_dir, output_dir, log_dir, run_root_path / "status.yaml", run_root_path / "run-manifest.yaml"


def _prepare_run_dirs(context: dict[str, Any], run_id: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = _run_paths(context, run_id)
    run_root_path, work_dir, output_dir, log_dir, _, _ = paths
    runs_root = run_root_path.parent
    ensure_dir(runs_root)
    if path_entry_exists(claim_path(context["paths"]["artifacts_root"], run_id)):
        raise RunLifecycleError.for_reuse(run_id, "an execution claim already exists")
    try:
        run_root_path.mkdir()
    except FileExistsError as exc:
        raise RunLifecycleError.for_reuse(run_id, "the run root already exists") from exc
    for directory in (work_dir, output_dir, log_dir):
        directory.mkdir()
    return paths


def _execution_block(
    *,
    mode: str,
    command: list[str],
    work_dir: Path | str,
    output_dir: Path | str,
    log_dir: Path | str,
    status_path: Path | str,
    workspace_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    work_dir_value = str(work_dir) if isinstance(work_dir, str) else to_workspace_relative(work_dir, workspace_root)
    output_dir_value = str(output_dir) if isinstance(output_dir, str) else to_workspace_relative(output_dir, workspace_root)
    log_dir_value = str(log_dir) if isinstance(log_dir, str) else to_workspace_relative(log_dir, workspace_root)
    status_path_value = str(status_path) if isinstance(status_path, str) else to_workspace_relative(status_path, workspace_root)
    return {
        "mode": mode,
        "dry_run": dry_run,
        "cwd": ".",
        "command": command,
        "work_dir": work_dir_value,
        "output_dir": output_dir_value,
        "log_dir": log_dir_value,
        "status_path": status_path_value,
    }


def _infer_slurm_required_executables(command: list[str]) -> list[str]:
    if not command:
        return []
    executable = str(command[0]).strip()
    if not executable or "/" in executable:
        return []
    return [executable]


def _resolve_slurm_run_paths(*, context: dict[str, Any], run_id: str) -> dict[str, str]:
    remote_run_root = resolve_remote_run_root(
        run_id=run_id,
        remote_workspace_root=context["compute"]["slurm"].get("remote_workspace_root"),
        remote_artifacts_root=context["compute"]["slurm"].get("remote_artifacts_root"),
    )
    if not remote_run_root:
        raise SystemExit(json.dumps({"error": "SLURM planning requires a remote run root."}, indent=2))
    return {
        "run_root": remote_run_root,
        "work_dir": str(PurePosixPath(remote_run_root) / "work"),
        "output_dir": str(PurePosixPath(remote_run_root) / "outputs"),
        "log_dir": str(PurePosixPath(remote_run_root) / "logs"),
        "status_path": str(PurePosixPath(remote_run_root) / "status.yaml"),
        "manifest_path": str(PurePosixPath(remote_run_root) / "run-manifest.yaml"),
    }


def _resolve_bids_execution_paths(*, context: dict[str, Any], mode: str) -> dict[str, str]:
    local_dataset_root = to_workspace_relative(context["dataset_root"], context["workspace_root"])
    local_input_derivative_root = to_workspace_relative(context["input_derivative_root"], context["workspace_root"])
    if mode != "slurm":
        return {
            "dataset_root": local_dataset_root,
            "input_derivative_root": local_input_derivative_root,
        }

    remote_dataset_root = context.get("remote_dataset_root")
    remote_input_derivative_root = context.get("remote_input_derivative_root")
    if str(local_dataset_root).startswith("/") and not remote_dataset_root:
        raise SystemExit(
            json.dumps(
                {"error": "BIDS SLURM planning requires dataset.remote_bids_root when the local BIDS root is outside the workspace."},
                indent=2,
            )
        )
    if str(local_input_derivative_root).startswith("/") and not remote_input_derivative_root:
        raise SystemExit(
            json.dumps(
                {
                    "error": (
                        "BIDS SLURM planning requires dataset.remote_input_derivative_root "
                        "when the local input derivative root is outside the workspace."
                    )
                },
                indent=2,
            )
        )
    return {
        "dataset_root": remote_dataset_root or local_dataset_root,
        "input_derivative_root": remote_input_derivative_root or local_input_derivative_root,
    }


def _render_run_shell_script(
    *,
    commands: list[list[str]],
    workspace_root_ref: str | Path,
) -> bytes:
    package_srcs = [
        "packages/research-analysis/src",
        "packages/research-ml/src",
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(workspace_root_ref))}",
        f'export PYTHONPATH="{":".join(package_srcs)}:${{PYTHONPATH:-}}"',
        "",
    ]
    lines.extend(" ".join(_quote_reviewed_command_part(part) for part in command) for command in commands)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _quote_reviewed_command_part(part: str) -> str:
    """Quote a reviewed command argument while retaining the logical output root."""

    prefix = f"{_TABULAR_OUTPUT_ROOT_TOKEN}/"
    if part.startswith(prefix):
        relative = part.removeprefix(prefix)
        candidate = PurePosixPath(relative)
        if not relative or candidate.is_absolute() or any(value in {"", ".", ".."} for value in candidate.parts):
            raise RunLifecycleError("Tabular output arguments must remain beneath the logical output root.")
        return f'"{_TABULAR_OUTPUT_ROOT_TOKEN}/{candidate.as_posix()}"'
    return shlex.quote(part)


def _write_run_shell_script(
    *,
    run_root_path: Path,
    script_bytes: bytes,
) -> Path:
    script_path = run_root_path / "execute.sh"
    script_path.write_bytes(script_bytes)
    script_path.chmod(0o755)
    return script_path


def _resolve_execution_workspace_root(*, context: dict[str, Any], mode: str) -> str | Path:
    if mode != "slurm":
        return context["workspace_root"]
    return str(context["compute"]["slurm"].get("remote_workspace_root", ""))


def _resolve_slurm_environment(slurm_config: dict[str, Any]) -> dict[str, str]:
    environment = slurm_config.get("environment")
    if not isinstance(environment, dict):
        return {}
    return {
        str(key).strip(): str(value).strip()
        for key, value in environment.items()
        if str(key).strip() and isinstance(value, str) and str(value).strip()
    }


def _resolve_slurm_prepare_directories(slurm_config: dict[str, Any]) -> list[str]:
    return normalize_slurm_prepare_directories(slurm_config.get("prepare_directories"))


def _complete_run_manifest(
    *,
    context: dict[str, Any],
    run_root_path: Path,
    manifest: dict[str, Any],
    command: list[str],
    mode: str,
    job_name: str,
    hpc_connection: dict[str, str] | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    completed_manifest = dict(manifest)
    slurm_script_bytes: bytes | None = None
    if mode == "slurm":
        remote_run_root = _resolve_slurm_run_paths(context=context, run_id=completed_manifest["run_id"])["run_root"]
        slurm_config = context["compute"]["slurm"]
        setup_commands = _build_repo_slurm_setup_commands(
            remote_workspace_root=str(slurm_config.get("remote_workspace_root", "")),
            workspace_root=Path(context["workspace_root"]),
            modules=_normalize_notebook_modules(slurm_config.get("modules")),
            environment=_resolve_slurm_environment(slurm_config),
            pre_activate_commands=_normalize_notebook_shell_commands(slurm_config.get("pre_activate_commands")),
            prepare_directories=_resolve_slurm_prepare_directories(slurm_config),
        )
        jobspec = build_slurm_jobspec(
            resources=completed_manifest.get("resources", {}),
            job_name=job_name,
            time=str(slurm_config.get("time", "02:00:00")),
            log_out=str(PurePosixPath(completed_manifest["execution"]["log_dir"]) / "slurm.out"),
            log_err=str(PurePosixPath(completed_manifest["execution"]["log_dir"]) / "slurm.err"),
            command=build_slurm_command_script(
                setup_commands=setup_commands,
                required_executables=_infer_slurm_required_executables(command),
                workflow_command=" ".join(shlex.quote(part) for part in command),
            ),
            slurm_site=resolve_hpc_slurm_site_settings(slurm_config),
        )
        script_path = run_root_path / "submit.sbatch"
        render = render_slurm_script(template_path=context["paths"]["ops_root"] / "slurm" / "job_templates" / "sbatch.job.sh", jobspec=jobspec)
        slurm_script_bytes = normalize_slurm_batch_script(render).encode("utf-8")
        completed_manifest["slurm"] = {"script_path": to_workspace_relative(script_path, context["workspace_root"]), "jobspec": jobspec, "job_id": ""}
        completed_manifest["hpc"] = {
            "target": (context.get("hpc_target") or {}).get("name", ""),
            "ssh_host": slurm_config.get("ssh_host", ""),
            "remote_workspace_root": slurm_config.get("remote_workspace_root", ""),
            "remote_artifacts_root": slurm_config.get("remote_artifacts_root", ""),
            "remote_run_root": remote_run_root,
            "site": {"slurm": resolve_hpc_slurm_site_settings(slurm_config)},
        }
        if hpc_connection is not None:
            completed_manifest["hpc"]["connection"] = dict(hpc_connection)
        completed_manifest["provision"] = build_provision_plan(context=context, manifest=completed_manifest)
        bootstrap = build_bootstrap_manifest(context=context, manifest=completed_manifest)
        if bootstrap is not None:
            completed_manifest["bootstrap"] = bootstrap

    publish_back_payload = dict(context.get("preprocessing", {}).get("publish_back", {}))
    if not publish_back_payload:
        publish_back_payload = dict(context.get("analysis", {}).get("publish_back", {}))
    if completed_manifest.get("slice") == "bids":
        adapter_scaffold = context["tool_adapter"].build_publish_back_scaffold(
            manifest=completed_manifest,
            run_root=str(run_root_path),
            workspace_root=str(context["workspace_root"]),
        )
        if adapter_scaffold.get("default_policy") and not publish_back_payload.get("default_policy"):
            publish_back_payload["default_policy"] = adapter_scaffold["default_policy"]
        if adapter_scaffold.get("items"):
            publish_back_payload["items"] = list(adapter_scaffold["items"])

    completed_manifest["publish_back"] = build_publish_back_scaffold(
        workspace_root=context["workspace_root"],
        run_root=run_root_path,
        publish_back=publish_back_payload,
    )
    return completed_manifest, slurm_script_bytes


def _read_run_control_mapping(path: Path, *, run_id: str, label: str) -> dict[str, Any]:
    try:
        value = load_yaml(path, resolve_env=False)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RunLifecycleError.for_reuse(run_id, f"{label} is missing or malformed") from exc
    if not isinstance(value, dict):
        raise RunLifecycleError.for_reuse(run_id, f"{label} is not a mapping")
    return value


def _require_real_directory(path: Path, *, run_id: str, label: str) -> None:
    import stat

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise RunLifecycleError.for_reuse(run_id, f"{label} is missing") from exc
    if not stat.S_ISDIR(mode) or path.is_symlink():
        raise RunLifecycleError.for_reuse(run_id, f"{label} is not a real directory")


def _real_directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    import stat

    try:
        identity = os.lstat(path)
    except OSError as exc:
        raise RunLifecycleError(f"{label} is missing or unreadable: {path}") from exc
    if not stat.S_ISDIR(identity.st_mode) or stat.S_ISLNK(identity.st_mode):
        raise RunLifecycleError(f"{label} is not a real directory: {path}")
    return identity.st_dev, identity.st_ino


def _require_unchanged_real_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    if _real_directory_identity(path, label=label) != expected_identity:
        raise RunLifecycleError(f"{label} changed filesystem identity during execution.")


def _require_real_file(path: Path, *, run_id: str, label: str) -> bytes:
    try:
        return _stable_regular_file_bytes(path, label=label)
    except (FileNotFoundError, OSError, RunLifecycleError) as exc:
        raise RunLifecycleError.for_reuse(run_id, f"{label} is missing, unsafe, or unreadable") from exc


def _inspect_tabular_reviewed_plan(
    *,
    context: dict[str, Any],
    run_id: str,
    requested_identity: dict[str, Any],
    expected_workflow: dict[str, Any],
    allow_claim: bool,
    allow_remote_material: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_root_path, work_dir, output_dir, log_dir, _, _ = _run_paths(context, run_id)
    _require_real_directory(run_root_path, run_id=run_id, label="run root")
    active_claim_path = claim_path(context["paths"]["artifacts_root"], run_id)
    if not allow_claim and path_entry_exists(active_claim_path):
        raise RunLifecycleError.for_reuse(run_id, "an execution claim already exists")

    manifest_path = run_root_path / "run-manifest.yaml"
    status_path = run_root_path / "status.yaml"
    execute_path = run_root_path / "execute.sh"
    manifest_bytes = _require_real_file(manifest_path, run_id=run_id, label="run manifest")
    status_bytes = _require_real_file(status_path, run_id=run_id, label="run status")
    execute_bytes = _require_real_file(execute_path, run_id=run_id, label="reviewed execution script")
    try:
        manifest = parse_yaml(manifest_bytes.decode("utf-8"), resolve_env=False)
        status = parse_yaml(status_bytes.decode("utf-8"), resolve_env=False)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RunLifecycleError.for_reuse(run_id, "run control files are malformed") from exc
    if not isinstance(manifest, dict) or not isinstance(status, dict):
        raise RunLifecycleError.for_reuse(run_id, "run control files must be mappings")
    if manifest.get("run_id") != run_id or status.get("run_id") != run_id:
        raise RunLifecycleError.for_reuse(run_id, "run control files identify a different run")
    required_status_keys = {"run_id", "state", "last_updated", "job_id", "mode"}
    if not required_status_keys.issubset(status):
        raise RunLifecycleError.for_reuse(run_id, "the run status is incomplete")
    if not all(isinstance(status.get(key), str) for key in ("state", "last_updated", "job_id", "mode")):
        raise RunLifecycleError.for_reuse(run_id, "the run status contains invalid metadata")
    if manifest.get("slice") != "tabular" or manifest.get("workflow") != expected_workflow:
        raise RunLifecycleError.for_reuse(run_id, "the run is owned by a different workflow")
    if status.get("state") != "planned":
        raise RunLifecycleError.for_reuse(run_id, f"status is {status.get('state')!r}, not 'planned'")

    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise RunLifecycleError.for_reuse(run_id, "the run manifest has no complete execution mapping")
    if status.get("mode") != execution.get("mode"):
        raise RunLifecycleError.for_reuse(run_id, "the run status mode conflicts with the reviewed manifest")
    slurm_bytes: bytes | None = None
    expected_names = {"execute.sh", "run-manifest.yaml", "status.yaml", "work", "logs"}
    if execution.get("mode") == "slurm":
        expected_names.add("outputs")
        slurm_bytes = _require_real_file(run_root_path / "submit.sbatch", run_id=run_id, label="reviewed SLURM script")
        expected_names.add("submit.sbatch")
        if allow_remote_material:
            _require_real_directory(
                run_root_path / "hpc",
                run_id=run_id,
                label="reviewed HPC plan directory",
            )
            _validate_reviewed_stage_material(run_root_path=run_root_path, run_id=run_id)
            expected_names.add("hpc")
    actual_names = {entry.name for entry in run_root_path.iterdir()}
    if actual_names != expected_names:
        raise RunLifecycleError.for_reuse(run_id, "the run root contains unexpected execution residue")
    reviewed_directories = [(work_dir, "work directory"), (log_dir, "log directory")]
    if execution.get("mode") == "slurm":
        reviewed_directories.append((output_dir, "output directory"))
    else:
        transaction_plan = manifest.get("output_transaction")
        if not isinstance(transaction_plan, dict):
            raise RunLifecycleError.for_reuse(run_id, "the local output transaction plan is missing")
        try:
            tabular_output_specs_from_plan(transaction_plan)
            preflight_tabular_transaction_root(run_root_path)
        except TabularOutputTransactionError as exc:
            raise RunLifecycleError.for_reuse(run_id, str(exc)) from exc
    for directory, label in reviewed_directories:
        _require_real_directory(directory, run_id=run_id, label=label)
        if any(directory.iterdir()):
            raise RunLifecycleError.for_reuse(run_id, f"the {label} is not empty")

    try:
        verify_plan_identity(
            manifest,
            execute_script=execute_bytes,
            slurm_script=slurm_bytes,
        )
    except (TypeError, ValueError) as exc:
        raise RunLifecycleError.for_reuse(run_id, "the persisted plan identity or reviewed script is invalid") from exc
    if manifest.get("plan_identity") != requested_identity:
        raise RunLifecycleError.for_reuse(run_id, "the requested configuration or reviewed command plan has changed")
    if allow_remote_material:
        _verify_reviewed_submission_identity(
            run_root_path=run_root_path,
            run_id=run_id,
            manifest=manifest,
            execute_script=execute_bytes,
            slurm_script=slurm_bytes,
        )
    return manifest, status, run_root_path


def _validate_reviewed_stage_material(*, run_root_path: Path, run_id: str) -> None:
    import stat

    hpc_dir = run_root_path / "hpc"
    stage_dir = hpc_dir / "stage"
    stage_plan_path = hpc_dir / "stage-plan.yaml"
    submit_plan_path = hpc_dir / "submit-plan.json"
    _require_real_directory(hpc_dir, run_id=run_id, label="HPC plan directory")
    _require_real_directory(stage_dir, run_id=run_id, label="HPC stage directory")
    _require_real_file(stage_plan_path, run_id=run_id, label="HPC stage plan")
    plan = _read_run_control_mapping(stage_plan_path, run_id=run_id, label="HPC stage plan")
    if plan.get("run_id") != run_id or not isinstance(plan.get("staged_files"), list):
        raise RunLifecycleError.for_reuse(run_id, "the HPC stage plan is incomplete")
    expected_files = {stage_plan_path.resolve()}
    if path_entry_exists(submit_plan_path):
        _require_real_file(submit_plan_path, run_id=run_id, label="HPC submit plan")
        expected_files.add(submit_plan_path.resolve())
    for value in plan["staged_files"]:
        if not isinstance(value, str):
            raise RunLifecycleError.for_reuse(run_id, "the HPC stage plan contains an invalid file reference")
        candidate = Path(value).resolve()
        try:
            candidate.relative_to(stage_dir.resolve())
        except ValueError as exc:
            raise RunLifecycleError.for_reuse(run_id, "the HPC stage plan escapes its stage directory") from exc
        expected_files.add(candidate)
    actual_files: set[Path] = set()
    for entry in hpc_dir.rglob("*"):
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise RunLifecycleError.for_reuse(run_id, "the HPC plan contains an unsafe filesystem entry")
        if stat.S_ISREG(mode):
            actual_files.add(entry.resolve())
    if actual_files != expected_files:
        raise RunLifecycleError.for_reuse(run_id, "the HPC plan contains unexpected or missing files")


def _reviewed_submission_extra_files(*, run_root_path: Path, run_id: str) -> dict[str, bytes]:
    _validate_reviewed_stage_material(run_root_path=run_root_path, run_id=run_id)
    stage_plan_path = run_root_path / "hpc" / "stage-plan.yaml"
    submit_plan_path = run_root_path / "hpc" / "submit-plan.json"
    stage_plan = _read_run_control_mapping(stage_plan_path, run_id=run_id, label="HPC stage plan")
    extra_files = {
        "hpc/stage-plan.yaml": _require_real_file(stage_plan_path, run_id=run_id, label="HPC stage plan"),
        "hpc/submit-plan.json": _require_real_file(submit_plan_path, run_id=run_id, label="HPC submit plan"),
    }
    stage_dir = (run_root_path / "hpc" / "stage").resolve()
    for raw_path in stage_plan["staged_files"]:
        candidate = Path(raw_path).resolve()
        try:
            relative = candidate.relative_to(run_root_path.resolve())
            candidate.relative_to(stage_dir)
        except ValueError as exc:
            raise RunLifecycleError.for_reuse(run_id, "the HPC stage plan escapes the run root") from exc
        label = relative.as_posix()
        extra_files[label] = _require_real_file(candidate, run_id=run_id, label=f"reviewed staged file {label}")
    return extra_files


def _persist_reviewed_submission_identity(
    *,
    run_root_path: Path,
    manifest: dict[str, Any],
    stage_report: dict[str, Any],
    submit_plan: dict[str, Any],
) -> None:
    run_id = str(manifest["run_id"])
    stage_plan_path = run_root_path / "hpc" / "stage-plan.yaml"
    stored_stage_plan = _read_run_control_mapping(stage_plan_path, run_id=run_id, label="HPC stage plan")
    if stored_stage_plan != stage_report:
        raise RunLifecycleError.for_reuse(run_id, "the persisted HPC stage plan differs from the rendered plan")
    submit_plan_path = run_root_path / "hpc" / "submit-plan.json"
    submit_plan_bytes = (json.dumps(submit_plan, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    submit_plan_path.write_bytes(submit_plan_bytes)
    execute_script = _require_real_file(run_root_path / "execute.sh", run_id=run_id, label="reviewed execution script")
    slurm_script = _require_real_file(run_root_path / "submit.sbatch", run_id=run_id, label="reviewed SLURM script")
    extra_files = _reviewed_submission_extra_files(run_root_path=run_root_path, run_id=run_id)
    manifest["submission_identity"] = build_plan_identity(
        manifest,
        execute_script=execute_script,
        slurm_script=slurm_script,
        extra_files=extra_files,
    )
    write_run_manifest(run_root_path, manifest)


def _verify_reviewed_submission_identity(
    *,
    run_root_path: Path,
    run_id: str,
    manifest: dict[str, Any],
    execute_script: bytes,
    slurm_script: bytes | None,
) -> None:
    stored = manifest.get("submission_identity")
    if not isinstance(stored, dict) or slurm_script is None:
        raise RunLifecycleError.for_reuse(run_id, "the reviewed remote submission identity is incomplete")
    extra_files = _reviewed_submission_extra_files(run_root_path=run_root_path, run_id=run_id)
    expected = build_plan_identity(
        manifest,
        execute_script=execute_script,
        slurm_script=slurm_script,
        extra_files=extra_files,
    )
    if stored != expected:
        raise RunLifecycleError.for_reuse(run_id, "the reviewed remote stage or submission plan has changed")


def _load_reviewed_submission_plans(
    *,
    run_root_path: Path,
    run_id: str,
    status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_report = _read_run_control_mapping(
        run_root_path / "hpc" / "stage-plan.yaml",
        run_id=run_id,
        label="HPC stage plan",
    )
    submit_plan_bytes = _require_real_file(
        run_root_path / "hpc" / "submit-plan.json",
        run_id=run_id,
        label="HPC submit plan",
    )
    try:
        submit_plan = json.loads(submit_plan_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunLifecycleError.for_reuse(run_id, "the reviewed HPC submit plan is malformed") from exc
    if not isinstance(submit_plan, dict):
        raise RunLifecycleError.for_reuse(run_id, "the reviewed HPC submit plan is not a mapping")
    next_status = dict(status) | {"state": "stage-prepared", "last_updated": _timestamp()}
    return {"report": stage_report, "status": next_status}, submit_plan


def _write_new_tabular_run(
    *,
    run_root_path: Path,
    manifest: dict[str, Any],
    execute_script: bytes,
    slurm_script: bytes | None,
    initial_state: str,
    mode: str,
) -> None:
    run_root_path.mkdir()
    directory_names = ("work", "outputs", "logs") if mode == "slurm" else ("work", "logs")
    for name in directory_names:
        (run_root_path / name).mkdir()
    script_path = _write_run_shell_script(run_root_path=run_root_path, script_bytes=execute_script)
    if not script_path.is_file():
        raise RuntimeError("Failed to persist the reviewed execution script.")
    if slurm_script is not None:
        (run_root_path / "submit.sbatch").write_bytes(slurm_script)
    write_run_manifest(run_root_path, manifest)
    _write_tabular_status_atomic(
        run_root_path=run_root_path,
        status={
            "run_id": manifest["run_id"],
            "state": initial_state,
            "last_updated": _timestamp(),
            "job_id": "",
            "mode": mode,
        },
    )


class _TabularStatusPersistenceError(RuntimeError):
    """An atomic status replacement failed before or after the rename point."""

    def __init__(
        self,
        message: str,
        *,
        replacement_committed: bool,
        recovery_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.replacement_committed = replacement_committed
        self.recovery_path = recovery_path


def _write_tabular_status_atomic(*, run_root_path: Path, status: dict[str, Any]) -> None:
    """Durably replace status.yaml without exposing truncated YAML."""

    status_path = run_root_path / "status.yaml"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=run_root_path,
        prefix=".status.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    temporary_identity = os.fstat(descriptor)
    replacement_committed = False
    try:
        try:
            payload = dump_yaml(status).encode("utf-8")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("Short status write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary_path, status_path)
        replacement_committed = True
        directory_descriptor = os.open(run_root_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException as exc:
        try:
            current = os.lstat(temporary_path)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            raise _TabularStatusPersistenceError(
                "Atomic tabular status persistence failed and temporary cleanup state is "
                f"uncertain at {temporary_path}.",
                replacement_committed=replacement_committed,
                recovery_path=temporary_path,
            ) from cleanup_error
        else:
            if (current.st_dev, current.st_ino) == (temporary_identity.st_dev, temporary_identity.st_ino):
                try:
                    temporary_path.unlink()
                except OSError as cleanup_error:
                    raise _TabularStatusPersistenceError(
                        "Atomic tabular status persistence failed and owned temporary cleanup "
                        f"requires recovery at {temporary_path}.",
                        replacement_committed=replacement_committed,
                        recovery_path=temporary_path,
                    ) from cleanup_error
        state = status.get("state")
        raise _TabularStatusPersistenceError(
            f"Atomic persistence of tabular status {state!r} failed"
            + (" after replacement; durability is uncertain." if replacement_committed else "."),
            replacement_committed=replacement_committed,
        ) from exc


def _write_tabular_terminal_status(
    *,
    run_root_path: Path,
    run_id: str,
    state: str,
    mode: str,
) -> None:
    _write_tabular_status_atomic(
        run_root_path=run_root_path,
        status={
            "run_id": run_id,
            "state": state,
            "last_updated": _timestamp(),
            "job_id": "",
            "mode": mode,
        },
    )


def _transaction_json(
    staging: TabularOwnedStaging,
    relative_path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    data = read_tabular_transaction_file(staging.path, relative_path)
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise TabularOutputTransactionError(
            f"Tabular output {relative_path!r} changed during cross-contract validation."
        )
    return _transaction_json_bytes(data, label=relative_path)


def _transaction_json_from_path(path: Path) -> dict[str, Any]:
    return _transaction_json_bytes(
        _stable_regular_file_bytes(path, label="Tabular transaction JSON"),
        label=str(path),
    )


def _transaction_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(f"non-finite {constant}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TabularOutputTransactionError(f"Tabular output {label!r} is not strict JSON.") from exc
    if not isinstance(value, dict):
        raise TabularOutputTransactionError(f"Tabular output {label!r} must contain a JSON object.")
    return value


def _transaction_tsv_rows(
    staging: TabularOwnedStaging,
    relative_path: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    data = read_tabular_transaction_file(staging.path, relative_path)
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise TabularOutputTransactionError(
            f"Tabular output {relative_path!r} changed during cross-contract validation."
        )
    try:
        reader = csv.DictReader(data.decode("utf-8").splitlines(), delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("missing header")
        rows = [dict(row) for row in reader]
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise TabularOutputTransactionError(f"Tabular output {relative_path!r} is not a valid TSV.") from exc
    return list(reader.fieldnames), rows


def _is_finite_json_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_nonnegative_json_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_summary_statistic(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"n", "mean", "std", "min", "max"}:
        return False
    if not _is_nonnegative_json_integer(value.get("n")):
        return False
    numeric = [value.get(name) for name in ("mean", "std", "min", "max")]
    if value["n"] == 0:
        return all(item is None for item in numeric)
    return (
        all(_is_finite_json_number(item) for item in numeric)
        and float(value["std"]) >= 0.0
        and float(value["min"]) <= float(value["mean"]) <= float(value["max"])
    )


def _validate_tabular_analysis_report_structure(
    *,
    report: dict[str, Any],
    kind: str,
    config: dict[str, Any],
    source_reference: str,
) -> None:
    if report.get("table") != source_reference:
        raise TabularOutputTransactionError(
            "Tabular analysis output conflicts with its reviewed source table."
        )
    if kind == "correlation":
        if (
            set(report) != {"kind", "method", "x", "y", "n", "r", "table"}
            or report.get("method") != (config.get("method") or "pearson")
            or report.get("x") != config.get("x")
            or report.get("y") != config.get("y")
            or not _is_nonnegative_json_integer(report.get("n"))
            or report["n"] < 2
            or not _is_finite_json_number(report.get("r"))
            or abs(float(report["r"])) > 1.0
        ):
            raise TabularOutputTransactionError("Correlation output has an invalid result contract.")
        return
    if kind == "summary_table":
        columns = _analysis_list_value(config.get("columns"))
        summaries = report.get("summaries")
        if (
            set(report) != {"kind", "columns", "summaries", "table"}
            or report.get("columns") != columns
            or not isinstance(summaries, dict)
            or list(summaries) != columns
            or not all(_validate_summary_statistic(summaries[column]) for column in columns)
        ):
            raise TabularOutputTransactionError("Summary-table output has an invalid result contract.")
        return
    if kind == "linear_model":
        predictors = _analysis_list_value(config.get("predictors"))
        coefficients = report.get("coefficients")
        if (
            set(report) != {"kind", "outcome", "predictors", "n", "coefficients", "table"}
            or report.get("outcome") != config.get("outcome")
            or report.get("predictors") != predictors
            or not _is_nonnegative_json_integer(report.get("n"))
            or report["n"] <= len(predictors)
            or not isinstance(coefficients, dict)
            or set(coefficients) != {"intercept", *predictors}
            or not all(_is_finite_json_number(value) for value in coefficients.values())
        ):
            raise TabularOutputTransactionError("Linear-model output has an invalid result contract.")
        return
    if kind == "anova":
        predictors = _analysis_list_value(config.get("predictors"))
        group = config.get("group") or (predictors[0] if predictors else None)
        groups = report.get("groups")
        if (
            set(report) != {
                "kind", "outcome", "group", "n", "groups", "f", "df_between", "df_within", "table"
            }
            or report.get("outcome") != config.get("outcome")
            or report.get("group") != group
            or not _is_nonnegative_json_integer(report.get("n"))
            or not isinstance(groups, dict)
            or len(groups) < 2
            or not all(isinstance(label, str) and label and _validate_summary_statistic(value) for label, value in groups.items())
            or not _is_finite_json_number(report.get("f"))
            or float(report["f"]) < 0.0
            or not _is_nonnegative_json_integer(report.get("df_between"))
            or not _is_nonnegative_json_integer(report.get("df_within"))
        ):
            raise TabularOutputTransactionError("ANOVA output has an invalid result contract.")
        return
    if kind == "mixed_effects":
        predictors = _analysis_list_value(config.get("predictors"))
        group = config.get("group")
        expected_keys = {"kind", "outcome", "predictors", "engine", "n", "table"}
        if group:
            expected_keys |= {"group", "groups"}
        if (
            set(report) != expected_keys
            or report.get("outcome") != config.get("outcome")
            or report.get("predictors") != predictors
            or report.get("engine") != "summary-only"
            or not _is_nonnegative_json_integer(report.get("n"))
            or (group and report.get("group") != group)
            or (
                group
                and (
                    not isinstance(report.get("groups"), dict)
                    or not all(
                        isinstance(label, str) and label and _validate_summary_statistic(value)
                        for label, value in report["groups"].items()
                    )
                )
            )
        ):
            raise TabularOutputTransactionError("Mixed-effects output has an invalid result contract.")
        return
    raise TabularOutputTransactionError(f"Unsupported tabular analysis output kind {kind!r}.")


def _confirm_tabular_scientific_records(
    *,
    staging: TabularOwnedStaging,
    specs: tuple[TabularOutputSpec, ...],
    expected: tuple[TabularOutputRecord, ...],
) -> tuple[TabularOutputRecord, ...]:
    observed = validate_tabular_staged_outputs(staging, specs)
    if observed != expected:
        raise TabularOutputTransactionError(
            "Staged tabular output bytes changed during cross-contract validation."
        )
    return observed


def _validate_tabular_scientific_contract(
    *,
    context: dict[str, Any],
    staging: TabularOwnedStaging,
    manifest: dict[str, Any],
    specs: tuple[TabularOutputSpec, ...],
) -> tuple[TabularOutputRecord, ...]:
    """Validate cross-file scientific provenance without recomputing statistics."""

    validated_records = validate_tabular_staged_outputs(staging, specs)
    digests = {record.relative_path: record.sha256 for record in validated_records}
    for spec in specs:
        data = read_tabular_transaction_file(staging.path, spec.relative_path)
        if hashlib.sha256(data).hexdigest() != digests[spec.relative_path]:
            raise TabularOutputTransactionError(
                f"Tabular output {spec.relative_path!r} changed during cross-contract validation."
            )
        if str(staging.path).encode("utf-8") in data or staging.path.name.encode("utf-8") in data:
            raise TabularOutputTransactionError(
                f"Tabular output {spec.relative_path!r} contains a temporary staging reference."
            )

    workflow = manifest.get("workflow")
    if workflow == {"action": "analysis", "target": "tabular"}:
        analysis = manifest.get("analysis")
        if not isinstance(analysis, dict):
            raise TabularOutputTransactionError("Tabular analysis provenance is incomplete.")
        report_path = f"{analysis.get('name')}.json"
        report = _transaction_json(
            staging,
            report_path,
            expected_sha256=digests[report_path],
        )
        if report.get("kind") != analysis.get("kind"):
            raise TabularOutputTransactionError("Tabular analysis output kind conflicts with its reviewed plan.")
        config = analysis.get("config")
        if not isinstance(config, dict):
            raise TabularOutputTransactionError("Tabular analysis configuration provenance is incomplete.")
        kind = analysis.get("kind")
        source_reference = config.get("input_table")
        if not isinstance(kind, str) or not isinstance(source_reference, str) or not source_reference:
            raise TabularOutputTransactionError("Tabular analysis provenance is incomplete.")
        _validate_tabular_analysis_report_structure(
            report=report,
            kind=kind,
            config=config,
            source_reference=source_reference,
        )
        return _confirm_tabular_scientific_records(
            staging=staging,
            specs=specs,
            expected=validated_records,
        )

    predictor = manifest.get("predictor_contract")
    if not isinstance(predictor, dict):
        raise TabularOutputTransactionError("Tabular predictor provenance is incomplete.")
    target = predictor.get("target_column")
    features = predictor.get("feature_columns")
    if (
        not isinstance(target, str)
        or not isinstance(features, list)
        or not all(isinstance(value, str) for value in features)
        or predictor.get("feature_count") != len(features)
    ):
        raise TabularOutputTransactionError("Tabular predictor provenance is invalid.")

    if workflow == {"action": "evaluate", "target": "model"}:
        evaluation = _transaction_json(
            staging,
            "evaluation.json",
            expected_sha256=digests["evaluation.json"],
        )
        input_run = manifest.get("input_run")
        if not isinstance(input_run, dict) or not isinstance(input_run.get("run_id"), str):
            raise TabularOutputTransactionError("Evaluation input-run provenance is incomplete.")
        source = _load_tabular_input_run(context, input_run["run_id"])
        source_split = _transaction_json_from_path(Path(source["consumed"]["split_manifest"]["path"]))
        test_rows = source_split.get("test_rows")
        metrics = evaluation.get("metrics")
        predictions = evaluation.get("predictions")
        if evaluation.get("target_column") != target or evaluation.get("feature_columns") != features:
            raise TabularOutputTransactionError("Evaluation output conflicts with the reviewed predictor contract.")
        confusion = evaluation.get("confusion_matrix")
        if (
            evaluation.get("kind") != "logistic_regression_evaluation"
            or not isinstance(test_rows, list)
            or not isinstance(metrics, dict)
            or metrics.get("test_count") != len(test_rows)
            or not isinstance(predictions, list)
            or len(predictions) != len(test_rows)
            or evaluation.get("table_path") != str(source["consumed"]["features_table"]["path"])
            or set(metrics) != {"accuracy", "log_loss", "precision", "recall", "test_count"}
            or not all(
                _is_finite_json_number(metrics.get(name))
                for name in ("accuracy", "log_loss", "precision", "recall")
            )
            or not all(0.0 <= float(metrics[name]) <= 1.0 for name in ("accuracy", "precision", "recall"))
            or float(metrics["log_loss"]) < 0.0
            or not isinstance(confusion, dict)
            or set(confusion) != {"tn", "fp", "fn", "tp"}
            or not all(_is_nonnegative_json_integer(value) for value in confusion.values())
            or sum(confusion.values()) != len(test_rows)
        ):
            raise TabularOutputTransactionError("Evaluation output conflicts with its source split or model contract.")
        source_features_bytes = _stable_regular_file_bytes(
            Path(source["consumed"]["features_table"]["path"]),
            label="Evaluation source features",
        )
        try:
            source_reader = csv.DictReader(
                source_features_bytes.decode("utf-8").splitlines(),
                delimiter="\t",
            )
            source_feature_rows = [dict(row) for row in source_reader]
        except (UnicodeDecodeError, csv.Error) as exc:
            raise TabularOutputTransactionError("Evaluation source features are malformed.") from exc
        if any(
            not isinstance(prediction, dict)
            or set(prediction) != {"actual", "predicted", "probability", "row_number"}
            or prediction.get("row_number") != index
            or prediction.get("actual") not in {0, 1}
            or prediction.get("predicted") not in {0, 1}
            or not _is_finite_json_number(prediction.get("probability"))
            or not 0.0 <= float(prediction["probability"]) <= 1.0
            or test_rows[index] < 0
            or test_rows[index] >= len(source_feature_rows)
            or float(source_feature_rows[test_rows[index]][target]) != float(prediction["actual"])
            for index, prediction in enumerate(predictions)
        ):
            raise TabularOutputTransactionError(
                "Evaluation predictions conflict with source row identity or binary result structure."
            )
        return _confirm_tabular_scientific_records(
            staging=staging,
            specs=specs,
            expected=validated_records,
        )

    split = _transaction_json(staging, "split.json", expected_sha256=digests["split.json"])
    prep = _transaction_json(staging, "prep.json", expected_sha256=digests["prep.json"])
    columns, feature_rows = _transaction_tsv_rows(
        staging,
        "features.tsv",
        expected_sha256=digests["features.tsv"],
    )
    source_reference = manifest.get("dataset", {}).get("feature_table")
    if not isinstance(source_reference, str) or not source_reference:
        raise TabularOutputTransactionError("The tabular source-table reference is incomplete.")
    source_path = _resolve_reference_path(context["workspace_root"], source_reference)
    source_bytes = _stable_regular_file_bytes(source_path, label="Tabular input table")
    try:
        source_reader = csv.reader(
            source_bytes.decode("utf-8").splitlines(),
            delimiter="," if source_path.suffix.casefold() == ".csv" else "\t",
        )
        source_columns = next(source_reader)
        source_rows = [dict(zip(source_columns, row, strict=True)) for row in source_reader]
    except (UnicodeDecodeError, csv.Error, StopIteration, ValueError) as exc:
        raise TabularOutputTransactionError("The reviewed tabular input has no valid header.") from exc
    expected_feature_columns = list(source_columns)
    if "split_set" not in expected_feature_columns:
        expected_feature_columns.append("split_set")
    train_rows = split.get("train_rows")
    test_rows = split.get("test_rows")
    row_count = split.get("row_count")
    preprocessing_settings = manifest.get("settings", {}).get("preprocessing", {})
    if (
        split.get("kind") != "tabular_split"
        or split.get("table_path") != source_reference
        or split.get("target_column") != target
        or split.get("seed") != preprocessing_settings.get("split_seed")
        or split.get("test_fraction") != preprocessing_settings.get("test_fraction")
        or split.get("split_strategy") != "stratified_binary"
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or not isinstance(train_rows, list)
        or not isinstance(test_rows, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in train_rows + test_rows)
        or set(train_rows) & set(test_rows)
        or sorted(train_rows + test_rows) != list(range(row_count))
    ):
        raise TabularOutputTransactionError("Split output conflicts with its target or row-identity contract.")
    if (
        prep.get("kind") != "standardize_numeric"
        or prep.get("table_path") != source_reference
        or prep.get("target_column") != target
        or prep.get("feature_columns") != features
        or not isinstance(prep.get("statistics"), dict)
        or set(prep["statistics"]) != set(features)
        or len(prep["statistics"]) != len(features)
        or any(
            not isinstance(prep["statistics"].get(column), dict)
            or set(prep["statistics"][column]) != {"mean", "std"}
            or not _is_finite_json_number(prep["statistics"][column].get("mean"))
            or not _is_finite_json_number(prep["statistics"][column].get("std"))
            or float(prep["statistics"][column]["std"]) <= 0.0
            for column in features
        )
    ):
        raise TabularOutputTransactionError("Preprocessing output conflicts with the reviewed predictor contract.")
    if (
        len(feature_rows) != row_count
        or len(source_rows) != row_count
        or columns != expected_feature_columns
        or target not in columns
        or "split_set" not in columns
    ):
        raise TabularOutputTransactionError("Prepared feature output conflicts with the split row contract.")
    if any(column not in columns for column in features):
        raise TabularOutputTransactionError("Prepared feature output is missing a reviewed predictor.")
    if any(
        any(
            not value.strip()
            or not math.isfinite(float(value))
            for column in features
            for value in [row.get(column, "")]
        )
        for row in feature_rows
    ):
        raise TabularOutputTransactionError(
            "Prepared feature output contains a blank, nonnumeric, or nonfinite reviewed predictor."
        )
    expected_membership = {index: "train" for index in train_rows} | {index: "test" for index in test_rows}
    if any(row.get("split_set") != expected_membership[index] for index, row in enumerate(feature_rows)):
        raise TabularOutputTransactionError("Prepared feature rows conflict with split membership.")
    preserved_columns = [column for column in source_columns if column not in features]
    if any(
        any(feature_rows[index].get(column) != source_rows[index].get(column) for column in preserved_columns)
        for index in range(row_count)
    ):
        raise TabularOutputTransactionError("Prepared feature output changed a row identity or non-predictor value.")

    if workflow == {"action": "train", "target": "model"}:
        model = _transaction_json(
            staging,
            "model.json",
            expected_sha256=digests["model.json"],
        )
        model_settings = manifest.get("settings", {}).get("model", {})
        training_metrics = model.get("training_metrics")
        if (
            model.get("kind") != manifest.get("tool", {}).get("model")
            or model.get("target_column") != target
            or model.get("feature_columns") != features
            or model.get("table_path") != "outputs/features.tsv"
            or model.get("learning_rate") != model_settings.get("learning_rate", 0.2)
            or model.get("iterations") != model_settings.get("iterations", 350)
            or not _is_finite_json_number(model.get("intercept"))
            or not isinstance(model.get("weights"), dict)
            or set(model["weights"]) != set(features)
            or len(model["weights"]) != len(features)
            or not all(_is_finite_json_number(value) for value in model["weights"].values())
            or not isinstance(training_metrics, dict)
            or set(training_metrics) != {"accuracy"}
            or not _is_finite_json_number(training_metrics.get("accuracy"))
            or not 0.0 <= float(training_metrics["accuracy"]) <= 1.0
        ):
            raise TabularOutputTransactionError("Model output conflicts with its reviewed predictor or provenance contract.")
    return _confirm_tabular_scientific_records(
        staging=staging,
        specs=specs,
        expected=validated_records,
    )


def _finalize_tabular_run(
    *,
    context: dict[str, Any],
    run_root_path: Path,
    manifest: dict[str, Any],
    execute_script: bytes,
    slurm_script: bytes | None,
    command: list[str],
    mode: str,
    execute: bool,
    quiet: bool,
    allow_existing_review: bool = False,
    review_result: dict[str, Any] | None = None,
) -> int:
    run_id = str(manifest["run_id"])
    expected_workflow = dict(manifest["workflow"])
    runs_root = run_root_path.parent
    ensure_dir(runs_root)
    manifest["plan_identity"] = build_plan_identity(
        manifest,
        execute_script=execute_script,
        slurm_script=slurm_script,
    )
    requested_identity = dict(manifest["plan_identity"])
    if review_result is not None:
        review_result.update(
            {
                "plan_identity": requested_identity,
                "workflow": expected_workflow,
            }
        )
    exists = path_entry_exists(run_root_path)

    if not execute:
        if exists:
            if allow_existing_review:
                _inspect_tabular_reviewed_plan(
                    context=context,
                    run_id=run_id,
                    requested_identity=requested_identity,
                    expected_workflow=expected_workflow,
                    allow_claim=False,
                    allow_remote_material=True,
                )
                return 0
            raise RunLifecycleError.for_reuse(run_id, "the run root already exists")
        if path_entry_exists(claim_path(context["paths"]["artifacts_root"], run_id)):
            raise RunLifecycleError.for_reuse(run_id, "an execution claim already exists")
        try:
            _write_new_tabular_run(
                run_root_path=run_root_path,
                manifest=manifest,
                execute_script=execute_script,
                slurm_script=slurm_script,
                initial_state="planned",
                mode=mode,
            )
        except FileExistsError as exc:
            raise RunLifecycleError.for_reuse(run_id, "the run root was claimed concurrently") from exc
        if not quiet:
            print(json.dumps({"run_id": run_id, "manifest": str(run_root_path / "run-manifest.yaml"), "command": command}, indent=2))
        return 0

    if mode != "local":
        raise RunLifecycleError(f"Run {run_id!r} cannot execute through the {mode!r} planner.")
    transaction_plan = manifest.get("output_transaction")
    if not isinstance(transaction_plan, dict):
        raise RunLifecycleError("Local tabular execution requires a reviewed output-transaction plan.")
    try:
        transaction_specs = tabular_output_specs_from_plan(transaction_plan)
    except TabularOutputTransactionError as exc:
        raise RunLifecycleError(str(exc)) from exc
    support_error = tabular_atomic_no_replace_support_error()
    if support_error is not None:
        raise RunLifecycleError(support_error)

    # Fail before claiming or creating a direct-execution run whenever current
    # configuration-owned inputs no longer match the reviewed request.
    _revalidate_tabular_source_inputs(context=context, manifest=manifest)
    _revalidate_tabular_evaluation_input(context=context, manifest=manifest)
    if exists:
        _inspect_tabular_reviewed_plan(
            context=context,
            run_id=run_id,
            requested_identity=requested_identity,
            expected_workflow=expected_workflow,
            allow_claim=False,
        )

    claim = acquire_execution_claim(context["paths"]["artifacts_root"], run_id)
    staging: TabularOwnedStaging | None = None
    running_persisted = False
    promotion_committed = False
    reviewed_execute_script = execute_script
    run_root_identity: tuple[int, int] | None = None
    try:
        if exists:
            try:
                stored_manifest, _, run_root_path = _inspect_tabular_reviewed_plan(
                    context=context,
                    run_id=run_id,
                    requested_identity=requested_identity,
                    expected_workflow=expected_workflow,
                    allow_claim=True,
                )
            except BaseException:
                claim.release()
                raise
            manifest = stored_manifest
        else:
            if path_entry_exists(run_root_path):
                claim.release()
                raise RunLifecycleError.for_reuse(run_id, "the run root was claimed concurrently")
            try:
                _write_new_tabular_run(
                    run_root_path=run_root_path,
                    manifest=manifest,
                    execute_script=execute_script,
                    slurm_script=slurm_script,
                    initial_state="planned",
                    mode="local",
                )
            except FileExistsError as exc:
                claim.release()
                raise RunLifecycleError.for_reuse(run_id, "the run root was claimed concurrently") from exc

        run_root_identity = _real_directory_identity(
            run_root_path,
            label="Tabular run root",
        )
        reviewed_execute_script = _stable_regular_file_bytes(
            run_root_path / "execute.sh",
            label="Reviewed tabular execution script",
        )
        try:
            verified_identity = verify_plan_identity(
                manifest,
                execute_script=reviewed_execute_script,
            )
        except (TypeError, ValueError) as exc:
            raise RunLifecycleError.for_reuse(
                run_id,
                "the reviewed execution script changed after the execution claim was acquired",
            ) from exc
        if verified_identity != requested_identity:
            raise RunLifecycleError.for_reuse(
                run_id,
                "the reviewed execution script changed after the execution claim was acquired",
            )

        # Close the claim/check race before announcing that computation began.
        _revalidate_tabular_source_inputs(context=context, manifest=manifest)
        _revalidate_tabular_evaluation_input(context=context, manifest=manifest)
        _require_unchanged_real_directory(
            run_root_path,
            expected_identity=run_root_identity,
            label="Tabular run root",
        )
        preflight_tabular_transaction_root(run_root_path)
        try:
            _write_tabular_terminal_status(
                run_root_path=run_root_path,
                run_id=run_id,
                state="running",
                mode="local",
            )
        except _TabularStatusPersistenceError as exc:
            if exc.replacement_committed or exc.recovery_path is not None:
                print(
                    json.dumps(
                        {
                            "error": "Running-status durability is uncertain; the execution claim is retained for recovery.",
                            "run_id": run_id,
                            "claim": str(claim.path),
                            "recovery_path": str(exc.recovery_path or (run_root_path / "status.yaml")),
                            "detail": str(exc),
                        },
                        indent=2,
                    )
                )
                return 1
            raise
        running_persisted = True
        staging = create_tabular_staging(run_root_path)

        if shutil.which(command[0]) is None:
            raise TabularOutputTransactionError(f"{command[0]} not found on PATH")
        execution_environment = dict(os.environ)
        execution_environment[_TABULAR_OUTPUT_ROOT_VARIABLE] = str(staging.path)
        _revalidate_tabular_source_inputs(context=context, manifest=manifest)
        _revalidate_tabular_evaluation_input(context=context, manifest=manifest)
        _require_unchanged_real_directory(
            run_root_path,
            expected_identity=run_root_identity,
            label="Tabular run root",
        )
        if _stable_regular_file_bytes(
            run_root_path / "execute.sh",
            label="Reviewed tabular execution script",
        ) != reviewed_execute_script:
            raise RunLifecycleError.for_reuse(
                run_id,
                "the reviewed execution script changed immediately before launch",
            )
        try:
            completed = subprocess.run(
                [command[0], "-s"],
                cwd=context["workspace_root"],
                env=execution_environment,
                input=reviewed_execute_script,
                check=False,
            )
        except OSError as exc:
            raise TabularOutputTransactionError(f"Run launch failed: {exc}") from exc
        if completed.returncode != 0:
            raise TabularOutputTransactionError(
                f"Scientific command sequence exited nonzero ({completed.returncode})."
            )

        validated_records = _validate_tabular_scientific_contract(
            context=context,
            staging=staging,
            manifest=manifest,
            specs=transaction_specs,
        )
        _revalidate_tabular_source_inputs(context=context, manifest=manifest)
        _revalidate_tabular_evaluation_input(context=context, manifest=manifest)
        plan_identity = manifest["plan_identity"]
        transaction_manifest = seal_tabular_staged_transaction(
            staging,
            outputs=transaction_specs,
            run_id=run_id,
            workflow_action=str(expected_workflow["action"]),
            workflow_target=str(expected_workflow["target"]),
            plan_identity_schema=str(plan_identity["schema_version"]),
            plan_identity_sha256=str(plan_identity["sha256"]),
            expected_records=validated_records,
        )
        _revalidate_tabular_source_inputs(context=context, manifest=manifest)
        _revalidate_tabular_evaluation_input(context=context, manifest=manifest)
        _require_unchanged_real_directory(
            run_root_path,
            expected_identity=run_root_identity,
            label="Tabular run root",
        )
        validate_sealed_tabular_transaction(
            staging,
            outputs=transaction_specs,
            expected_manifest=transaction_manifest,
        )
        _revalidate_tabular_source_inputs(context=context, manifest=manifest)
        _revalidate_tabular_evaluation_input(context=context, manifest=manifest)
        _require_unchanged_real_directory(
            run_root_path,
            expected_identity=run_root_identity,
            label="Tabular run root",
        )
        promote_tabular_staging_no_replace(
            staging,
            run_root_path / TABULAR_FINAL_OUTPUT_DIRECTORY,
        )
        promotion_committed = True
        staging = None
        try:
            _write_tabular_terminal_status(
                run_root_path=run_root_path,
                run_id=run_id,
                state="succeeded",
                mode="local",
            )
        except BaseException as exc:
            status_detail = (
                "status.yaml may already read 'succeeded', but its durability is uncertain"
                if isinstance(exc, _TabularStatusPersistenceError) and exc.replacement_committed
                else "the last confirmed durable state is 'running'"
            )
            recovery_paths = [str(run_root_path / TABULAR_FINAL_OUTPUT_DIRECTORY)]
            if isinstance(exc, _TabularStatusPersistenceError) and exc.recovery_path is not None:
                recovery_paths.append(str(exc.recovery_path))
            print(
                json.dumps(
                    {
                        "error": (
                            "Outputs were committed, but durable success-status persistence failed; "
                            "the execution claim is retained for recovery."
                        ),
                        "run_id": run_id,
                        "claim": str(claim.path),
                        "recovery_path": str(run_root_path / TABULAR_FINAL_OUTPUT_DIRECTORY),
                        "recovery_paths": recovery_paths,
                        "detail": str(exc),
                        "status_detail": status_detail,
                    },
                    indent=2,
                )
            )
            return 1
        try:
            claim.release()
        except RunLifecycleError as exc:
            print(
                json.dumps(
                    {
                        "error": "Outputs succeeded, but execution-claim cleanup requires recovery.",
                        "run_id": run_id,
                        "claim": str(claim.path),
                        "detail": str(exc),
                    },
                    indent=2,
                )
            )
            return 1
        if not quiet:
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "manifest": str(run_root_path / "run-manifest.yaml"),
                        "transaction_manifest": str(
                            run_root_path / TABULAR_FINAL_OUTPUT_DIRECTORY / "transaction-manifest.json"
                        ),
                        "return_code": 0,
                    },
                    indent=2,
                )
            )
        return 0
    except KeyboardInterrupt as exc:
        status_error: str | None = None
        try:
            if run_root_path.is_dir() and not run_root_path.is_symlink():
                _write_tabular_terminal_status(
                    run_root_path=run_root_path,
                    run_id=run_id,
                    state="failed",
                    mode="local",
                )
        except BaseException as write_error:
            status_error = str(write_error)
        recovery_paths = [str(claim.path)]
        if staging is not None:
            recovery_paths.append(str(staging.path))
        if promotion_committed:
            recovery_paths.append(str(run_root_path / TABULAR_FINAL_OUTPUT_DIRECTORY))
        print(
            json.dumps(
                {
                    "error": "Run interrupted; the execution claim and recovery evidence are retained.",
                    "run_id": run_id,
                    "recovery_paths": recovery_paths,
                    "status_error": status_error,
                    "detail": str(exc),
                },
                indent=2,
            )
        )
        return 130
    except (
        RunLifecycleError,
        TabularOutputTransactionError,
        _TabularStatusPersistenceError,
        OSError,
        ValueError,
    ) as exc:
        committed = promotion_committed or (
            isinstance(exc, TabularOutputTransactionError) and exc.promotion_committed
        )
        if committed:
            recovery_path = (
                exc.recovery_path
                if isinstance(exc, TabularOutputTransactionError) and exc.recovery_path is not None
                else run_root_path / TABULAR_FINAL_OUTPUT_DIRECTORY
            )
            print(
                json.dumps(
                    {
                        "error": "Tabular output promotion is committed or uncertain; the claim is retained for recovery.",
                        "run_id": run_id,
                        "claim": str(claim.path),
                        "recovery_path": str(recovery_path),
                        "detail": str(exc),
                    },
                    indent=2,
                )
            )
            return 1

        if (
            isinstance(exc, _TabularStatusPersistenceError)
            and (exc.replacement_committed or exc.recovery_path is not None)
        ):
            print(
                json.dumps(
                    {
                        "error": "Tabular status replacement is committed or uncertain; the claim is retained for recovery.",
                        "run_id": run_id,
                        "claim": str(claim.path),
                        "recovery_path": str(exc.recovery_path or (run_root_path / "status.yaml")),
                        "detail": str(exc),
                    },
                    indent=2,
                )
            )
            return 1

        if (
            isinstance(exc, TabularOutputTransactionError)
            and exc.recovery_path is not None
            and staging is None
        ):
            print(
                json.dumps(
                    {
                        "error": "Tabular transaction recovery residue may remain; the claim is retained.",
                        "run_id": run_id,
                        "claim": str(claim.path),
                        "recovery_path": str(exc.recovery_path),
                        "detail": str(exc),
                    },
                    indent=2,
                )
            )
            return 1

        if not running_persisted:
            claim.release()
            if isinstance(exc, RunLifecycleError):
                raise
            raise RunLifecycleError(f"Local tabular execution could not start safely: {exc}") from exc

        recovery_path: Path | None = None
        if staging is not None:
            try:
                cleanup_tabular_staging(staging)
                staging = None
            except TabularOutputTransactionError as cleanup_error:
                recovery_path = cleanup_error.recovery_path or staging.path
                print(
                    json.dumps(
                        {
                            "error": "Tabular execution failed and owned staging cleanup requires recovery.",
                            "run_id": run_id,
                            "claim": str(claim.path),
                            "recovery_path": str(recovery_path),
                            "detail": str(exc),
                            "cleanup_error": str(cleanup_error),
                        },
                        indent=2,
                    )
                )
                return 1
        try:
            _write_tabular_terminal_status(
                run_root_path=run_root_path,
                run_id=run_id,
                state="failed",
                mode="local",
            )
        except BaseException as status_error:
            print(
                json.dumps(
                    {
                        "error": "Tabular execution failed and terminal-status persistence requires recovery.",
                        "run_id": run_id,
                        "claim": str(claim.path),
                        "detail": str(exc),
                        "status_error": str(status_error),
                    },
                    indent=2,
                )
            )
            return 1
        try:
            claim.release()
        except RunLifecycleError as claim_error:
            print(
                json.dumps(
                    {
                        "error": "Tabular execution failed and execution-claim cleanup requires recovery.",
                        "run_id": run_id,
                        "claim": str(claim.path),
                        "detail": str(exc),
                        "claim_error": str(claim_error),
                    },
                    indent=2,
                )
            )
            return 1
        print(json.dumps({"error": str(exc), "run_id": run_id}, indent=2))
        return 1


def _finalize_run(
    *,
    context: dict[str, Any],
    run_root_path: Path,
    manifest: dict[str, Any],
    command: list[str],
    mode: str,
    execute: bool,
    log_dir: Path,
    manifest_path: Path,
    job_name: str,
    hpc_connection: dict[str, str] | None = None,
    quiet: bool = False,
) -> int:
    manifest, slurm_script_bytes = _complete_run_manifest(
        context=context,
        run_root_path=run_root_path,
        manifest=manifest,
        command=command,
        mode=mode,
        job_name=job_name,
        hpc_connection=hpc_connection,
    )
    if slurm_script_bytes is not None:
        script_path = run_root_path / "submit.sbatch"
        script_path.write_bytes(slurm_script_bytes)
    write_run_manifest(run_root_path, manifest)
    write_status(run_root_path, {"run_id": manifest["run_id"], "state": "planned" if mode != "local" or not execute else "running", "last_updated": _timestamp(), "job_id": "", "mode": mode})

    if mode == "local" and execute:
        if shutil.which(command[0]) is None:
            print(json.dumps({"error": f"{command[0]} not found on PATH", "command": command}, indent=2))
            return 1
        completed = subprocess.run(command, cwd=context["workspace_root"], check=False)
        write_status(
            run_root_path,
            {
                "run_id": manifest["run_id"],
                "state": "succeeded" if completed.returncode == 0 else "failed",
                "last_updated": _timestamp(),
                "job_id": "",
                "mode": mode,
            },
        )
        if not quiet:
            print(json.dumps({"run_id": manifest["run_id"], "manifest": str(manifest_path), "return_code": completed.returncode}, indent=2))
        return completed.returncode

    if not quiet:
        print(json.dumps({"run_id": manifest["run_id"], "manifest": str(manifest_path), "command": command}, indent=2))
    return 0


def _load_run(context: dict[str, Any], run_id: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_root_path = run_path(context["paths"]["artifacts_root"], run_id)
    return read_run_manifest(run_root_path), read_status(run_root_path), run_root_path


def _load_legacy_tabular_input_run(context: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Preserve the pre-transaction remote planning boundary for SLURM plans."""

    run_root_path = run_path(context["paths"]["artifacts_root"], run_id)
    try:
        manifest = read_run_manifest(run_root_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RunLifecycleError(f"Input run {run_id!r} could not be loaded: {exc}") from exc
    workflow = manifest.get("workflow")
    if manifest.get("slice") != "tabular" or workflow != {"action": "train", "target": "model"}:
        raise RunLifecycleError(f"Input run {run_id!r} must be a tabular train-model run.")
    if manifest.get("project", {}).get("name") != context["project"]["name"]:
        raise RunLifecycleError(f"Input run {run_id!r} belongs to a different project.")
    if manifest.get("batch", {}).get("name") != context["batch"]["name"]:
        raise RunLifecycleError(
            f"Input run {run_id!r} does not match batch {context['batch']['name']!r}."
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise RunLifecycleError(f"Input run {run_id!r} has no output plan.")
    missing = [name for name in ("split_manifest", "features_table", "model") if not outputs.get(name)]
    if missing:
        raise RunLifecycleError(
            f"Input run {run_id!r} is missing planned outputs: {', '.join(missing)}."
        )
    return {"run_id": run_id, "run_root": run_root_path, "manifest": manifest}


def _load_tabular_input_run(context: dict[str, Any], run_id: str) -> dict[str, Any]:
    run_root_path = run_path(context["paths"]["artifacts_root"], run_id)
    try:
        _require_real_directory(run_root_path, run_id=run_id, label="input run root")
        run_root_identity = _real_directory_identity(
            run_root_path,
            label="Input training run root",
        )
        if path_entry_exists(claim_path(context["paths"]["artifacts_root"], run_id)):
            raise RunLifecycleError.for_reuse(run_id, "the input run has an execution claim")
        if tabular_staging_entries(run_root_path):
            raise RunLifecycleError.for_reuse(run_id, "the input run has transaction recovery residue")
        manifest_bytes = _require_real_file(
            run_root_path / "run-manifest.yaml",
            run_id=run_id,
            label="input run manifest",
        )
        status_bytes = _require_real_file(
            run_root_path / "status.yaml",
            run_id=run_id,
            label="input run status",
        )
        execute_bytes = _require_real_file(
            run_root_path / "execute.sh",
            run_id=run_id,
            label="input reviewed execution script",
        )
        try:
            manifest = parse_yaml(manifest_bytes.decode("utf-8"), resolve_env=False)
            status = parse_yaml(status_bytes.decode("utf-8"), resolve_env=False)
        except (UnicodeDecodeError, ValueError) as exc:
            raise RunLifecycleError.for_reuse(run_id, "the input control files are malformed") from exc
        if not isinstance(manifest, dict) or not isinstance(status, dict):
            raise RunLifecycleError.for_reuse(run_id, "the input control files are not mappings")
        if manifest.get("run_id") != run_id or status.get("run_id") != run_id:
            raise RunLifecycleError.for_reuse(run_id, "the input control files identify another run")
        required_status_keys = {"run_id", "state", "last_updated", "job_id", "mode"}
        if not required_status_keys.issubset(status) or not all(
            isinstance(status.get(key), str) for key in required_status_keys
        ):
            raise RunLifecycleError.for_reuse(run_id, "the input run status is incomplete")
        workflow = manifest.get("workflow")
        if manifest.get("slice") != "tabular" or workflow != {"action": "train", "target": "model"}:
            raise RunLifecycleError.for_reuse(run_id, "the input is not a local tabular train-model run")
        if status.get("state") != "succeeded" or status.get("mode") != "local":
            raise RunLifecycleError.for_reuse(run_id, "the input training transaction is not locally succeeded")
        execution = manifest.get("execution")
        if not isinstance(execution, dict) or execution.get("mode") not in {"plan", "local"}:
            raise RunLifecycleError.for_reuse(run_id, "the input training plan is remote or malformed")
        if manifest.get("project", {}).get("name") != context["project"]["name"]:
            raise RunLifecycleError.for_reuse(run_id, "the input run belongs to a different project")
        expected_project_root = to_workspace_relative(
            context["project_root"],
            context["workspace_root"],
        )
        if manifest.get("project", {}).get("root") != expected_project_root:
            raise RunLifecycleError.for_reuse(run_id, "the input run has a different project root")
        if manifest.get("batch", {}).get("name") != context["batch"]["name"]:
            raise RunLifecycleError.for_reuse(run_id, f"the input run does not match batch {context['batch']['name']!r}")
        expected_batch_path = to_workspace_relative(
            Path(context["batch"]["path"]),
            context["workspace_root"],
        )
        if manifest.get("batch", {}).get("path") != expected_batch_path:
            raise RunLifecycleError.for_reuse(run_id, "the input run has a different batch path")
        if manifest.get("batch", {}).get("selected_row") != context["batch"].get("selected_row"):
            raise RunLifecycleError.for_reuse(run_id, "the input run has a different selected batch row")
        source_digests = manifest.get("source_digests")
        if not isinstance(source_digests, dict):
            raise RunLifecycleError.for_reuse(run_id, "the input run has no source digest contract")
        current_batch_digest = _stable_regular_file_sha256(
            Path(context["batch"]["path"]),
            label="Evaluation batch manifest",
        )
        if source_digests.get("batch_sha256") != current_batch_digest:
            raise RunLifecycleError.for_reuse(run_id, "the input run was trained from different batch bytes")
        expected_root_names = {"execute.sh", "run-manifest.yaml", "status.yaml", "work", "logs", "outputs"}
        if {entry.name for entry in run_root_path.iterdir()} != expected_root_names:
            raise RunLifecycleError.for_reuse(run_id, "the input run root has an unsafe or incomplete inventory")
        _require_real_directory(run_root_path / "work", run_id=run_id, label="input work directory")
        _require_real_directory(run_root_path / "logs", run_id=run_id, label="input log directory")
        plan_identity = verify_plan_identity(manifest, execute_script=execute_bytes)
        transaction_plan = manifest.get("output_transaction")
        if not isinstance(transaction_plan, dict):
            raise RunLifecycleError.for_reuse(run_id, "the input run has no transaction plan")
        specs = tabular_output_specs_from_plan(transaction_plan)
        expected_specs = (
            TabularOutputSpec("split_manifest", "split.json", "json"),
            TabularOutputSpec("prep_plan", "prep.json", "json"),
            TabularOutputSpec("features_table", "features.tsv", "tsv"),
            TabularOutputSpec("model", "model.json", "json"),
        )
        if specs != expected_specs:
            raise RunLifecycleError.for_reuse(run_id, "the input training transaction has an unexpected output contract")
        compact_plan_identity = {
            "schema_version": plan_identity["schema_version"],
            "sha256": plan_identity["sha256"],
        }
        transaction_manifest, transaction_digest, records = validate_committed_tabular_transaction(
            run_root_path / TABULAR_FINAL_OUTPUT_DIRECTORY,
            outputs=specs,
            expected_run_id=run_id,
            expected_workflow_action="train",
            expected_workflow_target="model",
            expected_plan_identity=compact_plan_identity,
        )
        _require_unchanged_real_directory(
            run_root_path,
            expected_identity=run_root_identity,
            label="Input training run root",
        )
        if path_entry_exists(claim_path(context["paths"]["artifacts_root"], run_id)):
            raise RunLifecycleError.for_reuse(run_id, "the input run acquired an execution claim during validation")
        if tabular_staging_entries(run_root_path):
            raise RunLifecycleError.for_reuse(run_id, "the input run acquired transaction recovery residue during validation")
    except (FileNotFoundError, OSError, ValueError, RuntimeError, TabularOutputTransactionError) as exc:
        if isinstance(exc, RunLifecycleError):
            raise
        raise RunLifecycleError.for_reuse(run_id, f"the input training transaction is invalid: {exc}") from exc

    records_by_name = {record.logical_name: record for record in records}
    consumed: dict[str, dict[str, Any]] = {}
    for logical_name in ("split_manifest", "features_table", "model"):
        record = records_by_name[logical_name]
        consumed[logical_name] = {
            "path": run_root_path / TABULAR_FINAL_OUTPUT_DIRECTORY / record.relative_path,
            "relative_path": record.relative_path,
            "sha256": record.sha256,
        }
    return {
        "run_id": run_id,
        "run_root": run_root_path,
        "manifest": manifest,
        "plan_identity": compact_plan_identity,
        "transaction_manifest": transaction_manifest,
        "transaction_manifest_sha256": transaction_digest,
        "output_records": [record.to_dict() for record in records],
        "consumed": consumed,
    }


def _stable_regular_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one regular nonsymlink file while detecting in-read replacement."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RunLifecycleError(f"{label} could not be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        import stat

        if not stat.S_ISREG(before.st_mode):
            raise RunLifecycleError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RunLifecycleError(f"{label} changed while it was hashed: {path}")
        current = os.lstat(path)
        if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
            raise RunLifecycleError(f"{label} changed filesystem identity while it was hashed: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stable_regular_file_sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(_stable_regular_file_bytes(path, label=label)).hexdigest()


def _revalidate_tabular_evaluation_input(*, context: dict[str, Any], manifest: dict[str, Any]) -> None:
    if manifest.get("workflow") != {"action": "evaluate", "target": "model"}:
        return
    stored = manifest.get("input_run")
    if not isinstance(stored, dict) or not isinstance(stored.get("run_id"), str):
        raise RunLifecycleError("The evaluation plan has no complete input-run attestation.")
    current = _load_tabular_input_run(context, stored["run_id"])
    predictor = _tabular_input_predictor_contract(run_id=current["run_id"], manifest=current["manifest"])
    expected = _tabular_evaluation_input_block(input_run=current, predictor_contract=predictor)
    if stored != expected:
        raise RunLifecycleError(
            "The successful training transaction changed after evaluation planning; choose a new evaluation run id."
        )


def _revalidate_tabular_source_inputs(*, context: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Recheck every configuration-owned source digest bound into the reviewed plan."""

    source_digests = manifest.get("source_digests")
    if not isinstance(source_digests, dict):
        raise RunLifecycleError("The reviewed tabular plan has no source-digest contract.")
    expected_table = source_digests.get("input_table_sha256")
    if not isinstance(expected_table, str):
        raise RunLifecycleError("The reviewed tabular plan has no input-table digest.")
    if manifest.get("workflow") == {"action": "analysis", "target": "tabular"}:
        input_reference = manifest.get("analysis", {}).get("config", {}).get("input_table")
    else:
        input_reference = manifest.get("dataset", {}).get("feature_table")
    if not isinstance(input_reference, str) or not input_reference:
        raise RunLifecycleError("The reviewed tabular plan has no portable input-table reference.")
    input_path = _resolve_reference_path(context["workspace_root"], input_reference)
    if _stable_regular_file_sha256(input_path, label="Tabular input table") != expected_table:
        raise RunLifecycleError("The tabular input table changed after planning; choose a new run id.")

    expected_batch = source_digests.get("batch_sha256")
    if expected_batch is not None:
        batch_reference = manifest.get("batch", {}).get("path")
        if not isinstance(expected_batch, str) or not isinstance(batch_reference, str) or not batch_reference:
            raise RunLifecycleError("The reviewed tabular plan has an invalid batch-digest contract.")
        batch_path = _resolve_reference_path(context["workspace_root"], batch_reference)
        if _stable_regular_file_sha256(batch_path, label="Tabular batch manifest") != expected_batch:
            raise RunLifecycleError("The tabular batch manifest changed after planning; choose a new run id.")


def _resolve_run_resources(*, context: dict[str, Any], workload: str, mode: str) -> dict[str, Any]:
    try:
        return resolve_resource_plan(compute_config=context["compute"], workload=workload, mode=mode)
    except ValueError as exc:
        raise SystemExit(json.dumps({"error": str(exc)}, indent=2)) from exc


def _run_resource_workload_key(action: str, target: str) -> str:
    mapping = {
        ("preprocess", "bids"): "bids_preprocess",
        ("analysis", "bids"): "bids_analysis_first_level",
        ("analysis", "tabular"): "tabular_analysis",
        ("preprocess", "tabular"): "tabular_preprocess",
        ("train", "model"): "tabular_train_model",
        ("evaluate", "model"): "tabular_evaluate_model",
    }
    return mapping[(action, target)]


def _snakemake_resource_args(*, rule_name: str, resources: dict[str, Any], slurm_time: Any = None) -> list[str]:
    mem_mb = int(resources["ram_gb"]) * 1024
    threads = int(resources.get("threads", resources["cpus"]))
    cpus_per_task = int(resources["cpus"])
    resource_args = [
        "--set-threads",
        f"{rule_name}={threads}",
        "--set-resources",
        f"{rule_name}:mem_mb={mem_mb}",
        f"{rule_name}:cpus_per_task={cpus_per_task}",
    ]
    runtime_value = _slurm_time_to_snakemake_runtime(slurm_time)
    if runtime_value is not None:
        resource_args.append(f"{rule_name}:runtime={runtime_value}")
    return resource_args


def _snakemake_core_args(*, resources: dict[str, Any]) -> list[str]:
    threads = int(resources.get("threads", resources["cpus"]))
    n_jobs = int(resources.get("n_jobs", 1))
    return ["--cores", str(max(1, threads * n_jobs))]


def _slurm_time_to_snakemake_runtime(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    days = 0
    time_text = text
    if "-" in time_text:
        day_text, time_text = time_text.split("-", 1)
        if not re.fullmatch(r"\d+", day_text):
            return None
        days = int(day_text)

    hours = 0
    minutes = 0
    seconds = 0
    parts = time_text.split(":")
    if len(parts) == 3 and all(re.fullmatch(r"\d+", part) for part in parts):
        hours, minutes, seconds = (int(part) for part in parts)
    elif len(parts) == 2 and all(re.fullmatch(r"\d+", part) for part in parts):
        hours, minutes = (int(part) for part in parts)
    elif len(parts) == 1 and re.fullmatch(r"\d+", parts[0]):
        minutes = int(parts[0])
    else:
        return None

    duration_parts: list[str] = []
    if days:
        duration_parts.append(f"{days}d")
    if hours:
        duration_parts.append(f"{hours}h")
    if minutes:
        duration_parts.append(f"{minutes}m")
    if seconds:
        duration_parts.append(f"{seconds}s")
    if not duration_parts:
        duration_parts.append("1m")
    return "".join(duration_parts)


def _resolve_reference_path(workspace_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (workspace_root / candidate).resolve()


def _resolve_bids_selection(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, str | None]:
    return {
        "task_id": _first_nonempty(
            getattr(args, "task_id", None),
            context["preprocessing"].get("task_id"),
            context["preprocessing"].get("task"),
            context["preprocessing"].get("task_label"),
            context["dataset"].get("task_id"),
            context["dataset"].get("task"),
            context["dataset"].get("task_label"),
        ),
        "run_id": _first_nonempty(
            getattr(args, "row_run_id", None),
            context["preprocessing"].get("run_id"),
            context["dataset"].get("run_id"),
        ),
        "session_id": _first_nonempty(
            context["preprocessing"].get("session_id"),
            context["dataset"].get("session_id"),
        ),
        "subject_id": _first_nonempty(
            context["preprocessing"].get("subject_id"),
            context["dataset"].get("subject_id"),
        ),
    }


def _resolve_analysis_selection(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, str | None]:
    return {
        "task_id": _first_nonempty(
            getattr(args, "task_id", None),
            context["analysis_stage"].get("task_id"),
            context["dataset"].get("task_id"),
            context["dataset"].get("task"),
        ),
        "run_id": _first_nonempty(
            getattr(args, "row_run_id", None),
            context["analysis_stage"].get("run_id"),
            context["dataset"].get("run_id"),
        ),
        "session_id": _first_nonempty(
            context["analysis_stage"].get("session_id"),
            context["dataset"].get("session_id"),
        ),
        "subject_id": _first_nonempty(
            context["analysis_stage"].get("subject_id"),
            context["dataset"].get("subject_id"),
        ),
    }


def _bids_selector_values(args: argparse.Namespace) -> dict[str, object]:
    return {
        "subject_id": normalize_filter_values(getattr(args, "subject_id", None)),
        "session_id": _normalize_optional_cli_value(getattr(args, "session_id", None)),
        "task_id": _normalize_optional_cli_value(getattr(args, "task_id", None)),
        "run_id": _normalize_optional_cli_value(getattr(args, "selector_run_id", None)),
    }


def _discover_bids_batch_for_submit(args: argparse.Namespace, *, context: dict[str, Any]) -> dict[str, Any]:
    rows = context["tool_adapter"].discover_batch_rows(
        derivative_root=str(context["input_derivative_root"]),
        selectors=_resolve_bids_selection(args, context),
    )
    rows = _filter_bids_rows(rows, _bids_run_batch_filters(args))
    if not rows:
        raise SystemExit(json.dumps({"error": "No matching BIDS rows were discovered."}, indent=2))

    batch_name = context["batch"]["name"]
    batch_path = Path(context["batch"]["path"])
    _write_batch_manifest(batch_path, rows)
    return {
        "batch_name": batch_name,
        "batch_path": to_workspace_relative(batch_path, context["workspace_root"]),
        "row_count": len(rows),
    }


def _discover_analysis_bids_batch_for_submit(args: argparse.Namespace, *, context: dict[str, Any]) -> dict[str, Any]:
    rows = context["tool_adapter"].discover_batch_rows(
        derivative_root=str(context["input_derivative_root"]),
        selectors=_resolve_analysis_selection(args, context),
        context=context,
    )
    rows = _filter_bids_rows(rows, _bids_run_batch_filters(args))
    if not rows:
        raise SystemExit(json.dumps({"error": "No matching derivative-backed rows were discovered."}, indent=2))

    batch_name = context["batch"]["name"]
    batch_path = Path(context["batch"]["path"])
    _write_batch_manifest(batch_path, rows)
    return {
        "batch_name": batch_name,
        "batch_path": to_workspace_relative(batch_path, context["workspace_root"]),
        "row_count": len(rows),
    }


def _manifest_hpc_connection_hint(args: argparse.Namespace) -> dict[str, str] | None:
    return build_manifest_hpc_connection_hint(
        profile_name=getattr(args, "profile", None),
        role=getattr(args, "role", None),
        config_path=getattr(args, "config", None),
    )


def _validate_project_init_name(value: str) -> str:
    try:
        return _validate_simple_scaffold_name(value, label="--project")
    except SystemExit as exc:
        if str(exc) == "--project must be a simple name, not a path.":
            raise SystemExit("--project must be a simple project name, not a path.") from exc
        raise


def _validate_simple_scaffold_name(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise SystemExit(f"{label} must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise SystemExit(f"{label} must be a simple name, not a path.")
    return normalized


def _validate_relative_scaffold_subpath(value: str, *, label: str) -> PurePosixPath:
    normalized = str(value).strip()
    if not normalized:
        raise SystemExit(f"{label} must not be empty.")
    if Path(normalized).is_absolute() or PurePosixPath(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
        raise SystemExit(f"{label} must be a relative path under the canonical dataset root.")
    candidate = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise SystemExit(f"{label} must be a relative path under the canonical dataset root.")
    return candidate


def _validate_existing_local_directory(value: str, *, label: str) -> Path:
    resolved = Path(value).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{label} was not found: {resolved}")
    if not resolved.is_dir():
        raise SystemExit(f"{label} must point to a directory: {resolved}")
    return resolved


def _validate_remote_posix_root(value: str | None, *, label: str) -> str | None:
    normalized = _normalize_optional_cli_value(value)
    if normalized is None:
        return None
    path = PurePosixPath(normalized)
    if not path.is_absolute():
        raise SystemExit(f"{label} must be an absolute remote POSIX path.")
    return str(path)


def _render_project_init_summary(payload: dict[str, Any]) -> list[str]:
    project_name = str(payload["project_name"])
    created_files = [str(path) for path in payload["created_files"]]
    lines = [
        f"Initialized BIDS preprocessing project: {project_name}",
        f"Tool: {payload['tool']}",
        f"Adapter: {payload['adapter']}",
        "",
        "Created files",
    ]
    lines.extend(f"- {to_workspace_relative(path, workspace_root())}" for path in created_files)
    lines.extend(
        [
            "",
            "Next commands",
            f"- rp config validate --project {project_name}",
            f"- rp hpc sync data --project {project_name} --profile <profile>",
            f"- rp run submit preprocess bids --project {project_name} --discover --run-id {project_name}-submit",
            "- Review the rendered sync and submission plans before authorizing remote changes.",
            f"- rp hpc sync data --project {project_name} --profile <profile> --execute",
            f"- rp run submit preprocess bids --project {project_name} --discover --run-id {project_name}-submit --execute",
        ]
    )
    return lines


def _render_analysis_project_init_summary(payload: dict[str, Any]) -> list[str]:
    project_name = str(payload["project_name"])
    created_files = [str(path) for path in payload["created_files"]]
    lines = [
        f"Initialized BIDS analysis project: {project_name}",
        f"Tool: {payload['tool']}",
        f"Adapter: {payload['adapter']}",
    ]
    if payload.get("template") is not None:
        lines.append(f"Template: {payload['template']}")
    if payload.get("template_reason") is not None:
        lines.append(f"Template reason: {payload['template_reason']}")
    if payload.get("hpc_target") is not None:
        lines.append(f"HPC target: {payload['hpc_target']}")
    lines.extend(["", "Created files"])
    lines.extend(f"- {to_workspace_relative(path, workspace_root())}" for path in created_files)
    lines.extend(
        [
            "",
            "Next commands",
            f"- rp config validate --project {project_name}",
            f"- rp batch discover analysis bids --project {project_name} --stage first_level",
            f"- rp run submit analysis bids --project {project_name} --stage first_level --discover --run-id {project_name}-submit",
            "- Review the rendered submission plan before authorizing remote changes.",
            f"- rp run submit analysis bids --project {project_name} --stage first_level --discover --run-id {project_name}-submit --execute",
        ]
    )
    return lines


def _render_tabular_model_project_init_summary(payload: dict[str, Any]) -> list[str]:
    project_name = str(payload["project_name"])
    created_files = [str(path) for path in payload["created_files"]]
    created_directories = [str(path) for path in payload["created_directories"]]
    lines = [
        f"Initialized tabular model project: {project_name}",
        f"Dataset: {payload['dataset_name']}",
        f"Canonical dataset: {payload['canonical_dataset_name']}",
        f"Canonical features root: {payload['canonical_features_root']}",
        "",
        "Created files",
    ]
    lines.extend(f"- {to_workspace_relative(path, workspace_root())}" for path in created_files)
    lines.extend(
        [
            "",
            "Created directories",
        ]
    )
    lines.extend(f"- {to_workspace_relative(path, workspace_root())}" for path in created_directories)
    lines.extend(
        [
            "",
            "Next commands",
            f"- rp config validate --project {project_name}",
            f"- edit project/{project_name}/manifests/batches/{payload['default_batch']}.tsv",
            f"- rp run plan preprocess tabular --project {project_name} --run-id {project_name}-preprocess",
        ]
    )
    return lines


def _write_tabular_batch_manifest_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("feature_table\ttarget_column\n", encoding="utf-8")


def _render_bids_submit_summary(
    *,
    context: dict[str, Any],
    run_id: str,
    stage_execution: dict[str, Any],
    submit_execution: dict[str, Any],
    discovery_report: dict[str, Any] | None,
) -> list[str]:
    stage_result = "staged" if stage_execution.get("ok") else "stage-failed"
    submit_result = "submitted" if submit_execution.get("ok") else "submit-failed"
    lines = [
        "BIDS submit summary",
        f"Project: {context['project']['name']}",
        f"Tool: {context['preprocessing']['tool']}",
        f"Adapter: {context['preprocessing']['tool_adapter']}",
        f"Batch: {to_workspace_relative(Path(context['batch']['path']), context['workspace_root'])}",
    ]
    if discovery_report is not None:
        lines.append(f"Discovered rows: {discovery_report['row_count']}")
    lines.extend(
        [
            f"Run id: {run_id}",
            f"Stage result: {stage_result}",
            f"Submit result: {submit_result}",
            f"Submit job id: {submit_execution.get('job_id') or '(none)'}",
        ]
    )
    return lines


def _render_analysis_bids_submit_summary(
    *,
    context: dict[str, Any],
    run_id: str,
    stage_execution: dict[str, Any],
    submit_execution: dict[str, Any],
    discovery_report: dict[str, Any] | None,
) -> list[str]:
    stage_result = "staged" if stage_execution.get("ok") else "stage-failed"
    submit_result = "submitted" if submit_execution.get("ok") else "submit-failed"
    lines = [
        "BIDS analysis submit summary",
        f"Project: {context['project']['name']}",
        f"Stage: {context['analysis_stage_name']}",
        f"Tool: {context['analysis_tool_name']}",
        f"Adapter: {context['analysis_tool']['adapter']}",
        f"Batch: {to_workspace_relative(Path(context['batch']['path']), context['workspace_root'])}",
    ]
    if discovery_report is not None:
        lines.append(f"Discovered rows: {discovery_report['row_count']}")
    lines.extend(
        [
            f"Run id: {run_id}",
            f"Stage result: {stage_result}",
            f"Submit result: {submit_result}",
            f"Submit job id: {submit_execution.get('job_id') or '(none)'}",
        ]
    )
    return lines


def _write_batch_manifest(
    path: Path,
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] | list[str] = (),
) -> None:
    from .manifests import write_manifest_table

    write_manifest_table(
        path,
        rows,
        columns=columns,
        preferred_columns=("subject_id", "session_id", "task_id", "run_id"),
    )


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        normalized = _normalize_optional_cli_value(value)
        if normalized is not None:
            return normalized
    return None


def _read_tsv(path: Path) -> list[dict[str, str]]:
    from .manifests import read_manifest_table

    return [
        {key: resolve_env_value(value) or "" for key, value in row.items()}
        for row in read_manifest_table(path).rows
    ]


def _default_run_id(mode: str, action: str, target: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{mode}-{action}-{target}"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _notebook_submit_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
