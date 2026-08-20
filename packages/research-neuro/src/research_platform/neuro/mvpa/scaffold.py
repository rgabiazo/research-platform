"""Scaffold starter MVPA config files without running MVPA.

The helpers here create project-overlay configuration stubs only. They do not
inspect datasets, discover FEATs, read masks, compute distances, export RDMs,
or publish derivatives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from io import StringIO
import re
from typing import Any


TEMPLATE_MATERIALIZED_CROSSNOBIS = "materialized-crossnobis"
TEMPLATE_FSL_FEAT_CROSSNOBIS = "fsl-feat-crossnobis"
TEMPLATE_LEGACY_DISTANCE_RDM = "distance-rdm"
SUPPORTED_TEMPLATES = (
    TEMPLATE_MATERIALIZED_CROSSNOBIS,
    TEMPLATE_FSL_FEAT_CROSSNOBIS,
    TEMPLATE_LEGACY_DISTANCE_RDM,
)
SUPPORTED_COMPONENTS = ("specs", "runtime", "tables", "figures", "rdms", "derivatives")
SUPPORTED_COMPARISON_MODES = ("explicit", "complete")
CONDITIONS_COLUMNS = ("condition_id", "condition_label", "condition_description", "source_selector", "notes")
COMPARISONS_COLUMNS = (
    "comparison_id",
    "condition_a",
    "condition_b",
    "comparison_label",
    "comparison_description",
    "comparison_mode",
    "notes",
)
ROIS_COLUMNS = ("roi_id", "roi_label", "roi_source", "space", "mask_selector", "notes")

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_BIDS_LABEL = re.compile(r"^[A-Za-z0-9]+$")
_COMPONENT_ALIASES = {
    "derivative-publisher": "derivatives",
    "derivative_publisher": "derivatives",
    "publish-derivatives": "derivatives",
    "publish_derivatives": "derivatives",
}


def build_mvpa_config_scaffold(
    *,
    analysis_id: str,
    analysis_label: str | None = None,
    task: str | None = None,
    session: str | None = None,
    direction: str | None = None,
    template: str = TEMPLATE_MATERIALIZED_CROSSNOBIS,
    metric: str = "crossnobis",
    comparison_mode: str = "explicit",
    components: Sequence[str] | str | None = None,
    condition_specs: Sequence[str] | None = None,
    comparison_specs: Sequence[str] | None = None,
    roi_specs: Sequence[str] | None = None,
    analysis_variant: str = "main",
    phase_id: str = "analysis",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    normalized_analysis_id = _safe_required(analysis_id, "analysis_id", errors) or "mvpa_analysis"
    normalized_label = _analysis_label(analysis_label, normalized_analysis_id, errors)
    normalized_template = template or TEMPLATE_MATERIALIZED_CROSSNOBIS
    if normalized_template not in SUPPORTED_TEMPLATES:
        errors.append(f"template must be one of: {', '.join(SUPPORTED_TEMPLATES)}.")
    normalized_metric = _safe_required(metric, "metric", errors) or "crossnobis"
    normalized_mode = comparison_mode or "explicit"
    if normalized_mode not in SUPPORTED_COMPARISON_MODES:
        errors.append(f"comparison_mode must be one of: {', '.join(SUPPORTED_COMPARISON_MODES)}.")
    default_components: Sequence[str] | str | None = (
        "runtime" if normalized_template != TEMPLATE_LEGACY_DISTANCE_RDM else None
    )
    selected_components = normalize_components(
        components if components is not None else default_components,
        errors=errors,
    )
    normalized_task = _optional_bids_label(task, "task", errors) or "TODOtask"
    normalized_session = _normalize_session(session, errors=errors) or "ses-TODO"
    normalized_direction = _optional_bids_label(direction, "direction", errors)
    normalized_variant = _safe_required(analysis_variant, "analysis_variant", errors) or "main"
    normalized_phase = _safe_required(phase_id, "phase_id", errors) or "analysis"

    conditions = _condition_rows(condition_specs, errors=errors)
    comparisons = _comparison_rows(comparison_specs, conditions=conditions, mode=normalized_mode, errors=errors)
    rois = _roi_rows(roi_specs, errors=errors)
    if normalized_mode == "complete" and len(conditions) < 2:
        errors.append("complete comparison mode requires at least two conditions.")
    if normalized_mode == "explicit" and not comparisons:
        warnings.append("No explicit comparisons were provided; placeholder comparisons were generated.")

    runtime_id = (
        f"{normalized_analysis_id}_{_template_suffix(normalized_template)}"
        if normalized_template == TEMPLATE_LEGACY_DISTANCE_RDM
        else normalized_analysis_id
    )
    filename_prefix = f"{normalized_session}_task-{normalized_task}"
    desc_prefix = f"task-{normalized_task}_desc-{normalized_label}"
    files: list[dict[str, Any]] = []

    if "specs" in selected_components:
        files.extend(
            [
                _file_record(
                    component="specs",
                    kind="tsv",
                    relative_path=f"config/analysis/mvpa_specs/{desc_prefix}_conditions.tsv",
                    content=_render_tsv(CONDITIONS_COLUMNS, conditions),
                ),
                _file_record(
                    component="specs",
                    kind="tsv",
                    relative_path=f"config/analysis/mvpa_specs/{desc_prefix}_comparisons.tsv",
                    content=_render_tsv(COMPARISONS_COLUMNS, comparisons),
                ),
                _file_record(
                    component="specs",
                    kind="tsv",
                    relative_path=f"config/analysis/mvpa_specs/{desc_prefix}_rois.tsv",
                    content=_render_tsv(ROIS_COLUMNS, rois),
                ),
            ]
        )
    if "runtime" in selected_components:
        files.append(
            _file_record(
                component="runtime",
                kind="yaml",
                relative_path=f"config/analysis/mvpa/{runtime_id}.yaml",
                document=(
                    _materialized_runtime_document(
                        runtime_id=runtime_id,
                        analysis_label=normalized_label,
                        metric=normalized_metric,
                        conditions=conditions,
                        comparisons=comparisons,
                    )
                    if normalized_template == TEMPLATE_MATERIALIZED_CROSSNOBIS
                    else _fsl_feat_runtime_document(
                        runtime_id=runtime_id,
                        analysis_label=normalized_label,
                        task=normalized_task,
                        session=normalized_session,
                        direction=normalized_direction,
                        metric=normalized_metric,
                        conditions=conditions,
                        comparisons=comparisons,
                    )
                    if normalized_template == TEMPLATE_FSL_FEAT_CROSSNOBIS
                    else _runtime_document(
                        runtime_id=runtime_id,
                        analysis_label=normalized_label,
                        task=normalized_task,
                        session=normalized_session,
                        direction=normalized_direction,
                        metric=normalized_metric,
                        comparison_mode=normalized_mode,
                        conditions=conditions,
                        comparisons=comparisons,
                    )
                ),
            )
        )
    if "tables" in selected_components:
        files.append(
            _file_record(
                component="tables",
                kind="yaml",
                relative_path=f"config/analysis/mvpa_tables/{normalized_analysis_id}.yaml",
                document=_table_document(
                    analysis_id=normalized_analysis_id,
                    runtime_id=runtime_id,
                    task=normalized_task,
                    session=normalized_session,
                    phase_id=normalized_phase,
                    analysis_variant=normalized_variant,
                    filename_prefix=filename_prefix,
                ),
            )
        )
    if "figures" in selected_components:
        files.append(
            _file_record(
                component="figures",
                kind="yaml",
                relative_path=f"config/analysis/mvpa_figures/{normalized_analysis_id}.yaml",
                document=_figure_document(
                    analysis_id=normalized_analysis_id,
                    analysis_label=normalized_label,
                    task=normalized_task,
                    session=normalized_session,
                    phase_id=normalized_phase,
                    analysis_variant=normalized_variant,
                    filename_prefix=filename_prefix,
                ),
            )
        )
    if "rdms" in selected_components:
        files.append(
            _file_record(
                component="rdms",
                kind="yaml",
                relative_path=f"config/analysis/mvpa_rdms/{normalized_analysis_id}.yaml",
                document=_rdm_document(
                    analysis_id=normalized_analysis_id,
                    analysis_label=normalized_label,
                    task=normalized_task,
                    session=normalized_session,
                    phase_id=normalized_phase,
                    analysis_variant=normalized_variant,
                    filename_prefix=filename_prefix,
                    comparison_mode=normalized_mode,
                    conditions=conditions,
                    comparisons=comparisons,
                ),
            )
        )
    if "derivatives" in selected_components:
        files.append(
            _file_record(
                component="derivatives",
                kind="yaml",
                relative_path=f"config/analysis/mvpa_derivatives/{normalized_analysis_id}.yaml",
                document=_derivative_document(
                    analysis_id=normalized_analysis_id,
                    analysis_label=normalized_label,
                    task=normalized_task,
                    session=normalized_session,
                    direction=normalized_direction,
                    conditions=conditions,
                    comparisons=comparisons,
                    rois=rois,
                ),
            )
        )

    dependencies = _component_dependencies(selected_components)
    warnings.extend(_todo_warnings(selected_components, template=normalized_template))
    return {
        "valid": not errors,
        "analysis_id": normalized_analysis_id,
        "analysis_label": normalized_label,
        "template": normalized_template,
        "metric": normalized_metric,
        "comparison_mode": normalized_mode,
        "components": list(selected_components),
        "files": files,
        "dependencies": dependencies,
        "conditions": conditions,
        "condition_comparisons": comparisons,
        "rois": rois,
        "warnings": warnings,
        "errors": errors,
    }


def normalize_components(components: Sequence[str] | str | None, *, errors: list[str] | None = None) -> tuple[str, ...]:
    if components is None:
        raw = ("specs", "runtime", "tables", "rdms", "derivatives")
    elif isinstance(components, str):
        raw = tuple(part.strip() for part in components.split(",") if part.strip())
    else:
        values: list[str] = []
        for component in components:
            values.extend(part.strip() for part in str(component).split(",") if part.strip())
        raw = tuple(values)
    selected: list[str] = []
    for component in raw:
        normalized = _COMPONENT_ALIASES.get(component, component)
        if normalized not in SUPPORTED_COMPONENTS:
            if errors is not None:
                errors.append(f"component {component!r} must be one of: {', '.join(SUPPORTED_COMPONENTS)}.")
            continue
        if normalized not in selected:
            selected.append(normalized)
    return tuple(selected)


def _materialized_runtime_document(
    *,
    runtime_id: str,
    analysis_label: str,
    metric: str,
    conditions: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "mvpa_set": {
            "name": runtime_id,
            "description": f"Prepared-vector MVPA scaffold for {analysis_label}.",
            "scaffold_status": "not_ready",
            "scaffold_notes": [
                "Reference this MVPA set from one analysis bundle with exact ordered units.",
                "Configure the mvpa_inputs named root and replace patterns.tsv with the prepared-vector table.",
                "Review condition, ROI identity, centering, noise, threshold, and runtime-root choices before execution.",
            ],
            "unit_selection": {
                "mode": "exact_units",
                "key_columns": ["subject_id", "run_id"],
            },
            "conditions": [
                {
                    "id": row["condition_id"],
                    "description": row["condition_description"],
                }
                for row in conditions
            ],
            "condition_pairs": [
                {
                    "id": row["comparison_id"],
                    "condition_a": row["condition_a"],
                    "condition_b": row["condition_b"],
                }
                for row in comparisons
            ],
            "pattern_sources": [
                {
                    "name": "prepared-patterns",
                    "backend": "materialized_pattern_table",
                    "root_ref": "mvpa_inputs",
                    "path": "patterns.tsv",
                    "schema_version": "research_platform.neuro.mvpa.materialized_pattern_table.v1",
                }
            ],
            "roi_sources": [
                {
                    "name": "prepared-rois",
                    "source": "materialized_features",
                    "roi_labels": ["SeedA"],
                    "feature_space_id": "example-feature-space",
                    "roi_definition_id": "example-roi-definition",
                }
            ],
            "event_thresholds": {
                "min_events_per_condition_per_run": 1,
                "min_runs_per_condition": 2,
            },
            "mean_centering": {"enabled": False, "scope": "none"},
            "distance": {
                "metrics": [metric],
                "engine": "native_reference",
                "cross_validation": {"unit": "run"},
                "noise_normalization": {
                    "method": "identity",
                    "nonpositive_policy": "strict",
                    "min_retained_features": 1,
                },
            },
            "outputs": {
                "runtime_root": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/{mvpa_set}",
                },
            },
            "runtime": {"existing_output": "fail"},
            "missing_input_policy": "fail",
        }
    }


def _fsl_feat_runtime_document(
    *,
    runtime_id: str,
    analysis_label: str,
    task: str,
    session: str,
    direction: str | None,
    metric: str,
    conditions: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    entities: dict[str, str] = {"task": task}
    if direction:
        entities["direction"] = direction
    return {
        "mvpa_set": {
            "name": runtime_id,
            "description": f"Advanced FSL FEAT image-input MVPA scaffold for {analysis_label}.",
            "scaffold_status": "not_ready",
            "scaffold_notes": [
                "Compatibility mode: replace the selector, FEAT root, image templates, and ROI mask before execution.",
                "FSL FEAT inputs and optional neuroimaging dependencies are external prerequisites.",
            ],
            "unit_selection": {"mode": "legacy_cartesian"},
            "subjects": ["sub-TODO"],
            "sessions": [session],
            "runs": ["run-TODO"],
            "entities": entities,
            "conditions": [
                {"id": row["condition_id"], "aliases": [row["condition_id"]]}
                for row in conditions
            ],
            "condition_pairs": [
                {
                    "id": row["comparison_id"],
                    "condition_a": row["condition_a"],
                    "condition_b": row["condition_b"],
                }
                for row in comparisons
            ],
            "pattern_sources": [
                {
                    "name": "feat-patterns",
                    "backend": "fsl_feat_pe",
                    "root_ref": "feat_inputs",
                    "feat_dir_template": "{subject_id}/{session_id}/run-{run_id}.feat",
                    "design_file": "design.fsf",
                    "pe_image_template": "stats/pe{pe_number}.nii.gz",
                }
            ],
            "roi_sources": [
                {
                    "name": "image-rois",
                    "source": "explicit_masks",
                    "root_ref": "roi_inputs",
                    "path": "SeedA_mask.nii.gz",
                }
            ],
            "mean_centering": {"enabled": False, "scope": "none"},
            "distance": {
                "metrics": [metric],
                "engine": "native_reference",
                "cross_validation": {"unit": "run"},
                "noise_normalization": {"method": "identity"},
            },
            "outputs": {
                "runtime_root": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/{mvpa_set}",
                },
            },
            "runtime": {"existing_output": "fail"},
            "missing_input_policy": "fail",
        }
    }


def _runtime_document(
    *,
    runtime_id: str,
    analysis_label: str,
    task: str,
    session: str,
    direction: str | None,
    metric: str,
    comparison_mode: str,
    conditions: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    entities: dict[str, str] = {"task": task}
    if direction:
        entities["direction"] = direction
    condition_ids = [row["condition_id"] for row in conditions]
    condition_pairs: Any
    if comparison_mode == "complete":
        condition_pairs = {
            "mode": "all_pairs",
            "conditions": condition_ids,
            "id_template": "{condition_a}_minus_{condition_b}",
            "notes": "Generated from scaffold condition_comparisons.mode=complete.",
        }
    else:
        condition_pairs = [
            {
                "id": row["comparison_id"],
                "condition_a": row["condition_a"],
                "condition_b": row["condition_b"],
                "notes": "comparison_id is used as the runtime/export contrast_id.",
            }
            for row in comparisons
        ]
    return {
        "mvpa_set": {
            "name": runtime_id,
            "description": f"TODO starter MVPA distance workflow for {analysis_label}.",
            "scaffold_status": "not_ready",
            "scaffold_notes": [
                "Replace TODO selectors, pattern sources, ROI masks, and thresholds before execution.",
                "Spec TSV comparison_id values map to runtime condition_pairs.id and exported contrast_id values.",
            ],
            "subjects": ["sub-TODO"],
            "sessions": [session],
            "runs": ["run-TODO"],
            "entities": entities,
            "conditions": [
                {
                    "id": row["condition_id"],
                    "selector": {"source_selector": row["source_selector"] or f"TODO_{row['condition_id']}"},
                    "description": row["condition_description"],
                }
                for row in conditions
            ],
            "condition_comparisons": {
                "mode": comparison_mode,
                "source": "config/analysis/mvpa_specs/*_comparisons.tsv",
                "notes": "Scaffold-facing generic spec. Runtime compatibility is represented in condition_pairs.",
            },
            "condition_pairs": condition_pairs,
            "pattern_sources": [
                {
                    "name": "todo_patterns",
                    "backend": "bids_derivative_pattern_table",
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/TODO/prepared-patterns.tsv",
                    "notes": "TODO replace with a real compatible pattern source.",
                }
            ],
            "roi_sources": [
                {
                    "name": "todo_rois",
                    "source": "explicit_masks",
                    "root_ref": "artifact_root",
                    "masks": [
                        {
                            "label": "ExampleROI",
                            "path": ".research-platform/mvpa/TODO/rois/ExampleROI_mask.nii.gz",
                        }
                    ],
                    "notes": "TODO replace with real ROI masks or an ROI set reference.",
                }
            ],
            "distance": {
                "metrics": [metric],
                "engine": "manual_diagonal_crossnobis_v1" if metric == "crossnobis" else "native_reference",
                "cross_validation": {"unit": "run"},
                "noise_normalization": {"method": "diagonal" if metric == "crossnobis" else "identity"},
            },
            "outputs": {
                "runtime_root": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/{mvpa_set}"},
            },
            "missing_input_policy": "warn",
        }
    }


def _table_document(
    *,
    analysis_id: str,
    runtime_id: str,
    task: str,
    session: str,
    phase_id: str,
    analysis_variant: str,
    filename_prefix: str,
) -> dict[str, Any]:
    return {
        "mvpa_table_export": {
            "name": analysis_id,
            "description": "TODO subject-level distance table export scaffold.",
            "entities": {"session_id": session, "task_id": task},
            "sources": [
                {
                    "mvpa_set": runtime_id,
                    "phase_id": phase_id,
                    "analysis_variant": analysis_variant,
                    "family_id": "scaffold",
                    "notes": "TODO update expected row counts after runtime execution.",
                }
            ],
            "outputs": {
                "root_ref": "artifact_root",
                "path": ".research-platform/mvpa/reports/{table_set}",
                "filename_prefix": filename_prefix,
                "include_absolute_source_paths": False,
            },
        }
    }


def _figure_document(
    *,
    analysis_id: str,
    analysis_label: str,
    task: str,
    session: str,
    phase_id: str,
    analysis_variant: str,
    filename_prefix: str,
) -> dict[str, Any]:
    return {
        "mvpa_figure_export": {
            "name": analysis_id,
            "description": "TODO publication figure scaffold from exported subject-level distances.",
            "input": {
                "table_set": analysis_id,
                "root_ref": "artifact_root",
                "path": f".research-platform/mvpa/reports/{{table_set}}/{filename_prefix}_desc-SubjectLevelCrossnobisDistances_mvpa.tsv",
            },
            "outputs": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/reports/{figure_set}/figures"},
            "figures": [
                {
                    "figure_id": f"{analysis_id}_roiwise",
                    "kind": "strip_mean_ci",
                    "filters": {"analysis_variant": analysis_variant, "phase_id": phase_id},
                    "x": "roi_label",
                    "y": "crossnobis",
                    "title": f"{analysis_label} distances by ROI",
                    "ylabel": "Distance",
                    "xlabel": "ROI",
                    "zero_line": True,
                    "mean_ci": True,
                    "output_basename": f"{filename_prefix}_desc-{analysis_label}RoiwiseDistances_mvpa",
                    "output_formats": ["svg", "pdf", "png"],
                }
            ],
        }
    }


def _rdm_document(
    *,
    analysis_id: str,
    analysis_label: str,
    task: str,
    session: str,
    phase_id: str,
    analysis_variant: str,
    filename_prefix: str,
    comparison_mode: str,
    conditions: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "mvpa_rdm_export": {
            "name": analysis_id,
            "description": "TODO RDM export scaffold from exported subject-level distances.",
            "entities": {"session_id": session, "task_id": task},
            "input": {
                "table_set": analysis_id,
                "root_ref": "artifact_root",
                "path": f".research-platform/mvpa/reports/{{table_set}}/{filename_prefix}_desc-SubjectLevelCrossnobisDistances_mvpa.tsv",
            },
            "outputs": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/reports/{rdm_set}/rdms"},
            "rdms": [
                {
                    "rdm_id": f"{analysis_id}_rdm",
                    "enabled": True,
                    "kind": "rdm_heatmap",
                    "value_column": "crossnobis",
                    "filters": {"analysis_variant": analysis_variant, "phase_id": phase_id},
                    "conditions": [
                        {"condition_id": row["condition_id"], "label": row["condition_label"]} for row in conditions
                    ],
                    "pair_mappings": [
                        {
                            "contrast_id": row["comparison_id"],
                            "condition_a": row["condition_a"],
                            "condition_b": row["condition_b"],
                        }
                        for row in comparisons
                    ],
                    "aggregate_within_participant": {"enabled": True, "across": "roi_label", "method": "mean"},
                    "group_summary": {"method": "mean", "ci_level": 0.95},
                    "strict_all_pairs": comparison_mode == "complete",
                    "diagonal_value": 0.0,
                    "symmetric": True,
                    "title": f"{analysis_label} RDM",
                    "colorbar_label": "Distance",
                    "output_basename": f"{filename_prefix}_desc-{analysis_label}RDM_mvpa",
                    "output_formats": ["svg", "pdf", "png"],
                }
            ],
        }
    }


def _derivative_document(
    *,
    analysis_id: str,
    analysis_label: str,
    task: str,
    session: str,
    direction: str | None,
    conditions: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
    rois: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    filename_prefix = f"{session}_task-{task}"
    entities: dict[str, str] = {"session_id": session, "task_id": task}
    if direction:
        entities["direction"] = direction
    return {
        "mvpa_derivative_publish": {
            "name": analysis_id,
            "description": "TODO derivative publisher scaffold for verified MVPA artifacts.",
            "analysis_label": analysis_label,
            "derivative_name": "mvpa-crossnobis",
            "entities": entities,
            "inputs": {
                "table_sets": [
                    {
                        "table_set": analysis_id,
                        "distances": f".research-platform/mvpa/reports/{analysis_id}/{filename_prefix}_desc-SubjectLevelCrossnobisDistances_mvpa.tsv",
                        "audit": f".research-platform/mvpa/reports/{analysis_id}/{filename_prefix}_desc-SubjectLevelCrossnobisAudit_mvpa.tsv",
                        "manifest": f".research-platform/mvpa/reports/{analysis_id}/{filename_prefix}_desc-CrossnobisTables_manifest.json",
                    }
                ],
                "rdm_set": {
                    "rdm_set": analysis_id,
                    "root": f".research-platform/mvpa/reports/{analysis_id}/rdms",
                    "rdms": [
                        {
                            "rdm_id": f"{analysis_id}_rdm",
                            "basename": f"{filename_prefix}_desc-{analysis_label}RDM_mvpa",
                            "publish_desc": f"{analysis_label}RDM",
                            "conditions": [
                                {"condition_id": row["condition_id"], "label": row["condition_label"]}
                                for row in conditions
                            ],
                        }
                    ],
                },
            },
            "targets": {
                "local_artifact": {
                    "root_ref": "artifact_root",
                    "relative_path": ".research-platform/mvpa/derivatives/mvpa-crossnobis",
                    "default": True,
                },
                "dataset_derivatives": {
                    "root_ref": "dataset_derivatives_root",
                    "relative_path": "mvpa-crossnobis",
                    "default": False,
                },
            },
            "conditions": [
                {
                    "condition_id": row["condition_id"],
                    "label": row["condition_label"],
                    "description": row["condition_description"],
                }
                for row in conditions
            ],
            "contrasts": [
                {
                    "contrast_id": row["comparison_id"],
                    "condition_a": row["condition_a"],
                    "condition_b": row["condition_b"],
                    "comparison_label": row["comparison_label"],
                }
                for row in comparisons
            ],
            "rois": [{"roi_label": row["roi_label"], "roi_source": row["roi_source"]} for row in rois],
        }
    }


def _condition_rows(specs: Sequence[str] | None, *, errors: list[str]) -> list[dict[str, str]]:
    raw_specs = list(specs or ("condition_a:Condition A", "condition_b:Condition B"))
    rows: list[dict[str, str]] = []
    for index, spec in enumerate(raw_specs, start=1):
        parts = _split_spec(spec, expected=4)
        condition_id = parts[0] or f"condition_{index}"
        if not _SAFE_LABEL.fullmatch(condition_id):
            errors.append(f"condition {index} id must be a safe label.")
        label = parts[1] or _title_from_id(condition_id)
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": label,
                "condition_description": parts[2] or "TODO describe this condition.",
                "source_selector": parts[3] or f"TODO_{condition_id}",
                "notes": "Edit before execution.",
            }
        )
    for duplicate in _duplicates(row["condition_id"] for row in rows):
        errors.append(f"conditions contain duplicate condition_id: {duplicate}.")
    return rows


def _comparison_rows(
    specs: Sequence[str] | None,
    *,
    conditions: Sequence[Mapping[str, str]],
    mode: str,
    errors: list[str],
) -> list[dict[str, str]]:
    condition_ids = [row["condition_id"] for row in conditions]
    if specs:
        rows = [_comparison_row_from_spec(spec, index=index, mode=mode, condition_ids=condition_ids, errors=errors) for index, spec in enumerate(specs, start=1)]
    elif mode == "complete":
        rows = [
            _comparison_row(left, right, f"{left}_minus_{right}", mode=mode)
            for left_index, left in enumerate(condition_ids)
            for right in condition_ids[left_index + 1 :]
        ]
    else:
        left, right = (condition_ids + ["condition_a", "condition_b"])[:2]
        rows = [_comparison_row(left, right, f"{left}_minus_{right}", mode=mode)]
    for duplicate in _duplicates(row["comparison_id"] for row in rows):
        errors.append(f"condition comparisons contain duplicate comparison_id: {duplicate}.")
    seen_pairs: list[frozenset[str]] = []
    for row in rows:
        if row["condition_a"] not in condition_ids:
            errors.append(f"comparison {row['comparison_id']} references unknown condition_a: {row['condition_a']}.")
        if row["condition_b"] not in condition_ids:
            errors.append(f"comparison {row['comparison_id']} references unknown condition_b: {row['condition_b']}.")
        pair_key = frozenset((row["condition_a"], row["condition_b"]))
        if row["condition_a"] == row["condition_b"]:
            errors.append(f"comparison {row['comparison_id']} must reference two distinct conditions.")
        elif pair_key in seen_pairs:
            errors.append(f"condition comparisons contain duplicate unordered comparison: {row['condition_a']}/{row['condition_b']}.")
        seen_pairs.append(pair_key)
    return rows


def _comparison_row_from_spec(
    spec: str,
    *,
    index: int,
    mode: str,
    condition_ids: Sequence[str],
    errors: list[str],
) -> dict[str, str]:
    parts = _split_spec(spec, expected=5)
    comparison_id = parts[0] or f"comparison_{index}"
    if not _SAFE_LABEL.fullmatch(comparison_id):
        errors.append(f"comparison {index} id must be a safe label.")
    condition_a = parts[1] or (condition_ids[0] if condition_ids else "condition_a")
    condition_b = parts[2] or (condition_ids[1] if len(condition_ids) > 1 else "condition_b")
    return _comparison_row(
        condition_a,
        condition_b,
        comparison_id,
        label=parts[3],
        description=parts[4],
        mode=mode,
    )


def _comparison_row(
    condition_a: str,
    condition_b: str,
    comparison_id: str,
    *,
    label: str = "",
    description: str = "",
    mode: str,
) -> dict[str, str]:
    return {
        "comparison_id": comparison_id,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "comparison_label": label or f"{_title_from_id(condition_a)} vs {_title_from_id(condition_b)}",
        "comparison_description": description or "TODO describe this distance comparison.",
        "comparison_mode": mode,
        "notes": "comparison_id maps to runtime condition_pairs.id and exported contrast_id.",
    }


def _roi_rows(specs: Sequence[str] | None, *, errors: list[str]) -> list[dict[str, str]]:
    raw_specs = list(specs or ("roi_1:ExampleROI:TODO_roi_source",))
    rows: list[dict[str, str]] = []
    for index, spec in enumerate(raw_specs, start=1):
        parts = _split_spec(spec, expected=5)
        roi_id = parts[0] or f"roi_{index}"
        if not _SAFE_LABEL.fullmatch(roi_id):
            errors.append(f"roi {index} id must be a safe label.")
        label = parts[1] or _title_from_id(roi_id)
        rows.append(
            {
                "roi_id": roi_id,
                "roi_label": label,
                "roi_source": parts[2] or "TODO_roi_source",
                "space": parts[3] or "TODOspace",
                "mask_selector": parts[4] or "TODO_mask_selector",
                "notes": "Edit before execution.",
            }
        )
    for duplicate in _duplicates(row["roi_id"] for row in rows):
        errors.append(f"rois contain duplicate roi_id: {duplicate}.")
    return rows


def _file_record(
    *,
    component: str,
    kind: str,
    relative_path: str,
    content: str | None = None,
    document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "component": component,
        "kind": kind,
        "relative_path": relative_path,
        "content": content,
        "document": dict(document) if document is not None else None,
    }


def _render_tsv(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def _component_dependencies(components: Sequence[str]) -> list[dict[str, str]]:
    dependencies: list[dict[str, str]] = []
    if "rdms" in components and "tables" not in components:
        dependencies.append({"component": "rdms", "depends_on": "tables", "note": "RDM exports need a subject-level distance table."})
    if "figures" in components and "tables" not in components:
        dependencies.append({"component": "figures", "depends_on": "tables", "note": "Figure exports need a subject-level distance table."})
    if "derivatives" in components:
        dependencies.append(
            {
                "component": "derivatives",
                "depends_on": "tables,rdms",
                "note": "Derivative publishing needs verified table and RDM artifacts.",
            }
        )
    return dependencies


def _todo_warnings(components: Sequence[str], *, template: str) -> list[str]:
    warnings = ["Generated files are starter scaffolds and are not execution-ready until inputs and identities are reviewed."]
    if "runtime" in components:
        if template == TEMPLATE_MATERIALIZED_CROSSNOBIS:
            warnings.append(
                "Configure a matching analysis bundle, mvpa_inputs root, and materialized v1 pattern table."
            )
        elif template == TEMPLATE_FSL_FEAT_CROSSNOBIS:
            warnings.append("FSL FEAT image inputs and ROI masks are advanced external prerequisites.")
        else:
            warnings.append("Compatibility scaffold uses placeholder pattern and ROI sources.")
    if "derivatives" in components:
        warnings.append("Derivative publisher scaffold expects verified table/RDM artifacts to exist later.")
    return warnings


def _safe_required(value: str | None, label: str, errors: list[str]) -> str | None:
    text = _clean(value)
    if text is None:
        errors.append(f"{label} must be defined.")
        return None
    if not _SAFE_LABEL.fullmatch(text):
        errors.append(f"{label} must be a safe label.")
    return text


def _analysis_label(value: str | None, analysis_id: str, errors: list[str]) -> str:
    text = _clean(value) or "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", analysis_id) if part)
    if not text:
        text = "MvpaAnalysis"
    if not _BIDS_LABEL.fullmatch(text):
        errors.append("analysis_label must contain only letters and digits for BIDS desc fields.")
    return text


def _optional_bids_label(value: str | None, label: str, errors: list[str]) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    if not _BIDS_LABEL.fullmatch(text):
        errors.append(f"{label} must contain only letters and digits.")
    return text


def _normalize_session(value: str | None, *, errors: list[str]) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    raw = text[4:] if text.startswith("ses-") else text
    if not _BIDS_LABEL.fullmatch(raw):
        errors.append("session must be a BIDS label such as 01 or ses-01.")
    return f"ses-{raw}"


def _template_suffix(template: str) -> str:
    return template.replace("-", "_")


def _split_spec(spec: str, *, expected: int) -> list[str]:
    parts = str(spec).split(":", expected - 1)
    return [*parts, *([""] * (expected - len(parts)))]


def _title_from_id(value: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[_ .-]+", value) if part) or value


def _duplicates(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "COMPARISONS_COLUMNS",
    "CONDITIONS_COLUMNS",
    "ROIS_COLUMNS",
    "SUPPORTED_COMPONENTS",
    "SUPPORTED_COMPARISON_MODES",
    "SUPPORTED_TEMPLATES",
    "TEMPLATE_FSL_FEAT_CROSSNOBIS",
    "TEMPLATE_LEGACY_DISTANCE_RDM",
    "TEMPLATE_MATERIALIZED_CROSSNOBIS",
    "build_mvpa_config_scaffold",
    "normalize_components",
]
