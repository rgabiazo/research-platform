from __future__ import annotations

from collections.abc import Mapping, Sequence
from bisect import bisect_left, bisect_right
import csv
import io
import itertools
import json
import math
from pathlib import Path
import random
import re
import tempfile
from typing import Any

from ._path_safety import configured_path_is_unsafe, published_text_contains_local_path_reference


DEFAULT_OUTPUT_RELATIVE_PATH = ".research-platform/mvpa/reports/{publication_set}/publication"
OUTPUT_FORMATS = ("png", "pdf", "svg")
ROI_STATS_COLUMNS = (
    "analysis_variant",
    "family_id",
    "phase_id",
    "roi_label",
    "roi_display_label",
    "contrast_id",
    "contrast_display_label",
    "aggregation_level",
    "within_participant_aggregation",
    "aggregation_across",
    "N",
    "mean_crossnobis",
    "SD",
    "variance",
    "SEM",
    "ci_low",
    "ci_high",
    "median",
    "q1",
    "q3",
    "iqr",
    "min",
    "max",
    "percent_positive",
    "t",
    "df",
    "p_two",
    "p_one_greater",
    "p_signflip",
    "p_wilcoxon",
    "q_FDR",
    "dz",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "feature_count_min",
    "feature_count_max",
    "cv_unit_count_min",
    "cv_unit_count_max",
    "observation_count_min",
    "observation_count_max",
    "mean_feature_count",
    "mean_cv_unit_count",
    "mean_observation_count",
    "fdr_family",
    "signflip_method",
    "ci_method",
    "bootstrap_method",
    "source_table_relpath",
)
PHASE_STATS_COLUMNS = (
    "analysis_variant",
    "family_id",
    "phase_id",
    "phase_display_label",
    "contrast_id",
    "contrast_display_label",
    "roi_set_id",
    "roi_family",
    "roi_set_display_label",
    "summary_level",
    "aggregation_level",
    "within_participant_aggregation",
    "aggregation_across",
    "N",
    "mean_crossnobis",
    "SD",
    "variance",
    "SEM",
    "ci_low",
    "ci_high",
    "median",
    "q1",
    "q3",
    "iqr",
    "min",
    "max",
    "percent_positive",
    "t",
    "df",
    "p_two",
    "p_one_greater",
    "p_signflip",
    "p_wilcoxon",
    "q_FDR",
    "dz",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "mean_roi_count_per_participant",
    "mean_feature_count",
    "mean_cv_unit_count",
    "mean_observation_count",
    "mean_valid_cv_units",
    "fdr_family",
    "signflip_method",
    "ci_method",
    "bootstrap_method",
    "source_table_relpath",
)
PHASE_SUBJECT_COLUMNS = (
    "participant_id",
    "analysis_variant",
    "family_id",
    "phase_id",
    "contrast_id",
    "roi_set_id",
    "roi_family",
    "roi_count",
    "crossnobis",
    "feature_count_mean",
    "cv_unit_count_mean",
    "observation_count_mean",
    "source_rows_count",
)
RDM_STATS_COLUMNS = (
    "rdm_id",
    "aggregation_level",
    "within_participant_aggregation",
    "aggregation_across",
    "condition_a",
    "condition_b",
    "condition_a_label",
    "condition_b_label",
    "source_contrast_id",
    "N",
    "mean_crossnobis",
    "ci_low",
    "ci_high",
    "SD",
    "SEM",
)
DEFAULT_COMPACT_COLUMNS = (
    "Functional ROIs",
    "Analysis",
    "Contrast",
    "N",
    "Mean Distance",
    "95% CI",
    "Effect Size",
    "Permutation",
    "qFDR",
)
MANUSCRIPT_PHASE_SUBJECT_COLUMNS = (
    "participant_id",
    "analysis_variant",
    "family_id",
    "phase_id",
    "phase_display_label",
    "contrast_id",
    "roi_set_id",
    "roi_family",
    "crossnobis",
    "plot_value",
    "x_position",
    "jittered_x_position",
)
MANUSCRIPT_ROI_SUBJECT_COLUMNS = (
    "participant_id",
    "analysis_variant",
    "family_id",
    "phase_id",
    "roi_label",
    "roi_display_label",
    "contrast_id",
    "crossnobis",
    "plot_value",
    "x_position",
    "y_position",
    "jittered_y_position",
)
MANUSCRIPT_PHASE_STATS_COLUMNS = (
    "phase_id",
    "phase_display_label",
    "analysis_variant",
    "family_id",
    "contrast_id",
    "roi_set_id",
    "N",
    "mean_crossnobis",
    "ci_low",
    "ci_high",
    "SD",
    "SEM",
    "p_two",
    "p_one_greater",
    "p_signflip",
    "p_wilcoxon",
    "q_FDR",
    "dz",
    "x_position",
)
MANUSCRIPT_ROI_STATS_COLUMNS = (
    "roi_label",
    "roi_display_label",
    "phase_id",
    "analysis_variant",
    "family_id",
    "contrast_id",
    "N",
    "mean_crossnobis",
    "ci_low",
    "ci_high",
    "SD",
    "SEM",
    "p_two",
    "p_one_greater",
    "p_signflip",
    "p_wilcoxon",
    "q_FDR",
    "dz",
    "y_position",
)


def validate_mvpa_publication_export_document(document: Mapping[str, Any] | Any) -> list[str]:
    if not isinstance(document, Mapping):
        return ["MVPA publication export config must contain a mapping."]
    payload = _payload(document)
    errors: list[str] = []
    publication_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("publication_set"))
    if publication_set is None:
        errors.append("mvpa_publication_export.name must be defined.")
    elif not _safe_label(publication_set):
        errors.append("mvpa_publication_export.name must be a safe label.")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("mvpa_publication_export.inputs must be a mapping.")
    else:
        for key in ("subject_level_table", "audit_table"):
            _validate_relative_path_value(_mapping(inputs.get(key)).get("path"), f"mvpa_publication_export.inputs.{key}.path", errors)
            root_ref = _optional_text(_mapping(inputs.get(key)).get("root_ref"))
            if root_ref is not None and not _safe_label(root_ref):
                errors.append(f"mvpa_publication_export.inputs.{key}.root_ref must be a safe label.")
    outputs = payload.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        errors.append("mvpa_publication_export.outputs must be a mapping when defined.")
    elif isinstance(outputs, Mapping):
        path = _optional_text(outputs.get("path") or outputs.get("relative_path"))
        if path is not None and configured_path_is_unsafe(path):
            errors.append("mvpa_publication_export.outputs.path must be relative and stay under root_ref.")
        root_ref = _optional_text(outputs.get("root_ref"))
        if root_ref is not None and not _safe_label(root_ref):
            errors.append("mvpa_publication_export.outputs.root_ref must be a safe label.")
    return errors


def plan_or_execute_mvpa_publication_export(
    document: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    root_refs: Mapping[str, str | Path],
    execute: bool = False,
) -> dict[str, Any]:
    payload = _payload(document)
    config_errors = validate_mvpa_publication_export_document(document)
    publication_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("publication_set")) or "mvpa_publication"
    roots = {str(key): Path(value).expanduser().resolve() for key, value in root_refs.items()}
    output_root, output_relroot, output_errors = _output_root(payload, publication_set=publication_set, roots=roots)
    inputs = _mapping(payload.get("inputs"))
    subject_path, subject_relpath, subject_errors = _configured_path(
        _mapping(inputs.get("subject_level_table")),
        roots=roots,
        default_root_ref="artifact_root",
    )
    audit_path, audit_relpath, audit_errors = _configured_path(
        _mapping(inputs.get("audit_table")),
        roots=roots,
        default_root_ref="artifact_root",
    )
    warnings: list[str] = []
    errors: list[str] = [*config_errors, *output_errors, *subject_errors, *audit_errors]
    subject_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    if not errors:
        if not subject_path.is_file():
            errors.append(f"MVPA publication subject-level table is missing: {subject_relpath}.")
        else:
            subject_rows = _read_tsv(subject_path)
        if not audit_path.is_file():
            errors.append(f"MVPA publication audit table is missing: {audit_relpath}.")
        else:
            audit_rows = _read_tsv(audit_path)
    if not errors:
        subject_rows = _attach_audit_metadata(subject_rows, audit_rows)
    labels = _mapping(payload.get("labels"))
    entities = _entities(payload)
    prefix = _entity_prefix(**entities)
    stats_config = _mapping(payload.get("statistics"))
    table_config = _mapping(payload.get("tables"))
    figure_config = _mapping(payload.get("figures"))
    roi_result = _empty_family_result("SubjectLevelROIGroupInference", output_root, output_relroot, prefix)
    phase_result = _empty_family_result("SubjectLevelPhasePooledGroupInference", output_root, output_relroot, prefix)
    rdm_result = _empty_rdm_result(output_root, output_relroot, prefix)
    figure_results: list[dict[str, Any]] = []
    manuscript_result = _empty_manuscript_primary_result(output_root, output_relroot, prefix)

    if not errors:
        roi_result = _roi_group_inference(
            subject_rows,
            labels=labels,
            config=_mapping(table_config.get("roi_group_inference")),
            stats_config=stats_config,
            output_root=output_root,
            output_relroot=output_relroot,
            prefix=prefix,
            source_table_relpath=subject_relpath,
        )
        phase_result = _phase_pooled_group_inference(
            subject_rows,
            labels=labels,
            config=_mapping(table_config.get("phase_pooled_group_inference")),
            stats_config=stats_config,
            output_root=output_root,
            output_relroot=output_relroot,
            prefix=prefix,
            source_table_relpath=subject_relpath,
        )
        rdm_result = _rdm_group_summary(
            payload,
            roots=roots,
            labels=labels,
            output_root=output_root,
            output_relroot=output_relroot,
            prefix=prefix,
        )
        warnings.extend(rdm_result["warnings"])
        errors.extend([*roi_result["errors"], *phase_result["errors"], *rdm_result["errors"]])

    if not errors:
        figure_results = _prepare_figures(
            roi_result=roi_result,
            phase_result=phase_result,
            config=figure_config,
            labels=labels,
            output_root=output_root,
            output_relroot=output_relroot,
            prefix=prefix,
        )
        errors.extend(error for figure in figure_results for error in figure["errors"])
        warnings.extend(warning for figure in figure_results for warning in figure["warnings"])

    if not errors:
        manuscript_result = _prepare_manuscript_primary_main_figures(
            phase_result=phase_result,
            roi_result=roi_result,
            config=_mapping(figure_config.get("manuscript_primary_main") or figure_config.get("manuscript_primary_figures")),
            labels=labels,
            output_root=output_root,
            output_relroot=output_relroot,
            prefix=prefix,
        )
        errors.extend(manuscript_result["errors"])
        warnings.extend(manuscript_result["warnings"])

    output_records = [
        *roi_result["outputs"].values(),
        *phase_result["outputs"].values(),
        *rdm_result["outputs"].values(),
        *(output for figure in figure_results for output in figure["outputs"].values()),
        *(manuscript_result["companion_outputs"].values() if manuscript_result.get("enabled", True) else []),
        *(output for figure in manuscript_result["figures"] for output in figure["outputs"].values() if manuscript_result.get("enabled", True)),
    ]
    if execute and not errors:
        existing = [record["relative_path"] for record in output_records if Path(record["path"]).exists()]
        if existing:
            errors.append(f"MVPA publication export refuses to overwrite existing output(s): {', '.join(sorted(existing))}.")
    if execute and not errors:
        _write_family_outputs(roi_result)
        _write_family_outputs(phase_result)
        _write_rdm_outputs(rdm_result)
        _write_figures(figure_results)
        manuscript_result = _write_manuscript_primary_outputs(manuscript_result)
        errors.extend(manuscript_result["errors"])
        warnings.extend(manuscript_result["warnings"])
    valid = not errors
    return {
        "valid": valid,
        "executed": bool(execute and valid),
        "publication_set": publication_set,
        "output_root": {"relative_path": output_relroot, "path": output_root.as_posix()},
        "input_tables": {
            "subject_level": {"relative_path": subject_relpath, "exists": subject_path.is_file(), "row_count": len(subject_rows)},
            "audit": {"relative_path": audit_relpath, "exists": audit_path.is_file(), "row_count": len(audit_rows)},
        },
        "table_families": {
            "SubjectLevelROIGroupInference": _family_public_result(roi_result),
            "SubjectLevelPhasePooledGroupInference": _family_public_result(phase_result),
            "RDMGroupSummary": _rdm_public_result(rdm_result),
        },
        "figures": [_figure_public_result(figure) for figure in figure_results],
        "manuscript_primary_main": _manuscript_public_result(manuscript_result),
        "outputs": {record["relative_path"]: {"path": record["path"], "filename": record["filename"]} for record in output_records},
        "warnings": _unique(warnings),
        "errors": _unique(errors),
        "recomputed_mvpa_distances": False,
        "absolute_source_paths_excluded": True,
    }


def _payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = document.get("mvpa_publication_export") or document.get("mvpa_publication") or document
    return payload if isinstance(payload, Mapping) else {}


def _roi_group_inference(
    rows: Sequence[Mapping[str, str]],
    *,
    labels: Mapping[str, Any],
    config: Mapping[str, Any],
    stats_config: Mapping[str, Any],
    output_root: Path,
    output_relroot: str,
    prefix: str,
    source_table_relpath: str,
) -> dict[str, Any]:
    desc = "SubjectLevelROIGroupInference"
    enabled = bool(config.get("enabled", True))
    result = _empty_family_result(desc, output_root, output_relroot, prefix)
    if not enabled:
        result["enabled"] = False
        return result
    filtered = _filter_rows(rows, _mapping(config.get("filters")))
    groups = _group_rows(filtered, ("analysis_variant", "family_id", "phase_id", "roi_label", "contrast_id"))
    stats_rows: list[dict[str, Any]] = []
    subject_values: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        analysis_variant, family_id, phase_id, roi_label, contrast_id = key
        values = [_float(row.get("crossnobis")) for row in group_rows]
        values = [value for value in values if value is not None and math.isfinite(value)]
        stat = _group_stats(values, stats_config)
        stat.update(
            {
                "analysis_variant": analysis_variant,
                "family_id": family_id,
                "phase_id": phase_id,
                "roi_label": roi_label,
                "roi_display_label": _display_label(labels, "roi_display_labels", roi_label),
                "contrast_id": contrast_id,
                "contrast_display_label": _display_label(labels, "contrast_display_labels", contrast_id),
                "aggregation_level": "subject_roi",
                "within_participant_aggregation": "none",
                "aggregation_across": "participant",
                "feature_count_min": _min_number(group_rows, "feature_count"),
                "feature_count_max": _max_number(group_rows, "feature_count"),
                "cv_unit_count_min": _min_number(group_rows, "cv_unit_count"),
                "cv_unit_count_max": _max_number(group_rows, "cv_unit_count"),
                "observation_count_min": _min_number(group_rows, "observation_count"),
                "observation_count_max": _max_number(group_rows, "observation_count"),
                "mean_feature_count": _mean_number(group_rows, "feature_count"),
                "mean_cv_unit_count": _mean_number(group_rows, "cv_unit_count"),
                "mean_observation_count": _mean_number(group_rows, "observation_count"),
                "source_table_relpath": source_table_relpath,
            }
        )
        stats_rows.append(stat)
        for row in group_rows:
            subject_values.append({**{column: row.get(column, "") for column in ("participant_id", "analysis_variant", "family_id", "phase_id", "roi_label", "contrast_id")}, "crossnobis": row.get("crossnobis", "")})
    _apply_fdr(stats_rows, _text_sequence(config.get("fdr_by")) or ("analysis_variant", "phase_id"))
    result["stats_rows"] = _ordered_rows(stats_rows, ROI_STATS_COLUMNS)
    result["subject_rows"] = subject_values
    result["table_rows"] = _compact_table_rows(result["stats_rows"], labels=labels, config=_mapping(config.get("compact_table")))
    result["table_full_precision_rows"] = _compact_table_rows(result["stats_rows"], labels=labels, config=_mapping(config.get("compact_table")), full_precision=True)
    result["manifest"] = _family_manifest(
        desc,
        source_table_relpath=source_table_relpath,
        note="One row per ROI/contrast. The independent unit is participant.",
        rows=result["stats_rows"],
    )
    return result


def _phase_pooled_group_inference(
    rows: Sequence[Mapping[str, str]],
    *,
    labels: Mapping[str, Any],
    config: Mapping[str, Any],
    stats_config: Mapping[str, Any],
    output_root: Path,
    output_relroot: str,
    prefix: str,
    source_table_relpath: str,
) -> dict[str, Any]:
    desc = "SubjectLevelPhasePooledGroupInference"
    enabled = bool(config.get("enabled", True))
    result = _empty_family_result(desc, output_root, output_relroot, prefix)
    if not enabled:
        result["enabled"] = False
        return result
    subject_values: list[dict[str, Any]] = []
    group_defs = config.get("groups")
    if not isinstance(group_defs, Sequence) or isinstance(group_defs, (str, bytes, bytearray)) or not group_defs:
        phases = sorted({row.get("phase_id", "") for row in rows if row.get("phase_id")})
        group_defs = [{"roi_set_id": phase, "phase_id": phase, "roi_labels": "*"} for phase in phases]
    for group_def in group_defs:
        if not isinstance(group_def, Mapping):
            continue
        roi_set_id = str(group_def.get("roi_set_id") or group_def.get("roi_family") or group_def.get("phase_id") or "roi_set")
        phase_id = str(group_def.get("phase_id") or "")
        roi_family = str(group_def.get("roi_family") or roi_set_id)
        roi_labels = group_def.get("roi_labels", "*")
        base_rows = _filter_rows(rows, {**_mapping(config.get("filters")), **({"phase_id": phase_id} if phase_id else {})})
        if roi_labels != "*":
            allowed = {str(label) for label in _text_sequence(roi_labels)}
            base_rows = [row for row in base_rows if row.get("roi_label") in allowed]
        grouped = _group_rows(base_rows, ("participant_id", "analysis_variant", "family_id", "phase_id", "contrast_id"))
        for (participant_id, analysis_variant, family_id, row_phase, contrast_id), group_rows in grouped.items():
            values = [_float(row.get("crossnobis")) for row in group_rows]
            values = [value for value in values if value is not None and math.isfinite(value)]
            if not values:
                continue
            subject_values.append(
                {
                    "participant_id": participant_id,
                    "analysis_variant": analysis_variant,
                    "family_id": family_id,
                    "phase_id": row_phase,
                    "contrast_id": contrast_id,
                    "roi_set_id": roi_set_id,
                    "roi_family": roi_family,
                    "roi_count": len({row.get("roi_label") for row in group_rows if row.get("roi_label")}),
                    "crossnobis": sum(values) / len(values),
                    "feature_count_mean": _mean_number(group_rows, "feature_count"),
                    "cv_unit_count_mean": _mean_number(group_rows, "cv_unit_count"),
                    "observation_count_mean": _mean_number(group_rows, "observation_count"),
                    "source_rows_count": len(group_rows),
                }
            )
    stats_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(_group_rows(subject_values, ("analysis_variant", "family_id", "phase_id", "contrast_id", "roi_set_id", "roi_family")).items()):
        analysis_variant, family_id, phase_id, contrast_id, roi_set_id, roi_family = key
        values = [_float(row.get("crossnobis")) for row in group_rows]
        values = [value for value in values if value is not None and math.isfinite(value)]
        stat = _group_stats(values, stats_config)
        stat.update(
            {
                "analysis_variant": analysis_variant,
                "family_id": family_id,
                "phase_id": phase_id,
                "phase_display_label": _display_label(labels, "phase_display_labels", phase_id),
                "contrast_id": contrast_id,
                "contrast_display_label": _display_label(labels, "contrast_display_labels", contrast_id),
                "roi_set_id": roi_set_id,
                "roi_family": roi_family,
                "roi_set_display_label": _display_label(labels, "roi_set_display_labels", roi_set_id),
                "summary_level": "phase_pooled_mean_across_rois_per_subject",
                "aggregation_level": "subject_phase_roi_set",
                "within_participant_aggregation": "mean_across_rois",
                "aggregation_across": "participant",
                "mean_roi_count_per_participant": _mean_number(group_rows, "roi_count"),
                "mean_feature_count": _mean_number(group_rows, "feature_count_mean"),
                "mean_cv_unit_count": _mean_number(group_rows, "cv_unit_count_mean"),
                "mean_observation_count": _mean_number(group_rows, "observation_count_mean"),
                "mean_valid_cv_units": _mean_number(group_rows, "cv_unit_count_mean"),
                "source_table_relpath": source_table_relpath,
            }
        )
        stats_rows.append(stat)
    _apply_fdr(stats_rows, _text_sequence(config.get("fdr_by")) or ("analysis_variant", "roi_set_id"))
    result["stats_rows"] = _ordered_rows(stats_rows, PHASE_STATS_COLUMNS)
    result["subject_rows"] = _ordered_rows(subject_values, PHASE_SUBJECT_COLUMNS)
    result["table_rows"] = _compact_table_rows(result["stats_rows"], labels=labels, config=_mapping(config.get("compact_table")))
    result["table_full_precision_rows"] = _compact_table_rows(result["stats_rows"], labels=labels, config=_mapping(config.get("compact_table")), full_precision=True)
    result["manifest"] = _family_manifest(
        desc,
        source_table_relpath=source_table_relpath,
        note="One row per phase/contrast/ROI set after averaging ROI values within participant. This is not ROI-level inference or RDMGroupSummary.",
        rows=result["stats_rows"],
        extra={
            "within_participant_aggregation": "mean_across_rois",
            "independent_unit": "participant",
            "native_replacement_for": "EnhancedPhasePooledStatsWithCI-style reporting output",
        },
    )
    return result


def _rdm_group_summary(
    payload: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    labels: Mapping[str, Any],
    output_root: Path,
    output_relroot: str,
    prefix: str,
) -> dict[str, Any]:
    result = _empty_rdm_result(output_root, output_relroot, prefix)
    config = _mapping(_mapping(payload.get("tables")).get("rdm_group_summary"))
    if not bool(config.get("enabled", True)):
        result["enabled"] = False
        return result
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for source in _sequence(config.get("sources")):
        if not isinstance(source, Mapping):
            continue
        path, relpath, path_errors = _configured_path(source, roots=roots, default_root_ref="artifact_root")
        errors.extend(path_errors)
        rdm_id = str(source.get("rdm_id") or "")
        if path_errors:
            continue
        if not path.is_file():
            errors.append(f"RDM group summary source is missing: {relpath}.")
            continue
        for row in _read_tsv(path):
            rows.append(
                {
                    "rdm_id": row.get("rdm_id") or rdm_id,
                    "aggregation_level": "rdm_condition_pair",
                    "within_participant_aggregation": str(source.get("within_participant_aggregation") or "mean_across_rois"),
                    "aggregation_across": "participant",
                    "condition_a": row.get("condition_a", ""),
                    "condition_b": row.get("condition_b", ""),
                    "condition_a_label": row.get("condition_a_label") or _display_label(labels, "condition_display_labels", row.get("condition_a", "")),
                    "condition_b_label": row.get("condition_b_label") or _display_label(labels, "condition_display_labels", row.get("condition_b", "")),
                    "source_contrast_id": row.get("source_contrast_id", ""),
                    "N": row.get("n", ""),
                    "mean_crossnobis": row.get("group_mean_crossnobis", ""),
                    "ci_low": row.get("ci_low", ""),
                    "ci_high": row.get("ci_high", ""),
                    "SD": row.get("sd", ""),
                    "SEM": row.get("sem", ""),
                }
            )
    result["stats_rows"] = _ordered_rows(rows, RDM_STATS_COLUMNS)
    result["table_rows"] = _rdm_compact_rows(result["stats_rows"])
    result["manifest"] = {
        "table_family": "RDMGroupSummary",
        "description": "RDM/condition-pair group summary output. This is not ROI-level group inference and not phase-pooled group inference.",
        "row_count": len(rows),
        "warnings": warnings,
        "errors": errors,
    }
    result["warnings"] = warnings
    result["errors"] = errors
    return result


def _group_stats(values: Sequence[float], stats_config: Mapping[str, Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    n = len(clean)
    if n == 0:
        return _empty_stats()
    mean = sum(clean) / n
    sd = _sample_sd(clean)
    variance = sd * sd if math.isfinite(sd) else ""
    sem = sd / math.sqrt(n) if n > 0 and math.isfinite(sd) else ""
    t_stat = mean / sem if sem not in ("", 0) else ""
    df = n - 1 if n else ""
    p_two = ""
    p_one = ""
    ci_low = ""
    ci_high = ""
    p_wilcoxon = ""
    try:
        from scipy import stats

        if n > 1 and sem not in ("", 0):
            p_two = float(stats.ttest_1samp(clean, 0.0, alternative="two-sided").pvalue)
            p_one = float(stats.ttest_1samp(clean, 0.0, alternative="greater").pvalue)
            critical = float(stats.t.ppf(0.975, df))
            ci_low = mean - critical * float(sem)
            ci_high = mean + critical * float(sem)
        if n > 0 and any(value != 0 for value in clean):
            p_wilcoxon = float(stats.wilcoxon(clean, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        if n > 1 and sem not in ("", 0):
            normal_critical = 1.959963984540054
            ci_low = mean - normal_critical * float(sem)
            ci_high = mean + normal_critical * float(sem)
    q1, median, q3 = _quartiles(clean)
    bootstrap_low, bootstrap_high, bootstrap_method = _bootstrap_ci(clean, stats_config)
    p_signflip, signflip_method = _signflip_p_two(clean, seed=int(stats_config.get("signflip_seed", 12345)))
    return {
        "N": n,
        "mean_crossnobis": mean,
        "SD": sd,
        "variance": variance,
        "SEM": sem,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1 if q1 != "" and q3 != "" else "",
        "min": min(clean),
        "max": max(clean),
        "percent_positive": 100.0 * sum(1 for value in clean if value > 0) / n,
        "t": t_stat,
        "df": df,
        "p_two": p_two,
        "p_one_greater": p_one,
        "p_signflip": p_signflip,
        "p_wilcoxon": p_wilcoxon,
        "q_FDR": "",
        "dz": mean / sd if sd not in ("", 0) else "",
        "bootstrap_ci_low": bootstrap_low,
        "bootstrap_ci_high": bootstrap_high,
        "fdr_family": "",
        "signflip_method": signflip_method,
        "ci_method": "t_95_two_sided" if ci_low != "" else "not_available",
        "bootstrap_method": bootstrap_method,
    }


def _signflip_p_two(values: Sequence[float], *, seed: int) -> tuple[float | str, str]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    n = len(clean)
    if n == 0:
        return "", "not_available"
    integer_weights = _common_denominator_integer_weights(clean)
    observed = abs(sum(integer_weights))
    if n <= 28:
        if observed == 0:
            return 1.0, "exact_meet_in_middle"
        left_count = n // 2
        left = _signed_sums(integer_weights[:left_count])
        right = sorted(_signed_sums(integer_weights[left_count:]))
        count = 0
        for left_sum in left:
            count += len(right) - bisect_left(right, observed - left_sum)
            count += bisect_right(right, -observed - left_sum)
        assignment_count = 2**n
        if not 0 <= count <= assignment_count:
            raise RuntimeError("exact sign-flip count is outside the permutation space")
        return count / float(assignment_count), "exact_meet_in_middle"
    rng = random.Random(seed)
    draws = 100000
    count = 0
    for _ in range(draws):
        total = sum(weight if rng.random() < 0.5 else -weight for weight in integer_weights)
        if abs(total) >= observed:
            count += 1
    return count / draws, f"monte_carlo_{draws}"


def _common_denominator_integer_weights(values: Sequence[float]) -> list[int]:
    ratios = [float(value).as_integer_ratio() for value in values]
    # Every finite float is a binary rational, so the largest denominator is
    # a common power-of-two denominator for the represented values.
    common_denominator = max(denominator for _, denominator in ratios)
    return [
        numerator * (common_denominator // denominator)
        for numerator, denominator in ratios
    ]


def _signed_sums(values: Sequence[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums = [existing + value for existing in sums] + [existing - value for existing in sums]
    return sums


def _apply_fdr(rows: list[dict[str, Any]], family_columns: Sequence[str]) -> None:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in family_columns)
        grouped.setdefault(key, []).append(row)
    for key, group_rows in grouped.items():
        p_values = [_float(row.get("p_signflip")) for row in group_rows]
        valid_pairs = [(index, value) for index, value in enumerate(p_values) if value is not None and math.isfinite(value)]
        q_values = _bh_fdr([value for _, value in valid_pairs])
        for (index, _), q_value in zip(valid_pairs, q_values, strict=False):
            group_rows[index]["q_FDR"] = q_value
            group_rows[index]["fdr_family"] = "|".join(key)


def _bh_fdr(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    q = [1.0] * m
    running = 1.0
    for rank, (original_index, p_value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, p_value * m / rank)
        q[original_index] = running
    return q


def _prepare_manuscript_primary_main_figures(
    *,
    phase_result: Mapping[str, Any],
    roi_result: Mapping[str, Any],
    config: Mapping[str, Any],
    labels: Mapping[str, Any],
    output_root: Path,
    output_relroot: str,
    prefix: str,
) -> dict[str, Any]:
    result = _empty_manuscript_primary_result(output_root, output_relroot, prefix)
    if not config or not bool(config.get("enabled", False)):
        result["enabled"] = False
        return result
    configured_source_tables = _mapping(config.get("source_tables"))
    if configured_source_tables:
        result["source_tables"] = {
            str(key): _join_relative(output_relroot, str(value))
            for key, value in configured_source_tables.items()
            if value is not None
        }

    filters = _mapping(config.get("filters")) or {"analysis_variant": "main", "family_id": "primary_main"}
    phase_subject_rows = _filter_rows_when_columns_exist(list(phase_result.get("subject_rows", [])), filters)
    phase_stats_rows = _filter_rows_when_columns_exist(list(phase_result.get("stats_rows", [])), filters)
    roi_subject_rows = _filter_rows_when_columns_exist(list(roi_result.get("subject_rows", [])), filters)
    roi_stats_rows = _filter_rows_when_columns_exist(list(roi_result.get("stats_rows", [])), filters)

    phase_order = _text_sequence(config.get("phase_order")) or tuple(
        phase for phase in ("encoding", "recognition") if any(row.get("phase_id") == phase for row in [*phase_subject_rows, *phase_stats_rows])
    )
    if not phase_order:
        phase_order = tuple(sorted({str(row.get("phase_id", "")) for row in [*phase_subject_rows, *phase_stats_rows] if row.get("phase_id")}))
    roi_order = _text_sequence(config.get("roi_order")) or tuple(
        sorted({str(row.get("roi_label", "")) for row in [*roi_subject_rows, *roi_stats_rows] if row.get("roi_label")})
    )
    encoding_roi_order = _text_sequence(config.get("encoding_roi_order")) or tuple(
        roi for roi in roi_order if any(row.get("roi_label") == roi and row.get("phase_id") == "encoding" for row in [*roi_subject_rows, *roi_stats_rows])
    )
    recognition_roi_order = _text_sequence(config.get("recognition_roi_order")) or tuple(
        roi for roi in roi_order if any(row.get("roi_label") == roi and row.get("phase_id") == "recognition" for row in [*roi_subject_rows, *roi_stats_rows])
    )
    jitter_seed = int(config.get("jitter_seed", 3107))
    jitter_width = float(config.get("jitter_width", 0.08))
    phase_subject_plot = _manuscript_phase_subject_rows(
        phase_subject_rows,
        labels=labels,
        phase_order=phase_order,
        jitter_seed=jitter_seed,
        jitter_width=jitter_width,
    )
    phase_stats_plot = _manuscript_phase_stats_rows(
        phase_stats_rows,
        labels=labels,
        phase_order=phase_order,
    )
    roi_subject_plot = _manuscript_roi_subject_rows(
        roi_subject_rows,
        labels=labels,
        roi_order=roi_order,
        jitter_seed=jitter_seed,
        jitter_width=jitter_width,
    )
    roi_stats_plot = _manuscript_roi_stats_rows(
        roi_stats_rows,
        labels=labels,
        roi_order=roi_order,
    )
    expected_errors = _manuscript_expected_count_errors(
        _mapping(config.get("expected_counts")),
        {
            "phase_subject_values": len(phase_subject_plot),
            "phase_stats": len(phase_stats_plot),
            "roi_subject_values": len(roi_subject_plot),
            "roi_stats": len(roi_stats_plot),
        },
    )
    result["errors"].extend(expected_errors)
    result["phase_subject_rows"] = _ordered_rows(phase_subject_plot, MANUSCRIPT_PHASE_SUBJECT_COLUMNS)
    result["phase_stats_rows"] = _ordered_rows(phase_stats_plot, MANUSCRIPT_PHASE_STATS_COLUMNS)
    result["roi_subject_rows"] = _ordered_rows(roi_subject_plot, MANUSCRIPT_ROI_SUBJECT_COLUMNS)
    result["roi_stats_rows"] = _ordered_rows(roi_stats_plot, MANUSCRIPT_ROI_STATS_COLUMNS)

    figure_defaults = _mapping(config.get("figure_defaults"))
    output_formats = _text_sequence(config.get("output_formats")) or OUTPUT_FORMATS
    figures = _mapping(config.get("figures"))
    result["figures"] = [
        _manuscript_figure_record(
            "manuscript_primary_main_phase_pooled",
            "manuscript_phase_pooled_violin_dot",
            _mapping(figures.get("phase_pooled_violin_dot")),
            figure_defaults=figure_defaults,
            output_root=output_root,
            output_relroot=output_relroot,
            basename=str(
                _mapping(figures.get("phase_pooled_violin_dot")).get("output_basename")
                or f"{prefix}_desc-ManuscriptPrimaryMainPhasePooledCrossnobisViolinDot"
            ),
            output_formats=output_formats,
            plot_rows=result["phase_subject_rows"],
            summary_rows=result["phase_stats_rows"],
            source_tables=result["source_tables"],
            filters=filters,
        ),
        _manuscript_figure_record(
            "manuscript_primary_main_roi_forest",
            "manuscript_roi_forest",
            _mapping(figures.get("roi_forest")),
            figure_defaults=figure_defaults,
            output_root=output_root,
            output_relroot=output_relroot,
            basename=str(
                _mapping(figures.get("roi_forest")).get("output_basename")
                or f"{prefix}_desc-ManuscriptPrimaryMainROICrossnobisForest"
            ),
            output_formats=output_formats,
            plot_rows=result["roi_stats_rows"],
            summary_rows=result["roi_stats_rows"],
            source_tables=result["source_tables"],
            filters=filters,
        ),
        _manuscript_figure_record(
            "manuscript_primary_main_encoding_roi_dot_ci",
            "manuscript_roi_dot_ci",
            _mapping(figures.get("encoding_roi_dot_ci")),
            figure_defaults=figure_defaults,
            output_root=output_root,
            output_relroot=output_relroot,
            basename=str(
                _mapping(figures.get("encoding_roi_dot_ci")).get("output_basename")
                or f"{prefix}_desc-ManuscriptPrimaryMainEncodingROICrossnobisDotCI"
            ),
            output_formats=output_formats,
            plot_rows=[row for row in result["roi_subject_rows"] if row.get("phase_id") == "encoding" and row.get("roi_label") in set(encoding_roi_order)],
            summary_rows=[row for row in result["roi_stats_rows"] if row.get("phase_id") == "encoding" and row.get("roi_label") in set(encoding_roi_order)],
            source_tables=result["source_tables"],
            filters={**dict(filters), "phase_id": "encoding"},
        ),
        _manuscript_figure_record(
            "manuscript_primary_main_recognition_roi_dot_ci",
            "manuscript_roi_dot_ci",
            _mapping(figures.get("recognition_roi_dot_ci")),
            figure_defaults=figure_defaults,
            output_root=output_root,
            output_relroot=output_relroot,
            basename=str(
                _mapping(figures.get("recognition_roi_dot_ci")).get("output_basename")
                or f"{prefix}_desc-ManuscriptPrimaryMainRecognitionROICrossnobisDotCI"
            ),
            output_formats=output_formats,
            plot_rows=[row for row in result["roi_subject_rows"] if row.get("phase_id") == "recognition" and row.get("roi_label") in set(recognition_roi_order)],
            summary_rows=[row for row in result["roi_stats_rows"] if row.get("phase_id") == "recognition" and row.get("roi_label") in set(recognition_roi_order)],
            source_tables=result["source_tables"],
            filters={**dict(filters), "phase_id": "recognition"},
        ),
    ]
    return result


def _empty_manuscript_primary_result(output_root: Path, output_relroot: str, prefix: str) -> dict[str, Any]:
    source_tables = {
        "phase_subject_values": _join_relative(output_relroot, f"tables/{prefix}_desc-SubjectLevelPhasePooledGroupInference_subjectValues.tsv"),
        "phase_stats": _join_relative(output_relroot, f"tables/{prefix}_desc-SubjectLevelPhasePooledGroupInference_stats.tsv"),
        "roi_subject_values": _join_relative(output_relroot, f"tables/{prefix}_desc-SubjectLevelROIGroupInference_subjectValues.tsv"),
        "roi_stats": _join_relative(output_relroot, f"tables/{prefix}_desc-SubjectLevelROIGroupInference_stats.tsv"),
    }
    return {
        "enabled": True,
        "family": "ManuscriptPrimaryMainFigures",
        "source_tables": source_tables,
        "companion_outputs": {
            "phase_subject_values": _target(output_root, output_relroot, f"tables/{prefix}_desc-ManuscriptPrimaryMainPhasePooled_subjectValues.tsv"),
            "phase_stats": _target(output_root, output_relroot, f"tables/{prefix}_desc-ManuscriptPrimaryMainPhasePooled_stats.tsv"),
            "roi_subject_values": _target(output_root, output_relroot, f"tables/{prefix}_desc-ManuscriptPrimaryMainROI_subjectValues.tsv"),
            "roi_stats": _target(output_root, output_relroot, f"tables/{prefix}_desc-ManuscriptPrimaryMainROI_stats.tsv"),
        },
        "phase_subject_rows": [],
        "phase_stats_rows": [],
        "roi_subject_rows": [],
        "roi_stats_rows": [],
        "figures": [],
        "warnings": [],
        "errors": [],
    }


def _manuscript_figure_record(
    figure_id: str,
    kind: str,
    config: Mapping[str, Any],
    *,
    figure_defaults: Mapping[str, Any],
    output_root: Path,
    output_relroot: str,
    basename: str,
    output_formats: Sequence[str],
    plot_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    source_tables: Mapping[str, str],
    filters: Mapping[str, Any],
) -> dict[str, Any]:
    merged_config = {**dict(figure_defaults), **dict(config)}
    outputs = _manuscript_figure_outputs(output_root, output_relroot, basename, output_formats=output_formats)
    return {
        "figure_id": str(config.get("figure_id") or figure_id),
        "kind": kind,
        "config": merged_config,
        "outputs": outputs,
        "plot_rows": [dict(row) for row in plot_rows],
        "summary_rows": [dict(row) for row in summary_rows],
        "warnings": [],
        "errors": [],
        "manifest": {
            "figure_id": str(config.get("figure_id") or figure_id),
            "figure_family": "ManuscriptPrimaryMainFigures",
            "kind": kind,
            "source_tables": dict(source_tables),
            "filters": dict(filters),
            "row_counts": {"plot_data": len(plot_rows), "summary": len(summary_rows)},
            "outputs": {key: value["relative_path"] for key, value in outputs.items()},
            "layout_warnings": [],
            "executed": False,
            "editable_text": {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42},
            "recomputed_statistics": False,
            "recomputed_mvpa_distances": False,
        },
    }


def _manuscript_figure_outputs(output_root: Path, output_relroot: str, basename: str, *, output_formats: Sequence[str]) -> dict[str, dict[str, Any]]:
    outputs = {f"figure_{fmt}": _target(output_root, output_relroot, f"figures/{basename}.{fmt}") for fmt in output_formats}
    outputs["plot_data_tsv"] = _target(output_root, output_relroot, f"figures/{basename}_plot-data.tsv")
    outputs["summary_tsv"] = _target(output_root, output_relroot, f"figures/{basename}_summary.tsv")
    outputs["manifest_json"] = _target(output_root, output_relroot, f"figures/{basename}_manifest.json")
    return outputs


def _write_manuscript_primary_outputs(result: Mapping[str, Any]) -> dict[str, Any]:
    if not result.get("enabled", True):
        return dict(result)
    updated = dict(result)
    companion_outputs = _mapping(updated.get("companion_outputs"))
    _write_tsv_atomic(Path(_mapping(companion_outputs.get("phase_subject_values")).get("path", "")), updated.get("phase_subject_rows", []))
    _write_tsv_atomic(Path(_mapping(companion_outputs.get("phase_stats")).get("path", "")), updated.get("phase_stats_rows", []))
    _write_tsv_atomic(Path(_mapping(companion_outputs.get("roi_subject_values")).get("path", "")), updated.get("roi_subject_rows", []))
    _write_tsv_atomic(Path(_mapping(companion_outputs.get("roi_stats")).get("path", "")), updated.get("roi_stats_rows", []))
    try:
        plt = _load_matplotlib_pyplot()
    except ImportError as exc:
        raise RuntimeError(str(exc)) from exc
    written_figures = []
    write_errors: list[str] = []
    for figure in updated.get("figures", []):
        figure = dict(figure)
        outputs = _mapping(figure.get("outputs"))
        _write_tsv_atomic(Path(_mapping(outputs.get("plot_data_tsv")).get("path", "")), figure.get("plot_rows", []))
        _write_tsv_atomic(Path(_mapping(outputs.get("summary_tsv")).get("path", "")), figure.get("summary_rows", []))
        layout_warnings: list[str] = []
        figure_errors = list(figure.get("errors", []))
        try:
            layout_warnings = _render_manuscript_figure(plt, figure)
        except Exception as exc:
            figure_errors.append(f"Manuscript-primary figure {figure.get('figure_id')} failed to render: {exc}")
        manifest = dict(_mapping(figure.get("manifest")))
        manifest["layout_warnings"] = layout_warnings
        manifest["layout_warning_count"] = len(layout_warnings)
        manifest["layout_status"] = "ok" if not layout_warnings else "warning"
        manifest["executed"] = True
        figure["warnings"] = [*list(figure.get("warnings", [])), *layout_warnings]
        if layout_warnings and bool(_mapping(figure.get("config")).get("fail_on_layout_warning", False)):
            figure_errors.append(f"Manuscript-primary figure {figure.get('figure_id')} failed layout QC: {'; '.join(layout_warnings)}")
        figure["errors"] = figure_errors
        figure["manifest"] = manifest
        _write_json_atomic(Path(_mapping(outputs.get("manifest_json")).get("path", "")), manifest)
        write_errors.extend(figure_errors)
        written_figures.append(figure)
    updated["figures"] = written_figures
    updated["warnings"] = _unique([warning for figure in written_figures for warning in figure.get("warnings", [])])
    updated["errors"] = _unique([*list(updated.get("errors", [])), *write_errors, *_missing_manuscript_figure_output_errors(written_figures)])
    return updated


def _missing_manuscript_figure_output_errors(figures: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for figure in figures:
        figure_id = str(figure.get("figure_id") or "unknown")
        for output in _mapping(figure.get("outputs")).values():
            path = Path(str(_mapping(output).get("path", "")))
            if not path.is_file():
                errors.append(f"Manuscript-primary figure {figure_id} did not write expected output: {path.name}.")
    return errors


def _manuscript_phase_subject_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels: Mapping[str, Any],
    phase_order: Sequence[str],
    jitter_seed: int,
    jitter_width: float,
) -> list[dict[str, Any]]:
    positions = {phase: index + 1 for index, phase in enumerate(phase_order)}
    rng = random.Random(jitter_seed)
    output: list[dict[str, Any]] = []
    for row in _sort_by_order(rows, "phase_id", phase_order):
        phase_id = str(row.get("phase_id", ""))
        x_position = float(positions.get(phase_id, len(positions) + 1))
        value = row.get("crossnobis", "")
        output.append(
            {
                **dict(row),
                "phase_display_label": _display_label(labels, "phase_display_labels", phase_id),
                "plot_value": value,
                "x_position": x_position,
                "jittered_x_position": x_position + rng.uniform(-jitter_width, jitter_width),
            }
        )
    return output


def _manuscript_phase_stats_rows(rows: Sequence[Mapping[str, Any]], *, labels: Mapping[str, Any], phase_order: Sequence[str]) -> list[dict[str, Any]]:
    positions = {phase: index + 1 for index, phase in enumerate(phase_order)}
    output: list[dict[str, Any]] = []
    for row in _sort_by_order(rows, "phase_id", phase_order):
        phase_id = str(row.get("phase_id", ""))
        output.append(
            {
                **dict(row),
                "phase_display_label": row.get("phase_display_label") or _display_label(labels, "phase_display_labels", phase_id),
                "x_position": float(positions.get(phase_id, len(positions) + 1)),
            }
        )
    return output


def _manuscript_roi_subject_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels: Mapping[str, Any],
    roi_order: Sequence[str],
    jitter_seed: int,
    jitter_width: float,
) -> list[dict[str, Any]]:
    positions = {roi: index for index, roi in enumerate(roi_order)}
    rng = random.Random(jitter_seed + 101)
    output: list[dict[str, Any]] = []
    for row in _sort_by_order(rows, "roi_label", roi_order):
        roi_label = str(row.get("roi_label", ""))
        y_position = float(positions.get(roi_label, len(positions)))
        value = row.get("crossnobis", "")
        output.append(
            {
                **dict(row),
                "roi_display_label": _display_label(labels, "roi_display_labels", roi_label),
                "plot_value": value,
                "x_position": value,
                "y_position": y_position,
                "jittered_y_position": y_position + rng.uniform(-jitter_width, jitter_width),
            }
        )
    return output


def _manuscript_roi_stats_rows(rows: Sequence[Mapping[str, Any]], *, labels: Mapping[str, Any], roi_order: Sequence[str]) -> list[dict[str, Any]]:
    positions = {roi: index for index, roi in enumerate(roi_order)}
    output: list[dict[str, Any]] = []
    for row in _sort_by_order(rows, "roi_label", roi_order):
        roi_label = str(row.get("roi_label", ""))
        output.append(
            {
                **dict(row),
                "roi_display_label": row.get("roi_display_label") or _display_label(labels, "roi_display_labels", roi_label),
                "y_position": float(positions.get(roi_label, len(positions))),
            }
        )
    return output


def _render_manuscript_figure(plt: Any, figure: Mapping[str, Any]) -> list[str]:
    config = _mapping(figure.get("config"))
    kind = str(figure.get("kind", ""))
    plot_rows = list(figure.get("plot_rows", []))
    summary_rows = list(figure.get("summary_rows", []))
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    size = _mapping(config.get("figure_size"))
    width = float(size.get("width_inches", 8.0))
    height = float(size.get("height_inches", 5.0))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
    if kind == "manuscript_phase_pooled_violin_dot":
        _draw_manuscript_phase_violin_dot(ax, plot_rows, summary_rows, config)
    elif kind == "manuscript_roi_forest":
        _draw_manuscript_roi_forest(ax, summary_rows, config)
    elif kind == "manuscript_roi_dot_ci":
        _draw_manuscript_roi_dot_ci(ax, plot_rows, summary_rows, config)
    else:
        raise RuntimeError(f"Unsupported manuscript-primary figure kind: {kind}.")
    ax.set_title(str(config.get("title") or figure.get("figure_id") or ""), pad=float(config.get("title_pad", 12)))
    fig.tight_layout(pad=float(config.get("layout_pad", 1.4)))
    _apply_manuscript_margins(fig, kind, plot_rows, summary_rows, config)
    fig.canvas.draw()
    warnings = _figure_layout_warnings(fig, config)
    outputs = _mapping(figure.get("outputs"))
    for key, record in outputs.items():
        if not key.startswith("figure_"):
            continue
        fig.savefig(str(_mapping(record).get("path")), dpi=int(config.get("dpi", 300)))
    plt.close(fig)
    return warnings


def _apply_manuscript_margins(
    fig: Any,
    kind: str,
    plot_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    if "left_margin" in config:
        left = float(config["left_margin"])
    elif kind in {"manuscript_roi_forest", "manuscript_roi_dot_ci"}:
        labels = [str(row.get("roi_display_label") or row.get("roi_label") or "") for row in [*plot_rows, *summary_rows]]
        max_chars = max((len(label) for label in labels), default=10)
        left = min(0.5, max(0.22, 0.12 + max_chars * 0.006))
    else:
        left = 0.17
    bottom = float(config.get("bottom_margin", 0.28))
    right = float(config.get("right_margin", 0.86))
    top = float(config.get("top_margin", 0.90))
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)


def _draw_manuscript_phase_violin_dot(ax: Any, plot_rows: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    positions = sorted({_float(row.get("x_position")) for row in [*plot_rows, *summary_rows] if _float(row.get("x_position")) is not None})
    labels_by_position = {
        _float(row.get("x_position")): str(row.get("phase_display_label") or row.get("phase_id") or "")
        for row in [*summary_rows, *plot_rows]
        if _float(row.get("x_position")) is not None
    }
    values_by_position: dict[float, list[float]] = {float(position): [] for position in positions}
    for row in plot_rows:
        position = _float(row.get("x_position"))
        value = _float(row.get("plot_value") or row.get("crossnobis"))
        if position is not None and value is not None:
            values_by_position.setdefault(float(position), []).append(value)
    if bool(config.get("violin", True)) and positions:
        ax.violinplot([values_by_position.get(float(position), []) for position in positions], positions=positions, showmeans=False, showextrema=False)
    for row in plot_rows:
        value = _float(row.get("plot_value") or row.get("crossnobis"))
        jittered = _float(row.get("jittered_x_position"))
        if value is not None and jittered is not None:
            ax.scatter([jittered], [value], s=float(config.get("dot_size", 18)), alpha=float(config.get("dot_alpha", 0.72)), color=str(config.get("dot_color", "#2f6f4e")), linewidths=0)
    for row in summary_rows:
        position = _float(row.get("x_position"))
        mean = _float(row.get("mean_crossnobis"))
        low = _float(row.get("ci_low"))
        high = _float(row.get("ci_high"))
        if position is None or mean is None:
            continue
        yerr = [[mean - low if low is not None else 0.0], [high - mean if high is not None else 0.0]]
        ax.errorbar([position], [mean], yerr=yerr, fmt="D", color=str(config.get("mean_color", "#111111")), ecolor=str(config.get("mean_color", "#111111")), capsize=4, markersize=6)
    ax.axhline(0.0, linestyle="--", color="0.4", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels_by_position.get(float(position), str(position)) for position in positions])
    ax.set_ylabel(str(config.get("ylabel") or "Mean crossnobis distance across configured ROIs"))
    ax.set_xlabel(str(config.get("xlabel") or ""))
    _set_numeric_ylim(ax, [*(_float(row.get("plot_value") or row.get("crossnobis")) for row in plot_rows), *(_float(row.get("ci_low")) for row in summary_rows), *(_float(row.get("ci_high")) for row in summary_rows)])


def _draw_manuscript_roi_forest(ax: Any, summary_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    ordered = sorted(summary_rows, key=lambda row: _float(row.get("y_position")) or 0.0)
    y_positions = [_float(row.get("y_position")) or 0.0 for row in ordered]
    labels = [str(row.get("roi_display_label") or row.get("roi_label") or "") for row in ordered]
    means = [_float(row.get("mean_crossnobis")) or 0.0 for row in ordered]
    lows = [_float(row.get("ci_low")) for row in ordered]
    highs = [_float(row.get("ci_high")) for row in ordered]
    xerr = [
        [mean - low if low is not None else 0.0 for mean, low in zip(means, lows, strict=False)],
        [high - mean if high is not None else 0.0 for mean, high in zip(means, highs, strict=False)],
    ]
    ax.errorbar(means, y_positions, xerr=xerr, fmt="D", color=str(config.get("mean_color", "#1f4e79")), ecolor=str(config.get("mean_color", "#1f4e79")), capsize=4, markersize=6)
    ax.axvline(0.0, linestyle="--", color="0.4", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(str(config.get("xlabel") or "Mean crossnobis distance"))
    ax.set_ylabel(str(config.get("ylabel") or ""))
    _set_numeric_xlim(ax, [*means, *(value for value in lows if value is not None), *(value for value in highs if value is not None)])
    ax.invert_yaxis()


def _draw_manuscript_roi_dot_ci(ax: Any, plot_rows: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    ordered_stats = sorted(summary_rows, key=lambda row: _float(row.get("y_position")) or 0.0)
    y_positions = [_float(row.get("y_position")) or 0.0 for row in ordered_stats]
    labels = [str(row.get("roi_display_label") or row.get("roi_label") or "") for row in ordered_stats]
    for row in plot_rows:
        value = _float(row.get("plot_value") or row.get("crossnobis"))
        y_position = _float(row.get("jittered_y_position") or row.get("y_position"))
        if value is not None and y_position is not None:
            ax.scatter([value], [y_position], s=float(config.get("dot_size", 14)), alpha=float(config.get("dot_alpha", 0.62)), color=str(config.get("dot_color", "#2f6f4e")), linewidths=0)
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for row in ordered_stats:
        y_position = _float(row.get("y_position"))
        mean = _float(row.get("mean_crossnobis"))
        low = _float(row.get("ci_low"))
        high = _float(row.get("ci_high"))
        if y_position is None or mean is None:
            continue
        means.append(mean)
        if low is not None:
            lows.append(low)
        if high is not None:
            highs.append(high)
        xerr = [[mean - low if low is not None else 0.0], [high - mean if high is not None else 0.0]]
        ax.errorbar([mean], [y_position], xerr=xerr, fmt="D", color=str(config.get("mean_color", "#111111")), ecolor=str(config.get("mean_color", "#111111")), capsize=4, markersize=6)
    ax.axvline(0.0, linestyle="--", color="0.4", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(str(config.get("xlabel") or "Crossnobis distance"))
    ax.set_ylabel(str(config.get("ylabel") or ""))
    _set_numeric_xlim(ax, [*(_float(row.get("plot_value") or row.get("crossnobis")) for row in plot_rows), *means, *lows, *highs])
    ax.invert_yaxis()


def _set_numeric_ylim(ax: Any, values: Sequence[float | None]) -> None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return
    low = min(clean)
    high = max(clean)
    spread = high - low or max(abs(high), 0.01)
    ax.set_ylim(low - spread * 0.18, high + spread * 0.22)


def _set_numeric_xlim(ax: Any, values: Sequence[float | None]) -> None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return
    low = min(clean)
    high = max(clean)
    spread = high - low or max(abs(high), 0.01)
    ax.set_xlim(low - spread * 0.18, high + spread * 0.22)


def _figure_layout_warnings(fig: Any, config: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    renderer = fig.canvas.get_renderer()
    width = fig.bbox.width
    height = fig.bbox.height
    max_label_chars = int(config.get("max_label_chars", 96))
    for artist in fig.findobj():
        if not hasattr(artist, "get_window_extent") or not hasattr(artist, "get_text"):
            continue
        text = str(artist.get_text())
        if not text:
            continue
        if len(text) > max_label_chars:
            warnings.append(f"text label exceeds configured maximum length: {text[:40]}")
        try:
            bbox = artist.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > width + 1 or bbox.y1 > height + 1:
            warnings.append(f"text artist is outside figure bounds: {text[:40]}")
    return _unique(warnings)


def _manuscript_expected_count_errors(expected: Mapping[str, Any], actual: Mapping[str, int]) -> list[str]:
    errors: list[str] = []
    for key, value in expected.items():
        expected_value = _float(value)
        if expected_value is None:
            continue
        actual_value = int(actual.get(str(key), 0))
        if actual_value != int(expected_value):
            errors.append(f"ManuscriptPrimaryMainFigures expected {key}={int(expected_value)} but found {actual_value}.")
    return errors


def _filter_rows_when_columns_exist(rows: Sequence[Mapping[str, Any]], filters: Mapping[str, Any]) -> list[dict[str, Any]]:
    available_filters = {
        key: value
        for key, value in filters.items()
        if any(str(row.get(str(key), "")) != "" for row in rows)
    }
    return _filter_rows(rows, available_filters)


def _sort_by_order(rows: Sequence[Mapping[str, Any]], key: str, order: Sequence[str]) -> list[dict[str, Any]]:
    positions = {value: index for index, value in enumerate(order)}
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (positions.get(str(row.get(key, "")), len(positions)), str(row.get(key, "")), str(row.get("participant_id", ""))),
    )


def _prepare_figures(
    *,
    roi_result: Mapping[str, Any],
    phase_result: Mapping[str, Any],
    config: Mapping[str, Any],
    labels: Mapping[str, Any],
    output_root: Path,
    output_relroot: str,
    prefix: str,
) -> list[dict[str, Any]]:
    figure_defs = config.get("items")
    if not isinstance(figure_defs, Sequence) or isinstance(figure_defs, (str, bytes, bytearray)):
        figure_defs = [
            {"figure_id": "roi_group_inference_forest", "kind": "roi_group_forest", "title": "ROI group inference"},
            {"figure_id": "roi_group_inference_violin_dot", "kind": "roi_group_violin_dot", "title": "ROI subject values"},
            {"figure_id": "phase_pooled_group_inference_violin_dot", "kind": "phase_pooled_violin_dot", "title": "Phase-pooled subject values"},
        ]
    results: list[dict[str, Any]] = []
    for figure in figure_defs:
        if not isinstance(figure, Mapping) or not bool(figure.get("enabled", True)):
            continue
        kind = str(figure.get("kind") or "")
        desc = str(figure.get("output_desc") or _default_figure_desc(kind))
        basename = str(figure.get("output_basename") or f"{prefix}_desc-{desc}")
        outputs = _figure_outputs(output_root, output_relroot, basename)
        normalized_kind = _normalize_figure_kind(kind)
        if normalized_kind == "roi_group_forest":
            plot_rows = list(roi_result.get("stats_rows", []))
            summary_rows = plot_rows
        elif normalized_kind == "roi_group_violin_dot":
            plot_rows = list(roi_result.get("subject_rows", []))
            summary_rows = list(roi_result.get("stats_rows", []))
        elif normalized_kind == "phase_pooled_violin_dot":
            plot_rows = list(phase_result.get("subject_rows", []))
            summary_rows = list(phase_result.get("stats_rows", []))
        else:
            results.append({"figure_id": str(figure.get("figure_id") or kind), "kind": kind, "outputs": outputs, "plot_rows": [], "summary_rows": [], "manifest": {}, "warnings": [], "errors": [f"Unsupported publication figure kind: {kind}."]})
            continue
        results.append(
            {
                "figure_id": str(figure.get("figure_id") or kind),
                "kind": normalized_kind,
                "config": dict(figure),
                "outputs": outputs,
                "plot_rows": plot_rows,
                "summary_rows": summary_rows,
                "manifest": {
                    "figure_id": str(figure.get("figure_id") or kind),
                    "kind": kind,
                    "editable_text": {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42},
                    "row_counts": {"plot_data": len(plot_rows), "summary": len(summary_rows)},
                },
                "warnings": [],
                "errors": [],
            }
        )
    return results


def _write_figures(figure_results: Sequence[Mapping[str, Any]]) -> None:
    if not figure_results:
        return
    try:
        plt = _load_matplotlib_pyplot()
    except ImportError as exc:
        raise RuntimeError(str(exc)) from exc
    for figure in figure_results:
        outputs = figure["outputs"]
        _write_tsv_atomic(Path(outputs["plot_data_tsv"]["path"]), figure.get("plot_rows", []))
        _write_tsv_atomic(Path(outputs["summary_tsv"]["path"]), figure.get("summary_rows", []))
        _write_json_atomic(Path(outputs["manifest_json"]["path"]), figure.get("manifest", {}))
        _render_publication_figure(plt, figure)


def _render_publication_figure(plt: Any, figure: Mapping[str, Any]) -> None:
    config = _mapping(figure.get("config"))
    plot_rows = list(figure.get("plot_rows", []))
    kind = _normalize_figure_kind(str(figure.get("kind")))
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    width = float(_mapping(config.get("figure_size")).get("width_inches", 8.0))
    height = float(_mapping(config.get("figure_size")).get("height_inches", 5.0))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    title = str(config.get("title") or figure.get("figure_id") or "")
    ylabel = str(config.get("ylabel") or "Crossnobis distance")
    xlabel = str(config.get("xlabel") or "")
    if kind == "roi_group_forest":
        labels = [str(row.get("roi_display_label") or row.get("roi_label")) for row in plot_rows]
        y_positions = list(range(len(labels)))
        means = [_float(row.get("mean_crossnobis")) or 0.0 for row in plot_rows]
        lows = [_float(row.get("ci_low")) for row in plot_rows]
        highs = [_float(row.get("ci_high")) for row in plot_rows]
        xerr = [
            [mean - low if low is not None else 0.0 for mean, low in zip(means, lows, strict=False)],
            [high - mean if high is not None else 0.0 for mean, high in zip(means, highs, strict=False)],
        ]
        ax.errorbar(means, y_positions, xerr=xerr, fmt="D", color="#1f4e79", ecolor="#1f4e79", capsize=3)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.axvline(0.0, linestyle="--", color="0.4", linewidth=1)
        ax.set_xlabel(ylabel)
    else:
        category_key = "roi_label" if kind == "roi_group_violin_dot" else "phase_id"
        categories = sorted({str(row.get(category_key, "")) for row in plot_rows if row.get(category_key)})
        by_category = {category: [_float(row.get("crossnobis")) for row in plot_rows if str(row.get(category_key, "")) == category] for category in categories}
        positions = list(range(1, len(categories) + 1))
        if bool(config.get("violin", True)) and categories:
            ax.violinplot([[value for value in by_category[category] if value is not None] for category in categories], positions=positions, showmeans=False, showextrema=False)
        rng = random.Random(int(config.get("jitter_seed", 123)))
        for position, category in zip(positions, categories, strict=False):
            values = [value for value in by_category[category] if value is not None]
            xs = [position + rng.uniform(-0.08, 0.08) for _ in values]
            ax.scatter(xs, values, s=16, alpha=0.75, color="#2f6f4e")
            if values:
                mean = sum(values) / len(values)
                sd = _sample_sd(values)
                sem = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
                ax.errorbar([position], [mean], yerr=[[1.96 * sem], [1.96 * sem]], fmt="D", color="#111111", capsize=4)
        display = _mapping(config.get("display_labels"))
        ax.set_xticks(positions)
        ax.set_xticklabels([str(display.get(category, category)) for category in categories], rotation=float(config.get("tick_rotation", 25)), ha="right")
        ax.axhline(0.0, linestyle="--", color="0.4", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    ax.set_title(title, pad=12)
    for fmt in OUTPUT_FORMATS:
        fig.savefig(figure["outputs"][f"figure_{fmt}"]["path"], dpi=int(config.get("dpi", 300)))
    plt.close(fig)


def _write_family_outputs(result: Mapping[str, Any]) -> None:
    outputs = result["outputs"]
    _write_tsv_atomic(Path(outputs["stats_tsv"]["path"]), result.get("stats_rows", []))
    _write_tsv_atomic(Path(outputs["table_tsv"]["path"]), result.get("table_rows", []))
    _write_tsv_atomic(Path(outputs["table_full_precision_tsv"]["path"]), result.get("table_full_precision_rows", []))
    _write_tsv_atomic(Path(outputs["subject_values_tsv"]["path"]), result.get("subject_rows", []))
    _write_json_atomic(Path(outputs["manifest_json"]["path"]), result.get("manifest", {}))


def _write_rdm_outputs(result: Mapping[str, Any]) -> None:
    outputs = result["outputs"]
    _write_tsv_atomic(Path(outputs["stats_tsv"]["path"]), result.get("stats_rows", []))
    _write_tsv_atomic(Path(outputs["table_tsv"]["path"]), result.get("table_rows", []))
    _write_json_atomic(Path(outputs["manifest_json"]["path"]), result.get("manifest", {}))


def _compact_table_rows(rows: Sequence[Mapping[str, Any]], *, labels: Mapping[str, Any], config: Mapping[str, Any], full_precision: bool = False) -> list[dict[str, Any]]:
    columns = _text_sequence(config.get("columns")) or DEFAULT_COMPACT_COLUMNS
    output: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("roi_display_label") or row.get("roi_set_display_label") or row.get("roi_label") or row.get("roi_set_id") or ""
        values = {
            "Functional ROIs": label,
            "Functional ROI": label,
            "ROI": label,
            "Analysis": _display_label(labels, "analysis_variant_display_labels", str(row.get("analysis_variant", ""))),
            "Contrast": row.get("contrast_display_label") or row.get("contrast_id", ""),
            "N": row.get("N", ""),
            "Mean Distance": _format_number(row.get("mean_crossnobis"), full_precision=full_precision),
            "95% CI": _format_ci(row.get("ci_low"), row.get("ci_high"), full_precision=full_precision),
            "Effect Size": _format_number(row.get("dz"), full_precision=full_precision),
            "Permutation": _format_p(row.get("p_signflip"), full_precision=full_precision),
            "qFDR": _format_p(row.get("q_FDR"), full_precision=full_precision),
            "SD": _format_number(row.get("SD"), full_precision=full_precision),
            "p_signflip": _format_p(row.get("p_signflip"), full_precision=full_precision),
            "q_FDR": _format_p(row.get("q_FDR"), full_precision=full_precision),
            "dz": _format_number(row.get("dz"), full_precision=full_precision),
            "Crossnobis M [95% CI]": f"{_format_number(row.get('mean_crossnobis'), full_precision=full_precision)} {_format_ci(row.get('ci_low'), row.get('ci_high'), full_precision=full_precision)}",
        }
        output.append({column: values.get(column, row.get(column, "")) for column in columns})
    return output


def _rdm_compact_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "RDM": row.get("rdm_id", ""),
            "Condition A": row.get("condition_a_label", ""),
            "Condition B": row.get("condition_b_label", ""),
            "N": row.get("N", ""),
            "Mean Distance": _format_number(row.get("mean_crossnobis")),
            "95% CI": _format_ci(row.get("ci_low"), row.get("ci_high")),
        }
        for row in rows
    ]


def _empty_family_result(desc: str, output_root: Path, output_relroot: str, prefix: str) -> dict[str, Any]:
    outputs = _family_outputs(output_root, output_relroot, prefix, desc)
    return {
        "enabled": True,
        "desc": desc,
        "outputs": outputs,
        "stats_rows": [],
        "table_rows": [],
        "table_full_precision_rows": [],
        "subject_rows": [],
        "manifest": {},
        "warnings": [],
        "errors": [],
    }


def _empty_rdm_result(output_root: Path, output_relroot: str, prefix: str) -> dict[str, Any]:
    desc = "RDMGroupSummary"
    return {
        "enabled": True,
        "desc": desc,
        "outputs": {
            "stats_tsv": _target(output_root, output_relroot, f"tables/{prefix}_desc-{desc}_stats.tsv"),
            "table_tsv": _target(output_root, output_relroot, f"tables/{prefix}_desc-{desc}_table.tsv"),
            "manifest_json": _target(output_root, output_relroot, f"tables/{prefix}_desc-{desc}_manifest.json"),
        },
        "stats_rows": [],
        "table_rows": [],
        "manifest": {},
        "warnings": [],
        "errors": [],
    }


def _family_outputs(output_root: Path, output_relroot: str, prefix: str, desc: str) -> dict[str, dict[str, Any]]:
    return {
        "stats_tsv": _target(output_root, output_relroot, f"tables/{prefix}_desc-{desc}_stats.tsv"),
        "table_tsv": _target(output_root, output_relroot, f"tables/{prefix}_desc-{desc}_table.tsv"),
        "table_full_precision_tsv": _target(output_root, output_relroot, f"tables/{prefix}_desc-{desc}_tableFullPrecision.tsv"),
        "subject_values_tsv": _target(output_root, output_relroot, f"tables/{prefix}_desc-{desc}_subjectValues.tsv"),
        "manifest_json": _target(output_root, output_relroot, f"tables/{prefix}_desc-{desc}_manifest.json"),
    }


def _figure_outputs(output_root: Path, output_relroot: str, basename: str) -> dict[str, dict[str, Any]]:
    outputs = {f"figure_{fmt}": _target(output_root, output_relroot, f"figures/{basename}.{fmt}") for fmt in OUTPUT_FORMATS}
    outputs["plot_data_tsv"] = _target(output_root, output_relroot, f"figures/{basename}_plotData.tsv")
    outputs["summary_tsv"] = _target(output_root, output_relroot, f"figures/{basename}_summary.tsv")
    outputs["manifest_json"] = _target(output_root, output_relroot, f"figures/{basename}_manifest.json")
    return outputs


def _target(output_root: Path, output_relroot: str, relative: str) -> dict[str, Any]:
    path = output_root / relative
    return {"relative_path": _join_relative(output_relroot, relative), "path": path.as_posix(), "filename": path.name}


def _family_public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": result.get("enabled", True),
        "outputs": result.get("outputs", {}),
        "row_counts": {
            "stats": len(result.get("stats_rows", [])),
            "table": len(result.get("table_rows", [])),
            "subject_values": len(result.get("subject_rows", [])),
        },
    }


def _rdm_public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": result.get("enabled", True),
        "outputs": result.get("outputs", {}),
        "row_counts": {"stats": len(result.get("stats_rows", [])), "table": len(result.get("table_rows", []))},
    }


def _figure_public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "figure_id": result.get("figure_id"),
        "kind": result.get("kind"),
        "outputs": result.get("outputs", {}),
        "row_counts": {"plot_data": len(result.get("plot_rows", [])), "summary": len(result.get("summary_rows", []))},
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
    }


def _manuscript_public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": result.get("enabled", True),
        "family": result.get("family", "ManuscriptPrimaryMainFigures"),
        "source_tables": result.get("source_tables", {}),
        "companion_outputs": result.get("companion_outputs", {}),
        "row_counts": {
            "phase_subject_values": len(result.get("phase_subject_rows", [])),
            "phase_stats": len(result.get("phase_stats_rows", [])),
            "roi_subject_values": len(result.get("roi_subject_rows", [])),
            "roi_stats": len(result.get("roi_stats_rows", [])),
        },
        "figures": [_figure_public_result(figure) for figure in result.get("figures", [])],
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
    }


def _family_manifest(desc: str, *, source_table_relpath: str, note: str, rows: Sequence[Mapping[str, Any]], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "table_family": desc,
        "description": note,
        "independent_unit": "participant",
        "source_table_relpath": source_table_relpath,
        "row_count": len(rows),
        "recomputed_mvpa_distances": False,
        **dict(extra or {}),
    }


def _default_figure_desc(kind: str) -> str:
    kind = _normalize_figure_kind(kind)
    return {
        "roi_group_forest": "SubjectLevelROIGroupInferenceForest",
        "roi_group_violin_dot": "SubjectLevelROIGroupInferenceViolinDot",
        "phase_pooled_violin_dot": "SubjectLevelPhasePooledGroupInferenceViolinDot",
    }.get(kind, kind)


def _normalize_figure_kind(kind: str) -> str:
    aliases = {
        "roi_forest": "roi_group_forest",
        "roi_violin_dot": "roi_group_violin_dot",
        "phase_violin_dot": "phase_pooled_violin_dot",
    }
    return aliases.get(kind, kind)


def _attach_audit_metadata(subject_rows: Sequence[Mapping[str, str]], audit_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    lookup = {
        tuple(row.get(column, "") for column in ("participant_id", "analysis_variant", "phase_id", "roi_label", "contrast_id")): row
        for row in audit_rows
    }
    output: list[dict[str, str]] = []
    for row in subject_rows:
        merged = dict(row)
        audit = lookup.get(tuple(row.get(column, "") for column in ("participant_id", "analysis_variant", "phase_id", "roi_label", "contrast_id")), {})
        merged["family_id"] = row.get("family_id") or audit.get("family_id") or row.get("analysis_variant", "")
        output.append(merged)
    return output


def _filter_rows(rows: Sequence[Mapping[str, Any]], filters: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        keep = True
        for key, expected in filters.items():
            if expected is None:
                continue
            if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
                if row.get(str(key)) not in {str(value) for value in expected}:
                    keep = False
                    break
            elif str(row.get(str(key), "")) != str(expected):
                keep = False
                break
        if keep:
            output.append(dict(row))
    return output


def _group_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in columns)
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def _ordered_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in columns} for row in rows]


def _empty_stats() -> dict[str, Any]:
    return {key: "" for key in ("N", "mean_crossnobis", "SD", "variance", "SEM", "ci_low", "ci_high", "median", "q1", "q3", "iqr", "min", "max", "percent_positive", "t", "df", "p_two", "p_one_greater", "p_signflip", "p_wilcoxon", "q_FDR", "dz", "bootstrap_ci_low", "bootstrap_ci_high", "fdr_family", "signflip_method", "ci_method", "bootstrap_method")}


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quartiles(values: Sequence[float]) -> tuple[float | str, float | str, float | str]:
    if not values:
        return "", "", ""
    ordered = sorted(values)
    return _percentile(ordered, 0.25), _percentile(ordered, 0.5), _percentile(ordered, 0.75)


def _percentile(ordered: Sequence[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _bootstrap_ci(values: Sequence[float], stats_config: Mapping[str, Any]) -> tuple[float | str, float | str, str]:
    bootstrap = _mapping(stats_config.get("bootstrap"))
    if not bool(bootstrap.get("enabled", False)) or not values:
        return "", "", "disabled"
    draws = int(bootstrap.get("draws", 1000))
    rng = random.Random(int(bootstrap.get("seed", 12345)))
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975), f"percentile_{draws}"


def _min_number(rows: Sequence[Mapping[str, Any]], key: str) -> float | str:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return min(values) if values else ""


def _max_number(rows: Sequence[Mapping[str, Any]], key: str) -> float | str:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else ""


def _mean_number(rows: Sequence[Mapping[str, Any]], key: str) -> float | str:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else ""


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any, *, full_precision: bool = False) -> str:
    numeric = _float(value)
    if numeric is None or not math.isfinite(numeric):
        return ""
    return f"{numeric:.17g}" if full_precision else f"{numeric:.3f}"


def _format_p(value: Any, *, full_precision: bool = False) -> str:
    numeric = _float(value)
    if numeric is None or not math.isfinite(numeric):
        return ""
    return f"{numeric:.17g}" if full_precision else f"{numeric:.3f}"


def _format_ci(low: Any, high: Any, *, full_precision: bool = False) -> str:
    low_text = _format_number(low, full_precision=full_precision)
    high_text = _format_number(high, full_precision=full_precision)
    return f"[{low_text}, {high_text}]" if low_text and high_text else ""


def _display_label(labels: Mapping[str, Any], section: str, value: str) -> str:
    mapping = _mapping(labels.get(section))
    return str(mapping.get(value, value))


def _output_root(payload: Mapping[str, Any], *, publication_set: str, roots: Mapping[str, Path]) -> tuple[Path, str, list[str]]:
    outputs = _mapping(payload.get("outputs"))
    root_ref = _optional_text(outputs.get("root_ref")) or "artifact_root"
    relative = _optional_text(outputs.get("path") or outputs.get("relative_path")) or DEFAULT_OUTPUT_RELATIVE_PATH.format(publication_set=publication_set)
    errors: list[str] = []
    if root_ref not in roots:
        errors.append(f"MVPA publication export output root_ref {root_ref!r} is not known.")
        root = Path(".")
    else:
        root = roots[root_ref]
    if configured_path_is_unsafe(relative):
        errors.append("MVPA publication export output path must be relative.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"MVPA publication export output path resolves outside root_ref {root_ref!r}.")
    return path, relative, errors


def _configured_path(config: Mapping[str, Any], *, roots: Mapping[str, Path], default_root_ref: str) -> tuple[Path, str, list[str]]:
    root_ref = _optional_text(config.get("root_ref")) or default_root_ref
    relative = _optional_text(config.get("path") or config.get("relative_path")) or ""
    errors: list[str] = []
    root = roots.get(root_ref)
    if root is None:
        errors.append(f"MVPA publication root_ref {root_ref!r} is not known.")
        root = Path(".")
    if configured_path_is_unsafe(relative):
        errors.append("MVPA publication paths must be relative.")
    return (root / relative).resolve(), relative, errors


def _entities(payload: Mapping[str, Any]) -> dict[str, str]:
    entities = _mapping(payload.get("entities"))
    session = str(entities.get("session_id") or payload.get("session_id") or "ses-01")
    if not session.startswith("ses-"):
        session = f"ses-{session}"
    return {
        "session_id": session,
        "task_id": str(entities.get("task_id") or payload.get("task_id") or "task"),
        "direction": str(entities.get("direction") or payload.get("direction") or ""),
    }


def _entity_prefix(*, session_id: str, task_id: str, direction: str) -> str:
    parts = [session_id, f"task-{task_id}"]
    if direction:
        parts.append(f"dir-{direction}")
    return "_".join(parts)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _write_tsv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = _columns(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    _write_text_atomic(path, buffer.getvalue())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    if path.suffix in {".tsv", ".json", ".md"} and published_text_contains_local_path_reference(content):
        raise RuntimeError(f"Published text output contains a local absolute path marker: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(str(column))
    return columns or ["empty"]


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("MVPA publication figure exports require matplotlib. Install it locally with: uv add matplotlib or run with: uv run --with matplotlib ...") from exc
    return plt


def _validate_relative_path_value(value: Any, label: str, errors: list[str]) -> None:
    text = _optional_text(value)
    if text is None:
        errors.append(f"{label} must be defined.")
    elif configured_path_is_unsafe(text):
        errors.append(f"{label} must be relative and stay under its source root.")


def _safe_label(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _text_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value)


def _join_relative(root: str, child: str) -> str:
    root = root.strip("/")
    child = child.strip("/")
    return f"{root}/{child}" if root and child else root or child


def _unique(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


__all__ = [
    "plan_or_execute_mvpa_publication_export",
    "validate_mvpa_publication_export_document",
]
