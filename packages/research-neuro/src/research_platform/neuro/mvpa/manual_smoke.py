"""Read-only manual diagonal crossnobis smoke/equivalence helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import csv
import json
import math

import numpy as np

from research_platform.analysis.mvpa.manual_crossnobis import (
    ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
    ManualCrossnobisRun,
    compute_manual_diagonal_crossnobis_v1,
)
from research_platform.neuro.fsl.feat_design import (
    map_conditions_to_pe_numbers,
    parse_fsl_design_file,
)
from research_platform.neuro.nifti import load_nifti_image, validate_compatible_geometry


def compute_manual_crossnobis_smoke(
    *,
    subject: str,
    roi_mask_path: str | Path,
    feat_runs: Mapping[str, str | Path],
    condition_a: str,
    condition_b: str,
    condition_a_aliases: Sequence[str] = (),
    condition_b_aliases: Sequence[str] = (),
    phase: str | None = None,
    roi_label: str | None = None,
    design_files: Mapping[str, str | Path] | None = None,
    event_files: Mapping[tuple[str, str], str | Path] | None = None,
    event_pattern: str | None = None,
    min_events: int = 1,
    min_valid_voxels: int = 1,
    excluded_runs: Sequence[str] = (),
    reference_tsv: str | Path | None = None,
    reference_column: str = "crossnobis",
    reference_subject_column: str = "subject_id",
    reference_roi_column: str = "roi_label",
    reference_phase_column: str = "phase",
    tolerance: float = 1e-8,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute one subject/ROI/condition-pair read-only smoke estimate."""

    mask_path = Path(roi_mask_path)
    mask_image = load_nifti_image(mask_path)
    mask = np.asarray(mask_image.get_fdata()) != 0
    mask_flat = mask.ravel(order="C")
    n_voxels_raw = int(np.count_nonzero(mask_flat))
    if n_voxels_raw < 1:
        payload = {
            "estimator": ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
            "status": "empty_roi_mask",
            "subject": subject,
            "phase": phase,
            "roi_label": roi_label or mask_path.stem,
            "crossnobis": None,
            "n_voxels_raw": 0,
            "n_voxels_used": 0,
            "errors": [f"ROI mask has zero nonzero voxels: {mask_path}"],
        }
        _write_optional_output(payload, output_path)
        return payload

    design_files = dict(design_files or {})
    event_files = dict(event_files or {})
    excluded = {_normalize_run_id(run_id) for run_id in excluded_runs}
    event_counts: dict[str, dict[str, int | None]] = {}
    pe_mapping_summary: list[dict[str, Any]] = []
    manual_runs: list[ManualCrossnobisRun] = []

    for raw_run_id, raw_feat_dir in feat_runs.items():
        run_id = _normalize_run_id(raw_run_id)
        feat_dir = Path(raw_feat_dir)
        if run_id in excluded:
            manual_runs.append(
                ManualCrossnobisRun(
                    run_id=run_id,
                    condition_a=[0.0] * n_voxels_raw,
                    condition_b=[0.0] * n_voxels_raw,
                    sigma_squared=[1.0] * n_voxels_raw,
                    excluded=True,
                    exclusion_reason="configured_run_exclusion",
                )
            )
            pe_mapping_summary.append({"run_id": run_id, "status": "excluded"})
            continue

        count_a = _event_count_for(
            run_id,
            condition_a,
            feat_dir=feat_dir,
            event_files=event_files,
            event_pattern=event_pattern,
            subject=subject,
            phase=phase,
        )
        count_b = _event_count_for(
            run_id,
            condition_b,
            feat_dir=feat_dir,
            event_files=event_files,
            event_pattern=event_pattern,
            subject=subject,
            phase=phase,
        )
        event_counts[run_id] = {condition_a: count_a, condition_b: count_b}
        if count_a is None or count_b is None or count_a < min_events or count_b < min_events:
            manual_runs.append(
                ManualCrossnobisRun(
                    run_id=run_id,
                    condition_a=[0.0] * n_voxels_raw,
                    condition_b=[0.0] * n_voxels_raw,
                    sigma_squared=[1.0] * n_voxels_raw,
                    event_count_a=count_a,
                    event_count_b=count_b,
                )
            )
            pe_mapping_summary.append({"run_id": run_id, "status": "invalid_events"})
            continue

        design_path = Path(design_files.get(run_id, feat_dir / "design.fsf"))
        mapping = _condition_mapping(
            design_path,
            condition_a=condition_a,
            condition_b=condition_b,
            condition_a_aliases=condition_a_aliases,
            condition_b_aliases=condition_b_aliases,
        )
        pe_mapping_summary.append({"run_id": run_id, "design_fsf": design_path.as_posix(), **mapping["summary"]})
        pe_a = _load_masked_image(feat_dir / "stats" / f"pe{mapping['pe_a']}.nii.gz", mask_image, mask_flat)
        pe_b = _load_masked_image(feat_dir / "stats" / f"pe{mapping['pe_b']}.nii.gz", mask_image, mask_flat)
        sigma = _load_masked_image(feat_dir / "stats" / "sigmasquareds.nii.gz", mask_image, mask_flat)
        manual_runs.append(
            ManualCrossnobisRun(
                run_id=run_id,
                condition_a=pe_a,
                condition_b=pe_b,
                sigma_squared=sigma,
                event_count_a=count_a,
                event_count_b=count_b,
            )
        )

    result = compute_manual_diagonal_crossnobis_v1(
        manual_runs,
        min_events=min_events,
        min_valid_voxels=min_valid_voxels,
    )
    payload: dict[str, Any] = {
        "estimator": ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        "status": result.status,
        "subject": subject,
        "phase": phase,
        "roi_label": roi_label or mask_path.stem,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "min_events": min_events,
        "crossnobis": result.crossnobis,
        "n_valid_runs": result.n_valid_runs,
        "valid_runs": list(result.valid_runs),
        "excluded_runs": list(result.excluded_runs),
        "invalid_runs": list(result.invalid_runs),
        "n_voxels_raw": result.n_voxels_raw,
        "n_voxels_used": result.n_voxels_used,
        "event_counts": event_counts,
        "pe_mapping_summary": pe_mapping_summary,
        "sigma_pooling_source_runs": list(result.sigma_pooling_source_runs),
        "run_pair_details": [row.to_dict() for row in result.run_pair_details],
        "run_qc": [row.to_dict() for row in result.run_qc],
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }
    if reference_tsv is not None:
        payload["reference_comparison"] = _reference_comparison(
            reference_tsv,
            computed=result.crossnobis,
            subject=subject,
            roi_label=str(payload["roi_label"]),
            phase=phase,
            reference_column=reference_column,
            reference_subject_column=reference_subject_column,
            reference_roi_column=reference_roi_column,
            reference_phase_column=reference_phase_column,
            tolerance=tolerance,
        )
    _write_optional_output(payload, output_path)
    return payload


def _condition_mapping(
    design_path: Path,
    *,
    condition_a: str,
    condition_b: str,
    condition_a_aliases: Sequence[str],
    condition_b_aliases: Sequence[str],
) -> dict[str, Any]:
    design = parse_fsl_design_file(design_path)
    mapping = map_conditions_to_pe_numbers(
        [
            {"id": condition_a, "aliases": _aliases(condition_a, condition_a_aliases)},
            {"id": condition_b, "aliases": _aliases(condition_b, condition_b_aliases)},
        ],
        design,
        case_sensitive=False,
    )
    if mapping.status == "error":
        raise ValueError(f"Could not map conditions to FEAT PEs for {design_path}: {'; '.join(mapping.errors)}")
    by_condition = {row.condition_id: row for row in mapping.mappings}
    row_a = by_condition[condition_a]
    row_b = by_condition[condition_b]
    if row_a.pe_number is None or row_b.pe_number is None:
        raise ValueError(f"Missing PE number in FEAT mapping for {design_path}.")
    return {
        "pe_a": row_a.pe_number,
        "pe_b": row_b.pe_number,
        "summary": {
            "status": mapping.status,
            "condition_a": {
                "condition_id": condition_a,
                "matched_ev_title": row_a.matched_ev_title,
                "ev_index": row_a.ev_index,
                "pe_number": row_a.pe_number,
            },
            "condition_b": {
                "condition_id": condition_b,
                "matched_ev_title": row_b.matched_ev_title,
                "ev_index": row_b.ev_index,
                "pe_number": row_b.pe_number,
            },
            "design_warnings": list(design.warnings),
        },
    }


def _load_masked_image(path: Path, mask_image: Any, mask_flat: np.ndarray) -> tuple[float, ...]:
    image = load_nifti_image(path)
    validate_compatible_geometry(mask_image, image)
    values = np.asarray(image.get_fdata(), dtype=float).ravel(order="C")[mask_flat]
    return tuple(float(value) for value in values)


def _event_count_for(
    run_id: str,
    condition_id: str,
    *,
    feat_dir: Path,
    event_files: Mapping[tuple[str, str], str | Path],
    event_pattern: str | None,
    subject: str,
    phase: str | None,
) -> int | None:
    path = event_files.get((run_id, condition_id))
    if path is None and event_pattern:
        rendered = event_pattern.format(
            subject=subject,
            subject_id=subject.removeprefix("sub-"),
            run=run_id,
            run_id=run_id,
            condition=condition_id,
            condition_id=condition_id,
            phase=phase or "",
            feat_dir=feat_dir.as_posix(),
        )
        candidate = Path(rendered)
        path = candidate if candidate.is_absolute() else feat_dir / candidate
    if path is None:
        return None
    event_path = Path(path)
    if not event_path.exists():
        return None
    return sum(1 for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))


def _reference_comparison(
    path: str | Path,
    *,
    computed: float | None,
    subject: str,
    roi_label: str,
    phase: str | None,
    reference_column: str,
    reference_subject_column: str,
    reference_roi_column: str,
    reference_phase_column: str,
    tolerance: float,
) -> dict[str, Any]:
    row = _matching_reference_row(
        path,
        subject=subject,
        roi_label=roi_label,
        phase=phase,
        subject_column=reference_subject_column,
        roi_column=reference_roi_column,
        phase_column=reference_phase_column,
    )
    if row is None:
        return {
            "status": "reference_row_not_found",
            "computed_crossnobis": computed,
            "reference_crossnobis": None,
            "absolute_difference": None,
            "tolerance": tolerance,
            "passed": False,
        }
    reference_value = _reference_value(row, reference_column)
    difference = None if computed is None or reference_value is None else abs(computed - reference_value)
    return {
        "status": "ok" if difference is not None and difference <= tolerance else "failed",
        "computed_crossnobis": computed,
        "reference_crossnobis": reference_value,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "passed": difference is not None and difference <= tolerance,
    }


def _matching_reference_row(
    path: str | Path,
    *,
    subject: str,
    roi_label: str,
    phase: str | None,
    subject_column: str,
    roi_column: str,
    phase_column: str,
) -> dict[str, str] | None:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        if subject_column in row and row[subject_column] not in {subject, subject.removeprefix("sub-")}:
            continue
        if roi_column in row and row[roi_column] != roi_label:
            continue
        if phase is not None and phase_column in row and row[phase_column] != phase:
            continue
        return row
    return rows[0] if rows else None


def _reference_value(row: Mapping[str, str], reference_column: str) -> float | None:
    raw = row.get(reference_column)
    if raw is None and reference_column == "crossnobis":
        raw = row.get("distance")
    try:
        value = float(raw) if raw is not None else math.nan
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _aliases(condition_id: str, configured_aliases: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((condition_id, *configured_aliases)))


def _normalize_run_id(value: str) -> str:
    text = str(value).strip()
    if ":" in text:
        text = text.split(":")[-1]
    return text.removeprefix("run-")


def _write_optional_output(payload: Mapping[str, Any], output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["compute_manual_crossnobis_smoke"]
