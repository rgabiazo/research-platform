"""Generic non-FSL ROI extraction metrics for NIfTI value maps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import math

import numpy as np

from research_platform.neuro.nifti import validate_compatible_geometry
from research_platform.neuro.roi_masks import validate_binary_mask


DEFAULT_EXTRACTION_METRICS = (
    "mean",
    "median",
    "sum",
    "std",
    "min",
    "max",
    "voxel_count",
    "valid_voxel_count",
)


@dataclass(frozen=True)
class ExtractionResult:
    """Metrics and provenance for one ROI/value-map extraction."""

    metrics: Mapping[str, float | int]
    provenance: Mapping[str, Any]


def extract_roi_metrics(
    value_image: Any,
    roi_mask_image: Any,
    *,
    metrics: Sequence[str] = DEFAULT_EXTRACTION_METRICS,
) -> dict[str, float | int]:
    """Compute generic summary metrics from a value map inside an ROI mask."""

    validate_compatible_geometry(value_image, roi_mask_image)
    values = np.asarray(value_image.get_fdata(dtype=np.float64))
    if values.ndim != 3:
        raise ValueError("value_image must be a 3D value map.")

    mask = validate_binary_mask(roi_mask_image.get_fdata(), allow_empty=True, label="roi_mask_image")
    selected = values[mask]
    valid = selected[np.isfinite(selected)]
    voxel_count = int(selected.size)
    valid_voxel_count = int(valid.size)

    requested = tuple(metrics)
    unknown = sorted(set(requested) - set(DEFAULT_EXTRACTION_METRICS))
    if unknown:
        raise ValueError(f"Unsupported extraction metrics: {', '.join(unknown)}.")

    output: dict[str, float | int] = {}
    for metric in requested:
        if metric == "voxel_count":
            output[metric] = voxel_count
        elif metric == "valid_voxel_count":
            output[metric] = valid_voxel_count
        elif valid_voxel_count == 0:
            output[metric] = math.nan
        elif metric == "mean":
            output[metric] = float(np.mean(valid))
        elif metric == "median":
            output[metric] = float(np.median(valid))
        elif metric == "sum":
            output[metric] = float(np.sum(valid))
        elif metric == "std":
            output[metric] = float(np.std(valid))
        elif metric == "min":
            output[metric] = float(np.min(valid))
        elif metric == "max":
            output[metric] = float(np.max(valid))
    return output


def build_extraction_result(
    value_image: Any,
    roi_mask_image: Any,
    *,
    roi_label: str,
    value_desc: str | None = None,
    metrics: Sequence[str] = DEFAULT_EXTRACTION_METRICS,
    provenance: Mapping[str, Any] | None = None,
) -> ExtractionResult:
    """Return metrics with a small JSON-friendly extraction provenance block."""

    metric_values = extract_roi_metrics(value_image, roi_mask_image, metrics=metrics)
    payload: dict[str, Any] = {
        "roi_label": roi_label,
        "backend": "generic_nifti",
        "metrics": list(metrics),
    }
    if value_desc:
        payload["value_desc"] = value_desc
    if provenance:
        payload.update(dict(provenance))
    return ExtractionResult(metrics=metric_values, provenance=payload)


def write_extraction_table(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write ROI extraction rows to TSV or CSV based on the output extension."""

    row_list = [dict(row) for row in rows]
    if not row_list:
        raise ValueError("At least one extraction row is required.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if output.suffix.lower() == ".tsv" else ","
    fieldnames = _ordered_fieldnames(row_list)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(row_list)
    return output


def extraction_result_to_row(result: ExtractionResult) -> dict[str, Any]:
    """Flatten an extraction result into one table row."""

    row = dict(result.provenance)
    row.update(result.metrics)
    return row


def _ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
