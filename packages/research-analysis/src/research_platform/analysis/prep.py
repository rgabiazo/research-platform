from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ._tabular import format_float, numeric_value
from .splits import split_membership


def fit_standardization_plan(
    *,
    rows: list[dict[str, str]],
    feature_columns: list[str],
    target_column: str,
    split_manifest: dict[str, Any],
    table_path: str | Path,
) -> dict[str, Any]:
    membership = split_membership(split_manifest)
    train_rows = [rows[index] for index, split in membership.items() if split == "train"]
    if not train_rows:
        raise ValueError("Split manifest does not contain any training rows.")

    statistics: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        values = [numeric_value(row[column], column=column, row_number=index) for index, row in enumerate(train_rows, start=1)]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance) or 1.0
        statistics[column] = {"mean": mean, "std": std}

    return {
        "kind": "standardize_numeric",
        "table_path": str(Path(table_path)),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "statistics": statistics,
    }


def apply_standardization_plan(
    *,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    plan: dict[str, Any],
    split_manifest: dict[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    output_rows: list[dict[str, str]] = []
    membership = split_membership(split_manifest) if split_manifest else {}

    output_fields = list(fieldnames)
    if split_manifest and "split_set" not in output_fields:
        output_fields.append("split_set")

    for row_number, row in enumerate(rows):
        transformed = dict(row)
        for column in plan["feature_columns"]:
            stats = plan["statistics"][column]
            value = numeric_value(row[column], column=column, row_number=row_number + 1)
            transformed[column] = format_float((value - float(stats["mean"])) / float(stats["std"]))
        if split_manifest:
            transformed["split_set"] = membership.get(row_number, "")
        output_rows.append(transformed)

    return output_fields, output_rows
