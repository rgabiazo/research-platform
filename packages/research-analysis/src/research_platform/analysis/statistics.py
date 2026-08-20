"""Reusable pure-Python statistics helpers for tabular analysis specs."""

from __future__ import annotations

from collections import defaultdict
import math


def numeric_series(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = str(row.get(column, "")).strip()
        if raw:
            values.append(float(raw))
    return values


def paired_numeric_series(rows: list[dict[str, str]], x_column: str, y_column: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        raw_x = str(row.get(x_column, "")).strip()
        raw_y = str(row.get(y_column, "")).strip()
        if not raw_x or not raw_y:
            continue
        x_values.append(float(raw_x))
        y_values.append(float(raw_y))
    return x_values, y_values


def correlation_report(rows: list[dict[str, str]], *, x_column: str, y_column: str, method: str) -> dict[str, object]:
    x_values, y_values = paired_numeric_series(rows, x_column, y_column)
    if len(x_values) < 2:
        raise ValueError("Correlation requires at least two complete numeric pairs.")
    if method == "spearman":
        x_values = _ranks(x_values)
        y_values = _ranks(y_values)
    elif method != "pearson":
        raise ValueError("Correlation method must be pearson or spearman.")
    return {
        "kind": "correlation",
        "method": method,
        "x": x_column,
        "y": y_column,
        "n": len(x_values),
        "r": _pearson(x_values, y_values),
    }


def summary_table_report(rows: list[dict[str, str]], *, columns: list[str]) -> dict[str, object]:
    summaries: dict[str, dict[str, float | int | None]] = {}
    for column in columns:
        values = numeric_series(rows, column)
        summaries[column] = _summary(values)
    return {"kind": "summary_table", "columns": columns, "summaries": summaries}


def linear_model_report(rows: list[dict[str, str]], *, outcome: str, predictors: list[str]) -> dict[str, object]:
    if not predictors:
        raise ValueError("Linear model requires at least one predictor.")
    matrix: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        raw_y = str(row.get(outcome, "")).strip()
        if not raw_y:
            continue
        raw_predictors = [str(row.get(column, "")).strip() for column in predictors]
        if any(not value for value in raw_predictors):
            continue
        targets.append(float(raw_y))
        matrix.append([1.0, *[float(value) for value in raw_predictors]])
    if len(matrix) <= len(predictors):
        raise ValueError("Linear model requires more complete rows than coefficients.")
    coefficients = _least_squares(matrix, targets)
    return {
        "kind": "linear_model",
        "outcome": outcome,
        "predictors": predictors,
        "n": len(targets),
        "coefficients": {"intercept": coefficients[0], **{name: coefficients[index + 1] for index, name in enumerate(predictors)}},
    }


def anova_report(rows: list[dict[str, str]], *, outcome: str, group: str) -> dict[str, object]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = str(row.get(group, "")).strip()
        raw_y = str(row.get(outcome, "")).strip()
        if label and raw_y:
            groups[label].append(float(raw_y))
    groups = {label: values for label, values in groups.items() if values}
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least two groups.")
    all_values = [value for values in groups.values() for value in values]
    grand_mean = _mean(all_values)
    between = sum(len(values) * (_mean(values) - grand_mean) ** 2 for values in groups.values())
    within = sum(sum((value - _mean(values)) ** 2 for value in values) for values in groups.values())
    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    ms_between = between / df_between
    ms_within = within / df_within if df_within else float("nan")
    return {
        "kind": "anova",
        "outcome": outcome,
        "group": group,
        "n": len(all_values),
        "groups": {label: _summary(values) for label, values in groups.items()},
        "f": ms_between / ms_within if ms_within else float("inf"),
        "df_between": df_between,
        "df_within": df_within,
    }


def mixed_effects_report(rows: list[dict[str, str]], *, outcome: str, predictors: list[str], group: str | None) -> dict[str, object]:
    # This v1 framework records a deterministic grouped summary. A full mixed-effects engine can plug into
    # the same analysis spec without changing the public rp command shape.
    report: dict[str, object] = {
        "kind": "mixed_effects",
        "outcome": outcome,
        "predictors": predictors,
        "engine": "summary-only",
        "n": len(numeric_series(rows, outcome)),
    }
    if group:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            label = str(row.get(group, "")).strip()
            raw_y = str(row.get(outcome, "")).strip()
            if label and raw_y:
                grouped[label].append(float(raw_y))
        report["group"] = group
        report["groups"] = {label: _summary(values) for label, values in grouped.items()}
    return report


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": _mean(values),
        "std": _std(values),
        "min": min(values),
        "max": max(values),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if not x_denom or not y_denom:
        raise ValueError("Correlation is undefined for a constant series.")
    return numerator / (x_denom * y_denom)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position
        while end + 1 < len(indexed) and indexed[end + 1][0] == indexed[position][0]:
            end += 1
        rank = (position + end + 2) / 2.0
        for _, original_index in indexed[position : end + 1]:
            ranks[original_index] = rank
        position = end + 1
    return ranks


def _least_squares(matrix: list[list[float]], targets: list[float]) -> list[float]:
    transposed = list(zip(*matrix, strict=True))
    xtx = [[sum(a * b for a, b in zip(row_a, row_b, strict=True)) for row_b in transposed] for row_a in transposed]
    xty = [sum(a * y for a, y in zip(row, targets, strict=True)) for row in transposed]
    return _solve_linear_system(xtx, xty)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("Linear model design matrix is singular.")
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                value - factor * augmented[pivot_index][column_index]
                for column_index, value in enumerate(augmented[row_index])
            ]
    return [row[-1] for row in augmented]
