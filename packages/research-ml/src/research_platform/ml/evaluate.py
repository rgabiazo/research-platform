
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .metrics import bootstrap_r2_confidence_interval, mean_squared_error, r2_score
from .train import predict_probability, predict_regression


def evaluate_logistic_regression(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[int],
    model: dict[str, Any],
    target_column: str,
    table_path: str | Path,
) -> dict[str, Any]:
    if not feature_rows:
        raise ValueError("Evaluation requires at least one row.")

    tp = fp = tn = fn = 0
    log_loss = 0.0
    predictions: list[dict[str, Any]] = []

    for row_number, (row, target) in enumerate(zip(feature_rows, targets, strict=True)):
        probability = predict_probability(row=row, model=model)
        predicted = 1 if probability >= 0.5 else 0
        clipped = min(max(probability, 1e-9), 1 - 1e-9)
        log_loss += -(target * math.log(clipped) + (1 - target) * math.log(1 - clipped))
        predictions.append({"row_number": row_number, "probability": probability, "predicted": predicted, "actual": target})

        if predicted == 1 and target == 1:
            tp += 1
        elif predicted == 1:
            fp += 1
        elif target == 1:
            fn += 1
        else:
            tn += 1

    total = len(feature_rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / total
    return {
        "kind": "logistic_regression_evaluation",
        "table_path": str(Path(table_path)),
        "target_column": target_column,
        "feature_columns": model["feature_columns"],
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "log_loss": log_loss / total,
            "test_count": total,
        },
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "predictions": predictions,
    }


def evaluate_regression_model(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[float],
    model: dict[str, Any],
    target_column: str,
    table_path: str | Path,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 23,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if model["kind"] not in {"elastic_net_regression", "xgboost_regression"}:
        raise ValueError(
            f"Unsupported regression model kind {model['kind']!r}. Use elastic_net_regression or xgboost_regression."
        )
    if model["kind"] == "elastic_net_regression":
        return evaluate_elastic_net_regression(
            feature_rows=feature_rows,
            targets=targets,
            model=model,
            target_column=target_column,
            table_path=table_path,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    return evaluate_xgboost_regression(
        feature_rows=feature_rows,
        targets=targets,
        model=model,
        target_column=target_column,
        table_path=table_path,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


def evaluate_elastic_net_regression(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[float],
    model: dict[str, Any],
    target_column: str,
    table_path: str | Path,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 23,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    return _evaluate_regression_predictions(
        feature_rows=feature_rows,
        targets=targets,
        model=model,
        target_column=target_column,
        table_path=table_path,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


def evaluate_xgboost_regression(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[float],
    model: dict[str, Any],
    target_column: str,
    table_path: str | Path,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 23,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    return _evaluate_regression_predictions(
        feature_rows=feature_rows,
        targets=targets,
        model=model,
        target_column=target_column,
        table_path=table_path,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


def _evaluate_regression_predictions(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[float],
    model: dict[str, Any],
    target_column: str,
    table_path: str | Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not feature_rows:
        raise ValueError("Evaluation requires at least one row.")

    predictions = [predict_regression(row=row, model=model) for row in feature_rows]
    r2_interval = bootstrap_r2_confidence_interval(
        actuals=targets,
        predictions=predictions,
        iterations=bootstrap_iterations,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
    )
    prediction_rows = [
        {
            "row_number": row_number,
            "predicted": predicted,
            "actual": actual,
            "residual": actual - predicted,
        }
        for row_number, (actual, predicted) in enumerate(zip(targets, predictions, strict=True))
    ]
    return {
        "kind": f"{model['kind']}_evaluation",
        "table_path": str(Path(table_path)),
        "target_column": target_column,
        "feature_columns": model["feature_columns"],
        "metrics": {
            "r2": r2_score(actuals=targets, predictions=predictions),
            "mse": mean_squared_error(actuals=targets, predictions=predictions),
            "test_count": len(targets),
            "r2_ci_lower": r2_interval["lower"],
            "r2_ci_upper": r2_interval["upper"],
            "bootstrap_iterations": r2_interval["bootstrap_iterations"],
            "confidence_level": r2_interval["confidence_level"],
        },
        "predictions": prediction_rows,
    }
