
from __future__ import annotations

import base64
import importlib
import math
from pathlib import Path
from typing import Any

from .metrics import mean_squared_error, r2_score


def _sigmoid(value: float) -> float:
    clipped = max(min(value, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def fit_logistic_regression(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[int],
    feature_columns: list[str],
    target_column: str,
    learning_rate: float,
    iterations: int,
    table_path: str | Path,
) -> dict[str, Any]:
    if not feature_rows:
        raise ValueError("Training requires at least one row.")

    weights = [0.0 for _ in feature_columns]
    intercept = 0.0

    for _ in range(iterations):
        weight_gradient = [0.0 for _ in feature_columns]
        intercept_gradient = 0.0
        for row, target in zip(feature_rows, targets, strict=True):
            features = [float(row[column]) for column in feature_columns]
            prediction = _sigmoid(intercept + sum(weight * value for weight, value in zip(weights, features, strict=True)))
            error = prediction - target
            intercept_gradient += error
            for feature_index, value in enumerate(features):
                weight_gradient[feature_index] += error * value

        scale = learning_rate / len(feature_rows)
        intercept -= scale * intercept_gradient
        weights = [weight - scale * gradient for weight, gradient in zip(weights, weight_gradient, strict=True)]

    metrics = _training_metrics(
        feature_rows=feature_rows,
        targets=targets,
        weights=weights,
        intercept=intercept,
        feature_columns=feature_columns,
    )
    return {
        "kind": "logistic_regression",
        "table_path": str(Path(table_path)),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "intercept": intercept,
        "weights": {column: weight for column, weight in zip(feature_columns, weights, strict=True)},
        "training_metrics": metrics,
    }


def _training_metrics(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[int],
    weights: list[float],
    intercept: float,
    feature_columns: list[str],
) -> dict[str, float]:
    correct = 0
    for row, target in zip(feature_rows, targets, strict=True):
        probability = predict_probability(
            row=row,
            model={
                "feature_columns": feature_columns,
                "intercept": intercept,
                "weights": dict(zip(feature_columns, weights, strict=True)),
            },
        )
        predicted = 1.0 if probability >= 0.5 else 0.0
        if predicted == target:
            correct += 1
    return {"accuracy": correct / len(feature_rows)}


def predict_probability(*, row: dict[str, float], model: dict[str, Any]) -> float:
    score = float(model["intercept"])
    for column in model["feature_columns"]:
        score += float(model["weights"][column]) * float(row[column])
    return _sigmoid(score)


def fit_regression_model(
    *,
    kind: str,
    feature_rows: list[dict[str, float]],
    targets: list[float],
    feature_columns: list[str],
    target_column: str,
    table_path: str | Path,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    random_state: int = 23,
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    max_depth: int = 6,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
) -> dict[str, Any]:
    if kind == "elastic_net_regression":
        return fit_elastic_net_regression(
            feature_rows=feature_rows,
            targets=targets,
            feature_columns=feature_columns,
            target_column=target_column,
            table_path=table_path,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            random_state=random_state,
        )
    if kind == "xgboost_regression":
        return fit_xgboost_regression(
            feature_rows=feature_rows,
            targets=targets,
            feature_columns=feature_columns,
            target_column=target_column,
            table_path=table_path,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
        )
    raise ValueError(
        f"Unsupported regression kind {kind!r}. Use elastic_net_regression or xgboost_regression."
    )


def fit_elastic_net_regression(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[float],
    feature_columns: list[str],
    target_column: str,
    table_path: str | Path,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    random_state: int = 23,
) -> dict[str, Any]:
    if not feature_rows:
        raise ValueError("Training requires at least one row.")

    try:
        from sklearn.linear_model import ElasticNet
    except ModuleNotFoundError as exc:
        raise RuntimeError("ElasticNet regression requires scikit-learn to be installed.") from exc

    design_matrix = _design_matrix(feature_rows=feature_rows, feature_columns=feature_columns)
    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state)
    estimator.fit(design_matrix, targets)
    predictions = [float(value) for value in estimator.predict(design_matrix)]
    return {
        "kind": "elastic_net_regression",
        "table_path": str(Path(table_path)),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "max_iter": max_iter,
        "random_state": random_state,
        "intercept": float(estimator.intercept_),
        "coefficients": {
            column: float(coefficient) for column, coefficient in zip(feature_columns, estimator.coef_, strict=True)
        },
        "training_metrics": _regression_training_metrics(targets=targets, predictions=predictions),
    }


def fit_xgboost_regression(
    *,
    feature_rows: list[dict[str, float]],
    targets: list[float],
    feature_columns: list[str],
    target_column: str,
    table_path: str | Path,
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    max_depth: int = 6,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    random_state: int = 23,
) -> dict[str, Any]:
    if not feature_rows:
        raise ValueError("Training requires at least one row.")

    xgboost = _require_xgboost()
    design_matrix = _design_matrix(feature_rows=feature_rows, feature_columns=feature_columns)
    estimator = xgboost.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
    )
    estimator.fit(design_matrix, targets)
    predictions = [float(value) for value in estimator.predict(design_matrix)]
    booster_bytes = estimator.get_booster().save_raw()
    return {
        "kind": "xgboost_regression",
        "table_path": str(Path(table_path)),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "booster": base64.b64encode(booster_bytes).decode("ascii"),
        "training_metrics": _regression_training_metrics(targets=targets, predictions=predictions),
    }


def predict_regression(*, row: dict[str, float], model: dict[str, Any]) -> float:
    if model["kind"] == "elastic_net_regression":
        score = float(model["intercept"])
        for column in model["feature_columns"]:
            score += float(model["coefficients"][column]) * float(row[column])
        return score
    if model["kind"] == "xgboost_regression":
        xgboost = _require_xgboost()
        booster = xgboost.Booster()
        booster.load_model(bytearray(base64.b64decode(model["booster"])))
        features = [[float(row[column]) for column in model["feature_columns"]]]
        matrix = xgboost.DMatrix(features, feature_names=list(model["feature_columns"]))
        prediction = booster.predict(matrix)
        return float(prediction[0])
    raise ValueError(f"Unsupported regression model kind {model['kind']!r}.")


def _design_matrix(*, feature_rows: list[dict[str, float]], feature_columns: list[str]) -> list[list[float]]:
    return [[float(row[column]) for column in feature_columns] for row in feature_rows]


def _regression_training_metrics(*, targets: list[float], predictions: list[float]) -> dict[str, float | int]:
    return {
        "r2": r2_score(actuals=targets, predictions=predictions),
        "mse": mean_squared_error(actuals=targets, predictions=predictions),
        "train_count": len(targets),
    }


def _require_xgboost() -> Any:
    try:
        return importlib.import_module("xgboost")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "XGBoost regression requires the optional xgboost dependency. Install research-ml[xgboost] or xgboost."
        ) from exc
