from __future__ import annotations

import importlib
import math

import pytest

from research_platform.ml.evaluate import evaluate_elastic_net_regression
from research_platform.ml.metrics import bootstrap_r2_confidence_interval, mean_squared_error, r2_score
from research_platform.ml.train import fit_elastic_net_regression, fit_xgboost_regression


def test_regression_metrics_helpers() -> None:
    actuals = [1.0, 2.0, 3.0, 4.0]
    predictions = [1.0, 2.0, 3.0, 5.0]

    assert math.isclose(mean_squared_error(actuals=actuals, predictions=predictions), 0.25)
    assert math.isclose(r2_score(actuals=actuals, predictions=predictions), 0.8)


def test_bootstrap_r2_confidence_interval() -> None:
    actuals = [2.0, 4.0, 6.0, 8.0, 10.0]
    predictions = [1.8, 4.2, 5.9, 8.1, 10.2]

    interval = bootstrap_r2_confidence_interval(actuals=actuals, predictions=predictions, iterations=200, seed=13)

    assert interval["lower"] <= interval["estimate"] <= interval["upper"]
    assert interval["confidence_level"] == 0.95
    assert interval["bootstrap_iterations"] == 200


def test_elastic_net_regression_train_and_evaluate() -> None:
    feature_rows = [
        {"feature_a": 1.0, "feature_b": 0.0},
        {"feature_a": 2.0, "feature_b": 1.0},
        {"feature_a": 3.0, "feature_b": 0.0},
        {"feature_a": 4.0, "feature_b": 1.0},
        {"feature_a": 5.0, "feature_b": 0.0},
        {"feature_a": 6.0, "feature_b": 1.0},
    ]
    targets = [3.0, 6.0, 7.0, 10.0, 11.0, 14.0]

    model = fit_elastic_net_regression(
        feature_rows=feature_rows,
        targets=targets,
        feature_columns=["feature_a", "feature_b"],
        target_column="score",
        table_path="table.tsv",
        alpha=0.0001,
        l1_ratio=0.2,
        max_iter=5000,
        random_state=23,
    )
    report = evaluate_elastic_net_regression(
        feature_rows=feature_rows,
        targets=targets,
        model=model,
        target_column="score",
        table_path="table.tsv",
        bootstrap_iterations=50,
        bootstrap_seed=5,
    )

    assert model["kind"] == "elastic_net_regression"
    assert model["training_metrics"]["r2"] > 0.999
    assert report["kind"] == "elastic_net_regression_evaluation"
    assert report["metrics"]["r2"] > 0.999
    assert report["metrics"]["mse"] < 0.001


def test_xgboost_regression_fails_clearly_when_dependency_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = importlib.import_module

    def _missing_xgboost(name: str, package: str | None = None) -> object:
        if name == "xgboost":
            raise ModuleNotFoundError("No module named 'xgboost'")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _missing_xgboost)

    with pytest.raises(RuntimeError, match="XGBoost regression requires the optional xgboost dependency"):
        fit_xgboost_regression(
            feature_rows=[{"feature_a": 1.0}],
            targets=[1.0],
            feature_columns=["feature_a"],
            target_column="score",
            table_path="table.tsv",
        )
