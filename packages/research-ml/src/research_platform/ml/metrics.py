from __future__ import annotations

import math
import random
from typing import Sequence


def mean_squared_error(*, actuals: Sequence[float], predictions: Sequence[float]) -> float:
    _validate_pairs(actuals=actuals, predictions=predictions)
    total_error = 0.0
    for actual, predicted in zip(actuals, predictions, strict=True):
        total_error += (float(actual) - float(predicted)) ** 2
    return total_error / len(actuals)


def r2_score(*, actuals: Sequence[float], predictions: Sequence[float]) -> float:
    _validate_pairs(actuals=actuals, predictions=predictions)
    mean_actual = sum(float(actual) for actual in actuals) / len(actuals)
    residual_sum_squares = 0.0
    total_sum_squares = 0.0
    for actual, predicted in zip(actuals, predictions, strict=True):
        actual_value = float(actual)
        residual_sum_squares += (actual_value - float(predicted)) ** 2
        total_sum_squares += (actual_value - mean_actual) ** 2
    if math.isclose(total_sum_squares, 0.0):
        return 1.0 if math.isclose(residual_sum_squares, 0.0) else 0.0
    return 1.0 - (residual_sum_squares / total_sum_squares)


def bootstrap_r2_confidence_interval(
    *,
    actuals: Sequence[float],
    predictions: Sequence[float],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 23,
) -> dict[str, float | int]:
    _validate_pairs(actuals=actuals, predictions=predictions)
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be at least 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    sample_count = len(actuals)
    randomizer = random.Random(seed)
    estimates: list[float] = []

    for _ in range(iterations):
        sample_indices = [randomizer.randrange(sample_count) for _ in range(sample_count)]
        sample_actuals = [float(actuals[index]) for index in sample_indices]
        sample_predictions = [float(predictions[index]) for index in sample_indices]
        estimates.append(r2_score(actuals=sample_actuals, predictions=sample_predictions))

    estimates.sort()
    alpha = 1.0 - confidence_level
    lower_index = max(0, min(iterations - 1, int(math.floor((alpha / 2.0) * iterations))))
    upper_index = max(0, min(iterations - 1, int(math.ceil((1.0 - (alpha / 2.0)) * iterations)) - 1))
    return {
        "estimate": r2_score(actuals=actuals, predictions=predictions),
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
        "confidence_level": confidence_level,
        "bootstrap_iterations": iterations,
    }


def _validate_pairs(*, actuals: Sequence[float], predictions: Sequence[float]) -> None:
    if not actuals:
        raise ValueError("At least one value is required.")
    if len(actuals) != len(predictions):
        raise ValueError("actuals and predictions must have the same length.")
