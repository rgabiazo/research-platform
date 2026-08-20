from __future__ import annotations

import math

import pytest

from research_platform.analysis.mvpa import (
    METRIC_CORRELATION,
    METRIC_CROSSNOBIS,
    METRIC_EUCLIDEAN,
    NOISE_NORMALIZATION_DIAGONAL,
    NativeReferenceDistanceEngine,
    NoiseNormalization,
    DistanceRequest,
    pattern_dataset_from_rows,
)


def _dataset(rows: list[dict[str, object]], feature_columns: list[str] | None = None):
    return pattern_dataset_from_rows(rows, feature_columns=feature_columns or ["f1", "f2"])


def test_crossnobis_identity_normalization() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 2},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 3, "f2": 4},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 0, "f2": 0},
        ]
    )
    engine = NativeReferenceDistanceEngine()

    estimates = engine.compute_distances(dataset, DistanceRequest(metric=METRIC_CROSSNOBIS))

    assert engine.name == "native_reference"
    assert {METRIC_CROSSNOBIS, METRIC_EUCLIDEAN, METRIC_CORRELATION} <= engine.supported_metrics
    assert len(estimates) == 1
    estimate = estimates[0]
    assert estimate.condition_id_a == "a"
    assert estimate.condition_id_b == "b"
    assert estimate.distance == pytest.approx(11.0)
    assert estimate.metric == METRIC_CROSSNOBIS
    assert estimate.engine_name == "native_reference"
    assert estimate.cv_unit_count == 2
    assert estimate.context == {
        "condition_index_a": 0,
        "condition_index_b": 1,
        "feature_count": 2,
        "observation_count": 4,
    }


def test_negative_crossnobis_remains_valid() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 0},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": -2, "f2": 0},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 0, "f2": 0},
        ]
    )

    estimates = NativeReferenceDistanceEngine().compute_distances(
        dataset,
        DistanceRequest(metric=METRIC_CROSSNOBIS),
    )

    assert estimates[0].distance == pytest.approx(-2.0)


def test_multiple_observations_are_averaged_before_crossnobis() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 1},
            {"pattern_id": "p2", "condition_id": "a", "cv_unit": "run-1", "f1": 3, "f2": 3},
            {"pattern_id": "p3", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p4", "condition_id": "a", "cv_unit": "run-2", "f1": 4, "f2": 0},
            {"pattern_id": "p5", "condition_id": "b", "cv_unit": "run-2", "f1": 1, "f2": 0},
        ]
    )

    estimates = NativeReferenceDistanceEngine().compute_distances(
        dataset,
        DistanceRequest(metric=METRIC_CROSSNOBIS),
    )

    assert estimates[0].distance == pytest.approx(6.0)
    assert estimates[0].context["observation_count"] == 5


def test_euclidean_condition_means() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 2},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 4, "f2": 6},
        ]
    )

    estimates = NativeReferenceDistanceEngine().compute_distances(
        dataset,
        DistanceRequest(metric=METRIC_EUCLIDEAN),
    )

    assert estimates[0].distance == pytest.approx(5.0)
    assert estimates[0].cv_unit_count == 1
    assert estimates[0].context["feature_count"] == 2
    assert estimates[0].context["observation_count"] == 2


def test_correlation_distance() -> None:
    dataset = _dataset(
        [
            {
                "pattern_id": "p1",
                "condition_id": "a",
                "cv_unit": "run-1",
                "f1": 1,
                "f2": 2,
                "f3": 3,
            },
            {
                "pattern_id": "p2",
                "condition_id": "b",
                "cv_unit": "run-1",
                "f1": 1,
                "f2": 3,
                "f3": 2,
            },
        ],
        feature_columns=["f1", "f2", "f3"],
    )

    estimates = NativeReferenceDistanceEngine().compute_distances(
        dataset,
        DistanceRequest(metric=METRIC_CORRELATION),
    )

    assert estimates[0].distance == pytest.approx(0.5)


def test_correlation_constant_vector_failure() -> None:
    dataset = _dataset(
        [
            {
                "pattern_id": "p1",
                "condition_id": "a",
                "cv_unit": "run-1",
                "f1": 1,
                "f2": 1,
                "f3": 1,
            },
            {
                "pattern_id": "p2",
                "condition_id": "b",
                "cv_unit": "run-1",
                "f1": 1,
                "f2": 2,
                "f3": 3,
            },
        ],
        feature_columns=["f1", "f2", "f3"],
    )

    with pytest.raises(ValueError, match="constant"):
        NativeReferenceDistanceEngine().compute_distances(
            dataset,
            DistanceRequest(metric=METRIC_CORRELATION),
        )


def test_deterministic_condition_ordering() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "b", "cv_unit": "run-1", "f1": 1, "f2": 0},
            {"pattern_id": "p2", "condition_id": "a", "cv_unit": "run-1", "f1": 4, "f2": 0},
        ]
    )
    engine = NativeReferenceDistanceEngine()

    default_estimate = engine.compute_distances(dataset, DistanceRequest(metric=METRIC_EUCLIDEAN))[0]
    explicit_estimate = engine.compute_distances(
        dataset,
        DistanceRequest(metric=METRIC_EUCLIDEAN, condition_order=["a", "b"]),
    )[0]

    assert (default_estimate.condition_id_a, default_estimate.condition_id_b) == ("b", "a")
    assert (explicit_estimate.condition_id_a, explicit_estimate.condition_id_b) == ("a", "b")


def test_diagonal_crossnobis() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 2, "f2": 4},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 6, "f2": 8},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 0, "f2": 0},
        ]
    )
    request = DistanceRequest(
        metric=METRIC_CROSSNOBIS,
        noise_normalization=NoiseNormalization(method=NOISE_NORMALIZATION_DIAGONAL),
    )

    estimates = NativeReferenceDistanceEngine(diagonal_variances=[2, 4]).compute_distances(dataset, request)

    assert estimates[0].distance == pytest.approx(14.0)
    assert estimates[0].normalization_method == NOISE_NORMALIZATION_DIAGONAL


@pytest.mark.parametrize(
    "variances,match",
    [
        (None, "required"),
        ([1.0], "length"),
        ([0.0, 1.0], "positive finite"),
        ([-1.0, 1.0], "positive finite"),
        ([math.nan, 1.0], "positive finite"),
        ([math.inf, 1.0], "positive finite"),
    ],
)
def test_diagonal_variance_validation(variances: list[float] | None, match: str) -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 0},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 2, "f2": 0},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 0, "f2": 0},
        ]
    )
    request = DistanceRequest(
        metric=METRIC_CROSSNOBIS,
        noise_normalization=NoiseNormalization(method=NOISE_NORMALIZATION_DIAGONAL),
    )

    with pytest.raises(ValueError, match=match):
        NativeReferenceDistanceEngine(diagonal_variances=variances).compute_distances(dataset, request)


@pytest.mark.parametrize("metric", [METRIC_EUCLIDEAN, METRIC_CORRELATION])
def test_diagonal_normalization_is_rejected_for_non_crossnobis_metrics(metric: str) -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 2},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 2, "f2": 1},
        ]
    )

    with pytest.raises(ValueError, match="only supported for crossnobis"):
        NativeReferenceDistanceEngine(diagonal_variances=[1, 1]).compute_distances(
            dataset,
            DistanceRequest(
                metric=metric,
                noise_normalization=NoiseNormalization(method=NOISE_NORMALIZATION_DIAGONAL),
            ),
        )


def test_unsupported_metrics_raise_value_error() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 2},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 3, "f2": 4},
        ]
    )

    with pytest.raises(ValueError, match="Unsupported distance metric"):
        NativeReferenceDistanceEngine().compute_distances(dataset, DistanceRequest(metric="manhattan"))
