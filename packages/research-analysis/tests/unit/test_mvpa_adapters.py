from __future__ import annotations

from collections.abc import Iterable

from research_platform.analysis.mvpa import (
    METRIC_CROSSNOBIS,
    DistanceEngine,
    DistanceEstimate,
    DistanceRequest,
    PatternDataset,
    pattern_dataset_from_rows,
)


class FakeDistanceEngine:
    name = "fake"
    supported_metrics = {METRIC_CROSSNOBIS}

    def compute_distances(
        self,
        dataset: PatternDataset,
        request: DistanceRequest,
    ) -> Iterable[DistanceEstimate]:
        assert len(dataset.observations) == 4
        assert request.metric in self.supported_metrics
        return [
            DistanceEstimate(
                condition_id_a="a",
                condition_id_b="b",
                distance=1.25,
                metric=request.metric,
                engine_name=self.name,
            )
        ]


def test_distance_engine_protocol_accepts_fake_engine_without_external_imports() -> None:
    dataset = pattern_dataset_from_rows(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 2},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 3},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 4},
        ],
        feature_columns=["f1"],
    )
    engine: DistanceEngine = FakeDistanceEngine()
    request = DistanceRequest(metric=METRIC_CROSSNOBIS, condition_order=["a", "b"], engine_name=engine.name)

    assert isinstance(engine, DistanceEngine)
    estimates = list(engine.compute_distances(dataset, request))
    assert estimates == [
        DistanceEstimate(
            condition_id_a="a",
            condition_id_b="b",
            distance=1.25,
            metric=METRIC_CROSSNOBIS,
            engine_name="fake",
        )
    ]
