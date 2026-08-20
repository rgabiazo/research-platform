from __future__ import annotations

import pytest

from research_platform.analysis.mvpa import (
    ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
    ManualCrossnobisRun,
    NOISE_NONPOSITIVE_POLICY_DROP_FEATURES,
    compute_manual_diagonal_crossnobis_v1,
)


def _run(
    run_id: str,
    diff: list[float],
    *,
    sigma: list[float] | None = None,
    events_a: int | None = 3,
    events_b: int | None = 3,
    excluded: bool = False,
) -> ManualCrossnobisRun:
    return ManualCrossnobisRun(
        run_id=run_id,
        condition_a=diff,
        condition_b=[0.0] * len(diff),
        sigma_squared=sigma or [1.0] * len(diff),
        event_count_a=events_a,
        event_count_b=events_b,
        excluded=excluded,
        exclusion_reason="configured exclusion" if excluded else None,
    )


def test_manual_estimator_name_is_explicit() -> None:
    assert ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1 == "manual_diagonal_crossnobis_v1"


def test_two_run_case_equals_single_run_pair_dot_divided_by_voxels() -> None:
    result = compute_manual_diagonal_crossnobis_v1(
        [
            _run("run-1", [1.0, -1.0]),
            _run("run-2", [1.0, -1.0]),
        ],
        min_events=1,
    )

    assert result.status == "ok"
    assert result.n_voxels_used == 2
    assert len(result.run_pair_details) == 1
    assert result.run_pair_details[0].dot_product == pytest.approx(2.0)
    assert result.run_pair_details[0].distance == pytest.approx(1.0)
    assert result.crossnobis == pytest.approx(1.0)


def test_three_run_case_averages_all_unordered_run_pair_dot_over_voxels() -> None:
    result = compute_manual_diagonal_crossnobis_v1(
        [
            _run("run-1", [1.0, 2.0]),
            _run("run-2", [3.0, 4.0]),
            _run("run-3", [5.0, 8.0]),
        ],
        min_events=1,
    )

    pair_distances = [pair.distance for pair in result.run_pair_details]
    assert pair_distances == pytest.approx([0.25, 0.75, 0.75])
    assert result.crossnobis == pytest.approx(sum(pair_distances) / 3.0)


def test_min_events_filtering_happens_before_sigma_pooling_and_run_pairs() -> None:
    result = compute_manual_diagonal_crossnobis_v1(
        [
            _run("run-1", [1.0, 3.0], sigma=[1.0, 1.0], events_a=3, events_b=3),
            _run("run-2", [1.0, 3.0], sigma=[1.0, 1.0], events_a=3, events_b=3),
            _run("run-3", [100.0, 300.0], sigma=[10000.0, 10000.0], events_a=3, events_b=2),
        ],
        min_events=3,
    )

    assert result.status == "ok"
    assert result.valid_runs == ("run-1", "run-2")
    assert result.invalid_runs == ("run-3",)
    assert result.sigma_pooling_source_runs == ("run-1", "run-2")
    assert [(pair.run_id_i, pair.run_id_j) for pair in result.run_pair_details] == [("run-1", "run-2")]
    assert result.crossnobis == pytest.approx(1.0)


def test_configured_run_exclusion_is_not_used_for_events_sigma_or_run_pairs() -> None:
    result = compute_manual_diagonal_crossnobis_v1(
        [
            _run("run-1", [1.0, 3.0], sigma=[1.0, 1.0]),
            _run("run-2", [1.0, 3.0], sigma=[1.0, 1.0]),
            _run("run-3", [100.0, 300.0], sigma=[10000.0, 10000.0], excluded=True),
        ],
        min_events=3,
    )

    assert result.status == "ok"
    assert result.valid_runs == ("run-1", "run-2")
    assert result.excluded_runs == ("run-3",)
    assert result.sigma_pooling_source_runs == ("run-1", "run-2")
    assert result.n_valid_runs == 2
    assert [(pair.run_id_i, pair.run_id_j) for pair in result.run_pair_details] == [("run-1", "run-2")]
    assert result.crossnobis == pytest.approx(1.0)


def test_centering_occurs_after_diagonal_normalization_with_nonuniform_sigma() -> None:
    result = compute_manual_diagonal_crossnobis_v1(
        [
            _run("run-1", [1.0, 5.0], sigma=[1.0, 4.0]),
            _run("run-2", [3.0, 7.0], sigma=[1.0, 4.0]),
        ],
        min_events=1,
    )

    pre_normalization_centering_value = 2.5
    assert result.crossnobis == pytest.approx(0.1875)
    assert result.crossnobis != pytest.approx(pre_normalization_centering_value)


def test_finite_pe_and_positive_pooled_sigma_define_valid_voxel_count() -> None:
    result = compute_manual_diagonal_crossnobis_v1(
        [
            _run("run-1", [1.0, float("nan"), 3.0], sigma=[1.0, 1.0, 0.0]),
            _run("run-2", [1.0, 2.0, 3.0], sigma=[1.0, 1.0, 0.0]),
        ],
        min_events=1,
        min_valid_voxels=2,
    )

    assert result.status == "insufficient_valid_voxels"
    assert result.crossnobis is None
    assert result.n_voxels_raw == 3
    assert result.n_voxels_used == 1


def test_drop_features_policy_requires_each_run_noise_value_to_be_positive() -> None:
    result = compute_manual_diagonal_crossnobis_v1(
        [
            _run("run-1", [1.0, 5.0], sigma=[1.0, 0.0]),
            _run("run-2", [3.0, 7.0], sigma=[1.0, 4.0]),
        ],
        min_events=1,
        min_valid_voxels=1,
        noise_nonpositive_policy=NOISE_NONPOSITIVE_POLICY_DROP_FEATURES,
    )

    assert result.status == "ok"
    assert result.n_voxels_raw == 2
    assert result.n_voxels_used == 1
    assert result.retained_feature_count == 1
    assert result.dropped_noise_feature_count == 1
    assert result.dropped_feature_fraction == pytest.approx(0.5)
    assert result.run_qc[0].dropped_noise_feature_count == 1
