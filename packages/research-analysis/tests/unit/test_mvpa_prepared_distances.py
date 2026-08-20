from __future__ import annotations

import ast
import builtins
import json
import math
from pathlib import Path

import pytest

import research_platform.analysis.mvpa.prepared_distances as prepared_distances
from research_platform.analysis.mvpa import (
    DistanceEstimate,
    ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
    METRIC_EUCLIDEAN,
    NOISE_NORMALIZATION_DIAGONAL,
    MvpaPatternRowPreparationQcRow,
    NOISE_NORMALIZATION_IDENTITY,
    PreparedMvpaPatternGroup,
    PreparedMvpaPatternRow,
    compute_mvpa_distances_from_prepared_groups,
    prepare_mvpa_pattern_row_groups,
)


def _row(
    condition_id: str,
    cv_label: str,
    feature_values: list[float],
    *,
    index: int,
    group_key: dict[str, str] | None = None,
    voxel_index_hash: str | None = None,
    noise_values: list[object] | tuple[object, ...] = (),
    noise_usable: bool | None = None,
    noise_feature_count: int | None = None,
    noise_voxel_index_hash: str | None = None,
) -> PreparedMvpaPatternRow:
    return PreparedMvpaPatternRow(
        pattern_id=f"p{index}",
        condition_id=condition_id,
        cv_unit="run",
        cv_label=cv_label,
        feature_values=feature_values,
        feature_count=len(feature_values),
        group_key=group_key or {"subject_id": "sub-01"},
        source_row_index=index,
        voxel_index_hash=voxel_index_hash,
        noise_values=noise_values,
        noise_usable=noise_usable,
        noise_feature_count=noise_feature_count,
        noise_voxel_index_hash=noise_voxel_index_hash,
    )


def _first_seen(values: list[str]) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for value in values:
        ordered[value] = None
    return tuple(ordered)


def _group(
    rows: list[PreparedMvpaPatternRow],
    *,
    group_id: str = "g1",
    group_key: dict[str, str] | None = None,
    condition_ids: tuple[str, ...] | None = None,
    voxel_index_hash: str | None = None,
) -> PreparedMvpaPatternGroup:
    resolved_group_key = group_key or {"subject_id": "sub-01"}
    return PreparedMvpaPatternGroup(
        group_id=group_id,
        group_key=resolved_group_key,
        group_by=("subject_id",),
        rows=tuple(rows),
        cv_unit="run",
        cv_labels=_first_seen([row.cv_label for row in rows]),
        condition_ids=condition_ids or _first_seen([row.condition_id for row in rows]),
        feature_count=len(rows[0].feature_values) if rows else 1,
        voxel_index_hash=voxel_index_hash,
    )


def _valid_negative_group() -> PreparedMvpaPatternGroup:
    rows = [
        _row("a", "fold-1", [1.0, 0.0], index=1),
        _row("b", "fold-1", [0.0, 0.0], index=2),
        _row("a", "fold-2", [-2.0, 0.0], index=3),
        _row("b", "fold-2", [0.0, 0.0], index=4),
    ]
    return _group(rows, condition_ids=("a", "b"))


def _valid_identity_group() -> PreparedMvpaPatternGroup:
    rows = [
        _row("a", "fold-1", [1.0, 2.0], index=1),
        _row("b", "fold-1", [0.0, 0.0], index=2),
        _row("a", "fold-2", [3.0, 4.0], index=3),
        _row("b", "fold-2", [0.0, 0.0], index=4),
    ]
    return _group(rows, condition_ids=("a", "b"))


def _diagonal_row(
    condition_id: str,
    cv_label: str,
    feature_values: list[float],
    *,
    index: int,
    noise_values: list[object] | tuple[object, ...],
    noise_usable: bool | None = True,
    noise_feature_count: int | None = None,
    voxel_index_hash: str | None = "hash-a",
    noise_voxel_index_hash: str | None = "hash-a",
) -> PreparedMvpaPatternRow:
    return _row(
        condition_id,
        cv_label,
        feature_values,
        index=index,
        voxel_index_hash=voxel_index_hash,
        noise_values=noise_values,
        noise_usable=noise_usable,
        noise_feature_count=len(noise_values) if noise_feature_count is None else noise_feature_count,
        noise_voxel_index_hash=noise_voxel_index_hash,
    )


def _valid_diagonal_group() -> PreparedMvpaPatternGroup:
    rows = [
        _diagonal_row("a", "fold-1", [2.0, 4.0], index=1, noise_values=[1.0, 4.0]),
        _diagonal_row("b", "fold-1", [0.0, 0.0], index=2, noise_values=[3.0, 4.0]),
        _diagonal_row("a", "fold-2", [6.0, 8.0], index=3, noise_values=[2.0, 2.0]),
        _diagonal_row("b", "fold-2", [0.0, 0.0], index=4, noise_values=[2.0, 6.0]),
    ]
    return _group(rows, condition_ids=("a", "b"), voxel_index_hash="hash-a")


def _diagonal_group_with_first_row(
    *,
    noise_values: list[object] | tuple[object, ...] = (1.0, 4.0),
    noise_usable: bool | None = True,
    noise_feature_count: int | None = 2,
    voxel_index_hash: str | None = "hash-a",
    noise_voxel_index_hash: str | None = "hash-a",
    group_voxel_index_hash: str | None = "hash-a",
) -> PreparedMvpaPatternGroup:
    rows = [
        _diagonal_row(
            "a",
            "fold-1",
            [2.0, 4.0],
            index=1,
            noise_values=noise_values,
            noise_usable=noise_usable,
            noise_feature_count=noise_feature_count,
            voxel_index_hash=voxel_index_hash,
            noise_voxel_index_hash=noise_voxel_index_hash,
        ),
        _diagonal_row("b", "fold-1", [0.0, 0.0], index=2, noise_values=[3.0, 4.0]),
        _diagonal_row("a", "fold-2", [6.0, 8.0], index=3, noise_values=[2.0, 2.0]),
        _diagonal_row("b", "fold-2", [0.0, 0.0], index=4, noise_values=[2.0, 6.0]),
    ]
    return _group(rows, condition_ids=("a", "b"), voxel_index_hash=group_voxel_index_hash)


def _raw_manual_pattern_row(
    condition_id: str,
    run_id: str,
    feature_values: list[float],
    *,
    event_count: int,
    index: int,
    noise_values: list[float] | tuple[float, ...] = (1.0, 1.0),
) -> dict[str, object]:
    return {
        "pattern_id": f"raw-{index}",
        "condition_id": condition_id,
        "feature_values": feature_values,
        "feature_count": len(feature_values),
        "subject_id": "sub-01",
        "session_id": "ses-01",
        "run_id": run_id,
        "task_id": "memory",
        "direction": "AP",
        "model": "modelA",
        "pattern_source_name": "success_feat",
        "roi_source_name": "primary8",
        "roi_label": "EncodingFrontalPole",
        "voxel_order": "c_flat_index",
        "voxel_index_hash": "hash-a",
        "usable": True,
        "event_count": event_count,
        "noise_values": list(noise_values),
        "noise_usable": True,
        "noise_feature_count": len(noise_values),
        "noise_voxel_index_hash": "hash-a",
    }


def _codes(result) -> set[str]:
    return {qc.code for qc in result.qc_rows}


def _provenance(result) -> dict[str, object]:
    return {row.key: row.value for row in result.provenance}


def _require_optional_rsatoolbox() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("rsatoolbox")


def _assert_prepared_distance_parity(rsatoolbox_result, native_result) -> None:
    assert rsatoolbox_result.errors == ()
    assert native_result.errors == ()
    assert len(rsatoolbox_result.distances) == len(native_result.distances)
    for rsa_distance, native_distance in zip(
        rsatoolbox_result.distances,
        native_result.distances,
        strict=True,
    ):
        assert rsa_distance.group_id == native_distance.group_id
        assert rsa_distance.group_key == native_distance.group_key
        assert rsa_distance.condition_id_a == native_distance.condition_id_a
        assert rsa_distance.condition_id_b == native_distance.condition_id_b
        assert rsa_distance.distance == pytest.approx(native_distance.distance)
        assert rsa_distance.metric == native_distance.metric
        assert rsa_distance.engine_name == "rsatoolbox"
        assert native_distance.engine_name == "native_reference"
        assert rsa_distance.normalization_method == native_distance.normalization_method
        assert rsa_distance.cv_unit_count == native_distance.cv_unit_count
        assert rsa_distance.feature_count == native_distance.feature_count
        assert rsa_distance.observation_count == native_distance.observation_count
        assert rsa_distance.context == native_distance.context


def test_native_reference_default_behavior_remains_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("rsatoolbox_adapter") or "rsatoolbox_adapter" in (fromlist or ()):
            raise AssertionError("default native_reference path imported the rsatoolbox adapter")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = compute_mvpa_distances_from_prepared_groups([_valid_negative_group()])

    assert result.errors == ()
    assert len(result.distances) == 1
    distance = result.distances[0]
    assert (distance.condition_id_a, distance.condition_id_b) == ("a", "b")
    assert distance.distance == pytest.approx(-2.0)
    assert distance.cv_unit_count == 2
    assert distance.engine_name == "native_reference"
    assert distance.normalization_method == NOISE_NORMALIZATION_IDENTITY
    provenance = _provenance(result)
    assert provenance["engine_name"] == "native_reference"
    assert provenance["native_reference_fallback_used"] is False
    assert provenance["rsatoolbox_adapter_used"] is False
    assert provenance["distance_engine_failure_count"] == 0


def test_explicit_rsatoolbox_selection_uses_optional_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_platform.analysis.mvpa import rsatoolbox_adapter

    calls: list[tuple[tuple[float, ...] | None, str]] = []

    def fake_compute_distances(self, dataset, request):
        calls.append((self.diagonal_variances, request.engine_name))
        return (
            DistanceEstimate(
                condition_id_a="a",
                condition_id_b="b",
                distance=-2.0,
                metric=request.metric,
                engine_name=self.name,
                normalization_method=request.noise_normalization.method,
                cv_unit_count=2,
                context={
                    "condition_index_a": 0,
                    "condition_index_b": 1,
                    "feature_count": len(dataset.feature_names),
                    "observation_count": len(dataset.observations),
                },
            ),
        )

    monkeypatch.setattr(rsatoolbox_adapter.RsatoolboxDistanceEngine, "compute_distances", fake_compute_distances)

    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_negative_group()],
        engine_name="rsatoolbox",
    )

    assert result.errors == ()
    assert calls == [(None, "rsatoolbox")]
    assert isinstance(result.distances[0], prepared_distances.PreparedMvpaDistanceRow)
    assert result.distances[0].engine_name == "rsatoolbox"
    assert result.distances[0].distance == pytest.approx(-2.0)
    provenance = _provenance(result)
    assert provenance["engine_name"] == "rsatoolbox"
    assert provenance["rsatoolbox_adapter_used"] is True
    assert provenance["native_reference_fallback_used"] is False
    assert provenance["distance_engine_failure_count"] == 0


def test_rsatoolbox_missing_dependency_returns_qc_without_native_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_platform.analysis.mvpa import rsatoolbox_adapter

    real_import_module = rsatoolbox_adapter.importlib.import_module

    def fake_import_module(name: str):
        if name == "numpy":
            raise ImportError("simulated missing numpy")
        return real_import_module(name)

    monkeypatch.setattr(rsatoolbox_adapter.importlib, "import_module", fake_import_module)

    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_negative_group()],
        engine_name="rsatoolbox",
    )

    assert result.distances == ()
    assert _codes(result) == {"optional_distance_engine_unavailable"}
    assert "numpy" in result.errors[0]
    assert "native_reference" in result.errors[0]
    qc_row = result.qc_rows[0]
    assert qc_row.context["engine_name"] == "rsatoolbox"
    assert qc_row.context["optional_dependency"] == "numpy"
    assert qc_row.context["native_reference_fallback_used"] is False
    provenance = _provenance(result)
    assert provenance["rsatoolbox_adapter_used"] is True
    assert provenance["native_reference_fallback_used"] is False
    assert provenance["distance_engine_failure_count"] == 1


def test_rsatoolbox_diagonal_receives_mean_prepared_noise_variances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_platform.analysis.mvpa import rsatoolbox_adapter

    calls: list[tuple[float, ...] | None] = []

    def fake_compute_distances(self, dataset, request):
        calls.append(None if self.diagonal_variances is None else tuple(self.diagonal_variances))
        return (
            DistanceEstimate(
                condition_id_a="a",
                condition_id_b="b",
                distance=14.0,
                metric=request.metric,
                engine_name=self.name,
                normalization_method=request.noise_normalization.method,
                cv_unit_count=2,
                context={
                    "condition_index_a": 0,
                    "condition_index_b": 1,
                    "feature_count": len(dataset.feature_names),
                    "observation_count": len(dataset.observations),
                },
            ),
        )

    monkeypatch.setattr(rsatoolbox_adapter.RsatoolboxDistanceEngine, "compute_distances", fake_compute_distances)

    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_diagonal_group()],
        engine_name="rsatoolbox",
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.errors == ()
    assert calls == [(2.0, 4.0)]
    assert result.distances[0].normalization_method == NOISE_NORMALIZATION_DIAGONAL
    provenance = _provenance(result)
    assert provenance["diagonal_noise_groups_succeeded"] == 1
    assert provenance["rsatoolbox_adapter_used"] is True


def test_valid_diagonal_crossnobis_uses_mean_prepared_noise_variances() -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_diagonal_group()],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.errors == ()
    assert len(result.distances) == 1
    distance = result.distances[0]
    assert distance.distance == pytest.approx(14.0)
    assert distance.normalization_method == NOISE_NORMALIZATION_DIAGONAL
    provenance = _provenance(result)
    assert provenance["phase"] == "3B.4"
    assert provenance["noise_aggregation"] == "mean"
    assert provenance["diagonal_noise_groups_attempted"] == 1
    assert provenance["diagonal_noise_groups_succeeded"] == 1
    assert provenance["diagonal_noise_groups_failed"] == 0
    assert provenance["output_written"] is False
    assert provenance["rsatoolbox_adapter_used"] is False


def test_manual_diagonal_engine_uses_run_pair_dot_per_valid_voxel() -> None:
    rows = [
        _diagonal_row("a", "run-1", [1.0, -1.0], index=1, noise_values=[1.0, 1.0]),
        _diagonal_row("b", "run-1", [0.0, 0.0], index=2, noise_values=[1.0, 1.0]),
        _diagonal_row("a", "run-2", [1.0, -1.0], index=3, noise_values=[1.0, 1.0]),
        _diagonal_row("b", "run-2", [0.0, 0.0], index=4, noise_values=[1.0, 1.0]),
    ]

    result = compute_mvpa_distances_from_prepared_groups(
        [_group(rows, condition_ids=("a", "b"), voxel_index_hash="hash-a")],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.errors == ()
    distance = result.distances[0]
    assert distance.engine_name == ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1
    assert distance.distance == pytest.approx(1.0)
    assert distance.feature_count == 2
    assert distance.context["run_pair_details"][0]["dot_product"] == pytest.approx(2.0)


def _four_condition_manual_group(
    *,
    missing_condition_d_in_run3: bool = False,
    missing_condition_d_in_run2: bool = False,
    noise_values_run1: list[object] | tuple[object, ...] = (1.0, 1.0),
    noise_values_run2: list[object] | tuple[object, ...] = (1.0, 1.0),
    noise_values_run3: list[object] | tuple[object, ...] = (1.0, 1.0),
    noise_usable_run1: bool | None = True,
    noise_usable_run2: bool | None = True,
    noise_usable_run3: bool | None = True,
    d_run2_noise_voxel_hash: str | None = "hash-a",
) -> PreparedMvpaPatternGroup:
    condition_vectors = {
        "a": [1.0, 0.0],
        "b": [0.0, 0.0],
        "c": [0.0, 1.0],
        "d": [1.0, 1.0],
    }
    rows: list[PreparedMvpaPatternRow] = []
    index = 1
    for run_label, noise_values, noise_usable in (
        ("run-1", noise_values_run1, noise_usable_run1),
        ("run-2", noise_values_run2, noise_usable_run2),
        ("run-3", noise_values_run3, noise_usable_run3),
    ):
        for condition_id, vector in condition_vectors.items():
            if condition_id == "d" and run_label == "run-3" and missing_condition_d_in_run3:
                continue
            if condition_id == "d" and run_label == "run-2" and missing_condition_d_in_run2:
                continue
            rows.append(
                _diagonal_row(
                    condition_id,
                    run_label,
                    vector,
                    index=index,
                    noise_values=noise_values,
                    noise_usable=noise_usable,
                    noise_voxel_index_hash=d_run2_noise_voxel_hash
                    if condition_id == "d" and run_label == "run-2"
                    else "hash-a",
                )
            )
            index += 1
    return _group(rows, condition_ids=("a", "b", "c", "d"), voxel_index_hash="hash-a")


def test_manual_diagonal_all_pairs_computes_six_pairs_from_four_conditions() -> None:
    condition_pairs = [
        {"id": "a_minus_b", "left": "a", "right": "b"},
        {"id": "a_minus_c", "left": "a", "right": "c"},
        {"id": "a_minus_d", "left": "a", "right": "d"},
        {"id": "b_minus_c", "left": "b", "right": "c"},
        {"id": "b_minus_d", "left": "b", "right": "d"},
        {"id": "c_minus_d", "left": "c", "right": "d"},
    ]

    result = compute_mvpa_distances_from_prepared_groups(
        [_four_condition_manual_group()],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
        condition_pairs=condition_pairs,
    )

    assert result.errors == ()
    assert len(result.distances) == 6
    assert [row.condition_pair_id for row in result.distances] == [pair["id"] for pair in condition_pairs]
    assert all(row.cv_unit_count == 3 for row in result.distances)


def test_manual_diagonal_all_pairs_missing_condition_warns_when_enough_cv_units_remain() -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [_four_condition_manual_group(missing_condition_d_in_run3=True)],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.errors == ()
    assert len(result.distances) == 6
    distances_by_pair = {
        frozenset((row.condition_id_a, row.condition_id_b)): row.cv_unit_count for row in result.distances
    }
    assert distances_by_pair[frozenset(("a", "d"))] == 2
    assert distances_by_pair[frozenset(("a", "b"))] == 3
    assert any(
        row.code == "manual_pair_missing_condition_in_run" and row.status == "warning"
        for row in result.qc_rows
    )


def test_manual_diagonal_all_pairs_too_few_cv_units_remains_error() -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [
            _four_condition_manual_group(
                missing_condition_d_in_run2=True,
                missing_condition_d_in_run3=True,
            )
        ],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert len(result.distances) == 3
    assert result.errors
    assert "insufficient_valid_runs" in _codes(result)


def test_manual_diagonal_all_pairs_incompatible_noise_hash_fails_cleanly() -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [_four_condition_manual_group(missing_condition_d_in_run3=True, d_run2_noise_voxel_hash="hash-b")],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert len(result.distances) == 3
    assert "manual_diagonal_noise_voxel_hash_mismatch" in _codes(result)
    assert any(row.status == "failed" for row in result.qc_rows if row.code == "manual_diagonal_noise_voxel_hash_mismatch")


def test_manual_diagonal_all_pairs_drop_features_policy_filters_nonpositive_noise() -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [
            _four_condition_manual_group(
                noise_values_run1=(1.0, 0.0),
                noise_usable_run1=False,
                noise_values_run2=(1.0, 4.0),
                noise_values_run3=(1.0, 4.0),
            )
        ],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
        noise_nonpositive_policy="drop_features",
        min_retained_features=1,
    )

    assert result.errors == ()
    assert len(result.distances) == 6
    assert all(row.context["retained_feature_count"] >= 1 for row in result.distances)
    assert any(
        row.code == "manual_diagonal_noise_feature_filter_warning" and row.status == "warning"
        for row in result.qc_rows
    )


def test_manual_sensitivity_threshold_exclusion_can_warn_when_distance_is_computed() -> None:
    threshold_sweeps = [{"id": "sensitivity", "min_events": 3}]
    prepared = prepare_mvpa_pattern_row_groups(
        [
            _raw_manual_pattern_row("a", "run-1", [1.0, -1.0], event_count=3, index=1),
            _raw_manual_pattern_row("b", "run-1", [0.0, 0.0], event_count=3, index=2),
            _raw_manual_pattern_row("a", "run-2", [1.0, -1.0], event_count=3, index=3),
            _raw_manual_pattern_row("b", "run-2", [0.0, 0.0], event_count=3, index=4),
            _raw_manual_pattern_row("a", "run-3", [5.0, -5.0], event_count=3, index=5),
            _raw_manual_pattern_row("b", "run-3", [0.0, 0.0], event_count=2, index=6),
        ],
        threshold_sweeps=threshold_sweeps,
    )

    result = compute_mvpa_distances_from_prepared_groups(
        prepared.groups,
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
        threshold_sweeps=threshold_sweeps,
        preparation_qc_rows=prepared.qc_rows,
    )

    assert prepared.errors == ()
    assert result.errors == ()
    assert len(result.distances) == 1
    assert result.distances[0].cv_unit_count == 2
    assert result.distances[0].distance == pytest.approx(1.0)
    statuses_by_code = {row.code: row.status for row in result.qc_rows}
    assert statuses_by_code["threshold_failure"] == "warning"
    assert statuses_by_code["manual_pair_missing_condition_in_run"] == "warning"


def test_manual_missing_condition_can_warn_when_distance_is_computed() -> None:
    rows = [
        _diagonal_row("a", "run-1", [1.0, -1.0], index=1, noise_values=[1.0, 1.0]),
        _diagonal_row("b", "run-1", [0.0, 0.0], index=2, noise_values=[1.0, 1.0]),
        _diagonal_row("a", "run-2", [1.0, -1.0], index=3, noise_values=[1.0, 1.0]),
        _diagonal_row("b", "run-2", [0.0, 0.0], index=4, noise_values=[1.0, 1.0]),
        _diagonal_row("a", "run-3", [3.0, -3.0], index=5, noise_values=[1.0, 1.0]),
    ]

    result = compute_mvpa_distances_from_prepared_groups(
        [_group(rows, condition_ids=("a", "b"), voxel_index_hash="hash-a")],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.errors == ()
    assert len(result.distances) == 1
    assert result.distances[0].cv_unit_count == 2
    assert any(
        row.code == "manual_pair_missing_condition_in_run" and row.status == "warning"
        for row in result.qc_rows
    )


def test_manual_sensitivity_threshold_exclusion_remains_error_when_too_few_cv_units_remain() -> None:
    threshold_sweeps = [{"id": "sensitivity", "min_events": 3}]
    prepared = prepare_mvpa_pattern_row_groups(
        [
            _raw_manual_pattern_row("a", "run-1", [1.0, -1.0], event_count=3, index=1),
            _raw_manual_pattern_row("b", "run-1", [0.0, 0.0], event_count=3, index=2),
            _raw_manual_pattern_row("a", "run-2", [1.0, -1.0], event_count=3, index=3),
            _raw_manual_pattern_row("b", "run-2", [0.0, 0.0], event_count=2, index=4),
        ],
        threshold_sweeps=threshold_sweeps,
    )

    result = compute_mvpa_distances_from_prepared_groups(
        prepared.groups,
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
        threshold_sweeps=threshold_sweeps,
        preparation_qc_rows=prepared.qc_rows,
    )

    assert result.distances == ()
    assert result.errors
    assert "insufficient_valid_runs" in _codes(result)
    assert any(row.code == "insufficient_valid_runs" and row.status == "failed" for row in result.qc_rows)


def test_manual_diagonal_strict_policy_preserves_nonpositive_noise_failure() -> None:
    rows = [
        _diagonal_row("a", "run-1", [1.0, 5.0], index=1, noise_values=[1.0, 0.0]),
        _diagonal_row("b", "run-1", [0.0, 0.0], index=2, noise_values=[1.0, 0.0]),
        _diagonal_row("a", "run-2", [3.0, 7.0], index=3, noise_values=[1.0, 4.0]),
        _diagonal_row("b", "run-2", [0.0, 0.0], index=4, noise_values=[1.0, 4.0]),
    ]

    result = compute_mvpa_distances_from_prepared_groups(
        [_group(rows, condition_ids=("a", "b"), voxel_index_hash="hash-a")],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.distances == ()
    assert "manual_diagonal_noise_nonpositive_value" in _codes(result)


def test_manual_diagonal_drop_features_policy_filters_invalid_noise_features() -> None:
    rows = [
        _diagonal_row("a", "run-1", [2.0, 4.0, 6.0], index=1, noise_values=[1.0, 0.0, 4.0], noise_usable=False),
        _diagonal_row("b", "run-1", [1.0, 1.0, 1.0], index=2, noise_values=[1.0, 0.0, 4.0], noise_usable=False),
        _diagonal_row("a", "run-2", [4.0, 6.0, 8.0], index=3, noise_values=[1.0, 9.0, 4.0]),
        _diagonal_row("b", "run-2", [1.0, 1.0, 1.0], index=4, noise_values=[1.0, 9.0, 4.0]),
    ]

    result = compute_mvpa_distances_from_prepared_groups(
        [_group(rows, condition_ids=("a", "b"), voxel_index_hash="hash-a")],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
        noise_nonpositive_policy="drop_features",
        min_retained_features=2,
    )

    assert result.errors == ()
    distance = result.distances[0]
    assert distance.distance == pytest.approx(0.1875)
    assert distance.feature_count == 2
    assert distance.context["noise_nonpositive_policy"] == "drop_features"
    assert distance.context["original_feature_count"] == 3
    assert distance.context["retained_feature_count"] == 2
    assert distance.context["dropped_noise_feature_count"] == 1
    assert distance.context["dropped_feature_fraction"] == pytest.approx(1.0 / 3.0)
    assert any(
        row.code == "manual_diagonal_noise_feature_filter_warning" and row.status == "warning"
        for row in result.qc_rows
    )
    provenance = _provenance(result)
    assert provenance["noise_nonpositive_policy"] == "drop_features"
    assert provenance["min_retained_features"] == 2


def test_manual_diagonal_drop_features_policy_fails_when_too_few_features_remain() -> None:
    rows = [
        _diagonal_row("a", "run-1", [2.0, 4.0, 6.0], index=1, noise_values=[1.0, 0.0, 0.0], noise_usable=False),
        _diagonal_row("b", "run-1", [1.0, 1.0, 1.0], index=2, noise_values=[1.0, 0.0, 0.0], noise_usable=False),
        _diagonal_row("a", "run-2", [4.0, 6.0, 8.0], index=3, noise_values=[1.0, 9.0, 4.0]),
        _diagonal_row("b", "run-2", [1.0, 1.0, 1.0], index=4, noise_values=[1.0, 9.0, 4.0]),
    ]

    result = compute_mvpa_distances_from_prepared_groups(
        [_group(rows, condition_ids=("a", "b"), voxel_index_hash="hash-a")],
        engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
        noise_nonpositive_policy="drop_features",
        min_retained_features=2,
    )

    assert result.distances == ()
    assert "insufficient_valid_voxels" in _codes(result)
    qc_context = next(row.context for row in result.qc_rows if row.code == "insufficient_valid_voxels")
    assert qc_context["retained_feature_count"] == 1
    assert qc_context["dropped_noise_feature_count"] == 2


def test_identity_mode_ignores_prepared_noise_values() -> None:
    group = _group(
        [
            _row("a", "fold-1", [1.0, 0.0], index=1, noise_values=[0.0, "bad"], noise_usable=False),
            _row("b", "fold-1", [0.0, 0.0], index=2, noise_values=[0.0, "bad"], noise_usable=False),
            _row("a", "fold-2", [-2.0, 0.0], index=3, noise_values=[0.0, "bad"], noise_usable=False),
            _row("b", "fold-2", [0.0, 0.0], index=4, noise_values=[0.0, "bad"], noise_usable=False),
        ],
        condition_ids=("a", "b"),
    )

    result = compute_mvpa_distances_from_prepared_groups([group])

    assert result.errors == ()
    assert result.qc_rows == ()
    assert result.distances[0].distance == pytest.approx(-2.0)
    assert result.distances[0].normalization_method == NOISE_NORMALIZATION_IDENTITY
    provenance = _provenance(result)
    assert provenance["diagonal_noise_groups_attempted"] == 0


def test_missing_diagonal_noise_fails_group_without_distance() -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_negative_group()],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.distances == ()
    assert _codes(result) == {"diagonal_noise_missing"}
    assert result.errors
    provenance = _provenance(result)
    assert provenance["diagonal_noise_groups_attempted"] == 1
    assert provenance["diagonal_noise_groups_succeeded"] == 0
    assert provenance["diagonal_noise_groups_failed"] == 1


def test_partial_missing_diagonal_noise_fails_whole_group_without_partial_distance() -> None:
    rows = [
        _diagonal_row("a", "fold-1", [2.0, 4.0], index=1, noise_values=[1.0, 4.0]),
        _diagonal_row("b", "fold-1", [0.0, 0.0], index=2, noise_values=[3.0, 4.0]),
        _diagonal_row("a", "fold-2", [6.0, 8.0], index=3, noise_values=[2.0, 2.0]),
        _row("b", "fold-2", [0.0, 0.0], index=4, voxel_index_hash="hash-a"),
    ]
    group = _group(rows, condition_ids=("a", "b"), voxel_index_hash="hash-a")

    result = compute_mvpa_distances_from_prepared_groups(
        [group],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.distances == ()
    assert {"diagonal_noise_partial_availability", "diagonal_noise_missing"} <= _codes(result)
    assert {qc.source_row_index for qc in result.qc_rows if qc.code == "diagonal_noise_missing"} == {4}


@pytest.mark.parametrize("bad_noise", ["bad", math.nan, math.inf, -math.inf, True])
def test_nonfinite_or_nonnumeric_diagonal_noise_is_rejected(bad_noise: object) -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [_diagonal_group_with_first_row(noise_values=[bad_noise, 4.0])],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.distances == ()
    assert "diagonal_noise_nonfinite_value" in _codes(result)


@pytest.mark.parametrize("bad_noise", [0.0, -1.0])
def test_zero_or_negative_diagonal_noise_is_rejected(bad_noise: float) -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [_diagonal_group_with_first_row(noise_values=[bad_noise, 4.0])],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.distances == ()
    assert "diagonal_noise_nonpositive_value" in _codes(result)


@pytest.mark.parametrize(
    "noise_values,noise_feature_count",
    [
        ([1.0], 1),
        ([1.0, 4.0], 3),
    ],
)
def test_diagonal_noise_feature_count_mismatch_is_rejected(
    noise_values: list[object],
    noise_feature_count: int,
) -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [_diagonal_group_with_first_row(noise_values=noise_values, noise_feature_count=noise_feature_count)],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.distances == ()
    assert "diagonal_noise_feature_count_mismatch" in _codes(result)


@pytest.mark.parametrize(
    "row_hash,noise_hash,group_hash",
    [
        ("hash-a", "hash-a", None),
        (None, "hash-a", "hash-a"),
        ("hash-a", None, "hash-a"),
        ("hash-b", "hash-a", "hash-a"),
        ("hash-a", "hash-b", "hash-a"),
    ],
)
def test_diagonal_noise_voxel_hash_mismatch_or_missing_hash_is_rejected(
    row_hash: str | None,
    noise_hash: str | None,
    group_hash: str | None,
) -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [
            _diagonal_group_with_first_row(
                voxel_index_hash=row_hash,
                noise_voxel_index_hash=noise_hash,
                group_voxel_index_hash=group_hash,
            )
        ],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.distances == ()
    assert "diagonal_noise_voxel_hash_mismatch" in _codes(result)


def test_negative_diagonal_crossnobis_remains_valid() -> None:
    rows = [
        _diagonal_row("a", "fold-1", [1.0, 0.0], index=1, noise_values=[2.0, 1.0]),
        _diagonal_row("b", "fold-1", [0.0, 0.0], index=2, noise_values=[2.0, 1.0]),
        _diagonal_row("a", "fold-2", [-2.0, 0.0], index=3, noise_values=[2.0, 1.0]),
        _diagonal_row("b", "fold-2", [0.0, 0.0], index=4, noise_values=[2.0, 1.0]),
    ]
    group = _group(rows, condition_ids=("a", "b"), voxel_index_hash="hash-a")

    result = compute_mvpa_distances_from_prepared_groups(
        [group],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    assert result.errors == ()
    assert result.distances[0].distance == pytest.approx(-1.0)
    assert result.distances[0].normalization_method == NOISE_NORMALIZATION_DIAGONAL


def test_multiple_groups_are_returned_in_deterministic_group_order() -> None:
    group_two = _group(
        [
            _row("a", "fold-1", [1.0], index=1, group_key={"subject_id": "sub-02"}),
            _row("b", "fold-1", [0.0], index=2, group_key={"subject_id": "sub-02"}),
            _row("a", "fold-2", [3.0], index=3, group_key={"subject_id": "sub-02"}),
            _row("b", "fold-2", [0.0], index=4, group_key={"subject_id": "sub-02"}),
        ],
        group_id="g2",
        group_key={"subject_id": "sub-02"},
        condition_ids=("a", "b"),
    )
    group_one = _group(
        [
            _row("a", "fold-1", [2.0], index=5, group_key={"subject_id": "sub-01"}),
            _row("b", "fold-1", [0.0], index=6, group_key={"subject_id": "sub-01"}),
            _row("a", "fold-2", [4.0], index=7, group_key={"subject_id": "sub-01"}),
            _row("b", "fold-2", [0.0], index=8, group_key={"subject_id": "sub-01"}),
        ],
        group_id="g1",
        group_key={"subject_id": "sub-01"},
        condition_ids=("a", "b"),
    )

    result = compute_mvpa_distances_from_prepared_groups([group_two, group_one])

    assert [row.group_id for row in result.distances] == ["g1", "g2"]
    assert [row.distance for row in result.distances] == [pytest.approx(8.0), pytest.approx(3.0)]


def test_missing_condition_per_cv_unit_returns_group_qc_instead_of_distance() -> None:
    group = _group(
        [
            _row("a", "fold-1", [1.0], index=1),
            _row("b", "fold-1", [0.0], index=2),
            _row("a", "fold-2", [2.0], index=3),
        ],
        condition_ids=("a", "b"),
    )

    result = compute_mvpa_distances_from_prepared_groups([group])

    assert result.distances == ()
    assert {qc.code for qc in result.qc_rows} == {"unbalanced_condition_cv_presence"}
    assert result.qc_rows[0].level == "group"
    assert result.errors


def test_fewer_than_two_cv_units_returns_group_qc() -> None:
    group = _group(
        [
            _row("a", "fold-1", [1.0], index=1),
            _row("b", "fold-1", [0.0], index=2),
        ],
        condition_ids=("a", "b"),
    )

    result = compute_mvpa_distances_from_prepared_groups([group])

    assert result.distances == ()
    assert {qc.code for qc in result.qc_rows} == {"insufficient_cv_units"}


def test_condition_pair_selection_filters_conditions_and_orders_pair_by_group_condition_order() -> None:
    group = _group(
        [
            _row("c", "fold-1", [10.0], index=1),
            _row("a", "fold-1", [1.0], index=2),
            _row("b", "fold-1", [0.0], index=3),
            _row("c", "fold-2", [20.0], index=4),
            _row("a", "fold-2", [2.0], index=5),
            _row("b", "fold-2", [0.0], index=6),
        ],
        condition_ids=("c", "a", "b"),
    )

    result = compute_mvpa_distances_from_prepared_groups([group], condition_pairs=[("b", "a")])

    assert result.errors == ()
    assert [(row.condition_id_a, row.condition_id_b) for row in result.distances] == [("a", "b")]


def test_condition_pair_rows_are_deterministic_for_unordered_configured_pairs() -> None:
    group = _group(
        [
            _row("b", "fold-1", [0.0], index=1),
            _row("a", "fold-1", [1.0], index=2),
            _row("c", "fold-1", [3.0], index=3),
            _row("b", "fold-2", [0.0], index=4),
            _row("a", "fold-2", [2.0], index=5),
            _row("c", "fold-2", [4.0], index=6),
        ],
        condition_ids=("b", "a", "c"),
    )

    result = compute_mvpa_distances_from_prepared_groups(
        [group],
        condition_pairs=[("c", "a"), ("a", "b")],
    )

    assert [(row.condition_id_a, row.condition_id_b) for row in result.distances] == [
        ("b", "a"),
        ("a", "c"),
    ]


def test_condition_pair_id_and_grouping_columns_are_preserved_in_distance_rows() -> None:
    group_key = {"participant_id": "participant-a", "task_id": "task-alpha"}
    group = _group(
        [
            _row("condition-a", "run-a", [1.0], index=1, group_key=group_key),
            _row("condition-b", "run-a", [0.0], index=2, group_key=group_key),
            _row("condition-a", "run-b", [2.0], index=3, group_key=group_key),
            _row("condition-b", "run-b", [0.0], index=4, group_key=group_key),
        ],
        group_id="group-alpha",
        group_key=group_key,
        condition_ids=("condition-a", "condition-b"),
    )

    result = compute_mvpa_distances_from_prepared_groups(
        [group],
        condition_pairs=[{"id": "pair-alpha", "left": "condition-a", "right": "condition-b"}],
    )

    assert result.errors == ()
    distance = result.distances[0]
    assert distance.condition_pair_id == "pair-alpha"
    assert distance.group_key == group_key
    assert distance.to_dict()["group_key"] == group_key
    json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)


def test_min_observations_threshold_failure_records_pair_provenance() -> None:
    group_key = {"participant_id": "participant-a", "task_id": "task-alpha"}
    group = _group(
        [
            _row("condition-a", "run-a", [1.0], index=1, group_key=group_key),
            _row("condition-b", "run-a", [0.0], index=2, group_key=group_key),
            _row("condition-a", "run-b", [2.0], index=3, group_key=group_key),
            _row("condition-b", "run-b", [0.0], index=4, group_key=group_key),
        ],
        group_id="group-alpha",
        group_key=group_key,
        condition_ids=("condition-a", "condition-b"),
    )

    result = compute_mvpa_distances_from_prepared_groups(
        [group],
        condition_pairs=[{"id": "pair-alpha", "left": "condition-a", "right": "condition-b"}],
        threshold_sweeps=[{"id": "threshold-observations-high", "min_observations": 3}],
    )

    assert result.distances == ()
    assert _codes(result) == {"threshold_failure"}
    qc = result.qc_rows[0]
    assert qc.condition_pair_id == "pair-alpha"
    assert qc.context == {
        "threshold_id": "threshold-observations-high",
        "threshold_type": "min_observations",
        "required_value": 3,
        "observed_value": 2,
        "failure_reason": "observed_observations_below_required_minimum",
        "condition_pair_id": "pair-alpha",
    }
    provenance = _provenance(result)
    assert provenance["threshold_sweep_count"] == 1
    assert provenance["threshold_failure_count"] == 1
    json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)


def test_preparation_qc_rows_are_propagated_into_distance_result() -> None:
    preparation_qc = MvpaPatternRowPreparationQcRow(
        level="noise",
        status="warning",
        code="invalid_noise_values",
        message="Source row 7 had invalid noise metadata.",
        source_row_index=7,
        group_id="g1",
        group_key={"subject_id": "sub-01"},
        cv_unit="run",
        cv_label="fold-1",
    )

    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_negative_group()],
        preparation_qc_rows=[preparation_qc],
    )

    assert result.qc_rows[0].source == "row_preparation"
    assert result.qc_rows[0].code == "invalid_noise_values"
    assert result.qc_rows[0].context == {"phase": "3B.1"}
    assert result.warnings == ("Source row 7 had invalid noise metadata.",)


def test_result_to_dict_is_json_safe() -> None:
    result = compute_mvpa_distances_from_prepared_groups([_valid_negative_group()])

    payload = result.to_dict()

    assert payload["distances"][0]["distance"] == pytest.approx(-2.0)
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_diagonal_result_to_dict_is_json_safe() -> None:
    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_diagonal_group()],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    payload = result.to_dict()

    assert payload["distances"][0]["normalization_method"] == NOISE_NORMALIZATION_DIAGONAL
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_rsatoolbox_missing_dependency_result_to_dict_is_json_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_platform.analysis.mvpa import rsatoolbox_adapter

    real_import_module = rsatoolbox_adapter.importlib.import_module

    def fake_import_module(name: str):
        if name == "numpy":
            raise ImportError("simulated missing numpy")
        return real_import_module(name)

    monkeypatch.setattr(rsatoolbox_adapter.importlib, "import_module", fake_import_module)

    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_negative_group()],
        engine_name="rsatoolbox",
    )

    payload = result.to_dict()

    assert payload["distances"] == []
    assert payload["qc_rows"][0]["code"] == "optional_distance_engine_unavailable"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_no_output_writing_occurs(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = builtins.open

    def guarded_open(file, mode: str = "r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"unexpected write open for {file!r}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    result = compute_mvpa_distances_from_prepared_groups([_valid_negative_group()])

    provenance = {row.key: row.value for row in result.provenance}
    assert len(result.distances) == 1
    assert provenance["output_written"] is False


def test_no_output_writing_occurs_in_diagonal_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = builtins.open

    def guarded_open(file, mode: str = "r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"unexpected write open for {file!r}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_diagonal_group()],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )

    provenance = _provenance(result)
    assert len(result.distances) == 1
    assert provenance["output_written"] is False
    assert provenance["rsatoolbox_adapter_used"] is False


def test_no_output_writing_occurs_with_explicit_rsatoolbox(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_platform.analysis.mvpa import rsatoolbox_adapter

    real_open = builtins.open

    def guarded_open(file, mode: str = "r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"unexpected write open for {file!r}")
        return real_open(file, mode, *args, **kwargs)

    def fake_compute_distances(self, dataset, request):
        return (
            DistanceEstimate(
                condition_id_a="a",
                condition_id_b="b",
                distance=-2.0,
                metric=request.metric,
                engine_name=self.name,
                normalization_method=request.noise_normalization.method,
                cv_unit_count=2,
                context={
                    "condition_index_a": 0,
                    "condition_index_b": 1,
                    "feature_count": len(dataset.feature_names),
                    "observation_count": len(dataset.observations),
                },
            ),
        )

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(rsatoolbox_adapter.RsatoolboxDistanceEngine, "compute_distances", fake_compute_distances)

    result = compute_mvpa_distances_from_prepared_groups(
        [_valid_negative_group()],
        engine_name="rsatoolbox",
    )

    provenance = _provenance(result)
    assert len(result.distances) == 1
    assert provenance["output_written"] is False
    assert provenance["rsatoolbox_adapter_used"] is True


def test_rejects_unknown_engine_unsupported_noise_and_aggregation() -> None:
    group = _valid_negative_group()

    with pytest.raises(ValueError, match="engine_name"):
        compute_mvpa_distances_from_prepared_groups([group], engine_name="unknown")
    with pytest.raises(ValueError, match="Phase 3B.4"):
        compute_mvpa_distances_from_prepared_groups([group], noise_normalization_method="full")
    with pytest.raises(ValueError, match="mean"):
        compute_mvpa_distances_from_prepared_groups([group], noise_aggregation="median")
    with pytest.raises(ValueError, match="crossnobis"):
        compute_mvpa_distances_from_prepared_groups(
            [group],
            metric=METRIC_EUCLIDEAN,
            noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
        )


def test_optional_prepared_identity_crossnobis_parity_with_rsatoolbox() -> None:
    _require_optional_rsatoolbox()
    group = _valid_identity_group()

    native_result = compute_mvpa_distances_from_prepared_groups([group])
    rsatoolbox_result = compute_mvpa_distances_from_prepared_groups([group], engine_name="rsatoolbox")

    _assert_prepared_distance_parity(rsatoolbox_result, native_result)


def test_optional_prepared_negative_crossnobis_parity_with_rsatoolbox() -> None:
    _require_optional_rsatoolbox()
    group = _valid_negative_group()

    native_result = compute_mvpa_distances_from_prepared_groups([group])
    rsatoolbox_result = compute_mvpa_distances_from_prepared_groups([group], engine_name="rsatoolbox")

    _assert_prepared_distance_parity(rsatoolbox_result, native_result)
    assert rsatoolbox_result.distances[0].distance < 0


def test_optional_prepared_condition_pair_filtering_parity_with_rsatoolbox() -> None:
    _require_optional_rsatoolbox()
    group = _group(
        [
            _row("b", "fold-1", [0.0, 1.0], index=1),
            _row("a", "fold-1", [1.0, 2.0], index=2),
            _row("c", "fold-1", [3.0, 1.0], index=3),
            _row("b", "fold-2", [0.0, 2.0], index=4),
            _row("a", "fold-2", [2.0, 3.0], index=5),
            _row("c", "fold-2", [4.0, 1.0], index=6),
        ],
        condition_ids=("b", "a", "c"),
    )

    native_result = compute_mvpa_distances_from_prepared_groups(
        [group],
        condition_pairs=[("c", "a"), ("a", "b")],
    )
    rsatoolbox_result = compute_mvpa_distances_from_prepared_groups(
        [group],
        engine_name="rsatoolbox",
        condition_pairs=[("c", "a"), ("a", "b")],
    )

    _assert_prepared_distance_parity(rsatoolbox_result, native_result)
    assert [(row.condition_id_a, row.condition_id_b) for row in rsatoolbox_result.distances] == [
        ("b", "a"),
        ("a", "c"),
    ]


def test_optional_prepared_diagonal_parity_with_rsatoolbox() -> None:
    _require_optional_rsatoolbox()
    group = _valid_diagonal_group()

    native_result = compute_mvpa_distances_from_prepared_groups(
        [group],
        noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
    )
    try:
        rsatoolbox_result = compute_mvpa_distances_from_prepared_groups(
            [group],
            engine_name="rsatoolbox",
            noise_normalization_method=NOISE_NORMALIZATION_DIAGONAL,
        )
    except RuntimeError as exc:
        pytest.skip(f"rsatoolbox diagonal precision input is unavailable: {exc}")
    if rsatoolbox_result.errors:
        pytest.skip(f"rsatoolbox diagonal precision input is unavailable: {rsatoolbox_result.errors[0]}")

    _assert_prepared_distance_parity(rsatoolbox_result, native_result)


def test_forbidden_import_guard_for_prepared_distances_module_and_tests() -> None:
    forbidden_modules = (
        "research_platform.neuro",
        "research_platform.bids",
        "research_platform.core",
        "research_platform.viz",
        "research_platform.ml",
        "nibabel",
        "nilearn",
        "rsatoolbox",
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "sklearn",
        "pipelines",
        "ops",
    )

    imported_modules: list[str] = []
    for path in (Path(prepared_distances.__file__), Path(__file__)):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)

    for imported_module in imported_modules:
        assert not any(
            imported_module == forbidden or imported_module.startswith(f"{forbidden}.")
            for forbidden in forbidden_modules
        )
