from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import random
import re
from statistics import NormalDist

import pytest

import research_platform.analysis.subject_inference as subject_inference
from research_platform.analysis.subject_inference import (
    SubjectInferenceProvenanceRow,
    summarize_subject_level_inference,
)


def _row(participant: str, value: object, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "participant": participant,
        "value": value,
    }
    row.update(overrides)
    return row


def _codes(result) -> set[str]:
    return {row.code for row in result.qc_rows}


def _failed_codes(result) -> set[str]:
    return {row.code for row in result.qc_rows if row.status == "failed"}


def _provenance(result) -> dict[str, object]:
    return {row.key: row.value for row in result.provenance_rows}


def _bootstrap_expected(values: list[float], *, iterations: int, seed: int, confidence_level: float) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(iterations))
    alpha = 1.0 - confidence_level
    lower_index = math.floor((alpha / 2.0) * (iterations - 1))
    upper_index = math.ceil((1.0 - alpha / 2.0) * (iterations - 1))
    return means[lower_index], means[upper_index]


def _monte_carlo_sign_flip_expected(
    differences: list[float],
    *,
    alternative: str,
    iterations: int,
    seed: int,
) -> float:
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    count = 0
    for _ in range(iterations):
        statistic = sum(difference if rng.random() < 0.5 else -difference for difference in differences) / len(differences)
        if alternative == "greater" and statistic >= observed - 1e-12:
            count += 1
        elif alternative == "less" and statistic <= observed + 1e-12:
            count += 1
        elif alternative == "two_sided" and abs(statistic) >= abs(observed) - 1e-12:
            count += 1
    return (count + 1) / (iterations + 1)


def test_one_sample_stats_median_normal_ci_sign_flip_percent_positive_and_loso() -> None:
    result = summarize_subject_level_inference(
        [
            _row("participant-a", 1.0),
            _row("participant-b", 2.0),
            _row("participant-c", 3.0),
            _row("participant-d", 4.0),
        ],
        subject_column="participant",
        value_column="value",
    )

    summary = result.summary_rows[0]
    sd = math.sqrt(5.0 / 3.0)
    se = sd / 2.0
    z_value = NormalDist().inv_cdf(0.975)

    assert result.errors == ()
    assert summary.n == 4
    assert summary.mean == 2.5
    assert summary.sd == pytest.approx(sd)
    assert summary.se == pytest.approx(se)
    assert summary.median == 2.5
    assert summary.ci_method == "normal_approximation"
    assert summary.ci_low == pytest.approx(2.5 - z_value * se)
    assert summary.ci_high == pytest.approx(2.5 + z_value * se)
    assert summary.t == pytest.approx(2.5 / se)
    assert summary.df == 3
    assert summary.effect_size == pytest.approx(2.5 / sd)
    assert summary.effect_size_type == "dz"
    assert summary.p_method == "sign_flip"
    assert summary.p_method_detail == "exact"
    assert summary.p_value == 0.125
    assert summary.percent_positive == 100.0
    assert summary.loo_min == 2.0
    assert summary.loo_max == 3.0
    assert [(row.left_out_subject, row.n, row.mean) for row in result.loso_rows] == [
        ("participant-a", 3, pytest.approx(3.0)),
        ("participant-b", 3, pytest.approx(8.0 / 3.0)),
        ("participant-c", 3, pytest.approx(7.0 / 3.0)),
        ("participant-d", 3, pytest.approx(2.0)),
    ]


def test_finite_null_value_and_sign_flip_alternatives() -> None:
    rows = [
        _row("participant-a", 2.0),
        _row("participant-b", 3.0),
        _row("participant-c", 4.0),
    ]

    greater = summarize_subject_level_inference(
        rows,
        subject_column="participant",
        value_column="value",
        null_value=1.0,
        alternative="greater",
    ).summary_rows[0]
    less = summarize_subject_level_inference(
        rows,
        subject_column="participant",
        value_column="value",
        null_value=1.0,
        alternative="less",
    ).summary_rows[0]
    two_sided = summarize_subject_level_inference(
        rows,
        subject_column="participant",
        value_column="value",
        null_value=1.0,
        alternative="two_sided",
    ).summary_rows[0]

    assert greater.null_value == 1.0
    assert greater.mean == 3.0
    assert greater.p_value == 0.125
    assert less.p_value == 1.0
    assert two_sided.p_value == 0.25
    assert greater.effect_size == pytest.approx(2.0)
    assert greater.t == pytest.approx(2.0 / (1.0 / math.sqrt(3.0)))


def test_null_value_column_must_be_constant_within_group() -> None:
    result = summarize_subject_level_inference(
        [
            _row("participant-a", 2.0, family="family-a", null=1.0),
            _row("participant-b", 3.0, family="family-a", null=1.5),
        ],
        subject_column="participant",
        value_column="value",
        group_columns=("family",),
        null_value_column="null",
    )

    assert result.summary_rows[0].status == "failed"
    assert result.summary_rows[0].p_value is None
    assert "nonconstant_null_value" in _failed_codes(result)


def test_bootstrap_ci_is_deterministic_for_the_mean() -> None:
    rows = [
        _row("participant-a", 1.0),
        _row("participant-b", 2.0),
        _row("participant-c", 5.0),
        _row("participant-d", 9.0),
    ]

    first = summarize_subject_level_inference(
        rows,
        subject_column="participant",
        value_column="value",
        ci_method="bootstrap",
        bootstrap_iterations=250,
        seed=22,
    )
    second = summarize_subject_level_inference(
        rows,
        subject_column="participant",
        value_column="value",
        ci_method="bootstrap",
        bootstrap_iterations=250,
        seed=22,
    )
    expected_low, expected_high = _bootstrap_expected([1.0, 2.0, 5.0, 9.0], iterations=250, seed=22, confidence_level=0.95)

    assert first.summary_rows[0].ci_method == "bootstrap"
    assert first.summary_rows[0].ci_low == expected_low
    assert first.summary_rows[0].ci_high == expected_high
    assert first.to_dict() == second.to_dict()


def test_monte_carlo_sign_flip_is_seeded_and_uses_plus_one_correction() -> None:
    rows = [
        _row("participant-a", 1.0),
        _row("participant-b", 2.0),
        _row("participant-c", 3.0),
        _row("participant-d", 4.0),
        _row("participant-e", 5.0),
    ]

    first = summarize_subject_level_inference(
        rows,
        subject_column="participant",
        value_column="value",
        exact_max_n=2,
        sign_flip_iterations=300,
        seed=77,
    )
    second = summarize_subject_level_inference(
        rows,
        subject_column="participant",
        value_column="value",
        exact_max_n=2,
        sign_flip_iterations=300,
        seed=77,
    )
    expected = _monte_carlo_sign_flip_expected([1.0, 2.0, 3.0, 4.0, 5.0], alternative="two_sided", iterations=300, seed=77)

    assert first.summary_rows[0].p_method_detail == "monte_carlo"
    assert first.summary_rows[0].p_value == expected
    assert first.to_dict() == second.to_dict()


def test_benjamini_hochberg_fdr_applies_within_configured_families() -> None:
    result = summarize_subject_level_inference(
        [
            _row("participant-a", 1.0, family="family-a", effect="effect-one"),
            _row("participant-b", 2.0, family="family-a", effect="effect-one"),
            _row("participant-c", 3.0, family="family-a", effect="effect-one"),
            _row("participant-a", 1.0, family="family-a", effect="effect-two"),
            _row("participant-b", -1.0, family="family-a", effect="effect-two"),
            _row("participant-c", 0.0, family="family-a", effect="effect-two"),
            _row("participant-a", 1.0, family="family-b", effect="effect-one"),
            _row("participant-b", 2.0, family="family-b", effect="effect-one"),
            _row("participant-c", 3.0, family="family-b", effect="effect-one"),
        ],
        subject_column="participant",
        value_column="value",
        group_columns=("family", "effect"),
        fdr_family_columns=("family",),
    )

    by_group = {(row.group_key["family"], row.group_key["effect"]): row for row in result.summary_rows}

    assert by_group[("family-a", "effect-one")].p_value == 0.25
    assert by_group[("family-a", "effect-one")].q_value == 0.5
    assert by_group[("family-a", "effect-two")].p_value == 1.0
    assert by_group[("family-a", "effect-two")].q_value == 1.0
    assert by_group[("family-b", "effect-one")].q_value == 0.25
    assert {row.q_method for row in result.multiplicity_rows} == {"benjamini_hochberg"}
    assert _provenance(result)["fdr_family_columns"] == ["family"]


def test_minimum_n_failure_returns_status_and_qc_without_inference_values() -> None:
    result = summarize_subject_level_inference(
        [_row("participant-a", 1.0)],
        subject_column="participant",
        value_column="value",
        min_n=2,
    )

    summary = result.summary_rows[0]
    assert summary.status == "failed"
    assert summary.n == 1
    assert summary.mean is None
    assert summary.ci_low is None
    assert summary.p_value is None
    assert "minimum_n_not_met" in _failed_codes(result)


def test_non_finite_and_bool_values_are_rejected_with_qc_rows() -> None:
    result = summarize_subject_level_inference(
        [
            _row("participant-a", 1.0),
            _row("participant-b", float("nan")),
            _row("participant-c", float("inf")),
            _row("participant-d", True),
        ],
        subject_column="participant",
        value_column="value",
        min_n=1,
    )

    assert result.summary_rows[0].n == 1
    assert result.summary_rows[0].mean == 1.0
    assert [row.code for row in result.qc_rows if row.field_name == "value"] == [
        "invalid_value",
        "invalid_value",
        "invalid_value",
    ]
    json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)


def test_duplicate_subjects_fail_by_default() -> None:
    result = summarize_subject_level_inference(
        [
            _row("participant-a", 1.0),
            _row("participant-a", 3.0),
            _row("participant-b", 5.0),
        ],
        subject_column="participant",
        value_column="value",
    )

    assert result.summary_rows[0].status == "failed"
    assert result.summary_rows[0].mean is None
    assert "duplicate_subject" in _failed_codes(result)


def test_duplicate_subjects_can_be_aggregated_by_mean() -> None:
    result = summarize_subject_level_inference(
        [
            _row("participant-a", 1.0),
            _row("participant-a", 3.0),
            _row("participant-b", 5.0),
        ],
        subject_column="participant",
        value_column="value",
        duplicate_subject_policy="mean",
    )

    summary = result.summary_rows[0]
    assert summary.status == "warning"
    assert summary.n == 2
    assert summary.mean == 3.5
    assert "duplicate_subject_aggregated" in _codes(result)


def test_missing_expected_subjects_are_reported_per_group() -> None:
    result = summarize_subject_level_inference(
        [
            _row("participant-a", 1.0, family="family-a"),
            _row("participant-b", 2.0, family="family-a"),
        ],
        subject_column="participant",
        value_column="value",
        group_columns=("family",),
        expected_subjects=("participant-a", "participant-b", "participant-c"),
    )

    assert [(row.subject_id, row.status, row.group_key) for row in result.missingness_rows] == [
        ("participant-c", "missing", {"family": "family-a"})
    ]


def test_grouping_columns_metadata_columns_and_labels_are_generic() -> None:
    result = summarize_subject_level_inference(
        [
            _row(
                "participant-a",
                1.0,
                region_label="region-alpha",
                effect_name="effect-alpha",
                session_id="session-a",
                roi_family="family-alpha",
                catalog_version="catalog-a",
            ),
            _row(
                "participant-b",
                2.0,
                region_label="region-alpha",
                effect_name="effect-alpha",
                session_id="session-a",
                roi_family="family-alpha",
                catalog_version="catalog-a",
            ),
        ],
        subject_column="participant",
        value_column="value",
        group_columns=("region_label", "effect_name"),
        metadata_columns=("session_id", "roi_family", "catalog_version"),
        group_label_column="region_label",
        effect_label_column="effect_name",
        measure="measure-alpha",
    )

    summary = result.summary_rows[0]
    assert summary.group_label == "region-alpha"
    assert summary.effect_label == "effect-alpha"
    assert summary.measure == "measure-alpha"
    assert summary.group_key == {"region_label": "region-alpha", "effect_name": "effect-alpha"}
    assert summary.metadata == {
        "session_id": "session-a",
        "roi_family": "family-alpha",
        "catalog_version": "catalog-a",
    }


def test_prepared_mvpa_like_rows_are_consumed_through_generic_configuration() -> None:
    result = summarize_subject_level_inference(
        [
            {
                "participant": "participant-a",
                "distance_value": 1.0,
                "roi_label": "region-alpha",
                "condition_pair_id": "pair-alpha",
                "metric": "metric-alpha",
                "engine_name": "engine-alpha",
                "normalization_method": "normalization-alpha",
                "group_key": {"task_id": "task-alpha", "run_id": "run-a"},
            },
            {
                "participant": "participant-b",
                "distance_value": 3.0,
                "roi_label": "region-alpha",
                "condition_pair_id": "pair-alpha",
                "metric": "metric-alpha",
                "engine_name": "engine-alpha",
                "normalization_method": "normalization-alpha",
                "group_key": {"task_id": "task-alpha", "run_id": "run-b"},
            },
        ],
        subject_column="participant",
        value_column="distance_value",
        group_columns=("roi_label", "condition_pair_id", "metric", "engine_name", "normalization_method"),
        metadata_columns=("group_key",),
        group_label_column="roi_label",
        effect_label_column="condition_pair_id",
        measure_column="metric",
    )

    summary = result.summary_rows[0]
    assert result.errors == ()
    assert summary.group_label == "region-alpha"
    assert summary.effect_label == "pair-alpha"
    assert summary.measure == "metric-alpha"
    assert summary.mean == 2.0
    assert summary.metadata["group_key"] is None
    assert "mixed_metadata" in _codes(result)


def test_json_safety_tsv_safe_rows_and_passthrough_provenance() -> None:
    result = summarize_subject_level_inference(
        [
            _row("participant-a", 1.0, family="family-a", effect="effect-one"),
            _row("participant-b", 2.0, family="family-a", effect="effect-one"),
            _row("participant-c", 3.0, family="family-a", effect="effect-one"),
        ],
        subject_column="participant",
        value_column="value",
        group_columns=("family", "effect"),
        fdr_family_columns=(),
        provenance_rows=(SubjectInferenceProvenanceRow("threshold_policy", {"minimum": 2}), {"key": "exclusion_policy", "value": "none"}),
    )

    json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)
    row_sets = (
        result.summary_rows,
        result.multiplicity_rows,
        result.loso_rows,
        result.missingness_rows,
        result.qc_rows,
        result.provenance_rows,
    )
    for rows in row_sets:
        for row in rows:
            tsv_row = row.to_tsv_row()
            assert tsv_row
            assert all(isinstance(value, str) for value in tsv_row.values())
            assert all("\t" not in value and "\n" not in value and "\r" not in value for value in tsv_row.values())
    assert _provenance(result)["threshold_policy"] == {"minimum": 2}
    assert _provenance(result)["exclusion_policy"] == "none"


def test_subject_inference_module_has_no_forbidden_imports_or_study_literals() -> None:
    source_path = Path(subject_inference.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_import_roots = {
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "nilearn",
        "rsatoolbox",
        "research_core",
        "research_neuro",
        "research_bids",
        "research_viz",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    forbidden_literals = {
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
    }

    assert not (imported_roots & forbidden_import_roots)
    assert all(literal not in source for literal in forbidden_literals)


def test_new_tests_use_neutral_subject_identifiers() -> None:
    source = Path(__file__).read_text(encoding="utf-8")

    assert "participant-a" in source
    assert re.search(r"sub-\d{3}", source) is None
