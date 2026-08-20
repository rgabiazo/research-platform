from __future__ import annotations

import csv
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import re

import pytest

from research_platform.analysis.mvpa.publication_export import (
    _signflip_p_two,
    _write_json_atomic,
    _write_text_atomic,
    _write_tsv_atomic,
    plan_or_execute_mvpa_publication_export,
    validate_mvpa_publication_export_document,
)


def _brute_force_signflip_p_two(values: list[float]) -> Fraction:
    exact_values = [Fraction(*value.as_integer_ratio()) for value in values]
    observed = abs(sum(exact_values, start=Fraction()))
    extreme_count = sum(
        abs(
            sum(
                (sign * value for sign, value in zip(signs, exact_values, strict=True)),
                start=Fraction(),
            )
        )
        >= observed
        for signs in itertools.product((-1, 1), repeat=len(exact_values))
    )
    return Fraction(extreme_count, 2 ** len(exact_values))


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.1, 0.2, 0.3], 0.25),
        ([1.0, -1.0], 1.0),
        ([-0.2, -0.2, 0.4], 1.0),
    ],
)
def test_exact_signflip_regression_probabilities(values: list[float], expected: float) -> None:
    assert _signflip_p_two(values, seed=11) == (expected, "exact_meet_in_middle")


@pytest.mark.parametrize(
    "values",
    [
        [0.1, -0.2, 0.4, -0.8],
        [1.0, -1.0, 2.0**-40, -(2.0**-41)],
        [2.0**100, -(2.0**100), 0.125, -0.25, 0.5],
        [1.0e-100, -2.0e-100, 4.0e-100, -8.0e-100, 16.0e-100],
    ],
)
def test_exact_signflip_agrees_with_independent_fraction_oracle(values: list[float]) -> None:
    expected = float(_brute_force_signflip_p_two(values))
    actual, method = _signflip_p_two(values, seed=17)

    assert method == "exact_meet_in_middle"
    assert actual == expected
    assert 0.0 <= actual <= 1.0


def test_exact_signflip_all_same_sign_counts_observed_and_inverse() -> None:
    values = [0.125, 0.25, 0.5, 1.0]
    probability, method = _signflip_p_two(values, seed=23)

    assert method == "exact_meet_in_middle"
    assert probability == 2 / (2 ** len(values))


def test_monte_carlo_signflip_has_stable_exact_comparison() -> None:
    values = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7] * 4 + [0.8]

    assert _signflip_p_two(values, seed=29) == (0.36209, "monte_carlo_100000")


def _config() -> dict[str, object]:
    return {
        "mvpa_publication_export": {
            "name": "baseline_crossnobis",
            "entities": {"session_id": "ses-01", "task_id": "memory", "direction": "AP"},
            "inputs": {
                "subject_level_table": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/reports/baseline_crossnobis/distances.tsv",
                },
                "audit_table": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/reports/baseline_crossnobis/audit.tsv",
                },
            },
            "outputs": {
                "root_ref": "artifact_root",
                "path": ".research-platform/mvpa/reports/baseline_crossnobis/publication",
            },
            "labels": {
                "roi_display_labels": {"RoiA": "ROI A", "RoiB": "ROI B", "RoiC": "ROI C"},
                "roi_set_display_labels": {"encoding_rois": "Encoding ROIs", "recognition_rois": "Recognition ROIs"},
                "phase_display_labels": {"encoding": "Encoding", "recognition": "Recognition"},
                "contrast_display_labels": {
                    "pair_minus_item": "Pair - item",
                    "recog_pair_minus_item": "Recognition pair - item",
                },
                "analysis_variant_display_labels": {"main": "Primary", "sensitivity": "Sensitivity"},
            },
            "statistics": {"signflip_seed": 11, "bootstrap": {"enabled": False}},
            "tables": {
                "roi_group_inference": {
                    "enabled": True,
                    "filters": {"analysis_variant": "main"},
                    "fdr_by": ["analysis_variant", "phase_id"],
                    "compact_table": {
                        "columns": ["Functional ROIs", "Contrast", "N", "Mean Distance", "95% CI", "Effect Size", "Permutation", "qFDR"]
                    },
                },
                "phase_pooled_group_inference": {
                    "enabled": True,
                    "groups": [
                        {"roi_set_id": "encoding_rois", "roi_family": "primary", "phase_id": "encoding", "roi_labels": ["RoiA", "RoiB"]},
                        {"roi_set_id": "recognition_rois", "roi_family": "primary", "phase_id": "recognition", "roi_labels": ["RoiC"]},
                    ],
                    "fdr_by": ["analysis_variant", "roi_family"],
                    "compact_table": {
                        "columns": ["Functional ROIs", "Analysis", "Contrast", "N", "Mean Distance", "95% CI", "Effect Size", "Permutation", "qFDR"]
                    },
                },
                "rdm_group_summary": {
                    "enabled": True,
                    "sources": [
                        {
                            "rdm_id": "condition_main",
                            "root_ref": "artifact_root",
                            "path": ".research-platform/mvpa/reports/baseline_crossnobis/rdms/condition_summary.tsv",
                        }
                    ],
                },
            },
            "figures": {
                "items": [
                    {
                        "figure_id": "roi_forest",
                        "kind": "roi_group_forest",
                        "output_basename": "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInferenceForest",
                        "figure_size": {"width_inches": 6, "height_inches": 4},
                    },
                    {
                        "figure_id": "roi_violin",
                        "kind": "roi_group_violin_dot",
                        "output_basename": "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInferenceViolinDot",
                        "figure_size": {"width_inches": 6, "height_inches": 4},
                    },
                    {
                        "figure_id": "phase_violin",
                        "kind": "phase_pooled_violin_dot",
                        "output_basename": "ses-01_task-memory_dir-AP_desc-SubjectLevelPhasePooledGroupInferenceViolinDot",
                        "display_labels": {"encoding": "Encoding", "recognition": "Recognition"},
                        "figure_size": {"width_inches": 6, "height_inches": 4},
                    },
                ]
            },
        }
    }


def _config_without_figures() -> dict[str, object]:
    config = _config()
    payload = config["mvpa_publication_export"]
    assert isinstance(payload, dict)
    payload["figures"] = {"items": []}
    return config


def _write_sources(artifact_root: Path) -> None:
    report_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis"
    rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    main_values = {
        ("encoding", "RoiA", "pair_minus_item"): [0.1, 0.2, 0.3],
        ("encoding", "RoiB", "pair_minus_item"): [0.3, 0.4, 0.5],
        ("recognition", "RoiC", "recog_pair_minus_item"): [0.2, 0.4, 0.6],
    }
    sensitivity_values = {
        ("encoding", "RoiA", "pair_minus_item"): [0.11, 0.21, 0.31],
        ("encoding", "RoiB", "pair_minus_item"): [0.31, 0.41, 0.51],
        ("recognition", "RoiC", "recog_pair_minus_item"): [0.18, 0.38, 0.58],
    }
    for analysis_variant, values_by_key in (("main", main_values), ("sensitivity", sensitivity_values)):
        for (phase_id, roi_label, contrast_id), values in values_by_key.items():
            for index, value in enumerate(values, start=1):
                participant_id = f"sub-{index:03d}"
                row = {
                    "participant_id": participant_id,
                    "analysis_variant": analysis_variant,
                    "phase_id": phase_id,
                    "roi_label": roi_label,
                    "contrast_id": contrast_id,
                    "crossnobis": str(value),
                    "feature_count": str(40 + index),
                    "cv_unit_count": "3",
                    "observation_count": "6",
                }
                rows.append(row)
                audit_rows.append(
                    {
                        **{key: row[key] for key in ("participant_id", "analysis_variant", "phase_id", "roi_label", "contrast_id")},
                        "subject_id": participant_id,
                        "session_id": "ses-01",
                        "task_id": "memory",
                        "mvpa_set": f"{analysis_variant}_{phase_id}",
                        "family_id": "primary_main" if analysis_variant == "main" else "sensitivity",
                        "metric": "crossnobis",
                        "engine_name": "manual_diagonal_crossnobis_v1",
                        "normalization_method": "diagonal",
                        "source_distances_relpath": f".research-platform/mvpa/{analysis_variant}_{phase_id}/distances.tsv",
                    }
                )
    _write_tsv(report_root / "distances.tsv", rows)
    _write_tsv(report_root / "audit.tsv", audit_rows)
    _write_tsv(
        report_root / "rdms" / "condition_summary.tsv",
        [
            {
                "rdm_id": "condition_main",
                "condition_a": "item",
                "condition_b": "pair",
                "condition_a_label": "Item",
                "condition_b_label": "Pair",
                "source_contrast_id": "pair_minus_item",
                "group_mean_crossnobis": "0.25",
                "n": "3",
                "sd": "0.1",
                "sem": "0.057735",
                "ci_low": "0.01",
                "ci_high": "0.49",
            }
        ],
    )


def _toy_publication_style_config() -> dict[str, object]:
    encoding_rois = [f"EncodingRoi{index}" for index in range(1, 7)]
    recognition_rois = [f"RecognitionRoi{index}" for index in range(1, 3)]
    return {
        "mvpa_publication_export": {
            "name": "baseline_crossnobis",
            "entities": {"session_id": "ses-01", "task_id": "exampletask", "direction": "AP"},
            "inputs": {
                "subject_level_table": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/reports/baseline_crossnobis/distances.tsv",
                },
                "audit_table": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/reports/baseline_crossnobis/audit.tsv",
                },
            },
            "outputs": {
                "root_ref": "artifact_root",
                "path": ".research-platform/mvpa/reports/baseline_crossnobis/publication",
            },
            "labels": {
                "phase_display_labels": {"encoding": "Encoding", "recognition": "Recognition"},
                "roi_display_labels": {roi: roi.replace("Roi", " ROI ") for roi in [*encoding_rois, *recognition_rois]},
                "roi_set_display_labels": {"encoding_rois": "Encoding ROIs", "recognition_rois": "Recognition ROIs"},
            },
            "tables": {
                "roi_group_inference": {"enabled": True, "filters": {}, "fdr_by": ["analysis_variant", "phase_id"]},
                "phase_pooled_group_inference": {
                    "enabled": True,
                    "groups": [
                        {"roi_set_id": "encoding_rois", "roi_family": "toy_roi_family", "phase_id": "encoding", "roi_labels": encoding_rois},
                        {"roi_set_id": "recognition_rois", "roi_family": "toy_roi_family", "phase_id": "recognition", "roi_labels": recognition_rois},
                    ],
                    "fdr_by": ["analysis_variant", "roi_family"],
                },
                "rdm_group_summary": {"enabled": False},
            },
            "figures": {
                "items": [],
                "manuscript_primary_main": {
                    "enabled": True,
                    "filters": {"analysis_variant": "main", "family_id": "primary_main"},
                    "expected_counts": {
                        "phase_subject_values": 6,
                        "phase_stats": 2,
                        "roi_subject_values": 24,
                        "roi_stats": 8,
                    },
                    "source_tables": {
                        "phase_subject_values": "tables/ses-01_task-exampletask_dir-AP_desc-SubjectLevelPhasePooledGroupInference_subjectValues.tsv",
                        "phase_stats": "tables/ses-01_task-exampletask_dir-AP_desc-SubjectLevelPhasePooledGroupInference_stats.tsv",
                        "roi_subject_values": "tables/ses-01_task-exampletask_dir-AP_desc-SubjectLevelROIGroupInference_subjectValues.tsv",
                        "roi_stats": "tables/ses-01_task-exampletask_dir-AP_desc-SubjectLevelROIGroupInference_stats.tsv",
                    },
                    "phase_order": ["encoding", "recognition"],
                    "roi_order": [*encoding_rois, *recognition_rois],
                    "encoding_roi_order": encoding_rois,
                    "recognition_roi_order": recognition_rois,
                    "jitter_seed": 123,
                    "output_formats": ["png", "pdf", "svg"],
                    "figure_defaults": {"fail_on_layout_warning": True, "max_label_chars": 80},
                    "figures": {
                        "phase_pooled_violin_dot": {
                            "output_basename": "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainPhasePooledCrossnobisViolinDot",
                            "figure_size": {"width_inches": 6.0, "height_inches": 4.5},
                        },
                        "roi_forest": {
                            "output_basename": "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainROICrossnobisForest",
                            "figure_size": {"width_inches": 7.5, "height_inches": 5.5},
                        },
                        "encoding_roi_dot_ci": {
                            "output_basename": "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainEncodingROICrossnobisDotCI",
                            "figure_size": {"width_inches": 7.5, "height_inches": 5.5},
                        },
                        "recognition_roi_dot_ci": {
                            "output_basename": "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainRecognitionROICrossnobisDotCI",
                            "figure_size": {"width_inches": 7.5, "height_inches": 3.6},
                        },
                    },
                },
            },
        }
    }


def _write_toy_publication_style_sources(artifact_root: Path) -> None:
    report_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis"
    encoding_rois = [f"EncodingRoi{index}" for index in range(1, 7)]
    recognition_rois = [f"RecognitionRoi{index}" for index in range(1, 3)]
    rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for analysis_variant, family_id, offset in (("main", "primary_main", 0.0), ("sensitivity", "sensitivity", 0.01)):
        for participant_index in range(1, 4):
            participant_id = f"sub-{participant_index:03d}"
            for roi_index, roi_label in enumerate(encoding_rois, start=1):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "analysis_variant": analysis_variant,
                        "phase_id": "encoding",
                        "roi_label": roi_label,
                        "contrast_id": "pair_enc_hit_minus_item_enc_hit",
                        "crossnobis": str(0.001 * participant_index + 0.002 * roi_index + offset),
                        "feature_count": str(40 + roi_index),
                        "cv_unit_count": "3",
                        "observation_count": "6",
                    }
                )
            for roi_index, roi_label in enumerate(recognition_rois, start=1):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "analysis_variant": analysis_variant,
                        "phase_id": "recognition",
                        "roi_label": roi_label,
                        "contrast_id": "pair_recog_correct_minus_item_recog_correct",
                        "crossnobis": str(0.0015 * participant_index + 0.003 * roi_index + offset),
                        "feature_count": str(48 + roi_index),
                        "cv_unit_count": "3",
                        "observation_count": "6",
                    }
                )
    for row in rows:
        audit_rows.append(
            {
                **{key: row[key] for key in ("participant_id", "analysis_variant", "phase_id", "roi_label", "contrast_id")},
                "subject_id": row["participant_id"],
                "session_id": "ses-01",
                "task_id": "exampletask",
                "mvpa_set": f"{row['analysis_variant']}_{row['phase_id']}",
                "family_id": "primary_main" if row["analysis_variant"] == "main" else "sensitivity",
                "metric": "crossnobis",
                "engine_name": "manual_diagonal_crossnobis_v1",
                "normalization_method": "diagonal",
                "source_distances_relpath": f".research-platform/mvpa/{row['analysis_variant']}_{row['phase_id']}/distances.tsv",
            }
        )
    _write_tsv(report_root / "distances.tsv", rows)
    _write_tsv(report_root / "audit.tsv", audit_rows)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_config_validation_accepts_publication_export_document() -> None:
    assert validate_mvpa_publication_export_document(_config()) == []


def test_plan_mode_is_read_only_and_separates_table_families(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_sources(artifact_root)

    result = plan_or_execute_mvpa_publication_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["executed"] is False
    assert result["table_families"]["SubjectLevelROIGroupInference"]["row_counts"]["stats"] == 3
    assert result["table_families"]["SubjectLevelPhasePooledGroupInference"]["row_counts"]["stats"] == 4
    assert result["table_families"]["RDMGroupSummary"]["row_counts"]["stats"] == 1
    assert result["recomputed_mvpa_distances"] is False
    assert not (artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication").exists()


def test_execute_writes_roi_and_phase_pooled_inference_tables(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_sources(artifact_root)

    result = plan_or_execute_mvpa_publication_export(
        _config_without_figures(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication"
    roi_stats = _read_tsv(output_root / "tables" / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInference_stats.tsv")
    phase_stats = _read_tsv(output_root / "tables" / "ses-01_task-memory_dir-AP_desc-SubjectLevelPhasePooledGroupInference_stats.tsv")
    phase_values = _read_tsv(output_root / "tables" / "ses-01_task-memory_dir-AP_desc-SubjectLevelPhasePooledGroupInference_subjectValues.tsv")
    compact = _read_tsv(output_root / "tables" / "ses-01_task-memory_dir-AP_desc-SubjectLevelPhasePooledGroupInference_table.tsv")

    roi_a = next(row for row in roi_stats if row["roi_label"] == "RoiA")
    encoding_sub1 = next(row for row in phase_values if row["participant_id"] == "sub-001" and row["analysis_variant"] == "main" and row["phase_id"] == "encoding")
    encoding_main = next(row for row in phase_stats if row["analysis_variant"] == "main" and row["phase_id"] == "encoding")

    assert result["valid"] is True
    assert result["executed"] is True
    assert roi_a["N"] == "3"
    assert math.isclose(float(roi_a["mean_crossnobis"]), 0.2)
    assert math.isclose(float(roi_a["SD"]), 0.1)
    assert roi_a["p_signflip"] == "0.25"
    assert encoding_sub1["crossnobis"] == "0.2"
    assert encoding_sub1["roi_count"] == "2"
    assert math.isclose(float(encoding_main["mean_crossnobis"]), 0.3)
    assert list(compact[0]) == [
        "Functional ROIs",
        "Analysis",
        "Contrast",
        "N",
        "Mean Distance",
        "95% CI",
        "Effect Size",
        "Permutation",
        "qFDR",
    ]
    assert compact[0]["Mean Distance"] == "0.300"


def test_publication_outputs_rdm_group_summary_without_plotting(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_sources(artifact_root)

    result = plan_or_execute_mvpa_publication_export(
        _config_without_figures(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication"
    rdm_manifest = json.loads(
        (output_root / "tables" / "ses-01_task-memory_dir-AP_desc-RDMGroupSummary_manifest.json").read_text(encoding="utf-8")
    )

    assert result["valid"] is True
    assert "not ROI-level group inference" in rdm_manifest["description"]
    assert "not phase-pooled group inference" in rdm_manifest["description"]
    assert not (output_root / "figures").exists()


def test_publication_outputs_figure_artifacts(tmp_path: Path) -> None:
    pytest.importorskip(
        "matplotlib",
        reason="publication figure assertions require the optional matplotlib dependency",
    )
    artifact_root = tmp_path / "artifacts"
    _write_sources(artifact_root)

    result = plan_or_execute_mvpa_publication_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication"
    svg = (
        output_root / "figures" / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInferenceForest.svg"
    ).read_text(encoding="utf-8")

    assert result["valid"] is True
    assert (output_root / "figures" / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInferenceForest_plotData.tsv").is_file()
    assert (output_root / "figures" / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInferenceForest_summary.tsv").is_file()
    assert "<text" in svg
    for path in list((output_root / "figures").rglob("*.tsv")) + list((output_root / "figures").rglob("*.json")):
        assert re.search(r"(^|[=\s\"])/[^\s\"]+", path.read_text(encoding="utf-8")) is None


def test_generated_text_outputs_exclude_absolute_local_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_sources(artifact_root)

    result = plan_or_execute_mvpa_publication_export(
        _config_without_figures(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication"
    assert result["valid"] is True
    for path in list(output_root.rglob("*.tsv")) + list(output_root.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"(^|[=\s\"])/[^\s\"]+", text) is None


def test_local_file_uri_is_rejected_before_publication_destination_creation(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_sources(artifact_root)
    config = _config()
    payload = config["mvpa_publication_export"]
    assert isinstance(payload, dict)
    labels = payload["labels"]
    assert isinstance(labels, dict)
    roi_labels = labels["roi_display_labels"]
    assert isinstance(roi_labels, dict)
    roi_labels["RoiA"] = "file:///home/alice/example.tsv"

    with pytest.raises(RuntimeError, match="local absolute path marker"):
        plan_or_execute_mvpa_publication_export(
            config,
            workspace_root=tmp_path,
            root_refs={"artifact_root": artifact_root},
            execute=True,
        )

    output_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication"
    forbidden_destination = (
        output_root
        / "tables"
        / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInference_stats.tsv"
    )
    assert not forbidden_destination.exists()
    assert not list(output_root.rglob("tmp*"))


@pytest.mark.parametrize("suffix", [".tsv", ".json", ".md"])
def test_publication_text_validation_preserves_sentinel_and_leaves_no_temporary_file(
    tmp_path: Path,
    suffix: str,
) -> None:
    destination = tmp_path / f"published{suffix}"
    destination.write_text("sentinel\n", encoding="utf-8")
    entries_before = set(tmp_path.iterdir())

    with pytest.raises(RuntimeError, match="local absolute path marker"):
        if suffix == ".tsv":
            _write_tsv_atomic(destination, [{"source": "command --input /mnt/example/data.tsv"}])
        elif suffix == ".json":
            _write_json_atomic(destination, {"source": r"D:\Data\example.tsv"})
        else:
            _write_text_atomic(destination, r"source \\cluster.example\example-share\data.tsv")

    assert destination.read_text(encoding="utf-8") == "sentinel\n"
    assert set(tmp_path.iterdir()) == entries_before


def test_manuscript_primary_main_filters_and_counts_in_plan_mode(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_toy_publication_style_sources(artifact_root)

    result = plan_or_execute_mvpa_publication_export(
        _toy_publication_style_config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication"
    assert result["valid"] is True
    assert result["executed"] is False
    assert result["manuscript_primary_main"]["row_counts"] == {
        "phase_subject_values": 6,
        "phase_stats": 2,
        "roi_subject_values": 24,
        "roi_stats": 8,
    }
    assert len(result["manuscript_primary_main"]["figures"]) == 4
    assert not output_root.exists()


def test_manuscript_primary_main_filters_counts_and_outputs(tmp_path: Path) -> None:
    pytest.importorskip(
        "matplotlib",
        reason="manuscript figure assertions require the optional matplotlib dependency",
    )
    artifact_root = tmp_path / "artifacts"
    _write_toy_publication_style_sources(artifact_root)

    result = plan_or_execute_mvpa_publication_export(
        _toy_publication_style_config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication"
    phase_subject = _read_tsv(output_root / "tables" / "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainPhasePooled_subjectValues.tsv")
    phase_stats = _read_tsv(output_root / "tables" / "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainPhasePooled_stats.tsv")
    roi_subject = _read_tsv(output_root / "tables" / "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainROI_subjectValues.tsv")
    roi_stats = _read_tsv(output_root / "tables" / "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainROI_stats.tsv")
    basenames = [
        "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainPhasePooledCrossnobisViolinDot",
        "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainROICrossnobisForest",
        "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainEncodingROICrossnobisDotCI",
        "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainRecognitionROICrossnobisDotCI",
    ]
    companion_tables = [
        output_root / "tables" / "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainPhasePooled_subjectValues.tsv",
        output_root / "tables" / "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainPhasePooled_stats.tsv",
        output_root / "tables" / "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainROI_subjectValues.tsv",
        output_root / "tables" / "ses-01_task-exampletask_dir-AP_desc-ManuscriptPrimaryMainROI_stats.tsv",
    ]

    assert result["valid"] is True
    assert len(result["manuscript_primary_main"]["figures"]) == 4
    assert result["manuscript_primary_main"]["row_counts"] == {
        "phase_subject_values": 6,
        "phase_stats": 2,
        "roi_subject_values": 24,
        "roi_stats": 8,
    }
    assert all(path.is_file() for path in companion_tables)
    assert len(phase_subject) == 6
    assert len(phase_stats) == 2
    assert len(roi_subject) == 24
    assert len(roi_stats) == 8
    assert {row["analysis_variant"] for row in phase_subject + roi_subject} == {"main"}
    assert {row["family_id"] for row in phase_subject + roi_subject} == {"primary_main"}
    assert not any(row["analysis_variant"] == "sensitivity" for row in phase_subject + roi_subject + phase_stats + roi_stats)
    manuscript_outputs = list((output_root / "figures").glob("*ManuscriptPrimaryMain*"))
    assert len(manuscript_outputs) == 24
    for basename in basenames:
        for suffix in (".png", ".pdf", ".svg", "_plot-data.tsv", "_summary.tsv", "_manifest.json"):
            path = output_root / "figures" / f"{basename}{suffix}"
            assert path.is_file()
        svg = (output_root / "figures" / f"{basename}.svg").read_text(encoding="utf-8")
        manifest = json.loads((output_root / "figures" / f"{basename}_manifest.json").read_text(encoding="utf-8"))
        plot_rows = _read_tsv(output_root / "figures" / f"{basename}_plot-data.tsv")
        assert "<text" in svg
        assert manifest["layout_status"] == "ok"
        assert manifest["layout_warning_count"] == 0
        assert manifest["executed"] is True
        assert manifest["recomputed_statistics"] is False
        assert manifest["source_tables"]["phase_subject_values"].endswith(
            "tables/ses-01_task-exampletask_dir-AP_desc-SubjectLevelPhasePooledGroupInference_subjectValues.tsv"
        )
        assert all(row.get("analysis_variant") != "sensitivity" for row in plot_rows)


def test_execute_refuses_overwrite(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_sources(artifact_root)

    first = plan_or_execute_mvpa_publication_export(
        _config_without_figures(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )
    second = plan_or_execute_mvpa_publication_export(
        _config_without_figures(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    assert first["valid"] is True
    assert second["valid"] is False
    assert any("refuses to overwrite" in error for error in second["errors"])
