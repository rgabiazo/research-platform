from __future__ import annotations

import csv
import json
from pathlib import Path
import re

from research_platform.analysis import _version as analysis_version
from research_platform.analysis.mvpa.derivative_publish import (
    DEFAULT_TARGET,
    plan_or_execute_mvpa_derivative_publish,
    validate_mvpa_derivative_publish_document,
)


def _config() -> dict[str, object]:
    return {
        "mvpa_derivative_publish": {
            "name": "baseline_crossnobis",
            "analysis_label": "BaselineCrossnobis",
            "derivative_name": "mvpa-crossnobis",
            "entities": {"session_id": "ses-01", "task_id": "memory", "direction": "AP"},
            "inputs": {
                "table_sets": [
                    {
                        "table_set": "baseline_crossnobis_allpairs",
                        "distances": ".research-platform/mvpa/reports/baseline_crossnobis_allpairs/distances.tsv",
                        "audit": ".research-platform/mvpa/reports/baseline_crossnobis_allpairs/audit.tsv",
                        "manifest": ".research-platform/mvpa/reports/baseline_crossnobis_allpairs/manifest.json",
                    }
                ],
                "rdm_set": {
                    "rdm_set": "baseline_crossnobis",
                    "root": ".research-platform/mvpa/reports/baseline_crossnobis/rdms",
                    "rdms": [
                        {
                            "rdm_id": "four_condition_main",
                            "basename": "ses-01_task-memory_desc-FourConditionRDM_mvpa",
                            "publish_desc": "FourConditionRDM",
                            "conditions": [
                                {"condition_id": "item_enc_hit", "label": "Item encoding hit"},
                                {"condition_id": "pair_enc_hit", "label": "Pair encoding hit"},
                                {"condition_id": "item_recog_correct", "label": "Item recognition correct"},
                                {"condition_id": "pair_recog_correct", "label": "Pair recognition correct"},
                            ],
                        }
                    ],
                },
            },
            "targets": {
                "local_artifact": {
                    "root_ref": "artifact_root",
                    "relative_path": ".research-platform/mvpa/derivatives/mvpa-crossnobis",
                    "default": True,
                },
                "dataset_derivatives": {
                    "root_ref": "dataset_derivatives_root",
                    "relative_path": "mvpa-crossnobis",
                    "default": False,
                },
            },
            "conditions": [
                {"condition_id": "item_enc_hit", "label": "Item encoding hit"},
                {"condition_id": "pair_enc_hit", "label": "Pair encoding hit"},
                {"condition_id": "item_recog_correct", "label": "Item recognition correct"},
                {"condition_id": "pair_recog_correct", "label": "Pair recognition correct"},
            ],
            "contrasts": [
                {"contrast_id": "item_enc_hit_minus_pair_enc_hit", "condition_a": "item_enc_hit", "condition_b": "pair_enc_hit"},
                {
                    "contrast_id": "item_enc_hit_minus_item_recog_correct",
                    "condition_a": "item_enc_hit",
                    "condition_b": "item_recog_correct",
                },
            ],
            "rois": [{"roi_label": "RoiA"}],
        }
    }


def _config_with_asset_groups() -> dict[str, object]:
    config = _config()
    payload = config["mvpa_derivative_publish"]
    assert isinstance(payload, dict)
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    inputs["asset_groups"] = [
        {
            "asset_group": "subject_mni_loso_roi_masks",
            "root_ref": "roi_tag_root",
            "source_glob": "rois/primary8_loso_localizer/sub-*/ses-01/func/*",
            "preserve_from": "rois/primary8_loso_localizer",
            "destination_root": ".",
            "required": True,
        },
        {
            "asset_group": "subject_t1w_roi_masks",
            "root_ref": "roi_tag_root",
            "source_glob": "rois/primary8_loso_t1w_mvpa/sub-*/ses-01/func/*",
            "preserve_from": "rois/primary8_loso_t1w_mvpa",
            "destination_root": ".",
            "required": True,
        },
        {
            "asset_group": "loso_groupmaps",
            "root_ref": "roi_tag_root",
            "source_glob": "loso_groupmaps/primary8_loso_localizer/group/ses-01/func/*",
            "preserve_from": "loso_groupmaps/primary8_loso_localizer",
            "destination_root": ".",
            "required": True,
        },
        {
            "asset_group": "generated_loso_group_masks",
            "root_ref": "roi_tag_root",
            "source_glob": "group_masks/group/ses-01/func/*",
            "preserve_from": "group_masks",
            "destination_root": ".",
            "required": True,
        },
    ]
    return config


def _write_sources(artifact_root: Path) -> None:
    report_root = artifact_root / ".research-platform" / "mvpa" / "reports"
    table_root = report_root / "baseline_crossnobis_allpairs"
    table_root.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        table_root / "distances.tsv",
        [
            {
                "participant_id": "sub-001",
                "analysis_variant": "main",
                "phase_id": "encoding_recognition",
                "roi_label": "RoiA",
                "contrast_id": "item_enc_hit_minus_pair_enc_hit",
                "crossnobis": "1.0",
                "feature_count": "10",
                "cv_unit_count": "3",
                "observation_count": "6",
            },
            {
                "participant_id": "sub-002",
                "analysis_variant": "main",
                "phase_id": "encoding_recognition",
                "roi_label": "RoiA",
                "contrast_id": "item_enc_hit_minus_pair_enc_hit",
                "crossnobis": "2.0",
                "feature_count": "11",
                "cv_unit_count": "3",
                "observation_count": "6",
            },
        ],
    )
    _write_tsv(
        table_root / "audit.tsv",
        [
            {
                "participant_id": "sub-001",
                "analysis_variant": "main",
                "phase_id": "encoding_recognition",
                "roi_label": "RoiA",
                "contrast_id": "item_enc_hit_minus_pair_enc_hit",
                "subject_id": "sub-001",
                "session_id": "ses-01",
                "task_id": "memory",
                "mvpa_set": "allpairs",
                "family_id": "allpairs_main",
                "metric": "crossnobis",
                "engine_name": "manual_diagonal_crossnobis_v1",
                "normalization_method": "diagonal",
                "source_distances_relpath": ".research-platform/mvpa/allpairs/distances.tsv",
            }
        ],
    )
    (table_root / "manifest.json").write_text('{"table_set":"baseline_crossnobis_allpairs"}\n', encoding="utf-8")

    rdm_root = report_root / "baseline_crossnobis" / "rdms"
    rdm_root.mkdir(parents=True, exist_ok=True)
    basename = "ses-01_task-memory_desc-FourConditionRDM_mvpa"
    conditions = ["item_enc_hit", "pair_enc_hit", "item_recog_correct", "pair_recog_correct"]
    matrix_rows = []
    for row_condition in conditions:
        row = {"condition_id": row_condition}
        for column_condition in conditions:
            row[column_condition] = "0.0" if row_condition == column_condition else "1.0"
        matrix_rows.append(row)
    _write_tsv(rdm_root / f"{basename}_matrix.tsv", matrix_rows)
    long_rows = [
        {
            "rdm_id": "four_condition_main",
            "condition_a": a,
            "condition_b": b,
            "condition_a_label": a,
            "condition_b_label": b,
            "group_mean_crossnobis": str(index + 1),
            "n": "2",
            "sd": "0.1",
            "sem": "0.1",
            "ci_low": "0.0",
            "ci_high": "2.0",
            "source_contrast_id": f"{a}_minus_{b}",
        }
        for index, (a, b) in enumerate(_pairs(conditions))
    ]
    _write_tsv(rdm_root / f"{basename}_long.tsv", long_rows)
    _write_tsv(rdm_root / f"{basename}_summary.tsv", long_rows)
    subject_rows = []
    for participant_id in ("sub-001", "sub-002"):
        for index, (a, b) in enumerate(_pairs(conditions), start=1):
            subject_rows.append(
                {
                    "participant_id": participant_id,
                    "rdm_id": "four_condition_main",
                    "condition_a": a,
                    "condition_b": b,
                    "crossnobis": str(index),
                    "source_contrast_id": f"{a}_minus_{b}",
                    "pooled_roi_count": "1",
                    "pooled_row_count": "1",
                }
            )
    _write_tsv(rdm_root / f"{basename}_subject-pairs.tsv", subject_rows)
    (rdm_root / f"{basename}_manifest.json").write_text('{"layout_qc":{"status":"ok"}}\n', encoding="utf-8")
    (rdm_root / f"{basename}.svg").write_text("<svg><text>RDM</text></svg>\n", encoding="utf-8")
    (rdm_root / f"{basename}.pdf").write_bytes(b"%PDF-1.4\n")
    (rdm_root / f"{basename}.png").write_bytes(b"png\n")


def _write_asset_sources(
    roi_tag_root: Path,
    *,
    volume_root: Path | None = None,
    include_unmapped_path: bool = False,
) -> None:
    tool_path = "/home/alice/tools/ants/bin/antsApplyTransforms"
    mni_mask = roi_tag_root / "rois/primary8_loso_localizer/sub-001/ses-01/func/sub-001_ses-01_task-memory_space-MNI152NLin6Asym_label-RoiA_mask.nii.gz"
    t1w_mask = roi_tag_root / "rois/primary8_loso_t1w_mvpa/sub-001/ses-01/func/sub-001_ses-01_task-memory_run-01_space-T1w_label-RoiA_mask.nii.gz"
    qc_payload = {
        "source_mask": str(t1w_mask),
        "command_argv": [tool_path, "-i", str(mni_mask), "-o", str(t1w_mask)],
        "nested": {"inputs": [str(mni_mask)]},
    }
    if volume_root is not None:
        qc_payload["nested"]["template"] = str(volume_root / "tpl-MNI152NLin6Asym_res-2_T1w.nii.gz")
    if include_unmapped_path:
        qc_payload["nested"]["unmapped"] = "/mnt/private-example/raw/sub-001/anat.nii.gz"
    files = {
        "rois/primary8_loso_localizer/sub-001/ses-01/func/sub-001_ses-01_task-memory_space-MNI152NLin6Asym_label-RoiA_mask.nii.gz": b"mni-mask\n",
        "rois/primary8_loso_localizer/sub-001/ses-01/func/sub-001_ses-01_task-memory_space-MNI152NLin6Asym_label-RoiA_mask.json": json.dumps({"space": "MNI152NLin6Asym", "source_mask": str(mni_mask)}).encode(),
        "rois/primary8_loso_t1w_mvpa/sub-001/ses-01/func/sub-001_ses-01_task-memory_run-01_space-T1w_label-RoiA_mask.nii.gz": b"t1w-mask\n",
        "rois/primary8_loso_t1w_mvpa/sub-001/ses-01/func/sub-001_ses-01_task-memory_run-01_space-T1w_label-RoiA_qc.json": json.dumps(qc_payload).encode(),
        "rois/primary8_loso_t1w_mvpa/sub-001/ses-01/func/sub-001_ses-01_task-memory_run-01_space-T1w_label-RoiA_provenance.json": json.dumps({"executable": tool_path, "transform": str(t1w_mask)}).encode(),
        "loso_groupmaps/primary8_loso_localizer/group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_zstat.nii.gz": b"zstat\n",
        "loso_groupmaps/primary8_loso_localizer/group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_zstat.json": json.dumps({"heldout_subject": "sub-001", "source_zstat": str(roi_tag_root / "loso_groupmaps/primary8_loso_localizer/group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_zstat.nii.gz")}).encode(),
        "group_masks/group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_mask.nii.gz": b"group-mask\n",
        "group_masks/group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_mask.json": json.dumps({"heldout_subject": "sub-001", "source_masks": [str(mni_mask)]}).encode(),
    }
    for relative_path, content in files.items():
        path = roi_tag_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _pairs(conditions: list[str]) -> list[tuple[str, str]]:
    return [(left, right) for index, left in enumerate(conditions) for right in conditions[index + 1 :]]


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
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


def test_config_validation_accepts_publish_document() -> None:
    assert validate_mvpa_derivative_publish_document(_config()) == []


def test_plan_mode_writes_nothing_and_defaults_to_local_artifact(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    _write_sources(artifact_root)

    result = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
    )

    assert result["valid"] is True
    assert result["executed"] is False
    assert result["target"] == DEFAULT_TARGET
    assert result["target_root"]["relative_path"] == ".research-platform/mvpa/derivatives/mvpa-crossnobis"
    assert result["row_counts"]["subject_rdm_rows"] == 8
    assert not (artifact_root / ".research-platform" / "mvpa" / "derivatives").exists()


def test_asset_group_plan_includes_roi_outputs_without_writing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    roi_tag_root = tmp_path / "roi_tag"
    _write_sources(artifact_root)
    _write_asset_sources(roi_tag_root)

    result = plan_or_execute_mvpa_derivative_publish(
        _config_with_asset_groups(),
        workspace_root=tmp_path,
        root_refs={
            "artifact_root": artifact_root,
            "dataset_derivatives_root": dataset_derivatives_root,
            "roi_tag_root": roi_tag_root,
        },
    )

    output_paths = {record["relative_path"] for record in result["outputs"].values()}
    assert result["valid"] is True
    assert result["executed"] is False
    assert result["asset_groups"][0]["asset_group"] == "subject_mni_loso_roi_masks"
    assert "sub-001/ses-01/func/sub-001_ses-01_task-memory_space-MNI152NLin6Asym_label-RoiA_mask.nii.gz" in output_paths
    assert "sub-001/ses-01/func/sub-001_ses-01_task-memory_run-01_space-T1w_label-RoiA_mask.nii.gz" in output_paths
    assert "group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_zstat.nii.gz" in output_paths
    assert "group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_mask.nii.gz" in output_paths
    assert not (artifact_root / ".research-platform" / "mvpa" / "derivatives").exists()


def test_execute_writes_bids_derivative_layout_and_subject_rdms(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    monkeypatch.setattr(analysis_version.metadata, "version", lambda name: "0.1.0a1")
    _write_sources(artifact_root)

    result = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "derivatives" / "mvpa-crossnobis"
    assert result["valid"] is True
    assert result["executed"] is True
    assert (output_root / "dataset_description.json").is_file()
    dataset_description = json.loads(
        (output_root / "dataset_description.json").read_text(encoding="utf-8")
    )
    assert dataset_description["GeneratedBy"] == [{"Name": "research-analysis", "Version": "0.1.0a1"}]
    software_versions = _read_tsv(Path(result["outputs"]["software_versions_tsv"]["path"]))
    assert software_versions == [{"name": "research-analysis", "version": "0.1.0a1", "role": "publisher"}]
    group_sidecar = json.loads(
        Path(result["outputs"]["group_distances_json"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert group_sidecar["GeneratedBy"] == "research-platform"
    assert (output_root / "README.md").is_file()
    assert (output_root / "config" / "task-memory_desc-BaselineCrossnobis_conditions.tsv").is_file()
    assert (
        output_root / "group" / "ses-01" / "figures" / "ses-01_task-memory_dir-AP_desc-FourConditionRDM.svg"
    ).is_file()
    svg = (
        output_root / "group" / "ses-01" / "figures" / "ses-01_task-memory_dir-AP_desc-FourConditionRDM.svg"
    ).read_text(encoding="utf-8")
    assert "<text" in svg
    subject_rdm = output_root / "sub-001" / "ses-01" / "rsa" / "sub-001_ses-01_task-memory_dir-AP_desc-BaselineCrossnobis_rdm.tsv"
    rows = _read_tsv(subject_rdm)
    item_row = next(row for row in rows if row["rdm_id"] == "four_condition_main" and row["condition_id"] == "item_enc_hit")
    pair_row = next(row for row in rows if row["rdm_id"] == "four_condition_main" and row["condition_id"] == "pair_enc_hit")
    assert item_row["item_enc_hit"] == "0.0"
    assert item_row["pair_enc_hit"] == pair_row["item_enc_hit"]
    assert result["preserves_editable_svg_pdf"] is True
    assert result["recomputed_values"] is False


def test_execute_copies_roi_asset_groups_and_records_relative_sources(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    roi_tag_root = tmp_path / "roi_tag"
    volume_root = Path("/mnt/portable-example/derivatives")
    _write_sources(artifact_root)
    _write_asset_sources(roi_tag_root, volume_root=volume_root)

    result = plan_or_execute_mvpa_derivative_publish(
        _config_with_asset_groups(),
        workspace_root=tmp_path,
        root_refs={
            "artifact_root": artifact_root,
            "dataset_derivatives_root": dataset_derivatives_root,
            "roi_tag_root": roi_tag_root,
            "volume_root": volume_root,
        },
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "derivatives" / "mvpa-crossnobis"
    assert result["valid"] is True
    assert result["executed"] is True
    assert (
        output_root
        / "sub-001/ses-01/func/sub-001_ses-01_task-memory_space-MNI152NLin6Asym_label-RoiA_mask.nii.gz"
    ).read_bytes() == b"mni-mask\n"
    assert (
        output_root
        / "sub-001/ses-01/func/sub-001_ses-01_task-memory_run-01_space-T1w_label-RoiA_qc.json"
    ).is_file()
    qc_payload = json.loads(
        (
            output_root
            / "sub-001/ses-01/func/sub-001_ses-01_task-memory_run-01_space-T1w_label-RoiA_qc.json"
        ).read_text(encoding="utf-8")
    )
    assert qc_payload["command_argv"][0] == "antsApplyTransforms"
    assert qc_payload["command_argv"][2]["root_ref"] == "roi_tag_root"
    assert qc_payload["nested"]["template"] == {
        "root_ref": "volume_root",
        "relative_path": "tpl-MNI152NLin6Asym_res-2_T1w.nii.gz",
    }
    assert re.search(r"(^|[=\s\"])/[^\s\"]+", json.dumps(qc_payload)) is None
    assert (
        output_root
        / "group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_zstat.nii.gz"
    ).read_bytes() == b"zstat\n"
    assert (
        output_root
        / "group/ses-01/func/ses-01_task-memory_space-MNI152NLin6Asym_desc-contrast_heldout-sub001_mask.nii.gz"
    ).read_bytes() == b"group-mask\n"
    manifest = json.loads(
        (
            output_root
            / "sourcedata/manifests/ses-01_task-memory_desc-BaselineCrossnobis_publishManifest.json"
        ).read_text(encoding="utf-8")
    )
    source_inputs = manifest["source_inputs"]
    assert any(row["source_kind"] == "asset_group" for row in source_inputs)
    assert all(tmp_path.as_posix() not in json.dumps(row) for row in source_inputs)
    assert any(row["relative_path"].startswith("roi_tag_root/") for row in source_inputs)


def test_execute_copies_publication_tables_and_figures_as_asset_groups(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    _write_sources(artifact_root)
    publication_root = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "publication"
    (publication_root / "tables").mkdir(parents=True)
    (publication_root / "figures").mkdir(parents=True)
    (publication_root / "tables" / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInference_stats.tsv").write_text(
        "analysis_variant\tN\tmean_crossnobis\nmain\t2\t0.1\n",
        encoding="utf-8",
    )
    (publication_root / "tables" / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInference_manifest.json").write_text(
        '{"source_table_relpath": ".research-platform/mvpa/reports/baseline_crossnobis/distances.tsv"}\n',
        encoding="utf-8",
    )
    (publication_root / "figures" / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInferenceForest.svg").write_text(
        "<svg><text>Forest</text></svg>\n",
        encoding="utf-8",
    )
    (publication_root / "figures" / "ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInferenceForest.png").write_bytes(b"png\n")
    manuscript_companions = [
        "ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainPhasePooled_subjectValues.tsv",
        "ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainPhasePooled_stats.tsv",
        "ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainROI_subjectValues.tsv",
        "ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainROI_stats.tsv",
    ]
    for filename in manuscript_companions:
        (publication_root / "tables" / filename).write_text(
            "participant_id\tanalysis_variant\tfamily_id\tphase_id\tcrossnobis\nsub-001\tmain\tprimary_main\tencoding\t0.1\n",
            encoding="utf-8",
        )
    manuscript_basenames = [
        "ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainPhasePooledCrossnobisViolinDot",
        "ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainROICrossnobisForest",
        "ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainEncodingROICrossnobisDotCI",
        "ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainRecognitionROICrossnobisDotCI",
    ]
    for manuscript_basename in manuscript_basenames:
        (publication_root / "figures" / f"{manuscript_basename}.svg").write_text(
            "<svg><text>Primary</text></svg>\n",
            encoding="utf-8",
        )
        (publication_root / "figures" / f"{manuscript_basename}.pdf").write_bytes(b"%PDF-1.4\n")
        (publication_root / "figures" / f"{manuscript_basename}.png").write_bytes(b"png\n")
        (publication_root / "figures" / f"{manuscript_basename}_plot-data.tsv").write_text(
            "participant_id\tanalysis_variant\tfamily_id\tphase_id\tplot_value\nsub-001\tmain\tprimary_main\tencoding\t0.1\n",
            encoding="utf-8",
        )
        (publication_root / "figures" / f"{manuscript_basename}_summary.tsv").write_text(
            "phase_id\tN\tmean_crossnobis\nencoding\t1\t0.1\n",
            encoding="utf-8",
        )
        (publication_root / "figures" / f"{manuscript_basename}_manifest.json").write_text(
            '{"layout_status":"ok","source_tables":{"phase_subject_values":"tables/phase.tsv"}}\n',
            encoding="utf-8",
        )

    config = _config()
    payload = config["mvpa_derivative_publish"]
    assert isinstance(payload, dict)
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    inputs["asset_groups"] = [
        {
            "asset_group": "publication_group_tables",
            "root_ref": "artifact_root",
            "source_glob": ".research-platform/mvpa/reports/baseline_crossnobis/publication/tables/*",
            "preserve_from": ".research-platform/mvpa/reports/baseline_crossnobis/publication/tables",
            "destination_root": "group/ses-01/tables",
            "required": True,
        },
        {
            "asset_group": "publication_group_figures",
            "root_ref": "artifact_root",
            "source_glob": ".research-platform/mvpa/reports/baseline_crossnobis/publication/figures/*",
            "preserve_from": ".research-platform/mvpa/reports/baseline_crossnobis/publication/figures",
            "destination_root": "group/ses-01/figures",
            "required": True,
        },
    ]

    result = plan_or_execute_mvpa_derivative_publish(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "derivatives" / "mvpa-crossnobis"
    assert result["valid"] is True
    assert (
        output_root
        / "group/ses-01/tables/ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInference_stats.tsv"
    ).is_file()
    assert (
        output_root
        / "group/ses-01/figures/ses-01_task-memory_dir-AP_desc-SubjectLevelROIGroupInferenceForest.svg"
    ).read_text(encoding="utf-8") == "<svg><text>Forest</text></svg>\n"
    assert (
        output_root
        / "group/ses-01/tables/ses-01_task-memory_dir-AP_desc-ManuscriptPrimaryMainPhasePooled_subjectValues.tsv"
    ).is_file()
    for filename in manuscript_companions:
        assert (output_root / f"group/ses-01/tables/{filename}").is_file()
    for manuscript_basename in manuscript_basenames:
        assert (
            output_root
            / f"group/ses-01/figures/{manuscript_basename}.svg"
        ).read_text(encoding="utf-8") == "<svg><text>Primary</text></svg>\n"
        for suffix in (".png", ".pdf", ".svg", "_plot-data.tsv", "_summary.tsv", "_manifest.json"):
            assert (output_root / f"group/ses-01/figures/{manuscript_basename}{suffix}").is_file()
    assert len(list((output_root / "group/ses-01/figures").glob("*ManuscriptPrimaryMain*"))) == 24
    manifest = json.loads(
        (
            output_root
            / "sourcedata/manifests/ses-01_task-memory_desc-BaselineCrossnobis_publishManifest.json"
        ).read_text(encoding="utf-8")
    )
    assert any(
        row.get("source_kind") == "asset_group" and row.get("source_id") == "publication_group_tables"
        for row in manifest["source_inputs"]
    )
    assert any(
        row.get("source_kind") == "asset_group" and row.get("source_id") == "publication_group_figures"
        for row in manifest["source_inputs"]
    )


def test_unmapped_asset_json_absolute_path_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    roi_tag_root = tmp_path / "roi_tag"
    _write_sources(artifact_root)
    _write_asset_sources(roi_tag_root, include_unmapped_path=True)

    result = plan_or_execute_mvpa_derivative_publish(
        _config_with_asset_groups(),
        workspace_root=tmp_path,
        root_refs={
            "artifact_root": artifact_root,
            "dataset_derivatives_root": dataset_derivatives_root,
            "roi_tag_root": roi_tag_root,
        },
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "derivatives" / "mvpa-crossnobis"
    assert result["valid"] is False
    assert result["executed"] is False
    assert any("unmapped local absolute path" in error for error in result["errors"])
    assert not output_root.exists()


def test_missing_optional_asset_group_warns_without_failing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    roi_tag_root = tmp_path / "roi_tag"
    _write_sources(artifact_root)
    config = _config_with_asset_groups()
    payload = config["mvpa_derivative_publish"]
    assert isinstance(payload, dict)
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    first_asset_group = inputs["asset_groups"][0]
    assert isinstance(first_asset_group, dict)
    first_asset_group["required"] = False
    inputs["asset_groups"] = [first_asset_group]

    result = plan_or_execute_mvpa_derivative_publish(
        config,
        workspace_root=tmp_path,
        root_refs={
            "artifact_root": artifact_root,
            "dataset_derivatives_root": dataset_derivatives_root,
            "roi_tag_root": roi_tag_root,
        },
    )

    assert result["valid"] is True
    assert result["asset_groups"][0]["file_count"] == 0
    assert any("matched no files" in warning for warning in result["warnings"])


def test_published_tsv_and_json_outputs_exclude_absolute_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    _write_sources(artifact_root)

    result = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
        execute=True,
    )

    assert result["valid"] is True
    output_root = artifact_root / ".research-platform" / "mvpa" / "derivatives" / "mvpa-crossnobis"
    forbidden = tmp_path.as_posix()
    for path in list(output_root.rglob("*.tsv")) + list(output_root.rglob("*.json")):
        assert forbidden not in path.read_text(encoding="utf-8")


def test_published_text_leak_fails_before_writing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    _write_sources(artifact_root)
    audit_path = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis_allpairs" / "audit.tsv"
    audit_path.write_text(
        "participant_id\tanalysis_variant\tphase_id\troi_label\tcontrast_id\tsubject_id\tsession_id\ttask_id\tmvpa_set\tfamily_id\tmetric\tengine_name\tnormalization_method\tsource_distances_relpath\n"
        "sub-001\tmain\tencoding\tRoiA\tpair_minus_item\tsub-001\tses-01\tmemory\tallpairs\tmain\tcrossnobis\tmanual_diagonal_crossnobis_v1\tdiagonal\t/home/alice/source.tsv\n",
        encoding="utf-8",
    )

    result = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "derivatives" / "mvpa-crossnobis"
    assert result["valid"] is False
    assert result["executed"] is False
    assert any("Published text output" in error for error in result["errors"])
    assert not output_root.exists()


def test_embedded_posix_path_in_published_cell_fails_before_writing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    _write_sources(artifact_root)
    audit_path = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis_allpairs" / "audit.tsv"
    audit_rows = _read_tsv(audit_path)
    audit_rows[0]["source_distances_relpath"] = "command --input /home/alice/example.tsv"
    _write_tsv(audit_path, audit_rows)

    result = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
        execute=True,
    )

    output_root = artifact_root / ".research-platform" / "mvpa" / "derivatives" / "mvpa-crossnobis"
    assert result["valid"] is False
    assert result["executed"] is False
    assert any("Published text output" in error for error in result["errors"])
    assert not output_root.exists()


def test_missing_required_rdm_input_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    _write_sources(artifact_root)
    missing = (
        artifact_root
        / ".research-platform"
        / "mvpa"
        / "reports"
        / "baseline_crossnobis"
        / "rdms"
        / "ses-01_task-memory_desc-FourConditionRDM_mvpa_subject-pairs.tsv"
    )
    missing.unlink()

    result = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
    )

    assert result["valid"] is False
    assert any("source RDM artifact is missing" in error for error in result["errors"])


def test_execute_refuses_overwrite(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    _write_sources(artifact_root)

    first = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
        execute=True,
    )
    second = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
        execute=True,
    )

    assert first["valid"] is True
    assert second["valid"] is False
    assert any("refuses to overwrite" in error for error in second["errors"])


def test_dataset_derivatives_target_is_only_selected_explicitly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_derivatives_root = tmp_path / "dataset" / "derivatives"
    _write_sources(artifact_root)

    local = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
    )
    dataset = plan_or_execute_mvpa_derivative_publish(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root, "dataset_derivatives_root": dataset_derivatives_root},
        target="dataset_derivatives",
    )

    assert local["target"] == "local_artifact"
    assert dataset["target"] == "dataset_derivatives"
    assert dataset["target_root"]["root_ref"] == "dataset_derivatives_root"
    assert not dataset_derivatives_root.exists()
