from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
ANALYSIS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-analysis"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(ANALYSIS_PACKAGE_ROOT / "src"))

from research_platform.core.cli import _build_parser, main
from research_platform.core.config import load_yaml, write_yaml


class _FakeMvpaPlan:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


class MvpaCliTests(unittest.TestCase):
    def _write_workspace(self, workspace_root: Path) -> tuple[Path, Path]:
        write_yaml(
            workspace_root / "WORKSPACE.yaml",
            {
                "paths": {
                    "artifacts_root": "./artifacts",
                    "datasets_root": "./datasets",
                    "ops_root": "./ops",
                },
                "repos": {
                    "project_root": "./project",
                    "pipelines_root": "./pipelines",
                },
                "projects": {"default": "project-default"},
            },
        )
        project_root = workspace_root / "project" / "project-default"
        derivatives_root = workspace_root / "datasets" / "study-a" / "derivatives"
        feat_root = workspace_root / "external" / "feat"
        (workspace_root / "artifacts").mkdir(parents=True, exist_ok=True)
        derivatives_root.mkdir(parents=True, exist_ok=True)
        feat_root.mkdir(parents=True, exist_ok=True)
        write_yaml(project_root / "project.yaml", {"name": "project-default", "version": "0.1.0"})
        write_yaml(
            project_root / "config" / "dataset.yaml",
            {"dataset": {"primary": "study-a", "derivatives_root": "datasets/study-a/derivatives"}},
        )
        write_yaml(
            project_root / "config" / "analysis.yaml",
            {"analysis": {"external_input_roots": {"feat_root": {"local_root": "external/feat"}}}},
        )
        return project_root, derivatives_root

    def _write_mvpa_configs(
        self,
        project_root: Path,
        derivatives_root: Path,
        *,
        invalid: bool = False,
        missing_runtime_root: bool = False,
        missing_publication_root: bool = False,
        publication_enabled: bool | None = True,
    ) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "roi_sets" / "memory_roi_set.yaml",
            {
                "roi_set": {
                    "name": "memory_roi_set",
                    "outputs": {"root_ref": "dataset_derivatives_root", "path": "roi-runtime"},
                    "rois": [
                        {
                            "label": "SeedSphere",
                            "family": "coordinate_sphere",
                            "coordinate": [0, -52, 26],
                            "radius_mm": 6,
                        }
                    ],
                }
            },
        )
        roi_runtime = derivatives_root / "roi-runtime"
        roi_runtime.mkdir(parents=True, exist_ok=True)
        (roi_runtime / "SeedSphere.nii.gz").write_text("placeholder mask\n", encoding="utf-8")
        (roi_runtime / "SeedSphere.json").write_text("{}\n", encoding="utf-8")
        document = self._mvpa_document()
        if invalid:
            document["mvpa_set"]["distance"]["metrics"] = ["unsupported"]  # type: ignore[index]
        if missing_runtime_root:
            document["mvpa_set"]["outputs"].pop("runtime_root")  # type: ignore[index]
        if missing_publication_root:
            document["mvpa_set"]["outputs"].pop("published_root")  # type: ignore[index]
        if publication_enabled is None:
            document["mvpa_set"].pop("publication")  # type: ignore[index]
        else:
            document["mvpa_set"]["publication"]["enabled"] = publication_enabled  # type: ignore[index]
        write_yaml(project_root / "config" / "analysis" / "mvpa" / "memory_mvpa.yaml", document)

    def _mvpa_document(self) -> dict[str, object]:
        return {
            "mvpa_set": {
                "name": "memory_mvpa",
                "subjects": ["sub-001"],
                "sessions": ["ses-01"],
                "runs": ["01"],
                "entities": {
                    "task": "${RP_TASK:-memory}",
                    "direction": "AP",
                    "model": "ModelA",
                    "space": "MNI152NLin6Asym",
                    "resolution": "2",
                },
                "conditions": [
                    {"id": "faces", "aliases": ["face_trials"], "selector": {"trial_type": "face"}},
                    {"id": "places", "aliases": ["place_trials"], "selector": {"trial_type": "place"}},
                ],
                "pattern_sources": [
                    {
                        "name": "prepared_patterns",
                        "backend": "bids_derivative_pattern_table",
                        "root_ref": "feat_root",
                        "path": "tables/{subject_id}.tsv",
                    }
                ],
                "roi_sources": [
                    {
                        "name": "memory_rois",
                        "source": "roi_set_runtime",
                        "roi_set_ref": "memory_roi_set",
                        "root_ref": "dataset_derivatives_root",
                        "roi_labels": ["SeedSphere"],
                        "mask_template": "roi-runtime/{roi_label}.nii.gz",
                        "sidecar_template": "roi-runtime/{roi_label}.json",
                    }
                ],
                "distance": {
                    "metrics": ["crossnobis"],
                    "engine": "native_reference",
                    "cross_validation": {"unit": "run"},
                    "noise_normalization": {"method": "identity"},
                },
                "outputs": {
                    "runtime_root": {"root_ref": "artifact_root", "path": ".research-platform/mvpa/{mvpa_set}"},
                    "published_root": {"root_ref": "dataset_derivatives_root", "path": "mvpa-crossnobis/{mvpa_set}"},
                },
                "publication": {
                    "enabled": True,
                    "derivative_name": "mvpa-crossnobis",
                    "write_json_sidecars": True,
                    "write_provenance": True,
                },
                "missing_input_policy": "warn",
            }
        }

    def _write_mvpa_phase_4b4_runtime_inputs(self, workspace_root: Path) -> Path:
        runtime_root = workspace_root / "artifacts" / ".research-platform" / "mvpa" / "memory_mvpa"
        files = {
            "analysis/prepared-summaries/summaries.tsv": "group_id\tmetric\tmean_distance\nunit\tcrossnobis\t0.25\n",
            "analysis/prepared-summaries/qc.tsv": "level\tstatus\nsummary\tok\n",
            "analysis/prepared-summaries/provenance.json": "{\"phase\":\"summary\"}\n",
            "analysis/prepared-distances/distances.tsv": "group_id\tmetric\tdistance\nunit\tcrossnobis\t0.25\n",
            "analysis/prepared-distances/qc.tsv": "level\tstatus\ndistance\tok\n",
            "analysis/prepared-distances/provenance.json": "{\"phase\":\"distance\"}\n",
            "neuro/pattern-extraction/provenance.json": "{\"phase\":\"extraction\"}\n",
            "analysis/prepared-patterns/provenance.json": "{\"phase\":\"prepared\"}\n",
        }
        for relative_path, content in files.items():
            path = runtime_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return runtime_root

    def _write_mvpa_table_export_config(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "mvpa_tables" / "baseline_crossnobis.yaml",
            {
                "mvpa_table_export": {
                    "name": "baseline_crossnobis",
                    "entities": {"session_id": "ses-01", "task_id": "memory"},
                    "sources": [
                        {
                            "mvpa_set": "encoding_main",
                            "phase_id": "encoding",
                            "analysis_variant": "main",
                            "family_id": "primary_main",
                            "expected_rows": 2,
                        }
                    ],
                    "expected": {"total_rows": 2, "participant_count": 2},
                    "outputs": {
                        "root_ref": "artifact_root",
                        "path": ".research-platform/mvpa/reports/{table_set}",
                        "filename_prefix": "ses-01_task-memory",
                    },
                }
            },
        )

    def _write_mvpa_table_source(self, workspace_root: Path) -> Path:
        path = (
            workspace_root
            / "artifacts"
            / ".research-platform"
            / "mvpa"
            / "encoding_main"
            / "analysis"
            / "prepared-distances"
            / "distances.tsv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\t".join(
                [
                    "group_id",
                    "group_key",
                    "condition_id_a",
                    "condition_id_b",
                    "condition_pair_id",
                    "distance",
                    "metric",
                    "engine_name",
                    "normalization_method",
                    "cv_unit_count",
                    "feature_count",
                    "observation_count",
                    "context",
                    "subject_id",
                    "session_id",
                    "task_id",
                    "roi_label",
                ]
            )
            + "\n"
            + "g1\t{}\tpair\titem\tpair_minus_item\t0.1\tcrossnobis\tmanual_diagonal_crossnobis_v1\tdiagonal\t3\t12\t6\t{}\tsub-001\tses-01\tmemory\tRoiA\n"
            + "g2\t{}\tpair\titem\tpair_minus_item\t0.2\tcrossnobis\tmanual_diagonal_crossnobis_v1\tdiagonal\t3\t11\t6\t{}\tsub-002\tses-01\tmemory\tRoiA\n",
            encoding="utf-8",
        )
        return path

    def _write_mvpa_figure_export_config(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "mvpa_figures" / "baseline_crossnobis.yaml",
            {
                "mvpa_figure_export": {
                    "name": "baseline_crossnobis",
                    "input": {
                        "table_set": "baseline_crossnobis",
                        "root_ref": "artifact_root",
                        "path": ".research-platform/mvpa/reports/{table_set}/subject.tsv",
                    },
                    "outputs": {
                        "root_ref": "artifact_root",
                        "path": ".research-platform/mvpa/reports/{figure_set}/figures",
                    },
                    "figures": [
                        {
                            "figure_id": "roiwise_encoding_main",
                            "kind": "strip_mean_ci",
                            "filters": {
                                "analysis_variant": "main",
                                "phase_id": "encoding",
                                "contrast_id": "pair_minus_item",
                            },
                            "x": "roi_label",
                            "y": "crossnobis",
                            "order": ["RoiA"],
                            "title": "Encoding crossnobis",
                            "ylabel": "Crossnobis",
                            "xlabel": "ROI",
                            "output_basename": "ses-01_task-memory_desc-RoiwiseEncodingCrossnobis_mvpa",
                            "output_formats": ["svg", "pdf", "png"],
                        }
                    ],
                }
            },
        )

    def _write_mvpa_figure_subject_table(self, workspace_root: Path) -> Path:
        path = workspace_root / "artifacts" / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "subject.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "participant_id\tanalysis_variant\tphase_id\troi_label\tcontrast_id\tcrossnobis\tfeature_count\tcv_unit_count\tobservation_count\n"
            "sub-001\tmain\tencoding\tRoiA\tpair_minus_item\t0.1\t12\t3\t6\n"
            "sub-002\tmain\tencoding\tRoiA\tpair_minus_item\t0.2\t11\t3\t6\n",
            encoding="utf-8",
        )
        return path

    def _write_mvpa_rdm_export_config(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "mvpa_rdms" / "baseline_crossnobis.yaml",
            {
                "mvpa_rdm_export": {
                    "name": "baseline_crossnobis",
                    "input": {
                        "table_set": "baseline_crossnobis",
                        "root_ref": "artifact_root",
                        "path": ".research-platform/mvpa/reports/{table_set}/subject.tsv",
                    },
                    "outputs": {
                        "root_ref": "artifact_root",
                        "path": ".research-platform/mvpa/reports/{rdm_set}/rdms",
                    },
                    "rdms": [
                        {
                            "rdm_id": "encoding_pair_item_main",
                            "enabled": True,
                            "kind": "rdm_heatmap",
                            "value_column": "crossnobis",
                            "filters": {
                                "analysis_variant": "main",
                                "phase_id": "encoding",
                                "contrast_id": "pair_minus_item",
                            },
                            "conditions": [
                                {"condition_id": "item", "label": "Item"},
                                {"condition_id": "pair", "label": "Pair"},
                            ],
                            "pair_mappings": [
                                {"contrast_id": "pair_minus_item", "condition_a": "item", "condition_b": "pair"},
                            ],
                            "aggregate_within_participant": {"enabled": False, "method": "mean"},
                            "strict_all_pairs": True,
                            "diagonal_value": 0.0,
                            "symmetric": True,
                            "title": "Encoding pair/item RDM",
                            "colorbar_label": "Crossnobis",
                            "output_basename": "ses-01_task-memory_desc-EncodingPairItemCrossnobisRDM_mvpa",
                            "output_formats": ["svg", "pdf", "png"],
                        }
                    ],
                }
            },
        )

    def _write_mvpa_derivative_publish_config(self, project_root: Path) -> None:
        write_yaml(
            project_root / "config" / "analysis" / "mvpa_derivatives" / "baseline_crossnobis.yaml",
            {
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
                                    "rdm_id": "encoding_pair_item_main",
                                    "basename": "ses-01_task-memory_desc-EncodingPairItemCrossnobisRDM_mvpa",
                                    "publish_desc": "EncodingPairItemCrossnobisRDM",
                                    "conditions": [
                                        {"condition_id": "item", "label": "Item"},
                                        {"condition_id": "pair", "label": "Pair"},
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
                        {"condition_id": "item", "label": "Item"},
                        {"condition_id": "pair", "label": "Pair"},
                    ],
                    "contrasts": [
                        {"contrast_id": "pair_minus_item", "condition_a": "item", "condition_b": "pair"},
                    ],
                    "rois": [{"roi_label": "RoiA"}],
                }
            },
        )

    def _write_mvpa_derivative_publish_sources(self, workspace_root: Path) -> None:
        table_root = workspace_root / "artifacts" / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis_allpairs"
        table_root.mkdir(parents=True, exist_ok=True)
        (table_root / "distances.tsv").write_text(
            "participant_id\tanalysis_variant\tphase_id\troi_label\tcontrast_id\tcrossnobis\tfeature_count\tcv_unit_count\tobservation_count\n"
            "sub-001\tmain\tencoding\tRoiA\tpair_minus_item\t0.1\t12\t3\t6\n"
            "sub-002\tmain\tencoding\tRoiA\tpair_minus_item\t0.2\t11\t3\t6\n",
            encoding="utf-8",
        )
        (table_root / "audit.tsv").write_text(
            "participant_id\tanalysis_variant\tphase_id\troi_label\tcontrast_id\tsubject_id\tsession_id\ttask_id\tmvpa_set\tfamily_id\tmetric\tengine_name\tnormalization_method\tsource_distances_relpath\n"
            "sub-001\tmain\tencoding\tRoiA\tpair_minus_item\tsub-001\tses-01\tmemory\tallpairs\tmain\tcrossnobis\tmanual_diagonal_crossnobis_v1\tdiagonal\t.research-platform/mvpa/allpairs/distances.tsv\n",
            encoding="utf-8",
        )
        (table_root / "manifest.json").write_text("{}\n", encoding="utf-8")
        rdm_root = workspace_root / "artifacts" / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "rdms"
        rdm_root.mkdir(parents=True, exist_ok=True)
        basename = "ses-01_task-memory_desc-EncodingPairItemCrossnobisRDM_mvpa"
        (rdm_root / f"{basename}_matrix.tsv").write_text("condition_id\tItem\tPair\nitem\t0.0\t0.15\npair\t0.15\t0.0\n", encoding="utf-8")
        (rdm_root / f"{basename}_long.tsv").write_text(
            "rdm_id\tcondition_a\tcondition_b\tcondition_a_label\tcondition_b_label\tgroup_mean_crossnobis\tn\tsd\tsem\tci_low\tci_high\tsource_contrast_id\n"
            "encoding_pair_item_main\titem\tpair\tItem\tPair\t0.15\t2\t0.1\t0.1\t0.0\t0.3\tpair_minus_item\n",
            encoding="utf-8",
        )
        (rdm_root / f"{basename}_summary.tsv").write_text(
            "rdm_id\tcondition_a\tcondition_b\tcondition_a_label\tcondition_b_label\tgroup_mean_crossnobis\tn\tsd\tsem\tci_low\tci_high\tsource_contrast_id\n"
            "encoding_pair_item_main\titem\tpair\tItem\tPair\t0.15\t2\t0.1\t0.1\t0.0\t0.3\tpair_minus_item\n",
            encoding="utf-8",
        )
        (rdm_root / f"{basename}_subject-pairs.tsv").write_text(
            "participant_id\trdm_id\tcondition_a\tcondition_b\tcrossnobis\tsource_contrast_id\tpooled_roi_count\tpooled_row_count\n"
            "sub-001\tencoding_pair_item_main\titem\tpair\t0.1\tpair_minus_item\t1\t1\n"
            "sub-002\tencoding_pair_item_main\titem\tpair\t0.2\tpair_minus_item\t1\t1\n",
            encoding="utf-8",
        )
        (rdm_root / f"{basename}_manifest.json").write_text("{}\n", encoding="utf-8")
        (rdm_root / f"{basename}.svg").write_text("<svg><text>Encoding</text></svg>\n", encoding="utf-8")
        (rdm_root / f"{basename}.pdf").write_bytes(b"%PDF-1.4\n")
        (rdm_root / f"{basename}.png").write_bytes(b"png\n")

    def _run_cli(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        buffer = io.StringIO()
        env = {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}
        if extra_env:
            env.update(extra_env)
        with mock.patch.dict(os.environ, env, clear=False):
            with redirect_stdout(buffer):
                exit_code = main(args)
        return exit_code, buffer.getvalue()

    def _run_cli_exit(self, args: list[str], *, workspace_root: Path) -> str:
        with mock.patch.dict(os.environ, {"RESEARCH_PLATFORM_ROOT": str(workspace_root)}, clear=False):
            with self.assertRaises(SystemExit) as error:
                main(args)
        return str(error.exception)

    def _subparser(self, parser: object, name: str) -> object:
        for action in getattr(parser, "_actions", []):
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict) and name in choices:
                return choices[name]
        raise AssertionError(f"Subparser {name!r} not found.")

    def _subparser_choices(self, parser: object) -> dict[str, object]:
        for action in getattr(parser, "_actions", []):
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict) and choices:
                return choices
        raise AssertionError("Subparser choices not found.")

    def _workspace_file_snapshot(self, workspace_root: Path) -> list[str]:
        return sorted(path.relative_to(workspace_root).as_posix() for path in workspace_root.rglob("*") if path.is_file())

    def _workspace_tree_snapshot(self, workspace_root: Path) -> list[str]:
        rows: list[str] = []
        for path in workspace_root.rglob("*"):
            prefix = "d" if path.is_dir() else "f"
            rows.append(f"{prefix}:{path.relative_to(workspace_root).as_posix()}")
        return sorted(rows)

    def _plan_payload(
        self,
        *,
        status: str = "valid",
        noise_method: str = "identity",
        backend: str = "materialized_pattern_table",
        adapter_available: bool = True,
        materialization_supported: bool = True,
        ready_for_execution: bool = True,
    ) -> dict[str, object]:
        warnings = ["Synthetic discovery warning."] if status == "warning" else []
        schema_valid = status != "invalid"
        representation_kind = (
            "prepared_features" if backend == "materialized_pattern_table" else "image"
        )
        return {
            "status": status,
            "valid": schema_valid,
            "schema_valid": schema_valid,
            "plan_valid": schema_valid and status in {"valid", "warning"},
            "ready_for_materialization": schema_valid and materialization_supported,
            "ready_for_execution": schema_valid and ready_for_execution,
            "errors": [],
            "warnings": warnings,
            "analysis_units": [
                {
                    "unit_id": "unit-sub-001-ses-01-run-01",
                    "subject_id": "sub-001",
                    "session_id": "ses-01",
                    "run_id": "01",
                    "task_id": "memory",
                }
            ],
            "pattern_rows": [
                {
                    "unit_id": "unit-sub-001-ses-01-run-01",
                    "subject_id": "sub-001",
                    "session_id": "ses-01",
                    "run_id": "01",
                    "task_id": "memory",
                    "condition_id": "faces",
                    "source_name": "prepared_patterns",
                    "backend_name": backend,
                    "representation_kind": representation_kind,
                    "cross_validation_label": "01",
                    "backend_metadata": (
                        {
                            "roi_source_name": "memory_rois",
                            "roi_label": "SeedSphere",
                            "feature_count": 2,
                            "voxel_index_hash": "synthetic-index-hash",
                            "feature_space_id": "synthetic-feature-space",
                            "roi_definition_id": "synthetic-roi-definition",
                            "mean_centering_applied": False,
                            "mean_centering_scope": "none",
                            "noise_status": "usable" if noise_method == "diagonal" else "unused",
                            "noise_usable": noise_method == "diagonal",
                        }
                        if representation_kind == "prepared_features"
                        else {}
                    ),
                }
            ],
            "roi_source_rows": [
                {
                    "source_name": "memory_rois",
                    "task_id": "${RP_TASK:-memory}",
                    "status": "ok",
                    "mask_exists": True,
                    "mask_path": "root_ref:roi_root/masks/seed.nii",
                }
            ],
            "pattern_source_summaries": (
                [
                    {
                        "source_name": "prepared_patterns",
                        "backend": "materialized_pattern_table",
                        "root_ref": "pattern_root",
                        "portable_reference": "root_ref:pattern_root/patterns.tsv",
                        "source_sha256": "a" * 64,
                        "counts": {
                            "selected_rows": 1,
                            "usable_selected_rows": 1,
                        },
                        "usable_coverage_complete": True,
                    }
                ]
                if representation_kind == "prepared_features"
                else []
            ),
            "input_checks": (
                [
                    {
                        "input_kind": "root",
                        "status": "ok",
                        "exists": True,
                    },
                    {
                        "input_kind": "pe_image",
                        "status": "ok",
                        "exists": True,
                    },
                ]
                if representation_kind == "image"
                else []
            ),
            "event_threshold_rows": [],
            "mean_centering": {"enabled": False, "scope": "none"},
            "distances": [
                {
                    "metric": "crossnobis",
                    "engine": "native_reference",
                    "cv_unit": "run",
                    "noise_normalization_method": noise_method,
                    "status": "planned",
                }
            ],
            "adapter_availability": [
                {
                    "source_name": "prepared_patterns",
                    "backend": backend,
                    "registered": True,
                    "available": adapter_available,
                    "schema_supported": True,
                    "planning_supported": adapter_available,
                    "materialization_supported": materialization_supported,
                    "execution_available": adapter_available and ready_for_execution,
                    "ready_for_execution": adapter_available and ready_for_execution,
                    "reason": (
                        "Synthetic adapter is ready."
                        if adapter_available and ready_for_execution
                        else "Synthetic adapter execution is deferred."
                    ),
                }
            ],
            "backend_summary": {
                "integration_attempted": True,
                "ready_for_materialization": schema_valid and materialization_supported,
                "ready_for_execution": adapter_available and ready_for_execution,
            },
            "executed": False,
        }

    def _fake_extraction_result(
        self,
        *,
        usable: bool = True,
        representation_kind: str = "prepared_features",
    ) -> dict[str, object]:
        pattern_rows: list[dict[str, object]] = []
        if usable:
            for run_id, base in (("01", 1.0), ("02", 2.0)):
                for condition_id, offset in (("faces", 0.0), ("places", 1.0)):
                    pattern_rows.append(
                        {
                            "pattern_id": f"pat-sub-001-{condition_id}-run-{run_id}-seed",
                            "condition_id": condition_id,
                            "cv_unit": "run",
                            "subject_id": "sub-001",
                            "session_id": "ses-01",
                            "run_id": run_id,
                            "task_id": "memory",
                            "direction": "AP",
                            "model": "ModelA",
                            "pattern_source_name": "first_level_pe",
                            "roi_source_name": "memory_rois",
                            "roi_label": "SeedSphere",
                            "feature_count": 2,
                            "voxel_order": "c_flat_index",
                            "voxel_index_hash": "abc123",
                            "feature_space_id": "synthetic-feature-space",
                            "roi_definition_id": "synthetic-roi-definition",
                            "cross_validation_label": run_id,
                            "mean_centering_applied": False,
                            "mean_centering_scope": "none",
                            "noise_status": "unused",
                            "noise_usable": False,
                            "usable": True,
                            "feature_values": [base + offset, base + offset + 0.5],
                        }
                    )
        return {
            "representation_kind": representation_kind,
            "source_audit_kind": (
                "pattern_materialization"
                if representation_kind == "prepared_features"
                else "pattern_extraction"
            ),
            "pattern_rows": pattern_rows,
            "qc_rows": [
                {
                    "subject_id": "sub-001",
                    "session_id": "ses-01",
                    "run_id": "01",
                    "condition_id": "faces",
                    "roi_label": "SeedSphere",
                    "pattern_source_name": "first_level_pe",
                    "roi_source_name": "memory_rois",
                    "status": "ok" if usable else "missing_condition_pe_row",
                    "usable": usable,
                    "warnings": [],
                    "errors": [],
                    "event_threshold_status": "not_evaluated",
                }
            ],
            "provenance": {
                "representation_kind": representation_kind,
                "source_audit_kind": (
                    "pattern_materialization"
                    if representation_kind == "prepared_features"
                    else "pattern_extraction"
                ),
                "sources": [
                    {
                        "source_name": "prepared_patterns",
                        "source_sha256": "a" * 64,
                        "portable_reference": "root_ref:pattern_root/patterns.tsv",
                    }
                ],
            },
            "warnings": [],
            "errors": [],
            "executed": True,
        }

    def _fake_prepared_result(self) -> dict[str, object]:
        rows = [
            {
                "pattern_id": "pat-sub-001-faces-run-01-seed",
                "condition_id": "faces",
                "cv_unit": "run",
                "cv_label": "01",
                "feature_count": 2,
                "feature_values": [1.0, 1.5],
            },
            {
                "pattern_id": "pat-sub-001-places-run-01-seed",
                "condition_id": "places",
                "cv_unit": "run",
                "cv_label": "01",
                "feature_count": 2,
                "feature_values": [2.0, 2.5],
            },
        ]
        return {
            "groups": [
                {
                    "group_id": "sub-001-seed",
                    "group_key": {"subject_id": "sub-001", "roi_label": "SeedSphere"},
                    "group_by": ["subject_id", "roi_label"],
                    "cv_unit": "run",
                    "cv_labels": ["01"],
                    "condition_ids": ["faces", "places"],
                    "feature_count": 2,
                    "voxel_order": "c_flat_index",
                    "voxel_index_hash": "abc123",
                    "rows": rows,
                }
            ],
            "qc_rows": [{"level": "group", "status": "ok", "code": "prepared", "message": "ok", "usable": True}],
            "provenance": {"phase": "unit"},
            "warnings": [],
            "errors": [],
            "executed": True,
        }

    def _fake_distance_result(self) -> dict[str, object]:
        return {
            "distances": [
                {
                    "group_id": "sub-001-seed",
                    "group_key": {"subject_id": "sub-001", "roi_label": "SeedSphere"},
                    "condition_id_a": "faces",
                    "condition_id_b": "places",
                    "distance": 0.25,
                    "metric": "crossnobis",
                    "engine_name": "native_reference",
                    "normalization_method": "identity",
                    "cv_unit_count": 1,
                    "feature_count": 2,
                    "observation_count": 2,
                }
            ],
            "qc_rows": [{"level": "group", "status": "ok", "code": "distance", "message": "ok", "usable": True}],
            "provenance": {"phase": "unit"},
            "warnings": [],
            "errors": [],
            "executed": True,
        }

    def _fake_summary_result(self, *, rows: bool = True, errors: list[str] | None = None) -> dict[str, object]:
        summary_rows: list[dict[str, object]] = []
        if rows:
            summary_rows.append(
                {
                    "group_id": "group-seed",
                    "condition_id_a": "faces",
                    "condition_id_b": "places",
                    "metric": "crossnobis",
                    "engine_name": "native_reference",
                    "normalization_method": "identity",
                    "n": 1,
                    "mean_distance": 0.25,
                    "std_distance": 0.0,
                    "sem_distance": 0.0,
                    "min_distance": 0.25,
                    "max_distance": 0.25,
                }
            )
        return {
            "summary_rows": summary_rows,
            "qc_rows": [{"level": "summary", "status": "ok", "code": "summary", "message": "ok"}],
            "provenance": {"phase": "unit"},
            "warnings": [],
            "errors": errors or [],
            "executed": True,
        }

    def _fake_runtime_transaction_functions(
        self,
        calls: dict[str, object] | None = None,
    ) -> tuple[object, object, object]:
        from research_platform.neuro.mvpa.runtime_transaction import (
            MvpaRuntimeTransactionResult,
            plan_mvpa_runtime_transaction,
            runtime_output_specs,
        )

        def execute_transaction(
            plan: object,
            *,
            write_outputs: object,
            manifest_payload: dict[str, object],
        ) -> MvpaRuntimeTransactionResult:
            with tempfile.TemporaryDirectory() as staging_dir:
                writer_records = write_outputs(Path(staging_dir))  # type: ignore[operator]
            if calls is not None:
                calls["transaction_invoked"] = True
                calls["transaction_writer_records"] = sorted(writer_records)
            return MvpaRuntimeTransactionResult(
                final_root=plan.final_root,  # type: ignore[attr-defined]
                representation_kind=plan.representation_kind,  # type: ignore[attr-defined]
                manifest={"status": "succeeded", "outputs": []},
                output_sha256={},
                writer_records=writer_records,
            )

        return plan_mvpa_runtime_transaction, execute_transaction, runtime_output_specs

    def test_mvpa_lifecycle_commands_list_show_validate_doctor_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)

            list_code, list_output = self._run_cli(["analysis", "mvpa", "list", "--project", "project-default"], workspace_root=workspace_root)
            show_code, show_output = self._run_cli(
                ["analysis", "mvpa", "show", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
                extra_env={"RP_TASK": "changed"},
            )
            validate_code, validate_output = self._run_cli(
                ["analysis", "mvpa", "validate", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            doctor_code, doctor_output = self._run_cli(
                ["analysis", "mvpa", "doctor", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            plan_code, plan_output = self._run_cli(
                ["analysis", "mvpa", "plan", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
                extra_env={"RP_TASK": "changed"},
            )

        list_payload = json.loads(list_output)
        show_payload = json.loads(show_output)
        validate_payload = json.loads(validate_output)
        doctor_payload = json.loads(doctor_output)
        plan_payload = json.loads(plan_output)
        self.assertEqual(list_code, 0)
        self.assertEqual(list_payload["mvpa_sets"], ["memory_mvpa"])
        self.assertEqual(show_code, 0)
        self.assertEqual(show_payload["document"]["mvpa_set"]["entities"]["task"], "${RP_TASK:-memory}")
        self.assertEqual(validate_code, 0)
        self.assertTrue(validate_payload["valid"])
        self.assertTrue(validate_payload["schema_valid"])
        self.assertEqual(doctor_code, 1)
        self.assertFalse(doctor_payload["valid"])
        self.assertTrue(doctor_payload["schema_valid"])
        self.assertFalse(doctor_payload["ready_for_execution"])
        self.assertEqual(doctor_payload["referenced_roi_sets"], ["memory_roi_set"])
        self.assertIn("feat_root", doctor_payload["root_refs"])
        self.assertEqual(plan_code, 1)
        self.assertFalse(plan_payload["valid"])
        self.assertTrue(plan_payload["schema_valid"])
        self.assertFalse(plan_payload["plan_valid"])
        self.assertFalse(plan_payload["ready_for_execution"])
        self.assertFalse(plan_payload["executed"])
        self.assertEqual(plan_payload["referenced_roi_sets"], ["memory_roi_set"])
        self.assertEqual(plan_payload["roi_source_rows"][0]["task_id"], "${RP_TASK:-memory}")
        self.assertTrue(all(row["status"] == "planned" for row in plan_payload["distances"]))

    def test_mvpa_doctor_separates_schema_validity_from_adapter_readiness(self) -> None:
        deferred_backends = (
            "bids_derivative_pattern_table",
            "nilearn_glm",
            "surface_cifti",
        )
        for backend in deferred_backends:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp_dir:
                workspace_root = Path(tmp_dir)
                project_root, derivatives_root = self._write_workspace(workspace_root)
                self._write_mvpa_configs(project_root, derivatives_root)
                before = self._workspace_tree_snapshot(workspace_root)

                def plan(
                    _document: dict[str, object],
                    context: dict[str, object] | None = None,
                    *,
                    roots: dict[str, Path] | None = None,
                    roi_sets: dict[str, dict[str, object]] | None = None,
                    enable_backend_discovery: bool = False,
                    exact_units: list[dict[str, object]] | None = None,
                    unit_key_columns: list[str] | None = None,
                ) -> _FakeMvpaPlan:
                    return _FakeMvpaPlan(
                        self._plan_payload(
                            status="deferred",
                            backend=backend,
                            adapter_available=False,
                            ready_for_execution=False,
                        )
                    )

                with mock.patch(
                    "research_platform.core.cli._mvpa_lifecycle_functions",
                    return_value=(lambda _document: [], plan),
                ):
                    code, output = self._run_cli(
                        ["analysis", "mvpa", "doctor", "memory_mvpa", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )
                after = self._workspace_tree_snapshot(workspace_root)

            payload = json.loads(output)
            self.assertEqual(code, 1)
            self.assertFalse(payload["valid"])
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["ready_for_execution"])
            self.assertEqual(payload["adapter_availability"][0]["backend"], backend)
            self.assertFalse(payload["adapter_availability"][0]["available"])
            self.assertEqual(after, before)

    def test_mvpa_materialized_adapter_is_ready_after_runtime_transaction_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            before = self._workspace_tree_snapshot(workspace_root)

            def plan(
                _document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                return _FakeMvpaPlan(
                    self._plan_payload(
                        backend="materialized_pattern_table",
                        adapter_available=True,
                        materialization_supported=True,
                        ready_for_execution=True,
                    )
                )

            with mock.patch(
                "research_platform.core.cli._mvpa_lifecycle_functions",
                return_value=(lambda _document: [], plan),
            ):
                validate_code, validate_output = self._run_cli(
                    ["analysis", "mvpa", "validate", "memory_mvpa", "--project", "project-default"],
                    workspace_root=workspace_root,
                )
                doctor_code, doctor_output = self._run_cli(
                    ["analysis", "mvpa", "doctor", "memory_mvpa", "--project", "project-default"],
                    workspace_root=workspace_root,
                )
                plan_code, plan_output = self._run_cli(
                    ["analysis", "mvpa", "plan", "memory_mvpa", "--project", "project-default"],
                    workspace_root=workspace_root,
                )
            after = self._workspace_tree_snapshot(workspace_root)

        validate_payload = json.loads(validate_output)
        doctor_payload = json.loads(doctor_output)
        plan_payload = json.loads(plan_output)
        self.assertEqual(validate_code, 0)
        self.assertTrue(validate_payload["valid"])
        self.assertTrue(validate_payload["schema_valid"])
        self.assertEqual(doctor_code, 0)
        self.assertTrue(doctor_payload["valid"])
        self.assertTrue(doctor_payload["schema_valid"])
        self.assertTrue(doctor_payload["ready_for_materialization"])
        self.assertTrue(doctor_payload["ready_for_execution"])
        self.assertEqual(doctor_payload["errors"], [])
        self.assertEqual(plan_code, 0)
        self.assertTrue(plan_payload["valid"])
        self.assertTrue(plan_payload["schema_valid"])
        self.assertTrue(plan_payload["ready_for_materialization"])
        self.assertTrue(plan_payload["ready_for_execution"])
        self.assertFalse(plan_payload["executed"])
        adapter = plan_payload["adapter_availability"][0]
        self.assertEqual(adapter["backend"], "materialized_pattern_table")
        self.assertTrue(adapter["schema_supported"])
        self.assertTrue(adapter["planning_supported"])
        self.assertTrue(adapter["materialization_supported"])
        self.assertTrue(plan_payload["backend_summary"]["ready_for_materialization"])
        self.assertTrue(plan_payload["backend_summary"]["ready_for_execution"])
        self.assertEqual(after, before)

    def test_mvpa_doctor_rejects_materialized_rows_without_usable_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)

            payload = self._plan_payload(ready_for_execution=True)
            source_summary = payload["pattern_source_summaries"][0]  # type: ignore[index]
            source_summary["counts"]["usable_selected_rows"] = 0  # type: ignore[index]
            source_summary["usable_coverage_complete"] = False  # type: ignore[index]

            def plan(
                _document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                return _FakeMvpaPlan(payload)

            with mock.patch(
                "research_platform.core.cli._mvpa_lifecycle_functions",
                return_value=(lambda _document: [], plan),
            ):
                code, output = self._run_cli(
                    ["analysis", "mvpa", "doctor", "memory_mvpa", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

        doctor_payload = json.loads(output)
        checks = {row["id"]: row for row in doctor_payload["checks"]}
        self.assertEqual(code, 1)
        self.assertFalse(doctor_payload["ready_for_execution"])
        self.assertTrue(checks["selected_coverage"]["ok"])
        self.assertFalse(checks["condition_roi_coverage"]["ok"])
        self.assertIn("usable", checks["condition_roi_coverage"]["messages"][0].lower())

    def test_mvpa_doctor_keeps_fsl_image_execution_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)

            def plan(
                _document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                return _FakeMvpaPlan(
                    self._plan_payload(
                        backend="fsl_feat_pe",
                        adapter_available=True,
                        ready_for_execution=False,
                    )
                )

            with mock.patch(
                "research_platform.core.cli._mvpa_lifecycle_functions",
                return_value=(lambda _document: [], plan),
            ):
                code, output = self._run_cli(
                    ["analysis", "mvpa", "doctor", "memory_mvpa", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["valid"])
        self.assertTrue(payload["schema_valid"])
        self.assertFalse(payload["ready_for_execution"])
        self.assertEqual(payload["adapter_availability"][0]["backend"], "fsl_feat_pe")
        self.assertTrue(payload["adapter_availability"][0]["available"])
        self.assertTrue(payload["backend_summary"]["integration_attempted"])
        checks = {row["id"]: row for row in payload["checks"]}
        self.assertFalse(checks["adapter_availability"]["ok"])
        self.assertFalse(checks["transaction_support"]["ok"])
        self.assertIn("deferred", " ".join(checks["adapter_availability"]["findings"]).lower())

    def test_mvpa_lifecycle_help_distinguishes_schema_readiness_and_planning(self) -> None:
        parser = _build_parser()
        analysis_parser = self._subparser(parser, "analysis")
        mvpa_parser = self._subparser(analysis_parser, "mvpa")
        help_text = " ".join(mvpa_parser.format_help().split())

        self.assertIn("Validate an MVPA configuration schema without checking runtime readiness.", help_text)
        self.assertIn("Check MVPA adapter and input readiness without executing analysis.", help_text)
        self.assertIn("Resolve MVPA pattern sources and render a no-write planning preview.", help_text)

    def test_mvpa_invalid_config_reports_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root, invalid=True)

            code, output = self._run_cli(
                ["analysis", "mvpa", "validate", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["valid"])
        self.assertFalse(payload["schema_valid"])
        self.assertTrue(any("distance.metrics" in error for error in payload["errors"]))

    def test_mvpa_missing_project_and_config_return_json_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, _derivatives_root = self._write_workspace(workspace_root)

            missing_project_error = self._run_cli_exit(
                ["analysis", "mvpa", "list", "--project", "missing-project"],
                workspace_root=workspace_root,
            )
            missing_config_error = self._run_cli_exit(
                ["analysis", "mvpa", "show", "missing_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            self.assertTrue(project_root.exists())

        self.assertIn("missing-project", missing_project_error)
        self.assertIn("project/missing-project", missing_project_error)
        self.assertIn("rp project init missing-project", missing_project_error)
        missing_payload = json.loads(missing_config_error)
        self.assertIn("MVPA set 'missing_mvpa' was not found", missing_payload["error"])
        self.assertIn("project/project-default/config/analysis/mvpa/missing_mvpa.yaml", missing_payload["expected_path"])
        self.assertEqual(
            missing_payload["next_step"],
            (
                "rp analysis mvpa init missing_mvpa --project project-default "
                "--template materialized-crossnobis"
            ),
        )
        self.assertNotIn("Traceback", missing_config_error)

    def test_mvpa_doctor_and_plan_reject_filename_and_config_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            path = project_root / "config" / "analysis" / "mvpa" / "memory_mvpa.yaml"
            document = load_yaml(path)
            document["mvpa_set"]["name"] = "different-name"
            write_yaml(path, document)

            validate_code, validate_output = self._run_cli(
                ["analysis", "mvpa", "validate", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            doctor_code, doctor_output = self._run_cli(
                ["analysis", "mvpa", "doctor", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            plan_code, plan_output = self._run_cli(
                ["analysis", "mvpa", "plan", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )

        validate_payload = json.loads(validate_output)
        self.assertEqual(validate_code, 0)
        self.assertTrue(validate_payload["schema_valid"])
        for code, output in ((doctor_code, doctor_output), (plan_code, plan_output)):
            payload = json.loads(output)
            self.assertEqual(code, 1)
            self.assertTrue(payload["schema_valid"])
            self.assertFalse(payload["plan_valid"])
            self.assertTrue(any("filename/name mismatch" in error for error in payload["errors"]))

    def test_mvpa_commands_do_not_write_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            before = self._workspace_file_snapshot(workspace_root)

            expected_codes = {"validate": 0, "doctor": 1, "plan": 1}
            for command, expected_code in expected_codes.items():
                code, _output = self._run_cli(
                    ["analysis", "mvpa", command, "memory_mvpa", "--project", "project-default"],
                    workspace_root=workspace_root,
                )
                self.assertEqual(code, expected_code)

            after = self._workspace_file_snapshot(workspace_root)

        self.assertEqual(after, before)

    def test_mvpa_export_tables_plan_is_read_only_and_uses_relative_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_table_export_config(project_root)
            self._write_mvpa_table_source(workspace_root)
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "export-tables", "baseline_crossnobis", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            after = self._workspace_file_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 0, output)
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["table_a_columns"], [
            "participant_id",
            "analysis_variant",
            "phase_id",
            "roi_label",
            "contrast_id",
            "crossnobis",
            "feature_count",
            "cv_unit_count",
            "observation_count",
        ])
        self.assertEqual(payload["row_counts"]["table_a"], 2)
        self.assertFalse(any("reports/baseline_crossnobis" in path for path in after))
        self.assertEqual(after, before)
        source_relpath = payload["sources"][0]["source_distances_relpath"]
        self.assertFalse(source_relpath.startswith("/"))

    def test_mvpa_export_tables_execute_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_table_export_config(project_root)
            self._write_mvpa_table_source(workspace_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "export-tables", "baseline_crossnobis", "--project", "project-default", "--execute"],
                workspace_root=workspace_root,
            )
            payload = json.loads(output)
            outputs = payload["outputs"]
            output_exists = {
                "subject_level_distances": Path(outputs["subject_level_distances"]["path"]).is_file(),
                "subject_level_audit": Path(outputs["subject_level_audit"]["path"]).is_file(),
                "manifest": Path(outputs["manifest"]["path"]).is_file(),
            }

        self.assertEqual(code, 0, output)
        self.assertTrue(payload["executed"])
        self.assertEqual(
            output_exists,
            {"subject_level_distances": True, "subject_level_audit": True, "manifest": True},
        )

    def test_mvpa_export_figures_plan_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_figure_export_config(project_root)
            self._write_mvpa_figure_subject_table(workspace_root)
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "export-figures", "baseline_crossnobis", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            after = self._workspace_file_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 0, output)
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["command"], "analysis mvpa export-figures")
        self.assertEqual(payload["figures"][0]["figure_id"], "roiwise_encoding_main")
        self.assertEqual(payload["figures"][0]["row_counts"], {"plot_data": 2, "summary": 1})
        self.assertEqual(payload["figures"][0]["outputs"]["figure_svg"]["status"], "planned")
        self.assertFalse(any("reports/baseline_crossnobis/figures" in path for path in after))
        self.assertEqual(after, before)

    def test_mvpa_export_rdms_plan_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_rdm_export_config(project_root)
            self._write_mvpa_figure_subject_table(workspace_root)
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "export-rdms", "baseline_crossnobis", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            after = self._workspace_file_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 0, output)
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["command"], "analysis mvpa export-rdms")
        self.assertEqual(payload["rdms"][0]["rdm_id"], "encoding_pair_item_main")
        self.assertEqual(payload["rdms"][0]["row_counts"], {"matrix": 2, "long": 1, "subject_pairs": 2, "summary": 1})
        self.assertEqual(payload["rdms"][0]["outputs"]["figure_svg"]["status"], "planned")
        self.assertFalse(any("reports/baseline_crossnobis/rdms" in path for path in after))
        self.assertEqual(after, before)

    def test_mvpa_publish_derivatives_plan_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_derivative_publish_config(project_root)
            self._write_mvpa_derivative_publish_sources(workspace_root)
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "publish-derivatives", "baseline_crossnobis", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            after = self._workspace_file_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 0, output)
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["command"], "analysis mvpa publish-derivatives")
        self.assertEqual(payload["target"], "local_artifact")
        self.assertEqual(payload["default_target"], "local_artifact")
        self.assertEqual(payload["target_root"]["relative_path"], ".research-platform/mvpa/derivatives/mvpa-crossnobis")
        self.assertFalse(any("derivatives/mvpa-crossnobis" in path for path in after))
        self.assertEqual(after, before)

    def test_mvpa_publish_derivatives_execute_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_derivative_publish_config(project_root)
            self._write_mvpa_derivative_publish_sources(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "mvpa",
                    "publish-derivatives",
                    "baseline_crossnobis",
                    "--project",
                    "project-default",
                    "--execute",
                ],
                workspace_root=workspace_root,
            )
            payload = json.loads(output)
            output_root = workspace_root / "artifacts" / ".research-platform" / "mvpa" / "derivatives" / "mvpa-crossnobis"
            outputs_exist = {
                "dataset_description": (output_root / "dataset_description.json").is_file(),
                "readme": (output_root / "README.md").is_file(),
                "group_figure": (
                    output_root
                    / "group"
                    / "ses-01"
                    / "figures"
                    / "ses-01_task-memory_dir-AP_desc-EncodingPairItemCrossnobisRDM.svg"
                ).is_file(),
            }

        self.assertEqual(code, 0, output)
        self.assertTrue(payload["executed"])
        self.assertEqual(outputs_exist, {"dataset_description": True, "readme": True, "group_figure": True})

    def test_mvpa_publish_derivatives_dataset_target_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_derivative_publish_config(project_root)
            self._write_mvpa_derivative_publish_sources(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "mvpa",
                    "publish-derivatives",
                    "baseline_crossnobis",
                    "--project",
                    "project-default",
                    "--target",
                    "dataset_derivatives",
                ],
                workspace_root=workspace_root,
            )

        payload = json.loads(output)
        self.assertEqual(code, 0, output)
        self.assertEqual(payload["target"], "dataset_derivatives")
        self.assertEqual(payload["target_root"]["root_ref"], "dataset_derivatives_root")
        self.assertFalse((derivatives_root / "mvpa-crossnobis").exists())

    def test_mvpa_init_dry_run_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            (project_root / "config" / "analysis").mkdir(parents=True)
            before = self._workspace_file_snapshot(workspace_root)

            code, output = self._run_cli(
                [
                    "analysis",
                    "mvpa",
                    "init",
                    "demo_crossnobis",
                    "--project",
                    "project-default",
                    "--analysis-label",
                    "DemoCrossnobis",
                    "--dry-run",
                ],
                workspace_root=workspace_root,
            )

            after = self._workspace_file_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 0, output)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["mode"], "dry-run")
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["analysis_id"], "demo_crossnobis")
        self.assertEqual(payload["comparison_mode"], "explicit")
        self.assertEqual(after, before)
        self.assertEqual(
            [record["relative_path"] for record in payload["planned_output_files"]],
            ["config/analysis/mvpa/demo_crossnobis.yaml"],
        )

    def test_mvpa_init_execute_creates_valid_scaffold_files_in_temp_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            (project_root / "config" / "analysis").mkdir(parents=True)

            code, output = self._run_cli(
                [
                    "analysis",
                    "mvpa",
                    "init",
                    "demo_crossnobis",
                    "--project",
                    "project-default",
                    "--analysis-label",
                    "DemoCrossnobis",
                ],
                workspace_root=workspace_root,
            )
            payload = json.loads(output)
            runtime_path = project_root / "config" / "analysis" / "mvpa" / "demo_crossnobis.yaml"
            generated_text = runtime_path.read_text(encoding="utf-8")
            runtime_exists = runtime_path.is_file()
            runtime_document = load_yaml(runtime_path)["mvpa_set"]

        self.assertEqual(code, 0, output)
        self.assertTrue(payload["executed"])
        self.assertTrue(runtime_exists)
        self.assertEqual(payload["mode"], "write")
        self.assertEqual(runtime_document["name"], "demo_crossnobis")
        self.assertEqual(runtime_document["unit_selection"]["mode"], "exact_units")
        self.assertEqual(runtime_document["runtime"], {"existing_output": "fail"})
        self.assertNotIn("subjects", runtime_document)
        self.assertNotIn("sessions", runtime_document)
        self.assertNotIn("runs", runtime_document)
        self.assertNotRegex(generated_text, r"(^|[=\s])/[^\s]+")
        self.assertNotIn("mask", generated_text.casefold())
        self.assertEqual(len(payload["written_files"]), 1)

    def test_mvpa_init_overwrite_refusal_and_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            (project_root / "config" / "analysis").mkdir(parents=True)
            args = [
                "analysis",
                "mvpa",
                "init",
                "demo_crossnobis",
                "--project",
                "project-default",
                "--analysis-label",
                "DemoCrossnobis",
            ]

            first_code, first_output = self._run_cli(args, workspace_root=workspace_root)
            second_code, second_output = self._run_cli(args, workspace_root=workspace_root)
            force_code, force_output = self._run_cli([*args, "--force"], workspace_root=workspace_root)

        self.assertEqual(first_code, 0, first_output)
        self.assertEqual(second_code, 1)
        self.assertIn("refuses to overwrite", json.loads(second_output)["errors"][0])
        self.assertEqual(force_code, 0, force_output)

    def test_mvpa_init_missing_analysis_config_root_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            config_root = project_root / "config" / "analysis"
            if config_root.exists():
                self.fail("test setup unexpectedly created config/analysis")

            code, output = self._run_cli(
                [
                    "analysis",
                    "mvpa",
                    "init",
                    "demo_crossnobis",
                    "--project",
                    "project-default",
                    "--analysis-label",
                    "DemoCrossnobis",
                ],
                workspace_root=workspace_root,
            )

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["valid"])
        self.assertTrue(any("analysis config root does not exist" in error for error in payload["errors"]))

    def test_mvpa_cli_surface_has_scaffold_and_runtime_authorization_flags(self) -> None:
        parser = _build_parser()
        analysis = self._subparser(parser, "analysis")
        mvpa = self._subparser(analysis, "mvpa")
        mvpa_subcommands = self._subparser_choices(mvpa)
        self.assertEqual(
            set(mvpa_subcommands),
            {
                "init",
                "list",
                "show",
                "validate",
                "doctor",
                "plan",
                "smoke-manual-crossnobis",
                "run",
                "export-tables",
                "export-figures",
                "export-rdms",
                "export-publication",
                "publish-derivatives",
                "publish",
            },
        )
        for forbidden in ("report",):
            self.assertNotIn(forbidden, mvpa_subcommands)
        for name, command in mvpa_subcommands.items():
            option_strings = {option for action in getattr(command, "_actions", []) for option in getattr(action, "option_strings", [])}
            if name in {"run", "export-tables", "export-figures", "export-rdms", "export-publication", "publish-derivatives"}:
                self.assertIn("--execute", option_strings)
            else:
                self.assertNotIn("--execute", option_strings)
        init_options = {
            option
            for action in getattr(mvpa_subcommands["init"], "_actions", [])
            for option in getattr(action, "option_strings", [])
        }
        self.assertIn("--dry-run", init_options)
        self.assertIn("--force", init_options)
        for name in ("doctor", "plan", "run"):
            option_strings = {
                option
                for action in getattr(mvpa_subcommands[name], "_actions", [])
                for option in getattr(action, "option_strings", [])
            }
            self.assertIn("--bundle", option_strings)
        initialized = parser.parse_args(
            ["analysis", "mvpa", "init", "prepared-demo", "--project", "project-default"]
        )
        self.assertEqual(initialized.template, "materialized-crossnobis")
        self.assertFalse(initialized.dry_run)
        bundled = parser.parse_args(
            [
                "analysis",
                "mvpa",
                "plan",
                "prepared-demo",
                "--project",
                "project-default",
                "--bundle",
                "exact-units",
            ]
        )
        self.assertEqual(bundled.bundle, "exact-units")
        parsed = parser.parse_args(["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"])
        self.assertTrue(parsed.execute)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "analysis",
                        "mvpa",
                        "run",
                        "memory_mvpa",
                        "--project",
                        "project-default",
                        "--overwrite",
                    ]
                )
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "analysis",
                        "mvpa",
                        "publish",
                        "memory_mvpa",
                        "--project",
                        "project-default",
                        "--execute",
                    ]
                )

    def test_mvpa_run_returns_plan_json_without_writing_files_or_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            before = self._workspace_tree_snapshot(workspace_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            after = self._workspace_tree_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertEqual(payload["mode"], "plan")
        self.assertEqual(payload["command"], "analysis mvpa run")
        self.assertFalse(payload["valid"])
        self.assertTrue(payload["schema_valid"])
        self.assertFalse(payload["plan_valid"])
        self.assertFalse(payload["ready_for_execution"])
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["project"], "project-default")
        self.assertEqual(payload["mvpa_set"], "memory_mvpa")
        self.assertEqual(payload["referenced_roi_sets"], ["memory_roi_set"])
        self.assertEqual(after, before)

    def test_mvpa_run_previews_runtime_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )

        payload = json.loads(output)
        runtime_root = Path(payload["runtime_root"]["path"])
        planned_outputs = payload["planned_outputs"]
        self.assertEqual(code, 1)
        self.assertEqual(payload["runtime_root"]["relative_path"], ".research-platform/mvpa/memory_mvpa")
        self.assertEqual(
            planned_outputs["neuro_patterns_tsv"]["relative_path"],
            "neuro/pattern-extraction/patterns.tsv",
        )
        self.assertEqual(
            planned_outputs["analysis_prepared_distance_rows_tsv"]["path"],
            str(runtime_root / "analysis/prepared-distances/distances.tsv"),
        )
        self.assertEqual(
            planned_outputs["analysis_prepared_summary_rows_tsv"]["path"],
            str(runtime_root / "analysis/prepared-summaries/summaries.tsv"),
        )
        self.assertTrue(all(not row["executed"] for row in planned_outputs.values()))
        self.assertIn("compute_distances", [step["name"] for step in payload["planned_steps"]])
        self.assertTrue(all(not step["executed"] for step in payload["planned_steps"]))

    def test_mvpa_publish_returns_plan_json_without_writing_files_or_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_phase_4b4_runtime_inputs(workspace_root)
            before = self._workspace_tree_snapshot(workspace_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "publish", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )

            after = self._workspace_tree_snapshot(workspace_root)

        payload = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(payload["mode"], "plan")
        self.assertEqual(payload["command"], "analysis mvpa publish")
        self.assertFalse(payload["executed"])
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["project"], "project-default")
        self.assertEqual(payload["mvpa_set"], "memory_mvpa")
        self.assertEqual(after, before)
        self.assertTrue(all(step["executed"] is False for step in payload["planned_steps"]))
        self.assertIn("publish_tables", {step["name"] for step in payload["planned_steps"]})

    def test_mvpa_publish_previews_publication_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            runtime_root = self._write_mvpa_phase_4b4_runtime_inputs(workspace_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "publish", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )

        payload = json.loads(output)
        publication_root = derivatives_root.resolve() / "mvpa-crossnobis" / "memory_mvpa"
        planned_outputs = payload["planned_outputs"]
        self.assertEqual(code, 0)
        self.assertEqual(payload["runtime_root"]["relative_path"], ".research-platform/mvpa/memory_mvpa")
        self.assertEqual(payload["publication_root"]["relative_path"], "mvpa-crossnobis/memory_mvpa")
        self.assertEqual(
            payload["required_runtime_inputs"]["analysis_prepared_summary_rows_tsv"]["path"],
            str(runtime_root.resolve() / "analysis/prepared-summaries/summaries.tsv"),
        )
        self.assertTrue(all(record["exists"] for record in payload["required_runtime_inputs"].values()))
        self.assertEqual(
            planned_outputs["group_summary_tsv"]["relative_path"],
            "tables/group/group_desc-Crossnobis_mvpasummary.tsv",
        )
        self.assertEqual(
            planned_outputs["group_summary_tsv"]["path"],
            str(publication_root / "tables/group/group_desc-Crossnobis_mvpasummary.tsv"),
        )
        self.assertEqual(
            planned_outputs["group_distances_tsv"]["relative_path"],
            "distances/group/desc-CrossnobisDistances_distances.tsv",
        )
        self.assertEqual(
            planned_outputs["manifest_json"]["path"],
            str(publication_root / "manifest.json"),
        )
        self.assertFalse(publication_root.exists())

    def test_mvpa_publish_requires_publication_enabled(self) -> None:
        for publication_enabled in (False, None):
            with self.subTest(publication_enabled=publication_enabled):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    workspace_root = Path(tmp_dir)
                    project_root, derivatives_root = self._write_workspace(workspace_root)
                    self._write_mvpa_configs(project_root, derivatives_root, publication_enabled=publication_enabled)
                    self._write_mvpa_phase_4b4_runtime_inputs(workspace_root)

                    code, output = self._run_cli(
                        ["analysis", "mvpa", "publish", "memory_mvpa", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )

                payload = json.loads(output)
                self.assertEqual(code, 1)
                self.assertFalse(payload["valid"])
                self.assertFalse(payload["executed"])
                self.assertEqual(payload["status"], "error")
                self.assertTrue(any("publication.enabled" in error for error in payload["errors"]))

    def test_mvpa_publish_missing_runtime_or_publication_roots_exit_nonzero(self) -> None:
        cases = (
            {"name": "runtime_root", "kwargs": {"missing_runtime_root": True}, "field": "runtime_root", "error": "runtime_root"},
            {"name": "publication_root", "kwargs": {"missing_publication_root": True}, "field": "publication_root", "error": "published_root"},
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    workspace_root = Path(tmp_dir)
                    project_root, derivatives_root = self._write_workspace(workspace_root)
                    self._write_mvpa_configs(project_root, derivatives_root, **case["kwargs"])
                    self._write_mvpa_phase_4b4_runtime_inputs(workspace_root)

                    code, output = self._run_cli(
                        ["analysis", "mvpa", "publish", "memory_mvpa", "--project", "project-default"],
                        workspace_root=workspace_root,
                    )

                payload = json.loads(output)
                self.assertEqual(code, 1)
                self.assertFalse(payload["valid"])
                self.assertIsNone(payload[case["field"]])
                self.assertTrue(any(case["error"] in error for error in payload["errors"]))

    def test_mvpa_publish_missing_required_runtime_inputs_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)

            code, output = self._run_cli(
                ["analysis", "mvpa", "publish", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["valid"])
        self.assertEqual(payload["status"], "error")
        self.assertTrue(all(record["status"] == "missing" for record in payload["required_runtime_inputs"].values()))
        self.assertTrue(any("Phase 4B.4 runtime input" in error for error in payload["errors"]))

    def test_mvpa_run_execute_writes_neuro_and_analysis_runtime_outputs_only(self) -> None:
        from research_platform.analysis.mvpa.runtime_outputs import (
            write_prepared_mvpa_distance_outputs,
            write_prepared_mvpa_pattern_outputs,
            write_prepared_mvpa_summary_outputs,
        )
        from research_platform.analysis.mvpa.prepared_summary import summarize_prepared_mvpa_distances
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            before_files = set(self._workspace_file_snapshot(workspace_root))
            calls: dict[str, object] = {}

            def validate(document: dict[str, object]) -> list[str]:
                calls["task"] = document["mvpa_set"]["entities"]["task"]  # type: ignore[index]
                return []

            def plan(
                document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                calls["enable_backend_discovery"] = enable_backend_discovery
                calls["roi_sets"] = sorted((roi_sets or {}).keys())
                calls["artifact_root"] = str((roots or {})["artifact_root"])
                calls["config_path"] = context["config_path"] if context else None
                return _FakeMvpaPlan(self._plan_payload())

            def extract(_plan: object, *, load_noise: bool = False) -> dict[str, object]:
                calls["load_noise"] = load_noise
                return self._fake_extraction_result()

            def prepare(rows: object, **kwargs: object) -> dict[str, object]:
                row_list = list(rows)  # type: ignore[arg-type]
                calls["prepared_input_rows"] = len(row_list)
                calls["prepare_cv_unit"] = kwargs.get("cv_unit")
                calls["prepare_group_by"] = kwargs.get("group_by")
                return self._fake_prepared_result()

            def compute(groups: object, **kwargs: object) -> dict[str, object]:
                calls["distance_group_count"] = len(list(groups))  # type: ignore[arg-type]
                calls["distance_metric"] = kwargs.get("metric")
                calls["distance_engine"] = kwargs.get("engine_name")
                calls["distance_noise"] = kwargs.get("noise_normalization_method")
                calls["distance_noise_nonpositive_policy"] = kwargs.get("noise_nonpositive_policy")
                return self._fake_distance_result()

            with mock.patch("research_platform.core.cli._mvpa_lifecycle_functions", return_value=(validate, plan)):
                with mock.patch(
                    "research_platform.core.cli._mvpa_pattern_extraction_runtime_functions",
                    return_value=(
                        extract,
                        mock.Mock(side_effect=AssertionError("prepared vectors must not use the image writer")),
                    ),
                ):
                    with mock.patch(
                        "research_platform.core.cli._mvpa_analysis_runtime_functions",
                        return_value=(
                            prepare,
                            compute,
                            summarize_prepared_mvpa_distances,
                            write_prepared_mvpa_pattern_outputs,
                            write_prepared_mvpa_distance_outputs,
                            write_prepared_mvpa_summary_outputs,
                        ),
                    ):
                        code, output = self._run_cli(
                            ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                            workspace_root=workspace_root,
                        )

            runtime_root = workspace_root / "artifacts" / ".research-platform" / "mvpa" / "memory_mvpa"
            runtime_files = sorted(path.relative_to(runtime_root).as_posix() for path in runtime_root.rglob("*") if path.is_file())
            after_files = set(self._workspace_file_snapshot(workspace_root))
            added_files = sorted(after_files - before_files)

        payload = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(payload["mode"], "execute")
        self.assertTrue(payload["executed"])
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["load_noise"])
        self.assertEqual(
            payload["planned_outputs"]["neuro_materialized_patterns_tsv"]["relative_path"],
            "neuro/pattern-materialization/patterns.tsv",
        )
        self.assertEqual(
            payload["planned_outputs"]["analysis_prepared_pattern_rows_tsv"]["relative_path"],
            "analysis/prepared-patterns/rows.tsv",
        )
        self.assertEqual(
            payload["planned_outputs"]["analysis_prepared_distance_rows_tsv"]["relative_path"],
            "analysis/prepared-distances/distances.tsv",
        )
        self.assertEqual(
            payload["planned_outputs"]["analysis_prepared_summary_rows_tsv"]["relative_path"],
            "analysis/prepared-summaries/summaries.tsv",
        )
        self.assertEqual(
            set(payload["planned_outputs"]),
            {
                "neuro_materialized_patterns_tsv",
                "neuro_materialized_pattern_qc_tsv",
                "neuro_materialized_pattern_provenance_json",
                "neuro_materialized_pattern_vector_metadata_json",
                "analysis_prepared_pattern_rows_tsv",
                "analysis_prepared_pattern_qc_tsv",
                "analysis_prepared_pattern_provenance_json",
                "analysis_prepared_distance_rows_tsv",
                "analysis_prepared_distance_qc_tsv",
                "analysis_prepared_distance_provenance_json",
                "analysis_prepared_summary_rows_tsv",
                "analysis_prepared_summary_qc_tsv",
                "analysis_prepared_summary_provenance_json",
                "successful_run_manifest_json",
            },
        )
        self.assertEqual(payload["extraction"]["pattern_rows"], 4)
        self.assertEqual(payload["extraction"]["usable_pattern_rows"], 4)
        self.assertEqual(payload["prepared"]["group_count"], 1)
        self.assertEqual(payload["prepared"]["prepared_row_count"], 2)
        self.assertEqual(payload["distances"]["distance_row_count"], 1)
        self.assertEqual(payload["distances"]["usable_distance_rows"], 1)
        self.assertEqual(payload["summaries"]["summary_row_count"], 1)
        self.assertEqual(payload["summaries"]["qc_row_count"], 0)
        self.assertEqual(payload["manifest"]["status"], "succeeded")
        self.assertEqual(payload["manifest"]["representation_kind"], "prepared_features")
        self.assertEqual(payload["manifest"]["errors"], [])
        self.assertEqual(
            sorted(artifact["relative_path"] for artifact in payload["outputs"]),
            [
                "analysis/prepared-distances/distances.tsv",
                "analysis/prepared-distances/provenance.json",
                "analysis/prepared-distances/qc.tsv",
                "analysis/prepared-patterns/provenance.json",
                "analysis/prepared-patterns/qc.tsv",
                "analysis/prepared-patterns/rows.tsv",
                "analysis/prepared-summaries/provenance.json",
                "analysis/prepared-summaries/qc.tsv",
                "analysis/prepared-summaries/summaries.tsv",
                "neuro/pattern-materialization/patterns.tsv",
                "neuro/pattern-materialization/provenance.json",
                "neuro/pattern-materialization/qc.tsv",
                "neuro/pattern-materialization/vector_metadata.json",
            ],
        )
        self.assertEqual(
            runtime_files,
            [
                "analysis/prepared-distances/distances.tsv",
                "analysis/prepared-distances/provenance.json",
                "analysis/prepared-distances/qc.tsv",
                "analysis/prepared-patterns/provenance.json",
                "analysis/prepared-patterns/qc.tsv",
                "analysis/prepared-patterns/rows.tsv",
                "analysis/prepared-summaries/provenance.json",
                "analysis/prepared-summaries/qc.tsv",
                "analysis/prepared-summaries/summaries.tsv",
                "manifest.json",
                "neuro/pattern-materialization/patterns.tsv",
                "neuro/pattern-materialization/provenance.json",
                "neuro/pattern-materialization/qc.tsv",
                "neuro/pattern-materialization/vector_metadata.json",
            ],
        )
        self.assertEqual(
            added_files,
            [f"artifacts/.research-platform/mvpa/memory_mvpa/{path}" for path in runtime_files],
        )
        self.assertFalse(any("report" in path or "figure" in path or "publication" in path for path in runtime_files))
        self.assertTrue(calls["enable_backend_discovery"])
        self.assertEqual(calls["roi_sets"], ["memory_roi_set"])
        self.assertEqual(calls["task"], "${RP_TASK:-memory}")
        self.assertEqual(calls["prepared_input_rows"], 4)
        self.assertEqual(calls["prepare_cv_unit"], "run")
        self.assertIsNone(calls["prepare_group_by"])
        self.assertEqual(calls["distance_group_count"], 1)
        self.assertEqual(calls["distance_metric"], "crossnobis")
        self.assertEqual(calls["distance_engine"], "native_reference")
        self.assertEqual(calls["distance_noise"], "identity")
        self.assertEqual(calls["distance_noise_nonpositive_policy"], "strict")

    def test_mvpa_run_execute_error_exits_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root, invalid=True)

            with mock.patch(
                "research_platform.core.cli._mvpa_pattern_extraction_runtime_functions",
                side_effect=AssertionError("invalid configs must not import runtime extraction"),
            ):
                invalid_code, invalid_output = self._run_cli(
                    ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                    workspace_root=workspace_root,
                )

        invalid_payload = json.loads(invalid_output)
        self.assertEqual(invalid_code, 1)
        self.assertEqual(invalid_payload["mode"], "execute")
        self.assertFalse(invalid_payload["executed"])
        self.assertFalse(invalid_payload["valid"])
        self.assertTrue(any("distance.metrics" in error for error in invalid_payload["errors"]))

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)

            def plan(
                _document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                return _FakeMvpaPlan(self._plan_payload(status="error"))

            with mock.patch("research_platform.core.cli._mvpa_lifecycle_functions", return_value=(lambda _document: [], plan)):
                with mock.patch(
                    "research_platform.core.cli._mvpa_pattern_extraction_runtime_functions",
                    side_effect=AssertionError("error discovery must not import runtime extraction"),
                ):
                    discovery_error_code, discovery_error_output = self._run_cli(
                        ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                        workspace_root=workspace_root,
                    )

        discovery_error_payload = json.loads(discovery_error_output)
        self.assertEqual(discovery_error_code, 1)
        self.assertFalse(discovery_error_payload["executed"])
        self.assertFalse(discovery_error_payload["plan_valid"])
        self.assertTrue(any("execution refused" in error for error in discovery_error_payload["errors"]))

    def test_mvpa_run_execute_refuses_overwrite(self) -> None:
        from research_platform.neuro.mvpa.runtime_outputs import write_mvpa_pattern_extraction_outputs

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            runtime_root = workspace_root / "artifacts" / ".research-platform" / "mvpa" / "memory_mvpa"
            existing_patterns = runtime_root / "neuro" / "pattern-materialization" / "patterns.tsv"
            existing_patterns.parent.mkdir(parents=True, exist_ok=True)
            existing_patterns.write_text("old patterns\n", encoding="utf-8")

            def plan(
                _document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                return _FakeMvpaPlan(self._plan_payload())

            with mock.patch("research_platform.core.cli._mvpa_lifecycle_functions", return_value=(lambda _document: [], plan)):
                with mock.patch(
                    "research_platform.core.cli._mvpa_pattern_extraction_runtime_functions",
                    return_value=(lambda _plan, *, load_noise=False: self._fake_extraction_result(), write_mvpa_pattern_extraction_outputs),
                ):
                    code, output = self._run_cli(
                        ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                        workspace_root=workspace_root,
                    )

            runtime_files = sorted(path.relative_to(runtime_root).as_posix() for path in runtime_root.rglob("*") if path.is_file())
            existing_text = existing_patterns.read_text(encoding="utf-8")

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["executed"])
        self.assertTrue(any("already exists" in error for error in payload["errors"]))
        self.assertEqual(existing_text, "old patterns\n")
        self.assertEqual(runtime_files, ["neuro/pattern-materialization/patterns.tsv"])

    def test_mvpa_run_execute_preflights_existing_summary_before_any_writes(self) -> None:
        from research_platform.analysis.mvpa.prepared_summary import summarize_prepared_mvpa_distances
        from research_platform.analysis.mvpa.runtime_outputs import (
            write_prepared_mvpa_distance_outputs,
            write_prepared_mvpa_pattern_outputs,
            write_prepared_mvpa_summary_outputs,
        )
        from research_platform.neuro.mvpa.runtime_outputs import write_mvpa_pattern_extraction_outputs

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            runtime_root = workspace_root / "artifacts" / ".research-platform" / "mvpa" / "memory_mvpa"
            existing_summary = runtime_root / "analysis" / "prepared-summaries" / "summaries.tsv"
            existing_summary.parent.mkdir(parents=True, exist_ok=True)
            existing_summary.write_text("old summaries\n", encoding="utf-8")

            def plan(
                _document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                return _FakeMvpaPlan(self._plan_payload())

            with mock.patch("research_platform.core.cli._mvpa_lifecycle_functions", return_value=(lambda _document: [], plan)):
                with mock.patch(
                    "research_platform.core.cli._mvpa_pattern_extraction_runtime_functions",
                    return_value=(lambda _plan, *, load_noise=False: self._fake_extraction_result(), write_mvpa_pattern_extraction_outputs),
                ):
                    with mock.patch(
                        "research_platform.core.cli._mvpa_analysis_runtime_functions",
                        return_value=(
                            lambda _rows, **_kwargs: self._fake_prepared_result(),
                            lambda _groups, **_kwargs: self._fake_distance_result(),
                            summarize_prepared_mvpa_distances,
                            write_prepared_mvpa_pattern_outputs,
                            write_prepared_mvpa_distance_outputs,
                            write_prepared_mvpa_summary_outputs,
                        ),
                    ):
                        code, output = self._run_cli(
                            ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                            workspace_root=workspace_root,
                        )

            runtime_files = sorted(path.relative_to(runtime_root).as_posix() for path in runtime_root.rglob("*") if path.is_file())
            existing_text = existing_summary.read_text(encoding="utf-8")

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["executed"])
        self.assertTrue(any("final MVPA runtime root already exists" in error for error in payload["errors"]))
        self.assertEqual(existing_text, "old summaries\n")
        self.assertEqual(runtime_files, ["analysis/prepared-summaries/summaries.tsv"])

    def test_mvpa_run_execute_loads_diagonal_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            calls: dict[str, object] = {}

            def plan(
                _document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                return _FakeMvpaPlan(self._plan_payload(noise_method="diagonal"))

            def extract(_plan: object, *, load_noise: bool = False) -> dict[str, object]:
                calls["load_noise"] = load_noise
                return self._fake_extraction_result()

            def write(_result: object, **kwargs: object) -> dict[str, object]:
                calls["overwrite"] = kwargs.get("overwrite")
                return {"artifact_kind": "mvpa_pattern_materialization_outputs", "artifacts": [], "overwrite": kwargs.get("overwrite")}

            def write_analysis(_result: object, **kwargs: object) -> dict[str, object]:
                return {"artifact_kind": "analysis", "artifacts": [], "overwrite": kwargs.get("overwrite")}

            with mock.patch("research_platform.core.cli._mvpa_lifecycle_functions", return_value=(lambda _document: [], plan)):
                with mock.patch("research_platform.core.cli._mvpa_pattern_extraction_runtime_functions", return_value=(extract, write)):
                    with mock.patch(
                        "research_platform.core.cli._mvpa_pattern_materialization_writer",
                        return_value=write,
                    ), mock.patch(
                        "research_platform.core.cli._mvpa_analysis_runtime_functions",
                        return_value=(
                            lambda _rows, **_kwargs: self._fake_prepared_result(),
                            lambda _groups, **_kwargs: self._fake_distance_result(),
                            lambda _distances, **_kwargs: self._fake_summary_result(),
                            write_analysis,
                            write_analysis,
                            write_analysis,
                        ),
                    ), mock.patch(
                        "research_platform.core.cli._mvpa_runtime_transaction_functions",
                        return_value=self._fake_runtime_transaction_functions(calls),
                    ):
                        code, output = self._run_cli(
                            ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                            workspace_root=workspace_root,
                        )

        payload = json.loads(output)
        self.assertEqual(code, 0)
        self.assertTrue(payload["load_noise"])
        self.assertTrue(calls["load_noise"])
        self.assertFalse(calls["overwrite"])
        self.assertTrue(calls["transaction_invoked"])

    def test_mvpa_runtime_refinement_helpers_are_config_driven(self) -> None:
        from research_platform.core import cli

        plan_payload = {
            "grouping_columns": ["participant_id", "task_id"],
            "condition_pairs": [
                {
                    "id": "pair-alpha",
                    "condition_id_a": "condition-a",
                    "condition_id_b": "condition-b",
                }
            ],
            "threshold_sweeps": [
                {"id": "threshold-alpha", "min_events": 2, "min_observations": 3}
            ],
        }
        document = {"mvpa_set": {"distance": {"cross_validation": {"unit": "run"}}}}

        self.assertEqual(
            cli._analysis_mvpa_condition_pairs(plan_payload),
            ({"id": "pair-alpha", "left": "condition-a", "right": "condition-b"},),
        )
        self.assertEqual(
            cli._analysis_mvpa_threshold_sweeps(plan_payload),
            ({"id": "threshold-alpha", "min_events": 2, "min_observations": 3},),
        )
        self.assertEqual(
            cli._analysis_mvpa_summary_group_by(document, plan_payload=plan_payload),
            (
                "group_id",
                "participant_id",
                "task_id",
                "condition_id_a",
                "condition_id_b",
                "condition_pair_id",
                "metric",
                "engine_name",
                "normalization_method",
            ),
        )

    def test_mvpa_run_execute_requires_usable_pattern_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)

            def plan(
                _document: dict[str, object],
                context: dict[str, object] | None = None,
                *,
                roots: dict[str, Path] | None = None,
                roi_sets: dict[str, dict[str, object]] | None = None,
                enable_backend_discovery: bool = False,
                exact_units: list[dict[str, object]] | None = None,
                unit_key_columns: list[str] | None = None,
            ) -> _FakeMvpaPlan:
                return _FakeMvpaPlan(self._plan_payload())

            with mock.patch("research_platform.core.cli._mvpa_lifecycle_functions", return_value=(lambda _document: [], plan)):
                with mock.patch(
                    "research_platform.core.cli._mvpa_pattern_extraction_runtime_functions",
                    return_value=(lambda _plan, *, load_noise=False: self._fake_extraction_result(usable=False), mock.Mock()),
                ):
                    code, output = self._run_cli(
                        ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                        workspace_root=workspace_root,
                    )

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["executed"])
        self.assertEqual(payload["extraction"]["usable_pattern_rows"], 0)
        self.assertTrue(any("no usable pattern rows" in error for error in payload["errors"]))

    def test_mvpa_run_execute_analysis_stage_failures_exit_nonzero(self) -> None:
        def run_case(
            *,
            prepare: object,
            compute: object,
            summarize: object,
            write_source: object,
            write_patterns: object,
            write_distances: object,
            write_summaries: object,
        ) -> tuple[int, dict[str, object], bool, list[str]]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                workspace_root = Path(tmp_dir)
                project_root, derivatives_root = self._write_workspace(workspace_root)
                self._write_mvpa_configs(project_root, derivatives_root)

                def plan(
                    _document: dict[str, object],
                    context: dict[str, object] | None = None,
                    *,
                    roots: dict[str, Path] | None = None,
                    roi_sets: dict[str, dict[str, object]] | None = None,
                    enable_backend_discovery: bool = False,
                    exact_units: list[dict[str, object]] | None = None,
                    unit_key_columns: list[str] | None = None,
                ) -> _FakeMvpaPlan:
                    return _FakeMvpaPlan(self._plan_payload())

                with mock.patch("research_platform.core.cli._mvpa_lifecycle_functions", return_value=(lambda _document: [], plan)):
                    with mock.patch(
                        "research_platform.core.cli._mvpa_pattern_extraction_runtime_functions",
                        return_value=(lambda _plan, *, load_noise=False: self._fake_extraction_result(), write_source),
                    ):
                        with mock.patch(
                            "research_platform.core.cli._mvpa_pattern_materialization_writer",
                            return_value=write_source,
                        ), mock.patch(
                            "research_platform.core.cli._mvpa_analysis_runtime_functions",
                            return_value=(prepare, compute, summarize, write_patterns, write_distances, write_summaries),
                        ):
                            code, output = self._run_cli(
                                ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                                workspace_root=workspace_root,
                            )
                runtime_parent = workspace_root / "artifacts" / ".research-platform" / "mvpa"
                final_exists = (runtime_parent / "memory_mvpa").exists()
                remnants = sorted(path.name for path in runtime_parent.glob(".memory_mvpa.*.tmp"))
            return code, json.loads(output), final_exists, remnants

        def write_analysis(_result: object, **kwargs: object) -> dict[str, object]:
            return {"artifact_kind": "analysis", "artifacts": [], "overwrite": kwargs.get("overwrite")}

        def write_source(_result: object, **kwargs: object) -> dict[str, object]:
            return {
                "artifact_kind": "mvpa_pattern_materialization_outputs",
                "artifacts": [],
                "overwrite": kwargs.get("overwrite"),
            }

        summarize_analysis = lambda _distances, **_kwargs: self._fake_summary_result()
        cases = [
            (
                "zero prepared",
                lambda _rows, **_kwargs: {"groups": [], "qc_rows": [], "provenance": {}, "warnings": [], "errors": [], "executed": True},
                lambda _groups, **_kwargs: self._fake_distance_result(),
                summarize_analysis,
                write_source,
                write_analysis,
                write_analysis,
                write_analysis,
                "no usable prepared groups or rows",
            ),
            (
                "distance fatal",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                mock.Mock(side_effect=RuntimeError("distance boom")),
                summarize_analysis,
                write_source,
                write_analysis,
                write_analysis,
                write_analysis,
                "MVPA distance computation failed",
            ),
            (
                "zero distances",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                lambda _groups, **_kwargs: {"distances": [], "qc_rows": [], "provenance": {}, "warnings": [], "errors": [], "executed": True},
                summarize_analysis,
                write_source,
                write_analysis,
                write_analysis,
                write_analysis,
                "zero usable distance rows",
            ),
            (
                "source writer",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                lambda _groups, **_kwargs: self._fake_distance_result(),
                summarize_analysis,
                mock.Mock(side_effect=OSError("source writer boom")),
                write_analysis,
                write_analysis,
                write_analysis,
                "MVPA runtime transaction failed",
            ),
            (
                "analysis writer",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                lambda _groups, **_kwargs: self._fake_distance_result(),
                summarize_analysis,
                write_source,
                mock.Mock(side_effect=FileExistsError("analysis/prepared-patterns/rows.tsv already exists")),
                write_analysis,
                write_analysis,
                "MVPA runtime transaction failed",
            ),
            (
                "distance writer",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                lambda _groups, **_kwargs: self._fake_distance_result(),
                summarize_analysis,
                write_source,
                write_analysis,
                mock.Mock(side_effect=OSError("distance writer boom")),
                write_analysis,
                "MVPA runtime transaction failed",
            ),
            (
                "summary fatal",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                lambda _groups, **_kwargs: self._fake_distance_result(),
                mock.Mock(side_effect=RuntimeError("summary boom")),
                write_source,
                write_analysis,
                write_analysis,
                write_analysis,
                "MVPA distance summary failed",
            ),
            (
                "summary result errors",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                lambda _groups, **_kwargs: self._fake_distance_result(),
                lambda _distances, **_kwargs: self._fake_summary_result(errors=["bad summary"]),
                write_source,
                write_analysis,
                write_analysis,
                write_analysis,
                "bad summary",
            ),
            (
                "zero summaries",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                lambda _groups, **_kwargs: self._fake_distance_result(),
                lambda _distances, **_kwargs: self._fake_summary_result(rows=False),
                write_source,
                write_analysis,
                write_analysis,
                write_analysis,
                "zero summary rows",
            ),
            (
                "summary writer",
                lambda _rows, **_kwargs: self._fake_prepared_result(),
                lambda _groups, **_kwargs: self._fake_distance_result(),
                summarize_analysis,
                write_source,
                write_analysis,
                write_analysis,
                mock.Mock(side_effect=FileExistsError("analysis/prepared-summaries/summaries.tsv already exists")),
                "MVPA runtime transaction failed",
            ),
        ]
        for name, prepare, compute, summarize, write_source_fn, write_patterns, write_distances, write_summaries, expected_error in cases:
            with self.subTest(name=name):
                code, payload, final_exists, remnants = run_case(
                    prepare=prepare,
                    compute=compute,
                    summarize=summarize,
                    write_source=write_source_fn,
                    write_patterns=write_patterns,
                    write_distances=write_distances,
                    write_summaries=write_summaries,
                )
                self.assertEqual(code, 1)
                self.assertFalse(payload["executed"])
                self.assertTrue(any(expected_error in error for error in payload["errors"]))
                self.assertFalse(final_exists)
                self.assertEqual(remnants, [])

    def test_mvpa_run_missing_runtime_root_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root, missing_runtime_root=True)

            code, output = self._run_cli(
                ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )
            execute_code, execute_output = self._run_cli(
                ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"],
                workspace_root=workspace_root,
            )

        payload = json.loads(output)
        execute_payload = json.loads(execute_output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["valid"])
        self.assertFalse(payload["plan_valid"])
        self.assertIsNone(payload["runtime_root"])
        self.assertTrue(any("runtime_root" in error for error in payload["errors"]))
        self.assertEqual(execute_code, 1)
        self.assertFalse(execute_payload["executed"])
        self.assertIsNone(execute_payload["runtime_root"])
        self.assertTrue(any("runtime_root" in error for error in execute_payload["errors"]))

    def test_mvpa_run_invalid_config_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root, invalid=True)

            code, output = self._run_cli(
                ["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default"],
                workspace_root=workspace_root,
            )

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["valid"])
        self.assertFalse(payload["executed"])
        self.assertTrue(any("distance.metrics" in error for error in payload["errors"]))

    def test_mvpa_plan_does_not_execute_nifti_fsl_or_distance_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            before_modules = set(sys.modules)

            with mock.patch("subprocess.run", side_effect=AssertionError("MVPA plan must not execute subprocesses")):
                code, output = self._run_cli(
                    ["analysis", "mvpa", "plan", "memory_mvpa", "--project", "project-default"],
                    workspace_root=workspace_root,
                )

            new_modules = set(sys.modules) - before_modules

        payload = json.loads(output)
        self.assertEqual(code, 1)
        self.assertFalse(payload["executed"])
        self.assertNotIn("research_platform.neuro.mvpa.extraction", new_modules)
        self.assertNotIn("research_platform.neuro.mvpa.runtime_outputs", new_modules)
        self.assertFalse(any(name.startswith("research_platform.analysis.mvpa") for name in new_modules))
        self.assertFalse({"nibabel", "nilearn", "rsatoolbox"} & new_modules)
        self.assertTrue(all(row["status"] == "planned" for row in payload["distances"]))

    def test_mvpa_run_does_not_import_runtime_distance_or_heavy_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            script = textwrap.dedent(
                f"""
                import contextlib
                import io
                import json
                import os
                import sys
                sys.path.insert(0, {str(CORE_PACKAGE_ROOT / "src")!r})
                sys.path.insert(0, {str(HPC_PACKAGE_ROOT / "src")!r})
                sys.path.insert(0, {str(NEURO_PACKAGE_ROOT / "src")!r})
                from research_platform.core.cli import main
                os.environ["RESEARCH_PLATFORM_ROOT"] = {str(workspace_root)!r}
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main(["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default"])
                forbidden_exact = {{
                    "nibabel",
                    "numpy",
                    "nilearn",
                    "rsatoolbox",
                    "pandas",
                    "polars",
                    "scipy",
                    "sklearn",
                }}
                forbidden_prefixes = (
                    "research_platform.neuro.mvpa.extraction",
                    "research_platform.neuro.mvpa.runtime_outputs",
                    "research_platform.analysis.mvpa",
                    "research_platform.bids",
                    "research_platform.viz",
                    "research_platform.io",
                    "research_platform.pipelines",
                    "research_platform.ops",
                    "pipelines",
                    "ops",
                )
                forbidden = sorted(
                    name
                    for name in sys.modules
                    if name in forbidden_exact or any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
                )
                print(json.dumps({{"code": code, "payload": json.loads(stdout.getvalue()), "forbidden": forbidden}}, sort_keys=True))
                """
            )

            result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

        payload = json.loads(result.stdout)
        self.assertEqual(payload["code"], 1)
        self.assertFalse(payload["payload"]["executed"])
        self.assertEqual(payload["forbidden"], [])

    def test_mvpa_publish_does_not_import_runtime_publication_or_heavy_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            self._write_mvpa_phase_4b4_runtime_inputs(workspace_root)
            script = textwrap.dedent(
                f"""
                import contextlib
                import io
                import json
                import os
                import sys
                sys.path.insert(0, {str(CORE_PACKAGE_ROOT / "src")!r})
                sys.path.insert(0, {str(HPC_PACKAGE_ROOT / "src")!r})
                sys.path.insert(0, {str(NEURO_PACKAGE_ROOT / "src")!r})
                from research_platform.core.cli import main
                os.environ["RESEARCH_PLATFORM_ROOT"] = {str(workspace_root)!r}
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main(["analysis", "mvpa", "publish", "memory_mvpa", "--project", "project-default"])
                forbidden_exact = {{
                    "fsl.wrappers",
                    "matplotlib",
                    "nibabel",
                    "nilearn",
                    "numpy",
                    "pandas",
                    "plotly",
                    "polars",
                    "rsatoolbox",
                    "scipy",
                    "seaborn",
                    "sklearn",
                }}
                forbidden_prefixes = (
                    "fsl.wrappers",
                    "research_platform.neuro.mvpa.extraction",
                    "research_platform.neuro.mvpa.runtime_outputs",
                    "research_platform.analysis.mvpa",
                    "research_platform.bids",
                    "research_platform.viz",
                    "research_platform.io",
                    "research_platform.pipelines",
                    "research_platform.ops",
                    "pipelines",
                    "ops",
                )
                forbidden = sorted(
                    name
                    for name in sys.modules
                    if name in forbidden_exact or any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
                )
                print(json.dumps({{"code": code, "payload": json.loads(stdout.getvalue()), "forbidden": forbidden}}, sort_keys=True))
                """
            )

            result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

        payload = json.loads(result.stdout)
        self.assertEqual(payload["code"], 0)
        self.assertFalse(payload["payload"]["executed"])
        self.assertEqual(payload["forbidden"], [])

    def test_mvpa_run_execute_uses_lazy_runtime_helpers_without_forbidden_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root, derivatives_root = self._write_workspace(workspace_root)
            self._write_mvpa_configs(project_root, derivatives_root)
            script = textwrap.dedent(
                f"""
                import contextlib
                import io
                import json
                import os
                from pathlib import Path
                import sys
                import tempfile
                sys.path.insert(0, {str(CORE_PACKAGE_ROOT / "src")!r})
                sys.path.insert(0, {str(HPC_PACKAGE_ROOT / "src")!r})
                sys.path.insert(0, {str(NEURO_PACKAGE_ROOT / "src")!r})
                from research_platform.core import cli

                class Plan:
                    def to_dict(self):
                        return {{
                            "status": "valid",
                            "valid": True,
                            "schema_valid": True,
                            "plan_valid": True,
                            "ready_for_materialization": True,
                            "ready_for_execution": True,
                            "errors": [],
                            "warnings": [],
                            "analysis_units": [{{"unit_id": "unit-1", "subject_id": "sub-001"}}],
                            "pattern_rows": [{{
                                "unit_id": "unit-1",
                                "subject_id": "sub-001",
                                "condition_id": "faces",
                                "source_name": "prepared_patterns",
                                "backend_name": "materialized_pattern_table",
                                "representation_kind": "prepared_features",
                                "cross_validation_label": "01",
                                "backend_metadata": {{
                                    "roi_source_name": "example_rois",
                                    "roi_label": "SeedA",
                                    "feature_count": 1,
                                    "voxel_index_hash": "synthetic-index-hash",
                                    "feature_space_id": "synthetic-feature-space",
                                    "roi_definition_id": "synthetic-roi-definition",
                                    "mean_centering_applied": False,
                                    "mean_centering_scope": "none",
                                    "noise_status": "unused",
                                    "noise_usable": False,
                                }},
                            }}],
                            "pattern_source_summaries": [{{
                                "source_name": "prepared_patterns",
                                "backend": "materialized_pattern_table",
                                "root_ref": "pattern_root",
                                "portable_reference": "root_ref:pattern_root/patterns.tsv",
                                "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                "counts": {{
                                    "selected_rows": 1,
                                    "usable_selected_rows": 1,
                                }},
                                "usable_coverage_complete": True,
                            }}],
                            "adapter_availability": [{{
                                "backend": "materialized_pattern_table",
                                "registered": True,
                                "available": True,
                                "execution_available": True,
                                "ready_for_execution": True,
                            }}],
                            "event_threshold_rows": [],
                            "mean_centering": {{"enabled": False, "scope": "none"}},
                            "distances": [{{
                                "metric": "crossnobis",
                                "engine": "native_reference",
                                "cv_unit": "run",
                                "noise_normalization_method": "identity",
                            }}],
                            "backend_summary": {{
                                "integration_attempted": True,
                                "ready_for_materialization": True,
                                "ready_for_execution": True,
                            }},
                        }}

                def validate(_document):
                    return []

                def plan(
                    _document,
                    context=None,
                    *,
                    roots=None,
                    roi_sets=None,
                    enable_backend_discovery=False,
                    exact_units=None,
                    unit_key_columns=None,
                ):
                    assert enable_backend_discovery is True
                    assert "memory_roi_set" in roi_sets
                    return Plan()

                def extract(_plan, *, load_noise=False):
                    assert load_noise is False
                    return {{
                        "representation_kind": "prepared_features",
                        "source_audit_kind": "pattern_materialization",
                        "pattern_rows": [{{
                            "pattern_id": "pat-1",
                            "usable": True,
                            "feature_values": [1.0],
                            "cross_validation_label": "01",
                            "roi_source_name": "example_rois",
                            "roi_label": "SeedA",
                            "feature_count": 1,
                            "voxel_order": "c_flat_index",
                            "voxel_index_hash": "synthetic-index-hash",
                            "feature_space_id": "synthetic-feature-space",
                            "roi_definition_id": "synthetic-roi-definition",
                            "mean_centering_applied": False,
                            "mean_centering_scope": "none",
                            "noise_status": "unused",
                            "noise_usable": False,
                        }}],
                        "qc_rows": [{{"status": "ok", "usable": True, "warnings": [], "errors": []}}],
                        "provenance": {{
                            "sources": [{{
                                "source_name": "prepared_patterns",
                                "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                            }}],
                        }},
                        "warnings": [],
                        "errors": [],
                        "executed": True,
                    }}

                def write(_result, **kwargs):
                    assert kwargs["patterns_path"] == "neuro/pattern-materialization/patterns.tsv"
                    assert kwargs["qc_path"] == "neuro/pattern-materialization/qc.tsv"
                    assert kwargs["provenance_path"] == "neuro/pattern-materialization/provenance.json"
                    assert kwargs["vector_metadata_path"] == "neuro/pattern-materialization/vector_metadata.json"
                    assert kwargs["overwrite"] is False
                    return {{"artifact_kind": "mvpa_pattern_materialization_outputs", "artifacts": [], "overwrite": False}}

                calls = []

                def prepare(rows, **kwargs):
                    calls.append(["prepare", len(list(rows)), kwargs["cv_unit"], kwargs["group_by"]])
                    return {{
                        "groups": [{{"group_id": "g1", "group_key": {{}}, "rows": [{{"pattern_id": "pat-1"}}]}}],
                        "qc_rows": [],
                        "provenance": {{}},
                        "warnings": [],
                        "errors": [],
                        "executed": True,
                    }}

                def compute(groups, **kwargs):
                    calls.append(["compute", len(list(groups)), kwargs["metric"], kwargs["engine_name"], kwargs["noise_normalization_method"]])
                    return {{
                        "distances": [{{
                            "group_id": "g1",
                            "group_key": {{}},
                            "condition_id_a": "faces",
                            "condition_id_b": "places",
                            "distance": 1.0,
                            "metric": "crossnobis",
                            "engine_name": "native_reference",
                            "normalization_method": "identity",
                        }}],
                        "qc_rows": [],
                        "provenance": {{}},
                        "warnings": [],
                        "errors": [],
                        "executed": True,
                    }}

                def summarize(distances, **kwargs):
                    calls.append(["summarize", len(list(distances["distances"])), kwargs.get("group_by")])
                    return {{
                        "summary_rows": [{{
                            "group_id": "g1",
                            "condition_id_a": "faces",
                            "condition_id_b": "places",
                            "metric": "crossnobis",
                            "engine_name": "native_reference",
                            "normalization_method": "identity",
                            "n": 1,
                            "mean_distance": 1.0,
                            "std_distance": 0.0,
                            "sem_distance": 0.0,
                            "min_distance": 1.0,
                            "max_distance": 1.0,
                        }}],
                        "qc_rows": [],
                        "provenance": {{}},
                        "warnings": [],
                        "errors": [],
                        "executed": True,
                    }}

                def write_analysis(_result, **kwargs):
                    calls.append(["write_analysis", kwargs["overwrite"]])
                    return {{"artifact_kind": "analysis", "artifacts": [], "overwrite": kwargs["overwrite"]}}

                real_plan_transaction, _, runtime_specs = cli._mvpa_runtime_transaction_functions()

                class TransactionResult:
                    warnings = ()
                    manifest = {{"status": "succeeded", "outputs": []}}
                    output_sha256 = {{}}
                    recovery_path = None

                def execute_transaction(_plan, *, write_outputs, manifest_payload):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        write_outputs(Path(tmp_dir))
                    calls.append(["transaction", manifest_payload["mvpa_set"]])
                    return TransactionResult()

                cli._mvpa_lifecycle_functions = lambda: (validate, plan)
                cli._mvpa_pattern_extraction_runtime_functions = lambda: (extract, write)
                cli._mvpa_pattern_materialization_writer = lambda: write
                cli._mvpa_analysis_runtime_functions = lambda: (prepare, compute, summarize, write_analysis, write_analysis, write_analysis)
                cli._mvpa_runtime_transaction_functions = lambda: (real_plan_transaction, execute_transaction, runtime_specs)
                os.environ["RESEARCH_PLATFORM_ROOT"] = {str(workspace_root)!r}
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(["analysis", "mvpa", "run", "memory_mvpa", "--project", "project-default", "--execute"])
                forbidden_exact = {{
                    "rsatoolbox",
                    "pandas",
                    "polars",
                    "scipy",
                    "sklearn",
                }}
                forbidden_prefixes = (
                    "research_platform.bids",
                    "research_platform.viz",
                    "research_platform.io",
                    "research_platform.analysis.cli",
                    "research_platform.pipelines",
                    "research_platform.ops",
                    "pipelines",
                    "ops",
                )
                forbidden = sorted(
                    name
                    for name in sys.modules
                    if name in forbidden_exact or any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
                )
                print(json.dumps({{"calls": calls, "code": code, "payload": json.loads(stdout.getvalue()), "forbidden": forbidden}}, sort_keys=True))
                """
            )

            result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

        payload = json.loads(result.stdout)
        self.assertEqual(payload["code"], 0)
        self.assertTrue(payload["payload"]["executed"])
        self.assertEqual(payload["forbidden"], [])
        self.assertEqual(
            [call[0] for call in payload["calls"]],
            ["prepare", "compute", "summarize", "write_analysis", "write_analysis", "write_analysis", "transaction"],
        )

    def test_importing_core_cli_does_not_import_mvpa_or_heavy_scientific_modules(self) -> None:
        script = textwrap.dedent(
            f"""
            import json
            import sys
            sys.path.insert(0, {str(CORE_PACKAGE_ROOT / "src")!r})
            sys.path.insert(0, {str(HPC_PACKAGE_ROOT / "src")!r})
            import research_platform.core.cli  # noqa: F401
            forbidden = [
                "research_platform.neuro.mvpa",
                "research_platform.neuro.mvpa.extraction",
                "research_platform.analysis.mvpa",
                "research_platform.bids",
                "research_platform.viz",
                "research_platform.io",
                "numpy",
                "nibabel",
                "nilearn",
                "rsatoolbox",
                "pandas",
                "polars",
                "scipy",
                "sklearn",
            ]
            print(json.dumps({{name: name in sys.modules for name in forbidden}}, sort_keys=True))
            """
        )

        result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)

        self.assertFalse(any(payload.values()), payload)


if __name__ == "__main__":
    unittest.main()
