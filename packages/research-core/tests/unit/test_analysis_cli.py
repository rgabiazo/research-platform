from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))

from research_platform.core.cli import main
from research_platform.core.config import (
    load_project_bundle,
    load_yaml,
    resolve_analysis_external_input_root_declarations,
    validate_project_bundle,
    write_yaml,
)
from research_platform.core.paths import run_path, workspace_paths


class AnalysisCliTests(unittest.TestCase):
    def _run_cli_for_workspace(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        artifact_root: Path,
        extra_env: dict[str, str | None] | None = None,
    ) -> tuple[int, str]:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(workspace_root),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_HPC_HOST": "example-hpc",
            "RP_REMOTE_WORKSPACE_ROOT": "/remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "/remote/workspace/artifacts",
        }
        unset_keys: list[str] = []
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                    unset_keys.append(key)
                    continue
                env[key] = value
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False):
            for key in unset_keys:
                os.environ.pop(key, None)
            with redirect_stdout(buffer):
                exit_code = main(args)
        return exit_code, buffer.getvalue()

    def _run_cli_for_workspace_system_exit(
        self,
        args: list[str],
        *,
        workspace_root: Path,
        artifact_root: Path,
        extra_env: dict[str, str | None] | None = None,
    ) -> str:
        env = {
            "RESEARCH_PLATFORM_ROOT": str(workspace_root),
            "ARTIFACTS_ROOT": str(artifact_root),
            "RP_HPC_HOST": "example-hpc",
            "RP_REMOTE_WORKSPACE_ROOT": "/remote/workspace",
            "RP_REMOTE_ARTIFACTS_ROOT": "/remote/workspace/artifacts",
        }
        unset_keys: list[str] = []
        if extra_env:
            for key, value in extra_env.items():
                if value is None:
                    env.pop(key, None)
                    unset_keys.append(key)
                    continue
                env[key] = value
        with mock.patch.dict(os.environ, env, clear=False):
            for key in unset_keys:
                os.environ.pop(key, None)
            with self.assertRaises(SystemExit) as exc_info:
                main(args)
        return str(exc_info.exception)

    def _write_analysis_workspace(self, workspace_root: Path, *, workspace_hpc: dict[str, object] | None = None) -> None:
        pipeline_root = workspace_root / "pipelines" / "analysis-bids"
        for path in (
            pipeline_root / "config",
            pipeline_root / "profiles" / "local",
            pipeline_root / "profiles" / "slurm",
            workspace_root / "ops" / "sync" / "rsync",
            workspace_root / "ops" / "slurm" / "job_templates",
        ):
            path.mkdir(parents=True, exist_ok=True)

        workspace_config: dict[str, object] = {
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
        }
        if workspace_hpc is not None:
            workspace_config["hpc"] = workspace_hpc
        write_yaml(workspace_root / "WORKSPACE.yaml", workspace_config)
        write_yaml(
            pipeline_root / "config" / "defaults.yaml",
            {
                "workflow": {
                    "default_target": "bids_analysis",
                    "rule_name": "bids_analysis",
                    "execution_rule_name": "bids_analysis_unit",
                },
                "planner": {
                    "outputs": {
                        "runtime_plan_filename": "fsl-feat-plan.json",
                        "command_script_filename": "run-fsl-feat.sh",
                        "completion_marker_filename": "fsl-feat-complete.txt",
                        "output_data_dirname": "fsl_feat",
                    }
                },
            },
        )
        write_yaml(pipeline_root / "profiles" / "local" / "config.yaml", {"executor": "local", "jobs": 1})
        write_yaml(pipeline_root / "profiles" / "slurm" / "config.yaml", {"executor": "slurm", "jobs": 10})
        (workspace_root / "ops" / "slurm" / "job_templates" / "sbatch.job.sh").write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "#SBATCH --job-name={{ job_name }}",
                    "#SBATCH --cpus-per-task={{ cpus }}",
                    "#SBATCH --mem={{ mem }}",
                    "#SBATCH --time={{ time }}",
                    "#SBATCH --output={{ log_out }}",
                    "#SBATCH --error={{ log_err }}",
                    "{{ command }}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _configure_external_analysis_inputs(
        self,
        *,
        project_root: Path,
        ev_env_name: str,
        remote_ev_env_name: str,
    ) -> None:
        analysis_path = project_root / "config" / "analysis.yaml"
        document = load_yaml(analysis_path)
        analysis = document["analysis"]
        analysis["external_input_roots"] = {
            "evs": {
                "local_root": f"${{{ev_env_name}:-}}",
                "remote_root": f"${{{remote_ev_env_name}:-}}",
                "sync_enabled": True,
            },
            "feat_confounds": {
                "local_root": f"${{{ev_env_name}:-}}",
                "remote_root": f"${{{remote_ev_env_name}:-}}",
                "sync_enabled": True,
            },
        }
        analysis["inputs"]["confounds"] = {
            "required": True,
            "root_ref": "feat_confounds",
            "patterns": [
                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-confounds_noGSR.txt",
                "{input_root}/{subject_dir}/func/{bids_base}_desc-confounds_noGSR.txt",
            ],
        }
        analysis["inputs"]["evs"] = {
            "required": True,
            "root_ref": "evs",
            "patterns": [
                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
                "{input_root}/{subject_dir}/func/{bids_base}_desc-{ev_name}_events.txt",
            ],
        }
        write_yaml(analysis_path, document)

    def test_project_init_bids_analysis_scaffolds_valid_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            exit_code, output = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-new-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                    "--task-id",
                    "rest",
                    "--remote-study-root",
                    "/remote/studies/demo",
                    "--remote-derivative-root",
                    "/remote/studies/demo/derivatives/fmriprep",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            bundle = load_project_bundle("project-new-analysis", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)
            compute_path = workspace_root / "project" / "project-new-analysis" / "config" / "compute.yaml"
            compute_text = compute_path.read_text(encoding="utf-8")
            compute_bundle = bundle["compute"]["compute"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])
        self.assertIn("Initialized BIDS analysis project: project-new-analysis", output)
        self.assertIn("project/project-new-analysis/config/analysis.yaml", output)
        self.assertIn(
            "project/project-new-analysis/config/analysis/models/task_glm.yaml",
            output,
        )
        self.assertEqual(bundle["dataset"]["dataset"]["remote_bids_root"], "/remote/studies/demo")
        self.assertEqual(
            bundle["dataset"]["dataset"]["remote_input_derivative_root"],
            "/remote/studies/demo/derivatives/fmriprep",
        )
        self.assertEqual(bundle["analysis"]["analysis"]["defaults"]["tool"], "feat")
        self.assertEqual(bundle["analysis"]["analysis"]["stages"]["first_level"]["default_batch"], "feat_first_level")
        self.assertEqual(bundle["analysis"]["analysis"]["stages"]["first_level"]["settings"]["norm"], 0)
        self.assertEqual(
            bundle["analysis"]["analysis"]["stages"]["first_level"]["validation"]["empty_ev_policy"],
            "as_zero",
        )
        self.assertEqual(
            compute_bundle["tool_profiles"]["fsl"]["local"]["environment"]["FSLOUTPUTTYPE"],
            "NIFTI_GZ",
        )
        self.assertEqual(compute_bundle["tool_profiles"]["fsl"]["local"]["execution_backend"], "native")
        self.assertEqual(compute_bundle["tool_profiles"]["fsl"]["slurm"]["execution_backend"], "apptainer")
        self.assertTrue(compute_bundle["tool_profiles"]["fsl"]["slurm"]["container"]["enabled"])
        self.assertNotIn("modules", compute_bundle["tool_profiles"]["fsl"]["slurm"])
        self.assertNotIn("pre_activate_commands", compute_bundle["tool_profiles"]["fsl"]["slurm"])
        self.assertEqual(
            compute_bundle["slurm"]["pre_activate_commands"],
            ['[ -n "$SCRATCH" ] || { echo "ERROR: SCRATCH is not set on the remote node." >&2; exit 1; }'],
        )
        self.assertEqual(
            compute_bundle["slurm"]["prepare_directories"],
            ["$APPTAINER_CACHEDIR", "$APPTAINER_TMPDIR", "$TMPDIR"],
        )
        self.assertEqual(compute_bundle["slurm"]["environment"]["APPTAINER_CACHEDIR"], "$SCRATCH/apptainer-cache")
        self.assertIn("${RP_FSL_CONTAINER_IMAGE:-docker://vnmd/fsl_6.0.7.4:latest}", compute_text)
        self.assertIn("${RP_FSL_CONTAINER_IMAGE_NAME:-fsl_6.0.7.4.sif}", compute_text)
        self.assertIn("${RP_REMOTE_CONTAINER_ROOT:-$SCRATCH/containers/fsl}", compute_text)
        self.assertNotIn(str(study_root), compute_text)
        self.assertNotIn("/remote/studies/demo", compute_text)
        self.assertNotIn("StdEnv/2023", compute_text)

    def test_project_init_bids_analysis_auto_template_generates_fmripost_aroma_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmripost_aroma"
            events_root = workspace_root / "external" / "events"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            events_root.mkdir(parents=True, exist_ok=True)

            exit_code, output = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-aroma-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                    "--events-root",
                    str(events_root),
                    "--remote-events-root",
                    "/remote/study/derivatives/evs",
                    "--remote-study-root",
                    "/remote/study",
                    "--remote-derivative-root",
                    "/remote/study/derivatives/fmripost_aroma",
                    "--task-id",
                    "exampletask",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            bundle = load_project_bundle("project-aroma-analysis", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)
            analysis = bundle["analysis"]["analysis"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])
        self.assertIn("Template: fmripost-aroma-first-level", output)
        self.assertEqual(analysis["scaffold"]["template"], "fmripost-aroma-first-level")
        self.assertEqual(analysis["external_input_roots"]["evs"]["local_root"], str(events_root.resolve()))
        self.assertEqual(analysis["external_input_roots"]["evs"]["remote_root"], "/remote/study/derivatives/evs")
        self.assertEqual(analysis["external_input_roots"]["feat_confounds"]["local_root"], str(events_root.resolve()))
        self.assertEqual(analysis["external_input_roots"]["feat_confounds"]["remote_root"], "/remote/study/derivatives/evs")
        self.assertEqual(analysis["inputs"]["confounds"]["root_ref"], "feat_confounds")
        self.assertTrue(analysis["inputs"]["confounds"]["required"])
        self.assertIn("desc-nonaggrDenoised_bold.nii.gz", "\n".join(analysis["inputs"]["bold"]["patterns"]))
        self.assertIn("desc-confounds_noGSR.txt", "\n".join(analysis["inputs"]["confounds"]["patterns"]))
        self.assertEqual(analysis["stages"]["first_level"]["task_id"], "exampletask")
        self.assertEqual(analysis["stages"]["first_level"]["settings"]["norm"], 1)

    def test_project_init_bids_analysis_explicit_template_requires_events_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            error = self._run_cli_for_workspace_system_exit(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-aroma-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                    "--template",
                    "fmripost-aroma-first-level",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertIn("--events-root is required", error)

    def test_project_init_bids_analysis_deepprep_t1w_template_generates_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = study_root / "derivatives" / "DeepPrep" / "BOLD"
            events_root = workspace_root / "external" / "events"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            events_root.mkdir(parents=True, exist_ok=True)

            exit_code, output = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-deepprep-t1w-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                    "--template",
                    "deepprep-t1w-first-level",
                    "--events-root",
                    str(events_root),
                    "--remote-events-root",
                    "/remote/study/derivatives/evs",
                    "--remote-study-root",
                    "/remote/study",
                    "--remote-derivative-root",
                    "/remote/study/derivatives/DeepPrep/BOLD",
                    "--task-id",
                    "exampletask",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            bundle = load_project_bundle("project-deepprep-t1w-analysis", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)
            analysis = bundle["analysis"]["analysis"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])
        self.assertIn("Template: deepprep-t1w-first-level", output)
        self.assertEqual(analysis["scaffold"]["template"], "deepprep-t1w-first-level")
        self.assertEqual(analysis["external_input_roots"]["feat_confounds"]["local_root"], str(events_root.resolve()))
        self.assertEqual(analysis["inputs"]["confounds"]["root_ref"], "feat_confounds")
        self.assertIn("space-T1w_desc-preproc_bold.nii.gz", "\n".join(analysis["inputs"]["bold"]["patterns"]))
        self.assertIn("desc-confounds_noGSR.txt", "\n".join(analysis["inputs"]["confounds"]["patterns"]))
        self.assertEqual(analysis["stages"]["first_level"]["settings"]["smooth_mm"], 0.0)
        self.assertEqual(analysis["stages"]["first_level"]["settings"]["norm"], 0)

    def test_analysis_design_configure_first_level_updates_generated_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = study_root / "derivatives" / "DeepPrep" / "BOLD"
            events_root = workspace_root / "external" / "events"
            confounds_root = workspace_root / "external" / "confounds"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)
            events_root.mkdir(parents=True, exist_ok=True)
            confounds_root.mkdir(parents=True, exist_ok=True)

            self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-design-config",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                    "--template",
                    "deepprep-t1w-first-level",
                    "--events-root",
                    str(events_root),
                    "--remote-events-root",
                    "/remote/study/derivatives/evs",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            exit_code, output = self._run_cli_for_workspace(
                [
                    "analysis",
                    "design",
                    "configure",
                    "first-level",
                    "--project",
                    "project-design-config",
                    "--bold-space",
                    "T1w",
                    "--bold-desc",
                    "preproc",
                    "--confounds-root",
                    str(confounds_root),
                    "--remote-confounds-root",
                    "/remote/study/derivatives/confounds",
                    "--confounds-pattern",
                    "desc-confounds_noGSR.txt",
                    "--tr",
                    "1.0",
                    "--hpf",
                    "100",
                    "--smooth-mm",
                    "0",
                    "--norm",
                    "off",
                    "--motion-correction",
                    "off",
                    "--slice-timing",
                    "off",
                    "--bet",
                    "off",
                    "--prewhiten",
                    "on",
                    "--empty-ev-policy",
                    "as_zero",
                    "--output-desc",
                    "ModelA",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            analysis = load_yaml(workspace_root / "project" / "project-design-config" / "config" / "analysis.yaml")[
                "analysis"
            ]

        self.assertEqual(exit_code, 0)
        self.assertIn('"updated": true', output)
        self.assertEqual(analysis["external_input_roots"]["feat_confounds"]["local_root"], str(confounds_root.resolve()))
        self.assertEqual(analysis["external_input_roots"]["feat_confounds"]["remote_root"], "/remote/study/derivatives/confounds")
        self.assertIn("space-T1w_desc-preproc_bold.nii.gz", "\n".join(analysis["inputs"]["bold"]["patterns"]))
        self.assertEqual(
            analysis["inputs"]["confounds"]["patterns"],
            [
                "{input_root}/{subject_dir}/{session_dir}/func/{bids_base}_desc-confounds_noGSR.txt",
                "{input_root}/{subject_dir}/func/{bids_base}_desc-confounds_noGSR.txt",
            ],
        )
        stage = analysis["stages"]["first_level"]
        self.assertTrue(stage["validation"]["require_confounds"])
        self.assertEqual(stage["validation"]["empty_ev_policy"], "as_zero")
        self.assertEqual(stage["settings"]["tr"], 1.0)
        self.assertEqual(stage["settings"]["hpf"], 100.0)
        self.assertEqual(stage["settings"]["smooth_mm"], 0.0)
        self.assertEqual(stage["settings"]["norm"], 0)
        self.assertEqual(stage["settings"]["mc"], 0)
        self.assertEqual(stage["settings"]["slice_timing"], 0)
        self.assertEqual(stage["settings"]["bet"], 0)
        self.assertEqual(stage["settings"]["prewhiten"], 1)
        self.assertEqual(stage["outputs"]["desc"], "ModelA")

    def test_onboard_analysis_reuses_bids_analysis_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            answers = [
                "",
                "project-onboard-analysis",
                str(study_root),
                str(derivative_root),
                "feat",
                "rest",
                "/remote/studies/demo",
                "/remote/studies/demo/derivatives/fmriprep",
            ]
            with mock.patch("builtins.input", side_effect=answers):
                exit_code, output = self._run_cli_for_workspace(
                    ["onboard", "analysis"],
                    workspace_root=workspace_root,
                    artifact_root=artifact_root,
                )
            bundle = load_project_bundle("project-onboard-analysis", workspace_root)
            errors = validate_project_bundle(bundle, root=workspace_root)

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors, [])
        self.assertIn("Initialized BIDS analysis project: project-onboard-analysis", output)
        self.assertEqual(bundle["analysis"]["analysis"]["defaults"]["tool"], "feat")
        self.assertEqual(bundle["analysis"]["analysis"]["stages"]["first_level"]["default_batch"], "feat_first_level")
        self.assertEqual(bundle["dataset"]["dataset"]["remote_input_derivative_root"], "/remote/studies/demo/derivatives/fmriprep")

    def test_analysis_external_input_roots_preserve_nested_sync_targets_when_remote_root_is_declared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = workspace_root / "project" / "project-analysis"
            ev_root = workspace_root / "external" / "study" / "derivatives" / "evs"
            ev_root.mkdir(parents=True, exist_ok=True)
            analysis_config = {
                "external_input_roots": {
                    "evs": {
                        "local_root": str(ev_root),
                        "remote_root": "${TEST_REMOTE_EV_ROOT:-}",
                        "sync_enabled": True,
                    }
                }
            }

            with mock.patch.dict(
                os.environ,
                {
                    "TEST_REMOTE_EV_ROOT": "/remote/study/derivatives/evs",
                },
                clear=False,
            ):
                resolved, errors = resolve_analysis_external_input_root_declarations(
                    analysis_config,
                    workspace_root=workspace_root,
                    project_root=project_root,
                    require_remote_root=False,
                )

        self.assertEqual(errors, [])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["label"], "evs-root")
        self.assertEqual(resolved[0]["path"], ev_root.resolve())
        self.assertEqual(resolved[0]["remote_root"], "/remote/study/derivatives/evs")
        self.assertTrue(resolved[0]["preserve_nested_sync_target"])

    def test_project_init_bids_analysis_materializes_workspace_runtime_defaults_without_tool_profile_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(
                workspace_root,
                workspace_hpc={
                    "runtime_defaults": {
                        "default": "site-default",
                        "catalog": {
                            "site-default": {
                                "slurm": {
                                    "modules": ["StdEnv/2023", "python/3.11"],
                                    "pre_activate_commands": [
                                        "module load arrow/23.0.1",
                                        "export RP_CLUSTER=1",
                                    ],
                                    "prepare_directories": [
                                        "$SITE_CACHE",
                                    ],
                                    "remote_workspace_root": "/remote/workspace-default",
                                    "remote_artifacts_root": "/remote/scratch/shared-artifacts",
                                }
                            }
                        },
                    }
                },
            )
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            exit_code, _ = self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-new-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            compute_bundle = load_project_bundle("project-new-analysis", workspace_root)["compute"]["compute"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(compute_bundle["slurm"]["modules"], ["StdEnv/2023", "python/3.11"])
        self.assertEqual(
            compute_bundle["slurm"]["pre_activate_commands"],
            [
                "module load arrow/23.0.1",
                "export RP_CLUSTER=1",
                '[ -n "$SCRATCH" ] || { echo "ERROR: SCRATCH is not set on the remote node." >&2; exit 1; }',
            ],
        )
        self.assertEqual(
            compute_bundle["slurm"]["prepare_directories"],
            ["$SITE_CACHE", "$APPTAINER_CACHEDIR", "$APPTAINER_TMPDIR", "$TMPDIR"],
        )
        self.assertEqual(compute_bundle["slurm"]["remote_workspace_root"], "/remote/workspace-default")
        self.assertEqual(compute_bundle["slurm"]["remote_artifacts_root"], "/remote/scratch/shared-artifacts")
        self.assertNotIn("modules", compute_bundle["tool_profiles"]["fsl"]["slurm"])
        self.assertNotIn("pre_activate_commands", compute_bundle["tool_profiles"]["fsl"]["slurm"])

    def test_batch_discover_and_plan_analysis_bids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            study_root.mkdir(parents=True, exist_ok=True)
            func_root = derivative_root / "sub-001" / "ses-01" / "func"
            func_root.mkdir(parents=True, exist_ok=True)
            (
                func_root
                / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")

            self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-new-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

            discover_exit, discover_output = self._run_cli_for_workspace(
                [
                    "batch",
                    "discover",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            plan_exit, _ = self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                    "--empty-ev-policy",
                    "fail",
                    "--output-desc",
                    "modelA",
                    "--run-id",
                    "feat-plan",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            slurm_exit, _ = self._run_cli_for_workspace(
                [
                    "run",
                    "slurm",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                    "--run-id",
                    "feat-slurm",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            with mock.patch("research_platform.core.cli.execute_stage_plan") as stage_mock:
                with mock.patch("research_platform.core.cli.execute_submit_plan") as submit_mock:
                    submit_plan_exit, submit_plan_output = self._run_cli_for_workspace(
                        [
                            "run",
                            "submit",
                            "analysis",
                            "bids",
                            "--project",
                            "project-new-analysis",
                            "--stage",
                            "first_level",
                            "--discover",
                            "--run-id",
                            "feat-submit-plan",
                        ],
                        workspace_root=workspace_root,
                        artifact_root=artifact_root,
                    )
            submit_plan_payload = json.loads(submit_plan_output)

            paths = workspace_paths(
                workspace_root,
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                },
            )
            plan_manifest = run_path(paths["artifacts_root"], "feat-plan") / "run-manifest.yaml"
            slurm_manifest = run_path(paths["artifacts_root"], "feat-slurm") / "run-manifest.yaml"
            batch_text = (
                workspace_root / "project" / "project-new-analysis" / "manifests" / "batches" / "feat_first_level.tsv"
            ).read_text(encoding="utf-8")
            plan_manifest_text = plan_manifest.read_text(encoding="utf-8")
            slurm_manifest_text = slurm_manifest.read_text(encoding="utf-8")
            plan_manifest = load_yaml(plan_manifest)

        self.assertEqual(discover_exit, 0)
        self.assertEqual(plan_exit, 0)
        self.assertEqual(slurm_exit, 0)
        self.assertEqual(submit_plan_exit, 0)
        stage_mock.assert_not_called()
        submit_mock.assert_not_called()
        self.assertIn(
            str(
                (
                    workspace_root
                    / "project"
                    / "project-new-analysis"
                    / "manifests"
                    / "batches"
                    / "feat_first_level.tsv"
                ).resolve()
            ),
            submit_plan_payload["local_files"],
        )
        self.assertIn('"row_count": 1', discover_output)
        self.assertIn("sub-001\tses-01\ttask-rest\trun-01", batch_text)
        self.assertIn("workflow:", plan_manifest_text)
        self.assertIn("action: analysis", plan_manifest_text)
        self.assertIn("tool:", plan_manifest_text)
        self.assertIn("name: feat", plan_manifest_text)
        self.assertEqual(
            plan_manifest["analysis"]["stage_config"]["validation"]["empty_ev_policy"],
            "fail",
        )
        self.assertEqual(
            plan_manifest["analysis"]["stage_config"]["outputs"]["desc"],
            "modelA",
        )
        self.assertIn("slurm:", slurm_manifest_text)

    def test_plan_analysis_bids_rejects_invalid_output_desc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-new-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            error = self._run_cli_for_workspace_system_exit(
                [
                    "run",
                    "plan",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                    "--output-desc",
                    "model A",
                    "--run-id",
                    "feat-plan",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )

        self.assertIn("--output-desc must contain only letters and numbers", error)

    def test_plan_analysis_bids_filters_batch_by_subject_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            study_root.mkdir(parents=True, exist_ok=True)
            derivative_root.mkdir(parents=True, exist_ok=True)

            self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-new-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            batch_path = (
                workspace_root
                / "project"
                / "project-new-analysis"
                / "manifests"
                / "batches"
                / "feat_first_level.tsv"
            )
            batch_path.write_text(
                "\n".join(
                    [
                        "subject_id\tsession_id\ttask_id\trun_id",
                        "sub-007\tses-01\ttask-rest\trun-01",
                        "sub-009\tses-01\ttask-rest\trun-01",
                        "sub-010\tses-01\ttask-rest\trun-01",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            exit_code, _ = self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                    "--subject-id",
                    "007",
                    "sub-009",
                    "--run-id",
                    "feat-filtered-plan",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            paths = workspace_paths(
                workspace_root,
                {
                    "paths": {"artifacts_root": "./artifacts", "datasets_root": "./datasets", "ops_root": "./ops"},
                    "repos": {"project_root": "./project", "pipelines_root": "./pipelines"},
                },
            )
            manifest_path = run_path(paths["artifacts_root"], "feat-filtered-plan") / "run-manifest.yaml"
            manifest = load_yaml(manifest_path)
            filtered_batch = workspace_root / manifest["batch"]["path"]
            filtered_text = filtered_batch.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["batch"]["row_count"], 2)
        self.assertEqual(manifest["batch"]["source_row_count"], 3)
        self.assertEqual(manifest["batch"]["filters"]["subject_id"], ["sub-007", "sub-009"])
        self.assertIn("sub-007\tses-01\ttask-rest\trun-01", filtered_text)
        self.assertIn("sub-009\tses-01\ttask-rest\trun-01", filtered_text)
        self.assertNotIn("sub-010", filtered_text)

    def test_config_validate_and_plan_analysis_bids_allow_unset_remote_external_root_envs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            ev_root = workspace_root.parent / f"{workspace_root.name}-external-events"
            func_root = derivative_root / "sub-001" / "ses-01" / "func"
            event_func_root = ev_root / "sub-001" / "ses-01" / "func"
            study_root.mkdir(parents=True, exist_ok=True)
            func_root.mkdir(parents=True, exist_ok=True)
            event_func_root.mkdir(parents=True, exist_ok=True)
            (
                func_root
                / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")
            (event_func_root / "sub-001_ses-01_task-rest_run-01_desc-confounds_noGSR.txt").write_text(
                "0 0\n",
                encoding="utf-8",
            )
            for ev_name in ("condition_a", "condition_b", "button_press"):
                (
                    event_func_root / f"sub-001_ses-01_task-rest_run-01_desc-{ev_name}_events.txt"
                ).write_text("0 1 1\n", encoding="utf-8")

            self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-new-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            project_root = workspace_root / "project" / "project-new-analysis"
            self._configure_external_analysis_inputs(
                project_root=project_root,
                ev_env_name="ANALYSIS_EV_ROOT",
                remote_ev_env_name="ANALYSIS_REMOTE_EV_ROOT",
            )

            validate_exit, validate_output = self._run_cli_for_workspace(
                ["config", "validate", "--project", "project-new-analysis"],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={"ANALYSIS_EV_ROOT": str(ev_root), "ANALYSIS_REMOTE_EV_ROOT": None},
            )
            discover_exit, _ = self._run_cli_for_workspace(
                [
                    "batch",
                    "discover",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={"ANALYSIS_EV_ROOT": str(ev_root), "ANALYSIS_REMOTE_EV_ROOT": None},
            )
            plan_exit, _ = self._run_cli_for_workspace(
                [
                    "run",
                    "plan",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                    "--run-id",
                    "feat-plan-external",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={"ANALYSIS_EV_ROOT": str(ev_root), "ANALYSIS_REMOTE_EV_ROOT": None},
            )

        self.assertEqual(validate_exit, 0)
        self.assertEqual(discover_exit, 0)
        self.assertEqual(plan_exit, 0)
        self.assertIn('"valid": true', validate_output)

    def test_run_slurm_analysis_bids_fails_cleanly_when_external_root_has_no_remote_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            ev_root = workspace_root.parent / f"{workspace_root.name}-external-events"
            func_root = derivative_root / "sub-001" / "ses-01" / "func"
            event_func_root = ev_root / "sub-001" / "ses-01" / "func"
            study_root.mkdir(parents=True, exist_ok=True)
            func_root.mkdir(parents=True, exist_ok=True)
            event_func_root.mkdir(parents=True, exist_ok=True)
            (
                func_root
                / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")
            (event_func_root / "sub-001_ses-01_task-rest_run-01_desc-confounds_noGSR.txt").write_text(
                "0 0\n",
                encoding="utf-8",
            )
            for ev_name in ("condition_a", "condition_b", "button_press"):
                (
                    event_func_root / f"sub-001_ses-01_task-rest_run-01_desc-{ev_name}_events.txt"
                ).write_text("0 1 1\n", encoding="utf-8")

            self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-new-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            project_root = workspace_root / "project" / "project-new-analysis"
            self._configure_external_analysis_inputs(
                project_root=project_root,
                ev_env_name="ANALYSIS_EV_ROOT",
                remote_ev_env_name="ANALYSIS_REMOTE_EV_ROOT",
            )
            self._run_cli_for_workspace(
                [
                    "batch",
                    "discover",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={"ANALYSIS_EV_ROOT": str(ev_root), "ANALYSIS_REMOTE_EV_ROOT": None},
            )
            error = self._run_cli_for_workspace_system_exit(
                [
                    "run",
                    "slurm",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                    "--run-id",
                    "feat-slurm-external-missing",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={"ANALYSIS_EV_ROOT": str(ev_root), "ANALYSIS_REMOTE_EV_ROOT": None},
            )

        self.assertIn("selected external analysis input roots", error)
        self.assertIn("feat-confounds-root", error)

    def test_run_slurm_analysis_bids_uses_remote_external_root_paths_in_manifest_and_provision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            artifact_root = workspace_root / "artifacts"
            self._write_analysis_workspace(workspace_root)
            study_root = workspace_root / "external" / "study"
            derivative_root = workspace_root / "external" / "derivatives" / "fmriprep"
            ev_root = workspace_root.parent / f"{workspace_root.name}-external-events"
            func_root = derivative_root / "sub-001" / "ses-01" / "func"
            event_func_root = ev_root / "sub-001" / "ses-01" / "func"
            study_root.mkdir(parents=True, exist_ok=True)
            func_root.mkdir(parents=True, exist_ok=True)
            event_func_root.mkdir(parents=True, exist_ok=True)
            (
                func_root
                / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            ).write_text("", encoding="utf-8")
            (event_func_root / "sub-001_ses-01_task-rest_run-01_desc-confounds_noGSR.txt").write_text(
                "0 0\n",
                encoding="utf-8",
            )
            for ev_name in ("condition_a", "condition_b", "button_press"):
                (
                    event_func_root / f"sub-001_ses-01_task-rest_run-01_desc-{ev_name}_events.txt"
                ).write_text("0 1 1\n", encoding="utf-8")

            self._run_cli_for_workspace(
                [
                    "project",
                    "init",
                    "bids-analysis",
                    "--project",
                    "project-new-analysis",
                    "--study-root",
                    str(study_root),
                    "--derivative-root",
                    str(derivative_root),
                    "--tool",
                    "feat",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
            )
            project_root = workspace_root / "project" / "project-new-analysis"
            self._configure_external_analysis_inputs(
                project_root=project_root,
                ev_env_name="ANALYSIS_EV_ROOT",
                remote_ev_env_name="ANALYSIS_REMOTE_EV_ROOT",
            )
            self._run_cli_for_workspace(
                [
                    "batch",
                    "discover",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "ANALYSIS_EV_ROOT": str(ev_root),
                    "ANALYSIS_REMOTE_EV_ROOT": "/remote/analysis/events",
                },
            )
            slurm_exit, _ = self._run_cli_for_workspace(
                [
                    "run",
                    "slurm",
                    "analysis",
                    "bids",
                    "--project",
                    "project-new-analysis",
                    "--stage",
                    "first_level",
                    "--run-id",
                    "feat-slurm-external",
                ],
                workspace_root=workspace_root,
                artifact_root=artifact_root,
                extra_env={
                    "ANALYSIS_EV_ROOT": str(ev_root),
                    "ANALYSIS_REMOTE_EV_ROOT": "/remote/analysis/events",
                },
            )
            manifest = load_yaml(artifact_root / "runs" / "feat-slurm-external" / "run-manifest.yaml")

        self.assertEqual(slurm_exit, 0)
        self.assertEqual(manifest["analysis"]["input_roots"]["evs"]["path"], "/remote/analysis/events")
        self.assertEqual(manifest["analysis"]["input_roots"]["feat_confounds"]["path"], "/remote/analysis/events")
        neuro_bids_scope = next(scope for scope in manifest["provision"]["scopes"] if scope["name"] == "neuro-bids")
        labels_to_destinations = {entry["label"]: entry["destination"] for entry in neuro_bids_scope["entries"]}
        self.assertEqual(labels_to_destinations["evs-root"], "/remote/analysis/events")


if __name__ == "__main__":
    unittest.main()
