from __future__ import annotations

from pathlib import Path
import gzip
import struct
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.fsl.common import (
    FslContainerSpec,
    build_fsl_container_prepare_shell,
    build_fsl_container_prepare_and_exec_shell,
    build_fsl_headless_env,
    collect_fsl_bind_roots,
    infer_nvols_and_tr,
    resolve_fsl_container_runtime_image,
    resolve_fsl_runtime_backend,
)


class FslCommonTests(unittest.TestCase):
    def test_resolve_fsl_runtime_backend_supports_native_and_apptainer(self) -> None:
        native = resolve_fsl_runtime_backend(
            {"local": {"execution_backend": "native", "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"}}},
            mode="local",
        )
        apptainer = resolve_fsl_runtime_backend(
            {
                "slurm": {
                    "execution_backend": "apptainer",
                    "environment": {"FSLOUTPUTTYPE": "NIFTI_GZ"},
                    "container": {
                        "enabled": True,
                        "backend": "apptainer",
                        "image": "docker://lab/fsl:6.0.7",
                        "pull_mode": "if_missing",
                        "image_root": "$SCRATCH/containers/fsl",
                    },
                }
            },
            mode="slurm",
        )

        self.assertEqual(native.execution_backend, "native")
        self.assertEqual(native.environment["FSLOUTPUTTYPE"], "NIFTI_GZ")
        self.assertEqual(apptainer.execution_backend, "apptainer")
        self.assertIsNotNone(apptainer.container)
        self.assertEqual(apptainer.container.pull_mode, "if_missing")
        self.assertTrue(apptainer.container.image_name.endswith(".sif"))

    def test_resolve_fsl_runtime_backend_rejects_invalid_container_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires container.enabled=true"):
            resolve_fsl_runtime_backend(
                {
                    "slurm": {
                        "execution_backend": "apptainer",
                        "container": {"enabled": False, "backend": "apptainer", "image": "docker://lab/fsl:6.0.7"},
                    }
                },
                mode="slurm",
            )

        with self.assertRaisesRegex(ValueError, "container.image_root is required"):
            resolve_fsl_runtime_backend(
                {
                    "slurm": {
                        "execution_backend": "singularity",
                        "container": {
                            "enabled": True,
                            "backend": "singularity",
                            "image": "docker://lab/fsl:6.0.7",
                            "pull_mode": "if_missing",
                        },
                    }
                },
                mode="slurm",
            )

    def test_resolve_fsl_container_runtime_image_uses_existing_sif_or_materializes_docker_uri(self) -> None:
        direct = resolve_fsl_container_runtime_image(
            FslContainerSpec(
                enabled=True,
                backend="apptainer",
                image="/shared/containers/fsl.sif",
                pull_mode="never",
            )
        )
        pulled = resolve_fsl_container_runtime_image(
            FslContainerSpec(
                enabled=True,
                backend="apptainer",
                image="docker://ghcr.io/example/fsl:6.0.7",
                pull_mode="if_missing",
                image_name="fsl-runtime",
                image_root="$SCRATCH/containers/fsl",
            )
        )

        self.assertEqual(direct["runtime_image"], "/shared/containers/fsl.sif")
        self.assertFalse(direct["requires_pull"])
        self.assertEqual(pulled["runtime_image"], "$SCRATCH/containers/fsl/fsl-runtime.sif")
        self.assertTrue(pulled["requires_pull"])

    def test_collect_fsl_bind_roots_includes_selected_analysis_roots_and_dedupes_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            dataset_root = Path(tmp_dir) / "datasets" / "study"
            derivative_root = Path(tmp_dir) / "derivatives" / "fmriprep"
            event_root = Path(tmp_dir) / "analysis-inputs" / "events"
            output_root = workspace_root / "artifacts" / "runs" / "feat-demo" / "outputs"
            output_root.mkdir(parents=True, exist_ok=True)

            manifest = {
                "dataset": {
                    "root": str(dataset_root),
                    "derivative_root": str(derivative_root),
                },
                "analysis": {
                    "input_roots": {
                        "evs": {"path": str(event_root)},
                    },
                    "inputs": {
                        "confounds": {"root": str(event_root / "confounds")},
                    },
                },
            }

            bind_roots = collect_fsl_bind_roots(
                manifest=manifest,
                output_root=output_root,
                workspace_root=workspace_root,
            )

        self.assertIn(str(workspace_root.resolve()), bind_roots)
        self.assertIn(str(dataset_root.resolve()), bind_roots)
        self.assertIn(str(derivative_root.resolve()), bind_roots)
        self.assertIn(str(event_root.resolve()), bind_roots)
        self.assertNotIn(str((event_root / "confounds").resolve()), bind_roots)
        self.assertNotIn(str(output_root.resolve()), bind_roots)

    def test_build_fsl_container_prepare_and_exec_shell_pulls_once_and_runs_headless(self) -> None:
        with mock.patch.dict("research_platform.neuro.fsl.common.os.environ", {"USER": "demo-user"}, clear=True):
            shell = build_fsl_container_prepare_and_exec_shell(
                backend="apptainer",
                container=FslContainerSpec(
                    enabled=True,
                    backend="apptainer",
                    image="docker://ghcr.io/example/fsl:6.0.7",
                    pull_mode="if_missing",
                    image_name="fsl-feat",
                    image_root="$SCRATCH/containers/fsl",
                ),
                bind_roots=["/remote/workspace", "/remote/study"],
                env={"FSLOUTPUTTYPE": "NIFTI_GZ"},
                command=["feat", "/remote/workspace/artifacts/runs/demo/outputs/fsf/demo.fsf"],
            )
            headless_env = build_fsl_headless_env({"FSLOUTPUTTYPE": "NIFTI_GZ"})

        self.assertIn('mkdir -p "$IMAGE_ROOT"', shell)
        self.assertIn('while ! mkdir "$LOCK_DIR" 2>/dev/null; do', shell)
        self.assertIn('apptainer pull "$TMP_IMAGE" "$IMAGE_SOURCE"', shell)
        self.assertIn('mv "$TMP_IMAGE" "$RUNTIME_IMAGE"', shell)
        self.assertIn("exec apptainer exec --cleanenv", shell)
        self.assertIn("--bind /remote/workspace:/remote/workspace", shell)
        self.assertIn("--bind /remote/study:/remote/study", shell)
        self.assertIn(
            "--env BROWSER=false,FSLOUTPUTTYPE=NIFTI_GZ,FSL_FEAT_WATCH=0,USER=demo-user",
            shell,
        )
        self.assertEqual(headless_env["BROWSER"], "false")
        self.assertEqual(headless_env["FSL_FEAT_WATCH"], "0")
        self.assertEqual(headless_env["USER"], "demo-user")

    def test_build_fsl_container_prepare_shell_exports_apptainer_temp_and_cache(self) -> None:
        shell = build_fsl_container_prepare_shell(
            backend="apptainer",
            container=FslContainerSpec(
                enabled=True,
                backend="apptainer",
                image="docker://ghcr.io/example/fsl:6.0.7",
                pull_mode="if_missing",
                image_name="fsl-feat",
                image_root="$SCRATCH/containers/fsl",
            ),
        )

        self.assertIn('export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH}/apptainer-cache}"', shell)
        self.assertIn('export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SLURM_TMPDIR:-$SCRATCH/apptainer-tmp}}"', shell)
        self.assertIn('rm -rf "${RUNTIME_IMAGE}.lock.d"', shell)
        self.assertIn('apptainer pull "$TMP_IMAGE" "$IMAGE_SOURCE"', shell)

    def test_infer_nvols_and_tr_reads_gzipped_nifti_header_without_fsl_or_nibabel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            nifti_path = Path(tmp_dir) / "demo_bold.nii.gz"
            _write_fake_nifti1_header(nifti_path, nvols=173, tr=1.0)

            real_import = __import__

            def _fake_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "nibabel":
                    raise ImportError("synthetic missing nibabel")
                return real_import(name, *args, **kwargs)

            with (
                mock.patch("research_platform.neuro.fsl.common.shutil.which", return_value=None),
                mock.patch("builtins.__import__", side_effect=_fake_import),
            ):
                nvols, tr = infer_nvols_and_tr(nifti_path)

        self.assertEqual(nvols, 173)
        self.assertEqual(tr, 1.0)

    def test_infer_nvols_and_tr_keeps_override_tr_when_header_tr_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            nifti_path = Path(tmp_dir) / "demo_bold.nii.gz"
            _write_fake_nifti1_header(nifti_path, nvols=200, tr=0.0)

            real_import = __import__

            def _fake_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "nibabel":
                    raise ImportError("synthetic missing nibabel")
                return real_import(name, *args, **kwargs)

            with (
                mock.patch("research_platform.neuro.fsl.common.shutil.which", return_value=None),
                mock.patch("builtins.__import__", side_effect=_fake_import),
            ):
                nvols, tr = infer_nvols_and_tr(nifti_path, override_tr=1.0)

        self.assertEqual(nvols, 200)
        self.assertEqual(tr, 1.0)

    def test_infer_nvols_and_tr_reads_repetition_time_from_related_json_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nifti_path = (
                root
                / "derivatives"
                / "fmripost_aroma"
                / "sub-001"
                / "ses-01"
                / "func"
                / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-nonaggrDenoised_bold.nii.gz"
            )
            nifti_path.parent.mkdir(parents=True, exist_ok=True)
            _write_fake_nifti1_header(nifti_path, nvols=180, tr=0.0)

            metadata_path = (
                root
                / "derivatives"
                / "upstream"
                / "sub-001"
                / "ses-01"
                / "func"
                / "sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json"
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text('{"RepetitionTime": 1.5}', encoding="utf-8")

            real_import = __import__

            def _fake_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "nibabel":
                    raise ImportError("synthetic missing nibabel")
                return real_import(name, *args, **kwargs)

            with (
                mock.patch("research_platform.neuro.fsl.common.shutil.which", return_value=None),
                mock.patch("builtins.__import__", side_effect=_fake_import),
            ):
                nvols, tr = infer_nvols_and_tr(
                    nifti_path,
                    metadata_search_roots=[root],
                )

        self.assertEqual(nvols, 180)
        self.assertEqual(tr, 1.5)


def _write_fake_nifti1_header(path: Path, *, nvols: int, tr: float) -> None:
    header = bytearray(348)
    struct.pack_into("<I", header, 0, 348)
    struct.pack_into("<8h", header, 40, 4, 64, 64, 36, nvols, 1, 1, 1)
    struct.pack_into("<8f", header, 76, 0.0, 2.0, 2.0, 2.0, tr, 0.0, 0.0, 0.0)
    with gzip.open(path, "wb") as handle:
        handle.write(header)
        handle.write(b"\0" * 16)


if __name__ == "__main__":
    unittest.main()
