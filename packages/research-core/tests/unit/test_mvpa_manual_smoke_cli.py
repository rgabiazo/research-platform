from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")

CORE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HPC_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-hpc"
NEURO_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-neuro"
ANALYSIS_PACKAGE_ROOT = WORKSPACE_ROOT / "packages" / "research-analysis"
sys.path.insert(0, str(CORE_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(HPC_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(NEURO_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(ANALYSIS_PACKAGE_ROOT / "src"))

from research_platform.core.cli import main


def test_manual_crossnobis_smoke_cli_is_read_only_by_default(tmp_path: Path) -> None:
    mask_path = _write_image(tmp_path / "roi.nii.gz", np.ones((2, 1, 1), dtype=np.uint8))
    run1 = _write_feat_run(tmp_path / "run1.feat", pair_values=[1.0, -1.0], item_values=[0.0, 0.0])
    run2 = _write_feat_run(tmp_path / "run2.feat", pair_values=[1.0, -1.0], item_values=[0.0, 0.0])
    event_args = _write_event_args(tmp_path, runs=("01", "02"))
    before = _file_snapshot(tmp_path)

    code, payload = _run_smoke(
        [
            "--subject",
            "sub-001",
            "--roi-mask",
            mask_path.as_posix(),
            "--roi-label",
            "SeedA",
            "--phase",
            "encoding",
            "--condition-a",
            "pair_enc_hit",
            "--condition-b",
            "item_enc_hit",
            "--feat-run",
            f"01={run1}",
            "--feat-run",
            f"02={run2}",
            *event_args,
        ]
    )
    after = _file_snapshot(tmp_path)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["crossnobis"] == pytest.approx(1.0)
    assert payload["n_valid_runs"] == 2
    assert payload["valid_runs"] == ["01", "02"]
    assert payload["n_voxels_used"] == 2
    assert payload["sigma_pooling_source_runs"] == ["01", "02"]
    assert after == before


def test_manual_crossnobis_smoke_cli_reference_comparison_reports_pass_and_fail(tmp_path: Path) -> None:
    mask_path = _write_image(tmp_path / "roi.nii.gz", np.ones((2, 1, 1), dtype=np.uint8))
    run1 = _write_feat_run(tmp_path / "run1.feat", pair_values=[1.0, -1.0], item_values=[0.0, 0.0])
    run2 = _write_feat_run(tmp_path / "run2.feat", pair_values=[1.0, -1.0], item_values=[0.0, 0.0])
    event_args = _write_event_args(tmp_path, runs=("01", "02"))
    pass_ref = tmp_path / "reference-pass.tsv"
    pass_ref.write_text("subject_id\troi_label\tphase\tcrossnobis\nsub-001\tSeedA\tencoding\t1.0\n", encoding="utf-8")
    fail_ref = tmp_path / "reference-fail.tsv"
    fail_ref.write_text("subject_id\troi_label\tphase\tcrossnobis\nsub-001\tSeedA\tencoding\t2.0\n", encoding="utf-8")

    base_args = [
        "--subject",
        "sub-001",
        "--roi-mask",
        mask_path.as_posix(),
        "--roi-label",
        "SeedA",
        "--phase",
        "encoding",
        "--condition-a",
        "pair_enc_hit",
        "--condition-b",
        "item_enc_hit",
        "--feat-run",
        f"01={run1}",
        "--feat-run",
        f"02={run2}",
        *event_args,
    ]
    pass_code, pass_payload = _run_smoke([*base_args, "--reference-tsv", pass_ref.as_posix()])
    fail_code, fail_payload = _run_smoke([*base_args, "--reference-tsv", fail_ref.as_posix()])

    assert pass_code == 0
    assert pass_payload["reference_comparison"]["computed_crossnobis"] == pytest.approx(1.0)
    assert pass_payload["reference_comparison"]["reference_crossnobis"] == pytest.approx(1.0)
    assert pass_payload["reference_comparison"]["absolute_difference"] == pytest.approx(0.0)
    assert pass_payload["reference_comparison"]["passed"] is True
    assert fail_code == 0
    assert fail_payload["reference_comparison"]["reference_crossnobis"] == pytest.approx(2.0)
    assert fail_payload["reference_comparison"]["passed"] is False


def _run_smoke(args: list[str]) -> tuple[int, dict[str, object]]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["analysis", "mvpa", "smoke-manual-crossnobis", *args])
    return code, json.loads(buffer.getvalue())


def _write_feat_run(path: Path, *, pair_values: list[float], item_values: list[float]) -> Path:
    stats = path / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (path / "design.fsf").write_text(
        "\n".join(
            [
                'set fmri(evtitle1) "pair_enc_hit"',
                "set fmri(deriv_yn1) 1",
                'set fmri(evtitle2) "item_enc_hit"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_image(stats / "pe1.nii.gz", np.asarray(pair_values, dtype=float).reshape((2, 1, 1)))
    _write_image(stats / "pe3.nii.gz", np.asarray(item_values, dtype=float).reshape((2, 1, 1)))
    _write_image(stats / "sigmasquareds.nii.gz", np.ones((2, 1, 1), dtype=float))
    return path


def _write_event_args(tmp_path: Path, *, runs: tuple[str, ...]) -> list[str]:
    args: list[str] = []
    for run_id in runs:
        for condition in ("pair_enc_hit", "item_enc_hit"):
            path = tmp_path / f"run-{run_id}_{condition}.txt"
            path.write_text("0 1 1\n2 1 1\n", encoding="utf-8")
            args.extend(["--event-file", f"{run_id}:{condition}:{path}"])
    return args


def _write_image(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4)), path)
    return path


def _file_snapshot(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
