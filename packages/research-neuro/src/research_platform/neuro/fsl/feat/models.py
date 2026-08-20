"""First-level FEAT models and FSF rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import shutil
import subprocess
from typing import Any


@dataclass(frozen=True)
class FeatContrast:
    name: str
    weights: list[float]


@dataclass(frozen=True)
class FeatFirstLevelModel:
    name: str
    ev_order: list[str]
    derivative_on: list[str]
    contrasts: list[FeatContrast]
    nonconvolved: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatFirstLevelPlan:
    row: dict[str, str]
    entities: dict[str, str]
    bids_base: str
    bold_path: Path
    confounds_path: Path | None
    ev_paths_by_name: dict[str, Path]
    missing_evs: list[str]
    out_dir: Path
    fsf_path: Path
    empty_evs: list[str] = field(default_factory=list)
    nvols: int | None = None
    tr: float | None = None


def load_model_spec(model_ref: str, payload: dict[str, Any]) -> FeatFirstLevelModel:
    ev_order = [str(name).strip() for name in payload.get("ev_order", []) if str(name).strip()]
    derivative_on = [str(name).strip() for name in payload.get("derivative_on", []) if str(name).strip()]
    nonconvolved = [str(name).strip() for name in payload.get("nonconvolved", []) if str(name).strip()]
    contrasts_payload = payload.get("contrasts", [])
    contrasts: list[FeatContrast] = []
    for raw_contrast in contrasts_payload if isinstance(contrasts_payload, list) else []:
        if not isinstance(raw_contrast, dict):
            continue
        name = str(raw_contrast.get("name", "")).strip()
        weights = [float(weight) for weight in raw_contrast.get("weights", [])]
        if name:
            contrasts.append(FeatContrast(name=name, weights=weights))
    return FeatFirstLevelModel(
        name=str(payload.get("name") or model_ref),
        ev_order=ev_order,
        derivative_on=derivative_on,
        contrasts=contrasts,
        nonconvolved=nonconvolved,
    )


def render_first_level_fsf(
    *,
    model: FeatFirstLevelModel,
    plan: FeatFirstLevelPlan,
    tr: float,
    npts: int,
    settings: dict[str, Any],
) -> str:
    version = _feat_header_version()
    evs_orig = len(model.ev_order)
    derivative_on = set(model.derivative_on)
    nonconvolved = set(model.nonconvolved)
    empty_evs = set(plan.empty_evs)
    evs_real = evs_orig + sum(1 for ev_name in model.ev_order if ev_name in derivative_on and ev_name not in empty_evs)
    ncon = len(model.contrasts)

    delete_vols = int(_setting(settings, "delete_vols", default=0) or 0)
    hpf = float(_setting(settings, "hpf", default=100.0))
    smooth = float(_setting(settings, "smooth_mm", "smooth", default=5.0))
    norm_yn = int(_setting(settings, "norm", default=1) or 0)
    prewhiten_yn = int(_setting(settings, "prewhiten", default=1) or 0)
    slice_timing = int(_setting(settings, "slice_timing", default=0) or 0)
    bet_yn = int(_setting(settings, "bet", default=0) or 0)
    mc_yn = int(_setting(settings, "mc", default=0) or 0)

    lines: list[str] = [
        "# FEAT version number",
        f"set fmri(version) {version}",
        "set fmri(inmelodic) 0",
        "set fmri(level) 1",
        "set fmri(analysis) 7",
        "set fmri(relative_yn) 0",
        "set fmri(help_yn) 1",
        "set fmri(featwatcher_yn) 1",
        "set fmri(sscleanup_yn) 0",
        f"set fmri(outputdir) \"{plan.out_dir}\"",
        f"set fmri(tr) {tr}",
        f"set fmri(npts) {npts}",
        f"set fmri(ndelete) {delete_vols}",
        "set fmri(tagfirst) 1",
        "set fmri(multiple) 1",
        "set fmri(inputtype) 2",
        "set fmri(filtering_yn) 1",
        "set fmri(brain_thresh) 10",
        "set fmri(critical_z) 5.3",
        "set fmri(noise) 0.66",
        "set fmri(noisear) 0.34",
        f"set fmri(mc) {mc_yn}",
        "set fmri(sh_yn) 0",
        "set fmri(regunwarp_yn) 0",
        "set fmri(gdc) \"\"",
        "set fmri(dwell) 0.0",
        "set fmri(te) 0.0",
        "set fmri(signallossthresh) 10",
        "set fmri(unwarp_dir) y-",
        f"set fmri(st) {slice_timing}",
        "set fmri(st_file) \"\"",
        f"set fmri(bet_yn) {bet_yn}",
        f"set fmri(smooth) {smooth}",
        f"set fmri(norm_yn) {norm_yn}",
        "set fmri(perfsub_yn) 0",
        "set fmri(temphp_yn) 1",
        "set fmri(templp_yn) 0",
        "set fmri(melodic_yn) 0",
        "set fmri(stats_yn) 1",
        f"set fmri(prewhiten_yn) {prewhiten_yn}",
        "set fmri(motionevs) 0",
        "set fmri(motionevsbeta) \"\"",
        "set fmri(scriptevsbeta) \"\"",
        "set fmri(robust_yn) 0",
        "set fmri(mixed_yn) 2",
        "set fmri(randomisePermutations) 5000",
        f"set fmri(evs_orig) {evs_orig}",
        f"set fmri(evs_real) {evs_real}",
        "set fmri(evs_vox) 0",
        "set fmri(con_mode_old) orig",
        "set fmri(con_mode) orig",
        f"set fmri(ncon_orig) {ncon}",
        f"set fmri(ncon_real) {ncon}",
        "set fmri(nftests_orig) 0",
        "set fmri(nftests_real) 0",
        "set fmri(constcol) 0",
        "set fmri(poststats_yn) 1",
        "set fmri(threshmask) \"\"",
        "set fmri(thresh) 3",
        "set fmri(prob_thresh) 0.05",
        "set fmri(z_thresh) 3.1",
        "set fmri(zdisplay) 0",
        "set fmri(zmin) 2",
        "set fmri(zmax) 8",
        "set fmri(rendertype) 1",
        "set fmri(bgimage) 1",
        "set fmri(tsplot_yn) 1",
        "set fmri(reginitial_highres_yn) 0",
        "set fmri(reginitial_highres_search) 90",
        "set fmri(reginitial_highres_dof) 3",
        "set fmri(reghighres_yn) 0",
        "set fmri(reghighres_search) 90",
        "set fmri(reghighres_dof) BBR",
        "set fmri(regstandard_yn) 0",
        "set fmri(alternateReference_yn) 0",
        "set fmri(regstandard) \"\"",
        "set fmri(regstandard_search) 90",
        "set fmri(regstandard_dof) 12",
        "set fmri(regstandard_nonlinear_yn) 0",
        "set fmri(regstandard_nonlinear_warpres) 10",
        f"set fmri(paradigm_hp) {hpf}",
        "set fmri(fnirt_config) \"T1_2_MNI152_2mm\"",
        "set fmri(ncopeinputs) 0",
        f"set feat_files(1) \"{plan.bold_path}\"",
    ]

    if plan.confounds_path is not None:
        lines.extend(
            [
                "set fmri(confoundevs) 1",
                f"set confoundev_files(1) \"{plan.confounds_path}\"",
            ]
        )
    else:
        lines.append("set fmri(confoundevs) 0")

    for index, ev_name in enumerate(model.ev_order, start=1):
        ev_path = plan.ev_paths_by_name.get(ev_name)
        ev_is_empty = ev_name in empty_evs
        conv_code = 0 if ev_is_empty or ev_name in nonconvolved else 3
        lines.extend(
            [
                f"set fmri(evtitle{index}) \"{ev_name}\"",
                f"set fmri(shape{index}) {10 if ev_is_empty else 3}",
                f"set fmri(convolve{index}) {conv_code}",
                f"set fmri(convolve_phase{index}) 0",
                f"set fmri(tempfilt_yn{index}) 1",
                f"set fmri(deriv_yn{index}) {1 if ev_name in derivative_on and not ev_is_empty else 0}",
                f"set fmri(skip{index}) 0",
                f"set fmri(off{index}) 0",
                f"set fmri(on{index}) 0",
                f"set fmri(phase{index}) 0",
                f"set fmri(stop{index}) -1",
                f"set fmri(gammasigma{index}) 3",
                f"set fmri(gammadelay{index}) 6",
                f"set fmri(ortho{index}.0) 0",
            ]
        )
        if not ev_is_empty and ev_path is not None:
            lines.append(f"set fmri(custom{index}) \"{ev_path}\"")
        for other_index in range(1, evs_orig + 1):
            lines.append(f"set fmri(ortho{index}.{other_index}) 0")

    for contrast_index, contrast in enumerate(model.contrasts, start=1):
        lines.extend(
            [
                f"set fmri(conpic_real.{contrast_index}) 1",
                f"set fmri(conname_real.{contrast_index}) \"{contrast.name}\"",
                f"set fmri(conpic_orig.{contrast_index}) 1",
                f"set fmri(conname_orig.{contrast_index}) \"{contrast.name}\"",
            ]
        )
        for ev_index in range(1, evs_orig + 1):
            weight = contrast.weights[ev_index - 1] if ev_index - 1 < len(contrast.weights) else 0.0
            lines.append(f"set fmri(con_orig{contrast_index}.{ev_index}) {weight}")

        real_weights: list[float] = []
        for ev_index, ev_name in enumerate(model.ev_order, start=1):
            real_weights.append(contrast.weights[ev_index - 1] if ev_index - 1 < len(contrast.weights) else 0.0)
            if ev_name in derivative_on and ev_name not in empty_evs:
                real_weights.append(0.0)
        for ev_index, weight in enumerate(real_weights, start=1):
            lines.append(f"set fmri(con_real{contrast_index}.{ev_index}) {weight}")

    lines.append("set fmri(conmask_zerothresh_yn) 0")
    for contrast_index in range(1, ncon + 1):
        lines.append(f"set fmri(conmask{contrast_index}_{contrast_index}) 0")

    lines.extend(
        [
            "set fmri(alternative_mask) \"\"",
            "set fmri(init_initial_highres) \"\"",
            "set fmri(init_highres) \"\"",
            "set fmri(init_standard) \"\"",
            f"set fmri(overwrite_yn) {1 if settings.get('overwrite_design') else 0}",
            "",
        ]
    )
    return "\n".join(lines)


def validate_ev_files(ev_paths_by_name: dict[str, Path], *, ev_order: list[str]) -> list[str]:
    problems: list[str] = []
    for ev_name in ev_order:
        path = ev_paths_by_name.get(ev_name)
        if path is None:
            problems.append(f"Missing EV file for {ev_name}.")
            continue
        tokens = _read_first_data_line(path)
        if not tokens or len(tokens) < 3:
            problems.append(f"EV file is not 3-column: {path}")
            continue
        try:
            float(tokens[0])
            float(tokens[1])
            float(tokens[2])
        except Exception:
            problems.append(f"EV file contains non-numeric values: {path}")
    return problems


def _setting(settings: dict[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        value = settings.get(name)
        if value is not None:
            return value
    return default


def validate_confounds_rows(confounds_path: Path | None, *, npts: int) -> tuple[bool, str | None]:
    if confounds_path is None:
        return True, None
    if not confounds_path.exists():
        return False, f"Confounds file is missing: {confounds_path}"
    row_count = 0
    try:
        with confounds_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row_count += 1
    except Exception as exc:
        return False, f"Could not read confounds file {confounds_path}: {exc}"
    if row_count != npts:
        return False, f"Confounds rows ({row_count}) do not match npts ({npts})."
    return True, None


def feat_results_complete(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    if not (out_dir / "design.fsf").exists():
        return False
    if (out_dir / "filtered_func_data.nii.gz").exists():
        return True
    stats_dir = out_dir / "stats"
    if stats_dir.exists():
        for filename in ("pe1.nii.gz", "cope1.nii.gz", "zstat1.nii.gz"):
            if (stats_dir / filename).exists():
                return True
    return (out_dir / "report.html").exists()


def preflight_feat_model(fsf_path: Path) -> tuple[bool, str]:
    feat_model_path = shutil.which("feat_model")
    if not feat_model_path:
        return True, "feat_model not found; skipping preflight."

    base_name = fsf_path.with_suffix("").name
    try:
        completed = subprocess.run(
            [feat_model_path, base_name],
            cwd=fsf_path.parent,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return False, f"feat_model preflight failed: {exc}"

    if completed.returncode == 0:
        return True, "feat_model ok"

    tail = "\n".join((completed.stderr or completed.stdout or "").strip().splitlines()[-12:])
    return False, f"feat_model exited with code {completed.returncode}.\n{tail}"


def write_text_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _read_first_data_line(path: Path) -> list[str] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped.split()
    except Exception:
        return None
    return None


def _feat_header_version() -> str:
    try:
        fsl_dir = os.environ.get("FSLDIR")
        if fsl_dir:
            version_file = Path(fsl_dir) / "etc" / "fslversion"
            if version_file.exists():
                match = re.search(r"(\d+\.\d+)", version_file.read_text(encoding="utf-8"))
                if match:
                    return f"{float(match.group(1)):.2f}"
    except Exception:
        pass
    return "6.00"
