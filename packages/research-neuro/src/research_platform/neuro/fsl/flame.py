"""Local FSL FLAME1 helpers for reusable LOSO ROI workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import shutil
import subprocess


CommandRunner = Callable[[Sequence[str]], Any]


@dataclass(frozen=True)
class OneSampleDesignPaths:
    """Paths for a one-sample group-mean FLAME1 design."""

    design_mat: Path
    design_con: Path
    design_grp: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "design_mat": str(self.design_mat),
            "design_con": str(self.design_con),
            "design_grp": str(self.design_grp),
        }


@dataclass(frozen=True)
class FixedEffectsDesignPaths:
    """Paths for a minimal fixed-effects design."""

    design_mat: Path
    design_con: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "design_mat": str(self.design_mat),
            "design_con": str(self.design_con),
        }


@dataclass(frozen=True)
class Flame1CommandPlan:
    """A minimal one-sample FLAME1 command plan."""

    cope_inputs: tuple[Path, ...]
    varcope_inputs: tuple[Path, ...]
    mask_path: Path
    work_dir: Path
    output_zstat_path: Path
    merged_cope_path: Path
    merged_varcope_path: Path
    flame_output_dir: Path
    design_paths: OneSampleDesignPaths
    commands: tuple[tuple[str, ...], ...]
    environment: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cope_inputs": [str(path) for path in self.cope_inputs],
            "varcope_inputs": [str(path) for path in self.varcope_inputs],
            "mask_path": str(self.mask_path),
            "work_dir": str(self.work_dir),
            "output_zstat_path": str(self.output_zstat_path),
            "merged_cope_path": str(self.merged_cope_path),
            "merged_varcope_path": str(self.merged_varcope_path),
            "flame_output_dir": str(self.flame_output_dir),
            "design_paths": self.design_paths.to_dict(),
            "commands": [list(command) for command in self.commands],
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class FixedEffectsCommandPlan:
    """A shell-safe plan for FSL fixed-effects merging and ``flameo``."""

    cope_inputs: tuple[Path, ...]
    varcope_inputs: tuple[Path, ...]
    mask_path: Path
    work_dir: Path
    output_dir: Path
    merged_cope_path: Path
    merged_varcope_path: Path
    design_file: Path
    t_contrast_file: Path | None
    commands: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cope_inputs": [str(path) for path in self.cope_inputs],
            "varcope_inputs": [str(path) for path in self.varcope_inputs],
            "mask_path": str(self.mask_path),
            "work_dir": str(self.work_dir),
            "output_dir": str(self.output_dir),
            "merged_cope_path": str(self.merged_cope_path),
            "merged_varcope_path": str(self.merged_varcope_path),
            "design_file": str(self.design_file),
            "t_contrast_file": str(self.t_contrast_file) if self.t_contrast_file is not None else None,
            "commands": [list(command) for command in self.commands],
        }


def one_sample_design_paths(output_dir: str | Path) -> OneSampleDesignPaths:
    """Return conventional design paths under ``output_dir``."""

    root = Path(output_dir)
    return OneSampleDesignPaths(
        design_mat=root / "design.mat",
        design_con=root / "design.con",
        design_grp=root / "design.grp",
    )


def fixed_effects_design_paths(output_dir: str | Path) -> FixedEffectsDesignPaths:
    """Return conventional fixed-effects design paths under ``output_dir``."""

    root = Path(output_dir)
    return FixedEffectsDesignPaths(
        design_mat=root / "design.mat",
        design_con=root / "design.con",
    )


def write_one_sample_group_mean_design(output_dir: str | Path, *, n_subjects: int) -> OneSampleDesignPaths:
    """Write one-sample group-mean design files for FLAME1."""

    n = int(n_subjects)
    if n <= 0:
        raise ValueError("n_subjects must be greater than zero.")

    paths = one_sample_design_paths(output_dir)
    paths.design_mat.parent.mkdir(parents=True, exist_ok=True)
    paths.design_mat.write_text(_design_mat_text(n), encoding="utf-8")
    paths.design_con.write_text(_design_con_text(), encoding="utf-8")
    paths.design_grp.write_text(_design_grp_text(n), encoding="utf-8")
    return paths


def write_fixed_effects_design(output_dir: str | Path, *, n_inputs: int) -> FixedEffectsDesignPaths:
    """Write a minimal one-EV fixed-effects design for ``flameo --runmode=fe``."""

    n = int(n_inputs)
    if n <= 0:
        raise ValueError("n_inputs must be greater than zero.")

    paths = fixed_effects_design_paths(output_dir)
    paths.design_mat.parent.mkdir(parents=True, exist_ok=True)
    paths.design_mat.write_text(_design_mat_text(n), encoding="utf-8")
    paths.design_con.write_text(_design_con_text(), encoding="utf-8")
    return paths


def build_flame1_command_plan(
    *,
    cope_inputs: Sequence[str | Path],
    varcope_inputs: Sequence[str | Path],
    mask_path: str | Path,
    work_dir: str | Path,
    output_zstat_path: str | Path,
    environment: Mapping[str, Any] | None = None,
) -> Flame1CommandPlan:
    """Build the minimal FSL command plan for one-sample FLAME1."""

    copes = tuple(Path(path).resolve() for path in cope_inputs)
    varcopes = tuple(Path(path).resolve() for path in varcope_inputs)
    if not copes:
        raise ValueError("At least one COPE input is required for FLAME1.")
    if len(copes) != len(varcopes):
        raise ValueError("COPE and VARCOPE input counts must match.")

    root = Path(work_dir).resolve()
    merged_cope = root / "merged_cope.nii.gz"
    merged_varcope = root / "merged_varcope.nii.gz"
    flame_out = root / "flame1"
    design_paths = one_sample_design_paths(root)
    env = {str(name): str(value) for name, value in dict(environment or {}).items()}
    commands: tuple[tuple[str, ...], ...] = (
        ("fslmerge", "-t", str(merged_cope), *(str(path) for path in copes)),
        ("fslmerge", "-t", str(merged_varcope), *(str(path) for path in varcopes)),
        (
            "flameo",
            f"--cope={merged_cope}",
            f"--vc={merged_varcope}",
            f"--mask={Path(mask_path).resolve()}",
            f"--dm={design_paths.design_mat}",
            f"--tc={design_paths.design_con}",
            f"--cs={design_paths.design_grp}",
            "--runmode=flame1",
            f"--ld={flame_out}",
        ),
    )
    return Flame1CommandPlan(
        cope_inputs=copes,
        varcope_inputs=varcopes,
        mask_path=Path(mask_path).resolve(),
        work_dir=root,
        output_zstat_path=Path(output_zstat_path).resolve(),
        merged_cope_path=merged_cope,
        merged_varcope_path=merged_varcope,
        flame_output_dir=flame_out,
        design_paths=design_paths,
        commands=commands,
        environment=env,
    )


def build_fixed_effects_command_plan(
    *,
    cope_inputs: Sequence[str | Path],
    varcope_inputs: Sequence[str | Path],
    mask_path: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    design_file: str | Path,
    t_contrast_file: str | Path | None = None,
    merged_cope_path: str | Path | None = None,
    merged_varcope_path: str | Path | None = None,
) -> FixedEffectsCommandPlan:
    """Build a plan-only FSL fixed-effects command vector sequence.

    The returned commands are argv tuples only. This helper does not run FSL,
    create directories, or write design files.
    """

    copes = tuple(Path(path).resolve() for path in cope_inputs)
    varcopes = tuple(Path(path).resolve() for path in varcope_inputs)
    if not copes:
        raise ValueError("At least one COPE input is required for fixed-effects planning.")
    if len(copes) != len(varcopes):
        raise ValueError("COPE and VARCOPE input counts must match.")

    root = Path(work_dir).resolve()
    merged_cope = Path(merged_cope_path).resolve() if merged_cope_path is not None else root / "merged_cope.nii.gz"
    merged_varcope = (
        Path(merged_varcope_path).resolve()
        if merged_varcope_path is not None
        else root / "merged_varcope.nii.gz"
    )
    design = Path(design_file).resolve()
    t_contrast = Path(t_contrast_file).resolve() if t_contrast_file is not None else None
    out_dir = Path(output_dir).resolve()
    flameo_command: tuple[str, ...]
    common_flameo_args = (
        "flameo",
        f"--cope={merged_cope}",
        f"--vc={merged_varcope}",
        f"--mask={Path(mask_path).resolve()}",
        f"--dm={design}",
    )
    if t_contrast is None:
        flameo_command = (
            *common_flameo_args,
            "--runmode=fe",
            f"--ld={out_dir}",
        )
    else:
        flameo_command = (
            *common_flameo_args,
            f"--tc={t_contrast}",
            "--runmode=fe",
            f"--ld={out_dir}",
        )
    commands: tuple[tuple[str, ...], ...] = (
        ("fslmerge", "-t", str(merged_cope), *(str(path) for path in copes)),
        ("fslmerge", "-t", str(merged_varcope), *(str(path) for path in varcopes)),
        flameo_command,
    )
    return FixedEffectsCommandPlan(
        cope_inputs=copes,
        varcope_inputs=varcopes,
        mask_path=Path(mask_path).resolve(),
        work_dir=root,
        output_dir=out_dir,
        merged_cope_path=merged_cope,
        merged_varcope_path=merged_varcope,
        design_file=design,
        t_contrast_file=t_contrast,
        commands=commands,
    )


def execute_flame1_command_plan(
    plan: Flame1CommandPlan,
    *,
    runner: CommandRunner | None = None,
) -> Path:
    """Execute a FLAME1 command plan and materialize the configured zstat path."""

    plan.work_dir.mkdir(parents=True, exist_ok=True)
    write_one_sample_group_mean_design(plan.work_dir, n_subjects=len(plan.cope_inputs))
    command_runner = runner or _subprocess_runner(plan.environment)
    for command in plan.commands:
        command_runner(command)

    produced_zstat = _find_flame1_zstat(plan.flame_output_dir)
    if produced_zstat is None:
        expected = ", ".join(str(path) for path in _flame1_zstat_candidates(plan.flame_output_dir))
        raise FileNotFoundError(f"FLAME1 did not produce expected zstat map at any supported location: {expected}")

    plan.output_zstat_path.parent.mkdir(parents=True, exist_ok=True)
    if produced_zstat.resolve() != plan.output_zstat_path:
        shutil.copyfile(produced_zstat, plan.output_zstat_path)
    return plan.output_zstat_path


def _find_flame1_zstat(flame_output_dir: Path) -> Path | None:
    for candidate in _flame1_zstat_candidates(flame_output_dir):
        if candidate.exists():
            return candidate
    return None


def _flame1_zstat_candidates(flame_output_dir: Path) -> tuple[Path, ...]:
    return (
        flame_output_dir / "zstat1.nii.gz",
        flame_output_dir / "stats" / "zstat1.nii.gz",
    )


def _subprocess_runner(environment: Mapping[str, str]) -> CommandRunner:
    def _run(command: Sequence[str]) -> None:
        subprocess.run(
            [str(part) for part in command],
            check=True,
            env={**os.environ, **dict(environment)} if environment else None,
        )

    return _run


def _design_mat_text(n_subjects: int) -> str:
    rows = "\n".join("1" for _ in range(n_subjects))
    return "\n".join(
        [
            "/NumWaves 1",
            f"/NumPoints {n_subjects}",
            "/PPheights 1",
            "",
            "/Matrix",
            rows,
            "",
        ]
    )


def _design_con_text() -> str:
    return "\n".join(
        [
            "/NumWaves 1",
            "/NumContrasts 1",
            "/PPheights 1",
            "/RequiredEffect 1",
            "",
            "/Matrix",
            "1",
            "",
        ]
    )


def _design_grp_text(n_subjects: int) -> str:
    rows = "\n".join("1" for _ in range(n_subjects))
    return "\n".join(
        [
            "/NumWaves 1",
            f"/NumPoints {n_subjects}",
            "",
            "/Matrix",
            rows,
            "",
        ]
    )
