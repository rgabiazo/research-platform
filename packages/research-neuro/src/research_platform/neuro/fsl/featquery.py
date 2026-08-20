"""Local FSL featquery helpers for reusable ROI extraction workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import os
import re
import subprocess


CommandRunner = Callable[[Sequence[str]], Any]

SUPPORTED_FEATQUERY_METRICS = frozenset({"percent_signal_change", "mean_cope", "roi_voxel_count"})

_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9])")


@dataclass(frozen=True)
class FeatqueryCommandPlan:
    """A shell-safe local featquery command plan."""

    feat_dir: Path
    value_image: str
    output_name: str
    roi_mask_path: Path
    output_dir: Path
    report_path: Path
    metrics: tuple[str, ...]
    include_percent_signal_change: bool
    commands: tuple[tuple[str, ...], ...]
    environment: Mapping[str, str]

    @property
    def command(self) -> tuple[str, ...]:
        """Return the single featquery command argument vector."""

        return self.commands[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feat_dir": str(self.feat_dir),
            "value_image": self.value_image,
            "output_name": self.output_name,
            "roi_mask_path": str(self.roi_mask_path),
            "output_dir": str(self.output_dir),
            "report_path": str(self.report_path),
            "metrics": list(self.metrics),
            "include_percent_signal_change": self.include_percent_signal_change,
            "commands": [list(command) for command in self.commands],
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class FeatqueryReport:
    """Parsed values and QC state from one featquery report.txt."""

    report_path: Path | None
    stats_image: str | None = None
    mean_psc: float | None = None
    median_psc: float | None = None
    mean_cope: float | None = None
    median_cope: float | None = None
    max_cope: float | None = None
    roi_voxel_count: int | None = None
    max_voxel_coordinate: tuple[int, int, int] | None = None
    max_mm_coordinate: tuple[float, float, float] | None = None
    warnings: tuple[str, ...] = ()
    qc_flags: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return "ambiguous_report_values" not in self.qc_flags and "missing_report_values" not in self.qc_flags

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "report_path": str(self.report_path) if self.report_path is not None else None,
                "stats_image": self.stats_image,
                "mean_psc": self.mean_psc,
                "median_psc": self.median_psc,
                "mean_cope": self.mean_cope,
                "median_cope": self.median_cope,
                "max_cope": self.max_cope,
                "roi_voxel_count": self.roi_voxel_count,
                "max_voxel_coordinate": self.max_voxel_coordinate,
                "max_mm_coordinate": self.max_mm_coordinate,
                "usable": self.usable,
                "qc_flags": list(self.qc_flags),
                "warnings": list(self.warnings),
            }.items()
            if value is not None
        }


def build_featquery_command_plan(
    *,
    feat_dir: str | Path,
    roi_mask_path: str | Path,
    output_name: str,
    value_image: str = "stats/cope1",
    metrics: Sequence[str] = ("mean_cope",),
    include_percent_signal_change: Any | None = None,
    environment: Mapping[str, Any] | None = None,
) -> FeatqueryCommandPlan:
    """Build a featquery command without requiring FSL to be installed."""

    requested = _normalize_metrics(metrics)
    include_psc = _normalize_percent_signal_change_request(
        requested,
        include_percent_signal_change=include_percent_signal_change,
    )
    value = str(value_image).strip()
    if not value:
        raise ValueError("featquery value_image must be defined.")
    if Path(value).is_absolute():
        raise ValueError("featquery value_image must be relative to the FEAT directory.")
    name = str(output_name).strip()
    if not _SAFE_OUTPUT_NAME.fullmatch(name):
        raise ValueError("featquery output_name must be a safe single path segment.")

    feat = Path(feat_dir).resolve()
    roi = Path(roi_mask_path).resolve()
    output_dir = feat / name
    command_parts = ["featquery", "1", str(feat), "1", value, name]
    if include_psc:
        command_parts.append("-p")
    command_parts.append(str(roi))
    command = tuple(command_parts)

    env = {str(key): str(value) for key, value in dict(environment or {}).items()}
    return FeatqueryCommandPlan(
        feat_dir=feat,
        value_image=value,
        output_name=name,
        roi_mask_path=roi,
        output_dir=output_dir,
        report_path=output_dir / "report.txt",
        metrics=requested,
        include_percent_signal_change=include_psc,
        commands=(command,),
        environment=env,
    )


def execute_featquery_command_plan(
    plan: FeatqueryCommandPlan,
    *,
    runner: CommandRunner | None = None,
) -> Path:
    """Execute a featquery command plan and return the expected report path."""

    command_runner = runner or _subprocess_runner(plan.environment)
    for command in plan.commands:
        command_runner(command)
    if not plan.report_path.exists():
        raise FileNotFoundError(f"featquery did not produce expected report: {plan.report_path}")
    return plan.report_path


def parse_featquery_report(
    report_path: str | Path,
    *,
    required_metrics: Sequence[str] = (),
) -> FeatqueryReport:
    """Parse an FSL featquery report.txt file."""

    path = Path(report_path)
    return parse_featquery_report_text(
        path.read_text(encoding="utf-8"),
        report_path=path,
        required_metrics=required_metrics,
    )


def parse_featquery_report_text(
    text: str,
    *,
    report_path: str | Path | None = None,
    required_metrics: Sequence[str] = (),
) -> FeatqueryReport:
    """Parse featquery report text while preserving ambiguous or missing values."""

    requested = _normalize_metrics(required_metrics) if required_metrics else ()
    psc_result_rows = "percent_signal_change" in requested and "mean_cope" not in requested
    state: dict[str, Any] = {
        "stats_image": None,
        "mean_psc": None,
        "median_psc": None,
        "mean_cope": None,
        "median_cope": None,
        "max_cope": None,
        "roi_voxel_count": None,
        "max_voxel_coordinate": None,
        "max_mm_coordinate": None,
    }
    warnings: list[str] = []
    qc_flags: set[str] = set()
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        warnings.append("featquery report is empty.")
        qc_flags.add("missing_report_values")

    for line in lines:
        label, value_text = _split_labeled_value(line)
        field = _field_for_label(label)
        if field is None:
            continue
        numbers = _numbers(value_text)
        if len(numbers) != 1:
            warnings.append(f"Could not unambiguously parse {field} from report line: {line}")
            qc_flags.add("ambiguous_report_values")
            continue
        _record_value(state, field, numbers[0], warnings=warnings, qc_flags=qc_flags)

    _parse_featquery_result_rows(lines, state, warnings=warnings, qc_flags=qc_flags, percent_signal_change=psc_result_rows)
    _parse_simple_tables(lines, state, warnings=warnings, qc_flags=qc_flags)

    missing_fields = []
    if "percent_signal_change" in requested and state["mean_psc"] is None:
        missing_fields.append("mean_psc")
    if "mean_cope" in requested and state["mean_cope"] is None:
        missing_fields.append("mean_cope")
    if "roi_voxel_count" in requested and state["roi_voxel_count"] is None:
        missing_fields.append("roi_voxel_count")
    if missing_fields:
        warnings.append(f"Missing requested featquery report value(s): {', '.join(missing_fields)}.")
        qc_flags.add("missing_report_values")

    if not any(state[field] is not None for field in _NUMERIC_REPORT_FIELDS):
        warnings.append("No recognized numeric featquery values were found.")
        qc_flags.add("missing_report_values")

    return FeatqueryReport(
        report_path=Path(report_path) if report_path is not None else None,
        stats_image=state["stats_image"],
        mean_psc=_optional_float(state["mean_psc"]),
        median_psc=_optional_float(state["median_psc"]),
        mean_cope=_optional_float(state["mean_cope"]),
        median_cope=_optional_float(state["median_cope"]),
        max_cope=_optional_float(state["max_cope"]),
        roi_voxel_count=_optional_int(state["roi_voxel_count"]),
        max_voxel_coordinate=state["max_voxel_coordinate"],
        max_mm_coordinate=state["max_mm_coordinate"],
        warnings=tuple(warnings),
        qc_flags=tuple(sorted(qc_flags)) or ("pass",),
    )


def _normalize_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(metric).strip() for metric in metrics if str(metric).strip())
    if not requested:
        raise ValueError("At least one featquery metric must be requested.")
    unknown = sorted(set(requested) - SUPPORTED_FEATQUERY_METRICS)
    if unknown:
        raise ValueError(f"Unsupported featquery metric(s): {', '.join(unknown)}.")
    return requested


def _normalize_percent_signal_change_request(
    metrics: Sequence[str],
    *,
    include_percent_signal_change: Any | None,
) -> bool:
    metric_requested = "percent_signal_change" in metrics
    raw_requested = "mean_cope" in metrics
    explicit = _optional_bool(include_percent_signal_change, label="include_percent_signal_change")
    include_psc = metric_requested if explicit is None else explicit
    if metric_requested and explicit is False:
        raise ValueError("percent_signal_change requires FSL featquery -p; remove include_percent_signal_change: false.")
    if include_psc and raw_requested:
        raise ValueError(
            "FSL featquery raw COPE and percent signal change require separate extraction targets because PSC uses the -p conversion flag."
        )
    if include_psc and not metric_requested:
        raise ValueError("include_percent_signal_change requires the percent_signal_change metric.")
    return include_psc


def _optional_bool(value: Any | None, *, label: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} must be a boolean value.")


def _subprocess_runner(environment: Mapping[str, str]) -> CommandRunner:
    def _run(command: Sequence[str]) -> None:
        subprocess.run(
            [str(part) for part in command],
            check=True,
            env={**os.environ, **dict(environment)} if environment else None,
        )

    return _run


def _split_labeled_value(line: str) -> tuple[str, str]:
    for separator in (":", "="):
        if separator in line:
            label, value = line.split(separator, 1)
            return label.strip(), value.strip()
    tab_parts = [part.strip() for part in line.split("\t") if part.strip()]
    if len(tab_parts) >= 2:
        return " ".join(tab_parts[:-1]), tab_parts[-1]
    space_parts = [part.strip() for part in re.split(r"\s{2,}", line) if part.strip()]
    if len(space_parts) >= 2:
        return " ".join(space_parts[:-1]), space_parts[-1]
    match = list(_NUMBER.finditer(line))
    if not match:
        return line, ""
    first = match[0]
    return line[: first.start()].strip(), line[first.start() :].strip()


def _field_for_label(label: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9%]+", " ", label.lower()).strip()
    if not normalized:
        return None
    words = normalized.split()
    is_mean = "mean" in words
    is_median = "median" in words
    is_max = "max" in words or "maximum" in words
    is_percent = "%" in normalized or "percent" in normalized or "psc" in normalized
    is_signal_change = "signal" in normalized and "change" in normalized
    if is_percent and (is_signal_change or "psc" in normalized):
        return "median_psc" if is_median else "mean_psc" if is_mean or not is_median else None
    if "cope" in normalized:
        if is_max:
            return "max_cope"
        if is_median:
            return "median_cope"
        if is_mean:
            return "mean_cope"
    if normalized in {"mean", "mean value", "mean intensity"} or normalized.startswith("mean "):
        return "mean_cope"
    if normalized in {"median", "median value", "median intensity"} or normalized.startswith("median "):
        return "median_cope"
    if normalized in {"max", "maximum", "max value", "maximum value", "max intensity"} or normalized.startswith(("max ", "maximum ")):
        return "max_cope"
    if "voxel" in normalized or "voxels" in normalized or "nvox" in normalized:
        return "roi_voxel_count"
    return None


def _numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in _NUMBER.finditer(text)]


def _record_value(
    state: dict[str, Any],
    field: str,
    value: float,
    *,
    warnings: list[str],
    qc_flags: set[str],
) -> None:
    normalized: float | int = int(value) if field == "roi_voxel_count" and float(value).is_integer() else value
    existing = state.get(field)
    if existing is not None and not math.isclose(float(existing), float(normalized), rel_tol=0, abs_tol=1e-12):
        warnings.append(f"Multiple conflicting values were found for {field}.")
        qc_flags.add("ambiguous_report_values")
        return
    state[field] = normalized


_NUMERIC_REPORT_FIELDS = frozenset(
    {
        "mean_psc",
        "median_psc",
        "mean_cope",
        "median_cope",
        "max_cope",
        "roi_voxel_count",
    }
)


def _parse_featquery_result_rows(
    lines: Sequence[str],
    state: dict[str, Any],
    *,
    warnings: list[str],
    qc_flags: set[str],
    percent_signal_change: bool,
) -> None:
    for line in lines:
        row = _featquery_result_row(line)
        if row is None:
            continue
        stats_image, voxel_count, values = row
        _record_text_value(state, "stats_image", stats_image, warnings=warnings, qc_flags=qc_flags)
        _record_value(state, "roi_voxel_count", float(voxel_count), warnings=warnings, qc_flags=qc_flags)
        if len(values) >= 3:
            _record_value(
                state,
                "mean_psc" if percent_signal_change else "mean_cope",
                values[2],
                warnings=warnings,
                qc_flags=qc_flags,
            )
        if len(values) >= 4:
            _record_value(
                state,
                "median_psc" if percent_signal_change else "median_cope",
                values[3],
                warnings=warnings,
                qc_flags=qc_flags,
            )
        if not percent_signal_change and len(values) >= 6:
            _record_value(state, "max_cope", values[5], warnings=warnings, qc_flags=qc_flags)
        if len(values) >= 12:
            voxel_coordinate = values[-6:-3]
            mm_coordinate = values[-3:]
            if all(float(value).is_integer() for value in voxel_coordinate):
                _record_coordinate(
                    state,
                    "max_voxel_coordinate",
                    tuple(int(value) for value in voxel_coordinate),
                    warnings=warnings,
                    qc_flags=qc_flags,
                )
            _record_coordinate(
                state,
                "max_mm_coordinate",
                tuple(float(value) for value in mm_coordinate),
                warnings=warnings,
                qc_flags=qc_flags,
            )


def _featquery_result_row(line: str) -> tuple[str, int, list[float]] | None:
    tokens = line.split()
    if len(tokens) < 8:
        return None

    stats_index = None
    if _single_number(tokens[0]) is not None and _single_number(tokens[1]) is None:
        stats_index = 1
    elif _single_number(tokens[0]) is None:
        stats_index = 0
    if stats_index is None or len(tokens) <= stats_index + 2:
        return None

    voxel_count = _integer_token(tokens[stats_index + 1])
    if voxel_count is None:
        return None
    values = []
    for token in tokens[stats_index + 2 :]:
        value = _single_number(token)
        if value is None:
            break
        values.append(value)
    if len(values) < 6:
        return None
    return tokens[stats_index], voxel_count, values


def _integer_token(text: str) -> int | None:
    value = _single_number(text)
    if value is None or not float(value).is_integer():
        return None
    return int(value)


def _record_text_value(
    state: dict[str, Any],
    field: str,
    value: str,
    *,
    warnings: list[str],
    qc_flags: set[str],
) -> None:
    existing = state.get(field)
    if existing is not None and existing != value:
        warnings.append(f"Multiple conflicting values were found for {field}.")
        qc_flags.add("ambiguous_report_values")
        return
    state[field] = value


def _record_coordinate(
    state: dict[str, Any],
    field: str,
    value: tuple[int, int, int] | tuple[float, float, float],
    *,
    warnings: list[str],
    qc_flags: set[str],
) -> None:
    existing = state.get(field)
    if existing is not None and tuple(existing) != value:
        warnings.append(f"Multiple conflicting values were found for {field}.")
        qc_flags.add("ambiguous_report_values")
        return
    state[field] = value


def _parse_simple_tables(
    lines: Sequence[str],
    state: dict[str, Any],
    *,
    warnings: list[str],
    qc_flags: set[str],
) -> None:
    for index, line in enumerate(lines[:-1]):
        header = _table_tokens(line)
        values = _table_tokens(lines[index + 1])
        if len(header) < 2 or len(values) < 2 or len(values) < len(header):
            continue
        numeric_values = [_single_number(token) for token in values[: len(header)]]
        if not any(value is not None for value in numeric_values):
            continue
        for column, value in zip(header, numeric_values):
            if value is None:
                continue
            field = _field_for_label(column)
            if field is not None:
                _record_value(state, field, value, warnings=warnings, qc_flags=qc_flags)


def _table_tokens(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"\t+|\s{2,}", line.strip()) if part.strip()]


def _single_number(text: str) -> float | None:
    values = _numbers(text)
    return values[0] if len(values) == 1 else None


def _optional_float(value: float | int | None) -> float | None:
    return float(value) if value is not None else None


def _optional_int(value: float | int | None) -> int | None:
    return int(value) if value is not None else None
