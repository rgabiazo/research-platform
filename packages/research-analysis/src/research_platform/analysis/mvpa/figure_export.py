from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from ._path_safety import configured_path_is_unsafe, published_value_contains_local_path_reference


SUPPORTED_FIGURE_KINDS = ("strip_mean_ci", "category_distribution_mean_ci")
SUPPORTED_OUTPUT_FORMATS = ("svg", "pdf", "png")
DEFAULT_INPUT_RELATIVE_PATH = ".research-platform/mvpa/reports/{table_set}/{filename}"
DEFAULT_OUTPUT_RELATIVE_PATH = ".research-platform/mvpa/reports/{figure_set}/figures"
DEFAULT_DISTANCE_FILENAME = "{filename_prefix}_desc-SubjectLevelCrossnobisDistances_mvpa.tsv"
PLOT_DATA_COLUMNS = (
    "figure_id",
    "kind",
    "participant_id",
    "category",
    "category_label",
    "x_position",
    "jittered_x",
    "crossnobis",
    "analysis_variant",
    "phase_id",
    "contrast_id",
    "roi_label",
    "pooled_row_count",
    "pooled_roi_count",
)
SUMMARY_COLUMNS = (
    "figure_id",
    "kind",
    "category",
    "category_label",
    "n",
    "participant_count",
    "mean",
    "sd",
    "sem",
    "ci_level",
    "ci_low",
    "ci_high",
)


def validate_mvpa_figure_export_document(document: Mapping[str, Any] | Any) -> list[str]:
    if not isinstance(document, Mapping):
        return ["MVPA figure export config must contain a mapping."]
    payload = _payload(document)
    errors: list[str] = []
    figure_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("figure_set"))
    if figure_set is None:
        errors.append("mvpa_figure_export.name must be defined.")
    elif not _safe_label(figure_set):
        errors.append("mvpa_figure_export.name must be a safe label.")

    input_config = _input_config(payload)
    if input_config and not isinstance(input_config, Mapping):
        errors.append("mvpa_figure_export.input must be a mapping when defined.")
    elif isinstance(input_config, Mapping):
        path = _optional_text(input_config.get("path") or input_config.get("relative_path"))
        if path is not None and configured_path_is_unsafe(path):
            errors.append("mvpa_figure_export.input.path must be relative and stay under its root_ref.")
        root_ref = _optional_text(input_config.get("root_ref"))
        if root_ref is not None and not _safe_label(root_ref):
            errors.append("mvpa_figure_export.input.root_ref must be a safe label.")

    outputs = payload.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        errors.append("mvpa_figure_export.outputs must be a mapping when defined.")
    elif isinstance(outputs, Mapping):
        path = _optional_text(outputs.get("path") or outputs.get("relative_path"))
        if path is not None and configured_path_is_unsafe(path):
            errors.append("mvpa_figure_export.outputs.path must be relative and stay under its root_ref.")
        root_ref = _optional_text(outputs.get("root_ref"))
        if root_ref is not None and not _safe_label(root_ref):
            errors.append("mvpa_figure_export.outputs.root_ref must be a safe label.")

    figures = payload.get("figures")
    if not isinstance(figures, Sequence) or isinstance(figures, (str, bytes, bytearray)) or not figures:
        errors.append("mvpa_figure_export.figures must define at least one figure mapping.")
    else:
        seen: set[str] = set()
        for index, figure in enumerate(figures, start=1):
            if not isinstance(figure, Mapping):
                errors.append(f"mvpa_figure_export.figures[{index}] must be a mapping.")
                continue
            figure_id = _optional_text(figure.get("figure_id") or figure.get("id"))
            if figure_id is None:
                errors.append(f"mvpa_figure_export.figures[{index}].figure_id must be defined.")
            elif not _safe_label(figure_id):
                errors.append(f"mvpa_figure_export.figures[{index}].figure_id must be a safe label.")
            elif figure_id in seen:
                errors.append(f"mvpa_figure_export.figures[{index}].figure_id duplicates {figure_id!r}.")
            if figure_id is not None:
                seen.add(figure_id)
            kind = _optional_text(figure.get("kind"))
            if kind not in SUPPORTED_FIGURE_KINDS:
                errors.append(
                    f"mvpa_figure_export.figures[{index}].kind must be one of: {', '.join(SUPPORTED_FIGURE_KINDS)}."
                )
            for key in ("x", "y"):
                value = _optional_text(figure.get(key))
                if value is None:
                    errors.append(f"mvpa_figure_export.figures[{index}].{key} must be defined.")
                elif not _safe_column(value):
                    errors.append(f"mvpa_figure_export.figures[{index}].{key} must be a safe column name.")
            output_basename = _optional_text(figure.get("output_basename"))
            if output_basename is None:
                errors.append(f"mvpa_figure_export.figures[{index}].output_basename must be defined.")
            elif configured_path_is_unsafe(output_basename) or "/" in output_basename:
                errors.append(f"mvpa_figure_export.figures[{index}].output_basename must be a filename stem.")
            formats = _output_formats(figure)
            unknown_formats = sorted(set(formats).difference(SUPPORTED_OUTPUT_FORMATS))
            if unknown_formats:
                errors.append(
                    f"mvpa_figure_export.figures[{index}].output_formats contains unsupported format(s): "
                    f"{', '.join(unknown_formats)}."
                )
            aggregate = figure.get("aggregate")
            if aggregate is not None and not isinstance(aggregate, Mapping):
                errors.append(f"mvpa_figure_export.figures[{index}].aggregate must be a mapping when defined.")
            elif isinstance(aggregate, Mapping):
                method = _optional_text(aggregate.get("method")) or "mean"
                if method != "mean":
                    errors.append(f"mvpa_figure_export.figures[{index}].aggregate.method must be mean.")
                if not _text_sequence(aggregate.get("group_by")):
                    errors.append(f"mvpa_figure_export.figures[{index}].aggregate.group_by must define columns.")

    return errors


def plan_or_execute_mvpa_figure_export(
    document: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    root_refs: Mapping[str, str | Path],
    execute: bool = False,
) -> dict[str, Any]:
    payload = _payload(document)
    config_errors = validate_mvpa_figure_export_document(document)
    figure_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("figure_set")) or "mvpa_figure_export"
    workspace = Path(workspace_root).resolve()
    roots = {str(key): Path(value).expanduser().resolve() for key, value in root_refs.items()}
    artifact_root = roots.get("artifact_root") or roots.get("artifacts_root") or workspace / "artifacts"
    input_path, input_relative_path, input_errors = _input_table_path(payload, figure_set=figure_set, roots=roots)
    output_root, output_relative_root, output_errors = _output_root(payload, figure_set=figure_set, roots=roots)
    figure_targets = {
        _figure_id(figure): _figure_targets(figure, output_root=output_root, output_relative_root=output_relative_root)
        for figure in _figure_mappings(payload)
    }
    errors: list[str] = [*config_errors, *input_errors, *output_errors]
    warnings: list[str] = []
    source_rows: list[dict[str, str]] = []
    input_columns: tuple[str, ...] = ()
    if not errors:
        if not input_path.is_file():
            errors.append(f"MVPA subject-level table is missing: {input_relative_path}.")
        else:
            source_rows, input_columns = _read_subject_table(input_path)

    figure_results: list[dict[str, Any]] = []
    if not errors:
        for figure in _figure_mappings(payload):
            result = _prepare_figure(
                figure,
                source_rows=source_rows,
                input_columns=input_columns,
                targets=figure_targets[_figure_id(figure)],
            )
            figure_results.append(result)
            warnings.extend(result["warnings"])
            errors.extend(result["errors"])
    else:
        for figure in _figure_mappings(payload):
            targets = figure_targets[_figure_id(figure)]
            figure_results.append(_empty_figure_result(figure, targets=targets))

    if execute and not errors:
        existing = [
            f"{figure_result['figure_id']}:{name}"
            for figure_result in figure_results
            for name, target in figure_result["outputs"].items()
            if Path(target["path"]).exists()
        ]
        if existing:
            errors.append(f"MVPA figure export refuses to overwrite existing output(s): {', '.join(sorted(existing))}.")

    if execute and not errors:
        try:
            plt = _load_matplotlib_pyplot()
        except ImportError as exc:
            errors.append(str(exc))
            plt = None
        if plt is not None:
            for figure_result in figure_results:
                render_warnings = _render_and_write_figure(
                    plt,
                    figure_result,
                    input_relative_path=input_relative_path,
                    execute=execute,
                )
                figure_result["layout_warnings"] = _unique(
                    [*figure_result.get("layout_warnings", []), *render_warnings]
                )
                figure_result["warnings"] = _unique([*figure_result["warnings"], *render_warnings])
                warnings.extend(render_warnings)
                if figure_result["layout_warnings"] and bool(_mapping(figure_result["config"]).get("fail_on_layout_warning")):
                    errors.append(f"Figure {figure_result['figure_id']} has layout warning(s).")

    valid = not errors
    return {
        "valid": valid,
        "executed": bool(execute and valid),
        "figure_set": figure_set,
        "input_table": {
            "relative_path": input_relative_path,
            "path": input_path.as_posix(),
            "exists": input_path.is_file(),
            "row_count": len(source_rows),
            "columns": list(input_columns),
        },
        "output_root": {
            "relative_path": output_relative_root,
            "path": output_root.as_posix(),
        },
        "figures": [
            _public_figure_result(result, executed=bool(execute and valid), input_relative_path=input_relative_path)
            for result in figure_results
        ],
        "figure_count": len(figure_results),
        "supported_figure_kinds": list(SUPPORTED_FIGURE_KINDS),
        "warnings": _unique(warnings),
        "errors": _unique(errors),
    }


def _payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = document.get("mvpa_figure_export") or document.get("mvpa_figure_set") or document
    return payload if isinstance(payload, Mapping) else {}


def _input_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config = payload.get("input") or payload.get("input_table") or {}
    return config if isinstance(config, Mapping) else {}


def _figure_mappings(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    figures = payload.get("figures")
    if not isinstance(figures, Sequence) or isinstance(figures, (str, bytes, bytearray)):
        return ()
    return tuple(figure for figure in figures if isinstance(figure, Mapping))


def _input_table_path(
    payload: Mapping[str, Any],
    *,
    figure_set: str,
    roots: Mapping[str, Path],
) -> tuple[Path, str, list[str]]:
    config = _input_config(payload)
    root_ref = _optional_text(config.get("root_ref")) or "artifact_root"
    table_set = _optional_text(config.get("table_set")) or figure_set
    filename_prefix = _optional_text(config.get("filename_prefix")) or _filename_prefix(payload)
    default_filename = DEFAULT_DISTANCE_FILENAME.format(filename_prefix=filename_prefix)
    filename = _optional_text(config.get("filename")) or default_filename
    relative_template = _optional_text(config.get("path") or config.get("relative_path")) or DEFAULT_INPUT_RELATIVE_PATH
    errors: list[str] = []
    if root_ref not in roots:
        errors.append(f"mvpa_figure_export.input.root_ref {root_ref!r} is not a known root_ref.")
        root = Path(".").resolve()
    else:
        root = roots[root_ref]
    if configured_path_is_unsafe(relative_template):
        errors.append("mvpa_figure_export.input.path must be relative and stay under its root_ref.")
        relative_path = DEFAULT_INPUT_RELATIVE_PATH.format(table_set=table_set, filename=filename)
    else:
        relative_path = _render_template(
            relative_template,
            figure_set=figure_set,
            table_set=table_set,
            filename=filename,
            filename_prefix=filename_prefix,
        )
    return (root / relative_path).resolve(), relative_path, errors


def _output_root(
    payload: Mapping[str, Any],
    *,
    figure_set: str,
    roots: Mapping[str, Path],
) -> tuple[Path, str, list[str]]:
    outputs = _mapping(payload.get("outputs"))
    root_ref = _optional_text(outputs.get("root_ref")) or "artifact_root"
    relative_template = _optional_text(outputs.get("path") or outputs.get("relative_path")) or DEFAULT_OUTPUT_RELATIVE_PATH
    errors: list[str] = []
    if root_ref not in roots:
        errors.append(f"mvpa_figure_export.outputs.root_ref {root_ref!r} is not a known root_ref.")
        root = Path(".").resolve()
    else:
        root = roots[root_ref]
    if configured_path_is_unsafe(relative_template):
        errors.append("mvpa_figure_export.outputs.path must be relative and stay under its root_ref.")
        relative_path = DEFAULT_OUTPUT_RELATIVE_PATH.format(figure_set=figure_set)
    else:
        relative_path = _render_template(relative_template, figure_set=figure_set)
    return (root / relative_path).resolve(), relative_path, errors


def _figure_targets(
    figure: Mapping[str, Any],
    *,
    output_root: Path,
    output_relative_root: str,
) -> dict[str, dict[str, Any]]:
    basename = _optional_text(figure.get("output_basename")) or _figure_id(figure)
    files: OrderedDict[str, str] = OrderedDict()
    for output_format in _output_formats(figure):
        files[f"figure_{output_format}"] = f"{basename}.{output_format}"
    files["plot_data_tsv"] = f"{basename}_plot-data.tsv"
    files["summary_tsv"] = f"{basename}_summary.tsv"
    files["manifest_json"] = f"{basename}_manifest.json"
    return {
        key: {
            "path": output_root / filename,
            "relative_path": f"{output_relative_root}/{filename}",
            "filename": filename,
        }
        for key, filename in files.items()
    }


def _read_subject_table(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
        return rows, tuple(reader.fieldnames or ())


def _prepare_figure(
    figure: Mapping[str, Any],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    input_columns: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    figure_id = _figure_id(figure)
    kind = str(figure["kind"])
    x_column = str(figure["x"])
    y_column = str(figure["y"])
    warnings: list[str] = []
    errors: list[str] = []
    required_columns = _required_columns(figure)
    missing_columns = sorted(required_columns.difference(input_columns))
    if missing_columns:
        errors.append(f"Figure {figure_id} input table is missing required column(s): {', '.join(missing_columns)}.")
        return _figure_result(
            figure,
            targets=targets,
            plot_rows=[],
            summary_rows=[],
            warnings=warnings,
            errors=errors,
            categories=[],
        )
    filtered_rows = _filter_rows(source_rows, _mapping(figure.get("filters")))
    if not filtered_rows:
        errors.append(f"Figure {figure_id} has no rows after filters.")
    figure_rows = (
        _aggregate_rows(filtered_rows, figure, errors=errors)
        if isinstance(figure.get("aggregate"), Mapping)
        else [dict(row) for row in filtered_rows]
    )
    categories = _categories(figure_rows, x_column=x_column, configured_order=_text_sequence(figure.get("order") or figure.get("category_order") or figure.get("roi_order")))
    label_mapping = _label_mapping(figure)
    plot_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rng = random.Random(int(_number_or_default(figure.get("random_seed"), 12345)))
    jitter_width = float(_number_or_default(figure.get("jitter_width"), 0.14))
    ci_level = float(_number_or_default(figure.get("ci_level"), 0.95))
    for category_index, category in enumerate(categories, start=1):
        category_rows = [row for row in figure_rows if str(row.get(x_column) or "") == category]
        values: list[float] = []
        participants: set[str] = set()
        for row in category_rows:
            participant_id = _optional_text(row.get("participant_id"))
            if participant_id is None:
                errors.append(f"Figure {figure_id} row has missing participant_id.")
                continue
            value = _finite_number(row.get(y_column), f"Figure {figure_id} {y_column}", errors)
            if math.isfinite(value):
                values.append(value)
                participants.add(participant_id)
                plot_rows.append(
                    _plot_row(
                        figure_id=figure_id,
                        kind=kind,
                        row=row,
                        category=category,
                        category_label=label_mapping.get(category, category),
                        category_index=category_index,
                        jitter_width=jitter_width,
                        rng=rng,
                        value=value,
                    )
                )
        if len(participants) < 2:
            errors.append(f"Figure {figure_id} category {category!r} must have at least two participants.")
            continue
        summary_rows.append(
            _summary_row(
                figure_id=figure_id,
                kind=kind,
                category=category,
                category_label=label_mapping.get(category, category),
                values=values,
                participant_count=len(participants),
                ci_level=ci_level,
            )
        )
    if published_value_contains_local_path_reference(plot_rows):
        errors.append(f"Figure {figure_id} plot-data rows contain an absolute local path.")
    return _figure_result(
        figure,
        targets=targets,
        plot_rows=plot_rows,
        summary_rows=summary_rows,
        warnings=warnings,
        errors=errors,
        categories=categories,
    )


def _required_columns(figure: Mapping[str, Any]) -> set[str]:
    columns = {"participant_id", str(figure.get("x") or ""), str(figure.get("y") or "")}
    columns.update(str(column) for column in _mapping(figure.get("filters")).keys())
    aggregate = figure.get("aggregate")
    if isinstance(aggregate, Mapping):
        columns.update(_text_sequence(aggregate.get("group_by")))
        columns.add(str(aggregate.get("value") or figure.get("y") or ""))
        across = _optional_text(aggregate.get("across"))
        if across:
            columns.add(across)
    return {column for column in columns if column}


def _filter_rows(rows: Sequence[Mapping[str, Any]], filters: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not filters:
        return list(rows)
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        keep = True
        for column, expected in filters.items():
            values = set(_text_sequence(expected))
            if not values:
                expected_text = _optional_text(expected)
                values = {expected_text} if expected_text is not None else set()
            if values and str(row.get(str(column)) or "") not in values:
                keep = False
                break
        if keep:
            selected.append(row)
    return selected


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], figure: Mapping[str, Any], *, errors: list[str]) -> list[dict[str, Any]]:
    aggregate = _mapping(figure.get("aggregate"))
    group_by = _text_sequence(aggregate.get("group_by"))
    value_column = _optional_text(aggregate.get("value")) or str(figure["y"])
    across_column = _optional_text(aggregate.get("across"))
    groups: OrderedDict[tuple[str, ...], list[Mapping[str, Any]]] = OrderedDict()
    for row in rows:
        key = tuple(str(row.get(column) or "") for column in group_by)
        groups.setdefault(key, []).append(row)
    aggregated: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        values = [_finite_number(row.get(value_column), f"Figure {_figure_id(figure)} {value_column}", errors) for row in group_rows]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            continue
        base = {column: key[index] for index, column in enumerate(group_by)}
        for column in ("participant_id", "analysis_variant", "phase_id", "contrast_id", "roi_label", str(figure["x"])):
            if column in base:
                continue
            column_values = sorted({str(row.get(column) or "") for row in group_rows if str(row.get(column) or "")})
            if len(column_values) == 1:
                base[column] = column_values[0]
        base[str(figure["y"])] = sum(values) / len(values)
        base["pooled_row_count"] = len(group_rows)
        if across_column:
            base["pooled_roi_count"] = len({str(row.get(across_column) or "") for row in group_rows if str(row.get(across_column) or "")})
        else:
            base["pooled_roi_count"] = ""
        aggregated.append(base)
    return aggregated


def _categories(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_column: str,
    configured_order: Sequence[str],
) -> list[str]:
    observed: OrderedDict[str, None] = OrderedDict()
    for row in rows:
        value = _optional_text(row.get(x_column))
        if value is not None:
            observed[value] = None
    if configured_order:
        return [category for category in configured_order if category in observed]
    return list(observed)


def _plot_row(
    *,
    figure_id: str,
    kind: str,
    row: Mapping[str, Any],
    category: str,
    category_label: str,
    category_index: int,
    jitter_width: float,
    rng: random.Random,
    value: float,
) -> dict[str, Any]:
    jittered_x = category_index + rng.uniform(-jitter_width, jitter_width)
    return {
        "figure_id": figure_id,
        "kind": kind,
        "participant_id": row.get("participant_id", ""),
        "category": category,
        "category_label": category_label,
        "x_position": category_index,
        "jittered_x": jittered_x,
        "crossnobis": value,
        "analysis_variant": row.get("analysis_variant", ""),
        "phase_id": row.get("phase_id", ""),
        "contrast_id": row.get("contrast_id", ""),
        "roi_label": row.get("roi_label", ""),
        "pooled_row_count": row.get("pooled_row_count", ""),
        "pooled_roi_count": row.get("pooled_roi_count", ""),
    }


def _summary_row(
    *,
    figure_id: str,
    kind: str,
    category: str,
    category_label: str,
    values: Sequence[float],
    participant_count: int,
    ci_level: float,
) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    sd = _sample_sd(values)
    sem = sd / math.sqrt(n) if n > 0 else math.nan
    margin = _t_critical(ci_level, n - 1) * sem if n > 1 else math.nan
    return {
        "figure_id": figure_id,
        "kind": kind,
        "category": category,
        "category_label": category_label,
        "n": n,
        "participant_count": participant_count,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_level": ci_level,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def _render_and_write_figure(
    plt: Any,
    figure_result: dict[str, Any],
    *,
    input_relative_path: str,
    execute: bool,
) -> list[str]:
    figure = figure_result["config"]
    targets = figure_result["outputs"]
    plot_rows = figure_result["plot_data_rows"]
    summary_rows = figure_result["summary_rows"]
    fig, ax, annotations = _draw_matplotlib_figure(plt, figure, plot_rows=plot_rows, summary_rows=summary_rows)
    layout_warnings = _layout_warnings(fig, ax, annotations=annotations, figure=figure)
    dpi = int(_number_or_default(figure.get("dpi"), 300))
    try:
        for output_format in _output_formats(figure):
            target = Path(targets[f"figure_{output_format}"]["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, format=output_format, dpi=dpi)
        _write_tsv_atomic(Path(targets["plot_data_tsv"]["path"]), plot_rows, PLOT_DATA_COLUMNS)
        _write_tsv_atomic(Path(targets["summary_tsv"]["path"]), summary_rows, SUMMARY_COLUMNS)
        manifest = _figure_manifest(
            figure_result,
            input_relative_path=input_relative_path,
            layout_warnings=layout_warnings,
            executed=execute,
        )
        _write_text_atomic(Path(targets["manifest_json"]["path"]), json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    finally:
        plt.close(fig)
    return layout_warnings


def _draw_matplotlib_figure(
    plt: Any,
    figure: Mapping[str, Any],
    *,
    plot_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, list[Any]]:
    from matplotlib.ticker import MaxNLocator

    width = float(_number_or_default(_mapping(figure.get("layout")).get("width_inches"), 9.0))
    height = float(_number_or_default(_mapping(figure.get("layout")).get("height_inches"), 5.4))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(w_pad=0.18, h_pad=0.18, wspace=0.02, hspace=0.02)
    categories = [str(row["category"]) for row in summary_rows]
    category_labels = [str(row["category_label"]) for row in summary_rows]
    values_by_category = {
        category: [float(row["crossnobis"]) for row in plot_rows if str(row["category"]) == category]
        for category in categories
    }
    positions = list(range(1, len(categories) + 1))
    if str(figure.get("kind")) == "category_distribution_mean_ci" and bool(figure.get("violin", figure.get("distribution") == "violin")):
        violin_values = [values_by_category[category] for category in categories]
        parts = ax.violinplot(violin_values, positions=positions, widths=0.72, showmeans=False, showextrema=False)
        for body in parts.get("bodies", []):
            body.set_facecolor("#d9e2ec")
            body.set_edgecolor("#7b8794")
            body.set_alpha(0.45)
    ax.scatter(
        [float(row["jittered_x"]) for row in plot_rows],
        [float(row["crossnobis"]) for row in plot_rows],
        s=26,
        color="#286983",
        alpha=0.68,
        linewidths=0,
        zorder=2,
    )
    for summary in summary_rows:
        x_position = positions[categories.index(str(summary["category"]))]
        mean = float(summary["mean"])
        ci_low = float(summary["ci_low"])
        ci_high = float(summary["ci_high"])
        ax.errorbar(
            [x_position],
            [mean],
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D",
            color="#1f2933",
            ecolor="#1f2933",
            elinewidth=1.8,
            markersize=6,
            capsize=4,
            zorder=3,
        )
    if bool(figure.get("zero_line", True)):
        ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xticks(positions)
    rotation = int(_number_or_default(figure.get("x_tick_rotation"), 35 if str(figure.get("kind")) == "strip_mean_ci" else 0))
    ax.set_xticklabels(category_labels, rotation=rotation, ha="right" if rotation else "center")
    ax.set_title(str(figure.get("title") or "MVPA crossnobis"))
    ax.set_ylabel(str(figure.get("ylabel") or figure.get("y") or "crossnobis"))
    xlabel = str(figure.get("xlabel") or figure.get("x") or "")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", pad=1.5)
    annotations: list[Any] = []
    y_values = [float(row["crossnobis"]) for row in plot_rows]
    y_values.extend(float(row["ci_low"]) for row in summary_rows)
    y_values.extend(float(row["ci_high"]) for row in summary_rows)
    if bool(figure.get("zero_line", True)):
        y_values.append(0.0)
    y_min = min(y_values) if y_values else -1.0
    y_max = max(y_values) if y_values else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    annotation_config = _mapping(figure.get("annotations"))
    show_annotations = bool(annotation_config.get("show", False))
    top_pad = 0.30 if show_annotations else 0.14
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * top_pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    if show_annotations:
        for summary in summary_rows:
            x_position = positions[categories.index(str(summary["category"]))]
            text = _annotation_text(summary)
            y_text = float(summary["ci_high"]) + y_range * 0.05
            annotations.append(ax.text(x_position, y_text, text, ha="center", va="bottom", fontsize=9, color="#1f2933"))
    caption = _optional_text(figure.get("caption") or figure.get("notes"))
    if caption:
        annotations.append(fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=9, color="#4b5563"))
    ax.margins(x=0.06)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    return fig, ax, annotations


def _layout_warnings(fig: Any, ax: Any, *, annotations: Sequence[Any], figure: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    qc = _mapping(figure.get("layout_qc"))
    max_title_chars = int(_number_or_default(qc.get("max_title_chars"), 120))
    title = _optional_text(figure.get("title"))
    if title and len(title) > max_title_chars:
        warnings.append(f"title exceeds {max_title_chars} characters.")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    text_artists = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *annotations]
    legend = ax.get_legend()
    if legend is not None:
        text_artists.extend(legend.get_texts())
    for artist in text_artists:
        text = _optional_text(artist.get_text()) if hasattr(artist, "get_text") else None
        if text is None:
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.x0 < fig_bbox.x0 or bbox.x1 > fig_bbox.x1 or bbox.y0 < fig_bbox.y0 or bbox.y1 > fig_bbox.y1:
            warnings.append(f"text artist is outside figure bounds: {text[:40]}.")
    return _unique(warnings)


def _figure_manifest(
    figure_result: Mapping[str, Any],
    *,
    input_relative_path: str,
    layout_warnings: Sequence[str],
    executed: bool,
) -> dict[str, Any]:
    outputs = {
        key: {"relative_path": value["relative_path"], "filename": value["filename"]}
        for key, value in figure_result["outputs"].items()
    }
    output_hashes = {
        key: _sha256(Path(value["path"]))
        for key, value in figure_result["outputs"].items()
        if Path(value["path"]).is_file()
    }
    return {
        "schema_version": "research_platform.analysis.mvpa.figure_export.v1",
        "figure_id": figure_result["figure_id"],
        "kind": figure_result["kind"],
        "executed": executed,
        "input_table_relpath": input_relative_path,
        "outputs": outputs,
        "output_hashes": output_hashes,
        "row_counts": {
            "plot_data": len(figure_result["plot_data_rows"]),
            "summary": len(figure_result["summary_rows"]),
        },
        "categories": figure_result["categories"],
        "summary_rows": figure_result["summary_rows"],
        "layout_qc": {
            "warnings": list(layout_warnings),
            "status": "warning" if layout_warnings else "ok",
        },
        "editable_text_settings": {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
        "absolute_source_paths_excluded": True,
        "warnings": _unique([*figure_result["warnings"], *layout_warnings]),
        "errors": figure_result["errors"],
    }


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "MVPA figure exports require matplotlib. Install it locally with: uv add matplotlib "
            "or run the command with: uv run --with matplotlib -- rp analysis mvpa export-figures ..."
        ) from exc
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    return plt


def _figure_result(
    figure: Mapping[str, Any],
    *,
    targets: Mapping[str, Mapping[str, Any]],
    plot_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    errors: Sequence[str],
    categories: Sequence[str],
) -> dict[str, Any]:
    return {
        "figure_id": _figure_id(figure),
        "kind": str(figure.get("kind") or ""),
        "config": dict(figure),
        "outputs": _json_targets(targets, executed=False),
        "plot_data_rows": [dict(row) for row in plot_rows],
        "summary_rows": [dict(row) for row in summary_rows],
        "categories": list(categories),
        "row_counts": {"plot_data": len(plot_rows), "summary": len(summary_rows)},
        "layout_warnings": [],
        "warnings": _unique(warnings),
        "errors": _unique(errors),
    }


def _empty_figure_result(figure: Mapping[str, Any], *, targets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return _figure_result(figure, targets=targets, plot_rows=[], summary_rows=[], warnings=[], errors=[], categories=[])


def _public_figure_result(result: Mapping[str, Any], *, executed: bool, input_relative_path: str) -> dict[str, Any]:
    public = {
        key: value
        for key, value in result.items()
        if key not in {"config"}
    }
    public["outputs"] = {
        key: {**dict(value), "status": "written" if executed else "planned", "executed": executed}
        for key, value in result["outputs"].items()
    }
    public["manifest"] = _figure_manifest(
        result,
        input_relative_path=input_relative_path,
        layout_warnings=result.get("layout_warnings", []),
        executed=executed,
    )
    return public


def _json_targets(targets: Mapping[str, Mapping[str, Any]], *, executed: bool) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "relative_path": str(target["relative_path"]),
            "path": Path(target["path"]).as_posix(),
            "filename": str(target["filename"]),
            "status": "written" if executed else "planned",
            "executed": executed,
        }
        for key, target in targets.items()
    }


def _write_tsv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    os.replace(tmp_path, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    os.replace(tmp_path, path)


def _output_formats(figure: Mapping[str, Any]) -> tuple[str, ...]:
    values = _text_sequence(figure.get("output_formats"))
    return tuple(value.lower().lstrip(".") for value in values) or SUPPORTED_OUTPUT_FORMATS


def _figure_id(figure: Mapping[str, Any]) -> str:
    return str(figure.get("figure_id") or figure.get("id") or "figure")


def _filename_prefix(payload: Mapping[str, Any]) -> str:
    input_config = _input_config(payload)
    configured = _optional_text(input_config.get("filename_prefix"))
    if configured:
        return configured
    entities = _mapping(payload.get("entities"))
    session = _optional_text(entities.get("session_id") or entities.get("session"))
    task = _optional_text(entities.get("task_id") or entities.get("task"))
    pieces: list[str] = []
    if session:
        pieces.append(session)
    if task:
        pieces.append(f"task-{task.removeprefix('task-')}")
    return "_".join(pieces) if pieces else "desc-Crossnobis"


def _label_mapping(figure: Mapping[str, Any]) -> dict[str, str]:
    configured = figure.get("display_labels") or figure.get("label_mapping") or {}
    if not isinstance(configured, Mapping):
        return {}
    return {str(key): str(value) for key, value in configured.items()}


def _annotation_text(summary: Mapping[str, Any]) -> str:
    return f"M={float(summary['mean']):.3g}\n95% CI [{float(summary['ci_low']):.3g}, {float(summary['ci_high']):.3g}]"


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _t_critical(ci_level: float, df: int) -> float:
    if df <= 0:
        return math.nan
    try:
        from scipy import stats

        return float(stats.t.ppf((1.0 + ci_level) / 2.0, df))
    except Exception:
        table_95 = {
            1: 12.7062047364,
            2: 4.3026527299,
            3: 3.1824463053,
            4: 2.7764451052,
            5: 2.5705818366,
            6: 2.4469118511,
            7: 2.3646242510,
            8: 2.3060041350,
            9: 2.2621571627,
            10: 2.2281388519,
            11: 2.2009851601,
            12: 2.1788128297,
            13: 2.1603686565,
            14: 2.1447866879,
            15: 2.1314495456,
            16: 2.1199052992,
            17: 2.1098155778,
            18: 2.1009220402,
            19: 2.0930240544,
            20: 2.0859634473,
            21: 2.0796138447,
            22: 2.0738730679,
            23: 2.0686576104,
            24: 2.0638985616,
            25: 2.0595385528,
            26: 2.0555294386,
            27: 2.0518305165,
            28: 2.0484071418,
            29: 2.0452296421,
            30: 2.0422724563,
        }
        if math.isclose(ci_level, 0.95) and df in table_95:
            return table_95[df]
        return 1.9599639845


def _finite_number(value: Any, label: str, errors: list[str]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be finite numeric.")
        return math.nan
    if not math.isfinite(number):
        errors.append(f"{label} must be finite numeric.")
    return number


def _number_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(text for text in (_optional_text(item) for item in value) if text)
    text = _optional_text(value)
    return (text,) if text else ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_label(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "._-" for char in value)


def _safe_column(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "._-" for char in value)


def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return format(value, ".17g")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _unique(values: Sequence[str]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for value in values:
        if value:
            seen[str(value)] = None
    return list(seen)


__all__ = [
    "PLOT_DATA_COLUMNS",
    "SUMMARY_COLUMNS",
    "SUPPORTED_FIGURE_KINDS",
    "SUPPORTED_OUTPUT_FORMATS",
    "plan_or_execute_mvpa_figure_export",
    "validate_mvpa_figure_export_document",
]
