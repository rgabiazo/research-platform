from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .grouping import RunGroup
from .spec import BuildSpec, CompiledPlan, CompiledTransform


def _as_string(value: Any, *, missing_value: str) -> str:
    if value is None:
        return missing_value
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _strip_prefix(value: str, prefix: str) -> str:
    normalized = value.strip()
    token = f"{prefix}/"
    if normalized.startswith(token):
        return normalized[len(token) :]
    return normalized


def _basename(path: str) -> str:
    return Path(path).name


def _evaluate_value_spec(
    value_spec: dict[str, Any],
    *,
    source_row: dict[str, str],
    derived_values: dict[str, Any],
    group: RunGroup,
    first_row: dict[str, str],
    spec: BuildSpec,
    lookups: dict[str, dict[tuple[str, ...], str]],
    row_index: int,
    exec_context: dict[str, str],
) -> Any:
    source_kind = value_spec["source"]
    if source_kind == "literal":
        return value_spec["value"]
    if source_kind == "missing":
        return spec.missing_value
    if source_kind == "source_field":
        return source_row.get(str(value_spec["column"]), "").strip()
    if source_kind == "context_value":
        name = str(value_spec["name"])
        if name in exec_context:
            return exec_context[name]
        return group.phase.context[name]
    if source_kind == "derived_value":
        return derived_values.get(str(value_spec["name"]), spec.missing_value)
    if source_kind == "first_group_value":
        return _evaluate_value_spec(
            dict(value_spec["value"]),
            source_row=first_row,
            derived_values=derived_values,
            group=group,
            first_row=first_row,
            spec=spec,
            lookups=lookups,
            row_index=row_index,
            exec_context=exec_context,
        )
    raise ValueError(f"Unsupported value source {source_kind!r}.")


def _evaluate_operator(
    operator_spec: dict[str, Any],
    *,
    source_row: dict[str, str],
    derived_values: dict[str, Any],
    group: RunGroup,
    first_row: dict[str, str],
    spec: BuildSpec,
    lookups: dict[str, dict[tuple[str, ...], str]],
    row_index: int,
    exec_context: dict[str, str],
) -> Any:
    operator = operator_spec["operator"]
    if operator == "present":
        return _present(
            _evaluate_value_spec(
                operator_spec["value"],
                source_row=source_row,
                derived_values=derived_values,
                group=group,
                first_row=first_row,
                spec=spec,
                lookups=lookups,
                row_index=row_index,
                exec_context=exec_context,
            )
        )
    if operator == "equals_value":
        left = _as_string(
            _evaluate_value_spec(
                operator_spec["left"],
                source_row=source_row,
                derived_values=derived_values,
                group=group,
                first_row=first_row,
                spec=spec,
                lookups=lookups,
                row_index=row_index,
                exec_context=exec_context,
            ),
            missing_value=spec.missing_value,
        )
        right = _as_string(
            _evaluate_value_spec(
                operator_spec["right"],
                source_row=source_row,
                derived_values=derived_values,
                group=group,
                first_row=first_row,
                spec=spec,
                lookups=lookups,
                row_index=row_index,
                exec_context=exec_context,
            ),
            missing_value=spec.missing_value,
        )
        return left == right
    if operator == "not_equals_value":
        return not _evaluate_operator(
            {"operator": "equals_value", "left": operator_spec["left"], "right": operator_spec["right"]},
            source_row=source_row,
            derived_values=derived_values,
            group=group,
            first_row=first_row,
            spec=spec,
            lookups=lookups,
            row_index=row_index,
            exec_context=exec_context,
        )
    if operator == "in_set":
        value = _as_string(
            _evaluate_value_spec(
                operator_spec["value"],
                source_row=source_row,
                derived_values=derived_values,
                group=group,
                first_row=first_row,
                spec=spec,
                lookups=lookups,
                row_index=row_index,
                exec_context=exec_context,
            ),
            missing_value=spec.missing_value,
        )
        return value in {str(item) for item in operator_spec["values"]}
    raise ValueError(f"Unsupported nested operator {operator!r}.")


def _execute_transform(
    transform: CompiledTransform,
    *,
    source_row: dict[str, str],
    derived_values: dict[str, Any],
    group: RunGroup,
    first_row: dict[str, str],
    spec: BuildSpec,
    lookups: dict[str, dict[tuple[str, ...], str]],
    row_index: int,
    exec_context: dict[str, str],
) -> Any:
    operator = transform.operator
    args = transform.args
    if operator == "concat":
        return "".join(
            _as_string(
                _evaluate_value_spec(
                    value_spec,
                    source_row=source_row,
                    derived_values=derived_values,
                    group=group,
                    first_row=first_row,
                    spec=spec,
                    lookups=lookups,
                    row_index=row_index,
                    exec_context=exec_context,
                ),
                missing_value=spec.missing_value,
            )
            for value_spec in args["values"]
        )
    if operator == "strip_prefix":
        value = _as_string(
            _evaluate_value_spec(
                args["value"],
                source_row=source_row,
                derived_values=derived_values,
                group=group,
                first_row=first_row,
                spec=spec,
                lookups=lookups,
                row_index=row_index,
                exec_context=exec_context,
            ),
            missing_value=spec.missing_value,
        )
        prefix = _as_string(
            _evaluate_value_spec(
                args["prefix"],
                source_row=source_row,
                derived_values=derived_values,
                group=group,
                first_row=first_row,
                spec=spec,
                lookups=lookups,
                row_index=row_index,
                exec_context=exec_context,
            ),
            missing_value=spec.missing_value,
        )
        return _strip_prefix(value, prefix)
    if operator == "basename":
        return _basename(
            _as_string(
                _evaluate_value_spec(
                    args["value"],
                    source_row=source_row,
                    derived_values=derived_values,
                    group=group,
                    first_row=first_row,
                    spec=spec,
                    lookups=lookups,
                    row_index=row_index,
                    exec_context=exec_context,
                ),
                missing_value=spec.missing_value,
            )
        )
    if operator == "map_value":
        value = _as_string(
            _evaluate_value_spec(
                args["value"],
                source_row=source_row,
                derived_values=derived_values,
                group=group,
                first_row=first_row,
                spec=spec,
                lookups=lookups,
                row_index=row_index,
                exec_context=exec_context,
            ),
            missing_value=spec.missing_value,
        )
        if value in args["mapping"]:
            return str(args["mapping"][value])
        return _evaluate_value_spec(
            args["default"],
            source_row=source_row,
            derived_values=derived_values,
            group=group,
            first_row=first_row,
            spec=spec,
            lookups=lookups,
            row_index=row_index,
            exec_context=exec_context,
        )
    if operator == "if_else":
        condition = _evaluate_operator(
            args["condition"],
            source_row=source_row,
            derived_values=derived_values,
            group=group,
            first_row=first_row,
            spec=spec,
            lookups=lookups,
            row_index=row_index,
            exec_context=exec_context,
        )
        branch = args["when_true"] if condition else args["when_false"]
        return _evaluate_value_spec(
            branch,
            source_row=source_row,
            derived_values=derived_values,
            group=group,
            first_row=first_row,
            spec=spec,
            lookups=lookups,
            row_index=row_index,
            exec_context=exec_context,
        )
    if operator == "subtract_value":
        left = float(
            _as_string(
                _evaluate_value_spec(
                    args["left"],
                    source_row=source_row,
                    derived_values=derived_values,
                    group=group,
                    first_row=first_row,
                    spec=spec,
                    lookups=lookups,
                    row_index=row_index,
                    exec_context=exec_context,
                ),
                missing_value=spec.missing_value,
            )
        )
        right = float(
            _as_string(
                _evaluate_value_spec(
                    args["right"],
                    source_row=source_row,
                    derived_values=derived_values,
                    group=group,
                    first_row=first_row,
                    spec=spec,
                    lookups=lookups,
                    row_index=row_index,
                    exec_context=exec_context,
                ),
                missing_value=spec.missing_value,
            )
        )
        return left - right
    if operator == "lookup_value":
        key = tuple(
            _as_string(
                _evaluate_value_spec(
                    value_spec,
                    source_row=source_row,
                    derived_values=derived_values,
                    group=group,
                    first_row=first_row,
                    spec=spec,
                    lookups=lookups,
                    row_index=row_index,
                    exec_context=exec_context,
                ),
                missing_value=spec.missing_value,
            )
            for value_spec in args["key"]
        )
        lookup_name = str(args["lookup"])
        if key in lookups.get(lookup_name, {}):
            return lookups[lookup_name][key]
        return _evaluate_value_spec(
            args["default"],
            source_row=source_row,
            derived_values=derived_values,
            group=group,
            first_row=first_row,
            spec=spec,
            lookups=lookups,
            row_index=row_index,
            exec_context=exec_context,
        )
    if operator == "sequence_number":
        return str(int(args["start"]) + row_index)
    raise ValueError(f"Unsupported transform operator {operator!r}.")


def _materialize_row(
    transforms: list[CompiledTransform],
    *,
    source_row: dict[str, str],
    group: RunGroup,
    spec: BuildSpec,
    lookups: dict[str, dict[tuple[str, ...], str]],
    row_index: int,
    exec_context: dict[str, str],
) -> dict[str, Any]:
    first_row = group.rows[0]
    values: dict[str, Any] = {}
    for transform in transforms:
        values[transform.target_column] = _execute_transform(
            transform,
            source_row=source_row,
            derived_values=values,
            group=group,
            first_row=first_row,
            spec=spec,
            lookups=lookups,
            row_index=row_index,
            exec_context=exec_context,
        )
    return values


def _project_row(materialized: dict[str, Any], spec: BuildSpec) -> dict[str, str]:
    return {key: _as_string(value, missing_value=spec.missing_value) for key, value in materialized.items()}


def _build_lookup_tables(
    groups: list[RunGroup],
    plan: CompiledPlan,
    spec: BuildSpec,
    exec_context: dict[str, str],
) -> tuple[dict[str, dict[tuple[str, ...], str]], dict[str, list[tuple[str, ...]]]]:
    lookup_tables: dict[str, dict[tuple[str, ...], str]] = {}
    lookup_key_occurrences: dict[str, list[tuple[str, ...]]] = {}
    for lookup in plan.lookups:
        table: dict[tuple[str, ...], str] = {}
        seen_keys: list[tuple[str, ...]] = []
        for source_group in groups:
            if source_group.phase.name not in lookup.source_row_sets:
                continue
            for row_index, row in enumerate(source_group.rows):
                materialized = _materialize_row(
                    source_group.phase.trial_transforms,
                    source_row=row,
                    group=source_group,
                    spec=spec,
                    lookups=lookup_tables,
                    row_index=row_index,
                    exec_context=exec_context,
                )
                projected = _project_row(materialized, spec)
                if any(projected.get(column) != expected for column, expected in lookup.when_column_equals.items()):
                    continue
                key = tuple(projected[column] for column in lookup.key_columns)
                seen_keys.append(key)
                table[key] = projected[lookup.value_column]
        lookup_tables[lookup.name] = table
        lookup_key_occurrences[lookup.name] = seen_keys
    return lookup_tables, lookup_key_occurrences


def _apply_validations(
    *,
    groups: list[RunGroup],
    plan: CompiledPlan,
    spec: BuildSpec,
    lookup_key_occurrences: dict[str, list[tuple[str, ...]]],
    lookup_tables: dict[str, dict[tuple[str, ...], str]],
    exec_context: dict[str, str],
) -> None:
    for validation in plan.validations:
        if validation.operator == "required_columns":
            continue
        if validation.operator == "lookup_unique_keys":
            lookup_name = str(validation.args["lookup_name"])
            occurrences = lookup_key_occurrences.get(lookup_name, [])
            counts: dict[tuple[str, ...], int] = defaultdict(int)
            for key in occurrences:
                counts[key] += 1
            duplicate_keys = sorted(key for key, count in counts.items() if count > 1)
            if duplicate_keys:
                raise ValueError(f"Lookup {lookup_name!r} produced duplicate keys: {duplicate_keys}.")
            continue
        if validation.operator == "distinct_key_values":
            by_partition: dict[tuple[str, ...], set[str]] = defaultdict(set)
            for group in groups:
                for row_index, row in enumerate(group.rows):
                    materialized = _materialize_row(
                        group.phase.trial_transforms,
                        source_row=row,
                        group=group,
                        spec=spec,
                        lookups=lookup_tables,
                        row_index=row_index,
                        exec_context=exec_context,
                    )
                    projected = _project_row(materialized, spec)
                    partition_key = tuple(projected[column] for column in validation.args["partition_columns"])
                    by_partition[partition_key].add(projected[validation.args["value_column"]])
            collisions = {
                partition_key: sorted(values)
                for partition_key, values in by_partition.items()
                if len(values) > 1
            }
            if collisions:
                details = []
                for partition_key, values in sorted(collisions.items()):
                    condition = partition_key[0] if partition_key else "n/a"
                    stim_id = partition_key[1] if len(partition_key) > 1 else "n/a"
                    details.append(f"condition={condition} stim_id={stim_id} sources={values}")
                raise ValueError(validation.args["message_prefix"] + "; " + "; ".join(details))
            continue
        raise ValueError(f"Unsupported validation operator {validation.operator!r}.")


def build_run_rows(
    groups: list[RunGroup],
    spec: BuildSpec,
    *,
    preserve_source_stim_file: bool = False,
) -> dict[int, list[dict[str, str]]]:
    by_run: dict[int, list[RunGroup]] = defaultdict(list)
    for group in groups:
        by_run[group.run].append(group)

    outputs: dict[int, list[dict[str, str]]] = {}
    for run, run_groups in by_run.items():
        exec_context = {"preserve_source_stim_file": "1" if preserve_source_stim_file else "0"}
        lookup_tables, lookup_key_occurrences = _build_lookup_tables(run_groups, spec.compiled_plan, spec, exec_context)
        _apply_validations(
            groups=run_groups,
            plan=spec.compiled_plan,
            spec=spec,
            lookup_key_occurrences=lookup_key_occurrences,
            lookup_tables=lookup_tables,
            exec_context=exec_context,
        )

        run_rows: list[dict[str, str]] = []
        for group in run_groups:
            if group.phase.instruction_transforms:
                run_rows.append(
                    _project_row(
                        _materialize_row(
                            group.phase.instruction_transforms,
                            source_row=group.rows[0],
                            group=group,
                            spec=spec,
                            lookups=lookup_tables,
                            row_index=0,
                            exec_context=exec_context,
                        ),
                        spec,
                    )
                )
            for row_index, source_row in enumerate(group.rows):
                run_rows.append(
                    _project_row(
                        _materialize_row(
                            group.phase.trial_transforms,
                            source_row=source_row,
                            group=group,
                            spec=spec,
                            lookups=lookup_tables,
                            row_index=row_index,
                            exec_context=exec_context,
                        ),
                        spec,
                    )
                )

        run_rows.sort(key=lambda item: float(item["onset"]))
        outputs[run] = [{column: row[column] for column in spec.columns} for row in run_rows]
    return outputs


def validate_compiled_plan_rows(
    rows: list[dict[str, str]],
    spec: BuildSpec,
    *,
    session_override: str | None = None,
) -> None:
    if not rows:
        raise ValueError("Source file contains no data rows.")
    available = set(rows[0].keys())
    for validation in spec.compiled_plan.validations:
        if validation.operator != "required_columns":
            continue
        required_columns = set(str(column) for column in validation.args["columns"])
        if spec.session_column and session_override is not None:
            required_columns.discard(spec.session_column)
        missing = sorted(column for column in required_columns if column not in available)
        if missing:
            raise ValueError(f"Source file is missing required columns: {', '.join(missing)}.")
