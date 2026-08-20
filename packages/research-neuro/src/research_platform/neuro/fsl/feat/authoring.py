"""User-friendly FEAT model authoring helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


def init_model_document(
    *,
    name: str,
    options: Mapping[str, Any],
    template: str | None = None,
) -> dict[str, Any]:
    if template is not None:
        document = _template_document(name=name, template=template)
    else:
        document = {
            "model": {
                "name": name,
                "ev_order": _normalize_name_tokens(options.get("ev_order")),
                "derivative_on": _normalize_name_tokens(options.get("derivative_on")),
                "nonconvolved": _normalize_name_tokens(options.get("nonconvolved")),
                "contrasts": [_parse_contrast_spec(raw) for raw in options.get("contrasts", []) or []],
            }
        }
    return deepcopy(document)


def interactive_init_model_document(
    *,
    name: str,
    template: str | None = None,
) -> dict[str, Any]:
    if template is not None:
        return init_model_document(name=name, options={}, template=template)

    while True:
        ev_order = _normalize_name_tokens(_prompt_value("EV order (space- or comma-separated): "))
        if ev_order:
            break
        print("Provide at least one EV name.")

    derivative_on = _normalize_name_tokens(
        _prompt_value("Derivative-on EVs (optional; space- or comma-separated): ")
    )
    nonconvolved = _normalize_name_tokens(
        _prompt_value("Nonconvolved EVs (optional; space- or comma-separated): ")
    )

    contrasts: list[dict[str, Any]] = []
    print("Add one or more contrasts in the form name:w1,w2,...")
    while True:
        raw_contrast = _prompt_value("Contrast (leave blank when done): ")
        if raw_contrast is None:
            continue
        if not raw_contrast:
            if contrasts:
                break
            print("Provide at least one contrast.")
            continue
        try:
            contrasts.append(_parse_contrast_spec(raw_contrast))
        except ValueError as exc:
            print(str(exc))

    return {
        "model": {
            "name": name,
            "ev_order": ev_order,
            "derivative_on": derivative_on,
            "nonconvolved": nonconvolved,
            "contrasts": contrasts,
        }
    }


def validate_model_document(
    *,
    model_name: str,
    document: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    model = document.get("model")
    if not isinstance(model, Mapping):
        return ["Model document must contain a top-level 'model' mapping."]

    declared_name = _optional_text(model.get("name"))
    if declared_name is None:
        errors.append("model.name must be defined.")
    elif declared_name != model_name:
        errors.append(f"model.name {declared_name!r} must match the file name {model_name!r}.")

    ev_order = _normalize_name_tokens(model.get("ev_order"))
    derivative_on = _normalize_name_tokens(model.get("derivative_on"))
    nonconvolved = _normalize_name_tokens(model.get("nonconvolved"))

    if not ev_order:
        errors.append("model.ev_order must define at least one EV.")
    duplicate_evs = _find_duplicates(ev_order)
    if duplicate_evs:
        errors.append(f"model.ev_order contains duplicate EV names: {', '.join(duplicate_evs)}.")

    ev_names = set(ev_order)
    unknown_derivative = [name for name in derivative_on if name not in ev_names]
    if unknown_derivative:
        errors.append(
            "model.derivative_on contains EV names not present in model.ev_order: "
            + ", ".join(unknown_derivative)
            + "."
        )

    unknown_nonconvolved = [name for name in nonconvolved if name not in ev_names]
    if unknown_nonconvolved:
        errors.append(
            "model.nonconvolved contains EV names not present in model.ev_order: "
            + ", ".join(unknown_nonconvolved)
            + "."
        )

    overlap = sorted(set(derivative_on) & set(nonconvolved))
    if overlap:
        errors.append(
            "model.derivative_on and model.nonconvolved must not overlap: " + ", ".join(overlap) + "."
        )

    contrasts_payload = model.get("contrasts")
    if not isinstance(contrasts_payload, list) or not contrasts_payload:
        errors.append("model.contrasts must define at least one contrast.")
        return errors

    contrast_names: list[str] = []
    for index, contrast_payload in enumerate(contrasts_payload):
        label = f"model.contrasts[{index}]"
        if not isinstance(contrast_payload, Mapping):
            errors.append(f"{label} must contain a mapping.")
            continue
        contrast_name = _optional_text(contrast_payload.get("name"))
        if contrast_name is None:
            errors.append(f"{label}.name must be defined.")
        else:
            contrast_names.append(contrast_name)
        weights = contrast_payload.get("weights")
        if not isinstance(weights, list):
            errors.append(f"{label}.weights must contain a list of numeric values.")
            continue
        normalized_weights: list[float] = []
        for weight_index, raw_weight in enumerate(weights):
            try:
                normalized_weights.append(float(raw_weight))
            except (TypeError, ValueError):
                errors.append(f"{label}.weights[{weight_index}] must be numeric.")
        if ev_order and len(weights) != len(ev_order):
            errors.append(
                f"{label}.weights must contain exactly {len(ev_order)} values to match model.ev_order."
            )
        _ = normalized_weights

    duplicate_contrasts = _find_duplicates(contrast_names)
    if duplicate_contrasts:
        errors.append(
            "model.contrasts contains duplicate contrast names: " + ", ".join(duplicate_contrasts) + "."
        )

    return errors


def summarize_model_document(
    *,
    model_name: str,
    document: Mapping[str, Any],
) -> str:
    model = document.get("model")
    if not isinstance(model, Mapping):
        return f"FEAT first-level model: {model_name}\nInvalid document: missing top-level model mapping."

    ev_order = _normalize_name_tokens(model.get("ev_order"))
    derivative_on = _normalize_name_tokens(model.get("derivative_on"))
    nonconvolved = _normalize_name_tokens(model.get("nonconvolved"))
    contrasts_payload = model.get("contrasts")
    contrast_names: list[str] = []
    if isinstance(contrasts_payload, list):
        for contrast_payload in contrasts_payload:
            if isinstance(contrast_payload, Mapping):
                name = _optional_text(contrast_payload.get("name"))
                if name is not None:
                    contrast_names.append(name)

    errors = validate_model_document(model_name=model_name, document=document)
    lines = [
        f"FEAT first-level model: {model_name}",
        f"EV order ({len(ev_order)}): {', '.join(ev_order) if ev_order else '<none>'}",
        f"Derivative-on ({len(derivative_on)}): {', '.join(derivative_on) if derivative_on else '<none>'}",
        f"Nonconvolved ({len(nonconvolved)}): {', '.join(nonconvolved) if nonconvolved else '<none>'}",
        f"Contrasts ({len(contrast_names)}): {', '.join(contrast_names) if contrast_names else '<none>'}",
    ]
    if errors:
        lines.append(f"Validation issues: {len(errors)}")
    return "\n".join(lines)


def rename_model_document(
    *,
    new_name: str,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(dict(document))
    model = updated.get("model")
    if not isinstance(model, dict):
        model = {}
        updated["model"] = model
    model["name"] = new_name
    return updated


def _template_document(*, name: str, template: str) -> dict[str, Any]:
    if template == "blank":
        return {
            "model": {
                "name": name,
                "ev_order": [],
                "derivative_on": [],
                "nonconvolved": [],
                "contrasts": [],
            }
        }
    if template == "basic":
        return {
            "model": {
                "name": name,
                "ev_order": ["condition_a", "condition_b"],
                "derivative_on": ["condition_a", "condition_b"],
                "nonconvolved": [],
                "contrasts": [
                    {
                        "name": "condition_a_gt_condition_b",
                        "weights": [1, -1],
                    }
                ],
            }
        }
    raise ValueError(f"Unsupported FEAT model template: {template!r}.")


def _parse_contrast_spec(raw_value: Any) -> dict[str, Any]:
    value = _optional_text(raw_value)
    if value is None:
        raise ValueError("Contrast values must use the form name:w1,w2,...")
    name, separator, weights_value = value.partition(":")
    contrast_name = name.strip()
    if separator != ":" or not contrast_name:
        raise ValueError("Contrast values must use the form name:w1,w2,...")
    weights: list[int | float] = []
    for weight_token in weights_value.split(","):
        token = weight_token.strip()
        if not token:
            raise ValueError(f"Contrast {contrast_name!r} contains an empty weight.")
        try:
            numeric = float(token)
        except ValueError as exc:
            raise ValueError(f"Contrast {contrast_name!r} contains a non-numeric weight {token!r}.") from exc
        weights.append(int(numeric) if numeric.is_integer() else numeric)
    if not weights:
        raise ValueError(f"Contrast {contrast_name!r} must define at least one weight.")
    return {"name": contrast_name, "weights": weights}


def _normalize_name_tokens(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    tokens: list[str] = []
    if isinstance(raw_value, str):
        raw_items = raw_value.split()
    elif isinstance(raw_value, Iterable):
        raw_items = []
        for item in raw_value:
            raw_items.extend(str(item).split())
    else:
        raw_items = [str(raw_value)]

    for raw_item in raw_items:
        for token in str(raw_item).split(","):
            normalized = token.strip()
            if normalized:
                tokens.append(normalized)
    return tokens


def _find_duplicates(values: list[str]) -> list[str]:
    duplicates: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _prompt_value(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except EOFError:
        return None
