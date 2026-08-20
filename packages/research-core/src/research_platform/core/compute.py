"""Generic compute policy normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

_MEMORY_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?B?)?\s*$", re.IGNORECASE)
_RESOURCE_KEYS = ("cpus", "ram_gb", "threads", "n_jobs")


def normalize_compute_policy(compute_config: Mapping[str, Any]) -> dict[str, Any]:
    policy = _mapping(compute_config.get("policy"))
    default_preset = _optional_text(policy.get("default_preset"))
    presets: dict[str, dict[str, Any]] = {}
    for name, raw in _mapping(policy.get("presets")).items():
        normalized = _resource_fields(_mapping(raw))
        if normalized:
            presets[str(name)] = normalized

    workloads: dict[str, dict[str, Any]] = {}
    for name, raw in _mapping(policy.get("workloads")).items():
        if isinstance(raw, str):
            workloads[str(name)] = {"preset": raw}
            continue
        entry = _mapping(raw)
        normalized = _resource_fields(entry)
        preset_name = _optional_text(entry.get("preset"))
        if preset_name:
            normalized["preset"] = preset_name
        if normalized:
            workloads[str(name)] = normalized

    return {
        "default_preset": default_preset,
        "presets": presets,
        "workloads": workloads,
    }


def parse_ram_gb(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(math.ceil(float(value)))

    text = str(value).strip()
    if not text:
        raise ValueError("RAM value must not be empty.")

    match = _MEMORY_PATTERN.match(text)
    if not match:
        raise ValueError(f"Unsupported RAM value: {value}")

    amount = float(match.group(1))
    unit = (match.group(2) or "G").upper()
    if unit in {"", "G", "GB"}:
        return int(math.ceil(amount))
    if unit in {"M", "MB"}:
        return int(math.ceil(amount / 1024.0))
    if unit in {"K", "KB"}:
        return int(math.ceil(amount / (1024.0 * 1024.0)))
    if unit in {"T", "TB"}:
        return int(math.ceil(amount * 1024.0))
    raise ValueError(f"Unsupported RAM unit: {unit}")


def validate_resource_plan(resources: Mapping[str, Any]) -> dict[str, Any]:
    cpus = int(resources["cpus"])
    ram_gb = int(resources["ram_gb"])
    threads = int(resources["threads"])
    n_jobs = int(resources["n_jobs"])

    if cpus < 1:
        raise ValueError("Resource plan must define cpus >= 1.")
    if ram_gb < 1:
        raise ValueError("Resource plan must define ram_gb >= 1.")
    if threads < 1:
        raise ValueError("Resource plan must define threads >= 1.")
    if n_jobs < 1:
        raise ValueError("Resource plan must define n_jobs >= 1.")
    if threads > cpus:
        raise ValueError("Resource plan must define threads <= cpus.")
    return {
        "cpus": cpus,
        "ram_gb": ram_gb,
        "threads": threads,
        "n_jobs": n_jobs,
        "workload": str(resources["workload"]),
        "preset": str(resources["preset"]),
    }


def resolve_resource_plan(*, compute_config: Mapping[str, Any], workload: str, mode: str) -> dict[str, Any]:
    policy = normalize_compute_policy(compute_config)
    legacy = _legacy_resource_plan(compute_config=compute_config, workload=workload, mode=mode)
    default_preset = policy["default_preset"]
    workload_policy = policy["workloads"].get(workload, {})
    preset_name = _optional_text(workload_policy.get("preset")) or default_preset

    if preset_name and preset_name not in policy["presets"]:
        raise ValueError(f"Unknown compute.policy preset for workload {workload}: {preset_name}")

    resolved_fields = dict(policy["presets"].get(preset_name or "", {}))
    for key in _RESOURCE_KEYS:
        if key in workload_policy:
            resolved_fields[key] = workload_policy[key]

    if not resolved_fields:
        return legacy

    resolved = {
        "cpus": int(resolved_fields.get("cpus", legacy["cpus"])),
        "ram_gb": parse_ram_gb(resolved_fields.get("ram_gb", legacy["ram_gb"])),
        "threads": int(resolved_fields.get("threads", legacy["threads"])),
        "n_jobs": int(resolved_fields.get("n_jobs", legacy["n_jobs"])),
        "workload": workload,
        "preset": preset_name or "custom",
    }
    validated = validate_resource_plan(resolved)
    if mode == "slurm":
        _validate_slurm_memory_contract(
            compute_config=compute_config,
            workload=workload,
            preset_name=preset_name,
            workload_policy=workload_policy,
            resolved_ram_gb=validated["ram_gb"],
        )
    return validated


def _legacy_resource_plan(*, compute_config: Mapping[str, Any], workload: str, mode: str) -> dict[str, Any]:
    local_config = _mapping(compute_config.get("local"))
    slurm_config = _mapping(compute_config.get("slurm"))

    local_jobs = _coerce_int(local_config.get("jobs"), default=1)
    slurm_cpus = _coerce_int(slurm_config.get("cpus"), default=max(local_jobs, 1))
    slurm_ram_gb = parse_ram_gb(slurm_config.get("mem", 4))

    cpus = slurm_cpus if mode == "slurm" else local_jobs
    n_jobs = local_jobs if mode != "slurm" and workload.startswith("tabular_") else 1
    threads = cpus if workload.startswith("bids_") else 1
    resources = {
        "cpus": cpus,
        "ram_gb": slurm_ram_gb,
        "threads": threads,
        "n_jobs": min(n_jobs, cpus),
        "workload": workload,
        "preset": "legacy",
    }
    return validate_resource_plan(resources)


def _validate_slurm_memory_contract(
    *,
    compute_config: Mapping[str, Any],
    workload: str,
    preset_name: str | None,
    workload_policy: Mapping[str, Any],
    resolved_ram_gb: int,
) -> None:
    slurm_config = _mapping(compute_config.get("slurm"))
    slurm_mem = _optional_text(slurm_config.get("mem"))
    if slurm_mem is None:
        return

    slurm_ram_gb = parse_ram_gb(slurm_mem)
    normalized_ram_gb = int(resolved_ram_gb)
    if slurm_ram_gb == normalized_ram_gb:
        return

    preset_label = preset_name or "custom"
    if "ram_gb" in workload_policy:
        policy_field = f"compute.policy.workloads.{workload}.ram_gb"
    elif preset_name:
        policy_field = f"compute.policy.presets.{preset_name}.ram_gb"
    else:
        policy_field = "selected compute.policy ram_gb"

    raise ValueError(
        "SLURM memory contract mismatch for "
        f"workload {workload!r} (preset {preset_label!r}): "
        f"compute.slurm.mem resolves to {slurm_mem} ({slurm_ram_gb} GiB) "
        f"but the active compute.policy memory resolves to ram_gb={normalized_ram_gb}. "
        "Downstream SLURM renderers use the normalized ram_gb to build Snakemake mem_mb "
        "and slurm.jobspec.mem. Reconcile compute.slurm.mem and "
        f"{policy_field} so they agree."
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resource_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in _RESOURCE_KEYS:
        if value.get(key) is not None:
            normalized[key] = value[key]
    return normalized


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(value)
