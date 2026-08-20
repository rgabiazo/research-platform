"""Shared helpers for FSL-oriented neuro adapters."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

_PLACEHOLDER_PATTERN = re.compile(r"{([A-Za-z0-9_]+)}")
_ENTITY_PATTERNS = (
    ("subject_id", "sub"),
    ("session_id", "ses"),
    ("task_id", "task"),
    ("acq", "acq"),
    ("rec", "rec"),
    ("dir", "dir"),
    ("run_id", "run"),
    ("echo", "echo"),
)
_BIDS_ENTITY_ORDER = ("subject_id", "session_id", "task_id", "acq", "rec", "dir", "run_id", "echo")
_SUPPORTED_FSL_EXECUTION_BACKENDS = frozenset({"native", "apptainer", "singularity"})
_SUPPORTED_FSL_PULL_MODES = frozenset({"never", "if_missing"})
_HEADLESS_FEAT_ENVIRONMENT = {
    "BROWSER": "false",
    "FSL_FEAT_WATCH": "0",
}
_FSL_REQUIRED_CLEANENV_VARIABLES = ("USER",)
DEFAULT_FSL_APPTAINER_CACHE_DIR = "${SCRATCH}/apptainer-cache"
DEFAULT_FSL_APPTAINER_TMPDIR = '${SLURM_TMPDIR:-$SCRATCH/apptainer-tmp}'


@dataclass(frozen=True)
class FslContainerSpec:
    enabled: bool
    backend: str
    image: str
    pull_mode: str
    image_name: str | None = None
    image_root: str | None = None


@dataclass(frozen=True)
class FslRuntimeBackendSpec:
    execution_backend: str
    environment: dict[str, str]
    container: FslContainerSpec | None = None


def strip_nii_suffix(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def normalize_entity_label(value: Any, *, prefix: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(f"{prefix}-"):
        return text
    return f"{prefix}-{text}"


def parse_bidsish_entities(path: str | Path) -> dict[str, str]:
    name = strip_nii_suffix(path)
    if "_desc-" in name:
        name = name.split("_desc-", 1)[0]
    if "_space-" in name:
        name = name.split("_space-", 1)[0]

    entities: dict[str, str] = {}
    for key, prefix in _ENTITY_PATTERNS:
        match = re.search(rf"(?:(?<=_)|^){prefix}-([A-Za-z0-9]+)(?=_|$)", name)
        if match:
            entities[key] = f"{prefix}-{match.group(1)}"
    return entities


def build_bids_base(entities: dict[str, str]) -> str:
    return "_".join(value for key in _BIDS_ENTITY_ORDER if (value := entities.get(key)))


def render_path_pattern(pattern: str, values: dict[str, str], *, default: str = "*") -> str:
    return _PLACEHOLDER_PATTERN.sub(lambda match: str(values.get(match.group(1), default)), pattern)


def resolve_reference_path(workspace_root: str | Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(workspace_root).resolve() / candidate).resolve()


def infer_nvols_and_tr(
    nifti: str | Path,
    *,
    override_tr: float | None = None,
    metadata_paths: Sequence[str | Path] | None = None,
    metadata_search_roots: Sequence[str | Path] | None = None,
) -> tuple[int | None, float | None]:
    nifti_path = Path(nifti)
    nvols: int | None = None
    tr: float | None = None

    fslnvols = shutil.which("fslnvols")
    fslval = shutil.which("fslval")
    try:
        if fslnvols:
            output = subprocess.check_output([fslnvols, str(nifti_path)], text=True).strip()
            nvols = int(output)
        if override_tr is not None:
            tr = float(override_tr)
        elif fslval:
            output = subprocess.check_output([fslval, str(nifti_path), "pixdim4"], text=True).strip()
            tr = float(output)
    except Exception:
        pass

    if nvols is None or tr is None:
        try:
            import nibabel as nib  # type: ignore

            image = nib.load(str(nifti_path))
            if nvols is None:
                nvols = image.shape[3] if image.ndim == 4 else 1
            if tr is None and image.ndim == 4:
                zooms = image.header.get_zooms()
                if len(zooms) > 3:
                    tr_value = float(zooms[3])
                    tr = tr_value if tr_value else None
        except Exception:
            pass
    if nvols is None or tr is None:
        header_nvols, header_tr = _read_nifti_header_nvols_and_tr(nifti_path)
        if nvols is None:
            nvols = header_nvols
        if tr is None:
            tr = header_tr
    if tr is None:
        tr = _read_tr_from_json_metadata(
            nifti_path,
            metadata_paths=metadata_paths,
            metadata_search_roots=metadata_search_roots,
        )
    return nvols, tr


def _read_nifti_header_nvols_and_tr(nifti_path: Path) -> tuple[int | None, float | None]:
    try:
        opener = gzip.open if nifti_path.name.endswith(".gz") else open
        with opener(nifti_path, "rb") as handle:
            header = handle.read(108)
    except Exception:
        return None, None
    if len(header) < 108:
        return None, None

    endian = _infer_nifti_endianness(header[:4])
    if endian is None:
        return None, None

    try:
        dim = struct.unpack(f"{endian}8h", header[40:56])
        pixdim = struct.unpack(f"{endian}8f", header[76:108])
    except struct.error:
        return None, None

    nvols: int | None = None
    tr: float | None = None
    ndim = int(dim[0]) if dim else 0
    if ndim >= 4:
        nvols = int(dim[4]) if dim[4] > 0 else None
        tr_value = float(pixdim[4])
        tr = tr_value if tr_value > 0 else None
    elif ndim > 0:
        nvols = 1
    return nvols, tr


def _infer_nifti_endianness(raw_sizeof_hdr: bytes) -> str | None:
    if len(raw_sizeof_hdr) != 4:
        return None
    if struct.unpack("<I", raw_sizeof_hdr)[0] == 348:
        return "<"
    if struct.unpack(">I", raw_sizeof_hdr)[0] == 348:
        return ">"
    return None


def _read_tr_from_json_metadata(
    nifti_path: Path,
    *,
    metadata_paths: Sequence[str | Path] | None,
    metadata_search_roots: Sequence[str | Path] | None,
) -> float | None:
    explicit_candidates = [Path(path).resolve() for path in metadata_paths or []]
    for candidate in explicit_candidates:
        tr = _read_repetition_time_from_json(candidate)
        if tr is not None:
            return tr

    candidates = _discover_related_bold_json_sidecars(
        nifti_path,
        metadata_search_roots=metadata_search_roots or (),
    )
    for candidate in candidates:
        tr = _read_repetition_time_from_json(candidate)
        if tr is not None:
            return tr
    return None


def _discover_related_bold_json_sidecars(
    nifti_path: Path,
    *,
    metadata_search_roots: Sequence[str | Path],
) -> list[Path]:
    stem = strip_nii_suffix(nifti_path)
    entities = parse_bidsish_entities(nifti_path)
    bids_base = build_bids_base(entities)
    if not bids_base:
        return []

    exact_sidecar = nifti_path.with_name(f"{stem}.json").resolve()
    candidates: dict[Path, None] = {}
    if exact_sidecar.exists():
        candidates[exact_sidecar] = None

    subject_dir = entities.get("subject_id")
    session_dir = entities.get("session_id")
    search_patterns = _metadata_search_patterns(
        bids_base=bids_base,
        subject_dir=subject_dir,
        session_dir=session_dir,
    )
    for root_value in metadata_search_roots:
        root = Path(root_value).resolve()
        if not root.exists():
            continue
        for pattern in search_patterns:
            for candidate in root.glob(pattern):
                if candidate.is_file():
                    candidates[candidate.resolve()] = None

    space_token = _extract_filename_token(stem, prefix="space")
    return sorted(
        candidates,
        key=lambda candidate: _metadata_candidate_sort_key(
            candidate,
            exact_sidecar=exact_sidecar,
            space_token=space_token,
        ),
    )


def _metadata_search_patterns(*, bids_base: str, subject_dir: str | None, session_dir: str | None) -> tuple[str, ...]:
    patterns: list[str] = []
    if subject_dir and session_dir:
        patterns.append(f"**/{subject_dir}/{session_dir}/func/{bids_base}*_bold.json")
    if subject_dir:
        patterns.append(f"**/{subject_dir}/func/{bids_base}*_bold.json")
    patterns.append(f"**/{bids_base}*_bold.json")
    deduped: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        deduped.append(pattern)
    return tuple(deduped)


def _metadata_candidate_sort_key(candidate: Path, *, exact_sidecar: Path, space_token: str | None) -> tuple[int, int, int, str]:
    name = candidate.name
    return (
        0 if candidate == exact_sidecar else 1,
        0 if _candidate_matches_space(name, space_token) else 1,
        len(candidate.parts),
        name,
    )


def _candidate_matches_space(name: str, space_token: str | None) -> bool:
    if space_token is None:
        return True
    return f"_space-{space_token}_" in name or name.endswith(f"_space-{space_token}_bold.json")


def _extract_filename_token(name: str, *, prefix: str) -> str | None:
    match = re.search(rf"(?:^|_){prefix}-([^_]+)(?:_|$)", name)
    if not match:
        return None
    return match.group(1)


def _read_repetition_time_from_json(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("RepetitionTime")
    try:
        tr = float(value)
    except (TypeError, ValueError):
        return None
    return tr if tr > 0 else None


def resolve_fsl_runtime_backend(profile_config: Mapping[str, Any], *, mode: str) -> FslRuntimeBackendSpec:
    profile_block = profile_config.get("slurm" if mode == "slurm" else "local", {})
    if not isinstance(profile_block, Mapping):
        return FslRuntimeBackendSpec(execution_backend="native", environment={})

    execution_backend = _optional_text(profile_block.get("execution_backend")) or "native"
    if execution_backend not in _SUPPORTED_FSL_EXECUTION_BACKENDS:
        supported = ", ".join(sorted(_SUPPORTED_FSL_EXECUTION_BACKENDS))
        raise ValueError(f"Unsupported FSL execution_backend {execution_backend!r}. Expected one of: {supported}.")

    environment = _normalize_environment(profile_block.get("environment"))
    if execution_backend == "native":
        return FslRuntimeBackendSpec(execution_backend=execution_backend, environment=environment)

    container_block = profile_block.get("container", {})
    if not isinstance(container_block, Mapping):
        raise ValueError(f"FSL {mode} runtime backend {execution_backend!r} requires a container mapping.")

    enabled = bool(container_block.get("enabled", False))
    if not enabled:
        raise ValueError(
            f"FSL {mode} runtime backend {execution_backend!r} requires container.enabled=true."
        )

    container_backend = _optional_text(container_block.get("backend")) or execution_backend
    if container_backend not in _SUPPORTED_FSL_EXECUTION_BACKENDS - {"native"}:
        raise ValueError(
            f"FSL {mode} container.backend {container_backend!r} must be 'apptainer' or 'singularity'."
        )
    if container_backend != execution_backend:
        raise ValueError(
            f"FSL {mode} execution_backend {execution_backend!r} must match container.backend {container_backend!r}."
        )

    image = _optional_text(container_block.get("image"))
    if image is None:
        raise ValueError(f"FSL {mode} runtime backend {execution_backend!r} requires container.image.")

    pull_mode = _optional_text(container_block.get("pull_mode")) or "never"
    if pull_mode not in _SUPPORTED_FSL_PULL_MODES:
        supported = ", ".join(sorted(_SUPPORTED_FSL_PULL_MODES))
        raise ValueError(f"Unsupported FSL container pull_mode {pull_mode!r}. Expected one of: {supported}.")

    image_name = _normalize_image_name(_optional_text(container_block.get("image_name")))
    if image.startswith("docker://") and pull_mode == "if_missing" and image_name is None:
        image_name = _derive_container_image_name(image)
    image_root = _optional_text(container_block.get("image_root"))
    if image.startswith("docker://") and pull_mode == "if_missing" and image_root is None:
        raise ValueError(
            f"FSL {mode} container.image_root is required when container.image is a docker URI and pull_mode=if_missing."
        )

    container = FslContainerSpec(
        enabled=enabled,
        backend=container_backend,
        image=image,
        pull_mode=pull_mode,
        image_name=image_name,
        image_root=image_root,
    )
    return FslRuntimeBackendSpec(execution_backend=execution_backend, environment=environment, container=container)


def build_fsl_headless_env(base_env: Mapping[str, Any] | None = None) -> dict[str, str]:
    environment = _normalize_environment(base_env)
    for name in _FSL_REQUIRED_CLEANENV_VARIABLES:
        if name in environment:
            continue
        value = _current_process_env(name)
        if value is not None:
            environment[name] = value
    environment.update(_HEADLESS_FEAT_ENVIRONMENT)
    return environment


def collect_fsl_bind_roots(
    *,
    manifest: Mapping[str, Any],
    output_root: str | Path,
    workspace_root: str | Path,
) -> list[str]:
    workspace_root_path = Path(workspace_root).resolve()
    candidates: list[Path] = [workspace_root_path]

    for value in (
        manifest.get("dataset", {}).get("root"),
        manifest.get("dataset", {}).get("derivative_root"),
        output_root,
    ):
        resolved = _resolve_optional_manifest_path(workspace_root_path, value)
        if resolved is not None:
            candidates.append(resolved)

    analysis = manifest.get("analysis", {})
    selected_input_root_refs: set[str] = set()
    inputs = analysis.get("inputs", {})
    if isinstance(inputs, Mapping):
        for input_config in inputs.values():
            if not isinstance(input_config, Mapping):
                continue
            root_ref = _optional_text(input_config.get("root_ref"))
            if root_ref is not None:
                selected_input_root_refs.add(root_ref)

    input_roots = analysis.get("input_roots", {})
    if isinstance(input_roots, Mapping):
        for name, root_spec in input_roots.items():
            if not isinstance(root_spec, Mapping):
                continue
            if selected_input_root_refs and str(name) not in selected_input_root_refs:
                continue
            resolved = _resolve_optional_manifest_path(workspace_root_path, root_spec.get("path"))
            if resolved is not None:
                candidates.append(resolved)

    if isinstance(inputs, Mapping):
        for input_config in inputs.values():
            if not isinstance(input_config, Mapping):
                continue
            if _optional_text(input_config.get("root_ref")) is not None:
                continue
            resolved = _resolve_optional_manifest_path(workspace_root_path, input_config.get("root"))
            if resolved is not None:
                candidates.append(resolved)

    return [str(path) for path in _dedupe_bind_roots(candidates)]


def resolve_fsl_container_runtime_image(container: FslContainerSpec) -> dict[str, Any]:
    image_reference = container.image
    runtime_image = image_reference
    requires_pull = False

    if image_reference.startswith("docker://") and container.pull_mode == "if_missing":
        image_name = _normalize_image_name(container.image_name) or _derive_container_image_name(image_reference)
        image_root = container.image_root
        if image_root is None:
            raise ValueError("container.image_root is required when materializing a docker:// FSL image.")
        runtime_image = _join_runtime_path(image_root, image_name)
        requires_pull = True

    return {
        "image_reference": image_reference,
        "runtime_image": runtime_image,
        "requires_pull": requires_pull,
    }


def build_fsl_container_exec_command(
    *,
    backend: str,
    image_reference: str,
    bind_roots: Sequence[str],
    env: Mapping[str, Any] | None,
    command: Sequence[str],
) -> list[str]:
    if backend not in _SUPPORTED_FSL_EXECUTION_BACKENDS - {"native"}:
        raise ValueError(f"Unsupported FSL container backend {backend!r}.")

    resolved_env = build_fsl_headless_env(env)
    exec_command = [backend, "exec", "--cleanenv"]
    if resolved_env:
        exec_command.extend(
            ["--env", ",".join(f"{key}={resolved_env[key]}" for key in sorted(resolved_env))]
        )
    for root in bind_roots:
        exec_command.extend(["--bind", f"{root}:{root}"])
    exec_command.append(image_reference)
    exec_command.extend(str(part) for part in command)
    return exec_command


def build_fsl_container_prepare_and_exec_shell(
    *,
    backend: str,
    container: FslContainerSpec,
    bind_roots: Sequence[str],
    env: Mapping[str, Any] | None,
    command: Sequence[str],
) -> str:
    image_details = resolve_fsl_container_runtime_image(container)
    exec_shell = _build_fsl_container_exec_shell(
        backend=backend,
        runtime_image_variable='"$RUNTIME_IMAGE"',
        bind_roots=bind_roots,
        env=env,
        command=command,
    )

    image_root = container.image_root or ""
    image_path = str(image_details["runtime_image"])
    image_source = str(image_details["image_reference"])
    lines = [
        "set -euo pipefail",
        f"RUNTIME_IMAGE={_double_quoted_shell_value(image_path)}",
    ]
    if not image_details["requires_pull"]:
        lines.append(exec_shell)
        return "\n".join(lines)

    lines.extend(
        [
            f"IMAGE_ROOT={_double_quoted_shell_value(image_root)}",
            f"IMAGE_SOURCE={_double_quoted_shell_value(image_source)}",
            'LOCK_DIR="${RUNTIME_IMAGE}.lock.d"',
            'TMP_IMAGE="${RUNTIME_IMAGE}.tmp.$$"',
            'cleanup(){',
            '  if [ -n "${LOCK_DIR:-}" ]; then',
            '    rmdir "$LOCK_DIR" 2>/dev/null || true',
            "  fi",
            '  if [ -n "${TMP_IMAGE:-}" ] && [ -f "$TMP_IMAGE" ]; then',
            '    rm -f "$TMP_IMAGE"',
            "  fi",
            "}",
            "trap cleanup EXIT INT TERM",
            'mkdir -p "$IMAGE_ROOT"',
            'while ! mkdir "$LOCK_DIR" 2>/dev/null; do',
            "  sleep 1",
            "done",
            'if [ ! -s "$RUNTIME_IMAGE" ]; then',
            f"  {shlex.quote(backend)} pull \"$TMP_IMAGE\" \"$IMAGE_SOURCE\"",
            '  mv "$TMP_IMAGE" "$RUNTIME_IMAGE"',
            "fi",
            exec_shell,
        ]
    )
    return "\n".join(lines)


def build_fsl_container_prepare_shell(
    *,
    backend: str,
    container: FslContainerSpec,
) -> str:
    if backend not in _SUPPORTED_FSL_EXECUTION_BACKENDS - {"native"}:
        raise ValueError(f"Unsupported FSL container backend {backend!r}.")

    image_details = resolve_fsl_container_runtime_image(container)
    if not image_details["requires_pull"]:
        return "set -euo pipefail\nexit 0"

    image_root = container.image_root or ""
    image_path = str(image_details["runtime_image"])
    image_source = str(image_details["image_reference"])
    lines = [
        "set -euo pipefail",
        f'export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-{DEFAULT_FSL_APPTAINER_CACHE_DIR}}}"',
        f'export APPTAINER_TMPDIR="${{APPTAINER_TMPDIR:-{DEFAULT_FSL_APPTAINER_TMPDIR}}}"',
        'export TMPDIR="${TMPDIR:-$APPTAINER_TMPDIR}"',
        f"RUNTIME_IMAGE={_double_quoted_shell_value(image_path)}",
        f"IMAGE_SOURCE={_double_quoted_shell_value(image_source)}",
        f"IMAGE_ROOT={_double_quoted_shell_value(image_root)}",
        'TMP_IMAGE="${RUNTIME_IMAGE}.tmp.$$"',
        'mkdir -p "$IMAGE_ROOT" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"',
        'if [ -s "$RUNTIME_IMAGE" ]; then',
        "  exit 0",
        "fi",
        'rm -rf "${RUNTIME_IMAGE}.lock.d"',
        'rm -f "${RUNTIME_IMAGE}"',
        'rm -f "${RUNTIME_IMAGE}.tmp."*',
        f'{shlex.quote(backend)} pull "$TMP_IMAGE" "$IMAGE_SOURCE"',
        'mv "$TMP_IMAGE" "$RUNTIME_IMAGE"',
    ]
    return "\n".join(lines)


def _resolve_optional_manifest_path(workspace_root: Path, value: Any) -> Path | None:
    text = _optional_text(value)
    if text is None:
        return None
    return resolve_reference_path(workspace_root, text)


def _normalize_environment(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name).strip(): str(raw_value).strip()
        for name, raw_value in value.items()
        if str(name).strip() and str(raw_value).strip()
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_image_name(value: str | None) -> str | None:
    if value is None:
        return None
    return value if value.endswith(".sif") else f"{value}.sif"


def _derive_container_image_name(image_reference: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", image_reference.removeprefix("docker://")).strip("-")
    if not sanitized:
        sanitized = "fsl-image"
    if not sanitized.endswith(".sif"):
        sanitized += ".sif"
    return sanitized


def _current_process_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _join_runtime_path(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf}" if root.rstrip("/") else leaf


def _build_fsl_container_exec_shell(
    *,
    backend: str,
    runtime_image_variable: str,
    bind_roots: Sequence[str],
    env: Mapping[str, Any] | None,
    command: Sequence[str],
) -> str:
    resolved_env = build_fsl_headless_env(env)
    env_part = ""
    if resolved_env:
        env_part = " --env " + shlex.quote(",".join(f"{key}={resolved_env[key]}" for key in sorted(resolved_env)))
    bind_part = "".join(f" --bind {shlex.quote(f'{root}:{root}')}" for root in bind_roots)
    command_part = " ".join(shlex.quote(str(part)) for part in command)
    return f"exec {shlex.quote(backend)} exec --cleanenv{env_part}{bind_part} {runtime_image_variable} {command_part}"


def _double_quoted_shell_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


def _dedupe_bind_roots(paths: Sequence[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if any(_is_relative_to(resolved, existing) for existing in selected):
            continue
        selected = [existing for existing in selected if not _is_relative_to(existing, resolved)]
        if resolved not in selected:
            selected.append(resolved)
    return selected


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
