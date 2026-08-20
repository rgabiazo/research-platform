"""Dynamic BIDS tool adapter loading for orchestration-only core code."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping, Protocol, runtime_checkable

FORBIDDEN_TOOL_OPTION_KEYS = frozenset(
    {
        "cpu",
        "cpus",
        "mem",
        "mem_gb",
        "mem_mb",
        "memory",
        "nprocs",
        "omp_nthreads",
        "ram",
        "ram_gb",
        "threads",
    }
)
_BIDS_TOOL_REGISTRY = {
    "deepprep": "research_platform.neuro.deepprep.adapter:DeepPrepAdapter",
    "fmripost_aroma": "research_platform.neuro.fmripost_aroma.adapter:FmripostAromaAdapter",
}
_BIDS_ANALYSIS_TOOL_REGISTRY = {
    "feat": "research_platform.neuro.fsl.feat.adapter:FeatAnalysisAdapter",
}


@runtime_checkable
class BidsToolAdapter(Protocol):
    def tool_name(self) -> str: ...

    def requires_input_derivative(self) -> bool: ...

    def supported_input_derivatives(self) -> tuple[str, ...]: ...

    def validate_project(
        self,
        *,
        bundle: dict[str, Any],
        pipeline_defaults: dict[str, Any],
        workspace_root: str,
    ) -> list[str]: ...

    def discover_batch_rows(
        self,
        *,
        derivative_root: str,
        selectors: dict[str, str | None],
    ) -> list[dict[str, str]]: ...

    def expected_remote_input_files(
        self,
        *,
        derivative_root: str,
        remote_derivative_root: str,
        row: Mapping[str, str],
    ) -> list[str]: ...

    def expected_remote_auxiliary_files(
        self,
        *,
        context: dict[str, Any],
    ) -> list[str]: ...

    def runtime_metadata(
        self,
        *,
        pipeline_defaults: dict[str, Any],
        output_dir: str,
    ) -> dict[str, str]: ...

    def build_runtime_plan(
        self,
        *,
        manifest: Mapping[str, Any],
        workspace_root: str,
        plan_path: str,
        command_script_path: str,
    ) -> dict[str, Any]: ...

    def sync_entries(
        self,
        *,
        workspace_root: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]: ...

    def build_publish_back_scaffold(
        self,
        *,
        manifest: dict[str, Any],
        run_root: str,
        workspace_root: str,
    ) -> dict[str, Any]: ...

    def scaffold_project_defaults(
        self,
        *,
        project_name: str,
        study_root: str,
        derivative_root: str | None,
        task_id: str | None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class BidsAnalysisToolAdapter(Protocol):
    def tool_name(self) -> str: ...

    def validate_project(
        self,
        *,
        bundle: dict[str, Any],
        pipeline_defaults: dict[str, Any],
        workspace_root: str,
    ) -> list[str]: ...

    def discover_batch_rows(
        self,
        *,
        derivative_root: str,
        selectors: dict[str, str | None],
        context: dict[str, Any],
    ) -> list[dict[str, str]]: ...

    def expected_remote_input_files(
        self,
        *,
        derivative_root: str,
        remote_derivative_root: str,
        row: Mapping[str, str],
        context: dict[str, Any],
    ) -> list[str]: ...

    def runtime_metadata(
        self,
        *,
        pipeline_defaults: dict[str, Any],
        output_dir: str,
    ) -> dict[str, str]: ...

    def sync_entries(
        self,
        *,
        workspace_root: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]: ...

    def build_runtime_plan(
        self,
        *,
        manifest: Mapping[str, Any],
        workspace_root: str,
        plan_path: str,
        command_script_path: str,
    ) -> dict[str, Any]: ...

    def build_publish_back_scaffold(
        self,
        *,
        manifest: dict[str, Any],
        run_root: str,
        workspace_root: str,
    ) -> dict[str, Any]: ...

    def scaffold_project_defaults(
        self,
        *,
        project_name: str,
        study_root: str,
        derivative_root: str,
        task_id: str | None,
        template: str | None = None,
        events_root: str | None = None,
        confounds_root: str | None = None,
        remote_events_root: str | None = None,
        remote_confounds_root: str | None = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class BidsAnalysisModelAuthoringAdapter(Protocol):
    def init_model_document(
        self,
        *,
        name: str,
        options: Mapping[str, Any],
        template: str | None = None,
    ) -> dict[str, Any]: ...

    def interactive_init_model_document(
        self,
        *,
        name: str,
        template: str | None = None,
    ) -> dict[str, Any]: ...

    def validate_model_document(
        self,
        *,
        model_name: str,
        document: Mapping[str, Any],
    ) -> list[str]: ...

    def summarize_model_document(
        self,
        *,
        model_name: str,
        document: Mapping[str, Any],
    ) -> str: ...

    def rename_model_document(
        self,
        *,
        new_name: str,
        document: Mapping[str, Any],
    ) -> dict[str, Any]: ...


def load_bids_tool_adapter(preprocessing: Mapping[str, Any]) -> BidsToolAdapter:
    adapter_ref = str(preprocessing.get("tool_adapter", "")).strip()
    if not adapter_ref:
        raise ValueError("config/preprocessing.yaml must define preprocessing.tool_adapter for the BIDS slice.")
    return _load_adapter(
        adapter_ref,
        required_methods=(
            "tool_name",
            "requires_input_derivative",
            "supported_input_derivatives",
            "validate_project",
            "discover_batch_rows",
            "expected_remote_input_files",
            "expected_remote_auxiliary_files",
            "runtime_metadata",
            "build_runtime_plan",
            "sync_entries",
            "build_publish_back_scaffold",
            "scaffold_project_defaults",
        ),
        ref_label="preprocessing.tool_adapter",
    )


def load_bids_analysis_tool_adapter(tool_entry: Mapping[str, Any]) -> BidsAnalysisToolAdapter:
    adapter_ref = str(tool_entry.get("adapter", "")).strip()
    if not adapter_ref:
        raise ValueError("config/analysis.yaml must define analysis.tools.<tool>.adapter for the BIDS analysis slice.")
    return _load_adapter(
        adapter_ref,
        required_methods=(
            "tool_name",
            "validate_project",
            "discover_batch_rows",
            "expected_remote_input_files",
            "runtime_metadata",
            "sync_entries",
            "build_runtime_plan",
            "build_publish_back_scaffold",
            "scaffold_project_defaults",
        ),
        ref_label="analysis.tools.<tool>.adapter",
    )


def registered_bids_tools() -> tuple[str, ...]:
    return tuple(sorted(_BIDS_TOOL_REGISTRY))


def resolve_bids_tool_adapter_ref(tool_name: str) -> str:
    normalized = str(tool_name).strip()
    adapter_ref = _BIDS_TOOL_REGISTRY.get(normalized)
    if adapter_ref:
        return adapter_ref
    supported = ", ".join(registered_bids_tools())
    raise ValueError(f"Unsupported BIDS tool {normalized!r}. Supported tools: {supported}.")


def load_registered_bids_tool_adapter(tool_name: str) -> BidsToolAdapter:
    adapter_ref = resolve_bids_tool_adapter_ref(tool_name)
    return load_bids_tool_adapter({"tool_adapter": adapter_ref})


def registered_bids_analysis_tools() -> tuple[str, ...]:
    return tuple(sorted(_BIDS_ANALYSIS_TOOL_REGISTRY))


def resolve_bids_analysis_tool_adapter_ref(tool_name: str) -> str:
    normalized = str(tool_name).strip()
    adapter_ref = _BIDS_ANALYSIS_TOOL_REGISTRY.get(normalized)
    if adapter_ref:
        return adapter_ref
    supported = ", ".join(registered_bids_analysis_tools())
    raise ValueError(f"Unsupported BIDS analysis tool {normalized!r}. Supported tools: {supported}.")


def load_registered_bids_analysis_tool_adapter(tool_name: str) -> BidsAnalysisToolAdapter:
    adapter_ref = resolve_bids_analysis_tool_adapter_ref(tool_name)
    return load_bids_analysis_tool_adapter({"adapter": adapter_ref})


def require_bids_analysis_model_authoring_adapter(
    adapter: BidsAnalysisToolAdapter,
    *,
    tool_name: str,
) -> BidsAnalysisModelAuthoringAdapter:
    required_methods = (
        "init_model_document",
        "interactive_init_model_document",
        "validate_model_document",
        "summarize_model_document",
        "rename_model_document",
    )
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing:
        raise ValueError(
            f"Analysis tool {tool_name!r} does not expose model authoring support. Missing methods: {', '.join(missing)}."
        )
    return adapter  # type: ignore[return-value]


def validate_tool_options_shape(preprocessing: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    tool_options = preprocessing.get("tool_options")
    if tool_options is None:
        return errors
    if not isinstance(tool_options, Mapping):
        return ["config/preprocessing.yaml preprocessing.tool_options must be a mapping."]

    for key in tool_options:
        normalized = str(key).strip().lower()
        if normalized in FORBIDDEN_TOOL_OPTION_KEYS:
            errors.append(
                f"preprocessing.tool_options must not define compute resources such as {key!r}; keep CPU/memory under compute."
            )
    return errors


def _instantiate_adapter(candidate: Any) -> Any:
    if isinstance(candidate, type):
        return candidate()
    if callable(candidate) and not _looks_like_adapter(candidate):
        return candidate()
    return candidate


def _load_adapter(
    adapter_ref: str,
    *,
    required_methods: tuple[str, ...],
    ref_label: str,
) -> Any:
    module_name, separator, attribute_name = adapter_ref.partition(":")
    if separator != ":" or not module_name or not attribute_name:
        raise ValueError(f"{ref_label} must use the form 'package.module:AdapterClass'.")

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised indirectly
        raise ValueError(f"Unable to import {ref_label} module {module_name!r}: {exc}") from exc

    try:
        candidate = getattr(module, attribute_name)
    except AttributeError as exc:  # pragma: no cover - exercised indirectly
        raise ValueError(
            f"Unable to resolve {ref_label} attribute {attribute_name!r} from {module_name!r}."
        ) from exc

    adapter = _instantiate_adapter(candidate)
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing:
        raise ValueError(
            f"Loaded adapter {adapter_ref!r} is missing required methods: {', '.join(missing)}."
        )
    return adapter


def _looks_like_adapter(candidate: Any) -> bool:
    return callable(getattr(candidate, "tool_name", None))
