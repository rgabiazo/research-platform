from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from ._backend_protocol import BackendProtocol, UnsupportedBackendError
from ._types import BackendName


def _polars_backend_factory() -> BackendProtocol:
    from . import polars_ops

    return polars_ops.PolarsBackend()


def _pandas_backend_factory() -> BackendProtocol:
    from . import pandas_ops

    return pandas_ops.PandasBackend()


_BACKEND_FACTORIES = {
    "polars": _polars_backend_factory,
    "pandas": _pandas_backend_factory,
}


@lru_cache(maxsize=None)
def _load_backend(name: BackendName) -> BackendProtocol:
    key = name.lower()
    if key not in _BACKEND_FACTORIES:
        raise UnsupportedBackendError(f"Unknown backend '{name}'. Supported: {sorted(_BACKEND_FACTORIES)}")

    try:
        return _BACKEND_FACTORIES[key]()
    except Exception as exc:  # pragma: no cover - dependency import guard
        raise UnsupportedBackendError(f"Backend '{name}' unavailable: {exc}") from exc


def resolve_backend(backend: str = "polars") -> BackendProtocol:
    return _load_backend(backend)


def supports_backend(backend: str) -> bool:
    try:
        _load_backend(backend)
        return True
    except UnsupportedBackendError:
        return False


def supported_formats(backend: str | None = None) -> list[str]:
    if backend is None:
        supported: set[str] = set()
        for name in _BACKEND_FACTORIES:
            if supports_backend(name):
                backend_instance = resolve_backend(name)
                supported.update(backend_instance.supported_formats)
        return sorted(supported)

    backend_instance = resolve_backend(backend)
    return sorted(backend_instance.supported_formats)


def resolve_backends() -> Iterable[str]:
    return _BACKEND_FACTORIES.keys()
