from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_platform.analysis.mvpa import (
    CrossValidationSpec,
    DistanceEstimate,
    DistanceRequest,
    METRIC_CORRELATION,
    METRIC_CROSSNOBIS,
    METRIC_EUCLIDEAN,
    NOISE_NORMALIZATION_DIAGONAL,
    NativeReferenceDistanceEngine,
    NoiseNormalization,
    pattern_dataset_from_rows,
)
from research_platform.analysis.mvpa import rsatoolbox_adapter
from research_platform.analysis.mvpa.rsatoolbox_adapter import RsatoolboxDistanceEngine


def _dataset(rows: list[dict[str, object]], feature_columns: list[str] | None = None):
    return pattern_dataset_from_rows(rows, feature_columns=feature_columns or ["f1", "f2"])


def _phase_1b_identity_dataset():
    return _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 2},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 3, "f2": 4},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 0, "f2": 0},
        ]
    )


def _valid_request() -> DistanceRequest:
    return DistanceRequest(metric=METRIC_CROSSNOBIS)


def test_import_guard_does_not_require_rsatoolbox_at_module_import_time() -> None:
    module = importlib.import_module("research_platform.analysis.mvpa.rsatoolbox_adapter")

    assert module.RsatoolboxDistanceEngine.name == "rsatoolbox"
    assert module.RsatoolboxDistanceEngine.supported_metrics == {METRIC_CROSSNOBIS}


@pytest.mark.parametrize("missing_module", ["numpy", "rsatoolbox"])
def test_missing_optional_dependency_raises_runtime_error(monkeypatch: pytest.MonkeyPatch, missing_module: str) -> None:
    dataset = _phase_1b_identity_dataset()
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if missing_module == "numpy" and name == "numpy":
            raise ImportError("simulated missing numpy")
        if missing_module == "rsatoolbox":
            if name == "numpy":
                return SimpleNamespace()
            if name.startswith("rsatoolbox"):
                raise ImportError("simulated missing rsatoolbox")
        return real_import_module(name)

    monkeypatch.setattr(rsatoolbox_adapter.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError) as excinfo:
        RsatoolboxDistanceEngine().compute_distances(dataset, _valid_request())

    message = str(excinfo.value)
    assert missing_module in message
    assert "native_reference" in message


@pytest.mark.parametrize("metric", [METRIC_EUCLIDEAN, METRIC_CORRELATION, "unknown"])
def test_unsupported_metrics_raise_before_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
) -> None:
    def fail_import_module(name: str):
        raise AssertionError(f"optional import {name!r} should not be required")

    monkeypatch.setattr(rsatoolbox_adapter.importlib, "import_module", fail_import_module)

    with pytest.raises(ValueError, match="Unsupported distance metric"):
        RsatoolboxDistanceEngine().compute_distances(
            _phase_1b_identity_dataset(),
            DistanceRequest(metric=metric),
        )


def test_dependency_error_does_not_leak_public_rsatoolbox_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str):
        if name == "numpy":
            return SimpleNamespace()
        if name.startswith("rsatoolbox"):
            raise ImportError("simulated missing rsatoolbox")
        return importlib.import_module(name)

    monkeypatch.setattr(rsatoolbox_adapter.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError) as excinfo:
        RsatoolboxDistanceEngine().compute_distances(_phase_1b_identity_dataset(), _valid_request())

    message = str(excinfo.value)
    assert "Dataset" not in message
    assert "RDM" not in message
    assert "rsatoolbox.data" not in message
    assert "rsatoolbox.rdm" not in message


def test_validation_errors_raise_before_optional_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import_module(name: str):
        raise AssertionError(f"optional import {name!r} should not be required")

    monkeypatch.setattr(rsatoolbox_adapter.importlib, "import_module", fail_import_module)
    dataset = _phase_1b_identity_dataset()

    with pytest.raises(ValueError, match="requires balanced CV units"):
        RsatoolboxDistanceEngine().compute_distances(
            dataset,
            DistanceRequest(
                metric=METRIC_CROSSNOBIS,
                cross_validation=CrossValidationSpec(require_balanced_units=False),
            ),
        )
    with pytest.raises(ValueError, match="missing observations"):
        RsatoolboxDistanceEngine().compute_distances(
            _dataset(
                [
                    {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 0},
                    {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
                    {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 1, "f2": 0},
                ]
            ),
            _valid_request(),
        )
    with pytest.raises(ValueError, match="diagonal_variances"):
        RsatoolboxDistanceEngine().compute_distances(
            dataset,
            DistanceRequest(
                metric=METRIC_CROSSNOBIS,
                noise_normalization=NoiseNormalization(method=NOISE_NORMALIZATION_DIAGONAL),
            ),
        )


def test_adapter_has_no_forbidden_imports() -> None:
    source = Path(rsatoolbox_adapter.__file__).read_text()
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    forbidden_modules = (
        "numpy",
        "rsatoolbox",
        "pandas",
        "polars",
        "scipy",
        "nilearn",
        "pymvpa",
        "sklearn",
        "research_platform.neuro",
        "research_platform.bids",
        "research_platform.viz",
        "research_platform.core",
        "research_platform.ml",
    )
    for imported_module in imported_modules:
        assert not any(
            imported_module == forbidden or imported_module.startswith(f"{forbidden}.")
            for forbidden in forbidden_modules
        )


def _require_optional_rsatoolbox() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("rsatoolbox")


def _assert_native_parity(
    rsa_estimates: tuple[DistanceEstimate, ...],
    native_estimates: tuple[DistanceEstimate, ...],
) -> None:
    assert all(isinstance(estimate, DistanceEstimate) for estimate in rsa_estimates)
    assert len(rsa_estimates) == len(native_estimates)
    for rsa_estimate, native_estimate in zip(rsa_estimates, native_estimates, strict=True):
        assert rsa_estimate.condition_id_a == native_estimate.condition_id_a
        assert rsa_estimate.condition_id_b == native_estimate.condition_id_b
        assert rsa_estimate.distance == pytest.approx(native_estimate.distance)
        assert rsa_estimate.metric == native_estimate.metric
        assert rsa_estimate.engine_name == "rsatoolbox"
        assert rsa_estimate.normalization_method == native_estimate.normalization_method
        assert rsa_estimate.cv_unit_count == native_estimate.cv_unit_count
        assert rsa_estimate.context == native_estimate.context


def test_optional_identity_crossnobis_parity_with_native_reference() -> None:
    _require_optional_rsatoolbox()
    dataset = _phase_1b_identity_dataset()
    request = _valid_request()

    rsa_estimates = RsatoolboxDistanceEngine().compute_distances(dataset, request)
    native_estimates = NativeReferenceDistanceEngine().compute_distances(dataset, request)

    _assert_native_parity(rsa_estimates, native_estimates)


def test_optional_negative_crossnobis_parity_and_preserves_negative_value() -> None:
    _require_optional_rsatoolbox()
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 0},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": -2, "f2": 0},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 0, "f2": 0},
        ]
    )
    request = _valid_request()

    rsa_estimates = RsatoolboxDistanceEngine().compute_distances(dataset, request)
    native_estimates = NativeReferenceDistanceEngine().compute_distances(dataset, request)

    _assert_native_parity(rsa_estimates, native_estimates)
    assert rsa_estimates[0].distance < 0


def test_optional_multiple_observations_are_averaged_like_native_reference() -> None:
    _require_optional_rsatoolbox()
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1, "f2": 1},
            {"pattern_id": "p2", "condition_id": "a", "cv_unit": "run-1", "f1": 3, "f2": 3},
            {"pattern_id": "p3", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p4", "condition_id": "a", "cv_unit": "run-2", "f1": 4, "f2": 0},
            {"pattern_id": "p5", "condition_id": "b", "cv_unit": "run-2", "f1": 1, "f2": 0},
        ]
    )
    request = _valid_request()

    rsa_estimates = RsatoolboxDistanceEngine().compute_distances(dataset, request)
    native_estimates = NativeReferenceDistanceEngine().compute_distances(dataset, request)

    _assert_native_parity(rsa_estimates, native_estimates)


def test_optional_explicit_non_alphabetical_condition_order_controls_pair_order() -> None:
    _require_optional_rsatoolbox()
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "z", "cv_unit": "run-1", "f1": 1, "f2": 0},
            {"pattern_id": "p2", "condition_id": "a", "cv_unit": "run-1", "f1": 0, "f2": 1},
            {"pattern_id": "p3", "condition_id": "m", "cv_unit": "run-1", "f1": 2, "f2": 1},
            {"pattern_id": "p4", "condition_id": "z", "cv_unit": "run-2", "f1": 2, "f2": 0},
            {"pattern_id": "p5", "condition_id": "a", "cv_unit": "run-2", "f1": 0, "f2": 2},
            {"pattern_id": "p6", "condition_id": "m", "cv_unit": "run-2", "f1": 3, "f2": 1},
        ]
    )
    request = DistanceRequest(metric=METRIC_CROSSNOBIS, condition_order=["m", "z", "a"])

    rsa_estimates = RsatoolboxDistanceEngine().compute_distances(dataset, request)
    native_estimates = NativeReferenceDistanceEngine().compute_distances(dataset, request)

    assert [(estimate.condition_id_a, estimate.condition_id_b) for estimate in rsa_estimates] == [
        ("m", "z"),
        ("m", "a"),
        ("z", "a"),
    ]
    _assert_native_parity(rsa_estimates, native_estimates)


def test_optional_diagonal_variance_parity_with_native_reference() -> None:
    _require_optional_rsatoolbox()
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 2, "f2": 4},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 0, "f2": 0},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 6, "f2": 8},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 0, "f2": 0},
        ]
    )
    request = DistanceRequest(
        metric=METRIC_CROSSNOBIS,
        noise_normalization=NoiseNormalization(method=NOISE_NORMALIZATION_DIAGONAL),
    )

    rsa_engine = RsatoolboxDistanceEngine(diagonal_variances=[2, 4])
    native_engine = NativeReferenceDistanceEngine(diagonal_variances=[2, 4])
    try:
        rsa_estimates = rsa_engine.compute_distances(dataset, request)
    except RuntimeError as exc:
        pytest.skip(f"rsatoolbox diagonal precision input is unavailable: {exc}")
    native_estimates = native_engine.compute_distances(dataset, request)

    _assert_native_parity(rsa_estimates, native_estimates)
