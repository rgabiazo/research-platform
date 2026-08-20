from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import importlib
import math
from typing import Any, ClassVar

from .contracts import (
    CrossValidationSpec,
    DistanceEstimate,
    DistanceRequest,
    METRIC_CROSSNOBIS,
    NOISE_NORMALIZATION_DIAGONAL,
    NOISE_NORMALIZATION_IDENTITY,
    PatternDataset,
)
from .cv import validate_cross_validation


_CONDITION_DESCRIPTOR = "condition"
_CV_DESCRIPTOR = "cv_unit"
_CHANNEL_DESCRIPTOR = "feature_index"


@dataclass(frozen=True)
class _OptionalDependencies:
    np: Any
    data: Any
    rdm: Any


@dataclass(frozen=True)
class _CellMean:
    vector: tuple[float, ...]
    observation_count: int


@dataclass(frozen=True)
class _Pair:
    index_a: int
    index_b: int
    condition_id_a: str
    condition_id_b: str


@dataclass(frozen=True)
class RsatoolboxDistanceEngine:
    """Optional rsatoolbox-backed crossnobis adapter for MVPA distances."""

    diagonal_variances: Sequence[float] | None = None

    name: ClassVar[str] = "rsatoolbox"
    supported_metrics: ClassVar[frozenset[str]] = frozenset({METRIC_CROSSNOBIS})

    def compute_distances(
        self,
        dataset: PatternDataset,
        request: DistanceRequest,
    ) -> tuple[DistanceEstimate, ...]:
        if request.metric not in self.supported_metrics:
            supported = ", ".join(sorted(self.supported_metrics))
            raise ValueError(f"Unsupported distance metric {request.metric!r}. Use one of: {supported}.")
        if not request.cross_validation.require_balanced_units:
            raise ValueError("RsatoolboxDistanceEngine requires balanced CV units in Phase 1D.")

        resolved_condition_order = validate_cross_validation(
            dataset,
            CrossValidationSpec(
                minimum_units=request.cross_validation.minimum_units,
                require_balanced_units=True,
            ),
            condition_order=request.condition_order,
        )
        cv_units = _ordered_cv_units(dataset)
        cells = _cell_means(dataset)
        feature_count = len(dataset.feature_names)
        diagonal_variances = self._resolve_diagonal_variances(
            request.noise_normalization.method,
            feature_count=feature_count,
        )

        deps = _require_rsatoolbox()
        try:
            rsa_dataset = _to_rsatoolbox_dataset(
                deps,
                cells,
                condition_order=resolved_condition_order,
                cv_units=cv_units,
                feature_count=feature_count,
            )
            noise = _noise_precision(deps, diagonal_variances, feature_count=feature_count)
            rsa_rdms = deps.rdm.calc_rdm(
                rsa_dataset,
                method=METRIC_CROSSNOBIS,
                descriptor=_CONDITION_DESCRIPTOR,
                noise=noise,
                cv_descriptor=_CV_DESCRIPTOR,
            )
            matrix = _rdm_matrix(rsa_rdms, condition_count=len(resolved_condition_order))
        except Exception as exc:  # pragma: no cover - exercised when third-party APIs drift.
            raise RuntimeError(
                "rsatoolbox crossnobis computation failed in the optional MVPA adapter. "
                "The adapter did not fall back to native_reference."
            ) from exc

        estimates: list[DistanceEstimate] = []
        for pair in _condition_pairs(resolved_condition_order):
            distance = _finite_distance(
                float(matrix[pair.index_a][pair.index_b]) * feature_count,
                label="rsatoolbox crossnobis distance",
            )
            estimates.append(
                DistanceEstimate(
                    condition_id_a=pair.condition_id_a,
                    condition_id_b=pair.condition_id_b,
                    distance=distance,
                    metric=METRIC_CROSSNOBIS,
                    engine_name=self.name,
                    normalization_method=request.noise_normalization.method,
                    cv_unit_count=len(cv_units),
                    context={
                        "condition_index_a": pair.index_a,
                        "condition_index_b": pair.index_b,
                        "feature_count": feature_count,
                        "observation_count": _pair_observation_count(cells, pair, cv_units=cv_units),
                    },
                )
            )
        return tuple(estimates)

    def _resolve_diagonal_variances(
        self,
        normalization_method: str,
        *,
        feature_count: int,
    ) -> tuple[float, ...] | None:
        if normalization_method == NOISE_NORMALIZATION_IDENTITY:
            return None
        if normalization_method != NOISE_NORMALIZATION_DIAGONAL:
            raise ValueError(f"Unsupported crossnobis normalization method {normalization_method!r}.")
        if self.diagonal_variances is None:
            raise ValueError("diagonal_variances are required for diagonal crossnobis normalization.")
        return _validated_diagonal_variances(self.diagonal_variances, feature_count=feature_count)


def _require_rsatoolbox() -> _OptionalDependencies:
    try:
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "Optional dependency 'numpy' is required to use RsatoolboxDistanceEngine. "
            "The adapter did not fall back to native_reference."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Optional dependency 'numpy' could not be imported for RsatoolboxDistanceEngine. "
            "The adapter did not fall back to native_reference."
        ) from exc

    try:
        importlib.import_module("rsatoolbox")
        data = importlib.import_module("rsatoolbox.data")
        rdm = importlib.import_module("rsatoolbox.rdm")
    except ImportError as exc:
        raise RuntimeError(
            "Optional dependency 'rsatoolbox' is required to use RsatoolboxDistanceEngine. "
            "The adapter did not fall back to native_reference."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Optional dependency 'rsatoolbox' could not be imported for RsatoolboxDistanceEngine. "
            "The adapter did not fall back to native_reference."
        ) from exc

    if not hasattr(data, "Dataset") or not hasattr(rdm, "calc_rdm"):
        raise RuntimeError("Installed rsatoolbox does not provide the API expected by Phase 1D.")

    return _OptionalDependencies(np=np, data=data, rdm=rdm)


def _to_rsatoolbox_dataset(
    deps: _OptionalDependencies,
    cells: dict[tuple[str, str], _CellMean],
    *,
    condition_order: Sequence[str],
    cv_units: Sequence[str],
    feature_count: int,
) -> Any:
    condition_index = {condition_id: index for index, condition_id in enumerate(condition_order)}
    cv_index = {cv_unit: index for index, cv_unit in enumerate(cv_units)}

    measurements: list[tuple[float, ...]] = []
    condition_descriptors: list[int] = []
    cv_descriptors: list[int] = []
    for cv_unit in cv_units:
        for condition_id in condition_order:
            cell = cells[(condition_id, cv_unit)]
            measurements.append(cell.vector)
            condition_descriptors.append(condition_index[condition_id])
            cv_descriptors.append(cv_index[cv_unit])

    return deps.data.Dataset(
        measurements=deps.np.asarray(measurements, dtype=float),
        descriptors={"source": "research-platform"},
        obs_descriptors={
            _CONDITION_DESCRIPTOR: deps.np.asarray(condition_descriptors, dtype=int),
            _CV_DESCRIPTOR: deps.np.asarray(cv_descriptors, dtype=int),
        },
        channel_descriptors={_CHANNEL_DESCRIPTOR: deps.np.arange(feature_count, dtype=int)},
    )


def _noise_precision(
    deps: _OptionalDependencies,
    diagonal_variances: Sequence[float] | None,
    *,
    feature_count: int,
) -> Any:
    if diagonal_variances is None:
        return None
    precision = deps.np.zeros((feature_count, feature_count), dtype=float)
    for index, variance in enumerate(diagonal_variances):
        precision[index, index] = 1.0 / variance
    return precision


def _rdm_matrix(rsa_rdms: Any, *, condition_count: int) -> Any:
    get_matrices = getattr(rsa_rdms, "get_matrices", None)
    if not callable(get_matrices):
        raise RuntimeError("rsatoolbox returned an unexpected result shape.")

    matrices = get_matrices()
    shape = getattr(matrices, "shape", None)
    expected_shape = (1, condition_count, condition_count)
    if shape != expected_shape:
        raise RuntimeError(f"rsatoolbox returned an unexpected RDM matrix shape {shape!r}; expected {expected_shape!r}.")
    return matrices[0]


def _cell_means(dataset: PatternDataset) -> dict[tuple[str, str], _CellMean]:
    sums_by_cell: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
    counts_by_cell: dict[tuple[str, str], int] = {}

    for observation in dataset.observations:
        key = (observation.condition_id, observation.cv_unit)
        if key not in sums_by_cell:
            sums_by_cell[key] = [0.0] * len(observation.features)
            counts_by_cell[key] = 0
        for index, value in enumerate(observation.features):
            sums_by_cell[key][index] += value
        counts_by_cell[key] += 1

    return {
        key: _CellMean(
            vector=tuple(value / counts_by_cell[key] for value in feature_sums),
            observation_count=counts_by_cell[key],
        )
        for key, feature_sums in sums_by_cell.items()
    }


def _ordered_cv_units(dataset: PatternDataset) -> tuple[str, ...]:
    units: OrderedDict[str, None] = OrderedDict()
    for observation in dataset.observations:
        units[observation.cv_unit] = None
    return tuple(units)


def _condition_pairs(condition_ids: Sequence[str]) -> tuple[_Pair, ...]:
    return tuple(
        _Pair(
            index_a=index_a,
            index_b=index_b,
            condition_id_a=condition_ids[index_a],
            condition_id_b=condition_ids[index_b],
        )
        for index_a in range(len(condition_ids))
        for index_b in range(index_a + 1, len(condition_ids))
    )


def _pair_observation_count(
    cells: dict[tuple[str, str], _CellMean],
    pair: _Pair,
    *,
    cv_units: Sequence[str],
) -> int:
    return sum(
        cells[(condition_id, cv_unit)].observation_count
        for cv_unit in cv_units
        for condition_id in (pair.condition_id_a, pair.condition_id_b)
    )


def _validated_diagonal_variances(
    diagonal_variances: Sequence[float],
    *,
    feature_count: int,
) -> tuple[float, ...]:
    if isinstance(diagonal_variances, (str, bytes)):
        raise ValueError("diagonal_variances must contain positive finite numeric values.")
    try:
        raw_variances = tuple(diagonal_variances)
    except TypeError as exc:
        raise ValueError("diagonal_variances must contain positive finite numeric values.") from exc

    if len(raw_variances) != feature_count:
        raise ValueError(
            f"diagonal_variances length must match feature count ({len(raw_variances)} != {feature_count})."
        )

    variances: list[float] = []
    for variance in raw_variances:
        if isinstance(variance, bool):
            raise ValueError("diagonal_variances must contain positive finite numeric values.")
        try:
            numeric = float(variance)
        except (TypeError, ValueError) as exc:
            raise ValueError("diagonal_variances must contain positive finite numeric values.") from exc
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("diagonal_variances must contain positive finite numeric values.")
        variances.append(numeric)
    return tuple(variances)


def _finite_distance(value: float, *, label: str) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"{label} returned by rsatoolbox must be finite.")
    return value


__all__ = ["RsatoolboxDistanceEngine"]
