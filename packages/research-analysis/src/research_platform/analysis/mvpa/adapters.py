from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Protocol, runtime_checkable

from .contracts import DistanceEstimate, DistanceRequest, PatternDataset


@runtime_checkable
class DistanceEngine(Protocol):
    name: str
    supported_metrics: Collection[str]

    def compute_distances(
        self,
        dataset: PatternDataset,
        request: DistanceRequest,
    ) -> Iterable[DistanceEstimate]:
        ...


__all__ = ["DistanceEngine"]
