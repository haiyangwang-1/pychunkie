"""Kernel object and geometry adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .singularities import SingularityInfo

Evaluator = Callable[[Any, Any], NDArray[np.generic]]


@dataclass(frozen=True)
class Kernel:
    family: str
    selector: str
    params: Mapping[str, Any]
    input_dim: int
    output_dim: int
    singularity: SingularityInfo
    evaluator: Evaluator

    def __call__(self, source: Any, target: Any) -> NDArray[np.generic]:
        values = np.asarray(self.evaluator(source, target))
        expected_prefix = (self.output_dim, self.input_dim)
        if values.ndim != 4 or values.shape[:2] != expected_prefix:
            raise ValueError(
                "kernel evaluators must return values with shape "
                "(output_dim, input_dim, target_point_count, source_point_count)"
            )
        return values

    def __mul__(self, factor: complex) -> Kernel:
        from .algebra import scale

        return scale(self, factor)

    def __rmul__(self, factor: complex) -> Kernel:
        return self.__mul__(factor)

    def __neg__(self) -> Kernel:
        return self.__mul__(-1.0)

    def __add__(self, other: Kernel) -> Kernel:
        from .algebra import add

        return add(self, other)


def flat_positions(points: Any) -> NDArray[np.floating]:
    if hasattr(points, "flat_positions"):
        return np.asarray(points.flat_positions, dtype=float)
    if hasattr(points, "positions"):
        positions = np.asarray(points.positions, dtype=float)
        if positions.ndim == 3:
            return positions.swapaxes(1, 2).reshape(positions.shape[0], -1)
        return positions.reshape(positions.shape[0], -1)
    arr = np.asarray(points, dtype=float)
    return arr.reshape(arr.shape[0], -1)


def flat_normals(points: Any, *, label: str) -> NDArray[np.floating]:
    if hasattr(points, "flat_normals"):
        return np.asarray(points.flat_normals, dtype=float)
    if hasattr(points, "normals"):
        normals = np.asarray(points.normals, dtype=float)
        if normals.ndim == 3:
            return normals.swapaxes(1, 2).reshape(normals.shape[0], -1)
        return normals.reshape(normals.shape[0], -1)
    raise ValueError(f"{label} normals are required for this kernel selector")
