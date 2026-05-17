"""Operational singularity metadata.

The metadata is based on canonical 2D Laplace singular basis functions. Kernel
families express their local singular part as tensor coefficients multiplying
these bases; quadrature and backend code can then consume one representation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Coefficient = complex | float | NDArray[np.generic] | Callable[[Any, Any], NDArray[np.generic]]
Regularity = Literal["smooth", "c1", "continuous", "bounded", "unknown"]
BoundaryLimit = Literal["smooth", "removable", "pv", "hs", "supersingular"]


@dataclass(frozen=True)
class GeometryRequirements:
    source_normals: bool = False
    target_normals: bool = False
    source_derivatives: bool = False
    target_derivatives: bool = False


@dataclass(frozen=True)
class LaplaceBasis:
    """A differentiated Laplace log basis.

    ``derivative=()`` is ``G``; ``(a,)`` is ``d_xa G``; ``(a, b)`` is
    ``d_xa d_xb G``. Higher derivatives are intentionally not accepted until a
    selector requires and tests them.
    """

    derivative: tuple[int, ...] = ()

    def evaluate(self, source: Any, target: Any) -> NDArray[np.floating]:
        source_positions = _flat_positions(source)
        target_positions = _flat_positions(target)
        rx = target_positions[0, :, None] - source_positions[0, None, :]
        ry = target_positions[1, :, None] - source_positions[1, None, :]
        r = (rx, ry)
        rho2 = rx**2 + ry**2

        with np.errstate(divide="ignore", invalid="ignore"):
            if len(self.derivative) == 0:
                return -np.log(rho2) / (4.0 * np.pi)
            if len(self.derivative) == 1:
                a = self.derivative[0]
                return -r[a] / (2.0 * np.pi * rho2)
            if len(self.derivative) == 2:
                a, b = self.derivative
                delta = 1.0 if a == b else 0.0
                return (2.0 * r[a] * r[b] - delta * rho2) / (2.0 * np.pi * rho2**2)
        raise NotImplementedError("LaplaceBasis currently supports derivatives through second order")

    @property
    def legacy_strength(self) -> Literal["log", "pv", "hs"]:
        if len(self.derivative) == 0:
            return "log"
        if len(self.derivative) == 1:
            return "pv"
        return "hs"


@dataclass(frozen=True)
class LaplaceSingularTerm:
    basis: LaplaceBasis
    coefficient: Coefficient = 1.0
    meaning: str = ""

    def evaluate(self, source: Any, target: Any, *, output_dim: int, input_dim: int) -> NDArray[np.generic]:
        basis_values = self.basis.evaluate(source, target)
        coefficient = self.coefficient(source, target) if callable(self.coefficient) else self.coefficient
        coefficient_array = np.asarray(coefficient)

        if coefficient_array.ndim == 0:
            coefficient_array = np.full((output_dim, input_dim, 1, 1), coefficient_array)
        elif coefficient_array.ndim == 2:
            coefficient_array = coefficient_array[:, :, None, None]
        elif coefficient_array.ndim != 4:
            raise ValueError("singularity coefficient must be scalar, [out, in], or [out, in, target, source]")

        return coefficient_array * basis_values[None, None, :, :]


@dataclass(frozen=True)
class LaplaceSingularExpansion:
    input_dim: int
    output_dim: int
    terms: tuple[LaplaceSingularTerm, ...] = ()

    def evaluate(self, source: Any, target: Any) -> NDArray[np.generic]:
        source_positions = _flat_positions(source)
        target_positions = _flat_positions(target)
        dtype = complex if any(
            callable(term.coefficient) or np.iscomplexobj(term.coefficient) for term in self.terms
        ) else float
        values = np.zeros(
            (self.output_dim, self.input_dim, target_positions.shape[1], source_positions.shape[1]),
            dtype=dtype,
        )
        for term in self.terms:
            values = values + term.evaluate(source, target, output_dim=self.output_dim, input_dim=self.input_dim)
        return values

    @property
    def legacy_strength(self) -> Literal["smooth", "log", "pv", "hs", "mixed"]:
        if not self.terms:
            return "smooth"
        strengths = {term.basis.legacy_strength for term in self.terms}
        if "hs" in strengths:
            return "hs" if len(strengths) == 1 else "mixed"
        if "pv" in strengths:
            return "pv" if len(strengths) == 1 else "mixed"
        return "log"


@dataclass(frozen=True)
class SingularityInfo:
    family: str
    selector: str
    input_dim: int
    output_dim: int
    expansion: LaplaceSingularExpansion
    boundary_limit: BoundaryLimit
    remainder_regular: Regularity
    requirements: GeometryRequirements = GeometryRequirements()
    side_sensitive: bool = False
    notes: str = ""

    @classmethod
    def smooth(cls, *, family: str, selector: str, input_dim: int, output_dim: int) -> SingularityInfo:
        return cls(
            family=family,
            selector=selector,
            input_dim=input_dim,
            output_dim=output_dim,
            expansion=LaplaceSingularExpansion(input_dim=input_dim, output_dim=output_dim),
            boundary_limit="smooth",
            remainder_regular="smooth",
        )

    @property
    def legacy_strength(self) -> Literal["smooth", "log", "pv", "hs", "mixed"]:
        return self.expansion.legacy_strength


def matrix_coefficient(output_dim: int, input_dim: int, output: int, input_: int = 0, value: complex = 1.0):
    coefficient = np.zeros((output_dim, input_dim), dtype=complex if np.iscomplexobj(value) else float)
    coefficient[output, input_] = value
    return coefficient


def _flat_positions(points: Any) -> NDArray[np.floating]:
    if hasattr(points, "flat_positions"):
        return np.asarray(points.flat_positions, dtype=float)
    if hasattr(points, "positions"):
        positions = np.asarray(points.positions, dtype=float)
        if positions.ndim == 3:
            return positions.swapaxes(1, 2).reshape(positions.shape[0], -1)
        return positions.reshape(positions.shape[0], -1)
    arr = np.asarray(points, dtype=float)
    return arr.reshape(arr.shape[0], -1)
