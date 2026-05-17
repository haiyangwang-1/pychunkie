"""Equations and integral-system records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .config import SystemConfig
from .density import DensitySpace
from .matrix import SystemMatrix


@dataclass(frozen=True)
class BoundaryEquation:
    name: str
    target: object
    terms: tuple[object, ...]
    rhs: object


@dataclass(frozen=True)
class ConstraintTerm:
    density: str
    coefficients: object
    component: int | None = None


@dataclass(frozen=True)
class Constraint:
    name: str
    terms: tuple[ConstraintTerm, ...]
    value: complex = 0.0


@dataclass(frozen=True)
class IntegralSystem:
    name: str
    geometry: object
    unknowns: tuple[DensitySpace, ...]
    equations: tuple[BoundaryEquation, ...]
    constraints: tuple[object, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)

    def assemble(self, *, config: SystemConfig | None = None) -> SystemMatrix:
        from .assembly import assemble_system_matrix

        return assemble_system_matrix(self, config=SystemConfig() if config is None else config)

    def solve(self, *, config: SystemConfig | None = None):
        from .solvers import solve_system

        return solve_system(self, config=SystemConfig() if config is None else config)
