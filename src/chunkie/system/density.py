"""Density storage and solver-vector layout adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

DensityLayout = Literal["component_panel_major"]


@dataclass(frozen=True)
class DensitySpace:
    name: str
    geometry: object
    component_count: int = 1
    layout: DensityLayout = "component_panel_major"


@dataclass
class Density:
    name: str
    geometry: object
    values: NDArray[np.generic]
    component_count: int = 1
    layout: DensityLayout = "component_panel_major"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        expected = _geometry_shape(self.geometry)
        if self.values.ndim == 2:
            if self.component_count != 1:
                raise ValueError("two-dimensional density values imply component_count=1")
            if self.values.shape != expected:
                raise ValueError("scalar density values must have shape (local_node, panel)")
        elif self.values.ndim == 3:
            if self.values.shape != (self.component_count, *expected):
                raise ValueError("component density values must have shape (component, local_node, panel)")
        else:
            raise ValueError("density values must be rank 2 or rank 3")

    @property
    def component_values(self) -> NDArray[np.generic]:
        if self.values.ndim == 2:
            return self.values[None, :, :]
        return self.values

    def to_vector(self) -> NDArray[np.generic]:
        values = self.component_values
        # Solver vectors are component-major over panel-major points. This is
        # an explicit adapter boundary, not a memory-order convention.
        return values.swapaxes(1, 2).reshape(-1)

    @classmethod
    def from_vector(
        cls,
        name: str,
        geometry: object,
        vector: NDArray[np.generic],
        *,
        component_count: int = 1,
    ) -> Density:
        local_node_count, panel_count = _geometry_shape(geometry)
        arr = np.asarray(vector)
        expected_size = component_count * local_node_count * panel_count
        if arr.size != expected_size:
            raise ValueError("density vector has incompatible size")
        values = arr.reshape(component_count, panel_count, local_node_count).swapaxes(1, 2)
        if component_count == 1:
            return cls(name, geometry, values[0], component_count=1)
        return cls(name, geometry, values, component_count=component_count)


def _geometry_shape(geometry: object) -> tuple[int, int]:
    if not hasattr(geometry, "quadrature_order") or not hasattr(geometry, "panel_count"):
        raise TypeError("density geometry must expose quadrature_order and panel_count")
    return int(geometry.quadrature_order), int(geometry.panel_count)
