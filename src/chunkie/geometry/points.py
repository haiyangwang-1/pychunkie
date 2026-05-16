"""Point and panel views for panel-major boundary geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class PointMap:
    """Reversible map between panel-major point ids and panel/local ids."""

    quadrature_order: int
    panel_count: int

    @property
    def point_count(self) -> int:
        return self.quadrature_order * self.panel_count

    def to_point_id(self, panel_id: int | NDArray[np.integer], local_node_id: int | NDArray[np.integer]):
        panel = np.asarray(panel_id, dtype=np.int64)
        local = np.asarray(local_node_id, dtype=np.int64)
        return panel * self.quadrature_order + local

    def from_point_id(self, point_id: int | NDArray[np.integer]):
        point = np.asarray(point_id, dtype=np.int64)
        panel = point // self.quadrature_order
        local = point % self.quadrature_order
        return panel, local


@dataclass(frozen=True)
class PointInfoView:
    """Read-only view of boundary point tensors in canonical panel-major form."""

    positions: FloatArray
    derivatives: FloatArray
    second_derivatives: FloatArray
    normals: FloatArray
    weights: FloatArray
    nodes: FloatArray
    panel_ids: NDArray[np.integer]
    point_map: PointMap

    @property
    def flat_positions(self) -> FloatArray:
        """Return positions as ``positions[R, point]`` using panel-major order."""

        return self.positions.swapaxes(1, 2).reshape(self.positions.shape[0], -1)

    @property
    def flat_normals(self) -> FloatArray:
        return self.normals.swapaxes(1, 2).reshape(self.normals.shape[0], -1)

    @property
    def flat_weights(self) -> FloatArray:
        return self.weights.T.reshape(-1)


@dataclass(frozen=True)
class PanelView:
    """Geometry data for one source panel."""

    parent: object
    panel_id: int
    positions: FloatArray
    derivatives: FloatArray
    second_derivatives: FloatArray
    normals: FloatArray
    weights: FloatArray
    nodes: FloatArray
