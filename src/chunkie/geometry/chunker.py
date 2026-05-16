"""Panel-major curve discretization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .points import PanelView, PointInfoView, PointMap


@dataclass
class Chunker:
    """One oriented curve discretized into Legendre panels."""

    positions: NDArray[np.floating]
    derivatives: NDArray[np.floating]
    second_derivatives: NDArray[np.floating]
    normals: NDArray[np.floating]
    weights: NDArray[np.floating]
    nodes: NDArray[np.floating]
    reference_weights: NDArray[np.floating]
    adjacency: NDArray[np.integer]
    closed: bool
    orientation: Literal["ccw", "cw", "open"] | None = None
    vertices: NDArray[np.floating] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.derivatives = np.asarray(self.derivatives, dtype=float)
        self.second_derivatives = np.asarray(self.second_derivatives, dtype=float)
        self.normals = np.asarray(self.normals, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.reference_weights = np.asarray(self.reference_weights, dtype=float)
        self.adjacency = np.asarray(self.adjacency, dtype=np.int64)

        if self.positions.ndim != 3:
            raise ValueError("positions must have shape (coordinate_dim, quadrature_order, panel_count)")
        if self.derivatives.shape != self.positions.shape:
            raise ValueError("derivatives must match positions")
        if self.second_derivatives.shape != self.positions.shape:
            raise ValueError("second_derivatives must match positions")
        if self.normals.shape != self.positions.shape:
            raise ValueError("normals must match positions")
        if self.weights.shape != self.positions.shape[1:]:
            raise ValueError("weights must have shape (quadrature_order, panel_count)")
        if self.nodes.shape != (self.quadrature_order,):
            raise ValueError("nodes must have shape (quadrature_order,)")
        if self.reference_weights.shape != (self.quadrature_order,):
            raise ValueError("reference_weights must have shape (quadrature_order,)")
        if self.adjacency.shape != (2, self.panel_count):
            raise ValueError("adjacency must have shape (2, panel_count)")

    @property
    def coordinate_dim(self) -> int:
        return int(self.positions.shape[0])

    @property
    def quadrature_order(self) -> int:
        return int(self.positions.shape[1])

    @property
    def panel_count(self) -> int:
        return int(self.positions.shape[2])

    @property
    def point_count(self) -> int:
        return self.quadrature_order * self.panel_count

    @property
    def point_map(self) -> PointMap:
        return PointMap(self.quadrature_order, self.panel_count)

    @property
    def pointinfo(self) -> PointInfoView:
        return PointInfoView(
            positions=self.positions,
            derivatives=self.derivatives,
            second_derivatives=self.second_derivatives,
            normals=self.normals,
            weights=self.weights,
            nodes=self.nodes,
            panel_ids=np.arange(self.panel_count, dtype=np.int64),
            point_map=self.point_map,
        )

    def panel(self, panel_id: int) -> PanelView:
        if not 0 <= panel_id < self.panel_count:
            raise IndexError("panel_id out of range")
        return PanelView(
            parent=self,
            panel_id=panel_id,
            positions=self.positions[:, :, panel_id],
            derivatives=self.derivatives[:, :, panel_id],
            second_derivatives=self.second_derivatives[:, :, panel_id],
            normals=self.normals[:, :, panel_id],
            weights=self.weights[:, panel_id],
            nodes=self.nodes,
        )

    def translated(self, vector: ArrayLike) -> Chunker:
        offset = np.asarray(vector, dtype=float).reshape(self.coordinate_dim, 1, 1)
        return replace(self, positions=self.positions + offset)

    def scaled(self, factor: float, *, center: ArrayLike | None = None) -> Chunker:
        center_array = (
            np.zeros((self.coordinate_dim, 1, 1))
            if center is None
            else np.asarray(center, dtype=float).reshape(self.coordinate_dim, 1, 1)
        )
        scale = float(factor)
        return replace(
            self,
            positions=center_array + scale * (self.positions - center_array),
            derivatives=scale * self.derivatives,
            second_derivatives=scale * self.second_derivatives,
            weights=abs(scale) * self.weights,
        )


def right_normals(derivatives: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return right normals for 2D curve derivatives.

    For a counter-clockwise outer boundary this convention points exterior,
    which is the orientation convention used by traces and system jumps.
    """

    if derivatives.shape[0] != 2:
        raise ValueError("right_normals currently supports two-dimensional curves")
    speed = np.linalg.norm(derivatives, axis=0)
    normals = np.empty_like(derivatives)
    normals[0] = derivatives[1] / speed
    normals[1] = -derivatives[0] / speed
    return normals
