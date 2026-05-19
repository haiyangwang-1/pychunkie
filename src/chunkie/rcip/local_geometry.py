"""Local corner geometry for RCIP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from chunkie.geometry.chunker import right_normals
from chunkie.geometry.points import PointInfoView


@dataclass(frozen=True)
class LocalCornerGeometry:
    vertex: NDArray[np.floating]
    directions: NDArray[np.floating]
    positions: NDArray[np.floating]
    derivatives: NDArray[np.floating]
    normals: NDArray[np.floating]
    weights: NDArray[np.floating]
    nodes: NDArray[np.floating]
    reference_weights: NDArray[np.floating]
    panel_scales: NDArray[np.floating]

    @property
    def quadrature_order(self) -> int:
        return int(self.nodes.size)

    @property
    def panel_count(self) -> int:
        return int(self.weights.shape[1])

    @property
    def point_count(self) -> int:
        return self.quadrature_order * self.panel_count

    @property
    def signed_curvature(self) -> NDArray[np.floating]:
        return np.zeros((self.quadrature_order, self.panel_count), dtype=float)

    @property
    def pointinfo(self) -> PointInfoView:
        second_derivatives = np.zeros_like(self.derivatives)
        return PointInfoView(
            positions=self.positions,
            derivatives=self.derivatives,
            second_derivatives=second_derivatives,
            normals=self.normals,
            weights=self.weights,
            nodes=self.nodes,
            panel_ids=np.arange(self.panel_count, dtype=np.int64),
        )


def build_local_corner_geometry(
    vertex: ArrayLike,
    directions: ArrayLike,
    *,
    quadrature_order: int = 16,
    levels: int = 8,
    base_length: float = 1.0,
) -> LocalCornerGeometry:
    """Build dyadically refined ray panels around one corner.

    ``directions`` has shape ``(2, ray_count)`` and each column points away from
    the corner. Level zero is the outermost panel; increasing levels approach
    the vertex by powers of two.
    """

    vertex_array = np.asarray(vertex, dtype=float).reshape(2)
    direction_array = np.asarray(directions, dtype=float)
    if direction_array.ndim != 2 or direction_array.shape[0] != 2:
        raise ValueError("directions must have shape (2, ray_count)")
    direction_array = direction_array / np.linalg.norm(direction_array, axis=0, keepdims=True)
    nodes, reference_weights = leggauss(int(quadrature_order))
    ray_count = direction_array.shape[1]
    panel_count = ray_count * int(levels)
    positions = np.empty((2, nodes.size, panel_count), dtype=float)
    derivatives = np.empty_like(positions)
    weights = np.empty((nodes.size, panel_count), dtype=float)
    panel_scales = np.empty(panel_count, dtype=float)

    cursor = 0
    for ray in range(ray_count):
        direction = direction_array[:, ray]
        for level in range(int(levels)):
            outer = float(base_length) * 2.0 ** (-level)
            inner = float(base_length) * 2.0 ** (-(level + 1))
            midpoint = 0.5 * (inner + outer)
            half_width = 0.5 * (outer - inner)
            radii = midpoint + half_width * nodes
            positions[:, :, cursor] = vertex_array[:, None] + direction[:, None] * radii[None, :]
            # The local panel parameter runs from the inner radius to the outer
            # radius, so dr/du is the ray direction scaled by the half-width.
            derivatives[:, :, cursor] = direction[:, None] * half_width
            weights[:, cursor] = reference_weights * half_width
            panel_scales[cursor] = outer - inner
            cursor += 1

    normals = right_normals(derivatives)
    return LocalCornerGeometry(
        vertex=vertex_array,
        directions=direction_array,
        positions=positions,
        derivatives=derivatives,
        normals=normals,
        weights=weights,
        nodes=nodes,
        reference_weights=reference_weights,
        panel_scales=panel_scales,
    )
