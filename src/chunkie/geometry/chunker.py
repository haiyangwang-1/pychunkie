"""Panel-major curve discretization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .points import PanelView, PointInfoView


@dataclass
class Chunker:
    """One oriented curve discretized into Legendre panels."""

    # Shape: (coordinate_dim, quadrature_order, panel_count).
    positions: NDArray[np.floating]
    # Shape: (coordinate_dim, quadrature_order, panel_count);
    # derivative with respect to each panel's Legendre reference coordinate.
    derivatives: NDArray[np.floating]
    # Shape: (coordinate_dim, quadrature_order, panel_count);
    # second derivative with respect to each panel's Legendre reference coordinate.
    second_derivatives: NDArray[np.floating]
    # Shape: (coordinate_dim, quadrature_order, panel_count);
    # normals[:, local_node_id, panel_id] is one normal vector.
    normals: NDArray[np.floating]
    # Shape: (quadrature_order, panel_count);
    # weights[local_node_id, panel_id] =
    #   ||derivatives[:, local_node_id, panel_id]|| * _legendre_weights[local_node_id].
    # For a smooth panel, sum(weights[:, panel_id]) is the Gauss-Legendre
    # approximation to that panel's arclength.
    weights: NDArray[np.floating]

    # Private shape: (quadrature_order,); Gauss-Legendre nodes on [-1, 1].
    _legendre_nodes: NDArray[np.floating] = field(repr=False)
    # Private shape: (quadrature_order,); Gauss-Legendre weights on [-1, 1].
    _legendre_weights: NDArray[np.floating] = field(repr=False)

    # Shape: (2, panel_count), storing previous/next adjacent panel ids.
    adjacency: NDArray[np.integer]
    # Scalar flag; true when panel endpoints form a closed oriented curve.
    closed: bool
    # Scalar label describing the oriented curve direction.
    orientation: Literal["ccw", "cw", "open"] | None = None
    # Optional shape: (coordinate_dim, vertex_count).
    vertices: NDArray[np.floating] | None = None
    # Free-form metadata; values are constructor- or algorithm-specific.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.derivatives = np.asarray(self.derivatives, dtype=float)
        self.second_derivatives = np.asarray(self.second_derivatives, dtype=float)
        self.normals = np.asarray(self.normals, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self._legendre_nodes = np.asarray(self._legendre_nodes, dtype=float)
        self._legendre_weights = np.asarray(self._legendre_weights, dtype=float)
        self.adjacency = np.asarray(self.adjacency, dtype=np.int64)

        if self.positions.ndim != 3:
            raise ValueError(
                "positions must have shape (coordinate_dim, quadrature_order, panel_count)"
            )
        if self.derivatives.shape != self.positions.shape:
            raise ValueError("derivatives must match positions")
        if self.second_derivatives.shape != self.positions.shape:
            raise ValueError("second_derivatives must match positions")
        if self.normals.shape != self.positions.shape:
            raise ValueError("normals must match positions")
        if self.weights.shape != self.positions.shape[1:]:
            raise ValueError("weights must have shape (quadrature_order, panel_count)")
        if self._legendre_nodes.shape != (self.quadrature_order,):
            raise ValueError("_legendre_nodes must have shape (quadrature_order,)")
        if self._legendre_weights.shape != (self.quadrature_order,):
            raise ValueError("_legendre_weights must have shape (quadrature_order,)")
        if self.adjacency.shape != (2, self.panel_count):
            raise ValueError("adjacency must have shape (2, panel_count)")

    @property
    def nodes(self) -> None:
        raise AttributeError("Chunker.nodes is private; Legendre nodes are internal storage")

    @nodes.setter
    def nodes(self, value) -> None:
        raise AttributeError("Chunker.nodes is private and cannot be assigned")

    @property
    def reference_weights(self) -> None:
        raise AttributeError(
            "Chunker.reference_weights is private; Legendre weights are internal storage"
        )

    @reference_weights.setter
    def reference_weights(self, value) -> None:
        raise AttributeError("Chunker.reference_weights is private and cannot be assigned")

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
    def pointinfo(self) -> PointInfoView:
        return PointInfoView(
            positions=self.positions,
            derivatives=self.derivatives,
            second_derivatives=self.second_derivatives,
            normals=self.normals,
            weights=self.weights,
            nodes=self._legendre_nodes,
            panel_ids=np.arange(self.panel_count, dtype=np.int64),
        )

    @property
    def arclength_density(self) -> NDArray[np.floating]:
        return np.linalg.norm(self.derivatives, axis=0)

    @property
    def panel_lengths(self) -> NDArray[np.floating]:
        return np.sum(self.weights, axis=0)

    @property
    def bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        flat = self.pointinfo.flat_positions
        return np.min(flat, axis=1), np.max(flat, axis=1)

    @property
    def panel_endpoints(self) -> NDArray[np.floating]:
        from chunkie.quadrature.legendre import interpolation_matrix

        interpolation = interpolation_matrix(self._legendre_nodes, np.array([-1.0, 1.0]))
        return np.einsum("es,RsS->ReS", interpolation, self.positions)

    @property
    def panel_endpoint_tangents(self) -> NDArray[np.floating]:
        from chunkie.quadrature.legendre import interpolation_matrix

        interpolation = interpolation_matrix(self._legendre_nodes, np.array([-1.0, 1.0]))
        derivatives = np.einsum("es,RsS->ReS", interpolation, self.derivatives)
        # Endpoint tangents are unit vectors; derivative magnitudes near corners
        # belong to panel length/weights, not orientation diagnostics.
        return derivatives / np.linalg.norm(derivatives, axis=0, keepdims=True)

    @property
    def length(self) -> float:
        return float(np.sum(self.weights))

    @property
    def area(self) -> float:
        if self.coordinate_dim != 2:
            raise ValueError("area currently supports two-dimensional curves")
        x, y = self.positions
        dx, dy = self.derivatives
        # Derivatives are with respect to each panel's Legendre reference
        # coordinate, so the line integral uses reference weights directly.
        integrand = x * dy - y * dx
        return float(0.5 * np.sum(integrand * self._legendre_weights[:, None]))

    @property
    def tangents(self) -> NDArray[np.floating]:
        return self.derivatives / self.arclength_density[None, :, :]

    @property
    def signed_curvature(self) -> NDArray[np.floating]:
        if self.coordinate_dim != 2:
            raise ValueError("signed_curvature currently supports two-dimensional curves")
        dx, dy = self.derivatives
        ddx, ddy = self.second_derivatives
        speed = np.linalg.norm(self.derivatives, axis=0)
        return (dx * ddy - dy * ddx) / speed**3

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
            nodes=self._legendre_nodes,
        )

    ### AFFINE TRANSFORMS
    def affine(self, matrix: ArrayLike, *, offset: ArrayLike | None = None) -> Chunker:
        matrix_array = np.asarray(matrix, dtype=float)
        if matrix_array.shape != (self.coordinate_dim, self.coordinate_dim):
            raise ValueError("affine matrix must have shape (coordinate_dim, coordinate_dim)")
        offset_array = (
            np.zeros((self.coordinate_dim, 1, 1))
            if offset is None
            else np.asarray(offset, dtype=float).reshape(self.coordinate_dim, 1, 1)
        )

        # Affine transforms are an adapter boundary for geometry: positions and
        # derivatives transform linearly, while normals/weights are recomputed
        # from the transformed tangent so orientation and scaling stay coherent.
        positions = np.einsum("ab,bsS->asS", matrix_array, self.positions) + offset_array
        derivatives = np.einsum("ab,bsS->asS", matrix_array, self.derivatives)
        second_derivatives = np.einsum("ab,bsS->asS", matrix_array, self.second_derivatives)
        weights = np.linalg.norm(derivatives, axis=0) * self._legendre_weights[:, None]
        normals = right_normals(derivatives)
        return replace(
            self,
            positions=positions,
            derivatives=derivatives,
            second_derivatives=second_derivatives,
            normals=normals,
            weights=weights,
            orientation=_transformed_orientation(self.orientation, matrix_array),
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
        return self.affine(
            scale * np.eye(self.coordinate_dim), offset=(1.0 - scale) * center_array[:, 0, 0]
        )

    def rotated(
        self,
        angle: float,
        *,
        center: ArrayLike | None = None,
        target_center: ArrayLike | None = None,
    ) -> Chunker:
        source_center = (
            np.zeros(self.coordinate_dim) if center is None else np.asarray(center, dtype=float)
        )
        destination = (
            source_center if target_center is None else np.asarray(target_center, dtype=float)
        )
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        matrix = np.array([[c, -s], [s, c]])
        return self.affine(matrix, offset=destination - matrix @ source_center)

    def reflected(
        self,
        angle: float,
        *,
        center: ArrayLike | None = None,
        target_center: ArrayLike | None = None,
    ) -> Chunker:
        source_center = (
            np.zeros(self.coordinate_dim) if center is None else np.asarray(center, dtype=float)
        )
        destination = (
            source_center if target_center is None else np.asarray(target_center, dtype=float)
        )
        c = float(np.cos(2.0 * angle))
        s = float(np.sin(2.0 * angle))
        matrix = np.array([[c, s], [s, -c]])
        return self.affine(matrix, offset=destination - matrix @ source_center)


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


def _transformed_orientation(
    orientation: Literal["ccw", "cw", "open"] | None,
    matrix: NDArray[np.floating],
) -> Literal["ccw", "cw", "open"] | None:
    if orientation in {None, "open"}:
        return orientation
    if np.linalg.det(matrix) >= 0.0:
        return orientation
    return "cw" if orientation == "ccw" else "ccw"
