"""Arclength parameterization and resampling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from .chunker import Chunker, right_normals
from .constructors import _adjacency


@dataclass(frozen=True)
class ArcLengthParameterization:
    """Lookup data for evaluating a `Chunker` by physical arclength."""

    chunker: Chunker
    panel_offsets: NDArray[np.floating]
    node_arclengths: NDArray[np.floating]

    @property
    def panel_lengths(self) -> NDArray[np.floating]:
        return np.diff(self.panel_offsets)

    @property
    def total_length(self) -> float:
        return float(self.panel_offsets[-1])


def arclength_parameterization(chunker: Chunker) -> ArcLengthParameterization:
    """Build cumulative arclength data for panel-local inverse evaluation."""

    panel_lengths = chunker.panel_lengths
    panel_offsets = np.concatenate(([0.0], np.cumsum(panel_lengths)))
    node_arclengths = np.empty((chunker.quadrature_order, chunker.panel_count), dtype=float)
    for panel_id in range(chunker.panel_count):
        for local_id, node in enumerate(chunker.nodes):
            node_arclengths[local_id, panel_id] = panel_offsets[panel_id] + _integrate_panel_speed(
                chunker,
                panel_id,
                -1.0,
                float(node),
            )
    return ArcLengthParameterization(
        chunker=chunker,
        panel_offsets=panel_offsets,
        node_arclengths=node_arclengths,
    )


def evaluate_arclength(
    parameterization: ArcLengthParameterization,
    arclengths: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate position, unit tangent, and second arclength derivative."""

    chunker = parameterization.chunker
    requested = np.asarray(arclengths, dtype=float).reshape(-1)
    if chunker.closed:
        values = np.mod(requested, parameterization.total_length)
    else:
        values = requested
        if np.any((values < 0.0) | (values > parameterization.total_length)):
            raise ValueError("arclength values for an open Chunker must lie inside [0, total_length]")

    panel_ids = np.searchsorted(parameterization.panel_offsets[1:], values, side="right")
    panel_ids = np.minimum(panel_ids, chunker.panel_count - 1)

    positions = np.empty((chunker.coordinate_dim, values.size), dtype=float)
    tangents = np.empty_like(positions)
    second_arclength_derivatives = np.empty_like(positions)
    for output_id, (panel_id, value) in enumerate(zip(panel_ids, values, strict=True)):
        local_s = float(value - parameterization.panel_offsets[panel_id])
        reference = _invert_panel_arclength(chunker, int(panel_id), local_s)
        position, derivative, second_derivative = _evaluate_reference(chunker, int(panel_id), np.array([reference]))
        speed = np.linalg.norm(derivative[:, 0])
        tangent = derivative[:, 0] / speed
        # The stored derivatives are with respect to the panel reference
        # coordinate u.  This chain rule converts r_uu to d^2 r / ds^2, the
        # geometric curvature vector used by arclength-resampled panels.
        curvature_vector = second_derivative[:, 0] / speed**2
        curvature_vector -= derivative[:, 0] * np.dot(derivative[:, 0], second_derivative[:, 0]) / speed**4
        positions[:, output_id] = position[:, 0]
        tangents[:, output_id] = tangent
        second_arclength_derivatives[:, output_id] = curvature_vector

    return positions, tangents, second_arclength_derivatives


def resample_by_arclength(
    chunker: Chunker,
    *,
    panel_count: int | None = None,
    quadrature_order: int | None = None,
) -> Chunker:
    """Return a `Chunker` whose panels have constant arclength speed."""

    new_panel_count = chunker.panel_count if panel_count is None else int(panel_count)
    new_order = chunker.quadrature_order if quadrature_order is None else int(quadrature_order)
    nodes, reference_weights = leggauss(new_order)
    parameterization = arclength_parameterization(chunker)
    panel_length = parameterization.total_length / new_panel_count

    positions = np.empty((chunker.coordinate_dim, new_order, new_panel_count), dtype=float)
    derivatives = np.empty_like(positions)
    second_derivatives = np.empty_like(positions)
    weights = np.empty((new_order, new_panel_count), dtype=float)
    for panel_id in range(new_panel_count):
        panel_start = panel_id * panel_length
        target_s = panel_start + 0.5 * panel_length * (nodes + 1.0)
        panel_positions, tangents, curvature_vectors = evaluate_arclength(parameterization, target_s)
        positions[:, :, panel_id] = panel_positions
        derivatives[:, :, panel_id] = tangents * (0.5 * panel_length)
        second_derivatives[:, :, panel_id] = curvature_vectors * (0.5 * panel_length) ** 2
        weights[:, panel_id] = reference_weights * (0.5 * panel_length)

    normals = right_normals(derivatives)
    metadata = dict(chunker.metadata)
    metadata["resampled_by_arclength"] = True
    return Chunker(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second_derivatives,
        normals=normals,
        weights=weights,
        nodes=nodes,
        reference_weights=reference_weights,
        adjacency=_adjacency(new_panel_count, chunker.closed),
        closed=chunker.closed,
        orientation=chunker.orientation,
        vertices=chunker.vertices,
        metadata=metadata,
    )


def _invert_panel_arclength(chunker: Chunker, panel_id: int, target_length: float) -> float:
    panel_length = chunker.panel_lengths[panel_id]
    if panel_length <= 0.0:
        raise ValueError("cannot invert arclength on a zero-length panel")
    if target_length <= 0.0:
        return -1.0
    if target_length >= panel_length:
        return 1.0

    reference = -1.0 + 2.0 * target_length / panel_length
    for _ in range(12):
        length = _integrate_panel_speed(chunker, panel_id, -1.0, reference)
        derivative = _evaluate_reference(chunker, panel_id, np.array([reference]))[1][:, 0]
        speed = np.linalg.norm(derivative)
        step = (length - target_length) / speed
        reference = float(np.clip(reference - step, -1.0, 1.0))
        if abs(step) < 1.0e-14:
            break
    return reference


def _integrate_panel_speed(chunker: Chunker, panel_id: int, left: float, right: float) -> float:
    if np.isclose(left, right):
        return 0.0
    nodes, weights = leggauss(max(24, chunker.quadrature_order + 8))
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    references = midpoint + half_width * nodes
    derivatives = _evaluate_reference(chunker, panel_id, references)[1]
    return float(abs(half_width) * np.sum(weights * np.linalg.norm(derivatives, axis=0)))


def _evaluate_reference(
    chunker: Chunker,
    panel_id: int,
    references: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    matrix = _interpolation_matrix(chunker.nodes, references)
    positions = np.einsum("qk,Rk->Rq", matrix, chunker.positions[:, :, panel_id])
    derivatives = np.einsum("qk,Rk->Rq", matrix, chunker.derivatives[:, :, panel_id])
    second_derivatives = np.einsum("qk,Rk->Rq", matrix, chunker.second_derivatives[:, :, panel_id])
    return positions, derivatives, second_derivatives


def _interpolation_matrix(nodes: NDArray[np.floating], targets: NDArray[np.floating]) -> NDArray[np.floating]:
    barycentric_weights = _barycentric_weights(nodes)
    matrix = np.empty((targets.size, nodes.size), dtype=float)
    for row, target in enumerate(targets):
        differences = target - nodes
        exact = np.isclose(differences, 0.0, atol=1.0e-15, rtol=0.0)
        if np.any(exact):
            matrix[row] = 0.0
            matrix[row, np.argmax(exact)] = 1.0
            continue
        scaled = barycentric_weights / differences
        matrix[row] = scaled / np.sum(scaled)
    return matrix


def _barycentric_weights(nodes: NDArray[np.floating]) -> NDArray[np.floating]:
    differences = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(differences, 1.0)
    return 1.0 / np.prod(differences, axis=1)
