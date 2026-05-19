"""Panel refinement helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from .chunker import Chunker, right_normals
from .constructors import _adjacency


def change_quadrature_order(
    chunker: Chunker,
    quadrature_order: int,
    values: NDArray[np.generic] | None = None,
):
    """Interpolate geometry, and optional panel data, to a new panel order."""

    new_nodes, new_legendre_weights = leggauss(int(quadrature_order))
    interpolation = _interpolation_matrix(chunker._legendre_nodes, new_nodes)
    positions = np.einsum("ql,RlS->RqS", interpolation, chunker.positions)
    derivatives = np.einsum("ql,RlS->RqS", interpolation, chunker.derivatives)
    second_derivatives = np.einsum("ql,RlS->RqS", interpolation, chunker.second_derivatives)
    weights = new_legendre_weights[:, None] * np.linalg.norm(derivatives, axis=0)
    metadata = dict(chunker.metadata)
    metadata["quadrature_order_changed_from"] = chunker.quadrature_order
    updated = Chunker(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second_derivatives,
        normals=right_normals(derivatives),
        weights=weights,
        _legendre_nodes=new_nodes,
        _legendre_weights=new_legendre_weights,
        adjacency=chunker.adjacency.copy(),
        closed=chunker.closed,
        orientation=chunker.orientation,
        vertices=chunker.vertices,
        metadata=metadata,
    )
    if values is None:
        return updated

    data = np.asarray(values)
    if data.shape[-2:] != (chunker.quadrature_order, chunker.panel_count):
        raise ValueError("values must end with shape (old_quadrature_order, panel_count)")
    # Optional panel data follows the same node axis as geometry; leading axes
    # are preserved so scalar and component densities use one adapter.
    interpolated_values = np.einsum("ql,...lS->...qS", interpolation, data)
    return updated, interpolated_values


def refine(chunker: Chunker, *, levels: int = 1) -> Chunker:
    """Split every panel uniformly by ``2**levels`` child panels."""

    if levels < 0:
        raise ValueError("levels must be non-negative")
    if levels == 0:
        return replace(chunker)

    child_count = 2**levels
    new_panel_count = chunker.panel_count * child_count
    positions = np.empty((chunker.coordinate_dim, chunker.quadrature_order, new_panel_count), dtype=float)
    derivatives = np.empty_like(positions)
    second_derivatives = np.empty_like(positions)
    weights = np.empty((chunker.quadrature_order, new_panel_count), dtype=float)

    out_panel = 0
    for parent_panel in range(chunker.panel_count):
        for child_id in range(child_count):
            left = -1.0 + 2.0 * child_id / child_count
            right = -1.0 + 2.0 * (child_id + 1) / child_count
            _fill_refined_child(
                chunker,
                parent_panel,
                out_panel,
                left,
                right,
                positions,
                derivatives,
                second_derivatives,
                weights,
            )
            out_panel += 1

    metadata = dict(chunker.metadata)
    metadata["refinement_levels"] = metadata.get("refinement_levels", 0) + levels
    return Chunker(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second_derivatives,
        normals=right_normals(derivatives),
        weights=weights,
        _legendre_nodes=chunker._legendre_nodes,
        _legendre_weights=chunker._legendre_weights,
        adjacency=_adjacency(new_panel_count, chunker.closed),
        closed=chunker.closed,
        orientation=chunker.orientation,
        vertices=chunker.vertices,
        metadata=metadata,
    )


def _fill_refined_child(
    chunker: Chunker,
    parent_panel: int,
    out_panel: int,
    left: float,
    right: float,
    positions: NDArray[np.floating],
    derivatives: NDArray[np.floating],
    second_derivatives: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> None:
    center = 0.5 * (left + right)
    scale = 0.5 * (right - left)
    parent_nodes = center + scale * chunker._legendre_nodes
    interpolation = _interpolation_matrix(chunker._legendre_nodes, parent_nodes)

    positions[:, :, out_panel] = np.einsum("qk,Rk->Rq", interpolation, chunker.positions[:, :, parent_panel])
    # Parent derivatives are with respect to the parent reference coordinate.
    # The child coordinate maps by u_parent = center + scale*u_child, so the
    # chain rule contributes scale and scale**2 to first and second derivatives.
    derivatives[:, :, out_panel] = scale * np.einsum(
        "qk,Rk->Rq",
        interpolation,
        chunker.derivatives[:, :, parent_panel],
    )
    second_derivatives[:, :, out_panel] = scale**2 * np.einsum(
        "qk,Rk->Rq",
        interpolation,
        chunker.second_derivatives[:, :, parent_panel],
    )
    weights[:, out_panel] = chunker._legendre_weights * np.linalg.norm(
        derivatives[:, :, out_panel], axis=0
    )


def _interpolation_matrix(
    nodes: NDArray[np.floating], targets: NDArray[np.floating]
) -> NDArray[np.floating]:
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
