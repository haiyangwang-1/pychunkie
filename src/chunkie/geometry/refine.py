"""Panel refinement helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from .arclength import _interpolation_matrix
from .chunker import Chunker, right_normals
from .constructors import _adjacency


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
        nodes=chunker.nodes,
        reference_weights=chunker.reference_weights,
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
    parent_nodes = center + scale * chunker.nodes
    interpolation = _interpolation_matrix(chunker.nodes, parent_nodes)

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
    weights[:, out_panel] = chunker.reference_weights * np.linalg.norm(derivatives[:, :, out_panel], axis=0)
