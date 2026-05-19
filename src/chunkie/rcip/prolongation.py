"""RCIP prolongation operators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chunkie.quadrature.legendre import interpolation_matrix


def build_prolongation(source_nodes: ArrayLike, target_nodes: ArrayLike) -> NDArray[np.floating]:
    """Return the interpolation matrix from source nodes to target nodes."""

    source = np.asarray(source_nodes, dtype=float).reshape(-1)
    target = np.asarray(target_nodes, dtype=float).reshape(-1)
    return interpolation_matrix(source, target)


def build_split_panel_prolongation(
    nodes: ArrayLike,
    weights: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return interpolation and weighted transfer matrices for two half panels."""

    source_nodes = np.asarray(nodes, dtype=float).reshape(-1)
    source_weights = np.asarray(weights, dtype=float).reshape(-1)
    if source_nodes.shape != source_weights.shape:
        raise ValueError("nodes and weights must have the same shape")
    target_nodes = np.concatenate(((source_nodes - 1.0) / 2.0, (source_nodes + 1.0) / 2.0))
    target_weights = np.concatenate((source_weights / 2.0, source_weights / 2.0))
    interpolation = build_prolongation(source_nodes, target_nodes)

    # RCIP alternates between interpolation of densities and weighted transfer
    # of quadrature data. This matrix maps coarse weighted values to split-panel
    # weighted values so summing the target entries preserves panel integrals.
    weighted_transfer = target_weights[:, None] * interpolation / source_weights[None, :]
    return target_nodes, target_weights, interpolation, weighted_transfer


def build_block_prolongation(
    interpolation: ArrayLike,
    *,
    edge_count: int,
    component_count: int,
) -> NDArray[np.floating]:
    """Lift a scalar panel interpolation matrix to edge/component local layout."""

    interpolation_array = np.asarray(interpolation, dtype=float)
    if interpolation_array.ndim != 2:
        raise ValueError("interpolation must be a matrix")
    # This is the local RCIP block layout inherited from corner compression:
    # edge blocks outside, point interpolation in the middle, and components
    # inside each point. Solver-vector adapters remain separate in system code.
    return np.kron(
        np.eye(int(edge_count)), np.kron(interpolation_array, np.eye(int(component_count)))
    )
