"""RCIP prolongation and Schur-Banachiewicz algebra."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chunkie.quadrature.legendre import legendre_rule


def split_panel_interpolation(
    nodes: ArrayLike,
    weights: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return the old RCIP interpolation matrix and weighted companion."""

    source_nodes = np.asarray(nodes, dtype=float).reshape(-1)
    source_weights = np.asarray(weights, dtype=float).reshape(-1)
    if source_nodes.shape != source_weights.shape:
        raise ValueError("nodes and weights must have the same length")

    target_nodes = np.concatenate((source_nodes - 1.0, source_nodes + 1.0)) / 2.0
    target_weights = np.concatenate((source_weights, source_weights)) / 2.0
    source_vander = np.ones((source_nodes.size, source_nodes.size), dtype=float)
    target_vander = np.ones((target_nodes.size, source_nodes.size), dtype=float)
    for degree in range(1, source_nodes.size):
        source_vander[:, degree] = source_vander[:, degree - 1] * source_nodes
        target_vander[:, degree] = target_vander[:, degree - 1] * target_nodes
    interpolation = target_vander @ np.linalg.inv(source_vander)
    weighted = interpolation * (target_weights[:, None] / source_weights[None, :])
    return interpolation, weighted


def block_interpolation(
    interpolation: ArrayLike,
    *,
    edge_count: int,
    dimension: int,
) -> NDArray[np.floating]:
    """Lift one scalar split-panel interpolation to RCIP edge/component blocks."""

    matrix = np.asarray(interpolation, dtype=float)
    return np.kron(np.eye(int(edge_count)), np.kron(matrix, np.eye(int(dimension))))


def setup(
    quadrature_order: int,
    dimension: int,
    edge_count: int,
    starts_at_corner: ArrayLike,
) -> tuple[NDArray[np.floating], ...]:
    """Return the old ``chnk.rcip.setup`` arrays using zero-based indices."""

    nodes, weights = legendre_rule(int(quadrature_order))
    interpolation, weighted = split_panel_interpolation(nodes, weights)
    prolongation = block_interpolation(
        interpolation,
        edge_count=int(edge_count),
        dimension=int(dimension),
    )
    weighted_prolongation = block_interpolation(
        weighted,
        edge_count=int(edge_count),
        dimension=int(dimension),
    )

    is_start = np.asarray(starts_at_corner, dtype=bool).reshape(-1)
    if is_start.size != int(edge_count):
        raise ValueError("starts_at_corner must have one entry per edge")

    quadrature_order = int(quadrature_order)
    dimension = int(dimension)
    edge_count = int(edge_count)
    ilist = np.zeros((2, edge_count), dtype=np.int64)
    star_l: list[int] = []
    circ_l: list[int] = []
    star_l_scalar: list[int] = []
    circ_l_scalar: list[int] = []
    star_s: list[int] = []
    circ_s: list[int] = []

    indg1 = 2 * quadrature_order * dimension + np.arange(quadrature_order * dimension)
    indb1 = np.arange(2 * quadrature_order * dimension)
    indg11 = 2 * quadrature_order + np.arange(quadrature_order)
    indb11 = np.arange(2 * quadrature_order)
    indg0 = np.arange(quadrature_order * dimension)
    indb0 = quadrature_order * dimension + np.arange(2 * quadrature_order * dimension)
    indg01 = np.arange(quadrature_order)
    indb01 = quadrature_order + np.arange(2 * quadrature_order)

    indg1s = quadrature_order * dimension + np.arange(quadrature_order * dimension)
    indb1s = np.arange(quadrature_order * dimension)
    indg0s = np.arange(quadrature_order * dimension)
    indb0s = quadrature_order * dimension + np.arange(quadrature_order * dimension)

    for edge_id, edge_starts_at_corner in enumerate(is_start):
        offset_l = 3 * edge_id * quadrature_order * dimension
        offset_l_scalar = 3 * edge_id * quadrature_order
        offset_s = 2 * edge_id * quadrature_order * dimension
        if edge_starts_at_corner:
            star_l.extend((indb1 + offset_l).tolist())
            circ_l.extend((indg1 + offset_l).tolist())
            star_l_scalar.extend((indb11 + offset_l_scalar).tolist())
            circ_l_scalar.extend((indg11 + offset_l_scalar).tolist())
            star_s.extend((indb1s + offset_s).tolist())
            circ_s.extend((indg1s + offset_s).tolist())
            ilist[:, edge_id] = [0, 1]
        else:
            star_l.extend((indb0 + offset_l).tolist())
            circ_l.extend((indg0 + offset_l).tolist())
            star_l_scalar.extend((indb01 + offset_l_scalar).tolist())
            circ_l_scalar.extend((indg01 + offset_l_scalar).tolist())
            star_s.extend((indb0s + offset_s).tolist())
            circ_s.extend((indg0s + offset_s).tolist())
            ilist[:, edge_id] = [1, 2]

    return (
        prolongation,
        weighted_prolongation,
        np.asarray(star_l, dtype=np.int64),
        np.asarray(circ_l, dtype=np.int64),
        np.asarray(star_s, dtype=np.int64),
        np.asarray(circ_s, dtype=np.int64),
        ilist,
        np.asarray(star_l_scalar, dtype=np.int64),
        np.asarray(circ_l_scalar, dtype=np.int64),
    )


def schur_banachiewicz(
    prolongation: ArrayLike,
    weighted_prolongation: ArrayLike,
    local_matrix: ArrayLike,
    compressed_inverse: ArrayLike,
    star_l: ArrayLike,
    circ_l: ArrayLike,
    star_s: ArrayLike,
    circ_s: ArrayLike,
) -> NDArray[np.generic]:
    """Apply the same local RCIP Schur update used by the old implementation."""

    p = np.asarray(prolongation)
    pw = np.asarray(weighted_prolongation)
    matrix = np.asarray(local_matrix)
    out = np.asarray(compressed_inverse).copy()
    star_l = np.asarray(star_l, dtype=np.int64)
    circ_l = np.asarray(circ_l, dtype=np.int64)
    star_s = np.asarray(star_s, dtype=np.int64)
    circ_s = np.asarray(circ_s, dtype=np.int64)

    va = matrix[np.ix_(circ_l, star_l)] @ out
    pta = pw.T @ out
    ptau = pta @ matrix[np.ix_(star_l, circ_l)]
    circ_inverse = np.linalg.inv(
        matrix[np.ix_(circ_l, circ_l)] - va @ matrix[np.ix_(star_l, circ_l)]
    )
    circ_inverse_va_p = circ_inverse @ (va @ p)
    out[np.ix_(star_s, star_s)] = pta @ p + ptau @ circ_inverse_va_p
    out[np.ix_(circ_s, circ_s)] = circ_inverse
    out[np.ix_(circ_s, star_s)] = -circ_inverse_va_p
    out[np.ix_(star_s, circ_s)] = -ptau @ circ_inverse
    return out
