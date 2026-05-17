"""RCIP prolongation operators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def build_prolongation(source_nodes: ArrayLike, target_nodes: ArrayLike) -> NDArray[np.floating]:
    """Return the interpolation matrix from source nodes to target nodes."""

    source = np.asarray(source_nodes, dtype=float).reshape(-1)
    target = np.asarray(target_nodes, dtype=float).reshape(-1)
    barycentric_weights = _barycentric_weights(source)
    matrix = np.empty((target.size, source.size), dtype=float)
    for row, value in enumerate(target):
        difference = value - source
        exact = np.where(np.abs(difference) <= 10.0 * np.finfo(float).eps)[0]
        if exact.size:
            matrix[row] = 0.0
            matrix[row, exact[0]] = 1.0
            continue
        terms = barycentric_weights / difference
        matrix[row] = terms / np.sum(terms)
    return matrix


def _barycentric_weights(nodes: NDArray[np.floating]) -> NDArray[np.floating]:
    weights = np.ones(nodes.size, dtype=float)
    for index, node in enumerate(nodes):
        weights[index] = 1.0 / np.prod(node - np.delete(nodes, index))
    return weights
