"""Generalized Gaussian quadrature integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .legendre import legendre_rule


@dataclass(frozen=True)
class GGQRuleSet:
    neighbor_nodes: NDArray[np.floating]
    neighbor_weights: NDArray[np.floating]
    self_nodes: tuple[NDArray[np.floating], ...]
    self_weights: tuple[NDArray[np.floating], ...]
    neighbor_interpolator: NDArray[np.floating]
    self_interpolators: tuple[NDArray[np.floating], ...]
    singularity: str = "log"


def setup(
    quadrature_order: int,
    singularity: str = "log",
    *,
    nfac_self: int = 2,
    nfac_near: int = 2,
) -> GGQRuleSet:
    """Build generated GGQ-style local rules for one panel order."""

    qtype = singularity.lower()
    if qtype not in {"log", "removable", "pv", "hs"}:
        raise ValueError("GGQ singularity must be one of log, removable, pv, or hs")
    order = int(quadrature_order)
    neighbor_nodes, neighbor_weights = legendre_rule(max(order, int(nfac_near) * order))
    self_nodes, self_weights = getremovablequad(order, nfac=max(1, int(nfac_self)))
    base_nodes, _ = legendre_rule(order)
    return GGQRuleSet(
        neighbor_nodes=neighbor_nodes,
        neighbor_weights=neighbor_weights,
        self_nodes=tuple(self_nodes),
        self_weights=tuple(self_weights),
        neighbor_interpolator=_lagrange_matrix(base_nodes, neighbor_nodes),
        self_interpolators=tuple(_lagrange_matrix(base_nodes, nodes) for nodes in self_nodes),
        singularity=qtype,
    )


def getremovablequad(
    quadrature_order: int,
    *,
    nfac: int = 1,
) -> tuple[list[NDArray[np.floating]], list[NDArray[np.floating]]]:
    """Return split Gauss rules that avoid each original Legendre node."""

    order = int(quadrature_order)
    base_nodes, _ = legendre_rule(order)
    over_nodes, over_weights = legendre_rule(max(order, int(nfac) * order))
    unit_nodes = 0.5 * (over_nodes + 1.0)
    unit_weights = 0.5 * over_weights
    nodes_by_source: list[NDArray[np.floating]] = []
    weights_by_source: list[NDArray[np.floating]] = []
    for node in base_nodes:
        left_nodes = unit_nodes * (node + 1.0) - 1.0
        left_weights = unit_weights * (node + 1.0)
        right_nodes = unit_nodes * (1.0 - node) + node
        right_weights = unit_weights * (1.0 - node)
        nodes_by_source.append(np.concatenate((left_nodes, right_nodes)))
        weights_by_source.append(np.concatenate((left_weights, right_weights)))
    return nodes_by_source, weights_by_source


def build_ggq_panel_matrix(*args, **kwargs):
    raise NotImplementedError("GGQ panel matrix assembly is a required rewrite milestone")


def _lagrange_matrix(nodes: NDArray[np.floating], evaluation_nodes: NDArray[np.floating]) -> NDArray[np.floating]:
    barycentric_weights = _barycentric_weights(nodes)
    matrix = np.empty((evaluation_nodes.size, nodes.size), dtype=float)
    for row, value in enumerate(evaluation_nodes):
        difference = value - nodes
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
