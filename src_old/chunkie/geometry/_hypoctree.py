"""Uniform hyperoctree helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class HypOctNode:
    ctr: np.ndarray
    xi: np.ndarray
    prnt: int | None
    chld: list[int]
    nbor: list[int]


@dataclass
class HypOctTree:
    nlvl: int
    lvp: np.ndarray
    lrt: float
    nodes: list[HypOctNode]


def hypoct_uni(
    points: ArrayLike,
    box_size: float,
    max_level: int | float = np.inf,
    extent: ArrayLike | None = None,
) -> HypOctTree:
    """Build a uniform-depth hyperoctree over points.

    Indices in the returned tree are zero-based, following the rest of the
    Python port rather than MATLAB's one-based struct arrays.
    """

    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2:
        raise ValueError("x must have shape (dim, n)")
    if box_size < 0:
        raise ValueError("target box size must be non-negative")
    if max_level < 1:
        raise ValueError("maximum tree depth must be at least 1")

    dim, npt = point_array.shape
    if extent is None:
        if npt == 0:
            domain_extent = np.zeros((dim, 2), dtype=float)
        else:
            domain_extent = np.column_stack(
                (np.min(point_array, axis=1), np.max(point_array, axis=1))
            )
    else:
        domain_extent = np.asarray(extent, dtype=float)
        if domain_extent.shape != (dim, 2):
            raise ValueError("extent must have shape (dim, 2)")

    root_len = float(np.max(domain_extent[:, 1] - domain_extent[:, 0])) if dim else 0.0
    root_ctr = 0.5 * (domain_extent[:, 0] + domain_extent[:, 1])
    nodes = [HypOctNode(root_ctr, np.arange(npt, dtype=int), None, [], [])]
    lvp = [0, 1]
    level = 1
    side = root_len
    resolved_max_level = np.inf if np.isinf(max_level) else int(max_level)

    while level < resolved_max_level:
        next_side = 0.5 * side
        if next_side <= box_size:
            break
        start, stop = lvp[level - 1], lvp[level]
        before = len(nodes)
        for inode in range(start, stop):
            parent = nodes[inode]
            if parent.xi.size == 0:
                continue
            child_codes = _child_codes(point_array[:, parent.xi], parent.ctr)
            for code in np.unique(child_codes):
                mask = child_codes == code
                bits = ((int(code) >> np.arange(dim)) & 1).astype(float)
                child_ctr = parent.ctr + next_side * (bits - 0.5)
                child = HypOctNode(child_ctr, parent.xi[mask].copy(), inode, [], [])
                nodes.append(child)
                parent.chld.append(len(nodes) - 1)
            parent.xi = np.zeros(0, dtype=int)
        if len(nodes) == before:
            break
        level += 1
        lvp.append(len(nodes))
        side = next_side

    tree = HypOctTree(level, np.asarray(lvp, dtype=int), root_len, nodes)
    _populate_hypoct_neighbors(tree)
    return tree


def _child_codes(points: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    dim = points.shape[0]
    codes = np.zeros(points.shape[1], dtype=int)
    for axis in range(dim):
        codes += (points[axis] > ctr[axis]).astype(int) << axis
    return codes


def _populate_hypoct_neighbors(tree: HypOctTree) -> None:
    levels = np.zeros(len(tree.nodes), dtype=int)
    side_by_level = np.zeros(tree.nlvl, dtype=float)
    side = tree.lrt
    for lvl in range(tree.nlvl):
        levels[tree.lvp[lvl] : tree.lvp[lvl + 1]] = lvl
        side_by_level[lvl] = side
        side *= 0.5

    for lvl in range(1, tree.nlvl):
        side = side_by_level[lvl]
        start, stop = tree.lvp[lvl], tree.lvp[lvl + 1]
        for i in range(start, stop):
            node = tree.nodes[i]
            neighbors: set[int] = set()
            if node.prnt is not None:
                siblings = tree.nodes[node.prnt].chld
                neighbors.update(j for j in siblings if j != i)
                parent_neighbors = tree.nodes[node.prnt].nbor
            else:
                parent_neighbors = []

            for j in parent_neighbors:
                other = tree.nodes[j]
                if other.xi.size:
                    if _boxes_adjacent(node.ctr, side, other.ctr, side_by_level[levels[j]]):
                        neighbors.add(j)
                for child in other.chld:
                    other_child = tree.nodes[child]
                    if _boxes_adjacent(
                        node.ctr, side, other_child.ctr, side_by_level[levels[child]]
                    ):
                        neighbors.add(child)
            node.nbor = sorted(neighbors)


def _boxes_adjacent(ctr1: np.ndarray, side1: float, ctr2: np.ndarray, side2: float) -> bool:
    return bool(np.all(np.abs(ctr1 - ctr2) <= 0.5 * (side1 + side2) + 1e-14))
