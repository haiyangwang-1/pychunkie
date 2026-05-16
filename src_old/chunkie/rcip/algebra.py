"""RCIP prolongation and Schur-Banachiewicz setup algebra."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .. import lege


def IPinit(nodes: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Build the RCIP prolongation matrix and weighted prolongation."""

    t = np.asarray(nodes, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if t.size != w.size:
        raise ValueError("nodes and weights must have the same length")

    ngl = t.size
    a = np.ones((ngl, ngl), dtype=float)
    aa = np.ones((2 * ngl, ngl), dtype=float)
    t2 = np.concatenate((t - 1.0, t + 1.0)) / 2.0
    w2 = np.concatenate((w, w)) / 2.0
    for k in range(1, ngl):
        a[:, k] = a[:, k - 1] * t
        aa[:, k] = aa[:, k - 1] * t2
    ip = aa @ np.linalg.inv(a)
    ipw = ip * (w2[:, None] / w[None, :])
    return ip, ipw


def Pbcinit(interpolation: ArrayLike, edge_count: int, dimension: int) -> np.ndarray:
    """Construct the block diagonal RCIP prolongation for all edge unknowns."""

    ip = np.asarray(interpolation)
    return np.kron(np.eye(int(edge_count)), np.kron(ip, np.eye(int(dimension))))


def setup(
    quadrature_order: int,
    dimension: int,
    edge_count: int,
    starts_at_corner: ArrayLike,
) -> tuple[np.ndarray, ...]:
    """Return MATLAB ``chnk.rcip.setup`` arrays using zero-based indices.

    The ``star`` indices are the fine nodes adjacent to the corner, while
    ``circ`` indices are the surrounding coarse/interface nodes. The ``L``
    arrays act on vector unknowns; ``L1`` is the scalar companion used when the
    same local topology is needed without operator components.
    """

    t, w, _, _ = lege.exps(int(quadrature_order))
    ip, ipw = IPinit(t, w)
    pbc = Pbcinit(ip, edge_count, dimension)
    pwbc = Pbcinit(ipw, edge_count, dimension)

    is_start = np.asarray(starts_at_corner, dtype=bool).reshape(-1)
    if is_start.size != int(edge_count):
        raise ValueError("starts_at_corner must have one entry per edge")

    quadrature_order = int(quadrature_order)
    dimension = int(dimension)
    edge_count = int(edge_count)
    ilist = np.zeros((2, edge_count), dtype=int)
    starL: list[int] = []
    circL: list[int] = []
    starL1: list[int] = []
    circL1: list[int] = []
    starS: list[int] = []
    circS: list[int] = []

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

    for iedge, edge_starts_at_corner in enumerate(is_start):
        offL = 3 * iedge * quadrature_order * dimension
        offL1 = 3 * iedge * quadrature_order
        offS = 2 * iedge * quadrature_order * dimension
        if edge_starts_at_corner:
            starL.extend((indb1 + offL).tolist())
            circL.extend((indg1 + offL).tolist())
            starL1.extend((indb11 + offL1).tolist())
            circL1.extend((indg11 + offL1).tolist())
            starS.extend((indb1s + offS).tolist())
            circS.extend((indg1s + offS).tolist())
            ilist[:, iedge] = [0, 1]
        else:
            starL.extend((indb0 + offL).tolist())
            circL.extend((indg0 + offL).tolist())
            starL1.extend((indb01 + offL1).tolist())
            circL1.extend((indg01 + offL1).tolist())
            starS.extend((indb0s + offS).tolist())
            circS.extend((indg0s + offS).tolist())
            ilist[:, iedge] = [1, 2]

    return (
        pbc,
        pwbc,
        np.array(starL, dtype=int),
        np.array(circL, dtype=int),
        np.array(starS, dtype=int),
        np.array(circS, dtype=int),
        ilist,
        np.array(starL1, dtype=int),
        np.array(circL1, dtype=int),
    )


def SchurBana(
    P: ArrayLike,
    PW: ArrayLike,
    K: ArrayLike,
    A: ArrayLike,
    starL: ArrayLike,
    circL: ArrayLike,
    starS: ArrayLike,
    circS: ArrayLike,
) -> np.ndarray:
    """Apply the Schur-Banachiewicz RCIP block inverse update.

    This is the local block algebra that folds a refined corner patch back into
    the coarse unknowns. ``P`` prolongs coarse values to fine values, ``PW`` is
    the weighted adjoint, and the ``star``/``circ`` index sets select the fine
    and interface blocks of the local matrix ``K``.
    """

    p = np.asarray(P)
    pw = np.asarray(PW)
    k = np.asarray(K)
    out = np.asarray(A).copy()
    star_l = np.asarray(starL, dtype=int)
    circ_l = np.asarray(circL, dtype=int)
    star_s = np.asarray(starS, dtype=int)
    circ_s = np.asarray(circS, dtype=int)

    va = k[np.ix_(circ_l, star_l)] @ out
    pta = pw.T @ out
    ptau = pta @ k[np.ix_(star_l, circ_l)]
    dvaui = np.linalg.inv(k[np.ix_(circ_l, circ_l)] - va @ k[np.ix_(star_l, circ_l)])
    dvauivap = dvaui @ (va @ p)
    out[np.ix_(star_s, star_s)] = pta @ p + ptau @ dvauivap
    out[np.ix_(circ_s, circ_s)] = dvaui
    out[np.ix_(circ_s, star_s)] = -dvauivap
    out[np.ix_(star_s, circ_s)] = -ptau @ dvaui
    return out
