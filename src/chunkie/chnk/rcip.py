"""RCIP utilities for corner-aware chunkgraph workflows.

This module mirrors the small linear-algebra setup helpers from MATLAB
``chnk.rcip`` and provides a conservative first Python interface for corner
refinement and compression metadata. The full recursive local compression
solver is intentionally not hidden behind this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from ..chunker import Chunker


@dataclass
class RCIPSaved:
    k: int
    ndim: int
    nedge: int
    Pbc: np.ndarray
    PWbc: np.ndarray
    starL: np.ndarray
    circL: np.ndarray
    starS: np.ndarray
    circS: np.ndarray
    ilist: np.ndarray
    starL1: np.ndarray
    circL1: np.ndarray
    nsub: int = 0
    savedepth: float = np.inf
    R: list[np.ndarray] | None = None
    MAT: list[np.ndarray] | None = None
    chnkrlocals: list[Chunker] | None = None
    starind: np.ndarray | None = None


def IPinit(T: ArrayLike, W: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Build the RCIP prolongation matrix and weighted prolongation."""

    t = np.asarray(T, dtype=float).reshape(-1)
    w = np.asarray(W, dtype=float).reshape(-1)
    if t.size != w.size:
        raise ValueError("T and W must have the same length")

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


def Pbcinit(IP: ArrayLike, nedge: int, ndim: int) -> np.ndarray:
    """Construct the block diagonal RCIP prolongation for all edge unknowns."""

    ip = np.asarray(IP)
    return np.kron(np.eye(int(nedge)), np.kron(ip, np.eye(int(ndim))))


def setup(ngl: int, ndim: int, nedge: int, isstart: ArrayLike) -> tuple[np.ndarray, ...]:
    """Return MATLAB ``chnk.rcip.setup`` arrays using zero-based indices."""

    t, w, _, _ = lege.exps(int(ngl))
    ip, ipw = IPinit(t, w)
    pbc = Pbcinit(ip, nedge, ndim)
    pwbc = Pbcinit(ipw, nedge, ndim)

    is_start = np.asarray(isstart, dtype=bool).reshape(-1)
    if is_start.size != int(nedge):
        raise ValueError("isstart must have one entry per edge")

    ngl = int(ngl)
    ndim = int(ndim)
    nedge = int(nedge)
    ilist = np.zeros((2, nedge), dtype=int)
    starL: list[int] = []
    circL: list[int] = []
    starL1: list[int] = []
    circL1: list[int] = []
    starS: list[int] = []
    circS: list[int] = []

    indg1 = 2 * ngl * ndim + np.arange(ngl * ndim)
    indb1 = np.arange(2 * ngl * ndim)
    indg11 = 2 * ngl + np.arange(ngl)
    indb11 = np.arange(2 * ngl)
    indg0 = np.arange(ngl * ndim)
    indb0 = ngl * ndim + np.arange(2 * ngl * ndim)
    indg01 = np.arange(ngl)
    indb01 = ngl + np.arange(2 * ngl)

    indg1s = ngl * ndim + np.arange(ngl * ndim)
    indb1s = np.arange(ngl * ndim)
    indg0s = np.arange(ngl * ndim)
    indb0s = ngl * ndim + np.arange(ngl * ndim)

    for iedge, starts_at_corner in enumerate(is_start):
        offL = 3 * iedge * ngl * ndim
        offL1 = 3 * iedge * ngl
        offS = 2 * iedge * ngl * ndim
        if starts_at_corner:
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
    """Apply the Schur-Banachiewicz RCIP block inverse update."""

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


def Rcompchunk(
    chnkr: list[Chunker] | tuple[Chunker, ...] | Chunker,
    iedgechunks: ArrayLike,
    fkern: Any,
    ndim: int,
    vert0: ArrayLike,
    Pbc: ArrayLike | None = None,
    PWbc: ArrayLike | None = None,
    starL: ArrayLike | None = None,
    circL: ArrayLike | None = None,
    starS: ArrayLike | None = None,
    circS: ArrayLike | None = None,
    opts: dict[str, Any] | None = None,
) -> tuple[np.ndarray, RCIPSaved]:
    """Return an identity RCIP compression and MATLAB-like saved metadata."""

    _ = fkern
    _ = np.asarray(vert0, dtype=float)
    options = {} if opts is None else dict(opts)
    chunks = [chnkr] if isinstance(chnkr, Chunker) else list(chnkr)
    if not chunks:
        raise ValueError("Rcompchunk requires at least one edge chunker")
    k = chunks[0].k
    edge_chunks = np.asarray(iedgechunks, dtype=int)
    nedge = int(edge_chunks.size if edge_chunks.ndim == 1 else edge_chunks.shape[-1])
    isstart = np.ones(nedge, dtype=bool)

    if Pbc is None or PWbc is None or starL is None or circL is None or starS is None or circS is None:
        pbc, pwbc, sl, cl, ss, cs, ilist, sl1, cl1 = setup(k, ndim, nedge, isstart)
    else:
        pbc = np.asarray(Pbc)
        pwbc = np.asarray(PWbc)
        sl = np.asarray(starL, dtype=int)
        cl = np.asarray(circL, dtype=int)
        ss = np.asarray(starS, dtype=int)
        cs = np.asarray(circS, dtype=int)
        ilist = np.vstack((np.zeros(nedge, dtype=int), np.ones(nedge, dtype=int)))
        sl1 = sl // int(ndim)
        cl1 = cl // int(ndim)

    size = 2 * nedge * k * int(ndim)
    rmat = np.eye(size)
    saved = RCIPSaved(
        k=k,
        ndim=int(ndim),
        nedge=nedge,
        Pbc=pbc,
        PWbc=pwbc,
        starL=sl,
        circL=cl,
        starS=ss,
        circS=cs,
        ilist=ilist,
        starL1=sl1,
        circL1=cl1,
        nsub=0,
        savedepth=float(options.get("rcip_savedepth", options.get("save_depth", np.inf))),
        R=[rmat],
        MAT=[],
        chnkrlocals=[],
        starind=np.arange(size, dtype=int),
    )
    return rmat, saved


def rhohatInterp(rhohat: ArrayLike, rcipsav: RCIPSaved | dict[str, Any], ndepth: int | None = None):
    """Return saved-level interpolants for an already compressed density."""

    rho = np.asarray(rhohat).reshape(-1, order="F")
    if isinstance(rcipsav, dict):
        nsub = int(rcipsav.get("nsub", 0))
        locals_ = rcipsav.get("chnkrlocals", [])
    else:
        nsub = rcipsav.nsub
        locals_ = [] if rcipsav.chnkrlocals is None else rcipsav.chnkrlocals
    depth = nsub if ndepth is None else min(int(ndepth), nsub)
    count = max(depth, 1)
    srcinfo = [local.sourceinfo if hasattr(local, "sourceinfo") else None for local in locals_[:count]]
    wts = [local.wts.reshape(-1, order="F") for local in locals_[:count]]
    if not srcinfo:
        srcinfo = [None]
        wts = [None]
    return [rho.copy() for _ in range(count)], srcinfo, wts


def corner_refine(cg: Any, vertices: ArrayLike | None = None, depth: int = 1, stype: str = "a") -> Any:
    """Dyadically refine chunks adjacent to selected chunkgraph vertices."""

    out = cg.copy()
    if vertices is None:
        vinds = range(len(out.vstruc))
    else:
        vinds = np.asarray(vertices, dtype=int).reshape(-1)
    for _ in range(int(depth)):
        for ivert in vinds:
            edges, signs = out.vstruc[int(ivert)]
            for edge, sign in zip(edges, signs):
                ch = out.echnks[int(edge)]
                ch.split(0 if sign < 0 else ch.nch - 1, stype=stype)
    return out


ipinit = IPinit
pbcinit = Pbcinit
schurbana = SchurBana
rcompchunk = Rcompchunk
rhohatinterp = rhohatInterp
