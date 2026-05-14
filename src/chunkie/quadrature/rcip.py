"""RCIP utilities for corner-aware chunkgraph workflows.

This module mirrors the small linear-algebra setup helpers from MATLAB
``chnk.rcip`` and provides a conservative first Python interface for corner
refinement and compression metadata. The full recursive local compression
solver is intentionally not hidden behind this baseline.

RCIP is the corner path: sharp vertices get local dyadic refinement and a
compressed correction matrix so a global smooth-grid solve can represent the
near-corner singular density. The helpers here expose the prolongation,
weighted prolongation, Schur-Banachiewicz block update, and saved metadata used
by higher-level chunkgraph assembly; they do not silently run an incomplete full
corner solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_field_matrix, as_boundary_point_matrix, as_boundary_vector

from .. import lege
from ..geometry.chunker import Chunker, ChunkerPref, merge


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
    ctr: np.ndarray | None = None
    rcs: np.ndarray | None = None
    dcs: np.ndarray | None = None
    d2cs: np.ndarray | None = None
    dscal: np.ndarray | None = None
    d2scal: np.ndarray | None = None
    ileftright: np.ndarray | None = None
    glxs: np.ndarray | None = None
    glws: np.ndarray | None = None


@dataclass
class RCIPChunkGraphResult:
    vertices: np.ndarray
    edge_indices: list[np.ndarray]
    R: list[np.ndarray]
    saved: list[RCIPSaved]
    kernels: list[Any]


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


def Rcompchunk(
    chunker: list[Chunker] | tuple[Chunker, ...] | Chunker,
    edge_chunks: ArrayLike,
    kernel: Any,
    dimension: int,
    vertex: ArrayLike,
    Pbc: ArrayLike | None = None,
    PWbc: ArrayLike | None = None,
    starL: ArrayLike | None = None,
    circL: ArrayLike | None = None,
    starS: ArrayLike | None = None,
    circS: ArrayLike | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, RCIPSaved]:
    """Compute the RCIP compression matrix for chunks adjacent to a corner."""

    options = {} if options is None else dict(options)
    chunks = [chunker] if isinstance(chunker, Chunker) else list(chunker)
    if not chunks:
        raise ValueError("Rcompchunk requires at least one edge chunker")
    k = chunks[0].k
    dim = chunks[0].dim
    ndim = int(dimension)
    vert = np.asarray(vertex, dtype=float).reshape(dim)
    nsub = int(options.get("nsub", options.get("rcip_nsub", 0)))
    if nsub <= 0:
        edge_chunk_indices = np.asarray(edge_chunks, dtype=int)
        nedge0 = int(
            edge_chunk_indices.size
            if edge_chunk_indices.ndim == 1
            else edge_chunk_indices.shape[-1]
        )
        pbc, pwbc, sl, cl, ss, cs, ilist, sl1, cl1 = setup(
            k, ndim, nedge0, np.ones(nedge0, dtype=bool)
        )
        size = 2 * nedge0 * k * ndim
        rmat = np.eye(size)
        saved = RCIPSaved(
            k=k,
            ndim=ndim,
            nedge=nedge0,
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
            savedepth=0,
            R=[rmat],
            MAT=[],
            chnkrlocals=[],
            starind=np.arange(size, dtype=int),
        )
        return rmat, saved

    sbclmat, sbcrmat, lvmat, rvmat, u = shiftedlegbasismats(k)
    records = _rcip_edge_records(chunks, edge_chunks, vert, sbclmat, sbcrmat, lvmat, rvmat, u)
    nedge = len(records)
    isstart = np.array([rec["ileftright"] == 1 for rec in records], dtype=bool)

    if (
        Pbc is None
        or PWbc is None
        or starL is None
        or circL is None
        or starS is None
        or circS is None
    ):
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
    savedepth = int(options.get("rcip_savedepth", options.get("save_depth", 10)))
    savedepth = min(max(savedepth, 0), nsub)
    nsys = 3 * k * nedge * ndim
    pref = ChunkerPref(k=k, dim=dim, nchstor=5, nchmax=5)
    rmat: np.ndarray | None = None
    saved_R: list[np.ndarray | None] = [None] * (nsub + 1)
    saved_MAT: list[np.ndarray | None] = [None] * nsub
    saved_locals: list[Chunker | None] = [None] * nsub

    for level in range(1, nsub + 1):
        h = np.ones(nedge) / (2 ** (nsub - level))
        locals_: list[Chunker] = []
        for iedge, rec in enumerate(records):
            if rec["ileftright"] == -1:
                ts = (
                    np.array([0.0, 0.5, 1.0]) * h[iedge]
                    if level == nsub
                    else np.array([0.0, 0.5, 1.0, 2.0]) * h[iedge]
                )
            else:
                ts = (
                    -np.array([1.0, 0.5, 0.0]) * h[iedge]
                    if level == nsub
                    else -np.array([2.0, 1.0, 0.5, 0.0]) * h[iedge]
                )
            locals_.append(
                chunkerfunclocal(
                    lambda t, rec=rec: _shiftedcurve(
                        t,
                        rec["rcs"],
                        rec["dcs"],
                        rec["dscal"],
                        rec["d2cs"],
                        rec["d2scal"],
                        rec["ileftright"],
                    ),
                    ts,
                    pref,
                    chunks[0].tstor,
                    chunks[0].wstor,
                )
            )

        if level == nsub:
            for iedge, rec in enumerate(records):
                local = locals_[iedge]
                next_chunk = rec["nextchunk"]
                source_chunker = chunks[rec["chunker"]]
                old_nch = local.nch
                local.addchunk(1)
                local.rstor[:, :, old_nch] = (
                    source_chunker.r[:, :, next_chunk] - rec["ctr"][:, None]
                )
                local.dstor[:, :, old_nch] = source_chunker.d[:, :, next_chunk]
                local.d2stor[:, :, old_nch] = source_chunker.d2[:, :, next_chunk]
                if rec["ileftright"] == -1:
                    local.adjstor[0, old_nch] = old_nch
                    local.adjstor[1, old_nch] = -1
                    local.adjstor[1, old_nch - 1] = old_nch + 1
                else:
                    local.adjstor[0, old_nch] = -1
                    local.adjstor[1, old_nch] = 1
                    local.adjstor[0, 0] = old_nch + 1
                    local = local.sort()[0]
                local.recompute_geometry()
                locals_[iedge] = local

        ilistl = None if level == 1 else ilist
        mat = np.eye(nsys) + _local_chunkermat(locals_, kernel, ndim, ilistl)
        if level == 1:
            rmat = np.linalg.inv(mat[np.ix_(sl, sl)])
            if level >= nsub - savedepth + 1:
                saved_R[0] = rmat
        if savedepth < nsub and level == nsub - savedepth + 1:
            saved_R[level - 1] = rmat
        rmat = SchurBana(pbc, pwbc, mat, rmat, sl, cl, ss, cs)
        if level >= nsub - savedepth + 1:
            saved_R[level] = rmat
            saved_MAT[level - 1] = mat[np.ix_(sl, cl)]
            saved_locals[level - 1] = merge(locals_)

    assert rmat is not None
    saved = RCIPSaved(
        k=k,
        ndim=ndim,
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
        nsub=nsub,
        savedepth=savedepth,
        R=[item for item in saved_R if item is not None],
        MAT=[item for item in saved_MAT if item is not None],
        chnkrlocals=[item for item in saved_locals if item is not None],
        starind=np.arange(size, dtype=int),
        ctr=np.column_stack([rec["ctr"] for rec in records]),
        rcs=np.stack([rec["rcs"] for rec in records], axis=2),
        dcs=np.stack([rec["dcs"] for rec in records], axis=2),
        d2cs=np.stack([rec["d2cs"] for rec in records], axis=2),
        dscal=np.array([rec["dscal"] for rec in records]),
        d2scal=np.array([rec["d2scal"] for rec in records]),
        ileftright=np.array([rec["ileftright"] for rec in records], dtype=int),
        glxs=chunks[0].tstor.copy(),
        glws=chunks[0].wstor.copy(),
    )
    return rmat, saved


def shiftedlegbasismats(
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quadrature_order = int(quadrature_order)
    t0, _, u, v = lege.exps(quadrature_order)
    t = (t0 + 1.0) / 2.0
    basis = t[:, None] * v[:, :-1]
    uu, ss, vv_t = np.linalg.svd(basis, full_matrices=False)
    sbclmat = vv_t[: quadrature_order - 1, :].T @ (
        np.diag(1.0 / ss[: quadrature_order - 1]) @ uu[:, : quadrature_order - 1].T
    )

    t = (t0 - 1.0) / 2.0
    basis = t[:, None] * v[:, :-1]
    uu, ss, vv_t = np.linalg.svd(basis, full_matrices=False)
    sbcrmat = vv_t[: quadrature_order - 1, :].T @ (
        np.diag(1.0 / ss[: quadrature_order - 1]) @ uu[:, : quadrature_order - 1].T
    )

    pm1 = lege.pols(np.array([-1.0]), quadrature_order - 1)[0].reshape(-1)
    p1 = lege.pols(np.array([1.0]), quadrature_order - 1)[0].reshape(-1)
    leftvalmat = pm1 @ u
    rightvalmat = p1 @ u
    return sbclmat, sbcrmat, leftvalmat, rightvalmat, u


def chunkerfunclocal(
    fcurve: Any,
    ts: ArrayLike,
    pref: ChunkerPref | dict[str, Any] | None = None,
    xs: ArrayLike | None = None,
    ws: ArrayLike | None = None,
) -> Chunker:
    p = ChunkerPref.from_any(pref)
    tbreaks = np.asarray(ts, dtype=float).reshape(-1)
    if tbreaks.size < 2:
        raise ValueError("ts must contain at least two endpoints")
    xnodes = lege.exps(p.k)[0] if xs is None else np.asarray(xs, dtype=float).reshape(-1)
    wnodes = lege.exps(p.k)[1] if ws is None else np.asarray(ws, dtype=float).reshape(-1)
    r0, _, _ = fcurve(np.array([tbreaks[0]]))
    dim = np.asarray(r0).reshape(-1, 1).shape[0]
    out = Chunker(
        ChunkerPref(
            nchmax=max(p.nchmax, tbreaks.size - 1),
            k=p.k,
            dim=dim,
            nchstor=max(p.nchstor, tbreaks.size - 1),
        ),
        xnodes,
        wnodes,
    )
    out.addchunk(tbreaks.size - 1)
    out.adj = np.vstack((np.arange(0, out.nch), np.arange(2, out.nch + 2)))
    out.adj[0, 0] = -1
    out.adj[1, -1] = -1
    for ich in range(out.nch):
        a = tbreaks[ich]
        b = tbreaks[ich + 1]
        ts_panel = a + (b - a) * (xnodes + 1.0) / 2.0
        r, d, d2 = fcurve(ts_panel)
        h = (b - a) / 2.0
        out.rstor[:, :, ich] = np.asarray(r).reshape(dim, p.k)
        out.dstor[:, :, ich] = np.asarray(d).reshape(dim, p.k) * h
        out.d2stor[:, :, ich] = np.asarray(d2).reshape(dim, p.k) * h * h
    out.recompute_geometry()
    return out


def rhohatInterp(rhohat: ArrayLike, rcipsav: RCIPSaved | dict[str, Any], ndepth: int | None = None):
    """Interpolate a compressed RCIP density back through saved levels."""

    rho = as_boundary_vector(rhohat, name="rhohat")
    if isinstance(rcipsav, dict):
        nsub = int(rcipsav.get("nsub", 0))
        savedepth = int(rcipsav.get("savedepth", nsub))
        locals_ = rcipsav.get("chnkrlocals", [])
        rlist = rcipsav.get("R", [])
        matlist = rcipsav.get("MAT", [])
        pbc = np.asarray(rcipsav["Pbc"])
        star_s = np.sort(np.asarray(rcipsav["starS"], dtype=int).reshape(-1))
        circ_s = np.sort(np.asarray(rcipsav["circS"], dtype=int).reshape(-1))
        star_l1 = np.sort(np.asarray(rcipsav["starL1"], dtype=int).reshape(-1))
        circ_l1 = np.sort(np.asarray(rcipsav["circL1"], dtype=int).reshape(-1))
        nedge = int(rcipsav["nedge"])
        ileftright = np.asarray(rcipsav.get("ileftright", np.ones(nedge)), dtype=int).reshape(-1)
    else:
        nsub = rcipsav.nsub
        savedepth = int(rcipsav.savedepth)
        locals_ = [] if rcipsav.chnkrlocals is None else rcipsav.chnkrlocals
        rlist = [] if rcipsav.R is None else rcipsav.R
        matlist = [] if rcipsav.MAT is None else rcipsav.MAT
        pbc = rcipsav.Pbc
        star_s = np.sort(rcipsav.starS)
        circ_s = np.sort(rcipsav.circS)
        star_l1 = np.sort(rcipsav.starL1)
        circ_l1 = np.sort(rcipsav.circL1)
        nedge = rcipsav.nedge
        ileftright = np.ones(nedge, dtype=int) if rcipsav.ileftright is None else rcipsav.ileftright

    if nsub <= 0:
        return [rho.copy()], [None], [None]

    depth = nsub if ndepth is None else min(int(ndepth), nsub)
    if depth > savedepth:
        raise ValueError("requested interpolation depth exceeds saved RCIP depth")
    if len(rlist) < depth + 1 or len(matlist) < depth or len(locals_) < depth:
        raise ValueError("rcipsav does not contain enough saved recursion data")

    nrho = circ_s.size + star_s.size
    if rho.size % nrho != 0:
        raise ValueError("rhohat has incompatible size for RCIP saved data")
    ndens = rho.size // nrho
    rhohat0 = as_boundary_field_matrix(rho, nrho, ndens, name="rhohat")

    circ_s_edges = _split_edge_indices(circ_s, nedge)
    star_s_edges = _split_edge_indices(star_s, nedge)
    circ_l1_edges = _split_edge_indices(circ_l1, nedge)
    star_l1_edges = _split_edge_indices(star_l1, nedge)

    cl = locals_[-1]
    wt = as_boundary_vector(cl.wts, name="weights")
    rhohat_interpolation = [rhohat0[idx, :].copy() for idx in circ_s_edges]
    srcinfo = [_pointinfo_subset(cl, idx) for idx in circ_l1_edges]
    wts = [wt[idx].copy() for idx in circ_l1_edges]

    r0 = rlist[-1]
    for idepth in range(1, depth + 1):
        r1 = rlist[-idepth - 1]
        mat = matlist[-idepth]
        rhotemp = np.linalg.solve(r0, rhohat0)
        rhohat0 = r1 @ (pbc @ rhotemp[star_s, :] - mat @ rhohat0[circ_s, :])
        if idepth == depth:
            for iedge in range(nedge):
                order = (
                    np.concatenate((circ_s_edges[iedge], star_s_edges[iedge]))
                    if int(ileftright[iedge]) == 1
                    else np.concatenate((star_s_edges[iedge], circ_s_edges[iedge]))
                )
                rhohat_interpolation[iedge] = np.vstack(
                    (rhohat_interpolation[iedge], rhohat0[order, :])
                )
                srcinfo[iedge] = _pointinfo_append(
                    srcinfo[iedge], _pointinfo_subset(cl, star_l1_edges[iedge])
                )
                wts[iedge] = np.concatenate((wts[iedge], wt[star_l1_edges[iedge]]))
        else:
            cl = locals_[-idepth - 1]
            wt = as_boundary_vector(cl.wts, name="weights")
            for iedge in range(nedge):
                rhohat_interpolation[iedge] = np.vstack(
                    (rhohat_interpolation[iedge], rhohat0[circ_s_edges[iedge], :])
                )
                srcinfo[iedge] = _pointinfo_append(
                    srcinfo[iedge], _pointinfo_subset(cl, circ_l1_edges[iedge])
                )
                wts[iedge] = np.concatenate((wts[iedge], wt[circ_l1_edges[iedge]]))
        r0 = r1

    if ndens == 1:
        rhohat_interpolation = [vals[:, 0] for vals in rhohat_interpolation]
    return rhohat_interpolation, srcinfo, wts


def corner_refine(
    cg: Any, vertices: ArrayLike | None = None, depth: int = 1, stype: str = "a"
) -> Any:
    """Dyadically refine chunks adjacent to selected chunkgraph vertices."""

    out = cg.copy()
    if vertices is None:
        vinds = range(len(out.vstruc))
    else:
        vinds = np.asarray(vertices, dtype=int).reshape(-1)
    for _ in range(int(depth)):
        for ivert in vinds:
            edges, signs = out.vstruc[int(ivert)]
            for edge, sign in zip(edges, signs, strict=True):
                ch = out.echnks[int(edge)]
                ch.split(0 if sign < 0 else ch.nch - 1, stype=stype)
    return out


def chunkgraph_rcip(
    graph: Any,
    kernel: Any,
    dimension: int,
    vertices: ArrayLike | None = None,
    options: dict[str, Any] | None = None,
    ignore_vertices: ArrayLike | None = None,
) -> RCIPChunkGraphResult:
    """Run RCIP compression at selected chunkgraph vertices.

    ``kernel`` may be a scalar kernel/callable used at every local corner or a
    global edge-by-edge block matrix. Global block matrices are restricted to
    the incident edges of each vertex before calling ``Rcompchunk``.
    """

    if not hasattr(graph, "echnks") or not hasattr(graph, "vstruc") or not hasattr(graph, "verts"):
        raise TypeError("chunkgraph_rcip expects a chunkgraph-like object")
    nvert = int(graph.verts.shape[1])
    vinds = _normalize_vertex_list(np.arange(nvert) if vertices is None else vertices, nvert)
    ignored = set(
        _normalize_vertex_list([] if ignore_vertices is None else ignore_vertices, nvert).tolist()
    )
    options = {} if options is None else dict(options)

    used_vertices: list[int] = []
    edge_indices: list[np.ndarray] = []
    rmats: list[np.ndarray] = []
    saved_list: list[RCIPSaved] = []
    kernels: list[Any] = []

    for ivert in vinds:
        iv = int(ivert)
        if iv in ignored:
            continue
        edges, _ = graph.vstruc[iv]
        edges = np.asarray(edges, dtype=int).reshape(-1)
        if edges.size < 2:
            continue
        local_kernel = _select_vertex_kernel(kernel, edges)
        rmat, saved = Rcompchunk(
            graph.echnks,
            edges,
            local_kernel,
            dimension,
            graph.verts[:, iv],
            options=options,
        )
        used_vertices.append(iv)
        edge_indices.append(edges.copy())
        rmats.append(rmat)
        saved_list.append(saved)
        kernels.append(local_kernel)

    return RCIPChunkGraphResult(
        vertices=np.array(used_vertices, dtype=int),
        edge_indices=edge_indices,
        R=rmats,
        saved=saved_list,
        kernels=kernels,
    )


def _rcip_edge_records(
    chunks: list[Chunker],
    edge_chunks: ArrayLike,
    vertex: np.ndarray,
    sbclmat: np.ndarray,
    sbcrmat: np.ndarray,
    lvmat: np.ndarray,
    rvmat: np.ndarray,
    u: np.ndarray,
) -> list[dict[str, Any]]:
    pairs = _normalize_edge_chunk_pairs(chunks, edge_chunks, vertex)
    out: list[dict[str, Any]] = []
    for chunker_idx, chunk_idx in pairs:
        ch = chunks[int(chunker_idx)]
        r = ch.r[:, :, int(chunk_idx)].copy()
        d = ch.d[:, :, int(chunk_idx)]
        d2 = ch.d2[:, :, int(chunk_idx)]
        left, right = ch.adj[:, int(chunk_idx)]
        if left > 0 and right < 0:
            nextchunk = int(left) - 1
            ileftright = 1
            ctr = (rvmat @ r.T).reshape(ch.dim)
            rcentered = r - ctr[:, None]
            rcs = sbcrmat @ rcentered.T
        elif left < 0 and right > 0:
            nextchunk = int(right) - 1
            ileftright = -1
            ctr = (lvmat @ r.T).reshape(ch.dim)
            rcentered = r - ctr[:, None]
            rcs = sbclmat @ rcentered.T
        else:
            raise ValueError("RCIP edge chunk must be adjacent to one vertex and one neighbor")
        out.append(
            {
                "chunker": int(chunker_idx),
                "chunk": int(chunk_idx),
                "nextchunk": nextchunk,
                "ctr": ctr,
                "rcs": rcs,
                "dcs": u @ d.T,
                "d2cs": u @ d2.T,
                "dscal": 2.0,
                "d2scal": 4.0,
                "ileftright": ileftright,
            }
        )
    return out


def _normalize_edge_chunk_pairs(
    chunks: list[Chunker], edge_chunks: ArrayLike, vertex: np.ndarray
) -> list[tuple[int, int]]:
    arr = np.asarray(edge_chunks, dtype=int)
    if arr.ndim == 1:
        pairs = []
        for chunker_idx in arr.reshape(-1):
            ch = chunks[int(chunker_idx)]
            pairs.append((int(chunker_idx), _chunk_adjacent_to_vertex(ch, vertex)))
        return pairs
    if arr.shape[0] != 2:
        raise ValueError("edge_chunks must be a 1D chunker list or a 2 x nedge array")
    pairs = []
    for col in range(arr.shape[1]):
        chunker_idx = int(arr[0, col])
        chunk_idx = int(arr[1, col])
        if chunker_idx >= len(chunks) and chunker_idx - 1 >= 0:
            chunker_idx -= 1
        if chunk_idx >= chunks[chunker_idx].nch and chunk_idx - 1 >= 0:
            chunk_idx -= 1
        pairs.append((chunker_idx, chunk_idx))
    return pairs


def _chunk_adjacent_to_vertex(chunker: Chunker, vertex: np.ndarray) -> int:
    rend, _ = chunker.chunkends()
    candidates: list[tuple[float, int]] = []
    for ich in range(chunker.nch):
        if chunker.adj[0, ich] < 0:
            candidates.append((float(np.linalg.norm(rend[:, 0, ich] - vertex)), ich))
        if chunker.adj[1, ich] < 0:
            candidates.append((float(np.linalg.norm(rend[:, 1, ich] - vertex)), ich))
    if not candidates:
        raise ValueError("could not identify a vertex-adjacent chunk")
    return min(candidates, key=lambda item: item[0])[1]


def _shiftedcurve(
    t: ArrayLike,
    rc: np.ndarray,
    dc: np.ndarray,
    scald: float,
    d2c: np.ndarray,
    scald2: float,
    ilr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tt_in = np.asarray(t, dtype=float).reshape(-1)
    tt = 2.0 * tt_in - 1.0 if int(ilr) == -1 else 2.0 * tt_in + 1.0
    pols = lege.pols(tt, dc.shape[0] - 1)[0].T
    r = (tt_in[:, None] * (pols[:, : rc.shape[0]] @ rc)).T
    d = (float(scald) * (pols @ dc)).T
    d2 = (float(scald2) * (pols @ d2c)).T
    return r, d, d2


def _local_chunkermat(
    chunks: list[Chunker],
    kernel: Any,
    dimension: int,
    ilist: np.ndarray | None = None,
) -> np.ndarray:
    from ..operators import chunkerkernevalmat, chunkermat
    from . import ggq as quadggq
    from . import native as quadnative

    starts = np.cumsum([0] + [ch.npt * int(dimension) for ch in chunks])
    out = np.zeros((starts[-1], starts[-1]))
    for itarg, target in enumerate(chunks):
        rows = slice(starts[itarg], starts[itarg + 1])
        for isrc, source in enumerate(chunks):
            cols = slice(starts[isrc], starts[isrc + 1])
            block_kernel = _select_local_kernel(kernel, itarg, isrc)
            opdims = _kernel_opdims(block_kernel, dimension)
            if itarg == isrc:
                if ilist is not None and getattr(block_kernel, "sing", "") in {"log", "pv", "hs"}:
                    block = quadggq.buildmat(
                        source,
                        block_kernel,
                        opdims,
                        getattr(block_kernel, "sing", "log"),
                        ilist=ilist[:, isrc],
                    )
                elif getattr(block_kernel, "sing", "") in {"log", "pv", "hs"}:
                    block = chunkermat(source, block_kernel)
                else:
                    block = quadnative.buildmat(source, block_kernel, opdims)
            else:
                block = chunkerkernevalmat(source, block_kernel, target, quadrature="smooth")
            block_arr = _zero_coincident_nonfinite_block(
                block,
                source,
                target,
                opdims,
                f"RCIP local matrix block ({itarg}, {isrc})",
            )
            out[rows, cols] = block_arr
    return out


def _zero_coincident_nonfinite_block(
    values: np.ndarray,
    source: Chunker,
    target: Chunker,
    opdims: tuple[int, int],
    context: str,
) -> np.ndarray:
    arr = np.array(values, copy=True)
    nonfinite = ~np.isfinite(arr)
    if not np.any(nonfinite):
        return arr

    expected = _coincident_chunker_mask(source, target, opdims, arr.shape)
    unexpected = nonfinite & ~expected
    if np.any(unexpected):
        raise ValueError(
            f"{context} contains non-finite values away from coincident source/target points"
        )
    arr[nonfinite] = 0.0
    return arr


def _coincident_chunker_mask(
    source: Chunker,
    target: Chunker,
    opdims: tuple[int, int],
    shape: tuple[int, ...],
) -> np.ndarray:
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    src_pts = as_boundary_point_matrix(source.r, source.dim, source.npt, name="source positions")
    targ_pts = as_boundary_point_matrix(target.r, target.dim, target.npt, name="target positions")
    expected_shape = (op0 * target.npt, op1 * source.npt)
    if tuple(shape) != expected_shape:
        return np.zeros(shape, dtype=bool)
    tol = (
        16.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.max(np.abs(src_pts))) if src_pts.size else 0.0,
            float(np.max(np.abs(targ_pts))) if targ_pts.size else 0.0,
        )
    )
    dist2 = np.sum((targ_pts[:, :, None] - src_pts[:, None, :]) ** 2, axis=0)
    coincident = dist2 <= tol**2
    return np.repeat(np.repeat(coincident, op0, axis=0), op1, axis=1)


def _select_local_kernel(kernel: Any, target_index: int, source_index: int) -> Any:
    arr = (
        np.asarray(kernel, dtype=object) if isinstance(kernel, (list, tuple, np.ndarray)) else None
    )
    if arr is not None and arr.ndim == 2:
        return arr[target_index, source_index]
    return kernel


def _select_vertex_kernel(kernel: Any, edges: np.ndarray) -> Any:
    arr = (
        np.asarray(kernel, dtype=object) if isinstance(kernel, (list, tuple, np.ndarray)) else None
    )
    if arr is None or arr.ndim != 2:
        return kernel
    if arr.shape == (edges.size, edges.size):
        return arr
    if edges.size and (np.max(edges) >= arr.shape[0] or np.max(edges) >= arr.shape[1]):
        raise ValueError("global RCIP block kernel matrix is too small for selected vertex edges")
    return arr[np.ix_(edges, edges)]


def _normalize_vertex_list(vertices: ArrayLike, nvert: int) -> np.ndarray:
    arr = np.asarray(vertices, dtype=int).reshape(-1)
    if arr.size and np.max(arr) >= int(nvert):
        if np.min(arr) >= 1 and np.max(arr) <= int(nvert):
            arr = arr - 1
        else:
            raise ValueError("vertex index out of range")
    if np.any(arr < 0) or np.any(arr >= int(nvert)):
        raise ValueError("vertex index out of range")
    return arr


def _kernel_opdims(kernel: Any, dimension: int) -> tuple[int, int]:
    opdims = getattr(kernel, "opdims", None)
    if opdims is None or opdims == (0, 0):
        return (int(dimension), int(dimension))
    return tuple(int(x) for x in opdims)


def _split_edge_indices(indices: np.ndarray, nedge: int) -> list[np.ndarray]:
    if indices.size % int(nedge) != 0:
        raise ValueError("RCIP saved indices are not evenly split by edge")
    per_edge = indices.size // int(nedge)
    return [indices[i * per_edge : (i + 1) * per_edge] for i in range(int(nedge))]


def _pointinfo_subset(chunker: Chunker, indices: np.ndarray) -> Any:
    from ..operators import PointInfo

    return PointInfo(
        r=as_boundary_point_matrix(chunker.r, chunker.dim, chunker.npt, name="positions")[
            :, indices
        ],
        d=as_boundary_point_matrix(chunker.d, chunker.dim, chunker.npt, name="derivatives")[
            :, indices
        ],
        d2=as_boundary_point_matrix(
            chunker.d2, chunker.dim, chunker.npt, name="second derivatives"
        )[:, indices],
        n=as_boundary_point_matrix(chunker.n, chunker.dim, chunker.npt, name="normals")[:, indices],
    )


def _pointinfo_append(left: Any, right: Any) -> Any:
    from ..operators import PointInfo

    return PointInfo(
        r=np.column_stack((left.r, right.r)),
        d=np.column_stack((left.d, right.d)),
        d2=np.column_stack((left.d2, right.d2)),
        n=np.column_stack((left.n, right.n)),
    )
