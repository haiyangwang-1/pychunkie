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
from ..chunker import Chunker, ChunkerPref, merge


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
    """Compute the RCIP compression matrix for chunks adjacent to a corner."""

    options = {} if opts is None else dict(opts)
    chunks = [chnkr] if isinstance(chnkr, Chunker) else list(chnkr)
    if not chunks:
        raise ValueError("Rcompchunk requires at least one edge chunker")
    k = chunks[0].k
    dim = chunks[0].dim
    ndim = int(ndim)
    vert = np.asarray(vert0, dtype=float).reshape(dim)
    nsub = int(options.get("nsub", options.get("rcip_nsub", 0)))
    if nsub <= 0:
        edge_chunks = np.asarray(iedgechunks, dtype=int)
        nedge0 = int(edge_chunks.size if edge_chunks.ndim == 1 else edge_chunks.shape[-1])
        pbc, pwbc, sl, cl, ss, cs, ilist, sl1, cl1 = setup(k, ndim, nedge0, np.ones(nedge0, dtype=bool))
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
    records = _rcip_edge_records(chunks, iedgechunks, vert, sbclmat, sbcrmat, lvmat, rvmat, u)
    nedge = len(records)
    isstart = np.array([rec["ileftright"] == 1 for rec in records], dtype=bool)

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
                ts = np.array([0.0, 0.5, 1.0]) * h[iedge] if level == nsub else np.array([0.0, 0.5, 1.0, 2.0]) * h[iedge]
            else:
                ts = -np.array([1.0, 0.5, 0.0]) * h[iedge] if level == nsub else -np.array([2.0, 1.0, 0.5, 0.0]) * h[iedge]
            locals_.append(
                chunkerfunclocal(
                    lambda t, rec=rec: _shiftedcurve(t, rec["rcs"], rec["dcs"], rec["dscal"], rec["d2cs"], rec["d2scal"], rec["ileftright"]),
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
                local.rstor[:, :, old_nch] = source_chunker.r[:, :, next_chunk] - rec["ctr"][:, None]
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
        mat = np.eye(nsys) + _local_chunkermat(locals_, fkern, ndim, ilistl)
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


def shiftedlegbasismats(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t0, _, u, v = lege.exps(int(k))
    t = (t0 + 1.0) / 2.0
    basis = t[:, None] * v[:, :-1]
    uu, ss, vv_t = np.linalg.svd(basis, full_matrices=False)
    sbclmat = vv_t[: k - 1, :].T @ (np.diag(1.0 / ss[: k - 1]) @ uu[:, : k - 1].T)

    t = (t0 - 1.0) / 2.0
    basis = t[:, None] * v[:, :-1]
    uu, ss, vv_t = np.linalg.svd(basis, full_matrices=False)
    sbcrmat = vv_t[: k - 1, :].T @ (np.diag(1.0 / ss[: k - 1]) @ uu[:, : k - 1].T)

    pm1 = lege.pols(np.array([-1.0]), k - 1)[0].reshape(-1)
    p1 = lege.pols(np.array([1.0]), k - 1)[0].reshape(-1)
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
    out = Chunker(ChunkerPref(nchmax=max(p.nchmax, tbreaks.size - 1), k=p.k, dim=dim, nchstor=max(p.nchstor, tbreaks.size - 1)), xnodes, wnodes)
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


def _rcip_edge_records(
    chunks: list[Chunker],
    iedgechunks: ArrayLike,
    vert: np.ndarray,
    sbclmat: np.ndarray,
    sbcrmat: np.ndarray,
    lvmat: np.ndarray,
    rvmat: np.ndarray,
    u: np.ndarray,
) -> list[dict[str, Any]]:
    pairs = _normalize_edge_chunk_pairs(chunks, iedgechunks, vert)
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


def _normalize_edge_chunk_pairs(chunks: list[Chunker], iedgechunks: ArrayLike, vert: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(iedgechunks, dtype=int)
    if arr.ndim == 1:
        pairs = []
        for chunker_idx in arr.reshape(-1):
            ch = chunks[int(chunker_idx)]
            pairs.append((int(chunker_idx), _chunk_adjacent_to_vertex(ch, vert)))
        return pairs
    if arr.shape[0] != 2:
        raise ValueError("iedgechunks must be a 1D chunker list or a 2 x nedge array")
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


def _chunk_adjacent_to_vertex(chnkr: Chunker, vert: np.ndarray) -> int:
    rend, _ = chnkr.chunkends()
    candidates: list[tuple[float, int]] = []
    for ich in range(chnkr.nch):
        if chnkr.adj[0, ich] < 0:
            candidates.append((float(np.linalg.norm(rend[:, 0, ich] - vert)), ich))
        if chnkr.adj[1, ich] < 0:
            candidates.append((float(np.linalg.norm(rend[:, 1, ich] - vert)), ich))
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
    fkern: Any,
    ndim: int,
    ilist: np.ndarray | None = None,
) -> np.ndarray:
    from ..operators import chunkerkernevalmat, chunkermat
    from . import quadggq, quadnative

    starts = np.cumsum([0] + [ch.npt * int(ndim) for ch in chunks])
    out = np.zeros((starts[-1], starts[-1]))
    for itarg, targ in enumerate(chunks):
        rows = slice(starts[itarg], starts[itarg + 1])
        for isrc, src in enumerate(chunks):
            cols = slice(starts[isrc], starts[isrc + 1])
            kern = _select_local_kernel(fkern, itarg, isrc)
            opdims = _kernel_opdims(kern, ndim)
            if itarg == isrc:
                if ilist is not None and getattr(kern, "sing", "") in {"log", "pv", "hs"}:
                    block = quadggq.buildmat(src, kern, opdims, getattr(kern, "sing", "log"), ilist=ilist[:, isrc])
                elif getattr(kern, "sing", "") in {"log", "pv", "hs"}:
                    block = chunkermat(src, kern)
                else:
                    block = quadnative.buildmat(src, kern, opdims)
            else:
                block = chunkerkernevalmat(src, kern, targ, {"forcesmooth": True})
            out[rows, cols] = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _select_local_kernel(fkern: Any, itarg: int, isrc: int) -> Any:
    arr = np.asarray(fkern, dtype=object) if isinstance(fkern, (list, tuple, np.ndarray)) else None
    if arr is not None and arr.ndim == 2:
        return arr[itarg, isrc]
    return fkern


def _kernel_opdims(kern: Any, ndim: int) -> tuple[int, int]:
    opdims = getattr(kern, "opdims", None)
    if opdims is None or opdims == (0, 0):
        return (int(ndim), int(ndim))
    return tuple(int(x) for x in opdims)


ipinit = IPinit
pbcinit = Pbcinit
schurbana = SchurBana
rcompchunk = Rcompchunk
rhohatinterp = rhohatInterp
shiftedlegbasismats = shiftedlegbasismats
chunkerfunclocal = chunkerfunclocal
