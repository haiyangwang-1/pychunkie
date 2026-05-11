"""Adaptive-special quadrature entry points.

The MATLAB implementation adaptively decides which close interactions need
replacement quadrature. This Python baseline exposes the same build path
and delegates the actual replacement blocks to ``quadggq``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..chunker import Chunker
from .. import lege
from ..operators import PointInfo
from . import quadggq, quadnative


def buildmat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    options = {} if opts is None else dict(opts)
    qtype = str(options.get("sing", getattr(kern, "sing", "log") or "log")).lower()
    if qtype != "log":
        return quadggq.buildmat(
            chnkr,
            kern,
            opdims if opdims is not None else getattr(kern, "opdims", None),
            type=qtype,
            ilist=options.get("ilist", None),
        )
    if opdims is None:
        opdims = getattr(kern, "opdims", None)
    if opdims is None or opdims == (0, 0):
        raise ValueError("opdims must be provided for adaptive quadrature assembly")

    op0 = int(opdims[0])
    op1 = int(opdims[1])
    aux = quadggq.setup(chnkr.k, qtype)
    mat = quadnative.buildmat(chnkr, kern, (op0, op1))
    nodes, weights = lege.exps(max(27, chnkr.k + 1))[:2]
    bary = lege.barywts(chnkr.k, chnkr.tstor)
    ignored = set() if options.get("ilist", None) is None else {
        int(idx) for idx in np.asarray(options.get("ilist"), dtype=int).reshape(-1)
    }

    for src_chunk in range(chnkr.nch):
        src_cols = _block_slice(src_chunk, chnkr.k, op1)
        left, right = chnkr.adj[:, src_chunk]
        for targ_chunk in (int(left) - 1, int(right) - 1):
            if targ_chunk < 0 or targ_chunk >= chnkr.nch:
                continue
            if src_chunk in ignored and targ_chunk in ignored:
                continue
            rows = _block_slice(targ_chunk, chnkr.k, op0)
            targinfo = _chunk_pointinfo(chnkr, targ_chunk)
            mat[rows, src_cols] = adapgausswts(
                chnkr, src_chunk, targinfo, kern, (op0, op1), nodes, weights, bary, options
            )[0]

        if src_chunk not in ignored:
            rows = _block_slice(src_chunk, chnkr.k, op0)
            mat[rows, src_cols] = quadggq.diagbuildmat(chnkr, src_chunk, kern, (op0, op1), aux)

    if bool(options.get("robust", False)):
        _apply_robust_close_corrections(mat, chnkr, kern, (op0, op1), nodes, weights, bary, options, ignored)

    return mat


def adapgausswts(
    chnkr: Chunker,
    src_chunk: int,
    targinfo: PointInfo,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    nodes: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    barywts: ArrayLike | None = None,
    opts: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive source-panel quadrature weights for one chunk and target set."""

    options = {} if opts is None else dict(opts)
    eps = float(options.get("eps", 1.0e-12))
    maxints = int(options.get("maxints", 100000))
    maxdepth = int(options.get("maxdepth", 52))
    t = np.asarray(lege.exps(max(27, chnkr.k + 1))[0] if nodes is None else nodes, dtype=float)
    w = np.asarray(lege.exps(max(27, chnkr.k + 1))[1] if weights is None else weights, dtype=float)
    bw = np.asarray(lege.barywts(chnkr.k, chnkr.tstor) if barywts is None else barywts, dtype=float)

    op0 = int(opdims[0])
    op1 = int(opdims[1])
    ntarg = int(targinfo.r.shape[1])
    mat = np.zeros((op0 * ntarg, op1 * chnkr.k), dtype=quadggq._kernel_dtype(chnkr, kern))
    maxrecs = np.zeros(ntarg, dtype=int)
    numints = np.zeros(ntarg, dtype=int)
    iers = np.zeros(ntarg, dtype=int)

    source = _chunk_source_arrays(chnkr, src_chunk)
    for itarg in range(ntarg):
        one_targ = _single_target(targinfo, itarg)
        initial = _adaptive_panel_integral(-1.0, 1.0, source, chnkr.tstor, bw, one_targ, kern, (op0, op1), t, w)
        stack = [(-1.0, 1.0, initial)]
        accum = np.zeros_like(initial)
        for count in range(1, maxints + 1):
            numints[itarg] = count
            maxrecs[itarg] = max(maxrecs[itarg], len(stack))
            a, b, parent = stack.pop()
            mid = 0.5 * (a + b)
            left = _adaptive_panel_integral(a, mid, source, chnkr.tstor, bw, one_targ, kern, (op0, op1), t, w)
            right = _adaptive_panel_integral(mid, b, source, chnkr.tstor, bw, one_targ, kern, (op0, op1), t, w)
            if np.max(np.abs(left + right - parent)) <= eps:
                accum = accum + left + right
                if not stack:
                    break
                continue
            if len(stack) + 2 > maxdepth:
                iers[itarg] = 8
                break
            stack.append((mid, b, right))
            stack.append((a, mid, left))
        else:
            iers[itarg] = 16
        mat[op0 * itarg : op0 * (itarg + 1), :] = accum
    return mat, maxrecs, numints, iers


def _adaptive_panel_integral(
    a: float,
    b: float,
    source: dict[str, np.ndarray | None],
    ct: np.ndarray,
    bw: np.ndarray,
    targ: PointInfo,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0
    tt = scale * nodes + shift
    interp = _barycentric_matrix(ct, bw, tt)
    src = PointInfo(
        r=source["r"] @ interp,
        d=source["d"] @ interp,
        d2=source["d2"] @ interp,
        n=source["n"] @ interp,
        data=None if source["data"] is None else source["data"] @ interp,
    )
    speed = np.sqrt(np.sum(np.abs(src.d) ** 2, axis=0))
    dsdt = scale * weights * speed
    kvals = quadggq._eval_kernel(kern, src, targ)
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    out = np.zeros((op0, op1 * ct.size), dtype=np.result_type(kvals, dsdt))
    for inode in range(tt.size):
        block = kvals[:, op1 * inode : op1 * (inode + 1)] * dsdt[inode]
        out += np.kron(interp[:, inode], block)
    return out


def _apply_robust_close_corrections(
    mat: np.ndarray,
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    nodes: np.ndarray,
    weights: np.ndarray,
    bary: np.ndarray,
    options: dict[str, Any],
    ignored: set[int],
) -> None:
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    points = chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F")
    deriv = chnkr.d.reshape(chnkr.dim, chnkr.npt, order="F")
    deriv2 = chnkr.d2.reshape(chnkr.dim, chnkr.npt, order="F")
    normals = chnkr.n.reshape(chnkr.dim, chnkr.npt, order="F")
    data = chnkr.data.reshape(chnkr.datadim, chnkr.npt, order="F") if chnkr.datadim else None
    chunk_lengths = chnkr.chunklen()
    for src_chunk in range(chnkr.nch):
        if src_chunk in ignored:
            continue
        src_cols = _block_slice(src_chunk, chnkr.k, op1)
        src_points = chnkr.r[:, :, src_chunk]
        dist = np.sqrt(np.min(np.sum((points[:, :, None] - src_points[:, None, :]) ** 2, axis=0), axis=1))
        close = np.flatnonzero(dist < chunk_lengths[src_chunk])
        if close.size == 0:
            continue
        left, right = chnkr.adj[:, src_chunk]
        ignore_chunks = {src_chunk}
        if left > 0:
            ignore_chunks.add(int(left) - 1)
        if right > 0:
            ignore_chunks.add(int(right) - 1)
        ignore_points = set()
        for chunk in ignore_chunks:
            ignore_points.update(range(chunk * chnkr.k, (chunk + 1) * chnkr.k))
        fix = np.array([idx for idx in close if idx not in ignore_points], dtype=int)
        if fix.size == 0:
            continue
        targinfo = PointInfo(
            r=points[:, fix],
            d=deriv[:, fix],
            d2=deriv2[:, fix],
            n=normals[:, fix],
            data=None if data is None else data[:, fix],
        )
        submat = adapgausswts(chnkr, src_chunk, targinfo, kern, opdims, nodes, weights, bary, options)[0]
        for local, global_idx in enumerate(fix):
            rows = slice(op0 * global_idx, op0 * (global_idx + 1))
            mat[rows, src_cols] = submat[op0 * local : op0 * (local + 1), :]


def _barycentric_matrix(ct: np.ndarray, bw: np.ndarray, tt: np.ndarray) -> np.ndarray:
    diff = ct[:, None] - tt[None, :]
    out = np.empty((ct.size, tt.size), dtype=float)
    exact = np.isclose(diff, 0.0, atol=0.0, rtol=1e-14)
    for col in range(tt.size):
        hits = np.flatnonzero(exact[:, col])
        if hits.size:
            out[:, col] = 0.0
            out[hits[0], col] = 1.0
            continue
        vals = bw / diff[:, col]
        out[:, col] = vals / np.sum(vals)
    return out


def _chunk_source_arrays(chnkr: Chunker, chunk: int) -> dict[str, np.ndarray | None]:
    return {
        "r": chnkr.r[:, :, chunk],
        "d": chnkr.d[:, :, chunk],
        "d2": chnkr.d2[:, :, chunk],
        "n": chnkr.n[:, :, chunk],
        "data": chnkr.data[:, :, chunk] if chnkr.datadim else None,
    }


def _chunk_pointinfo(chnkr: Chunker, chunk: int) -> PointInfo:
    return PointInfo(
        r=chnkr.r[:, :, chunk],
        d=chnkr.d[:, :, chunk],
        d2=chnkr.d2[:, :, chunk],
        n=chnkr.n[:, :, chunk],
        data=chnkr.data[:, :, chunk] if chnkr.datadim else None,
    )


def _single_target(targinfo: PointInfo, idx: int) -> PointInfo:
    return PointInfo(
        r=targinfo.r[:, idx : idx + 1],
        d=None if targinfo.d is None else targinfo.d[:, idx : idx + 1],
        d2=None if targinfo.d2 is None else targinfo.d2[:, idx : idx + 1],
        n=None if targinfo.n is None else targinfo.n[:, idx : idx + 1],
        data=None if targinfo.data is None else targinfo.data[:, idx : idx + 1],
    )


def _block_slice(chunk: int, k: int, opdim: int) -> slice:
    start = chunk * k * opdim
    return slice(start, start + k * opdim)
