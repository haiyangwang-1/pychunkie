"""Special quadrature builders for singular and near-panel interactions.

This module mirrors the MATLAB ``chnk.quadggq`` entry points. The current
Python rules are generated dynamically from split and oversampled
Gauss-Legendre rules, avoiding tabulated data while still replacing the
native diagonal and neighbor blocks where singular kernels need help.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie import lege
from chunkie.chunker import Chunker
from chunkie.operators import PointInfo
from chunkie.chnk import quadnative


@dataclass
class AuxQuad:
    xs1: np.ndarray
    wts1: np.ndarray
    xs0: list[np.ndarray]
    wts0: list[np.ndarray]
    ainterp1: np.ndarray
    ainterps0: list[np.ndarray]
    type: str = "log"


def setup(k: int, type: str = "log", nfac_self: int | None = None, nfac_near: int | None = None) -> AuxQuad:
    """Generate auxiliary quadrature rules for self and neighbor panels."""

    qtype = type.lower()
    if qtype not in {"log", "removable"}:
        raise NotImplementedError("generated quadggq rules currently support log/removable singularities")
    if nfac_self is None:
        nfac_self = max(4, int(np.ceil(48 / max(k, 1))))
    if nfac_near is None:
        nfac_near = max(4, int(np.ceil(48 / max(k, 1))))

    xs1, wts1 = lege.exps(int(nfac_near * k))[:2]
    xs0, wts0 = getremovablequad(k, nfac_self)
    return AuxQuad(
        xs1=xs1,
        wts1=wts1,
        xs0=xs0,
        wts0=wts0,
        ainterp1=lege.matrin(k, xs1)[0],
        ainterps0=[lege.matrin(k, xs)[0] for xs in xs0],
        type=qtype,
    )


def getlogquad(k: int, npolyfac: int = 2) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return generated neighbor and self rules for logarithmic kernels."""

    aux = setup(k, "log", nfac_self=npolyfac, nfac_near=npolyfac)
    return aux.xs1, aux.wts1, aux.xs0, aux.wts0


def logavail() -> np.ndarray:
    """Return panel orders supported by the generated log rules."""

    return np.arange(1, 65, dtype=int)


def getremovablequad(k: int, nfac: int = 1) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return split Gauss rules on each side of every Legendre node."""

    xleg, _ = lege.exps(k)[:2]
    xover, wover = lege.exps(max(int(np.ceil(k * nfac)), k))[:2]
    x01 = (xover + 1.0) / 2.0
    w01 = wover / 2.0
    xs0: list[np.ndarray] = []
    wts0: list[np.ndarray] = []
    for node in xleg:
        xleft = x01 * (node + 1.0) - 1.0
        wleft = w01 * (node + 1.0)
        xright = x01 * (1.0 - node) + node
        wright = w01 * (1.0 - node)
        xs0.append(np.concatenate((xleft, xright)))
        wts0.append(np.concatenate((wleft, wright)))
    return xs0, wts0


def buildmat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    type: str = "log",
    auxquads: AuxQuad | None = None,
    ilist: ArrayLike | None = None,
) -> np.ndarray:
    """Build a matrix with special self and neighbor quadrature blocks."""

    if opdims is None:
        opdims = getattr(kern, "opdims", None)
    if opdims is None or opdims == (0, 0):
        raise ValueError("opdims must be provided for special quadrature assembly")

    aux = setup(chnkr.k, type) if auxquads is None else auxquads
    ignored = set() if ilist is None else {int(idx) for idx in np.asarray(ilist, dtype=int).reshape(-1)}
    mat = quadnative.buildmat(chnkr, kern, opdims)

    for src_chunk in range(chnkr.nch):
        src_cols = _block_slice(src_chunk, chnkr.k, int(opdims[1]))
        left, right = chnkr.adj[:, src_chunk]

        for targ_chunk in (int(left) - 1, int(right) - 1):
            if targ_chunk < 0 or targ_chunk >= chnkr.nch:
                continue
            if src_chunk in ignored and targ_chunk in ignored:
                continue
            rows = _block_slice(targ_chunk, chnkr.k, int(opdims[0]))
            mat[rows, src_cols] = nearbuildmat(chnkr, targ_chunk, src_chunk, kern, opdims, aux)

        if src_chunk in ignored:
            continue
        rows = _block_slice(src_chunk, chnkr.k, int(opdims[0]))
        mat[rows, src_cols] = diagbuildmat(chnkr, src_chunk, kern, opdims, aux)

    return mat


def diagbuildmat(
    chnkr: Chunker,
    i: int,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    aux: AuxQuad | None = None,
) -> np.ndarray:
    """Build a special self-interaction block for chunk ``i``."""

    aux = setup(chnkr.k, "log") if aux is None else aux
    k = chnkr.k
    out = np.zeros((int(opdims[0]) * k, int(opdims[1]) * k), dtype=_kernel_dtype(chnkr, kern))
    rs = chnkr.r[:, :, i]
    ds = chnkr.d[:, :, i]
    d2s = chnkr.d2[:, :, i]
    ns = chnkr.n[:, :, i]
    dd = chnkr.data[:, :, i] if chnkr.datadim else None

    for inode in range(k):
        interp = aux.ainterps0[inode]
        src = _interpolated_pointinfo(rs, ds, d2s, ns, dd, interp)
        targ = PointInfo(
            r=rs[:, inode : inode + 1],
            d=ds[:, inode : inode + 1],
            d2=d2s[:, inode : inode + 1],
            n=ns[:, inode : inode + 1],
            data=dd[:, inode : inode + 1] if dd is not None else None,
        )
        weights = np.sqrt(np.sum(np.abs(src.d) ** 2, axis=0)) * aux.wts0[inode]
        block = _eval_kernel(kern, src, targ) * np.repeat(weights, int(opdims[1]))[None, :]
        rows = slice(int(opdims[0]) * inode, int(opdims[0]) * (inode + 1))
        out[rows, :] = block @ np.kron(interp, np.eye(int(opdims[1])))
    return out


def nearbuildmat(
    chnkr: Chunker,
    i: int,
    j: int,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    aux: AuxQuad | None = None,
) -> np.ndarray:
    """Build an oversampled near-neighbor block from source chunk ``j`` to target chunk ``i``."""

    aux = setup(chnkr.k, "log") if aux is None else aux
    interp = aux.ainterp1
    src = _interpolated_pointinfo(
        chnkr.r[:, :, j],
        chnkr.d[:, :, j],
        chnkr.d2[:, :, j],
        chnkr.n[:, :, j],
        chnkr.data[:, :, j] if chnkr.datadim else None,
        interp,
    )
    targ = PointInfo(
        r=chnkr.r[:, :, i],
        d=chnkr.d[:, :, i],
        d2=chnkr.d2[:, :, i],
        n=chnkr.n[:, :, i],
        data=chnkr.data[:, :, i] if chnkr.datadim else None,
    )
    weights = np.sqrt(np.sum(np.abs(src.d) ** 2, axis=0)) * aux.wts1
    mat = _eval_kernel(kern, src, targ) * np.repeat(weights, int(opdims[1]))[None, :]
    return mat @ np.kron(interp, np.eye(int(opdims[1])))


def _interpolated_pointinfo(
    r: np.ndarray,
    d: np.ndarray,
    d2: np.ndarray,
    n: np.ndarray,
    data: np.ndarray | None,
    interp: np.ndarray,
) -> PointInfo:
    ri = (interp @ r.T).T
    di = (interp @ d.T).T
    d2i = (interp @ d2.T).T
    speed = np.sqrt(np.sum(np.abs(di) ** 2, axis=0))
    ni = np.vstack((di[1], -di[0])) / speed[None, :]
    return PointInfo(
        r=ri,
        d=di,
        d2=d2i,
        n=ni if n is not None else None,
        data=(interp @ data.T).T if data is not None else None,
    )


def _eval_kernel(kern: Callable[[Any, Any], np.ndarray], src: PointInfo, targ: PointInfo) -> np.ndarray:
    if hasattr(kern, "eval") and getattr(kern, "eval") is not None:
        return kern.eval(src, targ)
    return kern(src, targ)


def _block_slice(chunk: int, k: int, opdim: int) -> slice:
    start = chunk * k * opdim
    return slice(start, start + k * opdim)


def _kernel_dtype(chnkr: Chunker, kern: Callable[[Any, Any], np.ndarray]) -> np.dtype:
    try:
        src = PointInfo(r=chnkr.r[:, :1, 0], d=chnkr.d[:, :1, 0], d2=chnkr.d2[:, :1, 0], n=chnkr.n[:, :1, 0])
        return np.asarray(_eval_kernel(kern, src, src)).dtype
    except Exception:
        return np.dtype(float)
