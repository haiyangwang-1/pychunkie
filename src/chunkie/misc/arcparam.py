"""Arc-length parameterization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from chunkie import lege
from chunkie.geometry.chunker import Chunker


@dataclass
class ArcParamData:
    plen: np.ndarray
    pstrt: np.ndarray
    cr: np.ndarray
    cd: np.ndarray
    cd2: np.ndarray
    k: int
    dim: int
    nch: int
    eps: float
    maxcond: float


def init(chunker: Chunker, chunks: ArrayLike | None = None) -> ArcParamData:
    """Initialize an arc-length parameterization for selected chunks."""

    chunk_ids = (
        np.arange(chunker.nch) if chunks is None else np.asarray(chunks, dtype=int).reshape(-1)
    )
    if np.any(chunk_ids < 0) or np.any(chunk_ids >= chunker.nch):
        raise IndexError("chunk index out of range")

    rs = chunker.r[:, :, chunk_ids]
    ds = chunker.d[:, :, chunk_ids]
    d2s = chunker.d2[:, :, chunk_ids]
    plen = np.sum(chunker.wts[:, chunk_ids], axis=0)
    pstrt = np.concatenate(([0.0], np.cumsum(plen)))

    amat = lege.intmat(chunker.k)[0]
    darc = np.sqrt(np.sum(ds**2, axis=0))
    snodes = 2.0 * (amat @ darc / plen[None, :]) - 1.0

    darc3 = darc[None, :, :]
    ddarc = np.sum(ds * d2s, axis=0) / darc
    d_arc = ds / darc3
    d2_arc = d2s / darc3**2 - ds * ddarc[None, :, :] / darc3**3

    nchs = chunk_ids.size
    cr = np.zeros((chunker.k, chunker.dim, nchs), dtype=rs.dtype)
    cd = np.zeros_like(cr)
    cd2 = np.zeros_like(cr)
    maxcond = 0.0
    for out_idx in range(nchs):
        vals = lege.pols(snodes[:, out_idx], chunker.k - 1)[0].T
        cr[:, :, out_idx] = np.linalg.solve(vals, rs[:, :, out_idx].T)
        cd[:, :, out_idx] = np.linalg.solve(vals, d_arc[:, :, out_idx].T)
        cd2[:, :, out_idx] = np.linalg.solve(vals, d2_arc[:, :, out_idx].T)
        maxcond = max(maxcond, float(np.linalg.cond(vals)))

    last = cr[-2:, :, :] if chunker.k >= 2 else cr[-1:, :, :]
    eps = float(np.max(last)) if last.size else 0.0
    return ArcParamData(plen, pstrt, cr, cd, cd2, chunker.k, chunker.dim, nchs, eps, maxcond)


def eval(s: ArrayLike, param_data: ArcParamData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate an arc-length parameterization at arclength values ``s``."""

    s_arr = np.asarray(s, dtype=float)
    flat = s_arr.reshape(-1)
    r = np.zeros((param_data.dim, flat.size), dtype=param_data.cr.dtype)
    d = np.zeros_like(r)
    d2 = np.zeros_like(r)

    for ich in range(param_data.nch):
        if ich == param_data.nch - 1:
            mask = (flat >= param_data.pstrt[ich]) & (flat <= param_data.pstrt[ich + 1])
        else:
            mask = (flat >= param_data.pstrt[ich]) & (flat < param_data.pstrt[ich + 1])
        if not np.any(mask):
            continue
        sloc = 2.0 * (flat[mask] - param_data.pstrt[ich]) / param_data.plen[ich] - 1.0
        legs = lege.pols(sloc, param_data.k - 1)[0].T
        r[:, mask] = (legs @ param_data.cr[:, :, ich]).T
        d[:, mask] = (legs @ param_data.cd[:, :, ich]).T
        d2[:, mask] = (legs @ param_data.cd2[:, :, ich]).T

    out_shape = s_arr.shape
    return (
        r.reshape((param_data.dim,) + out_shape),
        d.reshape((param_data.dim,) + out_shape),
        d2.reshape((param_data.dim,) + out_shape),
    )
