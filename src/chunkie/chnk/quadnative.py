"""Native smooth quadrature builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie.chunker import Chunker
from chunkie.operators import PointInfo


def buildmat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    i: ArrayLike | None = None,
    j: ArrayLike | None = None,
    wts: ArrayLike | None = None,
) -> np.ndarray:
    """Build a smooth quadrature submatrix for target chunks ``i`` and source chunks ``j``."""

    ich = np.arange(chnkr.nch) if i is None else np.asarray(i, dtype=int).reshape(-1)
    jch = np.arange(chnkr.nch) if j is None else np.asarray(j, dtype=int).reshape(-1)
    if np.any(ich < 0) or np.any(ich >= chnkr.nch) or np.any(jch < 0) or np.any(jch >= chnkr.nch):
        raise IndexError("chunk index out of range")

    srcinfo = _pointinfo_for_chunks(chnkr, jch)
    targinfo = _pointinfo_for_chunks(chnkr, ich)
    if hasattr(kern, "eval") and getattr(kern, "eval") is not None:
        mat = kern.eval(srcinfo, targinfo)
        if opdims is None and hasattr(kern, "opdims"):
            opdims = kern.opdims
    else:
        mat = kern(srcinfo, targinfo)

    if opdims is None:
        nsrc = chnkr.k * jch.size
        opdims = (mat.shape[0] // (chnkr.k * ich.size), mat.shape[1] // nsrc)

    base_w = chnkr.wstor if wts is None else np.asarray(wts, dtype=float).reshape(chnkr.k)
    speed = np.sqrt(np.sum(np.abs(chnkr.d[:, :, jch]) ** 2, axis=0))
    smooth_wts = (speed * base_w[:, None]).reshape(-1)
    return mat * np.repeat(smooth_wts, int(opdims[1]))[None, :]


def _pointinfo_for_chunks(chnkr: Chunker, chunks: np.ndarray) -> PointInfo:
    return PointInfo(
        r=chnkr.r[:, :, chunks].reshape(chnkr.dim, chnkr.k * chunks.size),
        d=chnkr.d[:, :, chunks].reshape(chnkr.dim, chnkr.k * chunks.size),
        d2=chnkr.d2[:, :, chunks].reshape(chnkr.dim, chnkr.k * chunks.size),
        n=chnkr.n[:, :, chunks].reshape(chnkr.dim, chnkr.k * chunks.size),
        data=chnkr.data[:, :, chunks].reshape(chnkr.datadim, chnkr.k * chunks.size)
        if chnkr.datadim
        else None,
    )
