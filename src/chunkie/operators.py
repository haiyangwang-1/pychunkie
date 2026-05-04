"""Dense direct operator assembly and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .chunker import Chunker


@dataclass
class PointInfo:
    r: np.ndarray
    d: np.ndarray | None = None
    d2: np.ndarray | None = None
    n: np.ndarray | None = None
    data: np.ndarray | None = None


def pointinfo(obj: Chunker | dict[str, Any] | ArrayLike | PointInfo) -> PointInfo:
    """Convert supported inputs to MATLAB-style point-info fields."""

    if isinstance(obj, PointInfo):
        return obj
    if isinstance(obj, Chunker):
        return PointInfo(
            r=obj.r.reshape(obj.dim, obj.npt),
            d=obj.d.reshape(obj.dim, obj.npt),
            d2=obj.d2.reshape(obj.dim, obj.npt),
            n=obj.n.reshape(obj.dim, obj.npt),
            data=obj.data.reshape(obj.datadim, obj.npt) if obj.datadim else None,
        )
    if isinstance(obj, dict):
        return PointInfo(
            r=np.asarray(obj["r"], dtype=float).reshape(np.asarray(obj["r"]).shape[0], -1),
            d=_optional_field(obj, "d"),
            d2=_optional_field(obj, "d2"),
            n=_optional_field(obj, "n"),
            data=_optional_field(obj, "data"),
        )
    arr = np.asarray(obj, dtype=float)
    return PointInfo(r=arr.reshape(arr.shape[0], -1))


def chunkermat(chnkr: Chunker, kern: Callable[[Any, Any], np.ndarray]) -> np.ndarray:
    """Build a dense native quadrature matrix for a chunker."""

    srcinfo = pointinfo(chnkr)
    mat = _eval_kernel(kern, srcinfo, srcinfo)
    wts = chnkr.wts.reshape(-1)
    if mat.shape[1] == chnkr.npt:
        return mat * wts[None, :]
    if mat.shape[1] % chnkr.npt != 0:
        raise ValueError("kernel column dimension is incompatible with chunker points")
    opdims_col = mat.shape[1] // chnkr.npt
    return mat * np.repeat(wts, opdims_col)[None, :]


def chunkermatapply(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dens: ArrayLike,
) -> np.ndarray:
    """Apply the dense native matrix for ``kern`` on ``chnkr``."""

    return chunkermat(chnkr, kern) @ np.asarray(dens).reshape(-1)


def chunkerkerneval(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dens: ArrayLike,
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
) -> np.ndarray:
    """Evaluate a dense direct layer potential at targets."""

    srcinfo = pointinfo(chnkr)
    targinfo = pointinfo(targobj)
    mat = _eval_kernel(kern, srcinfo, targinfo)
    dens_arr = np.asarray(dens)
    if dens_arr.size == chnkr.npt:
        weighted = dens_arr.reshape(-1) * chnkr.wts.reshape(-1)
    else:
        weighted = dens_arr.reshape(-1)
        if weighted.size % chnkr.npt != 0:
            raise ValueError("density has incompatible size")
        opdims_col = weighted.size // chnkr.npt
        weighted = weighted * np.repeat(chnkr.wts.reshape(-1), opdims_col)
    vals = mat @ weighted
    return vals.reshape(-1, targinfo.r.shape[1])


def _optional_field(obj: dict[str, Any], name: str) -> np.ndarray | None:
    if name not in obj or obj[name] is None:
        return None
    arr = np.asarray(obj[name])
    return arr.reshape(arr.shape[0], -1)


def _eval_kernel(kern: Callable[[Any, Any], np.ndarray], srcinfo: PointInfo, targinfo: PointInfo) -> np.ndarray:
    if hasattr(kern, "eval") and getattr(kern, "eval") is not None:
        return kern.eval(srcinfo, targinfo)
    return kern(srcinfo, targinfo)
