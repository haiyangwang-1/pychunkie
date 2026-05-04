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
    merged = _as_chunker(obj)
    if merged is not None:
        obj = merged
    if isinstance(obj, Chunker):
        return PointInfo(
            r=obj.r.reshape(obj.dim, obj.npt, order="F"),
            d=obj.d.reshape(obj.dim, obj.npt, order="F"),
            d2=obj.d2.reshape(obj.dim, obj.npt, order="F"),
            n=obj.n.reshape(obj.dim, obj.npt, order="F"),
            data=obj.data.reshape(obj.datadim, obj.npt, order="F") if obj.datadim else None,
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


def chunkermat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build a dense native quadrature matrix for a chunker."""

    chnkr = _require_chunker(chnkr)
    options = {} if opts is None else dict(opts)
    if _uses_special_quadrature(kern, options):
        from .chnk import quadggq

        return quadggq.buildmat(chnkr, kern, getattr(kern, "opdims", None), getattr(kern, "sing", "log"))

    srcinfo = pointinfo(chnkr)
    mat = _eval_kernel(kern, srcinfo, srcinfo)
    wts = chnkr.wts.reshape(-1, order="F")
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
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply the dense native matrix for ``kern`` on ``chnkr``."""

    return chunkermat(chnkr, kern, opts) @ np.asarray(dens).reshape(-1, order="F")


def chunkerintegral(
    chnkr: Chunker,
    f: Callable[[np.ndarray], ArrayLike] | ArrayLike,
    opts: dict[str, Any] | None = None,
) -> float:
    """Integrate scalar values over a chunker with the native smooth rule."""

    chnkr = _require_chunker(chnkr)
    _ = {} if opts is None else dict(opts)
    if callable(f):
        vals = np.asarray(f(chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F")))
    else:
        vals = np.asarray(f)
    if vals.size != chnkr.npt:
        raise ValueError("f must evaluate to one scalar value per chunker point")
    return float(np.dot(chnkr.wts.reshape(-1, order="F"), vals.reshape(-1, order="F")))


def chunkerinterior(
    chnkr: Chunker,
    ptsobj: Chunker | dict[str, Any] | ArrayLike | tuple[ArrayLike, ArrayLike] | list[ArrayLike],
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Classify target points as inside a closed 2D chunker.

    This is a dependency-light direct fallback based on the node polygon.
    FMM/FLAM acceleration and close-boundary correction are deferred.
    """

    chnkr = _require_chunker(chnkr)
    _ = {} if opts is None else dict(opts)
    if chnkr.dim != 2:
        raise ValueError("interior only well-defined for 2D chunkers")

    grid_shape = None
    if isinstance(ptsobj, (tuple, list)) and len(ptsobj) == 2:
        x = np.asarray(ptsobj[0], dtype=float)
        y = np.asarray(ptsobj[1], dtype=float)
        xx, yy = np.meshgrid(x, y)
        pts = np.vstack((xx.ravel(), yy.ravel()))
        grid_shape = xx.shape
    elif _as_chunker(ptsobj) is not None:
        ptschnkr = _require_chunker(ptsobj)
        pts = ptschnkr.r.reshape(ptschnkr.dim, ptschnkr.npt, order="F")
    elif isinstance(ptsobj, dict) and "r" in ptsobj:
        arr = np.asarray(ptsobj["r"], dtype=float)
        pts = arr.reshape(arr.shape[0], -1)
    else:
        arr = np.asarray(ptsobj, dtype=float)
        pts = arr.reshape(arr.shape[0], -1)

    if pts.shape[0] != 2:
        raise ValueError("target points must be two-dimensional")

    boundary = _chunker_polygon_points(chnkr)
    x = pts[0]
    y = pts[1]
    inside = np.zeros(pts.shape[1], dtype=bool)
    x0 = boundary[:, 0]
    y0 = boundary[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        crosses = (ya > y) != (yb > y)
        xhit = (xb - xa) * (y - ya) / (yb - ya + np.finfo(float).eps) + xa
        inside ^= crosses & (x < xhit)

    if grid_shape is not None:
        return inside.reshape(grid_shape)
    return inside


def chunkerkerneval(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dens: ArrayLike,
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate a dense direct layer potential at targets."""

    same_source_target = targobj is chnkr
    chnkr = _require_chunker(chnkr)
    if same_source_target and _uses_special_quadrature(kern, opts):
        vals = chunkermat(chnkr, kern, opts) @ np.asarray(dens).reshape(-1, order="F")
        opdims = getattr(kern, "opdims", (1, 1))[0]
        return vals.reshape(opdims, chnkr.npt, order="F")

    srcinfo = pointinfo(chnkr)
    targinfo = pointinfo(targobj)
    mat = _eval_kernel(kern, srcinfo, targinfo)
    dens_arr = np.asarray(dens)
    if dens_arr.size == chnkr.npt:
        weighted = dens_arr.reshape(-1, order="F") * chnkr.wts.reshape(-1, order="F")
    else:
        weighted = dens_arr.reshape(-1, order="F")
        if weighted.size % chnkr.npt != 0:
            raise ValueError("density has incompatible size")
        opdims_col = weighted.size // chnkr.npt
        weighted = weighted * np.repeat(chnkr.wts.reshape(-1, order="F"), opdims_col)
    vals = mat @ weighted
    return vals.reshape(-1, targinfo.r.shape[1], order="F")


def chunkerkernevalmat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build the dense native matrix mapping chunker densities to target values."""

    same_source_target = targobj is chnkr
    chnkr = _require_chunker(chnkr)
    if same_source_target and _uses_special_quadrature(kern, opts):
        return chunkermat(chnkr, kern, opts)

    _ = {} if opts is None else dict(opts)
    srcinfo = pointinfo(chnkr)
    targinfo = pointinfo(targobj)
    mat = _eval_kernel(kern, srcinfo, targinfo)
    wts = chnkr.wts.reshape(-1, order="F")
    if mat.shape[1] == chnkr.npt:
        return mat * wts[None, :]
    if mat.shape[1] % chnkr.npt != 0:
        raise ValueError("kernel column dimension is incompatible with chunker points")
    opdims_col = mat.shape[1] // chnkr.npt
    return mat * np.repeat(wts, opdims_col)[None, :]


def _optional_field(obj: dict[str, Any], name: str) -> np.ndarray | None:
    if name not in obj or obj[name] is None:
        return None
    arr = np.asarray(obj[name])
    return arr.reshape(arr.shape[0], -1)


def _as_chunker(obj: Any) -> Chunker | None:
    if isinstance(obj, Chunker):
        return obj
    merged = getattr(obj, "merged", None)
    if callable(merged):
        out = merged()
        if isinstance(out, Chunker):
            return out
    return None


def _require_chunker(obj: Any) -> Chunker:
    out = _as_chunker(obj)
    if out is None:
        raise TypeError("expected a chunker or chunkgraph-like object")
    return out


def _eval_kernel(kern: Callable[[Any, Any], np.ndarray], srcinfo: PointInfo, targinfo: PointInfo) -> np.ndarray:
    if hasattr(kern, "eval") and getattr(kern, "eval") is not None:
        return kern.eval(srcinfo, targinfo)
    return kern(srcinfo, targinfo)


def _uses_special_quadrature(kern: Callable[[Any, Any], np.ndarray], opts: dict[str, Any] | None) -> bool:
    options = {} if opts is None else dict(opts)
    if bool(options.get("forcesmooth", False)) or bool(options.get("usesmooth", False)):
        return False
    if bool(options.get("forceadap", False)):
        return True
    return getattr(kern, "sing", "") in {"log"}


def _chunker_polygon_points(chnkr: Chunker) -> np.ndarray:
    sorted_chnkr, _ = chnkr.sort()
    pieces: list[np.ndarray] = []
    for ich in range(sorted_chnkr.nch):
        rend, _ = sorted_chnkr.chunkends([ich])
        panel = np.column_stack((rend[:, 0, 0], sorted_chnkr.r[:, :, ich], rend[:, 1, 0]))
        if pieces:
            panel = panel[:, 1:]
        pieces.append(panel.T)
    if not pieces:
        raise ValueError("chunker has no boundary points")
    points = np.vstack(pieces)
    if np.linalg.norm(points[0] - points[-1]) > 1e-12:
        points = np.vstack((points, points[0]))
    return points
