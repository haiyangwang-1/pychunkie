"""Operator assembly, matrix-free application, and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

from . import lege
from .chunker import Chunker


@dataclass
class PointInfo:
    r: np.ndarray
    d: np.ndarray | None = None
    d2: np.ndarray | None = None
    n: np.ndarray | None = None
    data: np.ndarray | None = None


class ChunkerFMMMatrix(LinearOperator):
    """Matrix-free ``chunkermat`` operator using kernel FMM application."""

    def __init__(
        self,
        chnkr: Chunker,
        kern: Callable[[Any, Any], np.ndarray],
        opts: dict[str, Any] | None = None,
    ):
        self.chnkr = _require_chunker(chnkr)
        self.kern = kern
        self.opts = {} if opts is None else dict(opts)
        self.opdims = _kernel_opdims(self.chnkr, kern)
        self._correction_mat: spmatrix | None = None
        dtype = _operator_dtype(self.chnkr, kern)
        shape = (self.chnkr.npt * int(self.opdims[0]), self.chnkr.npt * int(self.opdims[1]))
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.shape[1]:
            raise ValueError("density has incompatible size")
        return _chunkermatapply_fmm(self.chnkr, self.kern, x, self.opts, self._correction())

    def _matmat(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.shape[1]:
            raise ValueError("density matrix has incompatible row count")
        if x.shape[1] == 0:
            return np.empty((self.shape[0], 0), dtype=np.result_type(self.dtype, x.dtype))
        return np.column_stack([self._matvec(x[:, col]) for col in range(x.shape[1])])

    def toarray(self) -> np.ndarray:
        """Materialize the operator by applying it to basis vectors."""

        return self._matmat(np.eye(self.shape[1], dtype=self.dtype))

    def todense(self) -> np.ndarray:
        return self.toarray()

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        arr = self.toarray()
        if dtype is not None:
            return np.array(arr, dtype=dtype, copy=True if copy is None else copy)
        if copy:
            return arr.copy()
        return arr

    def _correction(self) -> spmatrix | None:
        if not _uses_special_quadrature(self.kern, self.opts):
            return None
        if self._correction_mat is None:
            self._correction_mat = _special_correction_matrix(self.chnkr, self.kern, self.opts)
        return self._correction_mat


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
) -> np.ndarray | ChunkerFMMMatrix:
    """Build a native quadrature matrix, or an FMM-backed operator when requested."""

    chnkr = _require_chunker(chnkr)
    options = {} if opts is None else dict(opts)
    acceleration = _acceleration(options)
    if acceleration == "flam":
        _raise_flam_not_implemented()
    if acceleration == "fmm":
        _require_fmm(kern)
        return ChunkerFMMMatrix(chnkr, kern, options)
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
    """Apply the native matrix for ``kern`` on ``chnkr``."""

    chnkr = _require_chunker(chnkr)
    options = {} if opts is None else dict(opts)
    dens_vec = np.asarray(dens).reshape(-1, order="F")
    acceleration = _acceleration(options)
    if acceleration == "flam":
        _raise_flam_not_implemented()
    if acceleration == "fmm":
        _require_fmm(kern)
        return _chunkermatapply_fmm(chnkr, kern, dens_vec, options)
    return chunkermat(chnkr, kern, opts) @ dens_vec


def chunkerintegral(
    chnkr: Chunker,
    f: Callable[[np.ndarray], ArrayLike] | ArrayLike,
    opts: dict[str, Any] | None = None,
) -> float:
    """Integrate scalar values over a chunker with the native smooth rule."""

    chnkr = _require_chunker(chnkr)
    options = {} if opts is None else dict(opts)
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

    The default ``acceleration="dense"`` path is a dependency-light direct
    polygon test. With ``acceleration="fmm"``, the Laplace double-layer
    identity is used for accelerated classification, and near-boundary targets
    are corrected by the direct path. FLAM acceleration is deferred.
    """

    chnkr = _require_chunker(chnkr)
    options = {} if opts is None else dict(opts)
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

    acceleration = _acceleration(options)
    if acceleration == "flam":
        _raise_flam_not_implemented()
    if acceleration == "fmm":
        from .kernel import kernel

        lap_d = kernel("lap", "d")
        dens = np.ones(chnkr.npt)
        vals = chunkerkerneval(
            chnkr,
            lap_d,
            dens,
            PointInfo(r=pts),
            {"acceleration": "fmm", "eps": float(options.get("eps", options.get("tol", 1e-12)))},
        ).reshape(-1, order="F")
        inside = vals < -0.5
        if bool(options.get("closecorr", options.get("corrections", True))):
            near_fac = float(options.get("near_fac", options.get("fac", 1.0)))
            near = np.any(chnkr.flagnear(pts, {"fac": near_fac}), axis=1)
            if np.any(near):
                corrected = _chunkerinterior_direct(chnkr, pts[:, near], bool(options.get("axissym", False)))
                inside[near] = corrected
    else:
        inside = _chunkerinterior_direct(chnkr, pts, bool(options.get("axissym", False)))

    if grid_shape is not None:
        return inside.reshape(grid_shape)
    return inside


def _chunkerinterior_direct(chnkr: Chunker, pts: np.ndarray, axissym: bool = False) -> np.ndarray:
    inside = np.zeros(pts.shape[1], dtype=bool)
    for boundary in _chunker_component_polygons(chnkr, axissym):
        inside ^= _points_in_polygon(pts, boundary)
    return inside


def chunkerkerneval(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dens: ArrayLike,
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate a dense direct layer potential at targets."""

    options = {} if opts is None else dict(opts)
    same_source_target = targobj is chnkr
    chnkr = _require_chunker(chnkr)
    acceleration = _acceleration(options)
    if acceleration == "flam":
        _raise_flam_not_implemented()
    use_fmm = acceleration == "fmm"
    if use_fmm:
        _require_fmm(kern)
    if same_source_target and _uses_special_quadrature(kern, options) and not use_fmm:
        vals = chunkermat(chnkr, kern, opts) @ np.asarray(dens).reshape(-1, order="F")
        opdims = getattr(kern, "opdims", (1, 1))[0]
        return vals.reshape(opdims, chnkr.npt, order="F")

    srcinfo = pointinfo(chnkr)
    targinfo = pointinfo(targobj)
    if use_fmm:
        weighted = _weighted_density(chnkr, dens)
        vals = kern.fmm(float(options.get("eps", options.get("tol", 1e-12))), srcinfo, targinfo, weighted)
        if isinstance(vals, tuple):
            vals = vals[0]
        return np.asarray(vals).reshape(-1, targinfo.r.shape[1], order="F")

    mat = _eval_kernel(kern, srcinfo, targinfo)
    weighted = _weighted_density(chnkr, dens)
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
    options = {} if opts is None else dict(opts)
    acceleration = _acceleration(options)
    if acceleration == "flam":
        _raise_flam_not_implemented()
    if acceleration == "fmm":
        raise NotImplementedError("chunkerkernevalmat does not support FMM acceleration; use chunkerkerneval instead")
    if same_source_target and _uses_special_quadrature(kern, opts):
        return chunkermat(chnkr, kern, opts)

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


def _weighted_density(chnkr: Chunker, dens: ArrayLike) -> np.ndarray:
    dens_arr = np.asarray(dens)
    if dens_arr.size == chnkr.npt:
        return dens_arr.reshape(-1, order="F") * chnkr.wts.reshape(-1, order="F")
    weighted = dens_arr.reshape(-1, order="F")
    if weighted.size % chnkr.npt != 0:
        raise ValueError("density has incompatible size")
    opdims_col = weighted.size // chnkr.npt
    return weighted * np.repeat(chnkr.wts.reshape(-1, order="F"), opdims_col)


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
    return getattr(kern, "sing", "") in {"log", "pv", "hs"}


def _acceleration(options: dict[str, Any]) -> str:
    value = options.get("acceleration", "dense")
    if value is None:
        return "dense"
    acceleration = str(value).lower()
    if acceleration not in {"dense", "fmm", "flam"}:
        raise ValueError("acceleration must be one of 'dense', 'fmm', or 'flam'")
    return acceleration


def _require_fmm(kern: Callable[[Any, Any], np.ndarray]) -> None:
    if getattr(kern, "fmm", None) is None:
        raise NotImplementedError("FMM acceleration requested, but the kernel has no FMM evaluator")


def _raise_flam_not_implemented() -> None:
    raise NotImplementedError("FLAM acceleration is not implemented yet")


def _chunkermatapply_fmm(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dens: ArrayLike,
    options: dict[str, Any],
    correction: spmatrix | None = None,
) -> np.ndarray:
    dens_vec = np.asarray(dens).reshape(-1, order="F")
    fmm_options = dict(options)
    fmm_options["acceleration"] = "fmm"
    vals = chunkerkerneval(chnkr, kern, dens_vec, chnkr, fmm_options).reshape(-1, order="F")
    if _uses_special_quadrature(kern, options):
        corr = _special_correction_matrix(chnkr, kern, options) if correction is None else correction
        vals = vals + corr @ dens_vec
    return vals


def _special_correction_matrix(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> spmatrix:
    from .chnk import quadggq

    qtype = str(options.get("sing", getattr(kern, "sing", "log") or "log")).lower()
    return quadggq.buildmattd(
        chnkr,
        kern,
        getattr(kern, "opdims", None),
        type=qtype,
        ilist=options.get("ilist", None),
        corrections=True,
    )


def _kernel_opdims(chnkr: Chunker, kern: Callable[[Any, Any], np.ndarray]) -> tuple[int, int]:
    opdims = getattr(kern, "opdims", None)
    if opdims is not None and tuple(opdims) != (0, 0):
        return int(opdims[0]), int(opdims[1])
    src = _pointinfo_node(chnkr, 0)
    targ = _pointinfo_node(chnkr, 1 if chnkr.npt > 1 else 0)
    mat = _eval_kernel(kern, src, targ)
    return int(mat.shape[0]), int(mat.shape[1])


def _operator_dtype(chnkr: Chunker, kern: Callable[[Any, Any], np.ndarray]) -> np.dtype:
    try:
        src = _pointinfo_node(chnkr, 0)
        targ = _pointinfo_node(chnkr, 1 if chnkr.npt > 1 else 0)
        return np.asarray(_eval_kernel(kern, src, targ)).dtype
    except Exception:
        return np.dtype(float)


def _pointinfo_node(chnkr: Chunker, inode: int) -> PointInfo:
    src = pointinfo(chnkr)
    idx = int(inode)
    return PointInfo(
        r=src.r[:, idx : idx + 1],
        d=src.d[:, idx : idx + 1] if src.d is not None else None,
        d2=src.d2[:, idx : idx + 1] if src.d2 is not None else None,
        n=src.n[:, idx : idx + 1] if src.n is not None else None,
        data=src.data[:, idx : idx + 1] if src.data is not None else None,
    )


def _chunker_polygon_points(chnkr: Chunker) -> np.ndarray:
    return _chunker_component_polygons(chnkr)[0]


def _chunker_component_polygons(chnkr: Chunker, axissym: bool = False) -> list[np.ndarray]:
    sorted_chnkr, info = chnkr.sort()
    polygons: list[np.ndarray] = []
    start = 0
    for nch, closed in zip(np.asarray(info["nchs"], dtype=int), np.asarray(info["ifclosed"], dtype=bool)):
        polygons.append(_chunker_component_polygon(sorted_chnkr, start, int(nch), bool(closed), axissym))
        start += int(nch)
    if not polygons:
        raise ValueError("chunker has no boundary points")
    return polygons


def _chunker_component_polygon(chnkr: Chunker, start: int, nch: int, closed: bool, axissym: bool) -> np.ndarray:
    pieces: list[np.ndarray] = []
    ts = np.linspace(-1.0, 1.0, max(4 * chnkr.k, 64))
    interp = lege.matrin(chnkr.k, ts)[0]
    for ich in range(start, start + nch):
        panel = (interp @ chnkr.r[:, :, ich].T).T
        if pieces:
            panel = panel[:, 1:]
        pieces.append(panel.T)
    points = np.vstack(pieces)
    if axissym and not closed:
        axis_end = np.array([[0.0, points[-1, 1]], [0.0, points[0, 1]]])
        points = np.vstack((points, axis_end))
    if np.linalg.norm(points[0] - points[-1]) > 1e-12:
        points = np.vstack((points, points[0]))
    return points


def _points_in_polygon(pts: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    x = pts[0]
    y = pts[1]
    inside = np.zeros(pts.shape[1], dtype=bool)
    x0 = boundary[:, 0]
    y0 = boundary[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        crosses = (ya > y) != (yb > y)
        hits = np.zeros_like(crosses)
        if np.any(crosses):
            xhit = (xb - xa) * (y[crosses] - ya) / (yb - ya) + xa
            hits[crosses] = x[crosses] < xhit
        inside ^= hits
    return inside
