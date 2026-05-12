"""Operator assembly, matrix-free application, and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
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


class ChunkerFLAMMatrix(LinearOperator):
    """Matrix-free ``chunkermat`` operator backed by a PyFLAM factorization."""

    def __init__(
        self,
        chnkr: Chunker,
        kern: Callable[[Any, Any], np.ndarray],
        dval: ArrayLike | float | complex = 0.0,
        opts: dict[str, Any] | None = None,
    ):
        self.chnkr = _require_chunker(chnkr)
        self.kern = kern
        self.opts = {} if opts is None else dict(opts)
        self.factor = chunkerflam(self.chnkr, self.kern, dval, self.opts)
        self.flamtype = str(self.opts.get("flamtype", "rskelf")).lower()
        self.opdims = _kernel_opdims(self.chnkr, kern)
        dtype = _operator_dtype(self.chnkr, kern)
        if np.asarray(dval).size:
            dtype = np.result_type(dtype, np.asarray(dval).dtype)
        shape = (self.chnkr.npt * int(self.opdims[0]), self.chnkr.npt * int(self.opdims[1]))
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.shape[1]:
            raise ValueError("density has incompatible size")
        return self._apply(x)

    def _matmat(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.shape[1]:
            raise ValueError("density matrix has incompatible row count")
        if x.shape[1] == 0:
            return np.empty((self.shape[0], 0), dtype=np.result_type(self.dtype, x.dtype))
        return self._apply(x)

    def solve(self, rhs: ArrayLike) -> np.ndarray:
        """Apply the FLAM approximate inverse to one or more right-hand sides."""

        if self.flamtype != "rskelf":
            raise NotImplementedError("solve is only available for rskelf FLAM factors")
        pyflam = _require_pyflam()
        arr = np.asarray(rhs)
        one_dim = arr.ndim == 1
        if one_dim:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != self.shape[0]:
            raise ValueError("right-hand side has incompatible row count")
        out = pyflam.rskelf_partial_sv(self.factor, arr)
        return out[:, 0] if one_dim else out

    def logdet(self):
        """Return the FLAM log-determinant, including any residual skeleton block."""

        if self.flamtype != "rskelf":
            raise NotImplementedError("logdet is only available for rskelf FLAM factors")
        return _rskelf_logdet_complete(self.factor)

    def toarray(self) -> np.ndarray:
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

    def _apply(self, x: ArrayLike) -> np.ndarray:
        pyflam = _require_pyflam()
        arr = np.asarray(x)
        one_dim = arr.ndim == 1
        if one_dim:
            arr = arr.reshape(-1, 1)
        if self.flamtype == "rskelf":
            out = pyflam.rskelf_partial_mv(self.factor, arr)
        elif self.flamtype == "rskel":
            out = pyflam.rskel_mv(self.factor, arr)
        else:
            raise NotImplementedError(f"unsupported FLAM factor type {self.flamtype!r}")
        return out[:, 0] if one_dim else out


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


def chunkerflam(
    chnkobj: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dval: ArrayLike | float | complex = 0.0,
    opts: dict[str, Any] | None = None,
):
    """Build a PyFLAM compressed representation of a chunker system matrix."""

    pyflam = _require_pyflam()
    chnkr = _require_chunker(chnkobj)
    options = {} if opts is None else dict(opts)
    opdims = _kernel_opdims(chnkr, kern)
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    nrows = chnkr.npt * op0
    ncols = chnkr.npt * op1
    if nrows != ncols:
        raise ValueError("chunkerflam requires a square discretized operator")

    dval_vec = _dval_vector(dval, nrows)
    spmat = options.get("sp_nonsmooth", None)
    if spmat is None:
        spmat = _special_overwrite_matrix(chnkr, kern, options)
    else:
        spmat = spmat.tocsr() if sparse.issparse(spmat) else sparse.csr_matrix(spmat)
    if np.any(dval_vec != 0):
        spmat = spmat + sparse.diags(dval_vec, offsets=0, shape=(nrows, nrows), format="csr")

    from .chnk import flam

    l2scale = bool(options.get("l2scale", False))

    def matfun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return flam.kernbyindex(rows, cols, chnkr, kern, (op0, op1), spmat, l2scale)

    srcinfo = pointinfo(chnkr)
    xflam = np.repeat(np.real(srcinfo.r), op1, axis=1)
    flamtype = str(options.get("flamtype", "rskelf")).lower()
    occ = int(options.get("occ", 200))
    rank_or_tol = options.get("rank_or_tol", options.get("eps", options.get("tol", 1.0e-14)))
    rank_or_tol = int(rank_or_tol) if float(rank_or_tol).is_integer() and float(rank_or_tol) >= 1 else float(rank_or_tol)
    opts_flam = {
        "verb": int(bool(options.get("verb", False))),
        "lvlmax": options.get("lvlmax", np.inf),
    }
    useproxy = bool(options.get("useproxy", True)) and chnkr.datadim == 0 and op0 == op1
    pxyfun = None
    if useproxy:
        pxyfun = _chunkerflam_proxyfun(chnkr, kern, (op0, op1), options)

    if flamtype == "rskelf":
        return pyflam.rskelf(matfun, xflam, occ, rank_or_tol, pxyfun, opts_flam)
    if flamtype == "rskel":
        return pyflam.rskel(matfun, xflam, xflam, occ, rank_or_tol, None, opts_flam)
    raise NotImplementedError("flamtype must be 'rskelf' or 'rskel'")


def chunkermat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opts: dict[str, Any] | None = None,
) -> np.ndarray | ChunkerFMMMatrix | ChunkerFLAMMatrix:
    """Build a native quadrature matrix, or an FMM-backed operator when requested."""

    chnkr = _require_chunker(chnkr)
    options = {} if opts is None else dict(opts)
    acceleration = _acceleration(options)
    if acceleration == "flam":
        return ChunkerFLAMMatrix(chnkr, kern, options.get("dval", 0.0), options)
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
        return chunkermat(chnkr, kern, options) @ dens_vec
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
    polygon test. With ``acceleration="fmm"`` or ``acceleration="flam"``, the
    Laplace double-layer identity is used for accelerated classification, and
    near-boundary targets are corrected by the direct path.
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
    if acceleration in {"fmm", "flam"}:
        from .kernel import kernel

        lap_d = kernel("lap", "d")
        dens = np.ones(chnkr.npt)
        vals = chunkerkerneval(
            chnkr,
            lap_d,
            dens,
            PointInfo(r=pts),
            {"acceleration": acceleration, "eps": float(options.get("eps", options.get("tol", 1e-12)))},
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
    if acceleration == "flam" and same_source_target and _uses_special_quadrature(kern, options):
        vals = chunkermat(chnkr, kern, options) @ np.asarray(dens).reshape(-1, order="F")
        opdims = getattr(kern, "opdims", (1, 1))[0]
        return vals.reshape(opdims, chnkr.npt, order="F")
    if acceleration == "flam":
        if bool(options.get("forceadap", False)):
            targinfo = pointinfo(targobj)
            mat = _target_adaptive_matrix(chnkr, kern, targinfo, options)
            vals = mat @ np.asarray(dens).reshape(-1, order="F")
            return vals.reshape(-1, targinfo.r.shape[1], order="F")
        return _chunkerkerneval_flam(chnkr, kern, dens, targobj, options)
    use_fmm = acceleration == "fmm"
    if use_fmm:
        _require_fmm(kern)
    if same_source_target and _uses_special_quadrature(kern, options) and not use_fmm:
        vals = chunkermat(chnkr, kern, opts) @ np.asarray(dens).reshape(-1, order="F")
        opdims = getattr(kern, "opdims", (1, 1))[0]
        return vals.reshape(opdims, chnkr.npt, order="F")

    srcinfo = pointinfo(chnkr)
    targinfo = pointinfo(targobj)
    if bool(options.get("forceadap", False)):
        mat = _target_adaptive_matrix(chnkr, kern, targinfo, options)
        vals = mat @ np.asarray(dens).reshape(-1, order="F")
        return vals.reshape(-1, targinfo.r.shape[1], order="F")
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
        if same_source_target and _uses_special_quadrature(kern, opts):
            return np.asarray(chunkermat(chnkr, kern, options))
        if bool(options.get("forceadap", False)):
            return _target_adaptive_matrix(chnkr, kern, pointinfo(targobj), options)
        return _chunkerkernevalmat_flam(chnkr, kern, targobj, options)
    if acceleration == "fmm":
        raise NotImplementedError("chunkerkernevalmat does not support FMM acceleration; use chunkerkerneval instead")
    if same_source_target and _uses_special_quadrature(kern, opts):
        return chunkermat(chnkr, kern, opts)

    srcinfo = pointinfo(chnkr)
    targinfo = pointinfo(targobj)
    if bool(options.get("forceadap", False)):
        return _target_adaptive_matrix(chnkr, kern, targinfo, options)
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


def _require_pyflam():
    try:
        import pyflam
    except Exception as exc:  # pragma: no cover - dependency is required in packaged installs.
        raise ImportError("FLAM acceleration requires the pyflam package") from exc
    return pyflam


def _dval_vector(dval: ArrayLike | float | complex, size: int) -> np.ndarray:
    arr = np.asarray(dval)
    if arr.size == 1:
        return np.full(size, arr.reshape(-1)[0], dtype=arr.dtype)
    vec = arr.reshape(-1, order="F")
    if vec.size != size:
        raise ValueError(f"dval must be scalar or length {size}")
    return vec


def _special_overwrite_matrix(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> spmatrix:
    if not _uses_special_quadrature(kern, options):
        opdims = _kernel_opdims(chnkr, kern)
        return sparse.csr_matrix((chnkr.npt * int(opdims[0]), chnkr.npt * int(opdims[1])))
    from .chnk import quadggq

    qtype = str(options.get("sing", getattr(kern, "sing", "log") or "log")).lower()
    return quadggq.buildmattd(
        chnkr,
        kern,
        getattr(kern, "opdims", None),
        type=qtype,
        ilist=options.get("ilist", None),
        corrections=False,
    )


def _chunkerflam_proxyfun(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
):
    from .chnk import flam

    rank_or_tol = options.get("rank_or_tol", options.get("eps", options.get("tol", 1.0e-14)))
    optsnpxy = {"rank_or_tol": float(rank_or_tol), "nsrc": int(options.get("occ", 200))}
    width = float(np.max(chnkr.max() - chnkr.min()))
    proxybylevel = bool(options.get("proxybylevel", False))
    if not proxybylevel:
        npxy = flam.nproxy_square(kern, width, optsnpxy)
        if npxy == -1:
            return None
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)

        def pxyfun(x: np.ndarray, slf: np.ndarray, nbr: np.ndarray, l: np.ndarray, ctr: np.ndarray):
            _ = x
            return flam.proxyfun(slf, nbr, l, ctr, chnkr, kern, opdims, pr, ptau, pw, pin, True, bool(options.get("l2scale", False)))

        return pxyfun

    def pxyfun(x: np.ndarray, slf: np.ndarray, nbr: np.ndarray, l: np.ndarray, ctr: np.ndarray):
        _ = x
        level_width = float(np.max(np.asarray(l, dtype=float)))
        npxy = flam.nproxy_square(kern, level_width, optsnpxy)
        if npxy == -1:
            return np.zeros((0, np.asarray(slf).size)), np.asarray(nbr, dtype=np.int64)
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)
        return flam.proxyfun(slf, nbr, l, ctr, chnkr, kern, opdims, pr, ptau, pw, pin, True, bool(options.get("l2scale", False)))

    return pxyfun


def _chunkerkerneval_flam(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dens: ArrayLike,
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    factor, matfun, out_shape = _chunkerkerneval_flam_factor(chnkr, kern, targobj, options)
    pyflam = _require_pyflam()
    dens_vec = np.asarray(dens).reshape(-1, order="F")
    if dens_vec.size != out_shape[1]:
        raise ValueError("density has incompatible size")
    vals = pyflam.ifmm_mv(factor, dens_vec, matfun)
    return np.asarray(vals).reshape(-1, out_shape[0] // _kernel_opdims(chnkr, kern)[0], order="F")


def _chunkerkernevalmat_flam(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    factor, matfun, out_shape = _chunkerkerneval_flam_factor(chnkr, kern, targobj, options)
    pyflam = _require_pyflam()
    eye = np.eye(out_shape[1], dtype=_operator_dtype(chnkr, kern))
    return np.asarray(pyflam.ifmm_mv(factor, eye, matfun))


def _chunkerkerneval_flam_factor(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
):
    pyflam = _require_pyflam()
    from .chnk import flam

    targinfo = pointinfo(targobj)
    op0, op1 = _kernel_opdims(chnkr, kern)
    nrows = targinfo.r.shape[1] * op0
    ncols = chnkr.npt * op1

    def matfun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return flam.kernbyindexr(rows, cols, targinfo, chnkr, kern, (op0, op1))

    rx = np.repeat(np.real(targinfo.r), op0, axis=1)
    cx = np.repeat(np.real(pointinfo(chnkr).r), op1, axis=1)
    occ = int(options.get("occ", 200))
    rank_or_tol = options.get("rank_or_tol", options.get("eps", options.get("tol", 1.0e-14)))
    rank_or_tol = int(rank_or_tol) if float(rank_or_tol).is_integer() and float(rank_or_tol) >= 1 else float(rank_or_tol)
    opts_ifmm = {
        "verb": int(bool(options.get("verb", False))),
        "lvlmax": options.get("lvlmax", np.inf),
        # Store near and diagonal blocks; application still receives matfun for
        # interactions not retained by the compact representation.
        "store": options.get("store", "n"),
    }
    pxyfun = None
    if bool(options.get("useproxy", True)) and chnkr.datadim == 0:
        pxyfun = _chunkerkerneval_proxyfun(chnkr, kern, targinfo, (op0, op1), options)
    factor = pyflam.ifmm(matfun, rx, cx, occ, rank_or_tol, pxyfun, opts_ifmm)
    return factor, matfun, (nrows, ncols)


def _chunkerkerneval_proxyfun(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targinfo: PointInfo,
    opdims: tuple[int, int],
    options: dict[str, Any],
):
    from .chnk import flam

    rank_or_tol = options.get("rank_or_tol", options.get("eps", options.get("tol", 1.0e-14)))
    optsnpxy = {"rank_or_tol": float(rank_or_tol), "nsrc": int(options.get("occ", 200))}
    all_points = np.column_stack((np.real(targinfo.r), np.real(pointinfo(chnkr).r)))
    width = float(np.max(np.max(all_points, axis=1) - np.min(all_points, axis=1)))
    npxy = flam.nproxy_square(kern, width, optsnpxy)
    if npxy == -1:
        return None
    pr, ptau, pw, pin = flam.proxy_square_pts(npxy)

    def pxyfun(rc: str, rx: np.ndarray, cx: np.ndarray, slf: np.ndarray, nbr: np.ndarray, l: np.ndarray, ctr: np.ndarray):
        return flam.proxyfunr(rc, rx, cx, slf, nbr, l, ctr, chnkr, kern, opdims, pr, ptau, pw, pin, targobj=targinfo)

    return pxyfun


def _rskelf_logdet_complete(factor: Any):
    pyflam = _require_pyflam()
    ld = pyflam.rskelf_logdet(factor)
    si = getattr(factor, "Si", None)
    if si is None or np.asarray(si).size == 0:
        return ld
    if getattr(factor, "A_dense", None) is not None:
        skel = factor.A_dense[np.ix_(si, si)]
    elif getattr(factor, "A", None) is not None:
        skel = factor.A(si, si)
    else:
        return ld
    S = factor.S.toarray() if sparse.issparse(factor.S) else np.asarray(factor.S)
    sign, logabs = np.linalg.slogdet(np.asarray(skel) + S)
    return ld + np.log(np.asarray(sign, dtype=complex)) + logabs


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


def _target_adaptive_matrix(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targinfo: PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    """Build a target-evaluation matrix with adaptive close-panel replacements."""

    from .chnk import quadadap

    opdims = _kernel_opdims(chnkr, kern)
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    srcinfo = pointinfo(chnkr)
    mat = _eval_kernel(kern, srcinfo, targinfo)
    wts = chnkr.wts.reshape(-1, order="F")
    if mat.shape[1] == chnkr.npt:
        mat = mat * wts[None, :]
    elif mat.shape[1] % chnkr.npt == 0:
        mat = mat * np.repeat(wts, mat.shape[1] // chnkr.npt)[None, :]
    else:
        raise ValueError("kernel column dimension is incompatible with chunker points")

    flags = chnkr.flagnear(targinfo.r, {"fac": float(options.get("fac", 1.0))})
    if not np.any(flags):
        return mat

    for src_chunk in range(chnkr.nch):
        target_ids = np.flatnonzero(flags[:, src_chunk])
        if target_ids.size == 0:
            continue
        subinfo = PointInfo(
            r=targinfo.r[:, target_ids],
            d=None if targinfo.d is None else targinfo.d[:, target_ids],
            d2=None if targinfo.d2 is None else targinfo.d2[:, target_ids],
            n=None if targinfo.n is None else targinfo.n[:, target_ids],
            data=None if targinfo.data is None else targinfo.data[:, target_ids],
        )
        submat = quadadap.adapgausswts(chnkr, src_chunk, subinfo, kern, (op0, op1), opts=options)[0]
        col_start = src_chunk * chnkr.k * op1
        cols = slice(col_start, col_start + chnkr.k * op1)
        for local_idx, target_idx in enumerate(target_ids):
            rows = slice(op0 * target_idx, op0 * (target_idx + 1))
            local_rows = slice(op0 * local_idx, op0 * (local_idx + 1))
            mat[rows, cols] = submat[local_rows, :]
    return mat


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
