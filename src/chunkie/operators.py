"""Operator assembly, matrix-free application, and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

from . import lege
from .geometry.chunker import Chunker, ChunkerPref, merge
from .geometry.pointinfo import PointInfo


_KERNEL_PROBE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    FloatingPointError,
    NotImplementedError,
)


_NORMALIZED_OPTIONS_MARKER = "_chunkie_normalized_operator_options"


def _normalize_public_options(
    opts: dict[str, Any] | None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    l2scale: bool | None = None,
    dval: ArrayLike | float | complex | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    corrections: bool | None = None,
    side: str | None = None,
    near_factor: float | None = None,
) -> dict[str, Any]:
    if opts is None:
        options: dict[str, Any] = {}
    elif bool(opts.get(_NORMALIZED_OPTIONS_MARKER, False)):
        options = dict(opts)
    else:
        warnings.warn(
            "operator option dictionaries are deprecated; use keyword-only arguments instead",
            DeprecationWarning,
            stacklevel=3,
        )
        options = dict(opts)
    _set_option(options, "acceleration", acceleration)
    _set_option(options, "usepquad", use_panel_quadrature)
    _set_option(options, "l2scale", l2scale)
    _set_option(options, "dval", dval)
    _set_option(options, "tol", tol)
    _set_option(options, "flamtype", flam_type)
    _set_option(options, "rank_or_tol", rank_or_tol)
    _set_option(options, "useproxy", proxy)
    _set_option(options, "forceadap", force_adaptive)
    _set_option(options, "corrections", corrections)
    _set_option(options, "side", side)
    _set_option(options, "fac", near_factor)
    if quadrature is not None:
        qmode = str(quadrature).lower()
        if qmode == "smooth":
            options["forcesmooth"] = True
        elif qmode == "adaptive":
            options["forceadap"] = True
            options["adaptive_correction"] = True
        elif qmode != "auto":
            options["sing"] = qmode
    options[_NORMALIZED_OPTIONS_MARKER] = True
    return options


def _set_option(options: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        options[key] = value


@dataclass
class _BlockKernelLayout:
    chunkers: list[Chunker]
    kernels: np.ndarray
    opdims_mat: np.ndarray
    rowdims: np.ndarray
    coldims: np.ndarray
    row_offsets: np.ndarray
    col_offsets: np.ndarray


@dataclass
class RCIPContext:
    """Corner compression metadata produced by chunkgraph RCIP assembly."""

    source: Any
    system_kernel: Callable[[Any, Any], np.ndarray]
    saved: list[Any]
    nsub: int
    savedepth: int
    ndim: int = 1


class ChunkerRCIPMatrix(np.ndarray):
    """Dense matrix with attached RCIP interpolation metadata."""

    rcip: RCIPContext | None

    def __new__(cls, input_array: ArrayLike, rcip_context: RCIPContext | None = None):
        obj = np.asarray(input_array).view(cls)
        obj.rcip = rcip_context
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        self.rcip = None if obj is None else getattr(obj, "rcip", None)


@dataclass(frozen=True)
class _OperatorOptions:
    raw: dict[str, Any]

    @classmethod
    def from_any(cls, opts: dict[str, Any] | "_OperatorOptions" | None) -> "_OperatorOptions":
        if isinstance(opts, cls):
            return opts
        return cls({} if opts is None else dict(opts))

    @property
    def acceleration(self) -> str:
        value = self.raw.get("acceleration", "dense")
        if value is None:
            return "dense"
        acceleration = str(value).lower()
        if acceleration not in {"dense", "fmm", "flam"}:
            raise ValueError("acceleration must be one of 'dense', 'fmm', or 'flam'")
        return acceleration

    def flag(self, name: str, default: bool = False) -> bool:
        return _option_bool(self.raw.get(name, default))

    @property
    def l2scale(self) -> bool:
        return self.flag("l2scale")

    @property
    def flamtype(self) -> str:
        return str(self.raw.get("flamtype", "rskelf")).lower()

    @property
    def flam_occ(self) -> int:
        return int(self.raw.get("occ", 200))

    @property
    def flam_rank_or_tol(self) -> int | float:
        value = self.raw.get("rank_or_tol", self.raw.get("eps", self.raw.get("tol", 1.0e-14)))
        value_float = float(value)
        return int(value) if value_float.is_integer() and value_float >= 1 else value_float

    def flam_options(self, *, store_default: str | None = None) -> dict[str, Any]:
        opts = {
            "verb": int(self.flag("verb")),
            "lvlmax": self.raw.get("lvlmax", np.inf),
        }
        if store_default is not None:
            opts["store"] = self.raw.get("store", store_default)
        return opts

    def fmm_tol(self, default: float = 1.0e-12) -> float:
        return float(self.raw.get("eps", self.raw.get("tol", default)))

    def uses_special_quadrature(self, kern: Callable[[Any, Any], np.ndarray]) -> bool:
        if self.flag("forcesmooth") or self.flag("usesmooth"):
            return False
        if self.flag("forceadap"):
            return True
        return getattr(kern, "sing", "") in {"log", "pv", "hs"}

    def special_quadrature_type(self, kern: Callable[[Any, Any], np.ndarray]) -> str:
        qtype = str(self.raw.get("sing", getattr(kern, "sing", "log") or "log")).lower()
        return "log" if qtype == "smooth" else qtype


class ChunkerFMMMatrix(LinearOperator):
    """Matrix-free ``chunkermat`` operator using kernel FMM application."""

    def __init__(
        self,
        chnkr: Chunker,
        kern: Callable[[Any, Any], np.ndarray],
        opts: dict[str, Any] | None = None,
    ):
        self.kern = kern
        self.opts = {} if opts is None else dict(opts)
        self._correction_mat: spmatrix | None = None
        if _is_block_kernel_matrix(kern):
            self.chnkr = chnkr
            self.block_layout = _block_kernel_layout(chnkr, kern)
            self.opdims = None
            for item in self.block_layout.kernels.flat:
                _require_fmm(item)
            dtype = _block_operator_dtype(self.block_layout)
            shape = (int(self.block_layout.row_offsets[-1]), int(self.block_layout.col_offsets[-1]))
        else:
            self.chnkr = _require_chunker(chnkr)
            self.block_layout = None
            self.opdims = _kernel_opdims(self.chnkr, kern)
            _require_fmm(kern)
            dtype = _operator_dtype(self.chnkr, kern)
            shape = (self.chnkr.npt * int(self.opdims[0]), self.chnkr.npt * int(self.opdims[1]))
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.shape[1]:
            raise ValueError("density has incompatible size")
        if self.block_layout is not None:
            return _block_chunkermatapply_fmm(self.block_layout, x, self.opts, self._correction())
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
        if self.block_layout is not None:
            if not _block_uses_special_quadrature(self.block_layout, self.opts):
                return None
            if self._correction_mat is None:
                self._correction_mat = _block_special_correction_matrix(self.block_layout, self.opts)
            return self._correction_mat
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
        self.chnkr = _require_chunker(chnkr) if not _is_block_kernel_matrix(kern) else chnkr
        self.kern = kern
        self.opts = {} if opts is None else dict(opts)
        self.factor = chunkerflam(self.chnkr, self.kern, dval, self.opts)
        self.flamtype = _flamtype(self.opts)
        if _is_block_kernel_matrix(kern):
            self.block_layout = _block_kernel_layout(chnkr, kern)
            self.opdims = None
            dtype = _block_operator_dtype(self.block_layout)
            shape = (int(self.block_layout.row_offsets[-1]), int(self.block_layout.col_offsets[-1]))
        else:
            self.block_layout = None
            self.opdims = _kernel_opdims(self.chnkr, kern)
            dtype = _operator_dtype(self.chnkr, kern)
            shape = (self.chnkr.npt * int(self.opdims[0]), self.chnkr.npt * int(self.opdims[1]))
        if np.asarray(dval).size:
            dtype = np.result_type(dtype, np.asarray(dval).dtype)
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

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.shape[0]:
            raise ValueError("density has incompatible size")
        return self._apply(x, trans="c")

    def _rmatmat(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.shape[0]:
            raise ValueError("density matrix has incompatible row count")
        if x.shape[1] == 0:
            return np.empty((self.shape[1], 0), dtype=np.result_type(self.dtype, x.dtype))
        return self._apply(x, trans="c")

    def solve(self, rhs: ArrayLike, trans: str = "n") -> np.ndarray:
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
        out = pyflam.rskelf_partial_sv(self.factor, arr, trans=trans)
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

    def _apply(self, x: ArrayLike, trans: str = "n") -> np.ndarray:
        pyflam = _require_pyflam()
        arr = np.asarray(x)
        one_dim = arr.ndim == 1
        if one_dim:
            arr = arr.reshape(-1, 1)
        if self.flamtype == "rskelf":
            out = pyflam.rskelf_partial_mv(self.factor, arr, trans=trans)
        elif self.flamtype == "rskel":
            out = pyflam.rskel_mv(self.factor, arr, trans=trans)
        else:
            raise NotImplementedError(f"unsupported FLAM factor type {self.flamtype!r}")
        return out[:, 0] if one_dim else out


def chunkerflam(
    chnkobj: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dval: ArrayLike | float | complex = 0.0,
    opts: dict[str, Any] | None = None,
):
    """Build a PyFLAM compressed representation of a chunker system matrix.

    ``chnkobj`` may be a single chunker, a chunkgraph-like object, or an
    explicit sequence of edge chunkers when ``kern`` is an edge-by-edge block
    kernel matrix. ``dval`` adds a diagonal shift to the discretized operator.
    Important options include ``flamtype`` (``"rskelf"`` or ``"rskel"``),
    ``occ``, ``rank_or_tol``/``eps``/``tol``, ``useproxy``, and ``l2scale``.
    """

    pyflam = _require_pyflam()
    options = {} if opts is None else dict(opts)
    if _is_block_kernel_matrix(kern):
        return _chunkerflam_block(chnkobj, kern, dval, options, pyflam)

    chnkr = _require_chunker(chnkobj)
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
    has_dval = bool(np.any(dval_vec != 0))

    from .acceleration import flam

    l2scale = _l2scale(options)

    def matfun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        out = flam.kernbyindex(rows, cols, chnkr, kern, (op0, op1), spmat, l2scale)
        if has_dval:
            out = _add_diagonal_shift(out, rows, cols, dval_vec)
        return out

    srcinfo = PointInfo.from_any(chnkr)
    xflam = np.repeat(np.real(srcinfo.r), op1, axis=1)
    flamtype = _flamtype(options)
    occ = _flam_occ(options)
    rank_or_tol = _flam_rank_or_tol(options)
    opts_flam = _flam_options(options)
    useproxy = _flag(options, "useproxy", True) and chnkr.datadim == 0 and op0 == op1
    pxyfun = None
    if useproxy and flamtype == "rskelf":
        pxyfun = _chunkerflam_proxyfun(chnkr, kern, (op0, op1), options)
    elif useproxy and flamtype == "rskel":
        pxyfun = _chunkerkerneval_proxyfun(chnkr, kern, PointInfo.from_any(chnkr), (op0, op1), options)

    if flamtype == "rskelf":
        return pyflam.rskelf(matfun, xflam, occ, rank_or_tol, pxyfun, opts_flam)
    if flamtype == "rskel":
        return pyflam.rskel(matfun, xflam, xflam, occ, rank_or_tol, pxyfun, opts_flam)
    raise NotImplementedError("flamtype must be 'rskelf' or 'rskel'")


def _chunkerflam_block(
    chnkobj: Any,
    kerns: Any,
    dval: ArrayLike | float | complex,
    options: dict[str, Any],
    pyflam: Any,
):
    layout = _block_kernel_layout(chnkobj, kerns)
    nrows = int(layout.row_offsets[-1])
    ncols = int(layout.col_offsets[-1])
    if nrows != ncols:
        raise ValueError("chunkerflam requires a square discretized operator")

    dval_vec = _dval_vector(dval, nrows)
    spmat = options.get("sp_nonsmooth", None)
    if spmat is None:
        spmat = _block_special_overwrite_matrix(layout, options)
    else:
        spmat = spmat.tocsr() if sparse.issparse(spmat) else sparse.csr_matrix(spmat)
    has_dval = bool(np.any(dval_vec != 0))

    from .acceleration import flam

    l2scale = _l2scale(options)

    def matfun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        out = flam.kernbyindex(rows, cols, layout.chunkers, layout.kernels, layout.opdims_mat, spmat, l2scale)
        if has_dval:
            out = _add_diagonal_shift(out, rows, cols, dval_vec)
        return out

    xflam = np.column_stack(
        [
            np.repeat(np.real(PointInfo.from_any(chnkr).r), int(opdim), axis=1)
            for chnkr, opdim in zip(layout.chunkers, layout.coldims)
        ]
    )
    flamtype = _flamtype(options)
    occ = _flam_occ(options)
    rank_or_tol = _flam_rank_or_tol(options)
    opts_flam = _flam_options(options)
    if flamtype == "rskelf":
        return pyflam.rskelf(matfun, xflam, occ, rank_or_tol, None, opts_flam)
    if flamtype == "rskel":
        return pyflam.rskel(matfun, xflam, xflam, occ, rank_or_tol, None, opts_flam)
    raise NotImplementedError("flamtype must be 'rskelf' or 'rskel'")


def chunkermat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opts: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    l2scale: bool | None = None,
    dval: ArrayLike | float | complex | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    side: str | None = None,
) -> np.ndarray | ChunkerFMMMatrix | ChunkerFLAMMatrix | ChunkerRCIPMatrix | tuple[np.ndarray, RCIPContext]:
    """Assemble or factor a boundary-integral operator on a chunker-like object.

    The default path returns a dense NumPy matrix using native or special
    quadrature. ``acceleration="fmm"`` returns a matrix-free
    :class:`ChunkerFMMMatrix` for kernels with an FMM evaluator.
    ``acceleration="flam"`` returns a :class:`ChunkerFLAMMatrix`
    backed by PyFLAM and accepts ``dval`` for second-kind or shifted systems.
    Block kernel matrices are supported for explicit chunker sequences and
    chunkgraphs.
    """

    options = _normalize_public_options(
        opts,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        l2scale=l2scale,
        dval=dval,
        tol=tol,
        flam_type=flam_type,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        side=side,
    )
    if _is_block_kernel_matrix(kern):
        acceleration = _acceleration(options)
        if acceleration == "flam":
            return ChunkerFLAMMatrix(chnkr, kern, options.get("dval", 0.0), options)
        if acceleration == "fmm":
            return ChunkerFMMMatrix(chnkr, kern, options)
        return _block_kernel_mat(chnkr, kern, options)

    if _chunkgraph_rcip_mat_enabled(chnkr, kern, options):
        mat, context = _chunkgraph_rcip_mat(chnkr, kern, options)
        if _flag(options, "return_rcip"):
            return mat, context
        return mat
    if _is_chunkgraph_like(chnkr) and "rcip" in options and not _rcip_option_enabled(options):
        try:
            setattr(chnkr, "_last_rcip_context", None)
        except AttributeError:
            pass

    chnkr = _require_chunker(chnkr)
    acceleration = _acceleration(options)
    if acceleration == "flam":
        return ChunkerFLAMMatrix(chnkr, kern, options.get("dval", 0.0), options)
    if acceleration == "fmm":
        _require_fmm(kern)
        return ChunkerFMMMatrix(chnkr, kern, options)
    if _uses_special_quadrature(kern, options):
        if _flag(options, "adaptive_correction"):
            from .quadrature import adaptive as quadadap

            adap_options = dict(options)
            adap_options.setdefault("sing", _special_quadrature_type(kern, options))
            adap_options.setdefault("usepquad", _boundary_pquad_enabled(options))
            mat = quadadap.buildmat(chnkr, kern, getattr(kern, "opdims", None), adap_options)
        else:
            from .quadrature import ggq as quadggq

            mat = quadggq.buildmat(
                chnkr,
                kern,
                getattr(kern, "opdims", None),
                _special_quadrature_type(kern, options),
                pquad_side=_pquad_side(options),
                usepquad=_boundary_pquad_enabled(options),
            )
        return _apply_l2scale_matrix(chnkr, mat) if _l2scale(options) else mat

    srcinfo = PointInfo.from_any(chnkr)
    mat = _eval_kernel(kern, srcinfo, srcinfo)
    wts = chnkr.wts.reshape(-1, order="F")
    if mat.shape[1] == chnkr.npt:
        out = mat * wts[None, :]
        out = _apply_laplace_double_self_limit(chnkr, kern, out)
        out = _apply_laplace_sprime_self_limit(chnkr, kern, out)
        return _apply_l2scale_matrix(chnkr, out) if _l2scale(options) else out
    if mat.shape[1] % chnkr.npt != 0:
        raise ValueError("kernel column dimension is incompatible with chunker points")
    opdims_col = mat.shape[1] // chnkr.npt
    out = mat * np.repeat(wts, opdims_col)[None, :]
    out = _apply_stokes_strac_self_limit(chnkr, kern, out)
    return _apply_l2scale_matrix(chnkr, out) if _l2scale(options) else out


def chunkermatapply(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    dens: ArrayLike,
    opts: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    l2scale: bool | None = None,
    dval: ArrayLike | float | complex | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    side: str | None = None,
) -> np.ndarray:
    """Apply ``chunkermat(chnkr, kern, ...)`` without forcing dense materialization."""

    options = _normalize_public_options(
        opts,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        l2scale=l2scale,
        dval=dval,
        tol=tol,
        flam_type=flam_type,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        side=side,
    )
    acceleration = _acceleration(options)
    if acceleration == "fmm":
        if _is_block_kernel_matrix(kern):
            for item in np.asarray(kern, dtype=object).flat:
                _require_fmm(item)
        else:
            _require_fmm(kern)
    chnkobj = (
        chnkr
        if _is_block_kernel_matrix(kern) or _chunkgraph_rcip_mat_enabled(chnkr, kern, options)
        else _require_chunker(chnkr)
    )
    op = chunkermat(chnkobj, kern, options)
    dens_arg = _density_matmul_arg(op.shape[1], dens)
    return op @ dens_arg


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
    *,
    acceleration: str | None = None,
    tol: float | None = None,
    near_factor: float | None = None,
    close_correction: bool | None = None,
    axis_symmetric: bool | None = None,
) -> np.ndarray:
    """Classify target points as inside a closed 2D chunker.

    The default ``acceleration="dense"`` path is a dependency-light direct
    polygon test. With ``acceleration="fmm"`` or ``acceleration="flam"``, the
    Laplace double-layer identity is used for accelerated classification, and
    near-boundary targets are corrected by the direct path.
    """

    chnkr = _require_chunker(chnkr)
    options = _normalize_public_options(opts, acceleration=acceleration, tol=tol, near_factor=near_factor)
    _set_option(options, "closecorr", close_correction)
    _set_option(options, "axissym", axis_symmetric)
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
        from .kernels import kernel

        lap_d = kernel("lap", "d")
        dens = np.ones(chnkr.npt)
        vals = chunkerkerneval(
            chnkr,
            lap_d,
            dens,
            PointInfo(r=pts),
            {_NORMALIZED_OPTIONS_MARKER: True, "acceleration": acceleration, "eps": _fmm_tol(options)},
        ).reshape(-1, order="F")
        inside = vals < -0.5
        if _flag(options, "closecorr", _flag(options, "corrections", True)):
            near_fac = float(options.get("near_fac", options.get("fac", 1.0)))
            near = np.any(chnkr.flagnear(pts, fac=near_fac), axis=1)
            if np.any(near):
                corrected = _chunkerinterior_direct(chnkr, pts[:, near], _flag(options, "axissym"))
                inside[near] = corrected
    else:
        inside = _chunkerinterior_direct(chnkr, pts, _flag(options, "axissym"))

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
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    side: str | None = None,
) -> np.ndarray:
    """Evaluate a layer potential generated by a chunker density at targets.

    ``dens`` is flattened in Fortran order, including any vector components
    implied by ``kern.opdims``. ``targobj`` may be raw coordinates, a
    ``PointInfo``/mapping, another chunker, or a chunkgraph-like object.
    Options include ``acceleration="fmm"`` for supported FMM kernels,
    ``acceleration="flam"`` for PyFLAM target evaluation, and ``force_adaptive=True``
    for adaptive close-target correction.
    """

    options = _normalize_public_options(
        opts,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        tol=tol,
        flam_type=flam_type,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        force_adaptive=force_adaptive,
        side=side,
    )
    same_source_target = targobj is chnkr
    rcip_context = _chunkgraph_rcip_eval_context(chnkr, options)
    if rcip_context is not None and not same_source_target:
        return _chunkgraph_rcip_eval(chnkr, kern, dens, targobj, options, rcip_context)

    chnkr = _require_chunker(chnkr)
    acceleration = _acceleration(options)
    if same_source_target and _uses_special_quadrature(kern, options):
        mat_options = dict(options)
        mat_options.pop("acceleration", None)
        vals = chunkermat(chnkr, kern, mat_options) @ np.asarray(dens).reshape(-1, order="F")
        opdims = getattr(kern, "opdims", (1, 1))[0]
        return vals.reshape(opdims, chnkr.npt, order="F")
    if acceleration == "flam":
        targinfo = PointInfo.from_any(targobj)
        if _flag(options, "forceadap"):
            smooth_options = dict(options)
            smooth_options.pop("forceadap", None)
            vals = _chunkerkerneval_flam(chnkr, kern, dens, targinfo, smooth_options).reshape(-1, order="F")
            correction_options = dict(options)
            correction_options["recompute_source_normals"] = True
            correction_options.setdefault("transinv", False)
            correction = _target_adaptive_correction_matrix(chnkr, kern, targinfo, correction_options)
            vals = vals + correction @ np.asarray(dens).reshape(-1, order="F")
            return vals.reshape(-1, targinfo.r.shape[1], order="F")
        return _chunkerkerneval_flam(chnkr, kern, dens, targinfo, options)
    use_fmm = acceleration == "fmm"
    if use_fmm:
        _require_fmm(kern)
    if same_source_target and _uses_special_quadrature(kern, options) and not use_fmm:
        vals = chunkermat(chnkr, kern, options) @ np.asarray(dens).reshape(-1, order="F")
        opdims = getattr(kern, "opdims", (1, 1))[0]
        return vals.reshape(opdims, chnkr.npt, order="F")

    srcinfo = PointInfo.from_any(chnkr)
    targinfo = PointInfo.from_any(targobj)
    cormat = options.get("cormat", None)
    if cormat is not None:
        mat = _eval_kernel(kern, srcinfo, targinfo)
        weighted = _weighted_density(chnkr, dens)
        dens_vec = np.asarray(dens).reshape(-1, order="F")
        corr_vals = cormat @ dens_vec if sparse.issparse(cormat) else np.asarray(cormat) @ dens_vec
        vals = mat @ weighted + corr_vals
        return vals.reshape(-1, targinfo.r.shape[1], order="F")
    if _flag(options, "forceadap") and use_fmm:
        weighted = _weighted_density(chnkr, dens)
        vals = kern.fmm(_fmm_tol(options), srcinfo, targinfo, weighted)
        if isinstance(vals, tuple):
            vals = vals[0]
        vals = np.asarray(vals).reshape(-1, order="F")
        correction_options = dict(options)
        correction_options["recompute_source_normals"] = True
        correction_options.setdefault("transinv", False)
        correction = _target_adaptive_correction_matrix(chnkr, kern, targinfo, correction_options)
        vals = vals + correction @ np.asarray(dens).reshape(-1, order="F")
        return vals.reshape(-1, targinfo.r.shape[1], order="F")
    if _flag(options, "forceadap"):
        eval_options = dict(options)
        eval_options["recompute_source_normals"] = True
        eval_options.setdefault("transinv", False)
        mat = _target_adaptive_matrix(chnkr, kern, targinfo, eval_options)
        vals = mat @ np.asarray(dens).reshape(-1, order="F")
        return vals.reshape(-1, targinfo.r.shape[1], order="F")
    if use_fmm:
        weighted = _weighted_density(chnkr, dens)
        vals = kern.fmm(_fmm_tol(options), srcinfo, targinfo, weighted)
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
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    corrections: bool | None = None,
    side: str | None = None,
) -> np.ndarray | spmatrix:
    """Build the matrix mapping source densities to target values.

    This is the materialized companion to :func:`chunkerkerneval`. It is useful
    for diagnostics, custom solvers, and adaptive correction matrices. Ordinary
    evaluation matrices are dense; ``corrections=True`` returns the sparse
    near-target correction matrix accepted by legacy ``cormat`` evaluation
    calls. FMM is a
    matrix-free path and is intentionally unavailable here; use
    :func:`chunkerkerneval` or :func:`chunkermatapply` for FMM application.
    """

    same_source_target = targobj is chnkr
    chnkr = _require_chunker(chnkr)
    options = _normalize_public_options(
        opts,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        tol=tol,
        flam_type=flam_type,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        force_adaptive=force_adaptive,
        corrections=corrections,
        side=side,
    )
    if _flag(options, "corrections"):
        return _target_adaptive_correction_matrix(chnkr, kern, PointInfo.from_any(targobj), options)
    acceleration = _acceleration(options)
    if acceleration == "flam":
        if same_source_target and _uses_special_quadrature(kern, options):
            return np.asarray(chunkermat(chnkr, kern, options))
        if _flag(options, "forceadap"):
            targinfo = PointInfo.from_any(targobj)
            smooth_options = dict(options)
            smooth_options.pop("forceadap", None)
            return _chunkerkernevalmat_flam(chnkr, kern, targinfo, smooth_options) + _target_adaptive_correction_matrix(chnkr, kern, targinfo, options).toarray()
        return _chunkerkernevalmat_flam(chnkr, kern, targobj, options)
    if acceleration == "fmm":
        _require_fmm(kern)
        if same_source_target and _uses_special_quadrature(kern, options):
            return np.asarray(chunkermat(chnkr, kern, options))
        return _chunkerkernevalmat_fmm(chnkr, kern, targobj, options)
    if same_source_target and _uses_special_quadrature(kern, options):
        return chunkermat(chnkr, kern, options)

    srcinfo = PointInfo.from_any(chnkr)
    targinfo = PointInfo.from_any(targobj)
    if _flag(options, "forceadap"):
        return _target_adaptive_matrix(chnkr, kern, targinfo, options)
    mat = _eval_kernel(kern, srcinfo, targinfo)
    wts = chnkr.wts.reshape(-1, order="F")
    if mat.shape[1] == chnkr.npt:
        return mat * wts[None, :]
    if mat.shape[1] % chnkr.npt != 0:
        raise ValueError("kernel column dimension is incompatible with chunker points")
    opdims_col = mat.shape[1] // chnkr.npt
    return mat * np.repeat(wts, opdims_col)[None, :]


def _as_chunker(obj: Any) -> Chunker | None:
    if isinstance(obj, Chunker):
        return obj
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if items and all(isinstance(item, Chunker) for item in items):
            return merge(items)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        items = list(np.ravel(obj))
        if items and all(isinstance(item, Chunker) for item in items):
            return merge(items)
    merged = getattr(obj, "merged", None)
    if callable(merged):
        out = merged()
        if isinstance(out, Chunker):
            return out
    return None


def _as_chunker_sequence(obj: Any) -> list[Chunker] | None:
    if isinstance(obj, Chunker):
        return [obj]
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if items and all(isinstance(item, Chunker) for item in items):
            return items
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        items = list(np.ravel(obj, order="F"))
        if items and all(isinstance(item, Chunker) for item in items):
            return items
    edges = getattr(obj, "echnks", None)
    if edges is not None:
        edge_chunkers = list(edges)
        if edge_chunkers and all(isinstance(item, Chunker) for item in edge_chunkers):
            return edge_chunkers
    return None


def _require_chunker(obj: Any) -> Chunker:
    out = _as_chunker(obj)
    if out is None:
        raise TypeError("expected a chunker or chunkgraph-like object")
    return out


def _is_chunkgraph_like(obj: Any) -> bool:
    return (
        hasattr(obj, "echnks")
        and hasattr(obj, "vstruc")
        and hasattr(obj, "verts")
        and callable(getattr(obj, "merged", None))
    )


def _chunkgraph_nonsmooth_vertices(obj: Any, options: dict[str, Any]) -> list[int]:
    if not _is_chunkgraph_like(obj):
        return []
    raw_vertices = options.get("rcip_vertices", options.get("vertices", None))
    if raw_vertices is None:
        candidates = range(np.asarray(obj.verts).shape[1])
    else:
        candidates = _normalize_vertex_indices(raw_vertices, np.asarray(obj.verts).shape[1])
    ignored = set(
        _normalize_vertex_indices(
            options.get("rcip_ignore_vertices", options.get("ignore_vertices", [])),
            np.asarray(obj.verts).shape[1],
        ).tolist()
    )
    out: list[int] = []
    for ivert in candidates:
        if int(ivert) in ignored:
            continue
        edges, _ = obj.vstruc[int(ivert)]
        if np.asarray(edges).size >= 2:
            out.append(int(ivert))
    return out


def _normalize_vertex_indices(vertices: Any, nvert: int) -> np.ndarray:
    arr = np.asarray(vertices, dtype=int).reshape(-1)
    if arr.size and np.max(arr) >= int(nvert):
        if np.min(arr) >= 1 and np.max(arr) <= int(nvert):
            arr = arr - 1
        else:
            raise ValueError("vertex index out of range")
    if np.any(arr < 0) or np.any(arr >= int(nvert)):
        raise ValueError("vertex index out of range")
    return arr


def _rcip_option_enabled(options: dict[str, Any]) -> bool:
    value = options.get("rcip", True)
    if isinstance(value, RCIPContext):
        return True
    return _option_bool(value)


def _chunkgraph_rcip_mat_enabled(obj: Any, kern: Callable[[Any, Any], np.ndarray], options: dict[str, Any]) -> bool:
    if not _rcip_option_enabled(options):
        return False
    if not _is_chunkgraph_like(obj):
        return False
    if _acceleration(options) != "dense" or _l2scale(options):
        return False
    if not _is_rcip_second_kind_kernel(kern):
        return False
    merged = _as_chunker(obj)
    if merged is None:
        return False
    try:
        if _kernel_opdims(merged, kern) != (1, 1):
            return False
    except _KERNEL_PROBE_EXCEPTIONS:
        return False
    return bool(_chunkgraph_nonsmooth_vertices(obj, options))


def _is_rcip_second_kind_kernel(kern: Callable[[Any, Any], np.ndarray]) -> bool:
    name = str(getattr(kern, "name", "")).lower()
    typ = str(getattr(kern, "type", "")).lower()
    if name not in {"laplace", "helmholtz"}:
        return False
    return typ in {
        "d",
        "double",
        "sp",
        "sprime",
    }


def _chunkgraph_rcip_mat(
    cg: Any,
    kern: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> tuple[ChunkerRCIPMatrix, RCIPContext]:
    from .quadrature import rcip

    merged = cg.merged()
    base_options = _strip_rcip_options(options)
    mat = np.asarray(chunkermat(merged, kern, base_options)).copy()
    nsub = _rcip_nsub(options)
    savedepth = _rcip_savedepth(options, nsub)
    saved: list[Any] = []

    for ivert in _chunkgraph_nonsmooth_vertices(cg, options):
        edges, signs = cg.vstruc[ivert]
        edges = np.asarray(edges, dtype=int).reshape(-1)
        signs = np.asarray(signs, dtype=int).reshape(-1)
        isstart = signs < 0
        iedgechunks = _rcip_corner_edge_chunks(cg, edges, signs)
        pbc, pwbc, star_l, circ_l, star_s, circ_s, *_ = rcip.setup(cg.k, 1, edges.size, isstart)
        rmat, rcipsav = rcip.Rcompchunk(
            cg.echnks,
            iedgechunks,
            kern,
            1,
            cg.verts[:, ivert],
            Pbc=pbc,
            PWbc=pwbc,
            starL=star_l,
            circL=circ_l,
            starS=star_s,
            circS=circ_s,
            opts={"nsub": nsub, "rcip_savedepth": savedepth},
        )
        starind = _rcip_corner_star_indices(cg, edges, signs)
        rcipsav.starind = starind
        replacement = np.linalg.inv(rmat) - np.eye(rmat.shape[0], dtype=rmat.dtype)
        mat[np.ix_(starind, starind)] = replacement
        saved.append(rcipsav)

    context = RCIPContext(cg, kern, saved, nsub, savedepth, 1)
    try:
        setattr(cg, "_last_rcip_context", context)
    except AttributeError:
        pass
    return ChunkerRCIPMatrix(mat, context), context


def _rcip_nsub(options: dict[str, Any]) -> int:
    value = options.get("nsub", options.get("rcip_nsub", options.get("nsub_or_tol", 20)))
    value_float = float(value)
    if value_float <= 0:
        raise ValueError("RCIP nsub must be positive")
    if value_float < 1.0:
        return max(int(np.ceil(np.log2(1.0 / value_float**2))), 20)
    return int(value_float)


def _rcip_savedepth(options: dict[str, Any], nsub: int) -> int:
    savedepth = int(options.get("rcip_savedepth", options.get("save_depth", nsub)))
    return min(max(savedepth, 0), int(nsub))


def _strip_rcip_options(options: dict[str, Any]) -> dict[str, Any]:
    out = dict(options)
    for key in (
        "rcip",
        "return_rcip",
        "rcip_context",
        "rcip_saved",
        "rcip_vertices",
        "vertices",
        "rcip_ignore_vertices",
        "ignore_vertices",
        "nsub",
        "rcip_nsub",
        "nsub_or_tol",
        "rcip_savedepth",
        "save_depth",
        "rcip_eval_depth",
        "rcip_ndepth",
    ):
        out.pop(key, None)
    return out


def _rcip_corner_star_indices(cg: Any, edges: np.ndarray, signs: np.ndarray) -> np.ndarray:
    starts = np.cumsum([0] + [edge.npt for edge in cg.echnks])
    width = 2 * int(cg.k)
    out: list[np.ndarray] = []
    for edge, sign in zip(np.asarray(edges, dtype=int), np.asarray(signs, dtype=int)):
        lo = int(starts[int(edge)])
        hi = int(starts[int(edge) + 1])
        if hi - lo < width:
            raise ValueError("each RCIP edge needs at least two coarse panels")
        out.append(np.arange(lo, lo + width) if int(sign) < 0 else np.arange(hi - width, hi))
    return np.concatenate(out)


def _rcip_corner_edge_chunks(cg: Any, edges: np.ndarray, signs: np.ndarray) -> np.ndarray:
    edge_arr = np.asarray(edges, dtype=int).reshape(-1)
    sign_arr = np.asarray(signs, dtype=int).reshape(-1)
    out = np.zeros((2, edge_arr.size), dtype=int)
    out[0] = edge_arr
    for idx, (edge, sign) in enumerate(zip(edge_arr, sign_arr)):
        out[1, idx] = 0 if int(sign) < 0 else cg.echnks[int(edge)].nch - 1
    return out


def _chunkgraph_rcip_eval_context(obj: Any, options: dict[str, Any]) -> RCIPContext | None:
    value = options.get("rcip", None)
    if value is not None and not isinstance(value, RCIPContext) and not _option_bool(value):
        return None
    if isinstance(value, RCIPContext):
        return value
    context = options.get("rcip_context", None)
    if context is not None:
        return context
    if "rcip_saved" in options:
        return RCIPContext(
            obj,
            options.get("rcip_system_kernel", lambda s, t: np.zeros((t.r.shape[1], s.r.shape[1]))),
            list(options["rcip_saved"]),
            _rcip_nsub(options),
            _rcip_savedepth(options, _rcip_nsub(options)),
            1,
        )
    if not _is_chunkgraph_like(obj):
        return None
    return getattr(obj, "_last_rcip_context", None)


def _chunkgraph_rcip_eval(
    cg: Any,
    kern: Callable[[Any, Any], np.ndarray],
    dens: ArrayLike,
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
    context: RCIPContext,
) -> np.ndarray:
    from .quadrature import rcip

    if not context.saved:
        return chunkerkerneval(cg.merged(), kern, dens, targobj, _strip_rcip_options(options))

    dens_vec = np.asarray(dens).reshape(-1, order="F")
    merged = cg.merged()
    if dens_vec.size != merged.npt * int(context.ndim):
        raise ValueError("density has incompatible size for RCIP context")

    rho_coarse = dens_vec.copy()
    for rcipsav in context.saved:
        starind = np.asarray(rcipsav.starind, dtype=int).reshape(-1)
        rho_coarse[starind] = 0.0

    eval_options = _strip_rcip_options(options)
    targinfo = PointInfo.from_any(targobj)
    vals = chunkerkerneval(merged, kern, rho_coarse, targinfo, eval_options).reshape(-1, order="F")
    ndepth = int(options.get("rcip_eval_depth", options.get("rcip_ndepth", context.savedepth)))

    for rcipsav in context.saved:
        starind = np.asarray(rcipsav.starind, dtype=int).reshape(-1)
        rhocells, srcinfos, wtscells = rcip.rhohatInterp(dens_vec[starind], rcipsav, ndepth)
        local_targets = _shift_pointinfo(targinfo, np.asarray(rcipsav.ctr)[:, [0]])
        if _flag(eval_options, "forceadap"):
            for rho, srcinfo, wts in zip(rhocells, srcinfos, wtscells):
                if srcinfo is None or wts is None:
                    continue
                local_chunker = _chunker_from_pointinfo(srcinfo, wts, rcipsav.k)
                vals = vals + chunkerkerneval(local_chunker, kern, rho, local_targets, eval_options).reshape(-1, order="F")
        else:
            srcinfo = _merge_pointinfos([src for src in srcinfos if src is not None])
            if srcinfo is None:
                continue
            rho_local = np.concatenate([np.asarray(rho).reshape(-1, order="F") for rho in rhocells])
            wts_local = np.concatenate([np.asarray(wts).reshape(-1, order="F") for wts in wtscells if wts is not None])
            local_mat = _eval_kernel(kern, srcinfo, local_targets)
            vals = vals + local_mat @ _apply_weights_to_density(rho_local, wts_local)

    return vals.reshape(-1, targinfo.r.shape[1], order="F")


def _merge_pointinfos(infos: list[PointInfo]) -> PointInfo | None:
    if not infos:
        return None
    return PointInfo(
        r=np.column_stack([info.r for info in infos]),
        d=None if any(info.d is None for info in infos) else np.column_stack([info.d for info in infos]),
        d2=None if any(info.d2 is None for info in infos) else np.column_stack([info.d2 for info in infos]),
        n=None if any(info.n is None for info in infos) else np.column_stack([info.n for info in infos]),
        data=None if any(info.data is None for info in infos) else np.column_stack([info.data for info in infos]),
    )


def _shift_pointinfo(info: PointInfo, ctr: np.ndarray) -> PointInfo:
    return PointInfo(
        r=info.r - np.asarray(ctr).reshape(info.r.shape[0], 1),
        d=info.d,
        d2=info.d2,
        n=info.n,
        data=info.data,
    )


def _chunker_from_pointinfo(info: PointInfo, wts: np.ndarray, k: int) -> Chunker:
    npt = int(info.r.shape[1])
    k = int(k)
    if npt % k != 0:
        raise ValueError("RCIP local source points must be whole chunks")
    nch = npt // k
    out = Chunker(ChunkerPref(k=k, dim=info.r.shape[0], nchstor=nch, nchmax=nch))
    out.addchunk(nch)
    out.r = info.r.reshape(info.r.shape[0], k, nch, order="F")
    if info.d is not None:
        out.d = info.d.reshape(info.r.shape[0], k, nch, order="F")
    if info.d2 is not None:
        out.d2 = info.d2.reshape(info.r.shape[0], k, nch, order="F")
    if info.n is not None:
        out.n = info.n.reshape(info.r.shape[0], k, nch, order="F")
    out.wts = np.asarray(wts).reshape(k, nch, order="F")
    if nch:
        out.adj = np.vstack(
            (
                np.concatenate(([-1], np.arange(1, nch, dtype=int))),
                np.concatenate((np.arange(2, nch + 1, dtype=int), [-1])),
            )
        )
    return out


def _apply_weights_to_density(rho: np.ndarray, wts: np.ndarray) -> np.ndarray:
    if rho.size == wts.size:
        return rho * wts
    if rho.size % wts.size != 0:
        raise ValueError("RCIP local density and weights are incompatible")
    return rho * np.repeat(wts, rho.size // wts.size)


def _is_block_kernel_matrix(kern: Any) -> bool:
    if callable(kern):
        return False
    try:
        arr = np.asarray(kern, dtype=object)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.size > 0 and all(callable(item) for item in arr.flat)


def _block_kernel_layout(obj: Any, kerns: Any) -> _BlockKernelLayout:
    chunkers = _as_chunker_sequence(obj)
    if chunkers is None:
        raise TypeError("block kernel matrices require a chunker sequence or chunkgraph-like object")
    nchunker = len(chunkers)
    kernels = np.asarray(kerns, dtype=object)
    if kernels.shape != (nchunker, nchunker):
        raise ValueError("block kernel matrix shape must match the number of chunkers")

    rowdims = np.zeros(nchunker, dtype=int)
    coldims = np.zeros(nchunker, dtype=int)
    opdims_mat = np.zeros((2, nchunker, nchunker), dtype=int)
    for itarg, targ in enumerate(chunkers):
        for isrc, src in enumerate(chunkers):
            op0, op1 = _kernel_opdims_between(src, targ, kernels[itarg, isrc])
            opdims_mat[:, itarg, isrc] = (op0, op1)
            if rowdims[itarg] == 0:
                rowdims[itarg] = op0
            elif rowdims[itarg] != op0:
                raise ValueError("block kernel row operator dimensions are inconsistent")
            if coldims[isrc] == 0:
                coldims[isrc] = op1
            elif coldims[isrc] != op1:
                raise ValueError("block kernel column operator dimensions are inconsistent")

    row_offsets = np.concatenate(([0], np.cumsum([chnkr.npt * dim for chnkr, dim in zip(chunkers, rowdims)])))
    col_offsets = np.concatenate(([0], np.cumsum([chnkr.npt * dim for chnkr, dim in zip(chunkers, coldims)])))
    return _BlockKernelLayout(chunkers, kernels, opdims_mat, rowdims, coldims, row_offsets, col_offsets)


def _block_operator_dtype(layout: _BlockKernelLayout) -> np.dtype:
    dtype = np.dtype(float)
    for itarg, targ in enumerate(layout.chunkers):
        for isrc, src in enumerate(layout.chunkers):
            dtype = np.result_type(dtype, _operator_dtype_between(src, targ, layout.kernels[itarg, isrc]))
    return np.dtype(dtype)


def _block_kernel_mat(obj: Any, kerns: Any, opts: dict[str, Any]) -> np.ndarray:
    layout = _block_kernel_layout(obj, kerns)
    edge_chunkers = layout.chunkers
    kernels = layout.kernels
    nedge = len(edge_chunkers)
    out = np.zeros((int(layout.row_offsets[-1]), int(layout.col_offsets[-1])), dtype=_block_operator_dtype(layout))

    for kern in _unique_kernel_objects(kernels):
        pairs = [(itarg, isrc) for itarg in range(nedge) for isrc in range(nedge) if kernels[itarg, isrc] is kern]
        target_edges = sorted({itarg for itarg, _ in pairs})
        source_edges = sorted({isrc for _, isrc in pairs})
        targ_merged = merge([edge_chunkers[idx] for idx in target_edges])
        src_merged = merge([edge_chunkers[idx] for idx in source_edges])
        mat = _eval_kernel(kern, PointInfo.from_any(src_merged), PointInfo.from_any(targ_merged))
        sample_op1 = int(layout.opdims_mat[1, pairs[0][0], pairs[0][1]])
        src_weights = src_merged.wts.reshape(-1, order="F")
        if mat.shape[1] == src_merged.npt:
            weighted = mat * src_weights[None, :]
        elif mat.shape[1] == src_merged.npt * sample_op1:
            weighted = mat * np.repeat(src_weights, sample_op1)[None, :]
        else:
            raise ValueError("block kernel column dimension is incompatible with source edge points")
        local_rows = _block_offsets_for_edges(edge_chunkers, layout.rowdims, target_edges)
        local_cols = _block_offsets_for_edges(edge_chunkers, layout.coldims, source_edges)
        weighted = _apply_laplace_double_block_self_limits(
            edge_chunkers,
            kern,
            target_edges,
            source_edges,
            local_rows,
            local_cols,
            weighted,
        )
        target_lookup = {edge: idx for idx, edge in enumerate(target_edges)}
        source_lookup = {edge: idx for idx, edge in enumerate(source_edges)}
        for itarg, isrc in pairs:
            rows = slice(int(layout.row_offsets[itarg]), int(layout.row_offsets[itarg + 1]))
            cols = slice(int(layout.col_offsets[isrc]), int(layout.col_offsets[isrc + 1]))
            if itarg == isrc and _uses_special_quadrature(kern, opts):
                local_options = dict(opts)
                local_options.pop("l2scale", None)
                out[rows, cols] = chunkermat(edge_chunkers[itarg], kern, local_options)
                continue
            local_i = target_lookup[itarg]
            local_j = source_lookup[isrc]
            src_rows = slice(int(local_rows[local_i]), int(local_rows[local_i + 1]))
            src_cols = slice(int(local_cols[local_j]), int(local_cols[local_j + 1]))
            out[rows, cols] = weighted[src_rows, src_cols]
    if _l2scale(opts):
        return _apply_chunkgraph_l2scale(edge_chunkers, layout.rowdims, layout.coldims, out)
    return out


def _unique_kernel_objects(kernels: np.ndarray) -> list[Callable[[Any, Any], np.ndarray]]:
    unique: list[Callable[[Any, Any], np.ndarray]] = []
    for item in kernels.flat:
        if not any(item is existing for existing in unique):
            unique.append(item)
    return unique


def _block_offsets_for_edges(edge_chunkers: list[Chunker], dims: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.concatenate(
        ([0], np.cumsum([edge_chunkers[idx].npt * int(dims[idx]) for idx in indices]))
    )


def _kernel_opdims_between(src: Chunker, targ: Chunker, kern: Callable[[Any, Any], np.ndarray]) -> tuple[int, int]:
    opdims = getattr(kern, "opdims", None)
    if opdims is not None and tuple(opdims) != (0, 0):
        return int(opdims[0]), int(opdims[1])
    mat = _eval_kernel(kern, _pointinfo_node(src, 0), _pointinfo_node(targ, 0))
    return int(mat.shape[0]), int(mat.shape[1])


def _operator_dtype_between(src: Chunker, targ: Chunker, kern: Callable[[Any, Any], np.ndarray]) -> np.dtype:
    return _probe_kernel_dtype(kern, _pointinfo_node(src, 0), _pointinfo_node(targ, 0))


def _apply_chunkgraph_l2scale(edge_chunkers: list[Chunker], rowdims: np.ndarray, coldims: np.ndarray, mat: np.ndarray) -> np.ndarray:
    row_scales = np.concatenate(
        [np.repeat(np.sqrt(edge.wts.reshape(-1, order="F")), int(dim)) for edge, dim in zip(edge_chunkers, rowdims)]
    )
    col_scales = np.concatenate(
        [np.repeat(1.0 / np.sqrt(edge.wts.reshape(-1, order="F")), int(dim)) for edge, dim in zip(edge_chunkers, coldims)]
    )
    if sparse.issparse(mat):
        return sparse.diags(row_scales, format="csr") @ mat @ sparse.diags(col_scales, format="csr")
    return row_scales[:, None] * mat * col_scales[None, :]


def _chunker_l2_row_scale(chnkr: Chunker, rowdim: int) -> np.ndarray:
    return np.repeat(np.sqrt(chnkr.wts.reshape(-1, order="F")), int(rowdim))


def _chunker_l2_col_scale(chnkr: Chunker, coldim: int) -> np.ndarray:
    return np.repeat(1.0 / np.sqrt(chnkr.wts.reshape(-1, order="F")), int(coldim))


def _block_l2_row_scale(layout: _BlockKernelLayout) -> np.ndarray:
    return np.concatenate(
        [_chunker_l2_row_scale(edge, int(dim)) for edge, dim in zip(layout.chunkers, layout.rowdims)]
    )


def _block_l2_col_scale(layout: _BlockKernelLayout) -> np.ndarray:
    return np.concatenate(
        [_chunker_l2_col_scale(edge, int(dim)) for edge, dim in zip(layout.chunkers, layout.coldims)]
    )


def _weighted_density(chnkr: Chunker, dens: ArrayLike) -> np.ndarray:
    dens_arr = np.asarray(dens)
    if dens_arr.size == chnkr.npt:
        return dens_arr.reshape(-1, order="F") * chnkr.wts.reshape(-1, order="F")
    weighted = dens_arr.reshape(-1, order="F")
    if weighted.size % chnkr.npt != 0:
        raise ValueError("density has incompatible size")
    opdims_col = weighted.size // chnkr.npt
    return weighted * np.repeat(chnkr.wts.reshape(-1, order="F"), opdims_col)


def _density_matmul_arg(ncols: int, dens: ArrayLike) -> np.ndarray:
    arr = np.asarray(dens)
    if arr.ndim == 2 and arr.shape[0] == ncols:
        return arr
    vec = arr.reshape(-1, order="F")
    if vec.size != ncols:
        raise ValueError("density has incompatible size")
    return vec


def _eval_kernel(kern: Callable[[Any, Any], np.ndarray], srcinfo: PointInfo, targinfo: PointInfo) -> np.ndarray:
    if hasattr(kern, "eval") and getattr(kern, "eval") is not None:
        return kern.eval(srcinfo, targinfo)
    return kern(srcinfo, targinfo)


def _uses_special_quadrature(
    kern: Callable[[Any, Any], np.ndarray],
    opts: dict[str, Any] | _OperatorOptions | None,
) -> bool:
    return _OperatorOptions.from_any(opts).uses_special_quadrature(kern)


def _special_quadrature_type(kern: Callable[[Any, Any], np.ndarray], options: dict[str, Any] | _OperatorOptions) -> str:
    return _OperatorOptions.from_any(options).special_quadrature_type(kern)


def _is_laplace_double_kernel(kern: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kern, "name", "")).lower() == "laplace" and str(getattr(kern, "type", "")).lower() in {
        "d",
        "double",
    }


def _is_laplace_sprime_kernel(kern: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kern, "name", "")).lower() == "laplace" and str(getattr(kern, "type", "")).lower() in {
        "sp",
        "sprime",
    }


def _is_stokes_strac_kernel(kern: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kern, "name", "")).lower() in {"stokes", "stok"} and str(getattr(kern, "type", "")).lower() in {
        "strac",
        "straction",
    }


def _apply_laplace_double_self_limit(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_laplace_double_kernel(kern) or mat.shape != (chnkr.npt, chnkr.npt):
        return mat
    scale = getattr(kern, "params", {}).get("_scale", 1.0)
    speed = np.sqrt(np.sum(chnkr.d**2, axis=0))
    curvature = (chnkr.d[0] * chnkr.d2[1] - chnkr.d[1] * chnkr.d2[0]) / speed**3
    diag = scale * (-curvature.reshape(-1, order="F") / (4.0 * np.pi)) * chnkr.wts.reshape(-1, order="F")
    np.fill_diagonal(mat, diag)
    return mat


def _apply_laplace_sprime_self_limit(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_laplace_sprime_kernel(kern) or mat.shape != (chnkr.npt, chnkr.npt):
        return mat
    scale = getattr(kern, "params", {}).get("_scale", 1.0)
    speed = np.sqrt(np.sum(chnkr.d**2, axis=0))
    curvature = (chnkr.d[0] * chnkr.d2[1] - chnkr.d[1] * chnkr.d2[0]) / speed**3
    diag = scale * (-curvature.reshape(-1, order="F") / (4.0 * np.pi)) * chnkr.wts.reshape(-1, order="F")
    np.fill_diagonal(mat, diag)
    return mat


def _apply_stokes_strac_self_limit(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_stokes_strac_kernel(kern) or mat.shape != (2 * chnkr.npt, 2 * chnkr.npt):
        return mat
    scale = getattr(kern, "params", {}).get("_scale", 1.0)
    d = chnkr.d.reshape(chnkr.dim, chnkr.npt, order="F")
    speed = np.sqrt(np.sum(d**2, axis=0))
    tangents = d / speed[None, :]
    curvature = chnkr.signed_curvature().reshape(-1, order="F")
    weights = chnkr.wts.reshape(-1, order="F")
    for inode in range(chnkr.npt):
        block = scale * (-curvature[inode] / (2.0 * np.pi)) * np.outer(tangents[:, inode], tangents[:, inode]) * weights[inode]
        rows = slice(2 * inode, 2 * inode + 2)
        mat[rows, rows] = block
    return mat


def _apply_laplace_double_block_self_limits(
    edge_chunkers: list[Chunker],
    kern: Callable[[Any, Any], np.ndarray],
    target_edges: list[int],
    source_edges: list[int],
    target_offsets: np.ndarray,
    source_offsets: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_laplace_double_kernel(kern):
        return mat
    target_lookup = {edge: idx for idx, edge in enumerate(target_edges)}
    source_lookup = {edge: idx for idx, edge in enumerate(source_edges)}
    for edge in sorted(set(target_lookup) & set(source_lookup)):
        i = target_lookup[edge]
        j = source_lookup[edge]
        rows = slice(int(target_offsets[i]), int(target_offsets[i + 1]))
        cols = slice(int(source_offsets[j]), int(source_offsets[j + 1]))
        _apply_laplace_double_self_limit(edge_chunkers[edge], kern, mat[rows, cols])
    return mat


def _acceleration(options: dict[str, Any] | _OperatorOptions) -> str:
    return _OperatorOptions.from_any(options).acceleration


def _flag(options: dict[str, Any] | _OperatorOptions, name: str, default: bool = False) -> bool:
    return _OperatorOptions.from_any(options).flag(name, default)


def _l2scale(options: dict[str, Any] | _OperatorOptions) -> bool:
    return _OperatorOptions.from_any(options).l2scale


def _flamtype(options: dict[str, Any] | _OperatorOptions) -> str:
    return _OperatorOptions.from_any(options).flamtype


def _flam_occ(options: dict[str, Any] | _OperatorOptions) -> int:
    return _OperatorOptions.from_any(options).flam_occ


def _flam_rank_or_tol(options: dict[str, Any] | _OperatorOptions) -> int | float:
    return _OperatorOptions.from_any(options).flam_rank_or_tol


def _flam_options(options: dict[str, Any] | _OperatorOptions, *, store_default: str | None = None) -> dict[str, Any]:
    return _OperatorOptions.from_any(options).flam_options(store_default=store_default)


def _fmm_tol(options: dict[str, Any] | _OperatorOptions, default: float = 1.0e-12) -> float:
    return _OperatorOptions.from_any(options).fmm_tol(default)


def _pquad_enabled(options: dict[str, Any] | _OperatorOptions) -> bool:
    raw = _OperatorOptions.from_any(options).raw
    if "forcepquad" in raw:
        return _option_bool(raw["forcepquad"])
    return _option_bool(raw.get("usepquad", True))


def _pquad_side(options: dict[str, Any] | _OperatorOptions) -> str | None:
    raw = _OperatorOptions.from_any(options).raw
    if "side" not in raw or raw["side"] is None:
        return None
    side = str(raw["side"]).lower()
    if side not in {"i", "e"}:
        raise ValueError("side must be 'i' or 'e'")
    return side


def _boundary_pquad_enabled(options: dict[str, Any] | _OperatorOptions) -> bool:
    raw = _OperatorOptions.from_any(options).raw
    if "forcepquad" in raw or "usepquad" in raw:
        return _pquad_enabled(options)
    return _pquad_side(options) is not None


def _require_fmm(kern: Callable[[Any, Any], np.ndarray]) -> None:
    if getattr(kern, "fmm", None) is None:
        raise NotImplementedError("FMM acceleration requested, but the kernel has no FMM evaluator")


def _require_pyflam():
    try:
        import pyflam
    except (ImportError, OSError) as exc:  # pragma: no cover - dependency is required in packaged installs.
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
    from .quadrature import ggq as quadggq

    qtype = _special_quadrature_type(kern, options)
    spmat = quadggq.buildmattd(
        chnkr,
        kern,
        getattr(kern, "opdims", None),
        type=qtype,
        ilist=options.get("ilist", None),
        corrections=False,
        pquad_side=_pquad_side(options),
        usepquad=_boundary_pquad_enabled(options),
    )
    if _l2scale(options):
        spmat = _apply_l2scale_matrix(chnkr, spmat)
    return spmat


def _block_special_overwrite_matrix(layout: _BlockKernelLayout, options: dict[str, Any]) -> spmatrix:
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []
    for idx, chnkr in enumerate(layout.chunkers):
        kern = layout.kernels[idx, idx]
        if not _uses_special_quadrature(kern, options):
            continue
        local_options = dict(options)
        local_options.pop("l2scale", None)
        spmat = _special_overwrite_matrix(chnkr, kern, local_options).tocoo()
        if spmat.nnz == 0:
            continue
        rows_all.append(spmat.row + int(layout.row_offsets[idx]))
        cols_all.append(spmat.col + int(layout.col_offsets[idx]))
        vals_all.append(spmat.data)
    if not vals_all:
        return sparse.csr_matrix((int(layout.row_offsets[-1]), int(layout.col_offsets[-1])))
    out = sparse.csr_matrix(
        (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(int(layout.row_offsets[-1]), int(layout.col_offsets[-1])),
    )
    if _l2scale(options):
        return sparse.csr_matrix(_apply_chunkgraph_l2scale(layout.chunkers, layout.rowdims, layout.coldims, out))
    return out


def _block_uses_special_quadrature(layout: _BlockKernelLayout, options: dict[str, Any]) -> bool:
    for idx in range(len(layout.chunkers)):
        if _uses_special_quadrature(layout.kernels[idx, idx], options):
            return True
    return False


def _block_special_correction_matrix(layout: _BlockKernelLayout, options: dict[str, Any]) -> spmatrix:
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []
    for idx, chnkr in enumerate(layout.chunkers):
        kern = layout.kernels[idx, idx]
        if not _uses_special_quadrature(kern, options):
            continue
        local_options = dict(options)
        local_options.pop("l2scale", None)
        spmat = _special_correction_matrix(chnkr, kern, local_options).tocoo()
        if spmat.nnz == 0:
            continue
        rows_all.append(spmat.row + int(layout.row_offsets[idx]))
        cols_all.append(spmat.col + int(layout.col_offsets[idx]))
        vals_all.append(spmat.data)
    if not vals_all:
        return sparse.csr_matrix((int(layout.row_offsets[-1]), int(layout.col_offsets[-1])))
    return sparse.csr_matrix(
        (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(int(layout.row_offsets[-1]), int(layout.col_offsets[-1])),
    )


def _option_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _apply_l2scale_matrix(chnkr: Chunker, mat: np.ndarray | spmatrix) -> np.ndarray | spmatrix:
    npt = chnkr.npt
    if mat.shape[0] % npt != 0 or mat.shape[1] % npt != 0:
        raise ValueError("l2scale matrix dimensions must be multiples of chunker.npt")
    op0 = mat.shape[0] // npt
    op1 = mat.shape[1] // npt
    weights = chnkr.wts.reshape(-1, order="F")
    row_scale = np.sqrt(np.repeat(weights, op0))
    col_scale = 1.0 / np.sqrt(np.repeat(weights, op1))
    if sparse.issparse(mat):
        return sparse.diags(row_scale, format="csr") @ mat @ sparse.diags(col_scale, format="csr")
    return row_scale[:, None] * np.asarray(mat) * col_scale[None, :]


def _add_diagonal_shift(out: np.ndarray, rows: np.ndarray, cols: np.ndarray, dval: np.ndarray) -> np.ndarray:
    rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols_arr = np.asarray(cols, dtype=np.int64).reshape(-1)
    if rows_arr.size == 0 or cols_arr.size == 0:
        return out
    col_positions: dict[int, list[int]] = {}
    for pos, col in enumerate(cols_arr):
        col_positions.setdefault(int(col), []).append(pos)
    touched = False
    shifted = out
    for row_pos, row in enumerate(rows_arr):
        positions = col_positions.get(int(row))
        if positions is None:
            continue
        if not touched:
            shifted = np.array(out, dtype=np.result_type(out.dtype, dval.dtype), copy=True)
            touched = True
        shifted[row_pos, positions] += dval[int(row)]
    return shifted


def _chunkerflam_proxyfun(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
):
    from .acceleration import flam

    rank_or_tol = _flam_rank_or_tol(options)
    optsnpxy = {"rank_or_tol": float(rank_or_tol), "nsrc": _flam_occ(options)}
    width = float(np.max(chnkr.max() - chnkr.min()))
    proxybylevel = _flag(options, "proxybylevel")
    if not proxybylevel:
        npxy = flam.nproxy_square(kern, width, optsnpxy)
        if npxy == -1:
            return None
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)

        def pxyfun(x: np.ndarray, slf: np.ndarray, nbr: np.ndarray, l: np.ndarray, ctr: np.ndarray):
            _ = x
            return flam.proxyfun(slf, nbr, l, ctr, chnkr, kern, opdims, pr, ptau, pw, pin, True, _l2scale(options))

        return pxyfun

    def pxyfun(x: np.ndarray, slf: np.ndarray, nbr: np.ndarray, l: np.ndarray, ctr: np.ndarray):
        _ = x
        level_width = float(np.max(np.asarray(l, dtype=float)))
        npxy = flam.nproxy_square(kern, level_width, optsnpxy)
        if npxy == -1:
            return np.zeros((0, np.asarray(slf).size)), np.asarray(nbr, dtype=np.int64)
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)
        return flam.proxyfun(slf, nbr, l, ctr, chnkr, kern, opdims, pr, ptau, pw, pin, True, _l2scale(options))

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
    ntarget = PointInfo.from_any(targobj).r.shape[1]
    return np.asarray(vals).reshape(-1, ntarget, order="F")


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
    from .acceleration import flam

    targinfo = PointInfo.from_any(targobj)
    op0, op1 = _kernel_opdims(chnkr, kern, targinfo)
    nrows = targinfo.r.shape[1] * op0
    ncols = chnkr.npt * op1

    def matfun(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return flam.kernbyindexr(rows, cols, targinfo, chnkr, kern, (op0, op1))

    rx = np.repeat(np.real(targinfo.r), op0, axis=1)
    cx = np.repeat(np.real(PointInfo.from_any(chnkr).r), op1, axis=1)
    occ = _flam_occ(options)
    rank_or_tol = _flam_rank_or_tol(options)
    opts_ifmm = _flam_options(options, store_default="n")
    # Store near and diagonal blocks; application still receives matfun for
    # interactions not retained by the compact representation.
    pxyfun = None
    if _flag(options, "useproxy", True) and chnkr.datadim == 0 and targinfo.data is None:
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
    from .acceleration import flam

    rank_or_tol = _flam_rank_or_tol(options)
    optsnpxy = {"rank_or_tol": float(rank_or_tol), "nsrc": _flam_occ(options)}
    all_points = np.column_stack((np.real(targinfo.r), np.real(PointInfo.from_any(chnkr).r)))
    width = float(np.max(np.max(all_points, axis=1) - np.min(all_points, axis=1)))
    proxybylevel = _flag(options, "proxybylevel")
    if not proxybylevel:
        npxy = flam.nproxy_square(kern, width, optsnpxy)
        if npxy == -1:
            return None
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)

        def pxyfun(rc: str, rx: np.ndarray, cx: np.ndarray, slf: np.ndarray, nbr: np.ndarray, l: np.ndarray, ctr: np.ndarray):
            return flam.proxyfunr(rc, rx, cx, slf, nbr, l, ctr, chnkr, kern, opdims, pr, ptau, pw, pin, targobj=targinfo)

        return pxyfun

    def pxyfun(rc: str, rx: np.ndarray, cx: np.ndarray, slf: np.ndarray, nbr: np.ndarray, l: np.ndarray, ctr: np.ndarray):
        level_width = float(np.max(np.asarray(l, dtype=float)))
        npxy = flam.nproxy_square(kern, level_width, optsnpxy)
        if npxy == -1:
            slf_size = np.asarray(slf).size
            if str(rc).lower() == "c":
                return np.zeros((0, slf_size)), np.asarray(nbr, dtype=np.int64)
            return np.zeros((slf_size, 0)), np.asarray(nbr, dtype=np.int64)
        pr, ptau, pw, pin = flam.proxy_square_pts(npxy)
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
    op0, op1 = _kernel_opdims(chnkr, kern)
    use_l2scale = _l2scale(options)
    eval_dens = dens_vec * _chunker_l2_col_scale(chnkr, op1) if use_l2scale else dens_vec
    if _uses_special_quadrature(kern, options):
        mat_options = dict(options)
        mat_options.pop("acceleration", None)
        mat_options.pop("l2scale", None)
        vals = chunkermat(chnkr, kern, mat_options) @ eval_dens
        if use_l2scale:
            vals = _chunker_l2_row_scale(chnkr, op0) * vals
        return vals
    fmm_options = dict(options)
    fmm_options["acceleration"] = "fmm"
    fmm_options.pop("l2scale", None)
    vals = chunkerkerneval(chnkr, kern, eval_dens, chnkr, fmm_options).reshape(-1, order="F")
    if use_l2scale:
        vals = _chunker_l2_row_scale(chnkr, op0) * vals
    return vals


def _block_chunkermatapply_fmm(
    layout: _BlockKernelLayout,
    dens: ArrayLike,
    options: dict[str, Any],
    correction: spmatrix | None = None,
) -> np.ndarray:
    dens_vec = np.asarray(dens).reshape(-1, order="F")
    ncols = int(layout.col_offsets[-1])
    if dens_vec.size != ncols:
        raise ValueError("density has incompatible size")
    use_l2scale = _l2scale(options)
    eval_dens = dens_vec * _block_l2_col_scale(layout) if use_l2scale else dens_vec
    nrows = int(layout.row_offsets[-1])
    out = np.zeros(nrows, dtype=np.result_type(_block_operator_dtype(layout), eval_dens.dtype))
    fmm_options = dict(options)
    fmm_options["acceleration"] = "fmm"
    fmm_options.pop("l2scale", None)

    for itarg, targ in enumerate(layout.chunkers):
        rows = slice(int(layout.row_offsets[itarg]), int(layout.row_offsets[itarg + 1]))
        for isrc, src in enumerate(layout.chunkers):
            kern = layout.kernels[itarg, isrc]
            _require_fmm(kern)
            cols = slice(int(layout.col_offsets[isrc]), int(layout.col_offsets[isrc + 1]))
            vals = chunkerkerneval(src, kern, eval_dens[cols], targ, fmm_options).reshape(-1, order="F")
            out[rows] += vals

    if _block_uses_special_quadrature(layout, options):
        corr = _block_special_correction_matrix(layout, options) if correction is None else correction
        out = out + corr @ eval_dens
    if use_l2scale:
        out = _block_l2_row_scale(layout) * out
    return out


def _chunkerkernevalmat_fmm(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targobj: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    targinfo = PointInfo.from_any(targobj)
    op0, op1 = _kernel_opdims(chnkr, kern, targinfo)
    nrows = targinfo.r.shape[1] * op0
    ncols = chnkr.npt * op1
    eye = np.eye(ncols, dtype=_operator_dtype(chnkr, kern))
    out = np.empty((nrows, ncols), dtype=np.result_type(eye.dtype, complex if np.iscomplexobj(getattr(kern, "params", None)) else float))
    fmm_options = dict(options)
    fmm_options["acceleration"] = "fmm"
    for col in range(ncols):
        out[:, col] = chunkerkerneval(chnkr, kern, eye[:, col], targinfo, fmm_options).reshape(-1, order="F")
    return out


def _special_correction_matrix(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
) -> spmatrix:
    from .quadrature import ggq as quadggq

    qtype = _special_quadrature_type(kern, options)
    return quadggq.buildmattd(
        chnkr,
        kern,
        getattr(kern, "opdims", None),
        type=qtype,
        ilist=options.get("ilist", None),
        corrections=True,
        pquad_side=_pquad_side(options),
        usepquad=_boundary_pquad_enabled(options),
    )


def _target_adaptive_correction_matrix(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targinfo: PointInfo,
    options: dict[str, Any],
) -> spmatrix:
    """Sparse adaptive correction replacing smooth near-target blocks."""

    op0, op1 = _kernel_opdims(chnkr, kern, targinfo)
    ntarget = targinfo.r.shape[1]
    flags = chnkr.flagnear(targinfo.r, fac=float(options.get("fac", 1.0)))
    if not np.any(flags):
        return sparse.csr_matrix((op0 * ntarget, op1 * chnkr.npt))

    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []
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
        srcinfo = PointInfo(
            r=chnkr.r[:, :, src_chunk],
            d=chnkr.d[:, :, src_chunk],
            d2=chnkr.d2[:, :, src_chunk],
            n=chnkr.n[:, :, src_chunk],
            data=chnkr.data[:, :, src_chunk] if chnkr.datadim else None,
        )
        smooth = _eval_kernel(kern, srcinfo, subinfo) * np.repeat(chnkr.wts[:, src_chunk], op1)[None, :]
        close_panel = _target_close_panel_matrix(chnkr, src_chunk, subinfo, kern, (op0, op1), options)
        delta = close_panel - smooth

        global_rows = (target_ids[:, None] * op0 + np.arange(op0)[None, :]).reshape(-1)
        col_start = src_chunk * chnkr.k * op1
        global_cols = col_start + np.arange(chnkr.k * op1)
        rr, cc = np.indices(delta.shape)
        rows_all.append(global_rows[rr.reshape(-1)])
        cols_all.append(global_cols[cc.reshape(-1)])
        vals_all.append(delta.reshape(-1))

    if not vals_all:
        return sparse.csr_matrix((op0 * ntarget, op1 * chnkr.npt))
    return sparse.csr_matrix(
        (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(op0 * ntarget, op1 * chnkr.npt),
    )


def _target_adaptive_matrix(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targinfo: PointInfo,
    options: dict[str, Any],
) -> np.ndarray:
    """Build a target-evaluation matrix with adaptive close-panel replacements."""

    opdims = _kernel_opdims(chnkr, kern, targinfo)
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    srcinfo = PointInfo.from_any(chnkr)
    mat = _eval_kernel(kern, srcinfo, targinfo)
    wts = chnkr.wts.reshape(-1, order="F")
    if mat.shape[1] == chnkr.npt:
        mat = mat * wts[None, :]
    elif mat.shape[1] % chnkr.npt == 0:
        mat = mat * np.repeat(wts, mat.shape[1] // chnkr.npt)[None, :]
    else:
        raise ValueError("kernel column dimension is incompatible with chunker points")

    flags = chnkr.flagnear(targinfo.r, fac=float(options.get("fac", 1.0)))
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
        submat = _target_close_panel_matrix(chnkr, src_chunk, subinfo, kern, (op0, op1), options)
        col_start = src_chunk * chnkr.k * op1
        cols = slice(col_start, col_start + chnkr.k * op1)
        for local_idx, target_idx in enumerate(target_ids):
            rows = slice(op0 * target_idx, op0 * (target_idx + 1))
            local_rows = slice(op0 * local_idx, op0 * (local_idx + 1))
            mat[rows, cols] = submat[local_rows, :]
    return mat


def _target_close_panel_matrix(
    chnkr: Chunker,
    src_chunk: int,
    targinfo: PointInfo,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
) -> np.ndarray:
    from .quadrature import adaptive as quadadap

    pquad_mat, handled = _target_pquad_panel_matrix(chnkr, src_chunk, targinfo, kern, opdims, options)
    if pquad_mat is not None and np.all(handled):
        return np.real_if_close(pquad_mat)

    adaptive = quadadap.adapgausswts(chnkr, src_chunk, targinfo, kern, opdims, opts=options)[0]
    if pquad_mat is not None and np.any(handled):
        op0 = int(opdims[0])
        rows = _target_rows(np.flatnonzero(handled), op0)
        adaptive = np.asarray(adaptive, dtype=np.result_type(adaptive.dtype, pquad_mat.dtype))
        adaptive[rows, :] = pquad_mat[rows, :]
    return adaptive


def _target_pquad_panel_matrix(
    chnkr: Chunker,
    src_chunk: int,
    targinfo: PointInfo,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    options: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray]:
    if not _pquad_enabled(options):
        return None, np.zeros(targinfo.r.shape[1], dtype=bool)
    from .quadrature import panel as pquad

    splitinfo = pquad.splitinfo_for_kernel(kern)
    if splitinfo is None or tuple(splitinfo.opdims) != (int(opdims[0]), int(opdims[1])):
        return None, np.zeros(targinfo.r.shape[1], dtype=bool)
    side_tol = options.get("side_tol", None)
    block, handled = pquad.panel_matrix_auto_side(
        chnkr,
        src_chunk,
        targinfo,
        splitinfo,
        side=_pquad_side(options),
        side_tol=None if side_tol is None else float(side_tol),
    )
    return block, handled


def _kernel_opdims(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    targinfo: PointInfo | None = None,
) -> tuple[int, int]:
    opdims = getattr(kern, "opdims", None)
    if opdims is not None and tuple(opdims) != (0, 0):
        return int(opdims[0]), int(opdims[1])
    src = _pointinfo_node(chnkr, 0)
    targ = _pointinfo_first(targinfo) if targinfo is not None else _pointinfo_node(chnkr, 1 if chnkr.npt > 1 else 0)
    mat = _eval_kernel(kern, src, targ)
    return int(mat.shape[0]), int(mat.shape[1])


def _operator_dtype(chnkr: Chunker, kern: Callable[[Any, Any], np.ndarray]) -> np.dtype:
    return _probe_kernel_dtype(
        kern,
        _pointinfo_node(chnkr, 0),
        _pointinfo_node(chnkr, 1 if chnkr.npt > 1 else 0),
    )


def _probe_kernel_dtype(kern: Callable[[Any, Any], np.ndarray], src: PointInfo, targ: PointInfo) -> np.dtype:
    try:
        return np.asarray(_eval_kernel(kern, src, targ)).dtype
    except _KERNEL_PROBE_EXCEPTIONS:
        return np.dtype(float)


def _pointinfo_node(chnkr: Chunker, inode: int) -> PointInfo:
    src = PointInfo.from_any(chnkr)
    idx = int(inode)
    return _pointinfo_take(src, np.array([idx], dtype=np.int64))


def _pointinfo_first(info: PointInfo) -> PointInfo:
    return _pointinfo_take(info, np.array([0], dtype=np.int64))


def _pointinfo_take(info: PointInfo, indices: np.ndarray) -> PointInfo:
    return PointInfo(
        r=info.r[:, indices],
        d=info.d[:, indices] if info.d is not None else None,
        d2=info.d2[:, indices] if info.d2 is not None else None,
        n=info.n[:, indices] if info.n is not None else None,
        data=info.data[:, indices] if info.data is not None else None,
    )


def _target_rows(indices: np.ndarray, op0: int) -> np.ndarray:
    return (indices[:, None] * int(op0) + np.arange(int(op0))[None, :]).reshape(-1)


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
